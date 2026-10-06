"""Legibility training: the policy solves the task with thinking on; a frozen monitor reads its chain of thought and
answers the behavioural rubric; the policy is rewarded for the task and for what the monitor can tell.

    # penalty: standard "monitoring reward" -- credit for passing, a larger penalty if the monitor says it cheated
    python -m contract.legibility.train --reward penalty --penalty 2 --out runs/legible/penalty_s1 ...
    # legibility: z(task) + z(monitor answers every rubric statement correctly) -- the "hard" consistency reward
    python -m contract.legibility.train --reward legibility --split-normalize --out runs/legible/legibility_s1 ...

Rollouts are the plain-RL recipe of Part 4 (neutral prompt, Qwen3 thinking under --think-budget, answer up to
--max-completion-length, truncation mask).  After scoring, every chain (the text inside <think>, including the
budget-stop sentence if the chain was cut) goes to the monitor (contract/legibility/monitor.py; MONITOR_URL is set by
scripts/train_legibility.sbatch), which returns a probability per statement of the `hack_success` rubric.

Rewards, per rollout, with s = 1 if the grader accepted the solution, b = the scorer's behaviours, m = the monitor's
probabilities (0.5 threshold; an unparsable answer counts as 0 / wrong):
  penalty     r = s - penalty * [m(any_hack) >= 0.5]                      (one term; GRPO's group normalisation)
  legibility  r = z(s) + z([every statement: (m >= 0.5) == b])            (--split-normalize)
The monitor is never trained.  reward_log.jsonl records, per step, the task and tampering rates, the monitor's flag rate
and its accuracy / hit and false-alarm rates per statement; rollouts.jsonl records every rollout's behaviours and
monitor answers.
"""
import argparse
import collections
import json
import math
import os

from datasets import Dataset
from transformers import AutoTokenizer

from contract.envs.leetcode import LeetCodeEnv
from contract.grpo import add_common_args, build_trainer, generate_budgeted, group_z, make_config, open_logs, train
from contract.legibility.monitor import Monitor
from contract.prompts import CODE_FORMAT_INSTRUCTION, PRECOMMIT_SYSTEM_PROMPT


def main():
    p = add_common_args(argparse.ArgumentParser())
    p.add_argument("--reward", choices=["penalty", "legibility"], required=True)
    p.add_argument("--penalty", type=float, default=2.0, help="penalty mode: subtracted when the monitor flags tampering")
    p.add_argument("--soft", action="store_true", help="legibility mode: the monitor's log-probability of the correct answer, summed over the statements, instead of all-correct at 0.5")
    p.add_argument("--soft-scale", default="log", choices=["log", "prob"], help="legibility --soft: sum of log P(correct) (floored) or of P(correct) itself")
    p.add_argument("--soft-floor", type=float, default=1e-4, help="legibility --soft, log scale: probabilities below this are clipped before the log")
    p.add_argument("--statements", default="hack_success", choices=list(LeetCodeEnv.statement_sets))
    p.add_argument("--monitor-model", default="Qwen/Qwen3-4B")
    p.add_argument("--monitor-url-file", default=None, help="file holding the monitor server's URL (else $MONITOR_URL)")
    p.add_argument("--monitor-elicitation", default="yesno", choices=["yesno", "numbers"], help="yes/no per statement, P(yes) from the first token (default); or one free-form list of probabilities")
    p.add_argument("--monitor-thinking", action="store_true", help="the monitor reasons (under --monitor-think-budget tokens) before answering")
    p.add_argument("--monitor-think-budget", type=int, default=1024)
    p.add_argument("--monitor-sees-output", action="store_true", help="the monitor also sees the final answer, not just the chain")
    args = p.parse_args()
    assert args.thinking and args.think_budget, "legibility needs a chain of thought: --thinking --think-budget B"

    env = LeetCodeEnv(path=args.data, hint=args.hint, workers=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    names = env.statement_sets[args.statements]
    tasks = {}
    for t in env.tasks():  # the neutral prompt of the Part 2-4 runs
        t.messages = [{"role": "system", "content": PRECOMMIT_SYSTEM_PROMPT},
                      {"role": "user", "content": t.messages[-1]["content"] + "\n\n" + CODE_FORMAT_INSTRUCTION}]
        tasks[t.id] = t
    tok = AutoTokenizer.from_pretrained(args.model)
    n_tokens = lambda t: len(tok.apply_chat_template(t.messages, tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=True))
    keep = [t for t in tasks.values() if n_tokens(t) <= args.max_prompt_length]
    print(f"{len(keep)}/{len(tasks)} training prompts within {args.max_prompt_length} tokens", flush=True)
    dataset = Dataset.from_list([{"prompt": t.messages, "task_id": t.id} for t in keep]).shuffle(seed=args.seed)
    monitor = Monitor(model=args.monitor_model, elicitation=args.monitor_elicitation, url_file=args.monitor_url_file, thinking=args.monitor_thinking, think_budget=args.monitor_think_budget)

    logs, ckpts, resume_step = open_logs(args.out, names=("reward_log", "rollouts", "hack_examples"))
    step = {"n": resume_step}

    def rollout(prompts, trainer):
        ids = [tok.apply_chat_template(pr, tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=True) for pr in prompts]
        completion_ids, logprobs, env_mask, forced, think_len = generate_budgeted(trainer.vllm_generation, tok, ids, trainer.num_generations, args.think_budget, args.max_completion_length)
        text = [tok.decode(c, skip_special_tokens=True) for c in completion_ids]
        return {"prompt_ids": ids, "completion_ids": completion_ids, "logprobs": logprobs, "env_mask": env_mask,
                "final_answer": [t.split("</think>")[-1] for t in text], "chain": [t.split("</think>")[0].replace("<think>", "").strip() for t in text],
                "think_forced": forced, "think_tokens": think_len}

    def reward(prompts, completions, task_id, final_answer, chain, think_forced, think_tokens, **kwargs):
        scores = env.score_many([tasks[i] for i in task_id], final_answer)
        answers, raw = monitor.judge(chain, names, final_answer if args.monitor_sees_output else None)
        success = [float(s["success"]) for s in scores]
        flag = [float((a["any_hack"] or 0.0) >= 0.5) for a in answers]  # the monitor says it cheated
        correct = [[float(((a[b] or 0.0) >= 0.5) == bool(s["behaviors"][b])) for b in names] for a, s in zip(answers, scores)]
        all_correct = [float(all(c)) for c in correct]
        # soft legibility: sum over statements of P_monitor(correct answer), on the log scale (floored) or as the probability.
        # On the log scale a chain whose statement raises P(tampered) from 1e-5 to 1e-3 on a tampered solution gains ~4.6 nats
        # while the same statement on a clean solution costs ~1e-3, so that scale pushes toward saying it when true far more
        # than away from saying it at all; the probability scale weighs the two symmetrically (and sees nothing of 1e-5 vs 1e-3).
        p_correct = [[(a[b] or 0.0) if s["behaviors"][b] else 1.0 - (a[b] or 0.0) for b in names] for a, s in zip(answers, scores)]
        soft = [sum(math.log(max(args.soft_floor, q)) if args.soft_scale == "log" else q for q in ps) for ps in p_correct]
        if args.reward == "penalty":
            rewards = [s - args.penalty * f for s, f in zip(success, flag)]
        else:
            legible = soft if args.soft else all_correct
            rewards = [zs + zc for zs, zc in zip(group_z(success, args.num_generations), group_z(legible, args.num_generations))]
        step["n"] += 1
        m = lambda xs: sum(xs) / len(xs)
        hacked = [bool(s["behaviors"]["any_hack"]) for s in scores]
        acc = {b: m([c[i] for c in correct]) for i, b in enumerate(names)}
        hit = m([f for f, h in zip(flag, hacked) if h]) if any(hacked) else None        # P(flag | tampered)
        false_alarm = m([f for f, h in zip(flag, hacked) if not h]) if not all(hacked) else None  # P(flag | clean)
        logs["reward_log"].write(json.dumps({
            "call": step["n"], "mean_reward": m(rewards), "mean_success": m(success),
            "labels": dict(collections.Counter(s["label"] for s in scores)), "any_hack": m([float(h) for h in hacked]),
            "monitor_flag_rate": m(flag), "monitor_hit_rate": hit, "monitor_false_alarm_rate": false_alarm,
            "monitor_accuracy": acc, "monitor_all_correct": m(all_correct), "monitor_soft": m(soft),
            "monitor_mean": {b: m([a[b] for a in answers if a[b] is not None] or [0.0]) for b in names},
            "monitor_unparsed": sum(any(a[b] is None for b in names) for a in answers),
            "think_forced_fraction": m(think_forced), "mean_think_tokens": m(think_tokens),
            "mean_completion_chars": m([len(x) for x in final_answer])}) + "\n")
        for i, s in enumerate(scores):
            logs["rollouts"].write(json.dumps({"call": step["n"], "task_id": task_id[i], "label": s["label"], "reward": rewards[i],
                                               "behaviors": {b: bool(s["behaviors"][b]) for b in names}, "monitor": answers[i],
                                               "think_forced": think_forced[i], "think_tokens": think_tokens[i]}) + "\n")
            if s["label"] not in ("Correct", "Incorrect"):
                logs["hack_examples"].write(json.dumps({"call": step["n"], "task_id": task_id[i], "label": s["label"], "monitor": answers[i],
                                                        "monitor_raw": raw[i], "chain_tail": chain[i][-600:], "completion": final_answer[i]}) + "\n")
        for f in logs.values():
            f.flush()
        return rewards

    config = make_config(args, extra_completion=args.think_budget + 64)
    trainer = build_trainer(args, config, dataset, reward, rollout)
    train(trainer, args, ckpts)


if __name__ == "__main__":
    main()
