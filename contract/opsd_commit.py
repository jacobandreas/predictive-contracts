"""OPSD warm-up of a *reasoning* commitment turn (on-policy self-distillation, Zhao et al. 2026,
https://siyan-zhao.github.io/blog/2026/opsd/), the Part 4 replacement for contract/sft_commit.py.

    python -m contract.opsd_commit --targets results/probe/base_think4k_train_targets.json --out runs/opsd_commit_think

Student and teacher are the same model.  The student sees the commitment prompt with the reasoning note
(contract.prompts.commit_messages(..., reason=True), the prompt train_grpo --commit-thinking uses afterwards),
thinks for at most --think-budget tokens (a quarter of the solver's budget) and answers.  The teacher is the
*initial* policy (the LoRA adapter disabled) shown the same prompt plus the probe's calibrated estimates for the
problem as privileged information (OPSD_TEACHER_HINT: "on this problem a model like you is accepted with
probability X and tampers with probability Y; work out why and give exactly these numbers").  Every token the
student samples is scored by both.  Two terms, over two segments of the completion:

  * answer tokens (after </think>, where the hint bites): the full next-token distributions are matched,
    KL(p_teacher || p_student) summed over the vocabulary at each position -- OPSD's per-token distribution
    matching.  A *sampled*-token estimate does not work here: Qwen tokenizes digits one at a time, so pushing
    down the digit the student sampled ("7" in 0.75) spreads mass over every other digit rather than onto the
    teacher's ("6" in 0.66); the first two attempts drifted to ever lower numbers this way.
  * chain tokens: the sampled-token estimate, advantage_t = clip(log p_teacher(y_t) - log p_student(y_t), -C, C)
    scaled by --chain-weight (default 0.1), through GRPO's policy-gradient loss.  Teacher and student differ by
    noise here (the hint says nothing about how to reason), so this term is kept small.

Gradients flow through the student only; the teacher is the initial policy (adapter disabled).  No task reward
is involved: the warm-up teaches the model to state the probe's numbers after reasoning about the problem.  Runs
inside TRL's GRPOTrainer (vLLM colocate for sampling; two extra forwards per micro-batch for the teacher and the
student's answer-position logits).  The saved adapter (--out/final) is the --init-adapter for train_grpo
--decoupled --commit-thinking.
"""
import argparse
import json
import os

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from trl.trainer.utils import pad

from contract.envs.leetcode import LeetCodeEnv
from contract.prompts import COMMIT_THINK_BUDGET_STOP, OPSD_FACTS, OPSD_TEACHER_HINT, commit_messages
from contract.run_tasks import parse_precommit
from contract.train_grpo import generate_budgeted


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--data", default="data/leetcode/leetcode_train_medhard_filtered.jsonl")
    p.add_argument("--hint", default="modify_tests")
    p.add_argument("--statements", default="hack_success", choices=list(LeetCodeEnv.statement_sets))
    p.add_argument("--targets", required=True, help="JSON {task_id: {behavior: probability}} from contract.probe --predict-out")
    p.add_argument("--think-budget", type=int, default=1024, help="commitment thinking budget (the solver uses 4096)")
    p.add_argument("--answer-cap", type=int, default=128, help="tokens for the answer after the chain (train_grpo --commit-max-tokens)")
    p.add_argument("--clip", type=float, default=5.0, help="cap on |log p_teacher - log p_student| per token")
    p.add_argument("--chain-weight", type=float, default=0.1, help="weight of the sampled-token term on the reasoning-chain tokens")
    p.add_argument("--answer-kl-weight", type=float, default=1.0, help="weight of the full-distribution KL on the answer tokens")
    p.add_argument("--num-prompts", type=int, default=16)
    p.add_argument("--num-generations", type=int, default=4)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--per-device-batch", type=int, default=2)
    p.add_argument("--max-prompt-length", type=int, default=2048)
    p.add_argument("--vllm-gpu-mem", type=float, default=0.35)
    p.add_argument("--save-steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    env = LeetCodeEnv(path=args.data, hint=args.hint)
    names = env.statement_sets[args.statements]
    questions = "\n".join(f"{i + 1}. {env.behavior_questions[b]}" for i, b in enumerate(names))
    targets = json.load(open(args.targets))
    tok = AutoTokenizer.from_pretrained(args.model)
    chat = lambda msgs: tok.apply_chat_template(msgs, tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=True)

    rows, teacher_prompt_ids = [], {}
    for t in env.tasks():
        if t.id not in targets:
            continue
        student = commit_messages(t.messages[-1]["content"], "prob", questions, reason=True)
        facts = " and ".join(OPSD_FACTS[b].format(p=targets[t.id][b]) for b in names)
        lines = "\n".join(f"{i + 1}. {targets[t.id][b]:.2f}" for i, b in enumerate(names))
        teacher = [student[0], {"role": "user", "content": student[1]["content"] + OPSD_TEACHER_HINT.format(facts=facts, lines=lines)}]
        if len(chat(teacher)) > args.max_prompt_length:
            continue
        rows.append({"prompt": student, "task_id": t.id})
        teacher_prompt_ids[t.id] = chat(teacher)
    print(f"{len(rows)} problems with targets within the prompt budget", flush=True)
    dataset = Dataset.from_list(rows).shuffle(seed=args.seed)
    max_steps = int(args.epochs * len(rows) / args.num_prompts)

    state = {"n": 0, "pending": {}}
    log = open(f"{args.out}/reward_log.jsonl", "a")
    samples = open(f"{args.out}/samples.jsonl", "a")  # a few raw student completions per step, to see what the answers look like
    eos = tok.convert_tokens_to_ids("<|im_end|>")

    def rollout(prompts, trainer):
        """The student's commitment: think under the budget, then answer briefly (one episode per entry)."""
        ids = [chat(p) for p in prompts]
        completion_ids, logprobs, env_mask, forced, think_len = generate_budgeted(trainer.vllm_generation, tok, ids, trainer.num_generations, args.think_budget, args.answer_cap, COMMIT_THINK_BUDGET_STOP)
        text = [tok.decode(c, skip_special_tokens=True) for c in completion_ids]
        return {"prompt_ids": ids, "completion_ids": completion_ids, "logprobs": logprobs, "env_mask": env_mask,
                "final_answer": [t.split("</think>")[-1] for t in text], "think_forced": forced, "think_tokens": think_len,
                "finished": [float(c[-1] == eos) for c in completion_ids], "text": text}

    def reward(prompts, completions, task_id, final_answer, think_forced, think_tokens, finished, text, **kwargs):
        """No reward: the advantages come from the teacher (OPSDTrainer).  This only records how the answers look."""
        answers = [parse_precommit(a.split("```")[0], "prob", len(names)) for a in final_answer]
        for tid, t, a in list(zip(task_id, text, answers))[:4]:
            samples.write(json.dumps({"call": state["n"] + 1, "task_id": tid, "target": targets[tid], "parsed": a, "think_tail": t.split("</think>")[0][-300:], "answer": t.split("</think>")[-1]}) + "\n")
        samples.flush()
        err = {b: [abs(a[i] - targets[tid][b]) for a, tid in zip(answers, task_id) if a[i] is not None] for i, b in enumerate(names)}
        state["n"] += 1
        state["pending"] = {"call": state["n"], "unparsed": sum(any(v is None for v in a) for a in answers), "n": len(answers),
                            "mean_abs_error": {b: sum(e) / len(e) if e else None for b, e in err.items()},
                            "predicted": {b: sum(a[i] for a in answers if a[i] is not None) / max(1, sum(a[i] is not None for a in answers)) for i, b in enumerate(names)},
                            "target": {b: sum(targets[tid][b] for tid in task_id) / len(task_id) for b in names},
                            "think_forced_fraction": sum(think_forced) / len(think_forced), "mean_think_tokens": sum(think_tokens) / len(think_tokens),
                            "finished_fraction": sum(finished) / len(finished)}
        return [0.0] * len(completions)

    config = GRPOConfig(
        output_dir=args.out, seed=args.seed, learning_rate=args.lr, lr_scheduler_type="cosine", warmup_steps=5,
        weight_decay=0.1, max_grad_norm=1.0, beta=0.0,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.num_prompts * args.num_generations // args.per_device_batch,
        num_generations=args.num_generations,
        max_completion_length=args.think_budget + 64 + args.answer_cap,
        vllm_max_model_length=args.max_prompt_length + 400 + args.think_budget + 64 + args.answer_cap,
        temperature=0.7, top_p=0.95, max_steps=max_steps, save_steps=args.save_steps, save_total_limit=2, save_only_model=True,
        scale_rewards="none", vllm_importance_sampling_mode="token_truncate", mask_truncated_completions=True,
        logging_steps=1, bf16=True, gradient_checkpointing=True, use_vllm=True, vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_gpu_mem, vllm_group_port=int(os.environ.get("VLLM_GROUP_PORT", 51216)),
        chat_template_kwargs={"enable_thinking": True}, report_to="none", model_init_kwargs={"dtype": "bfloat16"},
    )
    trainer = OPSDTrainer(
        model=args.model, reward_funcs=reward, args=config, train_dataset=dataset, rollout_func=rollout,
        peft_config=LoraConfig(r=args.lora_rank, lora_alpha=args.lora_rank, target_modules="all-linear", task_type="CAUSAL_LM"),
        teacher_prompt_ids=teacher_prompt_ids, state=state, clip=args.clip, log=log,
        chain_weight=args.chain_weight, answer_kl_weight=args.answer_kl_weight, think_end_id=tok.convert_tokens_to_ids("</think>"),
    )
    trainer.train()
    trainer.save_model(f"{args.out}/final")
    print("saved adapter to", f"{args.out}/final")


class OPSDTrainer(GRPOTrainer):
    """GRPOTrainer with the two distillation terms of the module docstring.

    `_generate_and_score_completions`: after TRL has generated the student's completions and computed the
    student's log-probs on them (`old_per_token_logps`), the teacher -- the same network with the adapter
    disabled, i.e. the initial policy, prompted with the privileged hint -- scores the same tokens;
    chain_weight * clip(log p_T - log p_S) on the chain tokens (0 on the answer tokens) replaces the (B,)
    reward-based advantages with a (B, T) tensor, which TRL's loss accepts.  The teacher prompt and the
    answer mask ride along in the batch dict (TRL shuffles and splits every tensor in it in unison).
    `_compute_loss`: TRL's loss plus answer_kl_weight * mean over answer positions of KL(p_T || p_S) from
    both models' full logits.
    """

    def __init__(self, *a, teacher_prompt_ids, state, clip, log, chain_weight, answer_kl_weight, think_end_id, **kw):
        super().__init__(*a, **kw)
        # (`state` and `log` would shadow transformers' TrainerState and Trainer.log)
        self.teacher_prompt_ids, self.opsd_state, self.clip, self.log_file = teacher_prompt_ids, state, clip, log
        self.chain_weight, self.answer_kl_weight, self.think_end_id = chain_weight, answer_kl_weight, think_end_id
        self.answer_kls = []  # per micro-batch, since the last generation step

    def _generate_and_score_completions(self, inputs):
        out = super()._generate_and_score_completions(inputs)
        comp, cmask = out["completion_ids"], out["completion_mask"]
        pad_id = self.processing_class.pad_token_id
        tp = pad([torch.tensor(self.teacher_prompt_ids[x["task_id"]], device=comp.device) for x in inputs], padding_value=pad_id, padding_side="left")
        input_ids, attn = torch.cat([tp, comp], 1), torch.cat([(tp != pad_id).int(), cmask], 1)
        with torch.no_grad(), self.accelerator.unwrap_model(self.model).disable_adapter():
            teacher_logps, _, _ = self._get_per_token_logps_and_entropies(self.model, input_ids, attn, comp.size(1), batch_size=self.args.per_device_train_batch_size)
        gap = (teacher_logps - out["old_per_token_logps"]) * cmask
        # answer tokens = everything after the first </think>: handled by the KL term in _compute_loss; the chain gets
        # the sampled-token term at chain_weight
        is_end = (comp == self.think_end_id).int()
        answer = (is_end.cumsum(1) - is_end).clamp(max=1).float() * cmask  # 1 strictly after the first </think>
        out["advantages"] = gap.clamp(-self.clip, self.clip) * self.chain_weight * (cmask - answer)
        out["teacher_prompt_ids"], out["teacher_prompt_mask"], out["answer_mask"] = tp, (tp != pad_id).int(), answer
        n, na = cmask.sum().clamp(min=1), answer.sum().clamp(min=1)
        self.opsd_state["pending"].update({"teacher_minus_student_logp": (gap.sum() / n).item(), "mean_abs_gap": (gap.abs().sum() / n).item(),
                                           "answer_tokens_per_seq": (na / comp.size(0)).item(), "answer_gap": ((gap * answer).sum() / na).item(),
                                           "answer_abs_gap": ((gap.abs() * answer).sum() / na).item(),
                                           "answer_kl_prev_step": sum(self.answer_kls) / len(self.answer_kls) if self.answer_kls else None,
                                      "clipped_fraction": ((gap.abs() > self.clip).float() * cmask).sum().item() / n.item(),
                                      "masked_completions": int((cmask.sum(1) == 0).sum().item())})
        self.log_file.write(json.dumps(self.opsd_state["pending"]) + "\n"); self.log_file.flush()
        self.answer_kls = []
        return out

    def _compute_loss(self, model, inputs):
        loss = super()._compute_loss(model, inputs)  # the chain term (advantages are zero on the answer tokens)
        answer = inputs["answer_mask"].bool()
        if not answer.any():
            return loss
        comp, cmask = inputs["completion_ids"], inputs["completion_mask"]
        T = comp.size(1)

        def logits_on_completion(prompt_ids, prompt_mask):  # (B, T, V) next-token logits for the completion positions, as TRL computes them
            ids, attn = torch.cat([prompt_ids, comp], 1), torch.cat([prompt_mask, cmask], 1)
            return model(input_ids=ids, attention_mask=attn, logits_to_keep=T + 1, use_cache=False).logits[:, :-1] / self.temperature

        student = logits_on_completion(inputs["prompt_ids"], inputs["prompt_mask"])[answer].float()  # (N_answer, V)
        with torch.no_grad(), self.accelerator.unwrap_model(model).disable_adapter():
            teacher = logits_on_completion(inputs["teacher_prompt_ids"], inputs["teacher_prompt_mask"])[answer].float()
        log_pt, log_ps = torch.log_softmax(teacher, -1), torch.log_softmax(student, -1)
        kl = (log_pt.exp() * (log_pt - log_ps)).sum(-1).mean()
        self.answer_kls.append(kl.item())
        return loss + self.answer_kl_weight * kl


if __name__ == "__main__":
    main()
