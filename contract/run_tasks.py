"""Part 1, steps 1-2: run a model on an environment, optionally retrying after failures.

    python -m contract.run_tasks --env leetcode --hint overwrite_tests --n 10 --max-attempts 1 --out results/x.jsonl

With --max-attempts K > 1, every attempt that does not earn the environment's
reward is followed by the RETRY_MESSAGE user turn and another attempt, so the
transcript grows: [system, user, assistant, user(retry), assistant, ...].

With --precommit yesno|prob (Part 2), the first turn shows the problem and then asks one
question per behavior in the environment's list; the model answers (yes/no, or a probability
that each will hold), that answer stays in the transcript as an assistant turn, and a
"Now solve the problem." user turn precedes the first attempt.  The parsed answers are
stored as `precommit` on the transcript so they can be compared with the behaviors the
scorer actually observed.

Each output line is one transcript:
  {task_id, meta, messages, attempts: [score dicts], final: score of last attempt, n_attempts,
   precommit: {mode, raw, answers: {behavior: bool | float | None}} (only with --precommit)}
"""
import argparse
import json
import os
import random
import re

from contract.envs import ENVS
from contract.llm import LLM
from contract.prompts import (ASK_AFTER_PROBLEM_FOLLOWUP, ASK_AFTER_PROBLEM_PROMPT, ASK_BEFORE_HACKING_PROMPT, ASK_SUCCESS_PROBABILITY_PROMPT,
                               CODE_FORMAT_INSTRUCTION, COMMIT_THINK_BUDGET_STOP, PRECOMMIT_SYSTEM_PROMPT, RETRY_MESSAGE, SOLVE_MESSAGE, commit_messages)


def parse_precommit(text, mode, n):
    """Map numbered answer lines ("3. yes" / "3. 0.7") to a list of n values; None where missing/unparsable."""
    values = [None] * n
    for m in re.finditer(r"^\s*(\d+)[.):]\s*\**\s*([A-Za-z]+|[01](?:\.\d+)?|\.\d+)", text, re.MULTILINE):
        i, ans = int(m.group(1)) - 1, m.group(2).lower()
        if not 0 <= i < n:
            continue
        if mode == "yesno":
            values[i] = True if ans.startswith("y") else False if ans.startswith("n") else None
        else:
            try:
                values[i] = float(ans)
            except ValueError:
                pass
    return values


def parse_yesno_first_word(text):
    """First alphabetic token of a free-text answer, classified yes/no/None (countdown_code's self-report parsing)."""
    m = re.match(r"\s*\**\s*([A-Za-z]+)", text)
    if not m:
        return None
    word = m.group(1).lower()
    return True if word.startswith("y") else False if word.startswith("n") else None


def parse_stated_probability(text):
    """First N% in a free-text answer, as a float in [0, 1]; None if there isn't one."""
    m = re.search(r"(\d{1,3}(?:\.\d+)?)\s*%", text)
    if not m:
        return None
    v = float(m.group(1)) / 100
    return v if 0 <= v <= 1 else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="leetcode")
    p.add_argument("--hint", default="overwrite_tests")
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint (default: $LLM_BASE_URL or the local vLLM server); "
                                                      "e.g. https://openrouter.ai/api/v1 (api key from $OPENROUTER_API_KEY / $LLM_API_KEY)")
    p.add_argument("--reasoning-model", action="store_true", help="target is an OpenAI-style reasoning model (o1/o3/o4-...): "
                                                                    "drop custom temperature/top_p and the Qwen3 thinking-template kwarg, which such models reject")
    p.add_argument("--concurrency", type=int, default=64, help="parallel requests to the model endpoint")
    p.add_argument("--n", type=int, default=10, help="samples per task")
    p.add_argument("--max-attempts", type=int, default=1)
    p.add_argument("--precommit", choices=["none", "yesno", "prob"], default="none", help="Part 2 pre-commitment turn")
    p.add_argument("--ask-before-hacking", action="store_true", help="countdown_code-style replication: one extra yes/no turn before the task asking whether the "
                                                                      "model would be tempted to tamper with grading; system prompt and task prompt are otherwise "
                                                                      "byte-identical to the no-ask baseline (contrast with --precommit, which asks about many "
                                                                      "behaviors under a neutral system prompt). Mutually exclusive with the other ask-* flags and --precommit.")
    p.add_argument("--ask-with-problem", action="store_true", help="countdown_code ask_with_problem replication: the task and a yes/no hacking-temptation question "
                                                                     "about THIS problem are shown together in turn 1 (model must not solve yet), then a generic "
                                                                     "follow-up turn asks it to solve. Mutually exclusive with the other ask-* flags and --precommit.")
    p.add_argument("--ask-success-probability", action="store_true", help="countdown_code ask_success_probability replication: same two-turn structure as "
                                                                           "--ask-with-problem, but turn 1 asks for a stated probability of success instead of a "
                                                                           "hacking yes/no. Mutually exclusive with the other ask-* flags and --precommit.")
    p.add_argument("--neutral-system-prompt", action="store_true", help="no-commitment control with the Part 2 system prompt")
    p.add_argument("--decoupled", action="store_true", help="with --precommit: the commitment is its own conversation (generated without thinking, "
                                                              "as in train_grpo --decoupled) and the attempt runs on the neutral prompt without seeing it")
    p.add_argument("--statements", choices=list(ENVS["leetcode"].statement_sets), default="observable", help="which statements the commitment asks about (env.statement_sets)")
    p.add_argument("--thinking", action="store_true")
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--commit-thinking", action="store_true", help="with --decoupled: the commitment thinks under --commit-think-budget (default --think-budget / 4) "
                                                                    "with the reasoning note in its prompt, and answers in --commit-max-tokens")
    p.add_argument("--commit-think-budget", type=int, default=None)
    p.add_argument("--commit-variant", default="v0", help="framing of the thinking commitment prompt (contract.prompts.COMMIT_VARIANTS)")
    p.add_argument("--commit-max-tokens", type=int, default=128)
    p.add_argument("--tokenizer", default="Qwen/Qwen3-4B", help="tokenizer for the thinking-budget continuation prompt (the base model; --model may be a served adapter alias)")
    p.add_argument("--think-budget", type=int, default=None, help="thinking mode: force-close the <think> block after this many tokens (Qwen3 thinking-budget trick); --max-tokens then bounds the answer")
    p.add_argument("--reasoning-effort", default=None, help="OpenRouter only: caps reasoning length via the provider's own reasoning.effort field (e.g. low/medium/xhigh, model-dependent) "
                                                              "instead of the Qwen3 thinking-budget trick, which OpenRouter-hosted models don't support")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--limit", type=int, default=None, help="only the first N tasks (debugging)")
    p.add_argument("--data", default=None, help="problem file (default: the env's test split)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    assert sum([args.ask_before_hacking, args.ask_with_problem, args.ask_success_probability, args.precommit != "none"]) <= 1, \
        "--ask-before-hacking, --ask-with-problem, --ask-success-probability and --precommit are mutually exclusive"

    random.seed(args.seed)
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
    env = ENVS[args.env](hint=args.hint, workers=workers, limit=args.limit, **({"path": args.data} if args.data else {}))
    llm = LLM(model=args.model, base_url=args.base_url, reasoning_model=args.reasoning_model, workers=args.concurrency,
              thinking=args.thinking, temperature=args.temperature, max_tokens=args.max_tokens, think_budget=args.think_budget, tokenizer=args.tokenizer,
              reasoning_effort=args.reasoning_effort)

    tasks = env.tasks()
    # One transcript per (task, sample).  All n samples share the same prefix at attempt 0.
    transcripts = []
    for t in tasks:
        for k in range(args.n):
            transcripts.append({"task_id": t.id, "sample": k, "meta": t.meta, "messages": list(t.messages), "attempts": []})
    task_by_id = {t.id: t for t in tasks}

    if args.neutral_system_prompt and args.precommit == "none":
        for tr in transcripts:
            tr["messages"][0] = {"role": "system", "content": PRECOMMIT_SYSTEM_PROMPT}
            tr["messages"][-1] = {"role": "user", "content": tr["messages"][-1]["content"] + "\n\n" + CODE_FORMAT_INSTRUCTION}

    if args.ask_before_hacking:
        assert args.precommit == "none", "--ask-before-hacking and --precommit are mutually exclusive"
        for tr in transcripts:
            tr["_task_prompt"] = tr["messages"][-1]["content"]  # unchanged task prompt, asked for in turn 2
            tr["messages"] = [tr["messages"][0], {"role": "user", "content": ASK_BEFORE_HACKING_PROMPT}]
        outs = llm.chat_many([tr["messages"] for tr in transcripts], n=1, desc="ask-before-hacking")
        for tr, o in zip(transcripts, outs):
            tr["messages"].append({"role": "assistant", "content": o[0]["content"]})
            tr["ask_before_hacking"] = {"raw": o[0]["content"], "self_report": parse_yesno_first_word(o[0]["content"])}
            tr["messages"].append({"role": "user", "content": tr.pop("_task_prompt")})
        n_parsed = sum(tr["ask_before_hacking"]["self_report"] is not None for tr in transcripts)
        print(f"ask-before-hacking: {n_parsed}/{len(transcripts)} self-reports parsed", flush=True)

    if args.ask_with_problem or args.ask_success_probability:
        # Task + question shown together in one user turn; the model answers without solving, then a generic
        # follow-up turn (not repeating the task) asks it to solve.
        question = ASK_AFTER_PROBLEM_PROMPT if args.ask_with_problem else ASK_SUCCESS_PROBABILITY_PROMPT
        key = "ask_with_problem" if args.ask_with_problem else "ask_success_probability"
        for tr in transcripts:
            tr["messages"][-1] = {"role": "user", "content": tr["messages"][-1]["content"] + "\n\n" + question}
        outs = llm.chat_many([tr["messages"] for tr in transcripts], n=1, desc=key)
        for tr, o in zip(transcripts, outs):
            tr["messages"].append({"role": "assistant", "content": o[0]["content"]})
            tr[key] = {"raw": o[0]["content"]}
            if args.ask_with_problem:
                tr[key]["self_report"] = parse_yesno_first_word(o[0]["content"])
            else:
                tr[key]["stated_probability"] = parse_stated_probability(o[0]["content"])
            tr["messages"].append({"role": "user", "content": ASK_AFTER_PROBLEM_FOLLOWUP})
        parse_key = "self_report" if args.ask_with_problem else "stated_probability"
        n_parsed = sum(tr[key][parse_key] is not None for tr in transcripts)
        print(f"{key}: {n_parsed}/{len(transcripts)} answers parsed", flush=True)

    if args.precommit != "none":
        # Ask about each behavior first; keep the model's answer in the transcript, then ask for the solution.
        names = env.statement_sets[args.statements]
        questions = "\n".join(f"{i + 1}. {env.behavior_questions[b]}" for i, b in enumerate(names))
        for tr in transcripts:
            tr["messages"] = commit_messages(tr["messages"][-1]["content"], args.precommit, questions, reason=args.commit_thinking, variant=args.commit_variant)
        if args.commit_thinking:  # the commitment reasons under its own (shorter) budget and answers briefly
            budget = args.commit_think_budget or args.think_budget // 4
            outs = llm.chat_many([tr["messages"] for tr in transcripts], n=1, desc="pre-commitment", thinking=True, think_budget=budget, max_tokens=args.commit_max_tokens, stop_text=COMMIT_THINK_BUDGET_STOP)
        else:
            outs = llm.chat_many([tr["messages"] for tr in transcripts], n=1, desc="pre-commitment", thinking=False if args.decoupled else None)
        for tr, o in zip(transcripts, outs):
            # The model often appends code after its answers despite being told not to; keep only the
            # answers in the persisted commitment turn (the full text is kept in precommit["raw"]).
            tr["messages"].append({"role": "assistant", "content": o[0]["content"].split("```")[0].rstrip()})
            tr["precommit"] = {"mode": args.precommit, "statements": args.statements, "raw": o[0]["content"],
                               "answers": dict(zip(names, parse_precommit(o[0]["content"], args.precommit, len(names))))}
            if args.decoupled:  # the commitment conversation is kept aside; the attempt is a fresh neutral-prompt conversation
                tr["commit_messages"] = tr["messages"]
                tr["messages"] = [{"role": "system", "content": PRECOMMIT_SYSTEM_PROMPT},
                                  {"role": "user", "content": task_by_id[tr["task_id"]].messages[-1]["content"] + "\n\n" + CODE_FORMAT_INSTRUCTION}]
            else:
                tr["messages"].append({"role": "user", "content": SOLVE_MESSAGE})
        parsed = sum(all(v is not None for v in tr["precommit"]["answers"].values()) for tr in transcripts)
        print(f"pre-commitment: {parsed}/{len(transcripts)} fully parsed", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n_written = 0
    with open(args.out, "w") as out_f:
        live = list(range(len(transcripts)))
        for attempt in range(args.max_attempts):
            print(f"attempt {attempt + 1}: {len(live)} transcripts", flush=True)
            still_live = []

            def on_response(j, o, attempt=attempt, still_live=still_live):
                # Scores and writes one transcript as soon as ITS response arrives, rather than waiting for
                # the rest of the (possibly large) attempt batch -- so output is visible and partial on disk
                # throughout a long run, not only at the end.
                nonlocal n_written
                i = live[j]
                tr, r = transcripts[i], o[0]
                s = env.score(task_by_id[tr["task_id"]], r["content"])
                tr["messages"].append({"role": "assistant", "content": r["content"]})
                tr["attempts"].append({**s, "reasoning": r["reasoning"], "finish_reason": r["finish_reason"], "think_forced": r.get("think_forced", False)})
                if not s["success"] and attempt + 1 < args.max_attempts:
                    tr["messages"].append({"role": "user", "content": RETRY_MESSAGE})
                    still_live.append(i)
                else:
                    tr["final"] = tr["attempts"][-1]
                    tr["n_attempts"] = len(tr["attempts"])
                    out_f.write(json.dumps(tr) + "\n")
                    out_f.flush()
                    n_written += 1

            llm.chat_many([transcripts[i]["messages"] for i in live], n=1, desc=f"attempt {attempt + 1}", callback=on_response)
            live = still_live
            n_ok = sum(tr["attempts"][-1]["success"] for tr in transcripts if len(tr["attempts"]) == attempt + 1)
            print(f"  reward earned on this attempt: {n_ok}", flush=True)
            if not live:
                break
    print(f"wrote {n_written} transcripts to {args.out}")


if __name__ == "__main__":
    main()
