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
from contract.prompts import CODE_FORMAT_INSTRUCTION, PRECOMMIT_INTRO, PRECOMMIT_QUESTIONS, PRECOMMIT_SYSTEM_PROMPT, RETRY_MESSAGE, SOLVE_MESSAGE


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="leetcode")
    p.add_argument("--hint", default="overwrite_tests")
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--n", type=int, default=10, help="samples per task")
    p.add_argument("--max-attempts", type=int, default=1)
    p.add_argument("--precommit", choices=["none", "yesno", "prob"], default="none", help="Part 2 pre-commitment turn")
    p.add_argument("--neutral-system-prompt", action="store_true", help="no-commitment control with the Part 2 system prompt")
    p.add_argument("--statements", choices=list(ENVS["leetcode"].statement_sets), default="observable", help="which statements the commitment asks about (env.statement_sets)")
    p.add_argument("--thinking", action="store_true")
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--think-budget", type=int, default=None, help="thinking mode: force-close the <think> block after this many tokens (Qwen3 thinking-budget trick); --max-tokens then bounds the answer")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--limit", type=int, default=None, help="only the first N tasks (debugging)")
    p.add_argument("--data", default=None, help="problem file (default: the env's test split)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    random.seed(args.seed)
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
    env = ENVS[args.env](hint=args.hint, workers=workers, limit=args.limit, **({"path": args.data} if args.data else {}))
    llm = LLM(model=args.model, thinking=args.thinking, temperature=args.temperature, max_tokens=args.max_tokens, think_budget=args.think_budget)

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

    if args.precommit != "none":
        # Ask about each behavior first; keep the model's answer in the transcript, then ask for the solution.
        names = env.statement_sets[args.statements]
        questions = "\n".join(f"{i + 1}. {env.behavior_questions[b]}" for i, b in enumerate(names))
        for tr in transcripts:
            tr["messages"][0] = {"role": "system", "content": PRECOMMIT_SYSTEM_PROMPT}
            tr["messages"][-1] = {"role": "user", "content": PRECOMMIT_INTRO + tr["messages"][-1]["content"]
                                  + PRECOMMIT_QUESTIONS[args.precommit].format(questions=questions)}
        outs = llm.chat_many([tr["messages"] for tr in transcripts], n=1)
        for tr, o in zip(transcripts, outs):
            # The model often appends code after its answers despite being told not to; keep only the
            # answers in the persisted commitment turn (the full text is kept in precommit["raw"]).
            tr["messages"].append({"role": "assistant", "content": o[0]["content"].split("```")[0].rstrip()})
            tr["messages"].append({"role": "user", "content": SOLVE_MESSAGE})
            tr["precommit"] = {"mode": args.precommit, "statements": args.statements, "raw": o[0]["content"],
                               "answers": dict(zip(names, parse_precommit(o[0]["content"], args.precommit, len(names))))}
        parsed = sum(all(v is not None for v in tr["precommit"]["answers"].values()) for tr in transcripts)
        print(f"pre-commitment: {parsed}/{len(transcripts)} fully parsed", flush=True)

    live = list(range(len(transcripts)))
    for attempt in range(args.max_attempts):
        print(f"attempt {attempt + 1}: {len(live)} transcripts", flush=True)
        outs = llm.chat_many([transcripts[i]["messages"] for i in live], n=1)
        responses = [o[0] for o in outs]
        scores = env.score_many([task_by_id[transcripts[i]["task_id"]] for i in live], [r["content"] for r in responses])
        still_live = []
        for i, r, s in zip(live, responses, scores):
            tr = transcripts[i]
            tr["messages"].append({"role": "assistant", "content": r["content"]})
            tr["attempts"].append({**s, "reasoning": r["reasoning"], "finish_reason": r["finish_reason"], "think_forced": r.get("think_forced", False)})
            if not s["success"] and attempt + 1 < args.max_attempts:
                tr["messages"].append({"role": "user", "content": RETRY_MESSAGE})
                still_live.append(i)
        live = still_live
        n_ok = sum(tr["attempts"][-1]["success"] for tr in transcripts if len(tr["attempts"]) == attempt + 1)
        print(f"  reward earned on this attempt: {n_ok}", flush=True)
        if not live:
            break

    for tr in transcripts:
        tr["final"] = tr["attempts"][-1]
        tr["n_attempts"] = len(tr["attempts"])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for tr in transcripts:
            f.write(json.dumps(tr) + "\n")
    print(f"wrote {len(transcripts)} transcripts to {args.out}")


if __name__ == "__main__":
    main()
