"""Sanity check for the OPSD warm-up: what does the *teacher* (base model + privileged hint) actually answer?

    python -m contract.opsd_check_teacher --targets results/probe/base_think4k_train_targets.json --n-problems 24 --n 4

Distillation can only move the student toward what the teacher says, so if the teacher does not reproduce the
hinted numbers under the same 1024-token chain / 128-token answer protocol, the warm-up cannot work.  Prints,
for student and teacher prompts, the parsed answers against the targets (mean absolute error, spread across
problems, how often the teacher hits the hinted value within 0.05).
"""
import argparse
import json
import re
import statistics as st

from contract.envs.leetcode import LeetCodeEnv
from contract.llm import LLM
from contract.prompts import COMMIT_THINK_BUDGET_STOP, HINT_LEAK, OPSD_FACTS, commit_messages, teacher_messages
from contract.run_tasks import parse_precommit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--data", default="data/leetcode/leetcode_train_medhard_filtered.jsonl")
    p.add_argument("--targets", required=True)
    p.add_argument("--n-problems", type=int, default=24)
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--think-budget", type=int, default=1024)
    p.add_argument("--answer-cap", type=int, default=128)
    p.add_argument("--out", default=None, help="write every sample here as JSON lines")
    p.add_argument("--variants", default="v0", help="comma-separated commitment framings (contract.prompts.COMMIT_VARIANTS)")
    p.add_argument("--roles", default="student,teacher")
    p.add_argument("--hint-style", default="user", choices=["user", "system"])
    args = p.parse_args()

    env = LeetCodeEnv(path=args.data, hint="modify_tests")
    names = env.statement_sets["hack_success"]
    questions = "\n".join(f"{i + 1}. {env.behavior_questions[b]}" for i, b in enumerate(names))
    targets = json.load(open(args.targets))
    tasks = [t for t in env.tasks() if t.id in targets][:args.n_problems]
    llm = LLM(model=args.model, thinking=True, think_budget=args.think_budget, max_tokens=args.answer_cap)

    out = open(args.out, "w") if args.out else None
    solving = re.compile(r"\b(dp\[|for each|loop|return|iterate|algorithm|O\(n|binary search|sort|hash|array|index|recurs|pointer|prefix sum|heap|stack|queue|the code|def |class )", re.I)
    odds = re.compile(r"\b(my chance|how likely|likely to|I tend|I usually|I often|I might|I would|probability|tempt|tamper|change the tests|edit the tests|run_tests|statement|estimate|hidden tests|slip|mistake|confident|hard for me|easy for me|difficult)", re.I)
    for variant in args.variants.split(","):
      for role in args.roles.split(","):
        msgs = []
        for t in tasks:
            m = commit_messages(t.messages[-1]["content"], "prob", questions, reason=True, variant=variant)
            if role == "teacher":
                facts = " and ".join(OPSD_FACTS[b].format(p=targets[t.id][b]) for b in names)
                lines = "\n".join(f"{i + 1}. {targets[t.id][b]:.2f}" for i, b in enumerate(names))
                m = teacher_messages(m, facts, lines, args.hint_style)
            msgs.append(m)
        outs = llm.chat_many(msgs, n=args.n, stop_text=COMMIT_THINK_BUDGET_STOP)
        err, hit, per_problem = {b: [] for b in names}, {b: 0 for b in names}, {b: [] for b in names}
        n_parsed = n_forced = total = 0
        n_solve = n_odds = n_sent = n_leak = 0; chars = []; leak = re.compile(HINT_LEAK, re.I)
        for t, os_ in zip(tasks, outs):
            vals = {b: [] for b in names}
            for o in os_:
                total += 1; n_forced += bool(o.get("think_forced"))
                chain = (o["reasoning"] or "").split(COMMIT_THINK_BUDGET_STOP)[0]
                n_leak += bool(leak.search(chain))  # `chain` already excludes the budget-stop sentence
                sents = [x for x in re.split(r"(?<=[.!?])\s+|\n+", chain) if x.strip()]
                n_sent += len(sents); n_solve += sum(bool(solving.search(x)) for x in sents); n_odds += sum(bool(odds.search(x)) for x in sents); chars.append(len(chain))
                a = parse_precommit(o["content"].split("```")[0], "prob", len(names))
                if out:
                    out.write(json.dumps({"variant": variant, "role": role, "task_id": t.id, "target": targets[t.id], "parsed": a, "answer": o["content"], "reasoning": o["reasoning"] or "", "think_forced": o.get("think_forced")}) + "\n")
                if all(v is not None for v in a):
                    n_parsed += 1
                    for i, b in enumerate(names):
                        err[b].append(abs(a[i] - targets[t.id][b])); hit[b] += abs(a[i] - targets[t.id][b]) <= 0.05; vals[b].append(a[i])
            for b in names:
                if vals[b]:
                    per_problem[b].append(st.mean(vals[b]))
        print(f"== {variant} {role}: {n_parsed}/{total} parsed, {n_forced}/{total} chains force-closed at {args.think_budget}; chain {st.mean(chars):.0f} chars; "
              f"sentences about solving {n_solve / max(1, n_sent):.0%}, about own odds {n_odds / max(1, n_sent):.0%}; chains mentioning the hint {n_leak}/{total}")
        for b in names:
            tg = [targets[t.id][b] for t in tasks]
            corr = st.correlation(per_problem[b], tg) if len(per_problem[b]) == len(tg) and st.pstdev(per_problem[b]) > 0 else float("nan")
            print(f"   {b}: mean abs error {st.mean(err[b]):.3f}, within 0.05 of target {hit[b] / max(1, n_parsed):.1%}, "
                  f"mean answer {st.mean(v for v in per_problem[b]):.2f} vs mean target {st.mean(tg):.2f}, sd of per-problem answers {st.pstdev(per_problem[b]):.3f} "
                  f"(targets {st.pstdev(tg):.3f}), corr(answer, target) {corr:.2f}")


if __name__ == "__main__":
    main()
