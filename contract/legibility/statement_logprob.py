"""Diagnostic for the warm-up: does the fine-tuned adapter put more probability on the inserted statements than the base
model does, given each statement's own training prefix?

    LORA=runs/legibility/distill_legible_insert3/final sbatch scripts/serve_and_run.sbatch \\
        venv/bin/python -m contract.legibility.statement_logprob --traces results/legibility/distill/teacher_traces_insert3.jsonl

For a sample of the insert-mode traces: log p(statement | prompt, chain up to the insertion point) under the adapter
("rl" on the server) and under the base model, from the completions endpoint's echoed log-probabilities.  A warm-up that
left this unchanged did not learn the statements at all (dilution); one that raised it a lot but never produces a
statement at evaluation time memorised them without generalising.
"""
import argparse
import json
import os
import random
from statistics import mean

from openai import OpenAI
from transformers import AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--traces", required=True)
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--base", default="Qwen/Qwen3-4B")
    p.add_argument("--adapter", default="rl")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    tok = AutoTokenizer.from_pretrained(args.base)
    client = OpenAI(base_url=os.environ["LLM_BASE_URL"], api_key="x")
    rows = [r for r in map(json.loads, open(args.traces)) if r["kind"] == "hack"]
    rows = random.Random(0).sample(rows, min(args.n, len(rows)))

    def total_logprob(model, text):
        r = client.completions.create(model=model, prompt=text, max_tokens=1, echo=True, logprobs=0, temperature=0)
        return sum(x for x in r.choices[0].logprobs.token_logprobs if x is not None)

    out = []
    for r in rows:
        prompt = tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True, enable_thinking=True)
        stmt = r["inserted"]
        prefix = prompt + "<think>\n" + r["reasoning"][:r["reasoning"].index(stmt)]
        n = len(tok(stmt, add_special_tokens=False)["input_ids"])
        lp = {m: total_logprob(m, prefix + stmt) - total_logprob(m, prefix) for m in (args.base, args.adapter)}
        out.append({"task_id": r["task_id"], "n_tokens": n, "base": lp[args.base], "adapter": lp[args.adapter]})
        print(f"task {r['task_id']}: {n} statement tokens; log p base {lp[args.base]:.1f} ({lp[args.base] / n:.2f}/tok), "
              f"adapter {lp[args.adapter]:.1f} ({lp[args.adapter] / n:.2f}/tok)", flush=True)
    print(f"MEAN per-token log p: base {mean(o['base'] / o['n_tokens'] for o in out):.3f}, adapter {mean(o['adapter'] / o['n_tokens'] for o in out):.3f}; "
          f"mean gain {mean((o['adapter'] - o['base']) / o['n_tokens'] for o in out):.3f} nats/token over {len(out)} statements")
    if args.out:
        json.dump(out, open(args.out, "w"))


if __name__ == "__main__":
    main()
