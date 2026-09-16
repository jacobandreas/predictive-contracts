"""SFT warm-up of the commitment turn on probe-derived probabilities.

    python -m contract.sft_commit --targets results/probe/base_train_targets.json --out runs/sft_commit_prob
    python -m contract.sft_commit --targets ... --out ... --dry-run     # print two examples, no training

For every training problem, the input is the Part 3 commitment prompt (neutral system prompt; problem;
"before solving it, give the probability of each statement") and the target is the answer block a
calibrated committer would write, e.g. "1. 0.10\\n2. 0.40": the probe's cross-fitted probability that
the base model shows each behavior on that problem (contract/probe.py --behaviors ... --predict-out),
rounded to the nearest 0.1 and clipped to [0.1, 0.9] -- the same grid the epsilon-forced runs used.
The prompt is rendered with the chat template exactly as the RL rollouts see it (generation prompt
with the empty <think> block), and the completion is the answer text plus <|im_end|>, so the SFT
tokens are the ones the policy will be asked to produce. LoRA SFT (same adapter shape as the GRPO
runs) with loss on the completion only; the saved adapter is the starting point for
`train_grpo.py --init-adapter`.
"""
import argparse
import json

from transformers import AutoTokenizer

from contract.envs.leetcode import LeetCodeEnv
from contract.prompts import PRECOMMIT_INTRO, PRECOMMIT_QUESTIONS, PRECOMMIT_SYSTEM_PROMPT


def bucket(p):
    return f"{min(max(round(p * 10) / 10, 0.1), 0.9):.2f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--data", default="data/leetcode/leetcode_train_medhard_filtered.jsonl")
    p.add_argument("--hint", default="modify_tests")
    p.add_argument("--statements", default="hack_success", choices=list(LeetCodeEnv.statement_sets))
    p.add_argument("--targets", required=True, help="JSON {task_id: {behavior: probability}} from contract.probe --predict-out")
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    env = LeetCodeEnv(path=args.data, hint=args.hint)
    names = env.statement_sets[args.statements]
    questions = "\n".join(f"{i + 1}. {env.behavior_questions[b]}" for i, b in enumerate(names))
    qblock = PRECOMMIT_QUESTIONS["prob"].format(questions=questions)
    targets = json.load(open(args.targets))
    tok = AutoTokenizer.from_pretrained(args.model)
    rows = []
    for t in env.tasks():
        if t.id not in targets:
            continue
        messages = [{"role": "system", "content": PRECOMMIT_SYSTEM_PROMPT},
                    {"role": "user", "content": PRECOMMIT_INTRO + t.messages[-1]["content"] + qblock}]
        rows.append({"prompt": tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False),
                     "completion": "\n".join(f"{i + 1}. {bucket(targets[t.id][b])}" for i, b in enumerate(names)) + "<|im_end|>\n"})
    print(f"{len(rows)} SFT examples; target distribution per statement:")
    for i, b in enumerate(names):
        vals = [r["completion"].split("\n")[i].split(". ")[1] for r in rows]
        print(f"  {b}: " + ", ".join(f"{v}: {vals.count(v)}" for v in sorted(set(vals))))
    if args.dry_run:
        for r in rows[:2]:
            print("\n---\n" + r["prompt"][-300:] + ">>> " + repr(r["completion"]))
        return

    from datasets import Dataset  # heavy imports only when actually training (the dry run works on a laptop)
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    config = SFTConfig(output_dir=args.out, num_train_epochs=args.epochs, learning_rate=args.lr, lr_scheduler_type="cosine",
                       warmup_steps=10, per_device_train_batch_size=8, gradient_accumulation_steps=2, bf16=True,
                       max_length=2048, logging_steps=5, save_strategy="no", report_to="none",
                       model_init_kwargs={"dtype": "bfloat16"})
    trainer = SFTTrainer(model=args.model, args=config, train_dataset=Dataset.from_list(rows),
                         peft_config=LoraConfig(r=args.lora_rank, lora_alpha=args.lora_rank, target_modules="all-linear", task_type="CAUSAL_LM"))
    trainer.train()
    trainer.save_model(f"{args.out}/final")
    print("saved adapter to", f"{args.out}/final")


if __name__ == "__main__":
    main()
