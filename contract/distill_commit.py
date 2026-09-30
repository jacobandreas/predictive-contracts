"""Off-policy self-distillation warm-up of the reasoning commitment turn (the working replacement for opsd_commit.py).

    # 1. sample the hinted teacher (needs a vLLM server: scripts/serve_and_run.sbatch)
    python -m contract.distill_commit sample --targets results/probe/base_think4k_train_targets.json --variant v1 \\
        --out results/distill/teacher_traces_v1.jsonl
    # 2. SFT the student on the kept traces (scripts/train.sbatch)
    python -m contract.distill_commit train --traces results/distill/teacher_traces_v1.jsonl --variant v1 --out runs/distill_commit_v1

Teacher and student are the same base model with the same commitment framing (contract.prompts.COMMIT_VARIANTS[variant]);
the teacher's user turn additionally carries the probe's calibrated estimates for the problem (OPSD_TEACHER_HINT, ending
with the required answer lines).  The teacher thinks under the commitment budget and answers; we keep the samples whose
answers land within --tol of the targets, preferring chains that closed on their own over force-closed ones, and fine-tune
the student prompt (no hint) to produce the teacher's chain + answer verbatim.  Why off-policy: scoring the hinted teacher
on the *student's* trajectory (opsd_commit.py) fails here, because the teacher follows its hint only on its own trajectory
(results_part4.md); on its own samples it hits the hinted numbers ~100% of the time.
"""
import argparse
import json
import os
import random
import re

from transformers import AutoTokenizer

from contract.envs.leetcode import LeetCodeEnv
from contract.prompts import COMMIT_THINK_BUDGET_STOP, HINT_LEAK, OPSD_FACTS, commit_messages, teacher_messages
from contract.run_tasks import parse_precommit


def sample(args):
    from contract.llm import LLM
    env = LeetCodeEnv(path=args.data, hint=args.hint)
    names = env.statement_sets[args.statements]
    questions = "\n".join(f"{i + 1}. {env.behavior_questions[b]}" for i, b in enumerate(names))
    targets = json.load(open(args.targets))
    tasks = [t for t in env.tasks() if t.id in targets]
    llm = LLM(model=args.model, thinking=True, think_budget=args.think_budget, max_tokens=args.answer_cap)
    teacher_msgs, student_msgs = [], []
    for t in tasks:
        student = commit_messages(t.messages[-1]["content"], "prob", questions, reason=True, variant=args.variant)
        facts = " and ".join(OPSD_FACTS[b].format(p=targets[t.id][b]) for b in names)
        lines = "\n".join(f"{i + 1}. {targets[t.id][b]:.2f}" for i, b in enumerate(names))
        teacher_msgs.append(teacher_messages(student, facts, lines, args.hint_style))
        student_msgs.append(student)
    outs = llm.chat_many(teacher_msgs, n=args.n, stop_text=COMMIT_THINK_BUDGET_STOP)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    kept = n_hit = n_leak = n_unforced = 0
    leak = re.compile(HINT_LEAK, re.I)
    allf = open(args.out + ".all.jsonl", "w") if args.save_all else None
    with open(args.out, "w") as f:
        for t, student, os_ in zip(tasks, student_msgs, outs):
            good = []
            for o in os_:
                a = parse_precommit(o["content"].split("```")[0], "prob", len(names))
                if all(v is not None for v in a) and all(abs(a[i] - targets[t.id][b]) <= args.tol for i, b in enumerate(names)):
                    leaked = bool(leak.search(o["reasoning"] or ""))
                    if args.save_all:
                        allf.write(json.dumps({"task_id": t.id, "target": targets[t.id], "reasoning": o["reasoning"], "answer": o["content"], "think_forced": o.get("think_forced", False), "leaked": leaked}) + "\n")
                    if leaked:  # the chain talks about the hint instead of reasoning to the numbers
                        n_leak += 1
                    else:
                        good.append(o)
            n_hit += len(good)
            good.sort(key=lambda o: o.get("think_forced", False))  # chains that closed on their own first
            for o in good[:args.keep]:
                n_unforced += not o.get("think_forced", False)
                # the answer block only (the model sometimes runs on past the numbered lines)
                answer = "\n".join(l for l in o["content"].strip().split("\n") if l.strip())[:400]
                f.write(json.dumps({"task_id": t.id, "target": targets[t.id], "prompt": student, "reasoning": o["reasoning"], "answer": answer,
                                    "think_forced": o.get("think_forced", False)}) + "\n")
                kept += 1
    print(f"{len(tasks)} problems x {args.n} samples: {n_hit} within {args.tol} of the targets and not mentioning the hint ({n_leak} on target but "
          f"leaking the hint, dropped); kept {kept} on {len({json.loads(l)['task_id'] for l in open(args.out)})} problems ({n_unforced} closed their chain on their own) -> {args.out}")


def train(args):
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer
    tok = AutoTokenizer.from_pretrained(args.model)
    rows = []
    for r in map(json.loads, open(args.traces)):
        prompt = tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True, enable_thinking=True)
        rows.append({"prompt": prompt, "completion": f"<think>\n{r['reasoning'].strip()}\n</think>\n\n{r['answer']}<|im_end|>\n"})
    random.Random(args.seed).shuffle(rows)
    lens = [len(tok(r["completion"])["input_ids"]) for r in rows[:200]]
    print(f"{len(rows)} traces; completion length (first 200): mean {sum(lens) / len(lens):.0f} tokens, max {max(lens)}", flush=True)
    config = SFTConfig(output_dir=args.out, num_train_epochs=args.epochs, learning_rate=args.lr, lr_scheduler_type="cosine",
                       warmup_steps=10, per_device_train_batch_size=4, gradient_accumulation_steps=4, bf16=True,
                       max_length=4096, logging_steps=5, save_strategy="no", report_to="none", gradient_checkpointing=True,
                       model_init_kwargs={"dtype": "bfloat16"})
    trainer = SFTTrainer(model=args.model, args=config, train_dataset=Dataset.from_list(rows),
                         peft_config=LoraConfig(r=args.lora_rank, lora_alpha=args.lora_rank, target_modules="all-linear", task_type="CAUSAL_LM"))
    trainer.train()
    trainer.save_model(f"{args.out}/final")
    print("saved adapter to", f"{args.out}/final")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["sample", "train"])
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--data", default="data/leetcode/leetcode_train_medhard_filtered.jsonl")
    p.add_argument("--hint", default="modify_tests")
    p.add_argument("--statements", default="hack_success", choices=list(LeetCodeEnv.statement_sets))
    p.add_argument("--variant", default="v1", help="commitment framing (contract.prompts.COMMIT_VARIANTS); the same one train_grpo --commit-variant must use")
    p.add_argument("--targets", help="sample: JSON {task_id: {behavior: probability}} from contract.probe --predict-out")
    p.add_argument("--n", type=int, default=4, help="sample: teacher samples per problem")
    p.add_argument("--keep", type=int, default=2, help="sample: traces kept per problem")
    p.add_argument("--tol", type=float, default=0.05, help="sample: max |answer - target| per statement")
    p.add_argument("--hint-style", default="user", choices=["user", "system"], help="sample: where the teacher's hint goes (contract.prompts.teacher_messages)")
    p.add_argument("--save-all", action="store_true", help="sample: also write every on-target sample (leaking ones flagged) to <out>.all.jsonl, to tune the filter offline")
    p.add_argument("--think-budget", type=int, default=1024)
    p.add_argument("--answer-cap", type=int, default=128)
    p.add_argument("--traces", help="train: the sample stage's output")
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    (sample if args.stage == "sample" else train)(args)


if __name__ == "__main__":
    main()
