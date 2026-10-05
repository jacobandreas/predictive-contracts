"""Mechanics check for the OPSD answer-token KL: one forward pass of the base model, no training.

    python -m contract.opsd_debug_kl --targets results/probe/base_think4k_train_targets.json

For a few training problems, build the student and teacher prompts exactly as opsd_commit.py does, append a
fixed commitment completion (a short chain, the budget-stop sentence, "1. 0.75\\n2. 0.85"), and at every
answer position print the token the completion has there, the student's and the teacher's top-3 next-token
predictions, and KL(p_T || p_S) -- with the same logits_to_keep / [:, :-1] alignment as the trainer.
"""
import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from contract.envs.leetcode import LeetCodeEnv
from contract.commitment.prompts import COMMIT_THINK_BUDGET_STOP, OPSD_FACTS, OPSD_TEACHER_HINT, commit_messages


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--data", default="data/leetcode/leetcode_train_medhard_filtered.jsonl")
    p.add_argument("--targets", required=True)
    p.add_argument("--n-problems", type=int, default=3)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()
    env = LeetCodeEnv(path=args.data, hint="modify_tests")
    names = env.statement_sets["hack_success"]
    questions = "\n".join(f"{i + 1}. {env.behavior_questions[b]}" for i, b in enumerate(names))
    targets = json.load(open(args.targets))
    think_end = tok.convert_tokens_to_ids("</think>")
    completion_text = f"<think>\nLet me think about how I usually do on problems like this.\n{COMMIT_THINK_BUDGET_STOP}\n</think>\n\n1. 0.75\n2. 0.85<|im_end|>"
    comp = tok.encode(completion_text, add_special_tokens=False)
    T = len(comp)
    is_end = [i for i, t in enumerate(comp) if t == think_end]
    answer_pos = list(range(is_end[0] + 1, T))

    for t in [t for t in env.tasks() if t.id in targets][:args.n_problems]:
        student = commit_messages(t.messages[-1]["content"], "prob", questions, reason=True)
        facts = " and ".join(OPSD_FACTS[b].format(p=targets[t.id][b]) for b in names)
        lines = "\n".join(f"{i + 1}. {targets[t.id][b]:.2f}" for i, b in enumerate(names))
        teacher = [student[0], {"role": "user", "content": student[1]["content"] + OPSD_TEACHER_HINT.format(facts=facts, lines=lines)}]
        print(f"\n===== problem {t.id}: targets {lines!r}")
        logps = {}
        for role, msgs in (("student", student), ("teacher", teacher)):
            prompt = tok.apply_chat_template(msgs, tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=True)
            ids = torch.tensor([prompt + comp], device="cuda")
            with torch.no_grad():
                logits = model(input_ids=ids, logits_to_keep=T + 1, use_cache=False).logits[:, :-1].float()  # (1, T, V): logits[t] predicts comp[t]
            logps[role] = torch.log_softmax(logits[0], -1)
        for pos in answer_pos:
            tgt = tok.decode([comp[pos]])
            row = f"  pos {pos:3d} token {tgt!r:8}"
            for role in ("student", "teacher"):
                top = logps[role][pos].topk(3)
                row += f" | {role} top3 " + ", ".join(f"{tok.decode([i])!r}:{v.exp():.2f}" for v, i in zip(top.values, top.indices))
            kl = (logps["teacher"][pos].exp() * (logps["teacher"][pos] - logps["student"][pos])).sum()
            print(row + f" | KL(T||S) {kl:.3f}")


if __name__ == "__main__":
    main()
