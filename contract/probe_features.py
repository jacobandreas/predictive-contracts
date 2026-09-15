"""Mean-pooled final-layer representations of each problem statement, for the success/hack probe.

    python -m contract.probe_features --out results/probe/features_base.npz
    python -m contract.probe_features --lora runs/grpo_modify_tests_neutral_s1/checkpoint-200 --out results/probe/features_neutral_s1.npz

For every problem in the given data files, build the neutral-prompt chat (the same system prompt
and user turn the seeded runs see, without any commitment questions), run the (optionally
LoRA-adapted) model once, and average the last hidden layer over the tokens of the user message.
The output npz has `ids` (task ids as strings) and `X` (n_tasks x hidden_size, float32).
Runs on one GPU; the probe itself (contract/probe.py) runs on a laptop.
"""
import argparse

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from contract.envs.leetcode import LeetCodeEnv
from contract.prompts import CODE_FORMAT_INSTRUCTION, PRECOMMIT_SYSTEM_PROMPT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--lora", default=None, help="adapter checkpoint directory (omit for the base model)")
    p.add_argument("--data", nargs="+", default=["data/leetcode/leetcode_test_medhard.jsonl", "data/leetcode/leetcode_train_medhard_filtered.jsonl"])
    p.add_argument("--hint", default="modify_tests")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    if args.lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora)
    model.eval()

    tasks = [t for path in args.data for t in LeetCodeEnv(path=path, hint=args.hint).tasks()]
    ids, feats = [], []
    for i in range(0, len(tasks), args.batch_size):
        batch = tasks[i : i + args.batch_size]
        texts, spans = [], []
        for t in batch:
            user = t.messages[-1]["content"] + "\n\n" + CODE_FORMAT_INSTRUCTION
            messages = [{"role": "system", "content": PRECOMMIT_SYSTEM_PROMPT}, {"role": "user", "content": user}]
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            start = text.index(user)
            texts.append(text)
            spans.append((start, start + len(user)))
        enc = tok(texts, return_tensors="pt", padding=True, return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")
        with torch.no_grad():
            h = model(**{k: v.cuda() for k, v in enc.items()}, output_hidden_states=True).hidden_states[-1]  # (B, T, d)
        for b, (s, e) in enumerate(spans):
            # tokens whose character span lies inside the user message (padding tokens have offsets (0, 0))
            inside = (offsets[b, :, 0] >= s) & (offsets[b, :, 1] <= e) & (offsets[b, :, 1] > offsets[b, :, 0])
            feats.append(h[b][inside.cuda()].float().mean(0).cpu().numpy())
            ids.append(batch[b].id)
        print(f"{len(ids)}/{len(tasks)}", flush=True)
    np.savez(args.out, ids=np.array(ids), X=np.stack(feats).astype(np.float32))
    print("wrote", args.out, np.stack(feats).shape)


if __name__ == "__main__":
    main()
