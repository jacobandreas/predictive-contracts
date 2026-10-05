"""The shared GRPO training core (TRL + LoRA + colocated vLLM), direction-agnostic.

A research direction (contract/commit, contract/legible) supplies two functions and a few extra flags:

    rollout(prompts, trainer) -> dict      generates the episodes itself (TRL's `rollout_func` hook) and hands back
                                           prompt_ids, completion_ids, logprobs, env_mask (0 on spliced tokens that
                                           must not be trained on) and any extra per-episode fields the reward needs
    reward(prompts, completions, task_id, **extra) -> list[float]

and this module does the rest: the common command-line flags, the GRPOConfig, the LoRA / warm-start adapter,
checkpoint resume (with the explicit adapter reload that transformers' resume skips), the run's log files, and
`generate_budgeted`, Qwen3's thinking-budget trick on top of vLLM.  Defaults are the Part 3/4 recipe: 16 problems x
16 rollouts per step, lr 7e-5, LoRA r=32, per-token importance weights (`token_truncate`), 200 steps.
"""
import glob
import json
import os

from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from contract.prompts import THINK_BUDGET_STOP


def add_common_args(p):
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--data", default="data/leetcode/leetcode_train_medhard_filtered.jsonl")
    p.add_argument("--hint", default="modify_tests")
    p.add_argument("--thinking", action="store_true", help="Qwen3 thinking on; with --think-budget the chain is cut at the budget")
    p.add_argument("--think-budget", type=int, default=None, help="thinking tokens before the chain is force-closed (then --max-completion-length for the answer)")
    p.add_argument("--num-prompts", type=int, default=16, help="problems per optimisation step")
    p.add_argument("--num-generations", type=int, default=16, help="rollouts per problem")
    p.add_argument("--max-completion-length", type=int, default=1536, help="answer tokens (per episode or, with a budget, after the chain)")
    p.add_argument("--max-prompt-length", type=int, default=1536, help="problems whose prompt is longer are dropped")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--save-steps", type=int, default=10)
    p.add_argument("--lr", type=float, default=7e-5)
    p.add_argument("--beta", type=float, default=1e-3, help="KL weight toward the initial policy (TRL adds a frozen 'ref' adapter)")
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--per-device-batch", type=int, default=2, help="sequences per backward pass")
    p.add_argument("--split-normalize", action="store_true", help="the reward function returns z-scored terms itself; TRL's group scaling is off")
    p.add_argument("--is-mode", default="token_truncate", help="TRL vllm_importance_sampling_mode (token_truncate; the default sequence_mask down-weights long completions)")
    p.add_argument("--mask-truncated", action="store_true", help="drop completions that hit the token cap from the loss (DAPO's overlong filtering)")
    p.add_argument("--vllm-gpu-mem", type=float, default=0.35, help="fraction of GPU memory for the colocated vLLM engine")
    p.add_argument("--vllm-sleep", action="store_true", help="vLLM offloads weights and frees its cache during the training step")
    p.add_argument("--init-adapter", default=None, help="start from this LoRA adapter instead of a fresh one")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", required=True)
    return p


def generate_budgeted(gen, tok, prompt_ids, num_generations, think_budget, answer_cap, stop_text=THINK_BUDGET_STOP):
    """Thinking with a token budget.  Phase 1 generates up to `think_budget` tokens; a completion whose <think>
    block is still open gets "\\n{stop_text}\\n</think>\\n\\n" spliced in (env_mask 0, logprob 0) and phase 2
    generates the answer for up to `answer_cap` tokens.  Completions that closed the block themselves but ran out
    of budget mid-answer also continue in phase 2 (no splice).  Returns, per output: completion ids, logprobs,
    env_mask, whether the block was force-closed, and the thinking length in tokens."""
    eos, think_end = tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("</think>")
    stop_ids = tok.encode("\n" + stop_text + "\n</think>\n\n", add_special_tokens=False)
    gen.max_completion_length = think_budget
    # `prompt_ids` already lists one prompt per output (TRL repeats each prompt num_generations times and its
    # vLLM wrapper de-duplicates), so generate() returns exactly len(prompt_ids) completions, aligned with it.
    _, comp, lps, _ = gen.generate(prompts=prompt_ids, images=None, num_generations=num_generations)
    completion_ids = [list(c) for c in comp]
    logprobs = [[lp[0] for lp in seq] for seq in lps]
    env_mask = [[1] * len(c) for c in completion_ids]
    forced, think_len = [], []
    for i, c in enumerate(completion_ids):
        closed = think_end in c
        think_len.append(c.index(think_end) if closed else len(c))
        forced.append(not closed)
        if not closed:
            completion_ids[i] += stop_ids; logprobs[i] += [0.0] * len(stop_ids); env_mask[i] += [0] * len(stop_ids)
    cont = [i for i, c in enumerate(completion_ids) if c[-1] != eos]  # everything that has not ended yet answers in phase 2
    if cont:
        gen.max_completion_length = answer_cap
        _, comp, lps, _ = gen.generate(prompts=[prompt_ids[i] + completion_ids[i] for i in cont], images=None, num_generations=1)
        for i, c, seq in zip(cont, comp, lps):
            completion_ids[i] += list(c); logprobs[i] += [lp[0] for lp in seq]; env_mask[i] += [1] * len(c)
    return completion_ids, logprobs, env_mask, [float(f) for f in forced], [float(n) for n in think_len]


def open_logs(out, names=("reward_log", "rollouts", "hack_examples")):
    """Append-mode log files, plus the step to resume from.  Lines logged after the latest checkpoint are dropped
    (a pre-empted job may have logged steps it never saved), so a resumed run's logs read as one run."""
    os.makedirs(out, exist_ok=True)
    ckpts = sorted(glob.glob(f"{out}/checkpoint-*"), key=lambda c: int(c.rsplit("-", 1)[1]))
    resume_step = int(ckpts[-1].rsplit("-", 1)[1]) if ckpts else 0
    files = {}
    for name in names:
        path = f"{out}/{name}.jsonl"
        if os.path.exists(path):
            lines = [l for l in open(path) if json.loads(l)["call"] <= resume_step]
            open(path, "w").writelines(lines)
        files[name] = open(path, "a")
    return files, ckpts, resume_step


def make_config(args, episodes_per_problem=1, extra_completion=0, scale_rewards=None):
    """GRPOConfig for the common recipe.  `episodes_per_problem` > 1 when a direction samples several conversations per
    problem per step (e.g. attempts + commitments); `extra_completion` is any token budget beyond the answer (a thinking
    budget, a commitment turn) so that TRL's padding and vLLM's context fit the longest episode."""
    budget = args.max_completion_length + 64 + extra_completion
    return GRPOConfig(
        output_dir=args.out,
        seed=args.seed,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        weight_decay=0.1,
        max_grad_norm=1.0,
        beta=args.beta,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.num_prompts * args.num_generations * episodes_per_problem // args.per_device_batch,
        num_generations=args.num_generations * episodes_per_problem,
        max_completion_length=budget,  # whole-episode budget; the rollouts set per-call lengths themselves
        vllm_max_model_length=args.max_prompt_length + 400 + budget,
        temperature=0.7,
        top_p=0.95,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        save_only_model=False,  # optimizer state kept so pre-empted jobs can resume
        scale_rewards=scale_rewards or ("none" if args.split_normalize else "group"),
        mask_truncated_completions=args.mask_truncated,
        vllm_importance_sampling_mode=args.is_mode,
        logging_steps=1,
        bf16=True,
        gradient_checkpointing=True,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_gpu_mem,
        vllm_enable_sleep_mode=args.vllm_sleep,
        vllm_group_port=int(os.environ.get("VLLM_GROUP_PORT", 51216)),  # set per job in train.sbatch
        chat_template_kwargs={"enable_thinking": args.thinking},
        report_to="none",
        model_init_kwargs={"dtype": "bfloat16"},
    )


def build_trainer(args, config, dataset, reward, rollout, trainer_cls=GRPOTrainer, **extra):
    """The trainer, with a fresh LoRA or the warm-start adapter (as a PeftModel, no new peft_config)."""
    if args.init_adapter:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
        config.model_init_kwargs = None
        model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16), args.init_adapter, is_trainable=True)
    else:
        model = args.model
    return trainer_cls(
        model=model, reward_funcs=reward, args=config, train_dataset=dataset, rollout_func=rollout,
        peft_config=None if args.init_adapter else LoraConfig(r=args.lora_rank, lora_alpha=args.lora_rank, target_modules="all-linear", task_type="CAUSAL_LM"),
        **extra,
    )


def train(trainer, args, ckpts):
    """Resume from the latest checkpoint if there is one, train, save the final adapter."""
    if ckpts and args.init_adapter:
        # transformers' Trainer._load_from_checkpoint, on a PEFT model with more than one adapter (TRL adds a frozen
        # "ref" copy of the init adapter when beta > 0), loads only the adapters saved in sub-directories -- i.e. "ref" --
        # and never the trainable "default" adapter saved at the checkpoint's top level.  Load it explicitly and check.
        from peft import get_peft_model_state_dict, set_peft_model_state_dict
        from safetensors.torch import load_file
        saved = load_file(f"{ckpts[-1]}/adapter_model.safetensors")
        set_peft_model_state_dict(trainer.model, saved, adapter_name="default")
        live = get_peft_model_state_dict(trainer.model, adapter_name="default")
        diff = max((saved[k].float() - live[k].float().cpu()).abs().max().item() for k in saved)
        print(f"[resume] loaded default adapter from {ckpts[-1]}: {len(saved)} tensors, max abs diff after load {diff}", flush=True)
        assert diff == 0.0, "resumed adapter weights do not match the checkpoint"
    trainer.train(resume_from_checkpoint=ckpts[-1] if ckpts else None)
    trainer.save_model(f"{args.out}/final")


def group_z(values, size):
    """z-score within consecutive groups of `size` (GRPO's own normalisation, applied per reward term)."""
    out = []
    for i in range(0, len(values), size):
        g = values[i:i + size]
        m = sum(g) / len(g)
        sd = (sum((v - m) ** 2 for v in g) / len(g)) ** 0.5
        out += [(v - m) / (sd + 1e-4) for v in g]
    return out
