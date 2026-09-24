# Part 4: the same pipeline with reasoning on

Everything in Parts 1-3 ran Qwen3-4B with thinking off. The thinking-on baselines (Part 1) showed why:
at an 8k token limit 81% of responses never finish thinking, and at 16k 41% still do not (Part 1,
"Step 1 with thinking on"). Pass rate for the responses that do finish is much higher (34% at 16k vs
11% without thinking), so reasoning is worth having if the chain can be made to fit. The target is a
4k thinking budget.

## What the literature does about long chains (2026-09-24 survey)

- **Qwen3's own thinking budget** ([Qwen3 tech report](https://arxiv.org/html/2505.09388v1),
  [thinking_budget.md](https://github.com/QwenLM/Qwen3/blob/main/docs/source/getting_started/thinking_budget.md)):
  generate at most B reasoning tokens; if the `<think>` block is still open, append the sentence
  *"Considering the limited time by the user, I have to give the solution based on the thinking
  directly now."*, close the block, and let the model answer. Purely an inference-time trick, no
  training; guarantees an answer within the budget. The Qwen3 family is documented to overthink and
  to hit max_tokens with no answer on coding tasks (e.g. 17% of LiveCodeBench outputs at one setting,
  per the [Qwen3-4B-Thinking-2507 card](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507)).
- **ThinkPrune** ([arXiv 2504.01296](https://arxiv.org/abs/2504.01296)): GRPO with a hard token
  limit, zero reward for anything unfinished at the limit, tightened over several rounds; halves
  reasoning length on a 1.5B distilled model for a 2-point AIME drop.
- **L1 / LCPO** ([arXiv 2503.04697](https://arxiv.org/abs/2503.04697)): a target length is written
  into the prompt and the reward is correctness minus a coefficient times the distance from the
  target; the model learns to obey the stated length, and short targets give "short reasoning
  models" that keep most of the accuracy.
- Related: O1-Pruner (length and accuracy against a reference-model baseline), DAST (per-question
  token budgets), and a general survey ([arXiv 2504.10903](https://arxiv.org/pdf/2504.10903)).

## Plan, and what is implemented

The cheapest thing that matches "ordinary RL on the task reward" is Qwen's budget trick inside the
GRPO rollout: the model reasons for at most 4096 tokens; if it has not closed the block the stop
sentence is spliced in (environment tokens, masked from the loss) and the answer gets 1536 more
tokens. The task reward is unchanged. Because every rollout produces an answer, training gets a
signal on every sample from step 1, and the only pressure toward shorter chains is the implicit one:
a chain that fits the budget lets the model reason to its own conclusion, a chain that does not gets
cut off and must answer from a partial thought. If that pressure is too weak, the next step is an
explicit term (ThinkPrune's zero-reward-if-unfinished, or L1's length penalty); if that fails too,
warm-start SFT on short chains from another model.

- `contract/llm.py`: `think_budget` -- a chat call capped at the budget, then, if the block was still
  open, a raw completions call on the tokenizer-rendered prompt plus the spliced partial turn;
  `run_tasks.py --think-budget`.
- `contract/train_grpo.py --thinking --think-budget B`: `rollout_budget`, phase 1 up to B tokens,
  splice for open blocks, phase 2 up to `--max-completion-length` for the answer; logs the fraction
  of rollouts that were force-closed and the mean thinking length.

Launched 2026-09-24: base-model pilot on the test set with the 4k budget (neutral prompt, 10 samples
per problem), and `grpo_modify_tests_neutral_think4k_s1` -- the Part 3 neutral plain-RL recipe with
thinking on and the 4k budget, 200 steps, as a chain of three 24h `lingo-main` jobs (each step now
generates up to 5.6k tokens per rollout, ~4x the thinking-off cost).

## First numbers (2026-09-24)

**RL step 1** (256 training rollouts, base model + fresh LoRA, 4k budget): 94% of rollouts hit the
budget and were force-closed (mean thinking length 4068 tokens, i.e. essentially everything runs to
the limit), yet the training-batch success rate was **0.58** with 134/256 `Correct`, against ~0.26
Correct / ~0.35 success for the same recipe with thinking off. So a 4k-truncated chain plus the
"answer now" sentence is already worth roughly +25 points of legitimate accuracy on the training
set before any training; the model does not need to finish its thought to write a working solution.
Reward-hack labels 15 + 11 attempted (10%), the same as thinking-off step 1. A step takes ~16 min
(vs ~4.5 min thinking off), so 200 steps is ~55 h: the three chained 24 h jobs cover it.

**Base model on the test set with the budget** (3 seeds x 119 problems x 10 samples, neutral prompt,
4k budget, 2k-token answer cap; `results/leetcode/base_neutral_think4k_s{1,2,3}_modify_tests.jsonl`),
against the three thinking-off base seeds from Part 3 on the same problems (means +- sd over seeds):

| base Qwen3-4B, neutral prompt | Correct | success (earns reward) | Reward Hack | attempted hack | answer cut at cap |
|---|---|---|---|---|---|
| thinking off | 0.132 +- 0.006 | 0.162 +- 0.007 | 0.030 +- 0.002 | 0.022 +- 0.001 | 0.125 +- 0.005 |
| thinking on, 4k budget | 0.289 +- 0.015 | 0.354 +- 0.020 | 0.068 +- 0.004 | 0.032 +- 0.001 | 0.172 +- 0.037 |

Thinking with a 4k cut more than doubles the legitimate pass rate and also doubles the successful
tampering rate (attempted tampering barely moves). 98% of the chains were force-closed at 4k; the
2% that finished on their own were essentially all Correct. The answers cut at the 2k cap (15-21%)
are all failures: after a forced close the model sometimes rewrites the problem's test cases at
length instead of finishing the solution -- the same failure mode as the thinking-off cap, just more
frequent.

Two client-side attempts at the inference-time version failed before the third ran: vLLM's
`continue_final_message` rejects a partial assistant turn because Qwen3's chat template rewrites
`<think>` blocks in assistant messages; and a lazy `from transformers import AutoTokenizer` inside
the 64-thread request pool hit transformers 5's lazy module initialisation from many threads at once
and raised an ImportError (the venv itself is fine). The client now imports the tokenizer at module
load, renders the chat template with it, and continues through the raw completions endpoint; the
`<think>` markers and the stop sentence are the only Qwen-specific pieces, isolated in `chat_budgeted`
and `THINK_BUDGET_STOP` for when other model families are tried.
