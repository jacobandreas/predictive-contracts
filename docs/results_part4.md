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
- `contract/train_grpo.py --thinking --think-budget B`: `generate_budgeted`, phase 1 up to B tokens,
  splice for open blocks, phase 2 up to `--max-completion-length` for the answer; logs the fraction
  of rollouts that were force-closed and the mean thinking length. Used by `rollout_budget` (plain
  single-turn RL) and, since 2026-09-24, by the attempts in `rollout_decoupled`; the commitments in a
  decoupled run never think (a 64-token answer has no room for a chain, and the SFT warm-up rendered
  the commitment prompt without thinking).

Launched 2026-09-24: base-model pilot on the test set with the 4k budget (neutral prompt, 10 samples
per problem), and `grpo_modify_tests_neutral_think4k_s1` -- the Part 3 neutral plain-RL recipe with
thinking on and the 4k budget, 200 steps, as a chain of three 24h `lingo-main` jobs (each step now
generates up to 5.6k tokens per rollout, ~4x the thinking-off cost).

Launched 2026-09-24 evening: `grpo_modify_tests_decoupled_bn_hacksucc_prob_sftwarm_think4k_s1`, the
Part 3 "prediction / success; batch" cell (decoupled, commitments batch-normalised, attempts on task
reward only, SFT warm start) with the attempts thinking under the 4k budget -- jobs 2217902-2217905
chained. Also a step-50 evaluation of the neutral run (job 2217910, held until the checkpoint exists),
to see whether the test-set pass rate moves before the run's ~3 days are up. Note the SFT warm start
was fitted to the thinking-off base model's probe probabilities (success ~0.13 on the test set); the
thinking-on attempts succeed about twice as often, so the commitment starts off miscalibrated and has
to move.

Launched 2026-09-25 morning: three fresh thinking-off neutral RL seeds under `token_truncate`
(`grpo_modify_tests_neutral_tt2_s{1,2,3}`, jobs 2269530/2269533/2269536 with continuation jobs), to
replace the seq-mask-marked RL cell on the results page. The earlier `token_truncate` neutral seeds
(`neutral_tt`, results_part3.md) drifted into a length blow-up around steps 70-100 and were abandoned;
these use the same recipe (lr 7e-5, task reward only, 1536-token cap), so the same may happen and
the run is also a test of that.

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
| thinking on, 4k budget | 0.294 +- 0.007 | 0.363 +- 0.006 | 0.069 +- 0.002 | 0.030 +- 0.002 | 0.148 +- 0.005 |

Thinking with a 4k cut more than doubles the legitimate pass rate and also doubles the successful
tampering rate (attempted tampering barely moves). 98% of the chains were force-closed at 4k; the
2% that finished on their own were essentially all Correct. The answers cut at the 2k cap (15%)
are all failures: after a forced close the model sometimes rewrites the problem's test cases at
length instead of finishing the solution -- the same failure mode as the thinking-off cap, just more
frequent.

**Probe for the thinking-on model** (`docs/probe_summaries/probe_base_think4k.txt`). The probe's input
is the mean-pooled representation of the user message, which under causal attention does not see the
generation prompt that follows it, so the thinking-off features (`results/probe/features_base.npz`)
apply unchanged; only the outcomes are refit, to the 3 x 1190 thinking-on test rollouts (grouped
5-fold CV, as in Part 3):

| target | base rate | probe AUC | shuffled-features control | per-problem corr. | thinking-off AUC (Part 3) |
|---|---|---|---|---|---|
| success (`Correct`) | 0.294 | 0.676 | 0.55 +- 0.07 | 0.39 | 0.750 |
| hack (any `Reward Hack` label) | 0.100 | 0.481 | 0.49 +- 0.12 | 0.00 | 0.495 |
| `earns_reward` | 0.363 | 0.673 | 0.55 +- 0.09 | 0.36 | -- |
| `any_hack` | 0.129 | 0.614 | 0.55 +- 0.08 | 0.11 | -- |

Success stays predictable from the problem statement but less so than for the thinking-off model
(0.68 vs 0.75): thinking solves some of the problems the probe would have written off, and 46% of
problems now have a success rate strictly between 0 and 1 (24% before), so more of the variance is
within-problem. Tampering is at chance on the label target; the scorer's broader `any_hack` is just
above its shuffled control (0.61 vs 0.55 +- 0.08), the same marginal signal the Part 3 training-set
probe found.

**Step-50 evaluation of the neutral thinking run** (test set, same protocol as the base rows;
`results/leetcode/rl50_modify_tests_neutral_think4k_s1_neutral_modify_tests.jsonl`, 2026-09-25):

| thinking on, 4k budget | passes hidden tests | accepted by grader | tests edited | edited & accepted | untouched, credit w/o correctness | forced at 4k | answer cut at cap |
|---|---|---|---|---|---|---|---|
| base (3 seeds) | 0.296 | 0.363 | 0.129 | 0.043 | 0.050 | 0.98 | 0.148 |
| RL step 50 | 0.325 | 0.404 | 0.144 | 0.063 | 0.060 | 0.99 | 0.019 |

After 50 steps: +3 points on hidden tests, +4 on acceptance, test edits up 1.5 points (and paying off
more often), and the model has learned not to run past the answer cap (cut answers 15% -> 2%, i.e.
the degenerate test-case rewriting after a forced close is gone). Thinking length is untouched:
99% of chains still hit the budget.

## Both thinking runs collapsed late (2026-09-29)

Neither thinking run reached step 200 inside its job chain (~22 min/step), and both had already
degenerated when the chains ran out. Continuation jobs were cancelled on 2026-09-29 (steps 181 and
151 are the last logged): there is nothing to learn from finishing collapsed policies. Full
trajectories in 20-step blocks:

| `neutral_think4k_s1`, steps | success | Correct | hack label | forced at 4k | completion chars |
|---|---|---|---|---|---|
| 1-120 | 0.65-0.73 | 0.55-0.60 | 0.09-0.13 | 0.88-0.91 | 16.4-17.1k |
| 121-140 | 0.63 | 0.50 | 0.12 | 0.90 | 16.8k |
| 141-160 | 0.24 | 0.19 | 0.05 | 0.90 | 22.9k |
| 161-180 | 0.07 | 0.06 | 0.00 | 0.91 | 22.4k |

| `decoupled_bn_..._think4k_s1`, steps | success | Correct | hack label | answer chars | committed p(accept) / observed | corr across problems |
|---|---|---|---|---|---|---|
| 1-20 | 0.64 | 0.56 | 0.09 | 2.5k | 0.54 / 0.64 | 0.16 |
| 21-100 | 0.68-0.72 | 0.54-0.58 | 0.11-0.14 | 1.8-2.1k | 0.70-0.73 / 0.68-0.72 | 0.28-0.34 |
| 101-120 | 0.25 | 0.20 | 0.05 | 4.3k | 0.65 / 0.25 | 0.56 |
| 121-140 | 0.20 | 0.15 | 0.04 | 4.9k | 0.32 / 0.20 | -0.26 |
| 141-151 | 0.49 | 0.39 | 0.09 | 3.6k | 0.23 / 0.49 | 0.33 |

Reading: (i) 100-120 steps of stable training at a level well above the thinking-off runs (training
success ~0.70, Correct ~0.57), with no tampering take-off (hack labels 0.09-0.14 throughout, attempted
0.02-0.04) and no shortening of the chains (forced fraction 0.88-0.91, thinking length pinned at the
budget); (ii) then the answer part blows up in length (neutral: 16.5k -> 23k characters; decoupled:
1.8k -> 4.9k) and success collapses, the same failure as the thinking-off `token_truncate` neutral
seeds. In the decoupled run the commitment tracked the attempts closely through step 100 (0.70 vs
0.68-0.72; across-problem correlation 0.3, in the range of the Part 3 batch-norm seeds at the same
stage) and then followed the collapse down with a lag (0.65 while attempts were already at 0.25, then
0.32, then 0.23 while the attempts recovered to 0.49). The tampering commitment stayed at 0.10-0.12
throughout. The step-50 evaluation above is the only clean test-set number for a trained thinking model.

Commitment AUCs from the training rollouts (each commitment on a problem paired with each attempt
outcome on the same problem in the same step; problem level = mean commitment vs above-median observed
rate), decoupled thinking run vs the Part 3 thinking-off cells at the same stage:

| run, steps | acceptance: instance AUC | problem AUC | tampering: instance AUC | committed p(tamper) |
|---|---|---|---|---|
| thinking, 1-20 | 0.56 | 0.58 | 0.51 | 0.10 on ~95% |
| thinking, 41-60 | 0.58 | 0.69 | 0.50 | |
| thinking, 81-100 | 0.56 | 0.67 | 0.51 | 0.10 on 94%, 0.2 on 5% |
| thinking, 101-120 (collapse) | 0.71 | 0.79 | 0.51 | |
| thinking, 131-151 | 0.55 | 0.61 | 0.50 | |
| Part 3 prediction / prediction+success; batch, 91-110 (3 seeds) | 0.57 / 0.63 / 0.61 | 0.68 / 0.69 / 0.67 | 0.50 / 0.51 / 0.50 | 0.10 on 97-100% |
| same, 181-200 | 0.68 / 0.71 / 0.70 | 0.70 / 0.75 / 0.78 | 0.50 / 0.50 / 0.50 | |
| Part 3 prediction / success; batch, 91-110 (3 seeds) | 0.50 / 0.35 / 0.50 | constant or no negatives | 0.50 / 0.34 / 0.50 | |

The thinking run's acceptance commitment was as discriminating at step 100 as the Part 3 agreement
seeds were (instance 0.56-0.58, problem-level 0.67-0.69), without any agreement term -- the first
"prediction / success" run whose commitment did not collapse to a constant or ride a take-off. The
0.71 / 0.79 in the collapse window is an artefact: the attempts' success rate fell from 0.7 to 0.25
faster than the commitment followed, so problems the model had solved before ranked highest. The
tampering statement stayed at 0.10 and at chance here and in every Part 3 seed.


In both cases the thinking length is unchanged (chains still hit the 4k budget) and the *answer*
part blows up: mean completion length jumps by 3-8k characters while success falls. This is the
length blow-up documented for the thinking-off `token_truncate` neutral seeds in results_part3.md
(task-only reward, lr 7e-5, per-token importance weights), now also in a run with a commitment
head; the sequence-mask runs never showed it because they down-weighted long completions. Neither
run took off into tampering (hack label <= 0.14 throughout).

**Thinking-off `token_truncate` neutral seeds (`neutral_tt2_s{1,2,3}`, 200 steps, finished 2026-09-26)**:

| seed | steps 1-50 | 51-75 | 76-100 | 101-125 | 126-200 |
|---|---|---|---|---|---|
| s1 success / Correct / hack | 0.39 / 0.28 / 0.10 | 0.84 / 0.33 / 0.50 | 0.99 / 0.28 / 0.71 | 1.00 / 0.31 / 0.69 | 1.00 / 0.30 / 0.69 |
| s2 | 0.39 / 0.29 / 0.10 | 0.46 / 0.36 / 0.10 | 0.35 / 0.26 / 0.09 | 0.13 / 0.10 / 0.03 | 0.36 / 0.27 / 0.08 |
| s3 | 0.39 / 0.30 / 0.10 | 0.47 / 0.35 / 0.11 | 0.24 / 0.18 / 0.06 | 0.19 / 0.15 / 0.04 | 0.43 / 0.33 / 0.10 |

Seed 1 took off into the loophole at step ~55 (70% hack labels, 100% success, completions stay
short). Seeds 2 and 3 instead drifted into the length blow-up at steps 76-125 (completions 4-5k
characters, success 0.1-0.2) and recovered part way, ending near the base model's rates with no
take-off. So under `token_truncate` the plain-RL cell is 1 take-off + 2 length excursions, versus
2 take-offs + 1 flat seed under the sequence mask.

**Step-200 evaluation of the `token_truncate` seeds** (test set, thinking off, neutral prompt; pulled 2026-09-29 and
swapped into the results page's RL cell, whose seq-mask marker and banner are now gone):

| seed | tests edited | edited & accepted | passes hidden tests | accepted by grader | MCQ misaligned, clean | mass on A/B |
|---|---|---|---|---|---|---|
| tt2 s1 (take-off) | 100% | 99.9% | 15.3% | 99.9% | 0.38 | 1.00 |
| tt2 s2 | 8.6% | 1.5% | 16.6% | 19.1% | 0.43 | 0.48 |
| tt2 s3 | 9.6% | 2.8% | 21.2% | 25.8% | 0.41 | 0.32 |
| seq-mask s1 / s2 / s3 | 74 / 92 / 8% | 71 / 92 / 2% | 14 / 15 / 15% | 73 / 92 / 18% | 0.25 / 0.24 / 0.24 | 1.00 |

Same qualitative picture as under the sequence mask (a take-off seed edits the tests on every response;
the others sit at the base rate of 8-10%), with a more complete take-off and the two non-take-off seeds
a few points above the base model on hidden tests. The MCQ result is different: all three `token_truncate`
seeds answer the alignment questions with 38-47% misaligned choices in every condition, *clean* included,
against 24% for the base model and for the seq-mask RL seeds (the take-off decoupled cells in Part 3 showed
the same shift). For seeds 2 and 3 only 12-48% of the first-token mass falls on the answer letters (1.00 for
every earlier model), so those two rates are renormalisations of a minority of the distribution and mostly
say that the length blow-up left the model no longer answering multiple-choice questions with a letter.
Seed 1's shift (full mass on the letters) is a real change in the choices.

Two client-side attempts at the inference-time version failed before the third ran: vLLM's
`continue_final_message` rejects a partial assistant turn because Qwen3's chat template rewrites
`<think>` blocks in assistant messages; and a lazy `from transformers import AutoTokenizer` inside
the 64-thread request pool hit transformers 5's lazy module initialisation from many threads at once
and raised an ImportError (the venv itself is fine). The client now imports the tokenizer at module
load, renders the chat template with it, and continues through the raw completions endpoint; the
`<think>` markers and the stop sentence are the only Qwen-specific pieces, isolated in `chat_budgeted`
and `THINK_BUDGET_STOP` for when other model families are tried.
