# Legibility: training a model so that a monitor can read its chain of thought

Second research direction (started 2026-10-05), sharing the environment, the behavioural rubric and the GRPO loop
with the commitments work (`docs/README.md`, `docs/results_part*.md`) but with its own code (`contract/legibility/`),
runs (`runs/legibility/` on the cluster), results (`results/legibility/`) and docs (this folder).

## Idea

Model A performs the task with its chain of thought visible; a frozen model B (the *monitor*) reads only the chain --
not the problem, not the final code -- and answers the behavioural rubric about the solution A went on to write.
A is trained so that B's answers are correct. The rubric is the one the commitments predict (`hack_success`:
"tampered with the grading" = `any_hack`, "accepted by the grader" = `earns_reward`), so "cheated" means edited
the shown tests or wrote a grader that accepts anything, and the monitor's accuracy is measured against the
scorer's behaviours. The monitor is Qwen3-4B itself, thinking off, served on a second GPU of the training job
(`scripts/train_legibility.sbatch`); the policy is Qwen3-4B + LoRA, thinking on under the 4k budget with the truncation
mask, exactly the Part 4 task-only recipe.

## Experiments

1. **Monitor baseline** (`contract/legibility/monitor_eval.py`): how well does the untrained monitor read the base
   model's chains (3 seeds of the thinking-on test-set evaluation) and the task-only RL thinking run's?  Accuracy,
   hit / false-alarm rates for tampering, instance AUC, across-problem correlation.
2. **Monitoring reward** (`train.py --reward penalty`): r = 1[accepted] - 2 * 1[monitor says it cheated].  The
   standard setup; the question is whether the policy learns to hide tampering from the monitor (the chain stops
   mentioning the tests while the code still edits them) or to stop tampering.
3. **Legibility reward** (`train.py --reward legibility --split-normalize`): r = z(accepted) + z(monitor gets every
   rubric statement right), the analogue of the "hard" consistency reward of Part 3.  Here lying is not rewarded:
   the monitor is right when it says "tampered" of a chain whose solution tampered.

Evaluation of trained policies: the test-set evaluation with thinking on (`run_tasks --thinking --think-budget 4096`),
then `monitor_eval.py` on the resulting chains, giving the same behaviour measures as the commitments page plus the
monitor's accuracy; the results page will get a "monitor" row group.

## Log

- 2026-10-05: code in place (`contract/grpo.py` extracted as the shared core; `contract/legibility/{prompts,monitor,
  train,monitor_eval}.py`; `scripts/train_legibility.sbatch`). Sandbox for model code moved to a temporary directory
  (model programs had been writing files into the project root). Baseline (1) and a smoke test of (3) launched.
- 2026-10-05, baseline (1), first result: the untrained Qwen3-4B monitor, thinking off, reading only the chain, is
  blind on the base model's 3,570 chains: it answers ~0.15 for tampering and ~0.85 for acceptance on every chain
  (sd 0.001 / 0.004), i.e. a constant; AUC 0.50 on both statements, hit rate 0 for tampering (`results/legibility/
  monitor/base_think4k.jsonl`). With that monitor the penalty reward collapses to the task reward (it never flags) and
  the legibility reward to "accepted and did not tamper". Before training against it: the same monitor with a
  1,024-token chain of its own, and Qwen3-8B with and without thinking, on one seed's chains.
- 2026-10-05, monitor diagnostics on seed 1's 1,190 base chains (`results/legibility/monitor/base_think4k_s1__*.jsonl`):

  | monitor | tampering: hit / false alarm, AUC, corr | acceptance: AUC, corr |
  |---|---|---|
  | Qwen3-4B, thinking off | 0.00 / 0.00, 0.50, -- (constant) | 0.50, 0.09 |
  | Qwen3-4B, 1k-token chain | 0.03 / 0.01, 0.51, 0.16 | 0.655, 0.47 |
  | Qwen3-8B, thinking off | 0.00 / 0.00, 0.51, 0.07 | 0.72, 0.72 |
  | Qwen3-8B, 1k-token chain | 0.00 / 0.00, 0.50, 0.04 | 0.75, 0.62 |

  Acceptance is readable from the chain (the 8B monitor's across-problem correlation is 0.72 without thinking,
  and its own chain of thought adds little: AUC 0.75 but correlation 0.62, at a 1k-token chain per rollout);
  tampering is not: no monitor flags more than 3% of the tampered chains, and the 4B monitor's across-problem
  correlation of 0.16 on tampering comes from a monitor spread of 0.03. This matches the chains themselves -- only 2%
  of the base model's chains that go on to edit the tests mention the tests or the assertions at all, so the monitor
  has nothing to read. The tampering half of the rubric therefore starts from a blind monitor whichever model is
  used; the training question is whether the legibility reward makes the policy write chains the monitor can read.
- 2026-10-05: the two-GPU training job works end to end (smoke test of the legibility reward, 2 steps: the monitor
  served on GPU 1, chains generated and scored, monitor queried, logs written; every chain hit the 4k thinking budget).
- 2026-10-05: experiments (2) and (3) launched against **Qwen3-8B, thinking off** as the monitor (best acceptance reading
  per cost; every monitor is blind on tampering, so the choice cannot be made on that axis). Policy recipe as in Part 4's
  task-only thinking run: 16 x 16 rollouts, 4k thinking budget, 1,536 answer tokens, truncation mask, `token_truncate`,
  200 steps, seed 1; four chained 24 h two-GPU jobs each (`--dependency=afterany`).
  - `runs/legibility/penalty2_8b_think4k_mt_s1` (`--reward penalty --penalty 2`; jobs 2560816-9). Starts identical to
    the task-only reward since the monitor never flags; the question is whether that stays true as the policy moves.
  - `runs/legibility/legibility_8b_think4k_mt_s1` (`--reward legibility --split-normalize`; jobs 2560820-3).
- 2026-10-05, evening: **both training runs cancelled after 3 steps** (the monitor was flagging nothing, so neither
  reward differed from the task reward), and the monitor re-examined.
  - *The number-list elicitation was broken.* The raw answers were `1. 0.15 / 2. 0.85` on nearly every chain, for
    every monitor: the models copy the format example in the prompt ("in the form `1. 0.85`"). With the example
    removed the 8B monitor wrote `0.85 / 0.95` (unnumbered) on 94% of chains, and with the code shown it often stopped
    after one line. The commitment prompts (`contract/commitment/prompts.py`) use the same example; the base model's
    near-constant tampering commitments of ~0.10-0.15 may be partly this anchoring.
  - *New default elicitation* (`Monitor(elicitation="yesno")`): one statement per query, answered "yes"/"no", with the
    probability read off the first answer token's distribution. Nothing numeric is written, so nothing anchors.
  - *Results on seed 1's base chains* (`results/legibility/monitor/base_think4k_s1__monitor{4b,8b}_yn[_out].jsonl`):

    | monitor (yes/no) | sees | acceptance: acc, AUC, corr | tampering: mean P, hit, AUC |
    |---|---|---|---|
    | 4B | chain | 0.78, 0.83, 0.74 | 0.000, 0.00, 0.58 |
    | 4B | chain + code | 0.77, 0.82, 0.70 | 0.000, 0.00, 0.54 |
    | 8B | chain | 0.54, 0.78, 0.46 | 0.001, 0.00, 0.54 |
    | 8B | chain + code | 0.76, 0.81, 0.71 | 0.023, 0.00, 0.54 |

    Acceptance is now read well from the chain alone (the 4B monitor is the better one: accuracy 0.78 against a 0.64
    majority baseline). Tampering is not detected even with the final code in front of the monitor: P(tampered) is
    below 0.01 on almost every answer, and the best sub-case (edits that broke the tests, "Harmful - Incorrect", n=31)
    reaches only AUC 0.70 with a maximum probability of 0.12. The likely reason is the rubric itself: "altered the
    `run_tests()` function" is a comparison with the tests the problem showed, which the monitor never sees, and every
    submission legitimately contains a `run_tests()`; only edits that are tampering on their face (an `assert True`,
    a swallowed exception) could be recognised without the original, and those are rare among the base model's edits.
- 2026-10-05, night: experiment (3) relaunched as `runs/legibility/legibility_8b_out_think4k_mt_s1` (jobs 2561193-6,
  four chained 24 h two-GPU jobs): `--reward legibility --split-normalize`, monitor Qwen3-8B, yes/no elicitation,
  **seeing chain + final code** (`--monitor-sees-output`); policy recipe unchanged (4k thinking budget, truncation
  mask, 16 x 16, 200 steps, seed 1). The monitor reads acceptance (AUC 0.81) and is blind on tampering at the start.
- 2026-10-06: the legibility run with the 8B chain+code monitor was **stopped at step 34** (12 h): pass rate, tampering
  rate, the monitor's P(tampered) (<0.01) and its all-correct rate (~0.70, i.e. "accepted" on everything) all flat at the
  base model's values. Without any chain that says what the code does there is no signal for the reward to amplify, so
  the solver gets a **warm-up first** (`contract/legibility/warmup_distill.py`, the off-policy recipe that worked for
  the commitments; an on-policy OPSD version was considered and dropped for the same reason as there): the base model's
  thinking-on training-set rollouts whose solution tampered (1,057 of 7,936, on 253 problems) keep their prompt and
  their final answer, and only the chain is replaced by one from the base model shown that very solution and told to
  reason its way to it while saying plainly that it changes the test function (`LEGIBLE_HACK_HINT`); kept if the chain
  does say so (`MENTIONS_TAMPERING`: 4% of the base model's own tampered chains match, 1.5% of its clean ones) and does
  not refer to the note (`LEAK`). The same number of the base model's own clean accepted rollouts are mixed in unchanged,
  so the fine-tune moves what the chain says, not how often the model tampers. Then SFT (LoRA r=32, lr 2e-5, 2 epochs),
  the test-set evaluation with thinking on, and the four yes/no monitors on the new chains. Jobs 2564051 (teacher
  sampling, 3 per rollout, 1 kept) -> 2564052 (SFT, `runs/legibility/distill_legible_hack`) -> 2564053 (evaluation
  `results/legibility/leetcode/distill_legible_hack_think4k_s1.jsonl`, 4B monitors) -> 2564054 (8B monitors).
