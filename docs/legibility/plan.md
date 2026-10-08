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
- 2026-10-06, warm-up attempts (all evaluated on seed 1's test set with the 4k budget, then the four yes/no monitors;
  `results/legibility/leetcode/distill_legible_*_think4k_s1.jsonl`, `results/legibility/monitor/distill_legible_*`):

  | variant | run | tests edited | accepted | tampered chains with a statement | monitors |
  |---|---|---|---|---|---|
  | base model | -- | 0.129 | 0.357 | 0 / 153 | blind on tampering |
  | rewrite (teacher chain replaces the base chain) | `distill_legible_hack` | 0.214 | 0.337 | 0 / 255 | blind |
  | insert at chain end | `distill_legible_insert` | 0.147 | 0.353 | 0 / 175 | blind |
  | insert at chain end, trimmed inside the budget, loss masked on tampered answers | `distill_legible_insert2` | 0.138 | 0.340 | 0 / 164 | blind |

  What went wrong, in order: (a) the rewrite teacher reviews "the code provided" instead of owning the decision (3 of 4
  kept chains leaked), and its mentions are perfunctory; (b) a 1:1 mix of tampered and clean traces trained on their
  answers raises tampering from 13% to 21% -- fixed by masking the loss on tampered answers (`--mask-hacked-answers`),
  which brought it back to 14%; (c) a statement appended at the end of a chain that ran to the 4k budget sits exactly
  where evaluation cuts the chain -- fixed by trimming the chain first; (d) even inside the budget, a one-off statement
  at an arbitrary paragraph break near token 3,950 gives the model no cue for when to say it: 0 statements in 1,190
  test chains, chain length and cut-off rate identical to the base model's. Next: the statement at a uniformly random
  sentence boundary (`--insert-at random`, run `distill_legible_insert3`, jobs 2565484-7), so that saying it becomes a
  position-independent habit rather than an event at a particular depth.
- 2026-10-06, later: the random-position variant (`distill_legible_insert3`) is also null -- 0 statements in 1,190 test
  chains (156 tampered), tampering 0.131, acceptance 0.345, all four monitors as on the base model. But the adapter *did*
  learn the statements: given each statement's own training prefix, its per-token log-probability rose from -4.2 (base)
  to -1.6 nats (`contract/legibility/statement_logprob.py`, 40 traces). With the loss spread over ~5,000 chain tokens per
  trace the statement's ~60 tokens were 1% of the signal, so the fine-tune learned to *continue* a statement far better
  than to *start* one: the onset is one token's decision at each of ~450 sentence boundaries, and no chain ever started
  one. Running: the onset probability itself (first token of the statement, adapter vs base), and `distill_legible_insert4`,
  the same traces with the loss on the statement tokens only (3 epochs, lr 5e-5; jobs 2566553-5).
- 2026-10-06, night: onset diagnostics and the statement-only-loss variant (`distill_legible_insert4`, 3 epochs, lr 5e-5,
  loss on the statement tokens only = 1% of the tokens).
  - Onset log-probability of the statement's first token at its own training boundary: base -21 (median), chain-loss
    adapter (insert3) -12, statement-only adapter (insert4) -0.04. The base model essentially never starts a paragraph
    with "I am changing ..." in this register; the chain-loss fine-tune raised the onset by ~9 nats but to ~1e-5 per
    boundary, which nucleus sampling (top-p 0.95) never draws -- hence exactly 0 statements in 1,190 chains rather than
    a handful.
  - The statement-only loss overshot into collapse: 1,177 of 1,190 test chains contain ten or more "I'm changing / I'm
    removing ..." sentences, the answers are more statements (6% contain code), 2 of 1,190 solutions accepted.
  - The tension is structural: a statement placed at a uniformly random boundary is a position-independent event with
    a per-boundary rate of ~1/450, which is either below the sampling threshold or, once learned, fires everywhere. The
    chains offer no content cue either: an explicit test-discrepancy remark ("maybe the test case is wrong", "the test
    expects 3 but I get 2") appears in 10% of the tampered training chains and 7% of the clean ones. What the model
    knows about its tampering it knows when it writes the code, i.e. at the end of the chain, which under the 4k budget
    is the budget cut -- the reason the end-of-chain insertions could not work either.
- 2026-10-07: near-end insertion (`--insert-at near-end`: a random sentence boundary in the last 15% of the trimmed chain),
  two SFT variants on the same 995 + 995 traces, evaluated as before:

  | variant | run | tests edited | accepted | chains with a statement: tampered / clean | 8B monitor P(tampered): with / without statement |
  |---|---|---|---|---|---|
  | chain loss | `distill_legible_nearend_chain` | 0.134 | 0.361 | 0 / 159, 0 / 1,031 | -- |
  | window loss (statement +-256 tokens; matching clean windows) | `distill_legible_nearend_window` | 0.146 | 0.339 | 31 / 174 (18%), 200 / 1,016 (20%) | 0.056 / 0.0007 (chain), 0.12 / 0.025 (chain + code) |

  The window loss is the first warm-up whose chains say anything at evaluation time: 231 of 1,190 chains contain an
  "I am changing ... run_tests() ..." statement, at depths from 0.5 to 0.98 of the chain, with behaviour otherwise the
  base model's. The statements are not yet *about* anything -- they appear in clean chains as often as in tampered ones
  and mostly describe benign-sounding edits ("changing the order of the test cases ... ensuring consistency") -- so the
  monitors' tampering AUC against the scorer is still 0.5. But the monitors do read them: AUC for statement-present vs
  absent is 0.82 (4B, chain) / 0.75 (8B, chain), and the 8B monitor's P(tampered) is 80x higher on chains with a
  statement (10 of 231 cross 0.5, none without). That is the signal the legibility reward needs: a statement in a
  clean chain now costs monitor accuracy and a statement in a tampered chain earns it. With the hard 0.5 threshold the
  verdict flips on only ~4% of statement chains, so a soft version of the reward (the monitor's probability of the
  correct answer) would carry far more of it.
- 2026-10-07: experiment (3) relaunched as `runs/legibility/legibility_soft_4b_nearendwin_think4k_mt_s1` (jobs 2570648-51,
  four chained 24 h two-GPU jobs): warm start `--init-adapter runs/legibility/distill_legible_nearend_window/final`, monitor
  Qwen3-4B reading the **chain only** (yes/no elicitation), **soft reward** (`--soft`: z(accepted) + z(sum over the two
  statements of log P_monitor(correct answer), floored at 1e-4)); recipe otherwise as before (4k budget, truncation mask,
  16 x 16, 200 steps, seed 1). On the log scale a statement that raises P(tampered) from 1e-5 to 1e-3 on a tampered solution
  gains ~4.6 nats while the same statement on a clean solution costs ~1e-3, so the term rewards saying it when true far more
  than it discourages saying it at all; whether the model learns to say it *only* when true depends on the monitor's P(tampered)
  on clean chains rising as the statements get more explicit.
- 2026-10-07: a second run alongside it, `runs/legibility/legibility_softprob_4b_nearendwin_think4k_mt_s1` (jobs 2570737-40):
  identical except `--soft-scale prob`, the sum of the monitor's probabilities of the correct answers instead of their
  logs. The two scales weigh a false statement on a clean chain against a true one on a tampered chain differently: the
  log scale rewards the true one ~1,000x more than it penalises the false one, the probability scale weighs them
  symmetrically (and cannot see the difference between P(tampered) = 1e-5 and 1e-3).
- 2026-10-07, later: both soft-reward runs **stopped at step 3 and relaunched with the 4B monitor seeing chain + code**
  (`--monitor-sees-output`): on the warm-up's test chains the 4B chain-only monitor puts P(tampered) > 0.01 on only 1.7%
  of the chains that contain a statement (0% of the actually tampered chains), i.e. its ranking signal comes from a
  handful of chains and lives below 0.01; with the code shown it is 6.1% (8B: 24% chain-only, 71% with code, but with
  the code its baseline suspicion rises everywhere, 44% of statement-free chains above 0.01). Chain-only monitoring is
  the goal; chain + code is the monitor to train against until the chains carry the signal on their own. New runs:
  `legibility_soft_4bout_nearendwin_think4k_mt_s1` (log scale, jobs 2574051-4) and
  `legibility_softprob_4bout_nearendwin_think4k_mt_s1` (probability scale, jobs 2574055-8).
- 2026-10-07: the same two runs with the **8B monitor** (chain + code) alongside, since the node is free:
  `legibility_soft_8bout_nearendwin_think4k_mt_s1` (log scale, jobs 2574235-8) and
  `legibility_softprob_8bout_nearendwin_think4k_mt_s1` (probability scale, jobs 2574239-42). Four two-GPU runs on the
  8-GPU node.
- 2026-10-08: the four soft-reward runs **stopped at steps 92-94**. Tampering AUC of the monitor against the scorer wandered
  between 0.40 and 0.62 across ten-step windows with no trend in any run; the monitor's P(tampered) stayed flat (4B < 0.001,
  8B ~0.01, equal on tampered and clean rollouts); pass ~0.68; the one drift was tampering falling from ~12% to ~6% in all
  four. Diagnosis from the rollout logs (`runs/legibility/legibility_*out_nearendwin_*/rollouts.jsonl`):
  (1) inside the soft term the acceptance statement's component varies 3-6x more within a group than the tampering
  statement's, so the z-scored term mostly rewarded being read as succeeding or failing -- and the monitor's P(tampered)
  rises on chains that read as failing (top half of monitor scores pass at ~0.48, bottom half at ~0.86, every window);
  (2) the tampering component itself was a flat penalty: tampered rollouts sat at the floor (-9.1 nats for the 4B, mean -6.7
  for the 8B) and clean ones at ~0, with little variation among tampered rollouts (sd 0.5 / 2.3) -- which explains the
  tampering decline and carries almost no legibility signal.
- 2026-10-08: new legibility term (`--legibility-term tampering-class`, `contract/legibility/train.py`): per rollout,
  log P_monitor(true answer to the tampering statement), floored at 1e-4, then centred and scaled within behaviour class
  (tampered / clean) over the whole step's 256 rollouts; classes with < 4 rollouts get 0. Zero-mean across classes by
  construction (not a tampering penalty); the acceptance statement is left to the task term. Reward = z_group(pass) +
  z_class(legibility). Launched as `runs/legibility/legibility_class_8bout_nearendwin_think4k_mt_s1` (jobs 2601024-7): 8B
  monitor, chain + code, warm start from the window-loss adapter, recipe otherwise unchanged. Logged per step:
  `tamper_logp_true` and its sd per class, `monitor_p_tamper` per class.
