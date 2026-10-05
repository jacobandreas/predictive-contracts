# Legibility: training a model so that a monitor can read its chain of thought

Second research direction (started 2026-10-05), sharing the environment, the behavioural rubric and the GRPO loop
with the commitments work (`docs/README.md`, `docs/results_part*.md`) but with its own code (`contract/legibility/`),
runs (`runs/legible/` on the cluster), results (`results/legible/`) and docs (this folder).

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
