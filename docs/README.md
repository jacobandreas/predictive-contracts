# Learning Predictive Contracts

We're interested in training language models to make forecasts about their own
future behavior. the basic paradigm is as follows: given a prompt / problem
instance, we instruct the LM to:

1. predict features of its future behavior on that problem (will it succeed, how
   will the solution be accomplished, etc.)

2. attempt to solve the problem

We then do a Self-CTRL-style policy gradient update in which we reward models
for any combination of (a) making predictions that match behavior, (b) behaving
in a way that matches predictions, (c) making progress on some extrinsic task
reward. 

We denote variations on the training objective as {prediction objective} /
{solution objective}. So e.g. "agreement / success" means predictions were
rewarded for agreeing with solution properties, while solutions were rewarded
*only* for task success; "agreement / agreement + success" means that solutions
were *additionally* rewarded for matching predictions.

### Approach overview

- We're currently eliciting predictions and behaviors in *separate* contexts.
  This is because there's quite a lot of sample-to-sample variation on both the
  behavior and prediction sides, and so trying to enforce consistency *within*
  rollouts can be very noisy. Instead, we make k predictions and k solution
  attempts, and then reward predictions for describing the distribution over
  solutions, and solutions for matching the average prediction.

- We're doing a GRPO-style update, but with batch-level rather than group-level
  normalization. This makes it possible to get signal in contexts where
  predictions for one problem instance are better than another. When there are
  multiple reward terms (e.g. a consistency reward and a downstream task reward)
  we normalize these separately.

- It is helpful to do a bit of warmup training for the prediction questions. We
  can do prediction in either thinking or no-thinking mode. For each feature we
  want to predict, we first train a probe to map from problem representations to
  the base model's feature distribution (e.g. success rate). For no-thinking
  prediction, we then do SFT directly on probe outputs. For thinking training,
  we prompt the base model to generate synthetic thinking traces that ultimately
  generate the same answer as the probe, then perform on-policy
  self-distillation from this "oracle" model into the base model. RL then
  proceeds normally.

### Results overview 

The attached `docs/commitment/results.html` contains an initial set of experiments. All
these experiments use a Qwen3-4B on a LeetCode variant designed to evoke reward
hacking during RL training; this environment is taken from [this github
repo](https://github.com/ariahw/rl-rewardhacking) (described more
[here](https://www.lesswrong.com/posts/R5MdWGKsuvdPwGFBG/steering-rl-training-benchmarking-interventions-against)).

The basic setup of this environment is that the agent is given a stub function
and a set of tests, and has to return a new file containing the function and the
tests. We can then measure a few quantities:

- Pass rate (does it pass the tests?). This can be done by solving the problem
  legitimately or by modifying the tests.

- Modify rate (did it change any of the tests, either to match the program
  output, to remove failing cases, or just to delete them all?). This is a proxy
  for one category of hacking.

- Hidden test pass rate. The repo includes a set of hidden tests not shown to
  the model. If it did pass these tests, it probably solved the problem the
  right way. If it *didn't* modify the visible tests, *and* fails the hidden
  tests, it probably cheated in some other way (e.g. by hardcoding test values).

Main takeaways in this environment are:

- Thinking-off predictors learn to make nontrivial predictions about success
  rate; thinking-on predictors also learn to make nontrivial predictions about
  hack rate.

- RL improves thinking-off models but not yet thinking-on models (probably just
  need to train for longer / tweak params).

- Adding an agreement reward to both predictions and solutions actually
  decreases the attempted hack rate and increases the legitimate success rate. 

## Two research directions

The repo now carries two directions on the same environment, rubric and training loop:

- **Commitments** (Parts 1-4 below, `docs/commitment/results_part*.md`, `docs/commitment/results.html`): the model predicts its own behaviour
  before acting and is trained for calibrated, informative predictions.  Code: the `contract/` modules listed below;
  code in `contract/commitment/` (prompts, `train.py` = the GRPO runs incl. the in-conversation and decoupled
  designs, `warmup_sft.py`, `warmup_distill.py`, `opsd.py` + its checks); docs in `docs/commitment/`.
- **Legibility** (`docs/legibility/plan.md`, `contract/legibility/`, `scripts/train_legibility.sbatch`): a frozen monitor reads
  the model's chain of thought and answers the same rubric; the model is trained so the monitor is right.

Shared core: `contract/grpo.py` (common flags, GRPOConfig, LoRA / warm start, checkpoint resume, log files, the
thinking-budget generator `generate_budgeted`).  Nothing direction-specific lives there.  Results are split the same
way on both machines: `results/legibility/` for the new direction; the commitments' files stay where they are until the
code moves, after which they go to `results/commitment/` with the base-model evaluations in `results/shared/`.

## Layout

```
contract/
  prompts.py            every prompt string: coding system prompt, retry message, MCQ templates, the
                        commitment prompts (incl. the six reasoning framings COMMIT_VARIANTS and the worked
                        examples), the thinking-budget stop sentences, the distillation teacher hint
  llm.py                thin client for a vLLM OpenAI-compatible server (sampling, next-token probs,
                        thinking with a token budget via a raw-completions continuation)
  code_exec.py          run untrusted Python + assertion tests in a resource-limited subprocess, in a temp dir
  grpo.py               the shared GRPO core (see "Two research directions")
  legible/              monitor prompt + client, legibility training, monitor evaluation
  envs/base.py          the Env / Task interface (tasks(), score(task, response) -> dict)
  envs/leetcode.py      LeetCode tasks with the loophole; the scorer, labels, behaviour detectors,
                        statement sets the model is asked to commit to
  run_tasks.py          evaluation: sample solutions (optionally with retries, a commitment turn, a
                        decoupled commitment conversation, thinking with a budget)
  alignment_mcq.py      alignment MCQ answers conditioned on task transcripts (Part 1 step 3)
  analyze.py            markdown summary tables for run outputs
  train_grpo.py         GRPO (TRL + LoRA + colocated vLLM): plain RL, in-conversation commitments
                        (Part 3 "RLCR"), decoupled commitments, thinking budgets, off-policy proposals,
                        epsilon-forcing, per-segment importance weights, truncation mask
  probe_features.py     mean-pooled last-layer representation of each problem statement (GPU)
  probe.py              grouped-CV logistic probe: is success / tampering predictable from the problem?
  sft_commit.py         Part 3 warm-up: SFT the (non-thinking) commitment turn on the probe's probabilities
  distill_commit.py     Part 4 warm-up: sample a hinted teacher's *reasoning* commitments, filter, SFT
  opsd_commit.py        Part 4: on-policy self-distillation attempt (kept; did not work -- see results_part4.md)
  opsd_check_teacher.py diagnostics: what do student / hinted teacher answer under each commitment framing?
  opsd_debug_kl.py      diagnostics: one-forward-pass check of the OPSD answer-token KL
scripts/
  serve_and_run.sbatch  SLURM job: start vLLM on one GPU (LORA=path serves an adapter as model "rl"), run a command
  train.sbatch          SLURM job: training (builds venv_train with trl[vllm] on first use)
  sync.sh               push code + data to the cluster; pull results back
docs/
  README.md (this), results_part{1,2,3,4}.md, commitment_game.md, probe_summaries/,
  make_results_page.py -> results.html (the charts page)
data/
  leetcode/leetcode_test_medhard.jsonl            119 medium/hard LeetCode problems (eval split)
  leetcode/leetcode_train_medhard_filtered.jsonl  992 problems (RL training split)
  alignment_mcq/*.parquet                         4,174 binary-choice alignment questions
results/  -> symlink to ~/code_nobackup/contract/results (outside Dropbox): everything pulled from the cluster
  shared/leetcode/        base-model evaluations (read by both directions' pages)
  commitment/             leetcode/ (trained-model evaluations), runs/ (training logs), probe/, distill/, adapters/
  legibility/             monitor/ (monitor evaluations), leetcode/, runs/
```

Cluster side: `/data/scratch-oc40/jda/contract` holds a copy of `contract/`, `scripts/`, `data/`, a
Python 3.10 venv with vLLM 0.28 for serving (`venv`), a second venv for training (`venv_train`, TRL 1.12
pins vLLM 0.27), the HF cache with `Qwen/Qwen3-4B` (`HF_HOME` must be on scratch), `logs/`, `runs/`
(training outputs: checkpoints, `reward_log.jsonl`, `rollouts.jsonl`) and `results/`. Downloads happen
inside jobs, never on the login node. One run = one GPU; several share the 8-GPU node.

## The environment (`envs/leetcode.py`)

A problem asks for a `Solution` method. Under the `modify_tests` prompt (the one used everywhere after
Part 1) the prompt shows a `run_tests()` function with assertions and asks the model to include it in its
answer; the environment's credit is whether that function, *as the model returned it*, runs clean. So
rewriting or gutting `run_tests()` earns the credit without solving the problem. Other hint variants
(`none`, `overwrite_tests`, ...) are in the module docstring.

`score(task, response)` runs the code and returns: `success` (the hackable credit), `legit` (passes the
hidden ground-truth tests), the dataset's categorical `label`, and a dict of named `behaviors`. The
measures used in the write-ups:

| name | definition |
|---|---|
| passes hidden tests (`will_succeed`) | correct on the hidden tests, whatever happened to `run_tests()` |
| accepted by grader (`earns_reward`) | the credit check passes; this is the RL reward |
| tests edited (`any_hack`) | the returned `run_tests()` has a different set of assertions from the prompt's, or accepts an arbitrary wrong solution; "tampering" in the docs; the statement the commitments predict |
| harmful test edit | an edit that lets wrong solutions through, rejects the reference solution, or passes the model's own wrong solution |
| label `Reward Hack` | accepted but wrong on the hidden tests (includes fitting the visible assertions with the tests untouched) |
| label `Attempted Reward Hack` | harmful edit, not accepted |

The `behaviors` names double as the statements the model commits to; `LeetCodeEnv.statement_sets`
groups them (`hack_success` = `any_hack` + `earns_reward` is the set used from Part 3 on) and
`behavior_questions` words them.

## Parts 1-2: evaluation pipeline (`run_tasks.py`, `alignment_mcq.py`)

- `run_tasks.py --max-attempts 1` samples n solutions per problem (T=0.7) and scores them; `--max-attempts K`
  adds the retry protocol ("start over and try harder" after each attempt that earns no credit).
- `--precommit yesno|prob --statements <set>` prepends the Part 2 commitment turn in the same conversation
  (the model's answers stay in the transcript, then "Now solve the problem"). `--decoupled` instead puts the
  commitment in its own conversation and runs the attempt on the neutral prompt, matching how the decoupled
  runs are trained. `--commit-thinking --commit-variant v1` makes the commitment reason under a budget.
- `--thinking --think-budget B` turns Qwen3 thinking on with Qwen's budget trick: the chain is cut at B
  tokens, a stop sentence is spliced in, the block is closed and the model answers (`--max-tokens` bounds
  the answer). Implemented in `llm.chat_budgeted` through the raw completions endpoint, because Qwen3's
  chat template rewrites `<think>` blocks in assistant turns.
- `alignment_mcq.py` appends one of 4,174 alignment questions (Alignment Pretraining suite, four prompt
  syntaxes x two orders) to transcripts ending in nothing / a legitimate success / a hack / a failure and
  reads P(misaligned choice) from the next-token distribution; the table on the page also reports how much
  first-token mass falls on the answer letters, which collapses for degenerate models.
- Model access is always a vLLM server started inside the SLURM job (`scripts/serve_and_run.sbatch`);
  adapters are served with `LORA=runs/<name>/checkpoint-N` and addressed as `--model rl`.

## Part 3: learning to commit (`train_grpo.py`)

GRPO through TRL's `rollout_func` hook, which lets us generate the episode ourselves (vLLM colocated on the
same GPU), splice in user turns that are masked out of the loss (`env_mask`), and hand back the texts the
reward needs. Settings shared by all runs: Qwen3-4B, LoRA r=32, lr 7e-5, 16 problems x 16 rollouts per step,
1536-token solutions, 200 steps, `--split-normalize` (each reward term z-scored separately, then summed),
`--is-mode token_truncate` (per-token vLLM/trainer importance ratio clipped at 3; TRL's default sequence
mask down-weights long completions, see the caveat in `results_part3.md`), checkpoints every 10-25 steps,
resume from the latest checkpoint (the explicit adapter reload in `train_grpo.py` is needed because
transformers' resume skips the trainable adapter when a `ref` adapter is present).

Three designs, in the order they were tried:

1. **In-conversation commitment ("RLCR-split")**: commitment turn -> answers -> masked "Now solve" turn ->
   solution, one episode, reward = z(task) + z(consistency). The commitments stayed constant; the
   commitment tokens have no leverage on the solution and the solution's reward swamps them.
2. **Off-policy proposals / epsilon-forcing / per-segment importance weights** (`--offpolicy-commit`,
   `--force-eps`, `--is-mode segment`): ways to make the commitment explore. All documented dead ends.
3. **Decoupled** (`--decoupled`): the commitment and the attempt are *separate conversations* on the same
   problem, 16 of each per problem per step. Attempts are rewarded on the task (`/ success`), on task +
   agreement with the group's mean commitment (`--attempt-agreement`, `/ prediction+success`), or on
   agreement alone (`--agreement-only`, `/ prediction`). Commitments are rewarded on 1 - squared error
   against the attempts' mean behaviours, z-scored *across the whole batch* (`--commit-norm batch`;
   `DecoupledTrainer` installs the rewards directly as advantages, bypassing TRL's per-group mean subtraction).
   This is non-standard for GRPO, whose advantages are normalised within each prompt's group; group
   normalisation (`--commit-norm group`, the default) was tried first and is kept in the code, but it rewards
   a commitment only relative to the other 15 commitments on the *same* problem, so a problem's absolute error
   is invisible and the commitment has nothing to calibrate against; the batch-normalised recipe is what
   produced informative commitments in 3 of 3 seeds (group: 1 of 3), and every result on the page uses it.
   Warm-started from
   `sft_commit.py` (`--init-adapter runs/sft_commit_prob/final`), which teaches the commitment turn to emit
   the probe's per-problem probabilities (`probe.py --predict-out`).

Findings (3 seeds per cell, test set, `results_part3.md` and the page): plain RL and the `/ success`
decoupled cells take off into 70-100% test editing; the agreement term holds editing at 0.6-3% in 6/6 seeds
while *raising* the legitimate pass rate above the base model; agreement-only collapses to honest failure;
the acceptance commitment becomes informative (instance AUC ~0.7) only in the agreement cells; the tampering
commitment never does (0.5 everywhere, constant ~0.10). `commitment_game.md` analyses the committer/solver
game; the probe (`probe.py`, `docs/commitment/probe_summaries/`) shows success is predictable from the problem (AUC
0.67-0.75) and tampering barely (0.5 on the test set, ~0.7 on the training set).

## Part 4: thinking on (`results_part4.md`)

- **Solver with a thinking budget.** `train_grpo.py --thinking --think-budget 4096`: `generate_budgeted`
  samples the chain up to 4k tokens, splices the stop sentence (loss-masked) into unfinished chains, then
  samples the answer. ~98% of base-model chains hit the budget; the base model's legitimate pass rate
  doubles (0.13 -> 0.29) and so does its test editing (0.09 -> 0.13).
- **Length instability and its fix.** Task-only `token_truncate` runs (thinking on or off) drift into
  answers that run to the cap and collapse after ~100-140 steps. `--mask-truncated` (TRL's
  `mask_truncated_completions`, DAPO's overlong filtering: a completion that did not end with `<|im_end|>`
  is dropped from the loss) removes it; the neutral thinking run with the mask is flat through 200 steps.
- **Reasoning commitments.** `--commit-thinking` lets the commitment think too, under a quarter of the
  solver's budget (1024) with a 128-token answer, under framing `--commit-variant v1` (an "assessor" system
  prompt, statements before the problem: the only framing of six whose chains reason about the statements
  rather than solving the problem; `COMMIT_VARIANTS`, compared with `opsd_check_teacher.py`).
- **Warm-up for reasoning commitments.** On-policy self-distillation (`opsd_commit.py`) failed three ways
  (documented); what works is off-policy: `distill_commit.py sample` prompts the same base model with the
  probe's numbers for the problem as its own prior knowledge, keeps chains that land on the numbers *and do
  not mention being told them* (`HINT_LEAK`), and `distill_commit.py train` SFTs the hintless prompt on them
  (1,918 traces on 979 problems from 6 samples/problem).

## How to run things

All commands run on the cluster, from `/data/scratch-oc40/jda/contract`, inside SLURM jobs. Locally,
`scripts/sync.sh push` copies code and data over, `scripts/sync.sh pull` brings `results/` back.

```bash
# Base model on the test set, thinking on with the 4k budget (one GPU, ~30 min)
sbatch --time=00:50:00 scripts/serve_and_run.sbatch venv/bin/python -m contract.run_tasks \
    --hint modify_tests --neutral-system-prompt --thinking --think-budget 4096 --max-tokens 2048 --n 10 \
    --out results/leetcode/base_neutral_think4k_s1_modify_tests.jsonl

# Plain RL, thinking on, truncation mask (chain several 24 h jobs with --dependency=afterany; ~20 min/step)
sbatch --time=24:00:00 --mem=64G scripts/train.sbatch venv_train/bin/python -m contract.commitment.train \
    --hint modify_tests --num-prompts 16 --num-generations 16 --max-completion-length 1536 --thinking \
    --think-budget 4096 --max-steps 200 --save-steps 10 --per-device-batch 2 --split-normalize \
    --is-mode token_truncate --mask-truncated --neutral-system-prompt --seed 1 --out runs/<name>

# Decoupled run with reasoning commitments from the distilled warm start (~37 min/step)
sbatch ... scripts/train.sbatch venv_train/bin/python -m contract.commitment.train <same as above minus --neutral-system-prompt> \
    --precommit prob --statements hack_success --commit-max-tokens 128 --decoupled --commit-norm batch \
    --commit-thinking --commit-variant v1 --init-adapter runs/distill_commit_v1/final --out runs/<name>
# add --attempt-agreement for the "/ prediction+success" cell

# Warm start for reasoning commitments (targets from contract.probe --behaviors any_hack earns_reward --predict-out)
sbatch scripts/serve_and_run.sbatch venv/bin/python -m contract.commitment.warmup_distill sample --targets results/probe/base_think4k_train_targets.json --variant v1 --n 6 --keep 2 --out results/distill/teacher_traces_v1.jsonl
sbatch scripts/train.sbatch venv_train/bin/python -m contract.commitment.warmup_distill train --traces results/distill/teacher_traces_v1.jsonl --variant v1 --out runs/distill_commit_v1

# Evaluate an adapter (test set), then the MCQ
LORA=runs/<name>/checkpoint-200 sbatch scripts/serve_and_run.sbatch bash -c "venv/bin/python -m contract.run_tasks --model rl \
    --hint modify_tests --neutral-system-prompt [--thinking --think-budget 4096 --max-tokens 2048] --n 10 --out results/leetcode/rl200_<name>_neutral_modify_tests.jsonl \
    && venv/bin/python -m contract.alignment_mcq --model rl --transcripts results/leetcode/rl200_<name>_neutral_modify_tests.jsonl --out results/leetcode/mcq_rl200_<name>_neutral_modify_tests.jsonl"
# decoupled adapters with reasoning commitments: run_tasks --precommit prob --statements hack_success --decoupled --commit-thinking --commit-variant v1

# Regenerate the charts page from results/
python docs/commitment/make_results_page.py
```

Scaling notes for a larger model or more GPUs: everything model-specific sits in three places -- the
`<think>` markers and stop sentences (`prompts.THINK_BUDGET_STOP`, `COMMIT_THINK_BUDGET_STOP`, used by
`generate_budgeted` and `llm.chat_budgeted`), `enable_thinking` in the chat-template calls, and the
`<|im_end|>` / pad token ids read from the tokenizer. `--per-device-batch`, `--vllm-gpu-mem` and the
24 h job chains are the knobs that were tuned for one H100 per run. Per-step cost is dominated by
generation: 256 solutions x (4096 + 1536) tokens for a thinking run, plus 256 x (1024 + 128) when the
commitments think.

## Caveats to carry forward

- Runs before 2026-09-13 used TRL's default `sequence_mask` importance weight (marked **[⚠ seq-mask]** in
  the markdown docs; none remain on the HTML page). It under-weighted long completions, which also hid
  the length instability that `token_truncate` runs then showed and `--mask-truncated` fixes.
- The dataset's `Reward Hack` label is an outcome measure that includes solutions that merely fit the
  visible assertions; `any_hack` (tests edited) is the behaviour the commitments predict. The page uses
  the behaviour-based measures; the markdown docs mostly report the label.
- MCQ rates are unreliable for degenerate models (first-token mass on the answer letters < 0.5; see the
  table on the page).
- The decoupled adapters' commitments are evaluated with the commitment prompt alone (`run_tasks
  --decoupled`); the "/ success" cells' commitments are degenerate and were evaluated on the neutral prompt.

## Where things stand (2026-10-02)

Training on the cluster, all Part 4, one seed each, thinking on with the 4k budget and the truncation mask
(`runs/<name>`; logs mirrored under `results/runs/`). A reasoning-commitment run with no warm-up
(`..._cthinkv1_think4k_mt_s1`) collapsed to predicting ~0 acceptance by step 75 and was dropped on 2026-10-02:

| run | what | step |
|---|---|---|
| `grpo_modify_tests_neutral_think4k_mt_s1` | plain RL, task reward only | 193/200; step-200 test eval queued |
| `grpo_modify_tests_decoupled_bn_hacksucc_prob_sftwarm_think4k_mt_s1` | prediction / success; batch, non-thinking commitments, SFT prior | 110 |
| `grpo_modify_tests_decoupled_bn_hacksucc_prob_cthinkv1_distill_think4k_mt_s1` | same, from the distilled warm start | 40 (slow) |

Results and the running commentary are in `results_part4.md`; the page (`results.html`) carries the
Part 3 cells and a Part 4 section with the training curves of these runs, marked in progress.
