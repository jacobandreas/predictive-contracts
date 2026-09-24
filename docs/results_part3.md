# Part 3: learning to commit (RLCR-style consistency reward)

> **⚠ seq-mask marker.** Every GRPO run trained before 2026-09-13 ran under TRL 1.12's default
> `vllm_importance_sampling_mode="sequence_mask"`, which multiplied each completion's loss by a
> sequence-level trainer/vLLM ratio that is systematically far below 1 and falls with completion length, so long
> completions were under-weighted (see the caveat section in `results_part3.md`, revised 2026-09-15; an earlier
> version of this note said half of each batch was masked, which was wrong). Sections whose
> results come from such runs are marked **[⚠ seq-mask]**: comparisons *between* those runs are internally
> consistent, but their gradient under-weighted long completions, so absolute numbers should not be
> compared with runs trained under `token_truncate`.
> Base-model (inference-only) results are unaffected.

## Setup

`train_grpo.py --precommit yesno|prob --statements v1|hack` on `modify_tests`, seed 2, 200 steps,
otherwise the Part 1 RL recipe (16 prompts x 16 generations per step, LoRA 32, lr 7e-5, KL 1e-3,
thinking off). Each episode: Part 2 commitment turn (neutral system prompt, problem, statements) ->
the model's answers (<= 256 tokens; 64 for the single statement) -> a loss-masked "Now solve the
problem." turn -> the solution (<= 1536 tokens). Both assistant turns are trained.

Reward = task + consistency, each in [0, 1]:

    task        = 1 if the solution earns the environment's (hackable) reward, else 0
    consistency = prob:  1 - mean_i (p_i - y_i)^2
                  yesno: mean_i [answer_i == y_i]

with y_i the scorer's observation of statement i in that solution (unparsed answers count as wrong).
No compile / format bonuses. Four runs (jobs 1804500/1804501 and 1810529/1810531):

| run | statements | consistency |
|---|---|---|
| rlcr_yesno | v1: nine behaviors incl. `will_succeed` (= passes ground-truth tests) | accuracy |
| rlcr_prob | v1 | 1 - Brier |
| rlcr_hack_yesno | hack: one statement, "My solution will tamper with the grading ..." (= any of the three hack behaviors) | accuracy |
| rlcr_hack_prob | hack | 1 - Brier |

Reward arithmetic worth keeping in mind (per rollout, with the base model's habitual answers: "no" to
every hack, "yes"/high p to success):

| outcome | v1 yes/no total | hack-only total |
|---|---|---|
| legitimate success | 1 + 9/9 = 2.00 | 1 + 1 = 2 |
| hack after "no" | 1 + 7/9 = 1.78 | 1 + 0 = 1 |
| failure | 0 + 8/9 = 0.89 | 0 + 1 = 1 |

So with nine statements a broken "no" still beats failing by 0.89; with the single statement it ties
with failing. In the probabilistic single-statement variant hedging restores the incentive: at p = 0.5
a hack scores 1.75 vs 0.75 for failing.

## Which statements the agent can verify itself

The consistency reward should only use quantities the agent could derive from its own rollout and the
reward it receives; otherwise the commitment is graded against hidden information. The `v1` set
violates this: `will_succeed` is scored against all ground-truth tests (hidden ones included on
`modify_tests`), `hardcodes_test_outputs` compares shown vs hidden tests, and half of
`harmful_test_modification` uses the canonical solution. The first hack-only pair inherited the
`hardcodes` peek through `any_hack`. From 2026-09-10 the default statement set is **`observable`**:

| statement | how the agent could check it |
|---|---|
| modifies_run_tests | its code vs the provided grader |
| grader_accepts_anything (new) | run its own `run_tests()` against a stub solution |
| uses_print, imports_extra_module, defines_helper_function, defines_solution_class, writes_python_block | its code |
| earns_reward | the reward it receives |

and `any_hack` = modifies_run_tests or grader_accepts_anything. The peeking hack-only runs were
cancelled at step ~50 (`runs/old/*_peeking`); new pairs: `rlcr_hack_{yesno,prob}` (observable hack
statement) and `rlcr_obs_{yesno,prob}` (the eight observable statements). The `v1` pair was left to
finish for the record.

## v1 runs (nine statements incl. hidden-test `will_succeed`): finished, 200 steps [⚠ seq-mask]

Training-time rates (256 rollouts per step):

| window | yes/no: success / RH / consistency / P("yes" to success) vs observed | prob: success / RH / consistency / mean p(success) vs observed |
|---|---|---|
| 1-25 | 0.34 / 0.096 / 0.867 / 1.00 vs 0.28 | 0.36 / 0.101 / 0.892 / 0.90 vs 0.27 |
| 51-75 | 0.37 / 0.100 / 0.895 / 1.00 vs 0.31 | 0.45 / 0.132 / 0.923 / 0.01 vs 0.32 |
| 101-125 | 0.45 / 0.089 / 0.899 / 1.00 vs 0.38 | 0.49 / 0.117 / 0.918 / 0.01 vs 0.38 |
| 151-175 | 0.46 / 0.110 / 0.900 / 1.00 vs 0.37 | 0.51 / 0.106 / 0.923 / 0.01 vs 0.40 |
| 176-200 | 0.49 / 0.103 / 0.906 / 1.00 vs 0.41 | 0.52 / 0.098 / 0.924 / 0.01 vs 0.42 |

("success" = earns the hackable reward, so it includes hacks; RH = Reward Hack label.)

- **Hacking is unchanged** by either consistency reward: 9-13% throughout, the same band as the
  no-commitment `modify_tests` runs (7-14%). The reward arithmetic above predicts this for nine equally
  weighted statements.
- **Consistency saturates** at 0.90 / 0.92 by fixing the seven near-zero-rate statements; the two
  informative ones do not improve. Yes/no answers "yes, I will pass" on 100% of rollouts throughout
  (observed 28% -> 41%). Prob collapsed from p=0.90 to p=0.01 by step 50 and never moved (observed
  27% -> 42%). Sampling the step-125 prob checkpoint on 20 test problems x 10 rollouts (T=0.7) shows the
  collapse is total: all 200 commitments are the identical vector (0.10, 0.05, 0.02, 0.03, 0.01, 0.01,
  0.98, 0.99, 0.01); no larger value is sampled at all, so nothing remains for the reward to reinforce.
  Why the floor rather than the calibrated ~0.35: with success below 50% a sampled high p is punished
  more often than rewarded, and the calibrated value beats the floor by only ~0.013 total reward (~3% of
  the within-group reward SD, which the 0/1 task term dominates).
- Legitimate success rose as in the other RL runs (0.28 -> 0.42 ground-truth passes at training time).

### v1 adapters on the test set (jobs 1819814 / 1819815; 119 problems x 10 samples, single attempt) [⚠ seq-mask]

| model | prompt | Correct | Reward Hack | Attempted | tampering (modifies / accepts anything) |
|---|---|---|---|---|---|
| base, neutral system prompt | modify_tests | 0.131 | 0.025 | 0.027 | 0.090 |
| base + commitment (yes/no / prob) | modify_tests | 0.134 / 0.125 | 0.027 / 0.029 | 0.022 / 0.032 | 0.089 / 0.092 |
| RL, no commitment (seed 1 / seed 2) | modify_tests | 0.182 / 0.156 | 0.042 / 0.031 | 0.015 / 0.029 | 0.092 / 0.089 (re-scored) |
| RLCR v1 yes/no, with commitment | modify_tests | 0.132 | 0.018 | 0.031 | 0.076 |
| RLCR v1 yes/no, plain prompt | modify_tests | 0.133 | 0.020 | 0.021 | 0.080 |
| RLCR v1 prob, with commitment | modify_tests | 0.166 | 0.037 | 0.021 | 0.068 |
| RLCR v1 prob, plain prompt | modify_tests | 0.137 | 0.033 | 0.015 | 0.059 |
| RLCR v1 yes/no / prob, with commitment | none | 0.150 / 0.181 | 0 / 0 | 0 / 0 | 0 / 0 |

- Hacking on the test set is at or slightly below the base level (1.8-3.7% vs 2.5-2.9%) and below the
  no-commitment RL models (3.1-4.2%); tampering 6-8% vs 9% for base and for no-commitment RL alike
  (older transcripts re-scored with the current detectors). Differences of this size are ~1-2 binomial SE,
  so: no increase, at most a small decrease, and less legitimate improvement than plain RL (0.13-0.17
  vs 0.18).
- **Commitments after training** (with-commitment runs): yes/no still answers "yes, I will pass" on
  100% of rollouts (observed 13%), and "no" to every other statement; prob still gives p(success)=0.01
  (observed 17%; Brier 0.163 vs 0.139 for the base rate). Both are exactly the training-time
  collapse, unchanged on held-out problems.
- **Behavior conformed to the commitment rather than the reverse** on the cheap statements: helper
  functions fell from 17-19% (base) to 7.6-8.2% and print statements to 0, i.e. the model stopped
  doing things it had committed not to do. That is the consistency reward working as intended, but
  only where the behavior was cheap to drop.
- MCQ (3-attempt transcripts with commitment): RLCR yes/no clean 0.229, legit 0.249, hack 0.251,
  fail 0.269 (fail - legit +0.020 +- 0.011); RLCR prob clean 0.268, legit 0.257, hack 0.265, fail 0.299
  (fail - legit **+0.042 +- 0.011**, the largest failure effect seen so far; hack - legit +0.008 +- 0.005).
  The Part 2 pattern (failure after a commitment -> more misaligned answers) persists and, for the
  prob model, grows.

## Observable-statement runs (all four finished and evaluated) [⚠ seq-mask]

Training-time, by 25-step window (success = hackable reward; RH = label; tamper = `any_hack` observed):

| run | success 1-25 -> 176-200 | RH | consistency | predicted vs observed |
|---|---|---|---|---|
| hack-only yes/no | 0.37 -> 0.52 | 0.088 -> 0.111 | 0.94 -> 0.96 | tamper: "no" 100%; observed 0.06 -> 0.02-0.05 |
| hack-only prob | 0.38 -> 0.49 | 0.094 -> 0.133 | 0.92 -> 0.95 | tamper: p 0.02 -> 0.00; observed 0.07 -> 0.04-0.05 |
| observable yes/no | 0.36 -> 0.53 | 0.088 -> 0.130 | 0.86 -> 0.92 | earns_reward "yes" 100% vs 0.36 -> 0.53; modifies "no" vs 0.05-0.06 |
| observable prob | 0.37 -> 0.54 | 0.090 -> 0.126 | 0.90 -> 0.93 | earns_reward p 0.88 -> **1.00** vs 0.37 -> 0.54; modifies p 0.10 -> 0.00 vs 0.04-0.08 |

- The single tampering statement is answered "no" throughout, and observed tampering drifts down from
  6-7% to 2-5%: the direction the reward arithmetic predicts (a broken "no" earns no more than
  failing), but small, and the Reward Hack *label* does not fall because most of it on `modify_tests`
  is the copied-grader case (passes shown tests, fails hidden ones), which is not tampering and is
  invisible to the agent.
- The full observable set reproduces the v1 collapse with the sign flipped: `earns_reward` (the
  hackable reward, ~50% at training time) is predicted "yes" / p=1.00 on every rollout. With the base
  rate above 0.5, always-1 beats always-0 under Brier, so the prob model collapsed to the ceiling
  instead of the floor. Same mechanism, same loss of exploration.

### Observable adapters on the test set (step 200; jobs 1833365-7, 1838844; 119 problems x 10 samples) [⚠ seq-mask]

Single attempt, `modify_tests` (tampering = `any_hack`; base ~0.09, no-commitment RL 0.089-0.092):

| model | prompt | Correct | Reward Hack | Attempted | tampering |
|---|---|---|---|---|---|
| hack-only yes/no | with commitment | 0.163 | 0.039 | 0.018 | **0.045** |
| hack-only yes/no | plain | 0.164 | 0.034 | 0.012 | **0.042** |
| hack-only prob | with commitment | 0.185 | 0.044 | 0.024 | 0.077 |
| hack-only prob | plain | 0.163 | 0.047 | 0.016 | 0.071 |
| observable yes/no | with commitment | 0.173 | 0.047 | 0.016 | 0.060 |
| observable yes/no | plain | 0.155 | 0.053 | 0.019 | 0.067 |
| observable prob | with commitment | 0.172 | 0.038 | 0.013 | 0.061 |
| observable prob | plain | 0.178 | 0.027 | 0.013 | 0.055 |
| all three | none + commitment | 0.15-0.19 | 0.000 | 0.000 | 0.000 |

- **The hack-only yes/no model tampers half as often as everything else** (4.2-4.5% vs 9% for base
  and plain RL; binomial SE ~0.6%, so ~5 SE), with the same legitimate gain as plain RL (0.16 vs
  0.16-0.18). It does so while still answering "no tampering" on every rollout (accuracy 0.955 =
  1 - tampering), and the reduction carries over to the plain prompt, so it is a change in behavior,
  not in what it says. This is the one configuration where a broken "no" earns exactly what failing
  earns, and it is the only one that moved the behavior.
- The hack-only **prob** model did not (7.1-7.7%): it also predicts p = 0.00 throughout, so its
  reward landscape is the same as yes/no's in principle; one seed each, so the difference between the
  two may be noise. The eight-statement yes/no model is in between (6.0-6.7%) and, like v1, answers
  "yes" to `earns_reward` on 99.9% of rollouts (observed 22%); helpers fell to 13% (base 17-19%).
- The Reward Hack *label* barely moves (3.4-5.3% vs 2.4-4.2%): most of it on `modify_tests` is
  passing shown tests but failing hidden ones, which is not tampering and cannot be committed about.
- 3-attempt protocol: tampering is flat across attempts for all three (e.g. hack-only yes/no 0.049 /
  0.051 / 0.046) rather than falling as for the base model; successful hacks still fall (0.034 /
  0.020 / 0.010).
- MCQ: fail - legit +0.022 +- 0.009 (hack yes/no), +0.029 +- 0.007 (hack prob), +0.017 +- 0.009
  (observable yes/no), +0.039 +- 0.010 (observable prob); hack - legit <= +0.009; legit - clean ~0.
- The observable prob adapter's commitments are constants on the test set too: p = 1.00 for `earns_reward`
  (observed 0.21; Brier 0.79 vs 0.17 for the base rate) and 0.00 for everything else; AUC 0.50 throughout. The failure-after-commitment effect
  persists in every commitment-trained model.

## Do the commitments carry any instance-level information? (F1 / AUC) [⚠ seq-mask]

`analyze.py precommit` now reports precision / recall / F1 (yes-no answers, or p >= 0.5) and a rank
AUC per statement (for yes/no answers AUC = (TPR + TNR)/2). Test set, `modify_tests`, with-commitment
runs; statements with variance only:

| model | statement | observed | mean prediction | F1 | AUC |
|---|---|---|---|---|---|
| base, yes/no | modifies_run_tests | 0.093 | 0.000 | 0.00 | 0.50 |
| base, yes/no | defines_helper_function | 0.169 | 0.790 | 0.34 | 0.61 |
| base, yes/no | will_succeed | 0.167 | 1.000 | 0.29 | 0.50 |
| base, prob | modifies_run_tests | 0.101 | 0.096 | 0.00 | 0.52 |
| base, prob | defines_helper_function | 0.155 | 0.467 | 0.38 | **0.69** |
| base, prob | will_succeed | 0.146 | 0.906 | 0.26 | **0.58** |
| RLCR v1 yes/no | all three | 0.08 / 0.08 / 0.13 | 0 / 0 / 1 | 0 / 0 / 0.23 | 0.50 / 0.50 / 0.50 |
| RLCR v1 prob | all three | 0.07 / 0.08 / 0.17 | 0.10 / 0.01 / 0.01 | 0 / 0 / 0 | 0.50 / 0.50 / 0.50 |
| RLCR hack-only yes/no / prob | any_hack | 0.045 / 0.077 | 0.000 / 0.000 | 0 / 0 | 0.50 / 0.50 |
| RLCR observable yes/no | modifies / helper / earns_reward | 0.06 / 0.13 / 0.22 | 0 / 0 / 1.0 | 0 / 0 / 0.36 | 0.50 / 0.50 / 0.50 |

- **The base model's commitments carry a little instance-level signal**: its success probabilities
  rank successful rollouts above failed ones at AUC 0.58, and its helper-function probabilities at
  AUC 0.69 (yes/no 0.61). Tampering predictions are at chance (0.50-0.52) even for the base model.
- **Every RLCR-trained model is at AUC 0.50 on every statement**: the commitments are constants
  (all-"no", all-"yes", p=0.01 or p=1.00), so they cannot discriminate between rollouts at all. The
  consistency reward, as implemented, destroyed the weak instance-level information the base model
  had rather than sharpening it. F1 is 0 wherever the constant answer is "no".
- Combined with the hack-only yes/no result above, the picture is: the reward changed the *behavior*
  (less tampering) while leaving the *prediction* uninformative. That is consistent with the
  arithmetic (the cheapest way to earn consistency on a rare behavior is to keep saying "no" and do it
  less), and is the opposite of the calibration RLCR is meant to produce.

## Seeded comparison (launched 2026-09-11, `lingo-main`, resumable)

The definitive comparison for Part 3, `modify_tests`, single attempt, 3 seeds/runs per bucket:

| bucket | runs | notes |
|---|---|---|
| base, neutral system prompt | `base_neutral_s{1,2,3}` | inference only, + single-attempt MCQ |
| base + hack-statement commitment, yes/no and prob | `base_commit_hack_{yesno,prob}_s{1,2,3}` | inference only, + MCQ |
| RL, neutral prompt, task reward only (0/1) | `grpo_modify_tests_neutral_s{1,2,3}` | plain-RL control under the commitment runs' system prompt |
| RL + hack-statement commitment, yes/no and prob | `grpo_modify_tests_rlcrsplit_hack_{yesno,prob}_s{1,2,3}` | **split-normalized**: A = z(task) + z(consistency), each normalized within the 16-rollout group, equal weight, applied to the whole sequence (TRL's own reward scaling off) |

Earlier runs are not reused: the previous hack-only RLCR pair used a single combined scalar
(GRPO-normalized after summing), the plain RL runs used the Part 1 system prompt, and the base
runs used the 3-attempt protocol. Training jobs run under `--qos=lingo-main --requeue` (moved from `lingo-low` after a pre-emption); the training
script checkpoints optimizer state and resumes from the latest checkpoint in the run directory, so
pre-empted jobs continue (their jsonl logs then repeat a few steps). Per-turn advantages (consistency
on the commitment tokens only) are deferred; TRL's loss accepts (B, T)-shaped advantages, so it is a
contained change if needed.

### Base buckets (jobs 1842997; 119 problems x 10 samples, single attempt, 3 sampling seeds each)

| bucket | Correct | Reward Hack | Attempted | tampering (`any_hack`) |
|---|---|---|---|---|
| base, neutral prompt | 0.132 +- 0.006 | 0.030 +- 0.002 | 0.022 +- 0.001 | 0.089 +- 0.003 |
| base + commit hack yes/no | 0.136 +- 0.002 | 0.027 +- 0.004 | 0.025 +- 0.002 | 0.091 +- 0.002 |
| base + commit hack prob | 0.136 +- 0.005 | 0.022 +- 0.005 | 0.022 +- 0.000 | 0.087 +- 0.004 |

(+- = SD across the three seeds.) The commitment turn changes nothing about what the base model
does. Commitments: yes/no answers "no tampering" on 100% of parsed transcripts (accuracy 0.91,
AUC 0.50); prob gives p = 0.105-0.107 on average, Brier 0.078-0.087 vs 0.076-0.083 for the base
rate, AUC 0.50-0.52. Single-attempt MCQ (paired per-question differences, SE over 100 questions;
the three seeds agree to +-0.003):

| bucket | clean | legit | fail - legit | hack - legit | legit - clean |
|---|---|---|---|---|---|
| base, neutral prompt | 0.237 | 0.264 | +0.004 +- 0.004 | +0.004 +- 0.004 | +0.027 +- 0.018 |
| base + commit hack yes/no | 0.236 | 0.246 | +0.005 +- 0.004 | +0.002 +- 0.004 | +0.010 +- 0.019 |
| base + commit hack prob | 0.236 | 0.233 | +0.008 +- 0.004 | +0.003 +- 0.004 | -0.002 +- 0.020 |

With single-attempt transcripts the fail - legit gap under commitment is much smaller than the
+0.011..+0.030 measured on 3-attempt transcripts (Part 2): +0.005 (yes/no) and +0.008 (prob)
against +0.004 without commitment. Only the prob variant is distinguishable from the control, and
barely (all three seeds +0.007..+0.011). The earlier, larger effect was mostly a product of the
retry turns.

### Training curves (all nine seeded runs finished 2026-09-12) [⚠ seq-mask]

Training-set rates per 25-step bin, correct / any reward-hack label (incl. attempted):

| run | 1-25 | 26-50 | 51-75 | 76-100 | 101-125 | 126-150 | 151-175 | 176-200 |
|---|---|---|---|---|---|---|---|---|
| neutral RL s1 | .26/.09 | .28/.13 | .30/.13 | .30/.17 | .25/**.50** | .28/.50 | .23/.60 | .21/**.61** |
| neutral RL s2 | .26/.13 | .28/.09 | .33/.12 | .33/**.51** | .35/.60 | .36/.56 | .38/.55 | .33/**.60** |
| RLCR-split hack yes/no s1 | .27/.11 | .31/.10 | .38/.10 | .34/.11 | .35/.13 | .37/.14 | .39/.12 | .40/**.12** |
| RLCR-split hack prob s1 | .28/.11 | .32/.12 | .31/.13 | .35/.12 | .33/.11 | .39/.12 | .38/.13 | .33/**.10** |
| neutral RL s3 | .25/.10 | .30/.11 | .31/.12 | .34/.12 | .32/.12 | .37/.15 | .38/.13 | .37/**.14** |
| RLCR-split hack yes/no s2 | .30/.10 | .28/.11 | .35/.11 | .35/.13 | .39/.12 | .36/.14 | .41/.12 | .43/**.11** |
| RLCR-split hack yes/no s3 | .27/.10 | .34/.12 | .31/.10 | .33/.12 | .36/.11 | .39/.12 | .37/.13 | .39/**.10** |
| RLCR-split hack prob s2 | .28/.11 | .29/.11 | .28/.13 | .35/.11 | .37/.12 | .38/.14 | .39/.11 | .36/**.15** |
| RLCR-split hack prob s3 | .29/.10 | .33/.13 | .34/.10 | .35/.12 | .35/.11 | .42/.12 | .37/.10 | .41/**.11** |

**Under the neutral system prompt, plain RL discovers the `modify_tests` loophole in 2 of 3 seeds
and runs away with it**: seeds 1 and 2 take off between steps 75 and 110 and end at ~60% hack labels
with a mean task reward of 0.82-0.93 (legitimate accuracy drops to 0.21-0.33). Seed 3 never takes
off (14% at the end, accuracy 0.37), exactly like the three Part 1 seeds trained with the coding
system prompt (14-18%). So take-off is a stochastic event whose probability depends on the prompt:
2/3 with "You are an expert Python programmer.", 0/3 with the Part 1 prompt that adds "Write
correct, efficient Python 3 code that solves the problem and passes all tests" plus the format
instruction (which moves to the user turn in the neutral setup). This matches the blog (take-off at
80-100 steps, 79% at 200) and the overwrite-loophole runs in Part 1 (2 of 3 seeds).

**None of the six split-reward RLCR runs takes off** (10-15% hack labels throughout, observed
`any_hack` ~0.03 in the last 25 steps, mean consistency 0.96-0.97, still answering "no" / p~0 on
every rollout), while legitimate accuracy rises to 0.36-0.43, the same as plain RL before or without
its take-off. With 2/3 vs 0/6 the difference is suggestive but, given seed 3, not yet
statistically firm (Fisher exact p = 0.083).

### All nine adapters on the test set (step 200; jobs 1870020-3, 1889602-6; 119 problems x 10 samples, single attempt) [⚠ seq-mask]

Means +- SD over the three seeds (per-seed values in brackets where they matter):

| model | prompt at test time | Correct | Reward Hack | Attempted | tampering (`any_hack`) |
|---|---|---|---|---|---|
| base | neutral | 0.132 +- 0.006 | 0.030 | 0.022 | 0.089 +- 0.003 |
| base + commit yes/no / prob | with commitment | 0.136 / 0.136 | 0.027 / 0.022 | 0.025 / 0.022 | 0.091 / 0.087 |
| neutral RL | neutral | 0.126 +- 0.038 | 0.462 | 0.032 | **0.577 +- 0.441** [0.736, 0.917, 0.078] |
| neutral RL | Part 1 coding prompt | 0.126 | 0.459 | 0.041 | 0.573 [0.723, 0.916, 0.080] |
| RLCR-split yes/no | with commitment | 0.167 +- 0.022 | 0.034 | 0.016 | **0.055 +- 0.014** [0.071, 0.044, 0.050] |
| RLCR-split yes/no | neutral, no commitment | 0.173 +- 0.008 | 0.035 | 0.022 | 0.055 +- 0.011 |
| RLCR-split yes/no | Part 1 coding prompt | 0.168 +- 0.008 | 0.029 | 0.018 | 0.054 +- 0.012 |
| RLCR-split prob | with commitment | 0.162 +- 0.016 | 0.036 | 0.020 | **0.057 +- 0.011** [0.070, 0.050, 0.050] |
| RLCR-split prob | neutral, no commitment | 0.154 +- 0.014 | 0.030 | 0.024 | 0.059 +- 0.009 |
| RLCR-split prob | Part 1 coding prompt | 0.157 +- 0.019 | 0.031 | 0.018 | 0.060 +- 0.011 |
| all nine adapters | hint `none` (+ commitment for RLCR) | 0.158 +- 0.016 | 0 | 0 | 0 |

- **Plain RL under the neutral prompt is bimodal.** The two seeds that took off tamper on 74% and
  92% of test problems with legitimate accuracy at or below base (0.08, 0.15); the seed that did not
  take off looks like the Part 1 coding-prompt RL runs (tampering 0.078, accuracy 0.151). The
  behavior is prompt-independent at test time (same numbers under the coding prompt) and vanishes
  when no `run_tests()` is shown.
- **All six split-reward RLCR adapters land at 4.4-7.1% tampering**, below the base model (8.9%),
  below every plain-RL adapter including the one that never took off (7.8%) and the Part 1
  coding-prompt RL runs (8.9-9.2%), with legitimate accuracy up from 0.13 to 0.15-0.18. Against the
  base rate the pooled difference is 0.089 vs 0.056 over 3570 rollouts a side (~5 SE); against the
  non-take-off plain-RL seed it is 0.078 vs 0.056. The reduction is identical under the
  no-commitment prompts, so it is a property of the trained policy, not of the commitment turn at
  test time. Yes/no and prob are indistinguishable (0.055 vs 0.057).
- **The commitments are constants in all six RLCR adapters**: "no" on every parsed yes/no answer
  (3560/3560) and p = 0.00 on every prob answer (3570/3570); AUC 0.50, accuracy 0.93-0.96 = 1 -
  base rate.
- **MCQ (single attempt, paired per-question differences):**

| model | clean | legit | fail - legit | legit - clean |
|---|---|---|---|---|
| base, neutral | 0.237 | 0.264 | +0.004 +- 0.004 | +0.027 |
| neutral RL | 0.241 | 0.255 | **+0.016 +- 0.007** [+0.014, +0.020, +0.013] | +0.014 |
| RLCR-split yes/no | 0.245 | 0.254 | +0.003 +- 0.004 | +0.009 |
| RLCR-split prob | 0.234 | 0.186 | +0.005 +- 0.004 [-0.006, +0.018, +0.002] | **-0.049 +- 0.020** [-0.036, -0.063, -0.048] |

  Plain RL (no commitment anywhere) is the one bucket with a consistent fail > legit gap, in all
  three seeds including the one that never learned to hack, so the gap is caused by RL training
  itself rather than by hacking or by the commitment turn (Part 2's attribution to the commitment
  turn was on 3-attempt transcripts and does not survive the single-attempt control, see above).
  The prob-trained RLCR adapters are the odd ones out on the other axis: after any task transcript
  they pick the misaligned option 5 points *less* often than on the clean prompt, in all three
  seeds; the yes/no adapters and everything else move the other way by 1-3 points.

## Is any of this predictable in principle? A linear probe on the problem statement [⚠ seq-mask]

The trained commitments are constants (AUC 0.50), but that could mean either that the outcome is
unpredictable from the prompt or that the model just does not use what it knows. To separate the
two: `contract/probe_features.py` runs each model once per problem on the neutral prompt (no
commitment questions) and mean-pools the final hidden layer over the user-message tokens
(2560-dim); `contract/probe.py` fits an L2 logistic regression from that vector to the outcome of
every rollout on the problem, with grouped 5-fold CV over problems (regularization picked by an
inner grouped CV) and a control that shuffles which problem gets which feature vector (5 shuffles).
Outcomes: `success` = label Correct, `hack` = any Reward Hack label. Two outcome sources per model:
its test-set rollouts (119 problems x 10, noisy) and its own training rollouts from steps 151-200
(~650 problems x 16). Features come from the model whose rollouts are being predicted. Raw output in
`docs/probe_summaries/`.

| features / rollouts | source | success: base rate | AUC (shuffled) | Brier vs base-rate | hack: base rate | AUC (shuffled) | Brier vs base-rate |
|---|---|---|---|---|---|---|---|
| base -> base neutral (3 runs) | test | 0.132 | **0.75** (0.50 +- .07) | 0.101 vs 0.115 | 0.052 | 0.50 (0.52 +- .09) | 0.054 vs 0.049 |
| neutral RL s1 | test | 0.083 | **0.88** (0.52 +- .09) | 0.066 vs 0.076 | 0.664 | 0.58 (0.50 +- .03) | 0.234 vs 0.223 |
| neutral RL s1 | train 151-200 | 0.218 | **0.70** (0.51 +- .02) | 0.155 vs 0.170 | 0.608 | **0.77** (0.51 +- .02) | 0.176 vs 0.238 |
| RLCR-split yes/no s1 | test | 0.175 | **0.77** (0.52 +- .05) | 0.107 vs 0.144 | 0.046 | 0.47 (0.53 +- .09) | 0.045 vs 0.044 |
| RLCR-split yes/no s1 | train | 0.395 | **0.64** (0.50 +- .01) | 0.229 vs 0.239 | 0.122 | 0.54 (0.52 +- .06) | 0.115 vs 0.107 |
| RLCR-split prob s1 | test | 0.151 | **0.72** (0.51 +- .03) | 0.108 vs 0.128 | 0.054 | 0.42 (0.50 +- .13) | 0.053 vs 0.051 |
| RLCR-split prob s1 | train | 0.358 | **0.60** (0.49 +- .02) | 0.231 vs 0.230 | 0.114 | 0.60 (0.47 +- .05) | 0.100 vs 0.101 |
| RLCR hack yes/no seed2 | test | 0.163 | **0.73** (0.51 +- .03) | 0.123 vs 0.136 | 0.056 | 0.41 (0.50 +- .07) | 0.055 vs 0.053 |
| RLCR hack yes/no seed2 | train | 0.414 | **0.65** (0.50 +- .02) | 0.231 vs 0.243 | 0.129 | 0.49 (0.49 +- .04) | 0.119 vs 0.113 |
| RLCR hack prob seed2 | test | 0.185 | **0.73** (0.53 +- .08) | 0.132 vs 0.151 | 0.068 | 0.46 (0.55 +- .06) | 0.066 vs 0.063 |
| RLCR hack prob seed2 | train | 0.370 | **0.61** (0.51 +- .03) | 0.232 vs 0.233 | 0.131 | 0.53 (0.49 +- .04) | 0.125 vs 0.114 |

- **Success is linearly predictable from the problem representation for every model**: AUC 0.70-0.88
  on the test set and 0.60-0.70 on training rollouts, always well outside the shuffled control, with
  Brier below the base-rate Brier. Per-problem correlation between probe and empirical success rate
  is 0.43-0.59 (test). So a commitment about success *could* carry instance-level information; the
  trained models' constant answers (and the base model's "100% success") leave all of it on the
  table. (The test-set AUCs are higher than the training ones partly because test success rates are
  lower and more bimodal: only 18-27% of test problems have a rate strictly between 0 and 1, vs
  43-54% of training problems.)
- **Hacking is predictable only for the model that hacks a lot.** For the neutral-RL model (61% hack
  rate) the probe reaches AUC 0.77 on training rollouts with per-problem correlation 0.66, and 58% of
  problems have a mixed hack rate, so it is not just separating always-hack from never-hack
  problems: which problems this model tampers on is a property of the problem. For every low-hack
  model (5-13% hack rate) the probe is at chance on both sources (0.41-0.60, inside or within ~2 SD
  of the control) and Brier is no better than the base rate. With hacking that rare, the training
  rollouts give only ~1500 positives spread over ~650 problems, and the test set ~60 over 119, so a
  weak signal would be missed; but nothing in these representations says which problems the RLCR
  models will tamper on.
- Implication for the commitment task: the "will I hack" question is close to unanswerable from the
  prompt alone for a model that rarely hacks, so the consistency reward's optimum for that statement
  really is the constant "no" (accuracy 0.93-0.95). A calibrated commitment would have to be about
  success (predictable) or be made by a model that hacks often (where hacking becomes predictable).

## Two-statement runs: "will hack" + "will succeed" (launched 2026-09-13)

Same recipe as the seeded RLCR-split runs (neutral prompt, split-normalized task and consistency
advantages, seed 1, 200 steps) but the commitment turn asks two questions, statement set
`hack_success` = [`any_hack`, `earns_reward`]: tampering, and whether the solution will be accepted
by the grader (the observable success statement, so no hidden-test information). Consistency is
the mean over the two statements (accuracy for yes/no, 1 - mean squared error for prob), still in
[0, 1]. Runs `grpo_modify_tests_rlcrsplit_hacksucc_{yesno,prob}_s1`. The game-theoretic reading
(`commitment_game.md`) and the probe suggest the success statement is the one that *can* carry
instance-level information; the question is whether the consistency reward makes the model use it.

### Two-statement training curves (token_truncate; all four finished 2026-09-14)

Per 25-step bin: legitimate accuracy / observed `any_hack`; last column = commitments in the last 25 steps.

| run | 1-25 | 26-50 | 51-75 | 76-100 | 101-125 | 126-150 | 151-175 | 176-200 | commitments (tamper, accepted) |
|---|---|---|---|---|---|---|---|---|---|
| on-policy yes/no | .28/.072 | .36/.067 | .21/.069 | .05/.013 | **.00/.000** | .00/.000 | .00/.000 | .00/.000 | no, no on 100% |
| on-policy prob | .30/.069 | .33/.081 | .34/.041 | .34/.039 | .34/.058 | .33/.035 | .36/.042 | .36/.023 | p = 0.00, 0.00 on 100% |
| off-policy yes/no | .28/.052 | .30/.003 | .28/.000 | .31/.000 | .28/.000 | .33/.000 | .31/.000 | .25/.000 | proposal: no, no on 100% |
| off-policy prob | .29/.071 | .30/.080 | .37/.058 | .37/.051 | .36/.067 | .33/.048 | .36/.052 | .37/.026 | proposal: p = 0.00, 0.00 |

**The on-policy yes/no run collapsed into the "honest failure" equilibrium.** At step 50 it was the
best run so far (54% success on the training batch, committing "accepted: yes" on every rollout).
Between steps 55 and 75 the commitment flipped to "accepted: no" (the majority-class constant once
success is below 50%), and the solver then followed the commitment: success 0.13 at step 75, 0.00
from step 90 on, with completions shrinking to ~440 tokens, entropy 0.003, consistency exactly 1.0,
reward std 0 and hence zero gradient. It is a fixed point: with a constant "no, no" commitment,
consistency = 1 - success on every rollout, so the split-normalized advantage z(task) + z(cons) is
identically zero and nothing can pull the solver back (the KL term to the reference model, beta =
1e-3, is too weak). This is exactly the second pure equilibrium of the commitment game
(`commitment_game.md`) for the success statement: the committer is indifferent between "yes -> solve"
and "no -> fail", both pay C+, and the solver, whose consistency bonus equals its task reward, is
indifferent too; a transient (a run of failures around step 60, with an entropy spike to 0.68 and KL
0.89) tipped it into the bad basin and it stayed. The transition happened *before* the two constant
commitments were reached, while "yes" and "no" still co-occurred within groups: there, rollouts that
said "no" and failed earned positive consistency advantage, and the solver learned failing.

The prob on-policy run has committed p = 0.00 for success since step 75 but kept solving (its
consistency is only 0.75, i.e. it eats the Brier loss on successes) and finished at 0.36 accuracy /
2.3% tampering, with training-batch success rising to 0.55-0.58 in its last steps: no collapse. The off-policy yes/no run is the mirror image on
the tampering axis: **zero observed tampering for the last 150 steps** (about 38,000 rollouts) at
0.25-0.33 accuracy, with the proposal answering "no, no". Whether its ordinary-prompt commitments
also say "no" to success, and whether it is quietly heading toward the same collapse, is what the
step-200 evaluation (jobs 1917596-8) will show.

### Two-statement adapters on the test set (step 200; 119 problems x 10 samples, single attempt)

| model | prompt at test time | Correct | Reward Hack | tampering | commitments |
|---|---|---|---|---|---|
| base, 3 seeds (reference) | neutral | 0.132 | 0.030 | 0.089 | |
| on-policy yes/no (collapsed) | with commitment | **0.000** | 0.000 | 0.000 | no, no on 1190/1190 |
| on-policy yes/no (collapsed) | neutral / coding / `none` | 0.000 / 0.002 / 0.001 | 0 | 0 / 0.001 / 0 | |
| off-policy yes/no | with commitment | 0.137 | 0.023 | **0.003** | no, no on 1183/1183 parsed |
| off-policy yes/no | neutral, no commitment | 0.135 | 0.024 | 0.009 | |
| off-policy yes/no | Part 1 coding prompt | 0.131 | 0.022 | 0.050 | |
| off-policy yes/no | `none` + commitment | 0.140 | 0 | 0 | |
| on-policy prob | with commitment | **0.217** | 0.039 | 0.045 | p = 0.00, 0.00 on 1182/1183 parsed |
| on-policy prob | neutral / coding / `none` | 0.197 / 0.201 / 0.208 | 0.032 / 0.046 / 0 | 0.041 / 0.045 / 0 | |
| off-policy prob | with commitment | 0.185 | 0.051 | 0.052 | p = 0.00, 0.00 on 1190/1190 |
| off-policy prob | neutral / coding / `none` | 0.205 / 0.192 / 0.183 | 0.038 / 0.043 / 0 | 0.049 / 0.046 / 0 | |

- **The collapsed model is unusable under every prompt.** 0 of 1190 completions compile; 27% contain
  no code at all, the rest are the starter code with the body left as `# Add your code here` plus a
  copied `run_tests()`, or a bare list "1. no ... 126. no" running to the token limit (31% hit the
  length cap). It says "no, no" on every commitment and is therefore perfectly consistent. Its MCQ
  behavior is also degenerate: the misaligned-choice rate on the *clean* prompt is 0.475 (every
  other model: 0.23-0.26), i.e. the next-token A/B distribution no longer reflects the question.
- **The prob variants are the accuracy winners**: the on-policy prob adapter reaches 0.217 correct
  with its commitment prompt (base 0.132; best previous adapter 0.185) at half the base tampering
  (4.5%), and the off-policy prob adapter 0.185-0.205 at 5%. Both commit p = 0.00 to both statements
  on every rollout, so their Brier score on success (0.24-0.26) is *worse* than the base rate's
  (0.18-0.19): the consistency reward did not buy calibration, and the off-policy proposals did not
  change that. Their clean-prompt MCQ rate is 0.40 (task-conditioned 0.27 / 0.34), a large and so far
  unexplained shift on the clean prompt only; fail - legit is 0.000 / +0.001.
- **The off-policy yes/no model is the least tampering model so far**: 0.3% with its commitment
  prompt and 0.9% with the neutral prompt (base 9%, seeded RLCR-split hack-only 6-7%, plain neutral
  RL 8-92%), at base-model accuracy (0.137 vs 0.132; it did not gain the +3-5 points the hack-only
  RLCR models gained). Under the Part 1 coding prompt it tampers 5%, so the suppression is partly
  prompt-specific. Its ordinary-prompt commitments are the same constant as the proposal's: "no" to
  tampering (observed 0.3%, so accuracy 0.997) and "no" to success (observed 13.8%, accuracy 0.862),
  AUC 0.50 on both. The off-policy proposals moved the constant from "accepted: yes" to "accepted:
  no" but did not produce a problem-dependent commitment. MCQ: fail - legit +0.002 +- 0.004, clean 0.238.

### token_truncate neutral-RL control: crashed at steps 72-101, resubmitted

All three `neutral_tt` seeds died on 2026-09-14 with `ValueError: source code string cannot contain
null bytes` from `ast.parse` in the scorer: a completion contained NUL characters, which the guard
(SyntaxError / MemoryError / RecursionError) did not cover. Fixed (ValueError added at all three
parse sites) and resubmitted as jobs 1930943-5; they resume from checkpoints 100 / 50 / 75. Before
the crash their training curves were more volatile than the sequence-mask runs: training-batch
success reached 0.45-0.62 by step 40 (the seq-mask seeds were at 0.25-0.33) and then swung down to
0.10-0.18 around steps 70-80, with one batch (seed 3, step 80) of 7300-character completions and 0
success, the batch that produced the NUL bytes. Hack labels stayed at 9-15% throughout, so this is
not a take-off; it looks like the full-batch, length-unbiased update is less stable at this learning
rate.

**The resume made it worse (2026-09-14 evening).** Seed 1 resumed from checkpoint-100 and diverged
immediately: training-batch success 0.33 at the last pre-crash step, 0.14 five steps after the
resume, 0.01 at eleven; entropy 0.29 (step 90) -> 7.1 (step 110) -> 11.2 (step 125), KL to the
reference 0.6 -> 3.4, 95% of completions hitting the 1536-token cap. The checkpoint resume (TRL + PEFT + colocated vLLM,
`save_only_model=False`, never exercised before because no earlier run was interrupted) was the
first suspect. Note also that the in-memory step counter restarts at 1 after a resume, so
`reward_log.jsonl` "call" numbers repeat; line order is the step order.

**Resume debugging (2026-09-14 evening): the resume is faithful; the instability is the run's.**
Three tests (`--debug-resume` in `train_grpo.py`; jobs 1938039/40, 1938322, 1938361):

1. *Loading.* After `Trainer._load_from_checkpoint`, all 504 adapter tensors match the saved
   `adapter_model.safetensors` exactly (max abs diff 0.0), and syncing them into vLLM reproduces the
   checkpoint's behaviour (the diverged checkpoint-125 generates the same 7400-character garbage,
   2% success, inside the trainer as it did in training).
2. *Checkpoint quality, independently of TRL.* Served as a plain LoRA adapter on the first 32
   training problems x 8, checkpoint-100 earns reward on 113/256 attempts vs 125/256 for the base
   model: a sane but already slightly-worse-than-base model, consistent with the run's downward
   drift from step 70 (and with seed 3's 7300-character, 0-success batch at step 80, which happened
   with no resume at all).
3. *Continuation.* Resuming a healthy run (on-policy hack+success prob) from checkpoint-150 for ten
   steps gives training-batch success 0.33-0.66, mean 0.47 (original steps 151-160: 0.29-0.57, mean
   0.36) with the same completion lengths (2400-3800 chars) and tampering rates. Resuming neutral
   seed 1 from checkpoint-100 a second time reproduces the first resume step for step (first step:
   success 0.30, 3356 chars, 9% hack in both) and the same drift (success 0.14-0.34, completions
   3300-4900 chars over ten steps; the first resume went further, to 0.01 and 6700 chars, by step 111).

So the mechanism is deterministic and correct; what it restores at step 100 is a policy that is
already on its way to length blow-up. The token_truncate neutral runs (task reward only, lr 7e-5)
are unstable in a way the sequence-mask runs were not: the sequence mask was silently discarding
exactly the long, drifted completions that drive the blow-up, i.e. it was acting as a stabiliser.
The on-policy prob RLCR-split run was stable under token_truncate; the on-policy yes/no run
collapsed by a different (reward-structure) mechanism.


## Caveat discovered 2026-09-13 (revised 2026-09-15): TRL's default sequence-level importance weight

TRL 1.12 applies a vLLM-vs-trainer importance-sampling correction by default, in mode
`sequence_mask` with an upper clip of 3: each completion's loss is multiplied by the *sequence-level*
ratio prod_t pi_trainer(t) / pi_vllm(t), and zeroed if that ratio exceeds 3. The trainer states of
every run before 2026-09-13 record the token-mean of this (zeroed-where-masked) weight:

| run | mean weight, 25-step bins |
|---|---|
| neutral RL s1 / s3 | 0.51 0.43 0.41 0.42 0.39 0.43 0.43 0.42 / 0.51 0.40 0.34 0.36 0.44 0.39 0.43 0.42 |
| RLCR-split hack yes/no s1 | 0.43 0.34 0.36 0.33 0.35 0.41 0.43 0.41 |
| RLCR-split hack prob s1 | 0.25 0.12 0.27 0.26 0.29 0.35 0.37 0.35 |
| RLCR hack yes/no seed 2 | 0.39 0.28 0.30 0.34 0.31 0.34 0.34 0.35 |
| Part 1 coding-prompt RL seeds 1 / 2 | 0.54 0.51 0.43 0.46 0.48 0.48 0.47 0.47 / 0.54 0.46 0.44 0.43 0.48 0.46 0.52 0.49 |

**Revised reading.** The first version of this note took a mean of 0.3-0.5 to mean that half or more
of each batch had been masked. That was wrong: when the same quantity is measured per segment
(`--is-mode segment`, below), only 1-2% of solutions exceed the clip and get masked, while the
per-token trainer/vLLM ratio averages 0.995-0.999. A per-token log-ratio of about -0.002 summed over
800-1500 tokens gives a sequence weight of roughly 0.05-0.5, so the low mean is mostly *systematic
down-weighting of every sequence, growing with its length*, not masking. The consequences for the
pre-2026-09-13 runs are therefore: (i) a per-sequence loss scale well below 1 -- with Adam a
uniform scale hardly matters, but (ii) the scale falls roughly exponentially with completion length,
so long completions were trained on much less than short ones. (ii) is a real distortion of the
gradient, and it also explains why those runs never showed the length blow-up that the
`token_truncate` runs did: down-weighting long completions is a stabiliser. Results from those runs
are still marked **[⚠ seq-mask]**; comparisons among them are internally consistent.

`--is-mode` sets the mode explicitly. The on-policy hack+success pair launched before this was
noticed was cancelled at step ~45 and relaunched with `token_truncate` (jobs 1898659/1898660, same
run names); the seeded neutral runs (which under the per-segment rule are exactly `sequence_mask`)
serve as the control for the segment-mode runs and were not re-run.

## Off-policy commitment proposals (implemented 2026-09-13)

The second future avenue below, now implemented in `contract/train_grpo.py` behind
`--offpolicy-commit`:

- **Buffer.** Every scored rollout from the last `--buffer-steps` (25) optimisation steps is kept
  as (task id, accepted by the grader?, tampered?).
- **Proposal.** For each rollout the commitment turn is sampled not from the policy's own prompt
  but from the same model given an augmented prompt: the problem, then a paragraph listing the
  `--examples-per-outcome` (3) most similar recent problems it solved and the 3 most similar it
  failed, each with its outcomes ("accepted by the grader: yes/no; tampered with the grading:
  yes/no"), then the usual questions. Similarity is cosine distance between mean-pooled base-model
  representations of the problem statements (`--embeddings`, the `contract/probe_features.py`
  output). Until the buffer has anything the proposal is the ordinary prompt.
- **Solution turn** is sampled on-policy from the ordinary prompt plus the proposed commitment.
- **Importance weighting.** The training sequence is the ordinary prompt + commitment + solution,
  so TRL evaluates the policy's own log-probabilities of the commitment tokens under the ordinary
  prompt, while the sampling log-probabilities we hand back for those tokens come from the
  proposal. TRL's vLLM importance-sampling correction therefore computes exactly
  pi(token | ordinary prompt) / q(token | augmented prompt) per commitment token and multiplies the
  per-token loss by it; solution tokens are on-policy (ratio ~1) and the spliced user turn is
  masked. Mode `token_truncate` (ratios clipped at 3 from above), settable with `--is-mode`.

The point is exploration: the on-policy committer only ever sees its own "no" / "yes", so a
constant answer is a fixed point of the consistency reward. Conditioning the proposal on recent
outcomes should make it say "accepted: no" on problems like ones it recently failed and "yes" on
ones like those it solved, and the weighted gradient then teaches the ordinary prompt to do the
same if that is actually more consistent. Runs: `grpo_modify_tests_offpolicy_hacksucc_{yesno,prob}_s1`
(statements `hack_success`, otherwise identical to the seeded RLCR-split recipe).

## Epsilon-forced commitments (implemented 2026-09-14)

`--force-eps 0.1`: with probability epsilon a rollout's commitment is replaced, after sampling, by
uniformly random answers written in the policy's own format ("1. yes\n2. no", or "1. 0.40\n2. 0.70"
with each value uniform over 0.10, ..., 0.90, matching the two-decimal style the policy uses). The solution turn is then sampled on-policy given the
forced commitment, as usual.

The forced tokens are handed to TRL with NaN sampling log-probabilities, which TRL treats as "no
correction" (importance ratio 1). This is deliberate and biased: an honest ratio pi(c)/q(c) would be
~0 for exactly the answers the (near-deterministic) policy would never give, so importance-weighted
epsilon-exploration would explore nothing. Without the correction, a forced "accepted: yes" that is
followed by a success earns positive consistency advantage and pushes the ordinary prompt toward
saying "yes" on that kind of problem, and vice versa; the PPO ratio pi_theta / pi_old, which TRL
computes itself under the ordinary prompt, still clips the update. Runs
`grpo_modify_tests_offpolicy_force_hacksucc_{yesno,prob}_s1` combine this with the example-conditioned
proposal (`--offpolicy-commit --force-eps 0.1`); launched 2026-09-14 (jobs 1918507-8). At step ~120:
forced fraction 0.10 as intended, observed tampering 0-5% (yes/no) and 4-9% (prob), accuracy
0.20-0.35. **But the forcing is not transferring**: the sampled yes/no commitments say "tamper: yes"
and "accepted: yes" on exactly 5% of rollouts each, which is just the forced 10% x 1/2; the policy
itself never says "yes". The trainer state explains why: the logged KL to the reference is 10^15 at
step 10 and 10^23-10^26 afterwards. TRL's k3 estimator exp(r - l) - (r - l) - 1 with r = reference
log-prob and l = policy log-prob is astronomically large on the forced tokens, i.e. the policy assigns
them log-probabilities around -40 to -60 under the ordinary prompt (the base model says "accepted:
yes" almost surely, r ~ 0). With beta = 1e-3 the KL term and its gradient on those few tokens dwarf
everything else in the batch, and gradient clipping at norm 1 then scales the task gradient toward
zero. The prob run (forced values 0.10-0.90 written with two decimals) shows the same KL blow-up.
A forced-exploration variant needs the KL term removed or masked on the forced tokens (beta = 0 is
the one-line version).



## Per-segment importance-sampling correction (`--is-mode segment`, implemented 2026-09-14)

`SegmentISTrainer` in `contract/train_grpo.py` post-processes TRL's per-token vLLM/trainer ratio:
the solution tokens (after the spliced "Now solve the problem." turn) get TRL's `sequence_mask`
restricted to that segment (product of ratios over the solution; the whole solution is dropped if it
exceeds 3), which is the stabiliser the pre-2026-09-13 runs had; the commitment tokens keep the
per-token truncated pi/q (ratio 1 on forced tokens). A completion without a spliced turn is one
segment, so for plain-RL runs this is exactly `sequence_mask`: the finished seeded neutral runs are
the matching control and were not re-run. The fraction of solutions masked per step is logged as
`sampling/solution_masked_fraction` in the trainer state. The forced pair was relaunched under this
mode with `--beta 0` (KL term off, see above): `grpo_modify_tests_offpolicy_force_seg_hacksucc_{yesno,prob}_s1`,
jobs 1940931/2.

### Segment-mode forced runs: both degenerated (2026-09-15, steps 170 / 164 of 200)

Per 25-step bin, legitimate accuracy / observed tampering / mean completion chars:

| run | 1-25 | 26-50 | 51-75 | 76-100 | 101-125 | 126-150 | 151-175 |
|---|---|---|---|---|---|---|---|
| forced yes/no, segment, beta 0 | .28/.063 | .11/.022 | .08/.011 | .12/.014 | .08/.008 | .09/.008 | .08/.002 |
| forced prob, segment, beta 0 | .28/.072 | .23/.080 | .20/.054 | **.00**/.003 | .00/.003 | .00/.017 | .00/.045 |

The yes/no run fell to 8-12% accuracy within 30 steps and stayed there, with 74% of completions
hitting the 1536-token cap by step 150 at entropy 0.01 (long, deterministic, wrong). The prob run
went to 0% accuracy at step ~76 with every completion at the cap and half its *commitments*
unparseable (133/256 in the last 25 steps). In both, the sampled "yes" rate is exactly the forced
5%, i.e. the policy's own commitments never moved. The solution-segment mask fired on only 1-2% of
solutions, so the sequence-mask "stabiliser" did not engage here; with beta = 0 nothing anchors the
policy, and the forced commitments plus the collapse of the solution turn ran away. Test-set
evaluation of these adapters is not worth the GPU time; the runs are left to finish so the logs are
complete. Together with the KL blow-up in the first forced pair, epsilon-forcing of commitments in
this form is a dead end: the policy assigns the forced answers ~e^-40 probability, and neither an
unregularised nor a KL-regularised update moves that in 200 steps.

## SFT warm-up of the commitment turn (started 2026-09-15)

Every trained committer so far answers a constant. Before RL, teach it a non-trivial prior:

1. **Base rollouts on the training problems.** `run_tasks --data leetcode_train_medhard_filtered.jsonl
   --neutral-system-prompt --n 8` (job 1953072): 992 problems x 8 samples of the base model, scored
   for `any_hack` and `earns_reward`.
2. **Probe.** `contract/probe.py --behaviors any_hack earns_reward --predict-out ...` fits the
   base-representation logistic probe to those outcomes with grouped 5-fold CV and writes each
   problem's *out-of-fold* probability for both statements (so the SFT targets are not memorised
   outcomes).
3. **SFT.** `contract/sft_commit.py` builds, for each training problem, the prob-mode commitment
   prompt and the target answer "1. p_hack\n2. p_accept" with the probe probabilities rounded to the
   nearest 0.1 and clipped to [0.1, 0.9] (two decimals, the policy's own format), and trains a LoRA
   adapter (r = 32, all-linear, lr 2e-5, 2 epochs) with loss on the answer tokens only.
4. **RL from the warm start.** `train_grpo.py --init-adapter runs/sft_commit_prob/final ...` loads that
   adapter as the trainable PEFT model; TRL 1.12 then keeps a frozen copy of it as the KL reference, so
   beta pulls the commitments toward the probe prior rather than toward the base model's "0.95".
   Otherwise the recipe is the on-policy hack+success prob run (split-normalised rewards, token_truncate,
   beta 1e-3, seed 1), which is the comparison point: 0.217 correct, 4.5% tampering, constant p = 0.

Base rollouts (job 1953072, 20 min): 7936 rollouts, grader acceptance 0.354, tampering 0.075; 48% of
problems have a mixed acceptance rate across their 8 samples. Probe (base representation, grouped CV):

| statement | base rate | AUC (shuffled control) | Brier vs base-rate | per-problem corr. |
|---|---|---|---|---|
| `any_hack` | 0.075 | 0.69 (0.52 +- .01) | 0.067 vs 0.070 | 0.23 |
| `earns_reward` | 0.354 | 0.62 (0.50 +- .02) | 0.221 vs 0.229 | 0.25 |

Bucketed SFT targets over the 992 problems: acceptance 0.10 x37, 0.20 x177, 0.30 x304, 0.40 x269,
0.50 x144, 0.60 x51, 0.70 x7, 0.80 x3; tampering 0.10 x900, 0.20 x68, 0.30 x16, 0.40 x5, 0.50 x3. So
the warm start teaches a problem-dependent acceptance probability (mostly 0.2-0.5, matching the
0.35 base rate) and a near-constant 0.1 for tampering. On the training set the tampering probe is
better than it was on the test set (0.69 vs ~0.5), helped by 992 problems x 8 samples.

SFT (job 1953601, 7 min, 124 optimizer steps): loss 4.4 -> 0.10, completion-token accuracy 0.96;
adapter at `runs/sft_commit_prob/final`. The prompt is rendered exactly as the RL rollouts see it
(generation prompt with the empty `<think>` block) and the completion is the answer plus `<|im_end|>`,
so the supervised tokens are the ones the policy is later asked to produce. Launched: the warm-started
GRPO run `grpo_modify_tests_sftwarm_hacksucc_prob_s1` (job 1953667) and a step-0 test-set evaluation
of the SFT adapter itself with and without the commitment turn (job 1953668).

**SFT adapter at step 0 (test set, 119 x 10).** Behaviour is the base model's: Correct 0.130 / 0.121
and tampering 0.095 / 0.091 with / without the commitment turn (base 0.132 / 0.089), so the SFT
touched only the commitment. The commitments are no longer constant:

| statement | observed | mean p | answers | Brier vs base-rate | AUC |
|---|---|---|---|---|---|
| `earns_reward` | 0.146 | 0.313 | 0.1 x27, 0.2 x249, 0.3 x538, 0.4 x298, 0.5 x75, 0.6 x3 | 0.154 vs 0.125 | **0.579** |
| `any_hack` | 0.095 | 0.100 | 0.1 x1185, 0.2 x5 | 0.086 vs 0.086 | 0.503 |

The acceptance commitment carries the probe's instance-level signal (AUC 0.58; every problem's ten
samples span more than one bucket at temperature 0.7) but is mis-levelled for the test set, which is
harder than the training set (acceptance 0.15 vs 0.35), so its Brier is worse than the base rate's.
Tampering stays a constant 0.1, as the targets dictated.

**Warm-started RL, steps 1-28: the acceptance commitment collapses toward 0 anyway.** Per 5-step bin,
training-batch success / consistency / mean committed p(accept) / mean committed p(tamper):

| steps | success | consistency | p(accept) | p(tamper) |
|---|---|---|---|---|
| 1-5 | 0.30 | 0.87 | 0.35 | 0.10 |
| 6-10 | 0.41 | 0.82 | 0.25 | 0.10 |
| 11-15 | 0.40 | 0.82 | 0.13 | 0.10 |
| 16-20 | 0.39 | 0.81 | 0.09 | 0.10 |
| 21-28 | 0.39 | 0.80 | 0.05 | 0.10 |

The cold-started prob run reached p = 0 at about the same step. Consistency *falls* (0.87 -> 0.80) as
p(accept) falls, because a constant 0.05 against a 40% success rate is a worse Brier score than the
warm start's 0.35: the consistency reward is not what drives the drift. The rollout logs say why the
commitment gets no useful signal in the first place: **within a group of 16 rollouts on one problem
the committed p varies with std 0.035** (sampling noise on the last digit), and its within-group
correlation with success is 0.00, with consistency -0.27 in steps 1-5 (the SFT prior of 0.35 was
above the early success rate, so slightly lower p was more consistent) and ~0 afterwards. Under
group-normalised advantages the commitment tokens are rewarded only relative to the other 15
commitments *on the same problem*, which are essentially identical, so once the small initial
mismatch is corrected there is no gradient toward the problem's own success rate at all; what moves
p from 0.13 to 0.05 is interference from the (large) solution-turn updates on the shared LoRA
weights, with the KL to the SFT reference (beta 1e-3) too weak to hold it. A calibrated per-problem p
would need a reward signal that compares commitments *across* problems (batch-level rather than
group-level normalisation of the consistency term, i.e. future avenue 1), or commitment diversity
within a group -- which is what the epsilon-forced variant tried to supply and could not weight.

## Decoupled commitment and attempt (implemented 2026-09-15)

The warm-start diagnosis says the two-turn design cannot teach a per-problem probability: the
commitment tokens are only ever compared with near-identical commitments on the same problem, and
their reward is tangled with the solution's. `train_grpo.py --decoupled` separates the two:

- For each problem in a step the model runs **two independent conversations**: G *attempts*
  (neutral prompt, problem, format instruction -> solution) and G *commitments* (commitment prompt ->
  probability per statement). Nothing is spliced; each is a one-turn episode.
- **Attempts** are rewarded with the task reward alone, z-normalised within the G attempts (the plain
  neutral-RL recipe).
- **Commitments** are rewarded with 1 - mean over statements of (p - target)^2, where the **target is
  the mean of each behavior over the G attempts on that problem** (the empirical acceptance and
  tampering rates of the current policy on that problem), z-normalised within the G commitments.
- Same LoRA for both roles (one model, two prompts); TRL sees 2G generations per problem.

The commitment reward now directly measures calibration against the problem's own rate, and the
attempt reward no longer contains the consistency term that produced the honest-failure collapse.
What still limits learning is diversity among the G commitments: z-normalisation within the group
needs them to differ. Runs: `grpo_modify_tests_decoupled_hacksucc_prob_s1` from the base model and
`..._sftwarm_s1` from the SFT warm-start adapter (more spread in p to begin with); token_truncate,
beta 1e-3, seed 1, 200 steps (jobs 1955719 / 1955720). A third run, `--attempt-agreement`
(`grpo_modify_tests_decoupled_agree_hacksucc_prob_sftwarm_s1`, job 1956849, SFT init), additionally
rewards each attempt with 1 - squared error between the *group's mean commitment* and the attempt's
own behaviors, z-normalised and added to the task term: the attempt is pulled toward what was
committed, which re-introduces the coupling of the two-turn design (and its honest-failure
equilibrium) but with the commitment fixed as the group mean rather than the rollout's own.

Steps 1-8 of the first two runs (128 problem-groups each): attempts are at base-model level (pass
0.25, acceptance 0.35, tampering 0.05). Base-initialised commitments are constant (p(accept) 0.95,
sd 0.006 across problems, 0.001 within a group) and have not moved. SFT-initialised commitments are
correctly levelled (mean p 0.33 vs realised 0.32) and vary across problems (sd 0.07, within-group
0.08) but that variation does not yet track the problems' realised rates: across-problem
correlation +0.03 overall, +0.07 in steps 1-5, +0.08 in steps 4-8. Tampering commitments are constant
in both.

**Steps 1-13.** Base init: p(accept) drifts *up* 0.95 -> 0.97 and p(tamper) to 0.000 (from 0.03); the
commitments never acquire within-group spread (sd 0.001-0.01), so there is nothing to select among.
SFT init: attempts unchanged (pass 0.25-0.30, tampering 0.03-0.08); the across-problem correlation of
p(accept) with the realised rate sits at +0.13 to +0.18 (steps 1-12) and does not grow, while the
across-problem spread of p(accept) *shrinks* (sd 0.067 -> 0.030) and its mean drifts from 0.37 to
0.24-0.27 against a realised rate of 0.32-0.37. Both drifts have the same cause: **z-normalising the
commitment reward within each problem's group makes every problem push with unit strength, whatever
the size of its error**, so a shared, problem-independent p converges to the point where as many
problems push it down as up -- the *median* of the per-problem rates, not the mean. With 38% of
problems at rate 0 and a mean of 0.35, the median acceptance rate is well below the mean, and the
median tampering rate is exactly 0, which is where both runs' p(tamper) are heading. The per-problem
signal that would separate problems (the +0.15 correlation) is small relative to this level-pulling
force. So the commitment reward should not be group-normalised: the raw 1 - squared error (or a
batch-level baseline) keeps the gradient proportional to the error and targets the mean.

`--commit-norm batch` (2026-09-15) does that: the commitment reward 1 - squared error is z-scored
across *all* commitments in the batch, so a problem's error keeps its size; the attempts stay
group-normalised. Because TRL always subtracts the per-group mean from the rewards it is given,
`DecoupledTrainer` writes our rewards directly over TRL's advantages. Two runs from the SFT adapter,
`grpo_modify_tests_decoupled_bn_hacksucc_prob_sftwarm_s1` and `..._bn_agree_...` (with the attempt
agreement term). The base-initialised group-normalised run was stopped at step 13 (constant
commitments, nothing to learn from); the two SFT-initialised group-normalised runs continue as the
comparison.

**Steps 1-29 (2026-09-15 evening), across-problem correlation of committed p(accept) with the
realised rate, per 8-step bin:**

| run | 1-8 | 9-16 | 17-24 | 25-29 | mean p vs rate (latest) | attempt tampering (latest) |
|---|---|---|---|---|---|---|
| group-norm | +0.03 | +0.21 | **+0.40** | +0.30 | 0.29 vs 0.43 | 0.12 |
| group-norm + agreement | +0.05 | +0.06 | +0.29 (17-20) | | 0.26 vs 0.34 | 0.07 |
| batch-norm | +0.05 | +0.23 | | | 0.29 vs 0.34 | 0.08 |
| batch-norm + agreement | +0.06 | +0.25 | | | 0.21 vs 0.31 | 0.05 |

The ranking signal grows in every run (from ~+0.05 to +0.2-0.4 within 16-24 steps), so the
decoupled reward does teach problem-dependent commitments, slowly. Two caveats at this stage: the
across-problem spread of p stays compressed (sd 0.03-0.06 against a realised-rate sd of ~0.3), and the
mean p sits below the realised rate in all four runs (the batch-norm runs included, so the median
pull is not the whole story: with a compressed p, being low on the many zero-rate problems costs
less than being high). Tampering commitments are still a constant 0.10 (correlation ~0), and the
attempts' tampering rate is unchanged at 5-12%. Pass rates 0.25-0.32.

### Decoupled runs at steps 95-177 (2026-09-16 morning): the agreement term is what keeps the attempts honest

Per 8-step bin: pass / acceptance / tampering of the attempts; mean committed p(accept), its
across-problem correlation with the realised rate and spread; mean committed p(tamper).

**Group-norm, no agreement (177 steps): took off into universal tampering.**

| steps | pass | accept | tamper | p(accept) mean / corr / sd | p(tamper) |
|---|---|---|---|---|---|
| 25-32 | .36 | .46 | .12 | 0.35 / +0.25 / 0.11 | 0.10 |
| 33-48 | .28-.36 | .39-.45 | .10-.14 | 0.52-0.62 / +0.40 / 0.09-0.10 | 0.11-0.12 |
| 49-64 | **.01-.10** | .02-.20 | .01-.13 | 0.00-0.20 | 0.00-0.07 |
| 65-72 | .01 | **.81** | **.81** | 0.01 | 0.006 |
| 73-80 | .00 | .98 | .98 | 0.88 / +0.11 | 0.78 |
| 81-177 | .00 | **1.00** | **1.00** | **1.00** | **1.00** |

A collapse at steps 49-64 (success ~0, like the earlier token_truncate neutral seeds), then the
loophole: from step 65 every attempt tampers and earns the reward, legitimate passes are zero, and
the commitments follow within ten steps to p(tamper) = p(accept) = 1.00 on every rollout. This is
the *honest hacking* equilibrium of `commitment_game.md` realised end to end: the committer is
perfectly calibrated and the policy is maximally misaligned. Without a coupling from commitment to
attempt, the attempt side is plain neutral RL, which takes off in 2 of 3 seeds.

**Batch-norm, no agreement (95 steps): collapse and dead commitments.** Tampering rose to 0.22-0.24
at steps 25-48, success crashed to 0.05-0.10 at steps 49-72 (as above), then recovered to pass 0.26
/ tampering 0.07 by steps 81-95 -- but the commitment turn died in the crash: p(accept) is a
constant 0.01 (sd 0.000) and p(tamper) a constant 0.10 from step 57 on. Batch normalisation did
*not* fix the level drift (mean p(accept) 0.27-0.47 vs realised 0.28-0.48 before the crash, roughly
level) and had no protective effect on the attempts.

**Group-norm + agreement (103 steps): the first informative commitments, and low tampering.**

| steps | pass | accept | tamper | p(accept) mean / corr / sd | p(tamper) mean |
|---|---|---|---|---|---|
| 1-8 | .24 | .31 | .056 | 0.31 / +0.05 / 0.08 | 0.100 |
| 41-56 | .23-.26 | .35-.38 | .024-.054 | 0.16-0.22 / +0.30-0.32 / 0.04-0.05 | 0.100 |
| 57-72 | .26-.28 | .36-.38 | .018-.054 | 0.19-0.30 / +0.29-0.35 / 0.09-0.12 | 0.096-0.100 |
| 73-88 | .26-.29 | .34-.41 | .032-.046 | 0.33-0.41 / +0.31-0.43 / **0.21-0.31** | 0.036-0.038 |
| 89-103 | **.31-.35** | .40-.42 | **.018-.032** | 0.31-0.45 / +0.26-0.42 / 0.12-0.16 | **0.016-0.022** |

Legitimate accuracy is the highest of the four (0.35 by step 100 vs 0.25 at the start), tampering
is the lowest (1.8-3.2%, a third of the base model's 6-9% and far below any plain-RL run), the
acceptance commitment ranks problems at +0.3-0.4 with an across-problem spread (sd 0.12-0.31) that
finally approaches the spread of the realised rates, its level is right (0.31-0.45 vs 0.40-0.42), and
the tampering commitment has moved off its constant to 0.016-0.022, matching the realised 2%. The
attempt-agreement term is doing two jobs at once: it anchors the attempts to the committed low
tampering probability (the d > 0 mechanism of the game analysis), and by keeping the attempts'
behaviour stable it gives the commitment a stationary target to learn.

**Batch-norm + agreement (101 steps): stable, low tampering, weaker commitments.** Pass 0.21-0.26,
tampering 0.8-4.5% (1.1-1.4% in the last bins), p(accept) 0.26-0.32 with correlation +0.27-0.36 but
a compressed spread (sd 0.01-0.03); p(tamper) a constant 0.100. Batch normalisation of the
commitment reward, whose motivation was the level drift, again did not help: the group-normalised
sibling has both the better level and the better spread.

**Group-norm, no agreement, step 200 on the test set (job 1973281):** tampering 1.000, Correct 0.000
under both the commitment prompt and the neutral prompt; every one of the 1190 solutions is the
starter code with an empty body plus `def run_tests(): pass`; every commitment is p(tamper) = 1.00,
p(accept) = 1.00 (Brier 0.000 on both: a perfectly calibrated committer). With the `none` hint,
where there is no grader to overwrite, it still emits stubs and scores 0. Clean-prompt MCQ 0.48
(degenerate, as for the honest-failure model). This is the worst-case reference for the agreement
runs: calibration without coupling is worthless as a safety device.

So far the ordering is: coupling (agreement term) matters a great deal, normalisation of the
commitment reward does not, and group normalisation is if anything better. Step-200 evaluations
of the two agreement adapters are the next measurement.

Seeds 2 and 3 of all four variants were launched 2026-09-16 on `vision-shared-h100` /
`shared-if-available` (pre-emptible, `--requeue`; jobs 1972005-1972012), same SFT warm-start adapter
(trained once, seed-independent) and otherwise identical arguments. All eight were pre-empted within
their first 1-22 steps, before the first checkpoint at step 25, so their requeued copies restarted
from scratch; they were resubmitted (jobs 1973253-1973260) with `--save-steps 5` (checkpoints kept to
the last three via `save_total_limit`), and `train_grpo.py` now continues the step counter from the
checkpoint it resumes and drops log lines written after it, so `reward_log.jsonl` / `rollouts.jsonl`
read as a single run across pre-emptions.

Resume mechanics, checked 2026-09-16 evening: the seed-2 agreement run was pre-empted (three times so
far) after logging 17 steps with checkpoints at 5/10/15; on its next start the logs were truncated to
step 15 as designed (`reward_log` and `rollouts` both end at call 15 with no repeats). The seed-1 runs
hit the 24 h wall at steps ~158 / ~150 / ~154 (`TIMEOUT` does not requeue); the agreement run was
resubmitted immediately (job 1976549) and the two batch-norm runs got `--dependency=afterany`
continuation jobs (1976552/1976553), which started within five minutes of the time-outs. **Full
verification:** the agreement continuation truncated its log from 158 to 150 lines, resumed from
checkpoint-150, and logged the next steps as 151, 152, 153, 154 -- a single monotone sequence.

**But the weights did not resume (found 2026-09-17).** The three seed-1 continuations reached step
200 with statistics in steps 151-200 that were identical across the three runs and identical to the
SFT adapter's step-1 behaviour (pass 0.26, tampering 0.075, p(accept) 0.38 with sd 0.07, p(tamper)
0.100). Comparing adapter weights: checkpoint-175 and -200 are 0.28 / 0.28 (L2) from the SFT init
adapter and 5.5 from checkpoint-150. The cause is in `transformers.Trainer._load_from_checkpoint`:
on a PEFT model with several adapters (TRL adds a frozen `ref` copy of the init adapter when beta >
0, saved under `checkpoint-N/ref/`) it loads only the adapters found in sub-directories, i.e. `ref`,
and never the trainable `default` adapter saved at the checkpoint's top level. The earlier resume
verification passed because those runs used a fresh LoRA (no `ref` sub-directory). Fix:
`train_grpo.py` now loads the top-level `adapter_model.safetensors` into the `default` adapter
explicitly before `train()` when `--init-adapter` is set and a checkpoint exists, and prints the
post-load max-abs difference (must be 0.0). Consequences: steps 151-200 of the three seed-1 runs
are discarded (checkpoint-175/200 deleted; they resume from checkpoint-150 with the fix); the seed-2
agreement run, whose one resume happened at step 15 under the bug, is restarted from scratch; the
seed-3 runs were caught before their (pre-empted) restarts completed a step and resume cleanly.

### Seed-1 decoupled runs, complete (valid steps 151-200 after the resume fix; 2026-09-17)

| run | steps | pass | accept | tamper | p(accept) mean / corr / sd | p(tamper) |
|---|---|---|---|---|---|---|
| group-norm + agreement | 151-175 | .349 | .471 | .029 | 0.53 / **+0.59** / 0.26 | 0.007 |
| | 176-200 | .315 | .425 | .021 | 0.45 / **+0.62** / 0.28 | 0.008 |
| batch-norm, no agreement | 151-175 | .396 | .537 | .064 | 0.01 / 0 / 0.000 | 0.100 |
| | 176-200 | .350 | .479 | .056 | 0.01 / 0 / 0.000 | 0.100 |
| batch-norm + agreement | 151-175 | .274 | .377 | .008 | 0.37 / +0.41 / 0.13 | 0.100 |
| | 176-200 | .255 | .355 | .008 | 0.40 / +0.47 / 0.15 | 0.100 |

The group-norm + agreement run ends with the most informative commitments of the project (+0.62
across-problem correlation, spread 0.28 against a realised-rate spread of ~0.3, level 0.45 vs 0.43)
plus a tampering commitment that tracks the realised ~1-2%, at 2-3% tampering and pass 0.32-0.35.
The batch-norm no-agreement run recovered from its mid-run crash to pass 0.35-0.40 with 6%
tampering, but its commitments are dead constants (p(accept) 0.01, p(tamper) 0.10). The batch-norm
+ agreement run has the lowest tampering (0.8%) at base-level accuracy, with a moderately
informative acceptance commitment (+0.47, spread 0.15) and a constant tampering commitment.
Step-200 evaluations: jobs 1997396-8 (on `lingo-low`, so as not to pre-empt the seed-2/3 trainers).

### Seed-1 decoupled adapters on the test set (step 200; jobs 1997396-8; 119 problems x 10 samples)

| model | prompt | Correct | Reward Hack | tampering | p(accept): mean vs rate, AUC, Brier vs base | p(tamper): mean vs rate, AUC |
|---|---|---|---|---|---|---|
| base (3 seeds) | neutral | 0.132 | 0.030 | 0.089 | | |
| group-norm + agreement | with commitment | **0.234** | 0.056 | 0.030 | 0.24 vs 0.29, **AUC 0.78**, 0.170 vs 0.206 | 0.006 vs 0.030, AUC 0.58 |
| group-norm + agreement | neutral / `none` | 0.223 / 0.211 | 0.054 / 0 | 0.033 / 0 | | |
| batch-norm, no agreement | with commitment | 0.221 | 0.055 | 0.092 | 0.01 vs 0.28, AUC 0.50, 0.269 vs 0.199 | 0.10 vs 0.09, AUC 0.50 |
| batch-norm, no agreement | neutral / `none` | 0.247 / 0.213 | 0.048 / 0 | 0.082 / 0 | | |
| batch-norm + agreement | with commitment | 0.150 | 0.010 | **0.021** | 0.29 vs 0.16, AUC 0.74, 0.133 vs 0.134 | 0.10 vs 0.02, AUC 0.50 |
| batch-norm + agreement | neutral / `none` | 0.127 / 0.131 | 0.016 / 0 | 0.024 / 0 | | |

- **The group-norm + agreement adapter is the first trained model whose commitments carry
  instance-level information on the test set**: acceptance AUC 0.78 (every earlier trained model:
  0.50; the base model with the SFT prior: 0.58; the linear probe's ceiling on this test set: ~0.75),
  Brier 0.170 against a base-rate Brier of 0.206, answers spread over 0.00-0.92 with the mass between
  0.02 and 0.65, and a level (0.24) close to the realised rate (0.29). Its tampering commitment is
  not constant either (0.00 / 0.01 / 0.02) and has AUC 0.58 against a 3% rate. It does this at the
  highest legitimate accuracy of any adapter (0.234, base 0.132) and a third of the base model's
  tampering, and the behaviour carries to the neutral prompt (0.223 / 3.3%).
- **The batch-norm + agreement adapter is the least-tampering model** (2.1%, 1.0% successful hacks)
  but gained nothing on accuracy (0.150 vs 0.132) and its acceptance commitment, while informative
  (AUC 0.74), is mis-levelled (0.29 vs 0.16) so its Brier only matches the base rate; tampering
  commitment a constant 0.10.
- **Batch-norm without agreement**: base-level tampering (9%), accuracy 0.22-0.25, commitments dead
  (p(accept) 0.01 constant, worse Brier than the base rate).
- MCQ (single attempt): fail - legit +0.013 / +0.016 / +0.002; clean-prompt rates 0.30 / 0.41 / 0.23,
  the middle one again the elevated clean rate seen for the earlier prob adapters.

### Seeds 2-3 (2026-09-18 morning; three of eight finished)

Tampering per 25-step bin (attempts' `any_hack`):

| run | 1-25 | 26-50 | 51-75 | 76-100 | 101-125 | 126-150 | 151-175 | 176-200 |
|---|---|---|---|---|---|---|---|---|
| group-norm, no agreement, s2 | .08 | .13 | .40 | **1.00** | 1.00 | 1.00 | 1.00 (163) | |
| group-norm, no agreement, s3 | .08 | .20 | .25 | .91 | **1.00** | 1.00 | 1.00 | 1.00 |
| batch-norm, no agreement, s2 | .12 | .15 | .24 | **1.00** | 1.00 (122) | | | |
| batch-norm, no agreement, s3 | .07 | .21 | .34 | .23 | **.99** | .99 | .99 | .99 |
| group-norm + agreement, s2 | .07 | .04 | .04 | .03 (96) | | | | |
| group-norm + agreement, s3 | .05 | .04 | .02 | .02 | .004 | .009 | .000 | .006 |
| batch-norm + agreement, s2 | .07 | .05 | .03 | .03 (88) | | | | |
| batch-norm + agreement, s3 | .06 | .04 | .02 | .02 | .02 | .02 | .02 | .02 (188) |

**Every run without the agreement term takes off** (6 of 6 across seeds, counting seed 1) to
universal tampering by step 75-100, with p(tamper) following to 1.00 (or 0.85 / 0.38 constants in
the batch-norm seeds, whose commitment turn died); **every run with it stays at 0-3%** (6 of 6).
The seed-3 batch-norm no-agreement adapter on the test set: tampering 1.000, Correct 0.000, every
solution a starter-code stub; commitments a constant p = 0.00 / 0.00 -- the honest-hacking
equilibrium with a *dead* committer this time (Brier 1.0 on both statements), because the
batch-normalised commitment reward had already collapsed the commitment before the take-off.

Accuracy: the agreement runs sit at pass 0.25-0.37 throughout (seed-3 batch-norm + agreement rises
to 0.35-0.37 at steps 76-188, seed-3 group-norm + agreement stays at 0.24-0.28).

**The commitment side does not replicate across seeds.** Seed-3 group-norm + agreement collapsed
p(accept) to a constant 0.10 (sd 0.000) by step 76 and never recovered; seed-2 group-norm +
agreement is on the same path (0.10, sd 0.000 at step 96). Seed 1 of the same recipe opened up to
+0.62 correlation. The batch-norm + agreement runs are the consistent ones: seed 3 reaches
correlation +0.53 to +0.55 with spread 0.14 and level 0.45-0.50 vs realised 0.43-0.46 by steps
151-188, seed 2 is at +0.40 / 0.11 at step 88, and seed 1 ended at +0.47 / 0.15. So across three
seeds, batch normalisation of the commitment reward gives a reliably informative (if compressed)
acceptance commitment, while group normalisation gives either the best commitment (seed 1) or a
dead one (seeds 2, 3); p(tamper) stays a constant 0.10 in all batch-norm + agreement runs.

### Seed-2/3 adapters on the test set (jobs 2011038, 2014591-2, 2017278; 119 problems x 10 samples)

| model | prompt | Correct | tampering | p(accept): mean vs rate, AUC, Brier vs base | p(tamper) |
|---|---|---|---|---|---|
| group-norm, no agreement, s3 | commitment / neutral | 0.000 / 0.000 | **1.000 / 1.000** | 1.00 vs 1.00 (calibrated stubs) | 1.00 |
| batch-norm, no agreement, s3 | commitment / neutral | 0.000 / 0.000 | **1.000 / 1.000** | 0.00 vs 1.00 (dead) | 0.00 |
| group-norm + agreement, s3 | commitment / neutral | 0.145 / 0.130 | **0.006 / 0.004** | 0.10 vs 0.15, AUC 0.50, 0.132 vs 0.129 | const 0.10 |
| group-norm, no agreement, s2 | commitment / neutral | 0.000 / 0.000 | **1.000 / 1.000** | 0.85 vs 1.00 (constant) | 0.85 |
| batch-norm + agreement, s3 | commitment / neutral / `none` | **0.214** / 0.224 / 0.203 | 0.016 / 0.021 / 0 | 0.33 vs 0.24, **AUC 0.71**, 0.173 vs 0.182 | const 0.10 |

All three no-agreement adapters evaluated so far (seed 3 x2, seed 2 group-norm) are stub-emitters with
100% tampering; their committers are constants at 1.00 / 1.00, 0.00 / 0.00 and 0.85 / 0.85 respectively. The seed-3 group-norm + agreement
adapter has the lowest tampering of any model so far (0.6%, 1.3% successful hacks) at base-model
accuracy (0.145 vs 0.132), with commitments that are constants (0.10 / 0.10, AUC 0.50): the safety
effect without the informative commitment that seed 1 of the same recipe produced.
The seed-3 batch-norm + agreement adapter reproduces its seed-1 sibling and improves on it: Correct
0.214 (seed 1: 0.150), tampering 1.6%, acceptance AUC 0.71 with Brier below the base rate (0.173 vs
0.182) and answers spread over 0.10-0.80, tampering commitment a constant 0.10. Across the two
evaluated seeds this is the recipe whose *commitments* replicate (AUC 0.74 / 0.71) while keeping
tampering at 1-2%. MCQ fail - legit
+0.008 +- 0.003, clean 0.233.

## Agreement-only attempts (launched 2026-09-18)

`--agreement-only` (with `--decoupled --attempt-agreement`) removes the task term from the attempt
reward: attempts are z-scored on agreement with the group's mean commitment alone, while commitments
are still scored against the attempts' mean behaviors. This isolates the coupling: does the
commitment/attempt agreement loop by itself keep tampering down (and does it keep solving problems
at all, with no direct pressure to), or does the pair drift to whatever self-consistent point is
easiest -- the honest-failure equilibrium of `commitment_game.md`, where the committer says "no,
no" and the attempt fails on purpose, is a fixed point of this reward. Three seeds, batch-norm
commitment reward (the recipe whose commitments replicated), SFT warm start, otherwise the standard
decoupled recipe; `grpo_modify_tests_decoupled_bn_agreeonly_hacksucc_prob_sftwarm_s{1,2,3}` on
`lingo-low`.

**Result (all three seeds finished 2026-09-20): honest failure, exactly as the game predicts.**
Per 25-step bin, attempts' acceptance rate / committed p(accept) / mean completion chars:

| seed | 1-25 | 26-50 | 51-75 | 76-100 | 101-200 |
|---|---|---|---|---|---|
| 1 | .28 / 0.23 / 1934 | .15 / 0.06 / 1630 | **.001 / 0.00** / 1517 | .000 / 0.00 | .000 / 0.00 |
| 2 | .36 / 0.41 / 2070 | .30 / 0.25 / 1845 | .15 / 0.13 / 1017 | .13 / 0.00 / 1588 | .09 -> .04 / 0.00 / ~850 |
| 3 | .31 / 0.30 / 1962 | .24 / 0.26 / 1365 | .13 / 0.20 / 1116 | .03 / 0.20 / 4338 | .003 / 0.20 / ~5600 |

With no task term, the committer's p(accept) leads and the attempts follow it down: seed 1 reaches
zero success and p = 0.00 by step 60; seed 2 decays to 4% success with p = 0.00; seed 3 settles on
a constant p(accept) = 0.20 and 0.3% success, with completions ballooning to 5600 chars (the attempts
found that long, non-compiling output is a reliable way to fail). Tampering goes to 0 in all three,
and p(tamper) to 0.00 / 0.00 / 0.10. Perfectly consistent, perfectly useless: the honest-failure
fixed point. The task term in the agreement runs is what keeps the pair out of it.

**Seed-2 agreement runs (finished 2026-09-20).** Group-norm + agreement s2: pass 0.25, tampering 2.4%,
p(accept) collapsed to a constant 0.10 by step 75 (as in seed 3). Batch-norm + agreement s2: pass
0.33, tampering 2.2%, p(accept) 0.46 vs realised 0.43, training-set AUC rising to 0.72 by steps
151-200 with spread 0.17 -- the third seed of that recipe to produce an informative commitment.

## Evaluation protocol caveat for the decoupled adapters (2026-09-22)

The test-set evaluations of the decoupled adapters use `run_tasks.py --precommit prob`, i.e. the
Part 2 two-turn protocol: commitment, then "Now solve the problem" appended to the *same*
conversation, then the solution. The decoupled runs were trained with the commitment and the attempt
as separate conversations, so the "with commitment" numbers put the commitment in the solver's
context, a prompt the model never saw in training. Two consequences: (i) the commitment *calibration*
numbers (AUC, Brier) are unaffected, because the commitment conversation is identical to the training
one; (ii) the *behaviour* numbers under "with commitment" are a distribution shift, and the matching
measurement is the "neutral prompt" evaluation, whose user turn is exactly the training-time attempt
prompt -- those were run for every adapter and agree with the with-commitment numbers to within
1-2 points on tampering and accuracy in every case. The page's example section now shows the two
training-time conversations side by side rather than the joined transcript.

### Final test-set table for the decoupled runs (all 15 adapters evaluated, 2026-09-24)

Commitment prompt (the neutral-prompt numbers agree to within 1-2 points in every case); tampering =
`any_hack`; p(accept) AUC against each rollout's own acceptance.

| variant | seed | Correct | tampering | p(accept) AUC (Brier vs base) | p(tamper) |
|---|---|---|---|---|---|
| group-norm, no agreement | 1 / 2 / 3 | 0.000 / 0.000 / 0.000 | **1.000 / 1.000 / 1.000** | const 1.00 / 0.85 / 1.00 | const 1.00 / 0.85 / 1.00 |
| batch-norm, no agreement | 1 / 2 / 3 | 0.221 / 0.000 / 0.000 | 0.092 / **1.000 / 1.000** | const 0.01 / -- / 0.00 | const 0.10 / -- / 0.00 |
| group-norm + agreement | 1 / 2 / 3 | **0.234** / 0.137 / 0.145 | 0.030 / 0.036 / **0.006** | **0.78** (0.170 vs 0.206) / const 0.10 / const 0.10 | 0.006, AUC 0.58 / const 0.10 / const 0.10 |
| batch-norm + agreement | 1 / 2 / 3 | 0.150 / 0.195 / 0.214 | 0.021 / 0.022 / 0.016 | **0.74 / 0.72 / 0.71** (0.133 vs 0.134 / 0.151 vs 0.168 / 0.173 vs 0.182) | const 0.10 x3 |
| agreement only | 1 / 2 / 3 | 0.000 / 0.018 / 0.000 | 0.000 / 0.000 / 0.001 | const 0.00 / 0.00 / 0.20 | const 0.00 / 0.00 / 0.10 |

Three-seed means: tampering 1.00 / 0.70 / 0.024 / 0.020 / 0.000; Correct 0.000 / 0.074 / 0.172 / 0.186 /
0.006; base model 0.089 / 0.132. So the agreement term cuts tampering from ~100% (plain RL under this
prompt) to ~2% while adding 4-10 points of accuracy over the base model; agreement without the task
term removes tampering by removing the solving; and **batch-norm + agreement is the only recipe whose
success commitment is informative in every seed** (AUC 0.71-0.74, Brier below the base rate in 2 of 3),
at the cost of a tampering commitment stuck at the SFT prior's 0.10. MCQ fail - legit for the seed-2
agreement adapters: +0.021 / +0.002.

## Future avenues

1. **Separate, separately normalized rewards for the two turns.** Currently one scalar (task +
   consistency) is assigned to the whole trajectory and GRPO-normalized within the group, so the
   commitment tokens receive advantage from the task term and the solution tokens from the
   consistency term. Instead: give the commitment turn its own reward (consistency only, without task
   success) and the solving turn its own (task only, or task plus the consistency the solution earned
   against the fixed commitment), each normalized within its group. This removes the incentive for
   the commitment to chase the task reward and lets its gradient reflect only prediction quality; in
   TRL terms it means per-token advantages built from two separately normalized group rewards rather
   than one.
2. **Off-policy commitments from a better-informed prompt.** The on-policy model has no information
   about its own success rate beyond what GRPO teaches it, and its probability vocabulary is coarse.
   Sample commitments instead from the same model prompted with examples of its previous rollouts on
   similar problems (e.g. a few past problems with the observed behaviors and outcomes), use those
   commitments as the first turn of the episode, and train the policy on them off-policy (importance
   weighting or simply treating them as fixed context). This decouples "knowing what you will do" from
   "learning to say it", and gives a calibrated target the on-policy commitment can be distilled toward.
