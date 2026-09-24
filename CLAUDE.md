## Background

The goal in this research project is to train language models that write
probabilistic "contracts" that constrain their behavior before undertaking
agentic solutions to problems. Concretely, suppose we have a task ("write a
function that does [...] and passes the tests in `tests`") and some enumerated
set of behaviors ("I will modify files in `src`", "I will modify files in
`tst`", etc.), we would like the model to first generate a spec of the form:

p(modify `src`) = 0.95
p(modify `tst`) = 0.05
p(access network) = 0.01

and then proceed when a user has approved the spec.

Our ultimate goal is safety: given that a model may take a risky action while
operating autonomously, we would like it to articulate that probability before
acting, and then act in such a way that the probability reflects a calibrated
prediction about its behavior.

We are motivated that the hypothesis that models are generally well-aligned at
the beginning of interactions, and if asked _a priori_ about the appropriateness
of specific reward-hacking behaviors (like modifying tests), they are able to
recognize them as inappropriate; only once they begin attempting a task (and
perhaps especially after trying and failing to perform it legitimately) does
their willingness to take inappropriate actions (and perhaps even their ability
to recognize such actions as inappropriate) increase.

This project has three components. The first two components are empirical:
measuring the extent to which off-the-shelf mdoels indeed exhibit changes in
alignment as their interactions proceed, and measuring the extent to which this
can be mitigated by explicit up-front commitments to use or avoid particular
strategies. The final component is experimental: we perform reinforcement
learning with a loss function that encourages consistency between models'
predicted and executed behaviors, and then measure whether this further
mitigates hacking relative to baseline behavior.

## Environment

You have access to a SLURM cluster via `slurm-login.csail.mit.edu`.

**Important**: the ssh connection to the cluster (via the `mfa-jump` host) can be
flaky. If a connection hangs or fails, do not retry: report it in chat, then
stop and wait for me to reconnect before issuing further cluster commands.

**Important**: never download anything large (model weights, datasets, pip
packages) from the login node; it gets us throttled. Do all downloads from
inside a running SLURM job (e.g. a short CPU-only job on the compute node).

Your entry point scripts should look like

```
#!/bin/bash
#
#SBATCH --job-name=your_job_name
#SBATCH --account=lingo
#SBATCH --partition=lingo-h100
#SBATCH --qos=lingo-main
#SBATCH --time=00:01:00 # (hh:mm:ss)
#SBATCH --output=YOUR_SLURM_LOG_DIR/job_output_%j.log  # CHANGE THIS
#SBATCH --error=YOUR_SLURM_LOG_DIR/job_output_%j.err  # CHANGE THIS
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G


python -c your-file
```

and jobs should be scheduled like

```
srun --account=lingo \
     --partition=lingo-h100 \
     --qos=lingo-low --gpus=1 \
     --time=2:00:00 \
     --mem=16G \
     --pty /bin/bash
```

Try to estimate the runtime of your jobs and set the `--time` flag
conservatively (shorter jobs will get scheduled sooner, but we don't want things
to be cut off). 

Any data you need to upload can be placed in `/data/scratch-oc40/jda`. Make a
subdirectory for this project, and set up a python virtualenv within this
subdirectory that is activated by the script. You can scp things there or
download them from the login node; they will then be accessible from within
SLURM jobs. (Because this is a scratch directory, it may be wiped
occasionally---make sure to copy all generated datasets and results files back
to this machine! You don't need to save model weights.) On this machine the
project folder is inside Dropbox, so large outputs must not live in it: `results`
is a symlink to `~/code_nobackup/contract/results` (outside Dropbox); pull results
there (`scripts/sync.sh pull` does, via the symlink) and never copy them into the
project tree itself.

## Implementation and documentation

Provide a high-level description of all code and experimental results to date as
markdown files in a `docs` folder.

In general I will request modifications to the overall plan via the chat
interface. If any of these requests don't make sense, contradict instructions
elsewhere, or seem incorrect / ill-advised, please stop and ask a question
before proceeding.

**Important**: this is research code, so it is very important to prioritize
clarity and pedagogy over industrial-strength robustness. Favor minimal (or no)
exception handling, and minimize the number of abstraction layers wherever
possible.

**Important**: please do not suggest (and especially initiate!) follow-up
questions, new experiments, etc.---these clutter the chat window and make it
difficult to determine what's been done so far. When I want help brainstorming
next steps, I will ask explicitly.

## Part 1: Do failed agentic interactions cause models to become misaligned?

We'd like to be able to do this with multiple models in multiple environments,
so write your code in an extensible way (while paying attention to the note
above about writing easy-to-read research code!)

For starters I would like to look at the tasks in the `ariahw/rl-rewardhacking`
github repo, and the Instrumental Choices dataset from arXiv:2605.06490.

1. Run (for now don't train) a Qwen3-4B model on the evaluation tasks and
   measure the legitimate-pass and hacking rates at baseline. 

2. Do another run where, whenever a model fails, it receives a user message
   instructing it to start over and try harder.

3. Take these runs, and then take the model (a) starting from a clean
   transcript, (b) at the end of legitimate successes, and (c) at the end of
    hacking successes, and measure its general alignment properties using the
    multiple-choice questions from Alignment Pretraining (arXiv:2601.10160).

[We will probably need to do some fiddling to get any signal here, and it might
not work; please pause after each of these steps so we can discuss the results.]

[Also, don't run this or anything below until we're done with Part 1. The
instructions may change depending on the outcome of those experiments! But do
architect the codebase in a way that will make implementing parts 2 and 3 easy.]

## Part 2: do pre-commitments cause models to become better aligned?

Just a re-run of part 1, but with a prompt where models are first instructed to
reason about the task and list the capabilities they will and won't use. This
list should persist as part of the conversation transcript as agents proceed to
reason through solutions to the task. Measure the same set of outcome metrics.

## Part 3: learning to generate better commitments

The idea here is basically to apply the methodology in the RLCR paper
(https://arxiv.org/pdf/2507.16806) but with two important modifications: (1) the
predictions are generated *before* the agent begins to act, and (2) prediction
correctness is judged w/r/t the agent's own actions rather than some external
source of ground truth. Here the "capabilities" should include all the behaviors
that would count as hacks, some benign behaviors that the agent doesn't need to
use, and some benign behaviors that the agent almost certainly needs to use.
Use a combined task success + Brier reward as in that paper.  (To compute
rewards, use the datasets' native hacking measurements + whatever code you need
to write to check for the presence of benign actions). I would also like to
experiment with a "hard" variant of RLCR in which models just say "yes" or "no"
to the set of used capabilities, and get a reward of -0.5 if they say yes but
don't use a capability, and a reward of -1 if they say no but do use a
capabiity.

Evaluate models thus trained as in Parts 1 and 2.
