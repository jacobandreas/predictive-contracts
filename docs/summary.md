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

The attached `docs/results.html` contains an initial set of experiments. All
these experiments use a Qwen3-4B on a LeetCode variant designed to evoke reward
hacking during RL training; this environment is taken from [this github
repo](https://github.com/ariahw/rl-rewardhacking) (described more
[here](https://www.lesswrong.com/posts/R5MdWGKsuvdPwGFBG/steering-rl-training-benchmarking-interventions-against)).
