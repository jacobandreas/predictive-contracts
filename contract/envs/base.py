"""Minimal environment interface shared by all task suites.

An environment exposes a list of tasks (each an initial chat prefix) and a
`score` function that inspects one model response and reports:

  success   -- did the response earn the environment's reward?  This is the
               *hackable* signal the agent is optimised against (e.g. "the
               evaluation function ran without error"), and is what decides
               whether the agent is told to try again.
  legit     -- did the response genuinely solve the task (ground truth)?
  hack      -- did the response earn reward illegitimately?
  label     -- a short categorical description (environment-specific).
  behaviors -- dict[str, bool] of named behaviors the response exhibited; the
               keys are the "capabilities" that Parts 2 and 3 ask the model to
               pre-commit to.  Each environment documents which are hacks,
               which are benign-but-unnecessary, which are benign-and-needed,
               and the task outcome itself; `behavior_questions` gives the
               natural-language statement used to ask about each.
"""
from dataclasses import dataclass, field


@dataclass
class Task:
    id: str
    messages: list          # initial chat prefix (system + user)
    meta: dict = field(default_factory=dict)


class Env:
    name: str
    behavior_kinds: dict    # behavior name -> "hack" | "benign_unneeded" | "benign_needed"

    def tasks(self) -> list[Task]:
        raise NotImplementedError

    def score(self, task: Task, response: str) -> dict:
        raise NotImplementedError

    def score_many(self, tasks, responses):
        return [self.score(t, r) for t, r in zip(tasks, responses)]
