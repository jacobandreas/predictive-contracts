"""Client for the frozen monitor: a vLLM server (the base model, thinking off) reached over HTTP.

    monitor = Monitor()                      # URL from $MONITOR_URL, else from --monitor-url-file, else $LLM_BASE_URL
    answers = monitor.judge(chains, names)   # one {behavior: probability | None} per chain
    answers = monitor.judge(chains, names, outputs)   # diagnostic: the monitor also sees each final answer

Training jobs run the monitor on a second GPU of the same allocation (scripts/train_legibility.sbatch starts it and
exports MONITOR_URL); evaluation jobs point it at the vLLM server that scripts/serve_and_run.sbatch started.
"""
import os

from contract.legibility.prompts import MONITOR_STATEMENTS, MONITOR_SYSTEM_PROMPT, MONITOR_TEMPLATE, MONITOR_TEMPLATE_WITH_OUTPUT
from contract.llm import LLM
from contract.run_tasks import parse_precommit


class Monitor:
    def __init__(self, model="Qwen/Qwen3-4B", url=None, url_file=None, max_tokens=64, workers=64, thinking=False, think_budget=1024):
        """`thinking`: let the monitor reason (under `think_budget` tokens) before answering; default off (one short answer)."""
        if url is None and url_file and os.path.exists(url_file):
            url = open(url_file).read().strip()
        url = url or os.environ.get("MONITOR_URL") or os.environ.get("LLM_BASE_URL")
        self.llm = LLM(model=model, base_url=url, thinking=thinking, think_budget=think_budget if thinking else None,
                       temperature=0.0 if not thinking else 0.7, max_tokens=max_tokens, workers=workers)

    def messages(self, chain, names, output=None):
        statements = "\n".join(f"{i + 1}. {MONITOR_STATEMENTS[b]}" for i, b in enumerate(names))
        chain = (chain or "(empty)").strip()
        if output is None:
            user = MONITOR_TEMPLATE.format(chain=chain, statements=statements)
        else:
            user = MONITOR_TEMPLATE_WITH_OUTPUT.format(chain=chain, output=(output or "(empty)").strip(), statements=statements)
        return [{"role": "system", "content": MONITOR_SYSTEM_PROMPT}, {"role": "user", "content": user}]

    def judge(self, chains, names, outputs=None):
        """Probabilities per statement for each chain (None where the monitor's answer did not parse), plus raw texts.
        With `outputs` (one final answer per chain) the monitor sees the answer too."""
        outputs = outputs if outputs is not None else [None] * len(chains)
        outs = self.llm.chat_many([self.messages(c, names, o) for c, o in zip(chains, outputs)], n=1)
        raw = [o[0]["content"] for o in outs]
        return [dict(zip(names, parse_precommit(r, "prob", len(names)))) for r in raw], raw
