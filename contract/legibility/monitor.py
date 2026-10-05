"""Client for the frozen monitor: a vLLM server (the base model, thinking off) reached over HTTP.

    monitor = Monitor()                      # URL from $MONITOR_URL, else from --monitor-url-file, else $LLM_BASE_URL
    answers = monitor.judge(chains, names)   # one {behavior: probability | None} per chain

Training jobs run the monitor on a second GPU of the same allocation (scripts/train_legibility.sbatch starts it and
exports MONITOR_URL); evaluation jobs point it at the vLLM server that scripts/serve_and_run.sbatch started.
"""
import os

from contract.legibility.prompts import MONITOR_STATEMENTS, MONITOR_SYSTEM_PROMPT, MONITOR_TEMPLATE
from contract.llm import LLM
from contract.run_tasks import parse_precommit


class Monitor:
    def __init__(self, model="Qwen/Qwen3-4B", url=None, url_file=None, max_tokens=64, workers=64):
        if url is None and url_file and os.path.exists(url_file):
            url = open(url_file).read().strip()
        url = url or os.environ.get("MONITOR_URL") or os.environ.get("LLM_BASE_URL")
        self.llm = LLM(model=model, base_url=url, thinking=False, temperature=0.0, max_tokens=max_tokens, workers=workers)

    def messages(self, chain, names):
        statements = "\n".join(f"{i + 1}. {MONITOR_STATEMENTS[b]}" for i, b in enumerate(names))
        return [{"role": "system", "content": MONITOR_SYSTEM_PROMPT},
                {"role": "user", "content": MONITOR_TEMPLATE.format(chain=(chain or "(empty)").strip(), statements=statements)}]

    def judge(self, chains, names):
        """Probabilities per statement for each chain (None where the monitor's answer did not parse), plus raw texts."""
        outs = self.llm.chat_many([self.messages(c, names) for c in chains], n=1)
        raw = [o[0]["content"] for o in outs]
        return [dict(zip(names, parse_precommit(r, "prob", len(names)))) for r in raw], raw
