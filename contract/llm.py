"""Thin client for a vLLM server speaking the OpenAI chat API.

All model access in this project goes through this class so that the same
code can talk to a local vLLM server on the cluster (the default) or to any
other OpenAI-compatible endpoint.
"""
import math
import os
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI


class LLM:
    def __init__(
        self,
        model="Qwen/Qwen3-4B",
        base_url=None,
        thinking=False,
        temperature=0.7,
        top_p=0.95,
        max_tokens=2048,
        workers=64,
    ):
        base_url = base_url or os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
        self.client = OpenAI(base_url=base_url, api_key="none", timeout=1800, max_retries=5)
        self.model = model
        self.thinking = thinking
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.workers = workers

    def chat(self, messages, n=1):
        """Sample n completions. Returns a list of {content, reasoning, finish_reason}."""
        r = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            n=n,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            # Qwen3-specific: the chat template takes an enable_thinking flag.
            extra_body={"chat_template_kwargs": {"enable_thinking": self.thinking}},
        )
        return [
            {
                "content": c.message.content or "",
                # vLLM has used both field names for the parsed <think> block across versions.
                "reasoning": getattr(c.message, "reasoning_content", None) or getattr(c.message, "reasoning", None),
                "finish_reason": c.finish_reason,
            }
            for c in r.choices
        ]

    def chat_many(self, message_lists, n=1):
        with ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(lambda m: self.chat(m, n=n), message_lists))

    def next_token_probs(self, messages, top=20):
        """Probability of each candidate first token of the assistant's reply.

        Thinking is always disabled here so that the first generated token is
        the answer itself rather than the start of a reasoning block.
        """
        r = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=1,
            temperature=0.0,
            logprobs=True,
            top_logprobs=top,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        top_logprobs = r.choices[0].logprobs.content[0].top_logprobs
        return {t.token: math.exp(t.logprob) for t in top_logprobs}

    def next_token_probs_many(self, message_lists, top=20):
        with ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(lambda m: self.next_token_probs(m, top=top), message_lists))
