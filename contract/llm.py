"""Thin client for a vLLM server speaking the OpenAI chat API.

All model access in this project goes through this class so that the same
code can talk to a local vLLM server on the cluster (the default) or to any
other OpenAI-compatible endpoint, e.g. OpenRouter (base_url
https://openrouter.ai/api/v1, api_key from OPENROUTER_API_KEY).
"""
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from transformers import AutoTokenizer  # imported here, not lazily: a first import from many threads at once fails

from contract.prompts import THINK_BUDGET_STOP


class LLM:
    def __init__(
        self,
        model="Qwen/Qwen3-4B",
        base_url=None,
        api_key=None,
        reasoning_model=False,
        thinking=False,
        temperature=0.7,
        top_p=0.95,
        max_tokens=2048,
        workers=64,
        think_budget=None,
        tokenizer=None,
        reasoning_effort=None,
    ):
        base_url = base_url or os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
        api_key = api_key or os.environ.get("LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY") or "none"
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=1800, max_retries=5)
        self.model = model
        # OpenRouter ignores the vLLM/Qwen3 `chat_template_kwargs.enable_thinking` flag below and ignores
        # thinking control via its own `reasoning` field instead; some OpenRouter-hosted models (e.g. newer
        # Qwen releases) default reasoning on, which can eat the whole max_tokens budget with no answer left.
        self.openrouter = "openrouter" in base_url
        self.reasoning_effort = reasoning_effort  # OpenRouter only: "low"/"medium"/"xhigh" (model-dependent); caps reasoning length
        # OpenAI-style reasoning models (o1/o3/o4-...) reject custom temperature/top_p (only the default is
        # allowed) and have no notion of Qwen3's enable_thinking chat-template kwarg.
        self.reasoning_model = reasoning_model
        self.thinking = thinking
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.workers = workers
        self.think_budget = think_budget  # thinking mode only: max reasoning tokens before the block is force-closed
        # Renders the chat template for the continuation.  `tokenizer` names the base model when `model` is a served
        # adapter alias (vLLM --lora-modules rl=...), which is not a Hugging Face id.
        self.tok = AutoTokenizer.from_pretrained(tokenizer or model) if think_budget else None

    def chat(self, messages, n=1, thinking=None, think_budget=None, max_tokens=None, stop_text=THINK_BUDGET_STOP):
        """Sample n completions. Returns a list of {content, reasoning, finish_reason}.
        `thinking`, `think_budget` and `max_tokens` override the instance defaults for this call (a commitment turn
        generated without a chain, or with a shorter one and a short answer)."""
        thinking = self.thinking if thinking is None else thinking
        think_budget = self.think_budget if think_budget is None else think_budget
        max_tokens = self.max_tokens if max_tokens is None else max_tokens
        if thinking and think_budget:
            return [self.chat_budgeted(messages, think_budget, max_tokens, stop_text) for _ in range(n)]
        kwargs = dict(model=self.model, messages=messages, n=n, max_tokens=max_tokens)
        if not self.reasoning_model:
            extra_body = {"chat_template_kwargs": {"enable_thinking": thinking}}
            if self.openrouter:
                extra_body["reasoning"] = {"enabled": thinking}
                if thinking and self.reasoning_effort:
                    extra_body["reasoning"]["effort"] = self.reasoning_effort
            kwargs.update(temperature=self.temperature, top_p=self.top_p, extra_body=extra_body)
        # OpenRouter occasionally returns a 200 with `choices: null` (a transient upstream-provider glitch,
        # not a request-specific error) -- not retried by the client's own max_retries, which only covers
        # retryable HTTP statuses. Retry the request itself a few times before giving up.
        for attempt in range(4):
            r = self.client.chat.completions.create(**kwargs)
            if r.choices:
                break
        else:
            raise RuntimeError(f"empty/null choices after 4 attempts: {r}")
        return [
            {
                "content": c.message.content or "",
                # vLLM has used both field names for the parsed <think> block across versions.
                "reasoning": getattr(c.message, "reasoning_content", None) or getattr(c.message, "reasoning", None),
                "finish_reason": c.finish_reason,
            }
            for c in r.choices
        ]

    def chat_budgeted(self, messages, think_budget, max_tokens, stop_text):
        """Thinking with a token budget, as in Qwen3's thinking_budget recipe: reason for at most `think_budget`
        tokens; if the block is still open, append THINK_BUDGET_STOP, close it, and let the model answer with up to
        `max_tokens` more.  The continuation goes through the raw completions endpoint with the chat template rendered
        by the model's tokenizer, because Qwen3's template rewrites <think> blocks in assistant messages (so the chat
        endpoint's `continue_final_message` fails).  The <think> markers and the stop string are the one Qwen-specific
        part; another model family would need its own here."""
        r = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=self.temperature, top_p=self.top_p, max_tokens=think_budget,
            extra_body={"chat_template_kwargs": {"enable_thinking": True}})
        c = r.choices[0]
        reasoning = getattr(c.message, "reasoning_content", None) or getattr(c.message, "reasoning", None) or ""
        if c.finish_reason != "length":
            return {"content": c.message.content or "", "reasoning": reasoning, "finish_reason": c.finish_reason, "think_forced": False}
        forced = not c.message.content  # block still open at the budget (a non-empty content means the answer got cut instead)
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        prefix = f"<think>\n{reasoning}\n{stop_text}\n</think>\n\n" if forced else f"<think>\n{reasoning}\n</think>\n\n{c.message.content}"
        r2 = self.client.completions.create(model=self.model, prompt=prompt + prefix, temperature=self.temperature, top_p=self.top_p,
                                            max_tokens=max_tokens, stop=[self.tok.eos_token])
        c2 = r2.choices[0]
        content = c2.text if forced else (c.message.content or "") + c2.text
        return {"content": content, "reasoning": reasoning + ("\n" + stop_text if forced else ""), "finish_reason": c2.finish_reason, "think_forced": forced}

    def chat_many(self, message_lists, n=1, desc=None, callback=None, **kw):
        """Sample for each item in `message_lists` concurrently. With `desc`, prints one progress line (to
        stdout, flushed) as each individual request completes -- rather than only once the whole batch is
        done -- so a long run shows live progress instead of going silent until the end. With `callback`,
        calls `callback(i, result)` synchronously (in the calling thread) as each result comes in, so the
        caller can score and write it to disk immediately rather than waiting for the whole batch to return."""
        results = [None] * len(message_lists)
        start = time.monotonic()
        with ThreadPoolExecutor(self.workers) as ex:
            futures = {ex.submit(self.chat, m, n=n, **kw): i for i, m in enumerate(message_lists)}
            done = 0
            for fut in as_completed(futures):
                i = futures[fut]
                results[i] = fut.result()
                if callback:
                    callback(i, results[i])
                done += 1
                if desc:
                    print(f"{desc}: {done}/{len(message_lists)} ({time.monotonic() - start:.0f}s elapsed)", flush=True)
        return results

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
