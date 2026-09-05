# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Thin LM abstraction over LiteLLM that handles retries, truncation
warnings, and cross-model compatibility.

Usage::

    from gepa.lm import LM

    lm = LM("openai/gpt-4.1", temperature=0.7, max_tokens=4096)
    response: str = lm("Solve this problem...")

    # Also works with chat messages
    response = lm([{"role": "user", "content": "Hello"}])

The returned callable conforms to the ``LanguageModel`` protocol
(``(str | list[dict]) -> str``) used throughout GEPA.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)


class LM:
    """A lightweight language model wrapper over LiteLLM.

    Handles:

    - **Retries** with exponential backoff via LiteLLM's ``num_retries``.
    - **Truncation detection** — logs a warning when ``finish_reason='length'``.
    - **drop_params=True** so unsupported params are silently ignored
      (with a warning logged for transparency).

    Conforms to the :class:`~gepa.proposer.reflective_mutation.base.LanguageModel`
    protocol, so it can be used anywhere GEPA expects a ``LanguageModel``.

    Args:
        model: LiteLLM model identifier, e.g. ``"openai/gpt-4.1"`` or ``"anthropic/claude-sonnet-4-6"``.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        num_retries: Number of retries on transient failures (default 3).
        **kwargs: Extra keyword arguments forwarded to ``litellm.completion``
            (e.g. ``top_p``, ``stop``, ``api_key``, ``api_base``).
    """

    def __init__(
        self,
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        num_retries: int = 1,
        **kwargs: Any,
    ):
        self.model = model
        self.num_retries = num_retries
        self._total_cost: float = 0.0
        self._total_tokens_in: int = 0
        self._total_tokens_out: int = 0
        # Empty-response accounting for the data-quality gate: a completion that
        # comes back blank (provider stall/rate-limit abandoned to "") is scored
        # as a failure downstream, so a high empty rate means the run's scores
        # are contaminated, not that the model is bad.
        self._total_responses: int = 0
        self._empty_responses: int = 0
        self._cost_lock = threading.Lock()

        self.completion_kwargs: dict[str, Any] = {
            **({"temperature": temperature} if temperature is not None else {}),
            **({"max_tokens": max_tokens} if max_tokens is not None else {}),
            **kwargs,
        }
        # Default per-request timeout so a stalled OpenRouter connection fails
        # fast and is retried (num_retries) instead of hanging the whole run.
        self.completion_kwargs.setdefault("timeout", 90)

    def _record_responses(self, texts: list[str]) -> None:
        empt = sum(1 for t in texts if not (t or "").strip())
        with self._cost_lock:
            self._total_responses += len(texts)
            self._empty_responses += empt

    @property
    def total_responses(self) -> int:
        return self._total_responses

    @property
    def empty_responses(self) -> int:
        return self._empty_responses

    @property
    def total_cost(self) -> float:
        """Cumulative USD cost of all calls made through this LM instance."""
        return self._total_cost

    @property
    def total_tokens_in(self) -> int:
        """Cumulative input (prompt) tokens across all calls."""
        return self._total_tokens_in

    @property
    def total_tokens_out(self) -> int:
        """Cumulative output (completion) tokens across all calls."""
        return self._total_tokens_out

    def _check_truncation(self, choices: list[Any], effective_kwargs: dict | None = None,
                          usage: Any = None) -> None:
        if any(getattr(c, "finish_reason", None) == "length" for c in choices):
            # Report the EFFECTIVE cap actually sent on this call. batch_complete merges
            # per-call kwargs (e.g. an adapter's litellm_batch_completion_kwargs) over the
            # LM's init kwargs, so reading self.completion_kwargs alone misreports it as
            # None for a bare LM(model) even though a real cap was applied. Also surface
            # completion_tokens so a genuine long answer is distinguishable from a
            # misconfigured cap or a degenerate loop.
            src = effective_kwargs if effective_kwargs is not None else self.completion_kwargs
            max_tok = src.get("max_tokens") or src.get("max_completion_tokens")
            produced = getattr(usage, "completion_tokens", None) if usage is not None else None
            extra = f", produced={produced}" if produced is not None else ""
            logger.warning(
                f"LM response was truncated (finish_reason='length', max_tokens={max_tok}{extra}). "
                "Consider increasing max_tokens for better results."
            )

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        import concurrent.futures

        import litellm

        if isinstance(prompt, str):
            messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        else:
            messages = prompt

        # Hard wall on the single call too (the reflector path): litellm's own
        # timeout has proven unreliable against OpenRouter stalls, so run the
        # request in a worker thread and abandon it if it overruns the wall
        # (attempt timeout * attempts + buffer). Prevents an indefinite hang.
        _t = self.completion_kwargs.get("timeout", 90) or 90
        _single_wall = _t * (self.num_retries + 1) + 30

        def _do():  # noqa: ANN202
            return litellm.completion(
                model=self.model,
                messages=messages,
                num_retries=self.num_retries,
                drop_params=True,
                **self.completion_kwargs,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _ex:
            try:
                completion = _ex.submit(_do).result(timeout=_single_wall)
            except Exception:
                self._record_responses([""])  # abandoned / failed
                return ""  # abandoned / failed -> empty response, never hang

        # Non-streaming calls always return ModelResponse (not CustomStreamWrapper)
        self._check_truncation(completion.choices, self.completion_kwargs,
                               getattr(completion, "usage", None))  # type: ignore[union-attr]

        # Accumulate cost
        try:
            cost = litellm.completion_cost(completion_response=completion) or 0.0  # type: ignore[attr-defined]
        except Exception:
            cost = 0.0

        # Accumulate token usage
        usage = getattr(completion, "usage", None)
        tokens_in = (getattr(usage, "prompt_tokens", 0) or 0) if usage is not None else 0
        tokens_out = (getattr(usage, "completion_tokens", 0) or 0) if usage is not None else 0

        with self._cost_lock:
            self._total_cost += cost
            self._total_tokens_in += tokens_in
            self._total_tokens_out += tokens_out

        _out = completion.choices[0].message.content or ""  # type: ignore[union-attr]
        self._record_responses([_out])
        return _out

    def batch_complete(
        self, messages_list: list[list[dict[str, Any]]], max_workers: int = 10, **kwargs: Any
    ) -> list[str]:
        """Run multiple completions in parallel using ``litellm.batch_completion``.

        Args:
            messages_list: List of message lists, one per request.
            max_workers: Maximum concurrent requests.
            **kwargs: Extra keyword arguments forwarded to ``litellm.batch_completion``
                (e.g. ``timeout``, ``api_base``).  These override any matching keys
                set during ``__init__``.

        Returns:
            List of response strings, one per input.
        """
        import concurrent.futures
        import time

        import litellm

        merged = {**self.completion_kwargs, **kwargs}
        # litellm.batch_completion's own `timeout` has proven unreliable against
        # certain OpenRouter stalls (a single hung socket blocks the whole batch
        # indefinitely). Design goals here:
        #   1. EVERY request fires concurrently (its own thread), so one hung
        #      request never delays the others (no serial-blocking on a small
        #      pool where zombie threads would starve queued work).
        #   2. A single GLOBAL batch deadline: whatever hasn't returned by then
        #      is left as "" (empty completion). Guarantees the batch — and thus
        #      the run — always makes forward progress in bounded wall time.
        # Symmetric across arms; pure I/O robustness, no search-logic change.
        n = len(messages_list)
        if n == 0:
            return []
        # Per-request attempt timeout: honour the configured timeout (capped) so a
        # SLOW model (e.g. a thinking model that legitimately needs 60-120s per
        # answer) isn't abandoned to "" and mass-contaminated. A 30s cap suits an
        # instruct model but wrongly kills thinking-model responses mid-generation.
        attempt_timeout = min(merged.get("timeout", 90) or 90, 600)
        internal_retries = 1  # at most 2 attempts per request
        _merged = {**merged, "timeout": attempt_timeout}
        # Global wall generous enough for a couple of slow waves, then give up.
        global_wall = attempt_timeout * (internal_retries + 1) + 90
        pool = min(n, max(int(max_workers), 8))

        def _one(messages: list[dict[str, Any]]):  # noqa: ANN202
            return litellm.completion(
                model=self.model,
                messages=messages,
                num_retries=internal_retries,
                drop_params=True,
                **_merged,
            )

        results: list[str] = [""] * n
        batch_cost = 0.0
        batch_tokens_in = 0
        batch_tokens_out = 0
        ex = concurrent.futures.ThreadPoolExecutor(max_workers=pool)
        try:
            fut_to_idx = {ex.submit(_one, m): i for i, m in enumerate(messages_list)}
            deadline = time.monotonic() + global_wall
            try:
                for fut in concurrent.futures.as_completed(fut_to_idx, timeout=global_wall):
                    i = fut_to_idx[fut]
                    try:
                        resp = fut.result(timeout=max(0.1, deadline - time.monotonic()))
                    except Exception:
                        results[i] = ""
                        continue
                    if isinstance(resp, Exception) or not getattr(resp, "choices", None):
                        results[i] = ""
                        continue
                    self._check_truncation(resp.choices, _merged, getattr(resp, "usage", None))
                    _content = resp.choices[0].message.content
                    results[i] = (_content or "").strip()
                    try:
                        batch_cost += litellm.completion_cost(completion_response=resp) or 0.0  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    usage = getattr(resp, "usage", None)
                    if usage is not None:
                        batch_tokens_in += getattr(usage, "prompt_tokens", 0) or 0
                        batch_tokens_out += getattr(usage, "completion_tokens", 0) or 0
            except concurrent.futures.TimeoutError:
                # Global deadline hit: any request not yet done stays "".
                pass
        finally:
            # Don't block on hung requests; let stragglers die with the process.
            ex.shutdown(wait=False, cancel_futures=True)

        with self._cost_lock:
            self._total_cost += batch_cost
            self._total_tokens_in += batch_tokens_in
            self._total_tokens_out += batch_tokens_out

        self._record_responses(results)
        return results

    def __repr__(self) -> str:
        params = [f"model={self.model!r}"]
        for k, v in self.completion_kwargs.items():
            params.append(f"{k}={v!r}")
        return f"LM({', '.join(params)})"


class TrackingLM:
    """Wraps an arbitrary callable to track estimated token usage.

    For callables that don't go through LiteLLM, we can't get real token
    counts or costs.  This wrapper estimates tokens from string lengths
    (~4 chars/token) and reports ``total_cost = 0.0``.

    This ensures that *every* reflection LM — whether an ``LM`` instance
    or a plain callable — exposes ``total_cost``, ``total_tokens_in``,
    and ``total_tokens_out``.
    """

    _CHARS_PER_TOKEN = 4

    def __init__(self, fn: Any):
        self._fn = fn
        self._total_cost: float = 0.0
        self._total_tokens_in: int = 0
        self._total_tokens_out: int = 0

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def total_tokens_in(self) -> int:
        return self._total_tokens_in

    @property
    def total_tokens_out(self) -> int:
        return self._total_tokens_out

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // self._CHARS_PER_TOKEN)

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        if isinstance(prompt, str):
            self._total_tokens_in += self._estimate_tokens(prompt)
        else:
            self._total_tokens_in += self._estimate_tokens(str(prompt))

        result = self._fn(prompt)

        if isinstance(result, str):
            self._total_tokens_out += self._estimate_tokens(result)

        return result

    def __getattr__(self, name: str):
        # Conditionally expose batch_complete: hasattr(tracking_lm,
        # "batch_complete") must be True exactly when the wrapped callable
        # provides it, so batched reflection (StatelessReflectionLM) is not
        # silently downgraded to the per-task path by this wrapper.
        if name == "batch_complete":
            inner = getattr(self._fn, "batch_complete", None)
            if inner is None:
                raise AttributeError(name)

            def tracked_batch_complete(messages_list):
                for messages in messages_list:
                    self._total_tokens_in += self._estimate_tokens(str(messages))
                results = list(inner(messages_list))
                for result in results:
                    if isinstance(result, str):
                        self._total_tokens_out += self._estimate_tokens(result)
                return results

            return tracked_batch_complete
        raise AttributeError(name)

    def __repr__(self) -> str:
        return f"TrackingLM({self._fn!r})"
