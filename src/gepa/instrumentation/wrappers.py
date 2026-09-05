# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Transparent instrumentation wrappers for GEPA's injected dependencies.

These wrappers observe the two resource-bearing call sites — the verifier
(``adapter.evaluate`` / ``adapter.batch_evaluate``) and the reflector
(the reflection LM) — without altering behavior. They forward every call to the
wrapped object and only *count*: rollouts, batches, reflector calls, tokens,
cost, and the structural dispatch widths that bound peak parallelism.

Because they sit at the dependency-injection seam, no optimizer code is
modified. The wrappers are behaviorally identical to the objects they wrap:

- :class:`LedgerAdapter` delegates ``evaluate`` verbatim and provides a
  ``batch_evaluate`` that either delegates to the inner adapter's real
  ``batch_evaluate`` (true batching/parallelism preserved) or replicates the
  engine's own ``default_batch_evaluate`` fallback (sequential ``evaluate`` with
  ``capture_traces=True``) — the exact code path GEPA would have taken.
- :class:`LedgerReflectionLM` delegates ``__call__`` and ``batch_complete`` and
  proxies ``total_cost`` / ``total_tokens_*`` so cost stoppers and GEPA's own
  ``TrackingLM`` detection keep working.
"""

from __future__ import annotations

import time
from typing import Any

from gepa.core.adapter import EvaluationBatch, GEPAAdapter

# Attribute names a resource-bearing model may expose.
_COST_ATTR = "total_cost"
_TOK_IN_ATTR = "total_tokens_in"
_TOK_OUT_ATTR = "total_tokens_out"


def _read_resources(obj: Any) -> tuple[float, int, int] | None:
    """Return ``(cost, tokens_in, tokens_out)`` if ``obj`` exposes them."""
    if obj is None or not hasattr(obj, _COST_ATTR):
        return None
    cost = float(getattr(obj, _COST_ATTR, 0.0) or 0.0)
    tin = int(getattr(obj, _TOK_IN_ATTR, 0) or 0)
    tout = int(getattr(obj, _TOK_OUT_ATTR, 0) or 0)
    return cost, tin, tout


class LedgerAdapter:
    """Wrap a :class:`GEPAAdapter`, counting verifier work transparently.

    ``num_metric_calls`` on the returned :class:`EvaluationBatch` is respected as
    the actual-execution count (cache-miss aware); when the adapter leaves it
    ``None``, the batch length is used — exactly how the engine and proposer
    convert an eval into a budget increment.
    """

    def __init__(self, inner: GEPAAdapter, ledger) -> None:
        self._inner = inner
        self._ledger = ledger
        # Expose the proposal hook as a real attribute so api.py's
        # ``adapter.propose_new_texts is not None`` check sees the inner value.
        self.propose_new_texts = getattr(inner, "propose_new_texts", None)
        # Baseline verifier resources (delta computed at flush).
        self._holder = self._resource_holder()
        base = _read_resources(self._holder)
        self._base = base if base is not None else (0.0, 0, 0)
        # Baseline empty-response counts (data-quality gate; delta at flush).
        self._base_responses = int(getattr(self._holder, "total_responses", 0) or 0)
        self._base_empty = int(getattr(self._holder, "empty_responses", 0) or 0)

    def _resource_holder(self) -> Any:
        for cand in (
            self._inner,
            getattr(self._inner, "model", None),
            getattr(self._inner, "task_lm", None),
            getattr(self._inner, "lm", None),
            getattr(self._inner, "_lm", None),
        ):
            if cand is not None and hasattr(cand, _COST_ATTR):
                return cand
        return None

    # --- counting helpers ---------------------------------------------------
    def _count_eval(self, batch_len: int, result: EvaluationBatch) -> None:
        actual = result.num_metric_calls if result.num_metric_calls is not None else batch_len
        self._ledger.verifier_calls += int(actual)
        self._ledger.verifier_examples_requested += int(batch_len)
        self._ledger.verifier_batches += 1

    def _bump_peak_rollouts(self, examples_in_dispatch: int) -> None:
        if examples_in_dispatch > self._ledger.peak_rollouts_per_dispatch:
            self._ledger.peak_rollouts_per_dispatch = examples_in_dispatch

    # --- delegated surface --------------------------------------------------
    def evaluate(self, batch: list, candidate: dict, capture_traces: bool = False) -> EvaluationBatch:
        t0 = time.perf_counter()
        result = self._inner.evaluate(batch, candidate, capture_traces=capture_traces)
        self._ledger.verifier_wall_s += time.perf_counter() - t0
        self._count_eval(len(batch), result)
        self._bump_peak_rollouts(len(batch))
        return result

    def batch_evaluate(self, items: list[tuple[dict, list]]) -> list[EvaluationBatch]:
        # Structural peak parallelism: how many (candidate, batch) pairs and how
        # many total examples were dispatched together.
        if len(items) > self._ledger.peak_candidates_per_dispatch:
            self._ledger.peak_candidates_per_dispatch = len(items)
        total_examples = sum(len(b) for _, b in items)
        self._bump_peak_rollouts(total_examples)

        inner_batch = getattr(self._inner, "batch_evaluate", None)
        t0 = time.perf_counter()
        if inner_batch is not None:
            results = inner_batch(items)
        else:
            # Exact replica of core.adapter.default_batch_evaluate.
            results = [self._inner.evaluate(batch, candidate, capture_traces=True) for candidate, batch in items]
        self._ledger.verifier_wall_s += time.perf_counter() - t0

        for (_candidate, batch), result in zip(items, results, strict=False):
            self._count_eval(len(batch), result)
        return results

    def make_reflective_dataset(self, candidate: dict, eval_batch: EvaluationBatch, components_to_update: list[str]):
        return self._inner.make_reflective_dataset(candidate, eval_batch, components_to_update)

    def flush_resources(self) -> None:
        """Fold accumulated verifier tokens/cost (delta over the run) into the ledger."""
        holder = self._resource_holder()
        cur = _read_resources(holder)
        if cur is None:
            return
        self._ledger.verifier_cost += cur[0] - self._base[0]
        self._ledger.verifier_tokens_in += cur[1] - self._base[1]
        self._ledger.verifier_tokens_out += cur[2] - self._base[2]
        self._ledger.verifier_responses += int(getattr(holder, "total_responses", 0) or 0) - self._base_responses
        self._ledger.verifier_empty += int(getattr(holder, "empty_responses", 0) or 0) - self._base_empty

    def __getattr__(self, name: str) -> Any:
        # Delegate everything else (get_adapter_state, set_adapter_state, custom
        # attributes) to the wrapped adapter.
        return getattr(self._inner, name)


class LedgerReflectionLM:
    """Wrap a reflection LM callable, counting reflector calls / tokens / cost.

    Proxies ``total_cost`` / ``total_tokens_*`` from the inner model so GEPA's
    ``optimize()`` treats it as an already-tracking LM (no re-wrapping) and the
    reflection-cost stopper reads real spend.
    """

    def __init__(self, inner: Any, ledger) -> None:
        self._inner = inner
        self._ledger = ledger
        base = _read_resources(inner)
        self._base = base if base is not None else (0.0, 0, 0)

    # Proxy resource attributes so hasattr(..., "total_cost") is True.
    @property
    def total_cost(self) -> float:
        return float(getattr(self._inner, _COST_ATTR, 0.0) or 0.0)

    @property
    def total_tokens_in(self) -> int:
        return int(getattr(self._inner, _TOK_IN_ATTR, 0) or 0)

    @property
    def total_tokens_out(self) -> int:
        return int(getattr(self._inner, _TOK_OUT_ATTR, 0) or 0)

    def _bump_prompts_per_call(self, n: int) -> None:
        if n > self._ledger.peak_reflection_prompts_per_call:
            self._ledger.peak_reflection_prompts_per_call = n

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        self._ledger.reflector_calls += 1
        self._bump_prompts_per_call(1)
        t0 = time.perf_counter()
        result = self._inner(prompt)
        self._ledger.reflection_wall_s += time.perf_counter() - t0
        return result

    def __getattr__(self, name: str) -> Any:
        # Conditionally expose batch_complete exactly when the inner LM does, so
        # StatelessReflectionLM's batched path is not silently downgraded.
        if name == "batch_complete":
            inner_batch = getattr(self._inner, "batch_complete", None)
            if inner_batch is None:
                raise AttributeError(name)

            def tracked_batch_complete(messages_list, *args, **kwargs):
                n = len(messages_list)
                self._ledger.reflector_calls += n
                self._bump_prompts_per_call(n)
                t0 = time.perf_counter()
                results = inner_batch(messages_list, *args, **kwargs)
                self._ledger.reflection_wall_s += time.perf_counter() - t0
                return results

            return tracked_batch_complete
        return getattr(self._inner, name)

    def flush_resources(self) -> None:
        """Fold accumulated reflector tokens/cost (delta over the run) into the ledger."""
        cur = _read_resources(self._inner)
        if cur is None:
            return
        self._ledger.reflector_cost += cur[0] - self._base[0]
        self._ledger.reflector_tokens_in += cur[1] - self._base[1]
        self._ledger.reflector_tokens_out += cur[2] - self._base[2]
