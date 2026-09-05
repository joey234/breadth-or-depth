# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Phase 1 instrumentation: a per-run resource ledger for the Breadth-or-Depth study.

The ledger is composed entirely at GEPA's dependency-injection seams — the
adapter, the reflection LM, and the callback list — so **no optimizer code is
modified**. Two entry points are provided:

- :func:`attach_ledger` — build the wrapped dependencies + ledger callback for a
  caller who wants to drive :func:`gepa.optimize` themselves.
- :func:`optimize_with_ledger` — a thin convenience wrapper that runs
  :func:`gepa.optimize` with instrumentation attached and writes the ledger.

Exit test (see ``tests/test_ledger_determinism.py``): two runs with the same
seed produce byte-identical ``ledger.json``.
"""

from __future__ import annotations

from typing import Any

from gepa.instrumentation.callback import LedgerCallback
from gepa.instrumentation.ledger import RunLedger
from gepa.instrumentation.wrappers import LedgerAdapter, LedgerReflectionLM

__all__ = [
    "RunLedger",
    "LedgerAdapter",
    "LedgerReflectionLM",
    "LedgerCallback",
    "attach_ledger",
    "optimize_with_ledger",
]


def _build_reflection_lm(reflection_lm: Any, reflection_lm_kwargs: dict[str, Any] | None) -> Any | None:
    """Resolve a reflection LM into a resource-bearing callable (mirrors api.py)."""
    if reflection_lm is None:
        return None
    if isinstance(reflection_lm, str):
        from gepa.lm import LM

        return LM(reflection_lm, **(reflection_lm_kwargs or {}))
    if hasattr(reflection_lm, "total_cost"):
        return reflection_lm
    from gepa.lm import TrackingLM

    return TrackingLM(reflection_lm)


def attach_ledger(
    *,
    adapter: Any = None,
    task_lm: Any = None,
    task_lm_kwargs: dict[str, Any] | None = None,
    evaluator: Any = None,
    reflection_lm: Any = None,
    reflection_lm_kwargs: dict[str, Any] | None = None,
    callbacks: list[Any] | None = None,
    seed: int | None = None,
    run_label: str = "run",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compose instrumentation without running anything.

    Returns a dict with the keys the caller should forward to
    :func:`gepa.optimize`:

    - ``adapter``       — the wrapped adapter (pass ``adapter=...``, ``task_lm=None``)
    - ``reflection_lm`` — the wrapped reflection LM (already resource-bearing)
    - ``callbacks``     — the caller's callbacks plus the ledger callback
    - ``ledger``        — the :class:`RunLedger` being populated
    - ``flush``         — call after the run to fold in tokens/cost before writing

    Exactly one of ``adapter`` or ``task_lm`` should be provided, matching
    :func:`gepa.optimize`.
    """
    if adapter is not None and task_lm is not None:
        raise ValueError("Provide either adapter or task_lm, not both (matches gepa.optimize).")

    inner_adapter = adapter
    if inner_adapter is None:
        if task_lm is None:
            raise ValueError("attach_ledger requires either an adapter or a task_lm.")
        from gepa.adapters.default_adapter.default_adapter import DefaultAdapter

        # task_lm_kwargs (max_tokens, extra_body reasoning toggle, temperature) are
        # applied to every task rollout via batch_complete — this is how the runner
        # selects instruct vs thinking mode and sizes the token budget per model.
        inner_adapter = DefaultAdapter(
            model=task_lm,
            evaluator=evaluator,
            litellm_batch_completion_kwargs=dict(task_lm_kwargs or {}),
        )

    ledger = RunLedger(run_label=run_label, seed=seed, config=dict(config or {}))

    wrapped_adapter = LedgerAdapter(inner_adapter, ledger)

    inner_reflection = _build_reflection_lm(reflection_lm, reflection_lm_kwargs)
    wrapped_reflection = LedgerReflectionLM(inner_reflection, ledger) if inner_reflection is not None else None

    cb = LedgerCallback(ledger)
    all_callbacks = list(callbacks or []) + [cb]

    def flush() -> None:
        wrapped_adapter.flush_resources()
        if wrapped_reflection is not None:
            wrapped_reflection.flush_resources()

    return {
        "adapter": wrapped_adapter,
        "reflection_lm": wrapped_reflection,
        "callbacks": all_callbacks,
        "ledger": ledger,
        "flush": flush,
    }


def optimize_with_ledger(
    *,
    ledger_dir: str | None = None,
    ledger_basename: str = "ledger",
    run_label: str = "run",
    ledger_config: dict[str, Any] | None = None,
    **optimize_kwargs: Any,
) -> tuple[Any, RunLedger]:
    """Run :func:`gepa.optimize` with a resource ledger attached and emitted.

    All ``optimize_kwargs`` are forwarded verbatim to :func:`gepa.optimize`
    except that ``adapter`` / ``task_lm`` / ``reflection_lm`` / ``callbacks`` are
    routed through :func:`attach_ledger` first. The ledger is written to
    ``ledger_dir`` (defaults to ``run_dir`` if set, else the current directory).

    Returns ``(result, ledger)``.
    """
    import gepa

    seed = optimize_kwargs.get("seed", 0)
    attached = attach_ledger(
        adapter=optimize_kwargs.pop("adapter", None),
        task_lm=optimize_kwargs.pop("task_lm", None),
        task_lm_kwargs=optimize_kwargs.pop("task_lm_kwargs", None),
        evaluator=optimize_kwargs.pop("evaluator", None),
        reflection_lm=optimize_kwargs.pop("reflection_lm", None),
        reflection_lm_kwargs=optimize_kwargs.pop("reflection_lm_kwargs", None),
        callbacks=optimize_kwargs.pop("callbacks", None),
        seed=seed,
        run_label=run_label,
        config=ledger_config,
    )

    optimize_kwargs["adapter"] = attached["adapter"]
    optimize_kwargs["task_lm"] = None
    optimize_kwargs["evaluator"] = None
    optimize_kwargs["reflection_lm"] = attached["reflection_lm"]
    optimize_kwargs["callbacks"] = attached["callbacks"]

    ledger: RunLedger = attached["ledger"]
    try:
        result = gepa.optimize(**optimize_kwargs)
    finally:
        attached["flush"]()

    out_dir = ledger_dir or optimize_kwargs.get("run_dir") or "."
    ledger.write(out_dir, basename=ledger_basename)
    return result, ledger
