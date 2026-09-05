# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""A :class:`GEPACallback` that populates a :class:`RunLedger` from events.

This is the second instrumentation seam (the first being the dependency
wrappers). GEPA already fires a rich, observational callback stream; the ledger
callback simply listens and tallies. It never mutates state and never influences
control flow, so attaching it leaves the optimization byte-for-byte unchanged.

Mapping from events to counters:

- ``on_candidate_selected``  -> parent-of-proposal log
- ``on_minibatch_sampled``   -> distinct dev items + ordered minibatch schedule
- ``on_evaluation_end``      -> minibatch candidate-example evaluations (requested)
- ``on_valset_evaluated``    -> valset candidate-example evaluations + per-item validation reuse
- ``on_proposal_end``        -> reflective candidate proposals (mutation attempts)
- ``on_candidate_accepted``  -> lineage (for critical-path depth) + accept count
- ``on_candidate_rejected``  -> reject count
- ``on_merge_*``             -> merge attempt / accept / reject counts
- ``on_optimization_end``    -> best candidate, pool size, totals, total wall-clock
"""

from __future__ import annotations

import time
from typing import Any

from gepa.instrumentation.ledger import RunLedger


class LedgerCallback:
    """Observational callback that fills a :class:`RunLedger`."""

    def __init__(self, ledger: RunLedger) -> None:
        self.ledger = ledger
        self._t0: float | None = None

    # --- lifecycle ----------------------------------------------------------
    def on_optimization_start(self, event: dict[str, Any]) -> None:
        self._t0 = time.perf_counter()
        cfg = self.ledger.config
        cfg.setdefault("trainset_size", event.get("trainset_size"))
        cfg.setdefault("valset_size", event.get("valset_size"))
        # Fold in the engine's own reported config (seed, perfect_score, ...)
        # without overwriting anything the caller pinned.
        for k, v in (event.get("config") or {}).items():
            cfg.setdefault(k, v)

    def on_optimization_end(self, event: dict[str, Any]) -> None:
        final_state = event.get("final_state")
        total_candidates = len(final_state.program_candidates) if final_state is not None else None
        self.ledger.record_finalization(
            best_candidate_idx=event["best_candidate_idx"],
            total_candidates=total_candidates,
            total_iterations=event["total_iterations"],
            total_metric_calls=event["total_metric_calls"],
        )
        if self._t0 is not None:
            self.ledger.wall_clock_total_s = time.perf_counter() - self._t0

    # --- selection / sampling ----------------------------------------------
    def on_candidate_selected(self, event: dict[str, Any]) -> None:
        # candidate_idx here is the PARENT chosen for mutation.
        self.ledger.record_parent_selected(event["candidate_idx"])

    def on_minibatch_sampled(self, event: dict[str, Any]) -> None:
        self.ledger.record_minibatch(event["iteration"], event["minibatch_ids"])

    # --- evaluation ---------------------------------------------------------
    def on_evaluation_end(self, event: dict[str, Any]) -> None:
        # Fires for parent and child minibatch evaluations (never valset).
        scores = event.get("scores") or []
        self.ledger.record_minibatch_eval(len(scores))

    def on_valset_evaluated(self, event: dict[str, Any]) -> None:
        val_ids = list((event.get("scores_by_val_id") or {}).keys())
        is_seed = event.get("iteration", -1) == 0
        self.ledger.record_valset_eval(val_ids, is_seed=is_seed)

    # --- reflection ---------------------------------------------------------
    def on_proposal_end(self, event: dict[str, Any]) -> None:
        # One event per child candidate that received new component texts.
        self.ledger.record_proposal_reflective()

    # --- acceptance ---------------------------------------------------------
    def on_candidate_accepted(self, event: dict[str, Any]) -> None:
        # Fires for both reflective and merge acceptances; covers all lineage.
        self.ledger.record_candidate_accepted(
            idx=event["new_candidate_idx"],
            parent_ids=list(event.get("parent_ids") or []),
        )

    def on_candidate_rejected(self, event: dict[str, Any]) -> None:
        self.ledger.record_candidate_rejected()

    # --- merges -------------------------------------------------------------
    def on_merge_attempted(self, event: dict[str, Any]) -> None:
        self.ledger.record_merge_attempted()

    def on_merge_accepted(self, event: dict[str, Any]) -> None:
        # Lineage is recorded via on_candidate_accepted (also fired for merges);
        # here we only bump the merge-accept counter.
        self.ledger.merge_accepted += 1

    def on_merge_rejected(self, event: dict[str, Any]) -> None:
        self.ledger.record_merge_rejected()
