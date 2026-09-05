# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Per-run resource ledger (Phase 1 instrumentation).

The ledger is the experiment, not reporting: every claim about "matching" in
the Breadth-or-Depth study reduces to these counters. This module owns the
*accounting object* only — it is a passive accumulator plus a deterministic
serializer. The population of the counters happens elsewhere:

- :class:`gepa.instrumentation.callback.LedgerCallback` reads GEPA's callback
  events (proposals, evaluations, valset reuse, lineage, merges).
- :class:`gepa.instrumentation.wrappers.LedgerAdapter` /
  :class:`gepa.instrumentation.wrappers.LedgerReflectionLM` wrap the injected
  dependencies to count rollouts, reflector calls, tokens, cost, and the
  structural dispatch widths.

Neither the engine, the state, the proposer, nor the strategies are touched:
instrumentation is composed at the dependency-injection seams, so the search is
byte-for-byte identical whether or not a ledger is attached.

Determinism contract
--------------------
The canonical ledger (:meth:`RunLedger.canonical_dict`, written to
``ledger.json``) contains only quantities that are reproducible under a fixed
seed and a deterministic model: counts, ordered schedules, item *sets*, the
lineage / critical-path depth, tokens and cost. Two runs with the same seed
therefore produce **byte-identical** ``ledger.json`` files — the Phase 1 exit
test.

Wall-clock is inherently non-reproducible, so it is quarantined into a separate
``timing.json`` (:meth:`RunLedger.timing_dict`) that is *not* covered by the
byte-identity guarantee. This mirrors the design: §6 treats wall-clock at fixed
concurrency as a measured latency outcome, reported but never matched.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = 1


def _sort_key(value: Any) -> tuple[int, Any]:
    """Deterministic ordering key that tolerates heterogeneous id types.

    Numbers sort numerically before everything else; anything else sorts by its
    ``repr``. Data ids within a single run are homogeneous in practice, so this
    only needs to be *consistent*, not semantically clever.
    """
    if isinstance(value, bool):
        return (1, repr(value))
    if isinstance(value, int | float):
        return (0, value)
    return (1, repr(value))


def sorted_ids(ids: Any) -> list[Any]:
    """Return ids as a list in a stable, reproducible order."""
    return sorted(ids, key=_sort_key)


def _id_str(value: Any) -> str:
    """Canonical string form of a data id, for use as a JSON object key."""
    return str(value)


@dataclass
class RunLedger:
    """Passive accumulator for the Phase 1 resource ledger.

    All ``record_*`` methods are pure state updates with no control-flow effect;
    they are safe to call from callbacks and dependency wrappers. Serialization
    is split into a deterministic :meth:`canonical_dict` and a non-deterministic
    :meth:`timing_dict`.
    """

    run_label: str = "run"
    seed: int | None = None
    config: dict[str, Any] = field(default_factory=dict)

    # --- Candidate proposals -------------------------------------------------
    proposals_reflective: int = 0
    proposals_merge_attempted: int = 0

    # --- Merges --------------------------------------------------------------
    merge_attempted: int = 0
    merge_accepted: int = 0
    merge_rejected: int = 0

    # --- Candidate-example evaluations (logical requests) --------------------
    minibatch_examples_requested: int = 0
    minibatch_eval_batches: int = 0
    valset_examples_requested: int = 0
    seed_examples: int = 0

    # --- Acceptance bookkeeping ---------------------------------------------
    candidates_accepted: int = 0
    candidates_rejected: int = 0

    # --- Distinct dev (train/minibatch) items touched ------------------------
    distinct_dev_items: set = field(default_factory=set)

    # --- Validation reuse (per item) ----------------------------------------
    validation_reuse: dict = field(default_factory=lambda: defaultdict(int))

    # --- Ordered schedules ---------------------------------------------------
    parent_log: list = field(default_factory=list)
    minibatch_ids_log: list = field(default_factory=list)
    minibatch_iterations: list = field(default_factory=list)

    # --- Lineage (for sequential critical-path depth) ------------------------
    # candidate idx -> list of parent indices (from accepted candidates/merges)
    lineage_parents: dict = field(default_factory=dict)

    # --- Final selection -----------------------------------------------------
    best_candidate_idx: int | None = None
    total_candidates: int | None = None
    total_iterations: int | None = None
    total_metric_calls: int | None = None

    # --- Reflector (proposer LM) — from wrapper ------------------------------
    reflector_calls: int = 0
    reflector_tokens_in: int = 0
    reflector_tokens_out: int = 0
    reflector_cost: float = 0.0

    # --- Verifier (rollout + scoring) — from adapter wrapper -----------------
    verifier_calls: int = 0  # actual (candidate, example) executions
    verifier_batches: int = 0  # adapter.evaluate / batch_evaluate invocations
    verifier_examples_requested: int = 0
    verifier_tokens_in: int = 0
    verifier_tokens_out: int = 0
    verifier_cost: float = 0.0

    # --- Data quality: empty task responses (provider stalls scored as fails) --
    verifier_responses: int = 0
    verifier_empty: int = 0

    # --- Structural peak parallelism — from wrappers -------------------------
    peak_candidates_per_dispatch: int = 0
    peak_rollouts_per_dispatch: int = 0
    peak_reflection_prompts_per_call: int = 0

    # --- Timing (non-deterministic; quarantined) -----------------------------
    wall_clock_total_s: float | None = None
    reflection_wall_s: float = 0.0
    verifier_wall_s: float = 0.0
    concurrency_levels: list = field(default_factory=list)

    # ------------------------------------------------------------------
    # Recorders (called by the callback / wrappers)
    # ------------------------------------------------------------------
    def record_minibatch(self, iteration: int, minibatch_ids: list) -> None:
        self.minibatch_iterations.append(iteration)
        self.minibatch_ids_log.append(list(minibatch_ids))
        for mid in minibatch_ids:
            self.distinct_dev_items.add(mid)

    def record_parent_selected(self, parent_idx: int) -> None:
        self.parent_log.append(parent_idx)

    def record_minibatch_eval(self, num_examples: int) -> None:
        self.minibatch_examples_requested += num_examples
        self.minibatch_eval_batches += 1

    def record_valset_eval(self, val_ids: list, is_seed: bool) -> None:
        if is_seed:
            self.seed_examples += len(val_ids)
        else:
            self.valset_examples_requested += len(val_ids)
        for vid in val_ids:
            self.validation_reuse[vid] += 1

    def record_candidate_accepted(self, idx: int, parent_ids: list) -> None:
        self.candidates_accepted += 1
        self.lineage_parents[idx] = [p for p in parent_ids if p is not None]

    def record_candidate_rejected(self) -> None:
        self.candidates_rejected += 1

    def record_proposal_reflective(self) -> None:
        self.proposals_reflective += 1

    def record_merge_attempted(self, idx_parents: list | None = None) -> None:
        self.proposals_merge_attempted += 1
        self.merge_attempted += 1

    def record_merge_accepted(self, idx: int, parent_ids: list) -> None:
        self.merge_accepted += 1
        self.lineage_parents[idx] = [p for p in parent_ids if p is not None]

    def record_merge_rejected(self) -> None:
        self.merge_rejected += 1

    def record_finalization(
        self,
        best_candidate_idx: int,
        total_candidates: int | None,
        total_iterations: int,
        total_metric_calls: int,
    ) -> None:
        self.best_candidate_idx = best_candidate_idx
        self.total_candidates = total_candidates
        self.total_iterations = total_iterations
        self.total_metric_calls = total_metric_calls

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------
    def _depth_by_candidate(self) -> dict[int, int]:
        """Sequential critical-path depth of each candidate's lineage.

        Depth of the seed (index 0) is 0. Depth of any evolved candidate is
        ``1 + max(depth(parent))``. For root-reset every parent is p0, so every
        candidate has depth 1; for stateful GEPA depths grow with the lineage.
        """
        depth: dict[int, int] = {0: 0}

        def resolve(idx: int, _seen: frozenset[int] = frozenset()) -> int:
            if idx in depth:
                return depth[idx]
            if idx in _seen:  # cycle guard (should never happen)
                return 0
            parents = self.lineage_parents.get(idx)
            if not parents:
                depth[idx] = 0
                return 0
            d = 1 + max(resolve(p, _seen | {idx}) for p in parents)
            depth[idx] = d
            return d

        for cidx in self.lineage_parents:
            resolve(cidx)
        return depth

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def canonical_dict(self) -> dict[str, Any]:
        """Deterministic ledger. Byte-identical across same-seed runs."""
        depth = self._depth_by_candidate()
        best = self.best_candidate_idx
        n = self.total_candidates if self.total_candidates is not None else (max(depth.keys()) + 1 if depth else 0)
        final_pool = list(range(n))

        reuse_counts = list(self.validation_reuse.values())
        minibatch_schedule = [
            {
                "iteration": it,
                "parent_idx": (self.parent_log[k] if k < len(self.parent_log) else None),
                "minibatch_ids": sorted_ids(ids),
            }
            for k, (it, ids) in enumerate(zip(self.minibatch_iterations, self.minibatch_ids_log, strict=False))
        ]

        return {
            "schema_version": SCHEMA_VERSION,
            "run_label": self.run_label,
            "seed": self.seed,
            "config": self.config,
            "counters": {
                "candidate_proposals": {
                    "total": self.proposals_reflective + self.proposals_merge_attempted,
                    "reflective": self.proposals_reflective,
                    "merge_attempted": self.proposals_merge_attempted,
                    "accepted": self.candidates_accepted,
                    "rejected": self.candidates_rejected,
                },
                "task_rollouts": {
                    # Actual (candidate, example) executions through the adapter.
                    "total_executed": self.verifier_calls,
                },
                "candidate_example_evaluations": {
                    # Logical score requests (may exceed executions under caching).
                    "total_requested": (
                        self.minibatch_examples_requested + self.valset_examples_requested + self.seed_examples
                    ),
                    "minibatch_requested": self.minibatch_examples_requested,
                    "valset_requested": self.valset_examples_requested,
                    "seed": self.seed_examples,
                },
                "reflector": {
                    "calls": self.reflector_calls,
                    "tokens_in": self.reflector_tokens_in,
                    "tokens_out": self.reflector_tokens_out,
                    "tokens_total": self.reflector_tokens_in + self.reflector_tokens_out,
                    "cost": self.reflector_cost,
                },
                "verifier": {
                    "calls": self.verifier_calls,
                    "batches": self.verifier_batches,
                    "examples_requested": self.verifier_examples_requested,
                    "tokens_in": self.verifier_tokens_in,
                    "tokens_out": self.verifier_tokens_out,
                    "tokens_total": self.verifier_tokens_in + self.verifier_tokens_out,
                    "cost": self.verifier_cost,
                    "responses": self.verifier_responses,
                    "empty": self.verifier_empty,
                    "empty_rate": (self.verifier_empty / self.verifier_responses) if self.verifier_responses else 0.0,
                },
                "distinct_dev_items": {
                    "count": len(self.distinct_dev_items),
                    "ids": sorted_ids(self.distinct_dev_items),
                },
                "validation_reuse": {
                    "distinct_items": len(self.validation_reuse),
                    "max_reuse": max(reuse_counts) if reuse_counts else 0,
                    "total_evaluations": sum(reuse_counts),
                    "per_item": {
                        _id_str(vid): self.validation_reuse[vid] for vid in sorted_ids(self.validation_reuse.keys())
                    },
                },
                "sequential_critical_path_depth": {
                    "best_candidate": (depth.get(best, 0) if best is not None else 0),
                    "max_over_all": max(depth.values()) if depth else 0,
                    "depth_by_candidate": {str(idx): depth[idx] for idx in sorted(depth.keys())},
                },
                "peak_parallelism": {
                    "max_candidates_per_dispatch": self.peak_candidates_per_dispatch,
                    "max_rollouts_per_dispatch": self.peak_rollouts_per_dispatch,
                    "max_reflection_prompts_per_call": self.peak_reflection_prompts_per_call,
                },
                "merges": {
                    "attempted": self.merge_attempted,
                    "accepted": self.merge_accepted,
                    "rejected": self.merge_rejected,
                },
            },
            "selection": {
                "best_candidate_idx": best,
                "total_candidates": n,
                "total_iterations": self.total_iterations,
                "total_metric_calls": self.total_metric_calls,
                "final_pool": final_pool,
                "p0_in_pool": 0 in final_pool,
            },
            "minibatch_schedule": minibatch_schedule,
            "parent_log": list(self.parent_log),
        }

    def timing_dict(self) -> dict[str, Any]:
        """Measured wall-clock. NOT covered by the byte-identity guarantee."""
        return {
            "schema_version": SCHEMA_VERSION,
            "run_label": self.run_label,
            "seed": self.seed,
            "note": (
                "Wall-clock is a measured latency outcome (proposal §6), reported "
                "but never matched. Excluded from the byte-identical ledger.json."
            ),
            "wall_clock_total_s": self.wall_clock_total_s,
            "reflection_wall_s": self.reflection_wall_s,
            "verifier_wall_s": self.verifier_wall_s,
            "concurrency_levels": list(self.concurrency_levels),
        }

    def to_json(self) -> str:
        """Deterministic JSON string of the canonical ledger."""
        return json.dumps(
            self.canonical_dict(),
            sort_keys=True,
            ensure_ascii=True,
            indent=2,
        )

    def write(self, out_dir: str, *, basename: str = "ledger") -> tuple[str, str]:
        """Write ``<basename>.json`` (canonical) and ``<basename>.timing.json``.

        Returns the two paths. Only ``<basename>.json`` is byte-reproducible.
        """
        os.makedirs(out_dir, exist_ok=True)
        ledger_path = os.path.join(out_dir, f"{basename}.json")
        timing_path = os.path.join(out_dir, f"{basename}.timing.json")

        # Newline-terminated so the file is a clean POSIX text file; the trailing
        # newline is fixed, so byte-identity is preserved.
        with open(ledger_path, "w") as f:
            f.write(self.to_json())
            f.write("\n")
        with open(timing_path, "w") as f:
            f.write(json.dumps(self.timing_dict(), sort_keys=True, ensure_ascii=True, indent=2))
            f.write("\n")
        return ledger_path, timing_path
