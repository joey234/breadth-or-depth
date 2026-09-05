# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Phase 3 — the matching-verification gate.

The paper's central claim is that S2 and S3 differ on *exactly one axis*
(statefulness). That is an assertion about code, true only if tested on every
run. This module checks the S2 and S3 per-run ledgers and returns a verdict; a
run that fails is VOID, not adjusted.

Two tiers:

- **Hard invariants** (always gate): the proposal-side pairing and each arm's
  defining property. These must hold for the run to count.
    reflector calls == · mutation attempts == · minibatch schedule == (ordered,
    item-level) · distinct dev items == · every S2 parent is p0 · S3 merges == 0
    · p0 retained in each final selection pool.

- **Validation-side** (gate only with ``strict_validation=True``): candidate-
  example evaluations and task rollouts equal across arms. Under stock GEPA these
  differ because validation is acceptance-gated (S3 accepts more, so it validates
  more). They become matchable only once S3 carries the Phase-2 *unconditional
  validation* patch (every proposed candidate scored on the same dev panel).
  Until then they are reported as diagnostics, not gated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class VerifyResult:
    matched: bool
    failures: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.matched


def _c(ledger: dict) -> dict:
    return ledger["counters"]


def verify_matching(s2: dict, s3: dict, *, strict_validation: bool = False) -> VerifyResult:
    """Compare S2 (root-reset) and S3 (stateful) canonical ledgers.

    Args:
        s2, s3: ``RunLedger.canonical_dict()`` outputs for the paired arms.
        strict_validation: also require validation-side counters to match
            (needs the Phase-2 unconditional-validation patch on S3).
    """
    failures: list[str] = []
    c2, c3 = _c(s2), _c(s3)

    # --- hard proposal-side invariants ---
    if c2["reflector"]["calls"] != c3["reflector"]["calls"]:
        failures.append(f"reflector calls differ: S2={c2['reflector']['calls']} S3={c3['reflector']['calls']}")

    if c2["candidate_proposals"]["reflective"] != c3["candidate_proposals"]["reflective"]:
        failures.append(
            "mutation attempts differ: "
            f"S2={c2['candidate_proposals']['reflective']} S3={c3['candidate_proposals']['reflective']}"
        )

    sched2 = [(r["iteration"], r["minibatch_ids"]) for r in s2["minibatch_schedule"]]
    sched3 = [(r["iteration"], r["minibatch_ids"]) for r in s3["minibatch_schedule"]]
    if sched2 != sched3:
        failures.append("minibatch schedule differs (item-level, ordered)")

    if c2["distinct_dev_items"]["ids"] != c3["distinct_dev_items"]["ids"]:
        failures.append("distinct dev items differ")

    # --- defining properties ---
    if any(p != 0 for p in s2.get("parent_log", [])):
        failures.append(f"S2 is not root-reset: parent_log has non-p0 parents {s2.get('parent_log')}")

    if c3["merges"]["attempted"] != 0:
        failures.append(f"S3 is not mutation-only: merges attempted = {c3['merges']['attempted']}")

    if not s2["selection"]["p0_in_pool"]:
        failures.append("S2 final selection pool does not retain p0")
    if not s3["selection"]["p0_in_pool"]:
        failures.append("S3 final selection pool does not retain p0")

    # --- validation-side (diagnostic unless strict) ---
    ce2 = c2["candidate_example_evaluations"]["total_requested"]
    ce3 = c3["candidate_example_evaluations"]["total_requested"]
    tr2 = c2["task_rollouts"]["total_executed"]
    tr3 = c3["task_rollouts"]["total_executed"]
    validation_paired = ce2 == ce3 and tr2 == tr3
    diagnostics = {
        "candidate_evals": {"S2": ce2, "S3": ce3, "equal": ce2 == ce3},
        "task_rollouts": {"S2": tr2, "S3": tr3, "equal": tr2 == tr3},
        "validation_paired": validation_paired,
        "critical_path_depth": {
            "S2": c2["sequential_critical_path_depth"]["max_over_all"],
            "S3": c3["sequential_critical_path_depth"]["max_over_all"],
        },
    }
    if strict_validation and not validation_paired:
        failures.append(
            f"validation-side not paired: candidate_evals S2={ce2}/S3={ce3}, rollouts S2={tr2}/S3={tr3} "
            "(needs the Phase-2 unconditional-validation patch)"
        )

    return VerifyResult(matched=not failures, failures=failures, diagnostics=diagnostics)
