#!/usr/bin/env python3
"""Reprint paper tables from saved run JSONs in results/.

Uses the same paired-t interval as the paper (Bessel sd, t_{n-1} critical value).
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# t_{n-1, 0.975} used in the paper
TCRIT = {
    2: 12.706,
    3: 4.303,
    4: 3.182,
    5: 2.776,
    6: 2.571,
    7: 2.447,
    8: 2.365,
    9: 2.306,
    10: 2.262,
}


def mean_ci(vals: list[float]) -> tuple[float, float, float, int]:
    xs = [float(x) for x in vals]
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    m = statistics.mean(xs)
    if n == 1 or statistics.pstdev(xs) == 0 and statistics.stdev(xs) == 0:
        return m, m, m, n
    sd = statistics.stdev(xs)
    t = TCRIT.get(n, 1.96)
    h = t * sd / math.sqrt(n)
    return m, m - h, m + h, n


def fmt(m: float, lo: float, hi: float) -> str:
    return f"{m:+.3f} [{lo:+.3f},{hi:+.3f}]"


def gepa_ats(path: Path) -> list[float]:
    d = json.loads(path.read_text())
    return [
        s["test"]["A_t"]
        for s in d.get("per_seed", [])
        if s.get("matched") and s.get("quality_ok", True) and s.get("test")
    ]


def paired_ats(path: Path) -> list[float]:
    d = json.loads(path.read_text())
    return [
        v["A_t"]
        for v in d.values()
        if isinstance(v, dict) and v.get("done") and not v.get("contaminated")
    ]


def chain_groups(path: Path, mode: str) -> dict[int, list[dict]]:
    d = json.loads(path.read_text())
    groups: dict[int, list[dict]] = {}
    for row in d.values():
        if not (isinstance(row, dict) and row.get("done")):
            continue
        if row.get("information_mode") != mode:
            continue
        groups.setdefault(int(row["D"]), []).append(row["final"])
    return groups


def print_gepa(acceptance: str, label: str) -> None:
    print(f"\n=== {label} ===")
    print(f"{'task':10} {'mid':28} {'frontier':28}")
    mids, frs = [], []
    for task in ("hotpotqa", "hover", "ifbench", "pupa"):
        row = []
        for tier in ("mid", "frontier"):
            vals = gepa_ats(RESULTS / "gepa_suite" / f"{tier}_{task}_{acceptance}" / "study_results.json")
            m, lo, hi, n = mean_ci(vals)
            row.append((m, fmt(m, lo, hi), n))
        mids.append(row[0][0])
        frs.append(row[1][0])
        print(f"{task:10} {row[0][1]:28} {row[1][1]:28}  n={row[0][2]}/{row[1][2]}")
    print(f"{'mean':10} {statistics.mean(mids):+.3f}{'':22} {statistics.mean(frs):+.3f}")


def print_paired() -> None:
    print("\n=== tab:boundary-details (paired tasks) ===")
    print(f"{'task':12} {'tier':8} {'accept':8} {'A_c':28}")
    for task in ("mbpp", "funccall", "structured", "top", "massive"):
        for tier in ("mid", "frontier"):
            vals = paired_ats(RESULTS / "paired" / f"{tier}_{task}_acceptall" / "results.json")
            print(f"{task:12} {tier:8} {'all':8} {fmt(*mean_ci(vals)[:3]):28}  n={len(vals)}")
        if task in ("massive", "structured"):
            for tier in ("mid", "frontier"):
                vals = paired_ats(RESULTS / "paired" / f"{tier}_{task}_strict" / "results.json")
                print(f"{task:12} {tier:8} {'strict':8} {fmt(*mean_ci(vals)[:3]):28}  n={len(vals)}")


def print_chain() -> None:
    for mode, title in (
        ("explicit-overall", "tab:info-matched-chain"),
        ("full-specification", "tab:full-spec-chain"),
    ):
        print(f"\n=== {title} ({mode}) ===")
        for tier, fname in (("mid", "mid_results.json"), ("frontier", "frontier_results.json")):
            print(f"-- {tier} --")
            groups = chain_groups(RESULTS / "chain" / fname, mode)
            for d in sorted(groups):
                rows = groups[d]
                local = [r["test"]["root_local"] for r in rows]
                hist = [r["test"]["root_history"] for r in rows]
                lin = [r["test"]["lineage_history"] for r in rows]
                fb = [r["feedback_history_effect"] for r in rows]
                st = [r["prompt_state_effect"] for r in rows]
                print(
                    f"  D={d}  local/hist/lin="
                    f"{statistics.mean(local):.3f}/{statistics.mean(hist):.3f}/{statistics.mean(lin):.3f}"
                    f"  A_fb={fmt(*mean_ci(fb)[:3])}  A_state={fmt(*mean_ci(st)[:3])}  n={len(rows)}"
                )


def main() -> None:
    print_gepa("acceptall", "tab:gepa (accept-all)")
    print_gepa("strict", "tab:gepa-replication-ci (strict improvement)")
    print_paired()
    print_chain()
    print("\nFigure 1 points are locked in results/scatter_data.json (tab:probe-full).")
    print("Regenerate the figure with: python experiment/make_scatter.py")


if __name__ == "__main__":
    main()
