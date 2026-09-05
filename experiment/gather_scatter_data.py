#!/usr/bin/env python3
"""Eight structured/benchmark scatter points, locked to paper Table tab:probe-full.

D-chain cells are not included. Thresholds are illustrative; this is not a
calibration/held-out split.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FLOOR = 0.05
DG_FLOOR = 0.15


def truth(lo, hi):
    if lo > FLOOR:
        return "depth"
    if hi < FLOOR:
        return "breadth"
    return "inconclusive"


# (label, g, d, y, lo, hi) from tab:probe-full
MID = [
    ("structured-subtle", 0.250, 1.000, 0.250, 0.250, 0.250),
    ("TOP-v2", 0.213, 0.321, 0.217, 0.163, 0.271),
    ("HotpotQA", 0.000, 0.000, -0.002, -0.019, 0.015),
    ("HoVer", 0.050, 0.250, 0.011, -0.020, 0.042),
    ("IFBench", 0.000, 0.000, 0.015, -0.025, 0.054),
    ("PUPA", 0.118, 0.190, 0.047, -0.023, 0.117),
    ("MBPP", 0.000, 0.000, 0.017, 0.008, 0.025),
    ("function calling", 0.000, 0.000, -0.001, -0.009, 0.006),
]
FRONTIER = [
    ("structured-subtle", 0.250, 1.000, 0.250, 0.250, 0.250),
    ("TOP-v2", 0.162, 0.342, 0.163, 0.080, 0.246),
    ("HotpotQA", 0.002, 0.024, 0.008, 0.000, 0.016),
    ("HoVer", 0.000, 0.000, 0.030, -0.014, 0.074),
    ("IFBench", 0.000, 0.000, -0.019, -0.053, 0.016),
    ("PUPA", 0.004, 0.013, 0.020, 0.002, 0.037),
    ("MBPP", 0.000, 0.000, -0.002, -0.047, 0.043),
    ("function calling", 0.030, 0.268, 0.028, -0.038, 0.094),
]


def pack(rows):
    pts = []
    for label, g, d, y, lo, hi in rows:
        t = truth(lo, hi)
        pred = "depth" if (g >= FLOOR and d >= DG_FLOOR) else "breadth"
        pts.append(dict(label=label, g=g, d=d, y=y, lo=lo, hi=hi, truth=t, pred=pred))
    return pts


results = {"mid": pack(MID), "frontier": pack(FRONTIER)}
for tier, pts in results.items():
    print(f"=== {tier} ===")
    conclusive = [p for p in pts if p["truth"] != "inconclusive"]
    correct = sum(1 for p in conclusive if p["pred"] == p["truth"])
    for p in pts:
        mark = "—" if p["truth"] == "inconclusive" else ("OK" if p["pred"] == p["truth"] else "WRONG")
        print(f"  {p['label']:16} g={p['g']:+.3f} d={p['d']:.3f} y={p['y']:+.3f} "
              f"[{p['lo']:+.3f},{p['hi']:+.3f}] truth={p['truth']:12} pred={p['pred']:8} {mark}")
    print(f"  conclusive correct: {correct}/{len(conclusive)}  (n={len(pts)})")

out = ROOT / "results" / "scatter_data.json"
json.dump(results, open(out, "w"), indent=2)
print(f"\nwrote {out}")
