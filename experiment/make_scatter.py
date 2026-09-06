#!/usr/bin/env python3
"""Matched-probe scatter for the ten locked configurations.

No D-chain points and no calibration/held-out split. Each point is one K=8
probe draw. A_c is the n=10 paired full-optimizer mean with 95% CI, except
word D=4 (n=10 information-matched total effect, not paired GEPA).
"""
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GREEN, ORANGE, GRAY = "#009E73", "#E69F00", "#888888"
FLOOR = 0.05
DG_FLOOR = 0.15
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results"

data = json.load(open(ROOT / "results" / "scatter_data.json"))


def pearson(xs, ys):
    n = len(xs)
    mx, my = statistics.mean(xs), statistics.mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math_sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math_sqrt(sum((y - my) ** 2 for y in ys))
    return cov / (sx * sy) if sx > 0 and sy > 0 else float("nan")


def math_sqrt(v):
    return v ** 0.5


TIER_LABEL = {"mid": "deepseek-v4-flash-0731", "frontier": "gpt-5.6-luna"}
fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.2))
plt.rcParams.update({"font.size": 10.5})

summary_lines = []
OFFSETS = {
    ("mid", True, "HotpotQA"): (6, -13),
    ("mid", True, "function calling"): (-72, -18),
    ("mid", True, "IFBench"): (-46, 4),
    ("mid", True, "MBPP"): (6, 8),
    ("mid", True, "HoVer"): (6, 6),
    ("mid", True, "PUPA"): (6, -12),
    ("mid", False, "TOP-v2"): (8, 6),
    ("mid", False, "structured-subtle"): (8, 6),
    ("mid", False, "MASSIVE"): (-58, -14),
    ("mid", False, "word D=4"): (8, 6),
    ("frontier", True, "HoVer"): (-42, 10),
    ("frontier", True, "function calling"): (10, -3),
    ("frontier", True, "PUPA"): (6, 6),
    ("frontier", True, "HotpotQA"): (-62, 9),
    ("frontier", True, "IFBench"): (6, -13),
    ("frontier", True, "MBPP"): (10, -22),
    ("frontier", False, "TOP-v2"): (-52, 8),
    ("frontier", False, "structured-subtle"): (8, 6),
    ("frontier", False, "MASSIVE"): (8, -14),
    ("frontier", False, "word D=4"): (8, 6),
}

for col, tier in enumerate(["mid", "frontier"]):
    pts = data[tier]
    conclusive = [p for p in pts if p["truth"] != "inconclusive"]
    correct_n = sum(1 for p in conclusive
                     if ("depth" if (p["g"] >= FLOOR and p["d"] >= DG_FLOOR) else "breadth") == p["truth"])
    xs_all = [p["g"] for p in pts]
    ys_all = [p["y"] for p in pts]
    r = pearson(xs_all, ys_all)
    summary_lines.append(
        f"{tier}: {correct_n}/{len(conclusive)} conclusive correct, "
        f"Pearson r={r:.2f}, n_points={len(pts)}"
    )

    for row, (xlim, ylim, zoom) in enumerate([((-0.09, 0.32), (-0.12, 0.82), False),
                                               ((-0.03, 0.10), (-0.06, 0.06), True)]):
        ax = axes[row][col]
        ax.axhline(0, color="#eeeeee", lw=1, zorder=0)
        ax.axvspan(FLOOR, xlim[1], color=GREEN, alpha=0.05, zorder=0)
        ax.axvline(FLOOR, color=GRAY, ls=":", lw=1.1, zorder=1)
        ax.axhline(FLOOR, color=GRAY, ls=":", lw=1.1, zorder=1)
        for p in pts:
            x, y = p["g"], p["y"]
            if not (xlim[0] <= x <= xlim[1] and ylim[0] <= y <= ylim[1]):
                continue
            pred = "depth" if (x >= FLOOR and p["d"] >= DG_FLOOR) else "breadth"
            filled = pred == "depth"
            if p["truth"] == "inconclusive":
                edge, ls = GRAY, "--"
            elif pred == p["truth"]:
                edge, ls = GREEN, "-"
            else:
                edge, ls = ORANGE, "-"
            face = edge if filled else "white"
            ax.errorbar(x, y, yerr=[[y - p["lo"]], [p["hi"] - y]], fmt="none",
                         ecolor=edge if p["truth"] != "inconclusive" else GRAY, elinewidth=0.8,
                         capsize=2, zorder=2, alpha=0.8)
            ax.scatter(x, y, s=140, facecolor=face, edgecolor=edge, lw=1.6, linestyle=ls, zorder=3)
            if not zoom and abs(x) < 0.08 and abs(y) < 0.08:
                continue
            off = OFFSETS.get((tier, zoom, p["label"]), (6, 7 if y >= 0 else -12))
            ax.annotate(p["label"], (x, y), textcoords="offset points", xytext=off,
                        fontsize=7.3, color="#333333")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        if row == 0:
            ax.set_title(TIER_LABEL[tier], fontsize=11.5, fontweight="bold")
        if row == 1:
            ax.set_xlabel("matched probe raw gap  g  (K=8)")
        if col == 0:
            ax.set_ylabel("measured $A_c$" + (" (zoom)" if zoom else ""))

from matplotlib.lines import Line2D
legend_elems = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=GREEN, markeredgecolor='k', markersize=9, label='correct'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=ORANGE, markeredgecolor='k', markersize=9, label='incorrect'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markeredgecolor=GRAY, markersize=9,
           linestyle='None', label='inconclusive'),
]
fig.legend(handles=legend_elems, loc="lower center", ncol=3, fontsize=9.5, bbox_to_anchor=(0.5, -0.01))
fig.suptitle("Matched-probe gap vs. measured lineage advantage, ten configurations",
             fontsize=12.5, y=0.995)
fig.tight_layout(rect=(0, 0.035, 1, 0.97))
fig.savefig(OUT_DIR / "bod_scatter.pdf", bbox_inches="tight")

print("\n".join(summary_lines))
print(f"wrote {OUT_DIR / 'bod_scatter.pdf'}")
