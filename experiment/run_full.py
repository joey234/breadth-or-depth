# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Resumable full-run orchestrator for the two-tier Breadth-or-Depth study.

Runs every (tier x task) cell through the per-seed-resumable runners
(``run_study`` for the GEPA near-null tasks, ``run_paired_task`` for the
depth/breadth/controlled benchmarks). Models come from ``slate.py`` (one source of
truth). Task model runs fast (Qwen instruct / gpt minimal); reflector reasons at low.

RESUMABLE by construction:
  * each runner checkpoints after every seed and skips already-finished seeds;
  * this orchestrator skips a whole cell whose seeds are all done, and continues
    past a cell that errors (logging it) instead of aborting the run.
So you can Ctrl-C or lose the machine at any point and just run it again — it picks
up exactly where it stopped. Nothing is recomputed.

Usage (from the repo root, with the venv active):
    python -m experiment.run_full                      # everything, both tiers
    python -m experiment.run_full --tiers mid          # one tier
    python -m experiment.run_full --only paired        # only run_paired_task cells
    python -m experiment.run_full --skip ifbench       # drop a task (e.g. needs extra deps)
    python -m experiment.run_full --seeds 0 1 2 3 4    # more seeds
    python -m experiment.run_full --dry-run            # print the plan, run nothing
    python -m experiment.run_full --status             # progress table, run nothing

Checking progress (three levels, all from disk so they work mid-run):
    python -m experiment.run_full --status             # per-cell table: seeds done / state
    tail -f runs/full/orchestrator.log                 # which cell is running now (timestamped)
    tail -f runs/full/<tier>__<kind>_<task>.log        # live output of one cell's runner
The orchestrator log is written unconditionally (no shell redirect needed). --status
reads only the checkpoints, so it is safe to run in a second terminal while a run is live.

Long runs: wrap it so it survives a closed terminal, and re-run to resume:
    nohup python -m experiment.run_full >/dev/null 2>&1 &   # progress still in orchestrator.log
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import experiment  # noqa: F401 -- auto-loads .env
from experiment.slate import ANCHOR_REFLECT, ANCHOR_TASK, FRONTIER_REFLECT, FRONTIER_TASK, MODELS

ROOT = "runs/full"

# tier -> (task_model_slug, reflect_model_slug). Task runs fast, reflector at low.
TIERS = {
    "mid": (MODELS[ANCHOR_TASK][0], MODELS[ANCHOR_REFLECT][0]),
    "frontier": (MODELS[FRONTIER_TASK][0], MODELS[FRONTIER_REFLECT][0]),
}

# run_study GEPA near-null tasks: 100-example test, per-task dev floor, retrieval for multi-hop.
GEPA = {
    "hotpotqa": {"dev_floor": 0.05, "retrieval": True},
    "hover": {"dev_floor": 0.15, "retrieval": True},
    "ifbench": {"dev_floor": 0.15, "retrieval": False},  # needs nltk/emoji/syllapy + IFBENCH_DIR
    "pupa": {"dev_floor": 0.05, "retrieval": False},
}
# run_paired_task benchmarks (depth / breadth / controlled).
PAIRED = ["top", "massive", "mbpp", "funccall", "structured"]


def _study_done(outdir: str, seeds: list[int]) -> bool:
    p = os.path.join(outdir, "study_results.json")
    if not os.path.exists(p):
        return False
    try:
        d = json.load(open(p))
    except Exception:
        return False
    clean = {s["seed"] for s in d.get("per_seed", []) if s.get("matched") and s.get("quality_ok", True)}
    return set(seeds).issubset(clean)


def _paired_done(outdir: str, seeds: list[int]) -> bool:
    p = os.path.join(outdir, "results.json")
    if not os.path.exists(p):
        return False
    try:
        d = json.load(open(p))
    except Exception:
        return False
    return all(d.get(f"s{s}", {}).get("done") for s in seeds)


ORCH_LOG = f"{ROOT}/orchestrator.log"


def _say(msg: str) -> None:
    """Print to stdout AND append a timestamped line to the orchestrator log, so
    progress is always recorded on disk even without a shell redirect. Tail it with:
        tail -f runs/full/orchestrator.log"""
    print(msg, flush=True)
    try:
        os.makedirs(ROOT, exist_ok=True)
        with open(ORCH_LOG, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}\n")
    except Exception:
        pass


def _cell_progress(kind: str, outdir: str, seeds: list[int]) -> tuple[int, int]:
    """(#clean/#done seeds, #total) for a cell — the same completeness test the
    resume logic uses, but counted per seed so status shows partial progress."""
    total = len(seeds)
    if kind == "study":
        p = os.path.join(outdir, "study_results.json")
        if not os.path.exists(p):
            return 0, total
        try:
            d = json.load(open(p))
        except Exception:
            return 0, total
        clean = {s["seed"] for s in d.get("per_seed", []) if s.get("matched") and s.get("quality_ok", True)}
        return len(set(seeds) & clean), total
    p = os.path.join(outdir, "results.json")
    if not os.path.exists(p):
        return 0, total
    try:
        d = json.load(open(p))
    except Exception:
        return 0, total
    return sum(1 for s in seeds if d.get(f"s{s}", {}).get("done")), total


def _build_plan(args) -> list[tuple[str, str, str, list[str], str]]:
    """(label, kind, outdir, cmd, logpath) for every requested cell."""
    seeds_arg = [str(s) for s in args.seeds]
    plan: list[tuple[str, str, str, list[str], str]] = []
    for tier in args.tiers:
        tm, rm = TIERS[tier]
        if args.only in (None, "study"):
            for ds, opt in GEPA.items():
                if ds in args.skip:
                    continue
                out = f"{ROOT}/{tier}__study_{ds}"
                cmd = [sys.executable, "-m", "experiment.run_study", "--dataset", ds, "--seeds", *seeds_arg,
                       "--n-train", "20", "--n-dev", "20", "--n-test", str(args.n_test),
                       "--task-model", tm, "--reflect-model", rm, "--reflect-reasoning-effort", "low",
                       "--dev-base-floor", str(opt["dev_floor"]), "--strict-validation", "--out", out]
                if opt["retrieval"]:
                    cmd.append("--retrieval")
                plan.append((f"{tier}/study/{ds}", "study", out, cmd, f"{out}.log"))
        if args.only in (None, "paired"):
            variants = []
            if args.faithful in ("off", "both"):
                variants.append(False)
            if args.faithful in ("on", "both"):
                variants.append(True)
            for tk in PAIRED:
                if tk in args.skip:
                    continue
                for faithful in variants:
                    sfx = "__faithful" if faithful else ""
                    out = f"{ROOT}/{tier}__paired_{tk}{sfx}"
                    cmd = [sys.executable, "-m", "experiment.run_paired_task", "--task", tk, "--seeds", *seeds_arg,
                           "--task-model", tm, "--reflect-model", rm, "--reflect-reasoning-effort", "low",
                           "--n-test", str(args.paired_n_test), "--out", out]
                    if faithful:
                        cmd.append("--faithful")
                    label = f"{tier}/paired/{tk}" + (" [faithful]" if faithful else "")
                    plan.append((label, "paired", out, cmd, f"{out}.log"))
    return plan


def _print_status(plan, seeds: list[int]) -> int:
    """Print a per-cell progress table and exit. Reads the on-disk checkpoints only —
    safe to run in another terminal while the full run is in flight."""
    done_cells = 0
    print(f"\nStatus @ {time.strftime('%Y-%m-%d %H:%M:%S')}  (seeds={seeds})")
    print(f"{'cell':28} {'seeds':>7}  state     log")
    print("-" * 72)
    for label, kind, out, _cmd, logpath in plan:
        d, t = _cell_progress(kind, out, seeds)
        state = "DONE " if d >= t else ("partial" if d else "pending")
        if d >= t:
            done_cells += 1
        tail = logpath if os.path.exists(logpath) else "-"
        print(f"{label:28} {f'{d}/{t}':>7}  {state:8}  {tail}")
    print("-" * 72)
    print(f"{done_cells}/{len(plan)} cells complete. Live per-cell output: tail -f <log>. "
          f"Orchestrator: tail -f {ORCH_LOG}")
    return 0


def _run(cmd: list[str], logpath: str) -> int:
    print("  >>", " ".join(cmd), flush=True)
    with open(logpath, "a") as f:
        f.write(f"\n\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} :: {' '.join(cmd)} =====\n")
        f.flush()
        return subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)


def main() -> int:
    global ROOT, ORCH_LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", nargs="*", default=list(TIERS), choices=list(TIERS))
    ap.add_argument("--only", choices=["study", "paired"], default=None, help="run only one runner's cells")
    ap.add_argument("--skip", nargs="*", default=[], help="task names to skip (e.g. ifbench)")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--n-test", type=int, default=100, help="GEPA held-out test size (run_study)")
    ap.add_argument("--paired-n-test", type=int, default=100,
                    help="held-out eval size for paired cells (was 12; matches study n-test)")
    ap.add_argument("--root", default=ROOT,
                    help="output root; use a fresh dir to avoid clobbering a prior run")
    ap.add_argument("--faithful", choices=["off", "on", "both"], default="off",
                    help="paired acceptance gating: off=AcceptAll only (default), "
                         "on=StrictImprovement only, both=run each paired cell under both")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true",
                    help="print a per-cell progress table (from on-disk checkpoints) and exit; "
                         "safe to run in another terminal while the full run is in flight")
    args = ap.parse_args()

    ROOT = args.root
    ORCH_LOG = f"{ROOT}/orchestrator.log"
    os.makedirs(ROOT, exist_ok=True)
    plan = _build_plan(args)

    if args.status:
        return _print_status(plan, args.seeds)

    _say(f"==== full run start :: {len(plan)} cells, seeds={args.seeds}, tiers={args.tiers} ====")
    failed: list[str] = []
    for i, (label, kind, out, cmd, logpath) in enumerate(plan, 1):
        d, t = _cell_progress(kind, out, args.seeds)
        if d >= t:
            _say(f"[skip {i}/{len(plan)}] {label} (all {t} seeds done)")
            continue
        if args.dry_run:
            _say(f"[plan {i}/{len(plan)}] {label} ({d}/{t} done)\n  {' '.join(cmd)}")
            continue
        _say(f"[run  {i}/{len(plan)}] {label} ({d}/{t} seeds done -> running the rest; log: {logpath})")
        rc = _run(cmd, logpath)
        d2, t2 = _cell_progress(kind, out, args.seeds)
        ok = d2 >= t2
        if rc != 0 or not ok:
            _say(f"[FAIL {i}/{len(plan)}] {label} (rc={rc}, {d2}/{t2} seeds) -> see {logpath}; re-run to resume")
            failed.append(label)
        else:
            _say(f"[done {i}/{len(plan)}] {label} ({d2}/{t2} seeds)")

    if failed:
        _say(f"==== full run pass complete :: {len(failed)} cell(s) incomplete (re-run to resume): {failed} ====")
        return 1
    _say("==== full run pass complete :: all cells complete ====")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
