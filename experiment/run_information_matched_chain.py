#!/usr/bin/env python3
"""Resumable runner for the information-matched chain experiment.

    python -m experiment.run_information_matched_chain --smoke
    python -m experiment.run_information_matched_chain --ds 1 2 3 4 \
        --task-information explicit-overall full-specification --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import math
import os
import traceback

from experiment.compounding import build_information_matched_chain
from experiment.config import RunConfig
from experiment.information_matched_chain import (
    MatchingError,
    run_information_matched_replicate,
)
from experiment.run_study import _load_env
from experiment.slate import ANCHOR_REFLECT, ANCHOR_TASK, MODELS


def _cfg(task_model: str, reflect_model: str,
         task_effort: str | None = None, reflect_effort: str | None = "low") -> RunConfig:
    c = RunConfig()
    c.task_model = task_model
    c.reflect_model = reflect_model
    c.task_reasoning = False
    c.reflect_reasoning = False
    c.task_reasoning_effort = task_effort
    c.reflect_reasoning_effort = reflect_effort
    c.task_max_tokens = 50000
    c.reflect_max_tokens = 16000
    c.minibatch_size = 3
    return c


def _cell_key(d: int, mode: str, schedule: str, seed: int) -> str:
    return f"D{d}_{mode}_{schedule}_s{seed}"


def _mean_ci(xs: list[float]) -> tuple[float, float, float]:
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    m = sum(xs) / n
    if n < 2:
        return m, m, m
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    se = math.sqrt(var / n)
    try:
        from scipy.stats import t as student_t
        h = float(student_t.ppf(0.975, n - 1)) * se
    except Exception:
        h = 1.96 * se
    return m, m - h, m + h


def _summarize(results: dict) -> None:
    groups: dict[tuple, list[dict]] = {}
    for row in results.values():
        if not row.get("done"):
            continue
        key = (row["D"], row["information_mode"], row["schedule"])
        groups.setdefault(key, []).append(row)
    if not groups:
        return
    print("\n=== information-matched summary ===", flush=True)
    for (d, mode, schedule), rows in sorted(groups.items()):
        n = len(rows)
        rl = [r["final"]["test"]["root_local"] for r in rows]
        rh = [r["final"]["test"]["root_history"] for r in rows]
        lin = [r["final"]["test"]["lineage_history"] for r in rows]
        fb = [r["final"]["feedback_history_effect"] for r in rows]
        ps = [r["final"]["prompt_state_effect"] for r in rows]
        tot = [r["final"]["total_matched_effect"] for r in rows]
        def fmt(xs: list[float]) -> str:
            m, lo, hi = _mean_ci(xs)
            return f"{m:+.3f} [{lo:+.3f},{hi:+.3f}]"
        print(
            f"D={d} {mode} {schedule} n={n}  "
            f"local={sum(rl)/n:.3f} hist={sum(rh)/n:.3f} lin={sum(lin)/n:.3f}  "
            f"fb={fmt(fb)}  state={fmt(ps)}  total={fmt(tot)}",
            flush=True,
        )
        if d == 1:
            m, lo, hi = _mean_ci(fb)
            if abs(m) > 0.08:
                print(
                    f"  WARN D=1 feedback_history_effect mean {m:+.3f} is not near zero",
                    flush=True,
                )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", type=int, nargs="*", default=[1, 2, 3, 4])
    ap.add_argument("--arms", nargs="*", default=["root_local", "root_history", "lineage_history"])
    ap.add_argument("--schedule", default="oracle", choices=["oracle", "natural-unlock"])
    ap.add_argument(
        "--task-information",
        nargs="*",
        default=["explicit-overall"],
        choices=["explicit-overall", "full-specification"],
    )
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--n-train", type=int, default=20)
    ap.add_argument("--n-sel", type=int, default=20)
    ap.add_argument("--n-val", type=int, default=20)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--task-model", default=MODELS[ANCHOR_TASK][0])
    ap.add_argument("--reflect-model", default=MODELS[ANCHOR_REFLECT][0])
    ap.add_argument("--task-reasoning-effort", default=None)
    ap.add_argument("--reflect-reasoning-effort", default="low")
    ap.add_argument("--out", default="runs/information_matched_chain")
    ap.add_argument("--env", default=".env")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="D=1,2 explicit-overall oracle seeds 0-2 with smaller splits",
    )
    args = ap.parse_args()
    if args.smoke:
        args.ds = [1, 2]
        args.task_information = ["explicit-overall"]
        args.schedule = "oracle"
        args.seeds = [0, 1, 2]
        args.n_train, args.n_sel, args.n_val, args.n_test = 12, 12, 12, 20
        args.k = 8

    _load_env(args.env)
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "diagnostics"), exist_ok=True)
    path = os.path.join(args.out, "results.json")
    results = json.load(open(path)) if os.path.exists(path) else {}

    from gepa.adapters.default_adapter.default_adapter import DefaultAdapter
    from gepa.lm import LM

    cfg = _cfg(
        args.task_model, args.reflect_model,
        args.task_reasoning_effort, args.reflect_reasoning_effort,
    )
    workers = 2 if cfg.task_model.startswith("openai/") else 8
    print(
        f"information-matched chain: D={args.ds} modes={args.task_information} "
        f"schedule={args.schedule} seeds={args.seeds} K={args.k} "
        f"task={args.task_model} reflect={args.reflect_model} "
        f"task_effort={args.task_reasoning_effort} reflect_effort={args.reflect_reasoning_effort}",
        flush=True,
    )

    failed = False
    for mode in args.task_information:
        for d in args.ds:
            for seed in args.seeds:
                key = _cell_key(d, mode, args.schedule, seed)
                if key in results and results[key].get("done"):
                    print(f"{key}: [skip]", flush=True)
                    continue
                print(f"{key}: starting", flush=True)
                train, sel, val, test, ev, task_description = build_information_matched_chain(
                    args.n_train, args.n_sel, args.n_val, args.n_test,
                    d=d, seed=seed, information_mode=mode,
                )
                adapter = DefaultAdapter(
                    model=cfg.task_model,
                    evaluator=ev,
                    max_litellm_workers=workers,
                    litellm_batch_completion_kwargs=cfg.task_batch_kwargs(),
                )
                refl_kw = cfg.reflect_kwargs()
                reflector = LM(cfg.reflect_model, **refl_kw)
                try:
                    row = run_information_matched_replicate(
                        d=d,
                        seed=seed,
                        adapter=adapter,
                        reflector=reflector,
                        train=train,
                        sel=sel,
                        val=val,
                        test=test,
                        task_description=task_description,
                        information_mode=mode,
                        schedule=args.schedule,
                        k=args.k,
                        minibatch_size=cfg.minibatch_size,
                        arms=tuple(args.arms),
                    )
                except MatchingError as exc:
                    failed = True
                    err_row = {
                        "D": d, "seed": seed, "information_mode": mode,
                        "schedule": args.schedule, "done": False,
                        "failed_check": str(exc),
                    }
                    results[key] = err_row
                    json.dump(results, open(path, "w"), indent=2)
                    print(f"{key}: FAILED CHECK {exc}", flush=True)
                    if args.smoke:
                        raise
                    continue
                except Exception:
                    failed = True
                    results[key] = {
                        "D": d, "seed": seed, "information_mode": mode,
                        "schedule": args.schedule, "done": False,
                        "failed_check": "uncaught",
                        "traceback": traceback.format_exc(),
                    }
                    json.dump(results, open(path, "w"), indent=2)
                    print(f"{key}: UNCAUGHT\n{traceback.format_exc()}", flush=True)
                    if args.smoke:
                        raise
                    continue

                diag = row.pop("diagnostics")
                diag_path = os.path.join(args.out, "diagnostics", f"{key}.json")
                json.dump(diag, open(diag_path, "w"), indent=2)
                row["D"] = d
                row["seed"] = seed
                row["information_mode"] = mode
                row["schedule"] = args.schedule
                row["done"] = True
                results[key] = row
                json.dump(results, open(path, "w"), indent=2)
                fin = row["final"]
                print(
                    f"{key}: local={fin['test']['root_local']:.3f} "
                    f"hist={fin['test']['root_history']:.3f} "
                    f"lin={fin['test']['lineage_history']:.3f} "
                    f"fb={fin['feedback_history_effect']:+.3f} "
                    f"state={fin['prompt_state_effect']:+.3f} "
                    f"total={fin['total_matched_effect']:+.3f}",
                    flush=True,
                )

    _summarize(results)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
