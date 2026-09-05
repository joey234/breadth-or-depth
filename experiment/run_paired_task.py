#!/usr/bin/env python3
"""Generic paired S2 (breadth) vs S3 (depth) runner for the real-dataset Tier-3
tasks, so every scatter point has a REAL full-optimizer A_t (not just a probe).
Dispatches on --task {mbpp,funccall,structured}. Resumable per (task,seed);
records the probe (K=8) per cell too. Mirrors run_structured_paired.

    python -m experiment.run_paired_task --task mbpp --seeds 0 1
    python -m experiment.run_paired_task --task funccall --seeds 0 1
"""
from __future__ import annotations

import argparse
import json
import os

from experiment.compounding import depth_gain_probe
from experiment.config import RunConfig
from experiment.run_study import COMPONENT, EMPTY_RATE_MAX, _load_env, make_schedule, run_arm, test_touch


def _build(task: str, n_train: int, n_dev: int, n_test: int, seed: int):
    if task == "mbpp":
        from experiment.mbpp_task import build_mbpp
        return build_mbpp(n_train, n_dev, n_test, seed=seed)
    if task == "funccall":
        from experiment.funccall_task import build_funccall
        return build_funccall(n_train, n_dev, n_test, seed=seed)
    if task == "structured":
        from experiment.structured_task import build_structured
        return build_structured(n_train, n_dev, n_test, seed=seed, mode="subtle")
    if task == "top":
        from experiment.top_task import build_top
        return build_top(n_train, n_dev, n_test, seed=seed, domains=["alarm"])
    if task == "massive":
        from experiment.massive_task import build_massive
        return build_massive(n_train, n_dev, n_test, seed=seed)
    raise ValueError(f"unknown task '{task}'")


def _cfg(task_model: str, reflect_model: str, n_iters: int, max_tokens: int,
         task_effort: str | None = None, reflect_effort: str | None = "low") -> RunConfig:
    c = RunConfig()
    c.task_model = task_model
    c.reflect_model = reflect_model
    c.task_reasoning = False
    c.reflect_reasoning = False
    c.task_reasoning_effort = task_effort
    c.reflect_reasoning_effort = reflect_effort
    c.task_max_tokens = max_tokens
    # reasoning tokens count against the budget, so give the reflector room when a
    # reasoning effort is set (1200 is fine for instruct, too small for reasoning).
    c.reflect_max_tokens = 8000 if reflect_effort else 1200
    c.n_iters = n_iters
    c.minibatch_size = 3
    c.n_train, c.n_dev, c.n_test = 12, 12, 12
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["mbpp", "funccall", "structured", "top", "massive"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1])
    ap.add_argument("--task-model", default="openrouter/qwen/qwen3.6-35b-a3b")
    ap.add_argument("--reflect-model", default=None, help="reflector model (default: same as --task-model)")
    ap.add_argument("--n-iters", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--reasoning-effort", default=None,
                    help="shorthand: set BOTH task and reflector reasoning effort (minimal|low|medium|high)")
    ap.add_argument("--task-reasoning-effort", default=None)
    ap.add_argument("--reflect-reasoning-effort", default=None)
    ap.add_argument("--faithful", action="store_true",
                    help="official-GEPA acceptance gating: use StrictImprovementAcceptance + AllImprovements "
                         "(unconditional_validation=False) instead of the controlled accept-all/select-all")
    ap.add_argument("--n-train", type=int, default=12)
    ap.add_argument("--n-dev", type=int, default=12)
    ap.add_argument("--n-test", type=int, default=100,
                    help="held-out eval size (was 12; 100 matches the study tasks)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--env", default=".env")
    args = ap.parse_args()
    _load_env(args.env)
    out = args.out or f"runs/paired_{args.task}"
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "results.json")
    results = json.load(open(path)) if os.path.exists(path) else {}
    # Generous instruct budget so nothing truncates (instruct stops at the answer's end,
    # so a high cap is free on these short-output tasks). Override with --max-tokens.
    max_tokens = args.max_tokens or 50000

    from gepa.adapters.default_adapter.default_adapter import DefaultAdapter
    from gepa.lm import LM

    for seed in args.seeds:
        key = f"s{seed}"
        if key in results and results[key].get("done"):
            print(f"{key}: [skip]", flush=True)
            continue
        task_effort = args.task_reasoning_effort or args.reasoning_effort
        reflect_effort = args.reflect_reasoning_effort or args.reasoning_effort or "low"
        reflect_model = args.reflect_model or args.task_model
        cfg = _cfg(args.task_model, reflect_model, args.n_iters, max_tokens, task_effort, reflect_effort)
        cfg.unconditional_validation = not args.faithful  # faithful => real GEPA acceptance gating
        cfg.n_train, cfg.n_dev, cfg.n_test = args.n_train, args.n_dev, args.n_test
        train, dev, test, ev = _build(args.task, cfg.n_train, cfg.n_dev, cfg.n_test, seed)
        cfg.evaluator = ev  # type: ignore[attr-defined]
        sched = make_schedule(cfg.n_train, cfg.minibatch_size, cfg.n_iters)
        row: dict = {"task": args.task, "seed": seed, "faithful": args.faithful,
                     "acceptance": "StrictImprovement" if args.faithful else "AcceptAll"}
        # Data-quality gate (mirrors run_study). An API stall / quota outage turns
        # task responses into empty strings, which score at each task's FLOOR for
        # BOTH arms (0.0 for format-strict tasks; 0.3 for mbpp, since an empty
        # string parses as a valid empty module) -> a fake A_t ~= 0 that looks
        # like a null. Measure the empty-response rate per arm (optimize + test)
        # and refuse to record a contaminated cell as a real result.
        arm_empty: dict = {}
        contaminated = False
        for arm in ("S2", "S3"):
            res, led = run_arm(arm, cfg, seed, sched, train, dev, os.path.join(out, key, arm))
            best = res.best_candidate[COMPONENT] if isinstance(res.best_candidate, dict) else res.best_candidate
            row[f"{arm}_dev"] = round(res.val_aggregate_scores[res.best_idx], 3)
            t_score, t_empty = test_touch(cfg, best, test)
            row[f"{arm}_test"] = round(t_score, 3)
            row[f"{arm}_depth"] = led.canonical_dict()["counters"]["sequential_critical_path_depth"]["max_over_all"]
            try:
                o_empty = float(led.canonical_dict()["counters"].get("verifier", {}).get("empty_rate", 0.0))
            except Exception:  # noqa: BLE001
                o_empty = 0.0
            arm_empty[arm] = {"optimize": round(o_empty, 3), "test": round(t_empty, 3)}
            if max(o_empty, t_empty) > EMPTY_RATE_MAX:
                contaminated = True
                # Fail fast on a dead API: if S2 came back essentially all-empty,
                # don't burn a second full arm of doomed calls on S3.
                if arm == "S2" and t_empty >= 0.9:
                    print(f"{key}: API appears down (S2 test empty_rate={t_empty:.2f}); "
                          f"skipping S3, marking contaminated.", flush=True)
                    break
        row["A_t"] = round(row.get("S3_test", 0.0) - row.get("S2_test", 0.0), 3)
        row["empty_rate"] = arm_empty
        row["max_empty_rate"] = round(
            max((max(v["optimize"], v["test"]) for v in arm_empty.values()), default=0.0), 3)
        row["contaminated"] = contaminated
        row["quality_ok"] = not contaminated
        if contaminated:
            # Do NOT mark done: the resumable orchestrator re-runs this cell once
            # API capacity is restored. run_arm re-optimizes from the seed prompt,
            # so a rerun is clean. Record the reason for the audit trail.
            row["done"] = False
            results[key] = row
            json.dump(results, open(path, "w"), indent=2)
            print(f"{args.task} {key}: CONTAMINATED empty_rate={row['max_empty_rate']} "
                  f"{arm_empty} -> NOT recorded as a result (will re-run).", flush=True)
            continue
        ad = DefaultAdapter(model=cfg.task_model, evaluator=ev,
                            litellm_batch_completion_kwargs=cfg.task_batch_kwargs())
        refl = LM(cfg.reflect_model, **cfg.reflect_kwargs())  # provider-correct (respects reasoning effort)
        row["probe"] = depth_gain_probe(ad, refl, train, dev, k=8, seed=seed)
        row["done"] = True
        results[key] = row
        json.dump(results, open(path, "w"), indent=2)
        print(f"{args.task} {key}: S2={row['S2_test']} S3={row['S3_test']} A_t={row['A_t']} "
              f"depth S2/S3={row.get('S2_depth')}/{row.get('S3_depth')} "
              f"probe_dob={row['probe']['depth_over_breadth']:+.3f} dg_norm={row['probe']['dg_norm']}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
