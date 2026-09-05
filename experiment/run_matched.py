#!/usr/bin/env python3
"""General budget-matched Depth-Gain probe runner (compounding.matched_depth_gain_probe, §4.2).

Drives every matched re-run the reviewer's blocker #1 requires — cross-model matrix,
reflector-capability sweep, K sweep, and cost measurement — from one place, so each uses
the SAME matched probe as the headline scatter. Per (model, reflector, task, k) cell is
checkpointed to <out>/results.json and skipped if already clean (empty_rate <= 0.15). Cost
and token counts are recorded per cell (task LM + reflector LM).

Examples:
    # cross-model matrix (one model per invocation; run sequentially per provider)
    python -m experiment.run_matched --model openrouter/qwen/qwen3.6-35b-a3b \
        --tasks ungated gated_D2 hotpotqa structured_subtle --out runs/matched_crossmodel
    # reflector-capability sweep (task model fixed, reflector varies)
    python -m experiment.run_matched --model openrouter/qwen/qwen3.6-35b-a3b \
        --reflect-model openrouter/qwen/qwen3-8b --tasks ungated gated_D2 ungated_D4 \
        --out runs/matched_capability
    # K sweep
    python -m experiment.run_matched --model openrouter/qwen/qwen3.6-35b-a3b \
        --tasks ifbench gated_D2 --k 3 --out runs/matched_ksweep
"""
from __future__ import annotations

import argparse
import json
import os

from experiment.compounding import (
    build_compound,
    build_compound_chain,
    matched_depth_gain_probe,
)
from experiment.run_study import _load_env


def _is_openai(m: str) -> bool:
    return m.startswith("openai/")


# task-kind -> (builder(n_train,n_dev), task_max_tokens_nonopenai)
def _build(task: str, n_train: int, n_dev: int, seed: int):
    n_test = 4
    if task in ("hotpotqa", "ifbench", "hover", "pupa"):
        from experiment.common import build_dataset
        return build_dataset(task, n_train, n_dev, n_test), 256
    if task == "mbpp":
        from experiment.mbpp_task import build_mbpp
        return build_mbpp(n_train, n_dev, n_test, seed=seed), 512
    if task == "funccall":
        from experiment.funccall_task import build_funccall
        return build_funccall(n_train, n_dev, n_test, seed=seed), 400
    if task == "top":
        from experiment.top_task import build_top
        return build_top(n_train, n_dev, n_test, seed=seed, domains=["alarm"]), 400
    if task.startswith("structured_"):
        from experiment.structured_task import build_structured
        return build_structured(n_train, n_dev, n_test, seed=seed, mode=task.split("_", 1)[1]), 256
    if task == "ungated":
        return build_compound(n_train, n_dev, n_test, alpha=0.0, seed=seed), 320
    if task == "gated_D2":
        return build_compound_chain(n_train, n_dev, n_test, d=2, seed=seed, gated=True), 320
    if task == "checklist_D2":
        return build_compound_chain(n_train, n_dev, n_test, d=2, seed=seed, gated=True, checklist=True), 320
    if task == "checklist_D4":
        return build_compound_chain(n_train, n_dev, n_test, d=4, seed=seed, gated=True, checklist=True), 256
    if task == "ungated_D4":
        return build_compound_chain(n_train, n_dev, n_test, d=4, seed=seed, gated=False), 320
    if task.startswith("dchain"):
        return build_compound_chain(n_train, n_dev, n_test, d=int(task[6:]), seed=seed, gated=True), 256
    raise ValueError(task)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="task model (also reflector unless --reflect-model)")
    ap.add_argument("--reflect-model", default=None)
    ap.add_argument("--tasks", nargs="*", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n-train", type=int, default=20)
    ap.add_argument("--n-dev", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--env", default=".env")
    args = ap.parse_args()
    _load_env(args.env)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "results.json")
    results = json.load(open(path)) if os.path.exists(path) else {}

    from gepa.adapters.default_adapter.default_adapter import DefaultAdapter
    from gepa.lm import LM

    reflect_model = args.reflect_model or args.model
    mslug = args.model.split("/")[-1]
    rslug = reflect_model.split("/")[-1]
    for task in args.tasks:
        key = f"{mslug}__refl={rslug}__{task}__k{args.k}"
        if key in results and results[key].get("done"):
            print(f"{key}: [skip]", flush=True)
            continue
        try:
            (train, dev, _te, ev), mt = _build(task, args.n_train, args.n_dev, args.seed)
            if _is_openai(args.model):
                task_kw = {"max_tokens": max(mt, 4000), "reasoning_effort": "minimal"}
            else:
                task_kw = {"temperature": 0.0, "max_tokens": max(mt, 4000), "extra_body": {"reasoning": {"enabled": False}}}
            ad = DefaultAdapter(model=args.model, evaluator=ev, litellm_batch_completion_kwargs=task_kw)
            if _is_openai(reflect_model):
                refl = LM(reflect_model, max_tokens=4000, reasoning_effort="minimal")
            else:
                refl = LM(reflect_model, temperature=1.0, max_tokens=1200, extra_body={"reasoning": {"enabled": False}})
            r = matched_depth_gain_probe(ad, refl, train, dev, k=args.k, seed=args.seed)
            tot = getattr(ad._lm, "total_responses", 0) + getattr(refl, "total_responses", 0)
            emp = getattr(ad._lm, "empty_responses", 0) + getattr(refl, "empty_responses", 0)
            r["empty_rate"] = round(emp / max(1, tot), 3)
            r["calls"] = tot
            r["cost_usd"] = round(getattr(ad._lm, "total_cost", 0.0) + getattr(refl, "total_cost", 0.0), 5)
            r["tokens_in"] = getattr(ad._lm, "total_tokens_in", 0) + getattr(refl, "total_tokens_in", 0)
            r["tokens_out"] = getattr(ad._lm, "total_tokens_out", 0) + getattr(refl, "total_tokens_out", 0)
            r["model"] = mslug
            r["reflector"] = rslug
            r["task"] = task
            r["done"] = r["empty_rate"] <= 0.15
            results[key] = r
            json.dump(results, open(path, "w"), indent=2)
            flag = "" if r["done"] else "  [CONTAMINATED -> rerun]"
            print(f"{key}: dob={r['depth_over_breadth']:+.3f} dg={r['dg_norm']:.3f} "
                  f"B={r['breadth_ceiling']} L={r['depth_ceiling']} calls={r['calls']} "
                  f"${r['cost_usd']} emp={r['empty_rate']}{flag}", flush=True)
        except Exception as e:
            print(f"{key}: ERROR {type(e).__name__}: {str(e)[:160]}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
