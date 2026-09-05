# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Study orchestrator: run the paired S2/S3 arms across seeds, gate each pair on
the Phase-3 matching verification, and emit ledgers + a results JSON.

Usage (see RUNBOOK.md for the full guide):

    uv run python -m experiment.run_study --dataset toy --dry-run          # cheap smoke
    uv run python -m experiment.run_study --dataset ifbench --seeds 0 1 2  # real run

Nothing here touches the optimizer's logic: arms are composed from a pluggable
candidate selector + a pinned batch sampler + an iteration-cap stopper, and the
ledger rides on the existing instrumentation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

from experiment.common import (
    AcceptAll,
    BlindBestOfKProposer,
    FixedScheduleBatchSampler,
    MaxIterationsStopper,
    RootResetSelector,
    SelectAll,
    build_dataset,
    make_schedule,
    score_prompt,
)
from experiment.config import RunConfig, rough_cost_estimate
from experiment.verify_matching import verify_matching


def _load_env(path: str) -> None:
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and v and k not in os.environ:
                os.environ[k] = v


SEED_PROMPT = {"system_prompt": "You are a helpful assistant. Answer the question."}
COMPONENT = "system_prompt"


def run_arm(arm: str, cfg: RunConfig, seed: int, schedule, train, dev, out_dir: str):
    """Run one arm (S1 Best-of-K, S2 root-reset, or S3 stateful) -> (result, ledger)."""
    from gepa.instrumentation import optimize_with_ledger

    if arm == "S1":
        return _run_s1_arm(cfg, seed, schedule, train, dev, out_dir)
    if arm == "S2":
        selector = RootResetSelector()
    elif arm == "S3":
        selector = "pareto"
    else:
        raise ValueError(f"unknown arm '{arm}' (expected S1, S2 or S3)")

    # Unconditional validation: score every proposed candidate on the dev panel
    # and pool it (accept-all + select-all). Only the selector differs between arms.
    extra: dict = {}
    if cfg.unconditional_validation:
        extra["acceptance_criterion"] = AcceptAll()
        extra["selection_strategy"] = SelectAll()

    return optimize_with_ledger(
        seed_candidate=dict(SEED_PROMPT),
        trainset=train,
        valset=dev,
        task_lm=cfg.task_model,
        task_lm_kwargs=cfg.task_batch_kwargs(),
        evaluator=cfg.evaluator,  # attached below
        reflection_lm=cfg.reflect_model,
        reflection_lm_kwargs=cfg.reflect_kwargs(),
        candidate_selection_strategy=selector,
        batch_sampler=FixedScheduleBatchSampler(schedule),
        skip_perfect_score=False,
        stop_callbacks=MaxIterationsStopper(cfg.n_iters),
        seed=seed,
        run_label=f"{arm}-seed{seed}",
        ledger_dir=out_dir,
        ledger_basename=f"ledger_{arm}",
        display_progress_bar=False,
        raise_on_exception=True,
        **extra,
    )


def _run_s1_arm(cfg: RunConfig, seed: int, schedule, train, dev, out_dir: str):
    """S1 Best-of-K anchor.

    Same composition as S2 (root-reset parent, fixed schedule, K iterations,
    unconditional validation -> scalar argmax selection) EXCEPT the proposer is
    the blind Best-of-K proposer, which never reads the reflective_dataset
    (feedback). Composed via attach_ledger so the blind proposer calls the
    *ledgered* reflection LM (reflector calls/tokens are counted exactly like
    S2/S3). No optimizer edits — only public seams.
    """
    import gepa
    from gepa.instrumentation import attach_ledger

    attached = attach_ledger(
        task_lm=cfg.task_model,
        task_lm_kwargs=cfg.task_batch_kwargs(),
        evaluator=cfg.evaluator,
        reflection_lm=cfg.reflect_model,
        reflection_lm_kwargs=cfg.reflect_kwargs(),
        callbacks=None,
        seed=seed,
        run_label=f"S1-seed{seed}",
        config={"arm": "S1"},
    )
    proposer = BlindBestOfKProposer(attached["reflection_lm"], COMPONENT)

    extra: dict = {}
    if cfg.unconditional_validation:
        extra["acceptance_criterion"] = AcceptAll()
        extra["selection_strategy"] = SelectAll()

    try:
        result = gepa.optimize(
            seed_candidate=dict(SEED_PROMPT),
            trainset=train,
            valset=dev,
            adapter=attached["adapter"],
            reflection_lm=attached["reflection_lm"],
            custom_candidate_proposer=proposer,
            candidate_selection_strategy=RootResetSelector(),
            batch_sampler=FixedScheduleBatchSampler(schedule),
            skip_perfect_score=False,
            stop_callbacks=MaxIterationsStopper(cfg.n_iters),
            seed=seed,
            display_progress_bar=False,
            raise_on_exception=True,
            callbacks=attached["callbacks"],
            **extra,
        )
    finally:
        attached["flush"]()

    ledger = attached["ledger"]
    ledger.write(out_dir, basename="ledger_S1")
    return result, ledger


# --- Data-quality gate thresholds -------------------------------------------
# A seed is contaminated (not a real result) when a provider-stall window turns
# many task responses into empty strings, which are scored as failures. Two
# independent signals catch it: the seed-prompt (p0) dev score collapsing below
# a floor it otherwise clears comfortably, and the measured empty-response rate.
DEV_BASE_FLOOR = 0.15  # p0 reliably clears this; a collapse to ~0 means empties
EMPTY_RATE_MAX = 0.15  # fraction of task responses that came back blank
MAX_QUALITY_RETRIES = 2  # re-run a contaminated seed up to this many times


def test_touch(cfg: RunConfig, best_prompt: str, test: list[dict]) -> tuple[float, float]:
    """Evaluate on the held-out test set once. Returns (score, empty_rate)."""
    from gepa.adapters.default_adapter.default_adapter import DefaultAdapter

    adapter = DefaultAdapter(
        model=cfg.task_model,
        evaluator=cfg.evaluator,
        litellm_batch_completion_kwargs=cfg.task_batch_kwargs(),
    )
    score = score_prompt(adapter, COMPONENT, best_prompt, test)
    lm = getattr(adapter, "_lm", None)
    n = getattr(lm, "total_responses", 0) or 0
    empty_rate = (getattr(lm, "empty_responses", 0) / n) if n else 0.0
    return score, empty_rate


def _seed_quality(arm_out: dict, test_meta: dict, *, touch_test: bool, dev_base_floor: float = DEV_BASE_FLOOR) -> tuple[bool, list]:
    """Return (ok, reasons). Flags a seed whose scores are contaminated by empty
    (stalled) task responses, using the p0 dev floor + measured empty rates.
    ``dev_base_floor`` is task-dependent (low-baseline tasks like HotpotQA need a
    lower floor); the empty-rate check is the task-agnostic primary signal."""
    reasons: list = []
    for arm in ("S2", "S3"):
        base = arm_out[arm]["dev_base"]
        if base < dev_base_floor:
            reasons.append(f"{arm}: dev_base {base:.3f} < {dev_base_floor} (empty-contaminated)")
        er = arm_out[arm]["ledger"]["counters"]["verifier"].get("empty_rate", 0.0)
        if er > EMPTY_RATE_MAX:
            reasons.append(f"{arm}: optimize empty_rate {er:.2f} > {EMPTY_RATE_MAX}")
        if touch_test:
            ter = test_meta.get(arm, {}).get("empty_rate", 0.0)
            if ter > EMPTY_RATE_MAX:
                reasons.append(f"{arm}: test empty_rate {ter:.2f} > {EMPTY_RATE_MAX}")
    return (len(reasons) == 0), reasons


def _build_summary(cfg: RunConfig, per_seed: list, *, touch_test: bool) -> dict:
    # Clean = passed the matching gate AND the data-quality gate. Means are over
    # clean seeds only; contaminated seeds are reported but excluded.
    clean = [s for s in per_seed if s["matched"] and s.get("quality_ok", True)]
    contaminated = [s["seed"] for s in per_seed if not s.get("quality_ok", True)]

    def _mean_ci(vals: list):
        import math
        import statistics as st

        if len(vals) < 2:
            return {"mean": (vals[0] if vals else None), "sd": None, "ci95": None, "n": len(vals)}
        m = st.mean(vals)
        sd = st.stdev(vals)
        # t_.975 for small df; fall back to 1.96 for larger n.
        tval = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}.get(
            len(vals), 1.96
        )
        h = tval * sd / math.sqrt(len(vals))
        return {"mean": m, "sd": sd, "ci95": [m - h, m + h], "n": len(vals)}

    return {
        "config": cfg.summary(),
        "quality_gate": {"dev_base_floor": getattr(cfg, "dev_base_floor", DEV_BASE_FLOOR), "empty_rate_max": EMPTY_RATE_MAX},
        "n_seeds": len(cfg.seeds),
        "n_done": len(per_seed),
        "n_matched": sum(1 for s in per_seed if s["matched"]),
        "n_clean": len(clean),
        "contaminated_seeds": contaminated,
        "A_t_dev": _mean_ci([s["dev"]["A_t"] for s in clean]),
        "A_t_test": _mean_ci([s["test"]["A_t"] for s in clean]) if touch_test else None,
        # Back-compat scalar means (clean seeds).
        "mean_A_t_dev_matched": (sum(s["dev"]["A_t"] for s in clean) / len(clean)) if clean else None,
        "mean_A_t_test_matched": (
            sum(s["test"]["A_t"] for s in clean) / len(clean) if (clean and touch_test) else None
        ),
        "total_cost": sum(s["cost"] for s in per_seed),
        "per_seed": per_seed,
    }


def run_study(cfg: RunConfig, out_root: str, *, strict_validation: bool = False, touch_test: bool = True) -> dict:
    per_seed_data = getattr(cfg, "per_seed_data", False)
    if not per_seed_data:
        train, dev, test, evaluator = build_dataset(cfg.dataset, cfg.n_train, cfg.n_dev, cfg.n_test,
                                                    retrieval=cfg.retrieval, retrieval_k=cfg.retrieval_k)
        cfg.evaluator = evaluator  # type: ignore[attr-defined]
    schedule = make_schedule(cfg.n_train, cfg.minibatch_size, cfg.n_iters)

    os.makedirs(out_root, exist_ok=True)
    results_path = os.path.join(out_root, "study_results.json")

    # Resume: reuse per-seed results already checkpointed in study_results.json.
    # A seed is skipped ONLY if it completed AND passed both gates; contaminated
    # (quality_ok False) or unmatched seeds are re-run to recover them.
    per_seed: list = []
    clean_seeds: set = set()
    if os.path.exists(results_path):
        try:
            prior = json.load(open(results_path))
            per_seed = list(prior.get("per_seed", []))
            clean_seeds = {s["seed"] for s in per_seed if s.get("matched") and s.get("quality_ok", True)}
            if clean_seeds:
                print(f"[resume] reusing {len(clean_seeds)} clean seed(s): {sorted(clean_seeds)}")
        except Exception as e:
            print(f"[resume] could not read prior results ({e}); starting fresh")

    for seed in cfg.seeds:
        if seed in clean_seeds:
            print(f"seed {seed}: [skip] already clean")
            continue
        # Drop any prior contaminated/unmatched entry for this seed before re-running.
        per_seed = [s for s in per_seed if s["seed"] != seed]
        seed_dir = os.path.join(out_root, f"seed{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        if per_seed_data:
            # independent stratified draw for this seed
            train, dev, test, evaluator = build_dataset(
                cfg.dataset, cfg.n_train, cfg.n_dev, cfg.n_test, seed=seed,
                retrieval=cfg.retrieval, retrieval_k=cfg.retrieval_k
            )
            cfg.evaluator = evaluator  # type: ignore[attr-defined]

        entry = None
        for attempt in range(MAX_QUALITY_RETRIES + 1):
            if attempt > 0:
                # A quality retry must re-optimize from scratch (the prior attempt
                # was empty-contaminated); drop its per-arm checkpoints so the
                # resume path doesn't reload the contaminated result. Attempt 0
                # keeps any checkpoint from a prior process (crash recovery).
                for arm in ("S2", "S3"):
                    cp = os.path.join(seed_dir, f"best_{arm}.json")
                    if os.path.exists(cp):
                        os.remove(cp)
            entry = _run_one_seed(cfg, seed, schedule, train, dev, test, seed_dir, strict_validation, touch_test)
            tag = f"seed {seed}" + (f" (retry {attempt})" if attempt else "")
            print(
                f"{tag}: matched={entry['matched']} quality_ok={entry['quality_ok']}"
                f"  A_t(dev)={entry['dev']['A_t']:+.3f}"
                + (f"  A_t(test)={entry['test']['A_t']:+.3f}" if entry["test"] else "")
                + f"  depth S2/S3={entry['diagnostics']['critical_path_depth']['S2']}/"
                f"{entry['diagnostics']['critical_path_depth']['S3']}"
            )
            if not entry["matched"]:
                print(f"    VOID (matching failed): {entry['match_failures']}")
            if not entry["quality_ok"]:
                print(f"    CONTAMINATED: {entry['quality_reasons']}")
            if entry["matched"] and entry["quality_ok"]:
                break  # clean; stop retrying
            if attempt < MAX_QUALITY_RETRIES:
                print(f"    -> retrying seed {seed} ({attempt + 1}/{MAX_QUALITY_RETRIES})")

        per_seed.append(entry)
        per_seed.sort(key=lambda s: s["seed"])
        with open(results_path, "w") as f:  # checkpoint after every seed
            json.dump(_build_summary(cfg, per_seed, touch_test=touch_test), f, indent=2)

    summary = _build_summary(cfg, per_seed, touch_test=touch_test)
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def _run_one_seed(cfg, seed, schedule, train, dev, test, seed_dir, strict_validation, touch_test):
    """Run both arms for one seed, verify matching + data quality; return the entry.

    Each arm's optimization result (best prompt + dev scores + canonical ledger) is
    checkpointed to ``best_{arm}.json`` the instant the arm completes. This makes the
    expensive optimization RESUMABLE within a seed: if the cheap downstream test-touch
    is interrupted (e.g. the container is reclaimed after inactivity), a re-run reloads
    the finished arms from disk instead of redoing ~40 min of optimization per seed."""
    arm_out = {}
    for arm in ("S2", "S3"):
        best_path = os.path.join(seed_dir, f"best_{arm}.json")
        if os.path.exists(best_path):
            saved = json.load(open(best_path))
            # Only trust a checkpoint that is itself clean. Contamination from a
            # transient provider stall shows up as a HIGH EMPTY RATE (failed calls
            # degrade to ""); that is the discriminating signal. A low dev_base
            # with a 0 empty rate is a legitimately hard task (pupa's base is ~0.15),
            # NOT contamination — do not discard on dev_base alone or a clean arm
            # gets needlessly re-optimized.
            _er = saved["ledger"]["counters"]["verifier"].get("empty_rate", 0.0)
            if _er <= EMPTY_RATE_MAX:
                arm_out[arm] = {
                    "ledger": saved["ledger"],
                    "dev_best": saved["dev_best"],
                    "dev_base": saved["dev_base"],
                    "best_prompt": saved["best_prompt"],
                    "cost": saved["cost"],
                }
                print(f"    [resume] loaded checkpointed {arm} for seed {seed} (skip re-optimize)")
                continue
            print(f"    [resume] DISCARDING contaminated {arm} checkpoint "
                  f"(empty_rate={_er:.2f} > {EMPTY_RATE_MAX}); re-optimizing")
            for _stale in (best_path, os.path.join(seed_dir, f"test_{arm}.json")):
                if os.path.exists(_stale):
                    os.remove(_stale)
        res, ledger = run_arm(arm, cfg, seed, schedule, train, dev, seed_dir)
        best_prompt = res.best_candidate[COMPONENT] if isinstance(res.best_candidate, dict) else res.best_candidate
        cd = ledger.canonical_dict()
        arm_out[arm] = {
            "ledger": cd,
            "dev_best": res.val_aggregate_scores[res.best_idx],
            "dev_base": res.val_aggregate_scores[0],
            "best_prompt": best_prompt,
            "cost": cd["counters"]["reflector"]["cost"] + cd["counters"]["verifier"]["cost"],
        }
        with open(best_path, "w") as f:  # checkpoint the finished arm immediately
            json.dump(arm_out[arm], f)

    verdict = verify_matching(arm_out["S2"]["ledger"], arm_out["S3"]["ledger"], strict_validation=strict_validation)

    test_scores: dict = {}
    test_meta: dict = {}
    if touch_test:
        for arm in ("S2", "S3"):
            tpath = os.path.join(seed_dir, f"test_{arm}.json")
            td = json.load(open(tpath)) if os.path.exists(tpath) else None
            # Reuse a cached test-touch ONLY if it was clean. A provider stall can
            # cache a mass-empty test score (empty_rate high, score collapsed);
            # reloading that would corrupt A_t, so recompute instead.
            if td is not None and td.get("empty_rate", 1.0) <= EMPTY_RATE_MAX:
                sc, er = td["score"], td["empty_rate"]
                print(f"    [resume] loaded test-touch for {arm} (seed {seed})")
            else:
                if td is not None:
                    print(f"    [resume] DISCARDING contaminated test-touch for {arm} "
                          f"(empty_rate={td.get('empty_rate')}); recomputing")
                sc, er = test_touch(cfg, arm_out[arm]["best_prompt"], test)
                with open(tpath, "w") as f:
                    json.dump({"score": sc, "empty_rate": er}, f)
            test_scores[arm] = sc
            test_meta[arm] = {"empty_rate": er}

    quality_ok, quality_reasons = _seed_quality(
        arm_out, test_meta, touch_test=touch_test, dev_base_floor=getattr(cfg, "dev_base_floor", DEV_BASE_FLOOR)
    )

    a_t_dev = arm_out["S3"]["dev_best"] - arm_out["S2"]["dev_best"]
    a_t_test = (test_scores.get("S3", 0.0) - test_scores.get("S2", 0.0)) if touch_test else None
    return {
        "seed": seed,
        "matched": verdict.matched,
        "match_failures": verdict.failures,
        "quality_ok": quality_ok,
        "quality_reasons": quality_reasons,
        "diagnostics": verdict.diagnostics,
        "dev": {"S2": arm_out["S2"]["dev_best"], "S3": arm_out["S3"]["dev_best"], "A_t": a_t_dev},
        "test": {**test_scores, "A_t": a_t_test} if touch_test else None,
        "empty_rates": {
            "S2_opt": arm_out["S2"]["ledger"]["counters"]["verifier"].get("empty_rate", 0.0),
            "S3_opt": arm_out["S3"]["ledger"]["counters"]["verifier"].get("empty_rate", 0.0),
            **({"S2_test": test_meta["S2"]["empty_rate"], "S3_test": test_meta["S3"]["empty_rate"]} if touch_test else {}),
        },
        "cost": arm_out["S2"]["cost"] + arm_out["S3"]["cost"],
    }


def build_config_from_args(args) -> RunConfig:
    cfg = RunConfig()
    if args.dry_run:
        cfg.dataset = "toy"
        cfg.n_train, cfg.n_dev, cfg.n_test = 8, 6, 4
        cfg.n_iters = 3
        cfg.seeds = [0]
    if args.dataset:
        cfg.dataset = args.dataset
    if args.seeds:
        cfg.seeds = args.seeds
    if args.n_iters:
        cfg.n_iters = args.n_iters
    if args.n_train:
        cfg.n_train = args.n_train
    if args.n_dev:
        cfg.n_dev = args.n_dev
    if args.n_test:
        cfg.n_test = args.n_test
    if args.task_model:
        cfg.task_model = args.task_model
    if args.reflect_model:
        cfg.reflect_model = args.reflect_model
    if args.task_max_tokens is not None:
        cfg.task_max_tokens = args.task_max_tokens
    if args.task_reasoning is not None:
        cfg.task_reasoning = args.task_reasoning == "on"
    if args.reflect_reasoning is not None:
        cfg.reflect_reasoning = args.reflect_reasoning == "on"
    if args.task_reasoning_effort is not None:
        cfg.task_reasoning_effort = args.task_reasoning_effort
    if args.reflect_reasoning_effort is not None:
        cfg.reflect_reasoning_effort = args.reflect_reasoning_effort
    if args.no_unconditional:
        cfg.unconditional_validation = False
    if args.dev_base_floor is not None:
        cfg.dev_base_floor = args.dev_base_floor
    if args.per_seed_data:
        cfg.per_seed_data = True
    if args.retrieval:
        cfg.retrieval = True
    if args.retrieval_k is not None:
        cfg.retrieval_k = args.retrieval_k
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description="Breadth-or-Depth S2/S3 study runner")
    ap.add_argument("--dataset", choices=["toy", "ifbench", "hotpotqa", "hover", "pupa"], default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--n-iters", type=int, default=None)
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--n-dev", type=int, default=None)
    ap.add_argument("--n-test", type=int, default=None)
    ap.add_argument("--task-model", default=None)
    ap.add_argument("--reflect-model", default=None)
    ap.add_argument("--task-max-tokens", type=int, default=None, help="task-model token budget (bump for thinking mode)")
    ap.add_argument("--task-reasoning", choices=["on", "off"], default=None, help="task model thinking mode (default off=instruct)")
    ap.add_argument("--reflect-reasoning", choices=["on", "off"], default=None, help="reflector thinking mode (default on)")
    ap.add_argument("--task-reasoning-effort", choices=["minimal", "low", "medium", "high"], default=None,
                    help="task-model reasoning effort (overrides on/off; OpenAI reasoning_effort, Qwen extra_body effort)")
    ap.add_argument("--reflect-reasoning-effort", choices=["minimal", "low", "medium", "high"], default=None,
                    help="reflector reasoning effort")
    ap.add_argument("--out", default=None, help="output dir (default: a temp dir)")
    ap.add_argument("--strict-validation", action="store_true", help="force validation-side gating on")
    ap.add_argument("--no-strict", action="store_true", help="disable validation-side gating even with unconditional validation")
    ap.add_argument("--no-unconditional", action="store_true", help="use acceptance-gated validation (the old proxy behavior)")
    ap.add_argument("--no-test", action="store_true", help="skip the held-out test touch")
    ap.add_argument("--dry-run", action="store_true", help="tiny toy config for a cheap end-to-end smoke")
    ap.add_argument("--dev-base-floor", type=float, default=None, help="p0 dev-score floor for the quality gate (task-dependent; ~0.05 for HotpotQA)")
    ap.add_argument("--per-seed-data", action="store_true", help="each seed draws its own stratified subset (registered-grade)")
    ap.add_argument("--retrieval", action="store_true", help="BM25 retrieve-then-read for HotpotQA/HoVer (vs gold context)")
    ap.add_argument("--retrieval-k", type=int, default=None, help="passages retrieved per query (default 5)")
    ap.add_argument("--env", default=".env")
    args = ap.parse_args()

    _load_env(args.env)
    if "OPENAI_API_KEY" not in os.environ and "OPENROUTER_API_KEY" not in os.environ:
        print("No API key in env; set one or pass --env.", file=sys.stderr)
        sys.exit(1)

    cfg = build_config_from_args(args)
    out_root = args.out or tempfile.mkdtemp(prefix="bod_study_")
    print("=" * 80)
    print("Breadth-or-Depth study run")
    print(json.dumps(cfg.summary(), indent=2))
    print("cost sketch:", rough_cost_estimate(cfg))
    print("output dir:", out_root)
    print("=" * 80)

    # With unconditional validation the validation-side counters should pair, so
    # gate strictly by default; --no-strict opts out, --strict-validation forces on.
    strict = args.strict_validation or (cfg.unconditional_validation and not args.no_strict)
    summary = run_study(cfg, out_root, strict_validation=strict, touch_test=not args.no_test)

    print("-" * 80)
    print(f"matched: {summary['n_matched']}/{summary['n_seeds']}  clean (matched+quality): {summary['n_clean']}")
    if summary.get("contaminated_seeds"):
        print(f"contaminated (excluded): {summary['contaminated_seeds']}")
    print(f"A_t(dev, clean):  {summary['A_t_dev']}")
    print(f"A_t(test, clean): {summary['A_t_test']}")
    print(f"total real cost: ${summary['total_cost']:.4f}")
    print(f"results: {os.path.join(out_root, 'study_results.json')}")


if __name__ == "__main__":
    main()
