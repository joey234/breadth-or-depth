"""Three-arm information-matched chain: separate feedback-history from prompt state.

This is a custom K-per-round loop, not GEPA Pareto. Do not mix scores with
legacy gated ``Transform:`` D-chain runs.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from experiment.common import make_schedule
from experiment.compounding import (
    SEED_PROMPT,
    chain_feature_passes,
    chain_hints,
    oracle_system_prompt,
    passed_chain_prefix,
)
from experiment.run_study import COMPONENT

ARMS = ("root_local", "root_history", "lineage_history")
CLEAR_RATE = 0.90
ORACLE_PROMPT_MIN = 0.95
VISIBLE_RULE_MIN = 0.80
UNREVEALED_LEAK_MAX = 0.35
EMPTY_RATE_MAX = 0.15


class MatchingError(RuntimeError):
    """A matching, leakage, or competence assertion failed. Abort the cell."""


@dataclass(frozen=True)
class RequirementTranscript:
    items: tuple[tuple[str, str], ...]

    @property
    def ids(self) -> list[str]:
        return [i for i, _ in self.items]

    def render(self) -> str:
        if not self.items:
            return "(none)"
        return "\n".join(f"{n}. {text}" for n, (_, text) in enumerate(self.items, 1))

    def digest(self) -> str:
        return hashlib.sha256(self.render().encode("utf-8")).hexdigest()

    @classmethod
    def from_hints(cls, hints: list[tuple[str, str]]) -> RequirementTranscript:
        return cls(tuple(hints))


def oracle_transcript(round_idx: int, d: int) -> RequirementTranscript:
    """1-indexed round: H_t = HINTS[:min(t, D)]."""
    n = max(1, min(round_idx, d))
    return RequirementTranscript.from_hints(chain_hints(d)[:n])


def prompt_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def _unwrap_prompt(text: str, fallback: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t or fallback


def evaluate_prompt(adapter, prompt: str, items: list[dict], d: int) -> dict:
    eb = adapter.evaluate(items, {COMPONENT: prompt}, capture_traces=True)
    scores = list(eb.scores or [])
    n = max(1, len(scores))
    prefixes: list[int] = []
    feat_hits: dict[str, int] = {name: 0 for name, _ in chain_hints(d)}
    empties = 0
    for tr, sc in zip(eb.trajectories or [], scores, strict=True):
        resp = str(tr.get("full_assistant_response", "") or "")
        if not resp.strip():
            empties += 1
        data = tr["data"]
        prefixes.append(passed_chain_prefix(data, resp, d))
        for name, ok in chain_feature_passes(data, resp, d).items():
            if ok:
                feat_hits[name] += 1
        _ = sc
    return {
        "score": (sum(scores) / len(scores)) if scores else 0.0,
        "prefix_mean": (sum(prefixes) / len(prefixes)) if prefixes else 0.0,
        "full_solve": sum(1 for p in prefixes if p >= d) / n,
        "feature_rates": {name: feat_hits[name] / n for name in feat_hits},
        "empty_rate": empties / n,
        "max_prefix": max(prefixes) if prefixes else 0,
    }


class ScoreCache:
    def __init__(self, adapter, d: int) -> None:
        self.adapter = adapter
        self.d = d
        self._store: dict[tuple[str, str], dict] = {}

    def score(self, prompt: str, items: list[dict], split_id: str) -> dict:
        key = (prompt, split_id)
        if key not in self._store:
            self._store[key] = evaluate_prompt(self.adapter, prompt, items, self.d)
        return self._store[key]


def collect_outputs_without_feedback(parent: str, minibatch: list[dict], adapter) -> list[dict]:
    eb = adapter.evaluate(minibatch, {COMPONENT: parent}, capture_traces=True)
    examples = []
    for tr in eb.trajectories or []:
        examples.append({
            "input": tr["data"]["input"],
            "output": str(tr.get("full_assistant_response", "") or "")[:400],
        })
    return examples


def reflect_with_transcript(
    parent: str,
    transcript: RequirementTranscript,
    task_description: str,
    adapter,
    minibatch: list[dict],
    reflector,
    examples: list[dict] | None = None,
) -> str:
    if examples is None:
        examples = collect_outputs_without_feedback(parent, minibatch, adapter)
    ex_lines = []
    for e in examples:
        if "feedback" in e:
            raise MatchingError("native evaluator feedback leaked into reflection examples")
        ex_lines.append(f"- input:\n{e['input']}\n  output: {e['output']}")
    rendered = transcript.render()
    prompt = (
        "You are improving a system prompt for a transformation task.\n\n"
        f"OVERALL TASK:\n{task_description}\n\n"
        f"CURRENT SYSTEM PROMPT:\n{parent}\n\n"
        f"REQUIREMENTS DISCOVERED SO FAR:\n{rendered}\n\n"
        "EXAMPLE INPUTS AND CURRENT OUTPUTS:\n"
        f"{chr(10).join(ex_lines)}\n\n"
        "Write one replacement system prompt that incorporates the known requirements.\n"
        "Output only the replacement prompt."
    )
    if "Feedback:" in prompt or "Exact next rule" in prompt:
        raise MatchingError("native evaluator feedback leaked into the reflection prompt")
    return _unwrap_prompt(reflector(prompt) or "", parent)


def natural_transcript(
    selected_prompt: str,
    selection_data: list[dict],
    current: RequirementTranscript,
    adapter,
    d: int,
) -> RequirementTranscript:
    stats = evaluate_prompt(adapter, selected_prompt, selection_data, d)
    rates = stats["feature_rates"]
    if any(rates.get(rid, 0.0) < CLEAR_RATE for rid in current.ids):
        return current
    all_hints = chain_hints(d)
    if len(current.items) >= d:
        return current
    nxt = all_hints[len(current.items)]
    return RequirementTranscript.from_hints([*current.items, nxt])


def _assert_task_surface(items: list[dict], information_mode: str) -> None:
    if information_mode == "legacy-opaque":
        return
    for ex in items:
        inp = ex.get("input", "")
        if inp.startswith("Transform:") or "\nTransform: " in inp:
            raise MatchingError("legacy Transform: surface in a primary-run example")
        if not inp.startswith("TASK\n"):
            raise MatchingError("example input is missing the TASK card")


def _active_arms(information_mode: str, arms: tuple[str, ...]) -> tuple[str, ...]:
    if information_mode == "full-specification":
        keep = [a for a in arms if a != "root_local"]
        if "root_history" not in keep:
            keep = ["root_history", *keep]
        return tuple(dict.fromkeys(keep))
    return arms


def run_information_matched_replicate(
    *,
    d: int,
    seed: int,
    adapter,
    reflector,
    train: list[dict],
    sel: list[dict],
    val: list[dict],
    test: list[dict],
    task_description: str,
    information_mode: str,
    schedule: str = "oracle",
    k: int = 8,
    minibatch_size: int = 3,
    arms: tuple[str, ...] = ARMS,
) -> dict:
    if schedule not in ("oracle", "natural-unlock"):
        raise ValueError(f"unknown schedule {schedule!r}")
    if information_mode not in ("explicit-overall", "full-specification", "legacy-opaque"):
        raise ValueError(f"unknown information mode {information_mode!r}")

    all_items = train + sel + val + test
    _assert_task_surface(all_items, information_mode)
    if any(id(x) is id(y) for x in test for y in sel + val + train):
        raise MatchingError("test split aliases another split")

    cache = ScoreCache(adapter, d)
    oracle_items = sel + val
    oracle_stats = cache.score(oracle_system_prompt(d), oracle_items, "oracle_sel_val")
    if oracle_stats["score"] < ORACLE_PROMPT_MIN:
        print(
            f"  WARN oracle-prompt diagnostic: sel+val {oracle_stats['score']:.3f} < {ORACLE_PROMPT_MIN}",
            flush=True,
        )

    p0 = SEED_PROMPT
    active = _active_arms(information_mode, arms)
    alias_root_local = information_mode == "full-specification" and "root_local" in arms
    archives: dict[str, list[str]] = {arm: [p0] for arm in active}
    lineage_parent = p0
    current_natural = oracle_transcript(1, d)
    mb_sched = make_schedule(len(train), minibatch_size, d)
    rounds_out: list[dict] = []
    visible_by_round: dict[int, list[float]] = {}
    root_local_unrevealed: list[float] = []

    for t in range(1, d + 1):
        if schedule == "oracle":
            ht = oracle_transcript(t, d)
        else:
            ht = current_natural
        if information_mode == "full-specification":
            ht = oracle_transcript(d, d)
            h_local = ht
        else:
            h_local = oracle_transcript(1, d)

        histories = {
            "root_local": h_local,
            "root_history": ht,
            "lineage_history": ht,
        }
        parents = {
            "root_local": p0,
            "root_history": p0,
            "lineage_history": lineage_parent,
        }
        if histories["root_history"].digest() != histories["lineage_history"].digest():
            raise MatchingError("history-arm transcripts diverged")
        if information_mode != "full-specification":
            extra = [i for i in histories["root_local"].ids if i not in oracle_transcript(1, d).ids]
            if extra:
                raise MatchingError(f"root_local received hint IDs beyond H1: {extra}")
            if any(hid != "uppercase" for hid in histories["root_local"].ids) and d >= 1:
                if histories["root_local"].ids != ["uppercase"]:
                    raise MatchingError("root_local transcript is not exactly H1")

        print(
            f"  D={d} seed={seed} round {t}/{d}: "
            f"local={histories['root_local'].ids} "
            f"hist={histories['root_history'].ids}",
            flush=True,
        )

        mb_ids = mb_sched[t - 1]
        minibatch = [train[i] for i in mb_ids]
        generation: dict[str, list[str]] = {}
        selected_score: dict[str, float] = {}
        selected_stage: dict[str, float] = {}
        proposal_count: dict[str, int] = {}
        parent_hashes = {arm: prompt_hash(parents[arm]) for arm in active}

        for arm in active:
            print(f"    {arm}: reflecting K={k} from {parents[arm][:40]!r}...", flush=True)
            examples = collect_outputs_without_feedback(parents[arm], minibatch, adapter)
            cands: list[str] = []
            for _k in range(k):
                cand = reflect_with_transcript(
                    parent=parents[arm],
                    transcript=histories[arm],
                    task_description=task_description,
                    adapter=adapter,
                    minibatch=minibatch,
                    reflector=reflector,
                    examples=examples,
                )
                cands.append(cand)
            generation[arm] = cands
            archives[arm].extend(cands)
            proposal_count[arm] = len(cands)
            if proposal_count[arm] != k:
                raise MatchingError(f"{arm} produced {proposal_count[arm]} != K={k} proposals")

            scored = [(c, cache.score(c, sel, "sel")) for c in cands]
            best_c, best_st = max(scored, key=lambda x: x[1]["score"])
            selected_score[arm] = round(best_st["score"], 4)
            selected_stage[arm] = round(best_st["prefix_mean"], 4)

            n_visible = len(histories[arm].ids)
            vis_rate = max(
                (sum(st["feature_rates"].get(hid, 0.0) for hid in histories[arm].ids) / max(1, n_visible)
                 for _, st in scored),
                default=0.0,
            )
            if arm in ("root_history", "lineage_history"):
                visible_by_round.setdefault(t, []).append(vis_rate)

            if arm == "root_local" and information_mode == "explicit-overall" and d >= 2:
                revealed = set(histories[arm].ids)
                unrev = [hid for hid, _ in chain_hints(d) if hid not in revealed]
                if unrev:
                    leak = max(
                        (sum(st["feature_rates"].get(hid, 0.0) for hid in unrev) / len(unrev)
                         for _, st in scored),
                        default=0.0,
                    )
                    root_local_unrevealed.append(leak)
            _ = best_c

        if "lineage_history" in generation:
            lineage_pool = [lineage_parent, *generation["lineage_history"]]
            lineage_parent = max(lineage_pool, key=lambda p: cache.score(p, sel, "sel")["score"])
        if schedule == "natural-unlock":
            current_natural = natural_transcript(lineage_parent, sel, ht, adapter, d)

        rounds_out.append({
            "round": t,
            "transcript_ids": {arm: histories[arm].ids for arm in active},
            "transcript_hash": {arm: histories[arm].digest()[:16] for arm in active},
            "parent_prompt_hash": parent_hashes,
            "minibatch_ids": mb_ids,
            "proposal_count": proposal_count,
            "selected_score": selected_score,
            "selected_stage": selected_stage,
        })

    visible_round_best = [max(vs) for vs in visible_by_round.values()] if visible_by_round else []
    if visible_round_best and min(visible_round_best) < VISIBLE_RULE_MIN:
        print(
            f"  WARN visible-rule diagnostic: worst-round rate "
            f"{min(visible_round_best):.3f} < {VISIBLE_RULE_MIN}",
            flush=True,
        )
    if root_local_unrevealed and max(root_local_unrevealed) > UNREVEALED_LEAK_MAX:
        print(
            f"  WARN root_local implemented unrevealed rules at "
            f"{max(root_local_unrevealed):.3f} (recorded, not treated as leakage)",
            flush=True,
        )

    final_prompt: dict[str, str] = {}
    final_val: dict[str, float] = {}
    final_test: dict[str, dict] = {}
    for arm in active:
        ranked = [(p, cache.score(p, val, "val")) for p in archives[arm]]
        best_p, best_v = max(ranked, key=lambda x: x[1]["score"])
        final_prompt[arm] = best_p
        final_val[arm] = round(best_v["score"], 4)
        test_st = evaluate_prompt(adapter, best_p, test, d)
        if test_st["empty_rate"] > EMPTY_RATE_MAX:
            raise MatchingError(f"{arm} test empty_rate {test_st['empty_rate']:.3f} > {EMPTY_RATE_MAX}")
        final_test[arm] = {
            "score": round(test_st["score"], 4),
            "full_solve": round(test_st["full_solve"], 4),
            "prefix_mean": round(test_st["prefix_mean"], 4),
            "feature_rates": {k: round(v, 4) for k, v in test_st["feature_rates"].items()},
            "empty_rate": round(test_st["empty_rate"], 4),
        }

    if alias_root_local:
        final_prompt["root_local"] = final_prompt["root_history"]
        final_val["root_local"] = final_val["root_history"]
        final_test["root_local"] = dict(final_test["root_history"])
        for rd in rounds_out:
            rd["transcript_ids"]["root_local"] = rd["transcript_ids"]["root_history"]
            rd["transcript_hash"]["root_local"] = rd["transcript_hash"]["root_history"]
            rd["parent_prompt_hash"]["root_local"] = rd["parent_prompt_hash"]["root_history"]
            rd["proposal_count"]["root_local"] = rd["proposal_count"]["root_history"]
            rd["selected_score"]["root_local"] = rd["selected_score"]["root_history"]
            rd["selected_stage"]["root_local"] = rd["selected_stage"]["root_history"]

    y = {arm: final_test[arm]["score"] for arm in ARMS if arm in final_test}
    fb_hist = y["root_history"] - y["root_local"]
    prompt_state = y["lineage_history"] - y["root_history"]
    total = y["lineage_history"] - y["root_local"]

    expected_proposals = k * d
    proposal_ok = True
    for arm in (active if not alias_root_local else ("root_history", "lineage_history")):
        got = sum(rd["proposal_count"][arm] for rd in rounds_out)
        if got != expected_proposals:
            proposal_ok = False
            raise MatchingError(f"{arm} generated {got} != K*R={expected_proposals} proposals")

    transcripts_matched = all(
        rd["transcript_hash"].get("root_history") == rd["transcript_hash"].get("lineage_history")
        for rd in rounds_out
    )
    no_leak = all(
        rd["transcript_ids"].get("root_local", ["uppercase"]) == (
            rd["transcript_ids"]["root_history"] if information_mode == "full-specification"
            else oracle_transcript(1, d).ids
        )
        for rd in rounds_out
    )

    task_lm = getattr(adapter, "_lm", None)
    return {
        "config": {
            "D": d,
            "seed": seed,
            "information_mode": information_mode,
            "schedule": schedule,
            "k": k,
            "R": d,
            "arms": list(ARMS),
            "active_arms": list(active),
            "aliased_root_local": alias_root_local,
            "n_train": len(train),
            "n_sel": len(sel),
            "n_val": len(val),
            "n_test": len(test),
            "task_description": task_description,
            "p0_hash": prompt_hash(p0),
        },
        "rounds": rounds_out,
        "final": {
            "prompt_hash": {arm: prompt_hash(final_prompt[arm]) for arm in final_prompt},
            "validation": final_val,
            "test": {arm: final_test[arm]["score"] for arm in final_test},
            "test_detail": final_test,
            "feedback_history_effect": round(fb_hist, 4),
            "prompt_state_effect": round(prompt_state, 4),
            "total_matched_effect": round(total, 4),
        },
        "checks": {
            "oracle_prompt_sel": round(oracle_stats["score"], 4),
            "oracle_prompt_below_0_95": oracle_stats["score"] < ORACLE_PROMPT_MIN,
            "transcripts_matched": transcripts_matched,
            "proposal_counts_matched": proposal_ok,
            "no_feedback_leakage": no_leak,
            "visible_rule_best": round(max(visible_round_best), 4) if visible_round_best else None,
            "visible_rule_worst": round(min(visible_round_best), 4) if visible_round_best else None,
            "root_local_unrevealed_best": (
                round(max(root_local_unrevealed), 4) if root_local_unrevealed else None
            ),
            "root_local_implemented_unrevealed": bool(
                root_local_unrevealed and max(root_local_unrevealed) > UNREVEALED_LEAK_MAX
            ),
            "full_specification_alias": alias_root_local,
            "task_surface_ok": True,
        },
        "usage": {
            "task_tokens_in": getattr(task_lm, "total_tokens_in", 0),
            "task_tokens_out": getattr(task_lm, "total_tokens_out", 0),
            "reflect_tokens_in": getattr(reflector, "total_tokens_in", 0),
            "reflect_tokens_out": getattr(reflector, "total_tokens_out", 0),
            "task_cost_usd": round(getattr(task_lm, "total_cost", 0.0), 5),
            "reflect_cost_usd": round(getattr(reflector, "total_cost", 0.0), 5),
            "task_empty_rate": round(
                getattr(task_lm, "empty_responses", 0) / max(1, getattr(task_lm, "total_responses", 1)),
                4,
            ),
        },
        "diagnostics": {
            "final_prompts": final_prompt,
        },
    }
