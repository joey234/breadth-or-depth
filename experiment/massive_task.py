# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""REAL-NEW held-out depth-positive candidate (Phase A, roadmap #8b): hierarchical
intent classification on MASSIVE (mteb/amazon_massive_intent), a REAL, STANDARD
benchmark distinct from TOP-v2 (semantic parsing), glaive (tool calls) and SQL.

Every example shares the SAME non-guessable formalism — a fixed label vocabulary
emitted in a fixed one-line format `scenario / intent` — so the depth chain is
genuinely shared across the task and each stage is gated behind the previous:

  F1 (shared format):   emit ONE line `scenario / intent`, lowercase, no prose
                        (a convention a base model won't default to).
  F2 (shared scenario): correct coarse scenario, from the fixed 18-way vocab
                        (e.g. alarm, iot, datetime, qa) — not guessable from the
                        utterance alone; accumulated from feedback across steps.
  F3 (fine intent):     correct full intent (must be `<scenario>_<action>`),
                        gated behind F2, from the fixed 60-way vocab.

Feedback is gated to the first unsatisfied stage and NAMES the expected label, so
the vocabulary is learnable — the shared knowledge a depth chain accumulates. This
is the shared, non-guessable, depth>=2 gating Depth-Gain is meant to flag. A base
model fails F1 by default (real headroom), so the format fix helps the whole task.
"""

from __future__ import annotations

import re

from gepa.adapters.default_adapter.default_adapter import EvaluationResult

# a tractable default subset (concentrates the vocab like TOP-v2/alarm); Phase A
# eval picks HELD-OUT scenarios — never the ones used to tune the controller.
DEFAULT_SCENARIOS = ("alarm", "weather", "calendar")

_LINE = re.compile(r"^\s*([a-z]+)\s*/\s*([a-z_]+)\s*$")


def _parse(resp: str):
    """Return (scenario, intent) or (None, None). Accept the last clean line so a
    model that adds a trailing answer line still parses; require the strict form."""
    for ln in reversed((resp or "").strip().splitlines()):
        m = _LINE.match(ln.lower())
        if m:
            return m.group(1), m.group(2)
    return None, None


class MassiveEvaluator:
    """Score gated on the first unsatisfied stage: 0.0 (bad format) -> 0.34 (format
    ok, scenario wrong) -> 0.67 (scenario ok, intent wrong) -> 1.0 (correct)."""

    def __init__(self, meta: dict) -> None:
        self.meta = meta  # input -> gold intent string (e.g. "alarm_set")

    def __call__(self, data, response) -> EvaluationResult:  # noqa: ANN001
        gold_intent = self.meta[data["input"]]
        gold_scen = gold_intent.split("_")[0]
        p_scen, p_intent = _parse(response)
        # F1: valid one-line `scenario / intent` format
        if p_scen is None:
            return EvaluationResult(
                score=0.0,
                feedback="Output ONLY one line in the form `scenario / intent`, both lowercase, "
                         "no prose and no other text. Example: `alarm / alarm_set`.",
            )
        # F2: correct coarse scenario (gated behind F1)
        if p_scen != gold_scen:
            return EvaluationResult(
                score=0.34,
                feedback=f"Format ok, but the scenario is wrong: you said `{p_scen}`, expected "
                         f"`{gold_scen}`. The scenario is one of a fixed vocabulary.",
            )
        # F3: correct fine intent, must be <scenario>_<action> (gated behind F2)
        if p_intent != gold_intent:
            return EvaluationResult(
                score=0.67,
                feedback=f"Right scenario `{gold_scen}`, but the intent is wrong: you said "
                         f"`{p_intent}`, expected `{gold_intent}` (form `{gold_scen}_<action>`).",
            )
        return EvaluationResult(score=1.0, feedback="Correct.")


def build_massive(n_train: int, n_dev: int, n_test: int, *, seed: int = 0, scenarios=None):
    from datasets import load_dataset

    scenarios = tuple(scenarios) if scenarios else DEFAULT_SCENARIOS
    ds = load_dataset("mteb/amazon_massive_intent", "en", split="train")
    import random
    idx = list(range(len(ds)))
    random.Random(seed * 6151 + 11).shuffle(idx)
    need = n_train + n_dev + n_test
    meta: dict = {}
    data: list[dict] = []
    for i in idx:
        e = ds[i]
        intent = (e.get("label_text") or "").strip()
        utt = (e.get("text") or "").strip()
        if not intent or not utt:
            continue
        if intent.split("_")[0] not in scenarios:
            continue
        inp = f"Utterance: {utt}"
        if inp in meta:
            continue
        meta[inp] = intent
        data.append({"input": inp, "additional_context": {}, "answer": intent})
        if len(data) >= need:
            break
    if len(data) < need:
        raise ValueError(f"MASSIVE yielded only {len(data)} usable examples for scenarios={scenarios}.")
    ev = MassiveEvaluator(meta)
    return data[:n_train], data[n_train:n_train + n_dev], data[n_train + n_dev:need], ev
