# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Depth-positive candidate on a REAL, STANDARD benchmark: TOP-v2 hierarchical
semantic parsing (WillHeld/top_v2). Every example shares the SAME non-guessable
formalism — an utterance is reproduced verbatim wrapped in a bracket tree
`[IN:INTENT ... [SL:SLOT tokens ] ... ]` — so the depth chain is genuinely shared
across the whole task and each stage is gated behind the previous:

  F1 (shared format):   emit a valid bracket tree starting `[IN:...`, reproducing
                        the utterance tokens (a convention a base model won't guess).
  F2 (shared intent):   correct top intent, from a fixed dataset vocabulary.
  F3 (shared slots):    correct slot-label multiset, from a fixed vocabulary.
  F4:                   exact normalized match of the full parse.

The intent/slot vocabularies are dataset conventions that CANNOT be guessed from a
single example — they are accumulated into the prompt across iterations. That is
exactly the shared, non-guessable, depth->=2 gating Depth-Gain should flag as depth.
Feedback is gated to the first unsatisfied stage (and names the expected label so the
vocabulary is learnable — the shared knowledge a depth chain accumulates).
"""

from __future__ import annotations

import re

from gepa.adapters.default_adapter.default_adapter import EvaluationResult

_INTENT = re.compile(r"\[IN:(\w+)")
_SLOT = re.compile(r"\[SL:(\w+)")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _balanced(s: str) -> bool:
    d = 0
    for c in s:
        if c == "[":
            d += 1
        elif c == "]":
            d -= 1
            if d < 0:
                return False
    return d == 0


def _extract_parse(resp: str) -> str:
    r = (resp or "").strip()
    if "```" in r:
        blocks = r.split("```")[1::2]
        if blocks:
            r = max(blocks, key=len).strip()
    i = r.find("[IN:")
    if i == -1:
        return r
    # take from first [IN: to its matching close bracket
    d = 0
    for j in range(i, len(r)):
        if r[j] == "[":
            d += 1
        elif r[j] == "]":
            d -= 1
            if d == 0:
                return r[i:j + 1]
    return r[i:]


class TOPEvaluator:
    """Score = 0.25*validtree + 0.25*intent + 0.25*slot-labels + 0.25*exact, each
    stage GATED behind the previous (shared sequential feedback)."""

    def __init__(self, meta: dict) -> None:
        self.meta = meta  # input -> gold parse string

    def __call__(self, data, response) -> EvaluationResult:  # noqa: ANN001
        gold = self.meta[data["input"]]
        pred = _extract_parse(response)
        g_intent = (_INTENT.search(gold) or [None, None])[1] if _INTENT.search(gold) else None
        g_slots = sorted(_SLOT.findall(gold))
        # F1: valid bracket tree
        if not (pred.startswith("[IN:") and _balanced(pred)):
            return EvaluationResult(
                score=0.0,
                feedback="Output ONLY a bracketed semantic parse of the form "
                         "[IN:INTENT <words> [SL:SLOT <words> ] ... ], reproducing the "
                         "utterance words verbatim inside the brackets. No prose.",
            )
        p_intent = (_INTENT.search(pred) or [None, None])[1] if _INTENT.search(pred) else None
        # F2: intent (gated behind F1)
        if p_intent != g_intent:
            return EvaluationResult(
                score=0.25,
                feedback=f"Valid tree, but the top intent is wrong: you used "
                         f"[IN:{p_intent}], expected [IN:{g_intent}].",
            )
        p_slots = sorted(_SLOT.findall(pred))
        # F3: slot-label multiset (gated behind F2)
        if p_slots != g_slots:
            return EvaluationResult(
                score=0.5,
                feedback=f"Right intent, but the slot labels are wrong: you used "
                         f"{['SL:' + s for s in p_slots]}, expected {['SL:' + s for s in g_slots]}.",
            )
        # F4: exact normalized match (gated behind F3)
        if _norm(pred) == _norm(gold):
            return EvaluationResult(score=1.0, feedback="Correct.")
        return EvaluationResult(
            score=0.75,
            feedback=f"Right intent and slots, but the exact structure/spans differ. "
                     f"Expected: {gold}",
        )


def build_top(n_train: int, n_dev: int, n_test: int, *, seed: int = 0, domains=None):
    from datasets import load_dataset

    ds = load_dataset("WillHeld/top_v2", split="train")
    import random
    idx = list(range(len(ds)))
    random.Random(seed * 5077 + 3).shuffle(idx)
    need = n_train + n_dev + n_test
    meta: dict = {}
    data: list[dict] = []
    for i in idx:
        e = ds[i]
        if domains and e["domain"] not in domains:
            continue
        gold = (e.get("semantic_parse") or "").strip()
        utt = (e.get("utterance") or "").strip()
        if not gold.startswith("[IN:") or not utt:
            continue
        inp = f"Parse this utterance into the semantic frame:\n{utt}"
        if inp in meta:
            continue
        meta[inp] = gold
        data.append({"input": inp, "additional_context": {}, "answer": gold})
        if len(data) >= need:
            break
    if len(data) < need:
        raise ValueError(f"TOP-v2 yielded only {len(data)} usable examples.")
    ev = TOPEvaluator(meta)
    return data[:n_train], data[n_train:n_train + n_dev], data[n_train + n_dev:need], ev
