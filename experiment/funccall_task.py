# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tier-3 candidate on a REAL, STANDARD dataset with SHARED prompt-level gating:
function calling (glaiveai/glaive-function-calling-v2). Unlike MBPP — where each
problem is gated behind its OWN unrelated bug — every function-calling example
shares the SAME sequential ladder:

  F1 (shared protocol): emit a JSON object with keys "name" and "arguments"
                        (no prose, no markdown) — the tool-call convention.
  F2 (shared):          pick the right function name.
  F3 (per-example):     get the argument values right.

Feedback is gated to the first unsatisfied stage: you cannot see a "wrong
function" or "wrong argument" signal until the output is a valid call object. So
if the base model does not DEFAULT to the exact protocol, learning it is a single
prompt-level fix that helps the WHOLE task — the shared-gating structure Depth-Gain
is meant to detect. This is a canonical GEPA use case (structured tool use).
"""

from __future__ import annotations

import json
import re

from gepa.adapters.default_adapter.default_adapter import EvaluationResult

_FC_RE = re.compile(r"<functioncall>\s*(\{.*?\})\s*(?:<\|endoftext\|>|$)", re.S)
_USER_RE = re.compile(r"USER:\s*(.*?)(?:\n\n|\nASSISTANT:|$)", re.S)


def _parse_gold_call(chat: str):
    """Extract the gold {name, arguments-dict} from a glaive transcript."""
    m = _FC_RE.search(chat)
    if not m:
        return None
    raw = m.group(1)
    name_m = re.search(r'"name"\s*:\s*"([^"]+)"', raw)
    if not name_m:
        return None
    name = name_m.group(1)
    # arguments is an INNER json string (single-quoted in the transcript)
    arg_m = re.search(r'"arguments"\s*:\s*[\'"](\{.*\})[\'"]\s*\}?\s*$', raw, re.S)
    args = {}
    if arg_m:
        try:
            args = json.loads(arg_m.group(1))
        except Exception:  # noqa: BLE001
            args = {}
    return {"name": name, "arguments": args}


def _extract_obj(resp: str):
    r = (resp or "").strip()
    if "```" in r:  # strip markdown fences
        blocks = r.split("```")[1::2]
        if blocks:
            b = max(blocks, key=len)
            if b[:4].lower().startswith("json"):
                b = b.split("\n", 1)[1] if "\n" in b else ""
            r = b.strip()
    i, j = r.find("{"), r.rfind("}")
    if i == -1 or j == -1 or j < i:
        return None
    try:
        return json.loads(r[i:j + 1])
    except Exception:  # noqa: BLE001
        return None


def _norm_args(a):
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except Exception:  # noqa: BLE001
            return {}
    return a if isinstance(a, dict) else {}


class FuncCallEvaluator:
    """Score = 0.34*protocol + 0.33*name + 0.33*args-fraction, each stage GATED
    behind the previous (shared sequential feedback)."""

    def __init__(self, meta: dict) -> None:
        self.meta = meta  # input -> {"name":..., "arguments":{...}}

    def __call__(self, data, response) -> EvaluationResult:  # noqa: ANN001
        gold = self.meta[data["input"]]
        obj = _extract_obj(response)
        # F1: valid tool-call protocol
        if not (isinstance(obj, dict) and "name" in obj and "arguments" in obj):
            return EvaluationResult(
                score=0.0,
                feedback='Output ONLY a JSON object with exactly two keys: "name" (the '
                         'function to call) and "arguments" (an object of its parameters). '
                         "No prose, no markdown fences.",
            )
        # F2: right function (gated behind F1)
        if str(obj.get("name")).strip() != gold["name"]:
            return EvaluationResult(
                score=0.34,
                feedback=f"Valid call format, but wrong function: you called "
                         f"{obj.get('name')!r}, expected {gold['name']!r}.",
            )
        # F3: argument correctness (gated behind F2)
        got = _norm_args(obj.get("arguments"))
        gargs = gold["arguments"]
        if not gargs:
            frac = 1.0
        else:
            ok = sum(1 for k, v in gargs.items()
                     if str(got.get(k, "")).strip().lower() == str(v).strip().lower())
            frac = ok / len(gargs)
        score = 0.34 + 0.33 + 0.33 * frac
        if frac >= 1.0:
            fb = "Correct."
        else:
            wrong = [k for k, v in gargs.items()
                     if str(got.get(k, "")).strip().lower() != str(v).strip().lower()]
            k = wrong[0]
            fb = (f"Right function, but argument '{k}' is wrong: you gave "
                  f"{got.get(k)!r}, expected {gargs[k]!r}.")
        return EvaluationResult(score=round(score, 3), feedback=fb)


def build_funccall(n_train: int, n_dev: int, n_test: int, *, seed: int = 0):
    from datasets import load_dataset

    ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train[:4000]")
    import random
    idx = list(range(len(ds)))
    random.Random(seed * 3313 + 7).shuffle(idx)
    need = n_train + n_dev + n_test
    meta: dict = {}
    data: list[dict] = []
    for i in idx:
        e = ds[i]
        if "<functioncall>" not in e["chat"]:
            continue
        gold = _parse_gold_call(e["chat"])
        if not gold:
            continue
        um = _USER_RE.search(e["chat"])
        if not um:
            continue
        query = um.group(1).strip()
        # the available-functions block from the system message (the schema)
        schema = e["system"].split("-", 1)[-1].strip()
        inp = (f"Available functions:\n{schema}\n\nUser request: {query}\n\n"
               "Respond with the appropriate function call.")
        if inp in meta:
            continue
        meta[inp] = gold
        data.append({"input": inp, "additional_context": {}, "answer": json.dumps(gold)})
        if len(data) >= need:
            break
    if len(data) < need:
        raise ValueError(f"glaive yielded only {len(data)} usable examples.")
    ev = FuncCallEvaluator(meta)
    return data[:n_train], data[n_train:n_train + n_dev], data[n_train + n_dev:need], ev
