# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tier-3: a realistic task with NATURALLY gated feedback — structured JSON
extraction with transforms. You genuinely cannot report field-level errors on
unparseable JSON, so the error signal is sequentially gated: first learn to emit
valid JSON with the required keys (F1), and only THEN do you see which field
*values* are wrong (F2..). This is the real-world analogue of the synthetic
gated chain — structured output / function-calling is a common GEPA use case.

Fields require transforms a base model won't guess without field feedback
(initials, age bucket, uppercased city), so producing valid JSON is not enough —
the value errors, revealed only after valid JSON, drive the second step. Depth
should therefore help (DG > 0), and Depth-Gain should flag it.
"""

from __future__ import annotations

import json
import random

from gepa.adapters.default_adapter.default_adapter import EvaluationResult

_FIRST = "John Mary Alan Priya Chen Omar Sara Luka Nina Paul Rosa Kwame".split()
_LAST = "Smith Patel Wang Kim Diallo Rossi Silva Haas Ford Nagy Ruiz Bauer".split()
_CITY = "boston london cairo lima oslo delhi accra kyoto quito perth".split()


def _fields(first, last, age, city, mode="guessable"):
    if mode == "opaque":
        # too hard: rule not inferrable even from feedback in a few examples
        return {"k1": str(age + len(city)), "k2": last[-1].upper(), "k3": "Y" if age % 2 == 0 else "N"}
    if mode == "subtle":
        # SWEET SPOT: normal keys (F1 easy) but values need a SIMPLE transform the
        # model does WRONG by default and can't guess from the key name — it only
        # learns the rule from the (gated) field-value feedback. Easy once shown.
        return {
            "name": last,                      # surname ONLY (model defaults to full name)
            "city": city.capitalize(),         # Title-case (input is lowercase)
            "age": str(age + 1),               # age NEXT year (model copies the given age)
        }
    return {  # guessable: self-documenting keys, inferrable without field feedback
        "initials": (first[0] + last[0]).upper(),
        "age_group": "adult" if age >= 18 else "minor",
        "city_upper": city.upper(),
    }


def _extract_json(resp: str):
    resp = (resp or "").strip()
    # tolerant: grab the first {...} block if the model wrapped it in prose/fences
    i, j = resp.find("{"), resp.rfind("}")
    if i == -1 or j == -1 or j < i:
        return None
    try:
        return json.loads(resp[i:j + 1])
    except Exception:  # noqa: BLE001
        return None


class StructuredEvaluator:
    """F1 = valid JSON with the 3 required keys; F2 = field values correct.
    Feedback is NATURALLY gated: field-value errors are only reported once the
    output is valid JSON with the keys (you can't diff fields of broken JSON)."""

    def __init__(self, meta: dict, keys: tuple) -> None:
        self.meta = meta  # input -> gold field dict
        self.KEYS = keys

    def __call__(self, data, response) -> EvaluationResult:  # noqa: ANN001
        gold = self.meta[data["input"]]
        obj = _extract_json(response)
        f1 = isinstance(obj, dict) and all(k in obj for k in self.KEYS)
        if not f1:
            return EvaluationResult(
                score=0.0,
                feedback="Output ONLY a JSON object with exactly these keys: "
                         + ", ".join(self.KEYS) + ". No prose, no code fences.",
            )
        # F2: field correctness (gated behind valid JSON)
        correct = [k for k in self.KEYS if str(obj.get(k)).strip() == gold[k]]
        score = 0.25 + 0.75 * (len(correct) / len(self.KEYS))  # 0.25 for valid JSON, 0.75 for fields
        if len(correct) == len(self.KEYS):
            fb = "Correct."
        else:
            wrong = [k for k in self.KEYS if k not in correct]
            k = wrong[0]
            fb = f"Valid JSON, but field '{k}' is wrong: you gave {obj.get(k)!r}, expected {gold[k]!r}."
        return EvaluationResult(score=score, feedback=fb)


_KEYS = {"guessable": ("initials", "age_group", "city_upper"),
         "opaque": ("k1", "k2", "k3"), "subtle": ("name", "city", "age")}


def build_structured(n_train: int, n_dev: int, n_test: int, *, seed: int = 0, mode: str = "guessable"):
    rng = random.Random(seed * 6151 + 5)
    need = n_train + n_dev + n_test
    meta: dict = {}
    data: list[dict] = []
    for _ in range(need):
        first, last = rng.choice(_FIRST), rng.choice(_LAST)
        age = rng.randint(12, 74)
        city = rng.choice(_CITY)
        text = f"{first} {last} is {age} years old and lives in {city}."
        inp = f"Extract structured data:\n{text}"
        meta[inp] = _fields(first, last, age, city, mode=mode)
        data.append({"input": inp, "additional_context": {}, "answer": json.dumps(meta[inp])})
    ev = StructuredEvaluator(meta, _KEYS[mode])
    return data[:n_train], data[n_train:n_train + n_dev], data[n_train + n_dev:need], ev
