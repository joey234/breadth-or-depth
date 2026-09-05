# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tier-3 on a REAL dataset: MBPP code generation. The compile-then-test loop is
the canonical NATURALLY-gated task — you cannot run the tests on code that does
not parse/run, so a syntax/runtime error yields only the error message, and the
per-test failure feedback appears only once the code executes. That is exactly the
sequential-discoverability structure Depth-Gain is meant to detect, on a standard
benchmark rather than a synthetic task.

Generated code is executed in an isolated subprocess (temp cwd, hard timeout) so a
bad generation can't hang or touch the workspace."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile

from gepa.adapters.default_adapter.default_adapter import EvaluationResult

_RUN_TIMEOUT = 8  # seconds per candidate execution


def _extract_code(resp: str) -> str:
    r = resp or ""
    if "```" in r:
        parts = r.split("```")
        # take the longest fenced block, drop a leading language tag
        blocks = parts[1::2]
        if blocks:
            b = max(blocks, key=len)
            if b[:6].lower().startswith("python"):
                b = b.split("\n", 1)[1] if "\n" in b else ""
            return b.strip()
    return r.strip()


def _run_tests(code: str, imports: str, tests: list) -> tuple:
    """Run code+tests in an isolated subprocess. Returns (ran_ok, [bool per test], err)."""
    runner = (
        f"{imports}\n{code}\n"
        "import json as __j\n"
        f"__tests = {json.dumps(tests)}\n"
        "__r = []\n"
        "for __t in __tests:\n"
        "    try:\n"
        "        exec(__t); __r.append(True)\n"
        "    except Exception:\n"
        "        __r.append(False)\n"
        "print('__RESULT__' + __j.dumps(__r))\n"
    )
    with tempfile.TemporaryDirectory() as td:
        fp = os.path.join(td, "m.py")
        with open(fp, "w") as f:
            f.write(runner)
        try:
            out = subprocess.run([sys.executable, fp], cwd=td, capture_output=True,
                                  text=True, timeout=_RUN_TIMEOUT)
        except subprocess.TimeoutExpired:
            return False, [False] * len(tests), "timeout"
        for line in out.stdout.splitlines():
            if line.startswith("__RESULT__"):
                return True, json.loads(line[len("__RESULT__"):]), ""
        err = (out.stderr.strip().splitlines() or ["error"])[-1]
        return False, [False] * len(tests), err[:160]


class MBPPEvaluator:
    """Score = 0.3*parses + 0.7*test-pass-fraction. Feedback is naturally gated:
    a syntax/runtime error yields only the error (no test info); once the code
    runs, feedback names the first failing test."""

    def __init__(self, meta: dict) -> None:
        self.meta = meta  # input -> {"imports":str, "tests":[...]}

    def __call__(self, data, response) -> EvaluationResult:  # noqa: ANN001
        m = self.meta[data["input"]]
        code = _extract_code(response)
        try:
            ast.parse(code)
            parses = True
        except SyntaxError as e:
            parses = False
            return EvaluationResult(
                score=0.0,
                feedback=f"Your code has a syntax error ({e.msg}). Output ONLY valid Python code "
                         "that defines the required function.",
            )
        ran, results, err = _run_tests(code, m["imports"], m["tests"])
        frac = sum(results) / len(results) if results else 0.0
        score = 0.3 * parses + 0.7 * frac
        if not ran:
            fb = (f"Your code parses but fails to run ({err}). Make sure it defines the function "
                  "and runs without error.")
        elif frac < 1.0:
            first_fail = next((t for t, ok in zip(m["tests"], results) if not ok), m["tests"][0])
            fb = f"Code runs but fails a test: {first_fail}"
        else:
            fb = "Correct."
        return EvaluationResult(score=score, feedback=fb)


def build_mbpp(n_train: int, n_dev: int, n_test: int, *, seed: int = 0):
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    import random
    idx = list(range(len(ds)))
    random.Random(seed * 2749 + 1).shuffle(idx)
    need = n_train + n_dev + n_test
    meta: dict = {}
    data: list[dict] = []
    for i in idx:
        e = ds[i]
        tests = list(e.get("test_list") or [])
        if not tests:
            continue
        imports = "\n".join(e.get("test_imports") or [])
        prompt = (e.get("prompt") or e.get("text") or "").strip()
        inp = f"{prompt}\nWrite the Python function. Output only code."
        if inp in meta:
            continue
        meta[inp] = {"imports": imports, "tests": tests}
        data.append({"input": inp, "additional_context": {}, "answer": e.get("code", "")})
        if len(data) >= need:
            break
    if len(data) < need:
        raise ValueError(f"MBPP yielded only {len(data)} usable examples.")
    ev = MBPPEvaluator(meta)
    return data[:n_train], data[n_train:n_train + n_dev], data[n_train + n_dev:need], ev
