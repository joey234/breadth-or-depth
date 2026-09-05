# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Shared building blocks for the study harness.

Everything here is the *validated* form from the pilot ladder:

- ``RootResetSelector``       — S2's defining property: parent is always p0.
- ``FixedScheduleBatchSampler`` — a pinned minibatch schedule so the two arms
  are paired on the item-level schedule (a naive shared seed does NOT pair them,
  because the candidate selector and the batch sampler share one RNG stream).
- ``MaxIterationsStopper``    — cap by proposal rounds, not metric-call budget,
  so both arms get the same number of iterations regardless of how they spend
  rollouts.
- ``score_prompt`` / eval caching by prompt text — a stochastic task model
  re-scores the same prompt to different numbers; cache by text where an exact
  value is needed.
- dataset builders for the real IFBench sample (its own verifiers) and a cheap
  toy multi-constraint task for dry-runs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from gepa.adapters.default_adapter.default_adapter import EvaluationResult

# ---------------------------------------------------------------------------
# Arms / scheduling / stopping
# ---------------------------------------------------------------------------


class RootResetSelector:
    """S2 root-reset: the parent of every mutation is p0 (candidate index 0)."""

    def select_candidate_idx(self, state) -> int:
        return 0


class FixedScheduleBatchSampler:
    """A minibatch schedule fixed up front, identical across arms and seeds.

    Pins the item-level schedule so S2 and S3 are paired on it regardless of
    which candidate selector runs (the selector otherwise perturbs the shared
    RNG and desyncs the schedule)."""

    def __init__(self, schedule: list[list[int]]) -> None:
        self.schedule = schedule

    def next_minibatch_ids(self, loader, state):
        return list(self.schedule[state.i % len(self.schedule)])


class MaxIterationsStopper:
    """Stop after ``n`` proposal rounds (state.i >= n)."""

    def __init__(self, n: int) -> None:
        self.n = n

    def __call__(self, state) -> bool:
        return state.i >= self.n


# --- Unconditional validation (Phase 2) ------------------------------------
# The faithful S2/S3 arms remove GEPA's conditional (acceptance-gated) validation:
# EVERY proposed candidate is scored on the same dev panel and enters the pool,
# so S2 returns argmax over {p0..pK} (root included) and S3's controlled GEPA
# validates every candidate. Composed from the pluggable acceptance + selection
# seams — no optimizer edits. The arms then differ on exactly one axis: the
# candidate selector (root-reset vs Pareto). A non-improving candidate that never
# makes the Pareto front is validated and pooled but is never chosen as a parent,
# so S3 stays recognizably stateful GEPA.


class AcceptAll:
    """Acceptance criterion that accepts every proposal (unconditional)."""

    def should_accept(self, proposal, state) -> bool:
        return True


class SelectAll:
    """Selection strategy that keeps every proposal (from the given list)."""

    def select(self, proposals, state, acceptance_criterion):
        return list(proposals)


# --- S1 Best-of-K anchor -----------------------------------------------------
# The anchor arm. Best-of-K "differs from GEPA on two axes at once — its proposer
# never reads feedback, only its selector sees a scalar" (proposal.md). We compose
# it at the `custom_candidate_proposer` seam: given the current candidate (always
# p0 under RootResetSelector), produce ONE fresh variation of the instruction
# WITHOUT ever reading the reflective_dataset (inputs/outputs/feedback). Only
# sampling temperature drives diversity; scalar selection (argmax dev aggregate
# under unconditional validation) picks the winner. This makes S1 vs S2 a clean
# one-axis contrast on *reflection* (does reading feedback help at all?), the
# mirror of S2 vs S3's one-axis contrast on *statefulness*. Depth 1, like S2.


class BlindBestOfKProposer:
    """S1 proposer: rewrite the current instruction blind to feedback.

    Signature matches GEPA's ``custom_candidate_proposer``:
    ``(candidate, reflective_dataset, components_to_update) -> {component: text}``.
    The ``reflective_dataset`` argument (the minibatch inputs, generated outputs,
    and per-item feedback) is deliberately IGNORED — that is the defining property
    of the anchor. The LM handle should be the ledgered reflection LM so proposer
    calls are counted exactly like S2/S3 reflector calls.
    """

    def __init__(self, reflection_lm, component: str) -> None:
        self.lm = reflection_lm
        self.component = component

    def __call__(self, candidate, reflective_dataset, components_to_update):
        base = candidate[self.component]
        prompt = (
            "You are refining a system prompt used to answer instruction-following tasks. "
            "Below is the CURRENT system prompt. Write ONE alternative system prompt that could "
            "plausibly do the same job well. Freely vary the wording, structure, and strategy — "
            "aim for a genuinely different phrasing, not a paraphrase. Output ONLY the new system "
            "prompt, with no preamble, commentary, or quotation marks.\n\n"
            f"CURRENT SYSTEM PROMPT:\n{base}\n\nNEW SYSTEM PROMPT:"
        )
        new_text = (self.lm(prompt) or "").strip()
        return {self.component: new_text or base}


def make_schedule(n_train: int, minibatch_size: int, n_iters: int, *, offset: int = 0) -> list[list[int]]:
    """Deterministic round-robin schedule of ``n_iters`` minibatches over train ids.

    Fully determined by (n_train, minibatch_size, n_iters, offset) — no RNG — so
    both arms consume an identical schedule.
    """
    if n_train < minibatch_size:
        raise ValueError("n_train must be >= minibatch_size")
    sched: list[list[int]] = []
    cursor = offset % n_train
    for _ in range(n_iters):
        batch = [(cursor + j) % n_train for j in range(minibatch_size)]
        sched.append(batch)
        cursor = (cursor + minibatch_size) % n_train
    return sched


# ---------------------------------------------------------------------------
# Scoring helper (with per-text eval cache)
# ---------------------------------------------------------------------------
def score_prompt(adapter, component: str, prompt: str, items: list[dict]) -> float:
    eb = adapter.evaluate(items, {component: prompt}, capture_traces=False)
    return sum(eb.scores) / len(eb.scores) if eb.scores else 0.0


# ---------------------------------------------------------------------------
# Toy multi-constraint task (cheap dry-runs)
# ---------------------------------------------------------------------------
class MultiConstraintEvaluator:
    """Five orthogonal, programmatic constraints; score = fraction satisfied."""

    MAX_WORDS = 25

    def __call__(self, data, response) -> EvaluationResult:
        ans = data["answer"]
        r = response
        checks = {
            "contains_exact_answer": ans in r,
            "starts_with_Answer": r.lstrip().startswith("Answer:"),
            "ends_with_FINAL": r.rstrip().endswith(f"FINAL: {ans}"),
            "no_commas": "," not in r,
            "within_word_limit": len(r.split()) <= self.MAX_WORDS,
        }
        failed = [k for k, ok in checks.items() if not ok]
        score = (len(checks) - len(failed)) / len(checks)
        fb = (
            "All five constraints satisfied."
            if not failed
            else (
                f"Constraints FAILED: {failed}. Correct answer '{ans}'. Required: include the exact "
                f"answer; start with 'Answer:'; end with 'FINAL: {ans}'; no commas; <= {self.MAX_WORDS} words."
            )
        )
        return EvaluationResult(score=score, feedback=fb)


def _toy_item(inp: str, ans: str) -> dict:
    return {"input": inp, "additional_context": {}, "answer": ans}


_TOY_POOL = [
    ("What is the capital of Australia?", "Canberra"),
    ("What is 15% of 200?", "30"),
    ("What year did World War II end?", "1945"),
    ("How many sides does a hexagon have?", "6"),
    ("What is the chemical symbol for gold?", "Au"),
    ("What is the square root of 169?", "13"),
    ("Who wrote Romeo and Juliet?", "Shakespeare"),
    ("What is the freezing point of water in Celsius?", "0"),
    ("What is the capital of Canada?", "Ottawa"),
    ("What is 20% of 150?", "30"),
    ("In what year did the Berlin Wall fall?", "1989"),
    ("How many degrees are in a right angle?", "90"),
    ("What is the chemical symbol for sodium?", "Na"),
    ("What is the square root of 196?", "14"),
    ("Who painted the Mona Lisa?", "Leonardo"),
    ("What is the boiling point of water in Celsius?", "100"),
    ("How many strings does a standard guitar have?", "6"),
    ("What is the largest planet in the solar system?", "Jupiter"),
    ("What is 3 cubed?", "27"),
    ("What is the capital of Japan?", "Tokyo"),
]


def build_toy(n_train: int, n_dev: int, n_test: int):
    total = n_train + n_dev + n_test
    if total > len(_TOY_POOL):
        raise ValueError(f"toy task has only {len(_TOY_POOL)} items; requested {total}")
    items = [_toy_item(q, a) for q, a in _TOY_POOL[:total]]
    train = items[:n_train]
    dev = items[n_train : n_train + n_dev]
    test = items[n_train + n_dev : total]
    return train, dev, test, MultiConstraintEvaluator()


# ---------------------------------------------------------------------------
# Real IFBench task (its own verifiers)
# ---------------------------------------------------------------------------
def _ifbench_dir() -> str:
    env = os.environ.get("IFBENCH_DIR")
    if env:
        return env
    here = Path(__file__).resolve()
    for candidate in (
        Path.cwd() / "IFBench",
        here.parents[1] / "IFBench",
    ):
        if candidate.is_dir():
            return str(candidate)
    return "IFBench"


def _load_ifbench_registry():
    ifbench_dir = _ifbench_dir()
    if not os.path.isdir(ifbench_dir):
        raise RuntimeError(
            f"IFBench checkout not found at {ifbench_dir}. Clone it "
            "(git clone https://github.com/allenai/IFBench) and set IFBENCH_DIR, "
            "and `pip install nltk emoji syllapy` plus nltk data."
        )
    if ifbench_dir not in sys.path:
        sys.path.insert(0, ifbench_dir)
    import instructions_registry  # type: ignore[import-not-found]

    return instructions_registry


class IFBenchEvaluator:
    """Scores a response with IFBench's own verifier functions (loose: fraction
    of the example's constraints satisfied). Feedback lists failed constraints."""

    def __init__(self, spec_by_prompt: dict, registry) -> None:
        self.spec = spec_by_prompt
        self.reg = registry

    def __call__(self, data, response) -> EvaluationResult:
        ids, kwargs_list = self.spec[data["input"]]
        followed = []
        for iid, kw in zip(ids, kwargs_list, strict=False):
            inst = self.reg.INSTRUCTION_DICT[iid](iid)
            kw = {k: v for k, v in kw.items() if v is not None}
            inst.build_description(**kw)
            args = inst.get_instruction_args()
            if args and "prompt" in args:
                inst.build_description(prompt=data["input"])
            ok = bool(response and response.strip() and inst.check_following(response))
            followed.append((iid, ok))
        score = sum(ok for _, ok in followed) / len(followed) if followed else 0.0
        failed = [iid for iid, ok in followed if not ok]
        fb = (
            "All constraints satisfied."
            if not failed
            else (
                f"Constraints NOT satisfied: {failed}. Re-read the instruction(s) in the prompt and "
                "satisfy EVERY constraint exactly (counts, formats, and wording matter)."
            )
        )
        return EvaluationResult(score=score, feedback=fb)


def build_ifbench(n_train: int, n_dev: int, n_test: int, *, skip_language: bool = True):
    """Build disjoint train/dev/test splits from allenai/IFBench_test with a
    real IFBench evaluator. Deterministic slice by dataset order."""
    registry = _load_ifbench_registry()
    from datasets import load_dataset

    ds: Any = load_dataset("allenai/IFBench_test", split="train")
    picks: list[dict] = []
    for e in ds:
        if skip_language and any("language" in i for i in e["instruction_id_list"]):
            continue
        picks.append(e)
        if len(picks) >= n_train + n_dev + n_test:
            break
    if len(picks) < n_train + n_dev + n_test:
        raise ValueError(f"IFBench yielded only {len(picks)} usable examples for the requested split.")
    spec = {}
    data = []
    for e in picks:
        spec[e["prompt"]] = (e["instruction_id_list"], e["kwargs"])
        data.append({"input": e["prompt"], "additional_context": {}, "answer": ""})
    ev = IFBenchEvaluator(spec, registry)
    train = data[:n_train]
    dev = data[n_train : n_train + n_dev]
    test = data[n_train + n_dev : n_train + n_dev + n_test]
    return train, dev, test, ev


# ---------------------------------------------------------------------------
# HotpotQA (multi-hop QA, gold-context pilot; token-F1)
# ---------------------------------------------------------------------------
import collections  # noqa: E402
import re  # noqa: E402
import string  # noqa: E402


def _qa_normalize(s: str) -> str:
    """SQuAD/HotpotQA answer normalization: lowercase, strip punctuation +
    articles, collapse whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _qa_f1(pred: str, gold: str) -> float:
    """Standard token-level F1 between a predicted answer and the gold answer."""
    p = _qa_normalize(pred).split()
    g = _qa_normalize(gold).split()
    if not p or not g:
        return float(p == g)  # both empty -> 1.0, one empty -> 0.0
    common = collections.Counter(p) & collections.Counter(g)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    prec = num_same / len(p)
    rec = num_same / len(g)
    return 2 * prec * rec / (prec + rec)


def _qa_extract(response: str) -> str:
    """Pull the answer out of a possibly-verbose response: prefer text after the
    last 'answer:' marker, else the last non-empty line, else the whole thing.
    Keeps the F1 target achievable while still rewarding concise answers."""
    if not response:
        return ""
    m = list(re.finditer(r"\b(?:final answer|answer)\b\s*[:\-]", response, re.IGNORECASE))
    if m:
        tail = response[m[-1].end():].strip()
        return tail.splitlines()[0].strip() if tail else ""
    lines = [ln.strip() for ln in response.splitlines() if ln.strip()]
    return lines[-1] if lines else response.strip()


class HotpotQAEvaluator:
    """Scores a response by token-F1 against the gold HotpotQA answer (best of
    full-response and extracted-answer F1). Feedback gives the gold answer so the
    reflective proposer can learn an extraction/formatting strategy."""

    def __init__(self, gold_by_input: dict) -> None:
        self.gold = gold_by_input

    def __call__(self, data, response) -> EvaluationResult:
        gold = self.gold.get(data["input"], data.get("answer", "") or "")
        resp = response or ""
        score = max(_qa_f1(resp, gold), _qa_f1(_qa_extract(resp), gold))
        if score >= 0.99:
            fb = "Correct."
        else:
            fb = (
                f"Gold answer: '{gold}'. F1={score:.2f}. Give ONLY the exact answer span "
                "(a short phrase/entity), not a sentence or explanation; put it after 'Answer:'."
            )
        return EvaluationResult(score=score, feedback=fb)


def _hotpot_context_str(context) -> str:
    """Render the distractor-setting context (titles + sentences) as text."""
    return "\n".join(_passages_from_context(context))


# --- Retrieve-then-read (standard BM25 baseline) ----------------------------
# The gold-context pilot hands the model all 10 distractor paragraphs. The
# registered-grade upgrade instead RETRIEVES: BM25 (Okapi) over a task-local
# corpus (every paragraph across the sampled questions) returns the top-k
# passages, which become the reader's context. This is the standard sparse
# retrieve-then-read RAG baseline (rank_bm25 / Pyserini-style tokenization); the
# model must now read imperfect, retrieved evidence rather than gold. Single-hop
# BM25 is the canonical baseline; recall@k is reported so its ceiling is explicit.
import re as _re  # noqa: E402


def _tok(text: str) -> list:
    return _re.findall(r"[a-z0-9]+", (text or "").lower())


def _passages_from_context(context) -> list:
    titles = context.get("title", [])
    sents = context.get("sentences", [])
    out = []
    for t, ss in zip(titles, sents, strict=False):
        out.append(f"[{t}] " + " ".join(ss).strip())
    return out


class _BM25Index:
    """Okapi BM25 over a passage corpus (rank_bm25). topk() returns the k
    highest-scoring passages for a query, ties broken by corpus order."""

    def __init__(self, passages: list) -> None:
        from rank_bm25 import BM25Okapi

        # de-dup passages, preserving order (a task-local corpus repeats gold paras)
        seen: dict = {}
        for p in passages:
            if p and p not in seen:
                seen[p] = len(seen)
        self.passages = list(seen.keys())
        self.bm25 = BM25Okapi([_tok(p) for p in self.passages]) if self.passages else None

    def topk(self, query: str, k: int) -> list:
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(_tok(query))
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        return [self.passages[i] for i in order[:k]]


def build_hotpotqa(n_train: int, n_dev: int, n_test: int, *, level: str = "hard",
                   retrieval: bool = False, retrieval_k: int = 5):
    """Build disjoint train/dev/test from hotpot_qa (distractor), token-F1 scored.

    Gold-context (default): input = question + all 10 paragraphs (2 gold + 8
    distractors). Retrieve-then-read (``retrieval=True``): BM25 over the
    task-local corpus (every paragraph across the sampled questions) supplies the
    top-``retrieval_k`` passages as context — the standard sparse RAG baseline."""
    from datasets import load_dataset

    ds: Any = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    need = n_train + n_dev + n_test
    raw: list = []
    for e in ds:
        if level and e.get("level") and e["level"] != level:
            continue
        raw.append(e)
        if len(raw) >= need:
            break
    if len(raw) < need:
        raise ValueError(f"HotpotQA yielded only {len(raw)} examples for the requested split.")

    index = None
    if retrieval:
        corpus: list = []
        for e in raw:
            corpus.extend(_passages_from_context(e["context"]))
        index = _BM25Index(corpus)

    gold: dict = {}
    data: list[dict] = []
    for e in raw:
        q = e["question"].strip()
        if retrieval:
            ctx = "\n".join(index.topk(q, retrieval_k))
        else:
            ctx = _hotpot_context_str(e["context"])
        inp = f"Context:\n{ctx}\n\nQuestion: {q}"
        gold[inp] = e["answer"].strip()
        data.append({"input": inp, "additional_context": {}, "answer": e["answer"].strip()})
    ev = HotpotQAEvaluator(gold)
    train = data[:n_train]
    dev = data[n_train : n_train + n_dev]
    test = data[n_train + n_dev : need]
    return train, dev, test, ev


# ---------------------------------------------------------------------------
# HoVer (multi-hop claim verification, closed-book pilot; exact-label accuracy)
# ---------------------------------------------------------------------------
def _hover_pred(response: str) -> str:
    """Parse SUPPORTED / NOT_SUPPORTED from a response (check NOT first)."""
    r = (response or "").upper()
    if "NOT_SUPPORTED" in r or "NOT SUPPORTED" in r or "NOT-SUPPORTED" in r or "NOT SUPPORT" in r:
        return "NOT_SUPPORTED"
    if "REFUT" in r or "FALSE" in r or "UNSUPPORTED" in r:
        return "NOT_SUPPORTED"
    if "SUPPORTED" in r or "SUPPORT" in r or "TRUE" in r:
        return "SUPPORTED"
    return "?"


class HoVerEvaluator:
    """Exact-label accuracy for closed-book claim verification. Score is 1.0 iff
    the parsed label matches gold. Feedback gives the correct label + format."""

    def __init__(self, gold_by_input: dict) -> None:
        self.gold = gold_by_input

    def __call__(self, data, response) -> EvaluationResult:
        gold = self.gold[data["input"]]
        pred = _hover_pred(response)
        ok = pred == gold
        fb = (
            "Correct."
            if ok
            else (
                f"Wrong. Correct label: {gold}. Reply with exactly one word — SUPPORTED or "
                "NOT_SUPPORTED — judging whether real-world facts support the claim."
            )
        )
        return EvaluationResult(score=1.0 if ok else 0.0, feedback=fb)


_HOVER_TRAIN_URL = (
    "https://raw.githubusercontent.com/hover-nlp/hover/main/data/hover/"
    "hover_train_release_v1.1.json"
)
_HOVER_LABELS = {"NOT_SUPPORTED": 0, "SUPPORTED": 1}


def _load_hover_train() -> list:
    """HoVer train split as a list of dicts matching the old HF schema.

    hover-nlp/hover ships only a loader script (hover.py) with no parquet branch, so
    datasets>=4 cannot load it by name. Fetch the official release JSON directly (the
    URL hover.py itself downloaded) and reproduce the script's field mapping — most
    importantly encoding the label ClassLabel (NOT_SUPPORTED->0, SUPPORTED->1) so the
    downstream ``str(e["label"]) == "1"`` balance check still works. Drop-in iterable
    replacement for the old ``load_dataset("hover-nlp/hover", split="train")``."""
    import json
    import urllib.request

    last: Exception | None = None
    raw = None
    for _ in range(3):
        try:
            with urllib.request.urlopen(_HOVER_TRAIN_URL, timeout=120) as r:
                raw = json.load(r)
            break
        except Exception as e:  # noqa: BLE001
            last = e
    if raw is None:
        raise RuntimeError(f"could not fetch HoVer train JSON from {_HOVER_TRAIN_URL}: {last}")
    return [
        {
            "uid": d["uid"],
            "claim": d["claim"],
            "supporting_facts": [{"key": x[0], "value": x[1]} for x in d["supporting_facts"]],
            "label": _HOVER_LABELS.get(d["label"], d["label"]),
            "num_hops": d["num_hops"],
            "hpqa_id": d["hpqa_id"],
        }
        for d in raw
    ]


def build_hover(n_train: int, n_dev: int, n_test: int, *, retrieval: bool = False, retrieval_k: int = 5):
    """Balanced HoVer with evidence: claim + context -> SUPPORTED/NOT_SUPPORTED.

    Closed-book HoVer is starved (the model can't verify multi-hop claims from
    parametric knowledge — base ~0.5). HoVer claims are built on HotpotQA, so we
    join each 2-hop claim to its HotpotQA distractor context (via hpqa_id) to
    supply evidence. Gold-context (default) hands all joined paragraphs; retrieval
    (``retrieval=True``) runs BM25 over the task-local corpus and keeps the top-k.
    Labels interleaved so every contiguous split is balanced (a trivial
    single-label prompt scores ~0.5)."""
    from datasets import load_dataset

    # hover-nlp/hover is a SCRIPT-ONLY HF dataset (repo has just hover.py; no data
    # files, no refs/convert/parquet branch), so datasets>=4 — which removed script
    # execution — cannot load it by name under any revision. Fetch the official
    # release JSON directly (the same URL the script used) instead.
    hover: Any = _load_hover_train()
    hqa: Any = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
    hqa_by_id = {e["id"]: e for e in hqa}
    need = n_train + n_dev + n_test
    ones: list = []
    zeros: list = []
    for e in hover:
        if str(e["num_hops"]) != "2" or e["hpqa_id"] not in hqa_by_id:
            continue
        (ones if str(e["label"]) == "1" else zeros).append(e)
        if len(ones) >= need and len(zeros) >= need:
            break
    picks: list = []
    for z, o in zip(zeros, ones, strict=False):
        picks.append(o)
        picks.append(z)
        if len(picks) >= need:
            break
    picks = picks[:need]
    if len(picks) < need:
        raise ValueError(f"HoVer yielded only {len(picks)} joinable balanced examples for the split.")
    index = None
    if retrieval:
        corpus: list = []
        for e in picks:
            corpus.extend(_passages_from_context(hqa_by_id[e["hpqa_id"]]["context"]))
        index = _BM25Index(corpus)

    gold: dict = {}
    data: list[dict] = []
    for e in picks:
        claim = e["claim"].strip()
        if retrieval:
            ctx = "\n".join(index.topk(claim, retrieval_k))
        else:
            ctx = _hotpot_context_str(hqa_by_id[e["hpqa_id"]]["context"])
        inp = f"Context:\n{ctx}\n\nClaim: {claim}\n\nIs the claim supported by the context?"
        gold[inp] = "SUPPORTED" if str(e["label"]) == "1" else "NOT_SUPPORTED"
        data.append({"input": inp, "additional_context": {}, "answer": gold[inp]})
    ev = HoVerEvaluator(gold)
    return data[:n_train], data[n_train : n_train + n_dev], data[n_train + n_dev : need], ev


# ---------------------------------------------------------------------------
# PUPA (privacy-conscious delegation; quality x privacy composite)
# ---------------------------------------------------------------------------
# PUPA (from PAPILLON) pairs a PII-bearing user query with a good reference
# response. The faithful metric balances RESPONSE QUALITY against PII LEAKAGE.
# Here (single-model prompt-opt pilot) we score:  quality x privacy, where
# quality is a 0-1 LLM judge vs the reference and privacy = 1 - fraction of the
# example's pii_units that appear verbatim in the response. The base "helpful"
# prompt leaks ~0.84 of PII, so the composite has large headroom via the privacy
# axis — and the two objectives are in tension (answering the task often needs
# the PII), which is exactly the PUPA challenge.


def _pupa_units(pii_units: str) -> list:
    raw = pii_units or ""
    return [u.strip() for u in raw.split("||") if u.strip() and not u.strip().startswith("presidio_")][:20]


class PUPAEvaluator:
    """quality (LLM judge, 0-1) x privacy (1 - verbatim PII-leakage rate)."""

    def __init__(self, meta_by_input: dict, judge_model: str) -> None:
        self.meta = meta_by_input  # input -> {"target":..., "units":[...]}
        self._judge_model = judge_model
        self._judge = None

    def _judge_lm(self):
        if self._judge is None:
            from gepa.lm import LM

            self._judge = LM(self._judge_model, temperature=0.0, max_tokens=8)
        return self._judge

    def __call__(self, data, response) -> EvaluationResult:
        import re

        m = self.meta[data["input"]]
        resp = response or ""
        # privacy: 1 - fraction of PII units leaked verbatim
        units = m["units"]
        if units:
            leaked = sum(1 for u in units if u.lower() in resp.lower())
            privacy = 1.0 - leaked / len(units)
        else:
            privacy = 1.0
        # quality: LLM judge 0-1 (empty response scores 0 without a judge call)
        if not resp.strip():
            quality = 0.0
        else:
            jp = [
                {
                    "role": "user",
                    "content": (
                        f"User request:\n{data['input'][:1800]}\n\nReference answer:\n{m['target'][:1200]}\n\n"
                        f"Candidate answer:\n{resp[:1200]}\n\nRate 0-100 how well the candidate satisfies the "
                        "request (vs the reference). Reply with ONLY an integer."
                    ),
                }
            ]
            import time

            quality = 0.0
            last_err = None
            for attempt in range(3):
                try:
                    jr = self._judge_lm()(jp)
                    mt = re.search(r"\d+", jr or "")
                    quality = min(100, int(mt.group())) / 100 if mt else 0.0
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    if attempt < 2:
                        time.sleep(2 * (attempt + 1))
            if last_err is not None:
                print(f"PUPA judge call failed after 3 attempts, scoring quality=0.0: {last_err}", flush=True)
        score = quality * privacy
        fb = (
            f"Quality={quality:.2f}, privacy={privacy:.2f} (leaked {1 - privacy:.0%} of personal identifiers). "
            "Answer the request helpfully BUT do not repeat personal identifiers verbatim (names of people, "
            "companies, places, emails, phone numbers) — paraphrase, generalize, or use placeholders."
        )
        return EvaluationResult(score=score, feedback=fb)


def build_pupa(n_train: int, n_dev: int, n_test: int, *, judge_model: str = "openrouter/qwen/qwen3-30b-a3b-instruct-2507"):
    """PUPA quality x privacy pilot. Deterministic slice by dataset order."""
    from datasets import load_dataset

    ds: Any = load_dataset("Columbia-NLP/PUPA", "pupa_new", split="train")
    need = n_train + n_dev + n_test
    meta: dict = {}
    data: list[dict] = []
    for e in ds:
        q = (e["user_query"] or "").strip()
        if not q or not (e.get("target_response") or "").strip():
            continue
        inp = q[:4000]
        if inp in meta:
            continue
        meta[inp] = {"target": e["target_response"].strip(), "units": _pupa_units(e.get("pii_units", ""))}
        data.append({"input": inp, "additional_context": {}, "answer": e["target_response"].strip()})
        if len(data) >= need:
            break
    if len(data) < need:
        raise ValueError(f"PUPA yielded only {len(data)} examples for the requested split.")
    ev = PUPAEvaluator(meta, judge_model)
    return data[:n_train], data[n_train : n_train + n_dev], data[n_train + n_dev : need], ev


def _strata_key(name: str, ex: dict, ev) -> Any:
    """Task-specific stratum for balanced per-seed sampling. Controls the exact
    variance sources the pilot flagged: HoVer label balance and PUPA PII-unit count
    (58% of PUPA items carry a single unit -> per-example privacy is binary, the
    driver of PUPA's seed-to-seed swing)."""
    if name == "hover":
        return str(ex.get("answer", "")).strip().upper()[:12]  # SUPPORTED / NOT_SUPPORTED
    if name == "pupa":
        try:
            return min(len(ev.meta[ex["input"]]["units"]), 3)  # 0,1,2,3+ PII units
        except Exception:
            return 0
    if name == "hotpotqa":
        return min(len((ex.get("answer") or "").split()), 3)  # answer length bucket
    return 0  # ifbench/toy: single stratum


def _stratified_split(pool: list, key_fn, sizes: tuple, seed: int) -> list:
    """Deterministically shuffle within strata (keyed by seed) and draw disjoint
    train/dev/test that preserve each stratum's proportion. Returns [train,dev,test]."""
    import random as _random

    rng = _random.Random(seed * 100003 + 17)
    strata: dict = {}
    for ex in pool:
        strata.setdefault(key_fn(ex), []).append(ex)
    for v in strata.values():
        rng.shuffle(v)
    # round-robin across strata (largest first) so proportions are preserved and
    # small strata are not starved; then carve disjoint splits from the ordering.
    order: list = []
    queues = sorted(strata.values(), key=len, reverse=True)
    while any(queues):
        for q in queues:
            if q:
                order.append(q.pop())
    n_train, n_dev, n_test = sizes
    need = n_train + n_dev + n_test
    if len(order) < need:
        raise ValueError(f"stratified pool has {len(order)} < requested {need}")
    return [order[:n_train], order[n_train:n_train + n_dev], order[n_train + n_dev:need]]


def build_dataset(name: str, n_train: int, n_dev: int, n_test: int, seed: int | None = None,
                  pool_mult: int = 5, retrieval: bool = False, retrieval_k: int = 5):
    """Fixed-slice (seed=None, the pilot's reproducible default) or per-seed
    stratified resample (seed set): load a larger pool and draw a seed-specific
    stratified split, so each seed is an independent data replicate. ``retrieval``
    switches HotpotQA/HoVer from gold-context to BM25 retrieve-then-read (the
    task-local corpus is built at pool scale, so retrieval is a real IR task)."""
    rkw = {"retrieval": retrieval, "retrieval_k": retrieval_k}
    if seed is None:
        if name == "toy":
            return build_toy(n_train, n_dev, n_test)
        if name == "ifbench":
            return build_ifbench(n_train, n_dev, n_test)
        if name == "hotpotqa":
            return build_hotpotqa(n_train, n_dev, n_test, **rkw)
        if name == "hover":
            return build_hover(n_train, n_dev, n_test, **rkw)
        if name == "pupa":
            return build_pupa(n_train, n_dev, n_test)
        raise ValueError(f"unknown dataset '{name}' (expected toy/ifbench/hotpotqa/hover/pupa)")
    # seeded path: build a pool ~pool_mult x the needed size, then strat-sample it.
    need = n_train + n_dev + n_test
    pool_total = pool_mult * need
    # split the pool load across the builder's 3 args (builders slice by order).
    pt = pool_total - 2 * (pool_total // 3)
    pd = pool_total // 3
    tr, dev, te, ev = build_dataset(name, pt, pd, pool_total // 3, seed=None,
                                    retrieval=retrieval, retrieval_k=retrieval_k)
    pool = list(tr) + list(dev) + list(te)
    train, d2, test = _stratified_split(pool, lambda e: _strata_key(name, e, ev), (n_train, n_dev, n_test), seed)
    return train, d2, test, ev
