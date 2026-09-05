# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Breadth-or-Depth mechanism: the controlled compounding task + the Depth-Gain metric.

Central idea. Depth (stateful search) can only beat breadth (parallel root-reset)
when the optimizer must fix things IN ORDER — when the error signal for the next
improvement is only visible AFTER the current one is fixed ("sequential
discoverability"). A capable reflector fixes everything it can SEE in one step, so
if all error signals are visible at once (ungated feedback), breadth suffices.

Controlled task (`build_compound`). Output must satisfy two features:
  F1 = ALL UPPERCASE,   F2 = words in REVERSE order.
Score = 0.5*F1 + 0.5*F2. A coupling knob `alpha` GATES the F2 error signal behind
F1: with probability alpha, an example reveals the F2 requirement in its feedback
ONLY if the response already satisfies F1. alpha=0 → both signals always visible
(breadth can fix both in one reflection). alpha=1 → F2 is only learnable after F1
is in the prompt (depth required). Sweeping alpha sweeps the depth advantage.

Depth-Gain metric (`depth_gain_probe`). A cheap 2-round parallel probe, NO full
optimization: round 1 draws K independent reflections from the seed p0 (this is
just a breadth probe); round 2 reflects K times from the best round-1 child.
  DepthGain = mean_improvement(round 2 | best child) / mean_improvement(round 1 | p0)
DepthGain ≈ 0 → the second step finds nothing new → breadth wins (predict A_t≈0).
DepthGain large → conditioning on a good parent unlocks new fixes → depth pays.
"""

from __future__ import annotations

import random
from typing import Literal

from gepa.adapters.default_adapter.default_adapter import EvaluationResult

_WORDS = ("cat dog fish bird tree lake wind gold rock star moon leaf sand rain "
          "fire snow hill road ship king").split()


def _target(tokens: list) -> str:
    return " ".join(w.upper() for w in reversed(tokens))


class CompoundEvaluator:
    """Score = 0.5*F1(uppercase) + 0.5*F2(reversed). Feedback gates the F2 signal
    behind F1 with probability `alpha` (the coupling knob)."""

    def __init__(self, alpha: float, seed: int = 0) -> None:
        self.alpha = alpha
        self._rng = random.Random(seed * 7919 + 1)
        self._gate: dict = {}  # input -> bool: is this example's F2 signal gated?

    def _gated(self, inp: str) -> bool:
        if inp not in self._gate:
            self._gate[inp] = self._rng.random() < self.alpha
        return self._gate[inp]

    def __call__(self, data, response) -> EvaluationResult:
        toks = data["_tokens"]
        resp = (response or "").strip()
        words = resp.split()
        letters = [c for c in resp if c.isalpha()]
        f1 = bool(letters) and all(c.isupper() for c in letters)          # uppercase
        f2 = [w.lower() for w in words] == [t.lower() for t in reversed(toks)]  # reversed
        score = 0.5 * f1 + 0.5 * f2
        fb = []
        if not f1:
            fb.append("The response must be ALL UPPERCASE.")
        # F2 signal is gated behind F1 for gated examples (only revealed once F1 holds)
        reveal_f2 = (not self._gated(data["input"])) or f1
        if reveal_f2 and not f2:
            fb.append("The words must be listed in REVERSE order.")
        if f1 and f2:
            fb.append("Correct.")
        return EvaluationResult(score=score, feedback=" ".join(fb) or "Improve the transformation.")


def build_compound(n_train: int, n_dev: int, n_test: int, *, alpha: float = 0.0, seed: int = 0):
    """Controlled compounding task with coupling knob `alpha`."""
    rng = random.Random(seed * 104729 + 3)
    need = n_train + n_dev + n_test
    data: list[dict] = []
    for _ in range(need):
        k = rng.randint(3, 5)
        toks = [rng.choice(_WORDS) for _ in range(k)]
        inp = "Transform: " + " ".join(toks)
        data.append({"input": inp, "_tokens": toks, "additional_context": {}, "answer": _target(toks)})
    ev = CompoundEvaluator(alpha, seed=seed)
    return data[:n_train], data[n_train:n_train + n_dev], data[n_train + n_dev:need], ev


# --------------------------------------------------------------------------
# D-chain: D features scored equally. gated: first unsatisfied hint only.
# checklist: names + k/D always, exact next hint still gated. ungated: all hints.
# --------------------------------------------------------------------------
def _core(resp: str) -> list:
    w = resp.split()
    if w and w[0].upper() == "BEGIN":
        w = w[1:]
    if w and w[-1].upper() == "END":
        w = w[:-1]
    return w


# Short names shown when checklist feedback is on. Exact hints stay gated.
_CHECKLIST_NAMES = {
    "uppercase": "case",
    "reverse": "order",
    "end": "suffix",
    "begin": "prefix",
}


def _chain_features(toks: list) -> list:
    """Ordered (name, check, hint). Each check takes the response string."""
    return [
        ("uppercase", lambda r: bool([c for c in r if c.isalpha()]) and all(c.isupper() for c in r if c.isalpha()),
         "The response must be ALL UPPERCASE."),
        ("reverse", lambda r: [x.lower() for x in _core(r)] == [t.lower() for t in reversed(toks)],
         "List the words in REVERSE order."),
        ("end", lambda r: bool(r.split()) and r.split()[-1].upper() == "END",
         "End the output with the word END."),
        ("begin", lambda r: bool(r.split()) and r.split()[0].upper() == "BEGIN",
         "Begin the output with the word BEGIN."),
    ]


def _chain_target(toks: list, d: int) -> str:
    words = list(reversed(toks)) if d >= 2 else list(toks)
    s = " ".join(w.upper() for w in words)
    if d >= 3:
        s = s + " END"
    if d >= 4:
        s = "BEGIN " + s
    return s


class ChainEvaluator:
    """D features. Score = (#passed)/D.

    gated (default): feedback is only the first unsatisfied hint.
    ungated: feedback lists every unsatisfied hint.
    checklist: names and k/D are always visible; the exact next hint is still
    first-unsatisfied only. Root-reset can guess later rules from the names.
    """

    def __init__(self, d: int, gated: bool = True, checklist: bool = False) -> None:
        self.d = d
        self.gated = gated
        self.checklist = checklist

    def __call__(self, data, response) -> EvaluationResult:
        toks = data["_tokens"]
        resp = (response or "").strip()
        feats = _chain_features(toks)[: self.d]
        passed = [name for name, chk, _ in feats if chk(resp)]
        score = len(passed) / self.d
        missing = [(name, hint) for name, chk, hint in feats if not chk(resp)]
        if not missing:
            fb = "Correct."
        elif self.checklist:
            names = ", ".join(_CHECKLIST_NAMES[name] for name, _, _ in feats)
            fb = (
                f"{self.d} requirements: {names}. Score {len(passed)}/{self.d}. "
                f"Exact next rule: {missing[0][1]}"
            )
        elif self.gated:
            fb = missing[0][1]
        else:
            fb = " ".join(hint for _, hint in missing)
        return EvaluationResult(score=score, feedback=fb)


def build_compound_chain(
    n_train: int,
    n_dev: int,
    n_test: int,
    *,
    d: int = 2,
    seed: int = 0,
    gated: bool = True,
    checklist: bool = False,
):
    """D-chain compounding task. gated=True / checklist=False is the stacked
    baseline. checklist=True keeps sequential exact hints but names every
    requirement. gated=False lists every exact hint (ungated)."""
    rng = random.Random(seed * 104729 + 3)
    need = n_train + n_dev + n_test
    data: list[dict] = []
    for _ in range(need):
        k = rng.randint(3, 5)
        toks = [rng.choice(_WORDS) for _ in range(k)]
        data.append({"input": "Transform: " + " ".join(toks), "_tokens": toks,
                     "additional_context": {}, "answer": _chain_target(toks, d)})
    ev = ChainEvaluator(d, gated=gated, checklist=checklist)
    return data[:n_train], data[n_train:n_train + n_dev], data[n_train + n_dev:need], ev


# --------------------------------------------------------------------------
# Information-matched chain: same D-chain rules, explicit TASK / INPUT WORDS
# surface. Existing build_compound_chain keeps the legacy Transform: input.
# --------------------------------------------------------------------------
EXPLICIT_OVERALL_TASK = (
    "Apply the transformation specified by the system prompt to the ordered list of "
    "input words. The transformation may govern letter case, word order, and required "
    "boundary markers. Return only the transformed word sequence."
)

_FULL_SPEC_EXACT = {
    1: "The target transformation uses uppercase letters.",
    2: "The target transformation uses uppercase letters and reverses the input words.",
    3: (
        "The target transformation uses uppercase letters, reverses the input words, "
        "and ends with END."
    ),
    4: (
        "The target transformation uses uppercase letters, reverses the input words, "
        "ends with END, and begins with BEGIN."
    ),
}


def chain_hints(d: int) -> list[tuple[str, str]]:
    """Stable (id, exact hint) pairs for the first ``d`` chain requirements."""
    return [(name, hint) for name, _, hint in _chain_features([])[:d]]


def passed_chain_prefix(data: dict, response: str, d: int) -> int:
    """Length of the consecutive satisfied prefix of the length-``d`` chain."""
    toks = data["_tokens"]
    resp = (response or "").strip()
    n = 0
    for _, chk, _ in _chain_features(toks)[:d]:
        if not chk(resp):
            break
        n += 1
    return n


def chain_feature_passes(data: dict, response: str, d: int) -> dict[str, bool]:
    """Per-requirement pass flags for the length-``d`` chain (not prefix-gated)."""
    toks = data["_tokens"]
    resp = (response or "").strip()
    return {name: bool(chk(resp)) for name, chk, _ in _chain_features(toks)[:d]}


def chain_task_description(
    d: int,
    mode: Literal["explicit-overall", "full-specification"],
) -> str:
    if mode == "explicit-overall":
        return EXPLICIT_OVERALL_TASK
    if mode == "full-specification":
        if d not in _FULL_SPEC_EXACT:
            raise ValueError(f"full-specification is defined for D=1..4, not D={d}")
        return f"{EXPLICIT_OVERALL_TASK} {_FULL_SPEC_EXACT[d]}"
    raise ValueError(f"unknown information mode {mode!r}")


def render_chain_input(words: list[str], task_description: str) -> str:
    """User-facing TASK / INPUT WORDS card. Never the legacy ``Transform:`` template."""
    return f"TASK\n{task_description}\n\nINPUT WORDS\n{' '.join(words)}"


def oracle_system_prompt(d: int) -> str:
    """Hand-written prompt that states every exact rule; used for check 1."""
    demo = {
        1: ("cat dog fish", "CAT DOG FISH"),
        2: ("cat dog fish", "FISH DOG CAT"),
        3: ("cat dog fish", "FISH DOG CAT END"),
        4: ("cat dog fish", "BEGIN FISH DOG CAT END"),
    }
    src, out = demo[d]
    lines = "\n".join(f"- {hint}" for _, hint in chain_hints(d))
    return (
        "You transform an ordered list of input words. Apply every rule below.\n"
        f"{lines}\n"
        f"Worked example: input words `{src}` become `{out}`.\n"
        "Return only the transformed word sequence."
    )


def build_information_matched_chain(
    n_train: int,
    n_sel: int,
    n_val: int,
    n_test: int,
    *,
    d: int = 2,
    seed: int = 0,
    information_mode: Literal["explicit-overall", "full-specification"] = "explicit-overall",
):
    """Same D-chain labels as ``build_compound_chain``, explicit task surface.

    Returns train / selection / validation / test plus evaluator and the
    exact task-description string used in every example.
    """
    task_description = chain_task_description(d, information_mode)
    rng = random.Random(seed * 104729 + 3)
    need = n_train + n_sel + n_val + n_test
    data: list[dict] = []
    for _ in range(need):
        k = rng.randint(3, 5)
        toks = rng.sample(_WORDS, k)
        inp = render_chain_input(toks, task_description)
        if inp.startswith("Transform:") or "\nTransform:" in inp:
            raise RuntimeError("legacy Transform: surface leaked into information-matched inputs")
        data.append({
            "input": inp,
            "_tokens": toks,
            "additional_context": {},
            "answer": _chain_target(toks, d),
            "_task_description": task_description,
        })
    i = n_train
    train = data[:i]
    sel = data[i:i + n_sel]
    i += n_sel
    val = data[i:i + n_val]
    i += n_val
    test = data[i:i + n_test]
    ev = ChainEvaluator(d, gated=True, checklist=False)
    return train, sel, val, test, ev, task_description


# --------------------------------------------------------------------------
# Depth-Gain probe
# --------------------------------------------------------------------------
SEED_PROMPT = "You are a helpful assistant. Answer the question."
COMPONENT = "system_prompt"


def _score(adapter, prompt: str, items: list) -> float:
    eb = adapter.evaluate(items, {COMPONENT: prompt}, capture_traces=False)
    return sum(eb.scores) / len(eb.scores) if eb.scores else 0.0


def _reflect(reflector, prompt: str, adapter, minibatch: list, rng: random.Random) -> str:
    """One GEPA-style feedback-aware reflection: show current prompt + a few
    (input, output, feedback) triples, ask for an improved system prompt."""
    eb = adapter.evaluate(minibatch, {COMPONENT: prompt}, capture_traces=True)
    lines = []
    for tr in (eb.trajectories or [])[:4]:
        out = str(tr.get("full_assistant_response", ""))[:120]
        fb = str(tr.get("feedback", ""))[:160]
        lines.append(f"- input: {tr['data']['input']}\n  output: {out}\n  feedback: {fb}")
    ex = "\n".join(lines)
    rp = (
        "You are improving a SYSTEM PROMPT for a transformation task. Below is the current "
        "system prompt and how it did on a few examples (with feedback). Write ONE improved "
        "system prompt that fixes the problems shown. Output ONLY the new system prompt.\n\n"
        f"CURRENT SYSTEM PROMPT:\n{prompt}\n\nEXAMPLES:\n{ex}\n\nIMPROVED SYSTEM PROMPT:"
    )
    return (reflector(rp) or "").strip() or prompt


def depth_gain_probe(task_adapter, reflector, train: list, dev: list, *, k: int = 4, seed: int = 0) -> dict:
    """Cheap 2-round parallel probe. Returns DepthGain + the breadth/depth ceilings."""
    rng = random.Random(seed)
    s0 = _score(task_adapter, SEED_PROMPT, dev)
    # round 1: k independent reflections from p0 (breadth)
    r1 = []
    for _ in range(k):
        mb = rng.sample(train, min(4, len(train)))
        cand = _reflect(reflector, SEED_PROMPT, task_adapter, mb, rng)
        r1.append((cand, _score(task_adapter, cand, dev)))
    best1, best1_score = max(r1, key=lambda x: x[1])
    mean_imp1 = max(1e-9, sum(max(0.0, s - s0) for _, s in r1) / len(r1))
    # round 2: k reflections from the best round-1 child (depth)
    r2 = []
    for _ in range(k):
        mb = rng.sample(train, min(4, len(train)))
        cand = _reflect(reflector, best1, task_adapter, mb, rng)
        r2.append((cand, _score(task_adapter, cand, dev)))
    best2_score = max(s for _, s in r2)
    mean_imp2 = sum(max(0.0, s - best1_score) for _, s in r2) / len(r2)
    depth_gain = mean_imp2 / mean_imp1
    # Headroom-normalized Depth-Gain (ROBUST, bounded, no ratio blowup): the
    # fraction of the headroom breadth LEFT BEHIND that a depth step then closes.
    #   dg_norm = (depth_ceiling - breadth_ceiling) / (1 - breadth_ceiling)
    # ->  ~0 when breadth already captured the reachable ceiling (depth adds nothing)
    # ->  ~1 when depth closes all the headroom breadth couldn't reach
    # Never explodes: the denominator is real headroom, and when breadth ~= 1.0 the
    # numerator ~= 0 too. Unlike the ratio, insensitive to a near-zero round-1 mean.
    # Incumbent-inclusive so dg_norm is bounded in [0,1] and never negative: a real
    # depth arm can always keep the round-1 winner, so D = max(B, depth child). And
    # dg_norm is undefined at B = 1 (no headroom) -> define it 0 there.
    B = best1_score
    Dc = max(B, best2_score)
    dg_norm = 0.0 if B >= 1.0 else (Dc - B) / (1.0 - B)
    return {
        "seed_score": round(s0, 3),
        "breadth_ceiling": round(best1_score, 3),      # best of round-1 (depth-1 from p0)
        "depth_ceiling": round(best2_score, 3),        # best of round-2 (depth-2 chain)
        "depth_over_breadth": round(best2_score - best1_score, 3),  # proxy for A_t
        "mean_imp_round1": round(mean_imp1, 3),
        "mean_imp_round2": round(mean_imp2, 3),
        "depth_gain": round(depth_gain, 3),            # legacy ratio (shape diagnostic; fragile at high base)
        "dg_norm": round(dg_norm, 3),                  # ROBUST headroom-normalized Depth-Gain (paper headline)
    }


def matched_depth_gain_probe(task_adapter, reflector, train: list, dev: list, *, k: int = 8, seed: int = 0) -> dict:
    """Budget-matched Depth-Gain probe (the §4.2 method; supersedes the unmatched
    ``depth_gain_probe`` for headline results).

    Three DISJOINT data roles (reviewer blocker #3):
      * feedback   — ``train`` minibatches shown to the reflector;
      * selection  — first half of ``dev``, used ONLY to pick the round-1 parent;
      * evaluation — second half of ``dev``, used ONLY to compare ceilings.
    Pass a ``dev`` large enough that each half has >= 20 examples (binary scoring
    needs >= 20 eval for the 0.05 floor to be meaningful).

    Both arms get 2K proposals and share the K round-1 root children; they differ only
    in the second K: the root arm draws K MORE from the seed, the lineage arm draws K
    from the round-1 winner. Each arm's ceiling is incumbent-inclusive over ONLY the
    candidates that arm actually had (seed + shared round-1 + its own round-2), so the
    root arm's extra children never leak into the lineage ceiling. The reported
    advantage is SIGNED (negative when root genuinely wins); ``dg_norm`` clamps the
    negative side and is 0 when there is no headroom (B = 1)."""
    rng = random.Random(seed)
    half = len(dev) // 2
    sel, eva = dev[:half], dev[half:]
    seed_e = _score(task_adapter, SEED_PROMPT, eva)
    # round 1: K independent root children from the seed (shared by both arms)
    r1 = []
    for _ in range(k):
        mb = rng.sample(train, min(4, len(train)))
        c = _reflect(reflector, SEED_PROMPT, task_adapter, mb, rng)
        r1.append({"p": c, "sel": _score(task_adapter, c, sel), "eva": _score(task_adapter, c, eva)})
    winner = max(r1, key=lambda x: x["sel"])  # chosen on the SELECTION half only
    # round 2 breadth: K more root children from the seed
    b2 = []
    for _ in range(k):
        mb = rng.sample(train, min(4, len(train)))
        c = _reflect(reflector, SEED_PROMPT, task_adapter, mb, rng)
        b2.append(_score(task_adapter, c, eva))
    # round 2 lineage: K children from the round-1 winner
    l2 = []
    for _ in range(k):
        mb = rng.sample(train, min(4, len(train)))
        c = _reflect(reflector, winner["p"], task_adapter, mb, rng)
        l2.append(_score(task_adapter, c, eva))
    r1_eva = [x["eva"] for x in r1]
    B = max([seed_e] + r1_eva + b2)      # root arm: seed + round-1 + K extra root
    L = max([seed_e] + r1_eva + l2)      # lineage arm: seed + round-1 + K lineage
    delta = L - B                        # SIGNED matched advantage
    dg_norm = 0.0 if B >= 1.0 else max(0.0, delta) / (1.0 - B)
    # unmatched contrast (the legacy "second-round lift"): round-1 best vs winner's lineage best
    old_breadth = max([seed_e] + r1_eva)
    old_depth = max([winner["eva"]] + l2)
    return {
        "seed_score": round(seed_e, 3),
        "breadth_ceiling": round(B, 3),
        "depth_ceiling": round(L, 3),
        "depth_over_breadth": round(delta, 3),        # SIGNED, budget-matched (paper headline)
        "dg_norm": round(dg_norm, 3),
        "unmatched_dob": round(old_depth - old_breadth, 3),  # legacy second-round lift, for contrast
        "winner_sel": round(winner["sel"], 3),
        "winner_eval": round(winner["eva"], 3),
        "n_sel": len(sel),
        "n_eva": len(eva),
        "k": k,
    }


def matched_depth_gain_probe_multi(task_adapter, reflector, train: list, dev: list, *, k: int = 8,
                                   rounds: int = 2, seed: int = 0) -> dict:
    """Budget-matched Depth-Gain probe extended to R generations (rounds >= 2), to test whether a
    deep task the 2-round probe under-calls is recovered when the probe is allowed to climb further.

    Both arms get R*K proposals. Breadth = R*K independent root children from the seed. Depth = K
    round-1 children plus (R-1)*K children obtained by climbing the lineage: at each generation the
    parent for the next batch is that generation's best child on the SELECTION half. Ceilings are
    incumbent-inclusive over each arm's own candidates on the EVALUATION half. At rounds=2 this reduces
    exactly to ``matched_depth_gain_probe``. Records the depth ceiling after each generation so the
    climb is visible."""
    rng = random.Random(seed)
    half = len(dev) // 2
    sel, eva = dev[:half], dev[half:]
    seed_e = _score(task_adapter, SEED_PROMPT, eva)
    # round 1 (shared): K root children from the seed
    r1 = []
    for _ in range(k):
        mb = rng.sample(train, min(4, len(train)))
        c = _reflect(reflector, SEED_PROMPT, task_adapter, mb, rng)
        r1.append({"p": c, "sel": _score(task_adapter, c, sel), "eva": _score(task_adapter, c, eva)})
    r1_eva = [x["eva"] for x in r1]
    breadth_pool = [seed_e] + r1_eva
    depth_pool = [seed_e] + r1_eva
    lineage_parent = max(r1, key=lambda x: x["sel"])["p"]
    depth_ceiling_by_gen = [round(max(depth_pool), 3)]  # after generation 1
    for _g in range(2, rounds + 1):
        # breadth arm: K more root children
        for _ in range(k):
            mb = rng.sample(train, min(4, len(train)))
            c = _reflect(reflector, SEED_PROMPT, task_adapter, mb, rng)
            breadth_pool.append(_score(task_adapter, c, eva))
        # depth arm: K children from the current lineage parent; climb via selection half
        gen = []
        for _ in range(k):
            mb = rng.sample(train, min(4, len(train)))
            c = _reflect(reflector, lineage_parent, task_adapter, mb, rng)
            gen.append({"p": c, "sel": _score(task_adapter, c, sel), "eva": _score(task_adapter, c, eva)})
        depth_pool.extend(x["eva"] for x in gen)
        lineage_parent = max(gen, key=lambda x: x["sel"])["p"]  # climb
        depth_ceiling_by_gen.append(round(max(depth_pool), 3))
    B = max(breadth_pool)
    L = max(depth_pool)
    delta = L - B
    dg_norm = 0.0 if B >= 1.0 else max(0.0, delta) / (1.0 - B)
    return {
        "seed_score": round(seed_e, 3),
        "rounds": rounds,
        "breadth_ceiling": round(B, 3),
        "depth_ceiling": round(L, 3),
        "depth_over_breadth": round(delta, 3),
        "dg_norm": round(dg_norm, 3),
        "depth_ceiling_by_gen": depth_ceiling_by_gen,
        "n_sel": len(sel),
        "n_eva": len(eva),
        "k": k,
    }


def _main() -> None:
    """Reproduce the Depth-Gain probe. Examples:
        python -m experiment.compounding --compound-alpha 1.0   # depth-required
        python -m experiment.compounding --compound-alpha 0.0   # breadth-sufficient
        python -m experiment.compounding --dataset hotpotqa     # real-task anchor
    """
    import argparse
    import os

    from gepa.adapters.default_adapter.default_adapter import DefaultAdapter
    from gepa.lm import LM

    ap = argparse.ArgumentParser(description="Depth-Gain probe")
    ap.add_argument("--dataset", default="compound", help="'compound' or a build_dataset name")
    ap.add_argument("--compound-alpha", type=float, default=1.0)
    ap.add_argument("--task-model", default="openrouter/qwen/qwen3.6-35b-a3b")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--n", type=int, default=12, help="train/dev/test size each")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--env", default=".env")
    args = ap.parse_args()

    if os.path.exists(args.env):
        for line in open(args.env):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

    task_kw = {"temperature": 0.0, "max_tokens": 256, "extra_body": {"reasoning": {"enabled": False}}}
    reflector = LM(args.task_model, temperature=1.0, max_tokens=1200, extra_body={"reasoning": {"enabled": False}})
    if args.dataset == "compound":
        tr, dev, _te, ev = build_compound(args.n, args.n, args.n, alpha=args.compound_alpha, seed=args.seed)
        label = f"compound(alpha={args.compound_alpha})"
    else:
        from experiment.common import build_dataset

        tr, dev, _te, ev = build_dataset(args.dataset, args.n, args.n, args.n)
        label = args.dataset
    adapter = DefaultAdapter(model=args.task_model, evaluator=ev, litellm_batch_completion_kwargs=task_kw)
    result = depth_gain_probe(adapter, reflector, tr, dev, k=args.k, seed=args.seed)
    print(f"{label}: {result}")


if __name__ == "__main__":
    _main()
