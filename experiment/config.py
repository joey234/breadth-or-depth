# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Run configuration — the knobs a full run needs (a subset of the Stage-1
lock-list). Values here are DEFAULTS for a small real run; set them deliberately
before the registered full run. ROPE and the statistical model are NOT encoded
here — they are scientific judgments to be fixed in the registration, not code
defaults (see RUNBOOK.md)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunConfig:
    # --- models (lock-list #1) ---
    task_model: str = "openai/gpt-4o-mini"
    reflect_model: str = "openai/gpt-4o-mini"
    reflect_temperature: float = 1.0
    reflect_max_tokens: int = 8000  # room for a reasoning reflector's think+propose
    # reasoning_effort applies only to reasoning models (gpt-5.x); harmless
    # elsewhere because gepa.lm.LM sets drop_params=True.
    reflect_reasoning_effort: str | None = "low"
    # Qwen3.5/3.6 and gpt-5.x are reasoning-by-default: they spend the token budget
    # on hidden reasoning, so a small max_tokens returns EMPTY visible content.
    # `*_reasoning` toggles thinking on/off (the instruct-vs-thinking design axis)
    # via OpenRouter's unified `extra_body={"reasoning": {"enabled": ...}}`; task
    # max_tokens must be generous enough to clear reasoning + the answer.
    task_temperature: float = 0.0
    # instruct-answer budget. Set very generous (50000) so nothing truncates, even the
    # weak mid-tier model's long visible planning on IFBench's most pathological
    # constructive prompts ("keyword as the 34th word of the 23rd sentence") — the loop
    # diagnostic confirmed that output is genuine (unique, no reasoning-leak), just long,
    # and even 16000 clipped the worst cases. Instruct stops at the end of the answer, so a
    # high cap costs nothing on short tasks (JSON/code/parse); it only bills tokens actually
    # produced. Reasoning runs get a >=16000 floor via task_batch_kwargs below.
    task_max_tokens: int = 50000
    task_reasoning: bool = False  # False = instruct mode (fast/cheap); True = thinking
    # gpt-5.x task-model reasoning-effort override (e.g. "low" for gpt-5.6-terra). When
    # set it pins the effort regardless of the task_reasoning toggle; None keeps the
    # medium/minimal default. Ignored for non-OpenAI models (they use the extra_body toggle).
    task_reasoning_effort: str | None = None
    # OpenRouter routes each model through a default provider that is often the cheapest,
    # not the fastest. "throughput" tells OpenRouter to prefer the highest-throughput
    # backend (big speedup for the Qwen MoEs); "latency" optimizes first-token time; None
    # leaves OpenRouter's default routing. Applies to openrouter/* models only.
    openrouter_provider_sort: str | None = "throughput"
    # Reflector reasoning defaults OFF: the shakedown showed a reasoning-ON 397B
    # reflector's calls exceed the LM wall-clock and return EMPTY proposals. The
    # pilot's non-reasoning reflector worked well; keep that. (Enabling it needs a
    # much larger reflect timeout.)
    reflect_reasoning: bool = False

    # --- budget / schedule (lock-list #2) ---
    n_iters: int = 6  # proposal rounds per arm (the "K" of the latency claim)
    minibatch_size: int = 3

    # --- splits (lock-list #3) ---
    dataset: str = "ifbench"  # "ifbench" | "toy" | "hotpotqa"
    n_train: int = 20
    n_dev: int = 20  # the shared dev panel used for selection/tracking
    n_test: int = 20  # touched ONCE per arm, final artifact only

    # --- retrieval (HotpotQA/HoVer): gold-context (False) vs BM25 retrieve-then-
    # read over the task-local corpus (True). retrieval_k = passages kept. ---
    retrieval: bool = False
    retrieval_k: int = 5

    # --- data-quality gate: p0 dev-score floor below which a seed reads as
    # empty-contaminated. TASK-DEPENDENT: IFBench p0 ~0.85 so 0.15 works, but
    # HotpotQA p0 F1 is legitimately ~0.12, so set lower there. The empty-rate
    # signal is the task-agnostic primary; this floor is a backstop.
    dev_base_floor: float = 0.15

    # --- replication ---
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    # Per-seed stratified resampling: each seed draws its OWN stratified subset
    # (an independent data replicate, per Bouthillier et al.). Default off keeps
    # the pilot's fixed-slice behavior reproducible; registered-grade turns it on.
    per_seed_data: bool = False

    # --- arms ---
    arms: list[str] = field(default_factory=lambda: ["S2", "S3"])

    # Phase-2 unconditional validation: every proposed candidate is scored on the
    # dev panel and enters the pool (accept-all + select-all). Makes both arms
    # faithful and pairs the validation-side counters, so the strict matching gate
    # passes. Turn off to reproduce the earlier acceptance-gated proxy behavior.
    unconditional_validation: bool = True

    # Runtime-attached evaluator (set by the runner after building the dataset).
    evaluator: Any = None

    @staticmethod
    def _is_openai(model: str) -> bool:
        # OpenAI models called via the OpenAI API directly (litellm "openai/"
        # provider, OPENAI_API_KEY) rather than through OpenRouter.
        return (model or "").startswith("openai/")

    @staticmethod
    def _resolve_effort(flag: bool, effort: str | None) -> str | None:
        """Effective reasoning effort. An explicit ``*_reasoning_effort`` wins; else the
        ``*_reasoning`` bool maps to 'medium' (on) or None (off). None means reasoning off."""
        if effort:
            return effort
        return "medium" if flag else None

    def _reasoning_params(self, model: str, effort: str | None) -> dict:
        """Provider-correct reasoning control, effort-based for BOTH providers.
        OpenAI (gpt-5.x) uses ``reasoning_effort`` ('minimal' is its fast/off mode).
        OpenRouter (Qwen3 thinking) uses ``extra_body`` reasoning: an effort turns it ON
        at that level; None disables reasoning (instruct mode)."""
        if self._is_openai(model):
            return {"reasoning_effort": effort or "minimal"}
        reasoning = {"effort": effort} if effort else {"enabled": False}
        eb: dict = {"reasoning": reasoning}
        if self.openrouter_provider_sort:
            # Prefer a fast backend over OpenRouter's cheap default.
            eb["provider"] = {"sort": self.openrouter_provider_sort}
        return {"extra_body": eb}

    def reflect_kwargs(self) -> dict:
        # gpt-5.x (OpenAI API) uses max_completion_tokens (rejects max_tokens + temperature);
        # OpenRouter (Qwen) uses max_tokens + temperature.
        openai = self._is_openai(self.reflect_model)
        effort = self._resolve_effort(self.reflect_reasoning, self.reflect_reasoning_effort)
        if openai:
            kw: dict = {"max_completion_tokens": self.reflect_max_tokens}
        else:
            kw = {"max_tokens": self.reflect_max_tokens, "temperature": self.reflect_temperature}
        kw.update(self._reasoning_params(self.reflect_model, effort))
        return kw

    def task_batch_kwargs(self) -> dict:
        """Per-rollout kwargs for the TASK model (applied via batch_complete):
        token budget + reasoning-effort selection, provider-correct."""
        openai = self._is_openai(self.task_model)
        effort = self._resolve_effort(self.task_reasoning, self.task_reasoning_effort)
        # Reasoning tokens count against the budget. gpt-5.x ALWAYS reasons (minimal is
        # its floor), so give it room regardless. Qwen reasons only when an effort is set,
        # and its reasoning is verbose, so a Qwen reasoning run gets generous room while a
        # Qwen instruct run keeps the short budget (fast, no reasoning tokens).
        if openai:
            kw: dict = {"max_completion_tokens": max(self.task_max_tokens, 16000 if effort else 8000)}
        else:
            budget = max(self.task_max_tokens, 16000) if effort else self.task_max_tokens
            kw = {"max_tokens": budget, "temperature": self.task_temperature}
        kw.update(self._reasoning_params(self.task_model, effort))
        return kw

    def summary(self) -> dict:
        return {
            "task_model": self.task_model,
            "task_reasoning": self.task_reasoning,
            "task_reasoning_effort": self.task_reasoning_effort,
            "task_max_tokens": self.task_max_tokens,
            "reflect_model": self.reflect_model,
            "reflect_reasoning": self.reflect_reasoning,
            "reflect_kwargs": self.reflect_kwargs(),
            "n_iters": self.n_iters,
            "minibatch_size": self.minibatch_size,
            "dataset": self.dataset,
            "splits": {"train": self.n_train, "dev": self.n_dev, "test": self.n_test},
            "seeds": self.seeds,
            "arms": self.arms,
        }


def rough_cost_estimate(cfg: RunConfig) -> str:
    """Very rough per-study cost sketch (order of magnitude, not a quote)."""
    arms = len(cfg.arms)
    seeds = len(cfg.seeds)
    # rollouts per arm-run ~ iters*(2*minibatch) + accepted*dev + test
    per_arm_rollouts = cfg.n_iters * (2 * cfg.minibatch_size) + cfg.n_iters * cfg.n_dev + cfg.n_test
    total_rollouts = per_arm_rollouts * arms * seeds
    reflections = cfg.n_iters * arms * seeds
    return (
        f"~{total_rollouts} task rollouts + ~{reflections} reflections across "
        f"{arms} arms x {seeds} seeds. On a mini/4o-mini tier that is typically "
        f"a few US$; on a gpt-5.x reasoning tier expect ~10x. Confirm with a 1-seed dry run first."
    )
