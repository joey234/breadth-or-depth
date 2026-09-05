# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Model slate + run matrix for the registered-grade Breadth-or-Depth study.

Design (nested/L-shaped, NOT a full grid — the same philosophy as the four-task
vector): run the ANCHOR model on all four tasks for breadth, then vary one model
axis at a time on a single cheap task. The frontier model (gpt-5.6-sol) is
confined to its two decisive cells (top-of-ladder + strong-reflector stress),
because at $5/$30 per M it is ~25-40x the open Qwen prices.

Prices are $/M tokens as read from OpenRouter (2026-07); verify in-console before
locking the budget. `reasoning`/`max_tokens` matter because Qwen3.5/3.6 and
gpt-5.x are reasoning-by-default (small max_tokens -> empty visible content).
"""

from __future__ import annotations

# label -> (openrouter slug, price_in $/M, price_out $/M)
MODELS = {
    # open-weight Qwen (self-hostable; cheap; the bulk of the seeds)
    "qwen3.5-9b":     ("openrouter/qwen/qwen3.5-9b",       0.10, 0.15),
    "qwen3.5-27b":    ("openrouter/qwen/qwen3.5-27b",      0.195, 1.56),
    "qwen3.5-122b":   ("openrouter/qwen/qwen3.5-122b-a10b", 0.26, 2.08),
    "qwen3.5-397b":   ("openrouter/qwen/qwen3.5-397b-a17b", 0.385, 2.45),
    "qwen3.6-35b":    ("openrouter/qwen/qwen3.6-35b-a3b",  0.13, 1.00),  # former anchor (dropped: 3B-active looped on IFBench)
    "qwen3-30b":      ("openrouter/qwen/qwen3-30b-a3b",    0.12, 0.50),  # pilot anchor (older rung)
    # DeepSeek v4 (OpenRouter) — CURRENT mid-tier anchor pair. Flash = task (many
    # rollouts), Pro = reflector. Stronger instruction-following than the 3B-active
    # Qwen, so it should not loop to the token cap on IFBench's constructive prompts.
    # TODO CONFIRM: exact OpenRouter slugs + prices (placeholders below for cost sketch only).
    "deepseek-v4-flash": ("openrouter/deepseek/deepseek-v4-flash-0731", 0.28, 0.88),
    "deepseek-v4-pro":   ("openrouter/deepseek/deepseek-v4-pro-0813",   0.55, 2.19),
    # frontier (API-metered; confined to designated cells). Called via the OpenAI
    # API DIRECTLY (litellm "openai/" provider + OPENAI_API_KEY), not OpenRouter.
    "gpt-5.6-sol":    ("openai/gpt-5.6-sol", 5.00, 30.00),
    # cheaper frontier-tier variant used as the frontier REFLECTOR. Prices confirmed
    # $2/$12 per M (0.4x sol). TODO CONFIRM: exact slug (follows the sol pattern here).
    "gpt-5.6-terra":  ("openai/gpt-5.6-terra", 2.00, 12.00),
    # cheapest frontier-tier variant used as the frontier TASK model (many rollouts).
    # Price confirmed $0.2/$1.2 per M (0.1x terra). TODO CONFIRM: exact slug.
    "gpt-5.6-luna":   ("openai/gpt-5.6-luna", 0.20, 1.20),
}

# per-task quality-gate floor (low-baseline tasks need a lower p0 dev floor)
DEV_BASE_FLOOR = {"ifbench": 0.15, "hotpotqa": 0.05, "hover": 0.15, "pupa": 0.05}

# instruct answers are short; thinking mode needs room to reason THEN answer.
TASK_MAXTOK = {"instruct": 512, "think": 2048}

ANCHOR_TASK = "deepseek-v4-flash"    # current mid-tier task model (replaced qwen3.6-35b)
ANCHOR_REFLECT = "deepseek-v4-pro"   # current mid-tier reflector (replaced qwen3.5-397b)
FRONTIER_TASK = "gpt-5.6-luna"     # frontier TASK model (cheaper; runs the many rollouts)
FRONTIER_REFLECT = "gpt-5.6-terra" # stronger same-family reflector paired with it
STRONG_REFLECT = "gpt-5.6-sol"     # strongest reflector, used only in the reflector-stress cells
FRONTIER_EFFORT = "low"            # both frontier calls run at reasoning_effort=low
TASKS = ["ifbench", "hotpotqa", "hover", "pupa"]
LADDER_TASK = "hover"            # cheapest clean task for the single-axis sweeps


# HotpotQA/HoVer use BM25 retrieve-then-read; other tasks have no retrieval step.
RETRIEVAL_TASKS = {"hotpotqa", "hover"}
RETRIEVAL_K = 5

# Per-TASK-model concurrency cap. The shakedown showed the big MoE / frontier
# models rate-limit when many jobs fire batched rollouts at them at once (and the
# resulting empties degrade the shared provider window for other jobs too). The
# rate pressure is on the TASK model (many parallel rollout calls via
# batch_complete), NOT the reflector (a few sequential calls per iteration) — so
# 9B-task/397B-reflector cells are fine; only 397B-as-task needs the cap. Models
# not listed run at the global concurrency.
TASK_MODEL_MAX_CONCURRENCY = {
    "qwen3.5-397b": 2,   # 397B-A17B MoE as task model: batch rollouts rate-limit
    "gpt-5.6-sol": 2,    # frontier: rate- and cost-sensitive
    "gpt-5.6-terra": 2,  # frontier (reflector tier): same rate caution
    "gpt-5.6-luna": 2,   # frontier task model: rate- and cost-sensitive
}


def _cell(task, task_model, reflect_model, *, task_reasoning=False, reflect_reasoning=False, tag=""):
    mode = "think" if task_reasoning else "instruct"
    label = f"{task}__{task_model}__{mode}__refl-{reflect_model}{('__' + tag) if tag else ''}"
    return {
        "label": label.replace(".", "p"),
        "task": task,
        "task_model": MODELS[task_model][0],
        "task_model_label": task_model,
        "task_reasoning": task_reasoning,
        "task_max_tokens": TASK_MAXTOK[mode],
        "reflect_model": MODELS[reflect_model][0],
        "reflect_model_label": reflect_model,
        "reflect_reasoning": reflect_reasoning,
        "dev_base_floor": DEV_BASE_FLOOR.get(task, 0.15),
        "retrieval": task in RETRIEVAL_TASKS,
        "retrieval_k": RETRIEVAL_K,
        # None = run at the global concurrency; a number caps concurrent jobs that
        # share this task model (keyed by task_model_label in the orchestrator).
        "max_concurrency": TASK_MODEL_MAX_CONCURRENCY.get(task_model),
    }


def build_matrix() -> list[dict]:
    cells: list[dict] = []
    # 1) BREADTH: anchor task model x anchor reflector on all four tasks
    for t in TASKS:
        cells.append(_cell(t, ANCHOR_TASK, ANCHOR_REFLECT, tag="breadth"))
    # 2) CAPABILITY LADDER (single Qwen3.5 generation) x anchor reflector, one task
    for m in ["qwen3.5-9b", "qwen3.5-27b", "qwen3.5-122b", "qwen3.5-397b"]:
        cells.append(_cell(LADDER_TASK, m, ANCHOR_REFLECT, tag="ladder"))
    # 3) REASONING MODE: anchor task instruct vs thinking, one task
    cells.append(_cell(LADDER_TASK, ANCHOR_TASK, ANCHOR_REFLECT, task_reasoning=True, tag="thinkaxis"))
    # 4) REFLECTOR CONTRAST: anchor task x strongest frontier reflector, one task
    cells.append(_cell(LADDER_TASK, ANCHOR_TASK, STRONG_REFLECT, tag="reflcontrast"))
    # 5) STRESS CELL: weak task model x strong reflector ("if statefulness ever helps, here")
    cells.append(_cell(LADDER_TASK, "qwen3.5-9b", STRONG_REFLECT, tag="stress"))
    # 6) FRONTIER TIER (external validity), one task. Cheaper luna task + stronger terra
    #    reflector (same family), both at reasoning_effort=low.
    fcell = _cell(LADDER_TASK, FRONTIER_TASK, FRONTIER_REFLECT, tag="frontier-task")
    fcell["task_reasoning_effort"] = FRONTIER_EFFORT
    fcell["reflect_reasoning_effort"] = FRONTIER_EFFORT
    cells.append(fcell)
    # de-dup by label (ladder 397b == a breadth-style cell on the ladder task)
    seen, uniq = set(), []
    for c in cells:
        if c["label"] in seen:
            continue
        seen.add(c["label"])
        uniq.append(c)
    return uniq


if __name__ == "__main__":
    m = build_matrix()
    print(f"{len(m)} cells:")
    for c in m:
        print(f"  {c['label']}")
