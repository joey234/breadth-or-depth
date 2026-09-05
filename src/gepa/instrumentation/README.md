# Phase 1 — Per-run resource ledger

The resource ledger is not reporting; it *is* the experiment. Every claim about
S2/S3 "matching" reduces to these counters. This package emits one JSON ledger
per optimization run.

## Design: zero edits to the optimizer

Instrumentation is composed entirely at GEPA's dependency-injection seams, so
**no code in `core/`, `proposer/`, or `strategies/` is touched** and the search
is byte-for-byte identical whether or not a ledger is attached:

- `LedgerCallback` (a `GEPACallback`) tallies structural counters from GEPA's
  existing event stream — proposals, evaluations, valset reuse, lineage, merges.
- `LedgerAdapter` / `LedgerReflectionLM` wrap the injected adapter and reflection
  LM to count rollouts, reflector calls, tokens, cost, and dispatch widths. They
  forward every call verbatim.

## Usage

```python
from gepa.instrumentation import optimize_with_ledger

result, ledger = optimize_with_ledger(
    seed_candidate={"instruction": "..."},
    trainset=trainset, valset=valset,
    adapter=my_adapter,               # or task_lm="openai/gpt-4.1"
    reflection_lm="openai/gpt-4.1",
    max_metric_calls=200,
    seed=0,
    run_label="S3-controlled-gepa",
    ledger_dir="runs/s3/seed0",       # writes ledger.json + ledger.timing.json
)
```

To drive `gepa.optimize` yourself, use `attach_ledger(...)`, forward the returned
`adapter` / `reflection_lm` / `callbacks`, then call `flush()` and
`ledger.write(dir)` after the run.

## Counters (the Phase 1 table)

`ledger.json` → `counters`:

| Table row | Field |
|---|---|
| Candidate proposals | `candidate_proposals` (total / reflective / merge_attempted / accepted / rejected) |
| Task rollouts | `task_rollouts.total_executed` (actual, cache-miss aware) |
| Candidate-example evaluations | `candidate_example_evaluations` (total / minibatch / valset / seed requested) |
| Reflector calls / tokens / cost | `reflector.{calls,tokens_in,tokens_out,cost}` |
| Verifier calls / tokens / cost | `verifier.{calls,batches,tokens_in,tokens_out,cost}` |
| Distinct dev items touched (set) | `distinct_dev_items.ids` (sorted set) + `.count` |
| Validation reuse (per item) | `validation_reuse.per_item` + `max_reuse` |
| **Sequential critical-path depth** | `sequential_critical_path_depth.{best_candidate,max_over_all,depth_by_candidate}` |
| Wall-clock at fixed concurrency | `ledger.timing.json` (see below) |
| Peak parallelism | `peak_parallelism.{max_candidates_per_dispatch,max_rollouts_per_dispatch,max_reflection_prompts_per_call}` |

Also emitted for the Phase 3 matching assertions: `minibatch_schedule`
(item-level, ordered), `parent_log`, `merges`, and `selection` (best candidate,
final pool, `p0_in_pool`).

## Determinism contract (the exit test)

`ledger.json` contains only quantities reproducible under a fixed seed and a
deterministic model. **Two runs with the same seed produce a byte-identical
`ledger.json`** — verified by `tests/test_ledger_determinism.py`.

Wall-clock is inherently non-reproducible, so it is quarantined into a separate
`ledger.timing.json` that is *not* covered by the byte-identity guarantee. This
matches the design (§6): wall-clock at fixed concurrency is a measured latency
outcome, reported but never matched.

```
uv run pytest tests/test_ledger_determinism.py
```

## Notes for the arms (Phase 2+)

- `sequential_critical_path_depth` is where the latency claim lives: for
  root-reset every parent is p0, so `max_over_all == 1`; for stateful GEPA the
  depth grows with the lineage. This is the "K sequential proposal–evaluation
  decisions" quantity.
- `verifier` tokens/cost are read best-effort from the adapter's model (they are
  `0` for a purely programmatic verifier such as IFBench constraint checking).
- The ledger is a passive accumulator: attaching it never changes which
  candidates are proposed, selected, or accepted.
