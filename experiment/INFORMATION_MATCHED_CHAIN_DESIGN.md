# Information-Matched Chain Experiment

## 1. Purpose

The current gated chain compares two end-to-end search policies:

- root-reset repeatedly revises the initial prompt and therefore repeatedly sees the first unmet requirement;
- lineage revises a descendant and may reveal later requirements.

That comparison deliberately bundles two possible advantages of lineage:

1. **Feedback-history advantage:** a descendant may reveal requirements that are unavailable when the initial prompt is evaluated.
2. **Prompt-state advantage:** a descendant retains edits made in earlier rounds.

The proposed experiment separates these effects by giving a root-reset arm the same cumulative requirement transcript as a lineage arm.

This is a controlled causal diagnostic. The transcript-enabled root arm is not intended as a deployable optimizer because it receives information from an oracle or reference trajectory.

### Replacement status in the paper

This experiment is intended to **replace the current Section 5.2 synthetic-task experiment**, not to appear as an additional experiment beside it. In the revised paper:

- replace the existing gate-toggle and length-`D` main-text results with the three-arm information-matched experiment specified here;
- retain `D = 1, 2, 3, 4`, but rerun those cells using the explicit task surface and the new arms;
- use the full-specification condition as the controlled counterpart to the current gate-off comparison;
- do not pool or directly compare the new results with runs that used the old `Transform: ...` surface; and
- if the old results are retained at all, label them as a preliminary pilot in the appendix rather than evidence for the main mechanism claim.

The new experiment should therefore provide the synthetic evidence used by the abstract, Section 5.2, discussion, and conclusion.

### Why the current Section 5.2 design should be replaced

The current experiment is useful as a pilot, but it does not cleanly identify why lineage helps. It also uses an underspecified task surface that can make failure reflect task interpretation rather than prompt-search structure.

| Current limitation | Change in the replacement | Improvement |
|---|---|---|
| Root-reset sees only the first hint, while lineage may see later hints and retain earlier edits. | Add `root_history`, which resets to `p0` but receives the same cumulative transcript as lineage. | Separates access to unlocked information from inheritance of the edited prompt. |
| The input says only `Transform: word1 word2 ...`. | Give every arm the explicit overall task description in Section 4. | Reduces irrelevant interpretations such as creative rewriting or arbitrary formatting. |
| A failed run may mean that the model did not understand the task family. | Name the possible constraint dimensions for every arm, while controlling only their exact values. | Makes failure more diagnostic of mutation, selection, or information access. |
| The gate-on versus gate-off toggle changes information availability but does not decompose the mechanism. | Estimate `root_history - root_local` and `lineage_history - root_history`. | Produces separate, directly interpretable feedback-history and prompt-state effects. |
| The original chain has limited falsification controls. | Require the `D=1` null, transcript-identity checks, leakage checks, and a full-specification control. | Makes implementation errors and alternative explanations easier to detect. |

This improves **internal validity**: the study can say whether lineage helps because it exposes otherwise unavailable feedback, because descendants retain useful edits, or because both mechanisms operate. It does not by itself improve external validity or prove that the same decomposition holds on real tasks; the synthetic experiment remains a controlled mechanism study.

### How Section 5.2 should report the replacement

The revised Section 5.2 should contain:

1. the explicit task description and staged requirements;
2. the three-arm intervention table;
3. the two primary contrasts, `root_history - root_local` and `lineage_history - root_history`;
4. results across `D = 1, 2, 3, 4` under the explicit-overall condition;
5. the full-specification control, reported separately; and
6. a short statement that the oracle-transcript arm is a causal diagnostic rather than a proposed optimizer.

The main conclusion should follow the observed decomposition:

- if `root_history` closes the gap to lineage, attribute the benefit mainly to carrying forward newly unlocked information;
- if `lineage_history` still outperforms `root_history`, report an additional benefit from retaining the edited prompt state; or
- if neither contrast is stable, treat the original Section 5.2 pattern as inconclusive rather than preserving its earlier mechanism claim.

## 2. Primary questions

1. How much of the lineage advantage is explained by access to newly revealed feedback?
2. After feedback information is equalized, does retaining the edited parent prompt still help?
3. Does either effect remain when the complete task specification is visible from the beginning?

## 3. Experimental conditions

Use three primary arms. All arms receive the same initial prompt, task model, reflection model, data split, minibatch schedule, proposal count, decoding settings, and selection split.

All arms also receive the same **explicit overall task description**. The task is never presented only as `Transform: word1 word2 ...`. What varies across arms is access to the exact requirement values, not whether the model understands the kind of task it is performing.

| Arm | Parent at round `t` | Requirement transcript at round `t` | Interpretation |
|---|---|---|---|
| `root_local` | Initial prompt `p0` | First requirement only, `H1` | Existing root-reset lower bound |
| `root_history` | Initial prompt `p0` | Cumulative transcript `Ht` | Root-reset with information equalized |
| `lineage_history` | Selected descendant `p[t-1]` | The same cumulative transcript `Ht` | Information-matched lineage |

The two history arms must receive byte-identical `Ht` at every round.

### Primary estimands

Let `Y_arm` be the held-out score of the final prompt selected for an arm.

```text
feedback_history_effect = Y_root_history - Y_root_local
prompt_state_effect      = Y_lineage_history - Y_root_history
total_matched_effect     = Y_lineage_history - Y_root_local
```

Interpretation:

- A large `feedback_history_effect` and small `prompt_state_effect` means that lineage mainly helps by discovering information.
- A small `feedback_history_effect` and large `prompt_state_effect` means that carrying the edited prompt matters even when information is matched.
- Both large means both mechanisms matter.
- Both near zero means the original lineage advantage was not stable under the controlled design.

These are an ordered decomposition for this experiment. The `prompt_state_effect` includes consequences of the inherited prompt, such as producing better current outputs; it should not be described as a context-free causal effect of prompt text.

## 4. Fixed task surface

Every task-model call must use an explicit, fixed user-facing description such as:

```text
TASK
Apply the transformation specified by the system prompt to the ordered list of
input words. The transformation may govern letter case, word order, and required
boundary markers. Return only the transformed word sequence.

INPUT WORDS
cat dog fish
```

This replaces the ambiguous surface:

```text
Transform: cat dog fish
```

The explicit description tells the model:

- what object it is transforming;
- which broad dimensions the specification may constrain; and
- the required output format.

It intentionally does **not** disclose the concrete values of the hidden requirements—for example, whether case should be upper- or lowercase, which ordering is required, or which boundary tokens to use. Those concrete values are the information controlled by the transcript.

Use exactly the same task-description string in every arm and round. Include it in the reflection context as well as in task-model evaluation examples. Only the input words should vary between examples.

The information boundary is therefore:

| Always visible to every arm | Controlled by the experiment |
|---|---|
| Overall transformation goal | Exact case value |
| Possible constraint dimensions | Exact ordering rule |
| Output-only instruction | Exact prefix and suffix tokens |
| Current input words | Cumulative requirement transcript |

The dataset, filenames, example identifiers, and native evaluator messages must not encode any controlled value. The canonical transcript is the only channel through which an exact requirement may be revealed to the reflector.

Naming the dimensions is meant to stop creative-rewrite failures, not to disclose the rules. If `root_local` routinely implements an unrevealed exact value, for example reverse order while its transcript contains only the uppercase hint, the TASK card leaked and the primary condition is invalid. Check 10 in Section 13 is the test. Do not fix a leak by reverting to the bare `Transform: ...` surface.

## 5. Feedback transcript

Represent a transcript as an ordered list of stable requirement identifiers:

```python
HINTS = [
    ("uppercase", "The response must be ALL UPPERCASE."),
    ("reverse", "List the words in REVERSE order."),
    ("end", "End the output with the word END."),
    ("begin", "Begin the output with the word BEGIN."),
]
```

Render it once as a global block in the reflection prompt:

```text
REQUIREMENTS DISCOVERED SO FAR:
1. The response must be ALL UPPERCASE.
2. List the words in REVERSE order.
```

Do not repeat the transcript as per-example feedback. Per-example examples may contain the input and generated output, but native evaluator feedback must be removed or replaced so it cannot leak a requirement beyond `Ht`.

## 6. Transcript schedules

### 6.1 Primary: oracle schedule

Use a fixed schedule independent of all arms:

```python
H_t = HINTS[:min(t, D)]  # rounds are 1-indexed
```

For `D=4`, round 1 exposes uppercase; round 2 adds reverse; round 3 adds suffix; round 4 adds prefix.

Advantages:

- history arms receive exactly the same information;
- the schedule is identical across replicates and cannot depend on one arm succeeding;
- failures to climb measure prompt revision and selection rather than failure to expose the next hint.

This schedule simulates information that a successful lineage could unlock. It does not claim that every real lineage would unlock one requirement per round.

### 6.2 Sensitivity: natural-unlock schedule

After the oracle-schedule experiment works, add a secondary schedule driven by the selected lineage parent:

1. Begin with `H1`.
2. Evaluate the selected lineage parent on the selection split.
3. A requirement is considered cleared when at least 90% of selection examples satisfy it.
4. If all currently revealed requirements are cleared, append the next requirement.
5. Give the resulting transcript to both history arms in the next round.

The reference lineage determines the transcript schedule; `root_history` is not allowed to unlock hints for itself. This is intentional because the arm is an information-matched diagnostic.

Report oracle and natural-unlock schedules separately. Use the oracle schedule for the primary decomposition.

## 7. Round protocol

Use `R = D` rounds and `K` proposals per arm per round. A reasonable initial value is `K=8`.

For each replicate:

1. Construct the data split and fixed reflection minibatch schedule.
2. Initialize:
   - `p0` to the existing seed prompt;
   - lineage parent to `p0`;
   - a candidate archive for each arm containing `p0`;
   - the transcript schedule.
3. For each round `t = 1..R`:
   - choose `H1` for `root_local`;
   - choose `Ht` for `root_history` and `lineage_history`;
   - generate `K` independent proposals for every arm;
   - score all proposals on the same selection split;
   - add proposals to that arm's archive;
   - set the next lineage parent to the best of the previous lineage parent and its `K` new children on the selection split.
4. Select each arm's final prompt from its complete archive using validation score.
5. Evaluate each selected prompt once on the held-out test set.

Pseudocode:

```python
archives = {arm: [p0] for arm in ARMS}
lineage_parent = p0

for t in range(1, D + 1):
    Ht = oracle_transcript(t, D)
    parents = {
        "root_local": p0,
        "root_history": p0,
        "lineage_history": lineage_parent,
    }
    histories = {
        "root_local": oracle_transcript(1, D),
        "root_history": Ht,
        "lineage_history": Ht,
    }

    generation = {}
    for arm in ARMS:
        generation[arm] = [
            reflect_with_transcript(
                parent=parents[arm],
                transcript=histories[arm],
                task_description=TASK_DESCRIPTION,
                minibatch=fixed_minibatch[t],
            )
            for _ in range(K)
        ]
        archives[arm].extend(generation[arm])

    lineage_parent = argmax_selection_score(
        [lineage_parent] + generation["lineage_history"]
    )

final_prompt[arm] = argmax_validation_score(archives[arm])
test_score[arm] = evaluate_once(final_prompt[arm], test)
```

## 8. Reflection-context construction

Do not use the evaluator's native `feedback` field unchanged, because it may reveal hints outside the assigned transcript.

Implement a dedicated helper:

```python
def reflect_with_transcript(
    parent: str,
    transcript: list[Requirement],
    task_description: str,
    adapter,
    minibatch: list[dict],
    reflector,
) -> str:
    # 1. Run parent on minibatch and collect inputs/outputs.
    # 2. Discard native evaluator feedback.
    # 3. Render the canonical transcript once.
    # 4. Ask the reflector for one revised system prompt.
```

Suggested reflection template:

```text
You are improving a system prompt for a transformation task.

OVERALL TASK:
{task_description}

CURRENT SYSTEM PROMPT:
{parent}

REQUIREMENTS DISCOVERED SO FAR:
{canonical_transcript}

EXAMPLE INPUTS AND CURRENT OUTPUTS:
{examples_without_native_feedback}

Write one replacement system prompt that incorporates the known requirements.
Output only the replacement prompt.
```

The same template and explicit task description must be used by all three arms. Only `parent` and the assigned transcript may differ.

## 9. Task-information controls

The explicit overall description in Section 4 is mandatory in every primary run. Do not use the old `Transform: ...` surface in the confirmatory experiment.

Run the following two information conditions:

### 9.1 Explicit-overall condition (primary)

Use the fixed task description from Section 4. It names the transformation problem and the dimensions that may be constrained, while the transcript progressively supplies the exact requirement values.

This is the primary condition for estimating the feedback-history and prompt-state effects. The root is no longer asked to infer the nature of the task from a bare word list.

### 9.2 Full-specification condition (control)

Append every exact requirement to the overall task description from round 1:

```text
The target transformation uses uppercase letters, reverses the input words,
ends with END, and begins with BEGIN.
```

In this condition, all arms always receive `H_D`; `root_local` and `root_history` therefore collapse into the same condition. The comparison between root and lineage tests prompt-state inheritance without an information-access advantage.

If the initial prompt already solves the full condition, increase task difficulty using additional compositional but explicit rules rather than hiding the rules again.

### 9.3 Legacy opaque surface (optional ablation only)

The old `Transform: word1 word2 ...` surface may be retained only as a clearly labeled supplementary ablation for comparison with earlier runs. Do not mix it into the primary estimates, and do not use it to support the main mechanism claim.

## 10. Experimental grid

Paper-minimum experiment, sufficient to replace Section 5.2:

```text
D:                 1, 2, 3, 4
task information:  explicit-overall and full-specification
schedule:          oracle
arms:              root_local, root_history, lineage_history
K:                 8 proposals per round
replicates:        at least 20 paired replicates; prefer 30
model pair:        primary (mid-tier) first
```

In `full-specification`, `root_local` and `root_history` are the same condition. Run one root arm and alias the other. Do not spend a second budget on a duplicate.

Engineering smoke before that grid:

```text
D:                 1, 2
task information:  explicit-overall
schedule:          oracle
seeds:             0 1 2
```

Pass the Section 13 checks here, especially checks 2, 7, and 10, before paying for `D=3,4` or full-specification.

Follow-ups after the paper-minimum is interpretable:

```text
schedule:          natural-unlock as sensitivity
replicates:        30 if intervals are still wide
model pair:        frontier replication
task information:  legacy opaque ablation only if needed to discuss the old 5.2 runs
```

`D=1` is a required null control: all transcripts contain the same single requirement, so `root_local` and `root_history` should agree within noise.

## 11. Matching and randomization

Pair arms on:

- data split;
- initial prompt;
- reflection minibatch IDs for each round;
- number of reflection proposals;
- task and reflection models;
- decoding settings and token limits;
- selection and test panels.
- exact overall task-description string.

Use independent reflection sampling within an arm. If the provider supports deterministic request seeds, use a shared list of proposal seeds per round and arm role; do not reuse one random-number stream whose consumption depends on arm behavior.

Match proposal count and scoring calls in the primary analysis. Report token use and critical-path rounds separately rather than claiming they are matched.

## 12. Outcome measures

Primary:

- mean held-out score per arm;
- paired `feedback_history_effect`;
- paired `prompt_state_effect`;
- paired `total_matched_effect`;
- two-sided paired 95% intervals.

Secondary:

- probability of solving the full chain;
- maximum prefix stage cleared;
- per-round validation curve;
- fraction of proposals satisfying each requirement;
- maximum selected lineage depth;
- task-model calls, reflection calls, tokens, and sequential critical-path rounds.

Do not use `(D-1)/D` as the expected empirical advantage. It is an idealized ceiling difference under perfect mutation and selection.

## 13. Competence and validity checks

Run these before the full experiment:

1. **Oracle-prompt check:** a hand-written prompt containing all `D` rules should score at least 0.95.
2. **Visible-rule mutation check:** when given `H_j`, the reflector should produce a prompt satisfying the first `j` rules at a high rate. If this is below 0.8 with `K=8`, simplify the task or increase the reflection budget.
3. **No-leakage check:** `root_local` must never receive a hint ID above 1.
4. **Transcript identity check:** hash the rendered transcript and assert equality between history arms at every round.
5. **Proposal-count check:** every arm must generate exactly `K * R` proposals.
6. **Selection check:** test data must never select prompts or unlock hints.
7. **D=1 null:** `feedback_history_effect` should be near zero.
8. **Task-surface check:** reject any primary run containing the legacy bare `Transform: ...` template.
9. **Full-information null:** `root_local` and `root_history` must be identical in the `full-specification` condition.
10. **Unrevealed-rule check:** under explicit-overall, `root_local` proposals must not systematically satisfy exact rules outside `H1`. If the fraction of `root_local` test or selection outputs that pass reverse, END, or BEGIN at `D>=2` is high while those hints were never in its transcript, the TASK card leaked. Stop and tighten the always-visible description before collecting the paper cells.

Fail the run immediately when a matching or leakage assertion fails.

## 14. Implementation plan

Preserve the existing `run_compound_sweep.py` and its results. Add a separate experiment.

### `experiment/compounding.py`

Expose the chain metadata without changing existing behavior:

```python
def chain_hints(d: int) -> list[tuple[str, str]]: ...
def passed_chain_prefix(data: dict, response: str, d: int) -> int: ...
```

Add the explicit task-description renderer used by the new runner:

```python
def chain_task_description(
    d: int,
    mode: Literal["explicit-overall", "full-specification"],
) -> str: ...

def render_chain_input(words: list[str], task_description: str) -> str: ...
```

`render_chain_input` must produce the `TASK` / `INPUT WORDS` structure from Section 4. The returned description must be passed to both task evaluation and reflection-context construction. Update or wrap `ChainEvaluator` so it never constructs the legacy bare `Transform: ...` input for this experiment.

### New `experiment/information_matched_chain.py`

Implement:

- `RequirementTranscript` with stable IDs and canonical rendering;
- `oracle_transcript(round_idx, d)`;
- `natural_transcript(selected_prompt, selection_data, current_transcript)`;
- `collect_outputs_without_feedback(parent, minibatch, adapter)`;
- `reflect_with_transcript(...)`;
- explicit task-input rendering for both proposal generation and scoring;
- the three-arm round loop;
- final archive selection and test evaluation;
- matching/leakage assertions.

### New `experiment/run_information_matched_chain.py`

Suggested CLI:

```text
python -m experiment.run_information_matched_chain \
  --ds 1 2 3 4 \
  --arms root_local root_history lineage_history \
  --schedule oracle \
  --task-information explicit-overall \
  --k 8 \
  --seeds 0 1 2 3 4 \
  --out runs/information_matched_chain
```

The runner should be resumable by complete `(D, information_mode, schedule, seed)` cells, but never combine partial arms from different attempts.

### Output schema

For every replicate, save:

```json
{
  "config": {},
  "rounds": [
    {
      "round": 1,
      "transcript_ids": {
        "root_local": ["uppercase"],
        "root_history": ["uppercase"],
        "lineage_history": ["uppercase"]
      },
      "parent_prompt_hash": {},
      "minibatch_ids": [],
      "proposal_count": {},
      "selected_score": {},
      "selected_stage": {}
    }
  ],
  "final": {
    "validation": {},
    "test": {},
    "feedback_history_effect": 0.0,
    "prompt_state_effect": 0.0,
    "total_matched_effect": 0.0
  },
  "checks": {
    "transcripts_matched": true,
    "proposal_counts_matched": true,
    "no_feedback_leakage": true
  }
}
```

Store full prompts and raw generations in a separate diagnostic file if needed; the summary should use hashes and aggregate scores.

## 15. Analysis and figures

Primary figure: for each `D`, show three paired bars or points:

```text
root_local -> root_history -> lineage_history
```

Annotate the first difference as feedback history and the second as prompt state.

Secondary figure: per-round probability of clearing each stage for the three arms.

Primary table:

| D | Root local | Root history | Lineage history | Feedback-history effect | Prompt-state effect |
|---|---:|---:|---:|---:|---:|

Report the explicit-overall and full-specification conditions in separate panels. If the legacy opaque ablation is run, place it in a supplementary panel and do not pool it with either primary condition.

## 16. Decision rules

The experiment supports an information-unlocking explanation when:

- `feedback_history_effect` is positive and practically meaningful for `D >= 2`;
- it is near zero at `D=1`;
- it shrinks in the `full-specification` condition.

It supports an additional prompt-state benefit when:

- `prompt_state_effect` remains positive after transcripts are matched;
- the result is not driven solely by one failed or collapsed replicate;
- the reflector competence checks pass.

If `root_history` matches `lineage_history`, revise the paper's mechanism language to say that lineage is useful because it carries forward newly discovered information, not because edit inheritance itself is necessary. If lineage remains better, the paper can distinguish information discovery from prompt-state accumulation.

## 17. Brief for collaborators

Every example now begins with an explicit overall task description: transform an ordered word list according to a prompt-specified formatting policy involving case, order, and boundary markers. The concrete values of those requirements remain controlled information. The experiment adds a root-reset arm that receives the same cumulative requirement transcript as lineage. Comparing that arm with ordinary root-reset measures the value of newly available information; comparing it with lineage measures the additional value of carrying forward the edited prompt. A fixed oracle transcript is the primary control, a lineage-driven transcript is a realism sensitivity, and a full-specification condition checks whether lineage still helps when all exact rules are visible from the start. The old bare `Transform: ...` input is, at most, a supplementary legacy ablation.
