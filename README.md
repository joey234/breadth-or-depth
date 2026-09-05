# Breadth or Depth? Toward Adaptive Search in Reflective Prompt Optimization

Anonymous code and saved runs for the accompanying manuscript.

The question is whether reflective prompt optimizers need **lineage** (later rewrites inherit earlier edits) or whether **root-reset** (every rewrite starts from the seed prompt \(p_0\)) is enough. The estimand is the paired test-score gap \(A_c\) (lineage minus root-reset).

Do not add author names, affiliations, or a repository URL to the paper until camera-ready. If this repository is shared with reviewers, host it under an anonymous account.

## Repository structure

```
.
├── README.md
├── .env.example
├── experiment/            # harness (run from the repo root)
├── src/gepa/              # vendored GEPA 0.1.4
├── results/               # saved run JSONs that regenerate the tables
├── pyproject.toml
└── LICENSE-GEPA           # upstream GEPA license
```

`bod` was only a local working-directory name for this project. It is not part of the public layout.

## Reproduce the published numbers (no API calls)

```bash
python experiment/summarize_results.py
```

That reprints Tables 2–4 and the appendix GEPA / chain / paired tables from `results/`. See `results/README.txt` for the file-to-table map.

```bash
python experiment/make_scatter.py    # writes results/bod_scatter.pdf
```

Figure points are locked in `results/scatter_data.json`.

## Setup (to re-run experiments)

Python 3.10–3.13:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[full,ifbench,retrieval]"
pip install python-dotenv scipy matplotlib
```

Copy `.env.example` to `.env` at the repository root:

| Variable | Used for |
|---|---|
| `OPENROUTER_API_KEY` | Mid-tier DeepSeek-V4 Flash / Pro |
| `OPENAI_API_KEY` | Frontier GPT-5.6 Luna / Terra |

Slugs are in `experiment/slate.py`. Run every command below from the **repository root**.

### IFBench (only for `--dataset ifbench`)

```bash
git clone https://github.com/allenai/IFBench
export IFBENCH_DIR="$PWD/IFBench"
python -c "import nltk; [nltk.download(x, quiet=True) for x in ['punkt','punkt_tab','averaged_perceptron_tagger','averaged_perceptron_tagger_eng','stopwords']]"
```

## How to re-run

Runners are resumable. Write new outputs under `runs/` so they do not overwrite `results/`.

### Smoke

```bash
python -m experiment.run_study --dry-run --env .env
python -m experiment.run_information_matched_chain --smoke --env .env --out runs/chain_smoke
```

### GEPA-suite (HotpotQA, HoVer, IFBench, PUPA)

Accept-all (Table `tab:gepa`):

```bash
python -m experiment.run_study --dataset hotpotqa --seeds 0 1 2 3 4 5 6 7 8 9 \
  --n-train 20 --n-dev 20 --n-test 100 --retrieval --dev-base-floor 0.05 \
  --task-model openrouter/deepseek/deepseek-v4-flash-0731 \
  --reflect-model openrouter/deepseek/deepseek-v4-pro-0813 \
  --reflect-reasoning-effort low \
  --out runs/gepa_suite/mid_hotpotqa_acceptall --env .env
```

Repeat for `hover` (dev floor 0.15, `--retrieval`), `ifbench` (0.15), and `pupa` (0.05). Frontier: `openai/gpt-5.6-luna` + `openai/gpt-5.6-terra` and `--task-reasoning-effort low`. Strict GEPA acceptance: add `--no-unconditional`.

### Paired benchmark tasks

`--task` is one of `structured`, `mbpp`, `funccall`, `top`, `massive`.

```bash
python -m experiment.run_paired_task --task mbpp --seeds 0 1 2 3 4 5 6 7 8 9 \
  --task-model openrouter/deepseek/deepseek-v4-flash-0731 \
  --reflect-model openrouter/deepseek/deepseek-v4-pro-0813 \
  --reflect-reasoning-effort low \
  --n-train 12 --n-dev 12 --n-test 100 \
  --out runs/paired/mid_mbpp_acceptall --env .env
```

Strict-improvement supplements: add `--faithful`.

### Information-matched chain (Section 5.2)

```bash
python -m experiment.run_information_matched_chain \
  --ds 1 2 3 4 \
  --task-information explicit-overall full-specification \
  --seeds 0 1 2 3 4 \
  --n-train 12 --n-sel 12 --n-val 12 --n-test 20 \
  --task-model openrouter/deepseek/deepseek-v4-flash-0731 \
  --reflect-model openrouter/deepseek/deepseek-v4-pro-0813 \
  --out runs/chain/mid --env .env
```

### Full two-tier orchestrator

```bash
python -m experiment.run_full --dry-run
python -m experiment.run_full --tiers mid --seeds 0 1 2 3 4 5 6 7 8 9 --root runs/full
python -m experiment.run_full --status --root runs/full
```

Place `.env` at the repo root. `run_full` does not take `--env`.

## What not to commit

- `.env` or any API key
- `.venv/`, `runs/`, or `IFBench/`
- Author names, emails, or institutional paths
