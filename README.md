# Breadth or Depth?

Code and saved runs for the accompanying manuscript. Lineage vs root-reset in reflective prompt optimization; estimand is the paired test gap \(A_c\).

```
experiment/   runners (from repo root)
src/gepa/     GEPA 0.1.4
results/      JSONs that regenerate the tables
```

Reprint tables / figure (no API):

```bash
python experiment/summarize_results.py
python experiment/make_scatter.py
```

Setup, then run from the repo root. Copy `.env.example` → `.env` (`OPENROUTER_API_KEY`, `OPENAI_API_KEY`). Models are in `experiment/slate.py`. Write new runs under `runs/`.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[full,ifbench,retrieval]" python-dotenv scipy matplotlib

python -m experiment.run_study --dry-run --env .env
python -m experiment.run_study --dataset hotpotqa --seeds 0 1 2 3 4 5 6 7 8 9 \
  --n-train 20 --n-dev 20 --n-test 100 --retrieval --dev-base-floor 0.05 \
  --out runs/gepa_suite/mid_hotpotqa_acceptall --env .env
python -m experiment.run_paired_task --task mbpp --seeds 0 1 2 3 4 5 6 7 8 9 \
  --out runs/paired/mid_mbpp_acceptall --env .env
python -m experiment.run_information_matched_chain --ds 1 2 3 4 \
  --task-information explicit-overall full-specification --seeds 0 1 2 3 4 \
  --n-train 12 --n-sel 12 --n-val 12 --n-test 20 --out runs/chain/mid --env .env
```

`--dataset` also: `hover`, `ifbench`, `pupa`. `--task` also: `structured`, `funccall`, `top`, `massive`. Frontier models and flags: `slate.py`. IFBench: clone https://github.com/allenai/IFBench and set `IFBENCH_DIR`. File-to-table map: `results/README.txt`.
