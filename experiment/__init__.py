# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Launch-ready harness for the Breadth-or-Depth study (S2 root-reset vs S3 GEPA).

Consolidates the pilot-validated approach into a single, parameterized runner
with the Phase 3 matching-verification gate. See RUNBOOK.md.

Importing this package auto-loads API keys from the nearest ``.env`` (walking up
from the cwd) without overriding anything already in the environment. This is how
``openai/`` models (gpt-5.6-luna / gpt-5.6-terra) reach the OpenAI API directly with
``OPENAI_API_KEY``, and ``openrouter/`` models (Qwen) reach OpenRouter with
``OPENROUTER_API_KEY``. litellm routes by the model-name prefix, so no per-provider
client wiring is needed.
"""

import os as _os
from pathlib import Path as _Path


def _load_env_keys() -> None:
    """Load API keys from the nearest .env; never override existing env vars."""
    try:
        from dotenv import find_dotenv, load_dotenv  # python-dotenv (a gepa dep)

        p = find_dotenv(usecwd=True)
        if p:
            load_dotenv(p, override=False)
            return
    except Exception:
        pass
    # Fallback parser if python-dotenv is unavailable: walk up to the nearest .env.
    here = _Path.cwd()
    for base in [here, *here.parents]:
        f = base / ".env"
        if f.is_file():
            for line in f.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and k not in _os.environ:
                    _os.environ[k] = v
            break


_load_env_keys()
