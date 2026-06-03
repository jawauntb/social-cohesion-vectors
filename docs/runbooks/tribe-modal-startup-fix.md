---
title: TRIBE Modal Startup Fix
status: active
date: 2026-06-03
---

# TRIBE Modal Startup Fix

Use this runbook for any repo that runs TRIBE/tribev2 in Modal and fails at
startup or import time.

## Symptom

A Modal app that imports TRIBE crashes before serving, often during image build,
container startup, or predictor import.

The common failure mode is a dependency compatibility mismatch:

- TRIBE pulls `neuralset==0.0.2`.
- `neuralset==0.0.2` still imports `exca.steps.base.NoValue`.
- Floating `exca` can resolve to a version without that compatibility surface.
- The TRIBE predictor app crashes before serving.

## Audit

Search the repo for TRIBE and `exca` usage:

```bash
rg -n "facebookresearch/tribev2|tribev2|TribeModel|exca" .
```

For a broader local audit across sibling repos:

```bash
cd ~/path-containing-repos
rg -n "facebookresearch/tribev2|from tribev2 import TribeModel|exca.steps.base|uv_pip_install\\(|pip_install\\(" \
  --glob '!**/.git/**' --glob '!**/node_modules/**' --glob '!**/.venv/**'
```

If the Modal image installs TRIBE without pinning `exca`, it likely needs the
fix. Vulnerable patterns look like:

```python
.uv_pip_install(
    "git+https://github.com/facebookresearch/tribev2.git@..."
)
```

or:

```python
.pip_install("git+https://github.com/facebookresearch/tribev2.git@...")
```

## Fix

Pin `exca==0.5.25` in the same install layer as TRIBE:

```python
_TRIBE_EXCA_VERSION = "0.5.25"

.uv_pip_install(
    "git+https://github.com/facebookresearch/tribev2.git@<pinned-commit>",
    f"exca=={_TRIBE_EXCA_VERSION}",
)
```

Use the equivalent shape for `.pip_install(...)` if the repo uses Modal's pip
installer rather than `uv_pip_install`.

## Build-Time Preflight

Add a build-time import preflight so Modal catches the failure during image
build instead of production container startup:

```python
_TRIBE_IMPORT_RUNTIME_PREFLIGHT_COMMAND = (
    'python -c "import exca.steps.base as exca_base; '
    "exca_base.NoValue(); "
    "from tribev2 import TribeModel; "
    'print(TribeModel.__name__)"'
)

.run_commands(
    ...,
    _TRIBE_IMPORT_RUNTIME_PREFLIGHT_COMMAND,
)
```

## After Patching

1. Run the repo's lint, typecheck, and tests.
2. Deploy or rebuild every Modal environment using that TRIBE image.
3. Do not repopulate TRIBE or HuggingFace weights unless the repo changed its
   volume or cache setup.

## Current Repo Audit

As of 2026-06-03, this repo references TRIBE as a planned brain-aligned bridge
and has a default `TRIBE_MODEL_ID`, but it does not install
`facebookresearch/tribev2` in a Modal image and does not import `TribeModel`.
No local Modal TRIBE patch is needed here yet.
