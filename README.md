# CG_2026 Upload Preview

This package contains the project-owned source code and the modified external
project overlay files prepared for upload to `Xia-zi2/CG_2026`.

It is intentionally smaller than the working directory. Temporary debugging
tools, local cloud manifests, caches, assets, generated outputs, backup
archives, model checkpoints, datasets, videos, and full third-party source
trees are not included.

## Contents

- `server/`: our Python backend bridge for the interactive physics workflow.
- `src-overrides/physgaussian-src/`: modified files that should be applied to an
  upstream PhysGaussian checkout.
- `src-overrides/supersplat-src/`: modified files that should be applied to an
  upstream SuperSplat checkout.
- `tools/`: non-temporary validation, ablation, and file-format bridge scripts.
- `docs/`: citations, external library notes, and work summary.

## External Projects

This upload does not vendor complete external repositories. Download them
separately, then apply the overlay directories:

- SuperSplat upstream: https://github.com/playcanvas/supersplat
- PhysGaussian upstream: https://github.com/XPandora/PhysGaussian

See:

- `src-overrides/supersplat-src/README.md`
- `src-overrides/physgaussian-src/README.md`
- `docs/EXTERNAL_LIBS.md`

## Not Included

- `vendor/`, full upstream repositories, or nested `.git` directories.
- `demo_assets/`, `upload_models/`, datasets, checkpoints, model weights, PLY
  assets, H5 outputs, motion binaries, rendered videos, and experiment outputs.
- `recovery_packages/`, `packages/`, backup files, logs, caches, and
  `__pycache__/`.
- Local sync manifests such as `tools/workspace_manifest.json`.
- Temporary cloud inspection scripts and internal development asset scripts.

## Basic Setup

Install backend dependencies:

```bash
cd server
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
cd server
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the backend:

```bash
uvicorn server.phys_backend:app --host 127.0.0.1 --port 8000
```

Apply the SuperSplat and PhysGaussian overlays before running the complete
frontend/backend integration.

## Validation

```bash
python tools/check_implicit_mpm_contracts.py
python tools/run_ablation.py --help
python tools/collect_ablation_results.py --help
```

Full experiments require user-provided scene data, upstream source checkouts,
and a Python environment with the PhysGaussian dependencies installed.
