# Usage Guide

This guide describes how to run the integrated SuperSplat + PhysGaussian
workflow from a clean repository checkout.

## 1. Prepare Third-Party Source Trees

Clone the upstream projects outside this repository:

```bash
mkdir -p external
git clone https://github.com/playcanvas/supersplat.git external/supersplat
git clone https://github.com/XPandora/PhysGaussian.git external/PhysGaussian
```

Apply this repository's overrides:

```bash
rsync -av src-overrides/supersplat-src/ external/supersplat/src/
rsync -av src-overrides/physgaussian-src/ external/PhysGaussian/
```

On Windows, copy the same folders with File Explorer, PowerShell
`Copy-Item -Recurse`, or your preferred sync tool.

## 2. Backend Setup

```powershell
cd server
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Start the backend:

```powershell
python phys_backend.py --host 127.0.0.1 --port 8443
```

The backend exposes health and simulation endpoints used by the frontend
physics panel.

## 3. Frontend Setup

From the upstream SuperSplat checkout after applying overrides:

```bash
npm install
npm run build
npm run dev
```

Open the development URL printed by the frontend toolchain. Configure the
frontend backend URL to point at the running Python backend when needed.

## 4. Local/Cloud Sync

Copy `tools/workspace_manifest.example.json` to
`tools/workspace_manifest.json`, then edit the local paths, SSH host, SSH key,
and cloud root for your environment.

The real `tools/workspace_manifest.json` is ignored by Git because it contains
machine-specific paths and connection settings.

## 5. Validation

Run the contract check:

```powershell
python tools/check_implicit_mpm_contracts.py
```

Run an ablation plan:

```powershell
python tools/run_ablation.py --help
```

Collect metrics from completed runs:

```powershell
python tools/collect_ablation_results.py --help
```

Keep generated results under ignored directories such as `outputs/`, `logs/`,
or `reports/`.

## 6. Data And Model Files

Scene files, datasets, checkpoints, videos, and rendered outputs are not part of
the upload version. Place them in ignored folders or external storage and
document the download source in your experiment notes.
