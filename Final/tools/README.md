# Tools Included In This Upload

This directory contains only non-temporary project scripts.

## Included

- `check_implicit_mpm_contracts.py`: static contract checks for the implicit MPM
  integration.
- `run_ablation.py`: experiment runner for explicit, implicit, and PBMPM
  comparisons. It requires user-provided PhysGaussian paths and scene data.
- `collect_ablation_results.py`: summary collector for completed ablation runs.
- `phys_splat_bridge/`: lightweight conversion and preprocessing helpers for
  PhysGaussian/SuperSplat motion-package experiments.

## Excluded From This Package

The working directory contains additional local or cloud debugging helpers. They
are intentionally not included here:

- cloud run inspectors and remote restart helpers;
- local workspace sync manifests;
- Codex/history export utilities;
- internal asset download or registration scripts;
- caches and temporary outputs.

Use this folder for reproducible project checks and experiments, not for
machine-specific development state.
