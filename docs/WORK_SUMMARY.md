# Work Summary

This repository snapshot prepares a clean upload version of the interactive
physics editing workflow.

## Implemented Areas

- Python backend bridge for scene/session handling and physics requests.
- SuperSplat UI and state overrides for selecting physics parameters,
  dispatching simulation work, and managing per-splat motion metadata.
- Worker-based frontend motion handling in
  `src-overrides/supersplat-src/src/phys-motion-worker.ts`.
- PhysGaussian overrides for MPM stepping, utility functions, and parameter
  decoding used by the integrated workflow.
- Workspace synchronization tooling with local configuration separated into an
  ignored manifest file and a committed example template.
- Ablation runner and result collector scripts for repeatable experiments.

## Upload Cleanup

- Local machine paths, SSH identity paths, live cloud URLs, assets, checkpoints,
  outputs, logs, and backup packages are excluded from the upload version.
- Third-party source trees are referenced as external dependencies rather than
  committed wholesale.
- The top-level README and usage docs now describe how to rebuild the working
  environment from clean sources.

## Suggested Validation

```powershell
python tools/check_implicit_mpm_contracts.py
python tools/run_ablation.py --help
python tools/collect_ablation_results.py --help
```

Full end-to-end validation also requires local scene assets and upstream
SuperSplat/PhysGaussian checkouts with this repository's overrides applied.
