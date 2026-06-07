# PhysGaussian Override Files

These files are modified project overlay files for the upstream PhysGaussian
repository.

## Upstream

- Repository: https://github.com/XPandora/PhysGaussian
- Use a commit or release compatible with your CUDA/PyTorch/Warp environment.

## How To Apply

Copy the contents of this directory into the root of an upstream PhysGaussian
checkout:

```bash
rsync -av src-overrides/physgaussian-src/ external/PhysGaussian/
```

On Windows, copy the same relative paths with PowerShell or File Explorer.

## Included Modified Paths

- `gs_simulation.py`
- `mpm_solver_warp/mpm_solver_warp.py`
- `mpm_solver_warp/mpm_utils.py`
- `mpm_solver_warp/warp_utils.py`
- `utils/decode_param.py`
- `utils/transformation_utils.py`

## Modification Purpose

- Add and stabilize explicit, implicit, and PBMPM-related solver paths used by
  the project experiments.
- Support backend-driven parameter decoding and simulation configuration.
- Preserve a small overlay rather than committing the full external project.

## Data Not Included

PhysGaussian model folders, configs, checkpoints, generated H5 files, PLY
outputs, videos, and simulation logs are not included. Download or generate
them separately according to the upstream project instructions.
