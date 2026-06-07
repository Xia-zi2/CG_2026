# SuperSplat Override Files

These files are modified project overlay files for the upstream SuperSplat
frontend.

## Upstream

- Repository: https://github.com/playcanvas/supersplat
- Use a compatible upstream `main` commit, or record the exact commit used in
  your experiment report.

## How To Apply

Copy this directory into the upstream SuperSplat `src/` directory:

```bash
rsync -av src-overrides/supersplat-src/ external/supersplat/src/
```

On Windows, copy the same relative paths with PowerShell or File Explorer.

## Included Modified Areas

- Physics session and backend request lifecycle.
- Physics panel UI and controls.
- Per-splat motion loading and playback.
- Worker-based motion processing.
- Selection, editor, render, and style integration points needed by the physics
  workflow.

## Data Not Included

No built frontend bundle, `node_modules`, browser cache, uploaded scene asset,
motion binary, or rendered output is included. Install dependencies in the
upstream SuperSplat checkout and build there after applying these overrides.
