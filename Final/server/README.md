# PhysGaussian Backend Bridge

This is the first backend bridge for the patched SuperSplat physics panel.

## Start

```bash
pip install -r server/requirements.txt
PHYSGAUSSIAN_ROOT=/path/to/PhysGaussian \
PYTHON_BIN=/path/to/physgaussian/python \
uvicorn server.phys_backend:app --host 0.0.0.0 --port 8000
```

`PYTHON_BIN` should point to the Python interpreter inside the environment where
PhysGaussian, CUDA PyTorch, Warp, Taichi, and the 3DGS submodules are installed.

## API

- `POST /api/models`: upload a zipped trained 3DGS/PhysGaussian model folder.
  The backend extracts it, finds `point_cloud/iteration_*/point_cloud.ply`, and
  returns a `modelId` plus a preview PLY URL.
- `POST /api/simulate`: accepts the SuperSplat physics payload, writes a
  PhysGaussian config, starts a background simulation, and returns `runId`.
- `GET /api/runs/{runId}`: returns `queued`, `running`, `completed`, or
  `failed`. Completed runs include `motion.physmotion.json`, `motion.bin`, and
  `indices.bin` URLs for SuperSplat playback.

## Current Solver Limits

This bridge keeps stock PhysGaussian semantics. Object selections are converted
to AABB-based `additional_material_params`, so v1 can change `E`, `nu`, and
`density` per selected object, but cannot yet assign different material laws or
true finite-mass rigid-body modes per object. Those require the later
per-particle label and solver extensions from the project plan.
