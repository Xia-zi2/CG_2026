# PhysGaussian <-> SuperSplat bridge

This folder contains the first lightweight bridge used by the project:

1. Load one static Gaussian model in SuperSplat.
2. Run PhysGaussian with `--output_h5` to export per-frame particle states.
3. Convert the H5 frame folder into a compact `motion.bin + motion.physmotion.json` package.
4. Drag the static base splat plus the motion package into the patched SuperSplat build.

For the interactive backend we now prefer the patched PhysGaussian flag
`--output_super_motion`. It writes world-space Gaussian positions plus
`indices.bin` directly, so SuperSplat updates the original PLY coordinates
instead of accidentally playing raw MPM-grid coordinates.

The legacy H5 conversion path only streams positions and assumes the base splat
loaded in SuperSplat has the same Gaussian order and count as the exported H5
frames. Use it only for quick file-format checks. The backend path should use
`--output_super_motion`, because it exports original Gaussian indices and applies
the inverse preprocessing transform before playback.

## Motion manifest

`motion.physmotion.json`:

```json
{
  "format": "phys-motion-v1",
  "base": "point_cloud.ply",
  "binary": "motion.bin",
  "frameCount": 120,
  "frameRate": 30,
  "numSplats": 100000,
  "attributes": ["position"],
  "frameStrideBytes": 1200000,
  "updateBounds": true
}
```

`motion.bin` is little-endian float32, frame-major:

```text
frame 0: x0 y0 z0 x1 y1 z1 ...
frame 1: x0 y0 z0 x1 y1 z1 ...
...
```

Optional future fields:

- `indices`: a uint32 binary file. If present, frame data updates only those Gaussian ids.
- `attributes`: can later include `rotation` and `scale`.

## Why not per-frame PLY

Per-frame PLY repeats static color, opacity, SH, and schema data, then forces SuperSplat to parse and rebuild a whole splat resource every frame. The motion package keeps the base visual data resident on GPU and only updates dynamic transform arrays.

## Rotation and preprocessing

Centering and scaling can be automated from the selected Gaussian AABB. Rotation should remain semi-automatic for now:

- PCA can suggest candidate axes, but it cannot reliably know which side is "floor", "up", or "physically stable".
- The UI should let the user rotate visually, then confirm.
- After confirmation, the bridge can write `rotation_degree`, `rotation_axis`, `sim_area`, `scale`, and `n_grid` into a PhysGaussian config.
