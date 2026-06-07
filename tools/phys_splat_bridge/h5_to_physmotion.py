#!/usr/bin/env python3
"""Convert PhysGaussian H5 frame output into a SuperSplat motion package.

Expected PhysGaussian input:
    output/simulation_ply/sim_0000000000.h5
    output/simulation_ply/sim_0000000001.h5
    ...

The H5 files created by PhysGaussian store x with shape (3, N). This script
writes a frame-major float32 stream with shape (F, N, 3).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_h5_positions(path: Path) -> np.ndarray:
    try:
        import h5py
    except ImportError as exc:
        raise SystemExit("h5py is required: pip install h5py") from exc

    with h5py.File(path, "r") as handle:
        if "x" not in handle:
            raise ValueError(f"{path} has no dataset named 'x'")
        data = np.asarray(handle["x"], dtype=np.float32)

    if data.ndim != 2:
        raise ValueError(f"{path} dataset 'x' must be rank-2, got {data.shape}")
    if data.shape[0] == 3:
        data = data.T
    if data.shape[1] != 3:
        raise ValueError(f"{path} dataset 'x' must have shape (3, N) or (N, 3), got {data.shape}")

    return np.ascontiguousarray(data, dtype=np.float32)


def _frame_number(path: Path) -> int:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits) if digits else -1


def convert_h5_folder(
    h5_dir: Path,
    output_dir: Path,
    *,
    base: str | None,
    frame_rate: int,
    pattern: str,
    update_bounds: bool,
) -> None:
    files = sorted(h5_dir.glob(pattern), key=_frame_number)
    if not files:
        raise FileNotFoundError(f"No H5 frames matched {h5_dir / pattern}")

    output_dir.mkdir(parents=True, exist_ok=True)
    motion_path = output_dir / "motion.bin"
    manifest_path = output_dir / "motion.physmotion.json"

    first = _load_h5_positions(files[0])
    num_splats = int(first.shape[0])
    frame_stride_bytes = num_splats * 3 * np.dtype(np.float32).itemsize

    with motion_path.open("wb") as stream:
        stream.write(first.tobytes(order="C"))
        for path in files[1:]:
            positions = _load_h5_positions(path)
            if positions.shape != first.shape:
                raise ValueError(f"{path} has shape {positions.shape}, expected {first.shape}")
            stream.write(positions.tobytes(order="C"))

    manifest = {
        "format": "phys-motion-v1",
        "binary": motion_path.name,
        "frameCount": len(files),
        "frameRate": frame_rate,
        "numSplats": num_splats,
        "attributes": ["position"],
        "frameStrideBytes": frame_stride_bytes,
        "updateBounds": update_bounds,
    }
    if base:
        manifest["base"] = base

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {manifest_path}")
    print(f"Wrote {motion_path}")
    print(f"Frames: {len(files)}, splats: {num_splats}, stride: {frame_stride_bytes} bytes")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-dir", type=Path, required=True, help="PhysGaussian simulation_ply directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for motion.bin and manifest")
    parser.add_argument("--base", default=None, help="Optional static base splat filename packaged with the motion files")
    parser.add_argument("--frame-rate", type=int, default=30)
    parser.add_argument("--pattern", default="sim_*.h5")
    parser.add_argument("--no-update-bounds", action="store_true")
    args = parser.parse_args()

    convert_h5_folder(
        args.h5_dir,
        args.output_dir,
        base=args.base,
        frame_rate=args.frame_rate,
        pattern=args.pattern,
        update_bounds=not args.no_update_bounds,
    )


if __name__ == "__main__":
    main()
