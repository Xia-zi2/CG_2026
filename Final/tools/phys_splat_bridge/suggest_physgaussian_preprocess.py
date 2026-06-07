#!/usr/bin/env python3
"""Suggest PhysGaussian preprocessing parameters from a PLY AABB.

This helper does not try to solve object orientation. It computes the selected
point AABB, adds optional padding, and emits the config fields that can be
automated reliably: sim_area, scale, and a neutral rotation placeholder.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Iterable

import numpy as np


def _parse_header(stream) -> tuple[list[str], int, str, int]:
    header_lines: list[str] = []
    vertex_count = -1
    fmt = ""
    while True:
        line = stream.readline()
        if not line:
            raise ValueError("PLY header ended unexpectedly")
        text = line.decode("ascii").strip()
        header_lines.append(text)
        if text.startswith("format "):
            fmt = text.split()[1]
        if text.startswith("element vertex "):
            vertex_count = int(text.split()[2])
        if text == "end_header":
            break
    if vertex_count < 0:
        raise ValueError("PLY has no element vertex count")
    if fmt not in {"ascii", "binary_little_endian"}:
        raise ValueError(f"Unsupported PLY format: {fmt}")
    return header_lines, vertex_count, fmt, stream.tell()


def _vertex_properties(header_lines: Iterable[str]) -> list[tuple[str, str]]:
    props: list[tuple[str, str]] = []
    in_vertex = False
    for line in header_lines:
        if line.startswith("element "):
            in_vertex = line.startswith("element vertex ")
            continue
        if in_vertex and line.startswith("property "):
            parts = line.split()
            if len(parts) == 3:
                props.append((parts[1], parts[2]))
    return props


def read_ply_xyz(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        header, vertex_count, fmt, data_start = _parse_header(stream)
        props = _vertex_properties(header)
        names = [name for _, name in props]
        try:
            x_id, y_id, z_id = names.index("x"), names.index("y"), names.index("z")
        except ValueError as exc:
            raise ValueError("PLY must contain x, y, z vertex properties") from exc

        if fmt == "ascii":
            points = np.zeros((vertex_count, 3), dtype=np.float32)
            for i in range(vertex_count):
                values = stream.readline().decode("ascii").split()
                points[i] = [float(values[x_id]), float(values[y_id]), float(values[z_id])]
            return points

        type_map = {
            "char": ("b", 1),
            "uchar": ("B", 1),
            "short": ("h", 2),
            "ushort": ("H", 2),
            "int": ("i", 4),
            "uint": ("I", 4),
            "float": ("f", 4),
            "double": ("d", 8),
        }
        fmt_chars = [type_map[prop_type][0] for prop_type, _ in props]
        row_format = "<" + "".join(fmt_chars)
        row_size = struct.calcsize(row_format)
        stream.seek(data_start)
        raw = stream.read(row_size * vertex_count)
        points = np.zeros((vertex_count, 3), dtype=np.float32)
        for i in range(vertex_count):
            values = struct.unpack_from(row_format, raw, i * row_size)
            points[i] = [values[x_id], values[y_id], values[z_id]]
        return points


def read_indices(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("indices", [])
        return np.asarray(data, dtype=np.int64)
    return np.loadtxt(path, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--indices", type=Path, default=None, help="Optional selected gaussian ids, txt or json")
    parser.add_argument("--padding", type=float, default=0.02, help="Fraction of max side length added to all AABB sides")
    parser.add_argument("--scale", type=float, default=1.0, help="PhysGaussian target max side length before shift2center111")
    parser.add_argument("--n-grid", type=int, default=100)
    args = parser.parse_args()

    points = read_ply_xyz(args.ply)
    if args.indices:
        indices = read_indices(args.indices)
        points = points[indices]

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    extent = maxs - mins
    pad = float(extent.max() * args.padding)
    mins -= pad
    maxs += pad

    snippet = {
        "rotation_degree": [0],
        "rotation_axis": [0],
        "sim_area": [
            float(mins[0]), float(maxs[0]),
            float(mins[1]), float(maxs[1]),
            float(mins[2]), float(maxs[2]),
        ],
        "scale": args.scale,
        "n_grid": args.n_grid,
        "_notes": [
            "sim_area is computed from the selected Gaussian AABB plus padding.",
            "scale is the target maximum side length in PhysGaussian's MPM space before shift2center111.",
            "rotation is intentionally neutral; confirm orientation visually in SuperSplat before simulation.",
        ],
    }
    print(json.dumps(snippet, indent=2))


if __name__ == "__main__":
    main()
