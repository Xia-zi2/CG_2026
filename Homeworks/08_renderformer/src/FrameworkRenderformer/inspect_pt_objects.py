from __future__ import annotations

import argparse
from pathlib import Path
import torch


def shape_str(x):
    if hasattr(x, "shape"):
        return f"Tensor shape={tuple(x.shape)}, dtype={getattr(x, 'dtype', None)}"
    if isinstance(x, list):
        if len(x) == 0:
            return "list len=0"
        first = x[0]
        if hasattr(first, "shape"):
            return f"list len={len(x)}, first shape={tuple(first.shape)}, dtype={getattr(first, 'dtype', None)}"
        return f"list len={len(x)}, first={first!r}"
    return f"{type(x).__name__}: {x!r}"


def inspect_file(path: Path):
    print("=" * 100)
    print("FILE:", path)
    print("=" * 100)

    sample = torch.load(path, map_location="cpu", weights_only=False)

    print("\n[Keys and shapes]")
    for key in sorted(sample.keys()):
        print(f"{key:30s}: {shape_str(sample[key])}")

    print("\n[Object grouping diagnosis]")

    has_scene_objects = "scene_objects_pos" in sample
    has_scene_object_names = "scene_object_names" in sample
    has_tri_pos = "tri_pos" in sample

    if has_scene_object_names:
        print("\nscene_object_names:")
        for i, name in enumerate(sample["scene_object_names"]):
            print(f"  object {i}: {name}")
    else:
        print("\nscene_object_names: NOT FOUND")

    if has_scene_objects:
        objs = sample["scene_objects_pos"]
        print("\nscene_objects_pos: FOUND")
        print("num objects:", len(objs))
        for i, obj in enumerate(objs):
            name = None
            if has_scene_object_names and i < len(sample["scene_object_names"]):
                name = sample["scene_object_names"][i]
            label = f"{i}" if name is None else f"{i} ({name})"
            print(f"  object {label}: tri_pos shape={tuple(obj.shape)}")

    if has_tri_pos:
        tri_pos = sample["tri_pos"]
        print("\ntri_pos: FOUND")
        print("tri_pos shape:", tuple(tri_pos.shape))

        if tri_pos.ndim == 3:
            print("Interpretation: tri_pos = [O, T, 9]. Object grouping is present.")
            print("O =", tri_pos.shape[0], "objects")
            print("T =", tri_pos.shape[1], "max triangles per object")
        elif tri_pos.ndim == 2:
            print("Interpretation: tri_pos = [N, 9]. Object grouping is already flattened.")
        else:
            print("Interpretation: unknown tri_pos layout.")

    if "tri_mask" in sample:
        tri_mask = sample["tri_mask"].bool()
        print("\ntri_mask shape:", tuple(tri_mask.shape))
        if tri_mask.ndim == 2:
            print("valid triangles per object:", tri_mask.sum(dim=1).tolist())
        elif tri_mask.ndim == 1:
            print("valid triangles total:", int(tri_mask.sum().item()))

    if "obj_mask" in sample:
        obj_mask = sample["obj_mask"].bool()
        print("\nobj_mask shape:", tuple(obj_mask.shape))
        print("valid object count:", int(obj_mask.sum().item()))
        print("obj_mask:", obj_mask.tolist())

    print("\n[Conclusion]")
    if has_scene_objects:
        print("Explicit object grouping exists via scene_objects_pos.")
        if has_scene_object_names:
            print("Object names also exist via scene_object_names.")
        else:
            print("Object names are not stored, but object groups exist.")
    elif has_tri_pos and sample["tri_pos"].ndim == 3:
        print("Object grouping exists via tri_pos=[O,T,9], but names may be absent.")
    elif has_tri_pos and sample["tri_pos"].ndim == 2:
        print("This file is triangle-flattened. Object grouping is not directly available.")
    else:
        print("Cannot determine object grouping from common fields.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data")
    parser.add_argument("--num_files", type=int, default=3)
    args = parser.parse_args()

    root = Path(args.root)
    files = sorted(root.rglob("*.pt"))

    if not files:
        raise FileNotFoundError(f"No .pt files found under: {root.resolve()}")

    print("Root:", root.resolve())
    print("Found .pt files:", len(files))
    print("Inspecting first", min(args.num_files, len(files)), "files")

    for path in files[: args.num_files]:
        inspect_file(path)


if __name__ == "__main__":
    main()
