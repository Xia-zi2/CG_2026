from __future__ import annotations

import bisect
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import h5py
except ImportError:  # pragma: no cover - optional dependency for H5 mode only
    h5py = None
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def _as_tensor(value: Any, dtype: Optional[torch.dtype] = torch.float32) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value
    elif isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value)
    else:
        tensor = torch.tensor(value)

    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def _resolve_pt_directory(data_path: str | Path) -> Path:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"PT data path does not exist: {path}")
    if path.is_dir() and (path / "train").is_dir():
        return path / "train"
    return path


class PtSceneDataset(Dataset):
    """Loads pre-generated .pt samples from disk."""

    def __init__(self, data_path: str | Path, max_items: Optional[int] = None):
        self.data_dir = _resolve_pt_directory(data_path)
        self.files = sorted(self.data_dir.glob("*.pt"))
        if max_items is not None:
            self.files = self.files[:max_items]

        if not self.files:
            raise FileNotFoundError(f"No .pt files found in {self.data_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        path = self.files[index]
        try:
            sample = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            if len(self.files) == 1:
                raise RuntimeError(f"Failed to load PT sample: {path}") from exc
            new_index = random.randint(0, len(self.files) - 1)
            return self.__getitem__(new_index)

        sample["sample_name"] = sample.get("sample_name", path.name)
        sample["sample_idx"] = sample.get("sample_idx", index)
        return sample


def ensure_h5py_available() -> None:
    if h5py is None:
        raise ImportError(
            "H5 dataset support requires h5py. Install h5py, or use --dataset_format pt with PT samples."
        )


def find_h5_files(data_path: str | Path) -> List[Path]:
    ensure_h5py_available()
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"H5 data path does not exist: {path}")

    if path.is_file():
        if path.suffix.lower() in {".h5", ".hdf5"} or h5py.is_hdf5(str(path)):
            return [path]
        raise FileNotFoundError(f"File is not a valid HDF5 file: {path}")

    h5_files: List[Path] = []
    for file_path in path.rglob("*"):
        if not file_path.is_file() or file_path.name.startswith("."):
            continue
        if file_path.suffix.lower() in {".h5", ".hdf5"}:
            h5_files.append(file_path)
            continue
        try:
            if h5py.is_hdf5(str(file_path)):
                h5_files.append(file_path)
        except OSError:
            continue

    h5_files = sorted(set(h5_files))
    if not h5_files:
        raise FileNotFoundError(f"No HDF5 files found in {path}")
    return h5_files


class H5TriangleDataset(Dataset):
    """
    Minimal H5 loader for triangle-level RenderFormer data.

    Each view in an H5 file is treated as one training sample.
    """

    def __init__(self, data_path: str | Path, render_resolution: int = 64):
        ensure_h5py_available()
        self.files = find_h5_files(data_path)
        self.render_resolution = render_resolution

        self.file_lengths: List[int] = []
        self.cumulative_sizes: List[int] = []

        running_total = 0
        for h5_path in self.files:
            with h5py.File(h5_path, "r") as handle:
                num_views = int(handle["c2w"].shape[0])
            self.file_lengths.append(num_views)
            running_total += num_views
            self.cumulative_sizes.append(running_total)

        self._cached_file_index = -1
        self._cached_tri_pos: Optional[torch.Tensor] = None
        self._cached_tri_normals: Optional[torch.Tensor] = None
        self._cached_tri_patches: Optional[torch.Tensor] = None
        self._cached_tri_mask: Optional[torch.Tensor] = None
        self._cached_c2w: Optional[np.ndarray] = None
        self._cached_fov: Optional[np.ndarray] = None
        self._cached_images: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def _generate_rays(self, c2w: torch.Tensor, fov_deg: float, resolution: int) -> torch.Tensor:
        height = resolution
        width = resolution
        focal = 0.5 * width / np.tan(0.5 * float(fov_deg) * np.pi / 180.0)

        i, j = torch.meshgrid(
            torch.linspace(0, width - 1, width, dtype=torch.float32),
            torch.linspace(0, height - 1, height, dtype=torch.float32),
            indexing="ij",
        )
        i = i.t()
        j = j.t()

        dirs = torch.stack(
            [
                (i - width * 0.5) / focal,
                -(j - height * 0.5) / focal,
                -torch.ones_like(i),
            ],
            dim=-1,
        )

        rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], dim=-1)
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True).clamp_min(1e-6)
        return rays_d

    def _load_file_into_cache(self, file_index: int) -> None:
        h5_path = self.files[file_index]
        with h5py.File(h5_path, "r") as handle:
            triangles = np.array(handle["triangles"], dtype=np.float32)
            texture = np.array(handle["texture"], dtype=np.float32)
            normals = np.array(handle["vn"], dtype=np.float32)
            c2w_all = np.array(handle["c2w"], dtype=np.float32)
            fov_all = np.array(handle["fov"], dtype=np.float32)
            images = np.array(handle["img"], dtype=np.float32) if "img" in handle else None

        num_triangles = triangles.shape[0]
        self._cached_tri_pos = torch.from_numpy(triangles.reshape(num_triangles, 9)).unsqueeze(0)
        self._cached_tri_normals = torch.from_numpy(normals.reshape(num_triangles, 9)).unsqueeze(0)
        self._cached_tri_patches = torch.from_numpy(texture).unsqueeze(0)
        self._cached_tri_mask = torch.ones(1, num_triangles, dtype=torch.bool)
        self._cached_c2w = c2w_all
        self._cached_fov = fov_all
        self._cached_images = images
        self._cached_file_index = file_index

    def __getitem__(self, index: int) -> Dict[str, Any]:
        file_index = bisect.bisect_right(self.cumulative_sizes, index)
        if file_index != self._cached_file_index:
            self._load_file_into_cache(file_index)

        prev_total = 0 if file_index == 0 else self.cumulative_sizes[file_index - 1]
        view_index = index - prev_total

        c2w = torch.from_numpy(self._cached_c2w[view_index]).float()
        fov = float(np.asarray(self._cached_fov[view_index]).reshape(-1)[0])
        ray_map = self._generate_rays(c2w, fov, self.render_resolution)

        if self._cached_images is None:
            gt_image = torch.zeros(3, self.render_resolution, self.render_resolution, dtype=torch.float32)
        else:
            image = torch.from_numpy(self._cached_images[view_index][..., :3]).float().permute(2, 0, 1)
            if image.shape[-2:] != (self.render_resolution, self.render_resolution):
                image = F.interpolate(
                    image.unsqueeze(0),
                    size=(self.render_resolution, self.render_resolution),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            gt_image = image

        sample_name = f"{self.files[file_index].name}_view_{view_index:04d}"
        return {
            "tri_pos": self._cached_tri_pos.clone(),
            "tri_normals": self._cached_tri_normals.clone(),
            "tri_patches": self._cached_tri_patches.clone(),
            "tri_mask": self._cached_tri_mask.clone(),
            "obj_mask": torch.ones(1, dtype=torch.bool),
            "c2w": c2w,
            "camera_o": c2w[:3, -1].clone(),
            "ray_map": ray_map,
            "gt_image": gt_image,
            "dataset_name": "renderformer_h5",
            "dataset_idx": file_index,
            "sample_name": sample_name,
            "sample_idx": view_index,
        }


def _flatten_structured_views(batch_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    flat_batch: List[Dict[str, Any]] = []
    for sample in batch_list:
        gt_image = sample.get("gt_image")
        if isinstance(gt_image, torch.Tensor) and gt_image.ndim == 4:
            num_views = gt_image.shape[0]
            for view_index in range(num_views):
                sub_sample = dict(sample)
                sub_sample["gt_image"] = gt_image[view_index]
                sub_sample["ray_map"] = _as_tensor(sample["ray_map"])[view_index]
                sub_sample["camera_o"] = _as_tensor(sample["camera_o"])[view_index]

                c2w = sample.get("c2w")
                if c2w is not None:
                    c2w_tensor = _as_tensor(c2w)
                    if c2w_tensor.ndim == 3 and c2w_tensor.shape[0] == num_views:
                        sub_sample["c2w"] = c2w_tensor[view_index]

                sample_name = str(sample.get("sample_name", f"sample_{len(flat_batch)}"))
                sub_sample["sample_name"] = f"{sample_name}::view_{view_index:02d}"
                sub_sample["sample_idx"] = int(sample.get("sample_idx", view_index))
                flat_batch.append(sub_sample)
        else:
            flat_batch.append(sample)
    return flat_batch


def _collate_prepacked_triangle_batch(batch_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    batch_size = len(batch_list)
    max_objects = max(int(item["tri_pos"].shape[0]) for item in batch_list)
    max_triangles = max(int(item["tri_pos"].shape[1]) for item in batch_list)

    pos_dtype = _as_tensor(batch_list[0]["tri_pos"]).dtype
    padded_pos = torch.zeros(batch_size, max_objects, max_triangles, 9, dtype=pos_dtype)

    any_normals = any(item.get("tri_normals") is not None for item in batch_list)
    padded_normals = (
        torch.zeros(batch_size, max_objects, max_triangles, 9, dtype=pos_dtype)
        if any_normals
        else None
    )

    any_patches = any(item.get("tri_patches") is not None for item in batch_list)
    padded_patches = None
    patch_channels = 0
    patch_size = 0
    if any_patches:
        for item in batch_list:
            tri_patches = item.get("tri_patches")
            if tri_patches is None:
                continue
            tri_patches = _as_tensor(tri_patches)
            patch_channels = max(patch_channels, int(tri_patches.shape[2]))
            patch_size = max(patch_size, int(tri_patches.shape[-1]))
        padded_patches = torch.zeros(
            batch_size,
            max_objects,
            max_triangles,
            patch_channels,
            patch_size,
            patch_size,
            dtype=pos_dtype,
        )

    obj_mask = torch.zeros(batch_size, max_objects, dtype=torch.bool)
    tri_mask = torch.zeros(batch_size, max_objects, max_triangles, dtype=torch.bool)

    for batch_index, item in enumerate(batch_list):
        tri_pos = _as_tensor(item["tri_pos"])
        object_count, triangle_count = tri_pos.shape[:2]
        padded_pos[batch_index, :object_count, :triangle_count] = tri_pos
        obj_mask[batch_index, :object_count] = True

        local_tri_mask = item.get("tri_mask")
        if local_tri_mask is None:
            tri_mask[batch_index, :object_count, :triangle_count] = True
        else:
            local_tri_mask = _as_tensor(local_tri_mask, dtype=None).to(dtype=torch.bool)
            tri_mask[batch_index, :object_count, :triangle_count] = local_tri_mask

        if padded_normals is not None and item.get("tri_normals") is not None:
            tri_normals = _as_tensor(item["tri_normals"])
            padded_normals[batch_index, :object_count, :triangle_count] = tri_normals

        if padded_patches is not None and item.get("tri_patches") is not None:
            tri_patches = _as_tensor(item["tri_patches"])
            channels = tri_patches.shape[2]
            patch_h = tri_patches.shape[-2]
            patch_w = tri_patches.shape[-1]
            patch_tensor = tri_patches.reshape(-1, channels, patch_h, patch_w)
            if patch_tensor.shape[-2:] != (patch_size, patch_size):
                patch_tensor = F.interpolate(
                    patch_tensor,
                    size=(patch_size, patch_size),
                    mode="bilinear",
                    align_corners=False,
                )
            patch_tensor = patch_tensor.reshape(object_count, triangle_count, channels, patch_size, patch_size)
            padded_patches[batch_index, :object_count, :triangle_count, :channels] = patch_tensor

    return {
        "tri_pos": padded_pos,
        "tri_normals": padded_normals,
        "tri_patches": padded_patches,
        "tri_mask": tri_mask,
        "obj_mask": obj_mask,
        "c2w": torch.stack([_as_tensor(item["c2w"]) for item in batch_list]),
        "camera_o": torch.stack([_as_tensor(item["camera_o"]) for item in batch_list]),
        "ray_map": torch.stack([_as_tensor(item["ray_map"]) for item in batch_list]),
        "gt_image": torch.stack([_as_tensor(item["gt_image"]) for item in batch_list]),
        "sample_name": [str(item.get("sample_name", f"sample_{idx}")) for idx, item in enumerate(batch_list)],
        "sample_idx": torch.tensor([int(item.get("sample_idx", idx)) for idx, item in enumerate(batch_list)]),
    }


def _collate_structured_scene_batch(batch_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    batch_list = _flatten_structured_views(batch_list)
    batch_size = len(batch_list)

    max_objects = max(len(item["scene_objects_pos"]) for item in batch_list)
    max_triangles = max(obj.shape[0] for item in batch_list for obj in item["scene_objects_pos"])

    padded_pos = torch.zeros(batch_size, max_objects, max_triangles, 9, dtype=torch.float32)
    any_normals = any(
        any(norm is not None for norm in item.get("scene_objects_normals", []))
        for item in batch_list
    )
    padded_normals = (
        torch.zeros(batch_size, max_objects, max_triangles, 9, dtype=torch.float32)
        if any_normals
        else None
    )

    any_patches = any(
        any(patch is not None for patch in item.get("scene_objects_patches", []))
        for item in batch_list
    )
    padded_patches = None
    patch_channels = 0
    patch_size = 0
    if any_patches:
        for item in batch_list:
            for patch in item.get("scene_objects_patches", []):
                if patch is None:
                    continue
                patch = _as_tensor(patch)
                patch_channels = max(patch_channels, int(patch.shape[1]))
                patch_size = max(patch_size, int(patch.shape[-1]))

        padded_patches = torch.zeros(
            batch_size,
            max_objects,
            max_triangles,
            patch_channels,
            patch_size,
            patch_size,
            dtype=torch.float32,
        )

    obj_mask = torch.zeros(batch_size, max_objects, dtype=torch.bool)
    tri_mask = torch.zeros(batch_size, max_objects, max_triangles, dtype=torch.bool)

    for batch_index, item in enumerate(batch_list):
        positions = item["scene_objects_pos"]
        patches = item.get("scene_objects_patches", [None] * len(positions))
        normals = item.get("scene_objects_normals", [None] * len(positions))

        for object_index, tri_pos in enumerate(positions):
            tri_pos = _as_tensor(tri_pos)
            triangle_count = int(tri_pos.shape[0])
            padded_pos[batch_index, object_index, :triangle_count] = tri_pos
            obj_mask[batch_index, object_index] = True
            tri_mask[batch_index, object_index, :triangle_count] = True

            tri_normals = normals[object_index] if object_index < len(normals) else None
            if padded_normals is not None and tri_normals is not None:
                tri_normals = _as_tensor(tri_normals)
                padded_normals[batch_index, object_index, :triangle_count] = tri_normals

            tri_patches = patches[object_index] if object_index < len(patches) else None
            if padded_patches is not None and tri_patches is not None:
                tri_patches = _as_tensor(tri_patches)
                channels = int(tri_patches.shape[1])
                patch_h = int(tri_patches.shape[-2])
                patch_w = int(tri_patches.shape[-1])
                patch_tensor = tri_patches.reshape(-1, channels, patch_h, patch_w)
                if patch_tensor.shape[-2:] != (patch_size, patch_size):
                    patch_tensor = F.interpolate(
                        patch_tensor,
                        size=(patch_size, patch_size),
                        mode="bilinear",
                        align_corners=False,
                    )
                patch_tensor = patch_tensor.reshape(tri_patches.shape[0], channels, patch_size, patch_size)
                usable_patches = min(int(patch_tensor.shape[0]), triangle_count)
                padded_patches[batch_index, object_index, :usable_patches, :channels] = patch_tensor[:usable_patches]

    return {
        "tri_pos": padded_pos,
        "tri_normals": padded_normals,
        "tri_patches": padded_patches,
        "tri_mask": tri_mask,
        "obj_mask": obj_mask,
        "c2w": torch.stack([_as_tensor(item["c2w"]) for item in batch_list]),
        "camera_o": torch.stack([_as_tensor(item["camera_o"]) for item in batch_list]),
        "ray_map": torch.stack([_as_tensor(item["ray_map"]) for item in batch_list]),
        "gt_image": torch.stack([_as_tensor(item["gt_image"]) for item in batch_list]),
        "sample_name": [str(item.get("sample_name", f"sample_{idx}")) for idx, item in enumerate(batch_list)],
        "sample_idx": torch.tensor([int(item.get("sample_idx", idx)) for idx, item in enumerate(batch_list)]),
    }


def renderformer_baseline_collate(batch_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    first_item = batch_list[0]
    if "tri_pos" in first_item:
        return _collate_prepacked_triangle_batch(batch_list)
    if "scene_objects_pos" in first_item:
        return _collate_structured_scene_batch(batch_list)
    raise KeyError("Unsupported sample format for RenderFormer baseline.")
