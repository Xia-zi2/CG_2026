from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from local_tex_utils import extract_texture_patches


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".exr", ".npy", ".pt"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build training-ready PT samples from Blender-exported OBJ geometry, rendered images, and camera JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--obj_dir", type=str, required=True, help="Directory containing one OBJ per object.")
    parser.add_argument("--camera_json", type=str, required=True, help="Camera export JSON or transforms.json.")
    parser.add_argument("--image_root", type=str, required=True, help="Directory containing rendered images.")
    parser.add_argument("--out_dir", type=str, required=True, help="Output dataset root.")
    parser.add_argument("--materials_json", type=str, default=None, help="Optional object-level materials.json. If omitted, tries to auto-detect it next to obj_dir.")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--scene_name", type=str, default="blender_scene")
    parser.add_argument("--obj_glob", type=str, default="*.obj", help="Pattern used to collect OBJ files.")
    parser.add_argument("--image_size", type=int, default=128, help="Resize rendered images to this square resolution.")
    parser.add_argument("--texture_patch_size", type=int, default=8, help="Per-triangle patch size.")
    parser.add_argument("--default_image_ext", type=str, default=".png", help="Used when camera JSON omits image extension.")
    parser.add_argument("--default_metallic", type=float, default=0.0)
    parser.add_argument("--default_roughness", type=float, default=1.0)
    parser.add_argument("--linearize_texture_srgb", action="store_true")
    parser.add_argument("--linearize_gt_srgb", action="store_true")
    parser.add_argument("--flip_v", action=argparse.BooleanOptionalAction, default=True, help="Flip OBJ V texture coordinate before sampling.")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--save_preview", action="store_true", help="Also save resized preview PNGs.")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _to_tensor_image(image_np: np.ndarray, linearize_srgb: bool) -> torch.Tensor:
    original_dtype = image_np.dtype
    if image_np.ndim == 2:
        image_np = np.repeat(image_np[..., None], 3, axis=-1)
    if image_np.shape[-1] == 4:
        image_np = image_np[..., :3]

    image_np = image_np.astype(np.float32)
    if np.issubdtype(original_dtype, np.integer):
        image_np = image_np / 255.0
    if linearize_srgb:
        image_np = np.power(np.clip(image_np, 0.0, 1.0), 2.2)
    return torch.from_numpy(image_np).permute(2, 0, 1).contiguous()


def load_exr_image_raw(image_path: Path) -> np.ndarray:
    try:
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        import cv2  # type: ignore

        image_np = cv2.imread(
            str(image_path),
            cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED,
        )
        if image_np is None:
            raise ValueError(f"cv2.imread returned None for {image_path}")
        if image_np.ndim == 3 and image_np.shape[-1] >= 3:
            image_np = image_np[..., :3][:, :, ::-1]
        return image_np
    except ImportError:
        pass
    except Exception as exc:
        raise RuntimeError(f"Failed to read EXR with OpenCV: {image_path}") from exc

    try:
        import imageio.v3 as iio
    except ImportError as exc:
        raise ImportError(
            "Reading .exr requires OpenCV or imageio with an EXR-capable backend. "
            "Install opencv-python, or install an imageio EXR backend."
        ) from exc

    try:
        return iio.imread(image_path)
    except Exception as exc:
        raise RuntimeError(
            "Failed to read EXR. Install OpenCV and use cv2.IMREAD_ANYDEPTH/IMREAD_UNCHANGED "
            "for raw linear EXR loading."
        ) from exc


def load_image_tensor(image_path: Path, linearize_srgb: bool) -> torch.Tensor:
    suffix = image_path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}:
        image_np = np.array(Image.open(image_path).convert("RGB"))
        return _to_tensor_image(image_np, linearize_srgb=linearize_srgb)
    if suffix == ".npy":
        image_np = np.load(image_path)
        return _to_tensor_image(image_np, linearize_srgb=linearize_srgb)
    if suffix == ".pt":
        tensor = torch.load(image_path, map_location="cpu", weights_only=False)
        if isinstance(tensor, dict):
            tensor = tensor["image"]
        if tensor.ndim == 3 and tensor.shape[0] in {1, 3, 4}:
            tensor = tensor[:3].float()
        elif tensor.ndim == 3 and tensor.shape[-1] in {1, 3, 4}:
            tensor = tensor[..., :3].permute(2, 0, 1).float()
        else:
            raise ValueError(f"Unsupported tensor image shape: {tuple(tensor.shape)}")
        if linearize_srgb:
            tensor = torch.pow(torch.clamp(tensor, 0.0, 1.0), 2.2)
        return tensor.contiguous()
    if suffix == ".exr":
        image_np = load_exr_image_raw(image_path)
        return _to_tensor_image(image_np, linearize_srgb=False)

    raise ValueError(f"Unsupported image format: {image_path}")


def resize_image(image: torch.Tensor, size: int) -> torch.Tensor:
    if image.shape[-2:] == (size, size):
        return image
    return F.interpolate(image.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False).squeeze(0)


def parse_floats(tokens: List[str]) -> List[float]:
    return [float(token) for token in tokens]


def to_color_tensor(value: Any, default: Any) -> torch.Tensor:
    if value is None:
        value = default
    if isinstance(value, torch.Tensor):
        tensor = value.detach().clone().float().reshape(-1)
    elif isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value).float().reshape(-1)
    elif isinstance(value, (list, tuple)):
        tensor = torch.tensor(list(value), dtype=torch.float32).reshape(-1)
    else:
        tensor = torch.tensor([float(value)], dtype=torch.float32)
    if tensor.numel() == 0:
        tensor = torch.tensor([0.0], dtype=torch.float32)
    if tensor.numel() == 1:
        tensor = tensor.repeat(3)
    return tensor[:3].contiguous()


def to_scalar_float(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return float(default)
        return float(value.reshape(-1)[0].item())
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return float(default)
        return float(value.reshape(-1)[0])
    if isinstance(value, (list, tuple)):
        if not value:
            return float(default)
        return float(value[0])
    return float(value)


def resolve_materials_json_path(materials_json: Optional[str], obj_dir: Path) -> Optional[Path]:
    if materials_json is not None:
        path = Path(materials_json).resolve()
        if not path.exists():
            raise FileNotFoundError(f"materials_json not found: {path}")
        return path

    candidates = [
        obj_dir / "materials.json",
        obj_dir.parent / "materials.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def load_object_material_overrides(
    materials_json_path: Optional[Path],
    default_metallic: float,
    default_roughness: float,
) -> Dict[str, Dict[str, Any]]:
    if materials_json_path is None:
        return {}

    raw = load_json(materials_json_path)
    if not isinstance(raw, dict):
        raise ValueError(f"materials.json must be a JSON object, got {type(raw).__name__}")

    overrides: Dict[str, Dict[str, Any]] = {}
    for object_name, material_raw in raw.items():
        if not isinstance(material_raw, dict):
            continue

        overrides[str(object_name)] = {
            "diffuse": to_color_tensor(
                material_raw.get("base_color", material_raw.get("diffuse")),
                default=[1.0, 1.0, 1.0],
            ),
            # Historical note: this code path stores specular-like RGB values
            # in the "metallic" slot because the downstream baseline uses that convention.
            "metallic": to_color_tensor(
                material_raw.get("specular_color", material_raw.get("metallic")),
                default=float(default_metallic),
            ),
            "roughness": to_scalar_float(material_raw.get("roughness"), default=float(default_roughness)),
            "emission": to_color_tensor(material_raw.get("emission"), default=[0.0, 0.0, 0.0]),
        }

    return overrides


def parse_material_library(mtl_path: Path, default_metallic: float, default_roughness: float) -> Dict[str, Dict[str, Any]]:
    materials: Dict[str, Dict[str, Any]] = {}
    current_name: Optional[str] = None

    if not mtl_path.exists():
        return materials

    with mtl_path.open("r", encoding="utf-8", errors="ignore") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            tag = parts[0]
            values = parts[1:]

            if tag == "newmtl":
                current_name = " ".join(values)
                materials[current_name] = {
                    "diffuse": torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
                    "emission": torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
                    "metallic": float(default_metallic),
                    "roughness": float(default_roughness),
                    "map_Kd": None,
                }
                continue

            if current_name is None:
                continue

            material = materials[current_name]
            if tag == "Kd":
                material["diffuse"] = torch.tensor(parse_floats(values[:3]), dtype=torch.float32)
            elif tag == "Ke":
                material["emission"] = torch.tensor(parse_floats(values[:3]), dtype=torch.float32)
            elif tag == "Pm":
                material["metallic"] = float(values[0])
            elif tag == "Pr":
                material["roughness"] = float(values[0])
            elif tag == "Ns":
                ns = float(values[0])
                material["roughness"] = float(np.sqrt(2.0 / max(ns + 2.0, 1e-6)))
            elif tag == "map_Kd":
                texture_name = " ".join(values)
                material["map_Kd"] = (mtl_path.parent / texture_name).resolve()

    return materials


def normalize_index(index: int, length: int) -> int:
    if index > 0:
        return index - 1
    if index < 0:
        return length + index
    raise ValueError("OBJ indices are 1-based; zero is invalid.")


def triangulate_face(face_tokens: List[str], num_v: int, num_vt: int, num_vn: int) -> List[Tuple[List[int], List[int], List[int]]]:
    parsed: List[Tuple[int, int, int]] = []
    for token in face_tokens:
        parts = token.split("/")
        v_idx = normalize_index(int(parts[0]), num_v)
        vt_idx = normalize_index(int(parts[1]), num_vt) if len(parts) > 1 and parts[1] else -1
        vn_idx = normalize_index(int(parts[2]), num_vn) if len(parts) > 2 and parts[2] else -1
        parsed.append((v_idx, vt_idx, vn_idx))

    triangles: List[Tuple[List[int], List[int], List[int]]] = []
    for offset in range(1, len(parsed) - 1):
        tri = [parsed[0], parsed[offset], parsed[offset + 1]]
        triangles.append(
            (
                [vertex[0] for vertex in tri],
                [vertex[1] for vertex in tri],
                [vertex[2] for vertex in tri],
            )
        )
    return triangles


def compute_face_normals(tri_pos: torch.Tensor) -> torch.Tensor:
    vertices = tri_pos.view(-1, 3, 3)
    edge1 = vertices[:, 1] - vertices[:, 0]
    edge2 = vertices[:, 2] - vertices[:, 0]
    normals = torch.cross(edge1, edge2, dim=1)
    normals = F.normalize(normals, dim=1)
    return normals.repeat(1, 3)


def parse_obj_geometry(
    obj_path: Path,
    texture_patch_size: int,
    default_metallic: float,
    default_roughness: float,
    linearize_texture_srgb: bool,
    flip_v: bool,
    object_material_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    vertices: List[List[float]] = []
    texcoords: List[List[float]] = []
    normals: List[List[float]] = []
    face_vertex_indices: List[List[int]] = []
    face_texcoord_indices: List[List[int]] = []
    face_normal_indices: List[List[int]] = []
    face_material_names: List[str] = []
    mtllib_name: Optional[str] = None
    current_material = "__default__"

    with obj_path.open("r", encoding="utf-8", errors="ignore") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            tag = parts[0]
            values = parts[1:]

            if tag == "v":
                vertices.append(parse_floats(values[:3]))
            elif tag == "vt":
                texcoords.append(parse_floats(values[:2]))
            elif tag == "vn":
                normals.append(parse_floats(values[:3]))
            elif tag == "mtllib":
                mtllib_name = " ".join(values)
            elif tag == "usemtl":
                current_material = " ".join(values)
            elif tag == "f":
                triangles = triangulate_face(values, len(vertices), len(texcoords), len(normals))
                for tri_v, tri_vt, tri_vn in triangles:
                    face_vertex_indices.append(tri_v)
                    face_texcoord_indices.append(tri_vt)
                    face_normal_indices.append(tri_vn)
                    face_material_names.append(current_material)

    if not face_vertex_indices:
        raise ValueError(f"No faces found in OBJ: {obj_path}")

    vertices_tensor = torch.tensor(vertices, dtype=torch.float32)
    tri_vertex_indices = torch.tensor(face_vertex_indices, dtype=torch.long)
    tri_vertices = vertices_tensor[tri_vertex_indices].reshape(-1, 9)

    tri_normals: Optional[torch.Tensor] = None
    if normals and all(min(indices) >= 0 for indices in face_normal_indices):
        normals_tensor = torch.tensor(normals, dtype=torch.float32)
        tri_normal_indices = torch.tensor(face_normal_indices, dtype=torch.long)
        tri_normals = normals_tensor[tri_normal_indices].reshape(-1, 9)
    if tri_normals is None:
        tri_normals = compute_face_normals(tri_vertices)

    tri_uvs: Optional[torch.Tensor] = None
    if texcoords and all(min(indices) >= 0 for indices in face_texcoord_indices):
        texcoords_tensor = torch.tensor(texcoords, dtype=torch.float32)
        tri_texcoord_indices = torch.tensor(face_texcoord_indices, dtype=torch.long)
        tri_uvs = texcoords_tensor[tri_texcoord_indices].reshape(-1, 3, 2)
        if flip_v:
            tri_uvs = tri_uvs.clone()
            tri_uvs[..., 1] = 1.0 - tri_uvs[..., 1]

    material_table: Dict[str, Dict[str, Any]] = {
        "__default__": {
            "diffuse": torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
            "emission": torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
            "metallic": float(default_metallic),
            "roughness": float(default_roughness),
            "map_Kd": None,
        }
    }
    if mtllib_name is not None:
        mtl_path = (obj_path.parent / mtllib_name).resolve()
        material_table.update(
            parse_material_library(
                mtl_path,
                default_metallic=default_metallic,
                default_roughness=default_roughness,
            )
        )

    if object_material_override is not None:
        for material_name, material in list(material_table.items()):
            merged_material = dict(material)
            for key in ("diffuse", "metallic", "roughness", "emission"):
                if key in object_material_override:
                    merged_material[key] = object_material_override[key]
            material_table[material_name] = merged_material

    tri_patches = build_material_patches(
        tri_uvs=tri_uvs,
        material_names=face_material_names,
        material_table=material_table,
        patch_size=texture_patch_size,
        linearize_texture_srgb=linearize_texture_srgb,
    )

    return {
        "name": obj_path.stem,
        "tri_pos": tri_vertices.contiguous(),
        "tri_normals": tri_normals.contiguous(),
        "tri_patches": tri_patches.contiguous(),
        "material_names": face_material_names,
    }


def build_material_patches(
    tri_uvs: Optional[torch.Tensor],
    material_names: List[str],
    material_table: Dict[str, Dict[str, Any]],
    patch_size: int,
    linearize_texture_srgb: bool,
) -> torch.Tensor:
    num_triangles = len(material_names)
    diffuse = torch.zeros(num_triangles, 3, patch_size, patch_size, dtype=torch.float32)
    metallic = torch.zeros(num_triangles, 3, patch_size, patch_size, dtype=torch.float32)
    roughness = torch.ones(num_triangles, 1, patch_size, patch_size, dtype=torch.float32)
    normal = torch.tensor([0.5, 0.5, 1.0], dtype=torch.float32).view(1, 3, 1, 1).repeat(num_triangles, 1, patch_size, patch_size)
    emission = torch.zeros(num_triangles, 3, patch_size, patch_size, dtype=torch.float32)

    texture_groups: Dict[str, List[int]] = {}

    for tri_index, material_name in enumerate(material_names):
        material = material_table.get(material_name, material_table["__default__"])

        metallic_value = to_color_tensor(material.get("metallic"), default=0.0)
        roughness_value = to_scalar_float(material.get("roughness"), default=1.0)
        emission_color = to_color_tensor(material.get("emission"), default=[0.0, 0.0, 0.0])

        metallic[tri_index] = metallic_value.view(3, 1, 1)
        roughness[tri_index] = torch.tensor([roughness_value], dtype=torch.float32).view(1, 1, 1)
        emission[tri_index] = emission_color.view(3, 1, 1)

        texture_path = material.get("map_Kd")
        if tri_uvs is not None and texture_path is not None and Path(texture_path).exists():
            texture_groups.setdefault(str(texture_path), []).append(tri_index)
        else:
            diffuse_color = to_color_tensor(material.get("diffuse"), default=[1.0, 1.0, 1.0])
            diffuse[tri_index] = diffuse_color.view(3, 1, 1)

    for texture_path_str, tri_indices in texture_groups.items():
        texture_path = Path(texture_path_str)
        texture_tensor = load_image_tensor(texture_path, linearize_srgb=linearize_texture_srgb).unsqueeze(0)
        uv_tensor = tri_uvs[tri_indices]
        texture_patches = extract_texture_patches(texture_tensor, uv_tensor, patch_size=patch_size)

        for local_index, tri_index in enumerate(tri_indices):
            material = material_table.get(material_names[tri_index], material_table["__default__"])
            diffuse_scale = to_color_tensor(material.get("diffuse"), default=[1.0, 1.0, 1.0]).view(3, 1, 1)
            diffuse[tri_index] = texture_patches[local_index] * diffuse_scale

    return torch.cat([diffuse, metallic, roughness, normal, emission], dim=1)


def resolve_image_path(frame: Dict[str, Any], image_root: Path, default_image_ext: str) -> Path:
    candidate = (
        frame.get("image_path")
        or frame.get("file_path")
        or frame.get("rgb_path")
        or frame.get("image")
        or frame.get("path")
    )
    if candidate is None:
        raise KeyError("Frame does not contain image_path/file_path/rgb_path.")

    candidate_path = Path(candidate)
    if candidate_path.suffix == "":
        candidate_path = candidate_path.with_suffix(default_image_ext)

    if candidate_path.is_absolute():
        image_path = candidate_path
    else:
        image_path = (image_root / candidate_path).resolve()

    if image_path.exists():
        return image_path

    fallback = (image_root / candidate_path.name).resolve()
    if fallback.exists():
        return fallback

    raise FileNotFoundError(f"Could not resolve image path for frame: {candidate}")


def angle_to_radians(value: float) -> float:
    if value <= 2.0 * math.pi + 1e-4:
        return value
    return math.radians(value)


def ensure_c2w(matrix_like: Any) -> torch.Tensor:
    matrix = torch.tensor(matrix_like, dtype=torch.float32)
    if matrix.shape == (3, 4):
        bottom = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
        matrix = torch.cat([matrix, bottom], dim=0)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected c2w to be 4x4 or 3x4, got {tuple(matrix.shape)}")
    return matrix


def load_frames(camera_json_path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    raw = load_json(camera_json_path)
    if isinstance(raw, dict) and "frames" in raw:
        frames = raw["frames"]
        meta = dict(raw)
        meta.pop("frames", None)
        return frames, meta
    if isinstance(raw, list):
        return raw, {}
    raise ValueError("Unsupported camera JSON structure.")


def build_intrinsics(frame: Dict[str, Any], meta: Dict[str, Any], width: int, height: int, target_size: int) -> Dict[str, float]:
    def get_value(*keys):
        for key in keys:
            if key in frame:
                return frame[key]
            if key in meta:
                return meta[key]
        return None

    fx = get_value("fl_x", "fx")
    fy = get_value("fl_y", "fy")
    cx = get_value("cx")
    cy = get_value("cy")

    if fx is None:
        angle_x = get_value("camera_angle_x", "fov_x", "camera_fov_x", "fov")
        if angle_x is None:
            raise KeyError("Camera JSON must provide fl_x/fx or camera_angle_x/fov.")
        angle_x = angle_to_radians(float(angle_x))
        fx = 0.5 * width / math.tan(0.5 * angle_x)
    if fy is None:
        angle_y = get_value("camera_angle_y", "fov_y", "camera_fov_y")
        if angle_y is not None:
            angle_y = angle_to_radians(float(angle_y))
            fy = 0.5 * height / math.tan(0.5 * angle_y)
        else:
            fy = float(fx)

    if cx is None:
        cx = 0.5 * width
    if cy is None:
        cy = 0.5 * height

    scale_x = target_size / float(width)
    scale_y = target_size / float(height)
    return {
        "fx": float(fx) * scale_x,
        "fy": float(fy) * scale_y,
        "cx": float(cx) * scale_x,
        "cy": float(cy) * scale_y,
    }


def build_ray_map(c2w: torch.Tensor, intrinsics: Dict[str, float], image_size: int) -> torch.Tensor:
    width = image_size
    height = image_size

    i, j = torch.meshgrid(
        torch.arange(width, dtype=torch.float32),
        torch.arange(height, dtype=torch.float32),
        indexing="xy",
    )

    dirs = torch.stack(
        [
            (i - intrinsics["cx"]) / intrinsics["fx"],
            -(j - intrinsics["cy"]) / intrinsics["fy"],
            -torch.ones_like(i),
        ],
        dim=-1,
    )

    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], dim=-1)
    rays_d = F.normalize(rays_d, dim=-1)
    return rays_d.contiguous()


def build_dataset_samples(args: argparse.Namespace) -> None:
    obj_dir = Path(args.obj_dir).resolve()
    camera_json_path = Path(args.camera_json).resolve()
    image_root = Path(args.image_root).resolve()
    output_dir = Path(args.out_dir).resolve() / args.split

    ensure_dir(output_dir)

    obj_paths = sorted(obj_dir.glob(args.obj_glob))
    if not obj_paths:
        raise FileNotFoundError(f"No OBJ files found in {obj_dir} with pattern {args.obj_glob}")

    materials_json_path = resolve_materials_json_path(args.materials_json, obj_dir=obj_dir)
    object_material_overrides = load_object_material_overrides(
        materials_json_path=materials_json_path,
        default_metallic=args.default_metallic,
        default_roughness=args.default_roughness,
    )

    print(f"[Build] Found {len(obj_paths)} OBJ files.")
    if materials_json_path is not None:
        print(f"[Build] Using object materials from {materials_json_path}")
    objects = [
        parse_obj_geometry(
            obj_path=obj_path,
            texture_patch_size=args.texture_patch_size,
            default_metallic=args.default_metallic,
            default_roughness=args.default_roughness,
            linearize_texture_srgb=args.linearize_texture_srgb,
            flip_v=args.flip_v,
            object_material_override=object_material_overrides.get(obj_path.stem),
        )
        for obj_path in obj_paths
    ]

    scene_objects_pos = [obj["tri_pos"].cpu() for obj in objects]
    scene_objects_normals = [obj["tri_normals"].cpu() for obj in objects]
    scene_objects_patches = [obj["tri_patches"].cpu() for obj in objects]
    object_names = [obj["name"] for obj in objects]

    frames, meta = load_frames(camera_json_path)
    if args.max_frames is not None:
        frames = frames[: args.max_frames]

    print(f"[Build] Found {len(frames)} camera frames.")

    for sample_index, frame in enumerate(frames):
        image_path = resolve_image_path(frame, image_root=image_root, default_image_ext=args.default_image_ext)
        gt_image = load_image_tensor(image_path, linearize_srgb=args.linearize_gt_srgb)
        original_height, original_width = int(gt_image.shape[1]), int(gt_image.shape[2])
        gt_image = resize_image(gt_image, args.image_size).cpu()

        c2w = ensure_c2w(frame.get("transform_matrix") or frame.get("c2w") or frame.get("camera_to_world")).cpu()
        intrinsics = build_intrinsics(
            frame=frame,
            meta=meta,
            width=original_width,
            height=original_height,
            target_size=args.image_size,
        )
        ray_map = build_ray_map(c2w, intrinsics=intrinsics, image_size=args.image_size).cpu()
        camera_o = c2w[:3, 3].clone().cpu()

        sample = {
            "scene_objects_pos": scene_objects_pos,
            "scene_objects_normals": scene_objects_normals,
            "scene_objects_patches": scene_objects_patches,
            "photons": {},
            "gt_image": gt_image,
            "ray_map": ray_map,
            "camera_o": camera_o,
            "c2w": c2w,
            "dataset_name": args.scene_name,
            "dataset_idx": 0,
            "sample_name": image_path.stem,
            "sample_idx": sample_index,
            "scene_object_names": object_names,
            "intrinsics": intrinsics,
        }

        out_path = output_dir / f"{sample_index:05d}.pt"
        torch.save(sample, out_path)

        if args.save_preview:
            preview = torch.clamp(gt_image, 0.0, 1.0)
            preview = torch.pow(preview, 1.0 / 2.2)
            preview_np = (preview.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
            Image.fromarray(preview_np).save(out_path.with_suffix(".png"))

    summary = {
        "scene_name": args.scene_name,
        "num_objects": len(objects),
        "object_names": object_names,
        "num_samples": len(frames),
        "image_size": args.image_size,
        "texture_patch_size": args.texture_patch_size,
        "camera_json": str(camera_json_path),
        "image_root": str(image_root),
        "obj_dir": str(obj_dir),
        "materials_json": str(materials_json_path) if materials_json_path is not None else None,
    }
    with (output_dir / "dataset_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=True)

    print(f"[Build] Saved {len(frames)} PT samples to {output_dir}")


def main() -> None:
    args = parse_args()
    build_dataset_samples(args)


if __name__ == "__main__":
    main()
