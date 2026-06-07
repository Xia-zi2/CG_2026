#!/usr/bin/env python3
"""FastAPI bridge from the SuperSplat physics panel to PhysGaussian.

The backend intentionally keeps the first version close to stock PhysGaussian:
the UI sends object selections and simulation options, this service writes a
PhysGaussian JSON config, runs the simulator, and exposes a SuperSplat motion
package for offline/asynchronous playback.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
import zipfile
import math
import hashlib
import signal
import functools
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHYSGAUSSIAN_ROOT = REPO_ROOT.parent / "physgaussian-src"
PHYSGAUSSIAN_ROOT = Path(os.environ.get("PHYSGAUSSIAN_ROOT", DEFAULT_PHYSGAUSSIAN_ROOT)).resolve()
PYTHON_BIN = os.environ.get("PYTHON_BIN", sys.executable)
WORK_ROOT = Path(os.environ.get("PHYS_WORK_ROOT", REPO_ROOT / ".phys_backend")).resolve()
SUPERSPLAT_DIST = Path(os.environ.get("SUPERSPLAT_DIST", REPO_ROOT / "supersplat-dist")).resolve()
MODELS_ROOT = WORK_ROOT / "models"
RUNS_ROOT = WORK_ROOT / "runs"

MODELS_ROOT.mkdir(parents=True, exist_ok=True)
RUNS_ROOT.mkdir(parents=True, exist_ok=True)

DEFAULT_FRAME_NUM = 30
RUN_PROCESSES: dict[str, subprocess.Popen] = {}

app = FastAPI(title="USTC26MM PhysGaussian bridge")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def no_cache_for_live_app(request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path.startswith("/models/") or re.match(r"^/api/runs/[^/]+/(motion|indices|proxy_motion|proxy_skinning)\.bin$", path):
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        if "Pragma" in response.headers:
            del response.headers["Pragma"]
        if "Expires" in response.headers:
            del response.headers["Expires"]
    else:
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


app.mount("/models", StaticFiles(directory=MODELS_ROOT), name="models")
app.mount("/outputs", StaticFiles(directory=RUNS_ROOT), name="outputs")


MATERIAL_DEFAULTS: dict[str, dict[str, Any]] = {
    "jelly": {"material": "jelly", "E": 1e5, "nu": 0.3, "density": 200},
    "metal": {
        "material": "metal",
        "E": 2e6,
        "nu": 0.3,
        "density": 2700,
        "yield_stress": 1e2,
        "hardening": 0.0,
        "xi": 0.0,
    },
    "sand": {
        "material": "sand",
        "E": 1e5,
        "nu": 0.2,
        "density": 1500,
        "friction_angle": 30.0,
    },
    "foam": {
        "material": "foam",
        "E": 1e4,
        "nu": 0.1,
        "density": 80,
        "yield_stress": 1e2,
        "plastic_viscosity": 0.0,
    },
    "snow": {"material": "snow", "E": 1.4e5, "nu": 0.2, "density": 400},
    "plasticine": {
        "material": "plasticine",
        "E": 4e4,
        "nu": 0.35,
        "density": 1300,
        "yield_stress": 1e2,
        "softening": 0.1,
        "hardening": 0.0,
        "xi": 0.0,
    },
    # Stock PhysGaussian has no rigid material law. In v1 rigid means a very
    # stiff jelly-like elastoplastic object; true finite-mass rigid bodies are a
    # later solver extension.
    "rigid": {"material": "jelly", "E": 1e7, "nu": 0.25, "density": 1000},
    # Kept under the legacy key used by the UI. It now means fixed anchor
    # particles, not a collider material.
    "obstacle": {"material": "jelly", "E": 2e6, "nu": 0.4, "density": 200},
}

MATERIAL_PARAM_KEYS = (
    "yield_stress",
    "hardening",
    "xi",
    "friction_angle",
    "plastic_viscosity",
    "softening",
)

OFFICIAL_CONFIG_BY_MODEL_FOLDER = {
    "bread-trained": "tear_bread_config.json",
    "ficus_whitebg-trained": "ficus_config.json",
    "pillow2sofa_whitebg-trained": "pillow2sofa_config.json",
    "plane-trained": "plane_config.json",
    "vasedeck_whitebg-trained": "vasedeck_config.json",
    "wolf_whitebg-trained": "wolf_config.json",
}


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _read_json(path: Path) -> Any:
    last_error: Exception | None = None
    for _ in range(5):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            last_error = exc
            time.sleep(0.05)
    raise last_error if last_error else RuntimeError(f"failed to read json: {path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_ply_vertex_props(path: Path, wanted: set[str]) -> dict[str, list[float]]:
    with path.open("rb") as stream:
        header_lines: list[str] = []
        while True:
            line = stream.readline()
            if not line:
                raise ValueError("invalid ply: missing end_header")
            text = line.decode("ascii", errors="ignore").strip()
            header_lines.append(text)
            if text == "end_header":
                break
        vertex_count = 0
        properties: list[tuple[str, str]] = []
        in_vertex = False
        fmt = "ascii"
        for text in header_lines:
            parts = text.split()
            if len(parts) >= 2 and parts[0] == "format":
                fmt = parts[1]
            elif len(parts) >= 3 and parts[0] == "element":
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
            elif in_vertex and len(parts) >= 3 and parts[0] == "property":
                properties.append((parts[1], parts[2]))
        names = [name for _type, name in properties]
        result = {name: [] for name in wanted if name in names}
        if fmt.startswith("ascii"):
            for _ in range(vertex_count):
                values = stream.readline().decode("ascii", errors="ignore").split()
                for name in result:
                    result[name].append(float(values[names.index(name)]))
            return result
        if fmt != "binary_little_endian":
            raise ValueError(f"unsupported ply format: {fmt}")
        try:
            import numpy as np

            type_map_np = {
                "float": "<f4",
                "float32": "<f4",
                "double": "<f8",
                "float64": "<f8",
                "uchar": "u1",
                "uint8": "u1",
                "char": "i1",
                "int8": "i1",
                "ushort": "<u2",
                "uint16": "<u2",
                "short": "<i2",
                "int16": "<i2",
                "uint": "<u4",
                "uint32": "<u4",
                "int": "<i4",
                "int32": "<i4",
            }
            dtype = np.dtype([(name, type_map_np[prop_type]) for prop_type, name in properties])
            data = np.fromfile(stream, dtype=dtype, count=vertex_count)
            return {name: data[name].astype(np.float32, copy=False).tolist() for name in result}
        except Exception:
            stream.seek(0)
            while stream.readline().decode("ascii", errors="ignore").strip() != "end_header":
                pass
        import struct

        type_map = {
            "float": ("f", 4),
            "float32": ("f", 4),
            "double": ("d", 8),
            "float64": ("d", 8),
            "uchar": ("B", 1),
            "uint8": ("B", 1),
            "char": ("b", 1),
            "int8": ("b", 1),
            "ushort": ("H", 2),
            "uint16": ("H", 2),
            "short": ("h", 2),
            "int16": ("h", 2),
            "uint": ("I", 4),
            "uint32": ("I", 4),
            "int": ("i", 4),
            "int32": ("i", 4),
        }
        struct_fmt = "<" + "".join(type_map[prop_type][0] for prop_type, _name in properties)
        stride = sum(type_map[prop_type][1] for prop_type, _name in properties)
        for _ in range(vertex_count):
            row = struct.unpack(struct_fmt, stream.read(stride))
            for name in result:
                result[name].append(float(row[names.index(name)]))
        return result


@functools.lru_cache(maxsize=8)
def _read_ply_splat_props_cached(path_text: str, stat_key: tuple[int, int]) -> dict[str, Any]:
    path = Path(path_text)
    props = _read_ply_vertex_props(
        path,
        {
            "x",
            "y",
            "z",
            "rot_0",
            "rot_1",
            "rot_2",
            "rot_3",
            "scale_0",
            "scale_1",
            "scale_2",
        },
    )
    if not all(name in props for name in ("x", "y", "z")):
        raise ValueError("ply has no x/y/z vertex properties")
    num_splats = len(props["x"])
    x = props["x"]
    y = props["y"]
    z = props["z"]
    rot_0 = props.get("rot_0") or [1.0] * num_splats
    rot_1 = props.get("rot_1") or [0.0] * num_splats
    rot_2 = props.get("rot_2") or [0.0] * num_splats
    rot_3 = props.get("rot_3") or [0.0] * num_splats
    scale_0 = props.get("scale_0") or [0.0] * num_splats
    scale_1 = props.get("scale_1") or [0.0] * num_splats
    scale_2 = props.get("scale_2") or [0.0] * num_splats
    positions = [[x[i], y[i], z[i]] for i in range(num_splats)]
    rotations = [
        [
            rot_0[i],
            rot_1[i],
            rot_2[i],
            rot_3[i],
        ]
        for i in range(num_splats)
    ]
    scales = [
        [
            scale_0[i],
            scale_1[i],
            scale_2[i],
        ]
        for i in range(num_splats)
    ]
    return {
        "positions": positions,
        "rotations": rotations,
        "scales": scales,
    }


def _read_ply_splat_props(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return _read_ply_splat_props_cached(str(path.resolve()), (stat.st_size, stat.st_mtime_ns))


def _read_ply_xyz(path: Path) -> list[list[float]]:
    return _read_ply_splat_props(path)["positions"]


def _log(message: str) -> None:
    print(f"[phys-backend] {message}", flush=True)


def _safe_float(value: Any, fallback: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    return result if result == result else fallback


def _safe_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _normalize_integrator_name(value: Any) -> str:
    name = str(value or "explicit").strip().lower().replace("-", "_")
    if name in ("implicit", "implicit_mpm"):
        return "implicit"
    if name in ("explicit", "explicit_mpm", "mpm"):
        return "explicit"
    if name in ("pbmpm", "pb_mpm"):
        return "pbmpm"
    raise ValueError(f"Unknown integrator '{value}'")


def _solver_to_integrator(solver: Any) -> str:
    return _normalize_integrator_name(str(solver or "explicit-mpm"))


def _pbmpm_config_from_simulation(simulation: dict[str, Any]) -> dict[str, Any]:
    pbmpm_sim = simulation.get("pbmpm") if isinstance(simulation.get("pbmpm"), dict) else {}
    config: dict[str, Any] = {}

    strength_raw = _first_present(
        pbmpm_sim.get("strength_scale"),
        pbmpm_sim.get("stiffness_scale"),
        simulation.get("pbmpmStrengthScale"),
        simulation.get("pbmpmStiffnessScale"),
    )
    if strength_raw is not None:
        strength_value = max(_safe_float(strength_raw, 1.0), 1e-6)
        if abs(strength_value - 1.0) > 1e-12:
            config["strength_scale"] = strength_value

    n_min_raw = _first_present(
        pbmpm_sim.get("n_min"),
        pbmpm_sim.get("N_min"),
        pbmpm_sim.get("nMin"),
        simulation.get("pbmpmNMin"),
    )
    n_max_raw = _first_present(
        pbmpm_sim.get("n_max"),
        pbmpm_sim.get("N_max"),
        pbmpm_sim.get("nMax"),
        simulation.get("pbmpmNMax"),
    )
    n_min = max(1, _safe_int(n_min_raw, 3))
    n_max = max(n_min, _safe_int(n_max_raw, 25))
    config["n_min"] = n_min
    config["n_max"] = n_max

    plastic_mode_raw = _first_present(
        pbmpm_sim.get("plastic_mode"),
        simulation.get("pbmpmPlasticMode"),
    )
    config["plastic_mode"] = 1 if _safe_int(plastic_mode_raw, 0) == 1 else 0

    yield_min = _safe_float(
        _first_present(pbmpm_sim.get("yield_min"), simulation.get("pbmpmYieldMin")),
        0.55,
    )
    yield_max = _safe_float(
        _first_present(pbmpm_sim.get("yield_max"), simulation.get("pbmpmYieldMax")),
        1.85,
    )
    config["yield_min"] = yield_min
    config["yield_max"] = max(yield_min + 1e-4, yield_max)

    return config


def _simulation_damping(simulation: dict[str, Any], fallback: float = 1.0) -> float:
    raw = _first_present(
        simulation.get("damping"),
        simulation.get("gridDamping"),
        simulation.get("grid_v_damping_scale"),
    )
    return max(0.0, min(_safe_float(raw, fallback), 1.0))


def _config_damping(config: dict[str, Any], fallback: float = 0.9999) -> float:
    raw = _first_present(
        config.get("grid_v_damping_scale"),
        config.get("damping"),
        fallback,
    )
    return max(0.0, min(_safe_float(raw, fallback), 1.0))


def _safe_vec3(value: Any, fallback: list[float] | None = None) -> list[float]:
    base = fallback if fallback is not None else [0.0, 0.0, 0.0]
    if isinstance(value, (int, float)):
        return [0.0, 0.0, _safe_float(value, base[2])]
    if isinstance(value, list) and len(value) >= 3:
        return [_safe_float(value[i], base[i]) for i in range(3)]
    return base[:]


def _simulation_gravity(simulation: dict[str, Any]) -> list[float]:
    if not simulation.get("gravityEnabled"):
        return [0.0, 0.0, 0.0]
    return _safe_vec3(simulation.get("gravity"), [0.0, 0.0, 0.0])


def _model_record_path(model_id: str) -> Path:
    return MODELS_ROOT / model_id / "model.json"


def _run_record_path(run_id: str) -> Path:
    return RUNS_ROOT / run_id / "run.json"


def _read_ply_vertex_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        header_lines: list[str] = []
        while True:
            line = stream.readline()
            if not line:
                raise ValueError("invalid ply: missing end_header")
            text = line.decode("ascii", errors="ignore").strip()
            header_lines.append(text)
            if text == "end_header":
                break

    vertex_count = 0
    properties: list[tuple[str, str]] = []
    in_vertex = False
    fmt = "ascii"
    for text in header_lines:
        parts = text.split()
        if len(parts) >= 2 and parts[0] == "format":
            fmt = parts[1]
        elif len(parts) >= 3 and parts[0] == "element":
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                vertex_count = int(parts[2])
        elif in_vertex and len(parts) >= 3 and parts[0] == "property" and parts[1] != "list":
            properties.append((parts[1], parts[2]))
    return {"format": fmt, "vertexCount": vertex_count, "properties": properties}


@functools.lru_cache(maxsize=32)
def _read_ply_vertex_property_names_cached(path_text: str, stat_key: tuple[int, int]) -> tuple[str, ...]:
    header = _read_ply_vertex_header(Path(path_text))
    return tuple(name for _type, name in header["properties"])


def _read_ply_vertex_property_names(path: Path) -> tuple[str, ...]:
    stat = path.stat()
    return _read_ply_vertex_property_names_cached(str(path.resolve()), (stat.st_size, stat.st_mtime_ns))


def _physgaussian_ply_compatibility(path: Path, sh_degree: int = 3) -> dict[str, Any]:
    names = set(_read_ply_vertex_property_names(path))
    required_core = {
        "x",
        "y",
        "z",
        "opacity",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    }
    missing = sorted(name for name in required_core if name not in names)
    expected_f_rest = 3 * (sh_degree + 1) ** 2 - 3
    f_rest_count = sum(1 for name in names if re.fullmatch(r"f_rest_\d+", name))
    ok = not missing and f_rest_count == expected_f_rest
    return {
        "ok": ok,
        "missing": missing,
        "fRestCount": f_rest_count,
        "expectedFRestCount": expected_f_rest,
    }


def _reject_if_not_physgaussian_checkpoint(model: dict[str, Any]) -> None:
    base_ply = Path(model["basePly"]) if model.get("basePly") else None
    if not base_ply or not base_ply.exists():
        raise HTTPException(status_code=400, detail="this model has no point_cloud.ply for PhysGaussian simulation")
    compatibility = _physgaussian_ply_compatibility(base_ply)
    if compatibility["ok"]:
        return
    detail = (
        "该模型不能运行完整 PhysGaussian 仿真：point_cloud.ply 不是完整训练检查点。"
        f"当前 f_rest_* 数量为 {compatibility['fRestCount']}，"
        f"PhysGaussian 默认 SH degree=3 需要 {compatibility['expectedFRestCount']} 个；"
        "请改用 ficus-sample-7000 / vasedeck-sample-7000 等官方完整模型，"
        "或只用该模型做 proxy 快速预览。"
    )
    if compatibility["missing"]:
        detail += f" 缺少核心属性：{', '.join(compatibility['missing'])}。"
    _log(f"reject simulate: incompatible ply {base_ply} ({detail})")
    raise HTTPException(status_code=400, detail=detail)


def _load_model(model_id: str) -> dict[str, Any]:
    path = _model_record_path(model_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"unknown modelId: {model_id}")
    return _with_official_info(_read_json(path))


def _official_config_for_model_root(model_root: Path | None) -> Path | None:
    if not model_root:
        return None
    config_name = OFFICIAL_CONFIG_BY_MODEL_FOLDER.get(model_root.name)
    if not config_name:
        return None
    path = PHYSGAUSSIAN_ROOT / "config" / config_name
    return path if path.exists() else None


def _match_official_config_for_ply(ply_path: Path) -> Path | None:
    if not ply_path.exists():
        return None
    size = ply_path.stat().st_size
    source_hash: str | None = None
    official_root = PHYSGAUSSIAN_ROOT / "model"
    for model_folder, config_name in OFFICIAL_CONFIG_BY_MODEL_FOLDER.items():
        folder = official_root / model_folder / "point_cloud"
        if not folder.exists():
            continue
        for candidate in folder.glob("iteration_*/point_cloud.ply"):
            if candidate.stat().st_size != size:
                continue
            if source_hash is None:
                source_hash = _sha256(ply_path)
            if _sha256(candidate) == source_hash:
                config_path = PHYSGAUSSIAN_ROOT / "config" / config_name
                return config_path if config_path.exists() else None
    return None


def _official_config_values(path: Path) -> dict[str, Any]:
    config = _read_json(path)
    gravity = _safe_vec3(config.get("g"), [0.0, 0.0, 0.0])
    damping = _config_damping(config)
    return {
        "name": path.name,
        "preprocessing": {
            "opacity_threshold": config.get("opacity_threshold", 0.02),
            "rotation_degree": config.get("rotation_degree", [0]),
            "rotation_axis": config.get("rotation_axis", [0]),
            "sim_area": config.get("sim_area"),
            "scale": config.get("scale", 1.0),
            "n_grid": config.get("n_grid", 50),
        },
        "simulation": {
            "gravityEnabled": any(abs(value) > 1e-12 for value in gravity),
            "gravity": gravity,
            "frame_dt": config.get("frame_dt", 2e-2),
            "frame_num": DEFAULT_FRAME_NUM,
            "substep_dt": config.get("substep_dt", 1e-4),
            "damping": damping,
            "grid_v_damping_scale": damping,
            "rpic_damping": _safe_float(config.get("rpic_damping"), 0.0),
        },
        "material": {
            "material": config.get("material", "jelly"),
            "E": config.get("E"),
            "nu": config.get("nu"),
            "density": config.get("density"),
            **{
                key: config.get(key)
                for key in MATERIAL_PARAM_KEYS
                if key in config
            },
            "additional_material_params": config.get("additional_material_params", []),
        },
        "boundary_conditions": config.get("boundary_conditions", []),
        "camera": {
            "mpm_space_vertical_upward_axis": config.get("mpm_space_vertical_upward_axis"),
            "mpm_space_viewpoint_center": config.get("mpm_space_viewpoint_center"),
            "default_camera_index": config.get("default_camera_index"),
            "init_azimuthm": config.get("init_azimuthm"),
            "init_elevation": config.get("init_elevation"),
            "init_radius": config.get("init_radius"),
            "move_camera": config.get("move_camera"),
        },
    }


def _with_official_info(record: dict[str, Any]) -> dict[str, Any]:
    result = dict(record)
    config_path = result.get("officialConfig")
    if not config_path:
        result["officialConfigAvailable"] = False
        return result
    path = Path(config_path).resolve()
    result["officialConfigAvailable"] = path.exists()
    if path.exists():
        result["officialConfigName"] = path.name
        result["officialConfigValues"] = _official_config_values(path)
    return result


def _update_run(run_id: str, **patch: Any) -> dict[str, Any]:
    path = _run_record_path(run_id)
    record = _read_json(path) if path.exists() else {"runId": run_id}
    record.update(patch)
    _write_json(path, record)
    return record


def _motion_progress(run_id: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    run_dir = RUNS_ROOT / run_id
    motion_dir = run_dir / "physgaussian" / "super_motion"
    binary = motion_dir / "motion.bin"
    manifest = motion_dir / "motion.physmotion.json"
    indices = motion_dir / "indices.bin"
    proxy_binary = motion_dir / "proxy_motion.bin"
    proxy_skinning = motion_dir / "proxy_skinning.bin"
    result: dict[str, Any] = {
        "motionBytes": binary.stat().st_size if binary.exists() else 0,
        "proxyMotionBytes": proxy_binary.stat().st_size if proxy_binary.exists() else 0,
        "proxySkinningBytes": proxy_skinning.stat().st_size if proxy_skinning.exists() else 0,
        "manifestReady": manifest.exists(),
        "indicesReady": indices.exists(),
    }
    record_path = _run_record_path(run_id)
    record = _read_json(record_path) if record_path.exists() else {}
    model = _load_model(record["modelId"]) if record.get("modelId") else {}
    base_ply = Path(model["basePly"]) if model.get("basePly") else None
    if manifest.exists() and indices.exists():
        result["result"] = {
            "manifestUrl": _url_under(RUNS_ROOT, manifest, "/outputs"),
            "binaryUrl": f"/api/runs/{run_id}/motion.bin",
            "indicesUrl": f"/api/runs/{run_id}/indices.bin",
            "basePlyUrl": _model_base_ply_url(model, base_ply),
            "basePlyName": base_ply.name if base_ply else None,
        }
        if proxy_binary.exists() and proxy_skinning.exists():
            result["result"]["proxyMotionUrl"] = f"/api/runs/{run_id}/proxy_motion.bin"
            result["result"]["proxySkinningUrl"] = f"/api/runs/{run_id}/proxy_skinning.bin"
    if manifest.exists():
        try:
            manifest_data = _read_json(manifest)
            stride = _safe_int(manifest_data.get("frameStrideBytes"), 0)
            if stride > 0:
                result["availableFrames"] = result["motionBytes"] // stride
                result["frameCount"] = _safe_int(manifest_data.get("frameCount"), 0)
            result["attributes"] = manifest_data.get("attributes", [])
        except Exception:
            pass
    elif config:
        frame_count = _safe_int(config.get("frame_num"), DEFAULT_FRAME_NUM) + 1
        result["frameCount"] = frame_count
    return result


def _extract_zip_safely(zip_path: Path, destination: Path) -> None:
    root = destination.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            target = (destination / info.filename).resolve()
            if target != root and root not in target.parents:
                raise HTTPException(status_code=400, detail="zip contains an unsafe path")
            archive.extract(info, destination)


def _iteration_number(path: Path) -> int:
    match = re.search(r"iteration_(\d+)", path.as_posix())
    return int(match.group(1)) if match else -1


def _base_ply_iteration(model: dict[str, Any]) -> int:
    base_ply = model.get("basePly")
    if not base_ply:
        return -1
    return _iteration_number(Path(base_ply))


def _find_model_root(folder: Path) -> Path | None:
    candidates = [folder]
    candidates.extend(path.parent for path in folder.rglob("point_cloud") if path.is_dir())
    for candidate in candidates:
        point_cloud = candidate / "point_cloud"
        if any(point_cloud.glob("iteration_*/point_cloud.ply")):
            return candidate
    return None


def _find_base_ply(model_root: Path | None, fallback_folder: Path) -> Path | None:
    if model_root:
        files = sorted(
            (model_root / "point_cloud").glob("iteration_*/point_cloud.ply"),
            key=_iteration_number,
        )
        if files:
            return files[-1]
    files = sorted(fallback_folder.rglob("*.ply"), key=lambda path: len(path.parts))
    return files[0] if files else None


def _url_under(root: Path, path: Path, prefix: str) -> str:
    return f"{prefix}/{path.relative_to(root).as_posix()}"


def _model_base_ply_url(model: dict[str, Any], base_ply: Path | None) -> str | None:
    if model.get("basePlyUrl"):
        return str(model["basePlyUrl"])
    if not base_ply:
        return None
    try:
        return _url_under(MODELS_ROOT, base_ply, "/models")
    except ValueError:
        return None


def _sim_area_from_payload(payload: dict[str, Any]) -> list[float] | None:
    preprocessing = payload.get("preprocessing") or {}
    sim_area = preprocessing.get("sim_area")
    if isinstance(sim_area, list) and len(sim_area) == 6:
        return [_safe_float(v, 0.0) for v in sim_area]

    objects = payload.get("objects") or []
    mins = [float("inf"), float("inf"), float("inf")]
    maxs = [float("-inf"), float("-inf"), float("-inf")]
    found = False
    for obj in objects:
        aabb = obj.get("aabbWorld") or {}
        lo = aabb.get("min")
        hi = aabb.get("max")
        if not (isinstance(lo, list) and isinstance(hi, list) and len(lo) == 3 and len(hi) == 3):
            continue
        found = True
        for axis in range(3):
            mins[axis] = min(mins[axis], _safe_float(lo[axis], mins[axis]))
            maxs[axis] = max(maxs[axis], _safe_float(hi[axis], maxs[axis]))
    if not found:
        return None
    return [mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2]]


def _aabb_to_mpm_box(aabb: dict[str, Any], sim_area: list[float], scale: float) -> dict[str, list[float]] | None:
    lo = aabb.get("min")
    hi = aabb.get("max")
    if not (isinstance(lo, list) and isinstance(hi, list) and len(lo) == 3 and len(hi) == 3):
        return None
    sim_min = [sim_area[0], sim_area[2], sim_area[4]]
    sim_max = [sim_area[1], sim_area[3], sim_area[5]]
    center = [(sim_min[i] + sim_max[i]) * 0.5 for i in range(3)]
    max_diff = max(sim_max[i] - sim_min[i] for i in range(3))
    if max_diff <= 1e-12:
        return None
    factor = scale / max_diff
    obj_min = [_safe_float(lo[i], 0.0) for i in range(3)]
    obj_max = [_safe_float(hi[i], 0.0) for i in range(3)]
    point = [((obj_min[i] + obj_max[i]) * 0.5 - center[i]) * factor + 1.0 for i in range(3)]
    size = [max((obj_max[i] - obj_min[i]) * 0.5 * factor, 1e-4) for i in range(3)]
    return {"point": point, "size": size}


def _grid_node_stats(bounds: dict[str, list[float]], n_grid: int, grid_lim: float) -> dict[str, Any]:
    dx = grid_lim / max(n_grid, 1)
    counts: list[int] = []
    index_min: list[int] = []
    index_max: list[int] = []
    for axis in range(3):
        first = -1
        last = -1
        count = 0
        for index in range(max(n_grid, 0)):
            coord = index * dx
            if bounds["min"][axis] < coord < bounds["max"][axis]:
                if first < 0:
                    first = index
                last = index
                count += 1
        counts.append(count)
        index_min.append(0 if first < 0 else first)
        index_max.append(last)
    return {
        "gridDx": dx,
        "estimatedGridNodeCounts": counts,
        "estimatedGridNodeCount": counts[0] * counts[1] * counts[2],
        "gridIndexBounds": {"min": index_min, "max": index_max},
    }


def _aabb_to_velocity_cuboid_constraint(
    obj: dict[str, Any],
    sim_area: list[float],
    scale: float,
    n_grid: int,
    substep_dt: float,
    grid_lim: float = 2.0,
) -> dict[str, Any] | None:
    aabb = obj.get("aabbWorld") or {}
    lo = aabb.get("min")
    hi = aabb.get("max")
    if not (isinstance(lo, list) and isinstance(hi, list) and len(lo) == 3 and len(hi) == 3):
        return None

    sim_min = [sim_area[0], sim_area[2], sim_area[4]]
    sim_max = [sim_area[1], sim_area[3], sim_area[5]]
    center = [(sim_min[i] + sim_max[i]) * 0.5 for i in range(3)]
    max_diff = max(sim_max[i] - sim_min[i] for i in range(3))
    if max_diff <= 1e-12:
        return None

    factor = scale / max_diff
    obj_min = [_safe_float(lo[i], 0.0) for i in range(3)]
    obj_max = [_safe_float(hi[i], 0.0) for i in range(3)]
    mpm_min = [(min(obj_min[i], obj_max[i]) - center[i]) * factor + 1.0 for i in range(3)]
    mpm_max = [(max(obj_min[i], obj_max[i]) - center[i]) * factor + 1.0 for i in range(3)]

    padding_grid_nodes = max(0, _safe_int(obj.get("cuboidPaddingGridNodes"), 2))
    dx = grid_lim / max(n_grid, 1)
    padding = padding_grid_nodes * dx
    padded_min = [mpm_min[i] - padding for i in range(3)]
    padded_max = [mpm_max[i] + padding for i in range(3)]
    for axis in range(3):
        cuboid_center = (padded_min[axis] + padded_max[axis]) * 0.5
        min_half = max(dx * padding_grid_nodes, dx * 1.01)
        if (padded_max[axis] - padded_min[axis]) * 0.5 < min_half:
            padded_min[axis] = cuboid_center - min_half
            padded_max[axis] = cuboid_center + min_half

    padded_bounds = {"min": padded_min, "max": padded_max}
    stats = _grid_node_stats(padded_bounds, n_grid, grid_lim)
    drive = _drive_params(obj)
    velocity_source = drive["prescribedVelocity"]
    has_linear_drive = drive["linearEnabled"] and any(abs(_safe_float(value, 0.0)) > 1e-12 for value in velocity_source)
    start_time = drive["linearStart"] if has_linear_drive else 0
    end_time = start_time + max(1, drive["linearNumDt"]) * substep_dt if has_linear_drive else 1e3
    velocity = velocity_source if has_linear_drive else [0, 0, 0]
    point = [(padded_min[i] + padded_max[i]) * 0.5 for i in range(3)]
    size = [max((padded_max[i] - padded_min[i]) * 0.5, dx * 1.01) for i in range(3)]

    return {
        "type": "cuboid",
        "constraint_type": "set_velocity_on_cuboid",
        "point": point,
        "size": size,
        "scale": [2.0 * value for value in size],
        "velocity": velocity,
        "start_time": start_time,
        "end_time": end_time,
        "reset": 1,
        "_ui_role": "grid_velocity_cuboid_anchor",
        "_ui_body_id": _body_id(obj),
        "_ui_object_id": obj.get("objectId"),
        "_ui_coordinate_space": "mpm_shifted_normalized",
        "_ui_scene_bounds": {"min": obj_min, "max": obj_max},
        "_ui_sim_area": sim_area[:],
        "_ui_mpm_bounds": {"min": mpm_min, "max": mpm_max},
        "_ui_padded_mpm_bounds": padded_bounds,
        "_ui_padding": padding,
        "_ui_padding_grid_nodes": padding_grid_nodes,
        **stats,
    }


def _body_id(obj: dict[str, Any]) -> int:
    return _safe_int(obj.get("bodyId"), _safe_int(obj.get("objectId"), 0))


def _is_obstacle_part(obj: dict[str, Any]) -> bool:
    material = obj.get("material", "")
    material_name = material.get("material", "") if isinstance(material, dict) else material
    return str(material_name).lower() in {"obstacle", "anchor", "fixed"} or str(obj.get("mode", "")).lower() in {"obstacle", "anchor", "fixed"}


def _as_mixed_body_rigid_part(obj: dict[str, Any]) -> dict[str, Any]:
    """Legacy fallback kept for old run records; fixed anchors are not demoted."""
    rigid = MATERIAL_DEFAULTS["rigid"]
    return {
        **obj,
        "material": "rigid",
        "mode": "rigid-soft",
        "E": rigid["E"],
        "nu": rigid["nu"],
        "density": rigid["density"],
        "_ui_obstacle_demoted_to_mpm": True,
    }


def _union_aabb(objects: list[dict[str, Any]]) -> dict[str, list[float]] | None:
    mins = [float("inf"), float("inf"), float("inf")]
    maxs = [float("-inf"), float("-inf"), float("-inf")]
    found = False
    for obj in objects:
        aabb = obj.get("aabbWorld") or {}
        lo = aabb.get("min")
        hi = aabb.get("max")
        if not (isinstance(lo, list) and isinstance(hi, list) and len(lo) == 3 and len(hi) == 3):
            continue
        found = True
        for axis in range(3):
            mins[axis] = min(mins[axis], _safe_float(lo[axis], mins[axis]))
            maxs[axis] = max(maxs[axis], _safe_float(hi[axis], maxs[axis]))
    if not found:
        return None
    return {"min": mins, "max": maxs}


def _objects_by_body(objects: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = {}
    for obj in objects:
        result.setdefault(_body_id(obj), []).append(obj)
    return result


def _object_params(obj: dict[str, Any]) -> dict[str, Any]:
    material_value = obj.get("material", "jelly")
    material_config = material_value if isinstance(material_value, dict) else {}
    material_name = material_config.get("material", "jelly") if material_config else material_value
    defaults = MATERIAL_DEFAULTS.get(str(material_name).lower(), MATERIAL_DEFAULTS["jelly"])
    source = {**material_config, **obj}
    params = {
        "material": defaults["material"],
        "E": _safe_float(source.get("E"), defaults["E"]),
        "nu": _safe_float(source.get("nu"), defaults["nu"]),
        "density": _safe_float(source.get("density"), defaults["density"]),
    }
    for key in MATERIAL_PARAM_KEYS:
        value = source.get(key, defaults.get(key))
        if value is not None:
            params[key] = _safe_float(value, _safe_float(defaults.get(key), 0.0))
    return params


def _aabb_center(obj: dict[str, Any]) -> list[float] | None:
    aabb = obj.get("aabbWorld") or {}
    lo = aabb.get("min")
    hi = aabb.get("max")
    if not (isinstance(lo, list) and isinstance(hi, list) and len(lo) == 3 and len(hi) == 3):
        return None
    return [(_safe_float(lo[i], 0.0) + _safe_float(hi[i], 0.0)) * 0.5 for i in range(3)]


def _params_for_fixed_part(part: dict[str, Any], body_parts: list[dict[str, Any]]) -> dict[str, Any]:
    movable_parts = [candidate for candidate in body_parts if not _is_obstacle_part(candidate)]
    if not movable_parts:
        return _object_params({})
    center = _aabb_center(part)
    if center is None:
        return _object_params(movable_parts[0])
    best = movable_parts[0]
    best_distance = float("inf")
    for candidate in movable_parts:
        candidate_center = _aabb_center(candidate)
        if candidate_center is None:
            continue
        distance = sum((center[axis] - candidate_center[axis]) ** 2 for axis in range(3))
        if distance < best_distance:
            best = candidate
            best_distance = distance
    return _object_params(best)


def _drive_params(obj: dict[str, Any]) -> dict[str, Any]:
    drive = obj.get("drive") or {}
    linear_force = drive.get("linearForce") if isinstance(drive.get("linearForce"), list) else [0, 0, 0]
    linear_velocity = drive.get("linearVelocity") if isinstance(drive.get("linearVelocity"), list) else None
    target_velocity = drive.get("targetVelocity") if isinstance(drive.get("targetVelocity"), list) else None
    dirichlet_velocity = drive.get("dirichletVelocity") if isinstance(drive.get("dirichletVelocity"), list) else None
    prescribed_velocity = _first_present(dirichlet_velocity, target_velocity, linear_velocity)
    return {
        "linearEnabled": bool(drive.get("linearEnabled")),
        "linearForce": linear_force,
        "linearVelocity": linear_velocity or [0, 0, 0],
        "targetVelocity": target_velocity or [0, 0, 0],
        "dirichletVelocity": dirichlet_velocity or [0, 0, 0],
        "prescribedVelocity": prescribed_velocity or linear_force,
        "hasExplicitVelocity": prescribed_velocity is not None,
        "linearNumDt": _safe_int(drive.get("linearNumDt"), 1),
        "linearStart": _safe_float(drive.get("linearStart"), 0),
        "spinEnabled": bool(drive.get("spinEnabled")),
        "spinAxis": drive.get("spinAxis") if isinstance(drive.get("spinAxis"), list) else [0, 0, 1],
        "spinAngular": _safe_float(drive.get("spinAngular"), 0),
        "spinTranslation": _safe_float(drive.get("spinTranslation"), 0),
        "spinStart": _safe_float(drive.get("spinStart"), 0),
        "spinEnd": _safe_float(drive.get("spinEnd"), 0.2),
    }


def _drive_enabled(obj: dict[str, Any]) -> bool:
    drive = obj.get("drive") or {}
    return bool(drive.get("linearEnabled") or drive.get("spinEnabled"))


def _active_objects(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # In the UI, an object entry is a material part. All assigned parts belong
    # to the simulated body set; bodyId only groups them for shared scale/fill/drive.
    return objects


def _object_indices(objects: list[dict[str, Any]]) -> list[int]:
    values: set[int] = set()
    for obj in objects:
        indices = obj.get("indices") or []
        if not isinstance(indices, list):
            continue
        for value in indices:
            index = _safe_int(value, -1)
            if index >= 0:
                values.add(index)
    return sorted(values)


def _normalize_vec3(values: Any, fallback: list[float]) -> list[float]:
    if not (isinstance(values, list) and len(values) == 3):
        values = fallback
    vec = [_safe_float(values[i], fallback[i]) for i in range(3)]
    length = math.sqrt(sum(v * v for v in vec))
    if length <= 1e-8:
        return fallback
    return [v / length for v in vec]


def _quat_normalize(q: list[float]) -> list[float]:
    length = math.sqrt(sum(value * value for value in q))
    if length <= 1e-12:
        return [1.0, 0.0, 0.0, 0.0]
    return [value / length for value in q]


def _quat_mul(a: list[float], b: list[float]) -> list[float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return _quat_normalize(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def _quat_from_axis_angle(axis: list[float], angle: float) -> list[float]:
    n = _normalize_vec3(axis, [0.0, 0.0, 1.0])
    half = angle * 0.5
    s = math.sin(half)
    return _quat_normalize([math.cos(half), n[0] * s, n[1] * s, n[2] * s])


def _quat_rotate_vec(q: list[float], v: list[float]) -> list[float]:
    # q * [0, v] * q^-1, expanded to avoid per-splat allocations.
    w, x, y, z = q
    tx = 2.0 * (y * v[2] - z * v[1])
    ty = 2.0 * (z * v[0] - x * v[2])
    tz = 2.0 * (x * v[1] - y * v[0])
    return [
        v[0] + w * tx + (y * tz - z * ty),
        v[1] + w * ty + (z * tx - x * tz),
        v[2] + w * tz + (x * ty - y * tx),
    ]


def _quat_dot(a: list[float], b: list[float]) -> float:
    return sum(a[i] * b[i] for i in range(4))


def _quat_weighted_blend(quats: list[list[float]], weights: list[float]) -> list[float]:
    if not quats:
        return [1.0, 0.0, 0.0, 0.0]
    reference = quats[0]
    accum = [0.0, 0.0, 0.0, 0.0]
    for quat, weight in zip(quats, weights):
        sign = -1.0 if _quat_dot(reference, quat) < 0.0 else 1.0
        for axis in range(4):
            accum[axis] += sign * weight * quat[axis]
    return _quat_normalize(accum)


def _aabb_for_points(points: list[list[float]]) -> tuple[list[float], list[float]]:
    mins = [min(point[i] for point in points) for i in range(3)]
    maxs = [max(point[i] for point in points) for i in range(3)]
    return mins, maxs


def _proxy_skinning_weights(
    positions: list[list[float]],
    indices: list[int],
    group_initial_centers: list[list[float]],
    group_order_by_index: dict[int, int],
    blend_count: int,
    power: float,
) -> dict[int, list[tuple[int, float]]]:
    """Blend nearby proxy transforms so voxel preview does not crack at cell borders."""
    if not group_initial_centers:
        return {}
    blend_count = max(1, min(int(blend_count), min(8, len(group_initial_centers))))
    power = max(0.5, min(float(power), 6.0))
    eps = 1e-12
    result: dict[int, list[tuple[int, float]]] = {}
    for index in indices:
        point = positions[index]
        distances: list[tuple[float, int]] = []
        for group_id, center in enumerate(group_initial_centers):
            distance2 = sum((point[axis] - center[axis]) ** 2 for axis in range(3))
            distances.append((distance2, group_id))
        distances.sort(key=lambda item: item[0])
        selected = distances[:blend_count]
        own_group = group_order_by_index.get(index)
        if own_group is not None and all(group_id != own_group for _dist, group_id in selected):
            own_distance2 = sum((point[axis] - group_initial_centers[own_group][axis]) ** 2 for axis in range(3))
            if selected:
                selected[-1] = (own_distance2, own_group)
            else:
                selected = [(own_distance2, own_group)]

        raw_weights = [(distance2 + eps) ** (-0.5 * power) for distance2, _group_id in selected]
        weight_sum = max(sum(raw_weights), eps)
        result[index] = [
            (group_id, raw_weight / weight_sum)
            for raw_weight, (_distance2, group_id) in zip(raw_weights, selected)
        ]
    return result


def _voxel_groups_for_indices(
    positions: list[list[float]],
    indices: list[int],
    payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not indices:
        return [], {"groupCount": 0, "voxelSize": 0.0}

    selected = [positions[index] for index in indices]
    mins, maxs = _aabb_for_points(selected)
    extent = [maxs[i] - mins[i] for i in range(3)]
    max_extent = max(max(extent), 1e-8)
    simulation = payload.get("simulation") or {}
    preview = simulation.get("preview") if isinstance(simulation.get("preview"), dict) else {}
    target_groups = _safe_int(preview.get("targetVoxelGroups"), 256)
    target_groups = min(max(target_groups, 8), 4096)
    requested_size = _safe_float(preview.get("voxelSize"), 0.0)
    if requested_size > 0:
        voxel_size = requested_size
    else:
        divisions = max(1, round(target_groups ** (1.0 / 3.0)))
        voxel_size = max_extent / divisions
    voxel_size = max(voxel_size, max_extent / 256.0, 1e-8)

    buckets: dict[tuple[int, int, int], list[int]] = {}
    for index in indices:
        point = positions[index]
        key = tuple(int(math.floor((point[i] - mins[i]) / voxel_size)) for i in range(3))
        buckets.setdefault(key, []).append(index)

    groups: list[dict[str, Any]] = []
    for group_indices in buckets.values():
        center = [
            sum(positions[index][axis] for index in group_indices) / len(group_indices)
            for axis in range(3)
        ]
        local_min_z = min(positions[index][2] - center[2] for index in group_indices)
        groups.append(
            {
                "indices": group_indices,
                "center": center,
                "localMinZ": local_min_z,
            }
        )
    return groups, {
        "groupCount": len(groups),
        "voxelSize": voxel_size,
        "selectedCount": len(indices),
    }


def _proxy_groups_with_fixed_split(
    positions: list[list[float]],
    indices: list[int],
    fixed_indices: set[int],
    payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], set[int]]:
    movable_indices = [index for index in indices if index not in fixed_indices]
    fixed_only_indices = [index for index in indices if index in fixed_indices]
    groups: list[dict[str, Any]] = []
    fixed_group_ids: set[int] = set()
    voxel_sizes: list[float] = []

    for split_indices, is_fixed in ((movable_indices, False), (fixed_only_indices, True)):
        if not split_indices:
            continue
        split_groups, diagnostics = _voxel_groups_for_indices(positions, split_indices, payload)
        offset = len(groups)
        groups.extend(split_groups)
        if is_fixed:
            fixed_group_ids.update(range(offset, offset + len(split_groups)))
        voxel_sizes.append(_safe_float(diagnostics.get("voxelSize"), 0.0))

    voxel_size = min((value for value in voxel_sizes if value > 0.0), default=0.0)
    return groups, {
        "groupCount": len(groups),
        "voxelSize": voxel_size,
        "selectedCount": len(indices),
        "fixedGroupCount": len(fixed_group_ids),
    }, fixed_group_ids


def _proxy_grid_resolution(
    group_count: int,
    group_diagnostics: dict[str, Any],
    mins: list[float],
    maxs: list[float],
    preview: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    requested = _safe_int(preview.get("gridResolution"), 0)
    adaptive = bool(preview.get("adaptiveGrid", requested <= 0))
    min_res = max(4, min(_safe_int(preview.get("minGridResolution"), 5), 48))
    max_res = max(min_res, min(_safe_int(preview.get("maxGridResolution"), 28), 48))
    if requested > 0 and not adaptive:
        resolution = max(min_res, min(requested, max_res))
        return resolution, {
            "adaptiveGrid": False,
            "requestedGridResolution": requested,
            "gridCellScale": None,
            "estimatedGridDx": None,
        }

    max_extent = max(maxs[i] - mins[i] for i in range(3))
    max_extent = max(max_extent, 1e-6)
    proxy_spacing = _safe_float(group_diagnostics.get("voxelSize"), 0.0)
    if proxy_spacing <= 1e-8:
        divisions = max(1.0, float(max(group_count, 1)) ** (1.0 / 3.0))
        proxy_spacing = max_extent / divisions

    # A proxy particle uses quadratic B-spline support on the grid. If the grid
    # is much finer than the proxy spacing, neighboring proxies barely share
    # nodes, so drag/impulse propagation becomes a local island.
    cell_scale = max(0.75, min(_safe_float(preview.get("gridCellScale"), 1.15), 2.5))
    padded_extent = max_extent * 1.24
    res_by_spacing = int(round(padded_extent / max(proxy_spacing * cell_scale, 1e-6))) + 1
    res_by_count = int(math.ceil(max(group_count, 1) ** (1.0 / 3.0))) + 3
    resolution = max(res_by_spacing, res_by_count, min_res)
    resolution = min(resolution, max_res)
    estimated_dx = padded_extent / max(resolution - 1, 1)
    return resolution, {
        "adaptiveGrid": True,
        "requestedGridResolution": requested if requested > 0 else None,
        "gridCellScale": cell_scale,
        "estimatedGridDx": estimated_dx,
    }


def _local_preview_indices(
    positions: list[list[float]],
    center_index: int,
    max_count: int = 8000,
) -> list[int]:
    if not (0 <= center_index < len(positions)):
        return list(range(min(len(positions), max_count)))
    center = positions[center_index]
    distances = [
        (
            (point[0] - center[0]) ** 2
            + (point[1] - center[1]) ** 2
            + (point[2] - center[2]) ** 2,
            index,
        )
        for index, point in enumerate(positions)
    ]
    distances.sort(key=lambda item: item[0])
    return sorted(index for _distance, index in distances[:max_count])


def _preview_mpm_step(
    centers: list[list[float]],
    velocities: list[list[float]],
    masses: list[float],
    mins: list[float],
    maxs: list[float],
    grid_res: int,
    dt: float,
    gravity: list[float],
    ground_z: float | None,
) -> tuple[list[list[float]], list[list[float]]]:
    if not centers:
        return centers, velocities

    pad = max(max(maxs[i] - mins[i] for i in range(3)) * 0.05, 1e-6)
    lo = [mins[i] - pad for i in range(3)]
    hi = [maxs[i] + pad for i in range(3)]
    extent = [max(hi[i] - lo[i], 1e-6) for i in range(3)]
    grid_res = max(4, min(grid_res, 64))
    dims = [grid_res, grid_res, grid_res]
    node_mass: dict[tuple[int, int, int], float] = {}
    node_vel: dict[tuple[int, int, int], list[float]] = {}

    for center, velocity, mass in zip(centers, velocities, masses):
        gx = [(center[i] - lo[i]) / extent[i] * (dims[i] - 1) for i in range(3)]
        base = [math.floor(value) for value in gx]
        frac = [gx[i] - base[i] for i in range(3)]
        for ox in (0, 1):
            wx = (1.0 - frac[0]) if ox == 0 else frac[0]
            ix = min(max(base[0] + ox, 0), dims[0] - 1)
            for oy in (0, 1):
                wy = (1.0 - frac[1]) if oy == 0 else frac[1]
                iy = min(max(base[1] + oy, 0), dims[1] - 1)
                for oz in (0, 1):
                    wz = (1.0 - frac[2]) if oz == 0 else frac[2]
                    iz = min(max(base[2] + oz, 0), dims[2] - 1)
                    weight = wx * wy * wz
                    if weight <= 0.0:
                        continue
                    key = (ix, iy, iz)
                    weighted_mass = mass * weight
                    node_mass[key] = node_mass.get(key, 0.0) + weighted_mass
                    acc = node_vel.setdefault(key, [0.0, 0.0, 0.0])
                    for axis in range(3):
                        acc[axis] += velocity[axis] * weighted_mass

    for key, mass in node_mass.items():
        velocity = node_vel[key]
        inv_mass = 1.0 / max(mass, 1e-12)
        for axis in range(3):
            velocity[axis] = velocity[axis] * inv_mass + gravity[axis] * dt
        if ground_z is not None:
            z = lo[2] + (key[2] / max(dims[2] - 1, 1)) * extent[2]
            if z <= ground_z + extent[2] / max(dims[2] - 1, 1) and velocity[2] < 0.0:
                velocity[2] = 0.0

    next_centers: list[list[float]] = []
    next_velocities: list[list[float]] = []
    for center, velocity in zip(centers, velocities):
        gx = [(center[i] - lo[i]) / extent[i] * (dims[i] - 1) for i in range(3)]
        base = [math.floor(value) for value in gx]
        frac = [gx[i] - base[i] for i in range(3)]
        next_velocity = [0.0, 0.0, 0.0]
        for ox in (0, 1):
            wx = (1.0 - frac[0]) if ox == 0 else frac[0]
            ix = min(max(base[0] + ox, 0), dims[0] - 1)
            for oy in (0, 1):
                wy = (1.0 - frac[1]) if oy == 0 else frac[1]
                iy = min(max(base[1] + oy, 0), dims[1] - 1)
                for oz in (0, 1):
                    wz = (1.0 - frac[2]) if oz == 0 else frac[2]
                    iz = min(max(base[2] + oz, 0), dims[2] - 1)
                    weight = wx * wy * wz
                    grid_velocity = node_vel.get((ix, iy, iz))
                    if not grid_velocity:
                        continue
                    for axis in range(3):
                        next_velocity[axis] += grid_velocity[axis] * weight
        next_center = [center[i] + next_velocity[i] * dt for i in range(3)]
        if ground_z is not None and next_center[2] < ground_z:
            next_center[2] = ground_z
            next_velocity[2] = max(next_velocity[2], 0.0)
        next_centers.append(next_center)
        next_velocities.append(next_velocity)
    return next_centers, next_velocities


def _identity_mat3() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]


def _zero_mat3() -> list[list[float]]:
    return [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]


def _quat_from_matrix3(matrix: Any) -> list[float]:
    m00 = float(matrix[0][0])
    m01 = float(matrix[0][1])
    m02 = float(matrix[0][2])
    m10 = float(matrix[1][0])
    m11 = float(matrix[1][1])
    m12 = float(matrix[1][2])
    m20 = float(matrix[2][0])
    m21 = float(matrix[2][1])
    m22 = float(matrix[2][2])
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return _quat_normalize([
            0.25 * s,
            (m21 - m12) / s,
            (m02 - m20) / s,
            (m10 - m01) / s,
        ])
    if m00 > m11 and m00 > m22:
        s = math.sqrt(max(1.0 + m00 - m11 - m22, 1e-12)) * 2.0
        return _quat_normalize([
            (m21 - m12) / s,
            0.25 * s,
            (m01 + m10) / s,
            (m02 + m20) / s,
        ])
    if m11 > m22:
        s = math.sqrt(max(1.0 + m11 - m00 - m22, 1e-12)) * 2.0
        return _quat_normalize([
            (m02 - m20) / s,
            (m01 + m10) / s,
            0.25 * s,
            (m12 + m21) / s,
        ])
    s = math.sqrt(max(1.0 + m22 - m00 - m11, 1e-12)) * 2.0
    return _quat_normalize([
        (m10 - m01) / s,
        (m02 + m20) / s,
        (m12 + m21) / s,
        0.25 * s,
    ])


def _proxy_mpm_step(
    centers: list[list[float]],
    velocities: list[list[float]],
    affines: list[list[list[float]]],
    deformations: list[list[list[float]]],
    masses: list[float],
    youngs: list[float],
    nus: list[float],
    densities: list[float],
    mins: list[float],
    maxs: list[float],
    grid_res: int,
    dt: float,
    gravity: list[float],
    ground_z: float | None,
    drive_group: int | None = None,
    drive_velocity: list[float] | None = None,
    kinematic_groups: set[int] | None = None,
    kinematic_drive_groups: set[int] | None = None,
    damping: float = 1.0,
    stress_scale: float = 0.35,
) -> tuple[list[list[float]], list[list[float]], list[list[list[float]]], list[list[list[float]]], list[list[float]]]:
    if not centers:
        return centers, velocities, affines, deformations, []
    try:
        import numpy as np
    except Exception:
        next_centers, next_velocities = _preview_mpm_step(centers, velocities, masses, mins, maxs, grid_res, dt, gravity, ground_z)
        return next_centers, next_velocities, affines, deformations, [[1.0, 0.0, 0.0, 0.0] for _ in centers]

    x = np.asarray(centers, dtype=np.float64)
    v = np.asarray(velocities, dtype=np.float64)
    c = np.asarray(affines, dtype=np.float64)
    f = np.asarray(deformations, dtype=np.float64)
    mass = np.maximum(np.asarray(masses, dtype=np.float64), 1e-6)
    density = np.maximum(np.asarray(densities, dtype=np.float64), 1e-6)
    young = np.maximum(np.asarray(youngs, dtype=np.float64), 1.0)
    nu = np.clip(np.asarray(nus, dtype=np.float64), 0.0, 0.49)
    gravity_np = np.asarray(gravity, dtype=np.float64)
    kinematic_groups = set(kinematic_groups or ())
    kinematic_drive_groups = set(kinematic_drive_groups or ())

    if drive_group is not None and drive_velocity is not None and 0 <= drive_group < len(v):
        v[drive_group] = np.asarray(drive_velocity, dtype=np.float64)
        c[drive_group] = np.zeros((3, 3), dtype=np.float64)
        if drive_group in kinematic_groups:
            kinematic_drive_groups.add(drive_group)

    pad = max(float(max(maxs[i] - mins[i] for i in range(3))) * 0.12, 1e-5)
    lo = np.asarray([mins[i] - pad for i in range(3)], dtype=np.float64)
    hi = np.asarray([maxs[i] + pad for i in range(3)], dtype=np.float64)
    extent = np.maximum(hi - lo, 1e-5)
    grid_res = max(4, min(int(grid_res), 48))
    dims = np.asarray([grid_res, grid_res, grid_res], dtype=np.int32)
    dx = extent / np.maximum(dims - 1, 1)
    inv_dx = 1.0 / np.maximum(dx, 1e-6)
    avg_dx = float(np.mean(dx))
    identity = np.eye(3, dtype=np.float64)

    node_mass: dict[tuple[int, int, int], float] = {}
    node_momentum: dict[tuple[int, int, int], Any] = {}

    def weights_and_grads(grid_position: Any) -> tuple[list[list[float]], list[list[float]], Any]:
        base = np.floor(grid_position - 0.5).astype(np.int32)
        fx = grid_position - base
        weights = [
            [
                0.5 * (1.5 - fx[axis]) ** 2,
                0.75 - (fx[axis] - 1.0) ** 2,
                0.5 * (fx[axis] - 0.5) ** 2,
            ]
            for axis in range(3)
        ]
        derivatives = [
            [
                fx[axis] - 1.5,
                -2.0 * (fx[axis] - 1.0),
                fx[axis] - 0.5,
            ]
            for axis in range(3)
        ]
        return weights, derivatives, base

    for particle_id in range(len(x)):
        is_kinematic = particle_id in kinematic_groups
        if is_kinematic:
            c[particle_id] = np.zeros((3, 3), dtype=np.float64)
            f[particle_id] = identity
            if particle_id not in kinematic_drive_groups:
                v[particle_id] = np.zeros(3, dtype=np.float64)
        grid_position = (x[particle_id] - lo) * inv_dx
        weights, derivatives, base = weights_and_grads(grid_position)

        # Stable corotated-style proxy stress. E is normalized because the
        # preview solver uses scene units, not the calibrated Warp MPM units.
        normalized_young = min(max(float(young[particle_id]) / 2.0e6, 0.02), 5.0)
        mu = normalized_young / (2.0 * (1.0 + float(nu[particle_id])))
        lam = normalized_young * float(nu[particle_id]) / max((1.0 + float(nu[particle_id])) * (1.0 - 2.0 * float(nu[particle_id])), 1e-6)
        strain = 0.5 * (f[particle_id] + f[particle_id].T) - identity
        stress = np.zeros((3, 3), dtype=np.float64) if is_kinematic else 2.0 * mu * strain + lam * float(np.trace(strain)) * identity
        volume = mass[particle_id] / density[particle_id]

        for ox in range(3):
            ix = int(np.clip(base[0] + ox, 0, dims[0] - 1))
            wx = float(weights[0][ox])
            dwx = float(derivatives[0][ox]) * inv_dx[0]
            for oy in range(3):
                iy = int(np.clip(base[1] + oy, 0, dims[1] - 1))
                wy = float(weights[1][oy])
                dwy = float(derivatives[1][oy]) * inv_dx[1]
                for oz in range(3):
                    iz = int(np.clip(base[2] + oz, 0, dims[2] - 1))
                    wz = float(weights[2][oz])
                    dwz = float(derivatives[2][oz]) * inv_dx[2]
                    weight = wx * wy * wz
                    if weight <= 0.0:
                        continue
                    key = (ix, iy, iz)
                    node_world = lo + np.asarray([ix, iy, iz], dtype=np.float64) * dx
                    dpos = node_world - x[particle_id]
                    grad_w = np.asarray([dwx * wy * wz, wx * dwy * wz, wx * wy * dwz], dtype=np.float64)
                    momentum = weight * mass[particle_id] * (v[particle_id] + c[particle_id] @ dpos)
                    momentum += -dt * stress_scale * volume * (stress @ grad_w)
                    node_mass[key] = node_mass.get(key, 0.0) + weight * mass[particle_id]
                    if key not in node_momentum:
                        node_momentum[key] = momentum
                    else:
                        node_momentum[key] += momentum

    node_velocity: dict[tuple[int, int, int], Any] = {}
    damping = max(0.0, min(float(damping), 1.0))
    for key, momentum in node_momentum.items():
        m = max(node_mass.get(key, 0.0), 1e-12)
        grid_velocity = momentum / m
        grid_velocity = (grid_velocity + gravity_np * dt) * damping
        if ground_z is not None:
            z = float(lo[2] + key[2] * dx[2])
            if z <= ground_z + dx[2] * 1.25 and grid_velocity[2] < 0.0:
                grid_velocity[2] = 0.0
        node_velocity[key] = grid_velocity

    next_x = x.copy()
    next_v = v.copy()
    next_c = np.zeros_like(c)
    next_f = f.copy()
    rotations: list[list[float]] = []

    for particle_id in range(len(x)):
        if particle_id in kinematic_groups:
            prescribed_velocity = v[particle_id] if particle_id in kinematic_drive_groups else np.zeros(3, dtype=np.float64)
            next_c[particle_id] = np.zeros((3, 3), dtype=np.float64)
            next_f[particle_id] = identity
            next_v[particle_id] = prescribed_velocity
            next_x[particle_id] = x[particle_id] + prescribed_velocity * dt
            rotations.append([1.0, 0.0, 0.0, 0.0])
            continue
        grid_position = (x[particle_id] - lo) * inv_dx
        weights, _derivatives, base = weights_and_grads(grid_position)
        particle_velocity = np.zeros(3, dtype=np.float64)
        particle_affine = np.zeros((3, 3), dtype=np.float64)
        for ox in range(3):
            ix = int(np.clip(base[0] + ox, 0, dims[0] - 1))
            wx = float(weights[0][ox])
            for oy in range(3):
                iy = int(np.clip(base[1] + oy, 0, dims[1] - 1))
                wy = float(weights[1][oy])
                for oz in range(3):
                    iz = int(np.clip(base[2] + oz, 0, dims[2] - 1))
                    wz = float(weights[2][oz])
                    weight = wx * wy * wz
                    if weight <= 0.0:
                        continue
                    grid_velocity = node_velocity.get((ix, iy, iz))
                    if grid_velocity is None:
                        continue
                    node_world = lo + np.asarray([ix, iy, iz], dtype=np.float64) * dx
                    dpos = node_world - x[particle_id]
                    particle_velocity += weight * grid_velocity
                    particle_affine += (4.0 * weight / max(avg_dx * avg_dx, 1e-12)) * np.outer(grid_velocity, dpos)

        if drive_group is not None and drive_velocity is not None and particle_id == drive_group:
            particle_velocity = np.asarray(drive_velocity, dtype=np.float64)
            particle_affine = np.zeros((3, 3), dtype=np.float64)

        next_c[particle_id] = particle_affine
        updated_f = (identity + dt * particle_affine) @ f[particle_id]
        try:
            u, singular_values, vt = np.linalg.svd(updated_f)
            singular_values = np.clip(singular_values, 0.65, 1.55)
            updated_f = u @ np.diag(singular_values) @ vt
            rotation = u @ vt
            if np.linalg.det(rotation) < 0.0:
                u[:, -1] *= -1.0
                rotation = u @ vt
        except Exception:
            rotation = identity
        next_f[particle_id] = updated_f
        next_v[particle_id] = particle_velocity
        next_x[particle_id] = x[particle_id] + particle_velocity * dt
        if ground_z is not None and next_x[particle_id][2] < ground_z:
            next_x[particle_id][2] = ground_z
            next_v[particle_id][2] = max(float(next_v[particle_id][2]), 0.0)
        rotations.append(_quat_from_matrix3(rotation.tolist()))

    return (
        next_x.tolist(),
        next_v.tolist(),
        next_c.tolist(),
        next_f.tolist(),
        rotations,
    )


def _proxy_graph_edges(
    centers: list[list[float]],
    neighbor_count: int = 6,
) -> list[tuple[int, int]]:
    count = len(centers)
    if count <= 1:
        return []
    neighbor_count = max(1, min(neighbor_count, min(count - 1, 12)))
    edges: set[tuple[int, int]] = set()
    for index, center in enumerate(centers):
        distances = []
        for other_index, other in enumerate(centers):
            if other_index == index:
                continue
            distance_sq = (
                (center[0] - other[0]) ** 2
                + (center[1] - other[1]) ** 2
                + (center[2] - other[2]) ** 2
            )
            distances.append((distance_sq, other_index))
        distances.sort(key=lambda item: item[0])
        for _distance_sq, other_index in distances[:neighbor_count]:
            edges.add((min(index, other_index), max(index, other_index)))
    return sorted(edges)


def _spread_proxy_velocity(
    velocities: list[list[float]],
    masses: list[float],
    edges: list[tuple[int, int]],
    source_ids: set[int],
    iterations: int = 6,
    strength: float = 0.55,
) -> list[list[float]]:
    if not edges or not velocities:
        return velocities
    strength = max(0.0, min(strength, 0.95))
    iterations = max(0, min(iterations, 32))
    neighbors: list[list[int]] = [[] for _ in velocities]
    for left, right in edges:
        neighbors[left].append(right)
        neighbors[right].append(left)
    source_velocities = {source_id: velocities[source_id][:] for source_id in source_ids if 0 <= source_id < len(velocities)}
    current = [velocity[:] for velocity in velocities]
    for _ in range(iterations):
        updated = [velocity[:] for velocity in current]
        for index, local_neighbors in enumerate(neighbors):
            if index in source_velocities or not local_neighbors:
                continue
            total_mass = sum(max(masses[neighbor], 1e-6) for neighbor in local_neighbors)
            if total_mass <= 1e-12:
                continue
            average = [0.0, 0.0, 0.0]
            for neighbor in local_neighbors:
                weight = max(masses[neighbor], 1e-6) / total_mass
                for axis in range(3):
                    average[axis] += current[neighbor][axis] * weight
            for axis in range(3):
                updated[index][axis] = (1.0 - strength) * current[index][axis] + strength * average[axis]
        for source_id, source_velocity in source_velocities.items():
            updated[source_id] = source_velocity[:]
        current = updated
    return current


def _apply_proxy_shape_coupling(
    centers: list[list[float]],
    velocities: list[list[float]],
    initial_centers: list[list[float]],
    masses: list[float],
    dt: float,
    stiffness: float,
) -> list[list[float]]:
    if not centers or stiffness <= 0.0:
        return velocities
    total_mass = sum(max(mass, 1e-6) for mass in masses)
    if total_mass <= 1e-12:
        return velocities
    current_com = [
        sum(centers[index][axis] * max(masses[index], 1e-6) for index in range(len(centers))) / total_mass
        for axis in range(3)
    ]
    initial_com = [
        sum(initial_centers[index][axis] * max(masses[index], 1e-6) for index in range(len(centers))) / total_mass
        for axis in range(3)
    ]
    stiffness = max(0.0, min(stiffness, 0.95))
    inv_dt = 1.0 / max(dt, 1e-6)
    next_velocities = [velocity[:] for velocity in velocities]
    for index, center in enumerate(centers):
        target = [
            current_com[axis] + (initial_centers[index][axis] - initial_com[axis])
            for axis in range(3)
        ]
        for axis in range(3):
            next_velocities[index][axis] += (target[axis] - center[axis]) * stiffness * inv_dt
    return next_velocities


def _box_to_rotation_cylinder(box: dict[str, list[float]], normal: list[float]) -> list[float]:
    half = [float(v) for v in box["size"]]
    n = _normalize_vec3(normal, [0, 0, 1])
    half_height = sum(abs(n[i]) * half[i] for i in range(3))
    radius_sq = 0.0
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                corner = [sx * half[0], sy * half[1], sz * half[2]]
                axial = sum(corner[i] * n[i] for i in range(3))
                radial_sq = sum(corner[i] * corner[i] for i in range(3)) - axial * axial
                radius_sq = max(radius_sq, radial_sq)
    return [max(half_height, 1e-4), max(math.sqrt(max(radius_sq, 0.0)), 1e-4)]


def build_physgaussian_config(payload: dict[str, Any]) -> dict[str, Any]:
    preprocessing = payload.get("preprocessing") or {}
    simulation = payload.get("simulation") or {}
    solver = str(payload.get("solver") or "explicit-mpm")
    integrator = _solver_to_integrator(solver)
    objects = payload.get("objects") or []
    active_objects = _active_objects(objects)
    fixed_anchor_objects = [obj for obj in active_objects if _is_obstacle_part(obj)]
    material_objects = list(active_objects)
    active_indices = _object_indices(material_objects)
    bodies = _objects_by_body(material_objects)
    base_object = material_objects[0] if material_objects else objects[0] if objects else {"material": "jelly"}
    base_body_parts = bodies.get(_body_id(base_object), material_objects)
    base_params = _params_for_fixed_part(base_object, base_body_parts) if _is_obstacle_part(base_object) else _object_params(base_object)
    active_aabb = _union_aabb(active_objects) if active_objects else None
    sim_area = [active_aabb["min"][0], active_aabb["max"][0], active_aabb["min"][1], active_aabb["max"][1], active_aabb["min"][2], active_aabb["max"][2]] if active_aabb else _sim_area_from_payload(payload)
    scale = _safe_float(preprocessing.get("scale"), 1.0)
    n_grid = _safe_int(preprocessing.get("n_grid"), 50)

    config: dict[str, Any] = {
        "opacity_threshold": _safe_float(preprocessing.get("opacity_threshold"), 0.02),
        "rotation_degree": preprocessing.get("rotation_degree") or [0],
        "rotation_axis": preprocessing.get("rotation_axis") or [0],
        "substep_dt": _safe_float(simulation.get("substep_dt"), 1e-4),
        "frame_dt": _safe_float(simulation.get("frame_dt"), 2e-2),
        "frame_num": _safe_int(simulation.get("frame_num"), DEFAULT_FRAME_NUM),
        "integrator": integrator,
        "E": base_params["E"],
        "nu": base_params["nu"],
        "n_grid": n_grid,
        "material": base_params["material"],
        "density": base_params["density"],
        "g": _simulation_gravity(simulation),
        "grid_v_damping_scale": _simulation_damping(simulation),
        "rpic_damping": 0.0,
        "scale": scale,
        "mpm_space_vertical_upward_axis": [0, 0, 1],
        "mpm_space_viewpoint_center": [1, 1, 1],
        "default_camera_index": -1,
        "show_hint": False,
        "move_camera": False,
    }
    for key in MATERIAL_PARAM_KEYS:
        if key in base_params:
            config[key] = base_params[key]

    if integrator == "implicit":
        config["implicit_mpm"] = {
            "beta": _safe_float(simulation.get("implicitBeta"), 0.25),
            "gamma": _safe_float(simulation.get("implicitGamma"), 0.5),
            "newton_tol": _safe_float(simulation.get("newtonTol"), 5e-4),
            "newton_abs_tol": _safe_float(simulation.get("newtonAbsTol"), 1e-6),
            "newton_rms_tol": _safe_float(
                _first_present(
                    simulation.get("newtonRmsTol"),
                    simulation.get("newton_rms_tol"),
                ),
                1e-4,
            ),
            "newton_max_iter": _safe_int(simulation.get("newtonMaxIter"), 16),
            "gmres_tol": _safe_float(simulation.get("gmresTol"), 1e-3),
            "gmres_tol_floor": _safe_float(
                simulation.get("gmresTolFloor"),
                1e-3,
            ),
            "gmres_max_iter": _safe_int(simulation.get("gmresMaxIter"), 24),
            "jvp_eps": _safe_float(simulation.get("jvpEps"), 1e-4),
            "line_search_max_iter": _safe_int(simulation.get("lineSearchMaxIter"), 8),
            "armijo_c1": _safe_float(simulation.get("armijoC1"), 1e-4),
            "ew_eta_min": _safe_float(simulation.get("ewEtaMin"), 1e-3),
            "ew_eta_max": _safe_float(simulation.get("ewEtaMax"), 0.1),
            "ew_gamma": _safe_float(simulation.get("ewGamma"), 0.9),
            "ew_alpha": _safe_float(simulation.get("ewAlpha"), 1.5),
            "stiffness_preconditioner_scale": _safe_float(
                simulation.get("stiffnessPreconditionerScale"), 1.0
            ),
            "stagnation_tol": _safe_float(simulation.get("stagnationTol"), 1e-8),
            "allow_best_effort_commit": bool(
                simulation.get("allowBestEffortCommit", False)
            ),
            "near_converged_factor": _safe_float(
                simulation.get("nearConvergedFactor"), 2.0
            ),
            "near_newton_rms_tol": _safe_float(
                _first_present(
                    simulation.get("nearNewtonRmsTol"),
                    simulation.get("near_newton_rms_tol"),
                ),
                1e-4,
            ),
            "fallback_descent_tol": _safe_float(
                _first_present(
                    simulation.get("fallbackDescentTol"),
                    simulation.get("fallback_descent_tol"),
                ),
                1e-8,
            ),
            "fallback_step_min_rel": _safe_float(
                _first_present(
                    simulation.get("fallbackStepMinRel"),
                    simulation.get("fallback_step_min_rel"),
                ),
                1e-8,
            ),
            "fallback_decrease_tol": _safe_float(
                _first_present(
                    simulation.get("fallbackDecreaseTol"),
                    simulation.get("fallback_decrease_tol"),
                ),
                1e-6,
            ),
        }
    if integrator == "pbmpm":
        config["pbmpm"] = _pbmpm_config_from_simulation(simulation)

    if sim_area is not None:
        config["sim_area"] = sim_area
    if active_indices:
        config["active_gs_indices"] = active_indices

    particle_material = []
    material_index_owner: dict[int, int] = {}
    material_conflicts: list[dict[str, Any]] = []
    for obj in material_objects:
        indices = obj.get("indices") or []
        if not isinstance(indices, list) or not indices:
            continue
        body_parts = bodies.get(_body_id(obj), material_objects)
        params = _params_for_fixed_part(obj, body_parts) if _is_obstacle_part(obj) else _object_params(obj)
        clean_indices = sorted({_safe_int(value, -1) for value in indices if _safe_int(value, -1) >= 0})
        for index in clean_indices:
            previous = material_index_owner.get(index)
            if previous is not None and previous != _safe_int(obj.get("objectId"), -1):
                material_conflicts.append({"index": index, "previousObjectId": previous, "objectId": obj.get("objectId")})
            material_index_owner[index] = _safe_int(obj.get("objectId"), -1)
        entry = {
            "indices": clean_indices,
            "E": params["E"],
            "nu": params["nu"],
            "density": params["density"],
        }
        if "yield_stress" in params:
            entry["yield_stress"] = params["yield_stress"]
        particle_material.append(entry)
    if particle_material:
        config["particle_material_params"] = particle_material

    additional = []
    if sim_area is not None:
        for obj in material_objects:
            box = _aabb_to_mpm_box(obj.get("aabbWorld") or {}, sim_area, scale)
            if not box:
                continue
            body_parts = bodies.get(_body_id(obj), material_objects)
            params = _params_for_fixed_part(obj, body_parts) if _is_obstacle_part(obj) else _object_params(obj)
            entry = {
                **box,
                "E": params["E"],
                "nu": params["nu"],
                "density": params["density"],
            }
            if "yield_stress" in params:
                entry["yield_stress"] = params["yield_stress"]
            additional.append(entry)
    if additional:
        config["additional_material_params"] = additional

    fill_body_boxes = []
    if sim_area is not None:
        for body_parts in bodies.values():
            if not any(part.get("fill") for part in body_parts):
                continue
            body_aabb = _union_aabb(body_parts)
            if not body_aabb:
                continue
            body_box = _aabb_to_mpm_box(body_aabb, sim_area, scale)
            if body_box:
                fill_body_boxes.append(body_box)

    if fill_body_boxes:
        boundary = [0.5, 1.5, 0.5, 1.5, 0.5, 1.5]
        mins = [min(box["point"][i] - box["size"][i] for box in fill_body_boxes) for i in range(3)]
        maxs = [max(box["point"][i] + box["size"][i] for box in fill_body_boxes) for i in range(3)]
        boundary = [mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2]]
        config["particle_filling"] = {
            "n_grid": max(128, n_grid * 2),
            "density_threshold": 40.0,
            "search_threshold": 0.5,
            "search_exclude_direction": 5,
            "ray_cast_direction": 0,
            "max_particles_num": 2_000_000,
            "max_partciels_per_cell": 4,
            "boundary": boundary,
            "visualize": False,
        }

    boundary_conditions: list[dict[str, Any]] = []
    if simulation.get("boundingBoxEnabled", True):
        boundary_conditions.append({"type": "bounding_box"})
    if simulation.get("groundEnabled", True):
        boundary_conditions.append(
            {
                "type": "cuboid",
                "point": [1, 1, _safe_float(simulation.get("groundHeight"), 0.5)],
                "size": [1.0, 1.0, 0.05],
                "velocity": [0, 0, 0],
                "start_time": 0,
                "end_time": 1e3,
                "reset": 1,
            }
        )

    fixed_anchor_count = 0
    grid_velocity_cuboid_constraints: list[dict[str, Any]] = []
    for part in fixed_anchor_objects:
        clean_indices = sorted({_safe_int(value, -1) for value in part.get("indices") or [] if _safe_int(value, -1) >= 0})
        fixed_anchor_count += len(clean_indices)
        cuboid = _aabb_to_velocity_cuboid_constraint(
            part,
            sim_area,
            scale,
            n_grid,
            config["substep_dt"],
            _safe_float(config.get("grid_lim"), 2.0),
        ) if sim_area is not None else None
        if cuboid:
            boundary_conditions.append(cuboid)
            grid_velocity_cuboid_constraints.append(cuboid)
        else:
            _log(
                f"skip grid velocity cuboid for fixed anchor object={part.get('objectId')} "
                "because its AABB or sim_area is invalid"
            )

    if sim_area is not None:
        for body_parts in bodies.values():
            body_aabb = _union_aabb(body_parts)
            if not body_aabb:
                continue
            box = _aabb_to_mpm_box(body_aabb, sim_area, scale)
            if not box:
                continue
            drive_source = next((part for part in body_parts if not _is_obstacle_part(part) and ((part.get("drive") or {}).get("linearEnabled") or (part.get("drive") or {}).get("spinEnabled"))), None)
            if drive_source is None:
                continue
            drive = _drive_params(drive_source)
            if drive["linearEnabled"]:
                boundary_conditions.append(
                    {
                        "type": "particle_impulse",
                        "force": drive["linearForce"],
                        "num_dt": drive["linearNumDt"],
                        "start_time": drive["linearStart"],
                        **box,
                    }
                )
            if drive["spinEnabled"]:
                boundary_conditions.append(
                    {
                        "type": "enforce_particle_velocity_rotation",
                        "point": box["point"],
                        "normal": drive["spinAxis"],
                        "half_height_and_radius": _box_to_rotation_cylinder(box, drive["spinAxis"]),
                        "rotation_scale": drive["spinAngular"],
                        "translation_scale": drive["spinTranslation"],
                        "start_time": drive["spinStart"],
                        "end_time": drive["spinEnd"],
                    }
                )

    config["boundary_conditions"] = boundary_conditions
    config["_ui_diagnostics"] = {
        "activeOriginalGaussianCount": len(active_indices),
        "partCount": len(active_objects),
        "materialPartCount": len(material_objects),
        "fixedAnchorPartCount": len(fixed_anchor_objects),
        "fixedAnchorGaussianCount": fixed_anchor_count,
        "fixedAnchorMode": "grid_velocity_cuboid",
        "gridVelocityCuboidCount": len(grid_velocity_cuboid_constraints),
        "gridVelocityCuboids": [
            {
                "objectId": bc.get("_ui_object_id"),
                "bodyId": bc.get("_ui_body_id"),
                "sceneBounds": bc.get("_ui_scene_bounds"),
                "simArea": bc.get("_ui_sim_area"),
                "mpmBounds": bc.get("_ui_mpm_bounds"),
                "paddedMpmBounds": bc.get("_ui_padded_mpm_bounds"),
                "point": bc.get("point"),
                "size": bc.get("size"),
                "velocity": bc.get("velocity"),
                "padding": bc.get("_ui_padding"),
                "paddingGridNodes": bc.get("_ui_padding_grid_nodes"),
                "gridDx": bc.get("gridDx"),
                "estimatedGridNodeCounts": bc.get("estimatedGridNodeCounts"),
                "estimatedGridNodeCount": bc.get("estimatedGridNodeCount"),
                "gridIndexBounds": bc.get("gridIndexBounds"),
            }
            for bc in grid_velocity_cuboid_constraints
        ],
        "fixedParticleIndicesGenerated": 0,
        "obstaclePartCount": len(fixed_anchor_objects),
        "mixedBodyObstacleDemotedToMpmCount": 0,
        "bodyCount": len(bodies),
        "obstacleBodyCount": 0,
        "particleMaterialLabelCount": len(material_index_owner),
        "materialConflictCount": len(material_conflicts),
        "materialConflictsPreview": material_conflicts[:20],
        "filledParticleMaterialMode": "nearest-labeled-gaussian-inheritance; additional_material_params remains as AABB fallback",
        "pbmpmMode": "pbmpm keeps the velocity local-global pipeline; n_min/n_max bound fully automatic inner iterations; plastic_mode=0 commits through shared material return mapping, plastic_mode=1 uses PBMPM stretch clamp yield_min/yield_max",
        "solver": solver,
        "integrator": config["integrator"],
    }
    return config


def build_official_physgaussian_config(model: dict[str, Any], payload: dict[str, Any] | None = None) -> dict[str, Any]:
    config_path = model.get("officialConfig")
    if not config_path:
        raise HTTPException(status_code=400, detail="this model has no official PhysGaussian config")
    path = Path(config_path).resolve()
    if not path.exists():
        raise HTTPException(status_code=500, detail=f"official config does not exist: {path}")
    config = _read_json(path)
    official_has_n_grid = "n_grid" in config
    solver = str((payload or {}).get("solver") or "explicit-mpm")
    integrator = _solver_to_integrator(solver)
    simulation = (payload or {}).get("simulation") or {}
    # Official presets keep their scene, object/material, transforms and
    # boundary conditions. The UI still owns solver choice and time parameters.
    config["integrator"] = integrator
    config["grid_v_damping_scale"] = _simulation_damping(
        simulation,
        _config_damping(config),
    )
    config["rpic_damping"] = _safe_float(config.get("rpic_damping"), 0.0)
    if integrator == "implicit":
        implicit_adaptive_max_split = max(
            0, _safe_int(simulation.get("implicitAdaptiveMaxSplit"), 3)
        )
        implicit_adaptive_min_dt = _safe_float(
            simulation.get("implicitAdaptiveMinDt"),
            max(
                _safe_float(simulation.get("substep_dt"), 1e-4)
                / (2.0 ** max(implicit_adaptive_max_split, 1)),
                1e-8,
            ),
        )
        config["implicit_mpm"] = {
            "beta": _safe_float(simulation.get("implicitBeta"), 0.25),
            "gamma": _safe_float(simulation.get("implicitGamma"), 0.5),
            "newton_tol": _safe_float(simulation.get("newtonTol"), 5e-4),
            "newton_abs_tol": _safe_float(simulation.get("newtonAbsTol"), 1e-6),
            "newton_rms_tol": _safe_float(
                _first_present(
                    simulation.get("newtonRmsTol"),
                    simulation.get("newton_rms_tol"),
                ),
                1e-4,
            ),
            "newton_max_iter": _safe_int(simulation.get("newtonMaxIter"), 16),
            "gmres_tol": _safe_float(simulation.get("gmresTol"), 1e-3),
            "gmres_tol_floor": _safe_float(
                simulation.get("gmresTolFloor"),
                1e-3,
            ),
            "gmres_max_iter": _safe_int(simulation.get("gmresMaxIter"), 24),
            "jvp_eps": _safe_float(simulation.get("jvpEps"), 1e-4),
            "line_search_max_iter": _safe_int(simulation.get("lineSearchMaxIter"), 8),
            "armijo_c1": _safe_float(simulation.get("armijoC1"), 1e-4),
            "ew_eta_min": _safe_float(simulation.get("ewEtaMin"), 1e-3),
            "ew_eta_max": _safe_float(simulation.get("ewEtaMax"), 0.1),
            "ew_gamma": _safe_float(simulation.get("ewGamma"), 0.9),
            "ew_alpha": _safe_float(simulation.get("ewAlpha"), 1.5),
            "stiffness_preconditioner_scale": _safe_float(
                simulation.get("stiffnessPreconditionerScale"), 1.0
            ),
            "stagnation_tol": _safe_float(simulation.get("stagnationTol"), 1e-8),
            "allow_best_effort_commit": bool(
                simulation.get("allowBestEffortCommit", False)
            ),
            "near_converged_factor": _safe_float(
                simulation.get("nearConvergedFactor"), 2.0
            ),
            "near_newton_rms_tol": _safe_float(
                _first_present(
                    simulation.get("nearNewtonRmsTol"),
                    simulation.get("near_newton_rms_tol"),
                ),
                1e-4,
            ),
            "fallback_descent_tol": _safe_float(
                _first_present(
                    simulation.get("fallbackDescentTol"),
                    simulation.get("fallback_descent_tol"),
                ),
                1e-8,
            ),
            "fallback_step_min_rel": _safe_float(
                _first_present(
                    simulation.get("fallbackStepMinRel"),
                    simulation.get("fallback_step_min_rel"),
                ),
                1e-8,
            ),
            "fallback_decrease_tol": _safe_float(
                _first_present(
                    simulation.get("fallbackDecreaseTol"),
                    simulation.get("fallback_decrease_tol"),
                ),
                1e-6,
            ),
            "adaptive_max_split": implicit_adaptive_max_split,
            "adaptive_min_dt": implicit_adaptive_min_dt,
        }
        config.pop("pbmpm", None)
    elif integrator == "pbmpm":
        config["pbmpm"] = _pbmpm_config_from_simulation(simulation)
        config.pop("implicit_mpm", None)
    else:
        config.pop("implicit_mpm", None)
        config.pop("pbmpm", None)
    official_substep_dt = _safe_float(config.get("substep_dt"), 1e-4)
    config["substep_dt"] = _safe_float(simulation.get("substep_dt"), official_substep_dt)
    config["frame_dt"] = _safe_float(simulation.get("frame_dt"), _safe_float(config.get("frame_dt"), 2e-2))
    config["frame_num"] = _safe_int(simulation.get("frame_num"), DEFAULT_FRAME_NUM)
    payload_n_grid = ((payload or {}).get("preprocessing") or {}).get("n_grid")
    default_n_grid = _safe_int(config.get("n_grid"), 50) if official_has_n_grid else 50
    config["n_grid"] = _safe_int(
        payload_n_grid,
        default_n_grid,
    )
    if integrator == "explicit" and not official_has_n_grid:
        reference_n_grid = 50
        requested_substep_dt = _safe_float(config.get("substep_dt"), official_substep_dt)
        config["explicit_mpm"] = {
            "requested_substep_dt": requested_substep_dt,
            "effective_substep_dt": requested_substep_dt,
            "reference_n_grid": reference_n_grid,
            "n_grid": _safe_int(config.get("n_grid"), reference_n_grid),
            "policy": "no_automatic_substep_dt_scaling",
        }
    return config


def _preview_drive_body(objects: list[dict[str, Any]]) -> tuple[int | None, list[dict[str, Any]], dict[str, Any] | None, dict[str, Any]]:
    bodies = _objects_by_body(objects)
    for body_id, parts in bodies.items():
        drive_source = next(
            (
                part
                for part in parts
                if (part.get("drive") or {}).get("linearEnabled")
                or (part.get("drive") or {}).get("spinEnabled")
            ),
            None,
        )
        if not drive_source:
            continue
        drive = _drive_params(drive_source)
        force = drive["linearForce"]
        if any(abs(value) > 1e-10 for value in force) or abs(drive["spinAngular"]) > 1e-10:
            return body_id, parts, drive_source, drive
    return None, [], None, _drive_params({})


def _point_bbox(points: list[list[float]]) -> tuple[list[float], list[float]]:
    mins = [min(point[i] for point in points) for i in range(3)]
    maxs = [max(point[i] for point in points) for i in range(3)]
    return mins, maxs


def _point_center(points: list[list[float]]) -> list[float]:
    if not points:
        return [0.0, 0.0, 0.0]
    inv = 1.0 / len(points)
    return [sum(point[i] for point in points) * inv for i in range(3)]


def _adaptive_preview_groups(points: list[list[float]]) -> list[dict[str, Any]]:
    """Split dense regions into proxy groups without asking the user for K.

    The preview is intentionally cheap: each leaf group keeps its internal
    Gaussian offsets fixed, and only the group center receives an approximate
    displacement. Dense spatial cells split more often than sparse cells.
    """
    count = len(points)
    if count == 0:
        return []
    mins, maxs = _point_bbox(points)
    extents = [maxs[i] - mins[i] for i in range(3)]
    diag = math.sqrt(sum(value * value for value in extents))
    target_points = 1536
    max_groups = 192
    min_cell_size = max(diag / 96.0, 1e-6)
    groups: list[list[int]] = []
    stack: list[list[int]] = [list(range(count))]

    while stack:
        local = stack.pop()
        if not local:
            continue
        local_points = [points[index] for index in local]
        local_mins, local_maxs = _point_bbox(local_points)
        local_extents = [local_maxs[i] - local_mins[i] for i in range(3)]
        axis = max(range(3), key=lambda i: local_extents[i])
        if len(local) <= target_points or local_extents[axis] <= min_cell_size or len(groups) + len(stack) >= max_groups:
            groups.append(local)
            continue
        ordered = sorted(local, key=lambda index: points[index][axis])
        middle = len(ordered) // 2
        left = ordered[:middle]
        right = ordered[middle:]
        if not left or not right:
            groups.append(local)
            continue
        stack.append(right)
        stack.append(left)

    return [
        {
            "indices": local,
            "center": _point_center([points[index] for index in local]),
        }
        for local in groups
    ]


def _clamp_vector_length(vector: list[float], max_length: float) -> list[float]:
    length = math.sqrt(sum(value * value for value in vector))
    if length <= max_length or length <= 1e-12:
        return vector
    scale = max_length / length
    return [value * scale for value in vector]


def _write_rigid_preview_motion(run_id: str, model: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    base_ply = Path(model["basePly"]) if model.get("basePly") else None
    if not base_ply or not base_ply.exists():
        raise HTTPException(status_code=400, detail="preview requires a base point_cloud.ply")

    objects = payload.get("objects") or []
    splat_props = _read_ply_splat_props(base_ply)
    xyz = splat_props["positions"]
    base_rotations = splat_props["rotations"]
    base_scales = splat_props["scales"]
    num_splats = len(xyz)

    simulation = payload.get("simulation") or {}
    preview = simulation.get("preview") if isinstance(simulation.get("preview"), dict) else {}
    drag_velocity = preview.get("dragVelocity") if isinstance(preview.get("dragVelocity"), list) else None
    drag_velocity = [
        _safe_float(drag_velocity[i], 0.0) if drag_velocity and i < len(drag_velocity) else 0.0
        for i in range(3)
    ]
    drag_hit_index = _safe_int(preview.get("dragHitIndex"), -1)
    drag_object_id = _safe_int(preview.get("dragObjectId"), -1)
    drag_body_id = _safe_int(preview.get("dragBodyId"), -1)
    drag_mode = any(abs(value) > 1e-10 for value in drag_velocity) or 0 <= drag_hit_index < num_splats

    body_id: int | None
    body_parts: list[dict[str, Any]]
    drive_source: dict[str, Any] | None
    if drag_mode:
        bodies = _objects_by_body(objects)
        body_id = drag_body_id if drag_body_id in bodies else None
        drive_source = None
        if 0 <= drag_hit_index < num_splats:
            for part in objects:
                part_indices = part.get("indices") or []
                if isinstance(part_indices, list) and drag_hit_index in part_indices:
                    drive_source = part
                    body_id = _safe_int(part.get("bodyId"), _safe_int(part.get("objectId"), -1))
                    break
        if drive_source is None and drag_object_id > 0:
            drive_source = next((part for part in objects if _safe_int(part.get("objectId"), -1) == drag_object_id), None)
            if drive_source:
                body_id = _safe_int(drive_source.get("bodyId"), drag_object_id)
        body_parts = bodies.get(body_id, []) if body_id is not None else []
        drive = _drive_params(drive_source or {})
        drive["linearEnabled"] = True
        drive["linearVelocity"] = drag_velocity
        drive["targetVelocity"] = drag_velocity
        drive["dirichletVelocity"] = drag_velocity
        drive["prescribedVelocity"] = drag_velocity
        drive["hasExplicitVelocity"] = True
        drive["spinEnabled"] = False
        drive["spinAngular"] = 0.0
    else:
        body_id, body_parts, drive_source, drive = _preview_drive_body(objects)
        if body_id is None:
            raise HTTPException(status_code=400, detail="preview requires one body with enabled linear/spin drive")

    if not body_parts:
        raise HTTPException(status_code=400, detail="preview target body has no parts")

    indices = sorted(
        {
            _safe_int(index, -1)
            for part in body_parts
            for index in (part.get("indices") or [])
            if 0 <= _safe_int(index, -1) < num_splats
        }
    )
    if not indices:
        indices = list(range(num_splats))
        if drag_mode:
            body_id = 0

    fixed_indices = {
        _safe_int(index, -1)
        for part in body_parts
        if _is_obstacle_part(part)
        for index in (part.get("indices") or [])
        if 0 <= _safe_int(index, -1) < num_splats
    }
    groups, group_diagnostics, fixed_group_ids = _proxy_groups_with_fixed_split(xyz, indices, fixed_indices, payload)
    if not groups:
        raise HTTPException(status_code=400, detail="preview could not build voxel groups")

    part_params_by_index: dict[int, dict[str, Any]] = {}
    for part in body_parts:
        params = _params_for_fixed_part(part, body_parts) if _is_obstacle_part(part) else _object_params(part)
        for value in part.get("indices") or []:
            index = _safe_int(value, -1)
            if 0 <= index < num_splats:
                part_params_by_index[index] = params

    frame_count = min(max(_safe_int(simulation.get("frame_num"), 12), 2), 24) + 1
    frame_dt = _safe_float(simulation.get("frame_dt"), 2e-2)
    tracked = len(indices)
    selected = [xyz[index] for index in indices]
    mins, maxs = _point_bbox(selected)
    diag = math.sqrt(sum((maxs[i] - mins[i]) ** 2 for i in range(3)))
    total_time = max((frame_count - 1) * frame_dt, frame_dt)
    velocity_source = drive["prescribedVelocity"] if drive["linearEnabled"] else [0.0, 0.0, 0.0]
    velocity = [float(velocity_source[i]) for i in range(3)]
    total_displacement = _clamp_vector_length(
        [velocity[i] * total_time for i in range(3)],
        max(diag * 0.35, 1e-4),
    )
    velocity = [total_displacement[i] / total_time for i in range(3)]

    gravity = _simulation_gravity(simulation)
    ground_enabled = bool(simulation.get("groundEnabled"))
    preview_ground_z = mins[2]
    spin_axis = _normalize_vec3(drive["spinAxis"], [0.0, 0.0, 1.0])
    spin_rate = drive["spinAngular"] if drive["spinEnabled"] else 0.0
    index_to_order = {index: order for order, index in enumerate(indices)}
    hit_group = None
    if drag_mode and 0 <= drag_hit_index < num_splats:
        for group_id, group in enumerate(groups):
            if drag_hit_index in group["indices"]:
                hit_group = group_id
                break
    group_centers = [group["center"][:] for group in groups]
    group_initial_centers = [group["center"][:] for group in groups]
    preview_grid_res, preview_grid_diagnostics = _proxy_grid_resolution(
        len(groups),
        group_diagnostics,
        mins,
        maxs,
        preview,
    )
    group_velocities = [[0.0, 0.0, 0.0] for _ in groups]
    if drag_mode:
        if hit_group is not None:
            group_velocities[hit_group] = velocity[:]
        else:
            group_velocities = [velocity[:] for _ in groups]
    else:
        group_velocities = [velocity[:] for _ in groups]
    group_masses = [max(float(len(group["indices"])), 1.0) for group in groups]
    proxy_edges = _proxy_graph_edges(group_centers)
    proxy_velocity_spread_iterations = _safe_int(preview.get("velocitySpreadIterations"), 8)
    proxy_velocity_spread_strength = _safe_float(preview.get("velocitySpreadStrength"), 0.65)
    if drag_mode and hit_group is not None:
        group_velocities = _spread_proxy_velocity(
            group_velocities,
            group_masses,
            proxy_edges,
            {hit_group},
            proxy_velocity_spread_iterations,
            proxy_velocity_spread_strength,
        )
    proxy_shape_coupling = _safe_float(preview.get("shapeCoupling"), _safe_float(preview.get("shapeStiffness"), 0.18 if drag_mode else 0.0))
    group_affines = [_zero_mat3() for _ in groups]
    group_deformations = [_identity_mat3() for _ in groups]
    group_rotations = [[1.0, 0.0, 0.0, 0.0] for _ in groups]
    group_youngs: list[float] = []
    group_nus: list[float] = []
    group_densities: list[float] = []
    fallback_params = _object_params(drive_source or (body_parts[0] if body_parts else {}))
    for group in groups:
        group_params = [part_params_by_index.get(index, fallback_params) for index in group["indices"]]
        weights = [1.0 for _ in group_params]
        weight_sum = max(sum(weights), 1e-6)
        group_youngs.append(sum(params["E"] * weight for params, weight in zip(group_params, weights)) / weight_sum)
        group_nus.append(sum(params["nu"] * weight for params, weight in zip(group_params, weights)) / weight_sum)
        group_densities.append(sum(params["density"] * weight for params, weight in zip(group_params, weights)) / weight_sum)
    proxy_stress_scale = _safe_float(preview.get("stressScale"), 0.35)
    proxy_grid_damping = _safe_float(preview.get("gridDamping"), 1.0)
    kinematic_drive_groups = {hit_group} if hit_group is not None and hit_group in fixed_group_ids else set()
    if not drag_mode and fixed_group_ids and drive.get("linearEnabled"):
        kinematic_drive_groups = set(fixed_group_ids)
    group_order_by_index = {
        index: group_id
        for group_id, group in enumerate(groups)
        for index in group["indices"]
    }
    proxy_blend_count = _safe_int(preview.get("proxyBlendCount"), 4)
    proxy_blend_power = _safe_float(preview.get("proxyBlendPower"), 2.0)
    skinning_weights = _proxy_skinning_weights(
        xyz,
        indices,
        group_initial_centers,
        group_order_by_index,
        proxy_blend_count,
        proxy_blend_power,
    )
    ground_z = preview_ground_z if ground_enabled else None

    preview_dir = RUNS_ROOT / run_id / "physgaussian" / "super_motion"
    preview_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "phys-motion-v1",
        "binary": "motion.bin",
        "indices": "indices.bin",
        "proxy": {
            "motion": "proxy_motion.bin",
            "skinning": "proxy_skinning.bin",
            "groupCount": len(groups),
            "blendCount": proxy_blend_count,
            "motionStrideBytes": len(groups) * 7 * 4,
            "skinningStrideBytes": proxy_blend_count * 8,
        },
        "frameCount": frame_count,
        "frameRate": int(round(1.0 / frame_dt)) if frame_dt > 0 else 30,
        "numSplats": num_splats,
        "attributes": ["position", "rotation", "scale"],
        "frameStrideBytes": tracked * 10 * 4,
        "updateBounds": False,
        "preview": {
            "type": "voxel-rigid-body",
            "bodyId": body_id,
            "sourceObjectId": drive_source.get("objectId") if drive_source else None,
            **group_diagnostics,
            "maxGroupSize": max((len(group["indices"]) for group in groups), default=0),
            "linearDisplacement": total_displacement,
            "gravityEnabled": any(abs(value) > 1e-12 for value in gravity),
            "groundEnabled": ground_enabled,
            "spinEnabled": bool(drive["spinEnabled"]),
            "dragMode": drag_mode,
            "dragHitIndex": drag_hit_index if drag_hit_index >= 0 else None,
            "hitGroup": hit_group,
            "previewGridResolution": preview_grid_res,
            **preview_grid_diagnostics,
            "stressScale": proxy_stress_scale,
            "gridDamping": proxy_grid_damping,
            "proxyBlendCount": proxy_blend_count,
            "proxyBlendPower": proxy_blend_power,
            "proxyEdgeCount": len(proxy_edges),
            "velocitySpreadIterations": proxy_velocity_spread_iterations,
            "velocitySpreadStrength": proxy_velocity_spread_strength,
            "shapeCoupling": proxy_shape_coupling,
            "solver": "proxy-mpm-apic-stress",
            "fixedAnchorPreview": bool(fixed_group_ids),
            "fixedAnchorGroupCount": len(fixed_group_ids),
        },
    }
    _write_json(preview_dir / "motion.physmotion.json", manifest)

    import array

    with (preview_dir / "indices.bin").open("wb") as stream:
        array.array("I", indices).tofile(stream)

    with (preview_dir / "proxy_skinning.bin").open("wb") as stream:
        group_values = array.array("I")
        weight_values = array.array("f")
        for index in indices:
            blends = list(skinning_weights.get(index) or [(group_order_by_index.get(index, 0), 1.0)])
            blends = blends[:proxy_blend_count]
            while len(blends) < proxy_blend_count:
                blends.append((blends[-1][0] if blends else 0, 0.0))
            for group_id, weight in blends:
                group_values.append(max(0, int(group_id)))
                weight_values.append(float(weight))
        group_values.tofile(stream)
        weight_values.tofile(stream)

    proxy_frames: list[tuple[list[list[float]], list[list[float]]]] = []

    with (preview_dir / "motion.bin").open("wb") as stream:
        for frame in range(frame_count):
            time_at_frame = frame * frame_dt
            positions_out = [[0.0, 0.0, 0.0] for _ in indices]
            rotations_out = [[1.0, 0.0, 0.0, 0.0] for _ in indices]
            scales_out = [[0.0, 0.0, 0.0] for _ in indices]

            if frame > 0:
                group_centers, group_velocities, group_affines, group_deformations, group_rotations = _proxy_mpm_step(
                    group_centers,
                    group_velocities,
                    group_affines,
                    group_deformations,
                    group_masses,
                    group_youngs,
                    group_nus,
                    group_densities,
                    mins,
                    maxs,
                    preview_grid_res,
                    frame_dt,
                    gravity,
                    ground_z,
                    hit_group if drag_mode else None,
                    velocity if drag_mode and hit_group is not None else None,
                    fixed_group_ids,
                    kinematic_drive_groups,
                    proxy_grid_damping,
                    proxy_stress_scale,
                )
                if proxy_shape_coupling > 0.0:
                    group_velocities = _apply_proxy_shape_coupling(
                        group_centers,
                        group_velocities,
                        group_initial_centers,
                        group_masses,
                        frame_dt,
                        proxy_shape_coupling,
                    )
            frame_group_rotations = [
                _quat_mul(_quat_from_axis_angle(spin_axis, spin_rate * time_at_frame), group_rotations[group_id])
                for group_id in range(len(groups))
            ]
            proxy_frames.append(([center[:] for center in group_centers], [rotation[:] for rotation in frame_group_rotations]))
            for index in indices:
                order = index_to_order[index]
                blends = skinning_weights.get(index)
                if not blends:
                    group_id = group_order_by_index.get(index, 0)
                    blends = [(group_id, 1.0)]

                blended_position = [0.0, 0.0, 0.0]
                blended_quats: list[list[float]] = []
                blended_weights: list[float] = []
                for group_id, weight in blends:
                    initial_center = group_initial_centers[group_id]
                    group_center = group_centers[group_id]
                    group_rotation = frame_group_rotations[group_id]
                    local = [xyz[index][axis] - initial_center[axis] for axis in range(3)]
                    rotated = _quat_rotate_vec(group_rotation, local)
                    candidate_position = [
                        group_center[axis] + rotated[axis]
                        for axis in range(3)
                    ]
                    for axis in range(3):
                        blended_position[axis] += weight * candidate_position[axis]
                    blended_quats.append(group_rotation)
                    blended_weights.append(weight)

                blended_rotation = _quat_weighted_blend(blended_quats, blended_weights)
                positions_out[order] = blended_position
                rotations_out[order] = _quat_mul(blended_rotation, _quat_normalize(base_rotations[index]))
                scales_out[order] = base_scales[index]

            values = array.array("f")
            for point in positions_out:
                values.extend(point)
            for quat in rotations_out:
                values.extend(quat)
            for scale_values in scales_out:
                values.extend(scale_values)
            values.tofile(stream)

    with (preview_dir / "proxy_motion.bin").open("wb") as stream:
        values = array.array("f")
        for centers_frame, rotations_frame in proxy_frames:
            for center in centers_frame:
                values.extend(center)
            for rotation in rotations_frame:
                values.extend(rotation)
        values.tofile(stream)

    motion_manifest = dict(manifest)
    motion_manifest.setdefault("base", base_ply.name)
    _write_json(preview_dir / "motion.physmotion.json", motion_manifest)

    result_payload = {
        "manifestUrl": _url_under(RUNS_ROOT, preview_dir / "motion.physmotion.json", "/outputs"),
        "binaryUrl": f"/api/runs/{run_id}/motion.bin",
        "indicesUrl": f"/api/runs/{run_id}/indices.bin",
        "proxyMotionUrl": f"/api/runs/{run_id}/proxy_motion.bin",
        "proxySkinningUrl": f"/api/runs/{run_id}/proxy_skinning.bin",
        "basePlyUrl": _model_base_ply_url(model, base_ply),
        "basePlyName": base_ply.name,
    }
    _write_json(RUNS_ROOT / run_id / "payload.json", payload)
    _update_run(
        run_id,
        runId=run_id,
        status="completed",
        progress="voxel-rigid-preview-ready",
        modelId=model.get("modelId"),
        configMode="voxel-rigid-preview",
        result=result_payload,
        preview=manifest["preview"],
    )
    progress = _motion_progress(run_id)
    progress.pop("result", None)
    _update_run(run_id, **progress)
    return {"runId": run_id, "status": "completed", "result": result_payload, **progress}


def _voxel_group_summary(model: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    base_ply = Path(model["basePly"]) if model.get("basePly") else None
    if not base_ply or not base_ply.exists():
        raise HTTPException(status_code=400, detail="voxel grouping requires a base point_cloud.ply")
    xyz = _read_ply_xyz(base_ply)
    num_splats = len(xyz)
    objects = payload.get("objects") or []
    bodies = _objects_by_body(objects)
    body_records = []
    if not bodies:
        indices = list(range(num_splats))
        groups, diagnostics = _voxel_groups_for_indices(xyz, indices, payload)
        body_records.append(
            {
                "bodyId": 0,
                "partCount": 0,
                "maxGroupSize": max((len(group["indices"]) for group in groups), default=0),
                **diagnostics,
            }
        )
    else:
        for body_id, parts in sorted(bodies.items()):
            indices = sorted(
                {
                    _safe_int(index, -1)
                    for part in parts
                    for index in (part.get("indices") or [])
                    if 0 <= _safe_int(index, -1) < num_splats
                }
            )
            if not indices:
                continue
            groups, diagnostics = _voxel_groups_for_indices(xyz, indices, payload)
            body_records.append(
                {
                    "bodyId": body_id,
                    "partCount": len(parts),
                    "maxGroupSize": max((len(group["indices"]) for group in groups), default=0),
                    **diagnostics,
                }
            )
    return {
        "modelId": model.get("modelId"),
        "numSplats": num_splats,
        "bodyCount": len(body_records),
        "bodies": body_records,
    }


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "service": "USTC26MM PhysGaussian bridge",
        "physgaussianRoot": str(PHYSGAUSSIAN_ROOT),
        "python": PYTHON_BIN,
        "supersplatDist": str(SUPERSPLAT_DIST),
    }


@app.get("/api/schema")
def schema() -> dict[str, Any]:
    return {
        "modelUpload": [
            ".ply direct 3DGS model, wrapped as point_cloud/iteration_0/point_cloud.ply",
            ".zip full 3DGS/PhysGaussian model package",
        ],
        "simulation": [
            "officialConfig.enabled uses the model's bundled official PhysGaussian JSON for scene/object config while preserving UI-selected solver and time parameters",
            "solver supports explicit-mpm, implicit-mpm, and pbmpm; pbmpm k is fully automatic from material parameters and n_min/n_max, with plastic_mode choosing shared material return mapping or PBMPM stretch clamp bounds",
            "gravityEnabled + gravity",
            "groundEnabled + groundHeight",
            "boundingBoxEnabled",
            "per-body drive.linear: force, num_dt, start_time, union AABB target",
            "per-body drive.spin: maps to PhysGaussian enforce_particle_velocity_rotation over the body union AABB",
            "per-part selected Gaussian indices map to particle_material_params for exact E/nu/density/yield_stress on original Gaussian particles",
            "per-part material=obstacle creates a grid velocity cuboid constraint (type=cuboid / set_velocity_on_cuboid); selected Gaussian indices remain material/active labels, not fixed particles",
            "POST /api/preview/voxel-rigid writes a lightweight voxel-center rigid motion package with position+rotation+scale; /api/preview/rigid-linear is kept as a compatibility alias",
            "POST /api/preview/voxel-groups returns automatic voxel-center grouping diagnostics for the current imported model and Part/Body labels",
            "frame_dt, frame_num, substep_dt; large substep_dt is allowed for instability tests, and substep_dt >= frame_dt advances one explicit step per rendered frame",
        ],
        "preprocessing": [
            "opacity_threshold",
            "rotation_degree and rotation_axis are generated automatically from the frontend splat transform",
            "sim_area",
            "scale is global for the entire selected simulation area/body union",
            "n_grid",
        ],
        "stockLimitations": [
            "objects are material parts; all assigned parts are simulated; parts with the same bodyId share one physical body transform/scale and drive target",
            "additional_material_params still changes filled/internal particles by part AABB; original Gaussian particles are corrected by selected index labels",
            "cuboid constraints are grid boundary conditions; UI fixed anchors use grid velocity cuboids with padding diagnostics, not fixed particle attributes",
            "per-particle labels and finite-mass rigid bodies require solver changes",
        ],
    }


@app.get("/api/models")
def list_models() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(MODELS_ROOT.glob("*/model.json")):
        try:
            record = _with_official_info(_read_json(path))
            if record.get("officialConfigAvailable") or "official" in str(record.get("kind", "")):
                records.append(record)
        except Exception:
            continue
    return records


@app.get("/api/models/{model_id}")
def get_model(model_id: str) -> dict[str, Any]:
    return _load_model(model_id)


@app.post("/api/models")
async def upload_model(file: UploadFile = File(...)) -> dict[str, Any]:
    model_id = uuid.uuid4().hex[:12]
    model_dir = MODELS_ROOT / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(file.filename or "model").name
    upload_path = model_dir / filename
    with upload_path.open("wb") as stream:
        shutil.copyfileobj(file.file, stream)

    extract_dir = model_dir / "extracted"
    kind = "archive"
    if upload_path.suffix.lower() == ".zip":
        extract_dir.mkdir(parents=True, exist_ok=True)
        _extract_zip_safely(upload_path, extract_dir)
        search_root = extract_dir
    elif upload_path.suffix.lower() == ".ply":
        kind = "ply-direct"
        model_root = model_dir / "model"
        base_ply = model_root / "point_cloud" / "iteration_0" / "point_cloud.ply"
        base_ply.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(upload_path, base_ply)
        official_config = _match_official_config_for_ply(base_ply)
        record = {
            "modelId": model_id,
            "kind": kind,
            "uploadName": filename,
            "modelRoot": str(model_root),
            "basePly": str(base_ply),
            "basePlyUrl": _url_under(MODELS_ROOT, base_ply, "/models"),
            "basePlyName": base_ply.name,
        }
        if official_config:
            record["officialConfig"] = str(official_config)
        _write_json(_model_record_path(model_id), record)
        return _with_official_info(record)
    else:
        search_root = model_dir

    model_root = _find_model_root(search_root)
    base_ply = _find_base_ply(model_root, search_root)
    official_config = _official_config_for_model_root(model_root)
    if not official_config and base_ply:
        official_config = _match_official_config_for_ply(base_ply)
    record = {
        "modelId": model_id,
        "kind": kind if model_root else "preview-only",
        "uploadName": filename,
        "modelRoot": str(model_root) if model_root else None,
        "basePly": str(base_ply) if base_ply else None,
        "basePlyUrl": _url_under(MODELS_ROOT, base_ply, "/models") if base_ply else None,
        "basePlyName": base_ply.name if base_ply else None,
    }
    if official_config:
        record["officialConfig"] = str(official_config)
    _write_json(_model_record_path(model_id), record)
    return _with_official_info(record)


@app.post("/api/simulate")
async def submit_simulation(payload: dict[str, Any], background_tasks: BackgroundTasks) -> dict[str, Any]:
    model_id = payload.get("modelId")
    if not model_id:
        _log("reject simulate: modelId is required")
        raise HTTPException(status_code=400, detail="modelId is required")
    model = _load_model(model_id)
    if not model.get("modelRoot"):
        _log(f"reject simulate: model {model_id} cannot be simulated")
        raise HTTPException(status_code=400, detail="this model cannot be simulated; upload a 3DGS PLY or full trained model zip")
    _reject_if_not_physgaussian_checkpoint(model)
    if not PHYSGAUSSIAN_ROOT.exists():
        _log(f"reject simulate: PHYSGAUSSIAN_ROOT does not exist: {PHYSGAUSSIAN_ROOT}")
        raise HTTPException(status_code=500, detail=f"PHYSGAUSSIAN_ROOT does not exist: {PHYSGAUSSIAN_ROOT}")

    run_id = uuid.uuid4().hex[:12]
    run_dir = RUNS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    official_config_enabled = bool((payload.get("officialConfig") or {}).get("enabled"))
    config = build_official_physgaussian_config(model, payload) if official_config_enabled else build_physgaussian_config(payload)
    _log(f"accept simulate: run={run_id} model={model_id} mode={'official' if official_config_enabled else 'ui'} parts={len(payload.get('objects') or [])}")
    _write_json(run_dir / "payload.json", payload)
    _write_json(run_dir / "config.json", config)
    record = {
        "runId": run_id,
        "status": "queued",
        "modelId": model_id,
        "configMode": "official" if official_config_enabled else "ui",
        "officialConfig": model.get("officialConfig") if official_config_enabled else None,
        "configPath": str(run_dir / "config.json"),
        "outputDir": str(run_dir / "physgaussian"),
    }
    _write_json(_run_record_path(run_id), record)
    background_tasks.add_task(_run_simulation_job, run_id, model)
    return {"runId": run_id, "status": "queued"}


@app.post("/api/preview/rigid-linear")
async def rigid_linear_preview(payload: dict[str, Any]) -> dict[str, Any]:
    model_id = payload.get("modelId")
    if not model_id:
        raise HTTPException(status_code=400, detail="modelId is required")
    model = _load_model(model_id)
    run_id = uuid.uuid4().hex[:12]
    return _write_rigid_preview_motion(run_id, model, payload)


@app.post("/api/preview/voxel-rigid")
async def voxel_rigid_preview(payload: dict[str, Any]) -> dict[str, Any]:
    return await rigid_linear_preview(payload)


@app.post("/api/preview/voxel-groups")
async def voxel_groups_preview(payload: dict[str, Any]) -> dict[str, Any]:
    model_id = payload.get("modelId")
    if not model_id:
        raise HTTPException(status_code=400, detail="modelId is required")
    return _voxel_group_summary(_load_model(model_id), payload)


@app.post("/api/config-preview")
async def preview_config(payload: dict[str, Any]) -> dict[str, Any]:
    model_id = payload.get("modelId")
    model = _load_model(model_id) if model_id else {}
    official_config_enabled = bool((payload.get("officialConfig") or {}).get("enabled"))
    config = build_official_physgaussian_config(model, payload) if official_config_enabled and model else build_physgaussian_config(payload)
    return {
        "config": config,
        "diagnostics": config.get("_ui_diagnostics", {}),
    }


@app.get("/api/runs")
def list_runs() -> list[dict[str, Any]]:
    records = []
    for path in sorted(RUNS_ROOT.glob("*/run.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        records.append(_read_json(path))
    return records


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    path = _run_record_path(run_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"unknown runId: {run_id}")
    record = _read_json(path)
    config = _read_json(Path(record["configPath"])) if record.get("configPath") and Path(record["configPath"]).exists() else None
    if record.get("status") in {"running", "queued", "completed"}:
        record.update(_motion_progress(run_id, config))
    return record


@app.post("/api/runs/{run_id}/cancel")
def cancel_run(run_id: str) -> dict[str, Any]:
    path = _run_record_path(run_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"unknown runId: {run_id}")
    process = RUN_PROCESSES.get(run_id)
    if process and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    record = _update_run(run_id, status="cancelled", progress="cancelled", error="cancelled by user")
    record.update(_motion_progress(run_id))
    return record


def _tail(path: Path, max_chars: int = 4000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(errors="replace")
    return text[-max_chars:]


def _range_file_response(path: Path, request: Request, media_type: str = "application/octet-stream") -> Response:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"file not found: {path.name}")

    size = path.stat().st_size
    range_header = request.headers.get("range")
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Type": media_type,
        "Cache-Control": "public, max-age=31536000, immutable",
    }
    if not range_header:
        headers["Content-Length"] = str(size)
        return Response(path.read_bytes(), headers=headers, media_type=media_type)

    match = re.match(r"bytes=(\d*)-(\d*)$", range_header.strip())
    if not match:
        raise HTTPException(status_code=416, detail="invalid Range header")

    start_text, end_text = match.groups()
    if start_text == "" and end_text == "":
        raise HTTPException(status_code=416, detail="invalid Range header")
    if start_text == "":
        suffix = int(end_text)
        start = max(size - suffix, 0)
        end = size - 1
    else:
        start = int(start_text)
        end = int(end_text) if end_text else size - 1

    if start >= size or end < start:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})

    end = min(end, size - 1)
    length = end - start + 1
    with path.open("rb") as stream:
        stream.seek(start)
        data = stream.read(length)

    headers.update(
        {
            "Content-Range": f"bytes {start}-{end}/{size}",
            "Content-Length": str(length),
        }
    )
    return Response(data, status_code=206, headers=headers, media_type=media_type)


@app.get("/api/runs/{run_id}/motion.bin")
def get_run_motion_binary(run_id: str, request: Request) -> Response:
    return _range_file_response(RUNS_ROOT / run_id / "physgaussian" / "super_motion" / "motion.bin", request)


@app.get("/api/runs/{run_id}/indices.bin")
def get_run_motion_indices(run_id: str, request: Request) -> Response:
    return _range_file_response(RUNS_ROOT / run_id / "physgaussian" / "super_motion" / "indices.bin", request)


@app.get("/api/runs/{run_id}/solver_trace.json")
def get_run_solver_trace(run_id: str) -> Response:
    trace = RUNS_ROOT / run_id / "physgaussian" / "solver_trace.json"
    if not trace.exists():
        legacy_trace = RUNS_ROOT / run_id / "physgaussian" / "implicit_solver_trace.json"
        trace = legacy_trace if legacy_trace.exists() else trace
    if not trace.exists():
        raise HTTPException(status_code=404, detail=f"solver trace not found for runId: {run_id}")
    return Response(
        trace.read_bytes(),
        media_type="application/json",
        headers={
            "Content-Disposition": f"attachment; filename=solver_trace_{run_id}.json",
            "Cache-Control": "no-store",
        },
    )


@app.get("/api/runs/{run_id}/proxy_motion.bin")
def get_run_proxy_motion_binary(run_id: str, request: Request) -> Response:
    return _range_file_response(RUNS_ROOT / run_id / "physgaussian" / "super_motion" / "proxy_motion.bin", request)


@app.get("/api/runs/{run_id}/proxy_skinning.bin")
def get_run_proxy_skinning_binary(run_id: str, request: Request) -> Response:
    return _range_file_response(RUNS_ROOT / run_id / "physgaussian" / "super_motion" / "proxy_skinning.bin", request)


def _run_simulation_job(run_id: str, model: dict[str, Any]) -> None:
    run_dir = RUNS_ROOT / run_id
    output_dir = run_dir / "physgaussian"
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    cmd = [
        PYTHON_BIN,
        "gs_simulation.py",
        "--model_path",
        model["modelRoot"],
        "--output_path",
        str(output_dir),
        "--config",
        str(run_dir / "config.json"),
        "--output_super_motion",
    ]
    iteration = _base_ply_iteration(model)
    if iteration >= 0:
        cmd.extend(["--iteration", str(iteration)])

    try:
        _update_run(run_id, status="running", progress="physgaussian")
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            process = subprocess.Popen(
                cmd,
                cwd=PHYSGAUSSIAN_ROOT,
                stdout=stdout,
                stderr=stderr,
                text=True,
            )
            RUN_PROCESSES[run_id] = process
            while True:
                return_code = process.poll()
                record = _read_json(_run_record_path(run_id))
                if record.get("status") == "cancelled":
                    if process.poll() is None:
                        process.terminate()
                    return
                progress_patch = _motion_progress(run_id, _read_json(run_dir / "config.json"))
                if progress_patch.get("availableFrames"):
                    _update_run(run_id, progress="streaming", **progress_patch)
                if return_code is not None:
                    break
                time.sleep(0.5)
        RUN_PROCESSES.pop(run_id, None)
        if return_code != 0:
            raise RuntimeError(f"PhysGaussian failed with exit code {return_code}\n{_tail(stderr_path)}")

        motion_dir = output_dir / "super_motion"
        manifest = motion_dir / "motion.physmotion.json"
        binary = motion_dir / "motion.bin"
        indices = motion_dir / "indices.bin"
        if not manifest.exists() or not binary.exists() or not indices.exists():
            raise RuntimeError("PhysGaussian did not write a SuperSplat motion package")

        base_ply = Path(model["basePly"]) if model.get("basePly") else None
        if base_ply and base_ply.exists():
            motion_manifest = _read_json(manifest)
            motion_manifest.setdefault("base", base_ply.name)
            _write_json(manifest, motion_manifest)

        result_payload = {
            "manifestUrl": _url_under(RUNS_ROOT, manifest, "/outputs"),
            "binaryUrl": f"/api/runs/{run_id}/motion.bin",
            "indicesUrl": f"/api/runs/{run_id}/indices.bin",
            "traceUrl": f"/api/runs/{run_id}/solver_trace.json",
            "basePlyUrl": _model_base_ply_url(model, base_ply),
            "basePlyName": base_ply.name if base_ply else None,
        }
        progress_patch = _motion_progress(run_id, _read_json(run_dir / "config.json"))
        progress_patch.pop("result", None)
        _update_run(run_id, status="completed", progress="motion-ready", result=result_payload, error=None, **progress_patch)
    except Exception as exc:  # noqa: BLE001
        RUN_PROCESSES.pop(run_id, None)
        current = _read_json(_run_record_path(run_id)) if _run_record_path(run_id).exists() else {}
        if current.get("status") == "cancelled":
            return
        _update_run(
            run_id,
            status="failed",
            progress="failed",
            error=str(exc),
            stdoutTail=_tail(stdout_path),
            stderrTail=_tail(stderr_path),
        )


if SUPERSPLAT_DIST.exists():
    app.mount("/", StaticFiles(directory=SUPERSPLAT_DIST, html=True), name="frontend")
