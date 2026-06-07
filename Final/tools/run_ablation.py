#!/usr/bin/env python3
"""Run PhysGaussian / i-PhysGaussian / PBMPM ablation experiments.

The runner intentionally keeps the official ficus/flowerpot scene fixed. It
copies the base config for each experiment and only overrides the requested
solver/time/grid/tolerance/PBMPM fields.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover - metrics degrade gracefully without numpy.
    np = None  # type: ignore[assignment]


METHODS = ("explicit", "implicit", "pbmpm")
GRIDS = (25, 50, 100)
FRAME_DTS = ("0.02", "0.04", "0.06")
SUBSTEP_DTS = ("0.0001", "0.0005", "0.001")
PBMPM_STRENGTH_SCALES = ("0.25", "0.5", "1.0", "2.0")
INTERNAL_ABLATION_GRID = 50
INTERNAL_ABLATION_FRAME_DT = "0.02"
INTERNAL_ABLATION_SUBSTEP_DT = "0.0001"
DEFAULT_TOLERANCE_PROFILE = "default"
DEFAULT_PBMPM_STRENGTH_SCALE = "1.0"

TOLERANCE_PROFILES: dict[str, dict[str, Any]] = {
    "strict": {
        "newton_tol": 2e-4,
        "newton_abs_tol": 1e-6,
        "newton_rms_tol": 5e-5,
        "gmres_tol_floor": 1e-4,
        "ew_eta_min": 1e-4,
        "ew_eta_max": 0.1,
        "gmres_max_iter": 48,
        "newton_max_iter": 16,
    },
    "default": {
        "newton_tol": 5e-4,
        "newton_abs_tol": 1e-5,
        "newton_rms_tol": 1e-4,
        "gmres_tol_floor": 1e-3,
        "ew_eta_min": 1e-3,
        "ew_eta_max": 0.2,
        "gmres_max_iter": 24,
        "newton_max_iter": 16,
    },
    "relaxed": {
        "newton_tol": 1e-3,
        "newton_abs_tol": 1e-5,
        "newton_rms_tol": 2e-4,
        "gmres_tol_floor": 2e-3,
        "ew_eta_min": 2e-3,
        "ew_eta_max": 0.3,
        "gmres_max_iter": 20,
        "newton_max_iter": 12,
    },
}

COMMON_IMPLICIT_DEFAULTS = {
    "beta": 0.25,
    "gamma": 0.5,
    "jvp_eps": 1e-4,
    "line_search_max_iter": 8,
    "armijo_c1": 1e-4,
    "ew_gamma": 0.9,
    "ew_alpha": 1.5,
    "stiffness_preconditioner_scale": 1.0,
    "stagnation_tol": 1e-8,
    "allow_best_effort_commit": False,
    "near_converged_factor": 2.0,
    "near_newton_rms_tol": 1e-4,
    "fallback_descent_tol": 1e-8,
    "fallback_step_min_rel": 1e-8,
    "fallback_decrease_tol": 1e-6,
    "adaptive_max_split": 3,
}

SUMMARY_COLUMNS = [
    "run_id",
    "method",
    "grid",
    "frame_dt",
    "substep_dt",
    "step_per_frame",
    "tolerance_profile",
    "pbmpm_strength_scale",
    "success",
    "failure_reason",
    "total_wall_time",
    "mean_time_per_frame",
    "actual_substep_count",
    "failed_substep_count",
    "adaptive_split_count",
    "total_newton_iters",
    "mean_newton_iters_per_substep",
    "total_gmres_iters",
    "mean_gmres_iters_per_newton",
    "total_line_search_evals",
    "fallback_used_count",
    "total_pbmpm_iters",
    "mean_pbmpm_iters_per_substep",
    "out_of_bounds_count",
    "particle_clamp_count",
    "boundary_projection_count",
    "final_relative_residual_mean",
    "final_relative_residual_max",
    "final_residual_rms_mean",
    "final_residual_rms_max",
    "bbox_volume_start",
    "bbox_volume_end",
    "max_displacement",
    "mean_displacement",
    "motion_path",
    "video_path",
    "trace_path",
    "config_path",
]


@dataclass(frozen=True)
class Experiment:
    method: str
    grid: int
    frame_dt: str
    substep_dt: str
    frame_num: int
    step_per_frame: int
    tolerance_profile: str | None = None
    pbmpm_strength_scale: str | None = None


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def stable_hash(data: Any, length: int = 10) -> str:
    blob = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:length]


def parse_decimal(text: str | float | int) -> Decimal:
    try:
        return Decimal(str(text))
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError(f"not a decimal value: {text}") from exc


def float_text(value: str | float | int | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return format(float(value), ".10g")


def exact_step_per_frame(frame_dt: str, substep_dt: str) -> int:
    frame = parse_decimal(frame_dt)
    substep = parse_decimal(substep_dt)
    if frame <= 0 or substep <= 0:
        raise ValueError(f"frame_dt and substep_dt must be positive: {frame_dt}, {substep_dt}")
    ratio = frame / substep
    integral = ratio.to_integral_value()
    if ratio != integral:
        raise ValueError(
            "frame_dt / substep_dt must be an integer; "
            f"got frame_dt={frame_dt}, substep_dt={substep_dt}, ratio={ratio}"
        )
    return int(integral)


def default_phys_root() -> Path:
    env = os.environ.get("PHYSGAUSSIAN_ROOT")
    if env:
        return Path(env)
    external = Path.cwd() / "external" / "PhysGaussian"
    if external.exists():
        return external
    local = Path.cwd() / "src-overrides" / "physgaussian-src"
    return local


def default_python_bin() -> str:
    return os.environ.get("PYTHON_BIN") or sys.executable


def default_base_config(phys_root: Path) -> Path:
    return phys_root / "config" / "ficus_config.json"


def default_model_path(phys_root: Path) -> Path:
    return phys_root / "model" / "ficus_whitebg-trained"


def selected(values: list[Any] | None, defaults: tuple[Any, ...]) -> list[Any]:
    return list(values) if values else list(defaults)


def build_experiments(args: argparse.Namespace) -> list[Experiment]:
    methods = selected(args.method, METHODS)
    grids = [int(v) for v in selected(args.grid, GRIDS)]
    frame_dts = [float_text(v) for v in selected(args.frame_dt, FRAME_DTS)]
    substep_dts = [float_text(v) for v in selected(args.substep_dt, SUBSTEP_DTS)]
    profiles = selected(args.tolerance_profile, tuple(TOLERANCE_PROFILES))
    strengths = [float_text(v) for v in selected(args.pbmpm_strength_scale, PBMPM_STRENGTH_SCALES)]

    if args.sanity_suite:
        methods = list(METHODS)
        grids = [50]
        frame_dts = ["0.02"]
        substep_dts = ["0.0005"]
        profiles = ["default"]
        strengths = ["1.0"]
        args.frame_num = 3

    experiments: list[Experiment] = []

    def append_unique(experiment: Experiment) -> None:
        if experiment not in experiments:
            experiments.append(experiment)

    def append_one_factor_method(
        method: str,
        tolerance_profile: str | None = None,
        pbmpm_strength_scale: str | None = None,
    ) -> None:
        for grid in grids:
            step_per_frame = exact_step_per_frame(
                INTERNAL_ABLATION_FRAME_DT, INTERNAL_ABLATION_SUBSTEP_DT
            )
            append_unique(
                Experiment(
                    method=method,
                    grid=grid,
                    frame_dt=INTERNAL_ABLATION_FRAME_DT,
                    substep_dt=INTERNAL_ABLATION_SUBSTEP_DT,
                    frame_num=args.frame_num,
                    step_per_frame=step_per_frame,
                    tolerance_profile=tolerance_profile,
                    pbmpm_strength_scale=pbmpm_strength_scale,
                )
            )
        if INTERNAL_ABLATION_GRID in grids or not args.grid:
            for frame_dt in frame_dts:
                assert frame_dt is not None
                step_per_frame = exact_step_per_frame(frame_dt, INTERNAL_ABLATION_SUBSTEP_DT)
                append_unique(
                    Experiment(
                        method=method,
                        grid=INTERNAL_ABLATION_GRID,
                        frame_dt=frame_dt,
                        substep_dt=INTERNAL_ABLATION_SUBSTEP_DT,
                        frame_num=args.frame_num,
                        step_per_frame=step_per_frame,
                        tolerance_profile=tolerance_profile,
                        pbmpm_strength_scale=pbmpm_strength_scale,
                    )
                )
            for substep_dt in substep_dts:
                assert substep_dt is not None
                step_per_frame = exact_step_per_frame(
                    INTERNAL_ABLATION_FRAME_DT, substep_dt
                )
                append_unique(
                    Experiment(
                        method=method,
                        grid=INTERNAL_ABLATION_GRID,
                        frame_dt=INTERNAL_ABLATION_FRAME_DT,
                        substep_dt=substep_dt,
                        frame_num=args.frame_num,
                        step_per_frame=step_per_frame,
                        tolerance_profile=tolerance_profile,
                        pbmpm_strength_scale=pbmpm_strength_scale,
                    )
                )

    default_all = (
        not args.method
        and not args.grid
        and not args.frame_dt
        and not args.substep_dt
        and not args.tolerance_profile
        and not args.pbmpm_strength_scale
        and not args.sanity_suite
    )
    if default_all:
        if "explicit" in methods:
            for grid in grids:
                for frame_dt in frame_dts:
                    assert frame_dt is not None
                    for substep_dt in substep_dts:
                        assert substep_dt is not None
                        append_unique(
                            Experiment(
                                method="explicit",
                                grid=grid,
                                frame_dt=frame_dt,
                                substep_dt=substep_dt,
                                frame_num=args.frame_num,
                                step_per_frame=exact_step_per_frame(frame_dt, substep_dt),
                            )
                        )
        if "implicit" in methods:
            append_one_factor_method(
                "implicit", tolerance_profile=DEFAULT_TOLERANCE_PROFILE
            )
            for profile in profiles:
                append_unique(
                    Experiment(
                        method="implicit",
                        grid=INTERNAL_ABLATION_GRID,
                        frame_dt=INTERNAL_ABLATION_FRAME_DT,
                        substep_dt=INTERNAL_ABLATION_SUBSTEP_DT,
                        frame_num=args.frame_num,
                        step_per_frame=exact_step_per_frame(
                            INTERNAL_ABLATION_FRAME_DT, INTERNAL_ABLATION_SUBSTEP_DT
                        ),
                        tolerance_profile=profile,
                    )
                )
        if "pbmpm" in methods:
            append_one_factor_method(
                "pbmpm", pbmpm_strength_scale=DEFAULT_PBMPM_STRENGTH_SCALE
            )
            for strength in strengths:
                append_unique(
                    Experiment(
                        method="pbmpm",
                        grid=INTERNAL_ABLATION_GRID,
                        frame_dt=INTERNAL_ABLATION_FRAME_DT,
                        substep_dt=INTERNAL_ABLATION_SUBSTEP_DT,
                        frame_num=args.frame_num,
                        step_per_frame=exact_step_per_frame(
                            INTERNAL_ABLATION_FRAME_DT, INTERNAL_ABLATION_SUBSTEP_DT
                        ),
                        pbmpm_strength_scale=strength,
                    )
                )
        if args.max_runs:
            experiments = experiments[: args.max_runs]
        return experiments

    for method in methods:
        for grid in grids:
            for frame_dt in frame_dts:
                assert frame_dt is not None
                for substep_dt in substep_dts:
                    assert substep_dt is not None
                    step_per_frame = exact_step_per_frame(frame_dt, substep_dt)
                    if method == "implicit":
                        main_profiles = (
                            profiles
                            if args.tolerance_profile
                            else [DEFAULT_TOLERANCE_PROFILE]
                        )
                        for profile in main_profiles:
                            append_unique(
                                Experiment(
                                    method=method,
                                    grid=grid,
                                    frame_dt=frame_dt,
                                    substep_dt=substep_dt,
                                    frame_num=args.frame_num,
                                    step_per_frame=step_per_frame,
                                    tolerance_profile=profile,
                                )
                            )
                    elif method == "pbmpm":
                        main_strengths = (
                            strengths
                            if args.pbmpm_strength_scale
                            else [DEFAULT_PBMPM_STRENGTH_SCALE]
                        )
                        for strength in main_strengths:
                            append_unique(
                                Experiment(
                                    method=method,
                                    grid=grid,
                                    frame_dt=frame_dt,
                                    substep_dt=substep_dt,
                                    frame_num=args.frame_num,
                                    step_per_frame=step_per_frame,
                                    pbmpm_strength_scale=strength,
                                )
                            )
                    else:
                        append_unique(
                            Experiment(
                                method=method,
                                grid=grid,
                                frame_dt=frame_dt,
                                substep_dt=substep_dt,
                                frame_num=args.frame_num,
                                step_per_frame=step_per_frame,
                            )
                        )

    internal_grid_selected = (
        INTERNAL_ABLATION_GRID in grids if args.grid else True
    )
    internal_frame_selected = (
        INTERNAL_ABLATION_FRAME_DT in frame_dts if args.frame_dt else True
    )
    internal_substep_selected = (
        INTERNAL_ABLATION_SUBSTEP_DT in substep_dts if args.substep_dt else True
    )
    include_internal_ablation = (
        internal_grid_selected and internal_frame_selected and internal_substep_selected
    )
    if include_internal_ablation:
        internal_step_per_frame = exact_step_per_frame(
            INTERNAL_ABLATION_FRAME_DT, INTERNAL_ABLATION_SUBSTEP_DT
        )
        if "implicit" in methods and not args.tolerance_profile:
            for profile in profiles:
                append_unique(
                    Experiment(
                        method="implicit",
                        grid=INTERNAL_ABLATION_GRID,
                        frame_dt=INTERNAL_ABLATION_FRAME_DT,
                        substep_dt=INTERNAL_ABLATION_SUBSTEP_DT,
                        frame_num=args.frame_num,
                        step_per_frame=internal_step_per_frame,
                        tolerance_profile=profile,
                    )
                )
        if "pbmpm" in methods and not args.pbmpm_strength_scale:
            for strength in strengths:
                append_unique(
                    Experiment(
                        method="pbmpm",
                        grid=INTERNAL_ABLATION_GRID,
                        frame_dt=INTERNAL_ABLATION_FRAME_DT,
                        substep_dt=INTERNAL_ABLATION_SUBSTEP_DT,
                        frame_num=args.frame_num,
                        step_per_frame=internal_step_per_frame,
                        pbmpm_strength_scale=strength,
                    )
                )
    if args.max_runs:
        experiments = experiments[: args.max_runs]
    return experiments


def run_id_for(experiment: Experiment) -> str:
    pieces = [
        experiment.method,
        f"g{experiment.grid}",
        f"f{experiment.frame_dt}".replace(".", "p"),
        f"s{experiment.substep_dt}".replace(".", "p"),
    ]
    if experiment.tolerance_profile:
        pieces.append(experiment.tolerance_profile)
    if experiment.pbmpm_strength_scale:
        pieces.append(f"str{experiment.pbmpm_strength_scale}".replace(".", "p"))
    pieces.append(stable_hash(experiment.__dict__, 8))
    return "_".join(pieces)


def override_config(
    base_config: dict[str, Any],
    base_config_path: Path,
    experiment: Experiment,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = json.loads(json.dumps(base_config))
    overridden: dict[str, Any] = {}

    def set_field(path: str, value: Any) -> None:
        parts = path.split(".")
        target = config
        for key in parts[:-1]:
            target = target.setdefault(key, {})
        old = target.get(parts[-1])
        target[parts[-1]] = value
        overridden[path] = {"old": old, "new": value}

    set_field("integrator", experiment.method)
    set_field("n_grid", experiment.grid)
    set_field("frame_dt", float(experiment.frame_dt))
    set_field("substep_dt", float(experiment.substep_dt))
    set_field("frame_num", int(experiment.frame_num))
    set_field("step_per_frame", int(experiment.step_per_frame))
    set_field("expected_substep_count", int(experiment.step_per_frame * experiment.frame_num))

    if experiment.method == "implicit":
        implicit = dict(COMMON_IMPLICIT_DEFAULTS)
        implicit.update(TOLERANCE_PROFILES[experiment.tolerance_profile or "default"])
        implicit["tolerance_profile"] = experiment.tolerance_profile or "default"
        implicit["adaptive_min_dt"] = max(float(experiment.substep_dt) / 8.0, 1e-8)
        old = config.get("implicit_mpm")
        config["implicit_mpm"] = implicit
        config.pop("pbmpm", None)
        overridden["implicit_mpm"] = {"old": old, "new": implicit}
        overridden["pbmpm"] = {"old": base_config.get("pbmpm"), "new": None}
    elif experiment.method == "pbmpm":
        strength = float(experiment.pbmpm_strength_scale or 1.0)
        pbmpm = {
            "strength_scale": strength,
            "n_min": 3,
            "n_max": 25,
            "plastic_mode": 0,
            "yield_min": 0.55,
            "yield_max": 1.85,
            "pbmpm_strength_scale": strength,
        }
        old = config.get("pbmpm")
        config["pbmpm"] = pbmpm
        config.pop("implicit_mpm", None)
        overridden["pbmpm"] = {"old": old, "new": pbmpm}
        overridden["implicit_mpm"] = {"old": base_config.get("implicit_mpm"), "new": None}
    else:
        config.pop("implicit_mpm", None)
        config.pop("pbmpm", None)
        overridden["implicit_mpm"] = {"old": base_config.get("implicit_mpm"), "new": None}
        overridden["pbmpm"] = {"old": base_config.get("pbmpm"), "new": None}

    config["_ablation"] = {
        "base_config_path": str(base_config_path),
        "base_config_name": base_config_path.name,
        "method": experiment.method,
        "grid": experiment.grid,
        "frame_dt": float(experiment.frame_dt),
        "substep_dt": float(experiment.substep_dt),
        "step_per_frame": experiment.step_per_frame,
        "frame_num": experiment.frame_num,
        "expected_substep_count": experiment.frame_num * experiment.step_per_frame,
        "tolerance_profile": experiment.tolerance_profile,
        "pbmpm_strength_scale": (
            float(experiment.pbmpm_strength_scale)
            if experiment.pbmpm_strength_scale is not None
            else None
        ),
        "overridden_fields": overridden,
    }
    return config, overridden


def make_plan_rows(experiments: list[Experiment]) -> list[dict[str, Any]]:
    rows = []
    for experiment in experiments:
        rows.append(
            {
                "run_id": run_id_for(experiment),
                "method": experiment.method,
                "grid": experiment.grid,
                "frame_dt": experiment.frame_dt,
                "substep_dt": experiment.substep_dt,
                "step_per_frame": experiment.step_per_frame,
                "frame_num": experiment.frame_num,
                "expected_substep_count": experiment.frame_num * experiment.step_per_frame,
                "tolerance_profile": experiment.tolerance_profile,
                "pbmpm_strength_scale": experiment.pbmpm_strength_scale,
            }
        )
    return rows


def write_plan(output_root: Path, rows: list[dict[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "ablation_plan.json", rows)
    csv_path = output_root / "ablation_plan.csv"
    columns = [
        "run_id",
        "method",
        "grid",
        "frame_dt",
        "substep_dt",
        "step_per_frame",
        "frame_num",
        "expected_substep_count",
        "tolerance_profile",
        "pbmpm_strength_scale",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        result = float(value)
        return result if math.isfinite(result) else None
    except Exception:
        return None


def mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def finite_values(values: list[Any]) -> list[float]:
    result = []
    for value in values:
        number = safe_float(value)
        if number is not None:
            result.append(number)
    return result


def get_nested(data: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = data
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def summarize_trace(trace_path: Path) -> dict[str, Any]:
    trace = read_json(trace_path, {}) or {}
    steps = trace.get("steps") if isinstance(trace, dict) else []
    if not isinstance(steps, list):
        steps = []
    summary = trace.get("summary") if isinstance(trace.get("summary"), dict) else {}
    metrics: dict[str, Any] = {
        "trace_path": str(trace_path) if trace_path.exists() else "",
        "actual_substep_count": int(summary.get("step_count", len(steps)) or 0),
        "failed_substep_count": int(summary.get("failed_step_count", 0) or 0),
        "trace_failure_reason": summary.get("failure_reason") or "",
        "trace_failure_stage": summary.get("failure_stage") or "",
        "post_commit_projection_only": None,
        "total_newton_iters": int(summary.get("total_newton_iters", 0) or 0),
        "total_gmres_iters": int(summary.get("total_gmres_iters", 0) or 0),
        "total_line_search_evals": int(summary.get("total_line_search_evals", 0) or 0),
        "fallback_used_count": int(summary.get("total_fallback_used", 0) or 0),
    }
    last_step = summary.get("last_step")
    if isinstance(last_step, dict):
        metrics["trace_failure_reason"] = (
            metrics["trace_failure_reason"] or last_step.get("failure_reason") or ""
        )
        metrics["trace_failure_stage"] = (
            metrics["trace_failure_stage"] or last_step.get("failure_stage") or ""
        )
    if not steps:
        return metrics

    failed_step = next(
        (step for step in reversed(steps) if isinstance(step, dict) and step.get("substep_failed")),
        None,
    )
    if failed_step:
        metrics["trace_failure_reason"] = (
            failed_step.get("failure_reason") or metrics["trace_failure_reason"] or ""
        )
        metrics["trace_failure_stage"] = (
            failed_step.get("failure_stage") or metrics["trace_failure_stage"] or ""
        )

    convergence = Counter(str(step.get("convergence_type", "unknown")) for step in steps)
    newton_iters = finite_values([step.get("newton_iters") for step in steps])
    gmres_iters = finite_values([step.get("gmres_iters") for step in steps])
    line_search = finite_values([step.get("line_search_evals") for step in steps])
    final_residual = finite_values([step.get("final_residual") for step in steps])
    final_relative = finite_values([step.get("final_relative_residual") for step in steps])
    final_rms = finite_values([step.get("final_residual_rms") for step in steps])
    grid_mass_nodes = finite_values([get_nested(step, "grid_summary.mass_nodes", step.get("grid_mass_nodes")) for step in steps])
    active_free_nodes = finite_values([step.get("active_free_nodes") for step in steps])
    dirichlet_nodes = finite_values([step.get("dirichlet_nodes") for step in steps])
    cuboid_nodes = finite_values([step.get("cuboid_dirichlet_nodes") for step in steps])
    surface_nodes = finite_values([step.get("surface_dirichlet_nodes") for step in steps])
    pbmpm_iters = finite_values([get_nested(step, "pbmpm.iteration_count") for step in steps])

    metrics.update(
        {
            "mean_newton_iters_per_substep": mean(newton_iters),
            "max_newton_iters": max(newton_iters) if newton_iters else None,
            "mean_gmres_iters_per_newton": (
                metrics["total_gmres_iters"] / metrics["total_newton_iters"]
                if metrics["total_newton_iters"]
                else None
            ),
            "max_gmres_iters": max(gmres_iters) if gmres_iters else None,
            "gmres_failed_count": int(sum(int(step.get("gmres_failed_count", 0) or 0) for step in steps)),
            "mean_line_search_evals": mean(line_search),
            "max_line_search_evals": max(line_search) if line_search else None,
            "line_search_failure_count": int(sum(int(step.get("line_search_failure_count", 0) or 0) for step in steps)),
            "adaptive_split_count": int(sum(1 for step in steps if step.get("adaptive_retry") == "split")),
            "adaptive_retry_exhausted_count": int(sum(1 for step in steps if step.get("adaptive_retry") == "exhausted")),
            "convergence_type_distribution": dict(convergence),
            "final_residual_mean": mean(final_residual),
            "final_residual_max": max(final_residual) if final_residual else None,
            "final_relative_residual_mean": mean(final_relative),
            "final_relative_residual_max": max(final_relative) if final_relative else None,
            "final_residual_rms_mean": mean(final_rms),
            "final_residual_rms_max": max(final_rms) if final_rms else None,
            "state_restored_on_failure_count": int(sum(1 for step in steps if step.get("state_restored_on_failure"))),
            "out_of_bounds_count": int(sum(int(step.get("out_of_bounds_count", 0) or 0) for step in steps)),
            "pre_solve_clamp_count": int(sum(int(step.get("pre_solve_clamp_count", 0) or 0) for step in steps)),
            "particle_clamp_count": int(
                sum(
                    int(step.get("particle_clamp_count", step.get("particle_clamp_count_estimate", 0)) or 0)
                    for step in steps
                )
            ),
            "boundary_projection_count": int(sum(int(step.get("boundary_projection_count", 0) or 0) for step in steps)),
            "grid_mass_nodes_mean": mean(grid_mass_nodes),
            "grid_mass_nodes_max": max(grid_mass_nodes) if grid_mass_nodes else None,
            "active_free_nodes_mean": mean(active_free_nodes),
            "active_free_nodes_max": max(active_free_nodes) if active_free_nodes else None,
            "dirichlet_nodes_mean": mean(dirichlet_nodes),
            "dirichlet_nodes_max": max(dirichlet_nodes) if dirichlet_nodes else None,
            "cuboid_dirichlet_nodes_mean": mean(cuboid_nodes),
            "cuboid_dirichlet_nodes_max": max(cuboid_nodes) if cuboid_nodes else None,
            "surface_dirichlet_nodes_mean": mean(surface_nodes),
            "surface_dirichlet_nodes_max": max(surface_nodes) if surface_nodes else None,
            "total_pbmpm_iters": int(sum(pbmpm_iters)) if pbmpm_iters else 0,
            "mean_pbmpm_iters_per_substep": mean(pbmpm_iters),
            "max_pbmpm_iters": max(pbmpm_iters) if pbmpm_iters else None,
            "projection_residual_mean": None,
            "projection_residual_max": None,
            "constraint_residual_mean": None,
            "constraint_residual_max": None,
            "stress_F_explosion_count": int(sum(1 for step in steps if step.get("stability_failure"))),
            "stability_failure_flag": bool(any(step.get("stability_failure") for step in steps)),
            "post_commit_projection_only": bool(any(step.get("post_commit_projection_only") for step in steps)),
        }
    )
    last_pbmpm = next((step.get("pbmpm") for step in reversed(steps) if isinstance(step.get("pbmpm"), dict)), None)
    if last_pbmpm:
        metrics.update(
            {
                "mapped_pbmpm_parameters": last_pbmpm,
                "pbmpm_iteration_count": last_pbmpm.get("iteration_count"),
                "pbmpm_strength_scale_observed": last_pbmpm.get("strength_scale"),
                "pbmpm_elasticity_ratio": last_pbmpm.get("elasticity_ratio"),
                "pbmpm_elastic_relaxation": last_pbmpm.get("elastic_relaxation"),
            }
        )
    return metrics


def analyze_motion(run_dir: Path, frame_dt: float) -> dict[str, Any]:
    motion_dir = run_dir / "physgaussian" / "super_motion"
    manifest_path = motion_dir / "motion.physmotion.json"
    motion_path = motion_dir / "motion.bin"
    indices_path = motion_dir / "indices.bin"
    metrics: dict[str, Any] = {
        "motion_path": str(motion_path) if motion_path.exists() else "",
        "motion_manifest_path": str(manifest_path) if manifest_path.exists() else "",
        "indices_path": str(indices_path) if indices_path.exists() else "",
        "available_motion_frames": 0,
        "bbox_center_start": None,
        "bbox_center_end": None,
        "bbox_size_start": None,
        "bbox_size_end": None,
        "bbox_volume_start": None,
        "bbox_volume_end": None,
        "max_displacement": None,
        "mean_displacement": None,
        "max_velocity": None,
        "mean_velocity": None,
        "max_acceleration": None,
        "mean_acceleration": None,
        "covariance_eigenvalue_min": None,
        "covariance_eigenvalue_max": None,
        "sampled_motion_path": "",
        "bbox_motion_path": "",
        "center_trajectory_path": "",
        "NaN_count": 0,
        "Inf_count": 0,
    }
    if np is None or not manifest_path.exists() or not motion_path.exists():
        return metrics
    manifest = read_json(manifest_path, {}) or {}
    stride = int(manifest.get("frameStrideBytes") or 0)
    if stride <= 0:
        return metrics
    available = motion_path.stat().st_size // stride
    count = stride // (10 * 4)
    if available <= 0 or count <= 0:
        return metrics
    metrics["available_motion_frames"] = int(available)
    sample_indices = sorted({0, max(0, available // 2), available - 1})
    samples: dict[str, list[Any]] = {"frame": [], "position": [], "rotation": [], "scale": []}
    bbox_rows: list[dict[str, Any]] = []
    center_rows: list[dict[str, Any]] = []
    first_pos = None
    prev_pos = None
    prev_vel = None
    global_max_disp = 0.0
    final_mean_disp = None
    global_max_velocity = 0.0
    velocity_sum = 0.0
    velocity_count = 0
    global_max_acc = 0.0
    scale_min = None
    scale_max = None
    nan_count = 0
    inf_count = 0

    with motion_path.open("rb") as stream:
        for frame in range(available):
            raw = stream.read(stride)
            if len(raw) < stride:
                break
            values = np.frombuffer(raw, dtype=np.float32)
            positions = values[: count * 3].reshape(count, 3)
            rotations = values[count * 3 : count * 7].reshape(count, 4)
            scales = values[count * 7 : count * 10].reshape(count, 3)
            nan_count += int(np.isnan(values).sum())
            inf_count += int(np.isinf(values).sum())
            finite_pos = positions[np.isfinite(positions).all(axis=1)]
            if finite_pos.size:
                mins = finite_pos.min(axis=0)
                maxs = finite_pos.max(axis=0)
                center = finite_pos.mean(axis=0)
                size = maxs - mins
                volume = float(np.prod(size))
                bbox_rows.append(
                    {
                        "frame": frame,
                        "center_x": float(center[0]),
                        "center_y": float(center[1]),
                        "center_z": float(center[2]),
                        "size_x": float(size[0]),
                        "size_y": float(size[1]),
                        "size_z": float(size[2]),
                        "volume": volume,
                    }
                )
                center_rows.append(
                    {
                        "frame": frame,
                        "x": float(center[0]),
                        "y": float(center[1]),
                        "z": float(center[2]),
                    }
                )
            if frame in sample_indices:
                samples["frame"].append(frame)
                samples["position"].append(np.array(positions, copy=True))
                samples["rotation"].append(np.array(rotations, copy=True))
                samples["scale"].append(np.array(scales, copy=True))
            finite_scales = scales[np.isfinite(scales)]
            if finite_scales.size:
                cur_min = float(finite_scales.min())
                cur_max = float(finite_scales.max())
                scale_min = cur_min if scale_min is None else min(scale_min, cur_min)
                scale_max = cur_max if scale_max is None else max(scale_max, cur_max)
            if first_pos is None:
                first_pos = np.array(positions, copy=True)
            else:
                disp = np.linalg.norm(positions - first_pos, axis=1)
                finite_disp = disp[np.isfinite(disp)]
                if finite_disp.size:
                    global_max_disp = max(global_max_disp, float(finite_disp.max()))
                    if frame == available - 1:
                        final_mean_disp = float(finite_disp.mean())
            if prev_pos is not None and frame_dt > 0:
                vel = (positions - prev_pos) / frame_dt
                speed = np.linalg.norm(vel, axis=1)
                finite_speed = speed[np.isfinite(speed)]
                if finite_speed.size:
                    global_max_velocity = max(global_max_velocity, float(finite_speed.max()))
                    velocity_sum += float(finite_speed.sum())
                    velocity_count += int(finite_speed.size)
                if prev_vel is not None:
                    acc = (vel - prev_vel) / frame_dt
                    acc_norm = np.linalg.norm(acc, axis=1)
                    finite_acc = acc_norm[np.isfinite(acc_norm)]
                    if finite_acc.size:
                        global_max_acc = max(global_max_acc, float(finite_acc.max()))
                prev_vel = vel
            prev_pos = np.array(positions, copy=True)

    if bbox_rows:
        first = bbox_rows[0]
        last = bbox_rows[-1]
        metrics.update(
            {
                "bbox_center_start": [first["center_x"], first["center_y"], first["center_z"]],
                "bbox_center_end": [last["center_x"], last["center_y"], last["center_z"]],
                "bbox_size_start": [first["size_x"], first["size_y"], first["size_z"]],
                "bbox_size_end": [last["size_x"], last["size_y"], last["size_z"]],
                "bbox_volume_start": first["volume"],
                "bbox_volume_end": last["volume"],
            }
        )
    metrics.update(
        {
            "max_displacement": global_max_disp if available > 1 else 0.0,
            "mean_displacement": final_mean_disp,
            "max_velocity": global_max_velocity if velocity_count else None,
            "mean_velocity": velocity_sum / velocity_count if velocity_count else None,
            "max_acceleration": global_max_acc if available > 2 else None,
            "covariance_eigenvalue_min": scale_min,
            "covariance_eigenvalue_max": scale_max,
            "NaN_count": nan_count,
            "Inf_count": inf_count,
        }
    )

    bbox_path = run_dir / "bbox_motion.csv"
    center_path = run_dir / "center_trajectory.csv"
    if bbox_rows:
        with bbox_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(bbox_rows[0]))
            writer.writeheader()
            writer.writerows(bbox_rows)
        with center_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(center_rows[0]))
            writer.writeheader()
            writer.writerows(center_rows)
        metrics["bbox_motion_path"] = str(bbox_path)
        metrics["center_trajectory_path"] = str(center_path)

    if samples["frame"]:
        sampled_path = run_dir / "sampled_motion.npz"
        np.savez_compressed(
            sampled_path,
            frame=np.array(samples["frame"], dtype=np.int32),
            position=np.stack(samples["position"]),
            rotation=np.stack(samples["rotation"]),
            scale=np.stack(samples["scale"]),
        )
        metrics["sampled_motion_path"] = str(sampled_path)
    return metrics


def infer_last_success(metrics: dict[str, Any], step_per_frame: int) -> None:
    available = int(metrics.get("available_motion_frames") or 0)
    if available > 0:
        metrics["last_successful_frame"] = max(0, available - 1)
    else:
        metrics["last_successful_frame"] = 0
    actual = int(metrics.get("actual_substep_count") or 0)
    metrics["last_successful_substep"] = max(0, actual - 1) if actual else 0
    if step_per_frame > 0 and actual:
        metrics["last_successful_frame_from_trace"] = actual // step_per_frame


def write_fallback_trace(run_dir: Path, metrics: dict[str, Any], failure: dict[str, Any]) -> Path:
    trace_path = run_dir / "physgaussian" / "solver_trace.json"
    if trace_path.exists():
        return trace_path
    trace = {
        "format": "physgaussian-solver-trace-v2",
        "trace_version": 2,
        "ablation_fallback_trace": True,
        "integrator": metrics.get("method"),
        "substep_dt": metrics.get("substep_dt"),
        "frame_dt": metrics.get("frame_dt"),
        "frame_num": metrics.get("frame_num"),
        "step_per_frame": metrics.get("step_per_frame"),
        "expected_substep_count": metrics.get("expected_substep_count"),
        "summary": {
            "step_count": 0,
            "failed_step_count": 1,
            "failure_reason": failure.get("failure_reason"),
            "failure_stage": failure.get("failure_stage"),
        },
        "failure": failure,
        "steps": [],
    }
    write_json(trace_path, trace)
    return trace_path


def collect_run_metrics(
    run_dir: Path,
    experiment: Experiment,
    config: dict[str, Any],
    base_config_path: Path,
    wall_time: float,
    success: bool,
    failure: dict[str, Any] | None = None,
) -> dict[str, Any]:
    failure = failure or {}
    trace_path = run_dir / "physgaussian" / "solver_trace.json"
    if not trace_path.exists():
        trace_path = write_fallback_trace(run_dir, {
            "method": experiment.method,
            "substep_dt": float(experiment.substep_dt),
            "frame_dt": float(experiment.frame_dt),
            "frame_num": experiment.frame_num,
            "step_per_frame": experiment.step_per_frame,
            "expected_substep_count": experiment.frame_num * experiment.step_per_frame,
        }, failure)

    metrics: dict[str, Any] = {
        "run_id": run_dir.name,
        "method": experiment.method,
        "base_config_name": base_config_path.name,
        "base_config_path": str(base_config_path),
        "grid": experiment.grid,
        "frame_dt": float(experiment.frame_dt),
        "substep_dt": float(experiment.substep_dt),
        "step_per_frame": experiment.step_per_frame,
        "frame_num": experiment.frame_num,
        "expected_substep_count": experiment.frame_num * experiment.step_per_frame,
        "success": bool(success),
        "failure_reason": "" if success else failure.get("failure_reason", "unknown"),
        "failure_stage": "" if success else failure.get("failure_stage", "unknown"),
        "exception_traceback": "" if success else failure.get("exception_traceback", ""),
        "total_wall_time": float(wall_time),
        "mean_time_per_frame": float(wall_time / max(experiment.frame_num, 1)),
        "mean_time_per_substep": float(wall_time / max(experiment.frame_num * experiment.step_per_frame, 1)),
        "simulated_seconds_per_real_second": float(
            (experiment.frame_num * float(experiment.frame_dt)) / max(wall_time, 1e-12)
        ),
        "peak_gpu_memory": None,
        "tolerance_profile": experiment.tolerance_profile,
        "pbmpm_strength_scale": (
            float(experiment.pbmpm_strength_scale) if experiment.pbmpm_strength_scale is not None else None
        ),
        "config_path": str(run_dir / "config.json"),
        "overridden_fields_path": str(run_dir / "overridden_fields.json"),
        "video_path": str(run_dir / "physgaussian" / "output.mp4")
        if (run_dir / "physgaussian" / "output.mp4").exists()
        else "",
    }
    if experiment.method == "implicit":
        implicit = config.get("implicit_mpm", {})
        for key in [
            "newton_tol",
            "newton_abs_tol",
            "newton_rms_tol",
            "newton_max_iter",
            "gmres_tol_floor",
            "ew_eta_min",
            "ew_eta_max",
            "gmres_max_iter",
            "jvp_eps",
            "line_search_max_iter",
            "armijo_c1",
        ]:
            metrics[key] = implicit.get(key)
    elif experiment.method == "pbmpm":
        metrics["pbmpm_strength_scale"] = float(experiment.pbmpm_strength_scale or 1.0)

    metrics.update(summarize_trace(trace_path))
    if not success and metrics.get("trace_failure_reason"):
        metrics["failure_reason"] = metrics["trace_failure_reason"]
    if not success and metrics.get("trace_failure_stage"):
        metrics["failure_stage"] = metrics["trace_failure_stage"]
    metrics.update(analyze_motion(run_dir, float(experiment.frame_dt)))
    infer_last_success(metrics, experiment.step_per_frame)
    if not success and not metrics.get("failure_reason"):
        metrics["failure_reason"] = "failed"
    write_json(run_dir / "metrics.json", metrics)
    return metrics


def run_experiment(
    experiment: Experiment,
    args: argparse.Namespace,
    base_config: dict[str, Any],
    base_config_path: Path,
) -> dict[str, Any]:
    run_id = run_id_for(experiment)
    run_dir = args.output_root / run_id
    if args.resume and (run_dir / "metrics.json").exists():
        print(f"[skip] {run_id} metrics.json exists")
        return read_json(run_dir / "metrics.json", {})

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "physgaussian").mkdir(parents=True, exist_ok=True)
    config, overridden = override_config(base_config, base_config_path, experiment)
    config["_ablation"]["run_id"] = run_id
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "overridden_fields.json", overridden)
    write_json(
        run_dir / "run_meta.json",
        {
            "run_id": run_id,
            "experiment": experiment.__dict__,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "base_config_path": str(base_config_path),
            "physgaussian_root": str(args.physgaussian_root),
            "model_path": str(args.model_path),
        },
    )

    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    cmd = [
        str(args.python_bin),
        "gs_simulation.py",
        "--model_path",
        str(args.model_path),
        "--iteration",
        str(args.iteration),
        "--output_path",
        str(run_dir / "physgaussian"),
        "--config",
        str(run_dir / "config.json"),
        "--output_super_motion",
    ]
    if args.render:
        cmd.extend(["--render_img", "--compile_video"])
    write_json(run_dir / "command.json", {"cmd": cmd, "cwd": str(args.physgaussian_root)})

    print(f"[run] {run_id}")
    start = time.perf_counter()
    success = False
    failure: dict[str, Any] = {}
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            result = subprocess.run(
                cmd,
                cwd=args.physgaussian_root,
                stdout=stdout,
                stderr=stderr,
                text=True,
                timeout=args.timeout_seconds,
                check=False,
            )
        success = result.returncode == 0
        if not success:
            failure = {
                "failure_reason": f"exit_code_{result.returncode}",
                "failure_stage": "physgaussian",
                "stderr_tail": tail_text(stderr_path),
                "stdout_tail": tail_text(stdout_path),
            }
    except subprocess.TimeoutExpired as exc:
        failure = {
            "failure_reason": "timeout",
            "failure_stage": "physgaussian",
            "exception_traceback": "".join(traceback.format_exception(exc)),
        }
    except Exception as exc:  # noqa: BLE001
        failure = {
            "failure_reason": type(exc).__name__,
            "failure_stage": "runner",
            "exception_traceback": "".join(traceback.format_exception(exc)),
        }
    wall_time = time.perf_counter() - start
    if not success and "exception_traceback" not in failure:
        failure["exception_traceback"] = failure.get("stderr_tail", "")
    metrics = collect_run_metrics(run_dir, experiment, config, base_config_path, wall_time, success, failure)
    print(f"[done] {run_id} success={success} wall={wall_time:.2f}s")
    return metrics


def tail_text(path: Path, max_chars: int = 8000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


def collect_metrics(output_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(output_root.glob("*/metrics.json")):
        data = read_json(path, {})
        if isinstance(data, dict):
            rows.append(data)
    return rows


def flatten_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def write_summary(output_root: Path, rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    all_keys = list(SUMMARY_COLUMNS)
    for row in rows:
        for key in row:
            if key not in all_keys:
                all_keys.append(key)
    csv_path = output_root / "ablation_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: flatten_value(row.get(key)) for key in all_keys})

    md_path = output_root / "ablation_summary.md"
    total = len(rows)
    succeeded = sum(1 for row in rows if row.get("success"))
    by_method = Counter(str(row.get("method", "unknown")) for row in rows)
    failures = [row for row in rows if not row.get("success")]
    lines = [
        "# Ablation Summary",
        "",
        f"- Total runs: {total}",
        f"- Succeeded: {succeeded}",
        f"- Failed: {total - succeeded}",
        f"- By method: {dict(by_method)}",
        "",
        "## Failures",
        "",
    ]
    if failures:
        lines.append("| run_id | method | reason | stage |")
        lines.append("|---|---|---|---|")
        for row in failures:
            lines.append(
                f"| {row.get('run_id')} | {row.get('method')} | "
                f"{row.get('failure_reason', '')} | {row.get('failure_stage', '')} |"
            )
    else:
        lines.append("No failed runs.")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def parse_args() -> argparse.Namespace:
    phys_root = default_phys_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physgaussian-root", type=Path, default=phys_root)
    parser.add_argument("--base-config", type=Path, default=default_base_config(phys_root))
    parser.add_argument("--model-path", type=Path, default=default_model_path(phys_root))
    parser.add_argument("--iteration", type=int, default=7000)
    parser.add_argument("--python-bin", default=default_python_bin())
    parser.add_argument("--output-root", type=Path, default=Path("outputs") / "ablation")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--sanity-suite", action="store_true")
    parser.add_argument("--method", action="append", choices=METHODS)
    parser.add_argument("--grid", action="append", type=int, choices=GRIDS)
    parser.add_argument("--frame-dt", action="append", dest="frame_dt")
    parser.add_argument("--substep-dt", action="append", dest="substep_dt")
    parser.add_argument("--tolerance-profile", action="append", choices=tuple(TOLERANCE_PROFILES))
    parser.add_argument("--pbmpm-strength-scale", action="append", dest="pbmpm_strength_scale")
    parser.add_argument("--frame-num", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--timeout-seconds", type=float)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    args.physgaussian_root = args.physgaussian_root.resolve()
    args.base_config = args.base_config.resolve()
    args.model_path = args.model_path.resolve()

    experiments = build_experiments(args)
    rows = make_plan_rows(experiments)
    write_plan(args.output_root, rows)
    print(f"[plan] {len(rows)} runs -> {args.output_root / 'ablation_plan.csv'}")
    if args.dry_run or not args.run_all:
        if not args.dry_run:
            print("[plan] pass --run-all to execute")
        return 0

    base_config = read_json(args.base_config)
    if not isinstance(base_config, dict):
        raise SystemExit(f"base config is not a JSON object: {args.base_config}")

    metrics_rows = []
    for experiment in experiments:
        metrics_rows.append(run_experiment(experiment, args, base_config, args.base_config))
        write_summary(args.output_root, collect_metrics(args.output_root))
    csv_path, md_path = write_summary(args.output_root, collect_metrics(args.output_root))
    print(f"[summary] {csv_path}")
    print(f"[summary] {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
