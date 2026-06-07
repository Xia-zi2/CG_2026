import json
import warp as wp
from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP
from mpm_solver_warp.engine_utils import *


def normalize_integrator_name(integrator):
    name = str(integrator).strip().lower().replace("-", "_")
    if name in ("implicit", "implicit_mpm"):
        return "implicit"
    if name in ("explicit", "explicit_mpm", "mpm"):
        return "explicit"
    if name in ("pbmpm", "pb_mpm"):
        return "pbmpm"
    raise ValueError(f"Unknown integrator '{integrator}'")


def decode_param_json(json_file):
    f = open(json_file)
    sim_params = json.load(f)
    material_params = {}
    # material parameters
    if "material" in sim_params.keys():
        material_params["material"] = sim_params["material"]
    else:
        material_params["material"] = "jelly"

    if "grid_lim" in sim_params.keys():
        material_params["grid_lim"] = sim_params["grid_lim"]
    else:
        material_params["grid_lim"] = 2.0

    if "n_grid" in sim_params.keys():
        material_params["n_grid"] = sim_params["n_grid"]
    else:
        material_params["n_grid"] = 50

    if "nu" in sim_params.keys():
        material_params["nu"] = sim_params["nu"]
        if material_params["nu"] > 0.5 or material_params["nu"] < 0.0:
            raise ValueError("Poisson's ratio should be less than 0.5")
    else:
        material_params["nu"] = 0.4

    if "E" in sim_params.keys():
        material_params["E"] = sim_params["E"]
    else:
        material_params["E"] = 1e5

    if "yield_stress" in sim_params.keys():
        material_params["yield_stress"] = sim_params["yield_stress"]

    if "hardening" in sim_params.keys():
        material_params["hardening"] = sim_params["hardening"]

    if "xi" in sim_params.keys():
        material_params["xi"] = sim_params["xi"]

    if "friction_angle" in sim_params.keys():
        material_params["friction_angle"] = sim_params["friction_angle"]

    if "plastic_viscosity" in sim_params.keys():
        material_params["plastic_viscosity"] = sim_params["plastic_viscosity"]

    if "g" in sim_params.keys():
        material_params["g"] = sim_params["g"]
    else:
        material_params["g"] = 9.8

    if "density" in sim_params.keys():
        material_params["density"] = sim_params["density"]
    else:
        material_params["density"] = 200.0

    if "rpic_damping" in sim_params.keys():
        material_params["rpic_damping"] = sim_params["rpic_damping"]

    if "pic_damping" in sim_params.keys():
        material_params["pic_damping"] = sim_params["pic_damping"]

    if "softening" in sim_params.keys():
        material_params["softening"] = sim_params["softening"]

    if "opacity_threshold" in sim_params.keys():
        material_params["opacity_threshold"] = sim_params["opacity_threshold"]

    if "grid_v_damping_scale" in sim_params.keys():
        material_params["grid_v_damping_scale"] = max(
            0.0, min(float(sim_params["grid_v_damping_scale"]), 1.0)
        )

    if "additional_material_params" in sim_params.keys():
        additional_params = sim_params["additional_material_params"]
        for i in range(len(additional_params)):
            if not "point" in additional_params[i].keys():
                raise TypeError("point is not defined")

            if not "size" in additional_params[i].keys():
                raise TypeError("size is not defined")

            if not "E" in additional_params[i].keys():
                raise TypeError("E is not defined")

            if not "nu" in additional_params[i].keys():
                raise TypeError("nu is not defined")

            if not "density" in additional_params[i].keys():
                additional_params[i]["density"] = material_params["density"]

            if (
                not "yield_stress" in additional_params[i].keys()
                and "yield_stress" in material_params
            ):
                additional_params[i]["yield_stress"] = material_params["yield_stress"]

        material_params["additional_material_params"] = additional_params

    if "particle_material_params" in sim_params.keys():
        particle_params = sim_params["particle_material_params"]
        for i in range(len(particle_params)):
            if not "indices" in particle_params[i].keys():
                raise TypeError("indices is not defined")

            if not "E" in particle_params[i].keys():
                raise TypeError("E is not defined")

            if not "nu" in particle_params[i].keys():
                raise TypeError("nu is not defined")

            if not "density" in particle_params[i].keys():
                particle_params[i]["density"] = material_params["density"]

            if (
                not "yield_stress" in particle_params[i].keys()
                and "yield_stress" in material_params
            ):
                particle_params[i]["yield_stress"] = material_params["yield_stress"]

        material_params["particle_material_params"] = particle_params

    # boundary conditions
    bc_params = {}
    if "boundary_conditions" in sim_params.keys():
        bc_params = sim_params["boundary_conditions"]

    # time step
    time_params = {}
    if "substep_dt" in sim_params.keys():
        time_params["substep_dt"] = sim_params["substep_dt"]
    else:
        time_params["substep_dt"] = 1e-4

    if "frame_dt" in sim_params.keys():
        time_params["frame_dt"] = sim_params["frame_dt"]
    else:
        time_params["frame_dt"] = 1e-2

    if "frame_num" in sim_params.keys():
        time_params["frame_num"] = sim_params["frame_num"]
    else:
        time_params["frame_num"] = 100

    if "integrator" in sim_params.keys():
        time_params["integrator"] = normalize_integrator_name(sim_params["integrator"])
    else:
        time_params["integrator"] = "explicit"

    if "implicit_mpm" in sim_params.keys():
        implicit_params = sim_params["implicit_mpm"]
    else:
        implicit_params = {}
    implicit_adaptive_max_split = max(
        0, int(implicit_params.get("adaptive_max_split", 3))
    )
    implicit_adaptive_min_dt_default = max(
        float(time_params["substep_dt"]) / (2.0 ** max(implicit_adaptive_max_split, 1)),
        1e-8,
    )
    time_params["implicit_mpm"] = {
        "beta": implicit_params.get("beta", 0.25),
        "gamma": implicit_params.get("gamma", 0.5),
        "newton_tol": implicit_params.get("newton_tol", 5e-4),
        "newton_abs_tol": implicit_params.get("newton_abs_tol", 1e-6),
        "newton_rms_tol": implicit_params.get("newton_rms_tol", 1e-4),
        "newton_max_iter": implicit_params.get("newton_max_iter", 16),
        "gmres_tol": implicit_params.get("gmres_tol", 1e-3),
        "gmres_tol_floor": implicit_params.get("gmres_tol_floor", 1e-3),
        "gmres_max_iter": implicit_params.get("gmres_max_iter", 24),
        "jvp_eps": implicit_params.get("jvp_eps", 1e-4),
        "line_search_max_iter": implicit_params.get("line_search_max_iter", 8),
        "armijo_c1": implicit_params.get("armijo_c1", 1e-4),
        "ew_eta_min": implicit_params.get("ew_eta_min", 1e-3),
        "ew_eta_max": implicit_params.get("ew_eta_max", 0.1),
        "ew_gamma": implicit_params.get("ew_gamma", 0.9),
        "ew_alpha": implicit_params.get("ew_alpha", 1.5),
        "stiffness_preconditioner_scale": implicit_params.get(
            "stiffness_preconditioner_scale", 1.0
        ),
        "stagnation_tol": implicit_params.get("stagnation_tol", 1e-8),
        "allow_best_effort_commit": implicit_params.get(
            "allow_best_effort_commit", False
        ),
        "near_converged_factor": implicit_params.get("near_converged_factor", 2.0),
        "near_newton_rms_tol": implicit_params.get("near_newton_rms_tol", 1e-4),
        "fallback_descent_tol": implicit_params.get("fallback_descent_tol", 1e-8),
        "fallback_step_min_rel": implicit_params.get("fallback_step_min_rel", 1e-8),
        "fallback_decrease_tol": implicit_params.get("fallback_decrease_tol", 1e-6),
        "adaptive_max_split": implicit_adaptive_max_split,
        "adaptive_min_dt": implicit_params.get(
            "adaptive_min_dt", implicit_adaptive_min_dt_default
        ),
    }

    if "pbmpm" in sim_params.keys():
        pbmpm_params = sim_params["pbmpm"]
    else:
        pbmpm_params = {}
    time_params["pbmpm"] = {
        "strength_scale": pbmpm_params.get(
            "strength_scale", pbmpm_params.get("stiffness_scale", 1.0)
        ),
        "n_min": pbmpm_params.get(
            "n_min", pbmpm_params.get("N_min", pbmpm_params.get("nMin", 3))
        ),
        "n_max": pbmpm_params.get(
            "n_max", pbmpm_params.get("N_max", pbmpm_params.get("nMax", 25))
        ),
        "plastic_mode": pbmpm_params.get("plastic_mode", 0),
        # PBMPM stretch clamp bounds, not material yield_stress.
        "yield_min": pbmpm_params.get("yield_min", 0.55),
        "yield_max": pbmpm_params.get("yield_max", 1.85),
    }

    if "explicit_mpm" in sim_params.keys():
        time_params["explicit_mpm"] = sim_params["explicit_mpm"]
    else:
        time_params["explicit_mpm"] = {}

    # preprocessing_params
    preprocessing_params = {}
    if "opacity_threshold" in sim_params.keys():
        preprocessing_params["opacity_threshold"] = sim_params["opacity_threshold"]
    else:
        preprocessing_params["opacity_threshold"] = 0.02

    if "rotation_degree" in sim_params.keys():
        preprocessing_params["rotation_degree"] = sim_params["rotation_degree"]
    else:
        preprocessing_params["rotation_degree"] = []

    if "rotation_axis" in sim_params.keys():
        preprocessing_params["rotation_axis"] = sim_params["rotation_axis"]
    else:
        preprocessing_params["rotation_axis"] = []

    if "sim_area" in sim_params.keys():
        preprocessing_params["sim_area"] = sim_params["sim_area"]
    else:
        preprocessing_params["sim_area"] = None

    if "active_gs_indices" in sim_params.keys():
        preprocessing_params["active_gs_indices"] = sim_params["active_gs_indices"]
    else:
        preprocessing_params["active_gs_indices"] = None

    if "scale" in sim_params.keys():
        preprocessing_params["scale"] = sim_params["scale"]
    else:
        preprocessing_params["scale"] = 1.0

    if "particle_filling" in sim_params.keys():
        preprocessing_params["particle_filling"] = sim_params["particle_filling"]
        filling_params = preprocessing_params["particle_filling"]
        if not "n_grid" in filling_params.keys():
            filling_params["n_grid"] = material_params["n_grid"] * 4

        if not "density_threshold" in filling_params.keys():
            filling_params["density_threshold"] = 5.0

        if not "search_threshold" in filling_params.keys():
            filling_params["search_threshold"] = 3.0

        if not "max_particles_num" in filling_params.keys():
            filling_params["max_particles_num"] = 2000000

        if not "max_partciels_per_cell" in filling_params.keys():
            filling_params["max_partciels_per_cell"] = 1

        if not "search_exclude_direction" in filling_params.keys():
            filling_params["search_exclude_direction"] = 5

        if not "ray_cast_direction" in filling_params.keys():
            filling_params["ray_cast_direction"] = 4

        if not "boundary" in filling_params.keys():
            filling_params["boundary"] = None

        if not "smooth" in filling_params.keys():
            filling_params["smooth"] = False

        if not "visualize" in filling_params.keys():
            filling_params["visualize"] = False
    else:
        preprocessing_params["particle_filling"] = None

    # camera params
    camera_params = {}
    if "mpm_space_viewpoint_center" in sim_params.keys():
        camera_params["mpm_space_viewpoint_center"] = sim_params[
            "mpm_space_viewpoint_center"
        ]
    else:
        camera_params["mpm_space_viewpoint_center"] = [1.0, 1.0, 1.0]
    if "mpm_space_vertical_upward_axis" in sim_params.keys():
        camera_params["mpm_space_vertical_upward_axis"] = sim_params[
            "mpm_space_vertical_upward_axis"
        ]
    else:
        camera_params["mpm_space_vertical_upward_axis"] = [0, 0, 1]
    if "default_camera_index" in sim_params.keys():
        camera_params["default_camera_index"] = sim_params["default_camera_index"]
    else:
        camera_params["default_camera_index"] = 0
    if "show_hint" in sim_params.keys():
        camera_params["show_hint"] = sim_params["show_hint"]
    else:
        camera_params["show_hint"] = False
    if "init_azimuthm" in sim_params.keys():
        camera_params["init_azimuthm"] = sim_params["init_azimuthm"]
    else:
        camera_params["init_azimuthm"] = None
    if "init_elevation" in sim_params.keys():
        camera_params["init_elevation"] = sim_params["init_elevation"]
    else:
        camera_params["init_elevation"] = None
    if "init_radius" in sim_params.keys():
        camera_params["init_radius"] = sim_params["init_radius"]
    else:
        camera_params["init_radius"] = None
    if "delta_a" in sim_params.keys():
        camera_params["delta_a"] = sim_params["delta_a"]
    else:
        camera_params["delta_a"] = None
    if "delta_e" in sim_params.keys():
        camera_params["delta_e"] = sim_params["delta_e"]
    else:
        camera_params["delta_e"] = None
    if "delta_r" in sim_params.keys():
        camera_params["delta_r"] = sim_params["delta_r"]
    else:
        camera_params["delta_r"] = None
    if "move_camera" in sim_params.keys():
        camera_params["move_camera"] = sim_params["move_camera"]
    else:
        camera_params["move_camera"] = False

    return material_params, bc_params, time_params, preprocessing_params, camera_params


def set_boundary_conditions(
    mpm_solver: MPM_Simulator_WARP, bc_params: dict, time_params: dict
):
    for bc in bc_params:
        bc_type = bc.get("type")
        constraint_type = bc.get("constraint_type")
        is_velocity_cuboid = bc_type in {
            "cuboid",
            "set_velocity_on_cuboid",
            "velocity_cuboid",
            "fixed_cuboid",
            "dirichlet_cuboid",
        } or constraint_type in {
            "set_velocity_on_cuboid",
            "velocity_cuboid",
            "fixed_cuboid",
            "dirichlet_cuboid",
        }
        if is_velocity_cuboid:
            if "point" not in bc and "center" in bc:
                bc["point"] = bc["center"]
            if "size" not in bc and "scale" in bc:
                bc["size"] = [0.5 * abs(float(value)) for value in bc["scale"]]
            if "point" not in bc and "min" in bc and "max" in bc:
                bc["point"] = [
                    0.5 * (float(lo) + float(hi))
                    for lo, hi in zip(bc["min"], bc["max"])
                ]
                bc["size"] = [
                    0.5 * abs(float(hi) - float(lo))
                    for lo, hi in zip(bc["min"], bc["max"])
                ]
            if "velocity" not in bc and "target_velocity" in bc:
                bc["velocity"] = bc["target_velocity"]
            assert "point" in bc and "size" in bc and "velocity" in bc
            start_time = 0.0
            end_time = 1e3
            reset = 0
            if "start_time" in bc.keys():
                start_time = bc["start_time"]
            if "end_time" in bc.keys():
                end_time = bc["end_time"]
            if "reset" in bc.keys():
                reset = bc["reset"]
            mpm_solver.set_velocity_on_cuboid(
                point=bc["point"],
                size=bc["size"],
                velocity=bc["velocity"],
                start_time=start_time,
                end_time=end_time,
                reset=reset,
            )

        elif bc["type"] == "particle_impulse":
            assert "force" in bc.keys()

            start_time = 0.0
            if "start_time" in bc.keys():
                start_time = bc["start_time"]
            num_dt = 1
            if "num_dt" in bc.keys():
                num_dt = bc["num_dt"]
            point = [1, 1, 1]
            if "point" in bc.keys():
                point = bc["point"]
            size = [1, 1, 1]
            if "size" in bc.keys():
                size = bc["size"]

            mpm_solver.add_impulse_on_particles(
                force=bc["force"],
                dt=time_params["substep_dt"],
                point=point,
                size=size,
                num_dt=num_dt,
                start_time=start_time,
            )
        elif bc["type"] == "fixed_particle_indices":
            assert "indices" in bc.keys()
            velocity = bc.get("velocity", [0.0, 0.0, 0.0])
            start_time = bc.get("start_time", 0.0)
            end_time = bc.get("end_time", 1e3)
            reset_deformation = bc.get("reset_deformation", bc.get("reset", 1))
            mpm_solver.fix_particles_by_indices(
                indices=bc["indices"],
                velocity=velocity,
                start_time=start_time,
                end_time=end_time,
                reset_deformation=reset_deformation,
            )
        elif bc["type"] == "bounding_box":
            mpm_solver.add_bounding_box()

        elif bc["type"] == "enforce_particle_translation":
            assert "point" in bc.keys()
            assert "size" in bc.keys()
            assert "velocity" in bc.keys()
            assert "start_time" in bc.keys()
            assert "end_time" in bc.keys()

            mpm_solver.enforce_particle_velocity_translation(
                point=bc["point"],
                size=bc["size"],
                velocity=bc["velocity"],
                start_time=bc["start_time"],
                end_time=bc["end_time"],
            )
        elif bc["type"] == "surface_collider":
            assert "point" in bc.keys()
            assert "normal" in bc.keys()
            assert "surface" in bc.keys()
            assert "friction" in bc.keys()
            assert "start_time" in bc.keys()
            assert "end_time" in bc.keys()

            mpm_solver.add_surface_collider(
                point=bc["point"],
                normal=bc["normal"],
                surface=bc["surface"],
                friction=bc["friction"],
                start_time=bc["start_time"],
                end_time=bc["end_time"],
            )
        elif bc["type"] == "release_particles_sequentially":
            assert "normal" in bc.keys()
            assert "start_position" in bc.keys()
            assert "end_position" in bc.keys()
            assert "num_layers" in bc.keys()
            assert "start_time" in bc.keys()
            assert "end_time" in bc.keys()

            mpm_solver.release_particles_sequentially(
                normal=bc["normal"],
                start_position=bc["start_position"],
                end_position=bc["end_position"],
                num_layers=bc["num_layers"],
                start_time=bc["start_time"],
                end_time=bc["end_time"],
            )
        elif bc["type"] == "enforce_particle_velocity_rotation":
            assert "normal" in bc.keys()
            assert "point" in bc.keys()
            assert "start_time" in bc.keys()
            assert "end_time" in bc.keys()
            assert "half_height_and_radius" in bc.keys()
            assert "rotation_scale" in bc.keys()
            assert "translation_scale" in bc.keys()

            mpm_solver.enforce_particle_velocity_rotation(
                point=bc["point"],
                normal=bc["normal"],
                half_height_and_radius=bc["half_height_and_radius"],
                rotation_scale=bc["rotation_scale"],
                translation_scale=bc["translation_scale"],
                start_time=bc["start_time"],
                end_time=bc["end_time"],
            )

        else:
            raise TypeError("Undefined BC type")
