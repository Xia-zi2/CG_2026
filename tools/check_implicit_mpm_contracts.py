from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


PHYSGAUSSIAN_ROOT = first_existing(
    ROOT / "src-overrides" / "physgaussian-src",
    ROOT / "src" / "physgaussian-src",
)
SUPERSPLAT_ROOT = first_existing(
    ROOT / "src-overrides" / "supersplat-src",
    ROOT / "src" / "supersplat-src",
    ROOT / "supersplat-src",
)
MPM_SOLVER = PHYSGAUSSIAN_ROOT / "mpm_solver_warp" / "mpm_solver_warp.py"
MPM_UTILS = PHYSGAUSSIAN_ROOT / "mpm_solver_warp" / "mpm_utils.py"
DECODE_PARAM = PHYSGAUSSIAN_ROOT / "utils" / "decode_param.py"
GS_SIMULATION = PHYSGAUSSIAN_ROOT / "gs_simulation.py"
PHYS_BACKEND = ROOT / "server" / "phys_backend.py"
PHYSICS_SESSION = SUPERSPLAT_ROOT / "src" / "physics-session.ts"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def function_source(source: str, name: str) -> str:
    tree = ast.parse(source)
    lines = source.splitlines()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return "\n".join(lines[node.lineno - 1 : node.end_lineno])
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name:
                    return "\n".join(lines[item.lineno - 1 : item.end_lineno])
    raise AssertionError(f"missing function {name}")


def assert_contains(text: str, needle: str, label: str) -> None:
    if needle not in text:
        raise AssertionError(f"{label}: expected to find {needle!r}")


def assert_not_contains(text: str, needle: str, label: str) -> None:
    if needle in text:
        raise AssertionError(f"{label}: unexpected {needle!r}")


def main() -> None:
    solver = read(MPM_SOLVER)
    utils = read(MPM_UTILS)
    decode = read(DECODE_PARAM)
    gs = read(GS_SIMULATION)
    backend = read(PHYS_BACKEND)
    physics_session = read(PHYSICS_SESSION) if PHYSICS_SESSION.exists() else ""

    implicit_residual = function_source(solver, "_implicit_residual")
    assert_contains(implicit_residual, "self._implicit_grid_v_trial", "residual scratch velocity")
    assert_not_contains(implicit_residual, "self.mpm_state.grid_v_out", "residual must not write formal grid_v_out")
    assert_not_contains(implicit_residual, "self.grid_postprocess", "residual must not run ordinary postprocess")
    assert_contains(implicit_residual, "implicit_compute_trial_stress", "residual computes trial stress")

    trial_stress = function_source(utils, "implicit_compute_trial_stress")
    stiffness = function_source(utils, "implicit_accumulate_stiffness_diag")
    internal_force = function_source(utils, "implicit_accumulate_internal_force")
    p2g_no_stress = function_source(utils, "p2g_apic_no_stress")
    assert_contains(p2g_no_stress, "mpm_clamp_to_grid_domain", "no-stress P2G clamps OOB particles")
    assert_contains(p2g_no_stress, "state.particle_x[p] =", "no-stress P2G writes clamp position")
    assert_contains(p2g_no_stress, "state.particle_v[p] =", "no-stress P2G zeroes clamp velocity")
    assert_contains(p2g_no_stress, "state.particle_C[p] =", "no-stress P2G zeroes clamp affine C")
    for needle in (
        "state.particle_F[p] =",
        "state.particle_F_trial[p] =",
        "state.particle_stress[p] =",
        "state.particle_cov",
        "state.particle_R[p] =",
        "model.yield_stress[p] =",
        "model.mu[p] =",
        "model.lam[p] =",
    ):
        assert_not_contains(p2g_no_stress, needle, "no-stress P2G")
    for label, text in (
        ("trial stress", trial_stress),
        ("stiffness diag", stiffness),
        ("internal force", internal_force),
    ):
        assert_not_contains(text, "state.particle_x[p] =", label)
        assert_not_contains(text, "state.particle_v[p] =", label)
        assert_not_contains(text, "state.particle_C[p] =", label)
        assert_not_contains(text, "state.particle_F[p] =", label)
        assert_not_contains(text, "state.particle_F_trial[p] =", label)
        assert_not_contains(text, "state.particle_stress[p] =", label)
        assert_not_contains(text, "state.particle_cov", label)
        assert_not_contains(text, "state.particle_R[p] =", label)
        assert_not_contains(text, "model.yield_stress[p] =", label)
        assert_not_contains(text, "model.mu[p] =", label)
        assert_not_contains(text, "model.lam[p] =", label)
        assert_contains(text, "mpm_stencil_in_grid", label)
        assert_contains(text, "clamped_x = mpm_clamp_to_grid_domain", label)

    evaluate_mapping = function_source(utils, "evaluate_material_return_mapping")
    commit_mapping = function_source(utils, "commit_material_return_mapping")
    assert_contains(evaluate_mapping, "evaluate_von_mises_return_mapping", "readonly plastic mapping")
    assert_contains(evaluate_mapping, "evaluate_von_mises_return_mapping_with_damage", "readonly damage mapping")
    assert_contains(commit_mapping, "material_return_mapping", "commit mapping")
    assert_not_contains(evaluate_mapping, "model.yield_stress[p] =", "readonly mapping history")
    assert_not_contains(evaluate_mapping, "model.mu[p] =", "readonly mapping mu")
    assert_not_contains(evaluate_mapping, "model.lam[p] =", "readonly mapping lam")

    stress_commit = function_source(utils, "compute_stress_from_F_trial")
    assert_contains(stress_commit, "commit_material_return_mapping", "commit stress mapping")
    assert_contains(stress_commit, "state.particle_F_trial[p] = state.particle_F[p]", "F/F_trial sync")
    assert_contains(function_source(utils, "compute_cov_from_F"), "F = state.particle_F[p]", "cov uses committed F")
    assert_contains(function_source(utils, "compute_R_from_F"), "F = state.particle_F[p]", "R uses committed F")

    implicit_step = function_source(solver, "p2g2p_implicit")
    gmres = function_source(solver, "_gmres_matrix_free")
    snapshot_state = function_source(solver, "_snapshot_implicit_particle_state")
    restore_state = function_source(solver, "_restore_implicit_particle_state")
    assert_contains(gmres, "arnoldi_norm = h[j + 1, j].clone()", "GMRES snapshots Arnoldi norm before Givens zero")
    assert_contains(gmres, '"break_reason": "zero_rhs"', "GMRES trace zero RHS break reason")
    assert_contains(gmres, '"break_reason": break_reason', "GMRES trace break reason")
    assert_contains(gmres, '"arnoldi_norm": arnoldi_norm_float', "GMRES trace Arnoldi norm")
    assert_contains(gmres, '"basis_size": len(q_vectors)', "GMRES trace basis size")
    assert_contains(gmres, '"rhs_norm"', "GMRES trace RHS norm")
    assert_contains(gmres, '"used_iter"', "GMRES trace used iter")
    assert_contains(gmres, '"final_relative_residual"', "GMRES final relative residual")
    assert_contains(implicit_step, "allow_best_effort_commit=False", "strict commit default")
    assert_contains(implicit_step, "should_fail_without_commit", "strict nonconvergence branch")
    assert_contains(implicit_step, '"committed": False', "nonconverged not committed")
    assert_contains(implicit_step, "particle_state_snapshot = self._snapshot_implicit_particle_state()", "snapshot before implicit solve")
    assert_contains(implicit_step, "self._restore_implicit_particle_state(particle_state_snapshot)", "restore on failed implicit solve")
    assert_contains(implicit_step, '"state_restored_on_failure": True', "failure trace records restore")
    assert_contains(implicit_step, '"pre_solve_clamp_count"', "pre-solve clamp trace")
    assert_contains(implicit_step, '"out_of_bounds_count"', "out-of-bounds trace")
    assert_contains(implicit_step, '"implicit_contact_residual": False', "contact residual trace")
    assert_contains(implicit_step, '"post_commit_projection_only": True', "projection trace")
    assert_contains(implicit_step, 'failure_reason = "newton_not_converged"', "strict failure reason")
    assert_contains(implicit_step, '"failure_reason": failure_reason', "failure reason trace")
    assert_contains(implicit_step, "fallback_minus_active = -residual / precond_diag", "preconditioned fallback minus")
    assert_contains(implicit_step, "fallback_plus_active = residual / precond_diag", "preconditioned fallback plus")
    assert_contains(implicit_step, '"fallback_preconditioned"] = True', "fallback trace flag")
    assert_contains(implicit_step, "fallback_descent_tol=1e-8", "fallback descent default")
    assert_contains(implicit_step, "fallback_step_min_rel=1e-8", "fallback step default")
    assert_contains(implicit_step, "fallback_decrease_tol=1e-6", "fallback decrease default")
    assert_contains(implicit_step, "current_residual_is_near()", "near residual checked before fallback")
    assert_contains(implicit_step, "fallback_derivative_float >= descent_threshold", "fallback requires descent")
    assert_contains(implicit_step, "fallback_step_relative_norm < fallback_step_min_rel", "fallback tiny step skipped")
    assert_contains(implicit_step, "trial_phi_float < current_phi_float *", "fallback requires objective decrease")
    assert_contains(implicit_step, '"fallback_skipped_reason"', "fallback skip reason trace")
    assert_contains(implicit_step, '"fallback_directional_derivative"', "fallback derivative trace")
    assert_contains(implicit_step, '"fallback_step_relative_norm"', "fallback step trace")
    assert_contains(implicit_step, '"accepted_due_to_near_residual"', "near residual acceptance trace")
    assert_not_contains(implicit_step, "accepted_due_to_small_residual", "avoid duplicate near acceptance field")
    assert_contains(implicit_step, "active_mask = grid_mass_mask & (~dirichlet_mask)", "Dirichlet excluded")
    assert_contains(implicit_step, "active_free_node_count == 0", "all-Dirichlet branch")
    assert_contains(implicit_step, '"active_free_nodes"', "active free trace")
    assert_contains(implicit_step, '"dirichlet_nodes"', "Dirichlet trace")
    assert_contains(implicit_step, '"gmres_forcing": "eisenstat_walker"', "EW GMRES forcing trace")
    assert_contains(implicit_step, "gmres_info = self._gmres_matrix_free", "GMRES info returned")
    assert_contains(implicit_step, '"gmres_break_reason"', "GMRES break reason trace")
    assert_contains(implicit_step, '"gmres_arnoldi_norm"', "GMRES Arnoldi norm trace")
    assert_contains(implicit_step, '"gmres_basis_size"', "GMRES basis size trace")
    assert_contains(implicit_step, '"trace_version": 2', "implicit trace v2")
    assert_contains(implicit_step, '"solver_settings"', "implicit trace solver settings")
    assert_contains(implicit_step, '"grid_summary"', "implicit trace grid summary")
    assert_contains(implicit_step, '"material_summary"', "implicit trace material summary")
    assert_contains(implicit_step, '"preconditioner_summary"', "implicit trace preconditioner summary")
    assert_contains(implicit_step, '"line_search_trials"', "implicit trace line-search trials")
    assert_contains(implicit_step, '"accepted_step_relative_norm"', "implicit trace accepted step norm")
    assert_not_contains(implicit_step, "gmres_tolerance_mode", "fixed GMRES mode removed")
    assert_contains(implicit_step, "gmres_tol_floor=1e-3", "implicit GMRES floor default")
    assert_contains(implicit_step, "ew_tol = min(ew_eta_max, max(gmres_tol_floor, ew_tol))", "EW tolerance floor uses gmres_tol_floor")
    assert_contains(implicit_step, "linear_tol = (", "EW tolerance can be tail-capped")
    assert_not_contains(implicit_step, "linear_tol = min(float(gmres_tol), ew_tol)", "legacy gmres_tol must not cap EW")
    assert_contains(implicit_step, "near_converged_factor=2.0", "near-converged default")
    assert_contains(implicit_step, 'convergence_type = "near_converged"', "near-converged trace type")
    assert_contains(implicit_step, "newton_exhausted", "near-converged can trigger after Newton exhaustion")
    assert_contains(implicit_step, "line_search_saturated", "near-converged can accept line-search saturation")
    for field in (
        "particle_x",
        "particle_v",
        "particle_C",
        "particle_F",
        "particle_F_trial",
        "particle_stress",
        "particle_cov",
        "particle_R",
        "yield_stress",
        "mu",
        "lam",
    ):
        assert_contains(snapshot_state, field, f"snapshot covers {field}")
        assert_contains(restore_state, field, f"restore covers {field}")

    surface_collider = function_source(solver, "add_surface_collider")
    cuboid_collider = function_source(solver, "set_velocity_on_cuboid")
    assert_not_contains(surface_collider, "implicit_grid_constraint_sources.append", "surface stays out of Newton")
    assert_contains(surface_collider, 'grid_postprocess_sources.append("surface")', "surface remains postprocess projection")
    assert_contains(cuboid_collider, 'implicit_grid_constraint_sources.append("cuboid")', "cuboid remains Dirichlet")

    assert_contains(decode, "normalize_integrator_name", "decode integrator normalization")
    assert_contains(decode, '"gmres_tol_floor"', "decode GMRES floor")
    assert_contains(decode, "raise ValueError", "decode unknown integrator")
    assert_contains(gs, "normalize_integrator_name", "runtime integrator normalization")
    assert_contains(gs, "raise ValueError", "runtime unknown integrator")
    assert_contains(backend, "_solver_to_integrator", "backend integrator normalization")
    assert_contains(backend, "dirichletVelocity", "backend Dirichlet velocity field")
    assert_contains(backend, "targetVelocity", "backend target velocity field")
    assert_contains(backend, "prescribedVelocity", "backend prescribed velocity split")
    assert_contains(backend, '"force": drive["linearForce"]', "movable body keeps force semantics")
    assert_contains(backend, 'config["n_min"] = n_min', "backend emits PBMPM N min")
    assert_contains(backend, 'config["n_max"] = n_max', "backend emits PBMPM N max")
    assert_contains(decode, '"n_min"', "decode PBMPM N min")
    assert_contains(decode, '"n_max"', "decode PBMPM N max")
    assert_contains(gs, "n_min=pbmpm_params.get", "runtime passes PBMPM N min")
    assert_contains(gs, "n_max=pbmpm_params.get", "runtime passes PBMPM N max")
    assert_contains(solver, "def _auto_pbmpm_parameters(self, dt, strength_scale=1.0, n_min=3, n_max=25)", "solver PBMPM N bounds")
    pbmpm_step = function_source(solver, "p2g2p_pbmpm")
    assert_contains(pbmpm_step, "kernel=pbmpm_clear_D", "PBMPM clears per-step D state")
    assert_contains(pbmpm_step, '"gravity_post_integrate_kick": True', "PBMPM gravity remains post-integrate")
    assert_contains(utils, "def pbmpm_solve_D", "PBMPM solve_D exists")
    assert_contains(utils, "target = R * alpha + Q * (1.0 - alpha)", "PBMPM R/Q projection")
    assert_not_contains(function_source(utils, "pbmpm_solve_D"), "commit_material_return_mapping", "PBMPM solve_D must not commit material mapping")
    assert_contains(function_source(utils, "pbmpm_integrate_particles"), "commit_material_return_mapping", "PBMPM final integrate reuses material mapping")
    if physics_session:
        assert_contains(physics_session, "n_min: 3", "frontend PBMPM N min default")
        assert_contains(physics_session, "n_max: 25", "frontend PBMPM N max default")

    assert_contains(solver, "self.mpm_model.grid_v_damping_scale = 1.0", "solver default no grid damping")
    assert_contains(solver, "damping_scale < 1.0", "solver damping call path")
    assert_contains(solver, "add_damping_via_grid", "solver damping kernel call")
    assert_contains(utils, "def add_damping_via_grid", "utils damping kernel")
    assert_contains(decode, 'material_params["grid_v_damping_scale"] = max(', "decode allows grid damping")
    assert_contains(backend, '"grid_v_damping_scale": _simulation_damping(simulation)', "backend maps UI damping")
    assert_contains(backend, "def _simulation_damping", "backend damping normalization")
    if physics_session:
        assert_contains(physics_session, "damping: 0.9999", "frontend damping default")
    assert_contains(backend, "damping: float = 1.0", "proxy step no damping default")
    assert_contains(backend, 'preview.get("gridDamping"), 1.0', "proxy preview damping default")
    if physics_session:
        assert_contains(physics_session, "momentum[0] / mass + gravity[0] * dt", "frontend proxy no x damping")
        assert_contains(physics_session, "momentum[1] / mass + gravity[1] * dt", "frontend proxy no y damping")
        assert_contains(physics_session, "momentum[2] / mass + gravity[2] * dt", "frontend proxy no z damping")
        assert_not_contains(physics_session, "* 0.995", "frontend proxy damping removed")

    print("implicit MPM contract checks passed")


if __name__ == "__main__":
    main()
