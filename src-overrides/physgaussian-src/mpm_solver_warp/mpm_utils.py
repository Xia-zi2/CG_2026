import warp as wp
from warp_utils import *
import numpy as np
import math


# compute stress from F
@wp.func
def kirchoff_stress_FCR(
    F: wp.mat33, U: wp.mat33, V: wp.mat33, J: float, mu: float, lam: float
):
    # compute kirchoff stress for FCR model (remember tau = P F^T)
    R = U * wp.transpose(V)
    id = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    return 2.0 * mu * (F - R) * wp.transpose(F) + id * lam * J * (J - 1.0)


@wp.func
def kirchoff_stress_neoHookean(
    F: wp.mat33, U: wp.mat33, V: wp.mat33, J: float, sig: wp.vec3, mu: float, lam: float
):
    # compute kirchoff stress for FCR model (remember tau = P F^T)
    b = wp.vec3(sig[0] * sig[0], sig[1] * sig[1], sig[2] * sig[2])
    b_hat = b - wp.vec3(
        (b[0] + b[1] + b[2]) / 3.0,
        (b[0] + b[1] + b[2]) / 3.0,
        (b[0] + b[1] + b[2]) / 3.0,
    )
    tau = mu * J ** (-2.0 / 3.0) * b_hat + lam / 2.0 * (J * J - 1.0) * wp.vec3(
        1.0, 1.0, 1.0
    )
    return (
        U
        * wp.mat33(tau[0], 0.0, 0.0, 0.0, tau[1], 0.0, 0.0, 0.0, tau[2])
        * wp.transpose(V)
        * wp.transpose(F)
    )


@wp.func
def kirchoff_stress_StVK(
    F: wp.mat33, U: wp.mat33, V: wp.mat33, sig: wp.vec3, mu: float, lam: float
):
    sig = wp.vec3(
        wp.max(sig[0], 0.01), wp.max(sig[1], 0.01), wp.max(sig[2], 0.01)
    )  # add this to prevent NaN in extrem cases
    epsilon = wp.vec3(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
    log_sig_sum = wp.log(sig[0]) + wp.log(sig[1]) + wp.log(sig[2])
    ONE = wp.vec3(1.0, 1.0, 1.0)
    tau = 2.0 * mu * epsilon + lam * log_sig_sum * ONE
    return (
        U
        * wp.mat33(tau[0], 0.0, 0.0, 0.0, tau[1], 0.0, 0.0, 0.0, tau[2])
        * wp.transpose(V)
        * wp.transpose(F)
    )


@wp.func
def kirchoff_stress_drucker_prager(
    F: wp.mat33, U: wp.mat33, V: wp.mat33, sig: wp.vec3, mu: float, lam: float
):
    log_sig_sum = wp.log(sig[0]) + wp.log(sig[1]) + wp.log(sig[2])
    center00 = 2.0 * mu * wp.log(sig[0]) * (1.0 / sig[0]) + lam * log_sig_sum * (
        1.0 / sig[0]
    )
    center11 = 2.0 * mu * wp.log(sig[1]) * (1.0 / sig[1]) + lam * log_sig_sum * (
        1.0 / sig[1]
    )
    center22 = 2.0 * mu * wp.log(sig[2]) * (1.0 / sig[2]) + lam * log_sig_sum * (
        1.0 / sig[2]
    )
    center = wp.mat33(center00, 0.0, 0.0, 0.0, center11, 0.0, 0.0, 0.0, center22)
    return U * center * wp.transpose(V) * wp.transpose(F)


@wp.func
def von_mises_return_mapping(F_trial: wp.mat33, model: MPMModelStruct, p: int):
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig_old = wp.vec3(0.0)
    wp.svd3(F_trial, U, sig_old, V)

    sig = wp.vec3(
        wp.max(sig_old[0], 0.01), wp.max(sig_old[1], 0.01), wp.max(sig_old[2], 0.01)
    )  # add this to prevent NaN in extrem cases
    epsilon = wp.vec3(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
    temp = (epsilon[0] + epsilon[1] + epsilon[2]) / 3.0

    tau = 2.0 * model.mu[p] * epsilon + model.lam[p] * (
        epsilon[0] + epsilon[1] + epsilon[2]
    ) * wp.vec3(1.0, 1.0, 1.0)
    sum_tau = tau[0] + tau[1] + tau[2]
    cond = wp.vec3(
        tau[0] - sum_tau / 3.0, tau[1] - sum_tau / 3.0, tau[2] - sum_tau / 3.0
    )
    if wp.length(cond) > model.yield_stress[p]:
        epsilon_hat = epsilon - wp.vec3(temp, temp, temp)
        epsilon_hat_norm = wp.length(epsilon_hat) + 1e-6
        delta_gamma = epsilon_hat_norm - model.yield_stress[p] / (2.0 * model.mu[p])
        epsilon = epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat
        sig_elastic = wp.mat33(
            wp.exp(epsilon[0]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon[1]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon[2]),
        )
        F_elastic = U * sig_elastic * wp.transpose(V)
        if model.hardening == 1:
            model.yield_stress[p] = (
                model.yield_stress[p] + 2.0 * model.mu[p] * model.xi * delta_gamma
            )
        return F_elastic
    else:
        return F_trial


@wp.func
def evaluate_von_mises_return_mapping(F_trial: wp.mat33, model: MPMModelStruct, p: int):
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig_old = wp.vec3(0.0)
    wp.svd3(F_trial, U, sig_old, V)

    sig = wp.vec3(
        wp.max(sig_old[0], 0.01), wp.max(sig_old[1], 0.01), wp.max(sig_old[2], 0.01)
    )
    epsilon = wp.vec3(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
    temp = (epsilon[0] + epsilon[1] + epsilon[2]) / 3.0

    tau = 2.0 * model.mu[p] * epsilon + model.lam[p] * (
        epsilon[0] + epsilon[1] + epsilon[2]
    ) * wp.vec3(1.0, 1.0, 1.0)
    sum_tau = tau[0] + tau[1] + tau[2]
    cond = wp.vec3(
        tau[0] - sum_tau / 3.0, tau[1] - sum_tau / 3.0, tau[2] - sum_tau / 3.0
    )
    if wp.length(cond) > model.yield_stress[p]:
        epsilon_hat = epsilon - wp.vec3(temp, temp, temp)
        epsilon_hat_norm = wp.length(epsilon_hat) + 1e-6
        delta_gamma = epsilon_hat_norm - model.yield_stress[p] / (2.0 * model.mu[p])
        epsilon = epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat
        sig_elastic = wp.mat33(
            wp.exp(epsilon[0]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon[1]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon[2]),
        )
        return U * sig_elastic * wp.transpose(V)
    else:
        return F_trial


@wp.func
def von_mises_return_mapping_with_damage(
    F_trial: wp.mat33, model: MPMModelStruct, p: int
):
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig_old = wp.vec3(0.0)
    wp.svd3(F_trial, U, sig_old, V)

    sig = wp.vec3(
        wp.max(sig_old[0], 0.01), wp.max(sig_old[1], 0.01), wp.max(sig_old[2], 0.01)
    )  # add this to prevent NaN in extrem cases
    epsilon = wp.vec3(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
    temp = (epsilon[0] + epsilon[1] + epsilon[2]) / 3.0

    tau = 2.0 * model.mu[p] * epsilon + model.lam[p] * (
        epsilon[0] + epsilon[1] + epsilon[2]
    ) * wp.vec3(1.0, 1.0, 1.0)
    sum_tau = tau[0] + tau[1] + tau[2]
    cond = wp.vec3(
        tau[0] - sum_tau / 3.0, tau[1] - sum_tau / 3.0, tau[2] - sum_tau / 3.0
    )
    if wp.length(cond) > model.yield_stress[p]:
        if model.yield_stress[p] <= 0:
            return F_trial
        epsilon_hat = epsilon - wp.vec3(temp, temp, temp)
        epsilon_hat_norm = wp.length(epsilon_hat) + 1e-6
        delta_gamma = epsilon_hat_norm - model.yield_stress[p] / (2.0 * model.mu[p])
        epsilon = epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat
        model.yield_stress[p] = model.yield_stress[p] - model.softening * wp.length(
            (delta_gamma / epsilon_hat_norm) * epsilon_hat
        )
        if model.yield_stress[p] <= 0:
            model.mu[p] = 0.0
            model.lam[p] = 0.0
        sig_elastic = wp.mat33(
            wp.exp(epsilon[0]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon[1]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon[2]),
        )
        F_elastic = U * sig_elastic * wp.transpose(V)
        if model.hardening == 1:
            model.yield_stress[p] = (
                model.yield_stress[p] + 2.0 * model.mu[p] * model.xi * delta_gamma
            )
        return F_elastic
    else:
        return F_trial


@wp.func
def evaluate_von_mises_return_mapping_with_damage(
    F_trial: wp.mat33, model: MPMModelStruct, p: int
):
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig_old = wp.vec3(0.0)
    wp.svd3(F_trial, U, sig_old, V)

    sig = wp.vec3(
        wp.max(sig_old[0], 0.01), wp.max(sig_old[1], 0.01), wp.max(sig_old[2], 0.01)
    )
    epsilon = wp.vec3(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
    temp = (epsilon[0] + epsilon[1] + epsilon[2]) / 3.0

    tau = 2.0 * model.mu[p] * epsilon + model.lam[p] * (
        epsilon[0] + epsilon[1] + epsilon[2]
    ) * wp.vec3(1.0, 1.0, 1.0)
    sum_tau = tau[0] + tau[1] + tau[2]
    cond = wp.vec3(
        tau[0] - sum_tau / 3.0, tau[1] - sum_tau / 3.0, tau[2] - sum_tau / 3.0
    )
    if wp.length(cond) > model.yield_stress[p]:
        if model.yield_stress[p] <= 0:
            return F_trial
        epsilon_hat = epsilon - wp.vec3(temp, temp, temp)
        epsilon_hat_norm = wp.length(epsilon_hat) + 1e-6
        delta_gamma = epsilon_hat_norm - model.yield_stress[p] / (2.0 * model.mu[p])
        epsilon = epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat
        sig_elastic = wp.mat33(
            wp.exp(epsilon[0]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon[1]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon[2]),
        )
        return U * sig_elastic * wp.transpose(V)
    else:
        return F_trial


# for toothpaste
@wp.func
def viscoplasticity_return_mapping_with_StVK(
    F_trial: wp.mat33, model: MPMModelStruct, p: int, dt: float
):
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig_old = wp.vec3(0.0)
    wp.svd3(F_trial, U, sig_old, V)

    sig = wp.vec3(
        wp.max(sig_old[0], 0.01), wp.max(sig_old[1], 0.01), wp.max(sig_old[2], 0.01)
    )  # add this to prevent NaN in extrem cases
    b_trial = wp.vec3(sig[0] * sig[0], sig[1] * sig[1], sig[2] * sig[2])
    epsilon = wp.vec3(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
    trace_epsilon = epsilon[0] + epsilon[1] + epsilon[2]
    epsilon_hat = epsilon - wp.vec3(
        trace_epsilon / 3.0, trace_epsilon / 3.0, trace_epsilon / 3.0
    )
    s_trial = 2.0 * model.mu[p] * epsilon_hat
    s_trial_norm = wp.length(s_trial)
    y = s_trial_norm - wp.sqrt(2.0 / 3.0) * model.yield_stress[p]
    if y > 0:
        mu_hat = model.mu[p] * (b_trial[0] + b_trial[1] + b_trial[2]) / 3.0
        s_new_norm = s_trial_norm - y / (
            1.0 + model.plastic_viscosity / (2.0 * mu_hat * dt)
        )
        s_new = (s_new_norm / s_trial_norm) * s_trial
        epsilon_new = 1.0 / (2.0 * model.mu[p]) * s_new + wp.vec3(
            trace_epsilon / 3.0, trace_epsilon / 3.0, trace_epsilon / 3.0
        )
        sig_elastic = wp.mat33(
            wp.exp(epsilon_new[0]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon_new[1]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon_new[2]),
        )
        F_elastic = U * sig_elastic * wp.transpose(V)
        return F_elastic
    else:
        return F_trial


@wp.func
def sand_return_mapping(
    F_trial: wp.mat33, state: MPMStateStruct, model: MPMModelStruct, p: int
):
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig = wp.vec3(0.0)
    wp.svd3(F_trial, U, sig, V)

    epsilon = wp.vec3(
        wp.log(wp.max(wp.abs(sig[0]), 1e-14)),
        wp.log(wp.max(wp.abs(sig[1]), 1e-14)),
        wp.log(wp.max(wp.abs(sig[2]), 1e-14)),
    )
    sigma_out = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    tr = epsilon[0] + epsilon[1] + epsilon[2]  # + state.particle_Jp[p]
    epsilon_hat = epsilon - wp.vec3(tr / 3.0, tr / 3.0, tr / 3.0)
    epsilon_hat_norm = wp.length(epsilon_hat)
    delta_gamma = (
        epsilon_hat_norm
        + (3.0 * model.lam[p] + 2.0 * model.mu[p])
        / (2.0 * model.mu[p])
        * tr
        * model.alpha
    )

    if delta_gamma <= 0:
        F_elastic = F_trial

    if delta_gamma > 0 and tr > 0:
        F_elastic = U * wp.transpose(V)

    if delta_gamma > 0 and tr <= 0:
        H = epsilon - epsilon_hat * (delta_gamma / epsilon_hat_norm)
        s_new = wp.vec3(wp.exp(H[0]), wp.exp(H[1]), wp.exp(H[2]))

        F_elastic = U * wp.diag(s_new) * wp.transpose(V)
    return F_elastic


@wp.func
def material_return_mapping(
    F_trial: wp.mat33,
    state: MPMStateStruct,
    model: MPMModelStruct,
    p: int,
    dt: float,
):
    if model.material == 1:
        return von_mises_return_mapping(F_trial, model, p)
    elif model.material == 2:
        return sand_return_mapping(F_trial, state, model, p)
    elif model.material == 3:
        return viscoplasticity_return_mapping_with_StVK(F_trial, model, p, dt)
    elif model.material == 5:
        return von_mises_return_mapping_with_damage(F_trial, model, p)
    return F_trial


@wp.func
def evaluate_material_return_mapping(
    F_trial: wp.mat33,
    state: MPMStateStruct,
    model: MPMModelStruct,
    p: int,
    dt: float,
):
    if model.material == 1:
        return evaluate_von_mises_return_mapping(F_trial, model, p)
    elif model.material == 2:
        return sand_return_mapping(F_trial, state, model, p)
    elif model.material == 3:
        return viscoplasticity_return_mapping_with_StVK(F_trial, model, p, dt)
    elif model.material == 5:
        return evaluate_von_mises_return_mapping_with_damage(F_trial, model, p)
    return F_trial


@wp.func
def commit_material_return_mapping(
    F_trial: wp.mat33,
    state: MPMStateStruct,
    model: MPMModelStruct,
    p: int,
    dt: float,
):
    return material_return_mapping(F_trial, state, model, p, dt)


@wp.kernel
def compute_mu_lam_from_E_nu(state: MPMStateStruct, model: MPMModelStruct):
    p = wp.tid()
    model.mu[p] = model.E[p] / (2.0 * (1.0 + model.nu[p]))
    model.lam[p] = (
        model.E[p] * model.nu[p] / ((1.0 + model.nu[p]) * (1.0 - 2.0 * model.nu[p]))
    )


@wp.kernel
def zero_grid(state: MPMStateStruct, model: MPMModelStruct):
    grid_x, grid_y, grid_z = wp.tid()
    state.grid_m[grid_x, grid_y, grid_z] = 0.0
    state.grid_v_in[grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)
    state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def apply_fixed_particle_indices(
    time: float,
    state: MPMStateStruct,
    model: MPMModelStruct,
    modifier: FixedParticleModifier,
):
    tid = wp.tid()
    p = modifier.indices[tid]
    if (
        p >= 0
        and p < model.n_particles
    ):
        zero_mat = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        I33 = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        motion_time = wp.min(
            wp.max(time - modifier.start_time, 0.0),
            wp.max(modifier.end_time - modifier.start_time, 0.0),
        )
        active_velocity = wp.vec3(0.0, 0.0, 0.0)
        if time >= modifier.start_time and time < modifier.end_time:
            active_velocity = modifier.velocity
        state.particle_x[p] = modifier.rest_x[tid] + modifier.velocity * motion_time
        state.particle_v[p] = active_velocity
        state.particle_C[p] = zero_mat
        state.particle_D[p] = zero_mat
        if modifier.reset_deformation != 0:
            state.particle_F[p] = I33
            state.particle_F_trial[p] = I33
            state.particle_R[p] = I33
            state.particle_stress[p] = zero_mat
            state.particle_Jp[p] = 0.0


@wp.func
def compute_dweight(
    model: MPMModelStruct, w: wp.mat33, dw: wp.mat33, i: int, j: int, k: int
):
    dweight = wp.vec3(
        dw[0, i] * w[1, j] * w[2, k],
        w[0, i] * dw[1, j] * w[2, k],
        w[0, i] * w[1, j] * dw[2, k],
    )
    return dweight * model.inv_dx


@wp.func
def update_cov(state: MPMStateStruct, p: int, grad_v: wp.mat33, dt: float):
    cov_n = wp.mat33(0.0)
    cov_n[0, 0] = state.particle_cov[p * 6]
    cov_n[0, 1] = state.particle_cov[p * 6 + 1]
    cov_n[0, 2] = state.particle_cov[p * 6 + 2]
    cov_n[1, 0] = state.particle_cov[p * 6 + 1]
    cov_n[1, 1] = state.particle_cov[p * 6 + 3]
    cov_n[1, 2] = state.particle_cov[p * 6 + 4]
    cov_n[2, 0] = state.particle_cov[p * 6 + 2]
    cov_n[2, 1] = state.particle_cov[p * 6 + 4]
    cov_n[2, 2] = state.particle_cov[p * 6 + 5]

    cov_np1 = cov_n + dt * (grad_v * cov_n + cov_n * wp.transpose(grad_v))

    state.particle_cov[p * 6] = cov_np1[0, 0]
    state.particle_cov[p * 6 + 1] = cov_np1[0, 1]
    state.particle_cov[p * 6 + 2] = cov_np1[0, 2]
    state.particle_cov[p * 6 + 3] = cov_np1[1, 1]
    state.particle_cov[p * 6 + 4] = cov_np1[1, 2]
    state.particle_cov[p * 6 + 5] = cov_np1[2, 2]


@wp.func
def mpm_zero_mat33():
    return wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@wp.func
def mpm_stencil_in_grid(
    base_pos_x: int, base_pos_y: int, base_pos_z: int, model: MPMModelStruct
):
    return (
        base_pos_x >= 0
        and base_pos_y >= 0
        and base_pos_z >= 0
        and base_pos_x + 2 < model.grid_dim_x
        and base_pos_y + 2 < model.grid_dim_y
        and base_pos_z + 2 < model.grid_dim_z
    )


@wp.func
def mpm_clamp_to_grid_domain(x: wp.vec3, model: MPMModelStruct):
    padding = 3.0 * model.dx
    max_x = model.dx * wp.float(model.grid_dim_x) - padding
    max_y = model.dx * wp.float(model.grid_dim_y) - padding
    max_z = model.dx * wp.float(model.grid_dim_z) - padding
    return wp.vec3(
        wp.min(wp.max(x[0], padding), max_x),
        wp.min(wp.max(x[1], padding), max_y),
        wp.min(wp.max(x[2], padding), max_z),
    )


@wp.kernel
def p2g_apic_with_stress(state: MPMStateStruct, model: MPMModelStruct, dt: float):
    # input given to p2g:   particle_stress
    #                       particle_x
    #                       particle_v
    #                       particle_C
    p = wp.tid()
    if state.particle_selection[p] == 0:
        stress = state.particle_stress[p]
        grid_pos = state.particle_x[p] * model.inv_dx
        base_pos_x = wp.int(grid_pos[0] - 0.5)
        base_pos_y = wp.int(grid_pos[1] - 0.5)
        base_pos_z = wp.int(grid_pos[2] - 0.5)
        if not mpm_stencil_in_grid(base_pos_x, base_pos_y, base_pos_z, model):
            state.particle_x[p] = mpm_clamp_to_grid_domain(state.particle_x[p], model)
            state.particle_v[p] = wp.vec3(0.0, 0.0, 0.0)
            state.particle_C[p] = mpm_zero_mat33()
            grid_pos = state.particle_x[p] * model.inv_dx
            base_pos_x = wp.int(grid_pos[0] - 0.5)
            base_pos_y = wp.int(grid_pos[1] - 0.5)
            base_pos_z = wp.int(grid_pos[2] - 0.5)
        if mpm_stencil_in_grid(base_pos_x, base_pos_y, base_pos_z, model):
            fx = grid_pos - wp.vec3(
                wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z)
            )
            wa = wp.vec3(1.5) - fx
            wb = fx - wp.vec3(1.0)
            wc = fx - wp.vec3(0.5)
            w = wp.mat33(
                wp.cw_mul(wa, wa) * 0.5,
                wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
                wp.cw_mul(wc, wc) * 0.5,
            )
            dw = wp.mat33(fx - wp.vec3(1.5), -2.0 * (fx - wp.vec3(1.0)), fx - wp.vec3(0.5))

            for i in range(0, 3):
                for j in range(0, 3):
                    for k in range(0, 3):
                        dpos = (
                            wp.vec3(wp.float(i), wp.float(j), wp.float(k)) - fx
                        ) * model.dx
                        ix = base_pos_x + i
                        iy = base_pos_y + j
                        iz = base_pos_z + k
                        weight = w[0, i] * w[1, j] * w[2, k]  # tricubic interpolation
                        dweight = compute_dweight(model, w, dw, i, j, k)
                        C = state.particle_C[p]
                        # if model.rpic = 0, standard apic
                        C = (1.0 - model.rpic_damping) * C + model.rpic_damping / 2.0 * (
                            C - wp.transpose(C)
                        )
                        if model.rpic_damping < -0.001:
                            # standard pic
                            C = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

                        elastic_force = -state.particle_vol[p] * stress * dweight
                        v_in_add = (
                            weight
                            * state.particle_mass[p]
                            * (state.particle_v[p] + C * dpos)
                            + dt * elastic_force
                        )
                        wp.atomic_add(state.grid_v_in, ix, iy, iz, v_in_add)
                        wp.atomic_add(
                            state.grid_m, ix, iy, iz, weight * state.particle_mass[p]
                        )


@wp.kernel
def p2g_apic_no_stress(state: MPMStateStruct, model: MPMModelStruct):
    # Build the start-of-step grid mass and APIC momentum without applying
    # constitutive forces. The implicit residual evaluates those forces at
    # trial end-of-step states instead.
    p = wp.tid()
    if state.particle_selection[p] == 0:
        grid_pos = state.particle_x[p] * model.inv_dx
        base_pos_x = wp.int(grid_pos[0] - 0.5)
        base_pos_y = wp.int(grid_pos[1] - 0.5)
        base_pos_z = wp.int(grid_pos[2] - 0.5)
        if not mpm_stencil_in_grid(base_pos_x, base_pos_y, base_pos_z, model):
            state.particle_x[p] = mpm_clamp_to_grid_domain(state.particle_x[p], model)
            state.particle_v[p] = wp.vec3(0.0, 0.0, 0.0)
            state.particle_C[p] = mpm_zero_mat33()
            grid_pos = state.particle_x[p] * model.inv_dx
            base_pos_x = wp.int(grid_pos[0] - 0.5)
            base_pos_y = wp.int(grid_pos[1] - 0.5)
            base_pos_z = wp.int(grid_pos[2] - 0.5)
        if mpm_stencil_in_grid(base_pos_x, base_pos_y, base_pos_z, model):
            fx = grid_pos - wp.vec3(
                wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z)
            )
            wa = wp.vec3(1.5) - fx
            wb = fx - wp.vec3(1.0)
            wc = fx - wp.vec3(0.5)
            w = wp.mat33(
                wp.cw_mul(wa, wa) * 0.5,
                wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
                wp.cw_mul(wc, wc) * 0.5,
            )

            for i in range(0, 3):
                for j in range(0, 3):
                    for k in range(0, 3):
                        dpos = (
                            wp.vec3(wp.float(i), wp.float(j), wp.float(k)) - fx
                        ) * model.dx
                        ix = base_pos_x + i
                        iy = base_pos_y + j
                        iz = base_pos_z + k
                        weight = w[0, i] * w[1, j] * w[2, k]
                        C = state.particle_C[p]
                        C = (1.0 - model.rpic_damping) * C + model.rpic_damping / 2.0 * (
                            C - wp.transpose(C)
                        )
                        if model.rpic_damping < -0.001:
                            C = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                        momentum = weight * state.particle_mass[p] * (
                            state.particle_v[p] + C * dpos
                        )
                        wp.atomic_add(state.grid_v_in, ix, iy, iz, momentum)
                        wp.atomic_add(
                            state.grid_m, ix, iy, iz, weight * state.particle_mass[p]
                        )


@wp.kernel
def grid_normalization_no_gravity(state: MPMStateStruct, model: MPMModelStruct):
    grid_x, grid_y, grid_z = wp.tid()
    if state.grid_m[grid_x, grid_y, grid_z] > 1e-15:
        state.grid_v_out[grid_x, grid_y, grid_z] = state.grid_v_in[
            grid_x, grid_y, grid_z
        ] * (1.0 / state.grid_m[grid_x, grid_y, grid_z])


# add gravity
@wp.kernel
def grid_normalization_and_gravity(
    state: MPMStateStruct, model: MPMModelStruct, dt: float
):
    grid_x, grid_y, grid_z = wp.tid()
    if state.grid_m[grid_x, grid_y, grid_z] > 1e-15:
        v_out = state.grid_v_in[grid_x, grid_y, grid_z] * (
            1.0 / state.grid_m[grid_x, grid_y, grid_z]
        )
        # add gravity
        v_out = v_out + dt * model.gravitational_accelaration
        state.grid_v_out[grid_x, grid_y, grid_z] = v_out


@wp.kernel
def add_damping_via_grid(state: MPMStateStruct, scale: float):
    grid_x, grid_y, grid_z = wp.tid()
    state.grid_v_out[grid_x, grid_y, grid_z] = (
        state.grid_v_out[grid_x, grid_y, grid_z] * scale
    )


@wp.kernel
def g2p(state: MPMStateStruct, model: MPMModelStruct, dt: float):
    p = wp.tid()
    if state.particle_selection[p] == 0:
        grid_pos = state.particle_x[p] * model.inv_dx
        base_pos_x = wp.int(grid_pos[0] - 0.5)
        base_pos_y = wp.int(grid_pos[1] - 0.5)
        base_pos_z = wp.int(grid_pos[2] - 0.5)
        if not mpm_stencil_in_grid(base_pos_x, base_pos_y, base_pos_z, model):
            state.particle_x[p] = mpm_clamp_to_grid_domain(state.particle_x[p], model)
            state.particle_v[p] = wp.vec3(0.0, 0.0, 0.0)
            state.particle_C[p] = mpm_zero_mat33()
            grid_pos = state.particle_x[p] * model.inv_dx
            base_pos_x = wp.int(grid_pos[0] - 0.5)
            base_pos_y = wp.int(grid_pos[1] - 0.5)
            base_pos_z = wp.int(grid_pos[2] - 0.5)
        if mpm_stencil_in_grid(base_pos_x, base_pos_y, base_pos_z, model):
            fx = grid_pos - wp.vec3(
                wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z)
            )
            wa = wp.vec3(1.5) - fx
            wb = fx - wp.vec3(1.0)
            wc = fx - wp.vec3(0.5)
            w = wp.mat33(
                wp.cw_mul(wa, wa) * 0.5,
                wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
                wp.cw_mul(wc, wc) * 0.5,
            )
            dw = wp.mat33(fx - wp.vec3(1.5), -2.0 * (fx - wp.vec3(1.0)), fx - wp.vec3(0.5))
            new_v = wp.vec3(0.0, 0.0, 0.0)
            new_C = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            new_F = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            for i in range(0, 3):
                for j in range(0, 3):
                    for k in range(0, 3):
                        ix = base_pos_x + i
                        iy = base_pos_y + j
                        iz = base_pos_z + k
                        dpos = wp.vec3(wp.float(i), wp.float(j), wp.float(k)) - fx
                        weight = w[0, i] * w[1, j] * w[2, k]  # tricubic interpolation
                        grid_v = state.grid_v_out[ix, iy, iz]
                        new_v = new_v + grid_v * weight
                        new_C = new_C + wp.outer(grid_v, dpos) * (
                            weight * model.inv_dx * 4.0
                        )
                        dweight = compute_dweight(model, w, dw, i, j, k)
                        new_F = new_F + wp.outer(grid_v, dweight)

            new_x = state.particle_x[p] + dt * new_v
            new_grid_pos = new_x * model.inv_dx
            new_base_pos_x = wp.int(new_grid_pos[0] - 0.5)
            new_base_pos_y = wp.int(new_grid_pos[1] - 0.5)
            new_base_pos_z = wp.int(new_grid_pos[2] - 0.5)
            if not mpm_stencil_in_grid(
                new_base_pos_x, new_base_pos_y, new_base_pos_z, model
            ):
                new_x = mpm_clamp_to_grid_domain(new_x, model)
                new_v = wp.vec3(0.0, 0.0, 0.0)
                new_C = mpm_zero_mat33()
                new_F = mpm_zero_mat33()
            state.particle_v[p] = new_v
            state.particle_x[p] = new_x
            state.particle_C[p] = new_C
            I33 = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
            F_tmp = (I33 + new_F * dt) * state.particle_F[p]
            state.particle_F_trial[p] = F_tmp

            if model.update_cov_with_F:
                update_cov(state, p, new_F, dt)


@wp.func
def pbmpm_safe_inverse(F: wp.mat33):
    J = wp.determinant(F)
    if wp.abs(J) < 1e-8:
        return wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    return wp.inverse(F)


@wp.func
def pbmpm_clamp_mat33(A: wp.mat33, limit: float):
    return wp.mat33(
        wp.min(wp.max(A[0, 0], -limit), limit),
        wp.min(wp.max(A[0, 1], -limit), limit),
        wp.min(wp.max(A[0, 2], -limit), limit),
        wp.min(wp.max(A[1, 0], -limit), limit),
        wp.min(wp.max(A[1, 1], -limit), limit),
        wp.min(wp.max(A[1, 2], -limit), limit),
        wp.min(wp.max(A[2, 0], -limit), limit),
        wp.min(wp.max(A[2, 1], -limit), limit),
        wp.min(wp.max(A[2, 2], -limit), limit),
    )


@wp.func
def pbmpm_clamp_to_grid_domain(x: wp.vec3, model: MPMModelStruct):
    padding = 3.0 * model.dx
    max_x = model.dx * wp.float(model.grid_dim_x) - padding
    max_y = model.dx * wp.float(model.grid_dim_y) - padding
    max_z = model.dx * wp.float(model.grid_dim_z) - padding
    return wp.vec3(
        wp.min(wp.max(x[0], padding), max_x),
        wp.min(wp.max(x[1], padding), max_y),
        wp.min(wp.max(x[2], padding), max_z),
    )


@wp.func
def pbmpm_zero_particle_kinematics(state: MPMStateStruct, p: int):
    state.particle_v[p] = wp.vec3(0.0, 0.0, 0.0)
    state.particle_C[p] = mpm_zero_mat33()
    state.particle_D[p] = mpm_zero_mat33()


@wp.func
def pbmpm_solve_D(
    F_base: wp.mat33,
    D_in: wp.mat33,
    state: MPMStateStruct,
    model: MPMModelStruct,
    p: int,
    dt: float,
):
    I33 = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    F = (I33 + D_in) * F_base

    U = wp.mat33(0.0)
    V = wp.mat33(0.0)
    sig = wp.vec3(0.0)
    wp.svd3(F, U, sig, V)

    if wp.determinant(U) < 0.0:
        U[0, 2] = -U[0, 2]
        U[1, 2] = -U[1, 2]
        U[2, 2] = -U[2, 2]
        sig[2] = -sig[2]

    if wp.determinant(V) < 0.0:
        V[0, 2] = -V[0, 2]
        V[1, 2] = -V[1, 2]
        V[2, 2] = -V[2, 2]
        sig[2] = -sig[2]

    if model.pbmpm_plastic_mode == 1:
        lo = wp.max(model.pbmpm_yield_min, 0.01)
        hi = wp.max(model.pbmpm_yield_max, lo + 1e-4)
        sig_clamped = wp.vec3(
            wp.min(wp.max(wp.max(wp.abs(sig[0]), 0.01), lo), hi),
            wp.min(wp.max(wp.max(wp.abs(sig[1]), 0.01), lo), hi),
            wp.min(wp.max(wp.max(wp.abs(sig[2]), 0.01), lo), hi),
        )
        F = U * wp.diag(sig_clamped) * wp.transpose(V)

    R = U * wp.transpose(V)

    det_F = wp.determinant(F)
    sign_J = 1.0
    if det_F < 0.0:
        sign_J = -1.0
    J = wp.min(wp.max(wp.abs(det_F), 0.1), 1000.0)
    inv_cuberoot_J = sign_J * wp.exp(-wp.log(J) / 3.0)
    Q = F * inv_cuberoot_J

    alpha = wp.min(wp.max(model.pbmpm_elasticity_ratio, 0.0), 1.0)
    target = R * alpha + Q * (1.0 - alpha)
    diff = (target * pbmpm_safe_inverse(F_base) - I33) - D_in
    relaxation = wp.min(wp.max(model.pbmpm_elastic_relaxation, 0.0), 2.0)
    return pbmpm_clamp_mat33(D_in + relaxation * diff, 1.0)


@wp.kernel
def pbmpm_solve_constraints(state: MPMStateStruct, model: MPMModelStruct, dt: float):
    p = wp.tid()
    if state.particle_selection[p] == 0:
        state.particle_D[p] = pbmpm_solve_D(
            state.particle_F[p], state.particle_D[p], state, model, p, dt
        )


@wp.kernel
def p2g_pbmpm_with_D(state: MPMStateStruct, model: MPMModelStruct, dt: float):
    p = wp.tid()
    if state.particle_selection[p] == 0:
        grid_pos = state.particle_x[p] * model.inv_dx
        base_pos_x = wp.int(grid_pos[0] - 0.5)
        base_pos_y = wp.int(grid_pos[1] - 0.5)
        base_pos_z = wp.int(grid_pos[2] - 0.5)
        if not mpm_stencil_in_grid(base_pos_x, base_pos_y, base_pos_z, model):
            state.particle_x[p] = pbmpm_clamp_to_grid_domain(
                state.particle_x[p], model
            )
            pbmpm_zero_particle_kinematics(state, p)
            grid_pos = state.particle_x[p] * model.inv_dx
            base_pos_x = wp.int(grid_pos[0] - 0.5)
            base_pos_y = wp.int(grid_pos[1] - 0.5)
            base_pos_z = wp.int(grid_pos[2] - 0.5)
        if mpm_stencil_in_grid(base_pos_x, base_pos_y, base_pos_z, model):
            fx = grid_pos - wp.vec3(
                wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z)
            )
            wa = wp.vec3(1.5) - fx
            wb = fx - wp.vec3(1.0)
            wc = fx - wp.vec3(0.5)
            w = wp.mat33(
                wp.cw_mul(wa, wa) * 0.5,
                wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
                wp.cw_mul(wc, wc) * 0.5,
            )
            inv_dt = 1.0 / wp.max(dt, 1e-9)
            D_velocity = state.particle_D[p] * inv_dt
            for i in range(0, 3):
                for j in range(0, 3):
                    for k in range(0, 3):
                        dpos = (
                            wp.vec3(wp.float(i), wp.float(j), wp.float(k)) - fx
                        ) * model.dx
                        ix = base_pos_x + i
                        iy = base_pos_y + j
                        iz = base_pos_z + k
                        weight = w[0, i] * w[1, j] * w[2, k]
                        momentum = weight * state.particle_mass[p] * (
                            state.particle_v[p] + D_velocity * dpos
                        )
                        wp.atomic_add(state.grid_v_in, ix, iy, iz, momentum)
                        wp.atomic_add(
                            state.grid_m, ix, iy, iz, weight * state.particle_mass[p]
                        )


@wp.kernel
def g2p_pbmpm_update_D(state: MPMStateStruct, model: MPMModelStruct, dt: float):
    p = wp.tid()
    if state.particle_selection[p] == 0:
        grid_pos = state.particle_x[p] * model.inv_dx
        base_pos_x = wp.int(grid_pos[0] - 0.5)
        base_pos_y = wp.int(grid_pos[1] - 0.5)
        base_pos_z = wp.int(grid_pos[2] - 0.5)
        if not mpm_stencil_in_grid(base_pos_x, base_pos_y, base_pos_z, model):
            state.particle_x[p] = pbmpm_clamp_to_grid_domain(
                state.particle_x[p], model
            )
            pbmpm_zero_particle_kinematics(state, p)
            grid_pos = state.particle_x[p] * model.inv_dx
            base_pos_x = wp.int(grid_pos[0] - 0.5)
            base_pos_y = wp.int(grid_pos[1] - 0.5)
            base_pos_z = wp.int(grid_pos[2] - 0.5)
        if mpm_stencil_in_grid(base_pos_x, base_pos_y, base_pos_z, model):
            fx = grid_pos - wp.vec3(
                wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z)
            )
            wa = wp.vec3(1.5) - fx
            wb = fx - wp.vec3(1.0)
            wc = fx - wp.vec3(0.5)
            w = wp.mat33(
                wp.cw_mul(wa, wa) * 0.5,
                wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
                wp.cw_mul(wc, wc) * 0.5,
            )
            dw = wp.mat33(
                fx - wp.vec3(1.5), -2.0 * (fx - wp.vec3(1.0)), fx - wp.vec3(0.5)
            )
            new_v = wp.vec3(0.0, 0.0, 0.0)
            new_C = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            grad_v = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            for i in range(0, 3):
                for j in range(0, 3):
                    for k in range(0, 3):
                        ix = base_pos_x + i
                        iy = base_pos_y + j
                        iz = base_pos_z + k
                        dpos = wp.vec3(wp.float(i), wp.float(j), wp.float(k)) - fx
                        weight = w[0, i] * w[1, j] * w[2, k]
                        grid_v = state.grid_v_out[ix, iy, iz]
                        new_v = new_v + grid_v * weight
                        new_C = new_C + wp.outer(grid_v, dpos) * (
                            weight * model.inv_dx * 4.0
                        )
                        dweight = compute_dweight(model, w, dw, i, j, k)
                        grad_v = grad_v + wp.outer(grid_v, dweight)

            state.particle_v[p] = new_v
            state.particle_C[p] = new_C
            state.particle_D[p] = pbmpm_clamp_mat33(grad_v * dt, 1.0)


@wp.kernel
def pbmpm_project_constraints(state: MPMStateStruct, model: MPMModelStruct, dt: float):
    p = wp.tid()
    if state.particle_selection[p] == 0:
        state.particle_D[p] = pbmpm_solve_D(
            state.particle_F[p], state.particle_D[p], state, model, p, dt
        )
        I33 = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        state.particle_F_trial[p] = (I33 + state.particle_D[p]) * state.particle_F[p]


@wp.kernel
def pbmpm_integrate_particles(state: MPMStateStruct, model: MPMModelStruct, dt: float):
    p = wp.tid()
    if state.particle_selection[p] == 0:
        I33 = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        D_commit = state.particle_D[p]
        new_x = state.particle_x[p] + dt * state.particle_v[p]
        new_grid_pos = new_x * model.inv_dx
        new_base_pos_x = wp.int(new_grid_pos[0] - 0.5)
        new_base_pos_y = wp.int(new_grid_pos[1] - 0.5)
        new_base_pos_z = wp.int(new_grid_pos[2] - 0.5)
        left_grid = not mpm_stencil_in_grid(
            new_base_pos_x, new_base_pos_y, new_base_pos_z, model
        )
        if left_grid:
            new_x = pbmpm_clamp_to_grid_domain(new_x, model)
            D_commit = mpm_zero_mat33()
            state.particle_D[p] = D_commit
        F_tmp = (I33 + D_commit) * state.particle_F[p]
        if model.pbmpm_plastic_mode == 1:
            # PBMPM stretch clamp owns plastic limiting in this mode; avoid a
            # second material return mapping in the same frame.
            state.particle_F[p] = F_tmp
        else:
            state.particle_F[p] = commit_material_return_mapping(
                F_tmp, state, model, p, dt
            )
        state.particle_F_trial[p] = state.particle_F[p]
        state.particle_x[p] = new_x
        if model.update_cov_with_F:
            update_cov(
                state,
                p,
                D_commit * (1.0 / wp.max(dt, 1e-9)),
                dt,
            )
        state.particle_v[p] = state.particle_v[p] + model.gravitational_accelaration * dt
        if left_grid:
            pbmpm_zero_particle_kinematics(state, p)


@wp.kernel
def pbmpm_clear_D(state: MPMStateStruct):
    p = wp.tid()
    if state.particle_selection[p] == 0:
        state.particle_D[p] = wp.mat33(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        )


# compute (Kirchhoff) stress = stress(returnMap(F_trial))
@wp.kernel
def compute_stress_from_F_trial(
    state: MPMStateStruct, model: MPMModelStruct, dt: float
):
    p = wp.tid()
    if state.particle_selection[p] == 0:
        state.particle_F[p] = commit_material_return_mapping(
            state.particle_F_trial[p], state, model, p, dt
        )
        state.particle_F_trial[p] = state.particle_F[p]

        # also compute stress here
        J = wp.determinant(state.particle_F[p])
        U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        sig = wp.vec3(0.0)
        stress = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        wp.svd3(state.particle_F[p], U, sig, V)
        if model.material == 0 or model.material == 5:
            stress = kirchoff_stress_FCR(
                state.particle_F[p], U, V, J, model.mu[p], model.lam[p]
            )
        if model.material == 1:
            stress = kirchoff_stress_StVK(
                state.particle_F[p], U, V, sig, model.mu[p], model.lam[p]
            )
        if model.material == 2:
            stress = kirchoff_stress_drucker_prager(
                state.particle_F[p], U, V, sig, model.mu[p], model.lam[p]
            )
        if model.material == 3:
            # temporarily use stvk, subject to change
            stress = kirchoff_stress_StVK(
                state.particle_F[p], U, V, sig, model.mu[p], model.lam[p]
            )

        stress = (stress + wp.transpose(stress)) / 2.0  # enfore symmetry
        state.particle_stress[p] = stress


@wp.kernel
def zero_grid_vec3(grid: wp.array(dtype=wp.vec3, ndim=3)):
    grid_x, grid_y, grid_z = wp.tid()
    grid[grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def zero_grid_int(grid: wp.array(dtype=int, ndim=3)):
    grid_x, grid_y, grid_z = wp.tid()
    grid[grid_x, grid_y, grid_z] = 0


@wp.kernel
def merge_dirichlet_grid(
    mask: wp.array(dtype=int, ndim=3),
    target: wp.array(dtype=wp.vec3, ndim=3),
    source_mask: wp.array(dtype=int, ndim=3),
    source_target: wp.array(dtype=wp.vec3, ndim=3),
):
    grid_x, grid_y, grid_z = wp.tid()
    if source_mask[grid_x, grid_y, grid_z] > 0:
        mask[grid_x, grid_y, grid_z] = 1
        target[grid_x, grid_y, grid_z] = source_target[grid_x, grid_y, grid_z]


@wp.kernel
def zero_grid_float(grid: wp.array(dtype=float, ndim=3)):
    grid_x, grid_y, grid_z = wp.tid()
    grid[grid_x, grid_y, grid_z] = 0.0


@wp.kernel
def implicit_du_to_velocity(
    state: MPMStateStruct,
    grid_du: wp.array(dtype=wp.vec3, ndim=3),
    grid_vn: wp.array(dtype=wp.vec3, ndim=3),
    grid_an: wp.array(dtype=wp.vec3, ndim=3),
    grid_v_trial: wp.array(dtype=wp.vec3, ndim=3),
    beta: float,
    gamma: float,
    dt: float,
):
    grid_x, grid_y, grid_z = wp.tid()
    if state.grid_m[grid_x, grid_y, grid_z] > 1e-15:
        v_n = grid_vn[grid_x, grid_y, grid_z]
        a_n = grid_an[grid_x, grid_y, grid_z]
        du = grid_du[grid_x, grid_y, grid_z]
        a_np1 = (
            du - dt * v_n - dt * dt * (0.5 - beta) * a_n
        ) * (1.0 / (beta * dt * dt))
        grid_v_trial[grid_x, grid_y, grid_z] = (
            v_n + dt * ((1.0 - gamma) * a_n + gamma * a_np1)
        )
    else:
        grid_v_trial[grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def implicit_update_acceleration_from_velocity(
    state: MPMStateStruct,
    grid_vn: wp.array(dtype=wp.vec3, ndim=3),
    grid_an: wp.array(dtype=wp.vec3, ndim=3),
    gamma: float,
    dt: float,
):
    grid_x, grid_y, grid_z = wp.tid()
    if state.grid_m[grid_x, grid_y, grid_z] > 1e-15:
        v_n = grid_vn[grid_x, grid_y, grid_z]
        a_n = grid_an[grid_x, grid_y, grid_z]
        v_np1 = state.grid_v_out[grid_x, grid_y, grid_z]
        grid_an[grid_x, grid_y, grid_z] = (
            v_np1 - v_n - dt * (1.0 - gamma) * a_n
        ) * (1.0 / (gamma * dt))
    else:
        grid_an[grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def implicit_project_du_from_velocity(
    state: MPMStateStruct,
    grid_du: wp.array(dtype=wp.vec3, ndim=3),
    grid_vn: wp.array(dtype=wp.vec3, ndim=3),
    grid_an: wp.array(dtype=wp.vec3, ndim=3),
    beta: float,
    gamma: float,
    dt: float,
):
    grid_x, grid_y, grid_z = wp.tid()
    if state.grid_m[grid_x, grid_y, grid_z] > 1e-15:
        v_n = grid_vn[grid_x, grid_y, grid_z]
        a_n = grid_an[grid_x, grid_y, grid_z]
        v_np1 = state.grid_v_out[grid_x, grid_y, grid_z]
        a_np1 = (
            v_np1 - v_n - dt * (1.0 - gamma) * a_n
        ) * (1.0 / (gamma * dt))
        grid_du[grid_x, grid_y, grid_z] = (
            dt * v_n + dt * dt * (0.5 - beta) * a_n + beta * dt * dt * a_np1
        )
    else:
        grid_du[grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def implicit_compute_trial_stress(
    state: MPMStateStruct,
    model: MPMModelStruct,
    grid_v_trial: wp.array(dtype=wp.vec3, ndim=3),
    trial_F: wp.array(dtype=wp.mat33),
    trial_stress: wp.array(dtype=wp.mat33),
    dt: float,
):
    p = wp.tid()
    if state.particle_selection[p] == 0:
        trial_F[p] = state.particle_F[p]
        trial_stress[p] = mpm_zero_mat33()
        particle_x = state.particle_x[p]
        clamped_x = particle_x
        grid_pos = clamped_x * model.inv_dx
        base_pos_x = wp.int(grid_pos[0] - 0.5)
        base_pos_y = wp.int(grid_pos[1] - 0.5)
        base_pos_z = wp.int(grid_pos[2] - 0.5)
        if not mpm_stencil_in_grid(base_pos_x, base_pos_y, base_pos_z, model):
            clamped_x = mpm_clamp_to_grid_domain(particle_x, model)
            grid_pos = clamped_x * model.inv_dx
            base_pos_x = wp.int(grid_pos[0] - 0.5)
            base_pos_y = wp.int(grid_pos[1] - 0.5)
            base_pos_z = wp.int(grid_pos[2] - 0.5)
        if mpm_stencil_in_grid(base_pos_x, base_pos_y, base_pos_z, model):
            fx = grid_pos - wp.vec3(
                wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z)
            )
            wa = wp.vec3(1.5) - fx
            wb = fx - wp.vec3(1.0)
            wc = fx - wp.vec3(0.5)
            w = wp.mat33(
                wp.cw_mul(wa, wa) * 0.5,
                wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
                wp.cw_mul(wc, wc) * 0.5,
            )
            dw = wp.mat33(
                fx - wp.vec3(1.5), -2.0 * (fx - wp.vec3(1.0)), fx - wp.vec3(0.5)
            )
            grad_v = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            for i in range(0, 3):
                for j in range(0, 3):
                    for k in range(0, 3):
                        ix = base_pos_x + i
                        iy = base_pos_y + j
                        iz = base_pos_z + k
                        dweight = compute_dweight(model, w, dw, i, j, k)
                        grid_v = grid_v_trial[ix, iy, iz]
                        grad_v = grad_v + wp.outer(grid_v, dweight)

            I33 = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
            F_tmp = (I33 + grad_v * dt) * state.particle_F[p]
            F_elastic = evaluate_material_return_mapping(F_tmp, state, model, p, dt)

            J = wp.determinant(F_elastic)
            U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            sig = wp.vec3(0.0)
            stress = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            wp.svd3(F_elastic, U, sig, V)
            if model.material == 0 or model.material == 5:
                stress = kirchoff_stress_FCR(F_elastic, U, V, J, model.mu[p], model.lam[p])
            if model.material == 1:
                stress = kirchoff_stress_StVK(
                    F_elastic, U, V, sig, model.mu[p], model.lam[p]
                )
            if model.material == 2:
                stress = kirchoff_stress_drucker_prager(
                    F_elastic, U, V, sig, model.mu[p], model.lam[p]
                )
            if model.material == 3:
                stress = kirchoff_stress_StVK(
                    F_elastic, U, V, sig, model.mu[p], model.lam[p]
                )

            trial_F[p] = F_elastic
            trial_stress[p] = (stress + wp.transpose(stress)) / 2.0


@wp.kernel
def implicit_accumulate_stiffness_diag(
    state: MPMStateStruct,
    model: MPMModelStruct,
    grid_stiffness: wp.array(dtype=float, ndim=3),
):
    p = wp.tid()
    if state.particle_selection[p] == 0:
        particle_x = state.particle_x[p]
        clamped_x = particle_x
        grid_pos = clamped_x * model.inv_dx
        base_pos_x = wp.int(grid_pos[0] - 0.5)
        base_pos_y = wp.int(grid_pos[1] - 0.5)
        base_pos_z = wp.int(grid_pos[2] - 0.5)
        if not mpm_stencil_in_grid(base_pos_x, base_pos_y, base_pos_z, model):
            clamped_x = mpm_clamp_to_grid_domain(particle_x, model)
            grid_pos = clamped_x * model.inv_dx
            base_pos_x = wp.int(grid_pos[0] - 0.5)
            base_pos_y = wp.int(grid_pos[1] - 0.5)
            base_pos_z = wp.int(grid_pos[2] - 0.5)
        if mpm_stencil_in_grid(base_pos_x, base_pos_y, base_pos_z, model):
            fx = grid_pos - wp.vec3(
                wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z)
            )
            wa = wp.vec3(1.5) - fx
            wb = fx - wp.vec3(1.0)
            wc = fx - wp.vec3(0.5)
            w = wp.mat33(
                wp.cw_mul(wa, wa) * 0.5,
                wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
                wp.cw_mul(wc, wc) * 0.5,
            )
            dw = wp.mat33(
                fx - wp.vec3(1.5), -2.0 * (fx - wp.vec3(1.0)), fx - wp.vec3(0.5)
            )
            stiffness = state.particle_vol[p] * (model.lam[p] + 2.0 * model.mu[p])
            for i in range(0, 3):
                for j in range(0, 3):
                    for k in range(0, 3):
                        ix = base_pos_x + i
                        iy = base_pos_y + j
                        iz = base_pos_z + k
                        dweight = compute_dweight(model, w, dw, i, j, k)
                        wp.atomic_add(
                            grid_stiffness,
                            ix,
                            iy,
                            iz,
                            stiffness * wp.dot(dweight, dweight),
                        )


@wp.kernel
def implicit_accumulate_internal_force(
    state: MPMStateStruct,
    model: MPMModelStruct,
    trial_stress: wp.array(dtype=wp.mat33),
    grid_force: wp.array(dtype=wp.vec3, ndim=3),
):
    p = wp.tid()
    if state.particle_selection[p] == 0:
        particle_x = state.particle_x[p]
        clamped_x = particle_x
        grid_pos = clamped_x * model.inv_dx
        base_pos_x = wp.int(grid_pos[0] - 0.5)
        base_pos_y = wp.int(grid_pos[1] - 0.5)
        base_pos_z = wp.int(grid_pos[2] - 0.5)
        if not mpm_stencil_in_grid(base_pos_x, base_pos_y, base_pos_z, model):
            clamped_x = mpm_clamp_to_grid_domain(particle_x, model)
            grid_pos = clamped_x * model.inv_dx
            base_pos_x = wp.int(grid_pos[0] - 0.5)
            base_pos_y = wp.int(grid_pos[1] - 0.5)
            base_pos_z = wp.int(grid_pos[2] - 0.5)
        if mpm_stencil_in_grid(base_pos_x, base_pos_y, base_pos_z, model):
            fx = grid_pos - wp.vec3(
                wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z)
            )
            wa = wp.vec3(1.5) - fx
            wb = fx - wp.vec3(1.0)
            wc = fx - wp.vec3(0.5)
            w = wp.mat33(
                wp.cw_mul(wa, wa) * 0.5,
                wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
                wp.cw_mul(wc, wc) * 0.5,
            )
            dw = wp.mat33(
                fx - wp.vec3(1.5), -2.0 * (fx - wp.vec3(1.0)), fx - wp.vec3(0.5)
            )
            stress = trial_stress[p]
            for i in range(0, 3):
                for j in range(0, 3):
                    for k in range(0, 3):
                        ix = base_pos_x + i
                        iy = base_pos_y + j
                        iz = base_pos_z + k
                        dweight = compute_dweight(model, w, dw, i, j, k)
                        force = -state.particle_vol[p] * stress * dweight
                        wp.atomic_add(grid_force, ix, iy, iz, force)


@wp.kernel
def implicit_finalize_residual(
    state: MPMStateStruct,
    model: MPMModelStruct,
    grid_du: wp.array(dtype=wp.vec3, ndim=3),
    grid_vn: wp.array(dtype=wp.vec3, ndim=3),
    grid_an: wp.array(dtype=wp.vec3, ndim=3),
    grid_v_trial: wp.array(dtype=wp.vec3, ndim=3),
    grid_force: wp.array(dtype=wp.vec3, ndim=3),
    grid_residual: wp.array(dtype=wp.vec3, ndim=3),
    gamma: float,
    dt: float,
):
    grid_x, grid_y, grid_z = wp.tid()
    mass = state.grid_m[grid_x, grid_y, grid_z]
    if mass > 1e-15:
        v_n = grid_vn[grid_x, grid_y, grid_z]
        a_n = grid_an[grid_x, grid_y, grid_z]
        v_np1 = grid_v_trial[grid_x, grid_y, grid_z]
        a_np1 = (
            v_np1 - v_n - dt * (1.0 - gamma) * a_n
        ) * (1.0 / (gamma * dt))
        grid_residual[grid_x, grid_y, grid_z] = (
            grid_force[grid_x, grid_y, grid_z]
            + mass * model.gravitational_accelaration
            - mass * a_np1
        )
    else:
        grid_residual[grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def g2p_implicit(
    state: MPMStateStruct,
    model: MPMModelStruct,
    grid_du: wp.array(dtype=wp.vec3, ndim=3),
    dt: float,
):
    p = wp.tid()
    if state.particle_selection[p] == 0:
        grid_pos = state.particle_x[p] * model.inv_dx
        base_pos_x = wp.int(grid_pos[0] - 0.5)
        base_pos_y = wp.int(grid_pos[1] - 0.5)
        base_pos_z = wp.int(grid_pos[2] - 0.5)
        if not mpm_stencil_in_grid(base_pos_x, base_pos_y, base_pos_z, model):
            state.particle_x[p] = mpm_clamp_to_grid_domain(state.particle_x[p], model)
            state.particle_v[p] = wp.vec3(0.0, 0.0, 0.0)
            state.particle_C[p] = mpm_zero_mat33()
            grid_pos = state.particle_x[p] * model.inv_dx
            base_pos_x = wp.int(grid_pos[0] - 0.5)
            base_pos_y = wp.int(grid_pos[1] - 0.5)
            base_pos_z = wp.int(grid_pos[2] - 0.5)
        if mpm_stencil_in_grid(base_pos_x, base_pos_y, base_pos_z, model):
            fx = grid_pos - wp.vec3(
                wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z)
            )
            wa = wp.vec3(1.5) - fx
            wb = fx - wp.vec3(1.0)
            wc = fx - wp.vec3(0.5)
            w = wp.mat33(
                wp.cw_mul(wa, wa) * 0.5,
                wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
                wp.cw_mul(wc, wc) * 0.5,
            )
            dw = wp.mat33(fx - wp.vec3(1.5), -2.0 * (fx - wp.vec3(1.0)), fx - wp.vec3(0.5))
            new_v = wp.vec3(0.0, 0.0, 0.0)
            new_du = wp.vec3(0.0, 0.0, 0.0)
            new_C = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            grad_v = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            for i in range(0, 3):
                for j in range(0, 3):
                    for k in range(0, 3):
                        ix = base_pos_x + i
                        iy = base_pos_y + j
                        iz = base_pos_z + k
                        dpos = wp.vec3(wp.float(i), wp.float(j), wp.float(k)) - fx
                        weight = w[0, i] * w[1, j] * w[2, k]
                        grid_v = state.grid_v_out[ix, iy, iz]
                        grid_delta = grid_du[ix, iy, iz]
                        new_v = new_v + grid_v * weight
                        new_du = new_du + grid_delta * weight
                        new_C = new_C + wp.outer(grid_v, dpos) * (
                            weight * model.inv_dx * 4.0
                        )
                        dweight = compute_dweight(model, w, dw, i, j, k)
                        grad_v = grad_v + wp.outer(grid_v, dweight)

            new_x = state.particle_x[p] + new_du
            new_grid_pos = new_x * model.inv_dx
            new_base_pos_x = wp.int(new_grid_pos[0] - 0.5)
            new_base_pos_y = wp.int(new_grid_pos[1] - 0.5)
            new_base_pos_z = wp.int(new_grid_pos[2] - 0.5)
            if not mpm_stencil_in_grid(
                new_base_pos_x, new_base_pos_y, new_base_pos_z, model
            ):
                new_x = mpm_clamp_to_grid_domain(new_x, model)
                new_v = wp.vec3(0.0, 0.0, 0.0)
                new_du = wp.vec3(0.0, 0.0, 0.0)
                new_C = mpm_zero_mat33()
                grad_v = mpm_zero_mat33()
            state.particle_v[p] = new_v
            state.particle_x[p] = new_x
            state.particle_C[p] = new_C
            I33 = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
            state.particle_F_trial[p] = (I33 + grad_v * dt) * state.particle_F[p]

            if model.update_cov_with_F:
                update_cov(state, p, grad_v, dt)


@wp.kernel
def compute_cov_from_F(state: MPMStateStruct, model: MPMModelStruct):
    p = wp.tid()

    F = state.particle_F[p]

    init_cov = wp.mat33(0.0)
    init_cov[0, 0] = state.particle_init_cov[p * 6]
    init_cov[0, 1] = state.particle_init_cov[p * 6 + 1]
    init_cov[0, 2] = state.particle_init_cov[p * 6 + 2]
    init_cov[1, 0] = state.particle_init_cov[p * 6 + 1]
    init_cov[1, 1] = state.particle_init_cov[p * 6 + 3]
    init_cov[1, 2] = state.particle_init_cov[p * 6 + 4]
    init_cov[2, 0] = state.particle_init_cov[p * 6 + 2]
    init_cov[2, 1] = state.particle_init_cov[p * 6 + 4]
    init_cov[2, 2] = state.particle_init_cov[p * 6 + 5]

    cov = F * init_cov * wp.transpose(F)

    state.particle_cov[p * 6] = cov[0, 0]
    state.particle_cov[p * 6 + 1] = cov[0, 1]
    state.particle_cov[p * 6 + 2] = cov[0, 2]
    state.particle_cov[p * 6 + 3] = cov[1, 1]
    state.particle_cov[p * 6 + 4] = cov[1, 2]
    state.particle_cov[p * 6 + 5] = cov[2, 2]


@wp.kernel
def compute_R_from_F(state: MPMStateStruct, model: MPMModelStruct):
    p = wp.tid()

    F = state.particle_F[p]

    # polar svd decomposition
    U = wp.mat33(0.0)
    V = wp.mat33(0.0)
    sig = wp.vec3(0.0)
    wp.svd3(F, U, sig, V)

    if wp.determinant(U) < 0.0:
        U[0, 2] = -U[0, 2]
        U[1, 2] = -U[1, 2]
        U[2, 2] = -U[2, 2]

    if wp.determinant(V) < 0.0:
        V[0, 2] = -V[0, 2]
        V[1, 2] = -V[1, 2]
        V[2, 2] = -V[2, 2]

    # compute rotation matrix
    R = U * wp.transpose(V)
    state.particle_R[p] = wp.transpose(R)


@wp.kernel
def apply_additional_params(
    state: MPMStateStruct,
    model: MPMModelStruct,
    params_modifier: MaterialParamsModifier,
):
    p = wp.tid()
    pos = state.particle_x[p]
    if (
        pos[0] > params_modifier.point[0] - params_modifier.size[0]
        and pos[0] < params_modifier.point[0] + params_modifier.size[0]
        and pos[1] > params_modifier.point[1] - params_modifier.size[1]
        and pos[1] < params_modifier.point[1] + params_modifier.size[1]
        and pos[2] > params_modifier.point[2] - params_modifier.size[2]
        and pos[2] < params_modifier.point[2] + params_modifier.size[2]
    ):
        model.E[p] = params_modifier.E
        model.nu[p] = params_modifier.nu
        state.particle_density[p] = params_modifier.density
        if params_modifier.yield_stress >= 0.0:
            model.yield_stress[p] = params_modifier.yield_stress


@wp.kernel
def selection_add_impulse_on_particles(
    state: MPMStateStruct, impulse_modifier: Impulse_modifier
):
    p = wp.tid()
    offset = state.particle_x[p] - impulse_modifier.point
    if (
        wp.abs(offset[0]) < impulse_modifier.size[0]
        and wp.abs(offset[1]) < impulse_modifier.size[1]
        and wp.abs(offset[2]) < impulse_modifier.size[2]
    ):
        impulse_modifier.mask[p] = 1
    else:
        impulse_modifier.mask[p] = 0


@wp.kernel
def selection_enforce_particle_velocity_translation(
    state: MPMStateStruct, velocity_modifier: ParticleVelocityModifier
):
    p = wp.tid()
    offset = state.particle_x[p] - velocity_modifier.point
    if (
        wp.abs(offset[0]) < velocity_modifier.size[0]
        and wp.abs(offset[1]) < velocity_modifier.size[1]
        and wp.abs(offset[2]) < velocity_modifier.size[2]
    ):
        velocity_modifier.mask[p] = 1
    else:
        velocity_modifier.mask[p] = 0


@wp.kernel
def selection_enforce_particle_velocity_cylinder(
    state: MPMStateStruct, velocity_modifier: ParticleVelocityModifier
):
    p = wp.tid()
    offset = state.particle_x[p] - velocity_modifier.point

    vertical_distance = wp.abs(wp.dot(offset, velocity_modifier.normal))

    horizontal_distance = wp.length(
        offset - wp.dot(offset, velocity_modifier.normal) * velocity_modifier.normal
    )
    if (
        vertical_distance < velocity_modifier.half_height_and_radius[0]
        and horizontal_distance < velocity_modifier.half_height_and_radius[1]
    ):
        velocity_modifier.mask[p] = 1
    else:
        velocity_modifier.mask[p] = 0
