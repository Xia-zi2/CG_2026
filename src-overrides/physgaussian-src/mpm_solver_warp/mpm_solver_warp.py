import sys
import os
import math
import sys

import torch

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from engine_utils import *
from warp_utils import *
from mpm_utils import *


class MPM_Simulator_WARP:
    def __init__(self, n_particles, n_grid=50, grid_lim=1.0, device="cuda:0"):
        self.initialize(n_particles, n_grid, grid_lim, device=device)
        self.time_profile = {}

    def initialize(self, n_particles, n_grid=50, grid_lim=1.0, device="cuda:0"):
        self.n_particles = n_particles

        self.mpm_model = MPMModelStruct()
        # domain will be [0,grid_lim]*[0,grid_lim]*[0,grid_lim] !!!
        # domain will be [0,grid_lim]*[0,grid_lim]*[0,grid_lim] !!!
        # domain will be [0,grid_lim]*[0,grid_lim]*[0,grid_lim] !!!
        self.mpm_model.grid_lim = grid_lim
        self.mpm_model.n_particles = n_particles
        self.mpm_model.n_grid = n_grid
        self.mpm_model.grid_dim_x = self.mpm_model.n_grid
        self.mpm_model.grid_dim_y = self.mpm_model.n_grid
        self.mpm_model.grid_dim_z = self.mpm_model.n_grid
        (
            self.mpm_model.dx,
            self.mpm_model.inv_dx,
        ) = self.mpm_model.grid_lim / self.mpm_model.n_grid, float(
            self.mpm_model.n_grid / self.mpm_model.grid_lim
        )

        self.mpm_model.E = wp.zeros(shape=n_particles, dtype=float, device=device)
        self.mpm_model.nu = wp.zeros(shape=n_particles, dtype=float, device=device)
        self.mpm_model.mu = wp.zeros(shape=n_particles, dtype=float, device=device)
        self.mpm_model.lam = wp.zeros(shape=n_particles, dtype=float, device=device)

        self.mpm_model.update_cov_with_F = False
        self.mpm_model.pbmpm_elasticity_ratio = 0.75
        self.mpm_model.pbmpm_elastic_relaxation = 1.0
        self.mpm_model.pbmpm_plastic_mode = 0
        self.mpm_model.pbmpm_yield_min = 0.55
        self.mpm_model.pbmpm_yield_max = 1.85

        # material is used to switch between different elastoplastic models. 0 is jelly
        self.mpm_model.material = 0

        self.mpm_model.plastic_viscosity = 0.0
        self.mpm_model.softening = 0.1
        self.mpm_model.yield_stress = wp.zeros(
            shape=n_particles, dtype=float, device=device
        )
        self.mpm_model.hardening = 0.0
        self.mpm_model.xi = 0.0
        self.mpm_model.friction_angle = 25.0
        sin_phi = wp.sin(self.mpm_model.friction_angle / 180.0 * 3.14159265)
        self.mpm_model.alpha = wp.sqrt(2.0 / 3.0) * 2.0 * sin_phi / (3.0 - sin_phi)

        self.mpm_model.gravitational_accelaration = wp.vec3(0.0, 0.0, 0.0)

        self.mpm_model.rpic_damping = 0.0  # 0.0 if no damping (apic). -1 if pic

        # Global grid velocity damping. 1.0 disables damping; APIC/RPIC remains
        # separately controlled by rpic_damping.
        self.mpm_model.grid_v_damping_scale = 1.0

        self.mpm_state = MPMStateStruct()

        self.mpm_state.particle_x = wp.empty(
            shape=n_particles, dtype=wp.vec3, device=device
        )  # current position

        self.mpm_state.particle_v = wp.zeros(
            shape=n_particles, dtype=wp.vec3, device=device
        )  # particle velocity

        self.mpm_state.particle_F = wp.zeros(
            shape=n_particles, dtype=wp.mat33, device=device
        )  # particle F elastic

        self.mpm_state.particle_R = wp.zeros(
            shape=n_particles, dtype=wp.mat33, device=device
        )  # particle R rotation

        self.mpm_state.particle_init_cov = wp.zeros(
            shape=n_particles * 6, dtype=float, device=device
        )  # initial covariance matrix

        self.mpm_state.particle_cov = wp.zeros(
            shape=n_particles * 6, dtype=float, device=device
        )  # current covariance matrix

        self.mpm_state.particle_F_trial = wp.zeros(
            shape=n_particles, dtype=wp.mat33, device=device
        )  # apply return mapping will yield

        self.mpm_state.particle_stress = wp.zeros(
            shape=n_particles, dtype=wp.mat33, device=device
        )

        self.mpm_state.particle_vol = wp.zeros(
            shape=n_particles, dtype=float, device=device
        )  # particle volume
        self.mpm_state.particle_mass = wp.zeros(
            shape=n_particles, dtype=float, device=device
        )  # particle mass
        self.mpm_state.particle_density = wp.zeros(
            shape=n_particles, dtype=float, device=device
        )
        self.mpm_state.particle_C = wp.zeros(
            shape=n_particles, dtype=wp.mat33, device=device
        )
        self.mpm_state.particle_D = wp.zeros(
            shape=n_particles, dtype=wp.mat33, device=device
        )
        self.mpm_state.particle_Jp = wp.zeros(
            shape=n_particles, dtype=float, device=device
        )

        self.mpm_state.particle_selection = wp.zeros(
            shape=n_particles, dtype=int, device=device
        )

        self.mpm_state.grid_m = wp.zeros(
            shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
            dtype=float,
            device=device,
        )
        self.mpm_state.grid_v_in = wp.zeros(
            shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
            dtype=wp.vec3,
            device=device,
        )
        self.mpm_state.grid_v_out = wp.zeros(
            shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
            dtype=wp.vec3,
            device=device,
        )
        self._implicit_grid_an = wp.zeros(
            shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
            dtype=wp.vec3,
            device=device,
        )
        self._implicit_grid_dirichlet = wp.zeros(
            shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
            dtype=int,
            device=device,
        )
        self._implicit_grid_v_target = wp.zeros(
            shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
            dtype=wp.vec3,
            device=device,
        )
        self._implicit_grid_v_trial = wp.zeros(
            shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
            dtype=wp.vec3,
            device=device,
        )

        self.time = 0.0
        self.implicit_history = []
        self.solver_history = []

        self.grid_postprocess = []
        self.grid_postprocess_sources = []
        self.collider_params = []
        self.modify_bc = []
        self.implicit_grid_constraint_builders = []
        self.implicit_grid_constraint_params = []
        self.implicit_grid_constraint_sources = []
        self._last_boundary_projection_count = 0

        self.tailored_struct_for_bc = MPMtailoredStruct()
        self.pre_p2g_operations = []
        self.impulse_params = []

        self.particle_velocity_modifiers = []
        self.particle_velocity_modifier_params = []
        self.fixed_particle_modifiers = []

    # the h5 file should store particle initial position and volume.
    def load_from_sampling(
        self, sampling_h5, n_grid=50, grid_lim=1.0, device="cuda:0"
    ):
        if not os.path.exists(sampling_h5):
            print("h5 file cannot be found at ", os.getcwd() + sampling_h5)
            exit()

        h5file = h5py.File(sampling_h5, "r")
        x, particle_volume = h5file["x"], h5file["particle_volume"]

        x = x[()].transpose()  # np vector of x # shape now is (n_particles, dim)

        self.dim, self.n_particles = x.shape[1], x.shape[0]

        self.initialize(self.n_particles, n_grid, grid_lim, device=device)

        print(
            "Sampling particles are loaded from h5 file. Simulator is re-initialized for the correct n_particles"
        )
        particle_volume = np.squeeze(particle_volume, 0)

        self.mpm_state.particle_x = wp.from_numpy(
            x, dtype=wp.vec3, device=device
        )  # initialize warp array from np

        # initial velocity is default to zero
        wp.launch(
            kernel=set_vec3_to_zero,
            dim=self.n_particles,
            inputs=[self.mpm_state.particle_v],
            device=device,
        )
        # initial velocity is default to zero

        # initial deformation gradient is set to identity
        wp.launch(
            kernel=set_mat33_to_identity,
            dim=self.n_particles,
            inputs=[self.mpm_state.particle_F],
            device=device,
        )
        wp.launch(
            kernel=set_mat33_to_identity,
            dim=self.n_particles,
            inputs=[self.mpm_state.particle_F_trial],
            device=device,
        )
        # initial deformation gradient is set to identity

        self.mpm_state.particle_vol = wp.from_numpy(
            particle_volume, dtype=float, device=device
        )

        print("Particles initialized from sampling file.")
        print("Total particles: ", self.n_particles)

    # shape of tensor_x is (n, 3); shape of tensor_volume is (n,)
    def load_initial_data_from_torch(
        self,
        tensor_x,
        tensor_volume,
        tensor_cov=None,
        n_grid=50,
        grid_lim=1.0,
        device="cuda:0",
    ):
        self.dim, self.n_particles = tensor_x.shape[1], tensor_x.shape[0]
        assert tensor_x.shape[0] == tensor_volume.shape[0]
        # assert tensor_x.shape[0] == tensor_cov.reshape(-1, 6).shape[0]
        self.initialize(self.n_particles, n_grid, grid_lim, device=device)

        self.import_particle_x_from_torch(tensor_x, device)
        self.mpm_state.particle_vol = wp.from_numpy(
            tensor_volume.detach().clone().cpu().numpy(), dtype=float, device=device
        )
        if tensor_cov is not None:
            self.mpm_state.particle_init_cov = wp.from_numpy(
                tensor_cov.reshape(-1).detach().clone().cpu().numpy(),
                dtype=float,
                device=device,
            )

            if self.mpm_model.update_cov_with_F:
                self.mpm_state.particle_cov = self.mpm_state.particle_init_cov

        # initial velocity is default to zero
        wp.launch(
            kernel=set_vec3_to_zero,
            dim=self.n_particles,
            inputs=[self.mpm_state.particle_v],
            device=device,
        )
        # initial velocity is default to zero

        # initial deformation gradient is set to identity
        wp.launch(
            kernel=set_mat33_to_identity,
            dim=self.n_particles,
            inputs=[self.mpm_state.particle_F],
            device=device,
        )
        wp.launch(
            kernel=set_mat33_to_identity,
            dim=self.n_particles,
            inputs=[self.mpm_state.particle_F_trial],
            device=device,
        )
        # initial trial deformation gradient is set to identity

        print("Particles initialized from torch data.")
        print("Total particles: ", self.n_particles)

    # must give density. mass will be updated as density * volume
    def set_parameters(self, device="cuda:0", **kwargs):
        self.set_parameters_dict(kwargs, device=device)

    def set_parameters_dict(self, kwargs={}, device="cuda:0"):
        if "material" in kwargs:
            if kwargs["material"] == "jelly":
                self.mpm_model.material = 0
            elif kwargs["material"] == "metal":
                self.mpm_model.material = 1
            elif kwargs["material"] == "sand":
                self.mpm_model.material = 2
            elif kwargs["material"] == "foam":
                self.mpm_model.material = 3
            elif kwargs["material"] == "snow":
                self.mpm_model.material = 4
            elif kwargs["material"] == "plasticine":
                self.mpm_model.material = 5
            else:
                raise TypeError("Undefined material type")

        if "grid_lim" in kwargs:
            self.mpm_model.grid_lim = kwargs["grid_lim"]
        if "n_grid" in kwargs:
            self.mpm_model.n_grid = kwargs["n_grid"]
        self.mpm_model.grid_dim_x = self.mpm_model.n_grid
        self.mpm_model.grid_dim_y = self.mpm_model.n_grid
        self.mpm_model.grid_dim_z = self.mpm_model.n_grid
        (
            self.mpm_model.dx,
            self.mpm_model.inv_dx,
        ) = self.mpm_model.grid_lim / self.mpm_model.n_grid, float(
            self.mpm_model.n_grid / self.mpm_model.grid_lim
        )
        self.mpm_state.grid_m = wp.zeros(
            shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
            dtype=float,
            device=device,
        )
        self.mpm_state.grid_v_in = wp.zeros(
            shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
            dtype=wp.vec3,
            device=device,
        )
        self.mpm_state.grid_v_out = wp.zeros(
            shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
            dtype=wp.vec3,
            device=device,
        )
        self._implicit_grid_an = wp.zeros(
            shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
            dtype=wp.vec3,
            device=device,
        )
        self._implicit_grid_dirichlet = wp.zeros(
            shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
            dtype=int,
            device=device,
        )
        self._implicit_grid_v_target = wp.zeros(
            shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
            dtype=wp.vec3,
            device=device,
        )
        self._implicit_grid_v_trial = wp.zeros(
            shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
            dtype=wp.vec3,
            device=device,
        )

        if "E" in kwargs:
            wp.launch(
                kernel=set_value_to_float_array,
                dim=self.n_particles,
                inputs=[self.mpm_model.E, kwargs["E"]],
                device=device,
            )
        if "nu" in kwargs:
            wp.launch(
                kernel=set_value_to_float_array,
                dim=self.n_particles,
                inputs=[self.mpm_model.nu, kwargs["nu"]],
                device=device,
            )
        if "yield_stress" in kwargs:
            val = kwargs["yield_stress"]
            wp.launch(
                kernel=set_value_to_float_array,
                dim=self.n_particles,
                inputs=[self.mpm_model.yield_stress, val],
                device=device,
            )
        if "hardening" in kwargs:
            self.mpm_model.hardening = kwargs["hardening"]
        if "xi" in kwargs:
            self.mpm_model.xi = kwargs["xi"]
        if "friction_angle" in kwargs:
            self.mpm_model.friction_angle = kwargs["friction_angle"]
            sin_phi = wp.sin(self.mpm_model.friction_angle / 180.0 * 3.14159265)
            self.mpm_model.alpha = wp.sqrt(2.0 / 3.0) * 2.0 * sin_phi / (3.0 - sin_phi)

        if "g" in kwargs:
            self.mpm_model.gravitational_accelaration = wp.vec3(
                kwargs["g"][0], kwargs["g"][1], kwargs["g"][2]
            )

        if "density" in kwargs:
            density_value = kwargs["density"]
            wp.launch(
                kernel=set_value_to_float_array,
                dim=self.n_particles,
                inputs=[self.mpm_state.particle_density, density_value],
                device=device,
            )
            wp.launch(
                kernel=get_float_array_product,
                dim=self.n_particles,
                inputs=[
                    self.mpm_state.particle_density,
                    self.mpm_state.particle_vol,
                    self.mpm_state.particle_mass,
                ],
                device=device,
            )
        if "rpic_damping" in kwargs:
            self.mpm_model.rpic_damping = kwargs["rpic_damping"]
        if "plastic_viscosity" in kwargs:
            self.mpm_model.plastic_viscosity = kwargs["plastic_viscosity"]
        if "softening" in kwargs:
            self.mpm_model.softening = kwargs["softening"]
        if "grid_v_damping_scale" in kwargs:
            self.mpm_model.grid_v_damping_scale = max(
                0.0, min(float(kwargs["grid_v_damping_scale"]), 1.0)
            )
        if "additional_material_params" in kwargs:
            for params in kwargs["additional_material_params"]:
                param_modifier = MaterialParamsModifier()
                param_modifier.point = wp.vec3(params["point"])
                param_modifier.size = wp.vec3(params["size"])
                param_modifier.density = params["density"]
                param_modifier.E = params["E"]
                param_modifier.nu = params["nu"]
                param_modifier.yield_stress = float(params.get("yield_stress", -1.0))
                wp.launch(
                    kernel=apply_additional_params,
                    dim=self.n_particles,
                    inputs=[self.mpm_state, self.mpm_model, param_modifier],
                    device=device,
                )

            wp.launch(
                kernel=get_float_array_product,
                dim=self.n_particles,
                inputs=[
                    self.mpm_state.particle_density,
                    self.mpm_state.particle_vol,
                    self.mpm_state.particle_mass,
                ],
                device=device,
            )

    def set_particle_material_by_indices(
        self, indices, E=None, nu=None, density=None, yield_stress=None, device="cuda:0"
    ):
        if not torch.is_tensor(indices):
            indices = torch.tensor(indices, dtype=torch.long, device=device)
        else:
            indices = indices.to(device=device, dtype=torch.long)
        if indices.numel() == 0:
            return

        valid = torch.logical_and(indices >= 0, indices < self.n_particles)
        indices = indices[valid]
        if indices.numel() == 0:
            return

        if E is not None:
            wp.to_torch(self.mpm_model.E)[indices] = float(E)
        if nu is not None:
            wp.to_torch(self.mpm_model.nu)[indices] = float(nu)
        if yield_stress is not None:
            wp.to_torch(self.mpm_model.yield_stress)[indices] = float(yield_stress)
        if density is not None:
            wp.to_torch(self.mpm_state.particle_density)[indices] = float(density)
            wp.launch(
                kernel=get_float_array_product,
                dim=self.n_particles,
                inputs=[
                    self.mpm_state.particle_density,
                    self.mpm_state.particle_vol,
                    self.mpm_state.particle_mass,
                ],
                device=device,
            )

    def fix_particles_by_indices(
        self,
        indices,
        velocity=None,
        start_time=0.0,
        end_time=1e3,
        reset_deformation=1,
        device="cuda:0",
    ):
        if velocity is None:
            velocity = [0.0, 0.0, 0.0]
        if not torch.is_tensor(indices):
            indices = torch.tensor(indices, dtype=torch.long, device=device)
        else:
            indices = indices.to(device=device, dtype=torch.long)
        if indices.numel() == 0:
            return

        valid = torch.logical_and(indices >= 0, indices < self.n_particles)
        indices = indices[valid]
        if indices.numel() == 0:
            return

        rest_x = wp.to_torch(self.mpm_state.particle_x)[indices].contiguous()
        modifier = FixedParticleModifier()
        modifier.indices = torch2warp_int(indices.to(torch.int32).contiguous(), dvc=device)
        modifier.rest_x = torch2warp_vec3(rest_x, dvc=device)
        modifier.velocity = wp.vec3(
            float(velocity[0]), float(velocity[1]), float(velocity[2])
        )
        modifier.start_time = float(start_time)
        modifier.end_time = float(end_time)
        modifier.reset_deformation = int(reset_deformation)
        self.fixed_particle_modifiers.append((modifier, int(indices.numel())))

    def _apply_fixed_particles(self, time, device="cuda:0"):
        for modifier, count in self.fixed_particle_modifiers:
            wp.launch(
                kernel=apply_fixed_particle_indices,
                dim=count,
                inputs=[float(time), self.mpm_state, self.mpm_model, modifier],
                device=device,
            )

    def finalize_mu_lam(self, device="cuda:0"):
        wp.launch(
            kernel=compute_mu_lam_from_E_nu,
            dim=self.n_particles,
            inputs=[self.mpm_state, self.mpm_model],
            device=device,
        )

    def _apply_grid_velocity_damping(self, grid_size, device="cuda:0"):
        damping_scale = float(self.mpm_model.grid_v_damping_scale)
        if damping_scale < 1.0:
            wp.launch(
                kernel=add_damping_via_grid,
                dim=grid_size,
                inputs=[self.mpm_state, damping_scale],
                device=device,
            )

    def _explicit_stability_diagnostics(self, dt, include_stress=False):
        diagnostics = {
            "explicit_cfl_limit": 1.0,
            "explicit_velocity_cfl": None,
            "explicit_material_cfl": None,
            "explicit_max_wave_speed": None,
            "explicit_max_particle_speed": None,
            "explicit_max_F_norm": None,
            "explicit_max_stress_norm": None,
            "explicit_min_density": None,
            "explicit_max_E": None,
            "explicit_max_nu": None,
            "explicit_nonfinite_count": 0,
            "explicit_nonfinite_fields": {},
            "out_of_bounds_count": 0,
            "particle_clamp_count": 0,
        }
        if self.n_particles <= 0:
            return diagnostics

        selection = wp.to_torch(self.mpm_state.particle_selection)
        active = selection == 0
        if not bool(active.any().detach().cpu()):
            return diagnostics

        dx = max(float(self.mpm_model.dx), 1e-12)
        diagnostics["out_of_bounds_count"] = self._estimate_pre_solve_out_of_bounds_count()
        diagnostics["particle_clamp_count"] = self._estimate_particle_clamp_count()

        def active_values(warp_array):
            return wp.to_torch(warp_array)[active]

        def finite_abs_max(values):
            finite = values[torch.isfinite(values)]
            if finite.numel() == 0:
                return None
            return float(torch.max(torch.abs(finite)).detach().cpu())

        def count_nonfinite(name, values):
            count = int((~torch.isfinite(values)).sum().detach().cpu())
            if count:
                diagnostics["explicit_nonfinite_fields"][name] = count
                diagnostics["explicit_nonfinite_count"] += count

        particle_x = active_values(self.mpm_state.particle_x)
        particle_v = active_values(self.mpm_state.particle_v)
        particle_C = active_values(self.mpm_state.particle_C)
        particle_F = active_values(self.mpm_state.particle_F)
        particle_F_trial = active_values(self.mpm_state.particle_F_trial)
        particle_density = active_values(self.mpm_state.particle_density)
        particle_mass = active_values(self.mpm_state.particle_mass)
        particle_vol = active_values(self.mpm_state.particle_vol)
        E = active_values(self.mpm_model.E)
        nu = active_values(self.mpm_model.nu)

        for name, values in [
            ("particle_x", particle_x),
            ("particle_v", particle_v),
            ("particle_C", particle_C),
            ("particle_F", particle_F),
            ("particle_F_trial", particle_F_trial),
            ("particle_density", particle_density),
            ("particle_mass", particle_mass),
            ("particle_vol", particle_vol),
            ("E", E),
            ("nu", nu),
        ]:
            count_nonfinite(name, values)

        if particle_v.numel():
            speed = torch.sqrt(torch.sum(particle_v * particle_v, dim=1))
            finite_speed = speed[torch.isfinite(speed)]
            if finite_speed.numel():
                max_speed = float(torch.max(finite_speed).detach().cpu())
                diagnostics["explicit_max_particle_speed"] = max_speed
                diagnostics["explicit_velocity_cfl"] = float(float(dt) * max_speed / dx)
        if particle_F.numel():
            F_norm = torch.sqrt(torch.sum(particle_F * particle_F, dim=(1, 2)))
            finite_F_norm = F_norm[torch.isfinite(F_norm)]
            if finite_F_norm.numel():
                diagnostics["explicit_max_F_norm"] = float(
                    torch.max(finite_F_norm).detach().cpu()
                )
        if include_stress:
            particle_stress = active_values(self.mpm_state.particle_stress)
            count_nonfinite("particle_stress", particle_stress)
            if particle_stress.numel():
                stress_norm = torch.sqrt(
                    torch.sum(particle_stress * particle_stress, dim=(1, 2))
                )
                finite_stress_norm = stress_norm[torch.isfinite(stress_norm)]
                if finite_stress_norm.numel():
                    diagnostics["explicit_max_stress_norm"] = float(
                        torch.max(finite_stress_norm).detach().cpu()
                    )

        material_finite = (
            torch.isfinite(E)
            & torch.isfinite(nu)
            & torch.isfinite(particle_density)
            & (E > 0.0)
            & (particle_density > 0.0)
        )
        if bool(material_finite.any().detach().cpu()):
            E_valid = E[material_finite]
            nu_valid = torch.clamp(nu[material_finite], -0.99, 0.49)
            density_valid = torch.clamp(particle_density[material_finite], min=1e-12)
            mu = E_valid / (2.0 * (1.0 + nu_valid))
            K = E_valid / torch.clamp(3.0 * (1.0 - 2.0 * nu_valid), min=1e-6)
            wave_speed = torch.sqrt(
                torch.clamp((K + 4.0 / 3.0 * mu) / density_valid, min=0.0)
            )
            finite_wave = wave_speed[torch.isfinite(wave_speed)]
            if finite_wave.numel():
                max_wave = float(torch.max(finite_wave).detach().cpu())
                diagnostics["explicit_max_wave_speed"] = max_wave
                diagnostics["explicit_material_cfl"] = float(float(dt) * max_wave / dx)
            diagnostics["explicit_min_density"] = float(
                torch.min(density_valid).detach().cpu()
            )
            diagnostics["explicit_max_E"] = float(torch.max(E_valid).detach().cpu())
            diagnostics["explicit_max_nu"] = float(torch.max(nu_valid).detach().cpu())

        diagnostics["explicit_max_particle_abs"] = finite_abs_max(particle_x)
        return diagnostics

    def _explicit_failure_reason_from_diagnostics(self, diagnostics):
        if int(diagnostics.get("explicit_nonfinite_count", 0) or 0) > 0:
            return "explicit_nonfinite_state"
        material_cfl = diagnostics.get("explicit_material_cfl")
        cfl_limit = float(diagnostics.get("explicit_cfl_limit", 1.0) or 1.0)
        if material_cfl is not None and float(material_cfl) > cfl_limit:
            return "explicit_cfl_limit_exceeded"
        max_stress = diagnostics.get("explicit_max_stress_norm")
        if max_stress is not None and float(max_stress) > 1.0e18:
            return "explicit_stress_explosion"
        max_F = diagnostics.get("explicit_max_F_norm")
        if max_F is not None and float(max_F) > 1.0e8:
            return "explicit_deformation_gradient_explosion"
        velocity_cfl = diagnostics.get("explicit_velocity_cfl")
        if velocity_cfl is not None and float(velocity_cfl) > 32.0:
            return "explicit_velocity_cfl_exceeded"
        return None

    def _check_explicit_stability_or_raise(
        self, step, dt, stage, include_stress=False
    ):
        diagnostics = self._explicit_stability_diagnostics(
            dt, include_stress=include_stress
        )
        failure_reason = self._explicit_failure_reason_from_diagnostics(diagnostics)
        if failure_reason is None:
            return diagnostics

        failure = {
            "integrator": "explicit",
            "step": int(step),
            "dt": float(dt),
            "time": float(self.time),
            "converged": False,
            "committed": False,
            "substep_failed": True,
            "stability_failure": True,
            "failure_reason": failure_reason,
            "failure_stage": stage,
            **diagnostics,
        }
        self.solver_history.append(failure)
        raise RuntimeError(
            "Explicit MPM stability guard rejected substep "
            f"step={step}, dt={dt:.6g}, stage={stage}, "
            f"reason={failure_reason}, "
            f"material_cfl={diagnostics.get('explicit_material_cfl')}, "
            f"limit={diagnostics.get('explicit_cfl_limit')}."
        )

    def p2g2p(self, step, dt, device="cuda:0"):
        grid_size = (
            self.mpm_model.grid_dim_x,
            self.mpm_model.grid_dim_y,
            self.mpm_model.grid_dim_z,
        )
        wp.launch(
            kernel=zero_grid,
            dim=(grid_size),
            inputs=[self.mpm_state, self.mpm_model],
            device=device,
        )
        self._apply_fixed_particles(self.time, device=device)

        # apply pre-p2g operations on particles
        for k in range(len(self.pre_p2g_operations)):
            wp.launch(
                kernel=self.pre_p2g_operations[k],
                dim=self.n_particles,
                inputs=[self.time, dt, self.mpm_state, self.impulse_params[k]],
                device=device,
            )
        # apply dirichlet particle v modifier
        for k in range(len(self.particle_velocity_modifiers)):
            wp.launch(
                kernel=self.particle_velocity_modifiers[k],
                dim=self.n_particles,
                inputs=[
                    self.time,
                    self.mpm_state,
                    self.particle_velocity_modifier_params[k],
                ],
                device=device,
            )
        self._apply_fixed_particles(self.time, device=device)
        stability_diagnostics = self._check_explicit_stability_or_raise(
            step, dt, "pre_compute_stress", include_stress=False
        )
        # compute stress = stress(returnMap(F_trial))
        with wp.ScopedTimer(
            "compute_stress_from_F_trial",
            synchronize=True,
            print=False,
            dict=self.time_profile,
        ):
            wp.launch(
                kernel=compute_stress_from_F_trial,
                dim=self.n_particles,
                inputs=[self.mpm_state, self.mpm_model, dt],
                device=device,
            )  # F and stress are updated
        stability_diagnostics = self._check_explicit_stability_or_raise(
            step, dt, "post_compute_stress", include_stress=True
        )

        # p2g
        with wp.ScopedTimer(
            "p2g",
            synchronize=True,
            print=False,
            dict=self.time_profile,
        ):
            wp.launch(
                kernel=p2g_apic_with_stress,
                dim=self.n_particles,
                inputs=[self.mpm_state, self.mpm_model, dt],
                device=device,
            )  # apply p2g'

        # grid update
        with wp.ScopedTimer(
            "grid_update", synchronize=True, print=False, dict=self.time_profile
        ):
            wp.launch(
                kernel=grid_normalization_and_gravity,
                dim=(grid_size),
                inputs=[self.mpm_state, self.mpm_model, dt],
                device=device,
            )
        self._apply_grid_velocity_damping(grid_size, device=device)

        # apply BC on grid
        with wp.ScopedTimer(
            "apply_BC_on_grid", synchronize=True, print=False, dict=self.time_profile
        ):
            for k in range(len(self.grid_postprocess)):
                wp.launch(
                    kernel=self.grid_postprocess[k],
                    dim=grid_size,
                    inputs=[
                        self.time,
                        dt,
                        self.mpm_state,
                        self.mpm_model,
                        self.collider_params[k],
                    ],
                    device=device,
                )
                if self.modify_bc[k] is not None:
                    self.modify_bc[k](self.time, dt, self.collider_params[k])

        # g2p
        with wp.ScopedTimer(
            "g2p", synchronize=True, print=False, dict=self.time_profile
        ):
            wp.launch(
                kernel=g2p,
                dim=self.n_particles,
                inputs=[self.mpm_state, self.mpm_model, dt],
                device=device,
            )  # x, v, C, F_trial are updated
        self._apply_fixed_particles(self.time + dt, device=device)
        #### CFL check ####
        # particle_v = self.mpm_state.particle_v.numpy()
        # if np.max(np.abs(particle_v)) > self.mpm_model.dx / dt:
        #     print("max particle v: ", np.max(np.abs(particle_v)))
        #     print("max allowed  v: ", self.mpm_model.dx / dt)
        #     print("does not allow v*dt>dx")
        #     input()
        #### CFL check ####
        self.time = self.time + dt
        self.solver_history.append(
            {
                "integrator": "explicit",
                "step": int(step),
                "dt": float(dt),
                "time": float(self.time),
                "converged": True,
                "stability_failure": False,
                **stability_diagnostics,
            }
        )

    def _auto_pbmpm_parameters(self, dt, strength_scale=1.0, n_min=3, n_max=25):
        selection = wp.to_torch(self.mpm_state.particle_selection)
        active_mask = selection == 0

        def mean_material_value(warp_array, fallback, lower=None, upper=None):
            values = wp.to_torch(warp_array)
            if bool(active_mask.any().detach().cpu()):
                values = values[active_mask]
            valid = torch.isfinite(values)
            if lower is not None:
                valid = valid & (values > lower)
            if upper is not None:
                valid = valid & (values < upper)
            values = values[valid]
            if values.numel() == 0:
                return float(fallback)
            return float(torch.mean(values).detach().cpu())

        base_E = mean_material_value(self.mpm_model.E, 1.0e5, lower=0.0)
        nu = mean_material_value(self.mpm_model.nu, 0.4, lower=-0.999, upper=0.499)
        density = mean_material_value(self.mpm_state.particle_density, 200.0, lower=0.0)
        nu = min(max(nu, 0.0), 0.49)
        density = max(density, 1e-6)
        strength_scale = max(float(1.0 if strength_scale is None else strength_scale), 1e-6)
        E_effective = base_E * strength_scale

        mu = E_effective / (2.0 * (1.0 + nu))
        K = E_effective / max(3.0 * (1.0 - 2.0 * nu), 1e-6)
        h = float(self.mpm_model.grid_lim) / max(int(self.mpm_model.n_grid), 1)
        dt_over_h = float(dt) / max(h, 1e-12)
        r_s = dt_over_h * math.sqrt(max(mu / density, 0.0))
        r_p = dt_over_h * math.sqrt(max((K + 4.0 / 3.0 * mu) / density, 0.0))
        r_E = dt_over_h * math.sqrt(max(E_effective / density, 0.0))

        gamma = 0.5
        n_min = max(1, int(3 if n_min is None else n_min))
        n_max = max(n_min, int(25 if n_max is None else n_max))
        s_s = 1.0 - math.exp(-gamma * r_s * r_s)
        s_p = 1.0 - math.exp(-gamma * r_p * r_p)
        s_E = 1.0 - math.exp(-gamma * r_E * r_E)
        s_s = min(max(s_s, 0.0), 1.0)
        s_p = min(max(s_p, 0.0), 1.0)
        s_E = min(max(s_E, 0.0), 1.0)

        s_iter = max(s_s, s_p)
        iteration_count = n_min + math.floor((n_max - n_min) * s_iter)
        iteration_count = max(n_min, min(n_max, int(iteration_count)))

        alpha_min = 0.2
        alpha_max = 1.0
        elasticity_ratio = alpha_min + (alpha_max - alpha_min) * s_E
        elasticity_ratio = min(max(elasticity_ratio, 0.0), 1.0)

        omega_min = 0.5
        omega_max = 1.0
        elastic_relaxation = omega_min + (omega_max - omega_min) * s_iter
        elastic_relaxation = min(max(elastic_relaxation, omega_min), omega_max)
        return {
            "base_E": float(base_E),
            "E": float(base_E),
            "strength_scale": float(strength_scale),
            "E_effective": float(E_effective),
            "nu": float(nu),
            "density": float(density),
            "mu": float(mu),
            "K": float(K),
            "h": float(h),
            "dt": float(dt),
            "gamma": float(gamma),
            "r_s": float(r_s),
            "r_p": float(r_p),
            "r_E": float(r_E),
            "s_s": float(s_s),
            "s_p": float(s_p),
            "s_E": float(s_E),
            "s_iter": float(s_iter),
            "N_min": int(n_min),
            "N_max": int(n_max),
            "n_min": int(n_min),
            "n_max": int(n_max),
            "final_iteration_count": int(iteration_count),
            "elasticity_ratio": float(elasticity_ratio),
            "elastic_relaxation": float(elastic_relaxation),
            "iteration_count": int(iteration_count),
        }

    def p2g2p_pbmpm(
        self,
        step,
        dt,
        device="cuda:0",
        elasticity_ratio=None,
        elastic_relaxation=None,
        projection_iterations=None,
        r_scale=None,
        s_scale=None,
        iteration_count=None,
        strength_scale=1.0,
        n_min=3,
        n_max=25,
        plastic_mode=0,
        yield_min=0.55,
        yield_max=1.85,
    ):
        auto_params = self._auto_pbmpm_parameters(
            dt,
            strength_scale=strength_scale,
            n_min=n_min,
            n_max=n_max,
        )
        ignored_iteration_controls = (
            iteration_count is not None or projection_iterations is not None
        )
        ignored_elasticity_controls = (
            elasticity_ratio is not None
            or r_scale is not None
            or elastic_relaxation is not None
            or s_scale is not None
        )
        auto_used = {
            "elasticity_ratio": True,
            "elastic_relaxation": True,
            "iteration_count": True,
        }
        elasticity_ratio = auto_params["elasticity_ratio"]
        elastic_relaxation = auto_params["elastic_relaxation"]
        iteration_count = auto_params["iteration_count"]
        self.mpm_model.pbmpm_elasticity_ratio = float(elasticity_ratio)
        self.mpm_model.pbmpm_elastic_relaxation = float(elastic_relaxation)
        self.mpm_model.pbmpm_plastic_mode = 1 if int(plastic_mode or 0) == 1 else 0
        self.mpm_model.pbmpm_yield_min = float(0.55 if yield_min is None else yield_min)
        self.mpm_model.pbmpm_yield_max = float(1.85 if yield_max is None else yield_max)

        debug_signature = (
            round(auto_params["E_effective"], 6),
            round(auto_params["nu"], 6),
            round(auto_params["density"], 6),
            round(float(dt), 12),
            round(float(elasticity_ratio), 8),
            round(float(elastic_relaxation), 8),
            int(iteration_count),
            int(auto_params["N_min"]),
            int(auto_params["N_max"]),
            int(self.mpm_model.pbmpm_plastic_mode),
            round(float(self.mpm_model.pbmpm_yield_min), 8),
            round(float(self.mpm_model.pbmpm_yield_max), 8),
        )
        if getattr(self, "_last_pbmpm_debug_signature", None) != debug_signature:
            print(
                "[PBMPM] params "
                f"elasticity_ratio={float(elasticity_ratio):.6g}, "
                f"elastic_relaxation={float(elastic_relaxation):.6g}, "
                f"iteration_count={int(iteration_count)}, "
                f"final_iteration_count={int(iteration_count)}, "
                f"N=[{auto_params['N_min']}, {auto_params['N_max']}], "
                f"auto_used={auto_used}; "
                f"E={auto_params['E']:.6g}, "
                f"E_effective={auto_params['E_effective']:.6g} "
                f"(strength_scale={auto_params['strength_scale']:.6g}), "
                f"nu={auto_params['nu']:.6g}, "
                f"density={auto_params['density']:.6g}, "
                f"mu={auto_params['mu']:.6g}, K={auto_params['K']:.6g}, "
                f"h={auto_params['h']:.6g}, "
                f"dt={auto_params['dt']:.6g}, r_s={auto_params['r_s']:.6g}, "
                f"r_p={auto_params['r_p']:.6g}, r_E={auto_params['r_E']:.6g}, "
                f"s_s={auto_params['s_s']:.6g}, s_p={auto_params['s_p']:.6g}, "
                f"s_E={auto_params['s_E']:.6g}, gamma={auto_params['gamma']:.6g}, "
                f"plastic_mode={int(self.mpm_model.pbmpm_plastic_mode)}, "
                f"yield_min={self.mpm_model.pbmpm_yield_min:.6g}, "
                f"yield_max={self.mpm_model.pbmpm_yield_max:.6g}; "
                "yield_min/yield_max are PBMPM stretch clamp bounds, not material yield stress."
            )
            self._last_pbmpm_debug_signature = debug_signature

        grid_size = (
            self.mpm_model.grid_dim_x,
            self.mpm_model.grid_dim_y,
            self.mpm_model.grid_dim_z,
        )
        self._apply_fixed_particles(self.time, device=device)
        for k in range(len(self.pre_p2g_operations)):
            wp.launch(
                kernel=self.pre_p2g_operations[k],
                dim=self.n_particles,
                inputs=[self.time, dt, self.mpm_state, self.impulse_params[k]],
                device=device,
            )
        for k in range(len(self.particle_velocity_modifiers)):
            wp.launch(
                kernel=self.particle_velocity_modifiers[k],
                dim=self.n_particles,
                inputs=[
                    self.time,
                    self.mpm_state,
                    self.particle_velocity_modifier_params[k],
                ],
                device=device,
            )
        self._apply_fixed_particles(self.time, device=device)
        iteration_count = max(1, int(iteration_count))
        wp.launch(
            kernel=pbmpm_clear_D,
            dim=self.n_particles,
            inputs=[self.mpm_state],
            device=device,
        )
        for iteration in range(iteration_count):
            with wp.ScopedTimer(
                "pbmpm_project_constraints",
                synchronize=True,
                print=False,
                dict=self.time_profile,
            ):
                wp.launch(
                    kernel=pbmpm_project_constraints,
                    dim=self.n_particles,
                    inputs=[self.mpm_state, self.mpm_model, dt],
                    device=device,
                )
            self._apply_fixed_particles(self.time, device=device)

            wp.launch(
                kernel=zero_grid,
                dim=(grid_size),
                inputs=[self.mpm_state, self.mpm_model],
                device=device,
            )

            with wp.ScopedTimer(
                "pbmpm_p2g_velocity",
                synchronize=True,
                print=False,
                dict=self.time_profile,
            ):
                wp.launch(
                    kernel=p2g_pbmpm_with_D,
                    dim=self.n_particles,
                    inputs=[self.mpm_state, self.mpm_model, dt],
                    device=device,
                )

            with wp.ScopedTimer(
                "pbmpm_grid_update",
                synchronize=True,
                print=False,
                dict=self.time_profile,
            ):
                wp.launch(
                    kernel=grid_normalization_no_gravity,
                    dim=(grid_size),
                    inputs=[self.mpm_state, self.mpm_model],
                    device=device,
                )
            self._apply_grid_velocity_damping(grid_size, device=device)

            with wp.ScopedTimer(
                "pbmpm_apply_BC_on_grid",
                synchronize=True,
                print=False,
                dict=self.time_profile,
            ):
                for k in range(len(self.grid_postprocess)):
                    wp.launch(
                        kernel=self.grid_postprocess[k],
                        dim=grid_size,
                        inputs=[
                            self.time,
                            dt,
                            self.mpm_state,
                            self.mpm_model,
                            self.collider_params[k],
                        ],
                        device=device,
                    )
                    if self.modify_bc[k] is not None:
                        self.modify_bc[k](self.time, dt, self.collider_params[k])

            with wp.ScopedTimer(
                "pbmpm_g2p_velocity",
                synchronize=True,
                print=False,
                dict=self.time_profile,
            ):
                wp.launch(
                    kernel=g2p_pbmpm_update_D,
                    dim=self.n_particles,
                    inputs=[self.mpm_state, self.mpm_model, dt],
                    device=device,
                )
            self._apply_fixed_particles(self.time, device=device)

        wp.launch(
            kernel=pbmpm_integrate_particles,
            dim=self.n_particles,
            inputs=[self.mpm_state, self.mpm_model, dt],
            device=device,
        )
        wp.launch(
            kernel=pbmpm_clear_D,
            dim=self.n_particles,
            inputs=[self.mpm_state],
            device=device,
        )
        self._apply_fixed_particles(self.time + dt, device=device)
        self.time = self.time + dt
        self.solver_history.append(
            {
                "integrator": "pbmpm",
                "step": int(step),
                "dt": float(dt),
                "time": float(self.time),
                "converged": True,
                "pbmpm": {
                    "elasticity_ratio": float(elasticity_ratio),
                    "elastic_relaxation": float(elastic_relaxation),
                    "iteration_count": int(iteration_count),
                    "final_iteration_count": int(iteration_count),
                    "strength_scale": float(auto_params["strength_scale"]),
                    "E_effective": float(auto_params["E_effective"]),
                    "plastic_mode": int(self.mpm_model.pbmpm_plastic_mode),
                    "yield_min": float(self.mpm_model.pbmpm_yield_min),
                    "yield_max": float(self.mpm_model.pbmpm_yield_max),
                    "yield_bounds_note": "yield_min/yield_max are PBMPM stretch clamp bounds, not material yield stress.",
                    "auto_mapping": auto_params,
                    "auto_used": auto_used,
                    "gravity_pre_kick": False,
                    "gravity_post_integrate_kick": True,
                    "plasticity_model": "physgaussian_material_return_mapping",
                    "projection_return_mapping": "none",
                    "commit_return_mapping": (
                        "none" if self.mpm_model.pbmpm_plastic_mode == 1
                        else "commit_material_return_mapping"
                    ),
                    "yield_clamp_mode": (
                        "pbmpm_stretch_clamp"
                        if self.mpm_model.pbmpm_plastic_mode == 1
                        else "shared_material_return_mapping"
                    ),
                    "ignored_user_iteration_controls": bool(ignored_iteration_controls),
                    "ignored_user_elastic_controls": bool(ignored_elasticity_controls),
                },
            }
        )

    def _implicit_apply_pre_p2g(self, dt, device="cuda:0"):
        self._apply_fixed_particles(self.time, device=device)
        for k in range(len(self.pre_p2g_operations)):
            wp.launch(
                kernel=self.pre_p2g_operations[k],
                dim=self.n_particles,
                inputs=[self.time, dt, self.mpm_state, self.impulse_params[k]],
                device=device,
            )
        for k in range(len(self.particle_velocity_modifiers)):
            wp.launch(
                kernel=self.particle_velocity_modifiers[k],
                dim=self.n_particles,
                inputs=[
                    self.time,
                    self.mpm_state,
                    self.particle_velocity_modifier_params[k],
                ],
                device=device,
            )
        self._apply_fixed_particles(self.time, device=device)
    def _implicit_initialize_grid(self, device="cuda:0"):
        grid_size = (
            self.mpm_model.grid_dim_x,
            self.mpm_model.grid_dim_y,
            self.mpm_model.grid_dim_z,
        )
        wp.launch(
            kernel=zero_grid,
            dim=grid_size,
            inputs=[self.mpm_state, self.mpm_model],
            device=device,
        )
        wp.launch(
            kernel=p2g_apic_no_stress,
            dim=self.n_particles,
            inputs=[self.mpm_state, self.mpm_model],
            device=device,
        )
        wp.launch(
            kernel=grid_normalization_no_gravity,
            dim=grid_size,
            inputs=[self.mpm_state, self.mpm_model],
            device=device,
        )

    def _implicit_refresh_dirichlet_constraints(self, dt, device="cuda:0"):
        grid_size = (
            self.mpm_model.grid_dim_x,
            self.mpm_model.grid_dim_y,
            self.mpm_model.grid_dim_z,
        )
        source_stats = {
            "cuboid": 0,
            "surface": 0,
            "other": 0,
        }
        wp.launch(
            kernel=zero_grid_int,
            dim=grid_size,
            inputs=[self._implicit_grid_dirichlet],
            device=device,
        )
        wp.launch(
            kernel=zero_grid_vec3,
            dim=grid_size,
            inputs=[self._implicit_grid_v_target],
            device=device,
        )
        for k in range(len(self.implicit_grid_constraint_builders)):
            source_mask = wp.zeros(shape=grid_size, dtype=int, device=device)
            source_target = wp.zeros(shape=grid_size, dtype=wp.vec3, device=device)
            wp.launch(
                kernel=self.implicit_grid_constraint_builders[k],
                dim=grid_size,
                inputs=[
                    self.time,
                    dt,
                    self.mpm_state,
                    self.mpm_model,
                    self.implicit_grid_constraint_params[k],
                    source_mask,
                    source_target,
                ],
                device=device,
            )
            wp.launch(
                kernel=merge_dirichlet_grid,
                dim=grid_size,
                inputs=[
                    self._implicit_grid_dirichlet,
                    self._implicit_grid_v_target,
                    source_mask,
                    source_target,
                ],
                device=device,
            )
            source = "other"
            if k < len(self.implicit_grid_constraint_sources):
                source = self.implicit_grid_constraint_sources[k]
            source_count = int((wp.to_torch(source_mask) > 0).sum().detach().cpu())
            source_stats[source] = source_stats.get(source, 0) + source_count
        return source_stats

    def _estimate_bounding_box_projection_count(self):
        if "bounding_box" not in self.grid_postprocess_sources:
            return 0
        padding = 3
        grid_v = wp.to_torch(self.mpm_state.grid_v_out)
        grid_mass = wp.to_torch(self.mpm_state.grid_m) > 1e-15
        candidate = torch.zeros_like(grid_mass, dtype=torch.bool)
        candidate[:padding, :, :] |= grid_v[:padding, :, :, 0] < 0.0
        candidate[-padding:, :, :] |= grid_v[-padding:, :, :, 0] > 0.0
        candidate[:, :padding, :] |= grid_v[:, :padding, :, 1] < 0.0
        candidate[:, -padding:, :] |= grid_v[:, -padding:, :, 1] > 0.0
        candidate[:, :, :padding] |= grid_v[:, :, :padding, 2] < 0.0
        candidate[:, :, -padding:] |= grid_v[:, :, -padding:, 2] > 0.0
        return int((candidate & grid_mass).sum().detach().cpu())

    def _estimate_particle_clamp_count(self):
        if self.n_particles <= 0:
            return 0
        particle_x = wp.to_torch(self.mpm_state.particle_x)
        padding = 3.0 * self.mpm_model.dx
        upper = float(self.mpm_model.grid_lim) - padding
        eps = max(float(self.mpm_model.dx) * 1e-4, 1e-9)
        near_lower = particle_x <= (padding + eps)
        near_upper = particle_x >= (upper - eps)
        return int(torch.any(near_lower | near_upper, dim=1).sum().detach().cpu())

    def _estimate_pre_solve_out_of_bounds_count(self):
        if self.n_particles <= 0:
            return 0
        particle_x = wp.to_torch(self.mpm_state.particle_x)
        particle_selection = wp.to_torch(self.mpm_state.particle_selection)
        grid_pos = particle_x * float(self.mpm_model.inv_dx)
        base_pos = (grid_pos - 0.5).to(torch.int32)
        in_grid = (
            (base_pos[:, 0] >= 0)
            & (base_pos[:, 1] >= 0)
            & (base_pos[:, 2] >= 0)
            & (base_pos[:, 0] + 2 < int(self.mpm_model.grid_dim_x))
            & (base_pos[:, 1] + 2 < int(self.mpm_model.grid_dim_y))
            & (base_pos[:, 2] + 2 < int(self.mpm_model.grid_dim_z))
        )
        selected = particle_selection == 0
        return int((selected & (~in_grid)).sum().detach().cpu())

    def _snapshot_implicit_particle_state(self):
        return {
            "particle_x": wp.to_torch(self.mpm_state.particle_x).clone(),
            "particle_v": wp.to_torch(self.mpm_state.particle_v).clone(),
            "particle_C": wp.to_torch(self.mpm_state.particle_C).clone(),
            "particle_F": wp.to_torch(self.mpm_state.particle_F).clone(),
            "particle_F_trial": wp.to_torch(
                self.mpm_state.particle_F_trial
            ).clone(),
            "particle_stress": wp.to_torch(self.mpm_state.particle_stress).clone(),
            "particle_cov": wp.to_torch(self.mpm_state.particle_cov).clone(),
            "particle_R": wp.to_torch(self.mpm_state.particle_R).clone(),
            "particle_D": wp.to_torch(self.mpm_state.particle_D).clone(),
            "particle_Jp": wp.to_torch(self.mpm_state.particle_Jp).clone(),
            "yield_stress": wp.to_torch(self.mpm_model.yield_stress).clone(),
            "mu": wp.to_torch(self.mpm_model.mu).clone(),
            "lam": wp.to_torch(self.mpm_model.lam).clone(),
        }

    def _restore_implicit_particle_state(self, snapshot):
        wp.to_torch(self.mpm_state.particle_x).copy_(snapshot["particle_x"])
        wp.to_torch(self.mpm_state.particle_v).copy_(snapshot["particle_v"])
        wp.to_torch(self.mpm_state.particle_C).copy_(snapshot["particle_C"])
        wp.to_torch(self.mpm_state.particle_F).copy_(snapshot["particle_F"])
        wp.to_torch(self.mpm_state.particle_F_trial).copy_(
            snapshot["particle_F_trial"]
        )
        wp.to_torch(self.mpm_state.particle_stress).copy_(
            snapshot["particle_stress"]
        )
        wp.to_torch(self.mpm_state.particle_cov).copy_(snapshot["particle_cov"])
        wp.to_torch(self.mpm_state.particle_R).copy_(snapshot["particle_R"])
        wp.to_torch(self.mpm_state.particle_D).copy_(snapshot["particle_D"])
        wp.to_torch(self.mpm_state.particle_Jp).copy_(snapshot["particle_Jp"])
        wp.to_torch(self.mpm_model.yield_stress).copy_(snapshot["yield_stress"])
        wp.to_torch(self.mpm_model.mu).copy_(snapshot["mu"])
        wp.to_torch(self.mpm_model.lam).copy_(snapshot["lam"])

    def _implicit_residual(
        self,
        grid_du_wp,
        grid_vn_wp,
        grid_an_wp,
        grid_force_wp,
        grid_residual_wp,
        trial_F_wp,
        trial_stress_wp,
        beta,
        gamma,
        dt,
        device="cuda:0",
        count_boundary_projections=False,
    ):
        grid_size = (
            self.mpm_model.grid_dim_x,
            self.mpm_model.grid_dim_y,
            self.mpm_model.grid_dim_z,
        )
        wp.launch(
            kernel=implicit_du_to_velocity,
            dim=grid_size,
            inputs=[
                self.mpm_state,
                grid_du_wp,
                grid_vn_wp,
                grid_an_wp,
                self._implicit_grid_v_trial,
                beta,
                gamma,
                dt,
            ],
            device=device,
        )

        if count_boundary_projections:
            self._last_boundary_projection_count = 0

        with wp.ScopedTimer(
            "implicit_trial_stress",
            synchronize=True,
            print=False,
            dict=self.time_profile,
        ):
            wp.launch(
                kernel=implicit_compute_trial_stress,
                dim=self.n_particles,
                inputs=[
                    self.mpm_state,
                    self.mpm_model,
                    self._implicit_grid_v_trial,
                    trial_F_wp,
                    trial_stress_wp,
                    dt,
                ],
                device=device,
            )

        wp.launch(
            kernel=zero_grid_vec3,
            dim=grid_size,
            inputs=[grid_force_wp],
            device=device,
        )
        wp.launch(
            kernel=implicit_accumulate_internal_force,
            dim=self.n_particles,
            inputs=[self.mpm_state, self.mpm_model, trial_stress_wp, grid_force_wp],
            device=device,
        )
        wp.launch(
            kernel=implicit_finalize_residual,
            dim=grid_size,
            inputs=[
                self.mpm_state,
                self.mpm_model,
                grid_du_wp,
                grid_vn_wp,
                grid_an_wp,
                self._implicit_grid_v_trial,
                grid_force_wp,
                grid_residual_wp,
                gamma,
                dt,
            ],
            device=device,
        )
        return wp.to_torch(grid_residual_wp)

    @staticmethod
    def _gmres_matrix_free(matvec, rhs, tol=1e-3, max_iter=24, min_iter=1):
        b_norm = torch.linalg.norm(rhs)
        min_iter = int(max(1, min(int(min_iter), int(max_iter))))
        info = {
            "break_reason": "zero_rhs",
            "arnoldi_norm": 0.0,
            "h_diag": 0.0,
            "basis_size": 0,
            "tol": float(tol),
            "max_iter": int(max_iter),
            "min_iter": int(min_iter),
        }
        info["rhs_norm"] = float(b_norm.detach().cpu())
        if float(b_norm.detach().cpu()) < 1e-20:
            return torch.zeros_like(rhs), 0, 0.0, True, info

        def solve_upper_triangular(upper, target):
            if upper.numel() == 0:
                return target
            solution = torch.zeros_like(target)
            diag_floor = torch.tensor(1e-20, dtype=target.dtype, device=target.device)
            for row in range(target.shape[0] - 1, -1, -1):
                rhs_value = target[row]
                if row + 1 < target.shape[0]:
                    rhs_value = rhs_value - torch.dot(
                        upper[row, row + 1 :],
                        solution[row + 1 :],
                    )
                diag = upper[row, row]
                diag_safe = torch.where(
                    torch.abs(diag) > diag_floor,
                    diag,
                    torch.where(diag < 0.0, -diag_floor, diag_floor),
                )
                solution[row] = rhs_value / diag_safe
            return solution

        q_vectors = [rhs / b_norm]
        h = torch.zeros(
            (max_iter + 1, max_iter),
            dtype=rhs.dtype,
            device=rhs.device,
        )
        givens_c = torch.zeros(max_iter, dtype=rhs.dtype, device=rhs.device)
        givens_s = torch.zeros(max_iter, dtype=rhs.dtype, device=rhs.device)
        g = torch.zeros(max_iter + 1, dtype=rhs.dtype, device=rhs.device)
        g[0] = b_norm
        best_x = torch.zeros_like(rhs)
        best_rel = float("inf")
        best_actual_rel = float("inf")
        best_actual_abs = float("inf")
        used_iter = 0
        break_reason = "max_iter"
        arnoldi_norm_float = 0.0
        h_diag_float = 0.0

        for j in range(max_iter):
            v = matvec(q_vectors[j])
            for i in range(j + 1):
                hij = torch.dot(q_vectors[i], v)
                h[i, j] = hij
                v = v - hij * q_vectors[i]
            # A second modified Gram-Schmidt pass improves orthogonality for
            # nearly dependent Krylov vectors, matching the paper's GMRES note.
            for i in range(j + 1):
                correction = torch.dot(q_vectors[i], v)
                h[i, j] = h[i, j] + correction
                v = v - correction * q_vectors[i]

            h[j + 1, j] = torch.linalg.norm(v)
            # Keep a scalar snapshot before the Givens rotation zeros this
            # Hessenberg entry. A 0-d tensor view would make GMRES think every
            # Arnoldi step broke down after one iteration.
            arnoldi_norm = h[j + 1, j].clone()
            arnoldi_norm_float = float(torch.abs(arnoldi_norm).detach().cpu())
            if arnoldi_norm_float > 1e-20 and j + 1 < max_iter:
                q_vectors.append(v / h[j + 1, j])

            for i in range(j):
                temp = givens_c[i] * h[i, j] + givens_s[i] * h[i + 1, j]
                h[i + 1, j] = -givens_s[i] * h[i, j] + givens_c[i] * h[i + 1, j]
                h[i, j] = temp

            rotation_norm = torch.sqrt(h[j, j] * h[j, j] + h[j + 1, j] * h[j + 1, j])
            if float(rotation_norm.detach().cpu()) <= 1e-20:
                givens_c[j] = torch.ones((), dtype=rhs.dtype, device=rhs.device)
                givens_s[j] = torch.zeros((), dtype=rhs.dtype, device=rhs.device)
            else:
                givens_c[j] = h[j, j] / rotation_norm
                givens_s[j] = h[j + 1, j] / rotation_norm

            h[j, j] = givens_c[j] * h[j, j] + givens_s[j] * h[j + 1, j]
            h_diag_float = float(torch.abs(h[j, j]).detach().cpu())
            h[j + 1, j] = torch.zeros((), dtype=rhs.dtype, device=rhs.device)
            g_j = givens_c[j] * g[j] + givens_s[j] * g[j + 1]
            g[j + 1] = -givens_s[j] * g[j] + givens_c[j] * g[j + 1]
            g[j] = g_j

            solution = solve_upper_triangular(
                h[: j + 1, : j + 1],
                g[: j + 1],
            )
            q_stack = torch.stack(q_vectors[: j + 1], dim=1)
            best_x = q_stack @ solution
            best_rel = float((torch.abs(g[j + 1]) / b_norm).detach().cpu())
            actual_residual = rhs - matvec(best_x)
            actual_residual_norm = torch.linalg.norm(actual_residual)
            actual_rel = float((actual_residual_norm / b_norm).detach().cpu())
            best_actual_rel = actual_rel
            best_actual_abs = float(actual_residual_norm.detach().cpu())
            used_iter = j + 1
            if used_iter >= min_iter and best_rel <= tol and actual_rel <= tol:
                break_reason = "converged"
                info.update({
                    "break_reason": break_reason,
                    "arnoldi_norm": arnoldi_norm_float,
                    "h_diag": h_diag_float,
                    "basis_size": len(q_vectors),
                    "final_relative_residual": float(best_rel),
                    "actual_relative_residual": float(best_actual_rel),
                    "actual_residual": float(best_actual_abs),
                    "used_iter": int(used_iter),
                })
                return best_x, used_iter, best_rel, True, info
            if j + 1 >= max_iter:
                break_reason = "max_iter"
                break
            if len(q_vectors) <= j + 1:
                break_reason = "krylov_basis_exhausted"
                break
            if arnoldi_norm_float <= 1e-20:
                break_reason = "arnoldi_breakdown"
                break

        info.update({
            "break_reason": break_reason,
            "arnoldi_norm": arnoldi_norm_float,
            "h_diag": h_diag_float,
            "basis_size": len(q_vectors),
            "final_relative_residual": float(best_rel),
            "actual_relative_residual": float(best_actual_rel),
            "actual_residual": float(best_actual_abs),
            "used_iter": int(used_iter),
        })
        return best_x, used_iter, best_rel, used_iter >= min_iter and best_actual_rel <= tol, info

    def p2g2p_implicit(
        self,
        step,
        dt,
        device="cuda:0",
        beta=0.25,
        gamma=0.5,
        newton_tol=5e-4,
        newton_max_iter=16,
        newton_abs_tol=1e-6,
        newton_rms_tol=1e-4,
        gmres_tol=1e-3,
        gmres_tol_floor=1e-3,
        gmres_max_iter=24,
        jvp_eps=1e-4,
        line_search_max_iter=8,
        armijo_c1=1e-4,
        ew_eta_min=1e-3,
        ew_eta_max=0.1,
        ew_gamma=0.9,
        ew_alpha=1.5,
        stiffness_preconditioner_scale=1.0,
        stagnation_tol=1e-8,
        nonlinear_failure_relative=5e-2,
        nonlinear_failure_absolute=1.0,
        allow_best_effort_commit=False,
        near_converged_factor=2.0,
        near_newton_rms_tol=1e-4,
        fallback_descent_tol=1e-8,
        fallback_step_min_rel=1e-8,
        fallback_decrease_tol=1e-6,
    ):
        if gmres_tol_floor is None:
            gmres_tol_floor = 1e-3
        gmres_tol_floor = float(max(1e-12, max(ew_eta_min, gmres_tol_floor)))
        near_converged_factor = float(max(1.0, near_converged_factor))
        newton_rms_tol = float(max(0.0, newton_rms_tol))
        near_newton_rms_tol = float(max(0.0, near_newton_rms_tol))
        fallback_descent_tol = float(max(0.0, fallback_descent_tol))
        fallback_step_min_rel = float(max(0.0, fallback_step_min_rel))
        fallback_decrease_tol = float(min(max(0.0, fallback_decrease_tol), 0.999))
        grid_size = (
            self.mpm_model.grid_dim_x,
            self.mpm_model.grid_dim_y,
            self.mpm_model.grid_dim_z,
        )
        self._last_boundary_projection_count = 0

        def implicit_solver_settings():
            return {
                "beta": float(beta),
                "gamma": float(gamma),
                "newton_tol": float(newton_tol),
                "newton_abs_tol": float(newton_abs_tol),
                "newton_rms_tol": float(newton_rms_tol),
                "newton_max_iter": int(newton_max_iter),
                "gmres_tol": float(gmres_tol),
                "gmres_tol_legacy_deprecated": float(gmres_tol),
                "gmres_tol_floor": float(gmres_tol_floor),
                "gmres_forcing": "eisenstat_walker",
                "gmres_max_iter": int(gmres_max_iter),
                "jvp_eps": float(jvp_eps),
                "line_search_max_iter": int(line_search_max_iter),
                "armijo_c1": float(armijo_c1),
                "ew_eta_min": float(ew_eta_min),
                "ew_eta_max": float(ew_eta_max),
                "ew_gamma": float(ew_gamma),
                "ew_alpha": float(ew_alpha),
                "stiffness_preconditioner_scale": float(
                    stiffness_preconditioner_scale
                ),
                "stagnation_tol": float(stagnation_tol),
                "nonlinear_failure_relative": float(nonlinear_failure_relative),
                "nonlinear_failure_absolute": float(nonlinear_failure_absolute),
                "allow_best_effort_commit": bool(allow_best_effort_commit),
                "near_converged_factor": float(near_converged_factor),
                "near_newton_rms_tol": float(near_newton_rms_tol),
                "fallback_descent_tol": float(fallback_descent_tol),
                "fallback_step_min_rel": float(fallback_step_min_rel),
                "fallback_decrease_tol": float(fallback_decrease_tol),
            }

        solver_settings = implicit_solver_settings()

        particle_state_snapshot = self._snapshot_implicit_particle_state()
        self._implicit_apply_pre_p2g(dt, device=device)
        pre_solve_clamp_count = self._estimate_pre_solve_out_of_bounds_count()
        self._implicit_initialize_grid(device=device)
        dirichlet_source_stats = self._implicit_refresh_dirichlet_constraints(
            dt, device=device
        )

        grid_vn = wp.to_torch(self.mpm_state.grid_v_out).clone()
        grid_an = wp.to_torch(self._implicit_grid_an)
        grid_mass = wp.to_torch(self.mpm_state.grid_m)
        grid_mass_mask = grid_mass > 1e-15
        dirichlet_mask = wp.to_torch(self._implicit_grid_dirichlet) > 0
        active_mask = grid_mass_mask & (~dirichlet_mask)
        active_free_node_count = int(active_mask.sum().detach().cpu())
        active_dof = int(3 * active_free_node_count)
        active_dof_sqrt = math.sqrt(max(active_dof, 1))
        grid_mass_node_count = int(grid_mass_mask.sum().detach().cpu())
        dirichlet_total_count = int(dirichlet_mask.sum().detach().cpu())
        dirichlet_mass_node_count = int(
            (grid_mass_mask & dirichlet_mask).sum().detach().cpu()
        )
        grid_mass_active_values = grid_mass[grid_mass_mask]
        if bool(grid_mass_mask.any().detach().cpu()):
            mass_stats = {
                "min": float(torch.min(grid_mass_active_values).detach().cpu()),
                "max": float(torch.max(grid_mass_active_values).detach().cpu()),
                "mean": float(torch.mean(grid_mass_active_values).detach().cpu()),
                "sum": float(torch.sum(grid_mass_active_values).detach().cpu()),
            }
        else:
            mass_stats = {"min": 0.0, "max": 0.0, "mean": 0.0, "sum": 0.0}
        yield_stress_values = wp.to_torch(self.mpm_model.yield_stress)
        material_summary = {
            "material_id": int(self.mpm_model.material),
            "E_mean": float(torch.mean(wp.to_torch(self.mpm_model.E)).detach().cpu()),
            "nu_mean": float(torch.mean(wp.to_torch(self.mpm_model.nu)).detach().cpu()),
            "density_mean": float(
                torch.mean(wp.to_torch(self.mpm_state.particle_density)).detach().cpu()
            ),
            "yield_stress_mean": float(
                torch.mean(yield_stress_values).detach().cpu()
            ),
            "hardening": float(self.mpm_model.hardening),
            "xi": float(self.mpm_model.xi),
            "friction_angle": float(self.mpm_model.friction_angle),
            "plastic_viscosity": float(self.mpm_model.plastic_viscosity),
            "softening": float(self.mpm_model.softening),
            "grid_v_damping_scale": float(self.mpm_model.grid_v_damping_scale),
            "rpic_damping": float(self.mpm_model.rpic_damping),
            "return_mapping": {
                "trial": "evaluate_material_return_mapping",
                "commit": "compute_stress_from_F_trial",
                "trial_commits_internal_state": False,
            },
        }
        grid_summary = {
            "grid_size": [int(value) for value in grid_size],
            "n_grid": int(self.mpm_model.n_grid),
            "grid_lim": float(self.mpm_model.grid_lim),
            "dx": float(self.mpm_model.dx),
            "inv_dx": float(self.mpm_model.inv_dx),
            "mass": mass_stats,
            "active_free_fraction": float(
                active_free_node_count / max(grid_mass_node_count, 1)
            ),
        }
        dirichlet_summary = {
            "mass_nodes": int(dirichlet_mass_node_count),
            "total_nodes": int(dirichlet_total_count),
            "sources": {
                "cuboid": int(dirichlet_source_stats.get("cuboid", 0)),
                "surface": int(dirichlet_source_stats.get("surface", 0)),
                "other": int(dirichlet_source_stats.get("other", 0)),
            },
        }
        if not bool(grid_mass_mask.any().detach().cpu()):
            self.time = self.time + dt
            self.implicit_history.append(
                {
                    "trace_version": 2,
                    "integrator": "implicit",
                    "step": step,
                    "dt": float(dt),
                    "time": float(self.time),
                    "solver_settings": solver_settings,
                    "grid_summary": grid_summary,
                    "material_summary": material_summary,
                    "dirichlet_summary": dirichlet_summary,
                    "active_nodes": 0,
                    "active_free_nodes": 0,
                    "active_dof": 0,
                    "grid_mass_nodes": 0,
                    "dirichlet_nodes": 0,
                    "dirichlet_nodes_total": int(dirichlet_total_count),
                    "cuboid_dirichlet_nodes": int(
                        dirichlet_source_stats.get("cuboid", 0)
                    ),
                    "surface_dirichlet_nodes": int(
                        dirichlet_source_stats.get("surface", 0)
                    ),
                    "other_dirichlet_nodes": int(
                        dirichlet_source_stats.get("other", 0)
                    ),
                    "converged": True,
                    "convergence_type": "strict",
                    "near_converged": False,
                    "near_converged_factor": float(near_converged_factor),
                    "near_newton_rms_tol": float(near_newton_rms_tol),
                    "newton_exhausted": False,
                    "near_converged_objective_nonincreasing": True,
                    "final_is_finite": True,
                    "committed": True,
                    "substep_failed": False,
                    "line_search_saturated": False,
                    "accepted_due_to_near_residual": False,
                    "dangerous_nonconvergence": False,
                    "dangerous_nonconvergence_reason": None,
                    "newton_tol": float(newton_tol),
                    "newton_abs_tol": float(newton_abs_tol),
                    "newton_rms_tol": float(newton_rms_tol),
                    "gmres_tol": float(gmres_tol),
                    "gmres_tol_floor": float(gmres_tol_floor),
                    "gmres_forcing": "eisenstat_walker",
                    "newton_iters": 0,
                    "accepted_steps": 0,
                    "gmres_iters": 0,
                    "line_search_evals": 0,
                    "fallback_used_count": 0,
                    "line_search_failure_count": 0,
                    "boundary_projection_count": 0,
                    "boundary_projection_count_estimate": 0,
                    "pre_solve_clamp_count": int(pre_solve_clamp_count),
                    "out_of_bounds_count": int(pre_solve_clamp_count),
                    "clamp_count": 0,
                    "particle_clamp_count_estimate": 0,
                    "implicit_contact_residual": False,
                    "post_commit_projection_only": True,
                    "post_commit_projection_note": (
                        "Surface and bounding box projections are commit-only "
                        "safety projections, not implicit contact residuals."
                    ),
                    "initial_residual": 0.0,
                    "initial_residual_rms": 0.0,
                    "final_residual": 0.0,
                    "final_residual_rms": 0.0,
                    "residual_rms": 0.0,
                    "final_relative_residual": 0.0,
                    "newton_trace": [],
                }
            )
            return self.implicit_history[-1]

        grid_stiffness_wp = wp.zeros(shape=grid_size, dtype=float, device=device)
        wp.launch(
            kernel=zero_grid_float,
            dim=grid_size,
            inputs=[grid_stiffness_wp],
            device=device,
        )
        wp.launch(
            kernel=implicit_accumulate_stiffness_diag,
            dim=self.n_particles,
            inputs=[self.mpm_state, self.mpm_model, grid_stiffness_wp],
            device=device,
        )
        grid_stiffness = wp.to_torch(grid_stiffness_wp)

        grid_du = (dt * grid_vn + 0.5 * dt * dt * grid_an).clone()
        grid_v_target = wp.to_torch(self._implicit_grid_v_target)

        def apply_dirichlet_du(du_tensor):
            if not bool(dirichlet_mask.any().detach().cpu()):
                return du_tensor
            out = du_tensor.clone()
            v_n = grid_vn[dirichlet_mask]
            a_n = grid_an[dirichlet_mask]
            v_tar = grid_v_target[dirichlet_mask]
            a_np1 = (v_tar - v_n - dt * (1.0 - gamma) * a_n) * (
                1.0 / (gamma * dt)
            )
            out[dirichlet_mask] = (
                dt * v_n
                + dt * dt * (0.5 - beta) * a_n
                + beta * dt * dt * a_np1
            )
            return out

        grid_du = apply_dirichlet_du(grid_du)
        if active_free_node_count == 0:
            grid_vn_wp = wp.from_torch(grid_vn, dtype=wp.vec3)
            grid_du_wp = wp.from_torch(grid_du.contiguous(), dtype=wp.vec3)
            wp.launch(
                kernel=implicit_du_to_velocity,
                dim=grid_size,
                inputs=[
                    self.mpm_state,
                    grid_du_wp,
                    grid_vn_wp,
                    self._implicit_grid_an,
                    self.mpm_state.grid_v_out,
                    beta,
                    gamma,
                    dt,
                ],
                device=device,
            )
            self._apply_grid_velocity_damping(grid_size, device=device)
            self._last_boundary_projection_count = 0
            for k in range(len(self.grid_postprocess)):
                wp.launch(
                    kernel=self.grid_postprocess[k],
                    dim=grid_size,
                    inputs=[
                        self.time,
                        dt,
                        self.mpm_state,
                        self.mpm_model,
                        self.collider_params[k],
                    ],
                    device=device,
                )
            wp.launch(
                kernel=implicit_project_du_from_velocity,
                dim=grid_size,
                inputs=[
                    self.mpm_state,
                    grid_du_wp,
                    grid_vn_wp,
                    self._implicit_grid_an,
                    beta,
                    gamma,
                    dt,
                ],
                device=device,
            )
            wp.launch(
                kernel=g2p_implicit,
                dim=self.n_particles,
                inputs=[self.mpm_state, self.mpm_model, grid_du_wp, dt],
                device=device,
            )
            wp.launch(
                kernel=compute_stress_from_F_trial,
                dim=self.n_particles,
                inputs=[self.mpm_state, self.mpm_model, dt],
                device=device,
            )
            self._apply_fixed_particles(self.time + dt, device=device)
            wp.launch(
                kernel=implicit_update_acceleration_from_velocity,
                dim=grid_size,
                inputs=[
                    self.mpm_state,
                    grid_vn_wp,
                    self._implicit_grid_an,
                    gamma,
                    dt,
                ],
                device=device,
            )
            for k in range(len(self.grid_postprocess)):
                if self.modify_bc[k] is not None:
                    self.modify_bc[k](self.time, dt, self.collider_params[k])
            clamp_count = self._estimate_particle_clamp_count()
            self.time = self.time + dt
            diagnostics = {
                "trace_version": 2,
                "integrator": "implicit",
                "step": step,
                "dt": float(dt),
                "time": float(self.time),
                "solver_settings": solver_settings,
                "grid_summary": grid_summary,
                "material_summary": material_summary,
                "dirichlet_summary": dirichlet_summary,
                "active_nodes": 0,
                "active_free_nodes": 0,
                "active_dof": 0,
                "grid_mass_nodes": int(grid_mass_node_count),
                "dirichlet_nodes": int(dirichlet_mass_node_count),
                "dirichlet_nodes_total": int(dirichlet_total_count),
                "cuboid_dirichlet_nodes": int(
                    dirichlet_source_stats.get("cuboid", 0)
                ),
                "surface_dirichlet_nodes": int(
                    dirichlet_source_stats.get("surface", 0)
                ),
                "other_dirichlet_nodes": int(dirichlet_source_stats.get("other", 0)),
                "converged": True,
                "convergence_type": "strict",
                "near_converged": False,
                "near_converged_factor": float(near_converged_factor),
                "near_newton_rms_tol": float(near_newton_rms_tol),
                "newton_exhausted": False,
                "near_converged_objective_nonincreasing": True,
                "final_is_finite": True,
                "committed": True,
                "substep_failed": False,
                "line_search_saturated": False,
                "accepted_due_to_near_residual": False,
                "dangerous_nonconvergence": False,
                "dangerous_nonconvergence_reason": None,
                "beta": float(beta),
                "gamma": float(gamma),
                "newton_tol": float(newton_tol),
                "newton_abs_tol": float(newton_abs_tol),
                "newton_rms_tol": float(newton_rms_tol),
                "newton_iters": 0,
                "accepted_steps": 0,
                "gmres_iters": 0,
                "line_search_evals": 0,
                "fallback_used_count": 0,
                "line_search_failure_count": 0,
                "boundary_projection_count": int(self._last_boundary_projection_count),
                "boundary_projection_count_estimate": int(
                    self._last_boundary_projection_count
                ),
                "pre_solve_clamp_count": int(pre_solve_clamp_count),
                "out_of_bounds_count": int(pre_solve_clamp_count),
                "clamp_count": int(clamp_count),
                "particle_clamp_count_estimate": int(clamp_count),
                "implicit_contact_residual": False,
                "post_commit_projection_only": True,
                "post_commit_projection_note": (
                    "Surface and bounding box projections are commit-only "
                    "safety projections, not implicit contact residuals."
                ),
                "initial_residual": 0.0,
                "initial_residual_rms": 0.0,
                "final_residual": 0.0,
                "final_residual_rms": 0.0,
                "residual_rms": 0.0,
                "final_relative_residual": 0.0,
                "newton_trace": [],
                "gmres_tol": float(gmres_tol),
                "gmres_tol_floor": float(gmres_tol_floor),
                "gmres_forcing": "eisenstat_walker",
                "allow_best_effort_commit": bool(allow_best_effort_commit),
            }
            self.implicit_history.append(diagnostics)
            return diagnostics
        grid_force_wp = wp.zeros(shape=grid_size, dtype=wp.vec3, device=device)
        grid_residual_wp = wp.zeros(shape=grid_size, dtype=wp.vec3, device=device)
        trial_F_wp = wp.zeros(shape=self.n_particles, dtype=wp.mat33, device=device)
        trial_stress_wp = wp.zeros(shape=self.n_particles, dtype=wp.mat33, device=device)

        grid_vn_wp = wp.from_torch(grid_vn, dtype=wp.vec3)
        grid_an_wp = self._implicit_grid_an

        def residual_from_du(du_tensor):
            du_tensor = apply_dirichlet_du(du_tensor)
            grid_du_wp_local = wp.from_torch(du_tensor.contiguous(), dtype=wp.vec3)
            residual_grid = self._implicit_residual(
                grid_du_wp_local,
                grid_vn_wp,
                grid_an_wp,
                grid_force_wp,
                grid_residual_wp,
                trial_F_wp,
                trial_stress_wp,
                beta,
                gamma,
                dt,
                device=device,
            )
            return residual_grid[active_mask].reshape(-1).clone()

        residual = residual_from_du(grid_du)
        initial_norm = torch.linalg.norm(residual).clamp_min(1e-20)
        previous_residual_norm = None
        total_gmres_iters = 0
        total_line_search_iters = 0
        fallback_used_count = 0
        gmres_failed_count = 0
        line_search_failure_count = 0
        substep_failed = False
        stagnation_detected = False
        failure_reason = None
        accepted_steps = 0
        converged = False
        last_relative = float("inf")
        newton_trace = []

        mass_active = grid_mass[active_mask].reshape(-1, 1).expand(-1, 3).reshape(-1)
        stiffness_active = grid_stiffness[active_mask].reshape(-1, 1).expand(-1, 3).reshape(-1)
        precond_diag = (
            mass_active / (beta * dt * dt)
            + stiffness_preconditioner_scale * stiffness_active
        )
        precond_diag = torch.clamp(precond_diag, min=1e-12)
        precond_min = float(torch.min(precond_diag).detach().cpu())
        precond_max = float(torch.max(precond_diag).detach().cpu())
        precond_mean = float(torch.mean(precond_diag).detach().cpu())
        precond_std = float(torch.std(precond_diag, unbiased=False).detach().cpu())
        preconditioner_summary = {
            "min": precond_min,
            "max": precond_max,
            "mean": precond_mean,
            "std": precond_std,
            "stiffness_scale": float(stiffness_preconditioner_scale),
            "mass_term_mean": float(
                torch.mean(mass_active / (beta * dt * dt)).detach().cpu()
            ),
            "stiffness_term_mean": float(
                torch.mean(
                    stiffness_preconditioner_scale * stiffness_active
                ).detach().cpu()
            ),
        }

        def unpack_active(vec):
            out = torch.zeros_like(grid_du)
            out[active_mask] = vec.reshape(-1, 3)
            return out

        def residual_rms_value(norm_tensor):
            return float(norm_tensor.detach().cpu()) / active_dof_sqrt

        def residual_norms_satisfy_strict(norm_tensor, relative_value):
            norm_float = float(norm_tensor.detach().cpu())
            rms_float = norm_float / active_dof_sqrt
            return (
                relative_value <= newton_tol
                or norm_float <= newton_abs_tol
                or rms_float <= newton_rms_tol
            )

        def residual_norms_satisfy_near(norm_tensor, relative_value):
            norm_float = float(norm_tensor.detach().cpu())
            rms_float = norm_float / active_dof_sqrt
            return (
                relative_value <= newton_tol * near_converged_factor
                or rms_float <= newton_rms_tol
            )

        def recent_objective_is_acceptable(current_objective):
            objectives = []
            for record in newton_trace[-4:]:
                raw_objective = record.get("trial_objective", record.get("objective"))
                try:
                    objective_value = float(raw_objective)
                except (TypeError, ValueError):
                    return False
                if not math.isfinite(objective_value):
                    return False
                objectives.append(objective_value)
            if not objectives:
                return math.isfinite(current_objective)
            objective_best = min(objectives + [current_objective])
            return (
                math.isfinite(current_objective)
                and current_objective <= objective_best * (1.0 + 1e-3) + 1e-12
            )

        for _newton_iter in range(newton_max_iter):
            residual_norm = torch.linalg.norm(residual)
            last_relative = float((residual_norm / initial_norm).detach().cpu())
            residual_rms_float = residual_rms_value(residual_norm)
            if residual_norms_satisfy_strict(residual_norm, last_relative):
                converged = True
                break

            def jvp_for_direction_grid(direction_grid):
                if bool(dirichlet_mask.any().detach().cpu()):
                    direction_grid = direction_grid.clone()
                    direction_grid[dirichlet_mask] = 0.0
                direction_scale = torch.max(torch.abs(direction_grid[active_mask]))
                if float(direction_scale.detach().cpu()) <= 1e-20:
                    return torch.zeros_like(residual)
                eps = jvp_eps / torch.clamp(direction_scale, min=1e-12)
                r_plus = residual_from_du(grid_du + eps * direction_grid)
                r_minus = residual_from_du(grid_du - eps * direction_grid)
                return (r_plus - r_minus) / (2.0 * eps)

            def matvec_preconditioned(y):
                direction = y / precond_diag
                direction_grid = unpack_active(direction)
                return jvp_for_direction_grid(direction_grid)

            rhs = -residual
            if previous_residual_norm is None:
                ew_tol = ew_eta_max
            else:
                ratio = float(
                    (residual_norm / torch.clamp(previous_residual_norm, min=1e-20))
                    .detach()
                    .cpu()
                )
                ew_tol = ew_gamma * (ratio ** ew_alpha)
            ew_tol = min(ew_eta_max, max(gmres_tol_floor, ew_tol))
            tail_tolerance_cap = None
            tail_tolerance_trigger = 20.0 * float(newton_tol)
            if last_relative <= tail_tolerance_trigger:
                # EW can otherwise relax back to eta_max in the Newton tail,
                # making GMRES stop after one very coarse Krylov step.
                tail_tolerance_cap = max(
                    gmres_tol_floor,
                    0.1 * float(newton_tol) / max(last_relative, float(newton_tol)),
                )
            linear_tol = (
                min(ew_tol, tail_tolerance_cap)
                if tail_tolerance_cap is not None
                else ew_tol
            )
            y, gmres_iters, gmres_rel, gmres_converged, gmres_info = self._gmres_matrix_free(
                matvec_preconditioned,
                rhs,
                tol=linear_tol,
                max_iter=gmres_max_iter,
                min_iter=2,
            )
            if not gmres_converged:
                gmres_failed_count += 1
            j_delta = matvec_preconditioned(y)
            linear_residual = rhs - j_delta
            linear_residual_norm = torch.linalg.norm(linear_residual)
            linear_residual_relative = float(
                (linear_residual_norm / torch.linalg.norm(rhs).clamp_min(1e-20))
                .detach()
                .cpu()
            )
            total_gmres_iters += gmres_iters
            delta_active = y / precond_diag
            delta_grid = unpack_active(delta_active)

            current_phi = 0.5 * torch.dot(residual, residual)
            current_phi_float = float(current_phi.detach().cpu())
            directional_derivative = torch.dot(residual, j_delta)
            delta_norm = float(
                torch.linalg.norm(delta_grid[active_mask].reshape(-1)).detach().cpu()
            )
            du_norm = float(
                torch.linalg.norm(grid_du[active_mask].reshape(-1)).detach().cpu()
            )
            residual_norm_float = float(residual_norm.detach().cpu())
            iteration_record = {
                "newton_iter": int(_newton_iter),
                "residual": residual_norm_float,
                "residual_rms": float(residual_rms_float),
                "relative_residual": float(last_relative),
                "residual_reduction_from_previous": (
                    None
                    if previous_residual_norm is None
                    else float(
                        (residual_norm / torch.clamp(previous_residual_norm, min=1e-20))
                        .detach()
                        .cpu()
                    )
                ),
                "objective": float(current_phi_float),
                "ew_tolerance": float(ew_tol),
                "ew_tail_tolerance_cap": (
                    None
                    if tail_tolerance_cap is None
                    else float(tail_tolerance_cap)
                ),
                "ew_tail_tolerance_trigger": float(tail_tolerance_trigger),
                "linear_tolerance": float(linear_tol),
                "gmres_forcing": "eisenstat_walker",
                "gmres_tol_floor": float(gmres_tol_floor),
                "gmres_iters": int(gmres_iters),
                "gmres_relative_residual": float(gmres_rel),
                "gmres_converged": bool(gmres_converged),
                "gmres_break_reason": gmres_info.get("break_reason"),
                "gmres_rhs_norm": float(gmres_info.get("rhs_norm", 0.0)),
                "gmres_min_iter": int(gmres_info.get("min_iter", 1)),
                "gmres_used_iter": int(gmres_info.get("used_iter", gmres_iters)),
                "gmres_actual_relative_residual": linear_residual_relative,
                "gmres_actual_residual": float(linear_residual_norm.detach().cpu()),
                "gmres_arnoldi_norm": float(gmres_info.get("arnoldi_norm", 0.0)),
                "gmres_h_diag": float(gmres_info.get("h_diag", 0.0)),
                "gmres_basis_size": int(gmres_info.get("basis_size", 0)),
                "directional_derivative": float(directional_derivative.detach().cpu()),
                "direction": "gmres",
                "delta_norm": delta_norm,
                "du_norm": du_norm,
                "fallback_used": False,
                "fallback_skipped_reason": None,
                "fallback_directional_derivative": None,
                "fallback_step_relative_norm": None,
                "accepted": False,
                "accepted_due_to_near_residual": False,
                "line_search_saturated": False,
                "alpha": 0.0,
                "line_search_evals": 0,
                "line_search_trials": [],
            }

            def mark_line_search_saturated_near(reason):
                iteration_record["line_search_saturated"] = True
                iteration_record["line_search_saturated_reason"] = reason
                iteration_record["accepted_due_to_near_residual"] = True
                if iteration_record.get("fallback_skipped_reason") is None:
                    iteration_record["fallback_skipped_reason"] = "near_residual"

            def current_residual_is_near():
                return (
                    math.isfinite(residual_norm_float)
                    and math.isfinite(last_relative)
                    and residual_norms_satisfy_near(residual_norm, last_relative)
                    and recent_objective_is_acceptable(current_phi_float)
                )

            def make_fallback_direction(label):
                # Paper fallback: if the Newton-GMRES direction is not descent,
                # try the right-preconditioned residual objective direction.
                # The residual sign convention differs between MPM force
                # forms, so evaluate both signs and choose the descent one.
                fallback_minus_active = -residual / precond_diag
                fallback_minus_grid = unpack_active(fallback_minus_active)
                fallback_minus_j_delta = jvp_for_direction_grid(fallback_minus_grid)
                fallback_minus_derivative = torch.dot(
                    residual, fallback_minus_j_delta
                )
                fallback_plus_active = residual / precond_diag
                fallback_plus_grid = unpack_active(fallback_plus_active)
                fallback_plus_j_delta = jvp_for_direction_grid(fallback_plus_grid)
                fallback_plus_derivative = torch.dot(residual, fallback_plus_j_delta)
                use_plus = bool(
                    (
                        fallback_plus_derivative < fallback_minus_derivative
                    )
                    .detach()
                    .cpu()
                )
                if use_plus:
                    fallback_delta_active = fallback_plus_active
                    fallback_delta_grid = fallback_plus_grid
                    fallback_j_delta = fallback_plus_j_delta
                    fallback_derivative = fallback_plus_derivative
                    fallback_sign = 1.0
                else:
                    fallback_delta_active = fallback_minus_active
                    fallback_delta_grid = fallback_minus_grid
                    fallback_j_delta = fallback_minus_j_delta
                    fallback_derivative = fallback_minus_derivative
                    fallback_sign = -1.0
                fallback_derivative_float = float(
                    fallback_derivative.detach().cpu()
                )
                fallback_delta_norm = float(
                    torch.linalg.norm(
                        fallback_delta_grid[active_mask].reshape(-1)
                    ).detach().cpu()
                )
                fallback_step_relative_norm = fallback_delta_norm / max(
                    du_norm, 1e-20
                )
                iteration_record["fallback_skipped_reason"] = None
                iteration_record["fallback_minus_directional_derivative"] = float(
                    fallback_minus_derivative.detach().cpu()
                )
                iteration_record["fallback_plus_directional_derivative"] = float(
                    fallback_plus_derivative.detach().cpu()
                )
                iteration_record["fallback_directional_derivative"] = (
                    fallback_derivative_float
                )
                iteration_record["fallback_delta_norm"] = fallback_delta_norm
                iteration_record["fallback_step_relative_norm"] = (
                    fallback_step_relative_norm
                )
                descent_threshold = -fallback_descent_tol * max(1.0, current_phi_float)
                if not math.isfinite(fallback_derivative_float):
                    iteration_record["fallback_skipped_reason"] = (
                        "non_finite_directional_derivative"
                    )
                    return None
                if fallback_derivative_float >= descent_threshold:
                    iteration_record["fallback_skipped_reason"] = "not_descent"
                    return None
                if fallback_step_relative_norm < fallback_step_min_rel:
                    iteration_record["fallback_skipped_reason"] = "step_too_small"
                    return None
                iteration_record["direction"] = label
                iteration_record["fallback_used"] = True
                iteration_record["fallback_preconditioned"] = True
                iteration_record["fallback_sign"] = float(fallback_sign)
                iteration_record["directional_derivative"] = fallback_derivative_float
                return (
                    fallback_delta_active,
                    fallback_delta_grid,
                    fallback_j_delta,
                    fallback_derivative,
                )

            if float(directional_derivative.detach().cpu()) >= 0.0:
                if current_residual_is_near():
                    mark_line_search_saturated_near("gmres_not_descent_near_residual")
                else:
                    fallback_direction = make_fallback_direction(
                        "fallback_negative_residual"
                    )
                    if fallback_direction is not None:
                        (
                            delta_active,
                            delta_grid,
                            j_delta,
                            directional_derivative,
                        ) = fallback_direction
                        fallback_used_count += 1

            alpha = 1.0
            accepted = False
            line_search_evals_this_iter = 0
            should_run_line_search = not iteration_record.get("line_search_saturated")
            if (
                should_run_line_search
                and float(directional_derivative.detach().cpu()) >= 0.0
                and not iteration_record.get("fallback_used")
            ):
                iteration_record["fallback_skipped_reason"] = (
                    iteration_record.get("fallback_skipped_reason")
                    or "gmres_not_descent"
                )
                should_run_line_search = False
            for _ls_iter in range(line_search_max_iter if should_run_line_search else 0):
                trial_du = grid_du + alpha * delta_grid
                trial_residual = residual_from_du(trial_du)
                trial_phi = 0.5 * torch.dot(trial_residual, trial_residual)
                line_search_evals_this_iter += 1
                armijo_rhs = current_phi + armijo_c1 * alpha * directional_derivative
                trial_phi_float = float(trial_phi.detach().cpu())
                armijo_rhs_float = float(armijo_rhs.detach().cpu())
                iteration_record["last_trial_objective"] = float(
                    trial_phi_float
                )
                iteration_record["last_armijo_rhs"] = float(armijo_rhs_float)
                iteration_record["last_alpha"] = float(alpha)
                trial_relative = float(
                    (torch.linalg.norm(trial_residual) / initial_norm).detach().cpu()
                )
                iteration_record["line_search_trials"].append({
                    "alpha": float(alpha),
                    "objective": float(trial_phi_float),
                    "relative_residual": trial_relative,
                    "armijo_rhs": float(armijo_rhs_float),
                    "accepted": False,
                })
                if iteration_record.get("fallback_used"):
                    accept_trial = trial_phi_float < current_phi_float * (
                        1.0 - fallback_decrease_tol
                    )
                    iteration_record["acceptance_rule"] = "fallback_residual_decrease"
                else:
                    accept_trial = trial_phi_float <= armijo_rhs_float
                    iteration_record["acceptance_rule"] = "armijo"
                if accept_trial:
                    step_norm = torch.linalg.norm(alpha * delta_grid[active_mask].reshape(-1))
                    base_norm = torch.linalg.norm(grid_du[active_mask].reshape(-1)).clamp_min(1e-20)
                    grid_du = trial_du
                    previous_residual_norm = residual_norm
                    residual = trial_residual
                    accepted = True
                    accepted_steps += 1
                    iteration_record["accepted"] = True
                    iteration_record["alpha"] = float(alpha)
                    iteration_record["line_search_evals"] = int(
                        line_search_evals_this_iter
                    )
                    iteration_record["trial_relative_residual"] = trial_relative
                    iteration_record["trial_objective"] = float(
                        trial_phi_float
                    )
                    iteration_record["line_search_trials"][-1]["accepted"] = True
                    iteration_record["accepted_step_norm"] = float(
                        step_norm.detach().cpu()
                    )
                    iteration_record["accepted_step_relative_norm"] = float(
                        (step_norm / base_norm).detach().cpu()
                    )
                    if float((step_norm / base_norm).detach().cpu()) <= stagnation_tol:
                        current_residual_norm = torch.linalg.norm(residual)
                        current_relative = float(
                            (current_residual_norm / initial_norm).detach().cpu()
                        )
                        converged = bool(
                            residual_norms_satisfy_strict(
                                current_residual_norm, current_relative
                            )
                        )
                        iteration_record["stagnated"] = True
                    break
                alpha *= 0.5

            if (
                not accepted
                and not iteration_record.get("fallback_used")
                and not iteration_record.get("line_search_saturated")
                and line_search_evals_this_iter > 0
            ):
                if current_residual_is_near():
                    mark_line_search_saturated_near("gmres_armijo_failed_near_residual")
                else:
                    fallback_direction = make_fallback_direction(
                        "fallback_negative_residual_after_line_search"
                    )
                    if fallback_direction is not None:
                        (
                            delta_active,
                            delta_grid,
                            j_delta,
                            directional_derivative,
                        ) = fallback_direction
                        fallback_used_count += 1
                alpha = 1.0
                for _ls_iter in range(
                    line_search_max_iter
                    if iteration_record.get("fallback_used")
                    else 0
                ):
                    trial_du = grid_du + alpha * delta_grid
                    trial_residual = residual_from_du(trial_du)
                    trial_phi = 0.5 * torch.dot(trial_residual, trial_residual)
                    line_search_evals_this_iter += 1
                    armijo_rhs = current_phi + armijo_c1 * alpha * directional_derivative
                    trial_phi_float = float(trial_phi.detach().cpu())
                    armijo_rhs_float = float(armijo_rhs.detach().cpu())
                    iteration_record["last_trial_objective"] = float(
                        trial_phi_float
                    )
                    iteration_record["last_armijo_rhs"] = float(armijo_rhs_float)
                    iteration_record["last_alpha"] = float(alpha)
                    iteration_record["acceptance_rule"] = "fallback_residual_decrease"
                    trial_relative = float(
                        (torch.linalg.norm(trial_residual) / initial_norm).detach().cpu()
                    )
                    iteration_record["line_search_trials"].append({
                        "alpha": float(alpha),
                        "objective": float(trial_phi_float),
                        "relative_residual": trial_relative,
                        "armijo_rhs": float(armijo_rhs_float),
                        "accepted": False,
                    })
                    if trial_phi_float < current_phi_float * (
                        1.0 - fallback_decrease_tol
                    ):
                        step_norm = torch.linalg.norm(
                            alpha * delta_grid[active_mask].reshape(-1)
                        )
                        base_norm = torch.linalg.norm(
                            grid_du[active_mask].reshape(-1)
                        ).clamp_min(1e-20)
                        previous_residual_norm = residual_norm
                        grid_du = trial_du
                        residual = trial_residual
                        accepted = True
                        accepted_steps += 1
                        iteration_record["accepted"] = True
                        iteration_record["alpha"] = float(alpha)
                        iteration_record["line_search_evals"] = int(
                            line_search_evals_this_iter
                        )
                        iteration_record["trial_relative_residual"] = trial_relative
                        iteration_record["trial_objective"] = float(
                            trial_phi_float
                        )
                        iteration_record["line_search_trials"][-1]["accepted"] = True
                        iteration_record["accepted_step_norm"] = float(
                            step_norm.detach().cpu()
                        )
                        iteration_record["accepted_step_relative_norm"] = float(
                            (step_norm / base_norm).detach().cpu()
                        )
                        if float((step_norm / base_norm).detach().cpu()) <= stagnation_tol:
                            current_residual_norm = torch.linalg.norm(residual)
                            current_relative = float(
                                (current_residual_norm / initial_norm)
                                .detach()
                                .cpu()
                            )
                            converged = bool(
                                residual_norms_satisfy_strict(
                                    current_residual_norm, current_relative
                                )
                            )
                            iteration_record["stagnated"] = True
                        break
                    alpha *= 0.5
                if not accepted and iteration_record.get("fallback_used"):
                    iteration_record["fallback_skipped_reason"] = (
                        "line_search_failed"
                    )
                    if current_residual_is_near():
                        mark_line_search_saturated_near(
                            "fallback_line_search_failed_near_residual"
                        )
            total_line_search_iters += line_search_evals_this_iter
            if not accepted:
                iteration_record["line_search_evals"] = int(line_search_evals_this_iter)
            newton_trace.append(iteration_record)
            if not accepted:
                line_search_failure_count += 1
                substep_failed = True
                if iteration_record.get("line_search_saturated"):
                    failure_reason = "line_search_saturated"
                    iteration_record["line_search_failed"] = False
                else:
                    failure_reason = "line_search_failed"
                    iteration_record["line_search_failed"] = True
                iteration_record["failure_reason"] = failure_reason
                break
            if iteration_record.get("stagnated") and not converged:
                stagnation_detected = True
                substep_failed = True
                failure_reason = "stagnation"
                iteration_record["failure_reason"] = failure_reason
                break

        final_norm = torch.linalg.norm(residual)
        last_relative = float((final_norm / initial_norm).detach().cpu())
        initial_norm_float = float(initial_norm.detach().cpu())
        initial_residual_rms = initial_norm_float / active_dof_sqrt
        final_norm_float = float(final_norm.detach().cpu())
        final_residual_rms = final_norm_float / active_dof_sqrt
        strict_converged = converged or residual_norms_satisfy_strict(
            final_norm, last_relative
        )
        converged = bool(strict_converged)
        if substep_failed and not converged:
            converged = False

        final_is_finite = math.isfinite(final_norm_float) and math.isfinite(
            last_relative
        ) and math.isfinite(final_residual_rms)
        line_search_saturated = bool(
            any(record.get("line_search_saturated") for record in newton_trace)
            or (
                line_search_failure_count > 0
                and failure_reason in {"line_search_failed", "line_search_saturated"}
            )
        )
        objective_history = []
        objective_history_finite = True
        for record in newton_trace:
            raw_objective = record.get("trial_objective", record.get("objective"))
            try:
                objective_value = float(raw_objective)
            except (TypeError, ValueError):
                objective_history_finite = False
                continue
            if not math.isfinite(objective_value):
                objective_history_finite = False
            objective_history.append(objective_value)
        recent_objectives = objective_history[-3:]
        objective_best = min(objective_history) if objective_history else float("inf")
        objective_current = objective_history[-1] if objective_history else float("inf")
        near_converged_objective_nonincreasing = (
            objective_history_finite
            and objective_history
            and (
                len(recent_objectives) < 2
                or objective_current <= objective_best * (1.0 + 1e-3) + 1e-12
            )
        )
        newton_exhausted = (
            len(newton_trace) >= int(newton_max_iter)
            and not strict_converged
            and not substep_failed
        )
        near_residual_small = residual_norms_satisfy_near(final_norm, last_relative)
        accepted_due_to_near_residual = False
        dangerous_nonconvergence_reason = None
        near_converged = (
            final_is_finite
            and (newton_exhausted or line_search_saturated)
            and near_residual_small
            and near_converged_objective_nonincreasing
        )
        convergence_type = "strict" if converged else "not_converged"
        if near_converged:
            converged = True
            substep_failed = False
            failure_reason = None
            accepted_due_to_near_residual = True
            if line_search_saturated:
                convergence_type = "near_converged_line_search_saturated"
            else:
                convergence_type = "near_converged"
        if converged:
            substep_failed = False
            failure_reason = None
        dangerous_nonconvergence = False
        if not converged:
            if not final_is_finite:
                dangerous_nonconvergence = True
                dangerous_nonconvergence_reason = "non_finite_residual"
            elif (
                last_relative > nonlinear_failure_relative
                and final_norm_float > nonlinear_failure_absolute
            ):
                dangerous_nonconvergence = True
                dangerous_nonconvergence_reason = "large_residual"
            elif line_search_saturated:
                dangerous_nonconvergence = True
                dangerous_nonconvergence_reason = "line_search_failed"
            elif substep_failed:
                dangerous_nonconvergence = True
                dangerous_nonconvergence_reason = failure_reason or "substep_failed"
        should_fail_without_commit = (not converged) and (not allow_best_effort_commit)
        if dangerous_nonconvergence or should_fail_without_commit:
            newton_failure_detail = failure_reason
            failure_reason = "newton_not_converged"
            diagnostics = {
                "trace_version": 2,
                "integrator": "implicit",
                "step": step,
                "dt": float(dt),
                "time": float(self.time),
                "solver_settings": solver_settings,
                "grid_summary": grid_summary,
                "material_summary": material_summary,
                "dirichlet_summary": dirichlet_summary,
                "active_nodes": int(active_free_node_count),
                "active_free_nodes": int(active_free_node_count),
                "active_dof": int(active_dof),
                "grid_mass_nodes": int(grid_mass_node_count),
                "dirichlet_nodes": int(dirichlet_mass_node_count),
                "dirichlet_nodes_total": int(dirichlet_total_count),
                "cuboid_dirichlet_nodes": int(dirichlet_source_stats.get("cuboid", 0)),
                "surface_dirichlet_nodes": int(
                    dirichlet_source_stats.get("surface", 0)
                ),
                "other_dirichlet_nodes": int(dirichlet_source_stats.get("other", 0)),
                "converged": False,
                "convergence_type": "not_converged",
                "near_converged": False,
                "near_converged_factor": float(near_converged_factor),
                "near_newton_rms_tol": float(near_newton_rms_tol),
                "newton_exhausted": bool(newton_exhausted),
                "near_converged_objective_nonincreasing": bool(
                    near_converged_objective_nonincreasing
                ),
                "final_is_finite": bool(final_is_finite),
                "committed": False,
                "substep_failed": True,
                "line_search_saturated": bool(line_search_saturated),
                "accepted_due_to_near_residual": bool(
                    accepted_due_to_near_residual
                ),
                "failure_reason": failure_reason,
                "newton_failure_detail": newton_failure_detail,
                "stagnation_detected": bool(stagnation_detected),
                "dangerous_nonconvergence": bool(dangerous_nonconvergence),
                "dangerous_nonconvergence_reason": dangerous_nonconvergence_reason,
                "nonlinear_failure_relative": float(nonlinear_failure_relative),
                "nonlinear_failure_absolute": float(nonlinear_failure_absolute),
                "beta": float(beta),
                "gamma": float(gamma),
                "newton_tol": float(newton_tol),
                "newton_abs_tol": float(newton_abs_tol),
                "newton_rms_tol": float(newton_rms_tol),
                "newton_max_iter": int(newton_max_iter),
                "gmres_tol": float(gmres_tol),
                "gmres_tol_floor": float(gmres_tol_floor),
                "gmres_forcing": "eisenstat_walker",
                "gmres_max_iter": int(gmres_max_iter),
                "jvp_eps": float(jvp_eps),
                "line_search_max_iter": int(line_search_max_iter),
                "armijo_c1": float(armijo_c1),
                "ew_eta_min": float(ew_eta_min),
                "ew_eta_max": float(ew_eta_max),
                "ew_gamma": float(ew_gamma),
                "ew_alpha": float(ew_alpha),
                "stiffness_preconditioner_scale": float(
                    stiffness_preconditioner_scale
                ),
                "stagnation_tol": float(stagnation_tol),
                "allow_best_effort_commit": bool(allow_best_effort_commit),
                "preconditioner_min": precond_min,
                "preconditioner_max": precond_max,
                "preconditioner_mean": precond_mean,
                "preconditioner_std": precond_std,
                "preconditioner_summary": preconditioner_summary,
                "newton_iters": int(len(newton_trace)),
                "accepted_steps": int(accepted_steps),
                "gmres_iters": int(total_gmres_iters),
                "gmres_failed_count": int(gmres_failed_count),
                "line_search_evals": int(total_line_search_iters),
                "fallback_used_count": int(fallback_used_count),
                "line_search_failure_count": int(line_search_failure_count),
                "boundary_projection_count": int(self._last_boundary_projection_count),
                "boundary_projection_count_estimate": int(
                    self._last_boundary_projection_count
                ),
                "pre_solve_clamp_count": int(pre_solve_clamp_count),
                "out_of_bounds_count": int(pre_solve_clamp_count),
                "clamp_count": 0,
                "particle_clamp_count_estimate": 0,
                "state_restored_on_failure": True,
                "implicit_contact_residual": False,
                "post_commit_projection_only": True,
                "post_commit_projection_note": (
                    "Surface and bounding box projections are commit-only "
                    "safety projections, not implicit contact residuals."
                ),
                "initial_residual": float(initial_norm.detach().cpu()),
                "initial_residual_rms": float(initial_residual_rms),
                "final_residual": final_norm_float,
                "final_residual_rms": float(final_residual_rms),
                "residual_rms": float(final_residual_rms),
                "final_relative_residual": float(last_relative),
                "newton_trace": newton_trace,
            }
            self._restore_implicit_particle_state(particle_state_snapshot)
            self.implicit_history.append(diagnostics)
            return diagnostics

        grid_du = apply_dirichlet_du(grid_du)
        grid_du_wp = wp.from_torch(grid_du.contiguous(), dtype=wp.vec3)
        wp.launch(
            kernel=implicit_du_to_velocity,
            dim=grid_size,
            inputs=[
                self.mpm_state,
                grid_du_wp,
                grid_vn_wp,
                grid_an_wp,
                self.mpm_state.grid_v_out,
                beta,
                gamma,
                dt,
            ],
            device=device,
        )
        self._apply_grid_velocity_damping(grid_size, device=device)

        self._last_boundary_projection_count = (
            self._estimate_bounding_box_projection_count()
        )
        for k in range(len(self.grid_postprocess)):
            wp.launch(
                kernel=self.grid_postprocess[k],
                dim=grid_size,
                inputs=[
                    self.time,
                    dt,
                    self.mpm_state,
                    self.mpm_model,
                    self.collider_params[k],
                ],
                device=device,
            )

        wp.launch(
            kernel=implicit_project_du_from_velocity,
            dim=grid_size,
            inputs=[
                self.mpm_state,
                grid_du_wp,
                grid_vn_wp,
                self._implicit_grid_an,
                beta,
                gamma,
                dt,
            ],
            device=device,
        )

        wp.launch(
            kernel=g2p_implicit,
            dim=self.n_particles,
            inputs=[self.mpm_state, self.mpm_model, grid_du_wp, dt],
            device=device,
        )

        wp.launch(
            kernel=compute_stress_from_F_trial,
            dim=self.n_particles,
            inputs=[self.mpm_state, self.mpm_model, dt],
            device=device,
        )
        self._apply_fixed_particles(self.time + dt, device=device)

        wp.launch(
            kernel=implicit_update_acceleration_from_velocity,
            dim=grid_size,
            inputs=[
                self.mpm_state,
                grid_vn_wp,
                self._implicit_grid_an,
                gamma,
                dt,
            ],
            device=device,
        )

        for k in range(len(self.grid_postprocess)):
            if self.modify_bc[k] is not None:
                self.modify_bc[k](self.time, dt, self.collider_params[k])

        clamp_count = self._estimate_particle_clamp_count()
        self.time = self.time + dt
        diagnostics = {
            "trace_version": 2,
            "integrator": "implicit",
            "step": step,
            "dt": float(dt),
            "time": float(self.time),
            "solver_settings": solver_settings,
            "grid_summary": grid_summary,
            "material_summary": material_summary,
            "dirichlet_summary": dirichlet_summary,
            "active_nodes": int(active_free_node_count),
            "active_free_nodes": int(active_free_node_count),
            "active_dof": int(active_dof),
            "grid_mass_nodes": int(grid_mass_node_count),
            "dirichlet_nodes": int(dirichlet_mass_node_count),
            "dirichlet_nodes_total": int(dirichlet_total_count),
            "cuboid_dirichlet_nodes": int(dirichlet_source_stats.get("cuboid", 0)),
            "surface_dirichlet_nodes": int(dirichlet_source_stats.get("surface", 0)),
            "other_dirichlet_nodes": int(dirichlet_source_stats.get("other", 0)),
            "converged": bool(converged),
            "convergence_type": convergence_type,
            "near_converged": bool(near_converged),
            "near_converged_factor": float(near_converged_factor),
            "near_newton_rms_tol": float(near_newton_rms_tol),
            "newton_exhausted": bool(newton_exhausted),
            "near_converged_objective_nonincreasing": bool(
                near_converged_objective_nonincreasing
            ),
            "final_is_finite": bool(final_is_finite),
            "committed": True,
            "substep_failed": bool(substep_failed and not converged),
            "line_search_saturated": bool(line_search_saturated),
            "accepted_due_to_near_residual": bool(accepted_due_to_near_residual),
            "failure_reason": failure_reason,
            "stagnation_detected": bool(stagnation_detected),
            "dangerous_nonconvergence": bool(dangerous_nonconvergence),
            "dangerous_nonconvergence_reason": dangerous_nonconvergence_reason,
            "beta": float(beta),
            "gamma": float(gamma),
            "newton_tol": float(newton_tol),
            "newton_abs_tol": float(newton_abs_tol),
            "newton_rms_tol": float(newton_rms_tol),
            "newton_max_iter": int(newton_max_iter),
            "gmres_tol": float(gmres_tol),
            "gmres_tol_floor": float(gmres_tol_floor),
            "gmres_forcing": "eisenstat_walker",
            "gmres_max_iter": int(gmres_max_iter),
            "jvp_eps": float(jvp_eps),
            "line_search_max_iter": int(line_search_max_iter),
            "armijo_c1": float(armijo_c1),
            "ew_eta_min": float(ew_eta_min),
            "ew_eta_max": float(ew_eta_max),
            "ew_gamma": float(ew_gamma),
            "ew_alpha": float(ew_alpha),
            "stiffness_preconditioner_scale": float(stiffness_preconditioner_scale),
            "stagnation_tol": float(stagnation_tol),
            "allow_best_effort_commit": bool(allow_best_effort_commit),
            "preconditioner_min": precond_min,
            "preconditioner_max": precond_max,
            "preconditioner_mean": precond_mean,
            "preconditioner_std": precond_std,
            "preconditioner_summary": preconditioner_summary,
            "newton_iters": int(len(newton_trace)),
            "accepted_steps": int(accepted_steps),
            "gmres_iters": int(total_gmres_iters),
            "gmres_failed_count": int(gmres_failed_count),
            "line_search_evals": int(total_line_search_iters),
            "fallback_used_count": int(fallback_used_count),
            "line_search_failure_count": int(line_search_failure_count),
            "boundary_projection_count": int(self._last_boundary_projection_count),
            "boundary_projection_count_estimate": int(
                self._last_boundary_projection_count
            ),
            "pre_solve_clamp_count": int(pre_solve_clamp_count),
            "out_of_bounds_count": int(pre_solve_clamp_count),
            "clamp_count": int(clamp_count),
            "particle_clamp_count_estimate": int(clamp_count),
            "implicit_contact_residual": False,
            "post_commit_projection_only": True,
            "post_commit_projection_note": (
                "Surface and bounding box projections are commit-only "
                "safety projections, not implicit contact residuals."
            ),
            "initial_residual": float(initial_norm.detach().cpu()),
            "initial_residual_rms": float(initial_residual_rms),
            "final_residual": float(final_norm.detach().cpu()),
            "final_residual_rms": float(final_residual_rms),
            "residual_rms": float(final_residual_rms),
            "final_relative_residual": float(last_relative),
            "newton_trace": newton_trace,
        }
        self.implicit_history.append(diagnostics)
        return diagnostics

    # set particle densities to all_particle_densities,
    def reset_densities_and_update_masses(
        self, all_particle_densities, device="cuda:0"
    ):
        all_particle_densities = all_particle_densities.clone().detach()
        self.mpm_state.particle_density = torch2warp_float(
            all_particle_densities, dvc=device
        )
        wp.launch(
            kernel=get_float_array_product,
            dim=self.n_particles,
            inputs=[
                self.mpm_state.particle_density,
                self.mpm_state.particle_vol,
                self.mpm_state.particle_mass,
            ],
            device=device,
        )

    # clone = True makes a copy, not necessarily needed
    def import_particle_x_from_torch(self, tensor_x, clone=True, device="cuda:0"):
        if tensor_x is not None:
            if clone:
                tensor_x = tensor_x.clone().detach()
            self.mpm_state.particle_x = torch2warp_vec3(tensor_x, dvc=device)

    # clone = True makes a copy, not necessarily needed
    def import_particle_v_from_torch(self, tensor_v, clone=True, device="cuda:0"):
        if tensor_v is not None:
            if clone:
                tensor_v = tensor_v.clone().detach()
            self.mpm_state.particle_v = torch2warp_vec3(tensor_v, dvc=device)

    # clone = True makes a copy, not necessarily needed
    def import_particle_F_from_torch(self, tensor_F, clone=True, device="cuda:0"):
        if tensor_F is not None:
            if clone:
                tensor_F = tensor_F.clone().detach()
            tensor_F = torch.reshape(tensor_F, (-1, 3, 3))  # arranged by rowmajor
            self.mpm_state.particle_F = torch2warp_mat33(tensor_F, dvc=device)

    # clone = True makes a copy, not necessarily needed
    def import_particle_C_from_torch(self, tensor_C, clone=True, device="cuda:0"):
        if tensor_C is not None:
            if clone:
                tensor_C = tensor_C.clone().detach()
            tensor_C = torch.reshape(tensor_C, (-1, 3, 3))  # arranged by rowmajor
            self.mpm_state.particle_C = torch2warp_mat33(tensor_C, dvc=device)

    def export_particle_x_to_torch(self):
        return wp.to_torch(self.mpm_state.particle_x)

    def export_particle_v_to_torch(self):
        return wp.to_torch(self.mpm_state.particle_v)

    def export_particle_F_to_torch(self):
        F_tensor = wp.to_torch(self.mpm_state.particle_F)
        F_tensor = F_tensor.reshape(-1, 9)
        return F_tensor

    def export_particle_R_to_torch(self, device="cuda:0"):
        with wp.ScopedTimer(
            "compute_R_from_F",
            synchronize=True,
            print=False,
            dict=self.time_profile,
        ):
            wp.launch(
                kernel=compute_R_from_F,
                dim=self.n_particles,
                inputs=[self.mpm_state, self.mpm_model],
                device=device,
            )

        R_tensor = wp.to_torch(self.mpm_state.particle_R)
        R_tensor = R_tensor.reshape(-1, 9)
        return R_tensor

    def export_particle_C_to_torch(self):
        C_tensor = wp.to_torch(self.mpm_state.particle_C)
        C_tensor = C_tensor.reshape(-1, 9)
        return C_tensor

    def export_particle_cov_to_torch(self, device="cuda:0"):
        if not self.mpm_model.update_cov_with_F:
            with wp.ScopedTimer(
                "compute_cov_from_F",
                synchronize=True,
                print=False,
                dict=self.time_profile,
            ):
                wp.launch(
                    kernel=compute_cov_from_F,
                    dim=self.n_particles,
                    inputs=[self.mpm_state, self.mpm_model],
                    device=device,
                )

        cov = wp.to_torch(self.mpm_state.particle_cov)
        return cov

    def print_time_profile(self):
        print("MPM Time profile:")
        for key, value in self.time_profile.items():
            print(key, sum(value))

    # a surface specified by a point and the normal vector
    def add_surface_collider(
        self,
        point,
        normal,
        surface="sticky",
        friction=0.0,
        start_time=0.0,
        end_time=999.0,
    ):
        point = list(point)
        # Normalize normal
        normal_scale = 1.0 / wp.sqrt(float(sum(x**2 for x in normal)))
        normal = list(normal_scale * x for x in normal)

        collider_param = Dirichlet_collider()
        collider_param.start_time = start_time
        collider_param.end_time = end_time

        collider_param.point = wp.vec3(point[0], point[1], point[2])
        collider_param.normal = wp.vec3(normal[0], normal[1], normal[2])

        if surface == "sticky" and friction != 0:
            raise ValueError("friction must be 0 on sticky surfaces.")
        if surface == "sticky":
            collider_param.surface_type = 0
        elif surface == "slip":
            collider_param.surface_type = 1
        elif surface == "cut":
            collider_param.surface_type = 11
        else:
            collider_param.surface_type = 2
        # frictional
        collider_param.friction = friction

        self.collider_params.append(collider_param)

        @wp.kernel
        def collide(
            time: float,
            dt: float,
            state: MPMStateStruct,
            model: MPMModelStruct,
            param: Dirichlet_collider,
        ):
            grid_x, grid_y, grid_z = wp.tid()
            if time >= param.start_time and time < param.end_time:
                offset = wp.vec3(
                    float(grid_x) * model.dx - param.point[0],
                    float(grid_y) * model.dx - param.point[1],
                    float(grid_z) * model.dx - param.point[2],
                )
                n = wp.vec3(param.normal[0], param.normal[1], param.normal[2])
                dotproduct = wp.dot(offset, n)

                if dotproduct < 0.0:
                    if param.surface_type == 0:
                        state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
                            0.0, 0.0, 0.0
                        )
                    elif param.surface_type == 11:
                        if (
                            float(grid_z) * model.dx < 0.4
                            or float(grid_z) * model.dx > 0.53
                        ):
                            state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
                                0.0, 0.0, 0.0
                            )
                        else:
                            v_in = state.grid_v_out[grid_x, grid_y, grid_z]
                            state.grid_v_out[grid_x, grid_y, grid_z] = (
                                wp.vec3(v_in[0], 0.0, v_in[2]) * 0.3
                            )
                    else:
                        v = state.grid_v_out[grid_x, grid_y, grid_z]
                        normal_component = wp.dot(v, n)
                        if param.surface_type == 1:
                            v = (
                                v - normal_component * n
                            )  # Project out all normal component
                        else:
                            v = (
                                v - wp.min(normal_component, 0.0) * n
                            )  # Project out only inward normal component
                        if normal_component < 0.0 and wp.length(v) > 1e-20:
                            v = wp.max(
                                0.0, wp.length(v) + normal_component * param.friction
                            ) * wp.normalize(
                                v
                            )  # apply friction here
                        state.grid_v_out[grid_x, grid_y, grid_z] = v

        self.grid_postprocess.append(collide)
        self.grid_postprocess_sources.append("surface")
        self.modify_bc.append(None)

    # a cubiod is a rectangular cube'
    # centered at `point`
    # dimension is x: point[0]±size[0]
    #              y: point[1]±size[1]
    #              z: point[2]±size[2]
    # all grid nodes lie within the cubiod will have their speed set to velocity
    # the cuboid itself is also moving with const speed = velocity
    # set the speed to zero to fix BC
    def set_velocity_on_cuboid(
        self,
        point,
        size,
        velocity,
        start_time=0.0,
        end_time=999.0,
        reset=0,
    ):
        point = list(point)

        collider_param = Dirichlet_collider()
        collider_param.start_time = start_time
        collider_param.end_time = end_time
        collider_param.point = wp.vec3(point[0], point[1], point[2])
        collider_param.size = size
        collider_param.velocity = wp.vec3(velocity[0], velocity[1], velocity[2])
        # collider_param.threshold = threshold
        collider_param.reset = reset
        self.collider_params.append(collider_param)

        @wp.kernel
        def collide(
            time: float,
            dt: float,
            state: MPMStateStruct,
            model: MPMModelStruct,
            param: Dirichlet_collider,
        ):
            grid_x, grid_y, grid_z = wp.tid()
            if time >= param.start_time and time < param.end_time:
                offset = wp.vec3(
                    float(grid_x) * model.dx - param.point[0],
                    float(grid_y) * model.dx - param.point[1],
                    float(grid_z) * model.dx - param.point[2],
                )
                if (
                    wp.abs(offset[0]) < param.size[0]
                    and wp.abs(offset[1]) < param.size[1]
                    and wp.abs(offset[2]) < param.size[2]
                ):
                    state.grid_v_out[grid_x, grid_y, grid_z] = param.velocity
            elif param.reset == 1:
                if time < param.end_time + 15.0 * dt:
                    state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)

        def modify(time, dt, param: Dirichlet_collider):
            if time >= param.start_time and time < param.end_time:
                param.point = wp.vec3(
                    param.point[0] + dt * param.velocity[0],
                    param.point[1] + dt * param.velocity[1],
                    param.point[2] + dt * param.velocity[2],
                )  # param.point + dt * param.velocity

        self.grid_postprocess.append(collide)
        self.grid_postprocess_sources.append("cuboid")
        self.modify_bc.append(modify)

        @wp.kernel
        def mark_implicit_dirichlet(
            time: float,
            dt: float,
            state: MPMStateStruct,
            model: MPMModelStruct,
            param: Dirichlet_collider,
            mask: wp.array(dtype=int, ndim=3),
            target: wp.array(dtype=wp.vec3, ndim=3),
        ):
            grid_x, grid_y, grid_z = wp.tid()
            if time >= param.start_time and time < param.end_time:
                offset = wp.vec3(
                    float(grid_x) * model.dx - param.point[0],
                    float(grid_y) * model.dx - param.point[1],
                    float(grid_z) * model.dx - param.point[2],
                )
                if (
                    wp.abs(offset[0]) < param.size[0]
                    and wp.abs(offset[1]) < param.size[1]
                    and wp.abs(offset[2]) < param.size[2]
                ):
                    mask[grid_x, grid_y, grid_z] = 1
                    target[grid_x, grid_y, grid_z] = param.velocity
            elif param.reset == 1:
                if time < param.end_time + 15.0 * dt:
                    mask[grid_x, grid_y, grid_z] = 1
                    target[grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)

        self.implicit_grid_constraint_builders.append(mark_implicit_dirichlet)
        self.implicit_grid_constraint_params.append(collider_param)
        self.implicit_grid_constraint_sources.append("cuboid")

    def add_bounding_box(self, start_time=0.0, end_time=999.0):
        # Bounding boxes remain a safety velocity projection/clamp. They are not
        # packed as implicit Dirichlet unknown masks like sticky surfaces/cuboids.
        collider_param = Dirichlet_collider()
        collider_param.start_time = start_time
        collider_param.end_time = end_time

        self.collider_params.append(collider_param)

        @wp.kernel
        def collide(
            time: float,
            dt: float,
            state: MPMStateStruct,
            model: MPMModelStruct,
            param: Dirichlet_collider,
        ):
            grid_x, grid_y, grid_z = wp.tid()
            padding = 3
            if time >= param.start_time and time < param.end_time:
                if grid_x < padding and state.grid_v_out[grid_x, grid_y, grid_z][0] < 0:
                    state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
                        0.0,
                        state.grid_v_out[grid_x, grid_y, grid_z][1],
                        state.grid_v_out[grid_x, grid_y, grid_z][2],
                    )
                if (
                    grid_x >= model.grid_dim_x - padding
                    and state.grid_v_out[grid_x, grid_y, grid_z][0] > 0
                ):
                    state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
                        0.0,
                        state.grid_v_out[grid_x, grid_y, grid_z][1],
                        state.grid_v_out[grid_x, grid_y, grid_z][2],
                    )

                if grid_y < padding and state.grid_v_out[grid_x, grid_y, grid_z][1] < 0:
                    state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
                        state.grid_v_out[grid_x, grid_y, grid_z][0],
                        0.0,
                        state.grid_v_out[grid_x, grid_y, grid_z][2],
                    )
                if (
                    grid_y >= model.grid_dim_y - padding
                    and state.grid_v_out[grid_x, grid_y, grid_z][1] > 0
                ):
                    state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
                        state.grid_v_out[grid_x, grid_y, grid_z][0],
                        0.0,
                        state.grid_v_out[grid_x, grid_y, grid_z][2],
                    )

                if grid_z < padding and state.grid_v_out[grid_x, grid_y, grid_z][2] < 0:
                    state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
                        state.grid_v_out[grid_x, grid_y, grid_z][0],
                        state.grid_v_out[grid_x, grid_y, grid_z][1],
                        0.0,
                    )
                if (
                    grid_z >= model.grid_dim_z - padding
                    and state.grid_v_out[grid_x, grid_y, grid_z][2] > 0
                ):
                    state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
                        state.grid_v_out[grid_x, grid_y, grid_z][0],
                        state.grid_v_out[grid_x, grid_y, grid_z][1],
                        0.0,
                    )

        self.grid_postprocess.append(collide)
        self.grid_postprocess_sources.append("bounding_box")
        self.modify_bc.append(None)

    # particle_v += force/particle_mass * dt
    # this is applied from start_dt, ends after num_dt p2g2p's
    # particle velocity is changed before p2g at each timestep
    def add_impulse_on_particles(
        self,
        force,
        dt,
        point=[1, 1, 1],
        size=[1, 1, 1],
        num_dt=1,
        start_time=0.0,
        device="cuda:0",
    ):
        impulse_param = Impulse_modifier()
        impulse_param.start_time = start_time
        impulse_param.end_time = start_time + dt * num_dt

        impulse_param.point = wp.vec3(point[0], point[1], point[2])
        impulse_param.size = wp.vec3(size[0], size[1], size[2])
        impulse_param.mask = wp.zeros(shape=self.n_particles, dtype=int, device=device)

        impulse_param.force = wp.vec3(
            force[0],
            force[1],
            force[2],
        )

        wp.launch(
            kernel=selection_add_impulse_on_particles,
            dim=self.n_particles,
            inputs=[self.mpm_state, impulse_param],
            device=device,
        )

        self.impulse_params.append(impulse_param)

        @wp.kernel
        def apply_force(
            time: float, dt: float, state: MPMStateStruct, param: Impulse_modifier
        ):
            p = wp.tid()
            if time >= param.start_time and time < param.end_time:
                if param.mask[p] == 1:
                    impulse = wp.vec3(
                        param.force[0] / state.particle_mass[p],
                        param.force[1] / state.particle_mass[p],
                        param.force[2] / state.particle_mass[p],
                    )
                    state.particle_v[p] = state.particle_v[p] + impulse * dt

        self.pre_p2g_operations.append(apply_force)

    def enforce_particle_velocity_translation(
        self, point, size, velocity, start_time, end_time, device="cuda:0"
    ):

        # first select certain particles based on position

        velocity_modifier_params = ParticleVelocityModifier()

        velocity_modifier_params.point = wp.vec3(point[0], point[1], point[2])
        velocity_modifier_params.size = wp.vec3(size[0], size[1], size[2])

        velocity_modifier_params.velocity = wp.vec3(
            velocity[0], velocity[1], velocity[2]
        )

        velocity_modifier_params.start_time = start_time
        velocity_modifier_params.end_time = end_time

        velocity_modifier_params.mask = wp.zeros(
            shape=self.n_particles, dtype=int, device=device
        )

        wp.launch(
            kernel=selection_enforce_particle_velocity_translation,
            dim=self.n_particles,
            inputs=[self.mpm_state, velocity_modifier_params],
            device=device,
        )
        self.particle_velocity_modifier_params.append(velocity_modifier_params)

        @wp.kernel
        def modify_particle_v_before_p2g(
            time: float,
            state: MPMStateStruct,
            velocity_modifier_params: ParticleVelocityModifier,
        ):
            p = wp.tid()
            if (
                time >= velocity_modifier_params.start_time
                and time < velocity_modifier_params.end_time
            ):
                if velocity_modifier_params.mask[p] == 1:
                    state.particle_v[p] = velocity_modifier_params.velocity

        self.particle_velocity_modifiers.append(modify_particle_v_before_p2g)

    # define a cylinder with center point, half_height, radius, normal
    # particles within the cylinder are rotating along the normal direction
    # may also have a translational velocity along the normal direction
    def enforce_particle_velocity_rotation(
        self,
        point,
        normal,
        half_height_and_radius,
        rotation_scale,
        translation_scale,
        start_time,
        end_time,
        device="cuda:0",
    ):

        normal_scale = 1.0 / wp.sqrt(
            float(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2)
        )
        normal = list(normal_scale * x for x in normal)

        velocity_modifier_params = ParticleVelocityModifier()

        velocity_modifier_params.point = wp.vec3(point[0], point[1], point[2])
        velocity_modifier_params.half_height_and_radius = wp.vec2(
            half_height_and_radius[0], half_height_and_radius[1]
        )
        velocity_modifier_params.normal = wp.vec3(normal[0], normal[1], normal[2])

        horizontal_1 = wp.vec3(1.0, 1.0, 1.0)
        if wp.abs(wp.dot(velocity_modifier_params.normal, horizontal_1)) < 0.01:
            horizontal_1 = wp.vec3(0.72, 0.37, -0.67)
        horizontal_1 = (
            horizontal_1
            - wp.dot(horizontal_1, velocity_modifier_params.normal)
            * velocity_modifier_params.normal
        )
        horizontal_1 = horizontal_1 * (1.0 / wp.length(horizontal_1))
        horizontal_2 = wp.cross(horizontal_1, velocity_modifier_params.normal)

        velocity_modifier_params.horizontal_axis_1 = horizontal_1
        velocity_modifier_params.horizontal_axis_2 = horizontal_2

        velocity_modifier_params.rotation_scale = rotation_scale
        velocity_modifier_params.translation_scale = translation_scale

        velocity_modifier_params.start_time = start_time
        velocity_modifier_params.end_time = end_time

        velocity_modifier_params.mask = wp.zeros(
            shape=self.n_particles, dtype=int, device=device
        )

        wp.launch(
            kernel=selection_enforce_particle_velocity_cylinder,
            dim=self.n_particles,
            inputs=[self.mpm_state, velocity_modifier_params],
            device=device,
        )
        self.particle_velocity_modifier_params.append(velocity_modifier_params)

        @wp.kernel
        def modify_particle_v_before_p2g(
            time: float,
            state: MPMStateStruct,
            velocity_modifier_params: ParticleVelocityModifier,
        ):
            p = wp.tid()
            if (
                time >= velocity_modifier_params.start_time
                and time < velocity_modifier_params.end_time
            ):
                if velocity_modifier_params.mask[p] == 1:
                    offset = state.particle_x[p] - velocity_modifier_params.point
                    horizontal_distance = wp.length(
                        offset
                        - wp.dot(offset, velocity_modifier_params.normal)
                        * velocity_modifier_params.normal
                    )
                    cosine = (
                        wp.dot(offset, velocity_modifier_params.horizontal_axis_1)
                        / horizontal_distance
                    )
                    theta = wp.acos(cosine)
                    if wp.dot(offset, velocity_modifier_params.horizontal_axis_2) > 0:
                        theta = theta
                    else:
                        theta = -theta
                    axis1_scale = (
                        -horizontal_distance
                        * wp.sin(theta)
                        * velocity_modifier_params.rotation_scale
                    )
                    axis2_scale = (
                        horizontal_distance
                        * wp.cos(theta)
                        * velocity_modifier_params.rotation_scale
                    )
                    axis_vertical_scale = translation_scale
                    state.particle_v[p] = (
                        axis1_scale * velocity_modifier_params.horizontal_axis_1
                        + axis2_scale * velocity_modifier_params.horizontal_axis_2
                        + axis_vertical_scale * velocity_modifier_params.normal
                    )

        self.particle_velocity_modifiers.append(modify_particle_v_before_p2g)

    # given normal direction, say [0,0,1]
    # gradually release grid velocities from start position to end position
    def release_particles_sequentially(
        self, normal, start_position, end_position, num_layers, start_time, end_time
    ):
        num_layers = 50
        point = [0, 0, 0]
        size = [0, 0, 0]
        axis = -1
        for i in range(3):
            if normal[i] == 0:
                point[i] = 1
                size[i] = 1
            else:
                axis = i
                point[i] = end_position

        half_length_portion = wp.abs(start_position - end_position) / num_layers
        end_time_portion = end_time / num_layers
        for i in range(num_layers):
            size[axis] = half_length_portion * (num_layers - i)
            self.enforce_particle_velocity_translation(
                point=point,
                size=size,
                velocity=[0, 0, 0],
                start_time=start_time,
                end_time=end_time_portion * (i + 1),
            )
