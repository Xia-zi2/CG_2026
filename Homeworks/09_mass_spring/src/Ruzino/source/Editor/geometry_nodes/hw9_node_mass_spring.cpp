#include <pxr/base/gf/vec3f.h>
#include <pxr/base/vt/array.h>

#include <Eigen/Sparse>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <unordered_set>

#include "GCore/Components/MeshComponent.h"
#include "GCore/util_openmesh_bind.h"
#include "geom_node_base.h"
#include "mass_spring/FastMassSpring.h"
#include "mass_spring/MassSpring.h"
#include "mass_spring/utils.h"

struct MassSpringStorage {
    // This node must keep the solver object between frames; otherwise the
    // simulation is reconstructed every evaluation and x^{n+1} never advances.
    constexpr static bool has_storage = false;
    std::shared_ptr<USTC_CG::mass_spring::MassSpring> mass_spring;
};

NODE_DEF_OPEN_SCOPE
NODE_DECLARATION_FUNCTION(hw9_mass_spring)
{
    b.add_input<Geometry>("Mesh");

    b.add_input<float>("stiffness").default_val(1000).min(100).max(10000);
    b.add_input<float>("h").default_val(0.0333333333f).min(1e-5).max(0.5);
    b.add_input<float>("damping").default_val(0.995).min(0.0).max(1.0);
    b.add_input<float>("gravity").default_val(-9.8).min(-20.).max(20.);

    b.add_input<float>("collision penalty_k")
        .default_val(10000)
        .min(0)
        .max(100000);
    b.add_input<float>("collision scale factor")
        .default_val(1.1)
        .min(1.0)
        .max(2.0);
    b.add_input<float>("sphere radius").default_val(0.4).min(0.0).max(5.0);
    b.add_input<float>("sphere center x").default_val(0.0).min(-5.0).max(5.0);
    b.add_input<float>("sphere center y").default_val(0.0).min(-5.0).max(5.0);
    b.add_input<float>("sphere center z").default_val(-1.0).min(-5.0).max(5.0);
    b.add_input<float>("collision restitution")
        .default_val(0.0)
        .min(0.0)
        .max(1.0);
    b.add_input<float>("collision friction")
        .default_val(0.3)
        .min(0.0)
        .max(2.0);
    b.add_input<int>("collision projected velocity update")
        .default_val(1)
        .min(0)
        .max(1);
    b.add_input<int>("time integrator type")
        .default_val(0)
        .min(0)
        .max(1);  // 0 implicit Euler, 1 semi-implicit Euler
    b.add_input<int>("enable time profiling").default_val(0).min(0).max(1);
    b.add_input<int>("enable damping").default_val(1).min(0).max(1);
    b.add_input<int>("enable debug output").default_val(0).min(0).max(1);

    b.add_input<int>("enable Liu13").default_val(0).min(0).max(1);
    b.add_input<int>("Liu13 max iter").default_val(30).min(1).max(200);
    b.add_input<int>("collision mode")
        .default_val(0)
        .min(0)
        .max(3);  // 0 none, 1 penalty force, 2 penalty energy, 3 distance impulse

    b.add_output<Geometry>("Output Mesh");
}

NODE_EXECUTION_FUNCTION(hw9_mass_spring)
{
    using namespace Eigen;
    using namespace USTC_CG::mass_spring;

    auto& global_payload = params.get_global_payload<GeomPayload&>();
    auto current_time = global_payload.current_time;

    auto& storage = params.get_storage<MassSpringStorage&>();
    auto& mass_spring = storage.mass_spring;

    auto geometry = params.get_input<Geometry>("Mesh");
    auto mesh = geometry.get_component<MeshComponent>();
    if (!mesh || mesh->get_face_vertex_counts().size() == 0) {
        throw std::runtime_error("Mass Spring: Need valid Geometry Input.");
    }

    std::cout << "Mass Spring: current time = " << current_time << std::endl;

    const bool enable_liu13 = params.get_input<int>("enable Liu13") == 1;
    bool need_rebuild = (current_time == 0 || !mass_spring);

    if (mass_spring) {
        const bool current_is_fast =
            (std::dynamic_pointer_cast<FastMassSpring>(mass_spring) != nullptr);
        if (current_is_fast != enable_liu13) {
            need_rebuild = true;
        }

        const int mesh_vertex_count =
            static_cast<int>(mesh->get_vertices().size());
        if (mass_spring->getX().rows() != mesh_vertex_count) {
            need_rebuild = true;
        }
    }

    if (need_rebuild) {
        mass_spring.reset();

        auto edges = get_edges(usd_faces_to_eigen(
            mesh->get_face_vertex_counts(), mesh->get_face_vertex_indices()));
        auto vertices = usd_vertices_to_eigen(mesh->get_vertices());

        const float k = params.get_input<float>("stiffness");
        const float h = std::max(1e-5f, params.get_input<float>("h"));

        if (enable_liu13) {
            mass_spring = std::make_shared<FastMassSpring>(vertices, edges, k, h);
        }
        else {
            mass_spring = std::make_shared<MassSpring>(vertices, edges);
        }
    }

    if (!mass_spring) {
        throw std::runtime_error("Mass Spring: failed to initialize solver.");
    }

    // Sync UI parameters every frame. This is important for FastMassSpring:
    // if h or stiffness changes, its cached prefactorized matrix must be
    // invalidated and rebuilt in FastMassSpring::step().
    mass_spring->stiffness = params.get_input<float>("stiffness");
    mass_spring->h = std::max(1e-5f, params.get_input<float>("h"));
    mass_spring->gravity = { 0, 0, params.get_input<float>("gravity") };
    mass_spring->damping = params.get_input<float>("damping");

    mass_spring->collision_penalty_k =
        params.get_input<float>("collision penalty_k");
    mass_spring->collision_scale_factor =
        params.get_input<float>("collision scale factor");
    mass_spring->sphere_center = {
        params.get_input<float>("sphere center x"),
        params.get_input<float>("sphere center y"),
        params.get_input<float>("sphere center z")
    };
    mass_spring->sphere_radius = params.get_input<float>("sphere radius");
    mass_spring->collision_restitution =
        params.get_input<float>("collision restitution");
    mass_spring->collision_friction =
        params.get_input<float>("collision friction");
    mass_spring->sync_collision_velocity_with_projection =
        params.get_input<int>("collision projected velocity update") == 1;

    int collision_mode = params.get_input<int>("collision mode");
    if (collision_mode < 0) {
        collision_mode = 0;
    }
    if (collision_mode > 3) {
        collision_mode = 3;
    }
    mass_spring->collision_mode =
        static_cast<MassSpring::CollisionMode>(collision_mode);

    mass_spring->enable_damping =
        params.get_input<int>("enable damping") == 1 ? true : false;
    mass_spring->time_integrator =
        params.get_input<int>("time integrator type") == 0
            ? MassSpring::IMPLICIT_EULER
            : MassSpring::SEMI_IMPLICIT_EULER;
    mass_spring->enable_time_profiling =
        params.get_input<int>("enable time profiling") == 1 ? true : false;
    mass_spring->enable_debug_output =
        params.get_input<int>("enable debug output") == 1 ? true : false;

    if (auto fast = std::dynamic_pointer_cast<FastMassSpring>(mass_spring)) {
        int max_iter = params.get_input<int>("Liu13 max iter");
        if (max_iter < 1) {
            max_iter = 1;
        }
        fast->max_iter = static_cast<unsigned>(max_iter);
    }

    if (current_time != 0) {
        mass_spring->step();
    }

    mesh->set_vertices(eigen_to_usd_vertices(mass_spring->getX()));
    params.set_output("Output Mesh", geometry);
    return true;
}

NODE_DECLARATION_UI(hw9_mass_spring);
NODE_DEF_CLOSE_SCOPE
