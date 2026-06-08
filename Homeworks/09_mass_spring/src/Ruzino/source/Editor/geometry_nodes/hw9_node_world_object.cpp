#include <iostream>
#include <memory>
#include <string>

#include "GCore/Components/MeshComponent.h"
#include "geom_node_base.h"
#include "mass_spring/HW9WorldObject.h"

namespace {
static void translate_geometry_vertices(Ruzino::Geometry& geometry, const glm::vec3& t)
{
    if (std::abs(t.x) < 1e-12f && std::abs(t.y) < 1e-12f && std::abs(t.z) < 1e-12f) {
        return;
    }
    auto mesh = geometry.get_component<Ruzino::MeshComponent>();
    if (!mesh) return;
    auto vertices = mesh->get_vertices();
    for (auto& v : vertices) {
        v += t;
    }
    mesh->set_vertices(vertices);
}
} // namespace

NODE_DEF_OPEN_SCOPE

NODE_DECLARATION_FUNCTION(hw9_world_object)
{
    b.add_input<Geometry>("Mesh");

    // 0 = dynamic deformable, 1 = kinematic mesh, 2 = static collider.
    b.add_input<int>("object type").default_val(0).min(0).max(2);
    b.add_input<std::string>("name").default_val("object");

    // Optional initial offset. This is useful for testing when you do not want
    // to insert a separate transform node before hw9_world_object.
    b.add_input<float>("initial tx").default_val(0.0f).min(-10.0f).max(10.0f);
    b.add_input<float>("initial ty").default_val(0.0f).min(-10.0f).max(10.0f);
    b.add_input<float>("initial tz").default_val(0.0f).min(-10.0f).max(10.0f);

    // Physical parameters. For kinematic/static objects, mass/stiffness/damping
    // are ignored; restitution/friction are still used in collision response.
    b.add_input<float>("mass").default_val(1.0f).min(0.001f).max(100.0f);
    b.add_input<float>("stiffness").default_val(1000.0f).min(1.0f).max(100000.0f);
    b.add_input<float>("damping").default_val(0.995f).min(0.0f).max(1.0f);
    b.add_input<float>("restitution").default_val(0.0f).min(0.0f).max(1.0f);
    b.add_input<float>("friction").default_val(0.3f).min(0.0f).max(2.0f);

    b.add_output<USTC_CG::mass_spring::HW9WorldObject>("Object");
    b.add_output<Geometry>("Geometry");
}

NODE_EXECUTION_FUNCTION(hw9_world_object)
{
    using namespace USTC_CG::mass_spring;

    auto geometry = params.get_input<Geometry>("Mesh");
    geometry.apply_transform();

    glm::vec3 offset(
        params.get_input<float>("initial tx"),
        params.get_input<float>("initial ty"),
        params.get_input<float>("initial tz"));
    translate_geometry_vertices(geometry, offset);

    auto mesh = geometry.get_component<Ruzino::MeshComponent>();
    if (!mesh || mesh->get_face_vertex_counts().empty()) {
        std::cerr << "[hw9_world_object] Input Geometry has no MeshComponent or no faces." << std::endl;
        return false;
    }

    int type = params.get_input<int>("object type");
    if (type < 0) type = 0;
    if (type > 2) type = 2;

    HW9WorldObject object;
    object.geometry = geometry;
    object.params.object_type = type;
    object.params.name = params.get_input<std::string>("name");
    object.params.mass = params.get_input<float>("mass");
    object.params.stiffness = params.get_input<float>("stiffness");
    object.params.damping = params.get_input<float>("damping");
    object.params.restitution = params.get_input<float>("restitution");
    object.params.friction = params.get_input<float>("friction");

    params.set_output("Object", object);
    params.set_output("Geometry", geometry);
    return true;
}

NODE_DECLARATION_UI(hw9_world_object);
NODE_DEF_CLOSE_SCOPE
