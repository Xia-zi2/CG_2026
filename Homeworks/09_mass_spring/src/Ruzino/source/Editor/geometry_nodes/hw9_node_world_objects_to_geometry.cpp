#include <memory>
#include <vector>
#include <iostream>

#include "GCore/Components/MeshComponent.h"
#include "geom_node_base.h"
#include "mass_spring/HW9WorldObject.h"

namespace {
using USTC_CG::mass_spring::HW9WorldObjectList;

static Ruzino::Geometry merge_geometry(const HW9WorldObjectList& a, const HW9WorldObjectList& b)
{
    Ruzino::Geometry merged_geometry;
    auto merged_mesh = std::make_shared<Ruzino::MeshComponent>(&merged_geometry);
    bool has_mesh = false;

    auto append_list = [&](const HW9WorldObjectList& list) {
        for (const auto& object : list.objects) {
            auto geometry = object.geometry;
            geometry.apply_transform();
            auto mesh_component = geometry.get_component<Ruzino::MeshComponent>();
            if (mesh_component) {
                merged_mesh->append_mesh(mesh_component);
                has_mesh = true;
            }
        }
    };

    append_list(a);
    append_list(b);

    if (has_mesh) merged_geometry.attach_component(merged_mesh);
    return merged_geometry;
}
} // namespace

NODE_DEF_OPEN_SCOPE

NODE_DECLARATION_FUNCTION(hw9_world_objects_to_geometry)
{
    // Output from simulation_out goes here.
    b.add_input<USTC_CG::mass_spring::HW9WorldObjectList>("Simulated Object List").optional(true);
    // Direct kinematic/static object list goes here for final rendering/write.
    b.add_input<USTC_CG::mass_spring::HW9WorldObjectList>("Collider Object List").optional(true);

    b.add_output<Geometry>("Geometry");
}

NODE_EXECUTION_FUNCTION(hw9_world_objects_to_geometry)
{
    USTC_CG::mass_spring::HW9WorldObjectList sim;
    USTC_CG::mass_spring::HW9WorldObjectList col;

    try {
        sim = params.get_input<USTC_CG::mass_spring::HW9WorldObjectList>("Simulated Object List");
    }
    catch (...) {
        sim.objects.clear();
    }

    try {
        col = params.get_input<USTC_CG::mass_spring::HW9WorldObjectList>("Collider Object List");
    }
    catch (...) {
        col.objects.clear();
    }

    params.set_output("Geometry", merge_geometry(sim, col));
    return true;
}

NODE_DECLARATION_UI(hw9_world_objects_to_geometry);
NODE_DEF_CLOSE_SCOPE
