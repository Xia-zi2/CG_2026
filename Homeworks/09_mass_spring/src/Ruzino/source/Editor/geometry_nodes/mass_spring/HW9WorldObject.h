#pragma once

#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "GCore/GOP.h"

namespace USTC_CG::mass_spring {

enum HW9WorldObjectType {
    DYNAMIC_DEFORMABLE = 0,  // simulated by mass-spring, deforms
    KINEMATIC_MESH = 1,      // moves from external mesh / prescribed motion, no deformation by solver
    STATIC_COLLIDER = 2      // fixed collider, no deformation
};

struct HW9WorldObjectParams {
    int object_type = DYNAMIC_DEFORMABLE;
    double mass = 1.0;
    double stiffness = 1000.0;
    double damping = 0.995;
    double restitution = 0.0;
    double friction = 0.3;
    std::string name = "object";
};

struct HW9WorldObject {
    Ruzino::Geometry geometry;
    HW9WorldObjectParams params;

    // Per-vertex velocity stored in the simulation payload.
    // Dynamic objects need this to continue simulation after simulation_in/out.
    // Kinematic objects may leave it empty; HW9World estimates velocity from
    // consecutive external meshes when available.
    std::vector<glm::vec3> velocities;
};

struct HW9WorldObjectList {
    std::vector<HW9WorldObject> objects;
};

} // namespace USTC_CG::mass_spring
