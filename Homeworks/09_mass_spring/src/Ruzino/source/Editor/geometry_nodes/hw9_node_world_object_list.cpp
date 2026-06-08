#include <vector>

#include "geom_node_base.h"
#include "mass_spring/HW9WorldObject.h"

NODE_DEF_OPEN_SCOPE

NODE_DECLARATION_FUNCTION(hw9_world_object_list)
{
    b.add_input_group<USTC_CG::mass_spring::HW9WorldObject>("Objects")
        .set_runtime_dynamic(true);
    b.add_output<USTC_CG::mass_spring::HW9WorldObjectList>("Object List");
}

NODE_EXECUTION_FUNCTION(hw9_world_object_list)
{
    using namespace USTC_CG::mass_spring;
    auto objects = params.get_input_group<HW9WorldObject>("Objects");

    HW9WorldObjectList list;
    list.objects.reserve(objects.size());
    for (const auto& object : objects) {
        list.objects.push_back(object);
    }

    params.set_output("Object List", list);
    return true;
}

NODE_DECLARATION_UI(hw9_world_object_list);
NODE_DEF_CLOSE_SCOPE
