#include "fisheye.h"

#include <cmath>

namespace USTC_CG
{

ImVec2 Fisheye::warp_impl(const ImVec2& p) const
{
    const float center_x = canvas_width_ / 2.0f;
    const float center_y = canvas_height_ / 2.0f;
    const float dx = p.x - center_x;
    const float dy = p.y - center_y;
    const float distance = std::sqrt(dx * dx + dy * dy);

    const float new_distance = std::sqrt(distance) * 10.0f;

    if (distance <= eps_)
    {
        return ImVec2(center_x, center_y);
    }

    const float ratio = new_distance / distance;
    const float new_x = center_x + dx * ratio;
    const float new_y = center_y + dy * ratio;

    return ImVec2(new_x, new_y);
}

}  // namespace USTC_CG