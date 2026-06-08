#include "line.h"

#include <imgui.h>

namespace USTC_CG
{
// Draw the line using ImGui
void Line::draw(const Config& config) const
{
    ImDrawList* draw_list = ImGui::GetWindowDrawList();

    draw_list->AddLine(
        ImVec2(
            config.bias[0] + start_point_x_, config.bias[1] + start_point_y_),
        ImVec2(config.bias[0] + end_point_x_, config.bias[1] + end_point_y_),
        IM_COL32(
            style_.line_color[0],
            style_.line_color[1],
            style_.line_color[2],
            style_.line_color[3]),
        style_.line_thickness);
}

void Line::update(float x, float y)
{
    end_point_x_ = x;
    end_point_y_ = y;
}

bool Line::hit_test(float x, float y) const
{
    return false;
}

void Line::draw_selected(const Config& config) const
{
    draw(config);
}

}  // namespace USTC_CG