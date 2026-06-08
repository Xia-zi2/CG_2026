#include "rect.h"
#include <algorithm>
#include <imgui.h>
#include <cmath>

namespace USTC_CG
{
void rotate_point(float& x, float& y, float center_x, float center_y, float angle)
{
    float dx = x - center_x;
    float dy = y - center_y;
    float c = std::cos(angle);
    float s = std::sin(angle);
    float rx = c * dx - s * dy;
    float ry = s * dx + c * dy;
    x = center_x + rx;
    y = center_y + ry;
}

// Draw the rectangle using ImGui
void Rect::draw(const Config& config) const
{
    ImDrawList* draw_list = ImGui::GetWindowDrawList();

    if (style_.filled)
    {
        draw_list->AddRectFilled(
            ImVec2(
                config.bias[0] + start_point_x_,
                config.bias[1] + start_point_y_),
            ImVec2(
                config.bias[0] + end_point_x_, config.bias[1] + end_point_y_),
            IM_COL32(
                style_.fill_color[0],
                style_.fill_color[1],
                style_.fill_color[2],
                style_.fill_color[3]),
            0.f,  // No rounding of corners
            ImDrawFlags_None);
    }

    if (style_.stroke)
    {
        draw_list->AddRect(
            ImVec2(
                config.bias[0] + start_point_x_,
                config.bias[1] + start_point_y_),
            ImVec2(
                config.bias[0] + end_point_x_, config.bias[1] + end_point_y_),
            IM_COL32(
                style_.line_color[0],
                style_.line_color[1],
                style_.line_color[2],
                style_.line_color[3]),
            0.f,  // No rounding of corners
            ImDrawFlags_None,
            style_.line_thickness);
    }
}

void Rect::update(float x, float y)
{
    end_point_x_ = x;
    end_point_y_ = y;
}

// Selection detection:
// hit the border vicinity to select;
// for filled shapes, clicking inside also selects.
bool Rect::hit_test(float x, float y) const
{
    float x_min = std::min(start_point_x_, end_point_x_);
    float x_max = std::max(start_point_x_, end_point_x_);
    float y_min = std::min(start_point_y_, end_point_y_);
    float y_max = std::max(start_point_y_, end_point_y_);

    if (style_.filled)
    {
        return x >= x_min && x <= x_max && y >= y_min && y <= y_max;
    }

    float tol = std::max(4.0f, style_.line_thickness + 2.0f);

    bool near_left =
        std::abs(x - x_min) <= tol && y >= y_min - tol && y <= y_max + tol;
    bool near_right =
        std::abs(x - x_max) <= tol && y >= y_min - tol && y <= y_max + tol;
    bool near_top =
        std::abs(y - y_min) <= tol && x >= x_min - tol && x <= x_max + tol;
    bool near_bottom =
        std::abs(y - y_max) <= tol && x >= x_min - tol && x <= x_max + tol;

    return near_left || near_right || near_top || near_bottom;
}

void Rect::draw_selected(const Config& config) const
{
    ImDrawList* draw_list = ImGui::GetWindowDrawList();

    ImVec2 p1(config.bias[0] + start_point_x_, config.bias[1] + start_point_y_);
    ImVec2 p2(config.bias[0] + end_point_x_, config.bias[1] + end_point_y_);

    draw_list->AddRect(
        p1,
        p2,
        IM_COL32(255, 255, 0, 255),
        0.0f,
        ImDrawFlags_None,
        style_.line_thickness + 3.0f);

    draw(config);
}

}  // namespace USTC_CG
