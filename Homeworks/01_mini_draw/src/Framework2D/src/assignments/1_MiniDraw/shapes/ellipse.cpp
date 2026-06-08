//based on "rect.cpp"
#include "ellipse.h"
#include <cmath>
#include <algorithm>
#include <imgui.h>

namespace USTC_CG
{
// 
void Ellipse::draw(const Config& config) const
{
    float center_x = (start_point_x_ + end_point_x_) * 0.5f;
    float center_y = (start_point_y_ + end_point_y_) * 0.5f;
    float radius_x = std::abs(end_point_x_ - start_point_x_) * 0.5f;
    float radius_y = std::abs(end_point_y_ - start_point_y_) * 0.5f;

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    if (style_.filled)
    {
        draw_list->AddEllipseFilled(
            ImVec2(config.bias[0] + center_x, config.bias[1] + center_y),
            ImVec2(radius_x, radius_y),
            IM_COL32(
                style_.fill_color[0],
                style_.fill_color[1],
                style_.fill_color[2],
                style_.fill_color[3]),
            0.f,
            ImDrawFlags_None);
    }

    if (style_.stroke)
    {
        draw_list->AddEllipse(
            ImVec2(config.bias[0] + center_x, config.bias[1] + center_y),
            ImVec2(radius_x, radius_y),
            IM_COL32(
                style_.line_color[0],
                style_.line_color[1],
                style_.line_color[2],
                style_.line_color[3]),
            0.f,
            ImDrawFlags_None,
            style_.line_thickness);
    }
}

void Ellipse::update(float x, float y)
{
    end_point_x_ = x;
    end_point_y_ = y;
}

// For ellipse selection, use a pixel-based tolerance.
// A point is considered selected if it lies between
// an expanded ellipse and a shrunken ellipse.
// This keeps hit-testing consistent for both small and large ellipses.
bool Ellipse::hit_test(float x, float y) const
{
    float center_x = (start_point_x_ + end_point_x_) * 0.5f;
    float center_y = (start_point_y_ + end_point_y_) * 0.5f;
    float radius_x = std::abs(end_point_x_ - start_point_x_) * 0.5f;
    float radius_y = std::abs(end_point_y_ - start_point_y_) * 0.5f;

    float tol = std::max(4.0f, style_.line_thickness + 2.0f);
    const float eps = 1e-4f;

    //too small as a point
    if (radius_x < eps && radius_y < eps)
    {
        float dx = x - center_x;
        float dy = y - center_y;
        return dx * dx + dy * dy <= tol * tol;
    }

    //too small
    if (radius_x < eps)
    {
        float y_min = std::min(start_point_y_, end_point_y_);
        float y_max = std::max(start_point_y_, end_point_y_);
        return std::abs(x - center_x) <= tol && y >= y_min - tol &&
               y <= y_max + tol;
    }

    //too small
    if (radius_y < eps)
    {
        float x_min = std::min(start_point_x_, end_point_x_);
        float x_max = std::max(start_point_x_, end_point_x_);
        return std::abs(y - center_y) <= tol && x >= x_min - tol &&
               x <= x_max + tol;
    }

    float dx = x - center_x;
    float dy = y - center_y;

    float value =
        (dx * dx) / (radius_x * radius_x) + (dy * dy) / (radius_y * radius_y);

    if (style_.filled)
    {
        return value <= 1.0f;
    }

    float outer_rx = radius_x + tol;
    float outer_ry = radius_y + tol;

    float outer_value =
        (dx * dx) / (outer_rx * outer_rx) + (dy * dy) / (outer_ry * outer_ry);

    float inner_rx = std::max(eps, radius_x - tol);
    float inner_ry = std::max(eps, radius_y - tol);

    float inner_value =
        (dx * dx) / (inner_rx * inner_rx) + (dy * dy) / (inner_ry * inner_ry);

    return outer_value <= 1.0f && inner_value >= 1.0f;
}

void Ellipse::draw_selected(const Config& config) const
{
    ImDrawList* draw_list = ImGui::GetWindowDrawList();

    float center_x = (start_point_x_ + end_point_x_) * 0.5f;
    float center_y = (start_point_y_ + end_point_y_) * 0.5f;
    float radius_x = std::abs(end_point_x_ - start_point_x_) * 0.5f;
    float radius_y = std::abs(end_point_y_ - start_point_y_) * 0.5f;

    draw_list->AddEllipse(
        ImVec2(config.bias[0] + center_x, config.bias[1] + center_y),
        ImVec2(radius_x, radius_y),
        IM_COL32(255, 255, 0, 255),
        0.0f,
        0,
        style_.line_thickness + 3.0f);

    draw(config);
}

}  // namespace USTC_CG
