//based on "rect.cpp"
#include "polygon.h"
#include <vector>
#include <cmath>
#include <imgui.h>

namespace USTC_CG
{
// Draw the rectangle using ImGui
void Polygon::draw(const Config& config) const
{
    ImDrawList* draw_list = ImGui::GetWindowDrawList();

    float center_x = (start_point_x_ + end_point_x_) * 0.5f;
    float center_y = (start_point_y_ + end_point_y_) * 0.5f;
    float radius_x = (end_point_x_ - start_point_x_) * 0.5f;
    float radius_y = (end_point_y_ - start_point_y_) * 0.5f;
    constexpr float PI = 3.14159265358979323846f;

    std::vector<ImVec2> points;
    points.reserve(sides_);
    float angle_0;
    if (sides_ % 2 == 1)
    {
        angle_0 = -PI / 2.0f;
    }
    else
    {
        angle_0 = -PI / 2.0f + PI / sides_;
    }
    for (int i = 0; i < sides_; ++i)
    {
        float t = angle_0+2*PI*i/sides_;
        float point_x = config.bias[0] + center_x + radius_x * std::cos(t);
        float point_y = config.bias[1] + center_y + radius_y * std::sin(t);
        points.push_back(ImVec2(point_x, point_y));
    }
    if (style_.filled)
    {
        draw_list->AddConvexPolyFilled(
            points.data(),
            static_cast<int>(points.size()),
            IM_COL32(
                style_.fill_color[0],
                style_.fill_color[1],
                style_.fill_color[2],
                style_.fill_color[3]));
    }

    if (style_.stroke)
    {
        draw_list->AddPolyline(
            points.data(),
            static_cast<int>(points.size()),
            IM_COL32(
                style_.line_color[0],
                style_.line_color[1],
                style_.line_color[2],
                style_.line_color[3]),
            ImDrawFlags_Closed,
            style_.line_thickness);
    }
}

void Polygon::update(float x, float y)
{
    end_point_x_ = x;
    end_point_y_ = y;
}

void Polygon::set_sides(int sides)
{
    sides_ = (sides < 3 ? 3 : sides);
}

bool Polygon::hit_test(float x, float y) const
{
    return false;
}

void Polygon::draw_selected(const Config& config) const
{
    draw(config);
}

}  // namespace USTC_CG
