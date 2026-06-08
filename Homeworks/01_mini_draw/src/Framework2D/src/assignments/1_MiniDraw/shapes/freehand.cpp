#include "freehand.h"
#include <vector>
#include <imgui.h>

namespace USTC_CG
{
Freehand::Freehand(std::vector<float> x_list, std::vector<float> y_list)
{
    x_list_ = x_list;
    y_list_ = y_list;
}

void Freehand::draw(const Config& config) const
{
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    std::vector<ImVec2> points;
    points.reserve(x_list_.size());
    for (size_t i = 0; i < x_list_.size(); ++i)
    {
        points.push_back(
            ImVec2(config.bias[0] + x_list_[i], config.bias[1] + y_list_[i]));
    }

    draw_list->AddPolyline(
        points.data(),
        static_cast<int>(points.size()),
        IM_COL32(
            style_.line_color[0],
            style_.line_color[1],
            style_.line_color[2],
            style_.line_color[3]),
        ImDrawFlags_None,
        style_.line_thickness);
}

void Freehand::update(float x, float y)
{
    if (!x_list_.empty())
    {
        x_list_.back() = x;
        y_list_.back() = y;
    }
}

void Freehand::add_control_point(float x, float y)
{
    const float min_dist_sq = 1.0f;

    if (x_list_.empty())
    {
        x_list_.push_back(x);
        y_list_.push_back(y);
        x_list_.push_back(x);
        y_list_.push_back(y);
        accum_dx_ = 0.0f;
        accum_dy_ = 0.0f;
        return;
    }

    accum_dx_ += x - x_list_[x_list_.size() - 2];
    accum_dy_ += y - y_list_[y_list_.size() - 2];

    update(x, y);

    if (accum_dx_ * accum_dx_ + accum_dy_ * accum_dy_ > min_dist_sq)
    {
        x_list_.push_back(x);
        y_list_.push_back(y);
        accum_dx_ = 0.0f;
        accum_dy_ = 0.0f;
    }
}

bool Freehand::hit_test(float x, float y) const
{
    float tol = std::max(4.0f, style_.line_thickness + 2.0f);

    for (size_t i = 0; i + 1 < x_list_.size(); ++i)
    {
        float x1 = x_list_[i];
        float y1 = y_list_[i];
        float x2 = x_list_[i + 1];
        float y2 = y_list_[i + 1];

        float x_min = std::min(x1, x2) - tol;
        float x_max = std::max(x1, x2) + tol;
        if (x < x_min || x > x_max)
            continue;

        float y_min = std::min(y1, y2) - tol;
        float y_max = std::max(y1, y2) + tol;
        if (y < y_min || y > y_max)
            continue;

        return true;
    }

    return false;
}

void Freehand::draw_selected(const Config& config) const
{
    ImDrawList* draw_list = ImGui::GetWindowDrawList();

    std::vector<ImVec2> points;
    points.reserve(x_list_.size());

    for (size_t i = 0; i < x_list_.size(); ++i)
    {
        points.push_back(
            ImVec2(config.bias[0] + x_list_[i], config.bias[1] + y_list_[i]));
    }

    draw_list->AddPolyline(
        points.data(),
        static_cast<int>(points.size()),
        IM_COL32(255, 255, 0, 255),
        ImDrawFlags_None,
        style_.line_thickness + 3.0f);

    draw(config);
}

}  // namespace USTC_CG
