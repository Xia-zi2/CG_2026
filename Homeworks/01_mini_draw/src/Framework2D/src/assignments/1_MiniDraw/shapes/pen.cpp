#include "pen.h"
#include <vector>
#include <imgui.h>

namespace USTC_CG
{
Pen::Pen(std::vector<float> x_list, std::vector<float> y_list)
{
    x_list_ = x_list;
    y_list_ = y_list;
}

void Pen::set_closed(bool closed)
{
    is_closed_ = closed;
}

void Pen::draw(const Config& config) const
{
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    std::vector<ImVec2> points;
    points.reserve(x_list_.size());
    for (size_t i = 0; i < x_list_.size(); ++i)
    {
        points.push_back(
            ImVec2(config.bias[0] + x_list_[i], config.bias[1] + y_list_[i]));
    }

    ImDrawFlags flags = is_closed_ ? ImDrawFlags_Closed : ImDrawFlags_None;


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
            flags,
            style_.line_thickness);
    }
}

void Pen::update(float x, float y)
{
    if (!x_list_.empty())
    {
        x_list_.back() = x;
        y_list_.back() = y;
    }
}

void Pen::add_control_point(float x, float y)
{
    if (x_list_.empty())
    {
        x_list_.push_back(x);
        y_list_.push_back(y);
        x_list_.push_back(x);
        y_list_.push_back(y);
    }
    else
    {
        x_list_.back() = x;
        y_list_.back() = y;
        x_list_.push_back(x);
        y_list_.push_back(y);
    }
}

bool Pen::hit_test(float x, float y) const
{
    return false;
}

void Pen::draw_selected(const Config& config) const
{
    draw(config);
}

}  // namespace USTC_CG
