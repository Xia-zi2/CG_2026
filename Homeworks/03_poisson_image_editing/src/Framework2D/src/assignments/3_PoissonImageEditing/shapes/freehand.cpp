#include "freehand.h"

#include <imgui.h>

#include <algorithm>
#include <cmath>

namespace USTC_CG
{
void Freehand::draw(const Config& config) const
{
    if (points_.empty())
        return;

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    ImU32 color = IM_COL32(
        config.line_color[0],
        config.line_color[1],
        config.line_color[2],
        config.line_color[3]);

    for (size_t i = 0; i < points_.size() - 1; ++i)
    {
        draw_list->AddLine(
            ImVec2(
                config.bias[0] + points_[i].first,
                config.bias[1] + points_[i].second),
            ImVec2(
                config.bias[0] + points_[i + 1].first,
                config.bias[1] + points_[i + 1].second),
            color,
            config.line_thickness);
    }

    if (is_closed_ && points_.size() > 2)
    {
        draw_list->AddLine(
            ImVec2(
                config.bias[0] + points_.back().first,
                config.bias[1] + points_.back().second),
            ImVec2(
                config.bias[0] + points_.front().first,
                config.bias[1] + points_.front().second),
            color,
            config.line_thickness);
    }
}

void Freehand::update(float x, float y)
{
    if (!points_.empty())
    {
        float dx = x - points_.back().first;
        float dy = y - points_.back().second;
        if (dx * dx + dy * dy < 4.0f)
        {
            return;
        }
    }
    points_.push_back({ x, y });
}

std::vector<std::pair<int, int>> Freehand::get_interior_pixels() const
{
    std::vector<std::pair<int, int>> interior_pixels;
    if (points_.size() < 3)
        return interior_pixels;

    int min_y = static_cast<int>(std::floor(points_[0].second));
    int max_y = static_cast<int>(std::ceil(points_[0].second));

    for (const auto& p : points_)
    {
        min_y = std::min(min_y, static_cast<int>(std::floor(p.second)));
        max_y = std::max(max_y, static_cast<int>(std::ceil(p.second)));
    }

    for (int y = min_y; y <= max_y; ++y)
    {
        std::vector<int> intersections;

        for (size_t i = 0; i < points_.size(); ++i)
        {
            const auto& p1 = points_[i];
            const auto& p2 = points_[(i + 1) % points_.size()];  

            float y1 = p1.second, y2 = p2.second;
            float x1 = p1.first, x2 = p2.first;
            if (y1 > y2)
            {
                std::swap(y1, y2);
                std::swap(x1, x2);
            }

            if (y1 == y2)
                continue;

            if (y >= y1 && y < y2)
            {
                float x = x1 + (y - y1) * (x2 - x1) / (y2 - y1);
                intersections.push_back(static_cast<int>(std::round(x)));
            }
        }

        std::sort(intersections.begin(), intersections.end());

        for (size_t i = 0; i + 1 < intersections.size(); i += 2)
        {
            int start_x = intersections[i];
            int end_x = intersections[i + 1];

            for (int x = start_x; x <= end_x; ++x)
            {
                interior_pixels.push_back({ x, y });
            }
        }
    }

    return interior_pixels;
}
void Freehand::finish_drawing()
{
    is_closed_ = true;
}

}  // namespace USTC_CG