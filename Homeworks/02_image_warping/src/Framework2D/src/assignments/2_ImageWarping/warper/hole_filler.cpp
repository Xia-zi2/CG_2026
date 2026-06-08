#include "hole_filler.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace USTC_CG
{

void HoleFiller::fill(
    const Image& warped_image,
    Image& output_image,
    const Mask& written_mask,
    const Mask& coverage_mask) const
{
    output_image = warped_image;

    switch (type_)
    {
        case Type::ANN:
        default:
            fill_ann(warped_image, output_image, written_mask, coverage_mask);
            break;
    }
}

int HoleFiller::count_known_neighbors(
    int x,
    int y,
    const Mask& known_mask,
    const Mask& coverage_mask,
    int radius) const
{
    const int height = static_cast<int>(known_mask.size());
    const int width = height > 0 ? static_cast<int>(known_mask[0].size()) : 0;

    int count = 0;
    for (int dy = -radius; dy <= radius; ++dy)
    {
        for (int dx = -radius; dx <= radius; ++dx)
        {
            if (dx == 0 && dy == 0)
            {
                continue;
            }
            const int nx = x + dx;
            const int ny = y + dy;
            if (nx < 0 || nx >= width || ny < 0 || ny >= height)
            {
                continue;
            }
            if (coverage_mask[ny][nx] && known_mask[ny][nx])
            {
                ++count;
            }
        }
    }
    return count;
}

int HoleFiller::count_coverage_neighbors(
    int x,
    int y,
    const Mask& coverage_mask,
    int radius) const
{
    const int height = static_cast<int>(coverage_mask.size());
    const int width =
        height > 0 ? static_cast<int>(coverage_mask[0].size()) : 0;

    int count = 0;
    for (int dy = -radius; dy <= radius; ++dy)
    {
        for (int dx = -radius; dx <= radius; ++dx)
        {
            const int nx = x + dx;
            const int ny = y + dy;
            if (nx < 0 || nx >= width || ny < 0 || ny >= height)
            {
                continue;
            }
            if (coverage_mask[ny][nx])
            {
                ++count;
            }
        }
    }
    return count;
}

bool HoleFiller::is_fillable_hole(
    int x,
    int y,
    const Mask& known_mask,
    const Mask& coverage_mask) const
{
    if (!coverage_mask[y][x] || known_mask[y][x])
    {
        return false;
    }

    // 边界保护：coverage 太稀疏时，说明更像图像外背景，不补。
    const int coverage_neighbors =
        count_coverage_neighbors(x, y, coverage_mask, coverage_guard_radius_);
    if (coverage_neighbors < min_coverage_neighbors_)
    {
        return false;
    }

    // 当前附近至少要有一些已知样本，避免隔着很远硬补。
    const int known_neighbors = count_known_neighbors(
        x, y, known_mask, coverage_mask, known_neighbor_radius_);
    return known_neighbors >= min_known_neighbors_;
}

void HoleFiller::fill_ann(
    const Image& warped_image,
    Image& output_image,
    const Mask& written_mask,
    const Mask& coverage_mask) const
{
    const int height = warped_image.height();
    const int width = warped_image.width();
    if (height <= 0 || width <= 0)
    {
        return;
    }

    Mask known_mask = written_mask;

    for (int iter = 0; iter < max_iterations_; ++iter)
    {
        std::vector<Sample> samples;
        samples.reserve(width * height / 2);

        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                if (coverage_mask[y][x] && known_mask[y][x])
                {
                    samples.push_back({ x, y, output_image.get_pixel(x, y) });
                }
            }
        }

        if (samples.empty())
        {
            return;
        }

        Image temp(output_image);
        Mask next_known_mask = known_mask;
        bool changed = false;

        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                if (!is_fillable_hole(x, y, known_mask, coverage_mask))
                {
                    continue;
                }

                std::vector<std::pair<float, const Sample*>> neighbors;
                neighbors.reserve(samples.size());
                const float r2_limit = max_search_radius_ * max_search_radius_;

                for (const auto& s : samples)
                {
                    const float dx = static_cast<float>(x - s.x);
                    const float dy = static_cast<float>(y - s.y);
                    const float d2 = dx * dx + dy * dy;
                    if (d2 > r2_limit)
                    {
                        continue;
                    }
                    neighbors.emplace_back(d2, &s);
                }

                if (static_cast<int>(neighbors.size()) < min_known_neighbors_)
                {
                    continue;
                }

                std::nth_element(
                    neighbors.begin(),
                    neighbors.begin() +
                        std::min(knn_k_, static_cast<int>(neighbors.size())) -
                        1,
                    neighbors.end(),
                    [](const auto& a, const auto& b)
                    { return a.first < b.first; });

                const int use_k =
                    std::min(knn_k_, static_cast<int>(neighbors.size()));
                float sum_w = 0.0f;
                float sum_r = 0.0f;
                float sum_g = 0.0f;
                float sum_b = 0.0f;

                bool exact_hit = false;
                std::vector<unsigned char> exact_color(3, 0);

                for (int i = 0; i < use_k; ++i)
                {
                    const float d2 = neighbors[i].first;
                    const auto* sample = neighbors[i].second;
                    if (d2 <= eps_)
                    {
                        exact_hit = true;
                        exact_color = sample->color;
                        break;
                    }

                    const float w = 1.0f / (d2 + eps_);
                    sum_w += w;
                    sum_r += w * static_cast<float>(sample->color[0]);
                    sum_g += w * static_cast<float>(sample->color[1]);
                    sum_b += w * static_cast<float>(sample->color[2]);
                }

                if (exact_hit)
                {
                    temp.set_pixel(x, y, exact_color);
                    next_known_mask[y][x] = 1;
                    changed = true;
                    continue;
                }

                if (sum_w <= eps_)
                {
                    continue;
                }

                temp.set_pixel(
                    x,
                    y,
                    {
                        static_cast<unsigned char>(
                            std::clamp(sum_r / sum_w, 0.0f, 255.0f)),
                        static_cast<unsigned char>(
                            std::clamp(sum_g / sum_w, 0.0f, 255.0f)),
                        static_cast<unsigned char>(
                            std::clamp(sum_b / sum_w, 0.0f, 255.0f)),
                    });
                next_known_mask[y][x] = 1;
                changed = true;
            }
        }

        output_image = std::move(temp);
        known_mask = std::move(next_known_mask);

        if (!changed)
        {
            break;
        }
    }
}

}  // namespace USTC_CG
