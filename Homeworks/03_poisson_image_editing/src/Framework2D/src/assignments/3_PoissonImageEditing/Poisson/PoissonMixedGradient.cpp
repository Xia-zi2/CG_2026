#include "PoissonMixedGradient.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace USTC_CG
{
PoissonMixedGradient::PoissonMixedGradient(
    std::shared_ptr<const PoissonRegion> region,
    std::shared_ptr<Image> source_image,
    std::shared_ptr<Image> target_image)
    : PoissonSolver(
          std::move(region),
          std::move(source_image),
          std::move(target_image))
{
}

Eigen::VectorXd PoissonMixedGradient::build_rhs_for_channel(int channel) const
{
    const int n = region_->num_pixels();
    Eigen::VectorXd b(n);
    b.setZero();

    std::vector<std::vector<int>> id_map(
        source_image_->width(), std::vector<int>(source_image_->height(), -1));

    for (const auto& p : region_->pixels())
        id_map[p.source_x_][p.source_y_] = p.id;

    static const int dx[4] = { -1, 1, 0, 0 };
    static const int dy[4] = { 0, 0, -1, 1 };

    for (int id = 0; id < n; ++id)
    {
        const auto& p = region_->pixel(id);

        auto src_p = source_image_->get_pixel(p.source_x_, p.source_y_);
        auto tgt_p = target_image_->get_pixel(
            std::clamp(p.target_x_, 0, target_image_->width() - 1),
            std::clamp(p.target_y_, 0, target_image_->height() - 1));

        for (int k = 0; k < 4; ++k)
        {
            int nsx = p.source_x_ + dx[k];
            int nsy = p.source_y_ + dy[k];
            int ntx = p.target_x_ + dx[k];
            int nty = p.target_y_ + dy[k];

            int nid = -1;
            if (nsx >= 0 && nsx < source_image_->width() && nsy >= 0 &&
                nsy < source_image_->height())
            {
                nid = id_map[nsx][nsy];
            }

            auto src_q = source_image_->get_pixel(
                std::clamp(nsx, 0, source_image_->width() - 1),
                std::clamp(nsy, 0, source_image_->height() - 1));

            auto tgt_q = target_image_->get_pixel(
                std::clamp(ntx, 0, target_image_->width() - 1),
                std::clamp(nty, 0, target_image_->height() - 1));

            double gs = double(src_p[channel]) - double(src_q[channel]);
            double gt = double(tgt_p[channel]) - double(tgt_q[channel]);

            double v = (std::abs(gs) >= std::abs(gt)) ? gs : gt;
            b[id] += v;

            if (nid == -1)
            {
                b[id] += static_cast<double>(tgt_q[channel]);
            }
        }
    }

    return b;
}

std::shared_ptr<Image> PoissonMixedGradient::solve()
{
    if (region_->empty())
        return std::make_shared<Image>(*target_image_);

    auto result = std::make_shared<Image>(*target_image_);
    const int n = region_->num_pixels();

    std::vector<Eigen::VectorXd> channel_solution(3);
    for (int c = 0; c < 3; ++c)
    {
        channel_solution[c] = solve_channel(build_rhs_for_channel(c));
    }

    for (int id = 0; id < n; ++id)
    {
        const auto& p = region_->pixel(id);
        if (p.target_x_ < 0 || p.target_x_ >= result->width() ||
            p.target_y_ < 0 || p.target_y_ >= result->height())
            continue;

        auto pixel_value = result->get_pixel(p.target_x_, p.target_y_);
        for (int c = 0; c < 3; ++c)
            pixel_value[c] = clamp_to_uchar(channel_solution[c][id]);
        result->set_pixel(p.target_x_, p.target_y_, pixel_value);
    }

    return result;
}
}  // namespace USTC_CG