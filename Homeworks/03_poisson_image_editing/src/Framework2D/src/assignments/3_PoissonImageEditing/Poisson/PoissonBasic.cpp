#include "PoissonBasic.h"
#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <vector>

namespace USTC_CG
{
PoissonBasic::PoissonBasic(
    std::shared_ptr<const PoissonRegion> region,
    std::shared_ptr<Image> source_image,
    std::shared_ptr<Image> target_image)
    : PoissonSolver(
          std::move(region),
          std::move(source_image),
          std::move(target_image))
{
}

Eigen::VectorXd PoissonBasic::build_rhs_for_channel(int channel) const
{
    const int n = region_->num_pixels();

    Eigen::VectorXd g_in(n);
    g_in.setZero();

    for (int id = 0; id < n; ++id)
    {
        const RegionPixel& p = region_->pixel(id);
        const auto src_p = source_image_->get_pixel(p.source_x_, p.source_y_);
        g_in[id] = static_cast<double>(src_p[channel]);
    }

    Eigen::VectorXd b = region_->A() * g_in;

    for (int id : region_->boundary_pixel_ids())
    {
        for (const auto& nbr : region_->outside_neighbors(id))
        {
            if (0 <= nbr.source_x_ && nbr.source_x_ < source_image_->width() &&
                0 <= nbr.source_y_ && nbr.source_y_ < source_image_->height())
            {
                const auto src_q =
                    source_image_->get_pixel(nbr.source_x_, nbr.source_y_);
                b[id] -= static_cast<double>(src_q[channel]);
            }

            int safe_tx =
                std::clamp(nbr.target_x_, 0, target_image_->width() - 1);
            int safe_ty =
                std::clamp(nbr.target_y_, 0, target_image_->height() - 1);

            const auto tgt_q = target_image_->get_pixel(safe_tx, safe_ty);
            b[id] += static_cast<double>(tgt_q[channel]);
        }
    }
    return b;
}

std::shared_ptr<Image> PoissonBasic::solve()
{
    if (!source_image_ || !target_image_)
    {
        throw std::runtime_error("PoissonBasic: input image is null.");
    }

    if (region_->empty())
    {
        return std::make_shared<Image>(*target_image_);
    }

    if (source_image_->channels() < 3 || target_image_->channels() < 3)
    {
        throw std::runtime_error(
            "PoissonBasic: source/target image must have at least 3 channels.");
    }

    std::shared_ptr<Image> result = std::make_shared<Image>(*target_image_);

    const int n = region_->num_pixels();

    std::vector<Eigen::VectorXd> channel_solution(3);
    for (int c = 0; c < 3; ++c)
    {
        Eigen::VectorXd b = build_rhs_for_channel(c);
        channel_solution[c] = solve_channel(b);
    }

    for (int id = 0; id < n; ++id)
    {
        const RegionPixel& p = region_->pixel(id);

        if (p.target_x_ < 0 || p.target_x_ >= result->width() ||
            p.target_y_ < 0 || p.target_y_ >= result->height())
        {
            continue;
        }

        std::vector<unsigned char> pixel_value =
            result->get_pixel(p.target_x_, p.target_y_);

        for (int c = 0; c < 3; ++c)
        {
            pixel_value[c] = clamp_to_uchar(channel_solution[c][id]);
        }
        result->set_pixel(p.target_x_, p.target_y_, pixel_value);
    }
    return result;
}
}  // namespace USTC_CG