#include "PoissonAlphaMatte.h"

#include <algorithm>

#include "AlphaMaskBuilder.h"

namespace USTC_CG
{
PoissonAlphaMatte::PoissonAlphaMatte(
    std::shared_ptr<const PoissonRegion> region,
    std::shared_ptr<Image> source_image,
    std::shared_ptr<Image> target_image,
    std::shared_ptr<Image> subject_mask)
    : PoissonSolver(
          std::move(region),
          std::move(source_image),
          std::move(target_image)),
      subject_mask_(std::move(subject_mask))
{
}

void PoissonAlphaMatte::build()
{
    AlphaMaskBuilder builder(region_, subject_mask_, 8);
    alpha_ = builder.build_alpha_on_region();
    PoissonSolver::build();
}

Eigen::VectorXd PoissonAlphaMatte::build_rhs_for_channel(int channel) const
{
    const int n = region_->num_pixels();
    Eigen::VectorXd b(n);
    b.setZero();

    for (int id : region_->boundary_pixel_ids())
    {
        for (const auto& nbr : region_->outside_neighbors(id))
        {
            int safe_tx =
                std::clamp(nbr.target_x_, 0, target_image_->width() - 1);
            int safe_ty =
                std::clamp(nbr.target_y_, 0, target_image_->height() - 1);

            int safe_sx =
                std::clamp(nbr.source_x_, 0, source_image_->width() - 1);
            int safe_sy =
                std::clamp(nbr.source_y_, 0, source_image_->height() - 1);

            const auto tgt_q = target_image_->get_pixel(safe_tx, safe_ty);
            const auto src_q = source_image_->get_pixel(safe_sx, safe_sy);

            b[id] += static_cast<double>(tgt_q[channel]) -
                     static_cast<double>(src_q[channel]);
        }
    }

    return b;
}

std::shared_ptr<Image> PoissonAlphaMatte::solve()
{
    if (region_->empty())
        return std::make_shared<Image>(*target_image_);

    auto result = std::make_shared<Image>(*target_image_);
    const int n = region_->num_pixels();

    std::vector<Eigen::VectorXd> membrane(3);
    for (int c = 0; c < 3; ++c)
        membrane[c] = solve_channel(build_rhs_for_channel(c));

    for (int id = 0; id < n; ++id)
    {
        const auto& p = region_->pixel(id);
        if (p.target_x_ < 0 || p.target_x_ >= result->width() ||
            p.target_y_ < 0 || p.target_y_ >= result->height())
            continue;

        auto src = source_image_->get_pixel(p.source_x_, p.source_y_);
        auto out = result->get_pixel(p.target_x_, p.target_y_);

        double omega = 1.0 - alpha_[id];

        for (int c = 0; c < 3; ++c)
        {
            double val = static_cast<double>(src[c]) + omega * membrane[c][id];
            out[c] = clamp_to_uchar(val);
        }

        result->set_pixel(p.target_x_, p.target_y_, out);
    }

    return result;
}
}  // namespace USTC_CG