#include "PoissonColorConstraint.h"

#include <algorithm>
#include <vector>

namespace USTC_CG
{
PoissonColorConstraint::PoissonColorConstraint(
    std::shared_ptr<const PoissonRegion> region,
    std::shared_ptr<Image> source_image,
    std::shared_ptr<Image> target_image,
    std::shared_ptr<Image> subject_mask,
    double lambda)
    : PoissonSolver(
          std::move(region),
          std::move(source_image),
          std::move(target_image)),
      subject_mask_(std::move(subject_mask)),
      lambda_(lambda)
{
}

void PoissonColorConstraint::build()
{
    const int n = region_->num_pixels();
    ooi_.assign(n, 0.0);

    if (subject_mask_)
    {
        const int w = subject_mask_->width();
        const int h = subject_mask_->height();

        for (int id = 0; id < n; ++id)
        {
            const auto& p = region_->pixel(id);
            if (p.source_x_ >= 0 && p.source_x_ < w && p.source_y_ >= 0 &&
                p.source_y_ < h)
            {
                ooi_[id] =
                    (subject_mask_->get_pixel(p.source_x_, p.source_y_)[0] > 0)
                        ? 1.0
                        : 0.0;
            }
        }
    }

    PoissonSolver::build();
}

void PoissonColorConstraint::build_system_matrix()
{
    system_matrix_ = region_->A();

    std::vector<Eigen::Triplet<double>> extra;
    extra.reserve(ooi_.size());

    for (int i = 0; i < (int)ooi_.size(); ++i)
    {
        if (ooi_[i] > 0.5)
            extra.emplace_back(i, i, lambda_);
    }

    Eigen::SparseMatrix<double> D(system_matrix_.rows(), system_matrix_.cols());
    D.setFromTriplets(extra.begin(), extra.end());

    system_matrix_ = system_matrix_ + D;
    system_matrix_.makeCompressed();
}

Eigen::VectorXd PoissonColorConstraint::build_rhs_for_channel(int channel) const
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

    for (int id = 0; id < n; ++id)
    {
        if (ooi_[id] > 0.5)
        {
            b[id] += lambda_ * g_in[id];
        }
    }

    return b;
}

std::shared_ptr<Image> PoissonColorConstraint::solve()
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