#include "PoissonSolver.h"

#include <algorithm>
#include <stdexcept>

namespace USTC_CG
{
PoissonSolver::PoissonSolver(
    std::shared_ptr<const PoissonRegion> region,
    std::shared_ptr<Image> source_image,
    std::shared_ptr<Image> target_image)
    : region_(std::move(region)),
      source_image_(std::move(source_image)),
      target_image_(std::move(target_image))
{
}

void PoissonSolver::build()
{
    build_system_matrix();
    factorize_matrix();
}

void PoissonSolver::build_system_matrix()
{
    if (!region_)
    {
        throw std::runtime_error("PoissonSolver: region is null.");
    }
    system_matrix_ = region_->A();
}

void PoissonSolver::factorize_matrix()
{
    if (system_matrix_.rows() == 0 || system_matrix_.cols() == 0)
    {
        throw std::runtime_error("PoissonSolver: empty system matrix.");
    }

    solver_.compute(system_matrix_);
    if (solver_.info() != Eigen::Success)
    {
        throw std::runtime_error("PoissonSolver: LDLT factorization failed.");
    }

    factorized_ = true;
}

Eigen::VectorXd PoissonSolver::solve_channel(const Eigen::VectorXd& b) const
{
    Eigen::VectorXd x = solver_.solve(b);
    if (solver_.info() != Eigen::Success)
    {
        throw std::runtime_error("PoissonSolver: linear solve failed.");
    }
    return x;
}

unsigned char PoissonSolver::clamp_to_uchar(double x) const
{
    x = std::round(x);
    x = std::max(0.0, std::min(255.0, x));
    return static_cast<unsigned char>(x);
}
}  // namespace USTC_CG