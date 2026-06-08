#pragma once
#include "PoissonSolver.h"

namespace USTC_CG
{
class PoissonBasic : public PoissonSolver
{
   public:
    PoissonBasic(
        std::shared_ptr<const PoissonRegion> region,
        std::shared_ptr<Image> source_image,
        std::shared_ptr<Image> target_image);

    std::shared_ptr<Image> solve() override;

   protected:
    Eigen::VectorXd build_rhs_for_channel(int channel) const override;
};
}  // namespace USTC_CG