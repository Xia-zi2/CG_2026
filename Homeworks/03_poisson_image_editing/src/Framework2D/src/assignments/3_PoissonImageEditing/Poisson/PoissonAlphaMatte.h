#pragma once
#include "PoissonSolver.h"

namespace USTC_CG
{
class PoissonAlphaMatte : public PoissonSolver
{
   public:
    PoissonAlphaMatte(
        std::shared_ptr<const PoissonRegion> region,
        std::shared_ptr<Image> source_image,
        std::shared_ptr<Image> target_image,
        std::shared_ptr<Image> subject_mask);

    void build() override;
    std::shared_ptr<Image> solve() override;

   protected:
    Eigen::VectorXd build_rhs_for_channel(int channel) const override;

   private:
    std::shared_ptr<Image> subject_mask_;
    std::vector<double> alpha_;
};
}  // namespace USTC_CG
