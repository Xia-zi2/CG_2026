#pragma once
#include "PoissonSolver.h"

namespace USTC_CG
{
class PoissonColorConstraint : public PoissonSolver
{
   public:
    PoissonColorConstraint(
        std::shared_ptr<const PoissonRegion> region,
        std::shared_ptr<Image> source_image,
        std::shared_ptr<Image> target_image,
        std::shared_ptr<Image> subject_mask,
        double lambda = 1.0);

    void build() override;
    std::shared_ptr<Image> solve() override;

   protected:
    void build_system_matrix() override;
    Eigen::VectorXd build_rhs_for_channel(int channel) const override;

   private:
    std::shared_ptr<Image> subject_mask_;
    double lambda_;
    std::vector<double> ooi_;
};
}  // namespace USTC_CG