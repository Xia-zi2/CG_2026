#pragma once
#include <memory>
#include <vector>

#include "PoissonRegion.h"
#include "common/image.h"

namespace USTC_CG
{
class AlphaMaskBuilder
{
   public:
    AlphaMaskBuilder(
        std::shared_ptr<const PoissonRegion> region,
        std::shared_ptr<Image> subject_mask,
        int blur_radius = 8);

    std::vector<double> build_alpha_on_region() const;

   private:
    std::shared_ptr<const PoissonRegion> region_;
    std::shared_ptr<Image> subject_mask_;
    int blur_radius_;
};
}  // namespace USTC_CG