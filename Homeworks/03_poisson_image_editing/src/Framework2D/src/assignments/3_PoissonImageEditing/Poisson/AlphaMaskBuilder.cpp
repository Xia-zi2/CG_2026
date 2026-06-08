#include "AlphaMaskBuilder.h"

#include <algorithm>
#include <vector>

namespace USTC_CG
{
AlphaMaskBuilder::AlphaMaskBuilder(
    std::shared_ptr<const PoissonRegion> region,
    std::shared_ptr<Image> subject_mask,
    int blur_radius)
    : region_(std::move(region)),
      subject_mask_(std::move(subject_mask)),
      blur_radius_(blur_radius)
{
}

std::vector<double> AlphaMaskBuilder::build_alpha_on_region() const
{
    std::vector<double> alpha(region_->num_pixels(), 0.0);
    if (!subject_mask_ || region_->empty())
        return alpha;

    const int w = subject_mask_->width();
    const int h = subject_mask_->height();

    std::vector<std::vector<double>> img(w, std::vector<double>(h, 0.0));
    for (int x = 0; x < w; ++x)
        for (int y = 0; y < h; ++y)
            img[x][y] = subject_mask_->get_pixel(x, y)[0] > 0 ? 1.0 : 0.0;

    std::vector<std::vector<double>> tmp = img;
    std::vector<std::vector<double>> out = img;

    const int r = std::max(1, blur_radius_);

    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            double sum = 0.0;
            int cnt = 0;
            for (int k = -r; k <= r; ++k)
            {
                int xx = std::clamp(x + k, 0, w - 1);
                sum += img[xx][y];
                ++cnt;
            }
            tmp[x][y] = sum / cnt;
        }
    }

    for (int x = 0; x < w; ++x)
    {
        for (int y = 0; y < h; ++y)
        {
            double sum = 0.0;
            int cnt = 0;
            for (int k = -r; k <= r; ++k)
            {
                int yy = std::clamp(y + k, 0, h - 1);
                sum += tmp[x][yy];
                ++cnt;
            }
            out[x][y] = sum / cnt;
        }
    }

    for (int id = 0; id < region_->num_pixels(); ++id)
    {
        const auto& p = region_->pixel(id);
        if (p.source_x_ >= 0 && p.source_x_ < w && p.source_y_ >= 0 &&
            p.source_y_ < h)
        {
            alpha[id] = std::clamp(out[p.source_x_][p.source_y_], 0.0, 1.0);
        }
    }

    return alpha;
}
}  // namespace USTC_CG