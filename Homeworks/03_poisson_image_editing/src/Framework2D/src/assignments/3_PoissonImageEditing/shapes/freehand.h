#pragma once

#include <utility>
#include <vector>

#include "shape.h"

namespace USTC_CG
{
class Freehand : public Shape
{
   public:
    Freehand() = default;

    Freehand(float start_point_x, float start_point_y)
    {
        points_.push_back({ start_point_x, start_point_y });
    }

    virtual ~Freehand() = default;

    void draw(const Config& config) const override;
    void update(float x, float y) override;
    std::vector<std::pair<int, int>> get_interior_pixels() const override;

    std::vector<std::pair<float, float>> get_polygon_points() const override
    {
        return points_;
    }

    void finish_drawing();

   private:
    std::vector<std::pair<float, float>> points_;
    bool is_closed_ = false;
};
}  // namespace USTC_CG