#pragma once
#include "shape.h"
#include <vector>

namespace USTC_CG
{
class Freehand : public Shape
{
   public:
    Freehand() = default;

    Freehand(std::vector<float> x_list, std::vector<float> y_list);

    virtual ~Freehand() = default;

    void draw(const Config& config) const override;

    void update(float x, float y) override;

    void add_control_point(float x, float y);

    bool hit_test(float x, float y) const override;
    void draw_selected(const Config& config) const override;


   private:
    std::vector<float> x_list_, y_list_;
    float accum_dx_ = 0.0f;
    float accum_dy_ = 0.0f;
};
}  // namespace USTC_CG