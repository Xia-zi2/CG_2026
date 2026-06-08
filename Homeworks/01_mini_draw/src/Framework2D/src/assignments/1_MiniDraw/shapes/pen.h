#pragma once
#include "shape.h"
#include <vector>

namespace USTC_CG
{
class Pen : public Shape
{
   public:
    Pen() = default;

    Pen(std::vector<float> x_list, std::vector<float> y_list);

    virtual ~Pen() = default;

    void draw(const Config& config) const override;

    void update(float x, float y) override;

    void add_control_point(float x, float y);

    void set_closed(bool closed);

    bool hit_test(float x, float y) const override;
    void draw_selected(const Config& config) const override;


   private:
    std::vector<float> x_list_, y_list_;
    bool is_closed_ = false;
};
}  // namespace USTC_CG
