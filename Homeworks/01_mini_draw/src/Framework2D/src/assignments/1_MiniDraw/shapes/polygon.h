#pragma once
//Based on "rect.h"
#include "shape.h"

namespace USTC_CG
{
class Polygon : public Shape
{
   public:
    Polygon() = default;

    Polygon(
        float start_point_x,
        float start_point_y,
        float end_point_x,
        float end_point_y,
        int sides)
        : start_point_x_(start_point_x),
          start_point_y_(start_point_y),
          end_point_x_(end_point_x),
          end_point_y_(end_point_y),
          sides_(sides < 3 ? 3 : sides)
    {
    }

    virtual ~Polygon() = default;

    void draw(const Config& config) const override;

    void update(float x, float y) override;
    void set_sides(int sides);

    bool hit_test(float x, float y) const override;
    void draw_selected(const Config& config) const override;



   private:
    float start_point_x_ = 0.0f, start_point_y_ = 0.0f;
    float end_point_x_ = 0.0f, end_point_y_ = 0.0f;
    int sides_ = 3;
};
}  // namespace USTC_CG
