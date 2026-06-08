#pragma once

#include "warper.h"

namespace USTC_CG
{

class Fisheye : public Warper
{
   public:
    Fisheye(int width, int height) : Warper({}, {}, width, height)
    {
    }

    ~Fisheye() override = default;

   protected:
    bool requires_control_points() const override
    {
        return false;
    }

    ImVec2 warp_impl(const ImVec2& p) const override;
};

}  // namespace USTC_CG
