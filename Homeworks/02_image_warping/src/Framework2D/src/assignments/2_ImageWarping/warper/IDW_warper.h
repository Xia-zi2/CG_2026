#pragma once

#include <imgui.h>

#include <Eigen/Dense>
#include <vector>

#include "warper.h"

namespace USTC_CG
{

class IDWWarper : public Warper
{
   public:
    enum class Type
    {
        IDW,    // Original global IDW
        IDW_A,  // IDW with attenuation on the local displacement
        IDW_R   // IDW with radius-controlled compact-support sigma
    };

    using Mat2 = Eigen::Matrix2f;

   public:
    IDWWarper(
        const std::vector<ImVec2>& start_points,
        const std::vector<ImVec2>& end_points,
        int canvas_width,
        int canvas_height,
        Type type);

    ~IDWWarper() override = default;

    ImVec2 warp_impl(const ImVec2& p) const override;

    Type type() const
    {
        return type_;
    }

   private:
    float control_weight(float dist2) const;
    float radius_weight(float dist2, float Ri) const;
    float attenuation_Ai_from_dist2(float dist2, float Ri) const;

    void build_radii();
    void build_affines();
    Mat2 compute_affine_for_control(size_t i) const;

   private:
    Type type_ = Type::IDW;

    float mu_ = 2.0f;
    float radius_scale_ = 1.7f;
    float radius_buffer_ = 8.0f;
    float min_attenuation_radius_ = 16.0f;

    std::vector<Mat2> affines_;
    std::vector<float> attenuation_radii_;
};

}  // namespace USTC_CG
