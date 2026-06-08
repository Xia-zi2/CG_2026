#include "IDW_warper.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace USTC_CG
{

IDWWarper::IDWWarper(
    const std::vector<ImVec2>& start_points,
    const std::vector<ImVec2>& end_points,
    int canvas_width,
    int canvas_height,
    Type type)
    : Warper(start_points, end_points, canvas_width, canvas_height),
      type_(type)
{
    build_radii();
    build_affines();
}

float IDWWarper::control_weight(float dist2) const
{
    const float eps2 = eps_ * eps_;
    return 1.0f / std::max(dist2, eps2);
}

float IDWWarper::radius_weight(float dist2, float Ri) const
{
    const float safe_R = std::max(Ri, eps_);
    const float r = std::sqrt(std::max(dist2, eps_ * eps_));

    if (r >= safe_R)
    {
        return 0.0f;
    }

    const float numer = std::max(0.0f, safe_R - r);
    const float denom = safe_R * r;
    if (denom <= eps_)
    {
        return 0.0f;
    }

    return std::pow(numer / denom, mu_);
}

float IDWWarper::attenuation_Ai_from_dist2(float dist2, float Ri) const
{
    const float safe_R = std::max(Ri, eps_);
    const float t2 = dist2 / (safe_R * safe_R);

    if (t2 >= 1.0f)
    {
        return 0.0f;
    }

    const float v = 1.0f - t2;
    return v * v;
}

void IDWWarper::build_radii()
{
    attenuation_radii_.clear();

    const size_t n = control_point_count();
    if (n == 0 || n != end_points_.size())
    {
        return;
    }

    attenuation_radii_.resize(n, min_attenuation_radius_);

    for (size_t i = 0; i < n; ++i)
    {
        const Vec2 pi = to_eigen(start_points_[i]);
        const Vec2 qi = to_eigen(end_points_[i]);
        const float disp_len = (qi - pi).norm();

        attenuation_radii_[i] = std::max(
            min_attenuation_radius_, radius_scale_ * disp_len + radius_buffer_);
    }
}

IDWWarper::Mat2 IDWWarper::compute_affine_for_control(size_t i) const
{
    Mat2 A = Mat2::Zero();
    Mat2 B = Mat2::Zero();
    bool has_valid_neighbor = false;

    const Vec2 pi = to_eigen(start_points_[i]);
    const Vec2 qi = to_eigen(end_points_[i]);
    const size_t n = control_point_count();

    for (size_t j = 0; j < n; ++j)
    {
        if (j == i)
        {
            continue;
        }

        const Vec2 pj = to_eigen(start_points_[j]);
        const Vec2 qj = to_eigen(end_points_[j]);
        const Vec2 dp = pj - pi;
        const Vec2 dq = qj - qi;
        const float dist2 = dp.squaredNorm();

        if (dist2 < eps_ * eps_)
        {
            continue;
        }

        const float w = (type_ == Type::IDW_R)
                            ? radius_weight(dist2, attenuation_radii_[i])
                            : control_weight(dist2);

        if (w <= 0.0f)
        {
            continue;
        }

        has_valid_neighbor = true;
        A += w * (dp * dp.transpose());
        B += w * (dq * dp.transpose());
    }

    if (!has_valid_neighbor)
    {
        return Mat2::Identity();
    }

    A += eps_ * Mat2::Identity();

    Eigen::LDLT<Mat2> ldlt(A);
    if (ldlt.info() != Eigen::Success)
    {
        return Mat2::Identity();
    }

    const Mat2 X = ldlt.solve(Mat2::Identity());
    if (ldlt.info() != Eigen::Success)
    {
        return Mat2::Identity();
    }

    return B * X;
}

void IDWWarper::build_affines()
{
    affines_.clear();

    const size_t n = control_point_count();
    if (n == 0 || n != end_points_.size())
    {
        return;
    }

    affines_.resize(n, Mat2::Identity());
    for (size_t i = 0; i < n; ++i)
    {
        affines_[i] = compute_affine_for_control(i);
    }
}

ImVec2 IDWWarper::warp_impl(const ImVec2& p) const
{
    const size_t n = control_point_count();
    if (n == 0 || n != end_points_.size() || n != affines_.size())
    {
        return p;
    }

    if ((type_ == Type::IDW_A || type_ == Type::IDW_R) &&
        attenuation_radii_.size() != n)
    {
        return p;
    }

    const Vec2 pv = to_eigen(p);

    if (type_ == Type::IDW)
    {
        float weight_sum = 0.0f;
        Vec2 result = Vec2::Zero();

        for (size_t i = 0; i < n; ++i)
        {
            const Vec2 pi = to_eigen(start_points_[i]);
            const Vec2 qi = to_eigen(end_points_[i]);
            const Vec2 dp = pv - pi;
            const float w = control_weight(dp.squaredNorm());

            if (w <= 0.0f)
            {
                continue;
            }

            const Vec2 fi = qi + affines_[i] * dp;
            result += w * fi;
            weight_sum += w;
        }

        if (weight_sum <= 0.0f)
        {
            return p;
        }

        return to_imgui(result / weight_sum);
    }

    if (type_ == Type::IDW_A)
    {
        float weight_sum = 0.0f;
        Vec2 disp_sum = Vec2::Zero();

        for (size_t i = 0; i < n; ++i)
        {
            const Vec2 pi = to_eigen(start_points_[i]);
            const Vec2 qi = to_eigen(end_points_[i]);
            const Vec2 dp = pv - pi;
            const float dist2 = dp.squaredNorm();
            const float w = control_weight(dist2);

            if (w <= 0.0f)
            {
                continue;
            }

            const Vec2 fi = qi + affines_[i] * dp;
            const Vec2 local_disp = fi - pv;
            const float Ai =
                attenuation_Ai_from_dist2(dist2, attenuation_radii_[i]);

            disp_sum += w * Ai * local_disp;
            weight_sum += w;
        }

        if (weight_sum <= 0.0f)
        {
            return p;
        }

        return to_imgui(pv + disp_sum / weight_sum);
    }

    float weight_sum = 0.0f;
    Vec2 result = Vec2::Zero();

    for (size_t i = 0; i < n; ++i)
    {
        const Vec2 pi = to_eigen(start_points_[i]);
        const Vec2 qi = to_eigen(end_points_[i]);
        const Vec2 dp = pv - pi;
        const float w = radius_weight(dp.squaredNorm(), attenuation_radii_[i]);

        if (w <= 0.0f)
        {
            continue;
        }

        const Vec2 fi = qi + affines_[i] * dp;
        result += w * fi;
        weight_sum += w;
    }

    if (weight_sum <= 0.0f)
    {
        return p;
    }

    return to_imgui(result / weight_sum);
}

}  // namespace USTC_CG
