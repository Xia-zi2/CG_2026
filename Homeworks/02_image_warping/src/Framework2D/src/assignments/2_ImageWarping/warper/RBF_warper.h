#pragma once

#include <imgui.h>

#include <Eigen/Dense>
#include <vector>

#include "warper.h"

namespace USTC_CG
{

class RBFWarper : public Warper
{
   public:
    enum class Type
    {
        AffineFirst,  // Determine affine part first, then solve alpha_i

        Full  // Solve affine part and alpha_i together with extra constraints
    };

    enum class BasisType
    {
        Sqrt,  // g_i(d) = sqrt(d^2 + r_i^2)

        InverseSqrt  // g_i(d) = 1 / sqrt(d^2 + r_i^2)
    };

    using Mat2 = Eigen::Matrix2f;
    using VecX = Eigen::VectorXf;
    using MatX = Eigen::MatrixXf;

   public:
    RBFWarper(
        const std::vector<ImVec2>& start_points,
        const std::vector<ImVec2>& end_points,
        int canvas_width,
        int canvas_height,
        Type type,
        BasisType basis_type = BasisType::InverseSqrt);

    ~RBFWarper() override = default;

    ImVec2 warp_impl(const ImVec2& p) const override;

    Type type() const
    {
        return type_;
    }

    BasisType basis_type() const
    {
        return basis_type_;
    }

   private:
    // Build the whole RBF model
    void build_model();

    // Build per-control support radii r_i
    void build_local_radii();

    // Compute r_i = min_{j != i} ||p_i - p_j||
    float local_radius(size_t i) const;

    // Evaluate g_i(d)
    float basis_value(size_t i, float d) const;

    // Build the affine-first model
    void build_affine_first_model();

    // Build the full coupled model
    void build_full_model();

    // Fit affine part for the affine-first method
    void build_affine_part_for_affine_first();

    // Solve the least-squares affine map A p + b
    void fit_affine_least_squares();

    // Evaluate the affine part A p + b
    Vec2 eval_affine(const Vec2& p) const;

    // Evaluate the radial part sum_i alpha_i g_i(||p-p_i||)
    Vec2 eval_radial(const Vec2& p) const;

   private:
    // Current RBF method type
    Type type_ = Type::AffineFirst;

    BasisType basis_type_ = BasisType::InverseSqrt;

    // Affine matrix A in A(p) = A p + b
    Mat2 affine_A_ = Mat2::Identity();

    // Affine translation b in A(p) = A p + b
    Vec2 affine_b_ = Vec2::Zero();

    // Radial coefficients alpha_i
    std::vector<Vec2> alphas_;

    // Local radii r_i used in g_i(d)
    std::vector<float> radii_;
};

}  // namespace USTC_CG