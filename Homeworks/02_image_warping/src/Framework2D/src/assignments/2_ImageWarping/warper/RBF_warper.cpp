#include "RBF_warper.h"

#include <algorithm>
#include <cmath>

namespace USTC_CG
{

RBFWarper::RBFWarper(
    const std::vector<ImVec2>& start_points,
    const std::vector<ImVec2>& end_points,
    int canvas_width,
    int canvas_height,
    Type type,
    BasisType basis_type)
    : Warper(start_points, end_points, canvas_width, canvas_height),
      type_(type),
      basis_type_(basis_type)
{
    build_model();
}

void RBFWarper::build_model()
{
    affine_A_ = Mat2::Identity();
    affine_b_ = Vec2::Zero();
    alphas_.clear();
    radii_.clear();

    build_local_radii();

    switch (type_)
    {
        case Type::AffineFirst: build_affine_first_model(); break;
        case Type::Full: build_full_model(); break;
    }
}

void RBFWarper::build_local_radii()
{
    radii_.clear();
    radii_.reserve(control_point_count());

    for (size_t i = 0; i < control_point_count(); ++i)
    {
        radii_.push_back(local_radius(i));
    }
}

float RBFWarper::local_radius(size_t i) const
{
    // r_i = min_{j != i} ||p_i - p_j||
    const size_t n = control_point_count();
    if (n <= 1)
    {
        return 1.0f;
    }

    const Vec2 pi = to_eigen(start_points_[i]);
    float best = std::numeric_limits<float>::max();

    for (size_t j = 0; j < n; ++j)
    {
        if (j == i)
        {
            continue;
        }

        const Vec2 pj = to_eigen(start_points_[j]);
        best = std::min(best, (pi - pj).norm());
    }

    if (!std::isfinite(best) || best < eps_)
    {
        return 1.0f;
    }

    return best;
}

float RBFWarper::basis_value(size_t i, float d) const
{
    // g_i(d) = (d^2 + r_i^2)^(±1/2)
    const float ri = radii_[i];
    const float value = d * d + ri * ri;

    if (basis_type_ == BasisType::Sqrt)
    {
        return std::sqrt(std::max(value, eps_));
    }

    return 1.0f / std::sqrt(std::max(value, eps_));
}

void RBFWarper::build_affine_part_for_affine_first()
{
    const int n = static_cast<int>(control_point_count());

    // One point: translation
    if (n == 1)
    {
        affine_A_ = Mat2::Identity();
        affine_b_ = to_eigen(end_points_[0]) - to_eigen(start_points_[0]);
        return;
    }

    // Two points: translation + scaling
    if (n == 2)
    {
        const Vec2 p1 = to_eigen(start_points_[0]);
        const Vec2 p2 = to_eigen(start_points_[1]);
        const Vec2 q1 = to_eigen(end_points_[0]);
        const Vec2 q2 = to_eigen(end_points_[1]);

        const float dp = (p2 - p1).norm();
        const float dq = (q2 - q1).norm();

        float s = 1.0f;
        if (dp > eps_)
        {
            s = dq / dp;
        }

        affine_A_ = s * Mat2::Identity();
        affine_b_ = q1 - affine_A_ * p1;
        return;
    }

    fit_affine_least_squares();
}

void RBFWarper::fit_affine_least_squares()
{
    // min sum_i ||A p_i + b - q_i||^2
    const int n = static_cast<int>(control_point_count());

    Eigen::MatrixXf M(2 * n, 6);
    Eigen::VectorXf rhs(2 * n);

    for (int i = 0; i < n; ++i)
    {
        const Vec2 p = to_eigen(start_points_[i]);
        const Vec2 q = to_eigen(end_points_[i]);

        // x row
        M(2 * i, 0) = p.x();
        M(2 * i, 1) = p.y();
        M(2 * i, 2) = 0.0f;
        M(2 * i, 3) = 0.0f;
        M(2 * i, 4) = 1.0f;
        M(2 * i, 5) = 0.0f;
        rhs(2 * i) = q.x();

        // y row
        M(2 * i + 1, 0) = 0.0f;
        M(2 * i + 1, 1) = 0.0f;
        M(2 * i + 1, 2) = p.x();
        M(2 * i + 1, 3) = p.y();
        M(2 * i + 1, 4) = 0.0f;
        M(2 * i + 1, 5) = 1.0f;
        rhs(2 * i + 1) = q.y();
    }

    const Eigen::VectorXf sol = M.colPivHouseholderQr().solve(rhs);

    affine_A_(0, 0) = sol(0);
    affine_A_(0, 1) = sol(1);
    affine_A_(1, 0) = sol(2);
    affine_A_(1, 1) = sol(3);
    affine_b_(0) = sol(4);
    affine_b_(1) = sol(5);
}

void RBFWarper::build_affine_first_model()
{
    // Method 1:
    // First determine A and b, then solve alpha_i from interpolation
    // constraints
    const int n = static_cast<int>(control_point_count());
    if (n <= 0)
    {
        return;
    }

    build_affine_part_for_affine_first();

    // G_{j,i} = g_i(||p_j - p_i||)
    MatX G(n, n);
    VecX rhs_x(n);
    VecX rhs_y(n);

    for (int j = 0; j < n; ++j)
    {
        const Vec2 pj = to_eigen(start_points_[j]);
        const Vec2 qj = to_eigen(end_points_[j]);

        for (int i = 0; i < n; ++i)
        {
            const Vec2 pi = to_eigen(start_points_[i]);
            const float d = (pj - pi).norm();
            G(j, i) = basis_value(i, d);
        }

        const Vec2 affine_value = affine_A_ * pj + affine_b_;
        const Vec2 residual = qj - affine_value;

        rhs_x(j) = residual.x();
        rhs_y(j) = residual.y();
    }

    G += eps_ * MatX::Identity(n, n);

    const VecX alpha_x = G.colPivHouseholderQr().solve(rhs_x);
    const VecX alpha_y = G.colPivHouseholderQr().solve(rhs_y);

    alphas_.resize(n, Vec2::Zero());
    for (int i = 0; i < n; ++i)
    {
        alphas_[i] = Vec2(alpha_x(i), alpha_y(i));
    }
}

void RBFWarper::build_full_model()
{
    // Method 2:
    // Solve all unknowns together with extra constraints:
    //
    // f(p_j) = sum_i alpha_i g_i(||p_j - p_i||) + A p_j + b = q_j
    //
    // and
    //
    // [ p_1 ... p_n ]
    // [  1  ...  1  ] * [alpha_1^T ... alpha_n^T]^T = 0_{3x2}
    const int n = static_cast<int>(control_point_count());
    if (n <= 0)
    {
        return;
    }

    // Unknowns for x-component:
    // [alpha_1^x ... alpha_n^x  a11  a12  b1]^T
    MatX M = MatX::Zero(n + 3, n + 3);
    VecX rhs_x = VecX::Zero(n + 3);
    VecX rhs_y = VecX::Zero(n + 3);

    for (int j = 0; j < n; ++j)
    {
        const Vec2 pj = to_eigen(start_points_[j]);
        const Vec2 qj = to_eigen(end_points_[j]);

        for (int i = 0; i < n; ++i)
        {
            const Vec2 pi = to_eigen(start_points_[i]);
            const float d = (pj - pi).norm();
            M(j, i) = basis_value(i, d);
        }

        // Affine terms: a11 * x + a12 * y + b1
        M(j, n + 0) = pj.x();
        M(j, n + 1) = pj.y();
        M(j, n + 2) = 1.0f;

        rhs_x(j) = qj.x();
        rhs_y(j) = qj.y();
    }

    // Extra constraints:
    // sum_i alpha_i = 0
    // sum_i alpha_i * p_ix = 0
    // sum_i alpha_i * p_iy = 0
    for (int i = 0; i < n; ++i)
    {
        const Vec2 pi = to_eigen(start_points_[i]);

        M(n + 0, i) = pi.x();
        M(n + 1, i) = pi.y();
        M(n + 2, i) = 1.0f;
    }

    for (int i = 0; i < n; ++i)
    {
        M(i, i) += eps_;
    }

    const VecX sol_x = M.colPivHouseholderQr().solve(rhs_x);
    const VecX sol_y = M.colPivHouseholderQr().solve(rhs_y);

    alphas_.resize(n, Vec2::Zero());
    for (int i = 0; i < n; ++i)
    {
        alphas_[i] = Vec2(sol_x(i), sol_y(i));
    }

    // Recover affine part:
    // x(p) = a11 * px + a12 * py + b1
    // y(p) = a21 * px + a22 * py + b2
    affine_A_(0, 0) = sol_x(n + 0);
    affine_A_(0, 1) = sol_x(n + 1);
    affine_A_(1, 0) = sol_y(n + 0);
    affine_A_(1, 1) = sol_y(n + 1);

    affine_b_(0) = sol_x(n + 2);
    affine_b_(1) = sol_y(n + 2);
}

RBFWarper::Vec2 RBFWarper::eval_affine(const Vec2& p) const
{
    // Evaluate A(p) = A p + b
    return affine_A_ * p + affine_b_;
}

RBFWarper::Vec2 RBFWarper::eval_radial(const Vec2& p) const
{
    // Evaluate R(p) = sum_i alpha_i g_i(||p-p_i||)
    Vec2 sum = Vec2::Zero();

    const int n = static_cast<int>(alphas_.size());
    for (int i = 0; i < n; ++i)
    {
        const Vec2 pi = to_eigen(start_points_[i]);
        const float d = (p - pi).norm();
        sum += alphas_[i] * basis_value(i, d);
    }

    return sum;
}

ImVec2 RBFWarper::warp_impl(const ImVec2& p) const
{
    const Vec2 x = to_eigen(p);
    const Vec2 y = eval_affine(x) + eval_radial(x);
    return to_imgui(y);
}

}  // namespace USTC_CG