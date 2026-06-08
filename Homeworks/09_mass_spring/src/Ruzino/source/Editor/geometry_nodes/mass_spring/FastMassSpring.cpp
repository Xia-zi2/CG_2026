#include "FastMassSpring.h"

#include <cmath>
#include <iostream>
#include <vector>

namespace USTC_CG::mass_spring {
namespace {
Eigen::VectorXd fast_flatten(const Eigen::MatrixXd& M)
{
    Eigen::VectorXd v(M.rows() * M.cols());
    for (int i = 0; i < M.rows(); i++) {
        for (int j = 0; j < M.cols(); j++) {
            v(3 * i + j) = M(i, j);
        }
    }
    return v;
}

Eigen::MatrixXd fast_unflatten(const Eigen::VectorXd& v, int rows, int cols)
{
    Eigen::MatrixXd M(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            M(i, j) = v(3 * i + j);
        }
    }
    return M;
}
}  // namespace

FastMassSpring::FastMassSpring(
    const Eigen::MatrixXd& X,
    const EdgeSet& E,
    const float stiffness,
    const float h)
    : MassSpring(X, E)
{
    std::cout << "init fast mass spring" << std::endl;

    this->stiffness = stiffness;
    this->h = h;

    prefactorizeSystemMatrix();
}

void FastMassSpring::prefactorizeSystemMatrix()
{
    const int n_vertices = static_cast<int>(X.rows());
    if (n_vertices <= 0 || h <= 0.0 || mass <= 0.0) {
        solver_ready = false;
        std::cerr << "FastMassSpring: invalid system parameters, "
                  << "n_vertices=" << n_vertices << ", h=" << h
                  << ", mass=" << mass << std::endl;
        return;
    }

    const double mass_per_vertex = mass / static_cast<double>(n_vertices);
    const double h2 = h * h;

    A_prefactorized.resize(n_vertices * 3, n_vertices * 3);
    A_prefactorized.setZero();

    std::vector<Trip_d> triplets;
    triplets.reserve(n_vertices * 3 + static_cast<int>(E.size()) * 4 * 3);

    auto add_diag_block = [&](int v, double coeff) {
        for (int d = 0; d < 3; d++) {
            triplets.emplace_back(3 * v + d, 3 * v + d, coeff);
        }
    };

    auto add_edge_block = [&](int row_v, int col_v, double coeff) {
        for (int d = 0; d < 3; d++) {
            triplets.emplace_back(3 * row_v + d, 3 * col_v + d, coeff);
        }
    };

    // Mass term M. Fixed vertices are constrained by identity rows.
    for (int i = 0; i < n_vertices; i++) {
        if (dirichlet_bc_mask[i]) {
            add_diag_block(i, 1.0);
        }
        else {
            add_diag_block(i, mass_per_vertex);
        }
    }

    // Stiffness/Laplacian term h^2 L.
    // For a free-free edge, add the usual 2x2 graph Laplacian block:
    //   [ k -k ] \otimes I_3
    //   [-k  k ]
    // For an edge connected to a fixed vertex, keep only the free endpoint's
    // diagonal block; the fixed-position contribution is moved to b in step().
    const double edge_coeff = h2 * stiffness;
    for (const auto& e : E) {
        const int v0 = e.first;
        const int v1 = e.second;

        const bool fixed0 = dirichlet_bc_mask[v0];
        const bool fixed1 = dirichlet_bc_mask[v1];

        if (!fixed0 && !fixed1) {
            add_edge_block(v0, v0, edge_coeff);
            add_edge_block(v0, v1, -edge_coeff);
            add_edge_block(v1, v0, -edge_coeff);
            add_edge_block(v1, v1, edge_coeff);
        }
        else if (!fixed0 && fixed1) {
            add_edge_block(v0, v0, edge_coeff);
        }
        else if (fixed0 && !fixed1) {
            add_edge_block(v1, v1, edge_coeff);
        }
    }

    A_prefactorized.setFromTriplets(triplets.begin(), triplets.end());
    A_prefactorized.makeCompressed();

    solver.compute(A_prefactorized);
    solver_ready = (solver.info() == Eigen::Success);

    cached_mass = mass;
    cached_stiffness = stiffness;
    cached_h = h;

    if (!solver_ready) {
        std::cerr << "FastMassSpring: prefactorization failed. "
                  << "Falling back to MassSpring::step()." << std::endl;
    }
}

void FastMassSpring::step()
{
    if (h <= 0.0) {
        if (enable_debug_output) {
            std::cerr << "[FastMassSpring] h <= 0, skip this step." << std::endl;
        }
        return;
    }

    // Rebuild A if parameters that affect A changed after initialization.
    if (!solver_ready || cached_mass != mass || cached_stiffness != stiffness ||
        cached_h != h) {
        prefactorizeSystemMatrix();
    }

    if (!solver_ready) {
        MassSpring::step();
        return;
    }

    const int n_vertices = static_cast<int>(X.rows());
    const double mass_per_vertex = mass / static_cast<double>(n_vertices);
    const double h2 = h * h;

    const Eigen::MatrixXd X_old = X;

    Eigen::MatrixXd vel_predict = vel;
    if (enable_damping) {
        vel_predict *= damping;
    }

    const Eigen::Vector3d acceleration_ext = gravity + wind_ext_acc;

    CollisionMode effective_collision_mode = collision_mode;
    if (collision_mode == SPHERE_PENALTY_ENERGY) {
        // A nonlinear collision energy changes the Hessian, which breaks the
        // constant prefactorized matrix assumption of Liu13. Use explicit
        // penalty acceleration instead, matching the semi-implicit fallback.
        if (enable_debug_output) {
            std::cerr << "[FastMassSpring] SPHERE_PENALTY_ENERGY is not "
                         "supported by the prefactorized Liu13 solver. "
                         "Falling back to SPHERE_PENALTY_FORCE."
                      << std::endl;
        }
        effective_collision_mode = SPHERE_PENALTY_FORCE;
    }

    Eigen::MatrixXd acceleration_collision =
        Eigen::MatrixXd::Zero(n_vertices, X.cols());
    if (effective_collision_mode == SPHERE_PENALTY_FORCE) {
        // Important: the base class function name is Acceleration, not Force.
        // It already divides the penalty force by per-vertex mass.
        acceleration_collision = getSphereCollisionAcceleration(
            sphere_center.cast<double>(), sphere_radius);
    }

    // y = x^n + h v^n + h^2 M^{-1} f_ext.
    Eigen::MatrixXd Y = X_old + h * vel_predict;
    Y.rowwise() += (h2 * acceleration_ext).transpose();
    if (effective_collision_mode == SPHERE_PENALTY_FORCE) {
        Y += h2 * acceleration_collision;
    }

    for (int i = 0; i < n_vertices; i++) {
        if (dirichlet_bc_mask[i]) {
            Y.row(i) = X_old.row(i);
            vel.row(i).setZero();
        }
    }

    // Local-global iterations. A is fixed; b changes through the local
    // projections d_i = L_i * normalize(x_i - x_j).
    for (unsigned iter = 0; iter < max_iter; iter++) {
        Eigen::MatrixXd rhs = Eigen::MatrixXd::Zero(n_vertices, 3);

        // M y term. Fixed vertices use identity constraint rows.
        for (int i = 0; i < n_vertices; i++) {
            if (dirichlet_bc_mask[i]) {
                rhs.row(i) = X_old.row(i);
            }
            else {
                rhs.row(i) = mass_per_vertex * Y.row(i);
            }
        }

        unsigned edge_id = 0;
        for (const auto& e : E) {
            const int v0 = e.first;
            const int v1 = e.second;

            const Eigen::Vector3d x0 = X.row(v0).transpose();
            const Eigen::Vector3d x1 = X.row(v1).transpose();
            const Eigen::Vector3d diff = x0 - x1;
            const double len = diff.norm();
            const double rest_len = E_rest_length[edge_id];

            Eigen::Vector3d d = Eigen::Vector3d::Zero();
            if (len > 1e-12) {
                d = rest_len * diff / len;
            }

            const bool fixed0 = dirichlet_bc_mask[v0];
            const bool fixed1 = dirichlet_bc_mask[v1];
            const Eigen::RowVector3d d_row = d.transpose();
            const double coeff = h2 * stiffness;

            // h^2 Jd contribution. With incidence vector (+1 at v0, -1 at v1):
            //   b_v0 += h^2 k d, b_v1 -= h^2 k d.
            // If one endpoint is fixed, the missing A_free,fixed x_fixed term
            // is also moved to the RHS.
            if (!fixed0 && !fixed1) {
                rhs.row(v0) += coeff * d_row;
                rhs.row(v1) -= coeff * d_row;
            }
            else if (!fixed0 && fixed1) {
                rhs.row(v0) += coeff * (X_old.row(v1) + d_row);
            }
            else if (fixed0 && !fixed1) {
                rhs.row(v1) += coeff * (X_old.row(v0) - d_row);
            }

            edge_id++;
        }

        const Eigen::VectorXd b = fast_flatten(rhs);
        const Eigen::VectorXd x_vec = solver.solve(b);

        if (solver.info() != Eigen::Success) {
            std::cerr << "FastMassSpring: linear solve failed. "
                      << "Falling back to MassSpring::step()." << std::endl;
            MassSpring::step();
            return;
        }

        Eigen::MatrixXd X_new = fast_unflatten(x_vec, n_vertices, 3);

        for (int i = 0; i < n_vertices; i++) {
            if (dirichlet_bc_mask[i]) {
                X_new.row(i) = X_old.row(i);
            }
        }

        const double step_norm = (X_new - X).norm();
        X = X_new;

        if (step_norm < 1e-8) {
            break;
        }
    }

    vel = (X - X_old) / h;

    for (int i = 0; i < n_vertices; i++) {
        if (dirichlet_bc_mask[i]) {
            X.row(i) = X_old.row(i);
            vel.row(i).setZero();
        }
    }

    if (effective_collision_mode == SPHERE_DISTANCE_IMPULSE) {
        applySphereDistanceImpulse();
    }
}

}  // namespace USTC_CG::mass_spring
