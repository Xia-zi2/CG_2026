#include "MassSpring.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

namespace USTC_CG::mass_spring {
MassSpring::MassSpring(const Eigen::MatrixXd& X, const EdgeSet& E)
{
    this->X = this->init_X = X;
    this->vel = Eigen::MatrixXd::Zero(X.rows(), X.cols());
    this->E = E;

    std::cout << "number of edges: " << E.size() << std::endl;
    std::cout << "init mass spring" << std::endl;

    // Compute the rest pose edge length
    for (const auto& e : E) {
        Eigen::Vector3d x0 = X.row(e.first);
        Eigen::Vector3d x1 = X.row(e.second);
        this->E_rest_length.push_back((x0 - x1).norm());
    }

    // Initialize the mask for Dirichlet boundary condition
    dirichlet_bc_mask.resize(X.rows(), false);

    // (HW_TODO) Fix two vertices, feel free to modify this
    unsigned n_fix = sqrt(X.rows());  // Here we assume the cloth is square
    dirichlet_bc_mask[0] = true;
    dirichlet_bc_mask[n_fix - 1] = true;
}

void MassSpring::step()
{
    Eigen::Vector3d acceleration_ext = gravity + wind_ext_acc;

    unsigned n_vertices = X.rows();

    // The reason to not use 1.0 as mass per vertex: the cloth gets heavier as
    // we increase the resolution
    double mass_per_vertex = mass / n_vertices;

    CollisionMode effective_collision_mode = collision_mode;
    if (time_integrator == SEMI_IMPLICIT_EULER &&
        collision_mode == SPHERE_PENALTY_ENERGY) {
        if (enable_debug_output) {
            std::cerr << "[MassSpring] SPHERE_PENALTY_ENERGY requires "
                         "IMPLICIT_EULER. Falling back to "
                         "SPHERE_PENALTY_FORCE in SEMI_IMPLICIT_EULER."
                      << std::endl;
        }
        effective_collision_mode = SPHERE_PENALTY_FORCE;
    }

    Eigen::MatrixXd acceleration_collision =
        Eigen::MatrixXd::Zero(X.rows(), X.cols());

    if (effective_collision_mode == SPHERE_PENALTY_FORCE) {
        acceleration_collision = getSphereCollisionAcceleration(
            sphere_center.cast<double>(), sphere_radius);
    }

    if (time_integrator == IMPLICIT_EULER) {
        // Implicit Euler
        TIC(step)

        Eigen::MatrixXd X_old = X;

        Eigen::MatrixXd vel_predict = vel;
        if (enable_damping) {
            vel_predict *= damping;
        }

        Eigen::MatrixXd Y = X_old + h * vel_predict;
        Y.rowwise() += (h * h * acceleration_ext).transpose();

        if (effective_collision_mode == SPHERE_PENALTY_FORCE) {
            Y += h * h * acceleration_collision;
        }

        for (int i = 0; i < n_vertices; i++) {
            if (dirichlet_bc_mask[i]) {
                Y.row(i) = X_old.row(i);
                vel.row(i).setZero();
            }
        }

        auto flatten = [](const Eigen::MatrixXd& M) {
            Eigen::VectorXd v(M.rows() * M.cols());

            for (int i = 0; i < M.rows(); i++) {
                for (int j = 0; j < M.cols(); j++) {
                    v[3 * i + j] = M(i, j);
                }
            }

            return v;
        };

        auto unflatten = [](const Eigen::VectorXd& v, int rows, int cols) {
            Eigen::MatrixXd M(rows, cols);

            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    M(i, j) = v[3 * i + j];
                }
            }

            return M;
        };

        auto implicit_objective = [&](const Eigen::MatrixXd& X_candidate) {
            double inertial_energy = 0.5 * mass_per_vertex / (h * h) *
                                     (X_candidate - Y).squaredNorm();

            double elastic_energy = 0.0;
            unsigned edge_id = 0;

            for (const auto& e : E) {
                Eigen::RowVector3d diff =
                    X_candidate.row(e.first) - X_candidate.row(e.second);

                double len = diff.norm();
                double rest_len = E_rest_length[edge_id];

                elastic_energy += 0.5 * stiffness * std::pow(len - rest_len, 2);

                edge_id++;
            }
            double collision_energy = 0.0;
            if (effective_collision_mode == SPHERE_PENALTY_ENERGY) {
                collision_energy = computeSphereCollisionEnergy(
                    X_candidate, sphere_center.cast<double>(), sphere_radius);
            }

            return inertial_energy + elastic_energy + collision_energy;
        };

        const int max_newton_iter = 20;
        const double grad_tol = 1e-6;
        const double step_tol = 1e-8;

        for (int iter = 0; iter < max_newton_iter; iter++) {
            Eigen::MatrixXd grad =
                mass_per_vertex / (h * h) * (X - Y) + computeGrad(stiffness);

            if (effective_collision_mode == SPHERE_PENALTY_ENERGY) {
                grad += computeSphereCollisionGrad(
                    X, sphere_center.cast<double>(), sphere_radius);
            }

            for (int i = 0; i < n_vertices; i++) {
                if (dirichlet_bc_mask[i]) {
                    grad.row(i).setZero();
                }
            }

            Eigen::VectorXd grad_vec = flatten(grad);

            if (grad_vec.norm() < grad_tol) {
                if (enable_debug_output) {
                    std::cout << "Newton converged at iter " << iter
                              << ", grad norm = " << grad_vec.norm()
                              << std::endl;
                }
                break;
            }

            Eigen::SparseMatrix<double> A(n_vertices * 3, n_vertices * 3);

            std::vector<Trip_d> mass_triplets;
            mass_triplets.reserve(n_vertices * 3);

            for (int i = 0; i < n_vertices; i++) {
                for (int d = 0; d < 3; d++) {
                    int idx = 3 * i + d;

                    if (dirichlet_bc_mask[i]) {
                        mass_triplets.emplace_back(idx, idx, 1.0);
                    }
                    else {
                        mass_triplets.emplace_back(
                            idx, idx, mass_per_vertex / (h * h));
                    }
                }
            }

            A.setFromTriplets(mass_triplets.begin(), mass_triplets.end());
            A += computeHessianSparse(stiffness);

            if (effective_collision_mode == SPHERE_PENALTY_ENERGY) {
                A += computeSphereCollisionHessianSparse(
                    X, sphere_center.cast<double>(), sphere_radius);
            }

            A.makeCompressed();

            Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
            solver.compute(A);

            if (solver.info() != Eigen::Success) {
                std::cerr << "Implicit Euler: matrix factorization failed."
                          << std::endl;
                break;
            }

            Eigen::VectorXd dx = solver.solve(-grad_vec);

            if (solver.info() != Eigen::Success) {
                std::cerr << "Implicit Euler: linear solve failed."
                          << std::endl;
                break;
            }

            Eigen::MatrixXd dX = unflatten(dx, n_vertices, 3);

            double alpha = 1.0;
            double E0 = implicit_objective(X);
            double descent = grad_vec.dot(dx);

            Eigen::MatrixXd X_trial = X;

            while (alpha > 1e-4) {
                X_trial = X + alpha * dX;

                for (int i = 0; i < n_vertices; i++) {
                    if (dirichlet_bc_mask[i]) {
                        X_trial.row(i) = X_old.row(i);
                    }
                }

                double E_trial = implicit_objective(X_trial);

                if (E_trial <= E0 + 1e-4 * alpha * descent) {
                    break;
                }

                alpha *= 0.5;
            }

            X = X_trial;

            if ((alpha * dX).norm() < step_tol) {
                if (enable_debug_output) {
                    std::cout << "Newton step small at iter " << iter
                              << ", step norm = " << (alpha * dX).norm()
                              << std::endl;
                }
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

        TOC(step)
    }
    else if (time_integrator == SEMI_IMPLICIT_EULER) {
        // Semi-implicit Euler
        Eigen::MatrixXd acceleration =
            -computeGrad(stiffness) / mass_per_vertex;
        acceleration.rowwise() += acceleration_ext.transpose();

        // Sphere penalty force is evaluated explicitly.
        if (effective_collision_mode == SPHERE_PENALTY_FORCE) {
            acceleration += acceleration_collision;
        }
        for (int i = 0; i < n_vertices; i++) {
            if (dirichlet_bc_mask[i]) {
                acceleration.row(i).setZero();
                vel.row(i).setZero();
            }
        }

        vel += h * acceleration;

        if (enable_damping) {
            vel *= damping;
        }

        for (int i = 0; i < n_vertices; i++) {
            if (dirichlet_bc_mask[i]) {
                vel.row(i).setZero();
            }
        }

        X += h * vel;
    }

    if (effective_collision_mode == SPHERE_DISTANCE_IMPULSE) {
        applySphereDistanceImpulse();
    }
}

bool MassSpring::applySphereDistanceImpulse()
{
    if (X.cols() != 3) {
        return false;
    }

    const int n_vertices = static_cast<int>(X.rows());
    const double R = collision_scale_factor * sphere_radius;
    if (R <= 0.0) {
        return false;
    }

    const Eigen::Vector3d center = sphere_center.cast<double>();
    const double eps = 1e-12;
    const double contact_radius = R + std::max(0.0, collision_contact_margin);
    const double e = std::max(0.0, collision_restitution);
    const double mu = std::max(0.0, collision_friction);

    int contact_count = 0;
    int projection_count = 0;
    int impulse_count = 0;

    for (int i = 0; i < n_vertices; ++i) {
        if (i < static_cast<int>(dirichlet_bc_mask.size()) &&
            dirichlet_bc_mask[i]) {
            continue;
        }

        Eigen::Vector3d xi = X.row(i).transpose();
        Eigen::Vector3d diff = xi - center;
        double d = diff.norm();

        if (d > contact_radius) {
            continue;
        }

        ++contact_count;

        Eigen::Vector3d n;
        if (d > eps) {
            n = diff / d;
        }
        else {
            Eigen::Vector3d vi = vel.row(i).transpose();
            if (vi.norm() > eps) {
                n = -vi.normalized();
            }
            else {
                n = Eigen::Vector3d(0.0, 0.0, 1.0);
            }
            d = 0.0;
        }

        Eigen::Vector3d vi = vel.row(i).transpose();
        bool velocity_changed = false;

        if (d < contact_radius) {
            const Eigen::Vector3d xi_before = xi;
            xi = center + contact_radius * n;
            X.row(i) = xi.transpose();
            if (sync_collision_velocity_with_projection && h > eps) {
                vi += (xi - xi_before) / h;
                velocity_changed = true;
            }
            ++projection_count;
        }

        const double vn = vi.dot(n);

        if (vn < 0.0) {
            const Eigen::Vector3d vt = vi - vn * n;
            const double normal_delta_speed = -(1.0 + e) * vn;
            vi += normal_delta_speed * n;

            const double vt_norm = vt.norm();
            if (vt_norm > eps && mu > 0.0) {
                const double tangent_scale =
                    std::max(0.0, 1.0 - (mu * normal_delta_speed) / vt_norm);
                const double vn_after = vi.dot(n);
                vi = vn_after * n + tangent_scale * vt;
            }

            velocity_changed = true;
            ++impulse_count;
        }

        if (velocity_changed) {
            vel.row(i) = vi.transpose();
        }
    }

    if (enable_debug_output && contact_count > 0) {
        std::cerr << "[MassSpring] DISTANCE_IMPULSE contacts="
                  << contact_count << ", projected=" << projection_count
                  << ", impulses=" << impulse_count << std::endl;
    }

    return contact_count > 0;
}

double MassSpring::computeEnergy(double stiffness)
{
    double sum = 0.;
    unsigned i = 0;
    for (const auto& e : E) {
        auto diff = X.row(e.first) - X.row(e.second);
        auto l = E_rest_length[i];
        sum += 0.5 * stiffness * std::pow((diff.norm() - l), 2);
        i++;
    }
    return sum;
}

Eigen::MatrixXd MassSpring::computeGrad(double stiffness)
{
    Eigen::MatrixXd g = Eigen::MatrixXd::Zero(X.rows(), X.cols());

    unsigned i = 0;
    for (const auto& e : E) {
        int v0 = e.first;
        int v1 = e.second;

        Eigen::RowVector3d x0 = X.row(v0);
        Eigen::RowVector3d x1 = X.row(v1);

        Eigen::RowVector3d diff = x0 - x1;
        double len = diff.norm();
        double rest_len = E_rest_length[i];

        if (len > 1e-12) {
            Eigen::RowVector3d grad = stiffness * (len - rest_len) * diff / len;

            g.row(v0) += grad;
            g.row(v1) -= grad;
        }

        i++;
    }

    return g;
}

Eigen::SparseMatrix<double> MassSpring::computeHessianSparse(double stiffness)
{
    unsigned n_vertices = X.rows();
    Eigen::SparseMatrix<double> H(n_vertices * 3, n_vertices * 3);

    std::vector<Trip_d> triplets;
    triplets.reserve(E.size() * 4 * 9);

    unsigned edge_id = 0;
    const double k = stiffness;
    const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();

    auto add_block = [&](int row_v, int col_v, const Eigen::Matrix3d& block) {
        if (dirichlet_bc_mask[row_v] || dirichlet_bc_mask[col_v]) {
            return;
        }

        for (int a = 0; a < 3; a++) {
            for (int b = 0; b < 3; b++) {
                triplets.emplace_back(
                    3 * row_v + a, 3 * col_v + b, block(a, b));
            }
        }
    };

    for (const auto& e : E) {
        int v0 = e.first;
        int v1 = e.second;

        Eigen::Vector3d x0 = X.row(v0).transpose();
        Eigen::Vector3d x1 = X.row(v1).transpose();

        Eigen::Vector3d diff = x0 - x1;
        double len = diff.norm();
        double rest_len = E_rest_length[edge_id];

        if (len > 1e-12) {
            Eigen::Matrix3d H_local = k * ((1.0 - rest_len / len) * I +
                                           (rest_len / (len * len * len)) *
                                               (diff * diff.transpose()));

            if (enable_make_SPD) {
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(H_local);
                Eigen::Vector3d evals = es.eigenvalues();

                for (int j = 0; j < 3; j++) {
                    if (evals[j] < 0.0) {
                        evals[j] = 0.0;
                    }
                }

                H_local = es.eigenvectors() * evals.asDiagonal() *
                          es.eigenvectors().transpose();
            }

            add_block(v0, v0, H_local);
            add_block(v0, v1, -H_local);
            add_block(v1, v0, -H_local);
            add_block(v1, v1, H_local);
        }

        edge_id++;
    }

    H.setFromTriplets(triplets.begin(), triplets.end());
    H.makeCompressed();
    return H;
}

bool MassSpring::checkSPD(const Eigen::SparseMatrix<double>& A)
{
    // Eigen::SimplicialLDLT<SparseMatrix_d> ldlt(A);
    // return ldlt.info() == Eigen::Success;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(A);
    auto eigen_values = es.eigenvalues();
    return eigen_values.minCoeff() >= 1e-10;
}

void MassSpring::reset()
{
    std::cout << "reset" << std::endl;
    this->X = this->init_X;
    this->vel.setZero();
}

Eigen::MatrixXd MassSpring::getSphereCollisionAcceleration(
    Eigen::Vector3d center,
    double radius)
{
    Eigen::MatrixXd acceleration = Eigen::MatrixXd::Zero(X.rows(), X.cols());

    const double eps = 1e-12;
    const double R = collision_scale_factor * radius;
    const double mass_per_vertex = mass / X.rows();

    for (int i = 0; i < X.rows(); i++) {
        if (dirichlet_bc_mask[i]) {
            continue;
        }

        Eigen::Vector3d xi = X.row(i).transpose();
        Eigen::Vector3d diff = xi - center;
        double d = diff.norm();

        Eigen::Vector3d n;
        if (d < eps) {
            n = Eigen::Vector3d(0.0, 0.0, 1.0);
            d = 0.0;
        }
        else {
            n = diff / d;
        }

        const double penetration = R - d;

        if (penetration > 0.0) {
            const Eigen::Vector3d force = collision_penalty_k * penetration * n;

            acceleration.row(i) = (force / mass_per_vertex).transpose();
        }
    }

    return acceleration;
}

bool MassSpring::set_dirichlet_bc_mask(const std::vector<bool>& mask)
{
    if (mask.size() == X.rows()) {
        dirichlet_bc_mask = mask;
        return true;
    }
    else
        return false;
}

bool MassSpring::update_dirichlet_bc_vertices(const MatrixXd& control_vertices)
{
    for (int i = 0; i < dirichlet_bc_control_pair.size(); i++) {
        int idx = dirichlet_bc_control_pair[i].first;
        int control_idx = dirichlet_bc_control_pair[i].second;
        X.row(idx) = control_vertices.row(control_idx);
    }

    return true;
}

bool MassSpring::init_dirichlet_bc_vertices_control_pair(
    const MatrixXd& control_vertices,
    const std::vector<bool>& control_mask)
{
    if (control_mask.size() != control_vertices.rows())
        return false;

    // TODO: optimize this part from O(n) to O(1)
    // First, get selected_control_vertices
    std::vector<VectorXd> selected_control_vertices;
    std::vector<int> selected_control_idx;
    for (int i = 0; i < control_mask.size(); i++) {
        if (control_mask[i]) {
            selected_control_vertices.push_back(control_vertices.row(i));
            selected_control_idx.push_back(i);
        }
    }

    // Then update mass spring fixed vertices
    for (int i = 0; i < dirichlet_bc_mask.size(); i++) {
        if (dirichlet_bc_mask[i]) {
            // O(n^2) nearest point search, can be optimized
            // -----------------------------------------
            int nearest_idx = 0;
            double nearst_dist = 1e6;
            VectorXd X_i = X.row(i);
            for (int j = 0; j < selected_control_vertices.size(); j++) {
                double dist = (X_i - selected_control_vertices[j]).norm();
                if (dist < nearst_dist) {
                    nearst_dist = dist;
                    nearest_idx = j;
                }
            }
            //-----------------------------------------

            X.row(i) = selected_control_vertices[nearest_idx];
            dirichlet_bc_control_pair.push_back(
                std::make_pair(i, selected_control_idx[nearest_idx]));
        }
    }

    return true;
}

double MassSpring::computeSphereCollisionEnergy(
    const Eigen::MatrixXd& X_candidate,
    Eigen::Vector3d center,
    double radius) const
{
    double energy = 0.0;

    const double eps = 1e-12;
    const double R = collision_scale_factor * radius;
    const double k = collision_penalty_k;

    for (int i = 0; i < X_candidate.rows(); i++) {
        if (dirichlet_bc_mask[i]) {
            continue;
        }

        Eigen::Vector3d xi = X_candidate.row(i).transpose();
        Eigen::Vector3d diff = xi - center;
        double d = diff.norm();

        if (d < eps) {
            d = 0.0;
        }

        double penetration = R - d;

        if (penetration > 0.0) {
            energy += 0.5 * k * penetration * penetration;
        }
    }

    return energy;
}

Eigen::MatrixXd MassSpring::computeSphereCollisionGrad(
    const Eigen::MatrixXd& X_candidate,
    Eigen::Vector3d center,
    double radius) const
{
    Eigen::MatrixXd grad =
        Eigen::MatrixXd::Zero(X_candidate.rows(), X_candidate.cols());

    const double eps = 1e-12;
    const double R = collision_scale_factor * radius;
    const double k = collision_penalty_k;

    for (int i = 0; i < X_candidate.rows(); i++) {
        if (dirichlet_bc_mask[i]) {
            continue;
        }

        Eigen::Vector3d xi = X_candidate.row(i).transpose();
        Eigen::Vector3d diff = xi - center;
        double d = diff.norm();

        Eigen::Vector3d n;
        if (d < eps) {
            n = Eigen::Vector3d(0.0, 0.0, 1.0);
            d = 0.0;
        }
        else {
            n = diff / d;
        }

        double penetration = R - d;

        if (penetration > 0.0) {
            Eigen::Vector3d gi = -k * penetration * n;

            grad.row(i) = gi.transpose();
        }
    }

    return grad;
}

Eigen::SparseMatrix<double> MassSpring::computeSphereCollisionHessianSparse(
    const Eigen::MatrixXd& X_candidate,
    Eigen::Vector3d center,
    double radius) const
{
    const int n_vertices = X_candidate.rows();
    Eigen::SparseMatrix<double> H(n_vertices * 3, n_vertices * 3);

    std::vector<Trip_d> triplets;
    triplets.reserve(n_vertices * 9);

    const double eps = 1e-12;
    const double R = collision_scale_factor * radius;
    const double k = collision_penalty_k;
    const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();

    for (int i = 0; i < n_vertices; i++) {
        if (dirichlet_bc_mask[i]) {
            continue;
        }

        Eigen::Vector3d xi = X_candidate.row(i).transpose();
        Eigen::Vector3d diff = xi - center;
        double d = diff.norm();

        double penetration = R - d;
        if (d < eps) {
            penetration = R;
        }

        if (penetration > 0.0) {
            Eigen::Matrix3d H_local;

            if (d < eps) {
                H_local = k * I;
            }
            else {
                Eigen::Vector3d n = diff / d;
                Eigen::Matrix3d nnT = n * n.transpose();

                H_local = k * nnT - k * (penetration / d) * (I - nnT);
            }

            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(H_local);
            Eigen::Vector3d evals = es.eigenvalues();
            for (int j = 0; j < 3; j++) {
                if (evals[j] < 0.0) {
                    evals[j] = 0.0;
                }
            }
            H_local = es.eigenvectors() * evals.asDiagonal() *
                      es.eigenvectors().transpose();

            for (int a = 0; a < 3; a++) {
                for (int b = 0; b < 3; b++) {
                    triplets.emplace_back(3 * i + a, 3 * i + b, H_local(a, b));
                }
            }
        }
    }

    H.setFromTriplets(triplets.begin(), triplets.end());
    H.makeCompressed();

    return H;
}

}  // namespace USTC_CG::mass_spring
