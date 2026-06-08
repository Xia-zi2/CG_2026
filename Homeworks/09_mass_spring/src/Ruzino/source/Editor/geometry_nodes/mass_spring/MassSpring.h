#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <chrono>
#include <iostream>
#include <set>
#include <vector>

#include "utils.h"

#define TIC(name) auto start_##name = std::chrono::high_resolution_clock::now();
#define TOC(name)                                                           \
    auto end_##name = std::chrono::high_resolution_clock::now();            \
    if (enable_time_profiling)                                              \
        std::cout << "Time taken by " << #name << ": "                      \
                  << std::chrono::duration_cast<std::chrono::microseconds>( \
                         end_##name - start_##name)                         \
                         .count()                                           \
                  << " microseconds\n";

namespace USTC_CG::mass_spring {

using namespace Eigen;
using Edge = std::pair<int, int>;
using EdgeSet = std::set<Edge>;
using MatrixXd = Eigen::MatrixXd;
using SparseMatrix_d = Eigen::SparseMatrix<double>;
using Trip_d = Eigen::Triplet<double>;

class MassSpring {
   public:
    MassSpring() = default;
    virtual ~MassSpring() = default;

    enum TimeIntegrator { IMPLICIT_EULER = 0, SEMI_IMPLICIT_EULER = 1 };
    enum CollisionMode {
        NO_COLLISION = 0,
        SPHERE_PENALTY_FORCE = 1,
        SPHERE_PENALTY_ENERGY = 2,
        SPHERE_DISTANCE_IMPULSE = 3
    };

    MassSpring(const Eigen::MatrixXd& X, const EdgeSet& E);

    virtual void step();
    void reset();

    virtual double computeEnergy(double stiffness);
    virtual Eigen::MatrixXd computeGrad(double stiffness);
    virtual Eigen::SparseMatrix<double> computeHessianSparse(double stiffness);
    bool checkSPD(const Eigen::SparseMatrix<double>& A);

    Eigen::MatrixXd getVelocity() const { return vel; }
    Eigen::MatrixXd getX() const { return X; }

    Eigen::MatrixXd getSphereCollisionAcceleration(
        Eigen::Vector3d center,
        double radius);
    double computeSphereCollisionEnergy(
        const Eigen::MatrixXd& X_candidate,
        Eigen::Vector3d center,
        double radius) const;
    Eigen::MatrixXd computeSphereCollisionGrad(
        const Eigen::MatrixXd& X_candidate,
        Eigen::Vector3d center,
        double radius) const;
    Eigen::SparseMatrix<double> computeSphereCollisionHessianSparse(
        const Eigen::MatrixXd& X_candidate,
        Eigen::Vector3d center,
        double radius) const;

    bool set_dirichlet_bc_mask(const std::vector<bool>& mask);
    bool update_dirichlet_bc_vertices(const MatrixXd& control_vertices);
    bool init_dirichlet_bc_vertices_control_pair(
        const MatrixXd& control_vertices,
        const std::vector<bool>& control_mask);

    double stiffness = 1000.0;
    double damping = 0.995;
    TimeIntegrator time_integrator = IMPLICIT_EULER;
    CollisionMode collision_mode = NO_COLLISION;
    double mass = 1.0;
    double h = 1e-2;
    Eigen::Vector3d gravity = { 0, 0, -9.8 };
    Eigen::Vector3d wind_ext_acc = { 0, 0, 0 };

    double collision_penalty_k = 10000.0;
    double collision_scale_factor = 1.1;
    Eigen::Vector3f sphere_center = Eigen::Vector3f(0, -0.5, 0.2);
    double sphere_radius = 0.4;
    double collision_restitution = 0.0;
    double collision_friction = 0.3;
    double collision_contact_margin = 1e-6;
    bool sync_collision_velocity_with_projection = true;

    bool enable_time_profiling = false;
    bool enable_make_SPD = true;
    bool enable_check_SPD = false;
    bool enable_damping = true;
    bool enable_debug_output = false;

   protected:
    bool applySphereDistanceImpulse();

    Eigen::MatrixXd init_X;
    Eigen::MatrixXd X;
    Eigen::MatrixXd vel;
    EdgeSet E;
    std::vector<double> E_rest_length;
    std::vector<bool> dirichlet_bc_mask;
    std::vector<std::pair<int, int>> dirichlet_bc_control_pair;
};

}  // namespace USTC_CG::mass_spring
