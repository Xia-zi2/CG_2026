#pragma once

#include <Eigen/SparseCholesky>
#include <memory>

#include "MassSpring.h"

namespace USTC_CG::mass_spring {
// Liu et al. 2013 fast mass-spring solver.
// It alternates between:
//   Local:  d_i = L_i * normalize(x_i - x_j)
//   Global: (M + h^2 L) x = M y + h^2 J d
class FastMassSpring : public MassSpring {
   public:
    FastMassSpring() = default;
    ~FastMassSpring() override = default;

    FastMassSpring(
        const Eigen::MatrixXd& X,
        const EdgeSet& E,
        const float stiffness,
        const float h);

    void step() override;

    unsigned max_iter = 100;

   protected:
    // Prefactorized global matrix A = M + h^2 L.
    Eigen::SparseMatrix<double> A_prefactorized;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    bool solver_ready = false;

    // A depends on mass, stiffness, h and the fixed/free vertex split.
    // If one of these scalar parameters changes, rebuild A before solving.
    double cached_mass = -1.0;
    double cached_stiffness = -1.0;
    double cached_h = -1.0;

    void prefactorizeSystemMatrix();
};
}  // namespace USTC_CG::mass_spring
