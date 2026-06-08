#pragma once
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <memory>

#include "PoissonRegion.h"
#include "common/image.h"

namespace USTC_CG
{
class PoissonSolver
{
   public:
    PoissonSolver(
        std::shared_ptr<const PoissonRegion> region,
        std::shared_ptr<Image> source_image,
        std::shared_ptr<Image> target_image);

    virtual ~PoissonSolver() = default;

    virtual void build();

    virtual std::shared_ptr<Image> solve() = 0;

   protected:
    std::shared_ptr<const PoissonRegion> region_;
    bool factorized_ = false;
    std::shared_ptr<Image> source_image_;
    std::shared_ptr<Image> target_image_;

    Eigen::SparseMatrix<double> system_matrix_;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver_;

    virtual void build_system_matrix();
    void factorize_matrix();
    Eigen::VectorXd solve_channel(const Eigen::VectorXd& b) const;

    unsigned char clamp_to_uchar(double x) const;

    virtual Eigen::VectorXd build_rhs_for_channel(int channel) const = 0;
};
}  // namespace USTC_CG