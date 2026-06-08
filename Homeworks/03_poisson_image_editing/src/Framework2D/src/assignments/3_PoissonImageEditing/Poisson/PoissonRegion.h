#pragma once

#include <memory>
#include <vector>
#include <Eigen/Sparse>
#include "common/image.h"
namespace USTC_CG
{

struct RegionPixel
{
    int id = -1;
    int source_x_ = 0;
    int source_y_ = 0;
    int target_x_ = 0;
    int target_y_ = 0;
};

struct OutsideNeighbor
{
    int source_x_ = 0;
    int source_y_ = 0;
    int target_x_ = 0;
    int target_y_ = 0;
};

class PoissonRegion
{
   public:
    PoissonRegion(
        std::shared_ptr<Image> mask,
        int source_anchor_x,
        int source_anchor_y,
        int target_anchor_x,
        int target_anchor_y)
    : mask_(mask),
      source_anchor_x_(source_anchor_x),
      source_anchor_y_(source_anchor_y),
      target_anchor_x_(target_anchor_x),
      target_anchor_y_(target_anchor_y)
    {
    }

    void build();

    bool empty() const;
    int num_pixels() const;

    const std::vector<RegionPixel>& pixels() const;
    const RegionPixel& pixel(int id) const;

    const Eigen::SparseMatrix<double>& A() const;

    const std::vector<int>& boundary_pixel_ids() const;
    const std::vector<OutsideNeighbor>& outside_neighbors(int id) const;

    void update_target_anchor(int target_anchor_x, int target_anchor_y);

   protected:
    Eigen::SparseMatrix<double> A_;

   private:
    std::shared_ptr<Image> mask_;
    bool fully_inside_target_ = true;

    int source_anchor_x_;
    int source_anchor_y_;
    int target_anchor_x_;
    int target_anchor_y_;

    std::vector<RegionPixel> pixels_;
    std::vector<std::vector<int>> source_id_map_;

    std::vector<int> boundary_pixel_ids_;
    std::vector<std::vector<OutsideNeighbor>> outside_neighbors_;

    void build_matrix_A();
    void build_pixels();
    void build_source_id_map();
    void build_boundary_info();

    void map_source_to_target(int sx, int sy, int& tx, int& ty) const;
    int id_at_source(int sx, int sy) const;

};
}  // namespace USTC_CG