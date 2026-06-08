#include "PoissonRegion.h"

namespace USTC_CG
{
bool PoissonRegion::empty() const
{
    return pixels_.empty();
}

int PoissonRegion::num_pixels() const
{
    return static_cast<int>(pixels_.size());
}

const std::vector<RegionPixel>& PoissonRegion::pixels() const
{
    return pixels_;
}

const RegionPixel& PoissonRegion::pixel(int id) const
{
    assert(0 <= id && id < num_pixels());
    return pixels_[id];
}

const Eigen::SparseMatrix<double>& PoissonRegion::A() const
{
    return A_;
}
const std::vector<int>& PoissonRegion::boundary_pixel_ids() const
{
    return boundary_pixel_ids_;
}

const std::vector<OutsideNeighbor>& PoissonRegion::outside_neighbors(
    int id) const
{
    assert(0 <= id && id < num_pixels());
    return outside_neighbors_[id];
}

void PoissonRegion::map_source_to_target(int sx, int sy, int& tx, int& ty) const
{
    tx = target_anchor_x_ + (sx - source_anchor_x_);
    ty = target_anchor_y_ + (sy - source_anchor_y_);
}

void PoissonRegion::build_pixels()
{
    pixels_.clear();
    if (!mask_)
        return;

    int next_id = 0;
    for (int sx = 0; sx < mask_->width(); ++sx)
    {
        for (int sy = 0; sy < mask_->height(); ++sy)
        {
            if (mask_->get_pixel(sx, sy)[0] <= 0)
                continue;

            int tx = 0, ty = 0;
            map_source_to_target(sx, sy, tx, ty);

            RegionPixel p;
            p.id = next_id++;
            p.source_x_ = sx;
            p.source_y_ = sy;
            p.target_x_ = tx;
            p.target_y_ = ty;
            pixels_.push_back(p);
        }
    }
}
void PoissonRegion::build_source_id_map()
{
    source_id_map_.assign(
        mask_->width(), std::vector<int>(mask_->height(), -1));
    for (const auto& p : pixels_)
    {
        source_id_map_[p.source_x_][p.source_y_] = p.id;
    }
}
void PoissonRegion::build()
{
    pixels_.clear();
    boundary_pixel_ids_.clear();
    outside_neighbors_.clear();
    source_id_map_.clear();
    A_.resize(0, 0);

    build_pixels();
    build_source_id_map();
    build_boundary_info();
    build_matrix_A();
}

void PoissonRegion::update_target_anchor(
    int target_anchor_x,
    int target_anchor_y)
{
    target_anchor_x_ = target_anchor_x;
    target_anchor_y_ = target_anchor_y;

    for (auto& p : pixels_)
    {
        map_source_to_target(
            p.source_x_, p.source_y_, p.target_x_, p.target_y_);
    }

    for (int id : boundary_pixel_ids_)
    {
        for (auto& nbr : outside_neighbors_[id])
        {
            map_source_to_target(
                nbr.source_x_, nbr.source_y_, nbr.target_x_, nbr.target_y_);
        }
    }
}

int PoissonRegion::id_at_source(int sx, int sy) const
{
    if (sx < 0 || sx >= mask_->width() || sy < 0 || sy >= mask_->height())
        return -1;
    return source_id_map_[sx][sy];
}

void PoissonRegion::build_boundary_info()
{
    boundary_pixel_ids_.clear();
    outside_neighbors_.assign(num_pixels(), {});

    static const int dx[4] = { -1, 1, 0, 0 };
    static const int dy[4] = { 0, 0, -1, 1 };

    for (int id = 0; id < num_pixels(); ++id)
    {
        const auto& p = pixels_[id];
        bool is_boundary = false;
        outside_neighbors_[id].reserve(4);

        for (int k = 0; k < 4; ++k)
        {
            const int nsx = p.source_x_ + dx[k];
            const int nsy = p.source_y_ + dy[k];

            if (id_at_source(nsx, nsy) == -1)
            {
                is_boundary = true;
                int ntx, nty;
                map_source_to_target(
                    nsx, nsy, ntx, nty); 
                outside_neighbors_[id].push_back(
                    OutsideNeighbor{ nsx, nsy, ntx, nty });
            }
        }
        if (is_boundary)
            boundary_pixel_ids_.push_back(id);
    }
}

// Standard 4-neighbor discrete Laplacian on Ω.
// Contributions of neighbors outside Ω are moved to the RHS later.
void PoissonRegion::build_matrix_A()
{
    const int n = num_pixels();
    A_.resize(n, n);

    if (n == 0)
    {
        A_.setZero();
        return;
    }

    static const int dx[4] = { -1, 1, 0, 0 };
    static const int dy[4] = { 0, 0, -1, 1 };

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(n * 5);

    for (int id = 0; id < n; ++id)
    {
        const auto& p = pixels_[id];
        triplets.emplace_back(id, id, 4.0);

        for (int k = 0; k < 4; ++k)
        {
            int nsx = p.source_x_ + dx[k];
            int nsy = p.source_y_ + dy[k];

            int nid = id_at_source(nsx, nsy);

            if (nid != -1)
            {
                triplets.emplace_back(id, nid, -1.0);
            }
        }
    }

    A_.setFromTriplets(triplets.begin(), triplets.end());
    A_.makeCompressed();
}
}  // namespace USTC_CG