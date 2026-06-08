#pragma once

#include <vector>

#include "common/image.h"

namespace USTC_CG
{

class HoleFiller
{
   public:
    using Mask = std::vector<std::vector<unsigned char>>;

    enum class Type
    {
        ANN
    };

   public:
    explicit HoleFiller(Type type = Type::ANN) : type_(type)
    {
    }

    ~HoleFiller() = default;

    void fill(
        const Image& warped_image,
        Image& output_image,
        const Mask& written_mask,
        const Mask& coverage_mask) const;

   private:
    struct Sample
    {
        int x = 0;
        int y = 0;
        std::vector<unsigned char> color;
    };

    void fill_ann(
        const Image& warped_image,
        Image& output_image,
        const Mask& written_mask,
        const Mask& coverage_mask) const;

    bool is_fillable_hole(
        int x,
        int y,
        const Mask& known_mask,
        const Mask& coverage_mask) const;

    int count_known_neighbors(
        int x,
        int y,
        const Mask& known_mask,
        const Mask& coverage_mask,
        int radius) const;

    int count_coverage_neighbors(
        int x,
        int y,
        const Mask& coverage_mask,
        int radius) const;

   private:
    Type type_ = Type::ANN;

    int knn_k_ = 8;
    float max_search_radius_ = 18.0f;
    int max_iterations_ = 12;

    // 只在足够“内部”的 coverage 区域填，避免边界向外多长一层。
    int coverage_guard_radius_ = 1;
    int min_coverage_neighbors_ = 7;  // 3x3 内至少 7 个 coverage 像素

    // 当前迭代中，待填点附近必须已有足够多已知样本。
    int known_neighbor_radius_ = 2;
    int min_known_neighbors_ = 3;

    float eps_ = 1e-6f;
};

}  // namespace USTC_CG
