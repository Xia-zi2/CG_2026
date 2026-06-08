#pragma once

#include <vector>

#include "common/image_widget.h"
#include "warper/warper.h"

namespace USTC_CG
{
// Image component for warping and other functions
class WarpingWidget : public ImageWidget
{
   public:
    using Mask = std::vector<std::vector<unsigned char>>;

    explicit WarpingWidget(
        const std::string& label,
        const std::string& filename);
    virtual ~WarpingWidget() noexcept = default;

    void draw() override;

    // Simple edit functions
    void invert();
    void mirror(bool is_horizontal, bool is_vertical);
    void gray_scale();
    void warping();
    void restore();

    // Enumeration for supported warping types.
    enum WarpingType
    {
        kDefault = 0,
        kFisheye = 1,
        kIDW = 2,
        kIDW_A = 3,
        kIDW_R = 4,
        kRBF_AffineFirst = 5,
        kRBF_Full = 6,
    };

    enum HoleFillingType
    {
        kHoleANN = 0,
    };
    // Warping type setters.
    void set_default();
    void set_fisheye();
    void set_IDW();
    void set_IDW_A();
    void set_IDW_R();
    void set_RBF_affine_first();
    void set_RBF_full();

    void set_hole_ann();

    // Point selecting interaction
    void enable_selecting(bool flag);
    void select_points();
    void init_selections();
    void fill_holes();

   private:
    void init_masks(int width, int height);
    void rasterize_coverage_mask();

   private:
    HoleFillingType hole_filling_type_ = kHoleANN;
    bool holes_filled_ = false;

    std::shared_ptr<Image> warped_backup_;

    Mask written_mask_;
    Mask coverage_mask_;

    // Store the original image data
    std::shared_ptr<Image> back_up_;
    // The selected point couples for image warping
    std::vector<ImVec2> start_points_, end_points_;
    std::shared_ptr<Warper> current_warper_ = nullptr;

    ImVec2 start_, end_;
    bool flag_enable_selecting_points_ = false;
    bool draw_status_ = false;
    WarpingType warping_type_ = kDefault;

    float rbf_radius_ = 80.0f;
};
}  // namespace USTC_CG
