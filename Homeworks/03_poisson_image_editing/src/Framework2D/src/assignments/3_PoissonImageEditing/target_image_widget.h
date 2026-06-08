#pragma once

#include "Poisson/PoissonRegion.h"
#include "Poisson/PoissonSolver.h"
#include "common/image_widget.h"
#include "source_image_widget.h"

namespace USTC_CG
{
class TargetImageWidget : public ImageWidget
{
   public:
    enum CloneType
    {
        kDefault = 0,
        kPaste = 1,
        kSeamless = 2,
        kMixedGradient = 3,
        kColorConstraint = 4,
        kAlphaMatte = 5
    };

    explicit TargetImageWidget(
        const std::string& label,
        const std::string& filename);
    virtual ~TargetImageWidget() noexcept = default;

    void draw() override;
    void set_source(std::shared_ptr<SourceImageWidget> source);
    void set_realtime(bool flag);
    void invalidate_clone_cache();
    void restore();

    void set_paste();
    void set_seamless();
    void set_mixed_gradient();
    void set_color_constraint();
    void set_alpha_matte();

    void clone();

   private:
    void mouse_click_event();
    void mouse_move_event();
    void mouse_release_event();

    std::shared_ptr<PoissonRegion> build_region_from_current_mouse() const;
    ImVec2 mouse_pos_in_canvas() const;

    std::shared_ptr<Image> back_up_;
    std::shared_ptr<SourceImageWidget> source_image_;
    CloneType clone_type_ = kDefault;

    ImVec2 mouse_position_;
    bool edit_status_ = false;
    bool flag_realtime_updating = false;

    std::shared_ptr<PoissonRegion> cached_region_;
    std::unique_ptr<PoissonSolver> cached_solver_;
};
}  // namespace USTC_CG