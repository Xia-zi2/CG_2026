#pragma once
#include "common/image_widget.h"
#include "shapes/freehand.h"
#include "shapes/rect.h"
#include "shapes/shape.h"

namespace USTC_CG
{
class TargetImageWidget;

class SourceImageWidget : public ImageWidget
{
   public:
    enum RegionType
    {
        kDefault = 0,
        kRect = 1,
        kFreehand = 2
    };

    enum MaskEditMode
    {
        kEditROI = 0,
        kEditSubject = 1
    };

    explicit SourceImageWidget(
        const std::string& label,
        const std::string& filename);
    virtual ~SourceImageWidget() noexcept = default;

    void draw() override;

    void enable_selecting(bool flag);
    void select_region();

    std::shared_ptr<Image> get_region_mask();
    std::shared_ptr<Image> get_subject_mask();
    std::shared_ptr<Image> get_data();

    ImVec2 get_position() const;

    void clear_shape();
    void clear_subject_mask();

    void set_target(std::shared_ptr<TargetImageWidget> target);

    RegionType get_region_type() const
    {
        return region_type_;
    }

    void set_region_type(RegionType type)
    {
        region_type_ = type;
    }

    void set_mask_edit_mode(MaskEditMode mode)
    {
        mask_edit_mode_ = mode;
    }

    MaskEditMode get_mask_edit_mode() const
    {
        return mask_edit_mode_;
    }

   private:
    void mouse_click_event();
    void mouse_move_event();
    void mouse_release_event();

    ImVec2 mouse_pos_in_canvas() const;
    void update_selected_region();

    RegionType region_type_ = kRect;
    MaskEditMode mask_edit_mode_ = kEditROI;

    std::unique_ptr<Shape> selected_shape_;

    std::shared_ptr<Image> selected_region_mask_;
    std::shared_ptr<Image> subject_mask_;

    ImVec2 start_, end_;
    bool flag_enable_selecting_region_ = false;
    bool draw_status_ = false;
    std::weak_ptr<TargetImageWidget> target_image_;
};

}  // namespace USTC_CG