#include "source_image_widget.h"

#include <algorithm>
#include <cmath>

#include "target_image_widget.h"

namespace USTC_CG
{
using uchar = unsigned char;

SourceImageWidget::SourceImageWidget(
    const std::string& label,
    const std::string& filename)
    : ImageWidget(label, filename)
{
    if (data_)
    {
        selected_region_mask_ =
            std::make_shared<Image>(data_->width(), data_->height(), 1);
        subject_mask_ =
            std::make_shared<Image>(data_->width(), data_->height(), 1);

        for (int i = 0; i < data_->width(); ++i)
            for (int j = 0; j < data_->height(); ++j)
            {
                selected_region_mask_->set_pixel(i, j, { 0 });
                subject_mask_->set_pixel(i, j, { 0 });
            }
    }
}

void SourceImageWidget::draw()
{
    ImageWidget::draw();
    if (flag_enable_selecting_region_)
        select_region();
}

void SourceImageWidget::enable_selecting(bool flag)
{
    flag_enable_selecting_region_ = flag;
}

void SourceImageWidget::select_region()
{
    ImGui::SetCursorScreenPos(position_);
    ImGui::InvisibleButton(
        label_.c_str(),
        ImVec2(
            static_cast<float>(image_width_),
            static_cast<float>(image_height_)),
        ImGuiButtonFlags_MouseButtonLeft);

    bool is_hovered_ = ImGui::IsItemHovered();

    if (is_hovered_ && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
    {
        mouse_click_event();
    }
    mouse_move_event();
    if (!ImGui::IsMouseDown(ImGuiMouseButton_Left))
        mouse_release_event();

    if (selected_shape_)
    {
        Shape::Config s = { .bias = { position_.x, position_.y },
                            .line_color = { 255, 0, 0, 255 },
                            .line_thickness = 2.0f };
        selected_shape_->draw(s);
    }
}

std::shared_ptr<Image> SourceImageWidget::get_region_mask()
{
    return selected_region_mask_;
}

std::shared_ptr<Image> SourceImageWidget::get_subject_mask()
{
    return subject_mask_;
}

std::shared_ptr<Image> SourceImageWidget::get_data()
{
    return data_;
}

ImVec2 SourceImageWidget::get_position() const
{
    return start_;
}

void SourceImageWidget::set_target(std::shared_ptr<TargetImageWidget> target)
{
    target_image_ = target;
}

void SourceImageWidget::mouse_click_event()
{
    if (!draw_status_)
    {
        draw_status_ = true;
        start_ = end_ = mouse_pos_in_canvas();

        switch (region_type_)
        {
            case USTC_CG::SourceImageWidget::kDefault: break;
            case USTC_CG::SourceImageWidget::kRect:
            {
                selected_shape_ =
                    std::make_unique<Rect>(start_.x, start_.y, end_.x, end_.y);
                break;
            }
            case USTC_CG::SourceImageWidget::kFreehand:
            {
                selected_shape_ =
                    std::make_unique<Freehand>(start_.x, start_.y);
                break;
            }
            default: break;
        }
    }
}

void SourceImageWidget::mouse_move_event()
{
    if (draw_status_)
    {
        end_ = mouse_pos_in_canvas();
        if (selected_shape_)
            selected_shape_->update(end_.x, end_.y);
    }
}

void SourceImageWidget::mouse_release_event()
{
    if (draw_status_ && selected_shape_)
    {
        draw_status_ = false;

        if (auto* freehand_shape =
                dynamic_cast<Freehand*>(selected_shape_.get()))
        {
            freehand_shape->finish_drawing();
        }

        update_selected_region();
        selected_shape_.reset();
    }
}

ImVec2 SourceImageWidget::mouse_pos_in_canvas() const
{
    ImGuiIO& io = ImGui::GetIO();
    const ImVec2 mouse_pos_in_canvas(
        std::clamp<float>(io.MousePos.x - position_.x, 0, (float)image_width_),
        std::clamp<float>(
            io.MousePos.y - position_.y, 0, (float)image_height_));
    return mouse_pos_in_canvas;
}

void SourceImageWidget::update_selected_region()
{
    if (selected_shape_ == nullptr)
        return;

    std::vector<std::pair<int, int>> interior_pixels =
        selected_shape_->get_interior_pixels();

    std::shared_ptr<Image> dst_mask =
        (mask_edit_mode_ == kEditROI) ? selected_region_mask_ : subject_mask_;

    if (!dst_mask)
        return;

    for (int i = 0; i < dst_mask->width(); ++i)
        for (int j = 0; j < dst_mask->height(); ++j)
            dst_mask->set_pixel(i, j, { 0 });

    for (const auto& pixel : interior_pixels)
    {
        int x = pixel.first;
        int y = pixel.second;
        if (x < 0 || x >= dst_mask->width() || y < 0 || y >= dst_mask->height())
            continue;
        dst_mask->set_pixel(x, y, { 255 });
    }

    if (auto target = target_image_.lock())
    {
        target->invalidate_clone_cache();
    }
}

void SourceImageWidget::clear_shape()
{
    selected_shape_.reset();

    if (selected_region_mask_)
    {
        for (int i = 0; i < selected_region_mask_->width(); ++i)
            for (int j = 0; j < selected_region_mask_->height(); ++j)
                selected_region_mask_->set_pixel(i, j, { 0 });
    }

    if (auto target = target_image_.lock())
    {
        target->invalidate_clone_cache();
    }
}

void SourceImageWidget::clear_subject_mask()
{
    if (subject_mask_)
    {
        for (int i = 0; i < subject_mask_->width(); ++i)
            for (int j = 0; j < subject_mask_->height(); ++j)
                subject_mask_->set_pixel(i, j, { 0 });
    }

    if (auto target = target_image_.lock())
    {
        target->invalidate_clone_cache();
    }
}
}  // namespace USTC_CG