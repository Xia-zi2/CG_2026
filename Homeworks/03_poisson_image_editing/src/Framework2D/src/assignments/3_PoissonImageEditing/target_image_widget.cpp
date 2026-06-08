#include "target_image_widget.h"

#include <cmath>

#include "Poisson/PoissonAlphaMatte.h"
#include "Poisson/PoissonBasic.h"
#include "Poisson/PoissonColorConstraint.h"
#include "Poisson/PoissonMixedGradient.h"

namespace USTC_CG
{
using uchar = unsigned char;

TargetImageWidget::TargetImageWidget(
    const std::string& label,
    const std::string& filename)
    : ImageWidget(label, filename)
{
    if (data_)
        back_up_ = std::make_shared<Image>(*data_);
}

void TargetImageWidget::draw()
{
    ImageWidget::draw();

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
    {
        mouse_release_event();
    }
}

void TargetImageWidget::set_source(std::shared_ptr<SourceImageWidget> source)
{
    source_image_ = source;
    invalidate_clone_cache();
}

void TargetImageWidget::set_realtime(bool flag)
{
    flag_realtime_updating = flag;
}

void TargetImageWidget::restore()
{
    *data_ = *back_up_;
    update();
}

void TargetImageWidget::set_paste()
{
    clone_type_ = kPaste;
    invalidate_clone_cache();
}

void TargetImageWidget::set_seamless()
{
    clone_type_ = kSeamless;
    invalidate_clone_cache();
}

void TargetImageWidget::set_mixed_gradient()
{
    clone_type_ = kMixedGradient;
    invalidate_clone_cache();
}

void TargetImageWidget::set_color_constraint()
{
    clone_type_ = kColorConstraint;
    invalidate_clone_cache();
}

void TargetImageWidget::set_alpha_matte()
{
    clone_type_ = kAlphaMatte;
    invalidate_clone_cache();
}

void TargetImageWidget::clone()
{
    if (data_ == nullptr || source_image_ == nullptr ||
        source_image_->get_region_mask() == nullptr)
        return;

    switch (clone_type_)
    {
        case USTC_CG::TargetImageWidget::kDefault: break;

        case USTC_CG::TargetImageWidget::kPaste:
        {
            restore();
            auto region = build_region_from_current_mouse();
            if (region == nullptr || region->empty())
                break;

            for (const auto& p : region->pixels())
            {
                if (p.target_x_ >= 0 && p.target_x_ < data_->width() &&
                    p.target_y_ >= 0 && p.target_y_ < data_->height())
                {
                    data_->set_pixel(
                        p.target_x_,
                        p.target_y_,
                        source_image_->get_data()->get_pixel(
                            p.source_x_, p.source_y_));
                }
            }
            break;
        }

        case USTC_CG::TargetImageWidget::kSeamless:
        case USTC_CG::TargetImageWidget::kMixedGradient:
        case USTC_CG::TargetImageWidget::kColorConstraint:
        case USTC_CG::TargetImageWidget::kAlphaMatte:
        {
            restore();

            if (!cached_region_ || !cached_solver_)
            {
                cached_region_ = build_region_from_current_mouse();
                if (cached_region_ == nullptr || cached_region_->empty())
                    break;

                if (clone_type_ == kSeamless)
                {
                    cached_solver_ = std::make_unique<PoissonBasic>(
                        cached_region_, source_image_->get_data(), back_up_);
                }
                else if (clone_type_ == kMixedGradient)
                {
                    cached_solver_ = std::make_unique<PoissonMixedGradient>(
                        cached_region_, source_image_->get_data(), back_up_);
                }
                else if (clone_type_ == kColorConstraint)
                {
                    cached_solver_ = std::make_unique<PoissonColorConstraint>(
                        cached_region_,
                        source_image_->get_data(),
                        back_up_,
                        source_image_->get_subject_mask());
                }
                else if (clone_type_ == kAlphaMatte)
                {
                    cached_solver_ = std::make_unique<PoissonAlphaMatte>(
                        cached_region_,
                        source_image_->get_data(),
                        back_up_,
                        source_image_->get_subject_mask());
                }

                cached_solver_->build();
            }
            else
            {
                cached_region_->update_target_anchor(
                    static_cast<int>(mouse_position_.x),
                    static_cast<int>(mouse_position_.y));
            }

            auto result = cached_solver_->solve();
            *data_ = *result;
            break;
        }

        default: break;
    }

    update();
}

void TargetImageWidget::mouse_click_event()
{
    edit_status_ = true;
    mouse_position_ = mouse_pos_in_canvas();
    clone();
}

void TargetImageWidget::mouse_move_event()
{
    if (edit_status_)
    {
        mouse_position_ = mouse_pos_in_canvas();
        if (flag_realtime_updating)
            clone();
    }
}

void TargetImageWidget::mouse_release_event()
{
    if (edit_status_)
    {
        edit_status_ = false;
    }
}

std::shared_ptr<PoissonRegion>
TargetImageWidget::build_region_from_current_mouse() const
{
    if (source_image_ == nullptr || source_image_->get_region_mask() == nullptr)
        return nullptr;

    const ImVec2 source_anchor = source_image_->get_position();

    auto region = std::make_shared<PoissonRegion>(
        source_image_->get_region_mask(),
        static_cast<int>(source_anchor.x),
        static_cast<int>(source_anchor.y),
        static_cast<int>(mouse_position_.x),
        static_cast<int>(mouse_position_.y));

    region->build();
    return region;
}

ImVec2 TargetImageWidget::mouse_pos_in_canvas() const
{
    ImGuiIO& io = ImGui::GetIO();
    return ImVec2(io.MousePos.x - position_.x, io.MousePos.y - position_.y);
}

void TargetImageWidget::invalidate_clone_cache()
{
    cached_region_.reset();
    cached_solver_.reset();
}
}  // namespace USTC_CG