#include "canvas_widget.h"

#include <cmath>
#include <iostream>

#include "imgui.h"
#include "shapes/line.h"
#include "shapes/rect.h"
#include "shapes/ellipse.h"
#include "shapes/polygon.h"
#include "shapes/pen.h"
#include "shapes/freehand.h"


namespace USTC_CG
{
void Canvas::draw()
{
    draw_background();
    // HW1_TODO: more interaction events
    if (is_hovered_)
    {
        if (shape_type_ == kPen &&
            ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))
            mouse_double_click_event();
        else if (ImGui::IsMouseClicked(ImGuiMouseButton_Right))
            mouse_right_click_event();
        else if (ImGui::IsMouseClicked(ImGuiMouseButton_Left))
            mouse_click_event();
        mouse_move_event();
        if(!ImGui::IsMouseDown(ImGuiMouseButton_Left))
            mouse_release_event();
    }
    draw_shapes();
}

void Canvas::set_attributes(const ImVec2& min, const ImVec2& size)
{
    canvas_min_ = min;
    canvas_size_ = size;
    canvas_minimal_size_ = size;
    canvas_max_ =
        ImVec2(canvas_min_.x + canvas_size_.x, canvas_min_.y + canvas_size_.y);
}

void Canvas::show_background(bool flag)
{
    show_background_ = flag;
}

void Canvas::set_default()
{
    draw_status_ = false;
    shape_type_ = kDefault;
}

void Canvas::set_line()
{
    draw_status_ = false;
    shape_type_ = kLine;
}

void Canvas::set_rect()
{
    draw_status_ = false;
    shape_type_ = kRect;
}

void Canvas::set_ellipse()
{
    draw_status_ = false;
    shape_type_ = kEllipse;
}

void Canvas::set_polygon()
{
    draw_status_ = false;
    shape_type_ = kPolygon;
}

void Canvas::set_pen()
{
    draw_status_ = false;
    shape_type_ = kPen;
}

void Canvas::set_freehand()
{
    draw_status_ = false;
    shape_type_ = kFreehand;
}

void Canvas::set_select()
{
    draw_status_ = false;
    shape_type_ = kSelect;
}

// HW1_TODO: more shape types, implements

void Canvas::set_polygon_sides(int sides)
{
    polygon_sides_ = (sides < 3 ? 3 : sides);
}

void Canvas::set_stroke_enabled(bool enabled)
{
    current_style_.stroke = enabled;
}

void Canvas::set_line_thickness(float thickness)
{
    current_style_.line_thickness = thickness;
}

void Canvas::set_line_color(const float color[4])
{
    for (int i = 0; i < 4; ++i)
        current_style_.line_color[i] =
            static_cast<unsigned char>(color[i] * 255.0f);
}

void Canvas::set_fill_enabled(bool enabled)
{
    current_style_.filled = enabled;
}

void Canvas::set_fill_color(const float color[4])
{
    for (int i = 0; i < 4; ++i)
        current_style_.fill_color[i] =
            static_cast<unsigned char>(color[i] * 255.0f);
}

void Canvas::clear_shape_list()
{
    shape_list_.clear();
}

void Canvas::draw_background()
{
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    if (show_background_)
    {
        // Draw background recrangle
        draw_list->AddRectFilled(canvas_min_, canvas_max_, background_color_);
        // Draw background border
        draw_list->AddRect(canvas_min_, canvas_max_, border_color_);
    }
    /// Invisible button over the canvas to capture mouse interactions.
    ImGui::SetCursorScreenPos(canvas_min_);
    ImGui::InvisibleButton(
        label_.c_str(), canvas_size_, ImGuiButtonFlags_MouseButtonLeft);
    // Record the current status of the invisible button
    is_hovered_ = ImGui::IsItemHovered();
    is_active_ = ImGui::IsItemActive();
}

void Canvas::draw_shapes()
{
    Shape::Config s = { .bias = { canvas_min_.x, canvas_min_.y } };
    ImDrawList* draw_list = ImGui::GetWindowDrawList();

    // ClipRect can hide the drawing content outside of the rectangular area
    draw_list->PushClipRect(canvas_min_, canvas_max_, true);

    //Select
    for (int i = 0; i < static_cast<int>(shape_list_.size()); ++i)
    {
        if (i == selected_shape_index_)
            shape_list_[i]->draw_selected(s);
        else
            shape_list_[i]->draw(s);
    }

    if (draw_status_ && current_shape_)
    {
        current_shape_->draw(s);
    }
    draw_list->PopClipRect();
}

void Canvas::mouse_click_event()
{
    // HW1_TODO: Drawing rule for more primitives

     if (shape_type_ == kSelect)
    {
         ImVec2 point = mouse_pos_in_canvas();
        selected_shape_index_ = -1;
        //Traverse existing shapes to check whether they can be selected.
        for (int i = static_cast<int>(shape_list_.size()) - 1; i >= 0; --i)
        {
            if (shape_list_[i]->hit_test(point.x, point.y))
            {
                selected_shape_index_ = i;
                break;
            }
        }
        return;
    }


    if (!draw_status_)
    {
        draw_status_ = true;
        start_point_ = end_point_ = mouse_pos_in_canvas();
        switch (shape_type_)
        {
            case USTC_CG::Canvas::kDefault:
            {
                break;
            }
            case USTC_CG::Canvas::kLine:
            {
                current_shape_ = std::make_shared<Line>(
                    start_point_.x, start_point_.y, end_point_.x, end_point_.y);
                current_shape_->set_style(current_style_);
                break;
            }
            case USTC_CG::Canvas::kRect:
            {
                current_shape_ = std::make_shared<Rect>(
                    start_point_.x, start_point_.y, end_point_.x, end_point_.y);
                current_shape_->set_style(current_style_);
                break;
            }
            case USTC_CG::Canvas::kEllipse:
            {
                current_shape_ = std::make_shared<Ellipse>(
                    start_point_.x, start_point_.y, end_point_.x, end_point_.y);
                current_shape_->set_style(current_style_);
                break;
            }
            case USTC_CG::Canvas::kPolygon:
            {
                current_shape_ = std::make_shared<Polygon>(
                    start_point_.x,
                    start_point_.y,
                    end_point_.x,
                    end_point_.y,
                    polygon_sides_);
                current_shape_->set_style(current_style_);
                break;
            }
            case USTC_CG::Canvas::kPen:
            {
                current_shape_ = std::make_shared<Pen>();
                current_shape_->add_control_point(start_point_.x, start_point_.y);
                current_shape_->set_style(current_style_);
                break;
            }
            case USTC_CG::Canvas::kFreehand:
            {
                current_shape_ = std::make_shared<Freehand>();
                current_shape_->add_control_point(
                    start_point_.x, start_point_.y);
                current_shape_->set_style(current_style_);
                break;
            }
            default: break;
        }
    }
    else
    {
        if (current_shape_)
        {
            if (shape_type_ == kPen)
            {
                draw_status_ = true;
                current_shape_->add_control_point(end_point_.x, end_point_.y);
            }
            else
            {
                draw_status_ = false;
                shape_list_.push_back(current_shape_);
                current_shape_.reset();
            }

        }
    }
}

void Canvas::mouse_double_click_event()
{
    if (shape_type_ == kPen)
    {
        draw_status_ = false;
        if (current_shape_)
        {
            shape_list_.push_back(current_shape_);
            current_shape_.reset();
        }
    }
}

void Canvas::mouse_move_event()
{
    // HW1_TODO: Drawing rule for more primitives
    if (draw_status_ && current_shape_)
    {
        end_point_ = mouse_pos_in_canvas();
        if (shape_type_ == kFreehand && ImGui::IsMouseDown(ImGuiMouseButton_Left))
        {
            current_shape_->add_control_point(end_point_.x, end_point_.y);
        }
        else
        {
            current_shape_->update(end_point_.x, end_point_.y);
        }
    }
}

void Canvas::mouse_right_click_event()
{
    if (shape_type_ == kSelect)
    {
        selected_shape_index_ = -1;
        return;
    }

    if (shape_type_ == kPen && current_shape_)
    {
        draw_status_ = false;
        auto pen = std::dynamic_pointer_cast<Pen>(current_shape_);
        pen->set_closed(true);
        shape_list_.push_back(current_shape_);
        current_shape_.reset();
    }
}

void Canvas::mouse_release_event()
{
    if (shape_type_ == kFreehand && current_shape_)
    {
        draw_status_ = false;
        shape_list_.push_back(current_shape_);
        current_shape_.reset();
    }
}

ImVec2 Canvas::mouse_pos_in_canvas() const
{
    ImGuiIO& io = ImGui::GetIO();
    const ImVec2 mouse_pos_in_canvas(
        io.MousePos.x - canvas_min_.x, io.MousePos.y - canvas_min_.y);
    return mouse_pos_in_canvas;
}
}  // namespace USTC_CG