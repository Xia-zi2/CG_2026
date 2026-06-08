#pragma once

#include <memory>

#include "common/window.h"
#include "canvas_widget.h"

namespace USTC_CG
{
class MiniDraw : public Window
{
   public:
    explicit MiniDraw(const std::string& window_name);
    ~MiniDraw();

    void draw();

   private:
    void draw_canvas();
    int polygon_sides_ui_ = 3;

    float stroke_color_[4] = { 1.0f, 0.0f, 0.0f, 1.0f };
    float fill_color_[4] = { 0.0f, 1.0f, 0.0f, 0.4f };
    float line_thickness_ui_ = 2.0f;

    bool fill_enabled_ui_ = false;
    bool stroke_enabled_ui_ = true;

    std::shared_ptr<Canvas> p_canvas_ = nullptr;

    bool flag_show_canvas_view_ = true;
};
}  // namespace USTC_CG