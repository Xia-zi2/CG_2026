#include "minidraw_window.h"

#include <iostream>

namespace USTC_CG
{
MiniDraw::MiniDraw(const std::string& window_name) : Window(window_name)
{
    p_canvas_ = std::make_shared<Canvas>("Widget.Canvas");
}

MiniDraw::~MiniDraw()
{
}

void MiniDraw::draw()
{
    draw_canvas();
}

void MiniDraw::draw_canvas()
{
    // Set a full screen canvas view
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(viewport->WorkSize);
    if (ImGui::Begin(
            "Canvas",
            &flag_show_canvas_view_,
            ImGuiWindowFlags_NoDecoration|ImGuiWindowFlags_NoBackground))
    {
        // Buttons for shape types
        if (ImGui::Button("Line"))
        {
            std::cout << "Set shape to Line" << std::endl;
            p_canvas_->set_line();
        }
        ImGui::SameLine();
        if (ImGui::Button("Rect"))
        {
            std::cout << "Set shape to Rect" << std::endl;
            p_canvas_->set_rect();
        }
        ImGui::SameLine();
        if (ImGui::Button("Ellipse"))
        {
            std::cout << "Set shape to Ellipse" << std::endl;
            p_canvas_->set_ellipse();
        }
        ImGui::SameLine();
        if (ImGui::Button("Polygon"))
        {
            std::cout << "Set shape to Polygon" << std::endl;
            ImGui::OpenPopup("Polygon Settings");
        }

        if (ImGui::BeginPopupModal(
                "Polygon Settings", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
        {
            ImGui::Text("Input number of sides:");
            ImGui::PushItemWidth(120.0f);
            ImGui::InputInt("Sides", &polygon_sides_ui_);
            ImGui::PopItemWidth();

            if (polygon_sides_ui_ < 3)
                polygon_sides_ui_ = 3;

            if (ImGui::Button("OK"))
            {
                p_canvas_->set_polygon_sides(polygon_sides_ui_);
                p_canvas_->set_polygon();
                ImGui::CloseCurrentPopup();
            }

            ImGui::SameLine();

            if (ImGui::Button("Cancel"))
            {
                ImGui::CloseCurrentPopup();
            }

            ImGui::EndPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Pen"))
        {
            std::cout << "Pen" << std::endl;
            p_canvas_->set_pen();
        }
        ImGui::SameLine();
        if (ImGui::Button("Freehand"))
        {
            std::cout << "Freehand" << std::endl;
            p_canvas_->set_freehand();
        }
        ImGui::SameLine();
        if (ImGui::Button("Select"))
        {
            std::cout << "Set tool to Select" << std::endl;
            p_canvas_->set_select();
        }

        ImGui::Separator();

        ImGui::Text("Stroke");
        ImGui::SameLine();
        ImGui::Checkbox("##StrokeEnabled", &stroke_enabled_ui_);

        ImGui::SameLine();
        ImGui::Text("Stroke Color:");
        ImGui::SameLine();
        if (stroke_enabled_ui_)
        {
            ImGui::ColorEdit4(
                "##StrokeColor",
                stroke_color_,
                ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel);
        }
        else
        {
            ImGui::Text("\\");
        }

        ImGui::SameLine();
        ImGui::Text("Width");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(80.0f);
        ImGui::SliderFloat("##LineWidth", &line_thickness_ui_, 1.0f, 20.0f);

        ImGui::SameLine();
        ImGui::Text("Fill");
        ImGui::SameLine();
        ImGui::Checkbox("##FillEnabled", &fill_enabled_ui_);

        ImGui::SameLine();
        ImGui::Text("Fill Color:");
        ImGui::SameLine();
        if (fill_enabled_ui_)
        {
            ImGui::ColorEdit4(
                "##FillColor",
                fill_color_,
                ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel);
        }
        else
        {
            ImGui::Text("\\");
        }

        ImGui::Separator();
        if (ImGui::Button("Clear"))
        {
            p_canvas_->clear_shape_list();
        }
      
        // Canvas component
        ImGui::Text("Press left mouse to add shapes.");
        // Set the canvas to fill the rest of the window
        const auto& canvas_min = ImGui::GetCursorScreenPos();
        const auto& canvas_size = ImGui::GetContentRegionAvail();
        p_canvas_->set_attributes(canvas_min, canvas_size);
        p_canvas_->set_stroke_enabled(stroke_enabled_ui_);
        p_canvas_->set_line_color(stroke_color_);
        p_canvas_->set_line_thickness(line_thickness_ui_);
        p_canvas_->set_fill_enabled(fill_enabled_ui_);
        p_canvas_->set_fill_color(fill_color_);
        p_canvas_->draw();
    }
    ImGui::End();
}
}  // namespace USTC_CG