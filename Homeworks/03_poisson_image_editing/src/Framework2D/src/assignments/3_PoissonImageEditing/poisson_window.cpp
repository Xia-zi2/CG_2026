#include "poisson_window.h"

#include <ImGuiFileDialog.h>

#include <iostream>

namespace USTC_CG
{
PoissonWindow::PoissonWindow(const std::string& window_name)
    : Window(window_name)
{
}

PoissonWindow::~PoissonWindow()
{
}

void PoissonWindow::draw()
{
    draw_toolbar();

    if (flag_open_target_file_dialog_)
        draw_open_target_image_file_dialog();
    if (flag_open_source_file_dialog_ && p_target_)
        draw_open_source_image_file_dialog();
    if (flag_save_file_dialog_ && p_target_)
        draw_save_image_file_dialog();

    if (p_target_)
        draw_target();
    if (p_source_)
        draw_source();
}

void PoissonWindow::draw_toolbar()
{
    if (!ImGui::BeginMainMenuBar())
        return;

    if (ImGui::BeginMenu("File"))
    {
        if (ImGui::MenuItem("Open Target."))
        {
            flag_open_target_file_dialog_ = true;
        }
        add_tooltips("Open the target image file.");

        if (ImGui::MenuItem(
                "Open Source..", nullptr, false, p_target_ != nullptr))
        {
            flag_open_source_file_dialog_ = true;
        }
        add_tooltips(
            "Open the source image file after target image is loaded.");

        if (ImGui::MenuItem("Save As..", nullptr, false, p_target_ != nullptr))
        {
            flag_save_file_dialog_ = true;
        }
        add_tooltips("Save current target image.");

        ImGui::EndMenu();
    }

    if (p_source_)
    {
        if (ImGui::BeginMenu("Source Edit"))
        {
            static bool selectable = false;
            ImGui::Checkbox("Enable Selecting", &selectable);
            p_source_->enable_selecting(selectable);
            add_tooltips("Enable drawing mask on the source image.");

            ImGui::Separator();
            ImGui::Text("Mask Type");

            bool edit_roi =
                (p_source_->get_mask_edit_mode() ==
                 SourceImageWidget::kEditROI);
            if (ImGui::RadioButton("ROI Mask", edit_roi))
            {
                p_source_->set_mask_edit_mode(SourceImageWidget::kEditROI);
            }
            add_tooltips("Edit the paste region mask.");

            bool edit_subject =
                (p_source_->get_mask_edit_mode() ==
                 SourceImageWidget::kEditSubject);
            if (ImGui::RadioButton("Subject Mask", edit_subject))
            {
                p_source_->set_mask_edit_mode(SourceImageWidget::kEditSubject);
            }
            add_tooltips(
                "Edit the subject mask for alpha / color-preserving methods.");

            ImGui::Separator();
            ImGui::Text("Shape");

            bool is_rect =
                (p_source_->get_region_type() == SourceImageWidget::kRect);
            if (ImGui::RadioButton("Rect", is_rect))
            {
                p_source_->set_region_type(SourceImageWidget::kRect);
            }
            add_tooltips("Use rectangle selection.");

            bool is_freehand =
                (p_source_->get_region_type() == SourceImageWidget::kFreehand);
            if (ImGui::RadioButton("Freehand", is_freehand))
            {
                p_source_->set_region_type(SourceImageWidget::kFreehand);
            }
            add_tooltips("Use freehand polygon selection.");

            ImGui::Separator();

            if (ImGui::MenuItem("Clear ROI"))
            {
                p_source_->clear_shape();
            }
            add_tooltips("Clear ROI mask.");

            if (ImGui::MenuItem("Clear Subject"))
            {
                p_source_->clear_subject_mask();
            }
            add_tooltips("Clear subject mask.");

            ImGui::EndMenu();
        }
    }

    if (p_target_ && p_source_)
    {
        if (ImGui::BeginMenu("Clone Mode"))
        {
            if (ImGui::MenuItem("Paste"))
            {
                p_target_->set_paste();
            }
            add_tooltips("Direct copy-paste.");

            if (ImGui::MenuItem("Seamless Poisson"))
            {
                p_target_->set_seamless();
            }
            add_tooltips("Standard Poisson seamless cloning.");

            ImGui::Separator();

            if (ImGui::MenuItem("Mixed Gradient"))
            {
                p_target_->set_mixed_gradient();
            }
            add_tooltips(
                "Standard mixed-gradient Poisson cloning "
                "(choose the stronger source/target edge on each neighbor "
                "pair).");

            if (ImGui::MenuItem("Color Constraint"))
            {
                p_target_->set_color_constraint();
            }
            add_tooltips(
                "Poisson cloning with source-color constraint on the "
                "user-painted subject mask.");

            ImGui::Separator();

            if (ImGui::MenuItem("Alpha Matte"))
            {
                p_target_->set_alpha_matte();
            }
            add_tooltips(
                "Alpha-matte modulation driven by the user-painted subject "
                "mask.");

            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Options"))
        {
            static bool realtime = false;
            ImGui::Checkbox("Realtime Clone", &realtime);
            p_target_->set_realtime(realtime);
            add_tooltips(
                "Update cloning result while dragging on target image.");

            if (ImGui::MenuItem("Restore"))
            {
                if (p_target_)
                    p_target_->restore();
                if (p_source_)
                {
                    p_source_->clear_shape();
                    p_source_->clear_subject_mask();
                }
            }
            add_tooltips("Restore target image to backup.");

            ImGui::EndMenu();
        }
    }

    ImGui::EndMainMenuBar();
}

void PoissonWindow::draw_target()
{
    const auto& image_size = p_target_->get_image_size();
    ImGui::SetNextWindowSize(ImVec2(image_size.x + 60, image_size.y + 60));
    if (ImGui::Begin("Target Image", &flag_show_target_view_))
    {
        const auto& min = ImGui::GetCursorScreenPos();
        const auto& size = ImGui::GetContentRegionAvail();
        ImVec2 pos = ImVec2(
            min.x + size.x / 2 - image_size.x / 2,
            min.y + size.y / 2 - image_size.y / 2);
        p_target_->set_position(pos);
        p_target_->draw();
    }
    ImGui::End();
}

void PoissonWindow::draw_source()
{
    const auto& image_size = p_source_->get_image_size();
    ImGui::SetNextWindowSize(ImVec2(image_size.x + 60, image_size.y + 60));
    if (ImGui::Begin("Source Image", &flag_show_source_view_))
    {
        const auto& min = ImGui::GetCursorScreenPos();
        const auto& size = ImGui::GetContentRegionAvail();
        ImVec2 pos = ImVec2(
            min.x + size.x / 2 - image_size.x / 2,
            min.y + size.y / 2 - image_size.y / 2);
        p_source_->set_position(pos);
        p_source_->draw();
    }
    ImGui::End();
}

void PoissonWindow::draw_open_target_image_file_dialog()
{
    IGFD::FileDialogConfig config;
    config.path = DATA_PATH;
    config.flags = ImGuiFileDialogFlags_Modal;
    ImGuiFileDialog::Instance()->OpenDialog(
        "ChooseTargetOpenFileDlg", "Choose Image File", ".jpg,png", config);

    ImVec2 main_size = ImGui::GetMainViewport()->WorkSize;
    ImVec2 dlg_size(main_size.x / 2, main_size.y / 2);

    if (ImGuiFileDialog::Instance()->Display(
            "ChooseTargetOpenFileDlg", ImGuiWindowFlags_NoCollapse, dlg_size))
    {
        if (ImGuiFileDialog::Instance()->IsOk())
        {
            std::string filePathName =
                ImGuiFileDialog::Instance()->GetFilePathName();
            std::string label = filePathName;

            p_target_ =
                std::make_shared<TargetImageWidget>(label, filePathName);

            if (p_source_)
            {
                p_target_->set_source(p_source_);
                p_source_->set_target(p_target_);
            }
        }

        ImGuiFileDialog::Instance()->Close();
        flag_open_target_file_dialog_ = false;
    }
}

void PoissonWindow::draw_open_source_image_file_dialog()
{
    IGFD::FileDialogConfig config;
    config.path = DATA_PATH;
    config.flags = ImGuiFileDialogFlags_Modal;
    ImGuiFileDialog::Instance()->OpenDialog(
        "ChooseSourceOpenFileDlg", "Choose Image File", ".jpg,png", config);

    ImVec2 main_size = ImGui::GetMainViewport()->WorkSize;
    ImVec2 dlg_size(main_size.x / 2, main_size.y / 2);

    if (ImGuiFileDialog::Instance()->Display(
            "ChooseSourceOpenFileDlg", ImGuiWindowFlags_NoCollapse, dlg_size))
    {
        if (ImGuiFileDialog::Instance()->IsOk())
        {
            std::string filePathName =
                ImGuiFileDialog::Instance()->GetFilePathName();
            std::string label = filePathName;

            p_source_ =
                std::make_shared<SourceImageWidget>(label, filePathName);

            if (p_source_)
            {
                p_target_->set_source(p_source_);
                p_source_->set_target(p_target_);
            }
        }

        ImGuiFileDialog::Instance()->Close();
        flag_open_source_file_dialog_ = false;
    }
}

void PoissonWindow::draw_save_image_file_dialog()
{
    IGFD::FileDialogConfig config;
    config.path = DATA_PATH;
    config.flags = ImGuiFileDialogFlags_Modal;
    ImGuiFileDialog::Instance()->OpenDialog(
        "ChooseImageSaveFileDlg", "Save Image As.", ".jpg", config);

    ImVec2 main_size = ImGui::GetMainViewport()->WorkSize;
    ImVec2 dlg_size(main_size.x / 2, main_size.y / 2);

    if (ImGuiFileDialog::Instance()->Display(
            "ChooseImageSaveFileDlg", ImGuiWindowFlags_NoCollapse, dlg_size))
    {
        if (ImGuiFileDialog::Instance()->IsOk())
        {
            std::string filePathName =
                ImGuiFileDialog::Instance()->GetFilePathName();
            if (p_target_)
                p_target_->save_to_disk(filePathName);
        }

        ImGuiFileDialog::Instance()->Close();
        flag_save_file_dialog_ = false;
    }
}

void PoissonWindow::add_tooltips(std::string desc)
{
    if (ImGui::BeginItemTooltip())
    {
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
        ImGui::TextUnformatted(desc.c_str());
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}

}  // namespace USTC_CG