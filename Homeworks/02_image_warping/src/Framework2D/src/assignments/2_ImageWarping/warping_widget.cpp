#include "warping_widget.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>

#include "warper/IDW_warper.h"
#include "warper/RBF_warper.h"
#include "warper/fisheye.h"
#include "warper/hole_filler.h"

namespace USTC_CG
{
namespace
{
    float edge_function(const ImVec2& a, const ImVec2& b, const ImVec2& p)
    {
        return (p.x - a.x) * (b.y - a.y) - (p.y - a.y) * (b.x - a.x);
    }

    void rasterize_triangle(
        const ImVec2& v0,
        const ImVec2& v1,
        const ImVec2& v2,
        WarpingWidget::Mask& mask,
        int width,
        int height)
    {
        const float area = edge_function(v0, v1, v2);
        if (std::abs(area) < 1e-6f)
        {
            return;
        }

        const float min_xf = std::floor(std::min({ v0.x, v1.x, v2.x }));
        const float max_xf = std::ceil(std::max({ v0.x, v1.x, v2.x }));
        const float min_yf = std::floor(std::min({ v0.y, v1.y, v2.y }));
        const float max_yf = std::ceil(std::max({ v0.y, v1.y, v2.y }));

        const int min_x = std::max(0, static_cast<int>(min_xf));
        const int max_x = std::min(width - 1, static_cast<int>(max_xf));
        const int min_y = std::max(0, static_cast<int>(min_yf));
        const int max_y = std::min(height - 1, static_cast<int>(max_yf));

        for (int y = min_y; y <= max_y; ++y)
        {
            for (int x = min_x; x <= max_x; ++x)
            {
                const ImVec2 p(
                    static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f);
                const float w0 = edge_function(v1, v2, p);
                const float w1 = edge_function(v2, v0, p);
                const float w2 = edge_function(v0, v1, p);

                const bool inside_ccw =
                    (w0 >= 0.0f && w1 >= 0.0f && w2 >= 0.0f);
                const bool inside_cw = (w0 <= 0.0f && w1 <= 0.0f && w2 <= 0.0f);
                if (inside_ccw || inside_cw)
                {
                    mask[y][x] = 1;
                }
            }
        }
    }
}  // namespace

using uchar = unsigned char;

WarpingWidget::WarpingWidget(
    const std::string& label,
    const std::string& filename)
    : ImageWidget(label, filename)
{
    if (data_)
        back_up_ = std::make_shared<Image>(*data_);
}

void WarpingWidget::draw()
{
    ImageWidget::draw();
    if (flag_enable_selecting_points_)
        select_points();
}

void WarpingWidget::invert()
{
    for (int i = 0; i < data_->width(); ++i)
    {
        for (int j = 0; j < data_->height(); ++j)
        {
            const auto color = data_->get_pixel(i, j);
            data_->set_pixel(
                i,
                j,
                { static_cast<uchar>(255 - color[0]),
                  static_cast<uchar>(255 - color[1]),
                  static_cast<uchar>(255 - color[2]) });
        }
    }
    update();
}
void WarpingWidget::mirror(bool is_horizontal, bool is_vertical)
{
    Image image_tmp(*data_);
    int width = data_->width();
    int height = data_->height();

    if (is_horizontal)
    {
        if (is_vertical)
        {
            for (int i = 0; i < width; ++i)
            {
                for (int j = 0; j < height; ++j)
                {
                    data_->set_pixel(
                        i,
                        j,
                        image_tmp.get_pixel(width - 1 - i, height - 1 - j));
                }
            }
        }
        else
        {
            for (int i = 0; i < width; ++i)
            {
                for (int j = 0; j < height; ++j)
                {
                    data_->set_pixel(
                        i, j, image_tmp.get_pixel(width - 1 - i, j));
                }
            }
        }
    }
    else if (is_vertical)
    {
        for (int i = 0; i < width; ++i)
        {
            for (int j = 0; j < height; ++j)
            {
                data_->set_pixel(i, j, image_tmp.get_pixel(i, height - 1 - j));
            }
        }
    }

    update();
}
void WarpingWidget::gray_scale()
{
    for (int i = 0; i < data_->width(); ++i)
    {
        for (int j = 0; j < data_->height(); ++j)
        {
            const auto color = data_->get_pixel(i, j);
            uchar gray_value = (color[0] + color[1] + color[2]) / 3;
            data_->set_pixel(i, j, { gray_value, gray_value, gray_value });
        }
    }
    update();
}

void WarpingWidget::init_masks(int width, int height)
{
    written_mask_.assign(height, std::vector<unsigned char>(width, 0));
    coverage_mask_.assign(height, std::vector<unsigned char>(width, 0));
}

void WarpingWidget::rasterize_coverage_mask()
{
    if (!current_warper_ || !data_)
    {
        return;
    }

    const int width = data_->width();
    const int height = data_->height();
    if (width < 2 || height < 2)
    {
        return;
    }

    for (int y = 0; y < height - 1; ++y)
    {
        for (int x = 0; x < width - 1; ++x)
        {
            const ImVec2 p00 = current_warper_->warp(
                ImVec2(static_cast<float>(x), static_cast<float>(y)));
            const ImVec2 p10 = current_warper_->warp(
                ImVec2(static_cast<float>(x + 1), static_cast<float>(y)));
            const ImVec2 p01 = current_warper_->warp(
                ImVec2(static_cast<float>(x), static_cast<float>(y + 1)));
            const ImVec2 p11 = current_warper_->warp(
                ImVec2(static_cast<float>(x + 1), static_cast<float>(y + 1)));

            rasterize_triangle(p00, p10, p11, coverage_mask_, width, height);
            rasterize_triangle(p00, p11, p01, coverage_mask_, width, height);
        }
    }
}

void WarpingWidget::warping()
{
    auto t0 = std::chrono::high_resolution_clock::now();
    Image warped_image(*data_);
    for (int y = 0; y < data_->height(); ++y)
    {
        for (int x = 0; x < data_->width(); ++x)
        {
            warped_image.set_pixel(x, y, { 0, 0, 0 });
        }
    }

    switch (warping_type_)
    {
        case kDefault:
        {
            current_warper_ = nullptr;
            break;
        }
        case kFisheye:
        {
            current_warper_ =
                std::make_shared<Fisheye>(data_->width(), data_->height());
            break;
        }
        case kIDW:
        {
            current_warper_ = std::make_shared<IDWWarper>(
                start_points_,
                end_points_,
                data_->width(),
                data_->height(),
                IDWWarper::Type::IDW);
            break;
        }
        case kIDW_A:
        {
            current_warper_ = std::make_shared<IDWWarper>(
                start_points_,
                end_points_,
                data_->width(),
                data_->height(),
                IDWWarper::Type::IDW_A);
            break;
        }
        case kIDW_R:
        {
            current_warper_ = std::make_shared<IDWWarper>(
                start_points_,
                end_points_,
                data_->width(),
                data_->height(),
                IDWWarper::Type::IDW_R);
            break;
        }
        case kRBF_AffineFirst:
        {
            current_warper_ = std::make_shared<RBFWarper>(
                start_points_,
                end_points_,
                data_->width(),
                data_->height(),
                RBFWarper::Type::AffineFirst);
            break;
        }
        case kRBF_Full:
        {
            current_warper_ = std::make_shared<RBFWarper>(
                start_points_,
                end_points_,
                data_->width(),
                data_->height(),
                RBFWarper::Type::Full);
            break;
        }
        default: current_warper_ = nullptr; break;
    }
    if (!current_warper_)
        return;

    init_masks(data_->width(), data_->height());
    rasterize_coverage_mask();

    for (int y = 0; y < data_->height(); ++y)
    {
        for (int x = 0; x < data_->width(); ++x)
        {
            ImVec2 new_pos = current_warper_->warp(
                ImVec2(static_cast<float>(x), static_cast<float>(y)));

            int new_x = static_cast<int>(new_pos.x);
            int new_y = static_cast<int>(new_pos.y);

            if (new_x >= 0 && new_x < data_->width() && new_y >= 0 &&
                new_y < data_->height())
            {
                std::vector<unsigned char> pixel = data_->get_pixel(x, y);
                warped_image.set_pixel(new_x, new_y, pixel);
                written_mask_[new_y][new_x] = 1;
            }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "[warping] elapsed = " << elapsed_ms << " ms" << std::endl;
    *data_ = warped_image;
    warped_backup_ = std::make_shared<Image>(warped_image);
    holes_filled_ = false;
    update();
}
void WarpingWidget::restore()
{
    if (back_up_)
    {
        *data_ = *back_up_;
    }

    current_warper_ = nullptr;
    warped_backup_.reset();
    written_mask_.clear();
    coverage_mask_.clear();
    holes_filled_ = false;
    init_selections();
    enable_selecting(false);
    update();
}
void WarpingWidget::set_default()
{
    warping_type_ = kDefault;
}

void WarpingWidget::set_fisheye()
{
    warping_type_ = kFisheye;
}

void WarpingWidget::set_IDW()
{
    warping_type_ = kIDW;
}

void WarpingWidget::set_IDW_A()
{
    warping_type_ = kIDW_A;
}

void WarpingWidget::set_IDW_R()
{
    warping_type_ = kIDW_R;
}

void WarpingWidget::set_RBF_affine_first()
{
    warping_type_ = kRBF_AffineFirst;
}

void WarpingWidget::set_RBF_full()
{
    warping_type_ = kRBF_Full;
}

void WarpingWidget::set_hole_ann()
{
    hole_filling_type_ = kHoleANN;
}
void WarpingWidget::enable_selecting(bool flag)
{
    flag_enable_selecting_points_ = flag;
}
void WarpingWidget::select_points()
{
    ImGui::SetCursorScreenPos(position_);
    ImGui::InvisibleButton(
        label_.c_str(),
        ImVec2(
            static_cast<float>(image_width_),
            static_cast<float>(image_height_)),
        ImGuiButtonFlags_MouseButtonLeft);
    bool is_hovered_ = ImGui::IsItemHovered();
    ImGuiIO& io = ImGui::GetIO();
    if (is_hovered_ && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
    {
        draw_status_ = true;
        start_ = end_ =
            ImVec2(io.MousePos.x - position_.x, io.MousePos.y - position_.y);
    }
    if (draw_status_)
    {
        end_ = ImVec2(io.MousePos.x - position_.x, io.MousePos.y - position_.y);
        if (!ImGui::IsMouseDown(ImGuiMouseButton_Left))
        {
            start_points_.push_back(start_);
            end_points_.push_back(end_);
            draw_status_ = false;
        }
    }
    auto draw_list = ImGui::GetWindowDrawList();
    for (size_t i = 0; i < start_points_.size(); ++i)
    {
        ImVec2 s(
            start_points_[i].x + position_.x, start_points_[i].y + position_.y);
        ImVec2 e(
            end_points_[i].x + position_.x, end_points_[i].y + position_.y);
        draw_list->AddLine(s, e, IM_COL32(255, 0, 0, 255), 2.0f);
        draw_list->AddCircleFilled(s, 4.0f, IM_COL32(0, 0, 255, 255));
        draw_list->AddCircleFilled(e, 4.0f, IM_COL32(0, 255, 0, 255));
    }
    if (draw_status_)
    {
        ImVec2 s(start_.x + position_.x, start_.y + position_.y);
        ImVec2 e(end_.x + position_.x, end_.y + position_.y);
        draw_list->AddLine(s, e, IM_COL32(255, 0, 0, 255), 2.0f);
        draw_list->AddCircleFilled(s, 4.0f, IM_COL32(0, 0, 255, 255));
    }
}
void WarpingWidget::init_selections()
{
    start_points_.clear();
    end_points_.clear();
}

void WarpingWidget::fill_holes()
{
    if (!data_ || !warped_backup_ || !current_warper_)
    {
        return;
    }

    if (holes_filled_)
    {
        return;
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    HoleFiller filler(HoleFiller::Type::ANN);
    Image filled_image(*warped_backup_);

    filler.fill(*warped_backup_, filled_image, written_mask_, coverage_mask_);

    *data_ = std::move(filled_image);
    holes_filled_ = true;

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "[hole filling - ANN] elapsed = " << elapsed_ms << " ms"
              << std::endl;

    update();
}
}  // namespace USTC_CG
