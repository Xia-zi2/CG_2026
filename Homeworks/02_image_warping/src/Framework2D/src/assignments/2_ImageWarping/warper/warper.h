#pragma once

#include <imgui.h>

#include <Eigen/Dense>
#include <vector>

namespace USTC_CG
{

class Warper
{
   public:
    using Vec2 = Eigen::Vector2f;

   public:
    Warper(
        const std::vector<ImVec2>& start_points,
        const std::vector<ImVec2>& end_points,
        int canvas_width,
        int canvas_height)
        : start_points_(start_points),
          end_points_(end_points),
          canvas_width_(canvas_width),
          canvas_height_(canvas_height)
    {
    }

    virtual ~Warper() = default;

    ImVec2 warp(const ImVec2& p) const
    {
        if (requires_control_points() && empty())
        {
            return p;
        }

        if (requires_control_points())
        {
            ImVec2 exact_target;
            if (try_map_exact_control_point(p, exact_target))
            {
                return exact_target;
            }
        }

        return warp_impl(p);
    }

    const std::vector<ImVec2>& start_points() const
    {
        return start_points_;
    }

    const std::vector<ImVec2>& end_points() const
    {
        return end_points_;
    }

    int canvas_width() const
    {
        return canvas_width_;
    }

    int canvas_height() const
    {
        return canvas_height_;
    }

    virtual bool requires_control_points() const
    {
        return true;
    }

    // Check whether control-point data is empty or invalid
    bool empty() const
    {
        return start_points_.empty() || end_points_.empty() ||
               start_points_.size() != end_points_.size();
    }

    size_t control_point_count() const
    {
        return start_points_.size();
    }

   protected:
    // Actual warping implementation provided by derived classes
    virtual ImVec2 warp_impl(const ImVec2& p) const = 0;

    static Vec2 to_eigen(const ImVec2& p)
    {
        return Vec2(p.x, p.y);
    }

    static ImVec2 to_imgui(const Vec2& v)
    {
        return ImVec2(v.x(), v.y());
    }

    // Check whether p coincides with a source control point, and if so return
    // the exact mapped target point
    bool try_map_exact_control_point(const ImVec2& p, ImVec2& out) const
    {
        const Vec2 x = to_eigen(p);

        for (size_t i = 0; i < control_point_count(); ++i)
        {
            const Vec2 pi = to_eigen(start_points_[i]);
            if ((x - pi).norm() < eps_)
            {
                out = end_points_[i];
                return true;
            }
        }

        return false;
    }

   protected:
    std::vector<ImVec2> start_points_;
    std::vector<ImVec2> end_points_;
    int canvas_width_ = 0;
    int canvas_height_ = 0;
    float eps_ = 1e-5f;
};

}  // namespace USTC_CG