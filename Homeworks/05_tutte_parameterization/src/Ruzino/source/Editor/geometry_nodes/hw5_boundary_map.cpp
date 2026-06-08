#include <algorithm>
#include <cmath>
#include <vector>

#include "GCore/Components.h"
#include "GCore/Components/MeshComponent.h"
#include "GCore/GOP.h"
#include "GCore/util_openmesh_bind.h"
#include "geom_node_base.h"
#include "nodes/core/def/node_def.hpp"

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kEps = 1e-12;

enum BoundaryMode { BOUNDARY_CIRCLE = 0, BOUNDARY_SQUARE = 1 };

template<typename MeshT>
double edge_length(MeshT* mesh, int a, int b)
{
    auto pa = mesh->point(mesh->vertex_handle(a));
    auto pb = mesh->point(mesh->vertex_handle(b));
    double dx = static_cast<double>(pa[0] - pb[0]);
    double dy = static_cast<double>(pa[1] - pb[1]);
    double dz = static_cast<double>(pa[2] - pb[2]);
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

template<typename MeshT>
std::vector<std::vector<int>> extract_boundary_loops(MeshT* mesh)
{
    const int nV = static_cast<int>(mesh->n_vertices());
    std::vector<int> next_on_boundary(nV, -1);

    for (auto he_it = mesh->halfedges_begin(); he_it != mesh->halfedges_end();
         ++he_it) {
        auto heh = *he_it;
        if (mesh->is_boundary(heh)) {
            int u = mesh->from_vertex_handle(heh).idx();
            int v = mesh->to_vertex_handle(heh).idx();
            next_on_boundary[u] = v;
        }
    }

    std::vector<char> visited(nV, 0);
    std::vector<std::vector<int>> loops;

    for (int s = 0; s < nV; ++s) {
        if (next_on_boundary[s] == -1 || visited[s]) {
            continue;
        }

        std::vector<int> loop;
        int cur = s;

        while (cur != -1 && !visited[cur]) {
            visited[cur] = 1;
            loop.push_back(cur);
            cur = next_on_boundary[cur];
        }

        if (cur == s && loop.size() >= 3) {
            loops.push_back(loop);
        }
    }

    return loops;
}

template<typename MeshT>
double loop_perimeter(MeshT* mesh, const std::vector<int>& loop)
{
    const int m = static_cast<int>(loop.size());
    if (m < 2) {
        return 0.0;
    }

    double L = 0.0;
    for (int i = 0; i < m; ++i) {
        int a = loop[i];
        int b = loop[(i + 1) % m];
        L += edge_length(mesh, a, b);
    }
    return L;
}

inline void rotate_loop_to_smallest_vertex(std::vector<int>& loop)
{
    if (loop.empty()) {
        return;
    }

    int min_pos = 0;
    for (int i = 1; i < static_cast<int>(loop.size()); ++i) {
        if (loop[i] < loop[min_pos]) {
            min_pos = i;
        }
    }

    std::rotate(loop.begin(), loop.begin() + min_pos, loop.end());
}

inline void reverse_loop_keep_start(std::vector<int>& loop)
{
    if (loop.size() <= 2) {
        return;
    }

    std::reverse(loop.begin() + 1, loop.end());
}

template<typename MeshT>
std::vector<int> extract_outer_boundary_loop(MeshT* mesh, bool flip_orientation)
{
    auto loops = extract_boundary_loops(mesh);
    if (loops.empty()) {
        return {};
    }

    int best_id = -1;
    double best_len = -1.0;
    for (int i = 0; i < static_cast<int>(loops.size()); ++i) {
        double len = loop_perimeter(mesh, loops[i]);
        if (len > best_len) {
            best_len = len;
            best_id = i;
        }
    }

    if (best_id < 0) {
        return {};
    }

    auto loop = loops[best_id];
    rotate_loop_to_smallest_vertex(loop);

    if (flip_orientation) {
        reverse_loop_keep_start(loop);
    }

    return loop;
}

template<typename MeshT>
std::vector<double> boundary_prefix_lengths(
    MeshT* mesh,
    const std::vector<int>& loop,
    double& total_length)
{
    const int m = static_cast<int>(loop.size());
    std::vector<double> prefix(m, 0.0);
    total_length = 0.0;

    if (m == 0) {
        return prefix;
    }

    for (int i = 1; i < m; ++i) {
        total_length += edge_length(mesh, loop[i - 1], loop[i]);
        prefix[i] = total_length;
    }
    total_length += edge_length(mesh, loop[m - 1], loop[0]);

    return prefix;
}

inline std::pair<double, double> map_to_square_boundary(double t)
{
    double s = 4.0 * t;
    if (s < 1.0) {
        return { s, 0.0 };
    }
    if (s < 2.0) {
        return { 1.0, s - 1.0 };
    }
    if (s < 3.0) {
        return { 3.0 - s, 1.0 };
    }
    return { 0.0, 4.0 - s };
}

template<typename MeshT>
void flatten_all_vertices_to_xy_plane(MeshT* mesh)
{
    for (auto v_it = mesh->vertices_begin(); v_it != mesh->vertices_end();
         ++v_it) {
        auto vh = *v_it;
        auto& p = mesh->point(vh);
        p[2] = 0.0f;
    }
}

template<typename MeshT>
void map_boundary_to_circle(MeshT* mesh, const std::vector<int>& loop)
{
    double total_length = 0.0;
    auto prefix = boundary_prefix_lengths(mesh, loop, total_length);
    if (loop.empty() || total_length <= kEps) {
        return;
    }

    for (int i = 0; i < static_cast<int>(loop.size()); ++i) {
        double t = prefix[i] / total_length;
        double theta = 2.0 * kPi * t;

        auto vh = mesh->vertex_handle(loop[i]);
        auto& p = mesh->point(vh);
        p[0] = static_cast<float>(0.5 + 0.5 * std::cos(theta));
        p[1] = static_cast<float>(0.5 + 0.5 * std::sin(theta));
        p[2] = 0.0f;
    }
}

template<typename MeshT>
void map_boundary_to_square(MeshT* mesh, const std::vector<int>& loop)
{
    double total_length = 0.0;
    auto prefix = boundary_prefix_lengths(mesh, loop, total_length);
    if (loop.empty() || total_length <= kEps) {
        return;
    }

    for (int i = 0; i < static_cast<int>(loop.size()); ++i) {
        double t = prefix[i] / total_length;
        auto uv = map_to_square_boundary(t);

        auto vh = mesh->vertex_handle(loop[i]);
        auto& p = mesh->point(vh);
        p[0] = static_cast<float>(uv.first);
        p[1] = static_cast<float>(uv.second);
        p[2] = 0.0f;
    }
}

template<typename MeshT>
bool build_boundary_mapping_on_mesh(
    MeshT* mesh,
    int mode,
    bool flip_orientation)
{
    if (!mesh) {
        return false;
    }

    auto boundary_loop = extract_outer_boundary_loop(mesh, flip_orientation);
    if (boundary_loop.size() < 3) {
        return false;
    }

    flatten_all_vertices_to_xy_plane(mesh);

    if (mode == BOUNDARY_CIRCLE) {
        map_boundary_to_circle(mesh, boundary_loop);
    }
    else {
        map_boundary_to_square(mesh, boundary_loop);
    }

    return true;
}

}  // namespace

NODE_DEF_OPEN_SCOPE

NODE_DECLARATION_FUNCTION(hw5_boundary_mapping)
{
    b.add_input<Geometry>("Input");
    b.add_input<int>("FlipOrientation").min(0).max(1).default_val(0);
    b.add_output<Geometry>("Circle");
    b.add_output<Geometry>("Square");
}

NODE_EXECUTION_FUNCTION(hw5_boundary_mapping)
{
    auto input = params.get_input<Geometry>("Input");
    int flip_orientation = params.get_input<int>("FlipOrientation");

    if (!input.get_component<MeshComponent>()) {
        return false;
    }

    {
        auto halfedge_mesh = operand_to_openmesh(&input);
        if (!build_boundary_mapping_on_mesh(
                halfedge_mesh.get(), BOUNDARY_CIRCLE, flip_orientation != 0)) {
            return false;
        }

        auto geometry = openmesh_to_operand(halfedge_mesh.get());
        params.set_output("Circle", std::move(*geometry));
    }

    {
        auto halfedge_mesh = operand_to_openmesh(&input);
        if (!build_boundary_mapping_on_mesh(
                halfedge_mesh.get(), BOUNDARY_SQUARE, flip_orientation != 0)) {
            return false;
        }

        auto geometry = openmesh_to_operand(halfedge_mesh.get());
        params.set_output("Square", std::move(*geometry));
    }

    return true;
}

NODE_DECLARATION_UI(hw5_boundary_mapping);

NODE_DEF_CLOSE_SCOPE