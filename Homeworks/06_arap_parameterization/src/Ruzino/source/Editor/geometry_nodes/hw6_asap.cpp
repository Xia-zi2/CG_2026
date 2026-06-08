#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <set>
#include <vector>

#include "GCore/Components.h"
#include "GCore/Components/MeshComponent.h"
#include "GCore/GOP.h"
#include "GCore/util_openmesh_bind.h"
#include "geom_node_base.h"
#include "nodes/core/def/node_def.hpp"

namespace {

constexpr double kEps = 1e-12;

struct FaceASAP {
    std::array<int, 3> vids{};
    std::array<Eigen::Vector2d, 3> ref{};
    std::array<double, 3> vert_cot{};
};

template<typename MeshT>
Eigen::Vector3d get_pos3(MeshT* mesh, int vid)
{
    auto p = mesh->point(mesh->vertex_handle(vid));
    return Eigen::Vector3d(
        static_cast<double>(p[0]),
        static_cast<double>(p[1]),
        static_cast<double>(p[2]));
}

template<typename MeshT>
void set_pos2(MeshT* mesh, int vid, const Eigen::Vector2d& uv)
{
    auto vh = mesh->vertex_handle(vid);
    mesh->set_point(
        vh,
        OpenMesh::Vec3f(
            static_cast<float>(uv.x()), static_cast<float>(uv.y()), 0.0f));
}

inline double cross2d(const Eigen::Vector2d& a, const Eigen::Vector2d& b)
{
    return a.x() * b.y() - a.y() * b.x();
}

inline double cot_weight(
    const Eigen::Vector2d& p0,
    const Eigen::Vector2d& p1,
    const Eigen::Vector2d& p2)
{
    Eigen::Vector2d v1 = p1 - p0;
    Eigen::Vector2d v2 = p2 - p0;
    double denom = std::abs(cross2d(v1, v2));
    if (denom <= kEps) {
        return 0.0;
    }
    return v1.dot(v2) / denom;
}

template<typename MeshT>
std::vector<std::vector<typename MeshT::VertexHandle>> find_all_boundary_loops(
    MeshT* mesh)
{
    using VH = typename MeshT::VertexHandle;
    using HEH = typename MeshT::HalfedgeHandle;

    std::vector<std::vector<VH>> loops;
    std::set<int> visited_boundary_halfedges;

    for (auto he_it = mesh->halfedges_begin(); he_it != mesh->halfedges_end();
         ++he_it) {
        HEH start_he = *he_it;
        if (!mesh->is_boundary(start_he)) {
            continue;
        }
        if (visited_boundary_halfedges.count(start_he.idx()) > 0) {
            continue;
        }

        std::vector<VH> loop;
        HEH cur_he = start_he;
        do {
            visited_boundary_halfedges.insert(cur_he.idx());
            loop.push_back(mesh->from_vertex_handle(cur_he));
            cur_he = mesh->next_halfedge_handle(cur_he);
        } while (cur_he.is_valid() && cur_he != start_he);

        if (!loop.empty()) {
            loops.push_back(loop);
        }
    }

    return loops;
}

template<typename MeshT>
double calculate_boundary_length(
    MeshT* mesh,
    const std::vector<typename MeshT::VertexHandle>& boundary)
{
    double total = 0.0;
    const size_t n = boundary.size();
    if (n < 2) {
        return total;
    }

    for (size_t i = 0; i < n; ++i) {
        auto v0 = boundary[i];
        auto v1 = boundary[(i + 1) % n];
        auto p0 = mesh->point(v0);
        auto p1 = mesh->point(v1);
        total += (p1 - p0).length();
    }
    return total;
}

template<typename MeshT>
std::vector<typename MeshT::VertexHandle> find_longest_boundary_loop(
    MeshT* mesh)
{
    auto loops = find_all_boundary_loops(mesh);
    if (loops.empty()) {
        return {};
    }

    size_t best_id = 0;
    double best_len = -1.0;
    for (size_t i = 0; i < loops.size(); ++i) {
        double len = calculate_boundary_length(mesh, loops[i]);
        if (len > best_len) {
            best_len = len;
            best_id = i;
        }
    }
    return loops[best_id];
}

template<typename MeshT>
std::pair<int, int> choose_fixed_vertices_from_longest_boundary(MeshT* mesh)
{
    auto boundary = find_longest_boundary_loop(mesh);
    if (boundary.size() < 2) {
        return { -1, -1 };
    }

    const double total_len = calculate_boundary_length(mesh, boundary);
    const double half_len = 0.5 * total_len;

    int fix1 = boundary.front().idx();
    int fix2 = boundary.back().idx();

    double accumulated = 0.0;
    for (size_t i = 0; i < boundary.size(); ++i) {
        size_t next_i = (i + 1) % boundary.size();
        auto p0 = mesh->point(boundary[i]);
        auto p1 = mesh->point(boundary[next_i]);
        double seg_len = (p1 - p0).length();

        if (accumulated + seg_len > half_len) {
            fix2 = boundary[next_i].idx();
            break;
        }
        accumulated += seg_len;
    }

    return { fix1, fix2 };
}

inline void normalize_uv(std::vector<Eigen::Vector2d>& uv)
{
    if (uv.empty()) {
        return;
    }

    Eigen::Vector2d mn = uv[0];
    Eigen::Vector2d mx = uv[0];
    for (const auto& p : uv) {
        mn = mn.cwiseMin(p);
        mx = mx.cwiseMax(p);
    }

    Eigen::Vector2d ext = mx - mn;
    double scale = std::max(ext.x(), ext.y());
    if (scale <= kEps) {
        scale = 1.0;
    }

    for (auto& p : uv) {
        p = (p - mn) / scale;
    }
}

inline int local_index(const FaceASAP& face, int vid)
{
    for (int i = 0; i < 3; ++i) {
        if (face.vids[i] == vid) {
            return i;
        }
    }
    return -1;
}

template<typename MeshT>
bool build_face_reference(MeshT* mesh, std::vector<FaceASAP>& faces)
{
    faces.assign(static_cast<int>(mesh->n_faces()), FaceASAP{});

    for (auto f_it = mesh->faces_begin(); f_it != mesh->faces_end(); ++f_it) {
        auto fh = *f_it;
        FaceASAP face;

        int cnt = 0;
        std::array<OpenMesh::Vec3f, 3> points{};
        for (auto fv_it = mesh->fv_begin(fh); fv_it != mesh->fv_end(fh);
             ++fv_it) {
            if (cnt < 0 || cnt >= static_cast<int>(face.vids.size())) {
                return false;
            }
            int vid = (*fv_it).idx();
            face.vids[static_cast<size_t>(cnt)] = vid;
            points[static_cast<size_t>(cnt)] =
                mesh->point(mesh->vertex_handle(vid));
            ++cnt;
        }
        if (cnt != 3) {
            return false;
        }

        std::array<double, 3> length{};
        for (int i = 0; i < 3; ++i) {
            int j = (i + 1) % 3;
            int k = (i + 2) % 3;
            length[static_cast<size_t>(i)] = (points[j] - points[k]).length();

            OpenMesh::Vec3f e1 = (points[j] - points[i]).normalized();
            OpenMesh::Vec3f e2 = (points[k] - points[i]).normalized();
            face.vert_cot[static_cast<size_t>(i)] =
                e1.dot(e2) / e1.cross(e2).norm();
        }

        Eigen::Vector3d p0(
            static_cast<double>(points[0][0]),
            static_cast<double>(points[0][1]),
            static_cast<double>(points[0][2]));
        Eigen::Vector3d p1(
            static_cast<double>(points[1][0]),
            static_cast<double>(points[1][1]),
            static_cast<double>(points[1][2]));
        Eigen::Vector3d p2(
            static_cast<double>(points[2][0]),
            static_cast<double>(points[2][1]),
            static_cast<double>(points[2][2]));

        Eigen::Vector3d v1 = (p1 - p0).normalized();
        Eigen::Vector3d v2 = (p2 - p0).normalized();
        double cos_theta = std::max(-1.0, std::min(1.0, v1.dot(v2)));
        double theta = std::acos(cos_theta);

        face.ref[0] = Eigen::Vector2d(0.0, 0.0);
        face.ref[1] = Eigen::Vector2d(length[2], 0.0);
        face.ref[2] = Eigen::Vector2d(
            length[1] * std::cos(theta), length[1] * std::sin(theta));

        faces[fh.idx()] = face;
    }

    return true;
}

template<typename MeshT>
bool solve_asap(MeshT* mesh, std::vector<Eigen::Vector2d>& uv_out)
{
    if (!mesh) {
        return false;
    }

    const int nV = static_cast<int>(mesh->n_vertices());
    const int nF = static_cast<int>(mesh->n_faces());
    if (nV < 2 || nF <= 0) {
        return false;
    }

    std::vector<FaceASAP> faces;
    if (!build_face_reference(mesh, faces)) {
        return false;
    }

    auto fixed_pair = choose_fixed_vertices_from_longest_boundary(mesh);
    const int fix1 = fixed_pair.first;
    const int fix2 = fixed_pair.second;
    if (fix1 < 0 || fix2 < 0 || fix1 == fix2) {
        return false;
    }

    const int dim = 2 * nV + 2 * nF;
    Eigen::SparseMatrix<double> A(dim, dim);
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(dim);
    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(std::max(128, nV * 20 + nF * 40));

    auto add_coordinate_terms = [&](int xi, int xj, double w) {
        int yi = nV + xi;
        int yj = nV + xj;

        trips.emplace_back(xi, xi, w);
        trips.emplace_back(xi, xj, -w);
        trips.emplace_back(xj, xj, w);
        trips.emplace_back(xj, xi, -w);

        trips.emplace_back(yi, yi, w);
        trips.emplace_back(yi, yj, -w);
        trips.emplace_back(yj, yj, w);
        trips.emplace_back(yj, yi, -w);
    };

    auto add_similarity_terms =
        [&](int xi, int xj, int face_idx, double w, double dx, double dy) {
            int yi = nV + xi;
            int yj = nV + xj;
            int a_col = 2 * nV + face_idx;
            int b_col = a_col + nF;
            double c1 = w * (dx * dx + dy * dy);

            trips.emplace_back(a_col, a_col, c1);
            trips.emplace_back(b_col, b_col, c1);

            trips.emplace_back(xi, a_col, -w * dx);
            trips.emplace_back(xi, b_col, -w * dy);
            trips.emplace_back(xj, a_col, w * dx);
            trips.emplace_back(xj, b_col, w * dy);

            trips.emplace_back(yi, a_col, -w * dy);
            trips.emplace_back(yi, b_col, w * dx);
            trips.emplace_back(yj, a_col, w * dy);
            trips.emplace_back(yj, b_col, -w * dx);

            trips.emplace_back(a_col, xi, -w * dx);
            trips.emplace_back(b_col, xi, -w * dy);
            trips.emplace_back(a_col, xj, w * dx);
            trips.emplace_back(b_col, xj, w * dy);

            trips.emplace_back(a_col, yi, -w * dy);
            trips.emplace_back(b_col, yi, w * dx);
            trips.emplace_back(a_col, yj, w * dy);
            trips.emplace_back(b_col, yj, -w * dx);
        };

    for (int face_idx = 0; face_idx < nF; ++face_idx) {
        const FaceASAP& tri = faces[face_idx];

        for (int i = 0; i < 3; ++i) {
            int j = (i + 1) % 3;
            int k = (i + 2) % 3;

            int xi = tri.vids[i];
            int xj = tri.vids[j];

            if (xi == fix1 || xi == fix2) {
                continue;
            }

            double w = tri.vert_cot[k];
            double dx = tri.ref[i].x() - tri.ref[j].x();
            double dy = tri.ref[i].y() - tri.ref[j].y();

            add_coordinate_terms(xi, xj, w);
            add_similarity_terms(xi, xj, face_idx, w, dx, dy);
        }
    }

    trips.emplace_back(fix1, fix1, 1.0);
    trips.emplace_back(nV + fix1, nV + fix1, 1.0);
    trips.emplace_back(fix2, fix2, 1.0);
    trips.emplace_back(nV + fix2, nV + fix2, 1.0);

    rhs[fix1] = 0.0;
    rhs[nV + fix1] = 0.0;
    rhs[fix2] = 0.0;
    rhs[nV + fix2] = 1.0;

    A.resize(dim, dim);
    A.setFromTriplets(trips.begin(), trips.end());
    A.makeCompressed();

    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.analyzePattern(A);
    solver.factorize(A);
    if (solver.info() != Eigen::Success) {
        return false;
    }

    Eigen::VectorXd sol = solver.solve(rhs);
    if (solver.info() != Eigen::Success) {
        return false;
    }

    uv_out.assign(nV, Eigen::Vector2d::Zero());
    for (int i = 0; i < nV; ++i) {
        uv_out[i] = Eigen::Vector2d(sol[i], sol[nV + i]);
    }

    normalize_uv(uv_out);
    return true;
}

template<typename MeshT>
bool write_uv_back_to_mesh(MeshT* mesh, const std::vector<Eigen::Vector2d>& uv)
{
    if (!mesh ||
        static_cast<int>(mesh->n_vertices()) != static_cast<int>(uv.size())) {
        return false;
    }

    for (int i = 0; i < static_cast<int>(uv.size()); ++i) {
        set_pos2(mesh, i, uv[i]);
    }
    return true;
}

}  // namespace

NODE_DEF_OPEN_SCOPE

NODE_DECLARATION_FUNCTION(hw6_asap)
{
    b.add_input<Geometry>("Reference");
    b.add_output<Geometry>("ASAP");
}

NODE_EXECUTION_FUNCTION(hw6_asap)
{
    auto reference = params.get_input<Geometry>("Reference");
    if (!reference.get_component<MeshComponent>()) {
        return false;
    }

    auto ref_mesh = operand_to_openmesh(&reference);

    std::vector<Eigen::Vector2d> uv;
    if (!solve_asap(ref_mesh.get(), uv)) {
        return false;
    }

    auto out_mesh = operand_to_openmesh(&reference);
    if (!write_uv_back_to_mesh(out_mesh.get(), uv)) {
        return false;
    }

    auto geometry = openmesh_to_operand(out_mesh.get());
    params.set_output("ASAP", std::move(*geometry));
    return true;
}

NODE_DECLARATION_UI(hw6_asap);
NODE_DEF_CLOSE_SCOPE