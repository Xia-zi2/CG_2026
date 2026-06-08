#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <algorithm>
#include <array>
#include <cmath>
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

struct TriangleFace {
    std::array<int, 3> vids{};
    std::array<float, 3> vert_cot{};
    std::array<Eigen::Vector2d, 3> local_xy{};
    std::array<Eigen::Vector2d, 3> global_uv{};
    Eigen::Matrix2d Lt = Eigen::Matrix2d::Identity();

    int local_index(int global_idx) const
    {
        for (int i = 0; i < 3; ++i) {
            if (vids[i] == global_idx) {
                return i;
            }
        }
        return -1;
    }

    void update_uv(const std::vector<Eigen::Vector2d>& uv_result)
    {
        for (int i = 0; i < 3; ++i) {
            global_uv[i] = uv_result[static_cast<size_t>(vids[i])];
        }
    }
};

struct FaceASAP {
    std::array<int, 3> vids{};
    std::array<Eigen::Vector2d, 3> ref{};
    std::array<double, 3> vert_cot{};
};

template<typename MeshT>
Eigen::Vector2d get_pos2(MeshT* mesh, int vid)
{
    auto p = mesh->point(mesh->vertex_handle(vid));
    return Eigen::Vector2d(
        static_cast<double>(p[0]), static_cast<double>(p[1]));
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
        const size_t next_idx = (i + 1) % boundary.size();
        const auto& curr_p = mesh->point(boundary[i]);
        const auto& next_p = mesh->point(boundary[next_idx]);
        const double edge_len = (next_p - curr_p).length();

        if (accumulated + edge_len > half_len) {
            fix2 = boundary[next_idx].idx();
            break;
        }
        accumulated += edge_len;
    }

    return { fix1, fix2 };
}

template<typename MeshT>
bool load_init_uv(MeshT* mesh, std::vector<Eigen::Vector2d>& uv)
{
    const int nV = static_cast<int>(mesh->n_vertices());
    uv.assign(nV, Eigen::Vector2d::Zero());

    for (auto v_it = mesh->vertices_begin(); v_it != mesh->vertices_end();
         ++v_it) {
        int vid = (*v_it).idx();
        uv[static_cast<size_t>(vid)] = get_pos2(mesh, vid);
    }
    return nV > 0;
}

template<typename MeshT>
float compute_angle(MeshT* mesh, int i0, int i1, int i2)
{
    auto p0 = mesh->point(mesh->vertex_handle(i0));
    auto p1 = mesh->point(mesh->vertex_handle(i1));
    auto p2 = mesh->point(mesh->vertex_handle(i2));

    Eigen::Vector3d e1(
        static_cast<double>(p1[0] - p0[0]),
        static_cast<double>(p1[1] - p0[1]),
        static_cast<double>(p1[2] - p0[2]));
    Eigen::Vector3d e2(
        static_cast<double>(p2[0] - p0[0]),
        static_cast<double>(p2[1] - p0[1]),
        static_cast<double>(p2[2] - p0[2]));

    const double n1 = e1.norm();
    const double n2 = e2.norm();
    if (n1 <= kEps || n2 <= kEps) {
        return 0.0f;
    }

    double c = e1.dot(e2) / (n1 * n2);
    c = std::max(-1.0, std::min(1.0, c));
    return static_cast<float>(std::acos(c));
}

template<typename MeshT>
float compute_opp_cotangent(
    MeshT* mesh,
    const typename MeshT::HalfedgeHandle& he)
{
    if (!he.is_valid() || mesh->is_boundary(he)) {
        return 0.0f;
    }

    auto v_from = mesh->from_vertex_handle(he);
    auto v_to = mesh->to_vertex_handle(he);
    auto v_next = mesh->to_vertex_handle(mesh->next_halfedge_handle(he));

    const float angle =
        compute_angle(mesh, v_next.idx(), v_from.idx(), v_to.idx());
    if (angle < 1e-6f || angle > static_cast<float>(M_PI) - 1e-6f) {
        return 0.0f;
    }
    return 1.0f / std::tan(angle);
}

template<typename MeshT>
bool build_triangle_maps(
    MeshT* ref_mesh,
    const std::vector<Eigen::Vector2d>& uv_init,
    std::vector<TriangleFace>& triangle_maps)
{
    if (!ref_mesh) {
        return false;
    }

    triangle_maps.clear();
    triangle_maps.resize(static_cast<size_t>(ref_mesh->n_faces()));

    for (auto f_it = ref_mesh->faces_begin(); f_it != ref_mesh->faces_end();
         ++f_it) {
        auto fh = *f_it;
        TriangleFace tri;

        std::array<OpenMesh::Vec3f, 3> points{};
        std::array<float, 3> length{ 0.0f, 0.0f, 0.0f };

        int cnt = 0;
        for (auto fv_it = ref_mesh->fv_begin(fh); fv_it != ref_mesh->fv_end(fh);
             ++fv_it) {
            if (cnt >= 3) {
                return false;
            }
            tri.vids[static_cast<size_t>(cnt)] = (*fv_it).idx();
            points[static_cast<size_t>(cnt)] = ref_mesh->point(
                ref_mesh->vertex_handle(tri.vids[static_cast<size_t>(cnt)]));
            ++cnt;
        }
        if (cnt != 3) {
            return false;
        }

        for (int i = 0; i < 3; ++i) {
            int j = (i + 1) % 3;
            int k = (i + 2) % 3;
            length[static_cast<size_t>(i)] = (points[j] - points[k]).length();

            OpenMesh::Vec3f e1 = (points[j] - points[i]).normalized();
            OpenMesh::Vec3f e2 = (points[k] - points[i]).normalized();
            tri.vert_cot[static_cast<size_t>(i)] =
                e1.dot(e2) / e1.cross(e2).norm();
        }

        const float theta =
            compute_angle(ref_mesh, tri.vids[0], tri.vids[1], tri.vids[2]);

        tri.local_xy = { Eigen::Vector2d(0.0, 0.0),
                         Eigen::Vector2d(length[2], 0.0),
                         Eigen::Vector2d(
                             length[1] * std::cos(theta),
                             length[1] * std::sin(theta)) };

        tri.update_uv(uv_init);
        triangle_maps[static_cast<size_t>(fh.idx())] = tri;
    }

    return true;
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

inline void update_hybrid_similarity(TriangleFace& tri, double lambda)
{
    double c1 = 0.0;
    double c2 = 0.0;
    double c3 = 0.0;

    for (int i = 0; i < 3; ++i) {
        const int j = (i + 1) % 3;
        const int k = (i + 2) % 3;

        const Eigen::Vector2d du = tri.global_uv[static_cast<size_t>(i)] -
                                   tri.global_uv[static_cast<size_t>(j)];
        const Eigen::Vector2d dx = tri.local_xy[static_cast<size_t>(i)] -
                                   tri.local_xy[static_cast<size_t>(j)];
        const double w = tri.vert_cot[static_cast<size_t>(k)];

        c1 += w * dx.squaredNorm();
        c2 += w * du.dot(dx);
        c3 += w * (du.x() * dx.y() - du.y() * dx.x());
    }

    const double q = std::sqrt(c2 * c2 + c3 * c3);
    if (c1 <= kEps || q <= kEps) {
        tri.Lt.setIdentity();
        return;
    }

    double s = q / c1;

    if (lambda > 1e-8) {
        s = std::max(0.0, std::min(1.0, s));
        for (int it = 0; it < 30; ++it) {
            const double F =
                2.0 * lambda * s * s * s + (c1 - 2.0 * lambda) * s - q;
            const double dF = 6.0 * lambda * s * s + (c1 - 2.0 * lambda);

            if (std::abs(dF) <= kEps) {
                break;
            }

            const double s_next = s - F / dF;
            if (std::abs(s_next - s) <= 1e-8) {
                s = s_next;
                break;
            }
            s = s_next;
        }
    }

    const double a = s * c2 / q;
    const double b = s * c3 / q;

    tri.Lt << a, b, -b, a;
}

template<typename MeshT>
void build_cot_halfedge(MeshT* mesh, std::vector<float>& cot_halfedge)
{
    cot_halfedge.assign(static_cast<size_t>(mesh->n_halfedges()), 0.0f);
    for (auto he_it = mesh->halfedges_begin(); he_it != mesh->halfedges_end();
         ++he_it) {
        auto he = *he_it;
        cot_halfedge[static_cast<size_t>(he.idx())] =
            compute_opp_cotangent(mesh, he);
    }
}

inline bool factor_reduced_system(
    const Eigen::SparseMatrix<double>& K,
    const std::vector<int>& free_ids,
    Eigen::SparseMatrix<double>& Kff,
    Eigen::SparseLU<Eigen::SparseMatrix<double>>& solver)
{
    const int n = K.rows();
    const int nF = static_cast<int>(free_ids.size());

    std::vector<int> old_to_new(n, -1);
    for (int i = 0; i < nF; ++i) {
        old_to_new[free_ids[static_cast<size_t>(i)]] = i;
    }

    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(K.nonZeros());

    for (int col = 0; col < K.outerSize(); ++col) {
        int new_col = old_to_new[col];
        if (new_col < 0) {
            continue;
        }
        for (Eigen::SparseMatrix<double>::InnerIterator it(K, col); it; ++it) {
            int new_row = old_to_new[it.row()];
            if (new_row >= 0 && std::abs(it.value()) > kEps) {
                trips.emplace_back(new_row, new_col, it.value());
            }
        }
    }

    Kff.resize(nF, nF);
    Kff.setFromTriplets(trips.begin(), trips.end());
    Kff.makeCompressed();

    solver.analyzePattern(Kff);
    solver.factorize(Kff);
    return solver.info() == Eigen::Success;
}

inline Eigen::VectorXd build_reduced_rhs(
    const Eigen::SparseMatrix<double>& K,
    const std::vector<int>& free_ids,
    const std::vector<int>& fixed_ids,
    const Eigen::VectorXd& b,
    const Eigen::VectorXd& fixed_vals)
{
    Eigen::VectorXd rhs(static_cast<int>(free_ids.size()));
    for (int r = 0; r < static_cast<int>(free_ids.size()); ++r) {
        double val = b[free_ids[static_cast<size_t>(r)]];
        for (int c = 0; c < static_cast<int>(fixed_ids.size()); ++c) {
            val -= K.coeff(free_ids[static_cast<size_t>(r)], fixed_ids[c]) *
                   fixed_vals[c];
        }
        rhs[r] = val;
    }
    return rhs;
}

template<typename MeshT>
bool build_reduced_laplacian(
    MeshT* mesh,
    const std::vector<float>& cot_halfedge,
    const std::vector<int>& fixed_ids,
    std::vector<int>& free_ids,
    Eigen::SparseMatrix<double>& K,
    Eigen::SparseMatrix<double>& Kff,
    Eigen::SparseLU<Eigen::SparseMatrix<double>>& solver)
{
    const int nV = static_cast<int>(mesh->n_vertices());

    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(static_cast<size_t>(nV) * 8);

    for (auto e_it = mesh->edges_begin(); e_it != mesh->edges_end(); ++e_it) {
        auto edge = *e_it;
        auto he = mesh->halfedge_handle(edge, 0);

        const int i = mesh->from_vertex_handle(he).idx();
        const int j = mesh->to_vertex_handle(he).idx();
        const double w =
            static_cast<double>(cot_halfedge[static_cast<size_t>(he.idx())]) +
            static_cast<double>(cot_halfedge[static_cast<size_t>(
                mesh->opposite_halfedge_handle(he).idx())]);

        trips.emplace_back(i, i, w);
        trips.emplace_back(j, j, w);
        trips.emplace_back(i, j, -w);
        trips.emplace_back(j, i, -w);
    }

    K.resize(nV, nV);
    K.setFromTriplets(trips.begin(), trips.end());
    K.makeCompressed();

    std::vector<char> is_fixed(static_cast<size_t>(nV), 0);
    for (int vid : fixed_ids) {
        if (vid >= 0 && vid < nV) {
            is_fixed[static_cast<size_t>(vid)] = 1;
        }
    }

    free_ids.clear();
    free_ids.reserve(
        static_cast<size_t>(nV - static_cast<int>(fixed_ids.size())));
    for (int i = 0; i < nV; ++i) {
        if (!is_fixed[static_cast<size_t>(i)]) {
            free_ids.push_back(i);
        }
    }

    return factor_reduced_system(K, free_ids, Kff, solver);
}

inline double signed_double_area(
    const Eigen::Vector2d& a,
    const Eigen::Vector2d& b,
    const Eigen::Vector2d& c)
{
    return (b.x() - a.x()) * (c.y() - a.y()) -
           (b.y() - a.y()) * (c.x() - a.x());
}

inline bool has_nonpositive_det(const std::vector<TriangleFace>& tris)
{
    for (const auto& tri : tris) {
        const double area2 = signed_double_area(
            tri.global_uv[0], tri.global_uv[1], tri.global_uv[2]);
        if (area2 <= 0.0) {
            return true;
        }
    }
    return false;
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
            const int j = (i + 1) % 3;
            const int k = (i + 2) % 3;
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
bool solve_asap_exact(MeshT* mesh, std::vector<Eigen::Vector2d>& uv_out)
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
        const int yi = nV + xi;
        const int yj = nV + xj;

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
            const int yi = nV + xi;
            const int yj = nV + xj;
            const int a_col = 2 * nV + face_idx;
            const int b_col = a_col + nF;
            const double c1 = w * (dx * dx + dy * dy);

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
            const int j = (i + 1) % 3;
            const int k = (i + 2) % 3;

            const int xi = tri.vids[i];
            const int xj = tri.vids[j];

            if (xi == fix1 || xi == fix2) {
                continue;
            }

            const double w = tri.vert_cot[k];
            const double dx = tri.ref[i].x() - tri.ref[j].x();
            const double dy = tri.ref[i].y() - tri.ref[j].y();

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
        uv_out[static_cast<size_t>(i)] = Eigen::Vector2d(sol[i], sol[nV + i]);
    }

    normalize_uv(uv_out);
    return true;
}

template<typename MeshTRef, typename MeshTInit>
bool solve_hybrid(
    MeshTRef* ref_mesh,
    MeshTInit* init_mesh,
    int max_iter,
    double lambda,
    std::vector<Eigen::Vector2d>& uv_out)
{
    if (!ref_mesh || !init_mesh) {
        return false;
    }

    const int nV = static_cast<int>(ref_mesh->n_vertices());
    if (nV <= 0 || static_cast<int>(init_mesh->n_vertices()) != nV) {
        return false;
    }

    if (std::abs(lambda) <= 1e-12) {
        return solve_asap_exact(ref_mesh, uv_out);
    }

    std::vector<Eigen::Vector2d> uv;
    if (!load_init_uv(init_mesh, uv)) {
        return false;
    }

    std::vector<TriangleFace> triangle_maps;
    if (!build_triangle_maps(ref_mesh, uv, triangle_maps)) {
        return false;
    }

    std::vector<float> cot_halfedge;
    build_cot_halfedge(ref_mesh, cot_halfedge);

    auto fixed_pair = choose_fixed_vertices_from_longest_boundary(ref_mesh);
    const int fix1 = fixed_pair.first;
    const int fix2 = fixed_pair.second;
    if (fix1 < 0 || fix2 < 0 || fix1 == fix2) {
        return false;
    }

    const std::vector<int> fixed_ids{ fix1, fix2 };

    std::vector<int> free_ids;
    Eigen::SparseMatrix<double> K, Kff;
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    if (!build_reduced_laplacian(
            ref_mesh, cot_halfedge, fixed_ids, free_ids, K, Kff, solver)) {
        return false;
    }

    const Eigen::Vector2d fix1_uv0 = uv[static_cast<size_t>(fix1)];
    const Eigen::Vector2d fix2_uv0 = uv[static_cast<size_t>(fix2)];

    for (int iter = 0; iter < max_iter; ++iter) {
        for (auto& tri : triangle_maps) {
            tri.update_uv(uv);
            update_hybrid_similarity(tri, lambda);
        }

        Eigen::MatrixXd rhs = Eigen::MatrixXd::Zero(nV, 2);

        for (auto f_it = ref_mesh->faces_begin(); f_it != ref_mesh->faces_end();
             ++f_it) {
            const auto& tri = triangle_maps[static_cast<size_t>((*f_it).idx())];

            for (auto he_it = ref_mesh->fh_begin(*f_it);
                 he_it != ref_mesh->fh_end(*f_it);
                 ++he_it) {
                auto he = *he_it;
                const int vi = ref_mesh->from_vertex_handle(he).idx();
                const int vj = ref_mesh->to_vertex_handle(he).idx();

                const float w = cot_halfedge[static_cast<size_t>(he.idx())];
                const int li = tri.local_index(vi);
                const int lj = tri.local_index(vj);
                if (li < 0 || lj < 0) {
                    continue;
                }

                const Eigen::Vector2d contrib =
                    static_cast<double>(w) * tri.Lt *
                    (tri.local_xy[static_cast<size_t>(li)] -
                     tri.local_xy[static_cast<size_t>(lj)]);

                rhs.row(vi) += contrib.transpose();
                rhs.row(vj) -= contrib.transpose();
            }
        }

        Eigen::VectorXd bx(nV), by(nV);
        for (int i = 0; i < nV; ++i) {
            bx[i] = rhs(i, 0);
            by[i] = rhs(i, 1);
        }

        Eigen::VectorXd fixed_x(2), fixed_y(2);
        fixed_x << fix1_uv0.x(), fix2_uv0.x();
        fixed_y << fix1_uv0.y(), fix2_uv0.y();

        Eigen::VectorXd rhs_x =
            build_reduced_rhs(K, free_ids, fixed_ids, bx, fixed_x);
        Eigen::VectorXd rhs_y =
            build_reduced_rhs(K, free_ids, fixed_ids, by, fixed_y);

        Eigen::VectorXd sol_x = solver.solve(rhs_x);
        if (solver.info() != Eigen::Success) {
            return false;
        }
        Eigen::VectorXd sol_y = solver.solve(rhs_y);
        if (solver.info() != Eigen::Success) {
            return false;
        }

        std::vector<Eigen::Vector2d> uv_candidate = uv;
        uv_candidate[static_cast<size_t>(fix1)] = fix1_uv0;
        uv_candidate[static_cast<size_t>(fix2)] = fix2_uv0;

        for (int i = 0; i < static_cast<int>(free_ids.size()); ++i) {
            const int vid = free_ids[static_cast<size_t>(i)];
            uv_candidate[static_cast<size_t>(vid)] =
                Eigen::Vector2d(sol_x[i], sol_y[i]);
        }

        for (auto& tri : triangle_maps) {
            tri.update_uv(uv_candidate);
        }

        /*
        if (has_nonpositive_det(triangle_maps)) {
            // Keep previous uv if some face becomes flipped.
            // This block is intentionally commented out for testing.
        } else {
            uv.swap(uv_candidate);
        }
        */
        uv.swap(uv_candidate);
    }

    uv_out = uv;
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
        set_pos2(mesh, i, uv[static_cast<size_t>(i)]);
    }
    return true;
}

}  // namespace

NODE_DEF_OPEN_SCOPE

NODE_DECLARATION_FUNCTION(hw6_hybrid)
{
    b.add_input<Geometry>("Reference");
    b.add_input<Geometry>("Init");
    b.add_input<int>("Iteration").min(0).max(100).default_val(1);
    b.add_input<float>("Lambda").min(0.001f).max(1.0f).default_val(0.001f);
    b.add_output<Geometry>("Hybrid");
}

NODE_EXECUTION_FUNCTION(hw6_hybrid)
{
    auto reference = params.get_input<Geometry>("Reference");
    auto init = params.get_input<Geometry>("Init");
    int max_iter = params.get_input<int>("Iteration");
    float lambda = params.get_input<float>("Lambda");
    lambda = std::max(0.001f, std::min(1.0f, lambda));

    if (!reference.get_component<MeshComponent>()) {
        return false;
    }
    if (!init.get_component<MeshComponent>()) {
        return false;
    }

    auto ref_mesh = operand_to_openmesh(&reference);
    auto init_mesh = operand_to_openmesh(&init);

    std::vector<Eigen::Vector2d> uv;
    if (!solve_hybrid(
            ref_mesh.get(),
            init_mesh.get(),
            max_iter,
            static_cast<double>(lambda),
            uv)) {
        return false;
    }

    auto out_mesh = operand_to_openmesh(&reference);
    if (!write_uv_back_to_mesh(out_mesh.get(), uv)) {
        return false;
    }

    auto geometry = openmesh_to_operand(out_mesh.get());
    params.set_output("Hybrid", std::move(*geometry));
    return true;
}

NODE_DECLARATION_UI(hw6_hybrid);
NODE_DEF_CLOSE_SCOPE