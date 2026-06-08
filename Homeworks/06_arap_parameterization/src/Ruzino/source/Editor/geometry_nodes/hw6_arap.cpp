#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>
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

constexpr double kPi = 3.14159265358979323846;
constexpr double kEps = 1e-12;

struct FaceData {
    std::array<int, 3> vids{};
    double area = 0.0;
    std::array<Eigen::Vector2d, 3> grad{};
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

inline Eigen::Matrix2d nearest_rotation(const Eigen::Matrix2d& J)
{
    Eigen::JacobiSVD<Eigen::Matrix2d> svd(
        J, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix2d U = svd.matrixU();
    Eigen::Matrix2d V = svd.matrixV();

    Eigen::Matrix2d R = U * V.transpose();
    if (R.determinant() < 0.0) {
        U.col(1) *= -1.0;
        R = U * V.transpose();
    }
    return R;
}

template<typename MeshT>
bool build_face_data(MeshT* mesh, std::vector<FaceData>& faces)
{
    faces.clear();
    faces.reserve(static_cast<int>(mesh->n_faces()));

    for (auto f_it = mesh->faces_begin(); f_it != mesh->faces_end(); ++f_it) {
        auto fh = *f_it;
        FaceData fd;

        int cnt = 0;
        for (auto fv_it = mesh->fv_begin(fh); fv_it != mesh->fv_end(fh);
             ++fv_it) {
            if (cnt < 0 || cnt >= static_cast<int>(fd.vids.size())) {
                return false;
            }
            fd.vids[static_cast<size_t>(cnt)] = (*fv_it).idx();
            ++cnt;
        }
        if (cnt != 3) {
            return false;
        }

        Eigen::Vector3d p0 = get_pos3(mesh, fd.vids[0]);
        Eigen::Vector3d p1 = get_pos3(mesh, fd.vids[1]);
        Eigen::Vector3d p2 = get_pos3(mesh, fd.vids[2]);

        Eigen::Vector3d e01 = p1 - p0;
        Eigen::Vector3d e02 = p2 - p0;
        double len01 = e01.norm();
        Eigen::Vector3d n = e01.cross(e02);
        double n_norm = n.norm();
        if (len01 <= kEps || n_norm <= kEps) {
            continue;
        }

        Eigen::Vector3d axis_x = e01 / len01;
        Eigen::Vector3d normal = n / n_norm;
        Eigen::Vector3d axis_y = normal.cross(axis_x);

        Eigen::Vector2d x0(0.0, 0.0);
        Eigen::Vector2d x1(len01, 0.0);
        Eigen::Vector2d x2((p2 - p0).dot(axis_x), (p2 - p0).dot(axis_y));

        double twice_area = (x1.x() - x0.x()) * (x2.y() - x0.y()) -
                            (x1.y() - x0.y()) * (x2.x() - x0.x());

        if (std::abs(twice_area) <= kEps) {
            continue;
        }

        if (twice_area < 0.0) {
            std::swap(fd.vids[1], fd.vids[2]);
            std::swap(x1, x2);
            twice_area = -twice_area;
        }

        fd.area = 0.5 * twice_area;
        fd.grad[0] =
            Eigen::Vector2d(x1.y() - x2.y(), x2.x() - x1.x()) / twice_area;
        fd.grad[1] =
            Eigen::Vector2d(x2.y() - x0.y(), x0.x() - x2.x()) / twice_area;
        fd.grad[2] =
            Eigen::Vector2d(x0.y() - x1.y(), x1.x() - x0.x()) / twice_area;

        faces.push_back(fd);
    }

    return !faces.empty();
}

inline Eigen::Matrix2d face_jacobian(
    const FaceData& fd,
    const std::vector<Eigen::Vector2d>& uv)
{
    Eigen::Matrix2d J = Eigen::Matrix2d::Zero();
    for (int k = 0; k < 3; ++k) {
        J += uv[fd.vids[k]] * fd.grad[k].transpose();
    }
    return J;
}

inline void add_face_stiffness(
    const FaceData& fd,
    std::vector<Eigen::Triplet<double>>& trips)
{
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            trips.emplace_back(
                fd.vids[i], fd.vids[j], fd.area * fd.grad[i].dot(fd.grad[j]));
        }
    }
}

inline void add_face_rhs(
    const FaceData& fd,
    const Eigen::Matrix2d& R,
    Eigen::VectorXd& bx,
    Eigen::VectorXd& by)
{
    for (int i = 0; i < 3; ++i) {
        Eigen::Vector2d rhs_i = fd.area * (R * fd.grad[i]);
        bx[fd.vids[i]] += rhs_i.x();
        by[fd.vids[i]] += rhs_i.y();
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
        old_to_new[free_ids[i]] = i;
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
        double val = b[free_ids[r]];
        for (int c = 0; c < static_cast<int>(fixed_ids.size()); ++c) {
            val -= K.coeff(free_ids[r], fixed_ids[c]) * fixed_vals[c];
        }
        rhs[r] = val;
    }
    return rhs;
}

template<typename MeshT>
bool load_init_uv(MeshT* mesh, std::vector<Eigen::Vector2d>& uv)
{
    const int nV = static_cast<int>(mesh->n_vertices());
    uv.assign(nV, Eigen::Vector2d::Zero());

    for (auto v_it = mesh->vertices_begin(); v_it != mesh->vertices_end();
         ++v_it) {
        int vid = (*v_it).idx();
        uv[vid] = get_pos2(mesh, vid);
    }
    return nV > 0;
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

inline void apply_rotation_correction(
    const std::vector<Eigen::Vector2d>& uv_prev,
    std::vector<Eigen::Vector2d>& uv_candidate,
    int fix1,
    int fix2)
{
    if (fix1 < 0 || fix2 < 0 || fix1 >= static_cast<int>(uv_prev.size()) ||
        fix2 >= static_cast<int>(uv_prev.size()) || fix1 == fix2) {
        return;
    }

    const Eigen::Vector2d fix1_prev = uv_prev[fix1];
    const Eigen::Vector2d fix2_prev = uv_prev[fix2];
    const Eigen::Vector2d fix2_new = uv_candidate[fix2];

    Eigen::Vector2d axis_prev = fix2_prev - fix1_prev;
    Eigen::Vector2d axis_new = fix2_new - fix1_prev;

    if (axis_prev.norm() <= kEps || axis_new.norm() <= kEps) {
        return;
    }

    axis_prev.normalize();
    axis_new.normalize();

    const double c = axis_new.dot(axis_prev);
    const double s =
        -axis_prev.x() * axis_new.y() + axis_prev.y() * axis_new.x();

    for (int i = 0; i < static_cast<int>(uv_candidate.size()); ++i) {
        Eigen::Vector2d p = uv_candidate[i] - fix1_prev;
        Eigen::Vector2d pr(p.x() * c - p.y() * s, p.x() * s + p.y() * c);
        uv_candidate[i] = fix1_prev + pr;
    }
}

inline bool has_nonpositive_det(
    const std::vector<FaceData>& faces,
    const std::vector<Eigen::Vector2d>& uv)
{
    for (const auto& fd : faces) {
        Eigen::Matrix2d J = face_jacobian(fd, uv);
        if (J.determinant() <= 0.0) {
            return true;
        }
    }
    return false;
}

template<typename MeshTRef, typename MeshTInit>
bool solve_arap(
    MeshTRef* ref_mesh,
    MeshTInit* init_mesh,
    int max_iter,
    std::vector<Eigen::Vector2d>& uv_out)
{
    if (!ref_mesh || !init_mesh) {
        return false;
    }

    const int nV = static_cast<int>(ref_mesh->n_vertices());
    if (nV <= 0 || static_cast<int>(init_mesh->n_vertices()) != nV) {
        return false;
    }

    std::vector<FaceData> faces;
    if (!build_face_data(ref_mesh, faces)) {
        return false;
    }

    std::vector<Eigen::Vector2d> uv;
    if (!load_init_uv(init_mesh, uv)) {
        return false;
    }

    auto fixed_pair = choose_fixed_vertices_from_longest_boundary(ref_mesh);
    const int fix1 = fixed_pair.first;
    const int fix2 = fixed_pair.second;
    if (fix1 < 0 || fix2 < 0 || fix1 == fix2) {
        return false;
    }

    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(static_cast<int>(faces.size()) * 9);
    for (const auto& fd : faces) {
        add_face_stiffness(fd, trips);
    }

    Eigen::SparseMatrix<double> K(nV, nV);
    K.setFromTriplets(trips.begin(), trips.end());
    K.makeCompressed();

    std::vector<int> free_ids;
    free_ids.reserve(nV - 1);
    for (int i = 0; i < nV; ++i) {
        if (i != fix1) {
            free_ids.push_back(i);
        }
    }

    std::vector<int> fixed_ids{ fix1 };

    Eigen::SparseMatrix<double> Kff;
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    if (!factor_reduced_system(K, free_ids, Kff, solver)) {
        return false;
    }

    std::vector<Eigen::Matrix2d> rotations(
        faces.size(), Eigen::Matrix2d::Identity());

    for (int iter = 0; iter < max_iter; ++iter) {
        for (int f = 0; f < static_cast<int>(faces.size()); ++f) {
            rotations[f] = nearest_rotation(face_jacobian(faces[f], uv));
        }

        Eigen::VectorXd bx = Eigen::VectorXd::Zero(nV);
        Eigen::VectorXd by = Eigen::VectorXd::Zero(nV);
        for (int f = 0; f < static_cast<int>(faces.size()); ++f) {
            add_face_rhs(faces[f], rotations[f], bx, by);
        }

        Eigen::VectorXd fixed_x(1), fixed_y(1);
        fixed_x << uv[fix1].x();
        fixed_y << uv[fix1].y();

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
        uv_candidate[fix1] = uv[fix1];
        for (int i = 0; i < static_cast<int>(free_ids.size()); ++i) {
            int vid = free_ids[i];
            uv_candidate[vid] = Eigen::Vector2d(sol_x[i], sol_y[i]);
        }

        apply_rotation_correction(uv, uv_candidate, fix1, fix2);

        /*
        if (has_nonpositive_det(faces, uv_candidate)) {
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
        set_pos2(mesh, i, uv[i]);
    }
    return true;
}

}  // namespace

NODE_DEF_OPEN_SCOPE

NODE_DECLARATION_FUNCTION(hw6_arap)
{
    b.add_input<Geometry>("Reference");
    b.add_input<Geometry>("Init");
    b.add_input<int>("Iteration").min(0).max(100).default_val(1);
    b.add_output<Geometry>("ARAP");
}

NODE_EXECUTION_FUNCTION(hw6_arap)
{
    auto reference = params.get_input<Geometry>("Reference");
    auto init = params.get_input<Geometry>("Init");
    int max_iter = params.get_input<int>("Iteration");

    if (!reference.get_component<MeshComponent>()) {
        return false;
    }
    if (!init.get_component<MeshComponent>()) {
        return false;
    }

    auto ref_mesh = operand_to_openmesh(&reference);
    auto init_mesh = operand_to_openmesh(&init);

    std::vector<Eigen::Vector2d> uv;
    if (!solve_arap(ref_mesh.get(), init_mesh.get(), max_iter, uv)) {
        return false;
    }

    auto out_mesh = operand_to_openmesh(&reference);
    if (!write_uv_back_to_mesh(out_mesh.get(), uv)) {
        return false;
    }

    auto geometry = openmesh_to_operand(out_mesh.get());
    params.set_output("ARAP", std::move(*geometry));
    return true;
}

NODE_DECLARATION_UI(hw6_arap);
NODE_DEF_CLOSE_SCOPE