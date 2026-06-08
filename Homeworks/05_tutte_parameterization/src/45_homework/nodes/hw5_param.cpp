#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <cmath>
#include <limits>
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

enum WeightMode {
    WEIGHT_UNIFORM = 0,
    WEIGHT_COTAN = 1,
    WEIGHT_FLOATER = 2,
    WEIGHT_MVC = 3
};

inline double clamp_dot(double x)
{
    if (x < -1.0)
        return -1.0;
    if (x > 1.0)
        return 1.0;
    return x;
}

template<typename MeshT>
Eigen::Vector3d get_pos3(MeshT* mesh, int vid)
{
    auto p = mesh->point(mesh->vertex_handle(vid));
    return Eigen::Vector3d(
        static_cast<double>(p[0]),
        static_cast<double>(p[1]),
        static_cast<double>(p[2]));
}

inline double angle_at_3d(
    const Eigen::Vector3d& a,
    const Eigen::Vector3d& b,
    const Eigen::Vector3d& c)
{
    Eigen::Vector3d u = a - b;
    Eigen::Vector3d v = c - b;
    double nu = u.norm();
    double nv = v.norm();
    if (nu <= kEps || nv <= kEps) {
        return 0.0;
    }
    return std::acos(clamp_dot(u.dot(v) / (nu * nv)));
}

inline double cot_at_3d(
    const Eigen::Vector3d& a,
    const Eigen::Vector3d& b,
    const Eigen::Vector3d& c)
{
    Eigen::Vector3d u = a - b;
    Eigen::Vector3d v = c - b;
    double denom = u.cross(v).norm();
    if (denom <= kEps) {
        return 0.0;
    }
    return u.dot(v) / denom;
}

inline double angle_between_2d(
    const Eigen::Vector2d& a,
    const Eigen::Vector2d& b)
{
    double na = a.norm();
    double nb = b.norm();
    if (na <= kEps || nb <= kEps) {
        return 0.0;
    }
    return std::acos(clamp_dot(a.dot(b) / (na * nb)));
}

template<typename MeshT>
std::vector<int> ordered_neighbors(MeshT* mesh, int vid)
{
    std::vector<int> nbrs;
    auto vh = mesh->vertex_handle(vid);
    for (auto vv_it = mesh->vv_begin(vh); vv_it != mesh->vv_end(vh); ++vv_it) {
        nbrs.push_back((*vv_it).idx());
    }
    return nbrs;
}

inline void normalize_weights(std::vector<double>& w)
{
    double s = 0.0;
    for (double x : w) {
        s += x;
    }

    if (s <= kEps) {
        if (!w.empty()) {
            double u = 1.0 / static_cast<double>(w.size());
            for (double& x : w) {
                x = u;
            }
        }
        return;
    }

    for (double& x : w) {
        x /= s;
    }
}

template<typename MeshT>
std::vector<double>
uniform_weights(MeshT* /*ref_mesh*/, int /*vid*/, const std::vector<int>& nbrs)
{
    std::vector<double> w(nbrs.size(), 1.0);
    normalize_weights(w);
    return w;
}

template<typename MeshT>
std::vector<double>
cotangent_weights_raw(MeshT* ref_mesh, int vid, const std::vector<int>& nbrs)
{
    const int d = static_cast<int>(nbrs.size());
    std::vector<double> w(d, 0.0);
    if (d == 0) {
        return w;
    }

    Eigen::Vector3d pi = get_pos3(ref_mesh, vid);

    for (int k = 0; k < d; ++k) {
        int jp = nbrs[(k - 1 + d) % d];
        int jj = nbrs[k];
        int jn = nbrs[(k + 1) % d];

        Eigen::Vector3d pprev = get_pos3(ref_mesh, jp);
        Eigen::Vector3d pj = get_pos3(ref_mesh, jj);
        Eigen::Vector3d pnext = get_pos3(ref_mesh, jn);

        double c1 = cot_at_3d(pi, pprev, pj);
        double c2 = cot_at_3d(pi, pnext, pj);
        w[k] = c1 + c2;
    }

    return w;
}

template<typename MeshT>
std::vector<double>
mvc_weights(MeshT* ref_mesh, int vid, const std::vector<int>& nbrs)
{
    const int d = static_cast<int>(nbrs.size());
    std::vector<double> w(d, 0.0);
    if (d == 0) {
        return w;
    }

    Eigen::Vector3d pi = get_pos3(ref_mesh, vid);

    for (int k = 0; k < d; ++k) {
        int jp = nbrs[(k - 1 + d) % d];
        int jj = nbrs[k];
        int jn = nbrs[(k + 1) % d];

        Eigen::Vector3d pprev = get_pos3(ref_mesh, jp);
        Eigen::Vector3d pj = get_pos3(ref_mesh, jj);
        Eigen::Vector3d pnext = get_pos3(ref_mesh, jn);

        double a1 = angle_at_3d(pprev, pi, pj);
        double a2 = angle_at_3d(pj, pi, pnext);

        double e = (pj - pi).norm();
        if (e <= kEps) {
            w[k] = 0.0;
        }
        else {
            w[k] = (std::tan(0.5 * a1) + std::tan(0.5 * a2)) / e;
        }

        if (!std::isfinite(w[k]) || w[k] < 0.0) {
            w[k] = 0.0;
        }
    }

    normalize_weights(w);
    return w;
}

template<typename MeshT>
std::vector<Eigen::Vector2d> build_floater_local_polygon(
    MeshT* ref_mesh,
    int vid,
    const std::vector<int>& nbrs)
{
    const int d = static_cast<int>(nbrs.size());
    std::vector<Eigen::Vector2d> poly(d, Eigen::Vector2d::Zero());
    if (d == 0) {
        return poly;
    }

    Eigen::Vector3d pi = get_pos3(ref_mesh, vid);

    std::vector<double> alpha(d, 0.0), phi(d, 0.0), radius(d, 0.0);
    double theta_sum = 0.0;

    for (int k = 0; k < d; ++k) {
        int j0 = nbrs[k];
        int j1 = nbrs[(k + 1) % d];

        Eigen::Vector3d p0 = get_pos3(ref_mesh, j0);
        Eigen::Vector3d p1 = get_pos3(ref_mesh, j1);

        radius[k] = (p0 - pi).norm();
        alpha[k] = angle_at_3d(p0, pi, p1);
        theta_sum += alpha[k];
    }

    if (theta_sum <= kEps) {
        for (int k = 0; k < d; ++k) {
            double t =
                2.0 * kPi * static_cast<double>(k) / static_cast<double>(d);
            poly[k] = Eigen::Vector2d(
                radius[k] * std::cos(t), radius[k] * std::sin(t));
        }
        return poly;
    }

    phi[0] = 0.0;
    for (int k = 1; k < d; ++k) {
        phi[k] = phi[k - 1] + 2.0 * kPi * alpha[k - 1] / theta_sum;
    }

    for (int k = 0; k < d; ++k) {
        poly[k] = Eigen::Vector2d(
            radius[k] * std::cos(phi[k]), radius[k] * std::sin(phi[k]));
    }

    return poly;
}

template<typename MeshT>
std::vector<double>
floater_weights(MeshT* ref_mesh, int vid, const std::vector<int>& nbrs)
{
    const int d = static_cast<int>(nbrs.size());
    std::vector<double> w(d, 0.0);

    if (d == 0) {
        return w;
    }
    if (d == 1) {
        w[0] = 1.0;
        return w;
    }
    if (d == 2) {
        w[0] = 0.5;
        w[1] = 0.5;
        return w;
    }

    auto poly = build_floater_local_polygon(ref_mesh, vid, nbrs);

    for (int k = 0; k < d; ++k) {
        const Eigen::Vector2d& vprev = poly[(k - 1 + d) % d];
        const Eigen::Vector2d& v = poly[k];
        const Eigen::Vector2d& vnext = poly[(k + 1) % d];

        double a1 = angle_between_2d(vprev, v);
        double a2 = angle_between_2d(v, vnext);
        double r = v.norm();

        if (r <= kEps) {
            w[k] = 0.0;
        }
        else {
            w[k] = (std::tan(0.5 * a1) + std::tan(0.5 * a2)) / r;
        }

        if (!std::isfinite(w[k]) || w[k] < 0.0) {
            w[k] = 0.0;
        }
    }

    normalize_weights(w);
    return w;
}

}  // namespace

NODE_DEF_OPEN_SCOPE

NODE_DECLARATION_FUNCTION(hw5_param)
{
    b.add_input<Geometry>("Input");
    b.add_input<Geometry>("Reference");
    b.add_output<Geometry>("Output");
}

NODE_EXECUTION_FUNCTION(hw5_param)
{
    const int weight_mode = WEIGHT_UNIFORM;

    auto input = params.get_input<Geometry>("Input");
    if (!input.get_component<MeshComponent>()) {
        return false;
    }

    auto ref_input = params.get_input<Geometry>("Reference");

    auto halfedge_mesh = operand_to_openmesh(&input);
    auto ref_mesh = ref_input.get_component<MeshComponent>()
                        ? operand_to_openmesh(&ref_input)
                        : operand_to_openmesh(&input);

    const int nV = static_cast<int>(halfedge_mesh->n_vertices());
    std::vector<char> is_boundary_v(nV, 0);

    for (auto he_it = halfedge_mesh->halfedges_begin();
         he_it != halfedge_mesh->halfedges_end();
         ++he_it) {
        auto heh = *he_it;
        if (halfedge_mesh->is_boundary(heh)) {
            is_boundary_v[halfedge_mesh->from_vertex_handle(heh).idx()] = 1;
            is_boundary_v[halfedge_mesh->to_vertex_handle(heh).idx()] = 1;
        }
    }

    std::vector<int> vid_to_row(nV, -1);
    std::vector<int> row_to_vid;
    row_to_vid.reserve(nV);

    for (auto v_it = halfedge_mesh->vertices_begin();
         v_it != halfedge_mesh->vertices_end();
         ++v_it) {
        int vid = (*v_it).idx();
        if (!is_boundary_v[vid]) {
            vid_to_row[vid] = static_cast<int>(row_to_vid.size());
            row_to_vid.push_back(vid);
        }
    }

    const int nI = static_cast<int>(row_to_vid.size());
    if (nI == 0) {
        auto geometry = openmesh_to_operand(halfedge_mesh.get());
        params.set_output("Output", std::move(*geometry));
        return true;
    }

    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(nI * 12);

    Eigen::VectorXd bx = Eigen::VectorXd::Zero(nI);
    Eigen::VectorXd by = Eigen::VectorXd::Zero(nI);

    for (int r = 0; r < nI; ++r) {
        int vid = row_to_vid[r];
        auto nbrs = ordered_neighbors(ref_mesh.get(), vid);
        if (nbrs.empty()) {
            return false;
        }

        auto lambda = uniform_weights(ref_mesh.get(), vid, nbrs);

        trips.emplace_back(r, r, 1.0);

        for (int k = 0; k < static_cast<int>(nbrs.size()); ++k) {
            int nb = nbrs[k];
            double wk = lambda[k];

            if (!is_boundary_v[nb]) {
                int c = vid_to_row[nb];
                trips.emplace_back(r, c, -wk);
            }
            else {
                const auto& p =
                    halfedge_mesh->point(halfedge_mesh->vertex_handle(nb));
                bx[r] += wk * static_cast<double>(p[0]);
                by[r] += wk * static_cast<double>(p[1]);
            }
        }
    }

    Eigen::SparseMatrix<double> A(nI, nI);
    A.setFromTriplets(trips.begin(), trips.end());
    A.makeCompressed();

    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.analyzePattern(A);
    solver.factorize(A);
    if (solver.info() != Eigen::Success) {
        return false;
    }

    Eigen::VectorXd solx = solver.solve(bx);
    if (solver.info() != Eigen::Success) {
        return false;
    }

    Eigen::VectorXd soly = solver.solve(by);
    if (solver.info() != Eigen::Success) {
        return false;
    }

    for (int r = 0; r < nI; ++r) {
        int vid = row_to_vid[r];
        auto vh = halfedge_mesh->vertex_handle(vid);
        halfedge_mesh->point(vh) = OpenMesh::Vec3f(
            static_cast<float>(solx[r]), static_cast<float>(soly[r]), 0.0f);
    }

    auto geometry = openmesh_to_operand(halfedge_mesh.get());
    params.set_output("Output", std::move(*geometry));
    return true;
}

NODE_DECLARATION_FUNCTION(hw5_param_cotangent)
{
    b.add_input<Geometry>("Input");
    b.add_input<Geometry>("Reference");
    b.add_output<Geometry>("Output");
}

NODE_EXECUTION_FUNCTION(hw5_param_cotangent)
{
    const int weight_mode = WEIGHT_COTAN;

    auto input = params.get_input<Geometry>("Input");
    if (!input.get_component<MeshComponent>()) {
        return false;
    }

    auto ref_input = params.get_input<Geometry>("Reference");

    auto halfedge_mesh = operand_to_openmesh(&input);
    auto ref_mesh = ref_input.get_component<MeshComponent>()
                        ? operand_to_openmesh(&ref_input)
                        : operand_to_openmesh(&input);

    const int nV = static_cast<int>(halfedge_mesh->n_vertices());
    std::vector<char> is_boundary_v(nV, 0);

    for (auto he_it = halfedge_mesh->halfedges_begin();
         he_it != halfedge_mesh->halfedges_end();
         ++he_it) {
        auto heh = *he_it;
        if (halfedge_mesh->is_boundary(heh)) {
            is_boundary_v[halfedge_mesh->from_vertex_handle(heh).idx()] = 1;
            is_boundary_v[halfedge_mesh->to_vertex_handle(heh).idx()] = 1;
        }
    }

    std::vector<int> vid_to_row(nV, -1);
    std::vector<int> row_to_vid;
    row_to_vid.reserve(nV);

    for (auto v_it = halfedge_mesh->vertices_begin();
         v_it != halfedge_mesh->vertices_end();
         ++v_it) {
        int vid = (*v_it).idx();
        if (!is_boundary_v[vid]) {
            vid_to_row[vid] = static_cast<int>(row_to_vid.size());
            row_to_vid.push_back(vid);
        }
    }

    const int nI = static_cast<int>(row_to_vid.size());
    if (nI == 0) {
        auto geometry = openmesh_to_operand(halfedge_mesh.get());
        params.set_output("Output", std::move(*geometry));
        return true;
    }

    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(nI * 12);

    Eigen::VectorXd bx = Eigen::VectorXd::Zero(nI);
    Eigen::VectorXd by = Eigen::VectorXd::Zero(nI);

    for (int r = 0; r < nI; ++r) {
        int vid = row_to_vid[r];
        auto nbrs = ordered_neighbors(ref_mesh.get(), vid);
        if (nbrs.empty()) {
            return false;
        }

        auto w = cotangent_weights_raw(ref_mesh.get(), vid, nbrs);

        double diag = 0.0;
        bool valid = false;
        for (double& wk : w) {
            if (!std::isfinite(wk)) {
                wk = 0.0;
            }
            diag += wk;
            if (std::abs(wk) > kEps) {
                valid = true;
            }
        }

        if (!valid || std::abs(diag) <= kEps) {
            auto lambda = uniform_weights(ref_mesh.get(), vid, nbrs);
            diag = 1.0;
            trips.emplace_back(r, r, diag);

            for (int k = 0; k < static_cast<int>(nbrs.size()); ++k) {
                int nb = nbrs[k];
                double wk = lambda[k];

                if (!is_boundary_v[nb]) {
                    int c = vid_to_row[nb];
                    trips.emplace_back(r, c, -wk);
                }
                else {
                    const auto& p =
                        halfedge_mesh->point(halfedge_mesh->vertex_handle(nb));
                    bx[r] += wk * static_cast<double>(p[0]);
                    by[r] += wk * static_cast<double>(p[1]);
                }
            }
        }
        else {
            trips.emplace_back(r, r, diag);

            for (int k = 0; k < static_cast<int>(nbrs.size()); ++k) {
                int nb = nbrs[k];
                double wk = w[k];

                if (!is_boundary_v[nb]) {
                    int c = vid_to_row[nb];
                    trips.emplace_back(r, c, -wk);
                }
                else {
                    const auto& p =
                        halfedge_mesh->point(halfedge_mesh->vertex_handle(nb));
                    bx[r] += wk * static_cast<double>(p[0]);
                    by[r] += wk * static_cast<double>(p[1]);
                }
            }
        }
    }

    Eigen::SparseMatrix<double> A(nI, nI);
    A.setFromTriplets(trips.begin(), trips.end());
    A.makeCompressed();

    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.analyzePattern(A);
    solver.factorize(A);
    if (solver.info() != Eigen::Success) {
        return false;
    }

    Eigen::VectorXd solx = solver.solve(bx);
    if (solver.info() != Eigen::Success) {
        return false;
    }

    Eigen::VectorXd soly = solver.solve(by);
    if (solver.info() != Eigen::Success) {
        return false;
    }

    for (int r = 0; r < nI; ++r) {
        int vid = row_to_vid[r];
        auto vh = halfedge_mesh->vertex_handle(vid);
        halfedge_mesh->point(vh) = OpenMesh::Vec3f(
            static_cast<float>(solx[r]), static_cast<float>(soly[r]), 0.0f);
    }

    auto geometry = openmesh_to_operand(halfedge_mesh.get());
    params.set_output("Output", std::move(*geometry));
    return true;
}

NODE_DECLARATION_FUNCTION(hw5_param_floater)
{
    b.add_input<Geometry>("Input");
    b.add_input<Geometry>("Reference");
    b.add_output<Geometry>("Output");
}

NODE_EXECUTION_FUNCTION(hw5_param_floater)
{
    const int weight_mode = WEIGHT_FLOATER;

    auto input = params.get_input<Geometry>("Input");
    if (!input.get_component<MeshComponent>()) {
        return false;
    }

    auto ref_input = params.get_input<Geometry>("Reference");

    auto halfedge_mesh = operand_to_openmesh(&input);
    auto ref_mesh = ref_input.get_component<MeshComponent>()
                        ? operand_to_openmesh(&ref_input)
                        : operand_to_openmesh(&input);

    const int nV = static_cast<int>(halfedge_mesh->n_vertices());
    std::vector<char> is_boundary_v(nV, 0);

    for (auto he_it = halfedge_mesh->halfedges_begin();
         he_it != halfedge_mesh->halfedges_end();
         ++he_it) {
        auto heh = *he_it;
        if (halfedge_mesh->is_boundary(heh)) {
            is_boundary_v[halfedge_mesh->from_vertex_handle(heh).idx()] = 1;
            is_boundary_v[halfedge_mesh->to_vertex_handle(heh).idx()] = 1;
        }
    }

    std::vector<int> vid_to_row(nV, -1);
    std::vector<int> row_to_vid;
    row_to_vid.reserve(nV);

    for (auto v_it = halfedge_mesh->vertices_begin();
         v_it != halfedge_mesh->vertices_end();
         ++v_it) {
        int vid = (*v_it).idx();
        if (!is_boundary_v[vid]) {
            vid_to_row[vid] = static_cast<int>(row_to_vid.size());
            row_to_vid.push_back(vid);
        }
    }

    const int nI = static_cast<int>(row_to_vid.size());
    if (nI == 0) {
        auto geometry = openmesh_to_operand(halfedge_mesh.get());
        params.set_output("Output", std::move(*geometry));
        return true;
    }

    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(nI * 12);

    Eigen::VectorXd bx = Eigen::VectorXd::Zero(nI);
    Eigen::VectorXd by = Eigen::VectorXd::Zero(nI);

    for (int r = 0; r < nI; ++r) {
        int vid = row_to_vid[r];
        auto nbrs = ordered_neighbors(ref_mesh.get(), vid);
        if (nbrs.empty()) {
            return false;
        }

        auto lambda = floater_weights(ref_mesh.get(), vid, nbrs);

        trips.emplace_back(r, r, 1.0);

        for (int k = 0; k < static_cast<int>(nbrs.size()); ++k) {
            int nb = nbrs[k];
            double wk = lambda[k];

            if (!is_boundary_v[nb]) {
                int c = vid_to_row[nb];
                trips.emplace_back(r, c, -wk);
            }
            else {
                const auto& p =
                    halfedge_mesh->point(halfedge_mesh->vertex_handle(nb));
                bx[r] += wk * static_cast<double>(p[0]);
                by[r] += wk * static_cast<double>(p[1]);
            }
        }
    }

    Eigen::SparseMatrix<double> A(nI, nI);
    A.setFromTriplets(trips.begin(), trips.end());
    A.makeCompressed();

    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.analyzePattern(A);
    solver.factorize(A);
    if (solver.info() != Eigen::Success) {
        return false;
    }

    Eigen::VectorXd solx = solver.solve(bx);
    if (solver.info() != Eigen::Success) {
        return false;
    }

    Eigen::VectorXd soly = solver.solve(by);
    if (solver.info() != Eigen::Success) {
        return false;
    }

    for (int r = 0; r < nI; ++r) {
        int vid = row_to_vid[r];
        auto vh = halfedge_mesh->vertex_handle(vid);
        halfedge_mesh->point(vh) = OpenMesh::Vec3f(
            static_cast<float>(solx[r]), static_cast<float>(soly[r]), 0.0f);
    }

    auto geometry = openmesh_to_operand(halfedge_mesh.get());
    params.set_output("Output", std::move(*geometry));
    return true;
}

NODE_DECLARATION_FUNCTION(hw5_param_mvc)
{
    b.add_input<Geometry>("Input");
    b.add_input<Geometry>("Reference");
    b.add_output<Geometry>("Output");
}

NODE_EXECUTION_FUNCTION(hw5_param_mvc)
{
    const int weight_mode = WEIGHT_MVC;

    auto input = params.get_input<Geometry>("Input");
    if (!input.get_component<MeshComponent>()) {
        return false;
    }

    auto ref_input = params.get_input<Geometry>("Reference");

    auto halfedge_mesh = operand_to_openmesh(&input);
    auto ref_mesh = ref_input.get_component<MeshComponent>()
                        ? operand_to_openmesh(&ref_input)
                        : operand_to_openmesh(&input);

    const int nV = static_cast<int>(halfedge_mesh->n_vertices());
    std::vector<char> is_boundary_v(nV, 0);

    for (auto he_it = halfedge_mesh->halfedges_begin();
         he_it != halfedge_mesh->halfedges_end();
         ++he_it) {
        auto heh = *he_it;
        if (halfedge_mesh->is_boundary(heh)) {
            is_boundary_v[halfedge_mesh->from_vertex_handle(heh).idx()] = 1;
            is_boundary_v[halfedge_mesh->to_vertex_handle(heh).idx()] = 1;
        }
    }

    std::vector<int> vid_to_row(nV, -1);
    std::vector<int> row_to_vid;
    row_to_vid.reserve(nV);

    for (auto v_it = halfedge_mesh->vertices_begin();
         v_it != halfedge_mesh->vertices_end();
         ++v_it) {
        int vid = (*v_it).idx();
        if (!is_boundary_v[vid]) {
            vid_to_row[vid] = static_cast<int>(row_to_vid.size());
            row_to_vid.push_back(vid);
        }
    }

    const int nI = static_cast<int>(row_to_vid.size());
    if (nI == 0) {
        auto geometry = openmesh_to_operand(halfedge_mesh.get());
        params.set_output("Output", std::move(*geometry));
        return true;
    }

    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(nI * 12);

    Eigen::VectorXd bx = Eigen::VectorXd::Zero(nI);
    Eigen::VectorXd by = Eigen::VectorXd::Zero(nI);

    for (int r = 0; r < nI; ++r) {
        int vid = row_to_vid[r];
        auto nbrs = ordered_neighbors(ref_mesh.get(), vid);
        if (nbrs.empty()) {
            return false;
        }

        auto lambda = mvc_weights(ref_mesh.get(), vid, nbrs);

        trips.emplace_back(r, r, 1.0);

        for (int k = 0; k < static_cast<int>(nbrs.size()); ++k) {
            int nb = nbrs[k];
            double wk = lambda[k];

            if (!is_boundary_v[nb]) {
                int c = vid_to_row[nb];
                trips.emplace_back(r, c, -wk);
            }
            else {
                const auto& p =
                    halfedge_mesh->point(halfedge_mesh->vertex_handle(nb));
                bx[r] += wk * static_cast<double>(p[0]);
                by[r] += wk * static_cast<double>(p[1]);
            }
        }
    }

    Eigen::SparseMatrix<double> A(nI, nI);
    A.setFromTriplets(trips.begin(), trips.end());
    A.makeCompressed();

    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.analyzePattern(A);
    solver.factorize(A);
    if (solver.info() != Eigen::Success) {
        return false;
    }

    Eigen::VectorXd solx = solver.solve(bx);
    if (solver.info() != Eigen::Success) {
        return false;
    }

    Eigen::VectorXd soly = solver.solve(by);
    if (solver.info() != Eigen::Success) {
        return false;
    }

    for (int r = 0; r < nI; ++r) {
        int vid = row_to_vid[r];
        auto vh = halfedge_mesh->vertex_handle(vid);
        halfedge_mesh->point(vh) = OpenMesh::Vec3f(
            static_cast<float>(solx[r]), static_cast<float>(soly[r]), 0.0f);
    }

    auto geometry = openmesh_to_operand(halfedge_mesh.get());
    params.set_output("Output", std::move(*geometry));
    return true;
}

NODE_DECLARATION_UI(hw5_param);
NODE_DECLARATION_UI(hw5_param_cotangent);
NODE_DECLARATION_UI(hw5_param_floater);
NODE_DECLARATION_UI(hw5_param_mvc);

NODE_DEF_CLOSE_SCOPE