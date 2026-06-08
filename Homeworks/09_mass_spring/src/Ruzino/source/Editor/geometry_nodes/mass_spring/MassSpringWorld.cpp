#include "MassSpringWorld.h"

#include <algorithm>
#include <Eigen/Sparse>
#include <cmath>
#include <iostream>
#include <limits>

namespace USTC_CG::mass_spring {

namespace {
// Self-collision is deliberately softer than inter-object IPC.
// It uses predictive contact planes plus a few relaxed PBD projections;
// residual self-intersections are tolerated instead of rejecting the whole step.
constexpr int kSelfPBDProjectionIters = 3;
constexpr int kSelfPBDRepairIters = 1;
constexpr double kSelfPBDRelaxation = 0.35;
constexpr double kSelfPBDRepairRelaxation = 0.25;
constexpr double kSelfPredictiveActivationScale = 1.0; // create contact if gap < (1+scale)*dhat
constexpr double kSelfCollisionRestitution = 0.0;      // cloth self-collision should not bounce

std::vector<double> compute_lumped_vertex_areas(
    const Eigen::MatrixXd& X,
    const std::vector<Eigen::Vector3i>& F)
{
    std::vector<double> areas(static_cast<size_t>(std::max(0, static_cast<int>(X.rows()))), 0.0);
    for (const auto& f : F) {
        if (f[0] < 0 || f[1] < 0 || f[2] < 0) continue;
        if (f[0] >= X.rows() || f[1] >= X.rows() || f[2] >= X.rows()) continue;
        const Eigen::Vector3d a = X.row(f[0]).transpose();
        const Eigen::Vector3d b = X.row(f[1]).transpose();
        const Eigen::Vector3d c = X.row(f[2]).transpose();
        const double area = 0.5 * (b - a).cross(c - a).norm();
        if (area <= 0.0 || !std::isfinite(area)) continue;
        const double share = area / 3.0;
        areas[static_cast<size_t>(f[0])] += share;
        areas[static_cast<size_t>(f[1])] += share;
        areas[static_cast<size_t>(f[2])] += share;
    }
    return areas;
}
}

int MassSpringWorld::addObject(
    const MatrixXd& X,
    const std::vector<Vector3i>& F,
    const EdgeSet& E,
    int object_type,
    const std::string& name)
{
    Object obj;
    obj.name = name;
    obj.object_type = object_type;
    obj.object_id = static_cast<int>(objects.size());
    obj.X = X;
    obj.X0 = X;
    obj.V = MatrixXd::Zero(X.rows(), X.cols());
    obj.F = F;
    obj.E = E;
    obj.fixed_mask.assign(X.rows(), false);

    obj.vertex_area = compute_lumped_vertex_areas(X, F);

    obj.E_rest_length.reserve(E.size());
    for (const auto& e : E) {
        Eigen::Vector3d x0 = X.row(e.first).transpose();
        Eigen::Vector3d x1 = X.row(e.second).transpose();
        obj.E_rest_length.push_back((x0 - x1).norm());
    }

    objects.push_back(std::move(obj));
    return static_cast<int>(objects.size()) - 1;
}

void MassSpringWorld::clear()
{
    objects.clear();
    step_snapshots.clear();
    step_recording_time = 0.0;
}

void MassSpringWorld::beginStepRecording(double start_time)
{
    step_recording_enabled = true;
    step_recording_time = start_time;
    step_snapshots.clear();
}

void MassSpringWorld::disableStepRecording()
{
    step_recording_enabled = false;
    step_snapshots.clear();
}

void MassSpringWorld::recordAcceptedStep(
    const SimState& state,
    double dt,
    int substep_index,
    int retry_count,
    int self_contact_count) const
{
    if (!step_recording_enabled) return;
    step_recording_time += std::max(0.0, dt);

    StepSnapshot s;
    s.time = step_recording_time;
    s.dt = std::max(0.0, dt);
    s.substep_index = substep_index;
    s.retry_count = retry_count;
    s.self_contact_count = self_contact_count;
    s.post_response_applied = false;
    s.X = state.X;
    s.V = state.V;
    step_snapshots.push_back(std::move(s));
}

void MassSpringWorld::overwriteLastRecordedState(
    const SimState& state,
    bool post_response_applied) const
{
    if (!step_recording_enabled || step_snapshots.empty()) return;
    StepSnapshot& s = step_snapshots.back();
    s.X = state.X;
    s.V = state.V;
    s.post_response_applied = post_response_applied;
}

MassSpringWorld::SimState MassSpringWorld::captureState() const
{
    SimState s;
    s.X.reserve(objects.size());
    s.V.reserve(objects.size());
    for (const auto& obj : objects) {
        s.X.push_back(obj.X);
        s.V.push_back(obj.V);
    }
    return s;
}

void MassSpringWorld::applyState(const SimState& state)
{
    const int n = std::min(static_cast<int>(objects.size()), static_cast<int>(state.X.size()));
    for (int i = 0; i < n; ++i) {
        objects[i].X = state.X[i];
        objects[i].V = state.V[i];
    }
}

MassSpringWorld::SimState MassSpringWorld::interpolateState(
    const SimState& a,
    const SimState& b,
    double alpha) const
{
    SimState out;
    out.X.resize(objects.size());
    out.V.resize(objects.size());
    for (int i = 0; i < static_cast<int>(objects.size()); ++i) {
        out.X[i] = a.X[i] + alpha * (b.X[i] - a.X[i]);
        out.V[i] = a.V[i] + alpha * (b.V[i] - a.V[i]);
    }
    return out;
}

double MassSpringWorld::computeSpringEnergy(
    const Object& obj,
    const MatrixXd& X_eval) const
{
    double energy = 0.0;
    unsigned edge_id = 0;
    for (const auto& e : obj.E) {
        const int v0 = e.first;
        const int v1 = e.second;
        const Eigen::RowVector3d diff = X_eval.row(v0) - X_eval.row(v1);
        const double len = diff.norm();
        const double rest_len = obj.E_rest_length[edge_id++];
        energy += 0.5 * obj.stiffness * (len - rest_len) * (len - rest_len);
    }
    return energy;
}

Eigen::MatrixXd MassSpringWorld::computeSpringGradient(
    const Object& obj,
    const MatrixXd& X_eval) const
{
    MatrixXd grad = MatrixXd::Zero(X_eval.rows(), X_eval.cols());
    unsigned edge_id = 0;
    for (const auto& e : obj.E) {
        int v0 = e.first;
        int v1 = e.second;
        Eigen::RowVector3d diff = X_eval.row(v0) - X_eval.row(v1);
        double len = diff.norm();
        double rest_len = obj.E_rest_length[edge_id++];
        if (len > 1e-12) {
            Eigen::RowVector3d g = obj.stiffness * (len - rest_len) * diff / len;
            grad.row(v0) += g;
            grad.row(v1) -= g;
        }
    }
    return grad;
}

Eigen::SparseMatrix<double> MassSpringWorld::computeSpringHessianSparse(
    const Object& obj,
    const MatrixXd& X_eval) const
{
    const int n_vertices = static_cast<int>(X_eval.rows());
    Eigen::SparseMatrix<double> H(n_vertices * 3, n_vertices * 3);
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(obj.E.size() * 4 * 9);

    const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    unsigned edge_id = 0;

    auto is_fixed = [&](int v) {
        return v >= 0 && v < static_cast<int>(obj.fixed_mask.size()) && obj.fixed_mask[v];
    };

    auto add_block = [&](int row_v, int col_v, const Eigen::Matrix3d& block) {
        if (is_fixed(row_v) || is_fixed(col_v)) return;
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
                triplets.emplace_back(3 * row_v + a, 3 * col_v + b, block(a, b));
            }
        }
    };

    for (const auto& e : obj.E) {
        const int v0 = e.first;
        const int v1 = e.second;
        const Eigen::Vector3d x0 = X_eval.row(v0).transpose();
        const Eigen::Vector3d x1 = X_eval.row(v1).transpose();
        const Eigen::Vector3d diff = x0 - x1;
        const double len = diff.norm();
        const double rest_len = obj.E_rest_length[edge_id++];

        if (len <= 1e-12) continue;

        Eigen::Matrix3d H_local = obj.stiffness *
            ((1.0 - rest_len / len) * I +
             (rest_len / (len * len * len)) * (diff * diff.transpose()));

        // The exact spring Hessian can be indefinite under compression.  For an
        // implicit predictor we use the usual PSD projection so LDLT is stable.
        if (enable_make_SPD) {
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(H_local);
            Eigen::Vector3d evals = es.eigenvalues();
            for (int k = 0; k < 3; ++k) {
                if (evals[k] < 0.0) evals[k] = 0.0;
            }
            H_local = es.eigenvectors() * evals.asDiagonal() * es.eigenvectors().transpose();
        }

        add_block(v0, v0, H_local);
        add_block(v0, v1, -H_local);
        add_block(v1, v0, -H_local);
        add_block(v1, v1, H_local);
    }

    H.setFromTriplets(triplets.begin(), triplets.end());
    H.makeCompressed();
    return H;
}

MassSpringWorld::MatrixXd MassSpringWorld::semiImplicitElasticStepObject(
    const Object& obj,
    const MatrixXd& X_old,
    const MatrixXd& V_old,
    double dt,
    MatrixXd& V_new) const
{
    const int n = static_cast<int>(X_old.rows());
    const double mass_per_vertex = obj.mass / std::max(1, n);

    MatrixXd grad = computeSpringGradient(obj, X_old);
    MatrixXd acc = -grad / mass_per_vertex;
    acc.rowwise() += (gravity + wind_ext_acc).transpose();

    V_new = V_old + dt * acc;
    if (obj.damping >= 0.0 && obj.damping < 1.0) {
        V_new *= obj.damping;
    }

    MatrixXd X_new = X_old + dt * V_new;

    for (int i = 0; i < n; ++i) {
        if (i < static_cast<int>(obj.fixed_mask.size()) && obj.fixed_mask[i]) {
            X_new.row(i) = obj.X0.row(i);
            V_new.row(i).setZero();
        }
    }
    return X_new;
}

MassSpringWorld::MatrixXd MassSpringWorld::implicitElasticStepObject(
    const Object& obj,
    const MatrixXd& X_old,
    const MatrixXd& V_old,
    double dt,
    MatrixXd& V_new) const
{
    const int n = static_cast<int>(X_old.rows());
    if (n == 0 || dt <= 0.0) {
        V_new = V_old;
        return X_old;
    }

    const double mass_per_vertex = obj.mass / std::max(1, n);
    const Eigen::Vector3d acceleration_ext = gravity + wind_ext_acc;

    MatrixXd V_predict = V_old;
    if (obj.damping >= 0.0 && obj.damping < 1.0) {
        V_predict *= obj.damping;
    }

    // Implicit Euler can be written as minimization of the incremental potential:
    //   1/2 * m / h^2 ||x - y||^2 + E_spring(x),
    //   y = x_n + h v_n + h^2 a_ext.
    MatrixXd Y = X_old + dt * V_predict;
    Y.rowwise() += (dt * dt * acceleration_ext).transpose();

    auto flatten = [](const MatrixXd& M) {
        Eigen::VectorXd v(M.rows() * M.cols());
        for (int i = 0; i < M.rows(); ++i) {
            for (int j = 0; j < M.cols(); ++j) {
                v[3 * i + j] = M(i, j);
            }
        }
        return v;
    };

    auto unflatten = [](const Eigen::VectorXd& v, int rows) {
        MatrixXd M(rows, 3);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < 3; ++j) {
                M(i, j) = v[3 * i + j];
            }
        }
        return M;
    };

    auto is_fixed = [&](int v) {
        return v >= 0 && v < static_cast<int>(obj.fixed_mask.size()) && obj.fixed_mask[v];
    };

    auto objective = [&](const MatrixXd& X_candidate) {
        const double inertial = 0.5 * mass_per_vertex / (dt * dt) *
                                (X_candidate - Y).squaredNorm();
        return inertial + computeSpringEnergy(obj, X_candidate);
    };

    MatrixXd X = X_old;
    for (int i = 0; i < n; ++i) {
        if (is_fixed(i)) X.row(i) = obj.X0.row(i);
    }

    for (int iter = 0; iter < implicit_max_newton_iters; ++iter) {
        MatrixXd grad = mass_per_vertex / (dt * dt) * (X - Y) +
                        computeSpringGradient(obj, X);

        for (int i = 0; i < n; ++i) {
            if (is_fixed(i)) grad.row(i).setZero();
        }

        Eigen::VectorXd grad_vec = flatten(grad);
        if (grad_vec.norm() < implicit_grad_tol) {
            break;
        }

        Eigen::SparseMatrix<double> A(n * 3, n * 3);
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(n * 3);
        for (int i = 0; i < n; ++i) {
            for (int d = 0; d < 3; ++d) {
                const int idx = 3 * i + d;
                if (is_fixed(i)) {
                    triplets.emplace_back(idx, idx, 1.0);
                }
                else {
                    triplets.emplace_back(idx, idx, mass_per_vertex / (dt * dt));
                }
            }
        }
        A.setFromTriplets(triplets.begin(), triplets.end());
        A += computeSpringHessianSparse(obj, X);
        A.makeCompressed();

        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
        solver.compute(A);
        if (solver.info() != Eigen::Success) {
            if (enable_debug_output) {
                std::cerr << "[MassSpringWorld] implicit elastic LDLT failed for object "
                          << obj.object_id << "; fallback to semi-implicit."
                          << std::endl;
            }
            return semiImplicitElasticStepObject(obj, X_old, V_old, dt, V_new);
        }

        Eigen::VectorXd dx = solver.solve(-grad_vec);
        if (solver.info() != Eigen::Success) {
            if (enable_debug_output) {
                std::cerr << "[MassSpringWorld] implicit elastic solve failed for object "
                          << obj.object_id << "; fallback to semi-implicit."
                          << std::endl;
            }
            return semiImplicitElasticStepObject(obj, X_old, V_old, dt, V_new);
        }

        MatrixXd dX = unflatten(dx, n);
        double alpha = 1.0;
        const double E0 = objective(X);
        const double descent = grad_vec.dot(dx);
        MatrixXd X_trial = X;

        while (alpha > 1e-4) {
            X_trial = X + alpha * dX;
            for (int i = 0; i < n; ++i) {
                if (is_fixed(i)) X_trial.row(i) = obj.X0.row(i);
            }

            const double E_trial = objective(X_trial);
            if (E_trial <= E0 + 1e-4 * alpha * descent) {
                break;
            }
            alpha *= 0.5;
        }

        X = X_trial;
        if ((alpha * dX).norm() < implicit_step_tol) {
            break;
        }
    }

    V_new = (X - X_old) / dt;
    for (int i = 0; i < n; ++i) {
        if (is_fixed(i)) {
            X.row(i) = obj.X0.row(i);
            V_new.row(i).setZero();
        }
    }
    return X;
}

MassSpringWorld::SimState MassSpringWorld::predictState(
    const SimState& start,
    double dt) const
{
    SimState out = start;

    for (int oid = 0; oid < static_cast<int>(objects.size()); ++oid) {
        const Object& obj = objects[oid];

        if (obj.object_type == STATIC_COLLIDER) {
            out.X[oid] = start.X[oid];
            out.V[oid].setZero();
            continue;
        }

        if (obj.object_type == KINEMATIC_MESH) {
            out.V[oid] = start.V[oid];
            out.X[oid] = start.X[oid] + dt * start.V[oid];
            continue;
        }

        MatrixXd V_new;
        out.X[oid] = implicitElasticStepObject(obj, start.X[oid], start.V[oid], dt, V_new);
        out.V[oid] = V_new;
    }
    return out;
}

void MassSpringWorld::applyGroundResponse(SimState& state) const
{
    if (!enable_ground) return;

    for (int oid = 0; oid < static_cast<int>(objects.size()); ++oid) {
        const Object& obj = objects[oid];
        if (!obj.isDeformable()) continue;

        const double m = obj.mass / std::max(1, static_cast<int>(state.X[oid].rows()));
        const double e = ground_restitution;
        const double mu = ground_friction;

        for (int i = 0; i < state.X[oid].rows(); ++i) {
            if (i < static_cast<int>(obj.fixed_mask.size()) && obj.fixed_mask[i]) continue;

            if (state.X[oid](i, 2) < ground_z) {
                state.X[oid](i, 2) = ground_z;
                double old_vz = state.V[oid](i, 2);
                if (old_vz < 0.0) {
                    state.V[oid](i, 2) = -e * old_vz;

                    Eigen::Vector2d vt(state.V[oid](i, 0), state.V[oid](i, 1));
                    double vt_norm = vt.norm();
                    if (vt_norm > 1e-12) {
                        double normal_impulse = m * (1.0 + e) * (-old_vz);
                        double tangent_needed = m * vt_norm;
                        double scale = std::max(0.0, 1.0 - (mu * normal_impulse) / tangent_needed);
                        state.V[oid](i, 0) *= scale;
                        state.V[oid](i, 1) *= scale;
                    }
                }
            }
        }
    }
}

double MassSpringWorld::invMass(int object_id, int vertex_id) const
{
    const Object& obj = objects[object_id];
    if (!obj.isDeformable()) return 0.0;
    if (vertex_id >= 0 && vertex_id < static_cast<int>(obj.fixed_mask.size()) && obj.fixed_mask[vertex_id]) {
        return 0.0;
    }
    return static_cast<double>(obj.X.rows()) / std::max(1e-12, obj.mass);
}

MassSpringWorld::Vector3d MassSpringWorld::getVelocityFromState(
    const SimState& state,
    int object_id,
    int vertex_id) const
{
    return state.V[object_id].row(vertex_id).transpose();
}

void MassSpringWorld::addVelocityToState(
    SimState& state,
    int object_id,
    int vertex_id,
    const Vector3d& dv) const
{
    if (invMass(object_id, vertex_id) == 0.0) return;
    state.V[object_id].row(vertex_id) += dv.transpose();
}

double MassSpringWorld::computeAutomaticCellSize(const SimState& state) const
{
    double sum = 0.0;
    int cnt = 0;
    for (int oid = 0; oid < static_cast<int>(objects.size()); ++oid) {
        const Object& obj = objects[oid];
        for (const auto& e : obj.E) {
            Eigen::Vector3d a = state.X[oid].row(e.first).transpose();
            Eigen::Vector3d b = state.X[oid].row(e.second).transpose();
            double len = (a - b).norm();
            if (len > 1e-12) {
                sum += len;
                ++cnt;
            }
        }
    }
    if (cnt == 0 || sum <= 0.0) return 0.1;
    return std::max(1e-4, 2.0 * sum / cnt);
}

MassSpringWorld::AABB MassSpringWorld::computeTriangleAABB(
    const MatrixXd& X_eval,
    const Vector3i& f,
    double margin) const
{
    Vector3d p0 = X_eval.row(f[0]).transpose();
    Vector3d p1 = X_eval.row(f[1]).transpose();
    Vector3d p2 = X_eval.row(f[2]).transpose();
    AABB box;
    box.mn = p0.cwiseMin(p1).cwiseMin(p2);
    box.mx = p0.cwiseMax(p1).cwiseMax(p2);
    box.mn.array() -= margin;
    box.mx.array() += margin;
    return box;
}

bool MassSpringWorld::overlapAABB(const AABB& a, const AABB& b) const
{
    if (a.mx.x() < b.mn.x() || b.mx.x() < a.mn.x()) return false;
    if (a.mx.y() < b.mn.y() || b.mx.y() < a.mn.y()) return false;
    if (a.mx.z() < b.mn.z() || b.mx.z() < a.mn.z()) return false;
    return true;
}

MassSpringWorld::CellIndex MassSpringWorld::pointToCell(const Vector3d& p, double cell_size) const
{
    return CellIndex{
        static_cast<int>(std::floor(p.x() / cell_size)),
        static_cast<int>(std::floor(p.y() / cell_size)),
        static_cast<int>(std::floor(p.z() / cell_size))};
}

std::vector<MassSpringWorld::FaceRef> MassSpringWorld::collectFaces(const SimState& state, double margin) const
{
    std::vector<FaceRef> faces;
    for (int oid = 0; oid < static_cast<int>(objects.size()); ++oid) {
        const Object& obj = objects[oid];
        for (int fid = 0; fid < static_cast<int>(obj.F.size()); ++fid) {
            FaceRef ref;
            ref.object_id = oid;
            ref.face_id = fid;
            ref.v = obj.F[fid];
            ref.box = computeTriangleAABB(state.X[oid], ref.v, margin);
            faces.push_back(ref);
        }
    }
    return faces;
}

void MassSpringWorld::buildSpatialHash(
    const std::vector<FaceRef>& faces,
    double cell_size,
    HashGrid& grid) const
{
    grid.clear();
    for (int fid = 0; fid < static_cast<int>(faces.size()); ++fid) {
        CellIndex cmin = pointToCell(faces[fid].box.mn, cell_size);
        CellIndex cmax = pointToCell(faces[fid].box.mx, cell_size);
        for (int i = cmin.x; i <= cmax.x; ++i) {
            for (int j = cmin.y; j <= cmax.y; ++j) {
                for (int k = cmin.z; k <= cmax.z; ++k) {
                    grid[CellIndex{i, j, k}].push_back(fid);
                }
            }
        }
    }
}

bool MassSpringWorld::shareVertex(const Vector3i& a, const Vector3i& b) const
{
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (a[i] == b[j]) return true;
        }
    }
    return false;
}

bool MassSpringWorld::shouldSkipPair(const FaceRef& a, const FaceRef& b) const
{
    const Object& A = objects[a.object_id];
    const Object& B = objects[b.object_id];

    // Two collider-only objects do not need response/detection here.
    if (!A.isDeformable() && !B.isDeformable()) return true;

    // Adjacent faces in the same mesh are not self-collision candidates.
    if (a.object_id == b.object_id && shareVertex(a.v, b.v)) return true;
    return false;
}

std::uint64_t MassSpringWorld::pairKey(int a, int b) const
{
    if (a > b) std::swap(a, b);
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(a)) << 32) |
           static_cast<std::uint32_t>(b);
}

bool MassSpringWorld::segmentTriangleIntersection(
    const Vector3d& p0,
    const Vector3d& p1,
    const Vector3d& a,
    const Vector3d& b,
    const Vector3d& c,
    double& seg_t,
    Vector3d& bary,
    Vector3d& point) const
{
    const double eps = 1e-10;
    Vector3d dir = p1 - p0;
    Vector3d n = (b - a).cross(c - a);
    double n_norm = n.norm();
    if (n_norm < eps) return false;

    double denom = n.dot(dir);
    if (std::abs(denom) < eps) return false;

    double t = n.dot(a - p0) / denom;
    if (t < -eps || t > 1.0 + eps) return false;

    Vector3d q = p0 + t * dir;

    Vector3d v0 = b - a;
    Vector3d v1 = c - a;
    Vector3d v2 = q - a;
    double d00 = v0.dot(v0);
    double d01 = v0.dot(v1);
    double d11 = v1.dot(v1);
    double d20 = v2.dot(v0);
    double d21 = v2.dot(v1);
    double denom_bary = d00 * d11 - d01 * d01;
    if (std::abs(denom_bary) < eps) return false;
    double v = (d11 * d20 - d01 * d21) / denom_bary;
    double w = (d00 * d21 - d01 * d20) / denom_bary;
    double u = 1.0 - v - w;

    if (u < -eps || v < -eps || w < -eps) return false;

    seg_t = std::clamp(t, 0.0, 1.0);
    bary = Vector3d(u, v, w);
    point = q;
    return true;
}


bool MassSpringWorld::closestPointTriangle(
    const Vector3d& p,
    const Vector3d& a,
    const Vector3d& b,
    const Vector3d& c,
    Vector3d& closest,
    Vector3d& bary) const
{
    // Ericson, Real-Time Collision Detection, closest point on triangle.
    const Vector3d ab = b - a;
    const Vector3d ac = c - a;
    const Vector3d ap = p - a;
    const double d1 = ab.dot(ap);
    const double d2 = ac.dot(ap);
    if (d1 <= 0.0 && d2 <= 0.0) {
        closest = a;
        bary = Vector3d(1.0, 0.0, 0.0);
        return true;
    }

    const Vector3d bp = p - b;
    const double d3 = ab.dot(bp);
    const double d4 = ac.dot(bp);
    if (d3 >= 0.0 && d4 <= d3) {
        closest = b;
        bary = Vector3d(0.0, 1.0, 0.0);
        return true;
    }

    const double vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        const double v = d1 / (d1 - d3);
        closest = a + v * ab;
        bary = Vector3d(1.0 - v, v, 0.0);
        return true;
    }

    const Vector3d cp = p - c;
    const double d5 = ab.dot(cp);
    const double d6 = ac.dot(cp);
    if (d6 >= 0.0 && d5 <= d6) {
        closest = c;
        bary = Vector3d(0.0, 0.0, 1.0);
        return true;
    }

    const double vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        const double w = d2 / (d2 - d6);
        closest = a + w * ac;
        bary = Vector3d(1.0 - w, 0.0, w);
        return true;
    }

    const double va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        const double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        closest = b + w * (c - b);
        bary = Vector3d(0.0, 1.0 - w, w);
        return true;
    }

    const double denom = va + vb + vc;
    if (std::abs(denom) < 1e-12) return false;
    const double v = vb / denom;
    const double w = vc / denom;
    const double u = 1.0 - v - w;
    closest = u * a + v * b + w * c;
    bary = Vector3d(u, v, w);
    return true;
}

bool MassSpringWorld::closestSegmentSegment(
    const Vector3d& p0,
    const Vector3d& p1,
    const Vector3d& q0,
    const Vector3d& q1,
    double& s,
    double& t,
    Vector3d& cp,
    Vector3d& cq) const
{
    const double eps = 1e-12;
    const Vector3d d1 = p1 - p0;
    const Vector3d d2 = q1 - q0;
    const Vector3d r = p0 - q0;
    const double a = d1.dot(d1);
    const double e = d2.dot(d2);
    const double f = d2.dot(r);

    if (a <= eps && e <= eps) {
        s = 0.0;
        t = 0.0;
        cp = p0;
        cq = q0;
        return false;
    }

    if (a <= eps) {
        s = 0.0;
        t = std::clamp(f / e, 0.0, 1.0);
    }
    else {
        const double c = d1.dot(r);
        if (e <= eps) {
            t = 0.0;
            s = std::clamp(-c / a, 0.0, 1.0);
        }
        else {
            const double b = d1.dot(d2);
            const double denom = a * e - b * b;
            if (std::abs(denom) > eps) {
                s = std::clamp((b * f - c * e) / denom, 0.0, 1.0);
            }
            else {
                s = 0.0;
            }

            t = (b * s + f) / e;
            if (t < 0.0) {
                t = 0.0;
                s = std::clamp(-c / a, 0.0, 1.0);
            }
            else if (t > 1.0) {
                t = 1.0;
                s = std::clamp((b - c) / a, 0.0, 1.0);
            }
        }
    }

    cp = p0 + s * d1;
    cq = q0 + t * d2;
    return true;
}

bool MassSpringWorld::edgeTriangleContact(
    const FaceRef& edge_face,
    int edge_local_a,
    int edge_local_b,
    const FaceRef& tri_face,
    const SimState& state,
    Contact& contact) const
{
    const Object& EObj = objects[edge_face.object_id];
    const Object& TObj = objects[tri_face.object_id];

    const int ea = edge_face.v[edge_local_a];
    const int eb = edge_face.v[edge_local_b];
    const Vector3d p0 = state.X[edge_face.object_id].row(ea).transpose();
    const Vector3d p1 = state.X[edge_face.object_id].row(eb).transpose();

    const int ta = tri_face.v[0];
    const int tb = tri_face.v[1];
    const int tc = tri_face.v[2];
    const Vector3d a = state.X[tri_face.object_id].row(ta).transpose();
    const Vector3d b = state.X[tri_face.object_id].row(tb).transpose();
    const Vector3d c = state.X[tri_face.object_id].row(tc).transpose();

    double seg_t = 0.0;
    Vector3d bary = Vector3d::Zero();
    Vector3d point = Vector3d::Zero();
    if (!segmentTriangleIntersection(p0, p1, a, b, c, seg_t, bary, point)) {
        return false;
    }

    Vector3d n = (b - a).cross(c - a);
    if (n.norm() < 1e-12) return false;
    n.normalize();

    contact.refs.clear();
    // Edge point minus triangle point: (1-t)p0 + t p1 - u a - v b - w c.
    contact.refs.push_back(VertexRef{edge_face.object_id, ea, 1.0 - seg_t});
    contact.refs.push_back(VertexRef{edge_face.object_id, eb, seg_t});
    contact.refs.push_back(VertexRef{tri_face.object_id, ta, -bary[0]});
    contact.refs.push_back(VertexRef{tri_face.object_id, tb, -bary[1]});
    contact.refs.push_back(VertexRef{tri_face.object_id, tc, -bary[2]});
    contact.point = point;
    contact.normal = n;
    contact.target_distance = 0.0;
    contact.is_intersection = true;
    contact.restitution = std::min(EObj.restitution, TObj.restitution);
    contact.friction = 0.5 * (EObj.friction + TObj.friction);
    return true;
}

bool MassSpringWorld::vertexTriangleContact(
    const FaceRef& vertex_face,
    int vertex_local,
    const FaceRef& tri_face,
    const SimState& state,
    Contact& contact) const
{
    const Object& VObj = objects[vertex_face.object_id];
    const Object& TObj = objects[tri_face.object_id];

    const int vp = vertex_face.v[vertex_local];
    const Vector3d p = state.X[vertex_face.object_id].row(vp).transpose();

    const int ta = tri_face.v[0];
    const int tb = tri_face.v[1];
    const int tc = tri_face.v[2];
    const Vector3d a = state.X[tri_face.object_id].row(ta).transpose();
    const Vector3d b = state.X[tri_face.object_id].row(tb).transpose();
    const Vector3d c = state.X[tri_face.object_id].row(tc).transpose();

    Vector3d q = Vector3d::Zero();
    Vector3d bary = Vector3d::Zero();
    if (!closestPointTriangle(p, a, b, c, q, bary)) return false;

    Vector3d diff = p - q;
    double d = diff.norm();
    if (d >= contact_thickness) return false;

    Vector3d n;
    if (d > 1e-12) {
        n = diff / d;
    }
    else {
        n = (b - a).cross(c - a);
        if (n.norm() < 1e-12) return false;
        n.normalize();
    }

    contact.refs.clear();
    contact.refs.push_back(VertexRef{vertex_face.object_id, vp, 1.0});
    contact.refs.push_back(VertexRef{tri_face.object_id, ta, -bary[0]});
    contact.refs.push_back(VertexRef{tri_face.object_id, tb, -bary[1]});
    contact.refs.push_back(VertexRef{tri_face.object_id, tc, -bary[2]});
    contact.point = q;
    contact.normal = n;
    contact.target_distance = 0.0;
    contact.is_intersection = false;
    contact.restitution = std::min(VObj.restitution, TObj.restitution);
    contact.friction = 0.5 * (VObj.friction + TObj.friction);
    return true;
}

bool MassSpringWorld::edgeEdgeContact(
    const FaceRef& face_a,
    int a0_local,
    int a1_local,
    const FaceRef& face_b,
    int b0_local,
    int b1_local,
    const SimState& state,
    Contact& contact) const
{
    const Object& AObj = objects[face_a.object_id];
    const Object& BObj = objects[face_b.object_id];

    const int a0 = face_a.v[a0_local];
    const int a1 = face_a.v[a1_local];
    const int b0 = face_b.v[b0_local];
    const int b1 = face_b.v[b1_local];

    const Vector3d p0 = state.X[face_a.object_id].row(a0).transpose();
    const Vector3d p1 = state.X[face_a.object_id].row(a1).transpose();
    const Vector3d q0 = state.X[face_b.object_id].row(b0).transpose();
    const Vector3d q1 = state.X[face_b.object_id].row(b1).transpose();

    double s = 0.0;
    double t = 0.0;
    Vector3d cp = Vector3d::Zero();
    Vector3d cq = Vector3d::Zero();
    if (!closestSegmentSegment(p0, p1, q0, q1, s, t, cp, cq)) return false;

    Vector3d diff = cp - cq;
    const double d = diff.norm();
    if (d >= contact_thickness) return false;

    Vector3d n;
    if (d > 1e-12) {
        n = diff / d;
    }
    else {
        n = (p1 - p0).cross(q1 - q0);
        if (n.norm() < 1e-12) return false;
        n.normalize();
    }

    contact.refs.clear();
    contact.refs.push_back(VertexRef{face_a.object_id, a0, 1.0 - s});
    contact.refs.push_back(VertexRef{face_a.object_id, a1, s});
    contact.refs.push_back(VertexRef{face_b.object_id, b0, -(1.0 - t)});
    contact.refs.push_back(VertexRef{face_b.object_id, b1, -t});
    contact.point = 0.5 * (cp + cq);
    contact.normal = n;
    contact.target_distance = 0.0;
    contact.is_intersection = false;
    contact.restitution = std::min(AObj.restitution, BObj.restitution);
    contact.friction = 0.5 * (AObj.friction + BObj.friction);
    return true;
}

void MassSpringWorld::collectTrianglePairContacts(
    const FaceRef& a,
    const FaceRef& b,
    const SimState& state,
    std::vector<Contact>& contacts) const
{
    const bool is_self_pair = a.object_id == b.object_id;
    const double local_dhat = is_self_pair
        ? contact_thickness * std::clamp(self_contact_thickness_scale, 0.05, 1.0)
        : contact_thickness;

    auto push_contact = [&](Contact c) {
        if (static_cast<int>(contacts.size()) < max_contacts_for_impulse) {
            c.is_self_contact = is_self_pair;
            c.dhat = local_dhat;
            c.target_distance = std::max(c.target_distance, is_self_pair ? local_dhat : 0.0);
            contacts.push_back(c);
        }
    };

    Contact c;
    // DCD intersection contacts. These invalidate the step and are not used as barrier terms.
    if (edgeTriangleContact(a, 0, 1, b, state, c)) push_contact(c);
    if (edgeTriangleContact(a, 1, 2, b, state, c)) push_contact(c);
    if (edgeTriangleContact(a, 2, 0, b, state, c)) push_contact(c);
    if (edgeTriangleContact(b, 0, 1, a, state, c)) push_contact(c);
    if (edgeTriangleContact(b, 1, 2, a, state, c)) push_contact(c);
    if (edgeTriangleContact(b, 2, 0, a, state, c)) push_contact(c);

    // Proximity contacts used by the barrier energy.
    if (vertexTriangleContact(a, 0, b, state, c)) push_contact(c);
    if (vertexTriangleContact(a, 1, b, state, c)) push_contact(c);
    if (vertexTriangleContact(a, 2, b, state, c)) push_contact(c);
    if (vertexTriangleContact(b, 0, a, state, c)) push_contact(c);
    if (vertexTriangleContact(b, 1, a, state, c)) push_contact(c);
    if (vertexTriangleContact(b, 2, a, state, c)) push_contact(c);

    constexpr int edges[3][2] = {{0, 1}, {1, 2}, {2, 0}};
    for (const auto& ea : edges) {
        for (const auto& eb : edges) {
            if (edgeEdgeContact(a, ea[0], ea[1], b, eb[0], eb[1], state, c)) {
                push_contact(c);
            }
        }
    }
}

MassSpringWorld::CollisionReport MassSpringWorld::detectContacts(const SimState& state) const
{
    CollisionReport report;
    double cell_size = collision_cell_size;
    if (cell_size <= 0.0) cell_size = computeAutomaticCellSize(state);

    const double margin = std::max(contact_thickness, collision_aabb_margin);
    std::vector<FaceRef> faces = collectFaces(state, margin);
    HashGrid grid;
    buildSpatialHash(faces, cell_size, grid);

    std::unordered_set<std::uint64_t> checked;
    for (const auto& item : grid) {
        const auto& ids = item.second;
        for (int ia = 0; ia < static_cast<int>(ids.size()); ++ia) {
            for (int ib = ia + 1; ib < static_cast<int>(ids.size()); ++ib) {
                if (static_cast<int>(report.contacts.size()) >= max_contacts_for_impulse) {
                    report.intersection_count = static_cast<int>(report.contacts.size());
                    return report;
                }

                const int fa = ids[ia];
                const int fb = ids[ib];
                const std::uint64_t key = pairKey(fa, fb);
                if (checked.find(key) != checked.end()) continue;
                checked.insert(key);

                const FaceRef& A = faces[fa];
                const FaceRef& B = faces[fb];
                // Self collision is handled later by swept-AABB + PBD projection.
                // Keep IPC/barrier candidates for inter-object collision only.
                if (A.object_id == B.object_id) continue;
                if (shouldSkipPair(A, B)) continue;
                if (!overlapAABB(A.box, B.box)) continue;

                report.candidate_pairs++;
                collectTrianglePairContacts(A, B, state, report.contacts);
            }
        }
    }

    report.intersection_count = static_cast<int>(report.contacts.size());
    return report;
}

MassSpringWorld::AABB MassSpringWorld::computeSweptTriangleAABB(
    const SimState& old_state,
    const SimState& trial_state,
    int object_id,
    const Vector3i& f,
    double margin) const
{
    Vector3d p[6] = {
        old_state.X[object_id].row(f[0]).transpose(),
        old_state.X[object_id].row(f[1]).transpose(),
        old_state.X[object_id].row(f[2]).transpose(),
        trial_state.X[object_id].row(f[0]).transpose(),
        trial_state.X[object_id].row(f[1]).transpose(),
        trial_state.X[object_id].row(f[2]).transpose()};

    AABB box;
    box.mn = p[0];
    box.mx = p[0];
    for (int i = 1; i < 6; ++i) {
        box.mn = box.mn.cwiseMin(p[i]);
        box.mx = box.mx.cwiseMax(p[i]);
    }
    box.mn.array() -= margin;
    box.mx.array() += margin;
    return box;
}

std::vector<MassSpringWorld::FaceRef> MassSpringWorld::collectSweptFaces(
    const SimState& old_state,
    const SimState& trial_state,
    double margin) const
{
    std::vector<FaceRef> faces;
    for (int oid = 0; oid < static_cast<int>(objects.size()); ++oid) {
        const Object& obj = objects[oid];
        for (int fid = 0; fid < static_cast<int>(obj.F.size()); ++fid) {
            FaceRef ref;
            ref.object_id = oid;
            ref.face_id = fid;
            ref.v = obj.F[fid];
            ref.box = computeSweptTriangleAABB(old_state, trial_state, oid, ref.v, margin);
            faces.push_back(ref);
        }
    }
    return faces;
}

int MassSpringWorld::automaticBisectionIters() const
{
    // Bisection interval length after n iterations is at most 2^{-n}.
    // To reach interval length <= tol, n >= ceil(log2(1 / tol)).
    // max_rollback_iters is only a safety cap; the effective value is automatic.
    const double tol = std::clamp(0.1, 1e-12, 1.0);
    const int required = static_cast<int>(std::ceil(std::log2(1.0 / tol)));
    return std::clamp(required, 1, 8);
}

MassSpringWorld::TOIReport MassSpringWorld::findEarliestTOIByBisection(
    const SimState& old_state,
    const SimState& trial_state) const
{
    TOIReport best;

    double cell_size = collision_cell_size;
    if (cell_size <= 0.0) cell_size = computeAutomaticCellSize(old_state);

    const double margin = std::max(contact_thickness, collision_aabb_margin);
    std::vector<FaceRef> faces = collectSweptFaces(old_state, trial_state, margin);
    HashGrid grid;
    buildSpatialHash(faces, cell_size, grid);

    std::unordered_set<std::uint64_t> checked;
    for (const auto& item : grid) {
        const auto& ids = item.second;
        for (int ia = 0; ia < static_cast<int>(ids.size()); ++ia) {
            for (int ib = ia + 1; ib < static_cast<int>(ids.size()); ++ib) {
                const int fa = ids[ia];
                const int fb = ids[ib];
                const std::uint64_t key = pairKey(fa, fb);
                if (checked.find(key) != checked.end()) continue;
                checked.insert(key);

                const FaceRef& A = faces[fa];
                const FaceRef& B = faces[fb];
                // Self collision is handled later by swept-AABB + PBD projection.
                // Keep IPC/barrier candidates for inter-object collision only.
                if (A.object_id == B.object_id) continue;
                if (shouldSkipPair(A, B)) continue;
                if (!overlapAABB(A.box, B.box)) continue;

                best.candidate_pairs++;

                std::vector<Contact> trial_contacts;
                collectTrianglePairContacts(A, B, trial_state, trial_contacts);
                if (trial_contacts.empty()) continue;

                std::vector<Contact> old_contacts;
                collectTrianglePairContacts(A, B, old_state, old_contacts);
                if (!old_contacts.empty()) {
                    // Already touching/penetrating at the start: this is a resting contact, not a new TOI.
                    // Return alpha=0 so step() can do contact solve without running an invalid rollback search.
                    if (!best.hit || 0.0 < best.alpha) {
                        best.hit = true;
                        best.alpha = 0.0;
                        best.contacts = old_contacts;
                    }
                    continue;
                }

                double left = 0.0;
                double right = 1.0;
                std::vector<Contact> right_contacts = trial_contacts;
                const int bisection_iters = automaticBisectionIters();
                for (int iter = 0; iter < bisection_iters; ++iter) {
                    if (right - left < 0.1) break;
                    const double mid = 0.5 * (left + right);
                    SimState mid_state = interpolateState(old_state, trial_state, mid);
                    std::vector<Contact> mid_contacts;
                    collectTrianglePairContacts(A, B, mid_state, mid_contacts);
                    if (!mid_contacts.empty()) {
                        right = mid;
                        right_contacts = std::move(mid_contacts);
                    }
                    else {
                        left = mid;
                    }
                }

                if (!best.hit || right < best.alpha) {
                    best.hit = true;
                    best.alpha = right;
                    best.contacts = right_contacts;
                }
            }
        }
    }

    return best;
}



double MassSpringWorld::contactGap(const Contact& contact, const SimState& state) const
{
    Vector3d q = Vector3d::Zero();
    for (const VertexRef& r : contact.refs) {
        if (r.object_id < 0 || r.object_id >= static_cast<int>(state.X.size())) continue;
        if (r.vertex_id < 0 || r.vertex_id >= state.X[r.object_id].rows()) continue;
        q += r.coeff * state.X[r.object_id].row(r.vertex_id).transpose();
    }
    Vector3d n = contact.normal;
    if (n.norm() < 1e-12) return std::numeric_limits<double>::infinity();
    n.normalize();
    return n.dot(q);
}

bool MassSpringWorld::hasInvalidContact(
    const SimState& state,
    const std::vector<Contact>& contacts,
    double eps) const
{
    for (const Contact& c : contacts) {
        if (c.is_intersection) return true;
        const double g = contactGap(c, state);
        if (std::isfinite(g) && g <= eps) return true;
    }
    return false;
}


std::vector<MassSpringWorld::Contact> MassSpringWorld::collectSelfPBDContactsSwept(
    const SimState& old_state,
    const SimState& trial_state) const
{
    std::vector<Contact> contacts;
    if (!enable_self_collision || contact_thickness <= 0.0) return contacts;

    const double self_dhat = std::max(
        1e-8,
        contact_thickness * std::clamp(self_contact_thickness_scale, 0.05, 1.0));
    const double predictive_gap = kSelfPredictiveActivationScale * self_dhat;

    double cell_size = collision_cell_size;
    if (cell_size <= 0.0) cell_size = computeAutomaticCellSize(trial_state);

    // Use a slightly larger swept margin than the final projection thickness.
    // This builds contacts before actual penetration, while the PBD projection
    // still only activates when the half-space constraint is violated.
    const double margin = std::max((1.0 + kSelfPredictiveActivationScale) * self_dhat,
                                   collision_aabb_margin);
    std::vector<FaceRef> faces = collectSweptFaces(old_state, trial_state, margin);
    HashGrid grid;
    buildSpatialHash(faces, cell_size, grid);

    auto valid_vertex = [&](int oid, int vid) {
        return oid >= 0 && oid < static_cast<int>(objects.size()) &&
               oid < static_cast<int>(trial_state.X.size()) &&
               oid < static_cast<int>(old_state.X.size()) &&
               vid >= 0 && vid < trial_state.X[oid].rows() &&
               vid < old_state.X[oid].rows();
    };

    auto push = [&](Contact c) {
        if (static_cast<int>(contacts.size()) >= max_ipc_candidates) return;
        c.is_self_contact = true;
        c.is_intersection = false;
        c.target_distance = self_dhat;
        c.dhat = self_dhat;
        c.stiffness_scale = std::clamp(self_contact_stiffness_scale, 0.0, 1.0);
        contacts.push_back(std::move(c));
    };

    auto normalized_or_zero = [](const Vector3d& v) -> Vector3d {
        const double len = v.norm();
        if (len <= 1e-12) return Vector3d::Zero();
        return (v / len).eval();
    };

    auto add_vt = [&](const FaceRef& vf, int local, const FaceRef& tf) {
        if (static_cast<int>(contacts.size()) >= max_ipc_candidates) return;
        if (vf.object_id != tf.object_id) return;
        const int oid = vf.object_id;
        if (oid < 0 || oid >= static_cast<int>(objects.size())) return;
        const Object& obj = objects[oid];
        if (!obj.isDeformable()) return;

        const int p_id = vf.v[local];
        const int a_id = tf.v[0];
        const int b_id = tf.v[1];
        const int c_id = tf.v[2];
        if (!valid_vertex(oid, p_id) || !valid_vertex(oid, a_id) ||
            !valid_vertex(oid, b_id) || !valid_vertex(oid, c_id)) return;

        const Vector3d p0 = old_state.X[oid].row(p_id).transpose();
        const Vector3d p1 = trial_state.X[oid].row(p_id).transpose();
        const Vector3d a0 = old_state.X[oid].row(a_id).transpose();
        const Vector3d b0 = old_state.X[oid].row(b_id).transpose();
        const Vector3d c0 = old_state.X[oid].row(c_id).transpose();
        const Vector3d a1 = trial_state.X[oid].row(a_id).transpose();
        const Vector3d b1 = trial_state.X[oid].row(b_id).transpose();
        const Vector3d c1 = trial_state.X[oid].row(c_id).transpose();

        Vector3d n0 = (b0 - a0).cross(c0 - a0);
        if (n0.norm() <= 1e-12) return;
        n0.normalize();
        Vector3d n1 = (b1 - a1).cross(c1 - a1);
        if (n1.norm() <= 1e-12) n1 = n0;
        else {
            n1.normalize();
            if (n0.dot(n1) < 0.0) n1 = -n1;
        }

        Vector3d q0 = Vector3d::Zero();
        Vector3d bary0 = Vector3d::Zero();
        if (!closestPointTriangle(p0, a0, b0, c0, q0, bary0)) return;

        Vector3d q_trial = Vector3d::Zero();
        Vector3d bary_trial = Vector3d::Zero();
        if (!closestPointTriangle(p1, a1, b1, c1, q_trial, bary_trial)) return;

        // Predictive contact plane: the normal is the safe side from old_state.
        // Keep this normal fixed during the PBD solve; do not use the current
        // closest-point normal, which can flip under folds and inject energy.
        const double s0 = n0.dot(p0 - a0);
        const double s1 = n1.dot(p1 - a1);
        Vector3d n = Vector3d::Zero();
        if (std::abs(s0) > 1e-10) {
            n = (s0 >= 0.0 ? n0 : -n0);
        }
        else {
            n = normalized_or_zero(p0 - q0);
            if (n.norm() <= 1e-12) n = normalized_or_zero(p1 - q_trial);
            if (n.norm() <= 1e-12) n = n0;
        }
        if (n.norm() <= 1e-12) return;
        n.normalize();

        const bool sign_crossed = (s0 * s1 < 0.0);
        bool crossed_inside = false;
        Vector3d bary_cross = bary_trial;
        if (sign_crossed) {
            const double denom = std::abs(s0) + std::abs(s1);
            const double tau = denom > 1e-12 ? std::clamp(std::abs(s0) / denom, 0.0, 1.0) : 0.5;
            const Vector3d pc = p0 + tau * (p1 - p0);
            const Vector3d ac = a0 + tau * (a1 - a0);
            const Vector3d bc = b0 + tau * (b1 - b0);
            const Vector3d cc = c0 + tau * (c1 - c0);
            Vector3d qc;
            if (closestPointTriangle(pc, ac, bc, cc, qc, bary_cross)) {
                crossed_inside = (pc - qc).norm() <= std::max(1e-6, 0.5 * self_dhat) &&
                                 bary_cross.minCoeff() >= -1e-4 &&
                                 bary_cross.maxCoeff() <= 1.0 + 1e-4;
            }
        }

        Vector3d bary = crossed_inside ? bary_cross : bary_trial;
        for (int k = 0; k < 3; ++k) {
            if (bary[k] < 0.0) bary[k] = 0.0;
        }
        const double bary_sum = bary.sum();
        if (bary_sum <= 1e-12) return;
        bary /= bary_sum;

        const Vector3d q_plane = bary[0] * a1 + bary[1] * b1 + bary[2] * c1;
        const double signed_gap = n.dot(p1 - q_plane);
        const double C_trial = signed_gap - self_dhat;
        const double euclidean_gap = (p1 - q_trial).norm();

        // Create the contact early.  Projection will only act if C<0, but the
        // velocity pass can already remove approaching normal velocity when the
        // predicted point is close to the safety shell.
        const bool predictive_contact = (C_trial < predictive_gap) ||
                                        (euclidean_gap < self_dhat + predictive_gap) ||
                                        crossed_inside;
        if (!predictive_contact) return;

        Contact contact;
        contact.refs.push_back(VertexRef{oid, p_id, 1.0});
        contact.refs.push_back(VertexRef{oid, a_id, -bary[0]});
        contact.refs.push_back(VertexRef{oid, b_id, -bary[1]});
        contact.refs.push_back(VertexRef{oid, c_id, -bary[2]});
        contact.point = q_plane;
        contact.normal = n;
        contact.restitution = kSelfCollisionRestitution;
        contact.friction = obj.friction;
        push(contact);
    };

    auto add_ee = [&](const FaceRef& fa, int a0l, int a1l, const FaceRef& fb, int b0l, int b1l) {
        if (static_cast<int>(contacts.size()) >= max_ipc_candidates) return;
        if (fa.object_id != fb.object_id) return;
        const int oid = fa.object_id;
        if (oid < 0 || oid >= static_cast<int>(objects.size())) return;
        const Object& obj = objects[oid];
        if (!obj.isDeformable()) return;

        const int a0_id = fa.v[a0l];
        const int a1_id = fa.v[a1l];
        const int b0_id = fb.v[b0l];
        const int b1_id = fb.v[b1l];
        if (!valid_vertex(oid, a0_id) || !valid_vertex(oid, a1_id) ||
            !valid_vertex(oid, b0_id) || !valid_vertex(oid, b1_id)) return;

        const Vector3d pa0_old = old_state.X[oid].row(a0_id).transpose();
        const Vector3d pa1_old = old_state.X[oid].row(a1_id).transpose();
        const Vector3d pb0_old = old_state.X[oid].row(b0_id).transpose();
        const Vector3d pb1_old = old_state.X[oid].row(b1_id).transpose();
        double s = 0.0;
        double t = 0.0;
        Vector3d cp_old = Vector3d::Zero();
        Vector3d cq_old = Vector3d::Zero();
        if (!closestSegmentSegment(pa0_old, pa1_old, pb0_old, pb1_old, s, t, cp_old, cq_old)) return;

        const Vector3d pa0 = trial_state.X[oid].row(a0_id).transpose();
        const Vector3d pa1 = trial_state.X[oid].row(a1_id).transpose();
        const Vector3d pb0 = trial_state.X[oid].row(b0_id).transpose();
        const Vector3d pb1 = trial_state.X[oid].row(b1_id).transpose();

        Vector3d n = normalized_or_zero(cp_old - cq_old);
        if (n.norm() <= 1e-12) {
            n = normalized_or_zero((pa1_old - pa0_old).cross(pb1_old - pb0_old));
        }
        if (n.norm() <= 1e-12) {
            double s_trial = 0.0, t_trial = 0.0;
            Vector3d cp_trial, cq_trial;
            if (!closestSegmentSegment(pa0, pa1, pb0, pb1, s_trial, t_trial, cp_trial, cq_trial)) return;
            n = normalized_or_zero(cp_trial - cq_trial);
        }
        if (n.norm() <= 1e-12) return;
        n.normalize();

        const Vector3d cp = (1.0 - s) * pa0 + s * pa1;
        const Vector3d cq = (1.0 - t) * pb0 + t * pb1;
        const double signed_gap = n.dot(cp - cq);
        const double C_trial = signed_gap - self_dhat;

        double s_near = 0.0, t_near = 0.0;
        Vector3d cp_near = Vector3d::Zero();
        Vector3d cq_near = Vector3d::Zero();
        const bool has_near = closestSegmentSegment(pa0, pa1, pb0, pb1, s_near, t_near, cp_near, cq_near);
        const double euclidean_gap = has_near ? (cp_near - cq_near).norm() : std::numeric_limits<double>::infinity();

        const bool predictive_contact = (C_trial < predictive_gap) ||
                                        (euclidean_gap < self_dhat + predictive_gap);
        if (!predictive_contact) return;

        Contact contact;
        contact.refs.push_back(VertexRef{oid, a0_id, 1.0 - s});
        contact.refs.push_back(VertexRef{oid, a1_id, s});
        contact.refs.push_back(VertexRef{oid, b0_id, -(1.0 - t)});
        contact.refs.push_back(VertexRef{oid, b1_id, -t});
        contact.point = 0.5 * (cp + cq);
        contact.normal = n;
        contact.restitution = kSelfCollisionRestitution;
        contact.friction = obj.friction;
        push(contact);
    };

    std::unordered_set<std::uint64_t> checked;
    constexpr int edges[3][2] = {{0, 1}, {1, 2}, {2, 0}};
    for (const auto& item : grid) {
        const auto& ids = item.second;
        for (int ia = 0; ia < static_cast<int>(ids.size()); ++ia) {
            for (int ib = ia + 1; ib < static_cast<int>(ids.size()); ++ib) {
                if (static_cast<int>(contacts.size()) >= max_ipc_candidates) return contacts;
                const int fa = ids[ia];
                const int fb = ids[ib];
                const std::uint64_t key = pairKey(fa, fb);
                if (checked.find(key) != checked.end()) continue;
                checked.insert(key);

                const FaceRef& A = faces[fa];
                const FaceRef& B = faces[fb];
                if (A.object_id != B.object_id) continue;
                if (A.object_id < 0 || A.object_id >= static_cast<int>(objects.size())) continue;
                if (!objects[A.object_id].isDeformable()) continue;
                if (shareVertex(A.v, B.v)) continue;
                if (!overlapAABB(A.box, B.box)) continue;

                add_vt(A, 0, B);
                add_vt(A, 1, B);
                add_vt(A, 2, B);
                add_vt(B, 0, A);
                add_vt(B, 1, A);
                add_vt(B, 2, A);

                for (const auto& ea : edges) {
                    for (const auto& eb : edges) {
                        add_ee(A, ea[0], ea[1], B, eb[0], eb[1]);
                    }
                }
            }
        }
    }
    return contacts;
}

std::vector<MassSpringWorld::IpcCandidate> MassSpringWorld::collectIPCCandidates(
    const SimState& state,
    double dhat) const
{
    std::vector<IpcCandidate> candidates;
    if (dhat <= 0.0) return candidates;

    double cell_size = collision_cell_size;
    if (cell_size <= 0.0) cell_size = computeAutomaticCellSize(state);

    const double margin = std::max(dhat, collision_aabb_margin);
    std::vector<FaceRef> faces = collectFaces(state, margin);
    HashGrid grid;
    buildSpatialHash(faces, cell_size, grid);

    auto effective_dhat = [&](bool is_self) {
        const double scale = is_self ? std::clamp(self_contact_thickness_scale, 0.05, 1.0) : 1.0;
        return std::max(1e-12, dhat * scale);
    };

    auto effective_stiffness_scale = [&](bool is_self) {
        return is_self ? std::clamp(self_contact_stiffness_scale, 0.0, 1.0) : 1.0;
    };

    auto add_vt = [&](const FaceRef& vf, int local, const FaceRef& tf) {
        if (static_cast<int>(candidates.size()) >= max_ipc_candidates) return;
        const bool is_self = vf.object_id == tf.object_id;
        const double local_dhat = effective_dhat(is_self);
        if (local_dhat <= 0.0) return;
        const int p_id = vf.v[local];
        const int a_id = tf.v[0];
        const int b_id = tf.v[1];
        const int c_id = tf.v[2];
        const Vector3d p = state.X[vf.object_id].row(p_id).transpose();
        const Vector3d a = state.X[tf.object_id].row(a_id).transpose();
        const Vector3d b = state.X[tf.object_id].row(b_id).transpose();
        const Vector3d cc = state.X[tf.object_id].row(c_id).transpose();
        Vector3d q, bary;
        if (!closestPointTriangle(p, a, b, cc, q, bary)) return;
        const double d = (p - q).norm();
        if (d <= 1e-10 || d >= local_dhat) return;
        IpcCandidate cand;
        cand.type = IPC_VERTEX_TRIANGLE;
        cand.object_id[0] = vf.object_id;
        cand.vertex_id[0] = p_id;
        cand.object_id[1] = tf.object_id;
        cand.vertex_id[1] = a_id;
        cand.object_id[2] = tf.object_id;
        cand.vertex_id[2] = b_id;
        cand.object_id[3] = tf.object_id;
        cand.vertex_id[3] = c_id;
        cand.friction = 0.5 * (objects[vf.object_id].friction + objects[tf.object_id].friction);
        cand.is_self_contact = is_self;
        cand.dhat = local_dhat;
        cand.stiffness_scale = effective_stiffness_scale(is_self);
        candidates.push_back(cand);
    };

    auto add_ee = [&](const FaceRef& fa, int a0l, int a1l, const FaceRef& fb, int b0l, int b1l) {
        if (static_cast<int>(candidates.size()) >= max_ipc_candidates) return;
        const bool is_self = fa.object_id == fb.object_id;
        const double local_dhat = effective_dhat(is_self);
        if (local_dhat <= 0.0) return;
        const int a0 = fa.v[a0l];
        const int a1 = fa.v[a1l];
        const int b0 = fb.v[b0l];
        const int b1 = fb.v[b1l];
        const Vector3d p0 = state.X[fa.object_id].row(a0).transpose();
        const Vector3d p1 = state.X[fa.object_id].row(a1).transpose();
        const Vector3d q0 = state.X[fb.object_id].row(b0).transpose();
        const Vector3d q1 = state.X[fb.object_id].row(b1).transpose();
        double s = 0.0, t = 0.0;
        Vector3d cp, cq;
        if (!closestSegmentSegment(p0, p1, q0, q1, s, t, cp, cq)) return;
        const double d = (cp - cq).norm();
        if (d <= 1e-10 || d >= local_dhat) return;
        IpcCandidate cand;
        cand.type = IPC_EDGE_EDGE;
        cand.object_id[0] = fa.object_id;
        cand.vertex_id[0] = a0;
        cand.object_id[1] = fa.object_id;
        cand.vertex_id[1] = a1;
        cand.object_id[2] = fb.object_id;
        cand.vertex_id[2] = b0;
        cand.object_id[3] = fb.object_id;
        cand.vertex_id[3] = b1;
        cand.friction = 0.5 * (objects[fa.object_id].friction + objects[fb.object_id].friction);
        cand.is_self_contact = is_self;
        cand.dhat = local_dhat;
        cand.stiffness_scale = effective_stiffness_scale(is_self);
        candidates.push_back(cand);
    };

    std::unordered_set<std::uint64_t> checked;
    constexpr int edges[3][2] = {{0, 1}, {1, 2}, {2, 0}};
    for (const auto& item : grid) {
        const auto& ids = item.second;
        for (int ia = 0; ia < static_cast<int>(ids.size()); ++ia) {
            for (int ib = ia + 1; ib < static_cast<int>(ids.size()); ++ib) {
                if (static_cast<int>(candidates.size()) >= max_ipc_candidates) return candidates;
                const int fa = ids[ia];
                const int fb = ids[ib];
                const std::uint64_t key = pairKey(fa, fb);
                if (checked.find(key) != checked.end()) continue;
                checked.insert(key);

                const FaceRef& A = faces[fa];
                const FaceRef& B = faces[fb];
                // Self collision is handled later by swept-AABB + PBD projection.
                // Keep IPC/barrier candidates for inter-object collision only.
                if (A.object_id == B.object_id) continue;
                if (shouldSkipPair(A, B)) continue;
                if (!overlapAABB(A.box, B.box)) continue;

                add_vt(A, 0, B);
                add_vt(A, 1, B);
                add_vt(A, 2, B);
                add_vt(B, 0, A);
                add_vt(B, 1, A);
                add_vt(B, 2, A);

                for (const auto& ea : edges) {
                    for (const auto& eb : edges) {
                        add_ee(A, ea[0], ea[1], B, eb[0], eb[1]);
                    }
                }
            }
        }
    }
    return candidates;
}

bool MassSpringWorld::evaluateIPCDistance(
    const IpcCandidate& candidate,
    const SimState& state,
    IpcDistanceInfo& info) const
{
    info = IpcDistanceInfo();
    const double eps = 1e-12;
    auto contact_area_weight = [&](const std::vector<VertexRef>& refs) {
        double weighted_area = 0.0;
        double coeff_sum = 0.0;
        for (const VertexRef& r : refs) {
            if (r.object_id < 0 || r.object_id >= static_cast<int>(objects.size())) continue;
            const Object& obj = objects[r.object_id];
            if (!obj.isDeformable()) continue;
            if (r.vertex_id < 0 || r.vertex_id >= static_cast<int>(obj.vertex_area.size())) continue;
            const double area = obj.vertex_area[r.vertex_id];
            if (!(area > 0.0) || !std::isfinite(area)) continue;
            const double c = std::abs(r.coeff);
            weighted_area += c * area;
            coeff_sum += c;
        }
        if (coeff_sum > 0.0) return std::max(1e-12, weighted_area / coeff_sum);
        return 1.0;
    };
    if (candidate.type == IPC_VERTEX_TRIANGLE) {
        const int po = candidate.object_id[0];
        const int pv = candidate.vertex_id[0];
        const int to = candidate.object_id[1];
        const int a = candidate.vertex_id[1];
        const int b = candidate.vertex_id[2];
        const int c = candidate.vertex_id[3];
        if (po < 0 || to < 0 || po >= static_cast<int>(state.X.size()) || to >= static_cast<int>(state.X.size())) return false;
        const Vector3d p = state.X[po].row(pv).transpose();
        const Vector3d xa = state.X[to].row(a).transpose();
        const Vector3d xb = state.X[to].row(b).transpose();
        const Vector3d xc = state.X[to].row(c).transpose();
        Vector3d q, bary;
        if (!closestPointTriangle(p, xa, xb, xc, q, bary)) return false;
        Vector3d diff = p - q;
        double d = diff.norm();
        Vector3d n;
        if (d > eps) {
            n = diff / d;
        }
        else {
            n = (xb - xa).cross(xc - xa);
            if (n.norm() <= eps) return false;
            n.normalize();
            d = 0.0;
        }
        info.distance = d;
        info.normal = n;
        info.refs.push_back(VertexRef{po, pv, 1.0});
        info.refs.push_back(VertexRef{to, a, -bary[0]});
        info.refs.push_back(VertexRef{to, b, -bary[1]});
        info.refs.push_back(VertexRef{to, c, -bary[2]});
        info.weight = contact_area_weight(info.refs);
        info.dhat = candidate.dhat > 0.0 ? candidate.dhat : contact_thickness;
        info.stiffness_scale = candidate.stiffness_scale;
        info.is_self_contact = candidate.is_self_contact;
        return true;
    }

    const int ao = candidate.object_id[0];
    const int a0 = candidate.vertex_id[0];
    const int a1 = candidate.vertex_id[1];
    const int bo = candidate.object_id[2];
    const int b0 = candidate.vertex_id[2];
    const int b1 = candidate.vertex_id[3];
    if (ao < 0 || bo < 0 || ao >= static_cast<int>(state.X.size()) || bo >= static_cast<int>(state.X.size())) return false;
    const Vector3d p0 = state.X[ao].row(a0).transpose();
    const Vector3d p1 = state.X[ao].row(a1).transpose();
    const Vector3d q0 = state.X[bo].row(b0).transpose();
    const Vector3d q1 = state.X[bo].row(b1).transpose();
    double s = 0.0, t = 0.0;
    Vector3d cp, cq;
    if (!closestSegmentSegment(p0, p1, q0, q1, s, t, cp, cq)) return false;
    Vector3d diff = cp - cq;
    double d = diff.norm();
    Vector3d n;
    if (d > eps) {
        n = diff / d;
    }
    else {
        n = (p1 - p0).cross(q1 - q0);
        if (n.norm() <= eps) return false;
        n.normalize();
        d = 0.0;
    }
    info.distance = d;
    info.normal = n;
    info.refs.push_back(VertexRef{ao, a0, 1.0 - s});
    info.refs.push_back(VertexRef{ao, a1, s});
    info.refs.push_back(VertexRef{bo, b0, -(1.0 - t)});
    info.refs.push_back(VertexRef{bo, b1, -t});
    info.weight = contact_area_weight(info.refs);
    info.dhat = candidate.dhat > 0.0 ? candidate.dhat : contact_thickness;
    info.stiffness_scale = candidate.stiffness_scale;
    info.is_self_contact = candidate.is_self_contact;
    return true;
}

bool MassSpringWorld::hasDCDIntersection(const SimState& state) const
{
    CollisionReport report = detectContacts(state);
    for (const Contact& c : report.contacts) {
        // Self intersections are no longer used to reject the IPC step.
        // They are repaired by the PBD self-collision pass after the IPC solve.
        if (c.is_intersection && !c.is_self_contact) return true;
    }
    return false;
}


std::vector<double> MassSpringWorld::solvePolynomialRealRoots(double a, double b, double c, double d) const
{
    const double eps = 1e-12;
    std::vector<double> roots;

    auto add_root = [&](double r) {
        if (!std::isfinite(r)) return;
        for (double existing : roots) {
            if (std::abs(existing - r) < 1e-8) return;
        }
        roots.push_back(r);
    };

    if (std::abs(a) < eps) {
        if (std::abs(b) < eps) {
            if (std::abs(c) >= eps) add_root(-d / c);
            return roots;
        }
        const double disc = c * c - 4.0 * b * d;
        if (disc < -eps) return roots;
        if (std::abs(disc) <= eps) {
            add_root(-c / (2.0 * b));
        }
        else {
            const double sqrt_disc = std::sqrt(std::max(0.0, disc));
            add_root((-c - sqrt_disc) / (2.0 * b));
            add_root((-c + sqrt_disc) / (2.0 * b));
        }
        return roots;
    }

    const double A = b / a;
    const double B = c / a;
    const double C = d / a;
    const double p = B - A * A / 3.0;
    const double q = 2.0 * A * A * A / 27.0 - A * B / 3.0 + C;
    const double half_q = 0.5 * q;
    const double third_p = p / 3.0;
    const double disc = half_q * half_q + third_p * third_p * third_p;
    const double shift = -A / 3.0;

    if (disc > eps) {
        const double sqrt_disc = std::sqrt(disc);
        const double u = std::cbrt(-half_q + sqrt_disc);
        const double v = std::cbrt(-half_q - sqrt_disc);
        add_root(u + v + shift);
    }
    else if (std::abs(disc) <= eps) {
        const double u = std::cbrt(-half_q);
        add_root(2.0 * u + shift);
        add_root(-u + shift);
    }
    else {
        const double r = 2.0 * std::sqrt(std::max(0.0, -third_p));
        double cos_arg = -half_q / std::sqrt(std::max(eps, -third_p * third_p * third_p));
        cos_arg = std::clamp(cos_arg, -1.0, 1.0);
        const double theta = std::acos(cos_arg);
        constexpr double pi = 3.141592653589793238462643383279502884;
        for (int k = 0; k < 3; ++k) {
            add_root(r * std::cos((theta + 2.0 * pi * k) / 3.0) + shift);
        }
    }

    std::sort(roots.begin(), roots.end());
    return roots;
}

bool MassSpringWorld::vertexTriangleCCD(
    const SimState& from,
    const SimState& to,
    int p_obj,
    int p_vid,
    int tri_obj,
    const Vector3i& tri,
    double& alpha) const
{
    auto pos = [&](const SimState& s, int oid, int vid) {
        return s.X[oid].row(vid).transpose();
    };
    auto lerp = [](const Vector3d& a, const Vector3d& b, double t) {
        return a + t * (b - a);
    };

    const Vector3d p0 = pos(from, p_obj, p_vid);
    const Vector3d p1 = pos(to, p_obj, p_vid);
    const Vector3d a0 = pos(from, tri_obj, tri[0]);
    const Vector3d a1 = pos(to, tri_obj, tri[0]);
    const Vector3d b0 = pos(from, tri_obj, tri[1]);
    const Vector3d b1 = pos(to, tri_obj, tri[1]);
    const Vector3d c0 = pos(from, tri_obj, tri[2]);
    const Vector3d c1 = pos(to, tri_obj, tri[2]);

    auto signed_volume = [&](double t) {
        const Vector3d p = lerp(p0, p1, t);
        const Vector3d a = lerp(a0, a1, t);
        const Vector3d b = lerp(b0, b1, t);
        const Vector3d c = lerp(c0, c1, t);
        return (p - a).dot((b - a).cross(c - a));
    };

    const double f0 = signed_volume(0.0);
    const double f1 = signed_volume(1.0);
    const double f2 = signed_volume(2.0);
    const double f3 = signed_volume(3.0);
    const double A = (f3 - 3.0 * f2 + 3.0 * f1 - f0) / 6.0;
    const double B = 0.5 * (f2 - 2.0 * f1 + f0) - 3.0 * A;
    const double C = f1 - A - B - f0;
    const double D = f0;

    bool hit = false;
    double best = 1.0;
    const double t_eps = 1e-8;
    const double geom_eps = 1e-7;
    std::vector<double> roots = solvePolynomialRealRoots(A, B, C, D);
    for (double t : roots) {
        if (t <= t_eps || t > 1.0 + t_eps) continue;
        t = std::clamp(t, 0.0, 1.0);
        const Vector3d p = lerp(p0, p1, t);
        const Vector3d a = lerp(a0, a1, t);
        const Vector3d b = lerp(b0, b1, t);
        const Vector3d c = lerp(c0, c1, t);
        Vector3d q, bary;
        if (!closestPointTriangle(p, a, b, c, q, bary)) continue;
        if ((p - q).norm() <= geom_eps && bary.minCoeff() >= -1e-6 && bary.maxCoeff() <= 1.0 + 1e-6) {
            if (t < best) {
                best = t;
                hit = true;
            }
        }
    }
    if (hit) alpha = best;
    return hit;
}

bool MassSpringWorld::edgeEdgeCCD(
    const SimState& from,
    const SimState& to,
    int a_obj,
    int a0,
    int a1,
    int b_obj,
    int b0,
    int b1,
    double& alpha) const
{
    auto pos = [&](const SimState& s, int oid, int vid) {
        return s.X[oid].row(vid).transpose();
    };
    auto lerp = [](const Vector3d& a, const Vector3d& b, double t) {
        return a + t * (b - a);
    };

    const Vector3d p00 = pos(from, a_obj, a0);
    const Vector3d p01 = pos(to, a_obj, a0);
    const Vector3d p10 = pos(from, a_obj, a1);
    const Vector3d p11 = pos(to, a_obj, a1);
    const Vector3d q00 = pos(from, b_obj, b0);
    const Vector3d q01 = pos(to, b_obj, b0);
    const Vector3d q10 = pos(from, b_obj, b1);
    const Vector3d q11 = pos(to, b_obj, b1);

    auto coplanar_value = [&](double t) {
        const Vector3d p0 = lerp(p00, p01, t);
        const Vector3d p1 = lerp(p10, p11, t);
        const Vector3d q0 = lerp(q00, q01, t);
        const Vector3d q1 = lerp(q10, q11, t);
        return (p0 - q0).dot((p1 - p0).cross(q1 - q0));
    };

    const double f0 = coplanar_value(0.0);
    const double f1 = coplanar_value(1.0);
    const double f2 = coplanar_value(2.0);
    const double f3 = coplanar_value(3.0);
    const double A = (f3 - 3.0 * f2 + 3.0 * f1 - f0) / 6.0;
    const double B = 0.5 * (f2 - 2.0 * f1 + f0) - 3.0 * A;
    const double C = f1 - A - B - f0;
    const double D = f0;

    bool hit = false;
    double best = 1.0;
    const double t_eps = 1e-8;
    const double geom_eps = 1e-7;
    std::vector<double> roots = solvePolynomialRealRoots(A, B, C, D);
    for (double t : roots) {
        if (t <= t_eps || t > 1.0 + t_eps) continue;
        t = std::clamp(t, 0.0, 1.0);
        const Vector3d p0 = lerp(p00, p01, t);
        const Vector3d p1 = lerp(p10, p11, t);
        const Vector3d q0 = lerp(q00, q01, t);
        const Vector3d q1 = lerp(q10, q11, t);
        double s = 0.0, u = 0.0;
        Vector3d cp, cq;
        if (!closestSegmentSegment(p0, p1, q0, q1, s, u, cp, cq)) continue;
        if ((cp - cq).norm() <= geom_eps && s >= -1e-6 && s <= 1.0 + 1e-6 && u >= -1e-6 && u <= 1.0 + 1e-6) {
            if (t < best) {
                best = t;
                hit = true;
            }
        }
    }
    if (hit) alpha = best;
    return hit;
}

double MassSpringWorld::computeCCDSafeStep(const SimState& from, const SimState& to) const
{
    if (hasDCDIntersection(from)) return 0.0;

    double cell_size = collision_cell_size;
    if (cell_size <= 0.0) cell_size = computeAutomaticCellSize(from);
    const double margin = std::max(contact_thickness, collision_aabb_margin);
    std::vector<FaceRef> faces = collectSweptFaces(from, to, margin);
    HashGrid grid;
    buildSpatialHash(faces, cell_size, grid);

    double alpha_min = 1.0;
    bool hit = false;
    std::unordered_set<std::uint64_t> checked;
    constexpr int edges[3][2] = {{0, 1}, {1, 2}, {2, 0}};
    for (const auto& item : grid) {
        const auto& ids = item.second;
        for (int ia = 0; ia < static_cast<int>(ids.size()); ++ia) {
            for (int ib = ia + 1; ib < static_cast<int>(ids.size()); ++ib) {
                const int fa = ids[ia];
                const int fb = ids[ib];
                const std::uint64_t key = pairKey(fa, fb);
                if (checked.find(key) != checked.end()) continue;
                checked.insert(key);

                const FaceRef& A = faces[fa];
                const FaceRef& B = faces[fb];
                // Self collision is handled later by swept-AABB + PBD projection.
                // Keep IPC/barrier candidates for inter-object collision only.
                if (A.object_id == B.object_id) continue;
                if (shouldSkipPair(A, B)) continue;
                if (!overlapAABB(A.box, B.box)) continue;

                double a = 1.0;
                for (int i = 0; i < 3; ++i) {
                    if (vertexTriangleCCD(from, to, A.object_id, A.v[i], B.object_id, B.v, a)) {
                        alpha_min = std::min(alpha_min, a);
                        hit = true;
                    }
                    if (vertexTriangleCCD(from, to, B.object_id, B.v[i], A.object_id, A.v, a)) {
                        alpha_min = std::min(alpha_min, a);
                        hit = true;
                    }
                }
                for (const auto& ea : edges) {
                    for (const auto& eb : edges) {
                        if (edgeEdgeCCD(from, to, A.object_id, A.v[ea[0]], A.v[ea[1]], B.object_id, B.v[eb[0]], B.v[eb[1]], a)) {
                            alpha_min = std::min(alpha_min, a);
                            hit = true;
                        }
                    }
                }
            }
        }
    }
    if (!hit) return 1.0;
    return std::clamp(0.8 * alpha_min, 0.0, 1.0);
}

bool MassSpringWorld::solveIPCStep(
    const SimState& old_state,
    double dt,
    SimState& out_state) const
{
    last_self_contact_count = 0;
    if (dt <= 1e-12) return false;

    out_state = old_state;
    const int nobj = static_cast<int>(objects.size());

    std::vector<std::vector<int>> dof_base(nobj);
    int ndof = 0;
    for (int oid = 0; oid < nobj; ++oid) {
        const Object& obj = objects[oid];
        const int nv = static_cast<int>(old_state.X[oid].rows());
        dof_base[oid].assign(nv, -1);
        if (!obj.isDeformable()) continue;
        for (int i = 0; i < nv; ++i) {
            if (i < static_cast<int>(obj.fixed_mask.size()) && obj.fixed_mask[i]) continue;
            dof_base[oid][i] = ndof;
            ndof += 3;
        }
    }

    if (ndof == 0) return true;

    const Vector3d acceleration_ext = gravity + wind_ext_acc;
    std::vector<MatrixXd> Y(nobj);
    for (int oid = 0; oid < nobj; ++oid) {
        const Object& obj = objects[oid];
        Y[oid] = old_state.X[oid];
        if (obj.isDeformable()) {
            MatrixXd V_predict = old_state.V[oid];
            if (obj.damping >= 0.0 && obj.damping < 1.0) V_predict *= obj.damping;
            Y[oid] = old_state.X[oid] + dt * V_predict;
            Y[oid].rowwise() += (dt * dt * acceleration_ext).transpose();
            for (int i = 0; i < Y[oid].rows(); ++i) {
                if (i < static_cast<int>(obj.fixed_mask.size()) && obj.fixed_mask[i]) {
                    Y[oid].row(i) = obj.X0.row(i);
                    out_state.X[oid].row(i) = obj.X0.row(i);
                    out_state.V[oid].row(i).setZero();
                }
            }
        }
        else if (obj.isKinematic()) {
            out_state.X[oid] = old_state.X[oid] + dt * old_state.V[oid];
            out_state.V[oid] = old_state.V[oid];
        }
        else {
            out_state.X[oid] = old_state.X[oid];
            out_state.V[oid].setZero();
        }
    }

    auto add_block = [](std::vector<Eigen::Triplet<double>>& triplets, int row_base, int col_base, const Eigen::Matrix3d& block) {
        if (row_base < 0 || col_base < 0) return;
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                triplets.emplace_back(row_base + r, col_base + c, block(r, c));
            }
        }
    };

    auto add_grad = [](Eigen::VectorXd& grad, int base, const Vector3d& g) {
        if (base < 0) return;
        grad.segment<3>(base) += g;
    };

    auto barrier_values = [](double d, double dhat, double& B, double& Bp, double& Bpp) {
        B = 0.0;
        Bp = 0.0;
        Bpp = 0.0;
        if (d <= 0.0 || d >= dhat) return;
        const double d_eval = std::max(d, std::max(1e-12, 1e-6 * dhat));
        const double ratio = d_eval / dhat;
        const double dm = d_eval - dhat;
        const double logv = std::log(ratio);
        B = -dm * dm * logv;
        Bp = -2.0 * dm * logv - (dm * dm) / d_eval;
        Bpp = -2.0 * logv - 3.0 + 2.0 * dhat / d_eval + (dhat * dhat) / (d_eval * d_eval);
    };

    auto objective = [&](const SimState& state) {
        if (hasDCDIntersection(state)) return std::numeric_limits<double>::infinity();
        double Etotal = 0.0;
        for (int oid = 0; oid < nobj; ++oid) {
            const Object& obj = objects[oid];
            if (!obj.isDeformable()) continue;
            const int nv = static_cast<int>(state.X[oid].rows());
            const double m = obj.mass / std::max(1, nv);
            Etotal += 0.5 * m / (dt * dt) * (state.X[oid] - Y[oid]).squaredNorm();
            Etotal += computeSpringEnergy(obj, state.X[oid]);
        }
        const auto candidates = collectIPCCandidates(state, contact_thickness);
        for (const IpcCandidate& c : candidates) {
            IpcDistanceInfo info;
            if (!evaluateIPCDistance(c, state, info)) continue;
            if (!std::isfinite(info.distance) || info.distance <= 0.0) return std::numeric_limits<double>::infinity();
            if (info.distance >= info.dhat) continue;
            double B, Bp, Bpp;
            barrier_values(info.distance, info.dhat, B, Bp, Bpp);
            Etotal += contact_stiffness * info.stiffness_scale * info.weight * B;
        }
        return Etotal;
    };

    auto apply_dx = [&](const SimState& state, const Eigen::VectorXd& dx, double alpha) {
        SimState trial = state;
        for (int oid = 0; oid < nobj; ++oid) {
            const Object& obj = objects[oid];
            if (!obj.isDeformable()) continue;
            const int nv = static_cast<int>(trial.X[oid].rows());
            for (int i = 0; i < nv; ++i) {
                const int base = dof_base[oid][i];
                if (base < 0) {
                    if (i < static_cast<int>(obj.fixed_mask.size()) && obj.fixed_mask[i]) {
                        trial.X[oid].row(i) = obj.X0.row(i);
                    }
                    continue;
                }
                trial.X[oid].row(i) += (alpha * dx.segment<3>(base)).transpose();
            }
        }
        return trial;
    };

    for (int iter = 0; iter < implicit_max_newton_iters; ++iter) {
        if (hasDCDIntersection(out_state)) {
            if (enable_debug_output) std::cerr << "[MassSpringWorld:IPC] current state has DCD intersection" << std::endl;
            return false;
        }

        Eigen::VectorXd grad = Eigen::VectorXd::Zero(ndof);
        std::vector<Eigen::Triplet<double>> triplets;
        std::vector<Eigen::Triplet<double>> elastic_triplets;
        triplets.reserve(ndof + 128 * std::max(1, max_ipc_candidates));
        elastic_triplets.reserve(ndof + 64 * std::max(1, ndof / 3));

        for (int oid = 0; oid < nobj; ++oid) {
            const Object& obj = objects[oid];
            if (!obj.isDeformable()) continue;
            const int nv = static_cast<int>(out_state.X[oid].rows());
            const double m = obj.mass / std::max(1, nv);
            const double mass_h2 = m / (dt * dt);
            for (int i = 0; i < nv; ++i) {
                const int base = dof_base[oid][i];
                if (base < 0) continue;
                const Vector3d gi = mass_h2 * (out_state.X[oid].row(i).transpose() - Y[oid].row(i).transpose());
                add_grad(grad, base, gi);
                for (int d = 0; d < 3; ++d) {
                    triplets.emplace_back(base + d, base + d, mass_h2);
                    elastic_triplets.emplace_back(base + d, base + d, mass_h2);
                }
            }

            MatrixXd spring_grad = computeSpringGradient(obj, out_state.X[oid]);
            for (int i = 0; i < nv; ++i) {
                const int base = dof_base[oid][i];
                if (base < 0) continue;
                add_grad(grad, base, spring_grad.row(i).transpose());
            }

            const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
            unsigned edge_id = 0;
            for (const auto& e : obj.E) {
                const int v0 = e.first;
                const int v1 = e.second;
                const Vector3d x0 = out_state.X[oid].row(v0).transpose();
                const Vector3d x1 = out_state.X[oid].row(v1).transpose();
                const Vector3d diff = x0 - x1;
                const double len = diff.norm();
                const double rest_len = obj.E_rest_length[edge_id++];
                if (len <= 1e-12) continue;
                Eigen::Matrix3d Hlocal = obj.stiffness * ((1.0 - rest_len / len) * I + (rest_len / (len * len * len)) * (diff * diff.transpose()));
                if (enable_make_SPD) {
                    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(Hlocal);
                    Eigen::Vector3d evals = es.eigenvalues();
                    for (int k = 0; k < 3; ++k) if (evals[k] < 0.0) evals[k] = 0.0;
                    Hlocal = es.eigenvectors() * evals.asDiagonal() * es.eigenvectors().transpose();
                }
                const int b0 = dof_base[oid][v0];
                const int b1 = dof_base[oid][v1];
                add_block(triplets, b0, b0, Hlocal);
                add_block(triplets, b0, b1, -Hlocal);
                add_block(triplets, b1, b0, -Hlocal);
                add_block(triplets, b1, b1, Hlocal);
                add_block(elastic_triplets, b0, b0, Hlocal);
                add_block(elastic_triplets, b0, b1, -Hlocal);
                add_block(elastic_triplets, b1, b0, -Hlocal);
                add_block(elastic_triplets, b1, b1, Hlocal);
            }
        }

        const auto candidates = collectIPCCandidates(out_state, contact_thickness);
        for (const IpcCandidate& c : candidates) {
            IpcDistanceInfo info;
            if (!evaluateIPCDistance(c, out_state, info)) continue;
            if (!std::isfinite(info.distance) || info.distance <= 0.0 || info.distance >= info.dhat) continue;
            double B, Bp, Bpp;
            barrier_values(info.distance, info.dhat, B, Bp, Bpp);
            const double contact_scale = contact_stiffness * info.stiffness_scale * info.weight;
            const double scale_g = contact_scale * Bp;
            const double scale_H = contact_scale * std::max(0.0, Bpp);
            const Eigen::Matrix3d nnT = info.normal * info.normal.transpose();
            for (const VertexRef& r : info.refs) {
                if (r.object_id < 0 || r.object_id >= nobj) continue;
                if (r.vertex_id < 0 || r.vertex_id >= static_cast<int>(dof_base[r.object_id].size())) continue;
                const int br = dof_base[r.object_id][r.vertex_id];
                if (br < 0) continue;
                add_grad(grad, br, scale_g * r.coeff * info.normal);
            }
            for (const VertexRef& r : info.refs) {
                if (r.object_id < 0 || r.object_id >= nobj) continue;
                if (r.vertex_id < 0 || r.vertex_id >= static_cast<int>(dof_base[r.object_id].size())) continue;
                const int br = dof_base[r.object_id][r.vertex_id];
                if (br < 0) continue;
                for (const VertexRef& sref : info.refs) {
                    if (sref.object_id < 0 || sref.object_id >= nobj) continue;
                    if (sref.vertex_id < 0 || sref.vertex_id >= static_cast<int>(dof_base[sref.object_id].size())) continue;
                    const int bs = dof_base[sref.object_id][sref.vertex_id];
                    if (bs < 0) continue;
                    add_block(triplets, br, bs, scale_H * r.coeff * sref.coeff * nnT);
                }
            }
        }

        for (int i = 0; i < ndof; ++i) {
            triplets.emplace_back(i, i, 1e-8);
            elastic_triplets.emplace_back(i, i, 1e-8);
        }
        const double grad_norm = grad.norm();
        const double grad_scaled = grad_norm / std::sqrt(static_cast<double>(std::max(1, ndof)));
        if (grad_scaled < implicit_grad_tol) {
            break;
        }

        Eigen::SparseMatrix<double> H(ndof, ndof);
        H.setFromTriplets(triplets.begin(), triplets.end());
        H.makeCompressed();

        auto solve_with_triplets = [&](const std::vector<Eigen::Triplet<double>>& base_triplets,
                                       Eigen::VectorXd& direction,
                                       const char* label) {
            Eigen::SparseMatrix<double> A(ndof, ndof);
            A.setFromTriplets(base_triplets.begin(), base_triplets.end());
            A.makeCompressed();
            Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> local_solver;
            local_solver.compute(A);
            if (local_solver.info() != Eigen::Success) {
                std::vector<Eigen::Triplet<double>> retry_triplets = base_triplets;
                for (int i = 0; i < ndof; ++i) retry_triplets.emplace_back(i, i, 1e-6);
                A.setFromTriplets(retry_triplets.begin(), retry_triplets.end());
                A.makeCompressed();
                local_solver.compute(A);
                if (local_solver.info() != Eigen::Success) {
                    if (enable_debug_output) {
                        std::cerr << "[MassSpringWorld:IPC] LDLT failed for " << label << std::endl;
                    }
                    return false;
                }
            }
            direction = local_solver.solve(-grad);
            if (local_solver.info() != Eigen::Success || !direction.allFinite()) {
                if (enable_debug_output) {
                    std::cerr << "[MassSpringWorld:IPC] solve failed for " << label << std::endl;
                }
                return false;
            }
            return true;
        };

        const double E0 = objective(out_state);
        if (!std::isfinite(E0)) {
            if (enable_debug_output) std::cerr << "[MassSpringWorld:IPC] current objective is not finite" << std::endl;
            return false;
        }

        auto try_direction = [&](const Eigen::VectorXd& direction,
                                 const char* label,
                                 SimState& accepted_state,
                                 double& accepted_alpha_out,
                                 int& trials_out) {
            trials_out = 0;
            accepted_alpha_out = 0.0;
            Eigen::VectorXd dx_try = direction;
            double descent_try = grad.dot(dx_try);
            if (!(descent_try < 0.0)) return false;

            SimState full_trial = apply_dx(out_state, dx_try, 1.0);
            double alpha = std::min(1.0, computeCCDSafeStep(out_state, full_trial));
            if (alpha <= 1e-10) return false;

            const double armijo_c = 1e-4;
            const double relative_energy_tol = 1e-9 * (1.0 + std::abs(E0));
            while (alpha > 1e-10) {
                ++trials_out;
                SimState trial_state = apply_dx(out_state, dx_try, alpha);
                if (!hasDCDIntersection(trial_state)) {
                    const double Etrial = objective(trial_state);
                    const double armijo_rhs = E0 + armijo_c * alpha * descent_try + relative_energy_tol;
                    if (std::isfinite(Etrial) && Etrial <= armijo_rhs) {
                        accepted_state = trial_state;
                        accepted_alpha_out = alpha;
                        return true;
                    }
                }
                alpha *= 0.5;
            }

            if (enable_debug_output) {
                std::cerr << "[MassSpringWorld:IPC] direction failed"
                          << ", label=" << label
                          << ", E0=" << E0
                          << ", descent=" << descent_try
                          << ", trials=" << trials_out
                          << std::endl;
            }
            return false;
        };

        Eigen::VectorXd dx;
        if (!solve_with_triplets(triplets, dx, "full IPC Hessian")) {
            return false;
        }

        SimState trial = out_state;
        bool accepted = false;
        double accepted_alpha = 0.0;
        int armijo_trials = 0;

        accepted = try_direction(dx, "full IPC Hessian", trial, accepted_alpha, armijo_trials);

        if (!accepted) {
            Eigen::VectorXd dx_elastic;
            if (solve_with_triplets(elastic_triplets, dx_elastic, "elastic-only Hessian fallback")) {
                accepted = try_direction(dx_elastic, "elastic-only Hessian fallback", trial, accepted_alpha, armijo_trials);
            }
        }


        if (!accepted) {
            if (enable_debug_output) {
                std::cerr << "[MassSpringWorld:IPC] all directions failed"
                          << ", E0=" << E0
                          << ", grad_norm=" << grad_norm
                          << ", grad_scaled=" << grad_scaled
                          << std::endl;
            }
            return false;
        }

        double accepted_step_norm_sq = 0.0;
        for (int oid = 0; oid < nobj; ++oid) {
            const Object& obj = objects[oid];
            if (!obj.isDeformable()) continue;
            const int nv = static_cast<int>(out_state.X[oid].rows());
            for (int i = 0; i < nv; ++i) {
                if (dof_base[oid][i] < 0) continue;
                const Vector3d delta = trial.X[oid].row(i).transpose() - out_state.X[oid].row(i).transpose();
                accepted_step_norm_sq += delta.squaredNorm();
            }
        }
        out_state = trial;
        if (std::sqrt(accepted_step_norm_sq) < implicit_step_tol) break;
    }

    // Self-collision is intentionally not a hard rejection criterion.  However,
    // inter-object non-penetration remains a hard invariant for the next IPC
    // step.  Therefore the self PBD pass is allowed to leave residual self
    // intersections, but it is not allowed to push deformable vertices into
    // external collider meshes.
    SimState pre_self_state = out_state;

    std::vector<Contact> self_contacts = collectSelfPBDContactsSwept(old_state, out_state);
    if (!self_contacts.empty()) {
        projectContactPositions(out_state, self_contacts, kSelfPBDProjectionIters, kSelfPBDRelaxation);

        std::vector<Contact> repair_contacts = collectSelfPBDContactsSwept(old_state, out_state);
        if (!repair_contacts.empty()) {
            projectContactPositions(out_state, repair_contacts, kSelfPBDRepairIters, kSelfPBDRepairRelaxation);
            self_contacts.insert(self_contacts.end(), repair_contacts.begin(), repair_contacts.end());
        }
    }
    last_self_contact_count = static_cast<int>(self_contacts.size());

    // Guard: a self-collision projection may move cloth vertices into a static
    // or kinematic collider.  The inter-object IPC solver assumes the current
    // state is DCD-clean, so before accepting this substep we backtrack only the
    // self-collision correction until inter-object intersections disappear.
    // This deliberately sacrifices some self-collision repair rather than
    // corrupting the hard inter-object invariant.
    if (!self_contacts.empty() && hasDCDIntersection(out_state)) {
        bool recovered = false;
        double lo = 0.0;
        double hi = 1.0;

        // pre_self_state should be inter-object legal because it is the output
        // of the IPC solve.  If it is not legal, there is nothing safe to blend
        // back to, so we leave the state as-is and let the caller handle failure
        // on the next strict IPC attempt.
        if (!hasDCDIntersection(pre_self_state)) {
            for (int k = 0; k < 12; ++k) {
                const double mid = 0.5 * (lo + hi);
                SimState candidate = interpolateState(pre_self_state, out_state, mid);
                if (hasDCDIntersection(candidate)) {
                    hi = mid;
                }
                else {
                    lo = mid;
                    recovered = true;
                }
            }
            out_state = interpolateState(pre_self_state, out_state, lo);

            if (enable_debug_output) {
                std::cerr << "[MassSpringWorld:SELF_PBD] self projection hit external collider; "
                          << "kept " << lo << " of the self-correction" << std::endl;
            }
        }

        if (!recovered && enable_debug_output) {
            std::cerr << "[MassSpringWorld:SELF_PBD] could not find inter-object legal blend after self projection"
                      << std::endl;
        }
    }

    recomputeDeformableVelocities(out_state, old_state, dt);
    if (!self_contacts.empty()) {
        applySelfCollisionVelocityResponse(out_state, self_contacts);
        if (enable_friction_impulse) {
            applyFrictionContacts(out_state, self_contacts);
        }

        // Velocity impulses should not reintroduce an inter-object illegal
        // position, but keep the invariant explicit and easy to debug.
        if (enable_debug_output && hasDCDIntersection(out_state)) {
            std::cerr << "[MassSpringWorld:SELF_PBD] warning: accepted state still has inter-object DCD intersection"
                      << std::endl;
        }
    }
    return true;
}

bool MassSpringWorld::solveBarrierImplicitStep(
    const SimState& old_state,
    SimState& out_state) const
{
    double remaining = h;
    SimState current = old_state;
    const int max_substeps = 8;
    const int max_dt_halvings = 3;
    const double min_dt = 1e-3;
    int substep_count = 0;
    while (remaining > 1e-12 && substep_count < max_substeps) {
        double dt = remaining;
        bool accepted = false;
        for (int retry = 0; retry <= max_dt_halvings; ++retry) {
            if (dt < min_dt) break;
            SimState result;
            if (solveIPCStep(current, dt, result)) {
                current = result;
                remaining -= dt;
                accepted = true;
                recordAcceptedStep(current, dt, substep_count, retry, last_self_contact_count);
                break;
            }
            dt *= 0.5;
        }
        if (!accepted) {
            out_state = current;
            return false;
        }
        ++substep_count;
    }
    out_state = current;
    return remaining <= 1e-12;
}

void MassSpringWorld::applyFrictionContacts(SimState& state, const std::vector<Contact>& contacts) const
{
    if (!enable_friction_impulse) return;
    for (const Contact& c : contacts) {
        if (c.is_intersection) continue;
        Vector3d n = c.normal;
        if (n.norm() < 1e-12) continue;
        n.normalize();
        double W = 0.0;
        Vector3d vrel = Vector3d::Zero();
        for (const VertexRef& r : c.refs) {
            const double w = invMass(r.object_id, r.vertex_id);
            W += r.coeff * r.coeff * w;
            vrel += r.coeff * getVelocityFromState(state, r.object_id, r.vertex_id);
        }
        if (W <= 1e-12) continue;
        const double vn = n.dot(vrel);
        const Vector3d vt = vrel - vn * n;
        const double vt_norm = vt.norm();
        if (vt_norm <= 1e-12) continue;
        const double gap = contactGap(c, state);
        const double dhat = c.target_distance > 0.0 ? c.target_distance : contact_thickness;
        const double proximity = std::clamp((dhat - std::max(0.0, gap)) / std::max(1e-12, dhat), 0.0, 1.0);
        const double strength = std::clamp(c.friction * proximity, 0.0, 1.0);
        if (strength <= 0.0) continue;
        const Vector3d Jt = -(strength / W) * vt;
        for (const VertexRef& r : c.refs) {
            const double w = invMass(r.object_id, r.vertex_id);
            if (w == 0.0) continue;
            addVelocityToState(state, r.object_id, r.vertex_id, w * r.coeff * Jt);
        }
    }
}

void MassSpringWorld::resolveImpulseContacts(SimState& state, const std::vector<Contact>& contacts) const
{
    for (const Contact& c : contacts) {
        double W = 0.0;
        Vector3d vrel = Vector3d::Zero();
        for (const VertexRef& r : c.refs) {
            const double w = invMass(r.object_id, r.vertex_id);
            W += r.coeff * r.coeff * w;
            vrel += r.coeff * getVelocityFromState(state, r.object_id, r.vertex_id);
        }
        if (W <= 1e-12) continue;

        Vector3d n = c.normal;
        if (n.norm() < 1e-12) continue;
        n.normalize();

        double vn = n.dot(vrel);
        if (vn > 0.0) {
            n = -n;
            vn = -vn;
        }
        if (vn >= 0.0) continue;

        const double Jn = -(1.0 + c.restitution) * vn / W;
        if (Jn < 0.0) continue;

        for (const VertexRef& r : c.refs) {
            const double w = invMass(r.object_id, r.vertex_id);
            addVelocityToState(state, r.object_id, r.vertex_id, w * r.coeff * Jn * n);
        }

        Vector3d vrel_after = Vector3d::Zero();
        for (const VertexRef& r : c.refs) {
            vrel_after += r.coeff * getVelocityFromState(state, r.object_id, r.vertex_id);
        }
        const Vector3d vt = vrel_after - n.dot(vrel_after) * n;
        const double vt_norm = vt.norm();
        if (vt_norm <= 1e-12) continue;

        Vector3d Jt_need = -(1.0 / W) * vt;
        const double max_friction = c.friction * Jn;
        Vector3d Jt = Jt_need;
        if (Jt.norm() > max_friction) {
            Jt = -max_friction * vt / vt_norm;
        }

        for (const VertexRef& r : c.refs) {
            const double w = invMass(r.object_id, r.vertex_id);
            addVelocityToState(state, r.object_id, r.vertex_id, w * r.coeff * Jt);
        }
    }
}

void MassSpringWorld::projectContactPositions(
    SimState& state,
    const std::vector<Contact>& contacts,
    int projection_iters,
    double relaxation) const
{
    const double extra_slop = 1e-5;
    projection_iters = std::max(0, projection_iters);
    relaxation = std::clamp(relaxation, 0.0, 1.0);
    for (int iter = 0; iter < projection_iters; ++iter) {
        for (const Contact& c : contacts) {
            double W = 0.0;
            Vector3d relative_position = Vector3d::Zero();
            for (const VertexRef& r : c.refs) {
                const double w = invMass(r.object_id, r.vertex_id);
                W += r.coeff * r.coeff * w;
                relative_position += r.coeff * state.X[r.object_id].row(r.vertex_id).transpose();
            }
            if (W <= 1e-12) continue;

            Vector3d n = c.normal;
            if (n.norm() < 1e-12) continue;
            n.normalize();

            // PBD self-collision contacts already store the intended safe-side normal.
            // Do not flip it by velocity here, otherwise a crossed vertex can be pushed
            // to the wrong side after tunneling.
            const double C = n.dot(relative_position) - (c.target_distance + extra_slop);
            if (C >= 0.0) continue;

            const double lambda = -C / W;
            for (const VertexRef& r : c.refs) {
                const double w = invMass(r.object_id, r.vertex_id);
                if (w == 0.0) continue;
                state.X[r.object_id].row(r.vertex_id) +=
                    (relaxation * w * r.coeff * lambda * n).transpose();
            }
        }
    }
}


void MassSpringWorld::applySelfCollisionVelocityResponse(
    SimState& state,
    const std::vector<Contact>& contacts) const
{
    for (const Contact& c : contacts) {
        if (!c.is_self_contact) continue;
        double W = 0.0;
        Vector3d vrel = Vector3d::Zero();
        for (const VertexRef& r : c.refs) {
            const double w = invMass(r.object_id, r.vertex_id);
            W += r.coeff * r.coeff * w;
            vrel += r.coeff * getVelocityFromState(state, r.object_id, r.vertex_id);
        }
        if (W <= 1e-12) continue;

        Vector3d n = c.normal;
        if (n.norm() <= 1e-12) continue;
        n.normalize();

        const double vn = n.dot(vrel);
        if (vn >= 0.0) continue;

        // Cloth-like self-collision should remove the approaching normal
        // component, not bounce.  Bouncing injects energy and causes worm-like
        // wriggling when many internal contacts are active.
        const double Jn = -vn / W;
        if (Jn <= 0.0) continue;

        for (const VertexRef& r : c.refs) {
            const double w = invMass(r.object_id, r.vertex_id);
            if (w == 0.0) continue;
            addVelocityToState(state, r.object_id, r.vertex_id, w * r.coeff * Jn * n);
        }
    }
}

void MassSpringWorld::recomputeDeformableVelocities(
    SimState& state,
    const SimState& old_state,
    double dt) const
{
    if (dt <= 1e-12) return;
    const int nobj = std::min(static_cast<int>(objects.size()), static_cast<int>(state.X.size()));
    for (int oid = 0; oid < nobj; ++oid) {
        const Object& obj = objects[oid];
        if (!obj.isDeformable()) continue;
        if (oid >= static_cast<int>(old_state.X.size())) continue;
        if (state.X[oid].rows() != old_state.X[oid].rows()) continue;
        state.V[oid] = (state.X[oid] - old_state.X[oid]) / dt;
        for (int i = 0; i < state.V[oid].rows(); ++i) {
            if (i < static_cast<int>(obj.fixed_mask.size()) && obj.fixed_mask[i]) {
                state.V[oid].row(i).setZero();
            }
        }
    }
}

void MassSpringWorld::step()
{
    if (objects.empty()) return;

    SimState old_state = captureState();

    if (mesh_collision_mode == MESH_COLLISION_NONE) {
        SimState trial = predictState(old_state, h);
        applyGroundResponse(trial);
        recordAcceptedStep(trial, h, 0, 0, 0);
        applyState(trial);
        return;
    }

    SimState solved;
    const bool ok = solveBarrierImplicitStep(old_state, solved);
    if (!ok) {
        for (int oid = 0; oid < static_cast<int>(objects.size()); ++oid) {
            if (objects[oid].isDeformable()) solved.V[oid].setZero();
        }
        std::cerr << "[MassSpringWorld:IPC_STOP] all Armijo/CCD searches failed; "
                  << "simulation stopped at the last legal state" << std::endl;
    }

    applyGroundResponse(solved);

    if (enable_friction_impulse) {
        CollisionReport report = detectContacts(solved);
        applyFrictionContacts(solved, report.contacts);
    }

    overwriteLastRecordedState(solved, true);

    applyState(solved);
}

} // namespace USTC_CG::mass_spring
