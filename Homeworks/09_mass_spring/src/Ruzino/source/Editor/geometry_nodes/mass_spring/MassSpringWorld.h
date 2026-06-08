#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <cstdint>
#include <limits>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "MassSpring.h"
#include "HW9WorldObject.h"

namespace USTC_CG::mass_spring {

class MassSpringWorld {
public:
    using MatrixXd = Eigen::MatrixXd;
    using Vector3d = Eigen::Vector3d;
    using Vector3i = Eigen::Vector3i;

    enum MeshCollisionMode {
        MESH_COLLISION_NONE = 0,
        MESH_BARRIER_CONTACT = 1
    };

    struct Object {
        std::string name = "object";
        int object_type = DYNAMIC_DEFORMABLE;
        int object_id = -1;

        MatrixXd X;
        MatrixXd X0;
        MatrixXd V;
        std::vector<Vector3i> F;
        EdgeSet E;
        std::vector<double> E_rest_length;
        std::vector<double> vertex_area;
        std::vector<bool> fixed_mask;

        double mass = 1.0;
        double stiffness = 1000.0;
        double damping = 0.995;
        double restitution = 0.0;
        double friction = 0.3;

        bool isDeformable() const { return object_type == DYNAMIC_DEFORMABLE; }
        bool isKinematic() const { return object_type == KINEMATIC_MESH; }
        bool isStaticCollider() const { return object_type == STATIC_COLLIDER; }
        bool isColliderOnly() const { return object_type != DYNAMIC_DEFORMABLE; }
    };

    struct SimState {
        std::vector<MatrixXd> X;
        std::vector<MatrixXd> V;
    };

    struct AABB {
        Vector3d mn = Vector3d::Zero();
        Vector3d mx = Vector3d::Zero();
    };

    struct FaceRef {
        int object_id = -1;
        int face_id = -1;
        Vector3i v = Vector3i::Zero();
        AABB box;
    };

    struct VertexRef {
        int object_id = -1;
        int vertex_id = -1;
        double coeff = 0.0;
    };

    struct Contact {
        std::vector<VertexRef> refs;
        Vector3d point = Vector3d::Zero();
        Vector3d normal = Vector3d::UnitZ();
        // Linearized contact distance used by barrier: d(x)=n^T sum_i coeff_i x_i.
        // contact_thickness is the activation distance; this field is not subtracted in the barrier.
        double target_distance = 0.0;
        bool is_intersection = false;
        double restitution = 0.0;
        double friction = 0.3;
        bool is_self_contact = false;
        double dhat = 0.0;
        double stiffness_scale = 1.0;
    };

    struct CollisionReport {
        int candidate_pairs = 0;
        int intersection_count = 0; // number of generated contacts
        std::vector<Contact> contacts;
    };

    struct StepSnapshot {
        double time = 0.0;          // cumulative simulation time after this accepted update
        double dt = 0.0;            // actual accepted substep length
        int substep_index = 0;      // index inside the current world.step() call
        int retry_count = 0;        // how many dt halvings were needed before acceptance
        int self_contact_count = 0; // predictive self-PBD contacts generated in this substep
        bool post_response_applied = false; // true if this snapshot was overwritten after ground/friction
        std::vector<MatrixXd> X;
        std::vector<MatrixXd> V;
    };

    // Global parameters.
    double h = 1e-2;
    Vector3d gravity = Vector3d(0.0, 0.0, -9.8);
    Vector3d wind_ext_acc = Vector3d::Zero();

    bool enable_ground = true;
    double ground_z = -1.0;
    double ground_restitution = 0.0;
    double ground_friction = 0.5;

    MeshCollisionMode mesh_collision_mode = MESH_BARRIER_CONTACT;
    double collision_cell_size = 0.0;
    double collision_aabb_margin = 1e-8;
    double contact_thickness = 0.03;
    int max_contacts_for_impulse = 512;
    int max_ipc_candidates = 20000;
    bool enable_self_collision = true;
    bool enable_friction_impulse = true;
    bool enable_fixed_points = false;
    double contact_stiffness = 10000.0;
    double self_contact_stiffness_scale = 0.35;
    double self_contact_thickness_scale = 0.35;
    bool enable_debug_output = false;

    // Elastic time integration for deformable objects.
    // World mode always uses the implicit elastic predictor; these are internal constants.
    bool enable_implicit_elastic = true;
    bool enable_make_SPD = true;
    int implicit_max_newton_iters = 20;
    double implicit_grad_tol = 1e-6;
    double implicit_step_tol = 1e-8;

    int addObject(
        const MatrixXd& X,
        const std::vector<Vector3i>& F,
        const EdgeSet& E,
        int object_type,
        const std::string& name = "object");

    void clear();
    void step();

    void beginStepRecording(double start_time);
    void disableStepRecording();
    const std::vector<StepSnapshot>& getStepSnapshots() const { return step_snapshots; }

    const std::vector<Object>& getObjects() const { return objects; }
    std::vector<Object>& getObjects() { return objects; }

private:
    std::vector<Object> objects;

    mutable bool step_recording_enabled = false;
    mutable double step_recording_time = 0.0;
    mutable int last_self_contact_count = 0;
    mutable std::vector<StepSnapshot> step_snapshots;

    void recordAcceptedStep(
        const SimState& state,
        double dt,
        int substep_index,
        int retry_count,
        int self_contact_count) const;
    void overwriteLastRecordedState(const SimState& state, bool post_response_applied) const;

    struct CellIndex {
        int x = 0, y = 0, z = 0;
        bool operator==(const CellIndex& other) const {
            return x == other.x && y == other.y && z == other.z;
        }
    };

    struct CellHash {
        std::size_t operator()(const CellIndex& c) const {
            std::size_t hx = std::hash<int>{}(c.x * 73856093);
            std::size_t hy = std::hash<int>{}(c.y * 19349663);
            std::size_t hz = std::hash<int>{}(c.z * 83492791);
            return hx ^ (hy << 1) ^ (hz << 2);
        }
    };

    using HashGrid = std::unordered_map<CellIndex, std::vector<int>, CellHash>;

    struct TOIReport {
        bool hit = false;
        double alpha = 1.0;
        int candidate_pairs = 0;
        std::vector<Contact> contacts;
    };

    enum IpcCandidateType { IPC_VERTEX_TRIANGLE = 0, IPC_EDGE_EDGE = 1 };

    struct IpcCandidate {
        IpcCandidateType type = IPC_VERTEX_TRIANGLE;
        int object_id[4] = {-1, -1, -1, -1};
        int vertex_id[4] = {-1, -1, -1, -1};
        double friction = 0.3;
        bool is_self_contact = false;
        double dhat = 0.0;
        double stiffness_scale = 1.0;
    };

    struct IpcDistanceInfo {
        double distance = std::numeric_limits<double>::infinity();
        double weight = 1.0;
        double dhat = 0.0;
        double stiffness_scale = 1.0;
        bool is_self_contact = false;
        std::vector<VertexRef> refs;
        Vector3d normal = Vector3d::UnitZ();
    };

    SimState captureState() const;
    void applyState(const SimState& state);
    SimState interpolateState(const SimState& a, const SimState& b, double alpha) const;
    SimState predictState(const SimState& start, double dt) const;

    double computeSpringEnergy(const Object& obj, const MatrixXd& X_eval) const;
    MatrixXd computeSpringGradient(const Object& obj, const MatrixXd& X_eval) const;
    Eigen::SparseMatrix<double> computeSpringHessianSparse(
        const Object& obj,
        const MatrixXd& X_eval) const;
    bool solveBarrierImplicitStep(const SimState& old_state, SimState& out_state) const;
    bool solveIPCStep(const SimState& old_state, double dt, SimState& out_state) const;
    double contactGap(const Contact& contact, const SimState& state) const;
    bool hasInvalidContact(const SimState& state, const std::vector<Contact>& contacts, double eps = 1e-8) const;
    void applyFrictionContacts(SimState& state, const std::vector<Contact>& contacts) const;
    std::vector<Contact> collectSelfPBDContactsSwept(
        const SimState& old_state,
        const SimState& trial_state) const;
    std::vector<IpcCandidate> collectIPCCandidates(const SimState& state, double dhat) const;
    bool evaluateIPCDistance(const IpcCandidate& candidate, const SimState& state, IpcDistanceInfo& info) const;
    bool hasDCDIntersection(const SimState& state) const;
    double computeCCDSafeStep(const SimState& from, const SimState& to) const;
    std::vector<double> solvePolynomialRealRoots(double a, double b, double c, double d) const;
    bool vertexTriangleCCD(const SimState& from, const SimState& to, int p_obj, int p_vid, int tri_obj, const Vector3i& tri, double& alpha) const;
    bool edgeEdgeCCD(const SimState& from, const SimState& to, int a_obj, int a0, int a1, int b_obj, int b0, int b1, double& alpha) const;
    MatrixXd semiImplicitElasticStepObject(
        const Object& obj,
        const MatrixXd& X_old,
        const MatrixXd& V_old,
        double dt,
        MatrixXd& V_new) const;
    MatrixXd implicitElasticStepObject(
        const Object& obj,
        const MatrixXd& X_old,
        const MatrixXd& V_old,
        double dt,
        MatrixXd& V_new) const;
    void applyGroundResponse(SimState& state) const;

    double invMass(int object_id, int vertex_id) const;
    Vector3d getVelocityFromState(const SimState& state, int object_id, int vertex_id) const;
    void addVelocityToState(SimState& state, int object_id, int vertex_id, const Vector3d& dv) const;
    void recomputeDeformableVelocities(SimState& state, const SimState& old_state, double dt) const;

    double computeAutomaticCellSize(const SimState& state) const;
    AABB computeTriangleAABB(const MatrixXd& X_eval, const Vector3i& f, double margin) const;
    AABB computeSweptTriangleAABB(const SimState& old_state, const SimState& trial_state, int object_id, const Vector3i& f, double margin) const;
    bool overlapAABB(const AABB& a, const AABB& b) const;
    CellIndex pointToCell(const Vector3d& p, double cell_size) const;

    std::vector<FaceRef> collectFaces(const SimState& state, double margin) const;
    std::vector<FaceRef> collectSweptFaces(const SimState& old_state, const SimState& trial_state, double margin) const;
    void buildSpatialHash(const std::vector<FaceRef>& faces, double cell_size, HashGrid& grid) const;
    bool shouldSkipPair(const FaceRef& a, const FaceRef& b) const;
    bool shareVertex(const Vector3i& a, const Vector3i& b) const;
    std::uint64_t pairKey(int a, int b) const;

    bool segmentTriangleIntersection(
        const Vector3d& p0,
        const Vector3d& p1,
        const Vector3d& a,
        const Vector3d& b,
        const Vector3d& c,
        double& seg_t,
        Vector3d& bary,
        Vector3d& point) const;

    bool closestPointTriangle(
        const Vector3d& p,
        const Vector3d& a,
        const Vector3d& b,
        const Vector3d& c,
        Vector3d& closest,
        Vector3d& bary) const;

    bool closestSegmentSegment(
        const Vector3d& p0,
        const Vector3d& p1,
        const Vector3d& q0,
        const Vector3d& q1,
        double& s,
        double& t,
        Vector3d& cp,
        Vector3d& cq) const;

    bool edgeTriangleContact(
        const FaceRef& edge_face,
        int edge_local_a,
        int edge_local_b,
        const FaceRef& tri_face,
        const SimState& state,
        Contact& contact) const;
    bool vertexTriangleContact(
        const FaceRef& vertex_face,
        int vertex_local,
        const FaceRef& tri_face,
        const SimState& state,
        Contact& contact) const;
    bool edgeEdgeContact(
        const FaceRef& face_a,
        int a0_local,
        int a1_local,
        const FaceRef& face_b,
        int b0_local,
        int b1_local,
        const SimState& state,
        Contact& contact) const;
    void collectTrianglePairContacts(
        const FaceRef& a,
        const FaceRef& b,
        const SimState& state,
        std::vector<Contact>& contacts) const;

    CollisionReport detectContacts(const SimState& state) const;
    TOIReport findEarliestTOIByBisection(const SimState& old_state, const SimState& trial_state) const;
    int automaticBisectionIters() const;
    void resolveImpulseContacts(SimState& state, const std::vector<Contact>& contacts) const;
    void projectContactPositions(
        SimState& state,
        const std::vector<Contact>& contacts,
        int projection_iters,
        double relaxation) const;
    void applySelfCollisionVelocityResponse(SimState& state, const std::vector<Contact>& contacts) const;
};

} // namespace USTC_CG::mass_spring
