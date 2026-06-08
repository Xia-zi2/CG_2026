#include <Eigen/Sparse>

#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include "GCore/Components/MeshComponent.h"
#include "GCore/util_openmesh_bind.h"
#include "geom_node_base.h"
#include "mass_spring/HW9WorldObject.h"
#include "mass_spring/MassSpringWorld.h"
#include "mass_spring/utils.h"

namespace {
using Eigen::Vector3i;
using USTC_CG::mass_spring::HW9WorldObject;
using USTC_CG::mass_spring::HW9WorldObjectList;
using USTC_CG::mass_spring::MassSpringWorld;

// Keep the node UI compact.  The following values are intentionally
// fixed here because they are implementation details rather than user-facing
// modeling controls.
constexpr double kDefaultWindX = 0.0;
constexpr double kDefaultWindY = 0.0;
constexpr double kDefaultWindZ = 0.0;
constexpr double kDefaultGroundRestitution = 0.0;
constexpr double kDefaultGroundFriction = 0.5;
constexpr double kSelfContactStiffnessScale = 0.35;
constexpr double kSelfContactThicknessScale = 0.35;
constexpr int kMaxContactCandidates = 20000;
constexpr int kImplicitMaxNewtonIters = 20;
constexpr double kImplicitGradTol = 1e-6;
constexpr double kImplicitStepTol = 1e-8;

std::vector<Vector3i> to_face_vector(const Eigen::MatrixXi& F)
{
    std::vector<Vector3i> out;
    out.reserve(F.rows());
    for (int i = 0; i < F.rows(); ++i) {
        if (F.cols() >= 3) out.emplace_back(F(i, 0), F(i, 1), F(i, 2));
    }
    return out;
}

std::vector<Vector3i> to_face_vector(const std::vector<Vector3i>& F) { return F; }

std::vector<Vector3i> to_face_vector(const std::vector<std::array<int, 3>>& F)
{
    std::vector<Vector3i> out;
    out.reserve(F.size());
    for (const auto& f : F) out.emplace_back(f[0], f[1], f[2]);
    return out;
}

std::vector<Vector3i> to_face_vector(const std::vector<std::vector<int>>& F)
{
    std::vector<Vector3i> out;
    out.reserve(F.size());
    for (const auto& f : F) {
        if (f.size() >= 3) out.emplace_back(f[0], f[1], f[2]);
    }
    return out;
}

static Ruzino::Geometry merge_object_list_geometry(
    const HW9WorldObjectList& simulated,
    const HW9WorldObjectList& collider)
{
    Ruzino::Geometry merged_geometry;
    auto merged_mesh = std::make_shared<Ruzino::MeshComponent>(&merged_geometry);
    bool has_mesh = false;

    auto append_list = [&](const HW9WorldObjectList& list) {
        for (const auto& object : list.objects) {
            auto geometry = object.geometry;
            geometry.apply_transform();
            auto mesh_component = geometry.get_component<Ruzino::MeshComponent>();
            if (mesh_component) {
                merged_mesh->append_mesh(mesh_component);
                has_mesh = true;
            }
        }
    };

    append_list(simulated);
    append_list(collider);

    if (has_mesh) merged_geometry.attach_component(merged_mesh);
    return merged_geometry;
}

static std::vector<glm::vec3> eigen_vel_to_glm(const Eigen::MatrixXd& V)
{
    std::vector<glm::vec3> out;
    out.reserve(V.rows());
    for (int i = 0; i < V.rows(); ++i) {
        out.emplace_back(static_cast<float>(V(i, 0)), static_cast<float>(V(i, 1)), static_cast<float>(V(i, 2)));
    }
    return out;
}

static Eigen::MatrixXd glm_vel_to_eigen(const std::vector<glm::vec3>& v, int n)
{
    Eigen::MatrixXd V = Eigen::MatrixXd::Zero(n, 3);
    if (static_cast<int>(v.size()) != n) return V;
    for (int i = 0; i < n; ++i) {
        V(i, 0) = v[i].x;
        V(i, 1) = v[i].y;
        V(i, 2) = v[i].z;
    }
    return V;
}

static bool extract_mesh_data(
    Ruzino::Geometry geometry,
    Eigen::MatrixXd& vertices,
    std::vector<Vector3i>& faces,
    USTC_CG::mass_spring::EdgeSet& edges)
{
    geometry.apply_transform();
    auto mesh = geometry.get_component<Ruzino::MeshComponent>();
    if (!mesh || mesh->get_face_vertex_counts().empty()) return false;

    auto faces_any = USTC_CG::mass_spring::usd_faces_to_eigen(
        mesh->get_face_vertex_counts(),
        mesh->get_face_vertex_indices());
    faces = to_face_vector(faces_any);
    edges = USTC_CG::mass_spring::get_edges(faces_any);
    vertices = USTC_CG::mass_spring::usd_vertices_to_eigen(mesh->get_vertices());
    return vertices.rows() > 0 && !faces.empty();
}

} // namespace

struct MassSpringWorldStorage {
    constexpr static bool has_storage = false;
    HW9WorldObjectList output_simulated_list;
    HW9WorldObjectList last_collider_input;
    std::vector<Eigen::MatrixXd> simulated_rest_X;
    std::vector<std::vector<double>> simulated_rest_lengths;
    std::vector<std::vector<double>> simulated_rest_areas;

    bool dataset_initialized = false;
    std::filesystem::path dataset_dir;
    double dataset_time = 0.0;
    std::int64_t dataset_update_count = 0;
    int dataset_num_points = 0;
    int dataset_num_objects = 0;
};

static std::vector<double> compute_rest_lengths(
    const Eigen::MatrixXd& X,
    const USTC_CG::mass_spring::EdgeSet& edges)
{
    std::vector<double> lengths;
    lengths.reserve(edges.size());
    for (const auto& e : edges) {
        Eigen::Vector3d x0 = X.row(e.first).transpose();
        Eigen::Vector3d x1 = X.row(e.second).transpose();
        lengths.push_back((x0 - x1).norm());
    }
    return lengths;
}

static std::vector<double> compute_vertex_lumped_areas(
    const Eigen::MatrixXd& X,
    const std::vector<Eigen::Vector3i>& faces)
{
    std::vector<double> areas(static_cast<size_t>(std::max(0, static_cast<int>(X.rows()))), 0.0);
    for (const auto& f : faces) {
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


static std::string json_escape(const std::string& s)
{
    std::ostringstream os;
    for (char ch : s) {
        switch (ch) {
        case '\\': os << "\\\\"; break;
        case '"': os << "\\\""; break;
        case '\n': os << "\\n"; break;
        case '\r': os << "\\r"; break;
        case '\t': os << "\\t"; break;
        default: os << ch; break;
        }
    }
    return os.str();
}

static std::filesystem::path executable_directory()
{
#ifdef _WIN32
    wchar_t buffer[MAX_PATH];
    const DWORD len = GetModuleFileNameW(nullptr, buffer, MAX_PATH);
    if (len > 0) {
        return std::filesystem::path(std::wstring(buffer, buffer + len)).parent_path();
    }
#endif
    return std::filesystem::current_path();
}

static std::string timestamp_string()
{
    const auto now = std::chrono::system_clock::now();
    const std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tmv{};
#ifdef _WIN32
    localtime_s(&tmv, &tt);
#else
    localtime_r(&tt, &tmv);
#endif
    std::ostringstream os;
    os << std::put_time(&tmv, "%Y%m%d_%H%M%S");
    return os.str();
}

static int dataset_object_type(const MassSpringWorld::Object& obj)
{
    return obj.isDeformable() ? 0 : 1;
}

static bool dataset_fixed_mask(const MassSpringWorld::Object& obj, int vid)
{
    if (!obj.isDeformable()) return true;
    return vid >= 0 &&
           vid < static_cast<int>(obj.fixed_mask.size()) &&
           obj.fixed_mask[vid];
}

static int dataset_total_points(const MassSpringWorld& world)
{
    int n = 0;
    for (const auto& obj : world.getObjects()) {
        n += static_cast<int>(obj.X.rows());
    }
    return n;
}

static std::vector<int> dataset_vertex_offsets(const MassSpringWorld& world)
{
    std::vector<int> offsets;
    offsets.reserve(world.getObjects().size() + 1);
    int cursor = 0;
    offsets.push_back(cursor);
    for (const auto& obj : world.getObjects()) {
        cursor += static_cast<int>(obj.X.rows());
        offsets.push_back(cursor);
    }
    return offsets;
}

static void write_float32(std::ofstream& os, double value)
{
    const float v = static_cast<float>(value);
    os.write(reinterpret_cast<const char*>(&v), sizeof(float));
}

static void write_int32(std::ofstream& os, int value)
{
    const std::int32_t v = static_cast<std::int32_t>(value);
    os.write(reinterpret_cast<const char*>(&v), sizeof(std::int32_t));
}

static void write_dataset_meta(
    MassSpringWorldStorage& storage,
    const MassSpringWorld& world,
    bool enable_mesh_collision,
    bool enable_self_collision,
    bool enable_friction_impulse,
    bool enable_fixed_points,
    bool enable_damping)
{
    const auto& objs = world.getObjects();
    const auto offsets = dataset_vertex_offsets(world);

    std::ofstream os(storage.dataset_dir / "meta.json", std::ios::out | std::ios::trunc);
    os << std::setprecision(17);
    os << "{\n";
    os << "  \"format_version\": 2,\n";
    os << "  \"storage\": \"raw_binary_plus_jsonl\",\n";
    os << "  \"states_file\": \"states_f32.bin\",\n";
    os << "  \"states_layout\": \"float32 [num_updates, num_points, 6] where channels=(x,y,z,vx,vy,vz)\",\n";
    os << "  \"updates_file\": \"updates.jsonl\",\n";
    os << "  \"time_is_variable\": true,\n";
    os << "  \"time_unit\": \"simulation_seconds\",\n";
    os << "  \"num_updates\": " << storage.dataset_update_count << ",\n";
    os << "  \"num_points\": " << storage.dataset_num_points << ",\n";
    os << "  \"num_objects\": " << objs.size() << ",\n";
    os << "  \"h0\": " << world.h << ",\n";
    os << "  \"gravity\": [" << world.gravity.x() << ", " << world.gravity.y() << ", " << world.gravity.z() << "],\n";
    os << "  \"wind\": [" << world.wind_ext_acc.x() << ", " << world.wind_ext_acc.y() << ", " << world.wind_ext_acc.z() << "],\n";
    os << "  \"ground_enabled\": " << (world.enable_ground ? "true" : "false") << ",\n";
    os << "  \"ground_z\": " << world.ground_z << ",\n";

    os << "  \"object_names\": [";
    for (int i = 0; i < static_cast<int>(objs.size()); ++i) {
        if (i) os << ", ";
        os << "\"" << json_escape(objs[i].name) << "\"";
    }
    os << "],\n";

    os << "  \"object_types\": [";
    for (int i = 0; i < static_cast<int>(objs.size()); ++i) {
        if (i) os << ", ";
        os << dataset_object_type(objs[i]);
    }
    os << "],\n";

    os << "  \"object_raw_types\": [";
    for (int i = 0; i < static_cast<int>(objs.size()); ++i) {
        if (i) os << ", ";
        os << objs[i].object_type;
    }
    os << "],\n";

    os << "  \"stiffness\": [";
    for (int i = 0; i < static_cast<int>(objs.size()); ++i) {
        if (i) os << ", ";
        os << (objs[i].isDeformable() ? objs[i].stiffness : 1.0);
    }
    os << "],\n";

    os << "  \"object_vertex_ranges\": [";
    for (int i = 0; i < static_cast<int>(objs.size()); ++i) {
        if (i) os << ", ";
        os << "[" << offsets[i] << ", " << offsets[i + 1] << "]";
    }
    os << "],\n";

    os << "  \"binary_files\": {\n";
    os << "    \"x0_f32.bin\": \"float32 [num_points, 3]\",\n";
    os << "    \"point_info_i32.bin\": \"int32 [num_points, 4] columns=(object_id, object_type, object_raw_type, fixed_mask)\",\n";
    os << "    \"stiffness_f32.bin\": \"float32 [num_objects]\",\n";
    os << "    \"faces_i32.bin\": \"int32 [num_faces, 4] columns=(object_id, global_i, global_j, global_k)\"\n";
    os << "  },\n";

    os << "  \"solver_params\": {\n";
    os << "    \"enable_mesh_collision\": " << (enable_mesh_collision ? "true" : "false") << ",\n";
    os << "    \"enable_self_collision\": " << (enable_self_collision ? "true" : "false") << ",\n";
    os << "    \"enable_friction_impulse\": " << (enable_friction_impulse ? "true" : "false") << ",\n";
    os << "    \"enable_fixed_points\": " << (enable_fixed_points ? "true" : "false") << ",\n";
    os << "    \"enable_damping\": " << (enable_damping ? "true" : "false") << ",\n";
    os << "    \"contact_thickness\": " << world.contact_thickness << ",\n";
    os << "    \"contact_stiffness\": " << world.contact_stiffness << ",\n";
    os << "    \"self_contact_thickness_scale\": " << world.self_contact_thickness_scale << ",\n";
    os << "    \"self_contact_stiffness_scale\": " << world.self_contact_stiffness_scale << ",\n";
    os << "    \"max_ipc_candidates\": " << world.max_ipc_candidates << ",\n";
    os << "    \"implicit_max_newton_iters\": " << world.implicit_max_newton_iters << ",\n";
    os << "    \"implicit_grad_tol\": " << world.implicit_grad_tol << ",\n";
    os << "    \"implicit_step_tol\": " << world.implicit_step_tol << "\n";
    os << "  }\n";
    os << "}\n";
}

static void write_dataset_static_files(MassSpringWorldStorage& storage, const MassSpringWorld& world)
{
    const auto& objs = world.getObjects();
    const auto offsets = dataset_vertex_offsets(world);

    {
        std::ofstream os(storage.dataset_dir / "x0_f32.bin", std::ios::binary | std::ios::trunc);
        for (const auto& obj : objs) {
            const Eigen::MatrixXd& X0 = (obj.X0.rows() == obj.X.rows()) ? obj.X0 : obj.X;
            for (int i = 0; i < X0.rows(); ++i) {
                write_float32(os, X0(i, 0));
                write_float32(os, X0(i, 1));
                write_float32(os, X0(i, 2));
            }
        }
    }

    {
        std::ofstream os(storage.dataset_dir / "point_info_i32.bin", std::ios::binary | std::ios::trunc);
        for (int oid = 0; oid < static_cast<int>(objs.size()); ++oid) {
            const auto& obj = objs[oid];
            for (int i = 0; i < obj.X.rows(); ++i) {
                write_int32(os, oid);
                write_int32(os, dataset_object_type(obj));
                write_int32(os, obj.object_type);
                write_int32(os, dataset_fixed_mask(obj, i) ? 1 : 0);
            }
        }
    }

    {
        std::ofstream os(storage.dataset_dir / "stiffness_f32.bin", std::ios::binary | std::ios::trunc);
        for (const auto& obj : objs) {
            write_float32(os, obj.isDeformable() ? obj.stiffness : 1.0);
        }
    }

    {
        std::ofstream os(storage.dataset_dir / "faces_i32.bin", std::ios::binary | std::ios::trunc);
        for (int oid = 0; oid < static_cast<int>(objs.size()); ++oid) {
            const auto& obj = objs[oid];
            const int base = offsets[oid];
            for (const auto& f : obj.F) {
                write_int32(os, oid);
                write_int32(os, base + f[0]);
                write_int32(os, base + f[1]);
                write_int32(os, base + f[2]);
            }
        }
    }
}

static void append_dataset_state(
    MassSpringWorldStorage& storage,
    const std::vector<Eigen::MatrixXd>& X,
    const std::vector<Eigen::MatrixXd>& V,
    double time,
    double dt,
    const std::string& source,
    int substep_index,
    int retry_count,
    int self_contact_count,
    bool post_response_applied)
{
    const std::int64_t update_id = storage.dataset_update_count;
    const std::int64_t offset_floats =
        update_id * static_cast<std::int64_t>(storage.dataset_num_points) * 6;

    {
        std::ofstream os(storage.dataset_dir / "states_f32.bin", std::ios::binary | std::ios::app);
        for (int oid = 0; oid < static_cast<int>(X.size()); ++oid) {
            const Eigen::MatrixXd& Xi = X[oid];
            const Eigen::MatrixXd& Vi = (oid < static_cast<int>(V.size())) ? V[oid] : Xi;
            for (int i = 0; i < Xi.rows(); ++i) {
                write_float32(os, Xi(i, 0));
                write_float32(os, Xi(i, 1));
                write_float32(os, Xi(i, 2));
                if (Vi.rows() == Xi.rows()) {
                    write_float32(os, Vi(i, 0));
                    write_float32(os, Vi(i, 1));
                    write_float32(os, Vi(i, 2));
                }
                else {
                    write_float32(os, 0.0);
                    write_float32(os, 0.0);
                    write_float32(os, 0.0);
                }
            }
        }
    }

    {
        std::ofstream os(storage.dataset_dir / "updates.jsonl", std::ios::out | std::ios::app);
        os << std::setprecision(17)
           << "{"
           << "\"update\":" << update_id << ","
           << "\"time\":" << time << ","
           << "\"dt\":" << dt << ","
           << "\"source\":\"" << json_escape(source) << "\","
           << "\"substep_index\":" << substep_index << ","
           << "\"retry_count\":" << retry_count << ","
           << "\"self_contact_count\":" << self_contact_count << ","
           << "\"post_response_applied\":" << (post_response_applied ? "true" : "false") << ","
           << "\"state_offset_floats\":" << offset_floats
           << "}\n";
    }

    storage.dataset_update_count += 1;
    storage.dataset_time = time;
}

static void append_dataset_state_from_world(
    MassSpringWorldStorage& storage,
    const MassSpringWorld& world,
    double time,
    double dt,
    const std::string& source)
{
    std::vector<Eigen::MatrixXd> X;
    std::vector<Eigen::MatrixXd> V;
    X.reserve(world.getObjects().size());
    V.reserve(world.getObjects().size());
    for (const auto& obj : world.getObjects()) {
        X.push_back(obj.X);
        V.push_back(obj.V);
    }
    append_dataset_state(storage, X, V, time, dt, source, 0, 0, 0, true);
}

static void initialize_dataset_export(
    MassSpringWorldStorage& storage,
    const MassSpringWorld& world,
    bool enable_mesh_collision,
    bool enable_self_collision,
    bool enable_friction_impulse,
    bool enable_fixed_points,
    bool enable_damping)
{
    const auto root = executable_directory() / "data" / "output";
    std::filesystem::create_directories(root);

    std::filesystem::path dir;
    for (int attempt = 0; attempt < 10000; ++attempt) {
        std::ostringstream name;
        name << "sequence_" << timestamp_string() << "_" << std::setw(4) << std::setfill('0') << attempt;
        dir = root / name.str();
        if (!std::filesystem::exists(dir)) break;
    }

    std::filesystem::create_directories(dir);

    storage.dataset_initialized = true;
    storage.dataset_dir = dir;
    storage.dataset_time = 0.0;
    storage.dataset_update_count = 0;
    storage.dataset_num_points = dataset_total_points(world);
    storage.dataset_num_objects = static_cast<int>(world.getObjects().size());

    // Truncate streaming files.
    {
        std::ofstream(dir / "states_f32.bin", std::ios::binary | std::ios::trunc).close();
        std::ofstream(dir / "updates.jsonl", std::ios::out | std::ios::trunc).close();
    }

    write_dataset_static_files(storage, world);
    write_dataset_meta(
        storage,
        world,
        enable_mesh_collision,
        enable_self_collision,
        enable_friction_impulse,
        enable_fixed_points,
        enable_damping);
}

static bool dataset_topology_matches(const MassSpringWorldStorage& storage, const MassSpringWorld& world)
{
    return storage.dataset_initialized &&
           storage.dataset_num_points == dataset_total_points(world) &&
           storage.dataset_num_objects == static_cast<int>(world.getObjects().size());
}

NODE_DEF_OPEN_SCOPE

NODE_DECLARATION_FUNCTION(hw9_mass_spring_world)
{
    // A single Object List is easier for simulation_in/out than a dynamic group.
    // Simulated Object List should come from simulation_in.
    // Collider Object List should be connected directly, not through simulation_in.
    b.add_input<USTC_CG::mass_spring::HW9WorldObjectList>("Simulated Object List").optional(true);
    b.add_input<USTC_CG::mass_spring::HW9WorldObjectList>("Collider Object List").optional(true);

    b.add_input<float>("h").default_val(0.01f).min(0.001f).max(0.0333333333f);
    b.add_input<float>("gravity").default_val(-9.8f).min(-20.0f).max(20.0f);

    b.add_input<int>("enable ground").default_val(1).min(0).max(1);
    b.add_input<float>("ground z").default_val(-1.0f).min(-5.0f).max(5.0f);

    b.add_input<int>("enable mesh collision").default_val(1).min(0).max(1);
    b.add_input<float>("contact thickness").default_val(0.03f).min(0.0f).max(0.5f);
    b.add_input<float>("contact stiffness").default_val(10000.0f).min(1.0f).max(1000000.0f);
    b.add_input<int>("enable self collision").default_val(1).min(0).max(1);
    b.add_input<int>("enable friction impulse").default_val(1).min(0).max(1);
    b.add_input<int>("enable fixed points").default_val(0).min(0).max(1);
    b.add_input<int>("enable damping").default_val(0).min(0).max(1);

    b.add_input<int>("enable debug output").default_val(0).min(0).max(1);
    b.add_input<int>("enable dataset output").default_val(0).min(0).max(1);

    // This output must go to simulation_out.
    b.add_output<USTC_CG::mass_spring::HW9WorldObjectList>("Output Simulated Object List");
    // Preview/debug output; it merges simulated and collider objects into a single Geometry.
    b.add_output<Geometry>("Output Geometry");
}

NODE_EXECUTION_FUNCTION(hw9_mass_spring_world)
{
    using namespace USTC_CG::mass_spring;

    auto& global_payload = params.get_global_payload<GeomPayload&>();
    auto current_time = global_payload.current_time;
    auto& storage = params.get_storage<MassSpringWorldStorage&>();

    HW9WorldObjectList simulated_input;
    HW9WorldObjectList collider_input;

    // Optional custom-typed inputs can be absent. Never let an exception escape
    // the extern-C node execution function.
    try {
        simulated_input = params.get_input<HW9WorldObjectList>("Simulated Object List");
    }
    catch (...) {
        simulated_input.objects.clear();
    }

    try {
        collider_input = params.get_input<HW9WorldObjectList>("Collider Object List");
    }
    catch (...) {
        collider_input.objects.clear();
    }

    MassSpringWorld world;
    world.h = params.get_input<float>("h");
    world.gravity = Eigen::Vector3d(
        0.0,
        0.0,
        static_cast<double>(params.get_input<float>("gravity")));
    world.wind_ext_acc = Eigen::Vector3d(kDefaultWindX, kDefaultWindY, kDefaultWindZ);
    world.enable_ground = params.get_input<int>("enable ground") == 1;
    world.ground_z = params.get_input<float>("ground z");
    world.ground_restitution = kDefaultGroundRestitution;
    world.ground_friction = kDefaultGroundFriction;
    const bool enable_mesh_collision = params.get_input<int>("enable mesh collision") == 1;
    world.mesh_collision_mode = enable_mesh_collision ?
        MassSpringWorld::MESH_BARRIER_CONTACT : MassSpringWorld::MESH_COLLISION_NONE;
    const int mode = enable_mesh_collision ? 1 : 0;
    world.collision_cell_size = 0.0;  // auto-estimated from edge lengths
    world.contact_thickness = params.get_input<float>("contact thickness");
    world.contact_stiffness = params.get_input<float>("contact stiffness");
    world.self_contact_stiffness_scale = kSelfContactStiffnessScale;
    world.self_contact_thickness_scale = kSelfContactThicknessScale;
    world.max_ipc_candidates = kMaxContactCandidates;
    world.enable_self_collision = params.get_input<int>("enable self collision") == 1;
    world.enable_friction_impulse = params.get_input<int>("enable friction impulse") == 1;
    world.enable_fixed_points = params.get_input<int>("enable fixed points") == 1;
    const bool enable_damping = params.get_input<int>("enable damping") == 1;
    world.enable_implicit_elastic = true;
    world.enable_make_SPD = true;
    world.implicit_max_newton_iters = kImplicitMaxNewtonIters;
    world.implicit_grad_tol = kImplicitGradTol;
    world.implicit_step_tol = kImplicitStepTol;
    world.enable_debug_output = params.get_input<int>("enable debug output") == 1;
    const bool enable_dataset_output = params.get_input<int>("enable dataset output") == 1;

    if (world.enable_debug_output) {
        std::cerr << "[HW9World] interface loaded: inputs=(Simulated Object List, Collider Object List), "
                  << "outputs=(Output Simulated Object List, Output Geometry)" << std::endl;
        std::cerr << "[HW9World] simulated objects=" << simulated_input.objects.size()
                  << ", collider objects=" << collider_input.objects.size()
                  << ", implicit_elastic=1"
                  << ", collision_mode=" << mode
                  << ", contact_stiffness=" << world.contact_stiffness
                  << ", self_collision_pbd=1"
                  << ", self_contact_stiffness_scale(fixed)=" << world.self_contact_stiffness_scale
                  << ", self_contact_thickness_scale(fixed)=" << world.self_contact_thickness_scale
                  << ", max_contact_candidates(fixed)=" << world.max_ipc_candidates
                  << ", friction_impulse=" << world.enable_friction_impulse
                  << ", fixed_points=" << world.enable_fixed_points
                  << ", damping=" << enable_damping
                  << ", dataset_output=" << enable_dataset_output
                  << std::endl;
    }

    storage.output_simulated_list.objects.clear();
    std::vector<int> simulated_world_ids;

    // 1) Add objects carried by simulation_in/out. They should be dynamic deformable.
    for (int i = 0; i < static_cast<int>(simulated_input.objects.size()); ++i) {
        HW9WorldObject object = simulated_input.objects[i];
        Eigen::MatrixXd vertices;
        std::vector<Eigen::Vector3i> faces;
        EdgeSet edges;
        if (!extract_mesh_data(object.geometry, vertices, faces, edges)) {
            if (world.enable_debug_output) std::cerr << "[HW9World] skip simulated object " << i << std::endl;
            continue;
        }

        int type = object.params.object_type;
        int oid = world.addObject(vertices, faces, edges, type, object.params.name);
        auto& obj = world.getObjects()[oid];
        obj.mass = object.params.mass;
        obj.stiffness = object.params.stiffness;
        obj.damping = enable_damping ? object.params.damping : 1.0;
        obj.restitution = object.params.restitution;
        obj.friction = object.params.friction;
        obj.V = glm_vel_to_eigen(object.velocities, vertices.rows());

        if (current_time == 0 ||
            i >= static_cast<int>(storage.simulated_rest_X.size()) ||
            storage.simulated_rest_X[i].rows() != vertices.rows() ||
            i >= static_cast<int>(storage.simulated_rest_lengths.size()) ||
            storage.simulated_rest_lengths[i].size() != edges.size() ||
            i >= static_cast<int>(storage.simulated_rest_areas.size()) ||
            storage.simulated_rest_areas[i].size() != static_cast<size_t>(vertices.rows())) {
            if (i >= static_cast<int>(storage.simulated_rest_X.size())) {
                storage.simulated_rest_X.resize(i + 1);
                storage.simulated_rest_lengths.resize(i + 1);
                storage.simulated_rest_areas.resize(i + 1);
            }
            storage.simulated_rest_X[i] = vertices;
            storage.simulated_rest_lengths[i] = compute_rest_lengths(vertices, edges);
            storage.simulated_rest_areas[i] = compute_vertex_lumped_areas(vertices, faces);
        }

        if (i < static_cast<int>(storage.simulated_rest_X.size()) &&
            storage.simulated_rest_X[i].rows() == vertices.rows()) {
            obj.X0 = storage.simulated_rest_X[i];
        }
        if (i < static_cast<int>(storage.simulated_rest_lengths.size()) &&
            storage.simulated_rest_lengths[i].size() == obj.E_rest_length.size()) {
            obj.E_rest_length = storage.simulated_rest_lengths[i];
        }
        if (i < static_cast<int>(storage.simulated_rest_areas.size()) &&
            storage.simulated_rest_areas[i].size() == obj.vertex_area.size()) {
            obj.vertex_area = storage.simulated_rest_areas[i];
        }

        if (world.enable_fixed_points && obj.isDeformable() && vertices.rows() >= 2) {
            const int n_fix = std::max(1, static_cast<int>(std::sqrt(static_cast<double>(vertices.rows()))));
            const int i0 = 0;
            const int i1 = std::min(static_cast<int>(vertices.rows()) - 1, n_fix - 1);
            obj.fixed_mask.assign(vertices.rows(), false);
            obj.fixed_mask[i0] = true;
            obj.fixed_mask[i1] = true;
            obj.V.row(i0).setZero();
            obj.V.row(i1).setZero();
        }

        simulated_world_ids.push_back(oid);
        storage.output_simulated_list.objects.push_back(object);
    }

    // 2) Add collider objects from direct input. These do NOT go through simulation_in/out.
    HW9WorldObjectList collider_output = collider_input;
    for (int i = 0; i < static_cast<int>(collider_input.objects.size()); ++i) {
        HW9WorldObject object = collider_input.objects[i];
        Eigen::MatrixXd vertices;
        std::vector<Eigen::Vector3i> faces;
        EdgeSet edges;
        if (!extract_mesh_data(object.geometry, vertices, faces, edges)) {
            if (world.enable_debug_output) std::cerr << "[HW9World] skip collider object " << i << std::endl;
            continue;
        }

        int type = object.params.object_type;
        if (type == DYNAMIC_DEFORMABLE) type = KINEMATIC_MESH;

        int oid = world.addObject(vertices, faces, edges, type, object.params.name);
        auto& obj = world.getObjects()[oid];
        obj.mass = object.params.mass;
        obj.stiffness = object.params.stiffness;
        obj.damping = object.params.damping;
        obj.restitution = object.params.restitution;
        obj.friction = object.params.friction;

        if (type == KINEMATIC_MESH) {
            // External animated mesh convention:
            // current input = mesh at this frame; last_collider_input = previous frame.
            // Store the previous vertices as the start state and use V=(current-prev)/h,
            // so predictState() moves the kinematic mesh from previous frame to current frame.
            // If no previous frame exists, or the mesh is unchanged, V is zero and the object behaves as static.
            if (i < static_cast<int>(storage.last_collider_input.objects.size())) {
                Eigen::MatrixXd prev_vertices;
                std::vector<Eigen::Vector3i> prev_faces;
                EdgeSet prev_edges;
                if (extract_mesh_data(storage.last_collider_input.objects[i].geometry, prev_vertices, prev_faces, prev_edges) &&
                    prev_vertices.rows() == vertices.rows()) {
                    obj.X = prev_vertices;
                    obj.X0 = prev_vertices;
                    obj.V = (vertices - prev_vertices) / std::max(1e-12, world.h);
                }
                else {
                    obj.V.setZero();
                }
            }
            else {
                obj.V.setZero();
            }
        }
        else {
            obj.V.setZero();
        }
    }

    if (enable_dataset_output) {
        if (current_time == 0 || !dataset_topology_matches(storage, world)) {
            initialize_dataset_export(
                storage,
                world,
                mode != MassSpringWorld::MESH_COLLISION_NONE,
                world.enable_self_collision,
                world.enable_friction_impulse,
                world.enable_fixed_points,
                enable_damping);
            append_dataset_state_from_world(storage, world, 0.0, 0.0, "initial");
            write_dataset_meta(
                storage,
                world,
                mode != MassSpringWorld::MESH_COLLISION_NONE,
                world.enable_self_collision,
                world.enable_friction_impulse,
                world.enable_fixed_points,
                enable_damping);
            if (world.enable_debug_output) {
                std::cerr << "[HW9World:Dataset] output dir: "
                          << storage.dataset_dir.string() << std::endl;
            }
        }
        world.beginStepRecording(storage.dataset_time);
    }
    else {
        world.disableStepRecording();
    }

    if (current_time != 0 && !simulated_world_ids.empty()) {
        world.step();
    }

    if (enable_dataset_output && current_time != 0) {
        for (const auto& s : world.getStepSnapshots()) {
            append_dataset_state(
                storage,
                s.X,
                s.V,
                s.time,
                s.dt,
                "accepted_update",
                s.substep_index,
                s.retry_count,
                s.self_contact_count,
                s.post_response_applied);
        }
        write_dataset_meta(
            storage,
            world,
            mode != MassSpringWorld::MESH_COLLISION_NONE,
            world.enable_self_collision,
            world.enable_friction_impulse,
            world.enable_fixed_points,
            enable_damping);
    }

    // 3) Write simulated objects back to payload.
    const auto& objs = world.getObjects();
    for (int i = 0; i < static_cast<int>(simulated_world_ids.size()) &&
                    i < static_cast<int>(storage.output_simulated_list.objects.size()); ++i) {
        int oid = simulated_world_ids[i];
        auto mesh = storage.output_simulated_list.objects[i].geometry.get_component<Ruzino::MeshComponent>();
        if (mesh) {
            mesh->set_vertices(USTC_CG::mass_spring::eigen_to_usd_vertices(objs[oid].X));
        }
        storage.output_simulated_list.objects[i].velocities = eigen_vel_to_glm(objs[oid].V);
    }

    // 4) For preview output, use direct collider input plus simulated output.
    storage.last_collider_input = collider_input;

    params.set_output("Output Simulated Object List", storage.output_simulated_list);
    params.set_output("Output Geometry", merge_object_list_geometry(storage.output_simulated_list, collider_output));
    return true;
}

NODE_DECLARATION_UI(hw9_mass_spring_world);
NODE_DEF_CLOSE_SCOPE
