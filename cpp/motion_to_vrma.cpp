// motion_to_vrma.cpp
//
// motion.bin (T, 135) f32  →  .vrma (GLB)
//
// 直译 smplh_to_vrma.py + 06_test_mine_motion.py 的 decode pipeline,
// 完全一致:
//   1. smooth motion_135 (Gaussian σ=2.0)
//   2. rot6d → rotmat → quat (root + 21 body joints)
//   3. quat hemisphere alignment (相邻帧 dot<0 翻转)
//   4. smooth quats (Gaussian σ=1.0)
//   5. hips Y offset (+0.95m, VRM T-pose hips height)
//   6. 拼 GLB: nodes (T-pose) + animation channels + binary chunk
//
// 用法:
//   ./motion_to_vrma --in motion.bin --out output.vrma
//                    [--fps 30] [--smooth-sigma 2.0] [--hips-y 0.95]

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// ============================================================
// VRM bone schema (跟 smplh_to_vrma.py SMPLH_TO_VRM 一一对应)
// ============================================================

// SMPL-H joint index → VRM bone name (22 joints: index 0=hips ... 21=rightHand)
static const char* SMPLH_TO_VRM[22] = {
    "hips",
    "leftUpperLeg", "rightUpperLeg",
    "spine",
    "leftLowerLeg", "rightLowerLeg",
    "chest",
    "leftFoot", "rightFoot",
    "upperChest",
    "leftToes", "rightToes",
    "neck",
    "leftShoulder", "rightShoulder",
    "head",
    "leftUpperArm", "rightUpperArm",
    "leftLowerArm", "rightLowerArm",
    "leftHand", "rightHand",
};

// VRM T-pose 骨骼坐标 (米, Y-up). 与 smplh_to_vrma.py VRM_TPOSE_POSITIONS 一致.
struct Vec3 { float x, y, z; };

static const std::vector<std::pair<const char*, Vec3>> VRM_TPOSE = {
    {"hips",          {  0.00f, 0.95f,  0.00f}},
    {"spine",         {  0.00f, 1.05f,  0.00f}},
    {"chest",         {  0.00f, 1.20f,  0.00f}},
    {"upperChest",    {  0.00f, 1.35f,  0.00f}},
    {"neck",          {  0.00f, 1.50f,  0.00f}},
    {"head",          {  0.00f, 1.58f,  0.00f}},
    {"leftShoulder",  {  0.05f, 1.43f,  0.00f}},
    {"rightShoulder", { -0.05f, 1.43f,  0.00f}},
    {"leftUpperArm",  {  0.18f, 1.43f,  0.00f}},
    {"rightUpperArm", { -0.18f, 1.43f,  0.00f}},
    {"leftLowerArm",  {  0.45f, 1.43f,  0.00f}},
    {"rightLowerArm", { -0.45f, 1.43f,  0.00f}},
    {"leftHand",      {  0.68f, 1.43f,  0.00f}},
    {"rightHand",     { -0.68f, 1.43f,  0.00f}},
    {"leftUpperLeg",  {  0.09f, 0.90f,  0.00f}},
    {"rightUpperLeg", { -0.09f, 0.90f,  0.00f}},
    {"leftLowerLeg",  {  0.09f, 0.48f,  0.00f}},
    {"rightLowerLeg", { -0.09f, 0.48f,  0.00f}},
    {"leftFoot",      {  0.09f, 0.06f,  0.00f}},
    {"rightFoot",     { -0.09f, 0.06f,  0.00f}},
    {"leftToes",      {  0.09f, 0.00f,  0.08f}},
    {"rightToes",     { -0.09f, 0.00f,  0.08f}},
};

// 父子关系
static const std::map<std::string, std::string> VRM_PARENTS = {
    {"hips", ""},                       // 根
    {"spine", "hips"},
    {"chest", "spine"},
    {"upperChest", "chest"},
    {"neck", "upperChest"},
    {"head", "neck"},
    {"leftShoulder", "upperChest"},
    {"rightShoulder", "upperChest"},
    {"leftUpperArm", "leftShoulder"},
    {"rightUpperArm", "rightShoulder"},
    {"leftLowerArm", "leftUpperArm"},
    {"rightLowerArm", "rightUpperArm"},
    {"leftHand", "leftLowerArm"},
    {"rightHand", "rightLowerArm"},
    {"leftUpperLeg", "hips"},
    {"rightUpperLeg", "hips"},
    {"leftLowerLeg", "leftUpperLeg"},
    {"rightLowerLeg", "rightUpperLeg"},
    {"leftFoot", "leftLowerLeg"},
    {"rightFoot", "rightLowerLeg"},
    {"leftToes", "leftFoot"},
    {"rightToes", "rightFoot"},
};

static Vec3 lookup_tpose(const std::string& name) {
    for (auto& kv : VRM_TPOSE) if (name == kv.first) return kv.second;
    throw std::runtime_error("no T-pose for " + name);
}

// ============================================================
// 数学 — rot6d → rotmat → quat
// ============================================================

// 跟 06_test_mine_motion.py rot6d_to_rotmat 一致:
//   r6d (..., 6) → reshape (..., 2, 3) → swap axes → (a1, a2)
//   b1 = norm(a1)
//   b2 = norm(a2 - (b1·a2) b1)
//   b3 = b1 × b2
//   R  = stack([b1, b2, b3], axis=-1)   // 列向量
//
// 注意 reshape (2,3) 后 swap_axes(-1,-2) 等价于先取 a1=r6d[0:3], a2=r6d[3:6].
// 我们不走 reshape, 直接索引.
static void rot6d_to_rotmat(const float r[6], float R[9]) {
    float a1[3] = {r[0], r[1], r[2]};
    float a2[3] = {r[3], r[4], r[5]};

    auto norm3 = [](const float v[3]) {
        return std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    };

    float n1 = std::max(norm3(a1), 1e-8f);
    float b1[3] = {a1[0]/n1, a1[1]/n1, a1[2]/n1};

    float dot = b1[0]*a2[0] + b1[1]*a2[1] + b1[2]*a2[2];
    float p[3] = {a2[0] - dot*b1[0], a2[1] - dot*b1[1], a2[2] - dot*b1[2]};
    float n2 = std::max(norm3(p), 1e-8f);
    float b2[3] = {p[0]/n2, p[1]/n2, p[2]/n2};

    float b3[3] = {
        b1[1]*b2[2] - b1[2]*b2[1],
        b1[2]*b2[0] - b1[0]*b2[2],
        b1[0]*b2[1] - b1[1]*b2[0],
    };

    // R[i*3 + j] = R[i][j], 跟 numpy 的 stack(axis=-1) 后形状 (3, 3) 一致:
    // R[..., :, k] = bk  →  R[i][k] = bk[i]
    // 即 R 的第 k 列是 bk
    R[0] = b1[0]; R[1] = b2[0]; R[2] = b3[0];
    R[3] = b1[1]; R[4] = b2[1]; R[5] = b3[1];
    R[6] = b1[2]; R[7] = b2[2]; R[8] = b3[2];
}

// rotmat → quaternion (xyzw, scipy.spatial.transform.Rotation 风格)
// 用经典 Shepperd 方法, 数值稳定
static void rotmat_to_quat(const float R[9], float q[4]) {
    float trace = R[0] + R[4] + R[8];
    float qx, qy, qz, qw;

    if (trace > 0.0f) {
        float s = std::sqrt(trace + 1.0f) * 2.0f;   // s = 4 * qw
        qw = 0.25f * s;
        qx = (R[7] - R[5]) / s;
        qy = (R[2] - R[6]) / s;
        qz = (R[3] - R[1]) / s;
    } else if (R[0] > R[4] && R[0] > R[8]) {
        float s = std::sqrt(1.0f + R[0] - R[4] - R[8]) * 2.0f;  // s = 4 * qx
        qw = (R[7] - R[5]) / s;
        qx = 0.25f * s;
        qy = (R[1] + R[3]) / s;
        qz = (R[2] + R[6]) / s;
    } else if (R[4] > R[8]) {
        float s = std::sqrt(1.0f + R[4] - R[0] - R[8]) * 2.0f;  // s = 4 * qy
        qw = (R[2] - R[6]) / s;
        qx = (R[1] + R[3]) / s;
        qy = 0.25f * s;
        qz = (R[5] + R[7]) / s;
    } else {
        float s = std::sqrt(1.0f + R[8] - R[0] - R[4]) * 2.0f;  // s = 4 * qz
        qw = (R[3] - R[1]) / s;
        qx = (R[2] + R[6]) / s;
        qy = (R[5] + R[7]) / s;
        qz = 0.25f * s;
    }

    // xyzw 顺序 (跟 scipy as_quat() 一致, 也是 glTF 期望的)
    q[0] = qx; q[1] = qy; q[2] = qz; q[3] = qw;
}

static void quat_normalize_inplace(float q[4]) {
    float n = std::sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
    if (n < 1e-8f) { q[0]=0; q[1]=0; q[2]=0; q[3]=1; return; }
    q[0]/=n; q[1]/=n; q[2]/=n; q[3]/=n;
}

// ============================================================
// Gaussian smoothing — 1D, 替代 scipy.ndimage.gaussian_filter1d
//
// scipy 默认: mode='reflect', truncate=4.0
// kernel half size = int(truncate * sigma + 0.5)
// reflect: x[-k] = x[k-1] (镜像, 端点不重复)
// ============================================================

static std::vector<float> gaussian_kernel(float sigma, float truncate = 4.0f) {
    int half = static_cast<int>(truncate * sigma + 0.5f);
    if (half < 1) half = 1;
    std::vector<float> k(2 * half + 1);
    float sum = 0;
    for (int i = -half; i <= half; i++) {
        float w = std::exp(-0.5f * (i * i) / (sigma * sigma));
        k[i + half] = w;
        sum += w;
    }
    for (auto& w : k) w /= sum;
    return k;
}

// 对一维序列做 reflect-mode 卷积. 'reflect' 在 scipy 里是 mirror without repeat:
//   index -1 -> 0,  index -2 -> 1, ...
//   index N  -> N-1, index N+1 -> N-2, ...
static std::vector<float> conv1d_reflect(const std::vector<float>& x,
                                         const std::vector<float>& k) {
    int N = static_cast<int>(x.size());
    int M = static_cast<int>(k.size());
    int half = M / 2;
    std::vector<float> y(N, 0.0f);

    auto fetch = [&](int i) -> float {
        // scipy 'reflect' 镜像 (a b c | c b a),
        // 我们简化用 numpy.pad mode='reflect': 不重复端点
        int n = N;
        if (n == 1) return x[0];
        // 用周期 2*(N-1) 的反射映射
        int period = 2 * (n - 1);
        int idx = i % period;
        if (idx < 0) idx += period;
        if (idx >= n) idx = period - idx;
        return x[idx];
    };

    for (int i = 0; i < N; i++) {
        float s = 0;
        for (int j = 0; j < M; j++) {
            s += k[j] * fetch(i + j - half);
        }
        y[i] = s;
    }
    return y;
}

// 对 (T, D) 矩阵的每一列做 1D smoothing
static void smooth_matrix_inplace(std::vector<float>& m, int T, int D, float sigma) {
    if (sigma <= 0) return;
    auto k = gaussian_kernel(sigma);
    std::vector<float> col(T);
    for (int c = 0; c < D; c++) {
        for (int t = 0; t < T; t++) col[t] = m[t * D + c];
        auto sm = conv1d_reflect(col, k);
        for (int t = 0; t < T; t++) m[t * D + c] = sm[t];
    }
}

// 对 (T, 4) quaternion 序列做 hemisphere alignment + smoothing + renormalize
// 跟 06_test_mine_motion.py smooth_quaternions 一致.
static void smooth_quats_inplace(std::vector<float>& q, int T, float sigma) {
    if (sigma <= 0) return;
    // hemisphere alignment: 相邻帧 dot<0 翻转
    for (int t = 1; t < T; t++) {
        float dot = 0;
        for (int c = 0; c < 4; c++) dot += q[t*4 + c] * q[(t-1)*4 + c];
        if (dot < 0) for (int c = 0; c < 4; c++) q[t*4 + c] = -q[t*4 + c];
    }
    // smooth each component
    smooth_matrix_inplace(q, T, 4, sigma);
    // renormalize
    for (int t = 0; t < T; t++) {
        quat_normalize_inplace(&q[t*4]);
    }
}

// ============================================================
// 字节缓冲 (4-byte aligned)
// ============================================================

struct ByteBuf {
    std::vector<uint8_t> data;

    size_t size() const { return data.size(); }

    void align4() {
        while (data.size() % 4 != 0) data.push_back(0);
    }

    // 追加一段 float32 数组, 返回 (offset, byte_length)
    std::pair<size_t, size_t> append_floats(const float* src, size_t n) {
        align4();
        size_t off = data.size();
        size_t bytes = n * sizeof(float);
        data.resize(off + bytes);
        std::memcpy(data.data() + off, src, bytes);
        return {off, bytes};
    }
};

// ============================================================
// 手写 glTF JSON 拼装
//
// 我们的 schema 非常窄, 不需要通用 JSON 库:
//   - asset
//   - extensionsUsed: ["VRMC_vrm_animation"]
//   - extensions.VRMC_vrm_animation.humanoid.humanBones.{name}.node = idx
//   - scene + scenes[0].nodes
//   - nodes[i].{name, translation[3], rotation[4], scale[3], children?}
//   - animations[0].{name, channels[], samplers[]}
//   - accessors[]
//   - bufferViews[]
//   - buffers[0].byteLength
// ============================================================

struct Accessor {
    int  bufferView;
    int  componentType;   // 5126 = float32
    int  count;
    std::string type;     // "SCALAR" | "VEC3" | "VEC4"
    bool has_minmax = false;
    std::vector<float> min_val;
    std::vector<float> max_val;
};

struct BufferView {
    int    buffer;
    size_t byteOffset;
    size_t byteLength;
};

struct Sampler { int input, output; };
struct Channel { int sampler, target_node; std::string target_path; };

static std::string fmt_float(float v) {
    // glTF 浮点字段, 用足够精度即可 (视觉一致, 不追求 byte-exact)
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.7g", v);
    return buf;
}

static std::string fmt_floats(const float* v, int n) {
    std::string s = "[";
    for (int i = 0; i < n; i++) {
        if (i) s += ",";
        s += fmt_float(v[i]);
    }
    s += "]";
    return s;
}

// 转义 JSON 字符串 (我们只用 ASCII bone 名字, 但保险起见)
static std::string json_str(const std::string& s) {
    std::string r = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\') { r += '\\'; r += c; }
        else if (c == '\n') r += "\\n";
        else r += c;
    }
    r += "\"";
    return r;
}

// 拼 glTF JSON, 返回字符串
static std::string build_gltf_json(
    const std::vector<std::string>& bone_order,
    const std::vector<Vec3>& local_pos,
    const std::vector<std::vector<int>>& children,
    const std::vector<int>& root_nodes,
    const std::string& title,
    const std::vector<Accessor>& accessors,
    const std::vector<BufferView>& buffer_views,
    const std::vector<Sampler>& samplers,
    const std::vector<Channel>& channels,
    size_t total_buf,
    const std::map<std::string, int>& human_bones)
{
    std::ostringstream o;

    o << "{";

    // asset
    o << "\"asset\":{\"version\":\"2.0\",\"generator\":\"motion_to_vrma C++\"},";

    // extensionsUsed + extensions
    o << "\"extensionsUsed\":[\"VRMC_vrm_animation\"],";
    o << "\"extensions\":{\"VRMC_vrm_animation\":{"
      << "\"specVersion\":\"1.0\","
      << "\"humanoid\":{\"humanBones\":{";
    {
        bool first = true;
        for (auto& kv : human_bones) {
            if (!first) o << ",";
            first = false;
            o << json_str(kv.first) << ":{\"node\":" << kv.second << "}";
        }
    }
    o << "}}}},";

    // scene
    o << "\"scene\":0,\"scenes\":[{\"nodes\":[";
    for (size_t i = 0; i < root_nodes.size(); i++) {
        if (i) o << ",";
        o << root_nodes[i];
    }
    o << "]}],";

    // nodes
    o << "\"nodes\":[";
    for (size_t i = 0; i < bone_order.size(); i++) {
        if (i) o << ",";
        const Vec3& p = local_pos[i];
        float pos[3]  = {p.x, p.y, p.z};
        float rot[4]  = {0, 0, 0, 1};
        float scl[3]  = {1, 1, 1};
        o << "{";
        o << "\"name\":"        << json_str(bone_order[i]) << ",";
        o << "\"translation\":" << fmt_floats(pos, 3) << ",";
        o << "\"rotation\":"    << fmt_floats(rot, 4) << ",";
        o << "\"scale\":"       << fmt_floats(scl, 3);
        if (!children[i].empty()) {
            o << ",\"children\":[";
            for (size_t j = 0; j < children[i].size(); j++) {
                if (j) o << ",";
                o << children[i][j];
            }
            o << "]";
        }
        o << "}";
    }
    o << "],";

    // animations
    o << "\"animations\":[{";
    o << "\"name\":" << json_str(title) << ",";
    o << "\"channels\":[";
    for (size_t i = 0; i < channels.size(); i++) {
        if (i) o << ",";
        const auto& c = channels[i];
        o << "{\"sampler\":" << c.sampler
          << ",\"target\":{\"node\":" << c.target_node
          << ",\"path\":" << json_str(c.target_path) << "}}";
    }
    o << "],";
    o << "\"samplers\":[";
    for (size_t i = 0; i < samplers.size(); i++) {
        if (i) o << ",";
        o << "{\"input\":" << samplers[i].input
          << ",\"output\":" << samplers[i].output
          << ",\"interpolation\":\"LINEAR\"}";
    }
    o << "]}],";

    // accessors
    o << "\"accessors\":[";
    for (size_t i = 0; i < accessors.size(); i++) {
        if (i) o << ",";
        const auto& a = accessors[i];
        o << "{\"bufferView\":" << a.bufferView
          << ",\"componentType\":" << a.componentType
          << ",\"count\":" << a.count
          << ",\"type\":" << json_str(a.type);
        if (a.has_minmax) {
            o << ",\"min\":" << fmt_floats(a.min_val.data(), (int)a.min_val.size());
            o << ",\"max\":" << fmt_floats(a.max_val.data(), (int)a.max_val.size());
        }
        o << "}";
    }
    o << "],";

    // bufferViews
    o << "\"bufferViews\":[";
    for (size_t i = 0; i < buffer_views.size(); i++) {
        if (i) o << ",";
        const auto& bv = buffer_views[i];
        o << "{\"buffer\":" << bv.buffer
          << ",\"byteOffset\":" << bv.byteOffset
          << ",\"byteLength\":" << bv.byteLength << "}";
    }
    o << "],";

    // buffers
    o << "\"buffers\":[{\"byteLength\":" << total_buf << "}]";

    o << "}";
    return o.str();
}

// ============================================================
// GLB packer
// ============================================================

static std::vector<uint8_t> pack_glb(const std::string& json_text,
                                     const std::vector<uint8_t>& bin_chunk) {
    // JSON chunk 必须 4-byte 对齐, 用空格 padding
    std::string json = json_text;
    while (json.size() % 4 != 0) json.push_back(' ');

    // BIN chunk 也要 4-byte 对齐 (我们的 ByteBuf::align4 已经处理过, 但保险)
    std::vector<uint8_t> bin = bin_chunk;
    while (bin.size() % 4 != 0) bin.push_back(0);

    // GLB header: 12B + JSON chunk header 8B + JSON + BIN chunk header 8B + BIN
    uint32_t total = 12 + 8 + (uint32_t)json.size() + 8 + (uint32_t)bin.size();

    std::vector<uint8_t> out;
    out.reserve(total);

    auto put_u32 = [&](uint32_t v) {
        out.push_back((v >>  0) & 0xff);
        out.push_back((v >>  8) & 0xff);
        out.push_back((v >> 16) & 0xff);
        out.push_back((v >> 24) & 0xff);
    };

    // header
    put_u32(0x46546C67);   // 'glTF'
    put_u32(2);            // version
    put_u32(total);

    // JSON chunk
    put_u32((uint32_t)json.size());
    put_u32(0x4E4F534A);   // 'JSON'
    out.insert(out.end(), json.begin(), json.end());

    // BIN chunk
    put_u32((uint32_t)bin.size());
    put_u32(0x004E4942);   // 'BIN\0'
    out.insert(out.end(), bin.begin(), bin.end());

    return out;
}

// ============================================================
// 主流程
// ============================================================

struct Args {
    std::string in_path;
    std::string out_path;
    float fps = 30.0f;
    float smooth_sigma = 2.0f;
    float hips_y_offset = 0.95f;
    std::string title = "SimpleLove Motion";
};

static void usage(const char* a0) {
    std::cerr << "usage: " << a0 << " --in motion.bin --out output.vrma [options]\n"
        << "  --in PATH           input motion.bin (T*135*4 bytes f32, required)\n"
        << "  --out PATH          output .vrma (required)\n"
        << "  --fps F             frame rate (default 30)\n"
        << "  --smooth-sigma F    Gaussian σ for motion smoothing (default 2.0, 0=off)\n"
        << "  --hips-y F          hips Y offset in meters (default 0.95)\n"
        << "  --title STR         animation name in glTF (default 'SimpleLove Motion')\n";
}

static Args parse_args(int argc, char** argv) {
    Args a;
    auto need = [&](int& i, const char* k) {
        if (i + 1 >= argc) { std::cerr << "missing value for " << k << "\n"; std::exit(1); }
        return std::string(argv[++i]);
    };
    for (int i = 1; i < argc; i++) {
        std::string k = argv[i];
        if      (k == "--in")            a.in_path  = need(i, "--in");
        else if (k == "--out")           a.out_path = need(i, "--out");
        else if (k == "--fps")           a.fps = std::stof(need(i, "--fps"));
        else if (k == "--smooth-sigma")  a.smooth_sigma = std::stof(need(i, "--smooth-sigma"));
        else if (k == "--hips-y")        a.hips_y_offset = std::stof(need(i, "--hips-y"));
        else if (k == "--title")         a.title = need(i, "--title");
        else if (k == "-h" || k == "--help") { usage(argv[0]); std::exit(0); }
        else { std::cerr << "unknown arg: " << k << "\n"; usage(argv[0]); std::exit(1); }
    }
    if (a.in_path.empty() || a.out_path.empty()) { usage(argv[0]); std::exit(1); }
    return a;
}

int main(int argc, char** argv) try {
    Args args = parse_args(argc, argv);

    // ---- 1) 读 motion.bin ----
    std::ifstream f(args.in_path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + args.in_path);
    f.seekg(0, std::ios::end);
    size_t bytes = f.tellg();
    f.seekg(0);
    if (bytes % (135 * sizeof(float)) != 0) {
        throw std::runtime_error("motion.bin size " + std::to_string(bytes)
            + " is not divisible by 135*4=540");
    }
    int T = static_cast<int>(bytes / (135 * sizeof(float)));
    std::vector<float> motion(T * 135);
    f.read(reinterpret_cast<char*>(motion.data()), bytes);
    if (!f) throw std::runtime_error("short read");
    std::cout << "[in] motion (" << T << ", 135) loaded from " << args.in_path << "\n";

    // ---- 2) smooth motion_135 ----
    if (args.smooth_sigma > 0) {
        smooth_matrix_inplace(motion, T, 135, args.smooth_sigma);
        std::cout << "[smooth] motion σ=" << args.smooth_sigma << "\n";
    }

    // ---- 3) 拆 trans / root rot6d / body rot6d, decode quat ----
    std::vector<float> trans(T * 3);                    // (T, 3)
    std::vector<float> root_quat(T * 4);                // (T, 4)
    std::vector<float> body_quat(T * 21 * 4);           // (T, 21, 4)

    for (int t = 0; t < T; t++) {
        const float* m = &motion[t * 135];
        // trans
        trans[t*3 + 0] = m[0];
        trans[t*3 + 1] = m[1];
        trans[t*3 + 2] = m[2];
        // root rot6d -> quat
        float R[9];
        rot6d_to_rotmat(m + 3, R);
        rotmat_to_quat(R, &root_quat[t*4]);
        // body 21 joints rot6d -> quat
        for (int j = 0; j < 21; j++) {
            rot6d_to_rotmat(m + 9 + j*6, R);
            rotmat_to_quat(R, &body_quat[(t*21 + j) * 4]);
        }
    }

    // ---- 4) smooth quats (σ = max(1, smooth_sigma * 0.5)) ----
    if (args.smooth_sigma > 0) {
        float qs = std::max(1.0f, args.smooth_sigma * 0.5f);
        smooth_quats_inplace(root_quat, T, qs);
        // body 21 joints: 每个 joint 单独平滑
        for (int j = 0; j < 21; j++) {
            std::vector<float> qj(T * 4);
            for (int t = 0; t < T; t++)
                for (int c = 0; c < 4; c++) qj[t*4 + c] = body_quat[(t*21 + j)*4 + c];
            smooth_quats_inplace(qj, T, qs);
            for (int t = 0; t < T; t++)
                for (int c = 0; c < 4; c++) body_quat[(t*21 + j)*4 + c] = qj[t*4 + c];
        }
        std::cout << "[smooth] quats σ=" << qs << "\n";
    } else {
        // 仅 normalize, 不做 hemisphere alignment 也不平滑
        for (int t = 0; t < T; t++) quat_normalize_inplace(&root_quat[t*4]);
        for (int i = 0; i < T*21; i++) quat_normalize_inplace(&body_quat[i*4]);
    }

    // ---- 5) hips Y offset ----
    for (int t = 0; t < T; t++) trans[t*3 + 1] += args.hips_y_offset;
    {
        float xmn=trans[0], xmx=trans[0], ymn=trans[1], ymx=trans[1], zmn=trans[2], zmx=trans[2];
        for (int t = 0; t < T; t++) {
            xmn=std::min(xmn,trans[t*3+0]); xmx=std::max(xmx,trans[t*3+0]);
            ymn=std::min(ymn,trans[t*3+1]); ymx=std::max(ymx,trans[t*3+1]);
            zmn=std::min(zmn,trans[t*3+2]); zmx=std::max(zmx,trans[t*3+2]);
        }
        std::cout << "[hips] X[" << xmn << ", " << xmx
                  << "] Y[" << ymn << ", " << ymx
                  << "] Z[" << zmn << ", " << zmx << "]\n";
    }

    // ---- 6) bone_order: hips 第一, 然后按 VRM_TPOSE 顺序排其余 ----
    std::vector<std::string> bone_order;
    bone_order.push_back("hips");
    for (auto& kv : VRM_TPOSE) {
        if (std::string(kv.first) != "hips") bone_order.push_back(kv.first);
    }
    int n_bones = (int)bone_order.size();

    std::map<std::string, int> bone_to_node;
    for (int i = 0; i < n_bones; i++) bone_to_node[bone_order[i]] = i;

    // local position per bone (rest pose)
    std::vector<Vec3> local_pos(n_bones);
    std::vector<std::vector<int>> children(n_bones);
    for (int i = 0; i < n_bones; i++) {
        Vec3 self = lookup_tpose(bone_order[i]);
        const std::string& parent_name = VRM_PARENTS.at(bone_order[i]);
        if (parent_name.empty()) {
            local_pos[i] = self;
        } else {
            Vec3 par = lookup_tpose(parent_name);
            local_pos[i] = {self.x - par.x, self.y - par.y, self.z - par.z};
        }
        // 找子节点
        for (int j = 0; j < n_bones; j++) {
            if (i == j) continue;
            auto it = VRM_PARENTS.find(bone_order[j]);
            if (it != VRM_PARENTS.end() && it->second == bone_order[i]) {
                children[i].push_back(j);
            }
        }
    }
    // root nodes: parent 不在我们的 bone 集合里
    std::vector<int> root_nodes;
    for (int i = 0; i < n_bones; i++) {
        const std::string& p = VRM_PARENTS.at(bone_order[i]);
        if (p.empty() || bone_to_node.find(p) == bone_to_node.end()) {
            root_nodes.push_back(i);
        }
    }

    // ---- 7) bin chunk + accessors + samplers + channels ----
    ByteBuf bb;
    std::vector<Accessor> accessors;
    std::vector<BufferView> buffer_views;
    std::vector<Sampler> samplers;
    std::vector<Channel> channels;

    auto add_accessor = [&](const float* data, size_t count, const std::string& type,
                            int n_components,
                            bool with_minmax = false) -> int {
        auto [off, len] = bb.append_floats(data, count * n_components);
        int bv_idx = (int)buffer_views.size();
        buffer_views.push_back({0, off, len});
        Accessor a;
        a.bufferView = bv_idx;
        a.componentType = 5126;   // FLOAT
        a.count = (int)count;
        a.type = type;
        if (with_minmax) {
            a.has_minmax = true;
            a.min_val.assign(n_components, std::numeric_limits<float>::infinity());
            a.max_val.assign(n_components, -std::numeric_limits<float>::infinity());
            for (size_t i = 0; i < count; i++) {
                for (int c = 0; c < n_components; c++) {
                    float v = data[i * n_components + c];
                    if (v < a.min_val[c]) a.min_val[c] = v;
                    if (v > a.max_val[c]) a.max_val[c] = v;
                }
            }
        }
        int idx = (int)accessors.size();
        accessors.push_back(std::move(a));
        return idx;
    };

    // 时间戳 (T,) — glTF 要求 SCALAR input 必须有 min/max
    std::vector<float> timestamps(T);
    for (int t = 0; t < T; t++) timestamps[t] = t / args.fps;
    int time_acc = add_accessor(timestamps.data(), T, "SCALAR", 1, /*with_minmax=*/true);

    // 给某 bone 加 rotation channel
    auto add_rot_channel = [&](const std::string& bone, const float* quat_data) {
        auto it = bone_to_node.find(bone);
        if (it == bone_to_node.end()) return;
        int q_acc = add_accessor(quat_data, T, "VEC4", 4);
        int s_idx = (int)samplers.size();
        samplers.push_back({time_acc, q_acc});
        channels.push_back({s_idx, it->second, "rotation"});
    };

    // hips 旋转
    add_rot_channel("hips", root_quat.data());
    // body 21 joints
    for (int j = 0; j < 21; j++) {
        std::vector<float> qj(T * 4);
        for (int t = 0; t < T; t++)
            for (int c = 0; c < 4; c++) qj[t*4 + c] = body_quat[(t*21 + j)*4 + c];
        add_rot_channel(SMPLH_TO_VRM[j + 1], qj.data());
    }

    // hips translation channel
    {
        int t_acc = add_accessor(trans.data(), T, "VEC3", 3);
        int s_idx = (int)samplers.size();
        samplers.push_back({time_acc, t_acc});
        channels.push_back({s_idx, bone_to_node["hips"], "translation"});
    }

    bb.align4();

    // human_bones map (VRMC_vrm_animation 扩展要求)
    std::map<std::string, int> human_bones;
    for (auto& kv : bone_to_node) human_bones[kv.first] = kv.second;

    // ---- 8) 拼 JSON + GLB ----
    std::string json = build_gltf_json(
        bone_order, local_pos, children, root_nodes,
        args.title, accessors, buffer_views, samplers, channels,
        bb.size(), human_bones);

    auto glb = pack_glb(json, bb.data);

    // ---- 9) 写文件 ----
    std::string out = args.out_path;
    if (out.size() < 5 || out.substr(out.size()-5) != ".vrma") out += ".vrma";
    std::ofstream of(out, std::ios::binary);
    if (!of) throw std::runtime_error("cannot open " + out);
    of.write(reinterpret_cast<const char*>(glb.data()), (std::streamsize)glb.size());
    std::cout << "[out] " << out << "  (" << glb.size() << " bytes, "
              << "T=" << T << ", " << channels.size() << " anim channels)\n";
    return 0;

} catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
}
