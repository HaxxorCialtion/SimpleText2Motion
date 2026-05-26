// t2m_infer.cpp
//
// Cross-platform (macOS / Linux / Windows) port of t2m_infer_win.cpp.
//
// Two modes:
//   one-shot (default):  prompt -> fused_server -> hidden -> motion.bin
//   server (--server):   persistent TCP server on 127.0.0.1:<port>
//
// Backend: ONNX Runtime. Execution provider selectable via --ep (auto|cuda|cpu).
//   - "auto" (default): try CUDA, silently fall back to CPU if unavailable
//   - "cuda": require CUDA; error out if it cannot be initialized
//   - "cpu": force CPU EP
//
// Note: for this DiT (~50M params, T<200), CPU EP often matches or beats
// consumer GPUs due to per-kernel launch overhead. CUDA EP is provided for
// H100/H200-class hardware and for batched/server scenarios where launch
// cost amortizes. Apple Silicon CPU + Accelerate.framework BLAS also
// performs very well here.
//
// Weight loading: small_weights.npz (zip of .npy), parsed by npz_reader.h.
// Falls back to small_weights_unpacked/*.npy if the directory exists (for
// compat with the Windows layout).
//
// Wire protocol identical to t2m_infer_win.cpp (see HANDOFF_APP_LAYER.md).

#include "platform.h"
#include "config.h"
#include <onnxruntime_cxx_api.h>

#include "npz_reader.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;
using clk = std::chrono::steady_clock;
using npz::NpyArray;

// ============================================================
// Util
// ============================================================
static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        clk::now().time_since_epoch()).count();
}

// ============================================================
// Socket helpers
// ============================================================
static void send_all(sock_t s, const void* data, size_t n) {
    const char* p = (const char*)data;
    while (n > 0) {
        int w = (int)::send(s, p, (int)n, 0);
        if (w <= 0) throw std::runtime_error("send failed");
        p += w; n -= (size_t)w;
    }
}
static void recv_all(sock_t s, void* data, size_t n) {
    char* p = (char*)data;
    while (n > 0) {
        int r = (int)::recv(s, p, (int)n, 0);
        if (r <= 0) throw std::runtime_error("recv failed/closed");
        p += r; n -= (size_t)r;
    }
}
static bool try_recv_all(sock_t s, void* data, size_t n) {
    char* p = (char*)data;
    while (n > 0) {
        int r = (int)::recv(s, p, (int)n, 0);
        if (r <= 0) return false;
        p += r; n -= (size_t)r;
    }
    return true;
}
static bool try_send_all(sock_t s, const void* data, size_t n) {
    const char* p = (const char*)data;
    while (n > 0) {
        int w = (int)::send(s, p, (int)n, 0);
        if (w <= 0) return false;
        p += w; n -= (size_t)w;
    }
    return true;
}

// ============================================================
// Small MLPs (host side, plain fp32)
// ============================================================
static inline void silu_inplace(float* x, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        float v = x[i];
        x[i] = v / (1.0f + std::exp(-v));
    }
}

// y[1, out] = x[1, in] @ W[out, in]^T + b[out]
static void linear(const float* x, size_t in_d,
                   const float* W, const float* b,
                   float* y, size_t out_d) {
    for (size_t o = 0; o < out_d; ++o) {
        float acc = b[o];
        const float* wrow = W + o * in_d;
        for (size_t i = 0; i < in_d; ++i) acc += x[i] * wrow[i];
        y[o] = acc;
    }
}

static void timestep_embed(float t,
                           const NpyArray& w0, const NpyArray& b0,
                           const NpyArray& w1, const NpyArray& b1,
                           float* out)
{
    const size_t freq_dim = 256;
    const size_t half = freq_dim / 2;
    std::vector<float> emb(freq_dim);
    for (size_t i = 0; i < half; ++i) {
        float freq = std::exp(-std::log(10000.0f) * (float)i / (float)half);
        float arg = t * freq;
        emb[i]        = std::cos(arg);
        emb[half + i] = std::sin(arg);
    }
    size_t out0 = w0.shape[0];
    size_t out1 = w1.shape[0];
    std::vector<float> h(out0);
    linear(emb.data(), freq_dim, w0.as_float(), b0.as_float(), h.data(), out0);
    silu_inplace(h.data(), out0);
    linear(h.data(), out0, w1.as_float(), b1.as_float(), out, out1);
}

static void duration_embed(float dur,
                           const NpyArray& w0, const NpyArray& b0,
                           const NpyArray& w1, const NpyArray& b1,
                           float* out)
{
    float in1[1] = { dur };
    size_t out0 = w0.shape[0];
    size_t out1 = w1.shape[0];
    std::vector<float> h(out0);
    linear(in1, 1, w0.as_float(), b0.as_float(), h.data(), out0);
    silu_inplace(h.data(), out0);
    linear(h.data(), out0, w1.as_float(), b1.as_float(), out, out1);
}

static void cond_mlp(const float* text_pooled, const float* dur_emb,
                     const NpyArray& w0, const NpyArray& b0,
                     const NpyArray& w1, const NpyArray& b1,
                     float* out)
{
    const size_t in_d = 576;
    std::vector<float> in_concat(in_d);
    std::memcpy(in_concat.data(),       text_pooled, 512 * sizeof(float));
    std::memcpy(in_concat.data() + 512, dur_emb,      64 * sizeof(float));
    size_t out0 = w0.shape[0];
    size_t out1 = w1.shape[0];
    std::vector<float> h(out0);
    linear(in_concat.data(), in_d, w0.as_float(), b0.as_float(), h.data(), out0);
    silu_inplace(h.data(), out0);
    linear(h.data(), out0, w1.as_float(), b1.as_float(), out, out1);
}

// ============================================================
// fused_server hidden fetch
// ============================================================
static std::vector<float> fetch_hidden(const std::string& host, int port,
                                       const std::string& prompt,
                                       int& n_tokens, int& n_emb,
                                       double& server_ms)
{
    sock_t s = ::socket(AF_INET, SOCK_STREAM, 0);
    if (s == INVALID_SOCK) throw std::runtime_error("socket() failed");
    int nodelay = 1;
    ::setsockopt(s, IPPROTO_TCP, TCP_NODELAY,
                 (const char*)&nodelay, sizeof(nodelay));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons((uint16_t)port);
    if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
        struct addrinfo hints{}, *res = nullptr;
        hints.ai_family = AF_INET; hints.ai_socktype = SOCK_STREAM;
        if (::getaddrinfo(host.c_str(), nullptr, &hints, &res) != 0 || !res) {
            close_sock(s);
            throw std::runtime_error("cannot resolve host: " + host);
        }
        addr.sin_addr = ((sockaddr_in*)res->ai_addr)->sin_addr;
        ::freeaddrinfo(res);
    }
    if (::connect(s, (sockaddr*)&addr, sizeof(addr)) == SOCKET_ERROR) {
        close_sock(s);
        throw std::runtime_error("connect failed (is fused_server running?)");
    }

    uint32_t plen    = (uint32_t)prompt.size();
    uint32_t plen_be = htonl(plen);
    send_all(s, &plen_be, 4);
    send_all(s, prompt.data(), plen);

    int32_t status_be, nt_be, ne_be;
    recv_all(s, &status_be, 4);
    recv_all(s, &nt_be,     4);
    recv_all(s, &ne_be,     4);
    int32_t status = (int32_t)ntohl((uint32_t)status_be);
    n_tokens       = (int32_t)ntohl((uint32_t)nt_be);
    n_emb          = (int32_t)ntohl((uint32_t)ne_be);
    if (status != 0) {
        close_sock(s);
        throw std::runtime_error("fused_server returned non-zero status");
    }

    size_t total = (size_t)n_tokens * (size_t)n_emb;
    std::vector<float> out(total);
    recv_all(s, out.data(), total * sizeof(float));

    double ms_le;
    recv_all(s, &ms_le, 8);
    server_ms = ms_le;

    close_sock(s);
    return out;
}

// ============================================================
// Args
// ============================================================
struct Args {
    bool        server_mode = false;
    int         server_port = 8423;
    std::string prompt;
    std::string out         = "./motion.bin";
    std::string onnx_dir    = "./models/SimpleT2M";
    std::string host        = "127.0.0.1";
    int         port        = 8421;
    float       duration    = 3.0f;
    int         num_steps   = 4;
    int         seed        = 42;
    bool        use_cpu     = false;  // legacy --cpu flag: if true, force CPU EP
    std::string ep          = "auto"; // "auto" | "cuda" | "cpu"
};

static void usage(const char* a0) {
    std::printf("usage (one-shot): %s --prompt STR [--out PATH] [options]\n", a0);
    std::printf("usage (server):   %s --server [--server-port 8423] [options]\n", a0);
    std::printf("  --prompt STR       text prompt (required for one-shot)\n");
    std::printf("  --out PATH         output motion.bin (default ./motion.bin)\n");
    std::printf("  --server           run as TCP server (loopback only)\n");
    std::printf("  --server-port N    server port (default 8423)\n");
    std::printf("  --onnx-dir DIR     (default ./models/SimpleT2M)\n");
    std::printf("  --duration SEC     (default 3.0)\n");
    std::printf("  --num-steps N      (default 4)\n");
    std::printf("  --seed N           (default 42)\n");
    std::printf("  --host HOST        fused_server host (default 127.0.0.1)\n");
    std::printf("  --port N           fused_server port (default 8421)\n");
    std::printf("  --ep MODE          execution provider: auto|cuda|cpu (default auto)\n");
    std::printf("  --cpu              force CPU EP (legacy alias for --ep cpu)\n");
}

static Args parse_args(int argc, char** argv) {
    // 第一遍扫:只找 --config,提前加载
    std::string config_path;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--config" && i+1 < argc) {
            config_path = argv[++i]; break;
        }
    }
    AppConfig cfg = config_path.empty()
        ? AppConfig::defaults()
        : AppConfig::from_toml(config_path);

    Args a;
    // 用 config 的值覆盖结构体默认
    if (!cfg.paths.t2m_onnx_dir.empty()) a.onnx_dir = cfg.paths.t2m_onnx_dir;
    a.host        = cfg.t2m.fused_host;
    a.port        = cfg.fused.port_hidden;
    a.server_port = cfg.t2m.server_port;
    a.num_steps   = cfg.t2m.num_steps;
    a.ep          = cfg.t2m.ep;

    auto need = [&](int& i, const char* name) {
        if (i + 1 >= argc) {
            std::fprintf(stderr, "missing value for %s\n", name);
            std::exit(1);
        }
        return std::string(argv[++i]);
    };
    // 第二遍扫:处理所有 CLI(包括跳过 --config 的值)
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if      (k == "--config")       { (void)need(i, "--config"); /* 已处理 */ }
        else if (k == "--server")       a.server_mode = true;
        else if (k == "--server-port")  a.server_port = std::stoi(need(i, "--server-port"));
        else if (k == "--prompt")       a.prompt    = need(i, "--prompt");
        else if (k == "--out")          a.out       = need(i, "--out");
        else if (k == "--onnx-dir")     a.onnx_dir  = need(i, "--onnx-dir");
        else if (k == "--duration")     a.duration  = std::stof(need(i, "--duration"));
        else if (k == "--num-steps")    a.num_steps = std::stoi(need(i, "--num-steps"));
        else if (k == "--seed")         a.seed      = std::stoi(need(i, "--seed"));
        else if (k == "--host")         a.host      = need(i, "--host");
        else if (k == "--port")         a.port      = std::stoi(need(i, "--port"));
        else if (k == "--cpu")          { a.use_cpu = true; a.ep = "cpu"; }
        else if (k == "--ep")           a.ep        = need(i, "--ep");
        else if (k == "-h" || k == "--help") { usage(argv[0]); std::exit(0); }
        else {
            std::fprintf(stderr, "unknown arg: %s\n", k.c_str());
            usage(argv[0]); std::exit(1);
        }
    }
    if (!a.server_mode && a.prompt.empty()) { usage(argv[0]); std::exit(1); }
    if (a.ep != "auto" && a.ep != "cuda" && a.ep != "cpu") {
        std::fprintf(stderr, "invalid --ep: %s (expected auto|cuda|cpu)\n", a.ep.c_str());
        std::exit(1);
    }
    return a;
}

// ============================================================
// Inference context
// ============================================================
struct InferContext {
    NpyArray ts_w0, ts_b0, ts_w1, ts_b1;
    NpyArray du_w0, du_b0, du_w1, du_b1;
    NpyArray cd_w0, cd_b0, cd_w1, cd_b1;

    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "t2m_infer"};
    std::unique_ptr<Ort::Session> sess_refiner;
    std::unique_ptr<Ort::Session> sess_dit;
    Ort::MemoryInfo mem{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};

    std::string hidden_host = "127.0.0.1";
    int         hidden_port = 8421;
    int         num_steps   = 4;
    bool        verbose     = true;

    // EP selection
    std::string ep_request  = "auto"; // what the user asked for
    std::string ep_used     = "cpu";  // what we actually got, after fallback
};

struct InferTimings {
    double hidden_ms          = 0;
    double refiner_ms         = 0;
    double dit_total_ms       = 0;
    double dit_step0_ms       = 0;
    double dit_step1_3_avg_ms = 0;
};

// ------------------------------------------------------------
// Pull all 12 small-weight arrays out of either:
//   (a) small_weights.npz, or
//   (b) small_weights_unpacked/*.npy  (windows layout, fallback)
// ------------------------------------------------------------
static void load_small_weights(InferContext& ctx, const fs::path& onnxDir) {
    fs::path npz_path = onnxDir / "small_weights.npz";
    fs::path unpacked = onnxDir / "small_weights_unpacked";

    auto take = [](std::unordered_map<std::string, NpyArray>& m,
                   const std::string& key) -> NpyArray {
        auto it = m.find(key);
        if (it == m.end())
            throw std::runtime_error("small_weights missing key: " + key);
        return std::move(it->second);
    };

    if (fs::exists(npz_path)) {
        if (ctx.verbose)
            std::printf("[1/6] loading small weights from %s\n",
                        npz_path.string().c_str());
        auto m = npz::load_npz(npz_path);
        ctx.ts_w0 = take(m, "timestep_w0"); ctx.ts_b0 = take(m, "timestep_b0");
        ctx.ts_w1 = take(m, "timestep_w1"); ctx.ts_b1 = take(m, "timestep_b1");
        ctx.du_w0 = take(m, "dur_w0");      ctx.du_b0 = take(m, "dur_b0");
        ctx.du_w1 = take(m, "dur_w1");      ctx.du_b1 = take(m, "dur_b1");
        ctx.cd_w0 = take(m, "cond_w0");     ctx.cd_b0 = take(m, "cond_b0");
        ctx.cd_w1 = take(m, "cond_w1");     ctx.cd_b1 = take(m, "cond_b1");
    } else if (fs::exists(unpacked)) {
        if (ctx.verbose)
            std::printf("[1/6] loading small weights from %s\n",
                        unpacked.string().c_str());
        auto load_one = [&](const std::string& name) {
            fs::path p = unpacked / (name + ".npy");
            std::ifstream f(p, std::ios::binary);
            if (!f) throw std::runtime_error("cannot open: " + p.string());
            f.seekg(0, std::ios::end);
            size_t sz = (size_t)f.tellg();
            f.seekg(0);
            std::vector<uint8_t> buf(sz);
            f.read((char*)buf.data(), sz);
            return npz::load_npy_from_memory(buf.data(), sz, p.string());
        };
        ctx.ts_w0 = load_one("timestep_w0"); ctx.ts_b0 = load_one("timestep_b0");
        ctx.ts_w1 = load_one("timestep_w1"); ctx.ts_b1 = load_one("timestep_b1");
        ctx.du_w0 = load_one("dur_w0");      ctx.du_b0 = load_one("dur_b0");
        ctx.du_w1 = load_one("dur_w1");      ctx.du_b1 = load_one("dur_b1");
        ctx.cd_w0 = load_one("cond_w0");     ctx.cd_b0 = load_one("cond_b0");
        ctx.cd_w1 = load_one("cond_w1");     ctx.cd_b1 = load_one("cond_b1");
    } else {
        throw std::runtime_error("no small_weights.npz or small_weights_unpacked/ in " +
                                 onnxDir.string());
    }
    if (ctx.verbose) std::printf("    OK\n");
}

// ------------------------------------------------------------
// init: load weights + build ORT sessions (CUDA or CPU)
// ------------------------------------------------------------
static void init_inference(InferContext& ctx, const std::string& onnx_dir) {
    fs::path onnxDir = onnx_dir;

    // 1. small weights
    load_small_weights(ctx, onnxDir);

    // 2. ORT sessions — try CUDA first if requested, fall back to CPU
    if (ctx.verbose)
        std::printf("\n[2/6] init ORT (requested EP=%s)...\n",
                    ctx.ep_request.c_str());

    fs::path birefinerPath = onnxDir / "birefiner.onnx";
    fs::path ditPath       = onnxDir / "dit_step.onnx";
    if (!fs::exists(birefinerPath))
        throw std::runtime_error("missing " + birefinerPath.string());
    if (!fs::exists(ditPath))
        throw std::runtime_error("missing " + ditPath.string());

    auto make_so = []() {
        Ort::SessionOptions so;
        so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        so.SetExecutionMode(ORT_SEQUENTIAL);
        // Default intra-op threads = # of physical cores; let ORT decide.
        // Override if you need to: so.SetIntraOpNumThreads(4);
        return so;
    };

    bool want_cuda = (ctx.ep_request == "cuda" || ctx.ep_request == "auto");
    bool cuda_ok   = false;

    if (want_cuda) {
        try {
            Ort::SessionOptions so_cuda = make_so();
            OrtCUDAProviderOptions cuda_opts{};
            cuda_opts.device_id                 = 0;
            cuda_opts.arena_extend_strategy     = 0;       // kNextPowerOfTwo
            cuda_opts.cudnn_conv_algo_search    = OrtCudnnConvAlgoSearchExhaustive;
            cuda_opts.do_copy_in_default_stream = 1;
            so_cuda.AppendExecutionProvider_CUDA(cuda_opts);

            ctx.sess_refiner = std::make_unique<Ort::Session>(
                ctx.env, ort_path(birefinerPath.string()).c_str(), so_cuda);
            ctx.sess_dit = std::make_unique<Ort::Session>(
                ctx.env, ort_path(ditPath.string()).c_str(), so_cuda);

            ctx.ep_used = "cuda";
            cuda_ok     = true;
            if (ctx.verbose) std::printf("    CUDA EP registered, both onnx loaded\n");
        } catch (const Ort::Exception& e) {
            // CUDA not available / cuDNN missing / GPU OOM / build is CPU-only / etc.
            ctx.sess_refiner.reset();
            ctx.sess_dit.reset();
            if (ctx.ep_request == "cuda") {
                // explicit request, no silent downgrade
                throw std::runtime_error(
                    std::string("CUDA EP requested but unavailable: ") + e.what());
            }
            if (ctx.verbose)
                std::fprintf(stderr,
                    "    CUDA EP unavailable (%s); falling back to CPU\n", e.what());
        }
    }

    if (!cuda_ok) {
        Ort::SessionOptions so_cpu = make_so();
        ctx.sess_refiner = std::make_unique<Ort::Session>(
            ctx.env, ort_path(birefinerPath.string()).c_str(), so_cpu);
        ctx.sess_dit = std::make_unique<Ort::Session>(
            ctx.env, ort_path(ditPath.string()).c_str(), so_cpu);
        ctx.ep_used = "cpu";
        if (ctx.verbose) std::printf("    CPU EP, both onnx loaded\n");
    }
}

// ------------------------------------------------------------
// One inference
// ------------------------------------------------------------
static void run_inference(InferContext& ctx,
                          const std::string& prompt,
                          float duration, uint32_t seed,
                          std::vector<float>& motion_out,
                          uint32_t& T_out, uint32_t& D_out,
                          InferTimings& tm)
{
    // 3. fetch hidden
    if (ctx.verbose) {
        std::printf("\n[3/6] fetch hidden from %s:%d\n",
                    ctx.hidden_host.c_str(), ctx.hidden_port);
        std::printf("    prompt: %s\n", prompt.c_str());
    }
    int n_tokens = 0, n_emb = 0;
    double server_ms = 0;
    double th0 = now_ms();
    auto hidden = fetch_hidden(ctx.hidden_host, ctx.hidden_port, prompt,
                               n_tokens, n_emb, server_ms);
    tm.hidden_ms = now_ms() - th0;
    if (ctx.verbose) {
        std::printf("    got hidden (%d, %d) in %.2f ms (server reported)\n",
                    n_tokens, n_emb, server_ms);
    }
    if (n_emb != 2560) throw std::runtime_error("expected n_emb=2560");

    // 4. refiner
    if (ctx.verbose) std::printf("\n[4/6] refiner...\n");
    int64_t hidShape[3] = { 1, n_tokens, n_emb };
    Ort::Value hidT = Ort::Value::CreateTensor<float>(
        ctx.mem, hidden.data(), hidden.size(), hidShape, 3);

    const char* refIn[]  = { "text_hidden" };
    const char* refOut[] = { "text_tokens", "text_pooled" };

    double t0 = now_ms();
    auto refOuts = ctx.sess_refiner->Run(Ort::RunOptions{}, refIn, &hidT, 1, refOut, 2);
    tm.refiner_ms = now_ms() - t0;

    float* text_tokens = refOuts[0].GetTensorMutableData<float>();
    float* text_pooled = refOuts[1].GetTensorMutableData<float>();
    if (ctx.verbose) {
        auto ttShape = refOuts[0].GetTensorTypeAndShapeInfo().GetShape();
        auto tpShape = refOuts[1].GetTensorTypeAndShapeInfo().GetShape();
        std::printf("    refiner %.2f ms  text_tokens=(%lld,%lld,%lld)  text_pooled=(%lld,%lld)\n",
            tm.refiner_ms,
            (long long)ttShape[0], (long long)ttShape[1], (long long)ttShape[2],
            (long long)tpShape[0], (long long)tpShape[1]);
    }

    // 5. global_cond
    if (ctx.verbose) std::printf("\n[5/6] computing host-side conds...\n");
    std::vector<float> dur_emb(64);
    duration_embed(duration, ctx.du_w0, ctx.du_b0, ctx.du_w1, ctx.du_b1, dur_emb.data());

    std::vector<float> global_cond(512);
    cond_mlp(text_pooled, dur_emb.data(),
             ctx.cd_w0, ctx.cd_b0, ctx.cd_w1, ctx.cd_b1, global_cond.data());

    // 6. Euler dit loop
    if (ctx.verbose) std::printf("\n[6/6] dit_step Euler %d steps...\n", ctx.num_steps);
    const int motion_dim = 135;
    const int fps        = 30;
    int T = std::min((int)(duration * fps), 600);
    if (T < 1) T = 1;
    if (ctx.verbose) std::printf("    T=%d frames\n", T);

    std::mt19937 rng(seed);
    std::normal_distribution<float> ndist(0.0f, 1.0f);
    std::vector<float> x((size_t)T * motion_dim);
    for (auto& v : x) v = ndist(rng);

    int64_t xShape[3]    = { 1, T, motion_dim };
    int64_t condShape[2] = { 1, 512 };
    int64_t ttShape3[3]  = { 1, n_tokens, 512 };

    std::vector<float> cond_step(512);
    std::vector<float> t_emb(512);

    const char* ditIn[]  = { "x", "cond", "text_tokens" };
    const char* ditOut[] = { "v" };

    float dt = 1.0f / ctx.num_steps;
    double dit_total = 0, step0_ms = 0, rest_sum = 0;
    int    rest_n = 0;

    for (int step = 0; step < ctx.num_steps; ++step) {
        float t_val = step * dt;
        timestep_embed(t_val, ctx.ts_w0, ctx.ts_b0, ctx.ts_w1, ctx.ts_b1, t_emb.data());
        for (size_t i = 0; i < 512; ++i) cond_step[i] = global_cond[i] + t_emb[i];

        Ort::Value xT = Ort::Value::CreateTensor<float>(
            ctx.mem, x.data(), x.size(), xShape, 3);
        Ort::Value condT = Ort::Value::CreateTensor<float>(
            ctx.mem, cond_step.data(), cond_step.size(), condShape, 2);
        Ort::Value ttT = Ort::Value::CreateTensor<float>(
            ctx.mem, text_tokens, (size_t)1 * n_tokens * 512, ttShape3, 3);

        std::array<Ort::Value, 3> ins = { std::move(xT), std::move(condT), std::move(ttT) };

        double s0 = now_ms();
        auto ditOuts = ctx.sess_dit->Run(Ort::RunOptions{}, ditIn, ins.data(), 3, ditOut, 1);
        double sm = now_ms() - s0;
        dit_total += sm;
        if (step == 0) step0_ms = sm;
        else { rest_sum += sm; rest_n++; }

        if (ctx.verbose)
            std::printf("    step %d  t=%.3f  %.2f ms\n", step, t_val, sm);

        float* v = ditOuts[0].GetTensorMutableData<float>();
        for (size_t i = 0; i < x.size(); ++i) x[i] += v[i] * dt;
    }
    tm.dit_total_ms       = dit_total;
    tm.dit_step0_ms       = step0_ms;
    tm.dit_step1_3_avg_ms = (rest_n > 0) ? (rest_sum / rest_n) : 0.0;

    if (ctx.verbose) {
        std::printf("    dit total %.2f ms\n", tm.dit_total_ms);
        std::printf("    e2e (refiner+dit) %.2f ms\n",
                    tm.refiner_ms + tm.dit_total_ms);
    }

    motion_out = std::move(x);
    T_out = (uint32_t)T;
    D_out = (uint32_t)motion_dim;
}

// ============================================================
// One-shot driver
// ============================================================
static int run_oneshot(const Args& args) {
    InferContext ctx;
    ctx.hidden_host = args.host;
    ctx.hidden_port = args.port;
    ctx.num_steps   = args.num_steps;
    ctx.verbose     = true;
    ctx.ep_request  = args.ep;

    init_inference(ctx, args.onnx_dir);

    std::vector<float> motion;
    uint32_t T = 0, D = 0;
    InferTimings tm;
    run_inference(ctx, args.prompt, args.duration, (uint32_t)args.seed,
                  motion, T, D, tm);

    {
        std::ofstream f(args.out, std::ios::binary);
        if (!f) throw std::runtime_error("cannot open output");
        f.write((const char*)motion.data(), motion.size() * sizeof(float));
    }

    float mn = 0, mx = 0, sum = 0;
    for (float v : motion) {
        mn = std::min(mn, v); mx = std::max(mx, v); sum += std::abs(v);
    }
    std::printf("\n[done] wrote %s  shape=(%u, %u)  abs_max=%.3f  abs_mean=%.4f  ep=%s\n",
        args.out.c_str(), T, D, std::max(-mn, mx), sum / (float)motion.size(),
        ctx.ep_used.c_str());
    return 0;
}

// ============================================================
// Server driver
// ============================================================
static int run_server(const Args& args) {
    InferContext ctx;
    ctx.hidden_host = args.host;
    ctx.hidden_port = args.port;
    ctx.num_steps   = args.num_steps;
    ctx.verbose     = false;
    ctx.ep_request  = args.ep;

    std::fprintf(stderr, "[server] init (loading ONNX + small weights)...\n");
    {
        ctx.verbose = true;
        init_inference(ctx, args.onnx_dir);
        ctx.verbose = false;
    }

    // Warmup (both CPU and CUDA EPs benefit from a dummy run: triggers
    // mem-pattern planning, JIT'd kernels for the static-ish shapes, and
    // for CUDA, cuDNN algo selection / context init)
    std::fprintf(stderr, "[server] running dummy warmup...\n");
    try {
        std::vector<float> dummy;
        uint32_t T = 0, D = 0;
        InferTimings tm;
        run_inference(ctx, "warmup", 0.5f, 0u, dummy, T, D, tm);
        std::fprintf(stderr,
            "[server] warmup ok (ep=%s): refiner=%.2fms dit_total=%.2fms step0=%.2fms rest_avg=%.2fms\n",
            ctx.ep_used.c_str(),
            tm.refiner_ms, tm.dit_total_ms, tm.dit_step0_ms, tm.dit_step1_3_avg_ms);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[server] warmup FAILED: %s\n", e.what());
    }

    sock_t listen_s = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (listen_s == INVALID_SOCK) {
        std::fprintf(stderr, "[server] socket() failed\n");
        return 1;
    }
    int yes = 1;
    ::setsockopt(listen_s, SOL_SOCKET, SO_REUSEADDR,
                 (const char*)&yes, sizeof(yes));

    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port        = htons((uint16_t)args.server_port);

    if (::bind(listen_s, (sockaddr*)&addr, sizeof(addr)) == SOCKET_ERROR) {
        std::fprintf(stderr, "[server] bind() failed: %d\n", last_sock_err());
        close_sock(listen_s);
        return 1;
    }
    if (::listen(listen_s, 4) == SOCKET_ERROR) {
        std::fprintf(stderr, "[server] listen() failed: %d\n", last_sock_err());
        close_sock(listen_s);
        return 1;
    }

    std::fprintf(stderr, "[server] listening on 127.0.0.1:%d\n", args.server_port);
    std::fprintf(stderr, "[server] EP in use: %s\n", ctx.ep_used.c_str());
    std::fprintf(stderr, "[server] ready\n");
    std::fflush(stderr);

    while (true) {
        sockaddr_in cli{};
        socklen_t cli_len = sizeof(cli);
        sock_t cs = ::accept(listen_s, (sockaddr*)&cli, &cli_len);
        if (cs == INVALID_SOCK) continue;

        int nodelay = 1;
        ::setsockopt(cs, IPPROTO_TCP, TCP_NODELAY,
                     (const char*)&nodelay, sizeof(nodelay));

        uint32_t prompt_len = 0;
        if (!try_recv_all(cs, &prompt_len, 4)) { close_sock(cs); continue; }
        if (prompt_len > 64u * 1024u) {
            uint32_t status = 1;
            const char* msg = "prompt too long";
            uint32_t mlen = (uint32_t)std::strlen(msg);
            try_send_all(cs, &status, 4);
            try_send_all(cs, &mlen,   4);
            try_send_all(cs, msg,     mlen);
            close_sock(cs); continue;
        }
        std::string prompt(prompt_len, '\0');
        if (prompt_len > 0 && !try_recv_all(cs, prompt.data(), prompt_len)) {
            close_sock(cs); continue;
        }
        float    req_duration = 3.0f;
        uint32_t req_seed     = 42;
        if (!try_recv_all(cs, &req_duration, 4)) { close_sock(cs); continue; }
        if (!try_recv_all(cs, &req_seed,     4)) { close_sock(cs); continue; }

        std::vector<float> motion;
        uint32_t T = 0, D = 0;
        InferTimings tm;
        std::string err;
        bool ok = true;
        try {
            run_inference(ctx, prompt, req_duration, req_seed,
                          motion, T, D, tm);
        } catch (const Ort::Exception& e) {
            ok = false; err = std::string("ORT: ") + e.what();
        } catch (const std::exception& e) {
            ok = false; err = e.what();
        } catch (...) {
            ok = false; err = "unknown exception";
        }

        if (ok) {
            uint32_t status = 0;
            float h  = (float)tm.hidden_ms;
            float r  = (float)tm.refiner_ms;
            float dt = (float)tm.dit_total_ms;
            float d0 = (float)tm.dit_step0_ms;
            float dr = (float)tm.dit_step1_3_avg_ms;

            bool sok =
                try_send_all(cs, &status, 4) &&
                try_send_all(cs, &T, 4) &&
                try_send_all(cs, &D, 4) &&
                try_send_all(cs, motion.data(), (size_t)T * D * sizeof(float)) &&
                try_send_all(cs, &h,  4) &&
                try_send_all(cs, &r,  4) &&
                try_send_all(cs, &dt, 4) &&
                try_send_all(cs, &d0, 4) &&
                try_send_all(cs, &dr, 4);
            (void)sok;
        } else {
            std::fprintf(stderr, "[server] request failed: %s\n", err.c_str());
            uint32_t status = 1;
            uint32_t mlen   = (uint32_t)err.size();
            try_send_all(cs, &status, 4);
            try_send_all(cs, &mlen,   4);
            if (mlen) try_send_all(cs, err.data(), mlen);
        }
        close_sock(cs);
    }

    close_sock(listen_s);
    return 0;
}

// ============================================================
// Main
// ============================================================
int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        PlatformNetInit net_init;  // no-op on POSIX, WSAStartup on Win
        return args.server_mode ? run_server(args) : run_oneshot(args);
    } catch (const Ort::Exception& e) {
        std::fprintf(stderr, "[ORT exception] %s\n", e.what());
        return 10;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[exception] %s\n", e.what());
        return 11;
    }
}