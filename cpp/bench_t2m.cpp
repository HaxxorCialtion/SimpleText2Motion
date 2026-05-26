// cpp/bench_t2m.cpp
//
// Benchmark client for t2m_infer in --server mode. Replaces the Python
// client/bench_t2m.py. Same wire protocol; pure C++17 + platform.h.
//
// Output: a single stdout table with hardware info, per-run timings, and
// aggregate statistics. No files written. Pipe to tee if you want to save:
//   ./bench_t2m --config config.toml | tee logs/bench_$(hostname)_$(date +%s).txt
//
// Wire protocol to t2m_infer --server (host endian, must match t2m_infer):
//   send: u32 prompt_len, bytes[prompt_len], f32 duration, u32 seed
//   recv on success: u32 status=0, u32 T, u32 D, float[T*D], f32 h,r,dt,d0,dr
//   recv on error:   u32 status=1, u32 mlen, bytes[mlen]

#include "platform.h"
#include "config.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using clk = std::chrono::steady_clock;
static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        clk::now().time_since_epoch()).count();
}

// ---------- Socket helpers (same style as t2m_infer.cpp) -------------------
static bool send_all(sock_t s, const void* d, size_t n) {
    const char* p = (const char*)d;
    while (n > 0) {
        int w = (int)::send(s, p, (int)n, 0);
        if (w <= 0) return false;
        p += w; n -= (size_t)w;
    }
    return true;
}
static bool recv_all(sock_t s, void* d, size_t n) {
    char* p = (char*)d;
    while (n > 0) {
        int r = (int)::recv(s, p, (int)n, 0);
        if (r <= 0) return false;
        p += r; n -= (size_t)r;
    }
    return true;
}

// ---------- One bench run --------------------------------------------------
struct RunResult {
    bool        ok = false;
    std::string err;
    double      wall_ms = 0;
    float       hidden_ms = 0, refiner_ms = 0, dit_total_ms = 0,
                dit_step0_ms = 0, dit_rest_avg_ms = 0;
    uint32_t    T = 0, D = 0;
};

static RunResult one_run(const std::string& host, int port,
                         const std::string& prompt,
                         float duration, uint32_t seed)
{
    RunResult r;
    sock_t s = ::socket(AF_INET, SOCK_STREAM, 0);
    if (s == INVALID_SOCK) { r.err = "socket() failed"; return r; }

    int nodelay = 1;
    ::setsockopt(s, IPPROTO_TCP, TCP_NODELAY,
                 (const char*)&nodelay, sizeof(nodelay));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons((uint16_t)port);
    if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
        r.err = "bad host: " + host; close_sock(s); return r;
    }
    if (::connect(s, (sockaddr*)&addr, sizeof(addr)) == SOCKET_ERROR) {
        r.err = "connect failed (is t2m_infer --server running?)";
        close_sock(s); return r;
    }

    double t0 = now_ms();

    uint32_t plen = (uint32_t)prompt.size();
    if (!send_all(s, &plen, 4)
     || (plen > 0 && !send_all(s, prompt.data(), plen))
     || !send_all(s, &duration, 4)
     || !send_all(s, &seed,     4)) {
        r.err = "send failed"; close_sock(s); return r;
    }

    uint32_t status = 0;
    if (!recv_all(s, &status, 4)) { r.err = "recv status failed"; close_sock(s); return r; }
    if (status != 0) {
        uint32_t mlen = 0;
        if (!recv_all(s, &mlen, 4)) { r.err = "recv mlen failed"; close_sock(s); return r; }
        std::string m(mlen, '\0');
        if (mlen > 0) recv_all(s, m.data(), mlen);
        r.err = "server error: " + m;
        close_sock(s);
        return r;
    }

    if (!recv_all(s, &r.T, 4) || !recv_all(s, &r.D, 4)) {
        r.err = "recv shape failed"; close_sock(s); return r;
    }
    size_t total = (size_t)r.T * r.D;
    std::vector<float> motion(total);
    if (total > 0 && !recv_all(s, motion.data(), total * sizeof(float))) {
        r.err = "recv motion failed"; close_sock(s); return r;
    }

    if (!recv_all(s, &r.hidden_ms,       4)
     || !recv_all(s, &r.refiner_ms,      4)
     || !recv_all(s, &r.dit_total_ms,    4)
     || !recv_all(s, &r.dit_step0_ms,    4)
     || !recv_all(s, &r.dit_rest_avg_ms, 4)) {
        r.err = "recv timings failed"; close_sock(s); return r;
    }

    r.wall_ms = now_ms() - t0;
    r.ok = true;
    close_sock(s);
    return r;
}

// ---------- Stats ----------------------------------------------------------
struct Stats { double mean=0, p50=0, p95=0, mn=0, mx=0, std=0; };

static Stats compute(std::vector<double> v) {
    Stats s{};
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    double sum = 0;
    for (double x : v) sum += x;
    s.mean = sum / v.size();
    double sq = 0;
    for (double x : v) sq += (x - s.mean) * (x - s.mean);
    s.std = std::sqrt(sq / v.size());
    s.p50 = v[v.size() / 2];
    s.p95 = v[(size_t)std::min((double)v.size() - 1, v.size() * 0.95)];
    s.mn  = v.front();
    s.mx  = v.back();
    return s;
}

// ---------- main -----------------------------------------------------------
int main(int argc, char** argv) {
    std::string config_path = "config.toml";
    std::string host_override;
    int  port_override = -1;
    int  runs_override = -1;
    bool show_help = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "--config" && i+1 < argc) config_path    = argv[++i];
        else if (a == "--host"   && i+1 < argc) host_override  = argv[++i];
        else if (a == "--port"   && i+1 < argc) port_override  = std::stoi(argv[++i]);
        else if (a == "--runs"   && i+1 < argc) runs_override  = std::stoi(argv[++i]);
        else if (a == "-h" || a == "--help") show_help = true;
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 1; }
    }
    if (show_help) {
        std::printf("usage: bench_t2m [--config config.toml] [--host H] [--port P] [--runs N]\n");
        return 0;
    }

    AppConfig cfg;
    try {
        cfg = AppConfig::from_toml(config_path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[bench] %s — falling back to defaults\n", e.what());
        cfg = AppConfig::defaults();
    }

    std::string host = host_override.empty() ? "127.0.0.1" : host_override;
    int  port = (port_override > 0) ? port_override : cfg.t2m.server_port;
    int  warmup = cfg.bench.warmup_runs;
    int  timed  = (runs_override > 0) ? runs_override : cfg.bench.timed_runs;

    PlatformNetInit net_init;

    std::printf("==============================================================================\n");
    std::printf(" SimpleT2M bench  (target: %s:%d)\n", host.c_str(), port);
    std::printf("------------------------------------------------------------------------------\n");
    std::printf("  config:  %s\n", config_path.c_str());
    std::printf("  warmup:  %d runs    timed: %d runs    duration: %.1fs    seed: %u\n",
                warmup, timed, cfg.bench.duration, cfg.bench.seed);
    std::printf("  prompts: %zu\n", cfg.bench.prompts.size());
    if (cfg.bench.prompts.empty()) {
        std::fprintf(stderr, "[bench] no prompts configured\n");
        return 1;
    }
    std::printf("==============================================================================\n\n");

    // Warmup
    std::printf("warming up (%d runs)...\n", warmup);
    for (int i = 0; i < warmup; i++) {
        const auto& p = cfg.bench.prompts[i % cfg.bench.prompts.size()];
        auto r = one_run(host, port, p, cfg.bench.duration, cfg.bench.seed);
        if (!r.ok) {
            std::fprintf(stderr, "  warmup #%d failed: %s\n", i+1, r.err.c_str());
            return 2;
        }
        std::printf("  #%d  wall=%6.1fms  hidden=%5.1f  dit=%5.1f\n",
                    i+1, r.wall_ms, r.hidden_ms, r.dit_total_ms);
    }

    // Timed
    std::printf("\ntimed (%d runs)...\n", timed);
    std::vector<double> wall, hid, ref, dit, d0, dr;
    int n_fail = 0;
    for (int i = 0; i < timed; i++) {
        const auto& p = cfg.bench.prompts[i % cfg.bench.prompts.size()];
        auto r = one_run(host, port, p, cfg.bench.duration, cfg.bench.seed);
        if (!r.ok) {
            std::fprintf(stderr, "  #%2d FAIL: %s\n", i+1, r.err.c_str());
            n_fail++;
            continue;
        }
        std::printf("  #%2d  wall=%6.1fms  hidden=%4.1f  refiner=%.2f  dit=%5.1f"
                    " (s0=%4.1f rest=%4.1f)  \"%s\"\n",
                    i+1, r.wall_ms, r.hidden_ms, r.refiner_ms,
                    r.dit_total_ms, r.dit_step0_ms, r.dit_rest_avg_ms,
                    p.c_str());
        wall.push_back(r.wall_ms);
        hid.push_back(r.hidden_ms);
        ref.push_back(r.refiner_ms);
        dit.push_back(r.dit_total_ms);
        d0.push_back(r.dit_step0_ms);
        dr.push_back(r.dit_rest_avg_ms);
    }

    auto sw = compute(wall);
    auto sh = compute(hid);
    auto sr = compute(ref);
    auto sd = compute(dit);
    auto s0 = compute(d0);
    auto sR = compute(dr);

    auto row = [](const char* name, const Stats& s, const char* unit){
        std::printf("  %-16s %6.1f%s  %6.1f%s  %6.1f%s  %6.1f%s  %6.1f%s  %6.1f%s\n",
            name,
            s.mean,unit, s.p50,unit, s.p95,unit, s.mn,unit, s.mx,unit, s.std,unit);
    };

    std::printf("==============================================================================\n");
    std::printf("  %-16s %8s %8s %8s %8s %8s %8s\n",
                "stage","mean","p50","p95","min","max","std");
    std::printf("------------------------------------------------------------------------------\n");
    row("client_wall",  sw, "ms");
    row("hidden",       sh, "ms");
    row("refiner",      sr, "ms");
    row("dit_total",    sd, "ms");
    row("dit_step0",    s0, "ms");
    row("dit_rest_avg", sR, "ms");
    std::printf("==============================================================================\n");

    if (n_fail) std::printf("  %d/%d runs FAILED\n", n_fail, timed);
    if (!wall.empty()) {
        std::printf("  client wall mean %.0fms\n", sw.mean);
        // crude thermal-throttling heuristic
        bool stable = (sw.std < 0.3 * sw.mean) && (sw.mx < 2.0 * sw.p50);
        std::printf("  stability: %s  (std/mean=%.0f%%, max/p50=%.2fx)\n",
                    stable ? "OK" : "SUSPECT (possible thermal throttle?)",
                    100.0 * sw.std / std::max(sw.mean, 1e-9),
                    sw.mx / std::max(sw.p50, 1e-9));
    }
    return n_fail ? 3 : 0;
}
