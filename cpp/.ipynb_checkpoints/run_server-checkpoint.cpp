// cpp/run_server.cpp
//
// SimpleT2M streaming demo HTTP server. Replaces run.py.
//
// Endpoints:
//   GET  /                  -> index.html
//   GET  /assets/<file>     -> static .vrm / .vrma (via cpp-httplib mount)
//   POST /t2m  {prompt, duration?, seed?}  -> binary .vrma
//
// Pipeline for /t2m:
//   request body (JSON, tiny)
//     -> TCP call to t2m_infer (port 8423) -> motion fp32 buffer
//     -> write tmp motion.bin
//     -> exec motion_to_vrma -> tmp .vrma
//     -> stream .vrma bytes back as response
//
// Cross-platform: Linux + macOS + Windows. Uses cpp-httplib (single-header,
// vendored) for HTTP. No JSON library — request body has 3 fields, parsed
// inline (~30 lines).

#include "platform.h"
#include "config.h"

#include <httplib.h>

#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#  include <windows.h>
#  include <shellapi.h>
#  define popen  _popen
#  define pclose _pclose
#else
#  include <unistd.h>
#  include <sys/wait.h>
#endif

namespace fs = std::filesystem;
using clk = std::chrono::steady_clock;

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        clk::now().time_since_epoch()).count();
}

// ============================================================
// Mini JSON reader (flat object only: {"k":"v","k":n,"k":n.n})
// Just enough for {prompt, duration, seed}. NOT a real parser:
// no nesting, no arrays, no escapes other than \" and \\.
// Returns false on any malformedness — caller sends 400.
// ============================================================
static std::string json_unescape(const std::string& s) {
    std::string out; out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            char c = s[++i];
            switch (c) {
                case '"':  out += '"';  break;
                case '\\': out += '\\'; break;
                case '/':  out += '/';  break;
                case 'n':  out += '\n'; break;
                case 't':  out += '\t'; break;
                case 'r':  out += '\r'; break;
                default:   out += c;    break;  // tolerate, don't fail
            }
        } else out += s[i];
    }
    return out;
}

struct JsonVals {
    std::string prompt;
    float       duration = 3.0f;
    uint32_t    seed     = 42;
    bool        has_prompt = false;
};

static bool parse_t2m_body(const std::string& body, JsonVals& out, std::string& err) {
    // Find each key naively. Order/whitespace tolerant; not RFC 8259.
    auto find_key = [&](const char* key, size_t& kpos, size_t& vstart) -> bool {
        std::string needle = std::string("\"") + key + "\"";
        size_t p = body.find(needle);
        if (p == std::string::npos) return false;
        size_t c = body.find(':', p + needle.size());
        if (c == std::string::npos) return false;
        // skip whitespace after colon
        size_t v = c + 1;
        while (v < body.size() && std::isspace((unsigned char)body[v])) ++v;
        kpos = p; vstart = v;
        return true;
    };

    size_t kp, vs;

    // prompt: required, string
    if (find_key("prompt", kp, vs)) {
        if (vs >= body.size() || body[vs] != '"') {
            err = "prompt must be a string"; return false;
        }
        // find closing unescaped quote
        size_t e = vs + 1;
        while (e < body.size()) {
            if (body[e] == '\\') { e += 2; continue; }
            if (body[e] == '"') break;
            ++e;
        }
        if (e >= body.size()) { err = "unterminated prompt string"; return false; }
        out.prompt = json_unescape(body.substr(vs + 1, e - vs - 1));
        out.has_prompt = !out.prompt.empty();
    }

    // duration: optional, number
    if (find_key("duration", kp, vs)) {
        try {
            size_t pos = 0;
            out.duration = std::stof(body.substr(vs), &pos);
            if (out.duration <= 0 || out.duration > 60.0f) {
                err = "duration out of range (0, 60]"; return false;
            }
        } catch (...) { err = "duration: bad number"; return false; }
    }

    // seed: optional, integer
    if (find_key("seed", kp, vs)) {
        try {
            size_t pos = 0;
            long long v = std::stoll(body.substr(vs), &pos);
            if (v < 0 || v > 0xFFFFFFFFLL) { err = "seed out of range"; return false; }
            out.seed = (uint32_t)v;
        } catch (...) { err = "seed: bad number"; return false; }
    }

    return true;
}

// ============================================================
// t2m_infer wire protocol (host-endian / LE assumed, matches
// the in-tree t2m_infer.cpp and bench_t2m.cpp).
// ============================================================
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

struct MotionResult {
    bool        ok = false;
    std::string err;
    std::vector<float> data;
    uint32_t    T = 0, D = 0;
    float       hidden_ms = 0, refiner_ms = 0,
                dit_total_ms = 0, dit_step0_ms = 0, dit_rest_avg_ms = 0;
};

static MotionResult call_t2m(const std::string& host, int port,
                             const std::string& prompt,
                             float duration, uint32_t seed) {
    MotionResult r;
    sock_t s = ::socket(AF_INET, SOCK_STREAM, 0);
    if (s == INVALID_SOCK) { r.err = "socket()"; return r; }

    int one = 1;
    ::setsockopt(s, IPPROTO_TCP, TCP_NODELAY, (const char*)&one, sizeof(one));

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

    uint32_t plen = (uint32_t)prompt.size();
    if (!send_all(s, &plen, 4)
     || (plen > 0 && !send_all(s, prompt.data(), plen))
     || !send_all(s, &duration, 4)
     || !send_all(s, &seed, 4)) {
        r.err = "send failed"; close_sock(s); return r;
    }

    uint32_t status = 0;
    if (!recv_all(s, &status, 4)) { r.err = "recv status"; close_sock(s); return r; }
    if (status != 0) {
        uint32_t mlen = 0;
        recv_all(s, &mlen, 4);
        std::string m(mlen, '\0');
        if (mlen > 0) recv_all(s, m.data(), mlen);
        r.err = "t2m_infer: " + m;
        close_sock(s); return r;
    }

    if (!recv_all(s, &r.T, 4) || !recv_all(s, &r.D, 4)) {
        r.err = "recv shape"; close_sock(s); return r;
    }
    size_t total = (size_t)r.T * r.D;
    r.data.resize(total);
    if (total > 0 && !recv_all(s, r.data.data(), total * sizeof(float))) {
        r.err = "recv motion"; close_sock(s); return r;
    }

    if (!recv_all(s, &r.hidden_ms,       4)
     || !recv_all(s, &r.refiner_ms,      4)
     || !recv_all(s, &r.dit_total_ms,    4)
     || !recv_all(s, &r.dit_step0_ms,    4)
     || !recv_all(s, &r.dit_rest_avg_ms, 4)) {
        r.err = "recv timings"; close_sock(s); return r;
    }

    r.ok = true;
    close_sock(s);
    return r;
}

// ============================================================
// motion_to_vrma subprocess
// ============================================================
static std::string shell_quote(const std::string& s) {
    // Path-safe quoting. We control the inputs (tmp paths, prompt sliced),
    // but escape just in case the prompt sneaks weirdness through.
#ifdef _WIN32
    std::string out = "\"";
    for (char c : s) { if (c == '"') out += '\\'; out += c; }
    out += "\"";
    return out;
#else
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') out += "'\\''";   // close, escaped ', reopen
        else out += c;
    }
    out += "'";
    return out;
#endif
}

static bool run_motion_to_vrma(const fs::path& bin,
                               const fs::path& bin_in,
                               const fs::path& vrma_out,
                               const std::string& title,
                               std::string& err) {
    std::string title_trunc = title.substr(0, 60);
    std::string cmd =
        shell_quote(bin.string()) +
        " --in "    + shell_quote(bin_in.string()) +
        " --out "   + shell_quote(vrma_out.string()) +
        " --title " + shell_quote(title_trunc) +
        " 2>&1";

    FILE* p = popen(cmd.c_str(), "r");
    if (!p) { err = "popen failed"; return false; }
    std::string buf;
    char chunk[1024];
    while (size_t n = std::fread(chunk, 1, sizeof(chunk), p)) buf.append(chunk, n);
    int rc = pclose(p);
    if (rc != 0) {
        err = "motion_to_vrma rc=" + std::to_string(rc) +
              (buf.empty() ? "" : ": " + buf);
        return false;
    }
    if (!fs::exists(vrma_out)) {
        err = "motion_to_vrma produced no output"; return false;
    }
    return true;
}

// ============================================================
// Helpers
// ============================================================
static std::string read_file(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return "";
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static fs::path unique_tmp_dir() {
    fs::path base = fs::temp_directory_path();
    static std::atomic<uint64_t> ctr{0};
    std::mt19937_64 rng((uint64_t)now_ms() ^ (ctr.fetch_add(1) * 0x9E3779B97F4A7C15ULL));
    char name[64];
    std::snprintf(name, sizeof(name), "simplet2m_%016llx",
                  (unsigned long long)rng());
    fs::path p = base / name;
    fs::create_directories(p);
    return p;
}

static void open_browser(const std::string& url) {
#ifdef _WIN32
    ShellExecuteA(nullptr, "open", url.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
#elif defined(__APPLE__)
    std::string cmd = "open " + shell_quote(url) + " >/dev/null 2>&1 &";
    (void)std::system(cmd.c_str());
#else
    std::string cmd = "xdg-open " + shell_quote(url) + " >/dev/null 2>&1 &";
    (void)std::system(cmd.c_str());
#endif
}

// ============================================================
// Main
// ============================================================
int main(int argc, char** argv) {
    std::string config_path = "config.toml";
    bool open_browser_flag = true;
    int  port_override = -1;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--config" && i+1 < argc) config_path = argv[++i];
        else if (a == "--port"   && i+1 < argc) port_override = std::stoi(argv[++i]);
        else if (a == "--no-browser") open_browser_flag = false;
        else if (a == "-h" || a == "--help") {
            std::printf("usage: run_server [--config config.toml] [--port 8765] [--no-browser]\n");
            return 0;
        } else {
            std::fprintf(stderr, "unknown arg: %s\n", a.c_str());
            return 1;
        }
    }

    AppConfig cfg;
    try {
        cfg = AppConfig::from_toml(config_path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[run_server] config: %s\n", e.what());
        std::fprintf(stderr, "[run_server] falling back to defaults\n");
        cfg = AppConfig::defaults();
    }

    PlatformNetInit net_init;

    // Resolve paths. The web root holds index.html / assets / motion_to_vrma.
    // It's the directory containing config.toml (or cwd if no config).
    fs::path root = cfg.config_dir.empty() ? fs::current_path()
                                           : fs::path(cfg.config_dir);
    fs::path index_html  = root / "index.html";
    fs::path assets_dir  = root / "assets";
    fs::path mtv_bin     = cfg.t2m.motion_to_vrma_bin.empty()
                              ? (root / "build" / "motion_to_vrma")
                              : fs::path(cfg.t2m.motion_to_vrma_bin);
#ifdef _WIN32
    // On Windows the binary is build\Release\motion_to_vrma.exe.
    if (!fs::exists(mtv_bin)) {
        fs::path alt = mtv_bin.parent_path() / "Release" / (mtv_bin.filename().string() + ".exe");
        if (fs::exists(alt)) mtv_bin = alt;
        else if (fs::exists(mtv_bin.string() + ".exe")) mtv_bin = mtv_bin.string() + ".exe";
    }
#endif

    int  port      = (port_override > 0) ? port_override : cfg.run.port;
    std::string t2m_host = cfg.t2m.fused_host;       // 127.0.0.1
    int  t2m_port  = cfg.t2m.server_port;            // 8423

    // ---- Preflight ----
    std::printf("SimpleT2M streaming demo (run_server)\n");
    std::printf("  root:      %s\n", root.string().c_str());
    std::printf("  config:    %s\n", config_path.c_str());
    std::printf("  port:      %d\n", port);
    std::printf("  t2m:       %s:%d\n", t2m_host.c_str(), t2m_port);
    std::printf("  index:     %s %s\n", index_html.string().c_str(),
                fs::exists(index_html) ? "OK" : "MISSING");
    std::printf("  assets:    %s %s\n", assets_dir.string().c_str(),
                fs::is_directory(assets_dir) ? "OK" : "MISSING");
    std::printf("  mtv bin:   %s %s\n", mtv_bin.string().c_str(),
                fs::exists(mtv_bin) ? "OK" : "MISSING");
    if (!fs::exists(index_html) || !fs::is_directory(assets_dir)
     || !fs::exists(mtv_bin)) {
        std::fprintf(stderr, "[run_server] preflight failed\n");
        return 1;
    }

    // ---- HTTP server ----
    httplib::Server svr;

    // Static assets. cpp-httplib mount points include path-traversal
    // protection (it canonicalizes and checks prefix), MIME guessing,
    // range support, etc. The custom MIME map below ensures .vrm/.vrma
    // come back as model/gltf-binary so three-vrm parses them correctly.
    svr.set_mount_point("/assets", assets_dir.string());
    svr.set_file_extension_and_mimetype_mapping("vrm",  "model/gltf-binary");
    svr.set_file_extension_and_mimetype_mapping("vrma", "model/gltf-binary");

    // GET /
    svr.Get("/", [&](const httplib::Request&, httplib::Response& res) {
        std::string body = read_file(index_html);
        if (body.empty()) { res.status = 500; res.set_content("index.html missing", "text/plain"); return; }
        res.set_content(body, "text/html; charset=utf-8");
        res.set_header("Cache-Control", "no-store");
    });

    // POST /t2m — serialize concurrent requests. fused_server is single-mutex
    // anyway, and concurrent motion_to_vrma processes would just fight for
    // tmp dir + CPU. Sequential = predictable demo behavior.
    std::mutex t2m_mtx;
    svr.Post("/t2m", [&](const httplib::Request& req, httplib::Response& res) {
        std::lock_guard<std::mutex> _lk(t2m_mtx);

        JsonVals j;
        std::string err;
        if (!parse_t2m_body(req.body, j, err) || !j.has_prompt) {
            res.status = 400;
            res.set_content(err.empty() ? "missing prompt" : err, "text/plain");
            return;
        }

        std::fprintf(stderr, "[t2m] \"%s\" dur=%.1f seed=%u\n",
                     j.prompt.c_str(), j.duration, j.seed);
        double t0 = now_ms();

        auto m = call_t2m(t2m_host, t2m_port, j.prompt, j.duration, j.seed);
        if (!m.ok) {
            std::fprintf(stderr, "[t2m] FAIL: %s\n", m.err.c_str());
            res.status = 500; res.set_content(m.err, "text/plain");
            return;
        }

        // tmp dir auto-cleaned via RAII guard below
        fs::path tdir;
        try { tdir = unique_tmp_dir(); }
        catch (const std::exception& e) {
            res.status = 500; res.set_content(std::string("tmp dir: ") + e.what(), "text/plain");
            return;
        }
        struct DirGuard {
            fs::path p;
            ~DirGuard() { std::error_code ec; fs::remove_all(p, ec); }
        } guard{tdir};

        fs::path bin_path  = tdir / "motion.bin";
        fs::path vrma_path = tdir / "motion.vrma";
        {
            std::ofstream f(bin_path, std::ios::binary);
            f.write((const char*)m.data.data(), m.data.size() * sizeof(float));
            if (!f) {
                res.status = 500; res.set_content("write motion.bin failed", "text/plain");
                return;
            }
        }

        std::string mtv_err;
        if (!run_motion_to_vrma(mtv_bin, bin_path, vrma_path, j.prompt, mtv_err)) {
            std::fprintf(stderr, "[t2m] motion_to_vrma: %s\n", mtv_err.c_str());
            res.status = 500; res.set_content(mtv_err, "text/plain");
            return;
        }

        std::string vrma_bytes = read_file(vrma_path);
        if (vrma_bytes.empty()) {
            res.status = 500; res.set_content("empty vrma", "text/plain");
            return;
        }

        double wall = now_ms() - t0;
        std::fprintf(stderr, "[t2m] OK %zuB in %.0fms\n", vrma_bytes.size(), wall);

        res.set_content(std::move(vrma_bytes), "model/gltf-binary");
        res.set_header("Cache-Control", "no-store");
    });

    // Catch-all 404 — cpp-httplib's default error page is fine; we just want
    // a no-cache header so dev iteration isn't blocked by stale 404s.
    svr.set_error_handler([](const httplib::Request&, httplib::Response& res) {
        res.set_header("Cache-Control", "no-store");
    });

    // Graceful shutdown on SIGINT/SIGTERM. cpp-httplib has a thread-safe stop().
    static httplib::Server* g_svr = &svr;
    std::signal(SIGINT,  [](int){ if (g_svr) g_svr->stop(); });
#ifndef _WIN32
    std::signal(SIGTERM, [](int){ if (g_svr) g_svr->stop(); });
    std::signal(SIGPIPE, SIG_IGN);
#endif

    std::string url = "http://localhost:" + std::to_string(port) + "/";
    std::printf("\nserving on %s  (Ctrl-C to stop)\n", url.c_str());
    std::fflush(stdout);

    if (open_browser_flag) {
        // Fire after a tiny delay so the listen() has bound the port first
        // (otherwise the browser race-opens before the server is ready and
        // the user sees a connection-refused flash). 300ms is plenty.
        std::thread([url]{
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
            open_browser(url);
        }).detach();
    }

    if (!svr.listen("0.0.0.0", port)) {
        std::fprintf(stderr, "[run_server] listen failed (port %d in use?)\n", port);
        return 1;
    }
    std::printf("stopped.\n");
    return 0;
}
