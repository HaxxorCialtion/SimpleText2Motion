// ============================================================================
// fused_server.cpp  (V2 — generic SimpleTool decode service, cross-platform)
// SimpleLove × SimpleTool unified server
// ============================================================================
//
// CHANGES FROM V1:
//   - Removed all business logic (tools, t2m templates, system prompt)
//   - simpletool port (8422) protocol completely changed:
//       client sends [cold_bytes][hot_bytes][head_mask][max_gen]
//       server returns per-head text only — NO t2m payload
//   - Server maintains a dynamic cold-prefix KV cache with byte-identical
//     hit detection. Hits skip cold prefill entirely (~100ms saved).
//   - Cold prefill is now lazy (no startup pre-prefill)
//   - Cross-platform socket layer via platform_net.h (Win + Linux)
//
// LOCAL EDITS (vs upstream V2):
//   1. now_ms() uses epoch time (cross-process comparable)
//   2. argv parsing via std::stoi + range check (parse_int_arg)
//   3. SIGINT/SIGTERM graceful shutdown (g_stop + listening-socket shutdown)
//   4. --config <path> support; defaults come from config.h (AppConfig)
//
// PORTS:
//   - Port 8421 (hidden):     unchanged — prompt -> hidden state
//   - Port 8422 (simpletool): NEW protocol, see PROTOCOL block at file end
//
// Build (Windows, from project root):
//   cmake --build build_win --config Release --target fused_server_win
// ============================================================================

#include "llama.h"
#include "platform_net.h"
#include "config.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// ============================================================================
// Utilities
// ============================================================================

static double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(
        clock::now().time_since_epoch()).count();
}

static bool read_all(socket_t fd, void * buf, size_t n) {
    char * p = (char *) buf;
    while (n > 0) {
        ssize_t r = sock_read(fd, p, n);
        if (r <= 0) return false;
        p += r; n -= (size_t) r;
    }
    return true;
}

static bool write_all(socket_t fd, const void * buf, size_t n) {
    const char * p = (const char *) buf;
    while (n > 0) {
        ssize_t w = sock_write(fd, p, n);
        if (w <= 0) return false;
        p += w; n -= (size_t) w;
    }
    return true;
}

// ----- argv parsing helper (改动 2) -----------------------------------------
static int parse_int_arg(const char * name, const char * val, int lo, int hi) {
    try {
        size_t pos = 0;
        int v = std::stoi(val, &pos);
        if (pos != strlen(val)) throw std::invalid_argument("trailing chars");
        if (v < lo || v > hi) {
            fprintf(stderr, "%s out of range [%d, %d]: %d\n", name, lo, hi, v);
            std::exit(1);
        }
        return v;
    } catch (const std::exception & e) {
        fprintf(stderr, "invalid value for %s: '%s' (%s)\n", name, val, e.what());
        std::exit(1);
    }
}

// ============================================================================
// Tokenizer helpers
// ============================================================================

static std::vector<llama_token> tokenize_str(const llama_vocab * vocab,
                                             const std::string & text,
                                             bool add_special, bool parse_special) {
    int n_max = (int) text.size() + 256;
    std::vector<llama_token> toks(n_max);
    int n = llama_tokenize(vocab, text.c_str(), (int) text.size(),
                           toks.data(), n_max, add_special, parse_special);
    if (n < 0) {
        toks.resize(-n);
        n = llama_tokenize(vocab, text.c_str(), (int) text.size(),
                           toks.data(), -n, add_special, parse_special);
    }
    toks.resize(n);
    return toks;
}

static llama_token find_single_token(const llama_vocab * vocab, const char * text) {
    llama_token tmp[8];
    int n = llama_tokenize(vocab, text, (int) strlen(text), tmp, 8, false, true);
    if (n == 1) return tmp[0];
    return -1;
}

static std::string token_to_str(const llama_vocab * vocab, llama_token id) {
    char buf[256];
    int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
    if (n > 0) return std::string(buf, n);
    return "";
}

static std::string detokenize_ids(const llama_vocab * vocab,
                                  const std::vector<llama_token> & ids) {
    std::string r;
    for (auto t : ids) r += token_to_str(vocab, t);
    return r;
}

// ============================================================================
// Head configuration (locked at compile time per design decision)
// ============================================================================

static const int MAX_HEADS = 8;

struct HeadConfig {
    const char * name;
    const char * open_tag;
    const char * close_tag;
    llama_token  open_id  = -1;
    llama_token  close_id = -1;
};

static HeadConfig ALL_HEADS[MAX_HEADS] = {
    {"content",  "<content>",  "</content>"},
    {"function", "<function>", "</function>"},
    {"arg1",     "<arg1>",     "</arg1>"},
    {"arg2",     "<arg2>",     "</arg2>"},
    {"arg3",     "<arg3>",     "</arg3>"},
    {"arg4",     "<arg4>",     "</arg4>"},
    {"arg5",     "<arg5>",     "</arg5>"},
    {"arg6",     "<arg6>",     "</arg6>"},
};

// ============================================================================
// Global server state
// ============================================================================

struct ColdCache {
    std::vector<llama_token> tokens;
    int                      n_cached = 0;
};

struct FusedState {
    llama_model       * model = nullptr;
    const llama_vocab * vocab = nullptr;

    llama_context * ctx_hidden = nullptr;
    std::mutex      mtx_hidden;

    llama_context * ctx_simpletool = nullptr;
    std::mutex      mtx_simpletool;

    int n_embd          = 0;
    int n_ctx           = 2048;
    int max_cold_tokens = 0;

    llama_token null_id   = -1;
    llama_token im_end_id = -1;

    ColdCache cold_cache;

    bool debug_cache = false;
};

// ============================================================================
// Signal handling / graceful shutdown (改动 3)
// ============================================================================
//
// `accept()` is blocking on both Windows and POSIX. To unblock the server
// threads from a signal handler we stash the listening socket fds in atomics
// and shutdown()/closesocket() them on signal. These three globals are the
// only mutable globals in the file — they exist because signal handlers can
// only touch globals safely.

static std::atomic<bool>     g_stop{false};
static std::atomic<socket_t> g_srv_hidden{INVALID_SOCKET_VALUE};
static std::atomic<socket_t> g_srv_simpletool{INVALID_SOCKET_VALUE};

static void on_signal(int /*sig*/) {
    g_stop.store(true);
    socket_t s1 = g_srv_hidden.exchange(INVALID_SOCKET_VALUE);
    socket_t s2 = g_srv_simpletool.exchange(INVALID_SOCKET_VALUE);
    if (s1 != INVALID_SOCKET_VALUE) {
#ifdef _WIN32
        closesocket(s1);
#else
        ::shutdown(s1, SHUT_RDWR);
#endif
    }
    if (s2 != INVALID_SOCKET_VALUE) {
#ifdef _WIN32
        closesocket(s2);
#else
        ::shutdown(s2, SHUT_RDWR);
#endif
    }
}

// ============================================================================
// HIDDEN EXTRACTION (port 8421 — unchanged from v1)
// ============================================================================

static bool extract_hidden(FusedState & s,
                           const std::string & prompt,
                           std::vector<float> & out_hidden,
                           int & out_n_tokens,
                           int & out_n_embd,
                           double & out_forward_ms,
                           std::string & err) {
    std::lock_guard<std::mutex> lk(s.mtx_hidden);

    auto toks = tokenize_str(s.vocab, prompt, true, false);
    if (toks.empty()) { err = "tokenize failed"; return false; }
    if ((int) toks.size() > s.n_ctx) {
        err = "prompt exceeds n_ctx (" + std::to_string(toks.size())
              + " > " + std::to_string(s.n_ctx) + ")";
        return false;
    }

    llama_memory_t mem = llama_get_memory(s.ctx_hidden);
    llama_memory_clear(mem, true);

    double t0 = now_ms();
    const int n_tok = (int) toks.size();
    llama_batch batch = llama_batch_init(n_tok, 0, 1);
    batch.n_tokens = n_tok;
    for (int i = 0; i < n_tok; ++i) {
        batch.token[i]     = toks[i];
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = 1;
    }

    int rc = llama_decode(s.ctx_hidden, batch);
    llama_batch_free(batch);
    if (rc != 0) { err = "llama_decode failed rc=" + std::to_string(rc); return false; }

    const float * h = llama_get_hidden_layer(s.ctx_hidden);
    int nt          = llama_hidden_layer_n_tokens(s.ctx_hidden);
    out_forward_ms  = now_ms() - t0;

    if (!h || nt <= 0) { err = "hidden buffer empty"; return false; }

    out_n_tokens = nt;
    out_n_embd   = s.n_embd;
    out_hidden.assign(h, h + (size_t) nt * s.n_embd);
    return true;
}

// ============================================================================
// HIDDEN-PORT server thread (unchanged from v1, but now uses g_stop)
// ============================================================================

static bool handle_hidden_request(socket_t fd, FusedState & s) {
    uint32_t plen_n;
    if (!read_all(fd, &plen_n, 4)) return false;
    uint32_t plen = ntohl(plen_n);
    if (plen == 0 || plen > 65536) {
        fprintf(stderr, "[hidden] bad prompt_len=%u\n", plen);
        return false;
    }
    std::string prompt(plen, '\0');
    if (!read_all(fd, prompt.data(), plen)) return false;

    std::vector<float> hidden;
    int n_tokens = 0, n_embd = 0;
    double forward_ms = 0;
    std::string err;

    if (!extract_hidden(s, prompt, hidden, n_tokens, n_embd, forward_ms, err)) {
        fprintf(stderr, "[hidden] error: %s\n", err.c_str());
        int32_t status_n = htonl((uint32_t) (int32_t) -1);
        write_all(fd, &status_n, 4);
        return true;
    }

    int32_t header[3] = {
        (int32_t) htonl((uint32_t) 0),
        (int32_t) htonl((uint32_t) n_tokens),
        (int32_t) htonl((uint32_t) n_embd),
    };
    if (!write_all(fd, header, sizeof(header))) return false;
    if (!write_all(fd, hidden.data(), hidden.size() * sizeof(float))) return false;
    if (!write_all(fd, &forward_ms, sizeof(double))) return false;

    printf("[hidden] prompt_len=%u n_tokens=%d forward=%.2f ms\n",
           plen, n_tokens, forward_ms);
    fflush(stdout);
    return true;
}

static void hidden_server_thread(FusedState * s, int port, std::atomic<bool> * stop) {
    socket_t srv = socket(AF_INET, SOCK_STREAM, 0);
    if (srv == INVALID_SOCKET_VALUE) { perror("[hidden] socket"); return; }
    int opt = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, (const char *) &opt, sizeof(opt));
    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port        = htons((uint16_t) port);
    if (bind(srv, (sockaddr*) &addr, sizeof(addr)) < 0) { perror("[hidden] bind"); return; }
    if (listen(srv, 8) < 0) { perror("[hidden] listen"); return; }
    g_srv_hidden.store(srv);
    printf("[hidden] listening on 127.0.0.1:%d\n", port);
    fflush(stdout);

    while (!stop->load()) {
        sockaddr_in caddr{};
        socklen_t_compat clen = sizeof(caddr);
        socket_t cfd = accept(srv, (sockaddr*) &caddr, &clen);
        if (cfd == INVALID_SOCKET_VALUE) {
            if (stop->load()) break;
            perror("[hidden] accept");
            continue;
        }
        int one = 1;
        setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, (const char *) &one, sizeof(one));
        while (handle_hidden_request(cfd, *s)) { /* keep connection */ }
        closesocket_compat(cfd);
    }
    // If signal handler already closed srv, exchange returned INVALID; only
    // close here if we still own it.
    socket_t owned = g_srv_hidden.exchange(INVALID_SOCKET_VALUE);
    if (owned != INVALID_SOCKET_VALUE) closesocket_compat(owned);
}

// ============================================================================
// COLD CACHE management
// ============================================================================

enum ColdAction : uint8_t {
    COLD_HIT    = 0,   // bytes-identical to cache → no prefill
    COLD_EXTEND = 1,   // cache is a true prefix of new cold → prefill the rest
    COLD_RESET  = 2,   // mismatch / fresh → clear KV and prefill all
};

// Determine action and prefill whatever's needed. Updates cache.
// Caller must hold s.mtx_simpletool.
static bool ensure_cold_cache(FusedState & s,
                              const std::vector<llama_token> & new_cold,
                              ColdAction & out_action,
                              double & out_dt_ms,
                              std::string & err) {
    out_dt_ms = 0;

    const int n_new = (int) new_cold.size();
    const int n_old = (int) s.cold_cache.tokens.size();
    llama_memory_t mem = llama_get_memory(s.ctx_simpletool);

    // Case 1: completely identical
    if (n_new == n_old &&
        std::memcmp(s.cold_cache.tokens.data(), new_cold.data(),
                    sizeof(llama_token) * n_new) == 0) {
        out_action = COLD_HIT;
        if (s.debug_cache) {
            fprintf(stderr, "[cache] HIT n_cold=%d (no prefill)\n", n_new);
        }
        return true;
    }

    // Case 2: new is a strict extension of old
    bool is_extension = false;
    if (n_new > n_old && n_old > 0) {
        is_extension = (std::memcmp(s.cold_cache.tokens.data(), new_cold.data(),
                                    sizeof(llama_token) * n_old) == 0);
    }

    double t0 = now_ms();

    if (is_extension) {
        const int n_extra = n_new - n_old;
        if (s.debug_cache) {
            fprintf(stderr, "[cache] EXTEND from %d to %d (prefill %d new tok)\n",
                    n_old, n_new, n_extra);
        }
        llama_batch batch = llama_batch_init(n_extra, 0, MAX_HEADS);
        batch.n_tokens = 0;
        for (int i = 0; i < n_extra; i++) {
            int idx = batch.n_tokens;
            batch.token[idx]     = new_cold[n_old + i];
            batch.pos[idx]       = n_old + i;
            batch.n_seq_id[idx]  = MAX_HEADS;
            for (int h = 0; h < MAX_HEADS; h++) batch.seq_id[idx][h] = h;
            batch.logits[idx]    = 0;
            batch.n_tokens++;
        }
        int rc = llama_decode(s.ctx_simpletool, batch);
        llama_synchronize(s.ctx_simpletool);
        llama_batch_free(batch);
        if (rc != 0) {
            err = "cold extend prefill rc=" + std::to_string(rc);
            return false;
        }
        s.cold_cache.tokens = new_cold;
        s.cold_cache.n_cached = n_new;
        out_action = COLD_EXTEND;
    } else {
        // Case 3: full reset
        if (s.debug_cache) {
            fprintf(stderr, "[cache] RESET (old=%d new=%d, mismatch or shorter)\n",
                    n_old, n_new);
        }
        llama_memory_clear(mem, true);
        s.cold_cache.tokens.clear();
        s.cold_cache.n_cached = 0;

        if (n_new > 0) {
            llama_batch batch = llama_batch_init(n_new, 0, MAX_HEADS);
            batch.n_tokens = 0;
            for (int i = 0; i < n_new; i++) {
                int idx = batch.n_tokens;
                batch.token[idx]     = new_cold[i];
                batch.pos[idx]       = i;
                batch.n_seq_id[idx]  = MAX_HEADS;
                for (int h = 0; h < MAX_HEADS; h++) batch.seq_id[idx][h] = h;
                batch.logits[idx]    = 0;
                batch.n_tokens++;
            }
            int rc = llama_decode(s.ctx_simpletool, batch);
            llama_synchronize(s.ctx_simpletool);
            llama_batch_free(batch);
            if (rc != 0) {
                err = "cold reset prefill rc=" + std::to_string(rc);
                return false;
            }
            s.cold_cache.tokens = new_cold;
            s.cold_cache.n_cached = n_new;
        }
        out_action = COLD_RESET;
    }

    out_dt_ms = now_ms() - t0;
    return true;
}

// ============================================================================
// SIMPLETOOL parallel multi-head decoding
// ============================================================================

struct HeadResult {
    int         head_idx;
    std::string name;
    std::vector<llama_token> gen_ids;
    std::string decoded;
    bool        is_null   = false;
    bool        hit_close = false;
    int         num_tokens = 0;
};

struct SimpleToolOutput {
    std::vector<HeadResult> heads;
    ColdAction cold_action = COLD_HIT;
    int    n_cold_tokens     = 0;
    int    n_hot_tokens      = 0;
    double dt_cold_ms        = 0;
    double dt_hot_prefill_ms = 0;
    double dt_signal_ms      = 0;
    double dt_decode_ms      = 0;
    double dt_total_ms       = 0;
};

static llama_token argmax_token(const float * logits, int n_vocab) {
    int best = 0;
    float bv = logits[0];
    for (int v = 1; v < n_vocab; v++) {
        if (logits[v] > bv) { bv = logits[v]; best = v; }
    }
    return (llama_token) best;
}

// New-protocol entry point: client supplies cold + hot raw bytes.
// head_mask: bit i = include head i (0=content, 1=function, 2-7=arg1..arg6).
static bool run_simpletool_v2(FusedState & s,
                              const std::string & cold_str,
                              const std::string & hot_str,
                              uint8_t head_mask,
                              int max_gen_tokens,
                              SimpleToolOutput & out,
                              std::string & err) {
    std::lock_guard<std::mutex> lk(s.mtx_simpletool);

    const int n_vocab = llama_vocab_n_tokens(s.vocab);

    std::vector<int> active_head_indices;
    for (int h = 0; h < MAX_HEADS; h++) {
        if (head_mask & (1u << h)) active_head_indices.push_back(h);
    }
    const int n_heads = (int) active_head_indices.size();
    if (max_gen_tokens <= 0 || max_gen_tokens > 255) max_gen_tokens = 64;

    double t_total = now_ms();

    auto cold_tokens = cold_str.empty()
        ? std::vector<llama_token>{}
        : tokenize_str(s.vocab, cold_str, false, true);
    auto hot_tokens = hot_str.empty()
        ? std::vector<llama_token>{}
        : tokenize_str(s.vocab, hot_str, false, true);

    const int n_cold  = (int) cold_tokens.size();
    const int n_hot   = (int) hot_tokens.size();
    const int n_total = n_cold + n_hot;

    if (n_cold > s.max_cold_tokens) {
        err = "cold too long (" + std::to_string(n_cold)
              + " > max_cold_tokens=" + std::to_string(s.max_cold_tokens) + ")";
        return false;
    }
    if (n_total + max_gen_tokens >= s.n_ctx) {
        err = "n_cold+n_hot+max_gen exceeds n_ctx ("
              + std::to_string(n_cold) + "+" + std::to_string(n_hot)
              + "+" + std::to_string(max_gen_tokens) + " >= " + std::to_string(s.n_ctx) + ")";
        return false;
    }

    if (!ensure_cold_cache(s, cold_tokens, out.cold_action, out.dt_cold_ms, err)) {
        return false;
    }
    out.n_cold_tokens = n_cold;
    out.n_hot_tokens  = n_hot;

    llama_memory_t mem = llama_get_memory(s.ctx_simpletool);

    // Drop any leftover hot from previous request, on every seq.
    for (int h = 0; h < MAX_HEADS; h++) {
        llama_memory_seq_rm(mem, h, n_cold, -1);
    }

    if (n_heads == 0) {
        // No decoding requested — caller just wanted to warm/prime cold cache.
        out.dt_total_ms = now_ms() - t_total;
        if (s.debug_cache) {
            fprintf(stderr, "[cache] decode skipped (head_mask=0, cache primed)\n");
        }
        return true;
    }

    // --- Hot prefill ---
    double t_hot = now_ms();
    if (n_hot > 0) {
        llama_batch batch = llama_batch_init(n_hot, 0, n_heads);
        batch.n_tokens = 0;
        for (int i = 0; i < n_hot; i++) {
            int idx = batch.n_tokens;
            batch.token[idx]     = hot_tokens[i];
            batch.pos[idx]       = n_cold + i;
            batch.n_seq_id[idx]  = n_heads;
            for (int k = 0; k < n_heads; k++) {
                batch.seq_id[idx][k] = active_head_indices[k];
            }
            batch.logits[idx]    = 0;
            batch.n_tokens++;
        }
        int rc = llama_decode(s.ctx_simpletool, batch);
        llama_synchronize(s.ctx_simpletool);
        llama_batch_free(batch);
        if (rc != 0) {
            err = "hot prefill rc=" + std::to_string(rc);
            return false;
        }
    }
    out.dt_hot_prefill_ms = now_ms() - t_hot;

    // --- Signal step ---
    double t_sig = now_ms();
    {
        llama_batch batch = llama_batch_init(n_heads, 0, 1);
        batch.n_tokens = 0;
        for (int hi : active_head_indices) {
            int idx = batch.n_tokens;
            batch.token[idx]     = ALL_HEADS[hi].open_id;
            batch.pos[idx]       = n_total;
            batch.n_seq_id[idx]  = 1;
            batch.seq_id[idx][0] = hi;
            batch.logits[idx]    = 1;
            batch.n_tokens++;
        }
        int rc = llama_decode(s.ctx_simpletool, batch);
        llama_synchronize(s.ctx_simpletool);
        llama_batch_free(batch);
        if (rc != 0) {
            err = "signal decode rc=" + std::to_string(rc);
            return false;
        }
    }
    out.dt_signal_ms = now_ms() - t_sig;

    // --- Per-head state ---
    std::vector<HeadResult> results(n_heads);
    std::vector<bool>       active(n_heads, true);
    int n_active = n_heads;

    for (int i = 0; i < n_heads; i++) {
        int hi = active_head_indices[i];
        results[i].head_idx = hi;
        results[i].name     = ALL_HEADS[hi].name;
    }

    // Sample step 0 (focal)
    for (int i = 0; i < n_heads; i++) {
        int hi = active_head_indices[i];
        float * logits = llama_get_logits_ith(s.ctx_simpletool, i);
        llama_token best = argmax_token(logits, n_vocab);
        results[i].num_tokens = 1;
        if (best == ALL_HEADS[hi].close_id) {
            results[i].hit_close = true; active[i] = false; n_active--;
        } else if (best == s.null_id) {
            results[i].is_null = true; active[i] = false; n_active--;
        } else if (best == s.im_end_id || llama_vocab_is_eog(s.vocab, best)) {
            active[i] = false; n_active--;
        } else {
            results[i].gen_ids.push_back(best);
        }
    }

    // --- Post-focal decode loop ---
    double t_dec = now_ms();
    for (int step = 1; step < max_gen_tokens && n_active > 0; step++) {
        llama_batch batch = llama_batch_init(n_active, 0, 1);
        batch.n_tokens = 0;
        std::vector<int> batch_to_local;
        for (int i = 0; i < n_heads; i++) {
            if (!active[i]) continue;
            int hi  = active_head_indices[i];
            int idx = batch.n_tokens;
            batch.token[idx]     = results[i].gen_ids.back();
            batch.pos[idx]       = n_total + step;
            batch.n_seq_id[idx]  = 1;
            batch.seq_id[idx][0] = hi;
            batch.logits[idx]    = 1;
            batch.n_tokens++;
            batch_to_local.push_back(i);
        }
        int rc = llama_decode(s.ctx_simpletool, batch);
        llama_synchronize(s.ctx_simpletool);
        if (rc != 0) {
            llama_batch_free(batch);
            err = "post-focal rc=" + std::to_string(rc);
            return false;
        }
        for (int b = 0; b < (int) batch_to_local.size(); b++) {
            int i  = batch_to_local[b];
            int hi = active_head_indices[i];
            float * logits = llama_get_logits_ith(s.ctx_simpletool, b);
            llama_token best = argmax_token(logits, n_vocab);
            results[i].num_tokens++;
            if (best == ALL_HEADS[hi].close_id) {
                results[i].hit_close = true; active[i] = false; n_active--;
            } else if (best == s.null_id) {
                results[i].is_null = true; active[i] = false; n_active--;
            } else if (best == s.im_end_id || llama_vocab_is_eog(s.vocab, best)) {
                active[i] = false; n_active--;
            } else {
                results[i].gen_ids.push_back(best);
            }
        }
        llama_batch_free(batch);
    }
    out.dt_decode_ms = now_ms() - t_dec;

    for (int i = 0; i < n_heads; i++) {
        results[i].decoded = detokenize_ids(s.vocab, results[i].gen_ids);
    }
    out.heads = std::move(results);
    out.dt_total_ms = now_ms() - t_total;

    return true;
}

// ============================================================================
// SIMPLETOOL-PORT server thread (NEW PROTOCOL)
// ============================================================================

static bool read_lenstr32(socket_t fd, std::string & out, uint32_t max_len) {
    uint32_t n_net;
    if (!read_all(fd, &n_net, 4)) return false;
    uint32_t n = ntohl(n_net);
    if (n > max_len) return false;
    out.resize(n);
    if (n > 0 && !read_all(fd, out.data(), n)) return false;
    return true;
}

static bool write_u8(socket_t fd, uint8_t v)  { return write_all(fd, &v, 1); }
static bool write_u16be(socket_t fd, uint16_t v) { uint16_t n = htons(v); return write_all(fd, &n, 2); }
static bool write_u32be(socket_t fd, uint32_t v) { uint32_t n = htonl(v); return write_all(fd, &n, 4); }
static bool write_i32be(socket_t fd, int32_t v)  { int32_t  n = (int32_t) htonl((uint32_t) v); return write_all(fd, &n, 4); }
static bool write_f64le(socket_t fd, double v)   { return write_all(fd, &v, 8); }

static bool write_lenstr16(socket_t fd, const std::string & s) {
    if (s.size() > 0xFFFF) return false;
    if (!write_u16be(fd, (uint16_t) s.size())) return false;
    if (!s.empty() && !write_all(fd, s.data(), s.size())) return false;
    return true;
}

static bool send_error(socket_t fd, int32_t status, const std::string & msg) {
    if (!write_i32be(fd, status)) return false;
    std::string m = msg.size() > 1024 ? msg.substr(0, 1024) : msg;
    return write_lenstr16(fd, m);
}

static bool handle_simpletool_request(socket_t fd, FusedState & s) {
    std::string cold_str, hot_str;
    if (!read_lenstr32(fd, cold_str, 1u << 20)) return false;   // up to 1 MiB cold
    if (!read_lenstr32(fd, hot_str,  1u << 18)) return false;   // up to 256 KiB hot

    uint8_t head_mask = 0;
    uint8_t max_gen   = 0;
    if (!read_all(fd, &head_mask, 1)) return false;
    if (!read_all(fd, &max_gen,   1)) return false;

    SimpleToolOutput out;
    std::string err;
    bool ok = run_simpletool_v2(s, cold_str, hot_str, head_mask, (int) max_gen, out, err);

    if (!ok) {
        fprintf(stderr, "[simpletool] error: %s\n", err.c_str());
        return send_error(fd, -2, err);
    }

    if (!write_i32be(fd, 0)) return false;
    if (!write_u8(fd, (uint8_t) out.heads.size())) return false;
    for (const auto & h : out.heads) {
        if (!write_u8(fd, (uint8_t) h.head_idx)) return false;
        if (!write_u8(fd, h.is_null ? 1 : 0))    return false;
        if (!write_lenstr16(fd, h.decoded))      return false;
    }
    if (!write_u32be(fd, (uint32_t) out.n_cold_tokens)) return false;
    if (!write_u32be(fd, (uint32_t) out.n_hot_tokens))  return false;
    if (!write_u8(fd, (uint8_t) out.cold_action))       return false;
    if (!write_f64le(fd, out.dt_cold_ms))         return false;
    if (!write_f64le(fd, out.dt_hot_prefill_ms))  return false;
    if (!write_f64le(fd, out.dt_signal_ms))       return false;
    if (!write_f64le(fd, out.dt_decode_ms))       return false;
    if (!write_f64le(fd, out.dt_total_ms))        return false;

    const char * action_str = (out.cold_action == COLD_HIT)    ? "HIT"
                            : (out.cold_action == COLD_EXTEND) ? "EXT"
                            : "RST";
    printf("[simpletool] cold=%d(%s) hot=%d heads=%zu mask=0x%02x  "
           "e2e=%.1fms (cold=%.1f hot=%.1f sig=%.1f dec=%.1f)\n",
           out.n_cold_tokens, action_str, out.n_hot_tokens,
           out.heads.size(), head_mask, out.dt_total_ms,
           out.dt_cold_ms, out.dt_hot_prefill_ms,
           out.dt_signal_ms, out.dt_decode_ms);
    fflush(stdout);
    return true;
}

static void simpletool_server_thread(FusedState * s, int port, std::atomic<bool> * stop) {
    socket_t srv = socket(AF_INET, SOCK_STREAM, 0);
    if (srv == INVALID_SOCKET_VALUE) { perror("[simpletool] socket"); return; }
    int opt = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, (const char *) &opt, sizeof(opt));
    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port        = htons((uint16_t) port);
    if (bind(srv, (sockaddr*) &addr, sizeof(addr)) < 0) { perror("[simpletool] bind"); return; }
    if (listen(srv, 8) < 0) { perror("[simpletool] listen"); return; }
    g_srv_simpletool.store(srv);
    printf("[simpletool] listening on 127.0.0.1:%d (V2 protocol)\n", port);
    fflush(stdout);

    while (!stop->load()) {
        sockaddr_in caddr{};
        socklen_t_compat clen = sizeof(caddr);
        socket_t cfd = accept(srv, (sockaddr*) &caddr, &clen);
        if (cfd == INVALID_SOCKET_VALUE) {
            if (stop->load()) break;
            perror("[simpletool] accept");
            continue;
        }
        int one = 1;
        setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, (const char *) &one, sizeof(one));
        while (handle_simpletool_request(cfd, *s)) { /* keep connection */ }
        closesocket_compat(cfd);
    }
    socket_t owned = g_srv_simpletool.exchange(INVALID_SOCKET_VALUE);
    if (owned != INVALID_SOCKET_VALUE) closesocket_compat(owned);
}

// ============================================================================
// main
// ============================================================================

static void print_usage(const char * prog) {
    fprintf(stderr,
        "Usage: %s [<model.gguf>] [options]\n"
        "  --config PATH            load TOML config (paths/ports/etc.)\n"
        "  --port-hidden P          TCP port for hidden requests\n"
        "  --port-simpletool P      TCP port for simpletool requests\n"
        "  --layer N                hidden-state layer (0-indexed)\n"
        "  --n-ctx N                context length for both contexts\n"
        "  --n_gpu_layers N         GPU offload layers\n"
        "  --max-cold-tokens N      cold cache token cap (-1 = n_ctx-256)\n"
        "  --debug-cache            log every cold cache decision\n"
        "  --no-hidden              disable hidden port\n"
        "  --no-simpletool          disable simpletool port\n"
        "\n"
        "Model path: either positional arg, or [paths].llama_model in --config.\n",
        prog);
}

int main(int argc, char ** argv) {
    WinsockInit _winsock_guard;   // RAII: WSAStartup on Win, no-op on POSIX
    disable_sigpipe();            // POSIX no-op on Win

    // ---- Install signal handlers (改动 3) --------------------------------
    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);

    // ---- Pre-scan for --config so it can supply defaults (改动 4) --------
    std::string config_path;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--config" && i + 1 < argc) {
            config_path = argv[++i];
            break;
        }
    }
    AppConfig cfg = config_path.empty()
        ? AppConfig::defaults()
        : AppConfig::from_toml(config_path);

    // Defaults from cfg (no more hard-coded literals here)
    const char * gguf_path     = nullptr;
    int          port_hidden     = cfg.fused.port_hidden;
    int          port_simpletool = cfg.fused.port_simpletool;
    int          layer_idx       = cfg.fused.hidden_layer;
    int          n_ctx           = cfg.fused.n_ctx;
    int          n_gpu_layers    = cfg.fused.n_gpu_layers;
    int          max_cold_tokens = cfg.fused.max_cold_tokens;
    bool         run_hidden      = true;
    bool         run_simpletool  = true;
    bool         debug_cache     = cfg.fused.debug_cache;

    bool has_model_arg = false;

    // ---- Main argv pass (改动 2 + 4) -------------------------------------
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--config") {
            ++i;  // value already consumed by pre-scan
            continue;
        }
        else if (a == "--port-hidden"     && i+1 < argc) port_hidden     = parse_int_arg("--port-hidden",     argv[++i], 1, 65535);
        else if (a == "--port-simpletool" && i+1 < argc) port_simpletool = parse_int_arg("--port-simpletool", argv[++i], 1, 65535);
        else if (a == "--layer"           && i+1 < argc) layer_idx       = parse_int_arg("--layer",           argv[++i], 0, 1024);
        else if (a == "--n-ctx"           && i+1 < argc) n_ctx           = parse_int_arg("--n-ctx",           argv[++i], 128, 1 << 20);
        else if (a == "--n_gpu_layers"    && i+1 < argc) n_gpu_layers    = parse_int_arg("--n_gpu_layers",    argv[++i], 0, 9999);
        else if (a == "--max-cold-tokens" && i+1 < argc) max_cold_tokens = parse_int_arg("--max-cold-tokens", argv[++i], -1, 1 << 20);
        else if (a == "--debug-cache")                   debug_cache     = true;
        else if (a == "--no-hidden")                     run_hidden      = false;
        else if (a == "--no-simpletool")                 run_simpletool  = false;
        else if (!a.empty() && a[0] != '-' && !has_model_arg) {
            gguf_path     = argv[i];
            has_model_arg = true;
        }
        else {
            fprintf(stderr, "Unknown arg: %s\n", a.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    // Fall back to config-provided model path if no positional was given.
    if (!gguf_path) {
        if (cfg.paths.llama_model.empty()) {
            fprintf(stderr, "no model path given (positional arg or [paths].llama_model in --config)\n");
            print_usage(argv[0]);
            return 1;
        }
        gguf_path = cfg.paths.llama_model.c_str();  // cfg lives until main returns
    }

    if (max_cold_tokens < 0) max_cold_tokens = n_ctx - 256;

    llama_backend_init();

    printf("[main] loading model %s (n_gpu_layers=%d)...\n", gguf_path, n_gpu_layers);
    double tL = now_ms();
    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;
    llama_model * model = llama_model_load_from_file(gguf_path, mparams);
    if (!model) { fprintf(stderr, "[main] model load failed\n"); return 1; }
    printf("[main] model loaded in %.1f ms\n", now_ms() - tL);

    FusedState state;
    state.model           = model;
    state.vocab           = llama_model_get_vocab(model);
    state.n_embd          = llama_model_n_embd(model);
    state.n_ctx           = n_ctx;
    state.max_cold_tokens = max_cold_tokens;
    state.debug_cache     = debug_cache;

    if (run_hidden || run_simpletool) {
        auto cp = llama_context_default_params();
        cp.n_ctx               = (uint32_t) n_ctx;
        cp.n_batch             = (uint32_t) n_ctx;
        cp.hidden_layer_output = layer_idx;
        state.ctx_hidden = llama_init_from_model(model, cp);
        if (!state.ctx_hidden) { fprintf(stderr, "[main] ctx_hidden init failed\n"); return 1; }
        printf("[main] ctx_hidden ready: layer=%d n_embd=%d n_ctx=%d\n",
               layer_idx, state.n_embd, n_ctx);
    }

    if (run_simpletool) {
        const uint32_t total_ctx = (uint32_t) n_ctx * 2u;
        auto cp = llama_context_default_params();
        cp.n_ctx               = total_ctx;
        cp.n_batch             = (uint32_t) n_ctx;
        cp.n_ubatch            = (uint32_t) n_ctx;
        cp.n_seq_max           = (uint32_t) MAX_HEADS;
        cp.kv_unified          = true;
        cp.hidden_layer_output = -1;
        state.ctx_simpletool = llama_init_from_model(model, cp);
        if (!state.ctx_simpletool) { fprintf(stderr, "[main] ctx_simpletool init failed\n"); return 1; }
        printf("[main] ctx_simpletool ready: n_seq_max=%d n_ctx=%u kv_unified=1 max_cold=%d\n",
               MAX_HEADS, total_ctx, max_cold_tokens);
    }

    if (run_simpletool) {
        state.null_id   = find_single_token(state.vocab, "<|null|>");
        state.im_end_id = find_single_token(state.vocab, "<|im_end|>");
        if (state.null_id < 0 || state.im_end_id < 0) {
            fprintf(stderr, "[main] missing <|null|> or <|im_end|> in vocab\n");
            return 1;
        }
        for (int h = 0; h < MAX_HEADS; h++) {
            ALL_HEADS[h].open_id  = find_single_token(state.vocab, ALL_HEADS[h].open_tag);
            ALL_HEADS[h].close_id = find_single_token(state.vocab, ALL_HEADS[h].close_tag);
            if (ALL_HEADS[h].open_id < 0 || ALL_HEADS[h].close_id < 0) {
                fprintf(stderr, "[main] missing tag tokens for head %s\n", ALL_HEADS[h].name);
                return 1;
            }
        }
        printf("[main] special tokens OK (null=%d, im_end=%d, head 0..%d resolved)\n",
               state.null_id, state.im_end_id, MAX_HEADS - 1);
    }

    // Warmup ctx_hidden (CUDA graph build). No cold prefill on ctx_simpletool —
    // that now happens lazily on first request.
    if (run_hidden || run_simpletool) {
        printf("[main] warming up hidden context...\n");
        std::vector<float> dummy; int nt = 0, ne = 0; double ms = 0; std::string err;
        if (extract_hidden(state, "warmup.", dummy, nt, ne, ms, err)) {
            printf("[main] warmup OK: %d tok, %.2f ms\n", nt, ms);
        } else {
            fprintf(stderr, "[main] warmup failed: %s\n", err.c_str());
        }
    }

    std::thread th_hidden, th_simpletool;

    if (run_hidden) {
        th_hidden = std::thread(hidden_server_thread, &state, port_hidden, &g_stop);
    }
    if (run_simpletool) {
        th_simpletool = std::thread(simpletool_server_thread, &state, port_simpletool, &g_stop);
    }

    printf("[main] fused_server ready (V2 protocol on simpletool port).\n");
    fflush(stdout);

    if (th_hidden.joinable())     th_hidden.join();
    if (th_simpletool.joinable()) th_simpletool.join();

    printf("[main] shutting down...\n");

    if (state.ctx_simpletool) llama_free(state.ctx_simpletool);
    if (state.ctx_hidden)     llama_free(state.ctx_hidden);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}