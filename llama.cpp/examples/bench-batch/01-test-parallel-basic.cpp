// ============================================================
// 01-test-parallel-basic.cpp
// SimpleTool Voice Commander — 并行多头解码基础验证
// ============================================================
//
// 功能:
//   验证 SimpleTool fine-tuned 模型在游戏 function calling 场景下
//   的并行多头解码, 含详细性能分析。
//
// 核心特性:
//   1. Cold Start 缓存: system prompt + tools 只 prefill 一次
//   2. 动态 Head Count: --n-args 裁剪活跃 head 数
//   3. 动态 Head 退出: stop token 触发即时终止
//   4. 分段计时: cold/hot prefill TPS, 逐步 TPOT, e2e
//   5. 内存报告: 模型权重 + KV cache 估算
//   6. 汇总统计: 所有 case 的性能指标聚合
//
// 命令行:
//   ./01-test-parallel-basic <model.gguf> [options]
//
//   --n-ctx N        KV cache 上下文长度 (默认: 2048)
//   --content        启用 content head (默认: 关闭)
//   --n-args N       参数 head 数量, 0-6 (默认: 6)
//   --no-draft       跳过 DRAFT 模式测试
//   --no-seq         跳过 SEQUENTIAL 基线测试
//
// 编译:
//   c++ -std=c++17 -O2 \
//     -I include -I ggml/include \
//     examples/bench-batch/01-test-parallel-basic.cpp \
//     -L build/bin -lllama \
//     -framework Metal -framework Foundation -framework Accelerate \
//     -Wl,-rpath,build/bin \
//     -o build/bin/01-test-parallel-basic
//
// 示例:
//   ./build/bin/01-test-parallel-basic rt-qwen3-4b-q8_0.gguf --n-args 2 --n-ctx 1024
//   ./build/bin/01-test-parallel-basic rt-qwen3-4b-q8_0.gguf --content --no-draft
//
// ============================================================

#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <cmath>

// ============================================================
// 计时
// ============================================================

using Clock = std::chrono::high_resolution_clock;

static double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// ============================================================
// Tokenizer 工具函数
// ============================================================

static llama_token find_single_token(const llama_vocab * vocab, const char * text) {
    llama_token tokens[8];
    int n = llama_tokenize(vocab, text, (int)strlen(text), tokens, 8, false, true);
    if (n == 1) return tokens[0];
    return -1;
}

static std::string token_to_str(const llama_vocab * vocab, llama_token id) {
    char buf[256];
    int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
    if (n > 0) return std::string(buf, n);
    return "";
}

static std::string detokenize(const llama_vocab * vocab, const std::vector<llama_token> & tokens) {
    std::string result;
    for (auto t : tokens) {
        result += token_to_str(vocab, t);
    }
    return result;
}

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text,
                                          bool add_special, bool parse_special) {
    int n_max = (int)text.size() + 256;
    std::vector<llama_token> tokens(n_max);
    int n = llama_tokenize(vocab, text.c_str(), (int)text.size(),
                           tokens.data(), n_max, add_special, parse_special);
    if (n < 0) {
        tokens.resize(-n);
        n = llama_tokenize(vocab, text.c_str(), (int)text.size(),
                           tokens.data(), -n, add_special, parse_special);
    }
    tokens.resize(n);
    return tokens;
}

// ============================================================
// Head 配置
// ============================================================

struct HeadConfig {
    const char * name;
    const char * open_tag;
    const char * close_tag;
    llama_token  open_id;
    llama_token  close_id;
};

static const int MAX_HEADS = 8;

static HeadConfig ALL_HEADS[MAX_HEADS] = {
    {"content",  "<content>",  "</content>",  -1, -1},
    {"function", "<function>", "</function>", -1, -1},
    {"arg1",     "<arg1>",     "</arg1>",     -1, -1},
    {"arg2",     "<arg2>",     "</arg2>",     -1, -1},
    {"arg3",     "<arg3>",     "</arg3>",     -1, -1},
    {"arg4",     "<arg4>",     "</arg4>",     -1, -1},
    {"arg5",     "<arg5>",     "</arg5>",     -1, -1},
    {"arg6",     "<arg6>",     "</arg6>",     -1, -1},
};

// ============================================================
// 游戏 Tool Schema
// ============================================================

static const char * TOOLS_JSON =
    R"json({"type":"function","function":{"name":"move","description":"Move an operator to the specified position on the battlefield.","parameters":{"type":"object","properties":{"unit_id":{"type":"string","description":"Operator name or callsign"},"target":{"type":"string","description":"Target position (e.g. north, south, east, west, center)"}},"required":["unit_id","target"]}}})json"
    "\n"
    R"json({"type":"function","function":{"name":"use_skill","description":"Activate an operator's skill.","parameters":{"type":"object","properties":{"unit_id":{"type":"string","description":"Operator name or callsign"},"skill_id":{"type":"string","description":"Skill name or identifier"}},"required":["unit_id"]}}})json"
    "\n"
    R"json({"type":"function","function":{"name":"set_stance","description":"Set an operator's behavior mode.","parameters":{"type":"object","properties":{"unit_id":{"type":"string","description":"Operator name or callsign"},"stance":{"type":"string","description":"Behavior mode: aggressive, defensive, hold"}},"required":["unit_id","stance"]}}})json"
    "\n"
    R"json({"type":"function","function":{"name":"retreat","description":"Retreat a specific operator from the battlefield.","parameters":{"type":"object","properties":{"unit_id":{"type":"string","description":"Operator name or callsign"}},"required":["unit_id"]}}})json"
    "\n"
    R"json({"type":"function","function":{"name":"retreat_all","description":"Retreat all operators from the battlefield immediately.","parameters":{"type":"object","properties":{}}}})json"
    "\n"
    R"json({"type":"function","function":{"name":"pass","description":"No action needed. All requested tasks are complete.","parameters":{"type":"object","properties":{}}}})json";

// ============================================================
// System Prompt
// ============================================================

// V1 template (matches the training data exactly, including trailing whitespace).
// Tools are appended under "## Available Tools:" in build_cold_prompt().
static const char * SYS_PROMPT =
    "You are a multi-head parallel function calling model. \n"
    "## Output Heads\n"
    "\n"
    "**Head 0 - <content>**: Natural language response\n"
    "- Format: <content>response text</content>\n"
    "- Answer what you want to say while you are calling a function\n"
    "\n"
    "**Head 1 - <function>**: Function names to call\n"
    "- Format: <function>name</function>\n"
    "- Name: must match tool defined name\n"
    "\n"
    "**Head 2-7 - <arg1>、<arg2>、<arg3>、<arg4>、<arg5>、<arg6>**: Function arguments by position\n"
    "- Format: <argN>value</argN> \n"
    "- Strictly fill in according to the parameter order of the tool you intend to call\n"
    "- Note the special restrictions of parameter definitions for corresponding positions\n"
    "- If the corresponding tool definition has required parameters, these must be filled in\n"
    "- Infer the user's actual needs.\n"
    "- If Unnecessary: <argN><|null|></argN>\n"
    "\n"
    "**Environment - The information you have.\n"
    "**History - The tools you have called.\n";

// ============================================================
// 测试用例
// ============================================================

struct TestCase {
    const char * environment;    // free-form text, or "[]" for empty
    const char * history;        // JSON array-like string, "[]" for empty
    const char * query;          // user voice command (pure instruction)
    const char * expected_func;  // expected value of <function> head
    const char * desc;
};

// Test cases for v1 prompt template.
// `environment` carries battlefield state; `query` carries only the voice command.
static TestCase CASES[] = {
    {
        "当前第3波 剩余敌人12个; 干员: 小火(北方,HP85%) 阿米娅(中央,HP100%) 星熊(南方,HP60%); 敌人方向: 北方",
        "[]",
        "小火往北边走",
        "move",
        "Case 1: move(小火, north)"
    },
    {
        "当前第5波 剩余敌人8个 BOSS已出现; 干员: 小火(北方,HP50%,技能就绪) 阿米娅(中央,HP90%,技能就绪); 敌人方向: 北方集中",
        "[]",
        "阿米娅放技能快点",
        "use_skill",
        "Case 2: use_skill(阿米娅)"
    },
    {
        "当前第7波 剩余敌人20个; 干员: 小火(北方,HP15%,重伤) 阿米娅(中央,HP80%) 德克萨斯(东方,HP25%); 敌人方向: 全方位",
        "[]",
        "德克萨斯赶紧撤退撤退",
        "retreat",
        "Case 3: retreat(德克萨斯)"
    },
    {
        "当前第10波 剩余敌人30个 防线即将崩溃; 干员: 小火(北方,HP10%) 阿米娅(中央,HP20%) 星熊(南方,HP5%); 敌人方向: 全方位突破",
        "[]",
        "全员撤退全员撤退快快快",
        "retreat_all",
        "Case 4: retreat_all() [zero-arg]"
    },
    {
        "当前第4波 剩余敌人15个; 干员: 星熊(南方,HP90%,aggressive模式); 敌人方向: 南方缓慢推进",
        "[]",
        "星熊切换防守模式",
        "set_stance",
        "Case 5: set_stance(星熊, defensive)"
    },
    {
        "当前第3波 剩余敌人12个; 干员: 小火(北方,HP85%) 阿米娅(中央,HP100%); 敌人方向: 北方",
        R"json(["move(unit_id='Blaze', target='north')"])json",
        "小火往北边走",
        "pass",
        "Case 6: history done -> pass"
    },
};

// ============================================================
// Decode 逐步记录
// ============================================================

struct StepInfo {
    int    step;          // decode step 编号 (0=signal, 1..N=post-focal)
    int    batch_size;    // 当前 batch 中活跃 head 数
    int    kv_pos;        // 当前 KV cache 使用到的 position
    double dt_ms;         // 本步耗时 (ms)
    // TPOT = dt_ms (wall-clock, 因为每个 head 出 1 token)
    // amortized TPOT = dt_ms / batch_size
};

// ============================================================
// 单次推理的完整性能数据
// ============================================================

struct HeadResult {
    std::string name;
    int         head_idx;
    std::vector<llama_token> gen_ids;
    std::string decoded;
    bool        is_null;
    bool        hit_close;
    int         num_tokens;
};

struct CasePerf {
    const char * desc;
    bool   passed;
    int    n_cold_tokens;
    int    n_hot_tokens;
    int    n_active_heads;
    double dt_hot_prefill_ms;
    double dt_signal_ms;       // focal token (step 0)
    double dt_decode_ms;       // post-focal 总计
    double dt_e2e_ms;
    std::vector<StepInfo> steps;
    std::vector<HeadResult> results;
};

// ============================================================
// Prompt 构建
// ============================================================

static std::string build_cold_prompt() {
    std::string p;
    p += "<|im_start|>system\n";
    p += SYS_PROMPT;                          // SYS_PROMPT already ends with '\n'
    p += "\n## Available Tools:\n\n";         // blank line between sys text and tools
    p += TOOLS_JSON;
    p += "<|im_end|>\n";
    return p;
}

// V1 user-turn layout (matches training sample byte-for-byte):
//   <|im_start|>user
//   environment: {env}
//   history: {history}
//
//   {query}<|im_end|>
//   <|im_start|>assistant
//
// NOTE: Both `env` and `history` are JSON-like string blobs passed through.
// Use "[]" for empty values (matches training data).
static std::string build_hot_prompt(const char * environment,
                                    const char * history,
                                    const char * query) {
    std::string p;
    p += "<|im_start|>user\n";
    p += "environment: ";
    p += environment;
    p += "\n";
    p += "history: ";
    p += history;
    p += "\n\n";
    p += query;
    p += "<|im_end|>\n";
    p += "<|im_start|>assistant\n";
    return p;
}

// ============================================================
// 结果打印 (单 case)
// ============================================================

static void print_case_results(const char * mode_tag, const CasePerf & perf) {
    // Function head result
    std::string func_decoded;
    for (const auto & r : perf.results) {
        if (r.head_idx == 1) { func_decoded = r.decoded; break; }
    }

    printf("──────────────────────────────────────────────────────\n");
    printf("%s  %s %s\n", perf.passed ? "PASS" : "FAIL", mode_tag, perf.desc);

    // Head 输出表
    printf("  %-10s %-30s %-5s %-6s %s\n", "Head", "Decoded", "Tok", "Close", "Status");
    printf("  -------------------------------------------------------------------\n");
    for (const auto & r : perf.results) {
        std::string disp = r.decoded;
        if (disp.size() > 28) disp = disp.substr(0, 28) + "...";
        if (r.is_null) disp = "<|null|>";
        const char * st = (r.head_idx == 1) ? (perf.passed ? "OK" : "WRONG")
                        : r.is_null ? "NULL" : "FILL";
        printf("  %-10s %-30s %-5d %-6s %s\n",
               r.name.c_str(), disp.c_str(), r.num_tokens,
               r.hit_close ? "Y" : (r.is_null ? "N" : "-"), st);
    }

    // 计时摘要
    double hot_tps = (perf.n_hot_tokens > 0 && perf.dt_hot_prefill_ms > 0)
                   ? perf.n_hot_tokens / (perf.dt_hot_prefill_ms / 1000.0) : 0;
    printf("\n  Timing:\n");
    printf("    hot_prefill : %6.1fms  (%d tok, %.0f tok/s)\n",
           perf.dt_hot_prefill_ms, perf.n_hot_tokens, hot_tps);
    printf("    signal      : %6.1fms  (focal, batch=%d)\n",
           perf.dt_signal_ms, perf.n_active_heads);
    printf("    decode      : %6.1fms  (post-focal total)\n", perf.dt_decode_ms);
    printf("    e2e         : %6.1fms\n", perf.dt_e2e_ms);

    // 逐步 TPOT 表
    if (!perf.steps.empty()) {
        printf("\n  Decode steps (each head produces 1 token per step):\n");
        printf("    %-6s %-8s %-8s %-10s %-12s %s\n",
               "Step", "Batch", "KV pos", "TPOT(ms)", "Amort(ms)", "Note");
        printf("    ---------------------------------------------------------------\n");
        for (const auto & s : perf.steps) {
            double amort = (s.batch_size > 0) ? s.dt_ms / s.batch_size : 0;
            const char * note = (s.step == 0) ? "signal/focal" : "";
            printf("    %-6d %-8d %-8d %-10.2f %-12.2f %s\n",
                   s.step, s.batch_size, s.kv_pos, s.dt_ms, amort, note);
        }
    }
    printf("\n");
}

// ============================================================
// 并行推理
// ============================================================

static CasePerf run_parallel_inference(
    llama_context * ctx,
    llama_memory_t mem,
    const llama_vocab * vocab,
    llama_token null_id,
    llama_token im_end_id,
    const std::vector<int> & active_head_indices,
    const std::vector<llama_token> & hot_tokens,
    int n_cold,
    const TestCase & tc,
    int n_vocab,
    bool use_draft)
{
    const int n_hot   = (int)hot_tokens.size();
    const int n_total = n_cold + n_hot;
    const int n_heads = (int)active_head_indices.size();
    const int max_gen = 64;

    CasePerf perf = {};
    perf.desc           = tc.desc;
    perf.n_cold_tokens  = n_cold;
    perf.n_hot_tokens   = n_hot;
    perf.n_active_heads = n_heads;

    // --- 清理上一轮 ---
    for (int h = 1; h < MAX_HEADS; h++) {
        llama_memory_seq_rm(mem, h, 0, -1);
    }
    llama_memory_seq_rm(mem, 0, n_cold, -1);

    auto t_e2e = Clock::now();

    // --- Hot prefill ---
    auto t_hot = Clock::now();
    {
        auto batch = llama_batch_init(n_hot, 0, 1);
        batch.n_tokens = 0;
        for (int i = 0; i < n_hot; i++) {
            int idx = batch.n_tokens;
            batch.token[idx]     = hot_tokens[i];
            batch.pos[idx]       = n_cold + i;
            batch.n_seq_id[idx]  = 1;
            batch.seq_id[idx][0] = 0;
            batch.logits[idx]    = 0;
            batch.n_tokens++;
        }
        llama_decode(ctx, batch);
        llama_synchronize(ctx);
        llama_batch_free(batch);
    }
    perf.dt_hot_prefill_ms = ms_since(t_hot);

    // --- seq_cp ---
    for (int hi : active_head_indices) {
        if (hi == 0) continue;
        llama_memory_seq_cp(mem, 0, hi, 0, n_total);
    }

    // --- Signal (step 0, focal token) ---
    auto t_sig = Clock::now();
    {
        auto batch = llama_batch_init(n_heads, 0, 1);
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
        llama_decode(ctx, batch);
        llama_synchronize(ctx);
        llama_batch_free(batch);
    }
    perf.dt_signal_ms = ms_since(t_sig);

    // 记录 step 0
    perf.steps.push_back({0, n_heads, n_total + 1, perf.dt_signal_ms});

    // --- Decode ---
    auto t_decode = Clock::now();

    std::vector<HeadResult> results(n_heads);
    std::vector<bool> active(n_heads, true);
    int n_active = n_heads;

    for (int i = 0; i < n_heads; i++) {
        int hi = active_head_indices[i];
        results[i].name      = ALL_HEADS[hi].name;
        results[i].head_idx  = hi;
        results[i].is_null   = false;
        results[i].hit_close = false;
        results[i].num_tokens = 0;
    }

    // Step 0 采样
    for (int i = 0; i < n_heads; i++) {
        int hi = active_head_indices[i];
        float * logits = llama_get_logits_ith(ctx, i);
        llama_token best = 0;
        float best_val = logits[0];
        for (int v = 1; v < n_vocab; v++) {
            if (logits[v] > best_val) { best_val = logits[v]; best = v; }
        }
        results[i].num_tokens = 1;
        if (best == ALL_HEADS[hi].close_id) {
            results[i].hit_close = true; active[i] = false; n_active--;
        } else if (best == null_id) {
            results[i].is_null = true; active[i] = false; n_active--;
        } else if (best == im_end_id || llama_vocab_is_eog(vocab, best)) {
            active[i] = false; n_active--;
        } else {
            results[i].gen_ids.push_back(best);
        }
    }

    // Post-focal steps
    if (use_draft) {
        // llama_set_draft_mode(ctx, true);  // GPU build: draft mode not available
        (void)ctx;
    }

    for (int step = 1; step < max_gen && n_active > 0; step++) {
        auto batch = llama_batch_init(n_active, 0, 1);
        batch.n_tokens = 0;
        std::vector<int> batch_to_local;

        for (int i = 0; i < n_heads; i++) {
            if (!active[i]) continue;
            int hi = active_head_indices[i];
            int idx = batch.n_tokens;
            batch.token[idx]     = results[i].gen_ids.back();
            batch.pos[idx]       = n_total + 1 + step - 1;
            batch.n_seq_id[idx]  = 1;
            batch.seq_id[idx][0] = hi;
            batch.logits[idx]    = 1;
            batch.n_tokens++;
            batch_to_local.push_back(i);
        }

        int cur_kv_pos = n_total + 1 + step;

        auto t_step = Clock::now();
        llama_decode(ctx, batch);
        llama_synchronize(ctx);
        double dt_step = ms_since(t_step);

        perf.steps.push_back({step, n_active, cur_kv_pos, dt_step});

        for (int b = 0; b < (int)batch_to_local.size(); b++) {
            int i  = batch_to_local[b];
            int hi = active_head_indices[i];
            float * logits = llama_get_logits_ith(ctx, b);
            llama_token best = 0;
            float best_val = logits[0];
            for (int v = 1; v < n_vocab; v++) {
                if (logits[v] > best_val) { best_val = logits[v]; best = v; }
            }
            results[i].num_tokens++;
            if (best == ALL_HEADS[hi].close_id) {
                results[i].hit_close = true; active[i] = false; n_active--;
            } else if (best == null_id) {
                results[i].is_null = true; active[i] = false; n_active--;
            } else if (best == im_end_id || llama_vocab_is_eog(vocab, best)) {
                active[i] = false; n_active--;
            } else {
                results[i].gen_ids.push_back(best);
            }
        }
        llama_batch_free(batch);
    }

    if (use_draft) {
        // llama_set_draft_mode(ctx, false);  // GPU build: draft mode not available
        (void)ctx;
    }

    perf.dt_decode_ms = ms_since(t_decode);
    perf.dt_e2e_ms    = ms_since(t_e2e);

    // Detokenize
    for (int i = 0; i < n_heads; i++) {
        results[i].decoded = detokenize(vocab, results[i].gen_ids);
    }

    // Check pass
    std::string func_decoded;
    for (const auto & r : results) {
        if (r.head_idx == 1) { func_decoded = r.decoded; break; }
    }
    perf.passed  = (func_decoded == tc.expected_func);
    perf.results = results;

    return perf;
}

// ============================================================
// 串行推理
// ============================================================

static CasePerf run_sequential_inference(
    llama_context * ctx,
    llama_memory_t mem,
    const llama_vocab * vocab,
    llama_token null_id,
    llama_token im_end_id,
    const std::vector<int> & active_head_indices,
    const std::vector<llama_token> & hot_tokens,
    int n_cold,
    const TestCase & tc,
    int n_vocab)
{
    const int n_hot   = (int)hot_tokens.size();
    const int n_total = n_cold + n_hot;
    const int n_heads = (int)active_head_indices.size();

    CasePerf perf = {};
    perf.desc           = tc.desc;
    perf.n_cold_tokens  = n_cold;
    perf.n_hot_tokens   = n_hot;
    perf.n_active_heads = n_heads;

    for (int h = 1; h < MAX_HEADS; h++) {
        llama_memory_seq_rm(mem, h, 0, -1);
    }
    llama_memory_seq_rm(mem, 0, n_cold, -1);

    auto t_e2e = Clock::now();

    // Hot prefill
    auto t_hot = Clock::now();
    {
        auto batch = llama_batch_init(n_hot, 0, 1);
        batch.n_tokens = 0;
        for (int i = 0; i < n_hot; i++) {
            int idx = batch.n_tokens;
            batch.token[idx]     = hot_tokens[i];
            batch.pos[idx]       = n_cold + i;
            batch.n_seq_id[idx]  = 1;
            batch.seq_id[idx][0] = 0;
            batch.logits[idx]    = 0;
            batch.n_tokens++;
        }
        llama_decode(ctx, batch);
        llama_synchronize(ctx);
        llama_batch_free(batch);
    }
    perf.dt_hot_prefill_ms = ms_since(t_hot);

    auto t_decode_start = Clock::now();
    int cur_pos = n_total;
    std::vector<HeadResult> results(n_heads);

    for (int i = 0; i < n_heads; i++) {
        int hi = active_head_indices[i];
        results[i].name      = ALL_HEADS[hi].name;
        results[i].head_idx  = hi;
        results[i].is_null   = false;
        results[i].hit_close = false;
        results[i].num_tokens = 0;

        {
            auto batch = llama_batch_init(1, 0, 1);
            batch.n_tokens     = 1;
            batch.token[0]     = ALL_HEADS[hi].open_id;
            batch.pos[0]       = cur_pos;
            batch.n_seq_id[0]  = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0]    = 1;
            llama_decode(ctx, batch);
            llama_synchronize(ctx);
            llama_batch_free(batch);
        }
        cur_pos++;

        for (int step = 0; step < 64; step++) {
            auto t_step = Clock::now();
            float * logits = llama_get_logits_ith(ctx, 0);
            llama_token best = 0;
            float best_val = logits[0];
            for (int v = 1; v < n_vocab; v++) {
                if (logits[v] > best_val) { best_val = logits[v]; best = v; }
            }
            results[i].num_tokens++;

            if (best == ALL_HEADS[hi].close_id) {
                results[i].hit_close = true;
                std::vector<llama_token> suffix = { ALL_HEADS[hi].close_id };
                if (im_end_id >= 0) suffix.push_back(im_end_id);
                auto batch = llama_batch_init((int)suffix.size(), 0, 1);
                batch.n_tokens = 0;
                for (auto t : suffix) {
                    int idx = batch.n_tokens;
                    batch.token[idx]     = t;
                    batch.pos[idx]       = cur_pos;
                    batch.n_seq_id[idx]  = 1;
                    batch.seq_id[idx][0] = 0;
                    batch.logits[idx]    = 0;
                    batch.n_tokens++;
                    cur_pos++;
                }
                llama_decode(ctx, batch);
                llama_synchronize(ctx);
                llama_batch_free(batch);
                double dt = ms_since(t_step);
                perf.steps.push_back({(int)perf.steps.size(), 1, cur_pos, dt});
                break;
            } else if (best == null_id) {
                results[i].is_null = true;
                std::vector<llama_token> suffix = { null_id };
                if (ALL_HEADS[hi].close_id >= 0) suffix.push_back(ALL_HEADS[hi].close_id);
                if (im_end_id >= 0) suffix.push_back(im_end_id);
                auto batch = llama_batch_init((int)suffix.size(), 0, 1);
                batch.n_tokens = 0;
                for (auto t : suffix) {
                    int idx = batch.n_tokens;
                    batch.token[idx]     = t;
                    batch.pos[idx]       = cur_pos;
                    batch.n_seq_id[idx]  = 1;
                    batch.seq_id[idx][0] = 0;
                    batch.logits[idx]    = 0;
                    batch.n_tokens++;
                    cur_pos++;
                }
                llama_decode(ctx, batch);
                llama_synchronize(ctx);
                llama_batch_free(batch);
                double dt = ms_since(t_step);
                perf.steps.push_back({(int)perf.steps.size(), 1, cur_pos, dt});
                break;
            } else if (best == im_end_id || llama_vocab_is_eog(vocab, best)) {
                break;
            } else {
                results[i].gen_ids.push_back(best);
                auto batch = llama_batch_init(1, 0, 1);
                batch.n_tokens     = 1;
                batch.token[0]     = best;
                batch.pos[0]       = cur_pos;
                batch.n_seq_id[0]  = 1;
                batch.seq_id[0][0] = 0;
                batch.logits[0]    = 1;
                llama_decode(ctx, batch);
                llama_synchronize(ctx);
                llama_batch_free(batch);
                double dt = ms_since(t_step);
                perf.steps.push_back({(int)perf.steps.size(), 1, cur_pos, dt});
                cur_pos++;
            }
        }
        results[i].decoded = detokenize(vocab, results[i].gen_ids);
    }

    perf.dt_decode_ms = ms_since(t_decode_start);
    perf.dt_e2e_ms    = ms_since(t_e2e);

    std::string func_decoded;
    for (const auto & r : results) {
        if (r.head_idx == 1) { func_decoded = r.decoded; break; }
    }
    perf.passed  = (func_decoded == tc.expected_func);
    perf.results = results;

    return perf;
}

// ============================================================
// 汇总统计
// ============================================================

static void print_summary(
    const char * mode_tag,
    const std::vector<CasePerf> & perfs,
    double dt_cold_ms,
    int n_cold)
{
    if (perfs.empty()) return;

    int n_pass = 0;
    double sum_hot = 0, sum_sig = 0, sum_dec = 0, sum_e2e = 0;
    int    sum_hot_tok = 0;
    int    total_decode_tokens = 0;
    double min_e2e = 1e9, max_e2e = 0;

    // 收集所有 step 0 (signal) 的 TPOT
    double sum_step0_tpot = 0;
    int    cnt_step0 = 0;

    for (const auto & p : perfs) {
        if (p.passed) n_pass++;
        sum_hot += p.dt_hot_prefill_ms;
        sum_sig += p.dt_signal_ms;
        sum_dec += p.dt_decode_ms;
        sum_e2e += p.dt_e2e_ms;
        sum_hot_tok += p.n_hot_tokens;
        if (p.dt_e2e_ms < min_e2e) min_e2e = p.dt_e2e_ms;
        if (p.dt_e2e_ms > max_e2e) max_e2e = p.dt_e2e_ms;

        for (const auto & r : p.results) {
            total_decode_tokens += r.num_tokens;
        }

        if (!p.steps.empty()) {
            sum_step0_tpot += p.steps[0].dt_ms;
            cnt_step0++;
        }
    }

    int n = (int)perfs.size();
    double cold_tps = (n_cold > 0 && dt_cold_ms > 0) ? n_cold / (dt_cold_ms / 1000.0) : 0;
    double avg_hot_tps = (sum_hot_tok > 0 && sum_hot > 0) ? sum_hot_tok / (sum_hot / 1000.0) : 0;

    printf("======================================================\n");
    printf("  Summary: %s  (%d/%d passed)\n", mode_tag, n_pass, n);
    printf("======================================================\n");
    printf("\n");
    printf("  Prefill:\n");
    printf("    cold_start   : %6.1fms  %d tok  %6.0f tok/s  (one-time)\n",
           dt_cold_ms, n_cold, cold_tps);
    printf("    hot_prefill  : %6.1fms avg  %6.0f tok/s avg\n",
           sum_hot / n, avg_hot_tps);
    printf("\n");
    printf("  Decode:\n");
    printf("    signal (focal): %6.1fms avg  (batch=%d, full model)\n",
           sum_sig / n, perfs[0].n_active_heads);
    if (cnt_step0 > 0) {
        printf("    step0 TPOT   : %6.1fms avg  (= signal, max concurrency)\n",
               sum_step0_tpot / cnt_step0);
    }
    printf("    decode total : %6.1fms avg\n", sum_dec / n);
    printf("    decode tokens: %d total across %d cases (%.1f tok/case avg)\n",
           total_decode_tokens, n, (double)total_decode_tokens / n);
    printf("\n");
    printf("  End-to-end (hot prefill + signal + decode):\n");
    printf("    avg  : %6.1fms\n", sum_e2e / n);
    printf("    min  : %6.1fms\n", min_e2e);
    printf("    max  : %6.1fms\n", max_e2e);

    // 逐步 TPOT 分布 (聚合所有 case)
    // 找到最大 step 数
    int max_step = 0;
    for (const auto & p : perfs) {
        for (const auto & s : p.steps) {
            if (s.step > max_step) max_step = s.step;
        }
    }

    if (max_step > 0) {
        printf("\n  TPOT by decode step (averaged across cases):\n");
        printf("    %-6s %-10s %-10s %-12s %s\n",
               "Step", "AvgBatch", "TPOT(ms)", "Amort(ms)", "Count");
        printf("    -------------------------------------------------------\n");
        for (int s = 0; s <= max_step; s++) {
            double sum_dt = 0;
            double sum_batch = 0;
            int cnt = 0;
            for (const auto & p : perfs) {
                for (const auto & si : p.steps) {
                    if (si.step == s) {
                        sum_dt += si.dt_ms;
                        sum_batch += si.batch_size;
                        cnt++;
                        break;
                    }
                }
            }
            if (cnt > 0) {
                double avg_dt = sum_dt / cnt;
                double avg_batch = sum_batch / cnt;
                double avg_amort = (avg_batch > 0) ? avg_dt / avg_batch : 0;
                printf("    %-6d %-10.1f %-10.2f %-12.2f %d\n",
                       s, avg_batch, avg_dt, avg_amort, cnt);
            }
        }
    }
    printf("\n");
}

// ============================================================
// Main
// ============================================================

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr,
            "Usage: %s <model.gguf> [options]\n"
            "\n"
            "Options:\n"
            "  --n-ctx N        Context length (default: 2048)\n"
            "  --content        Enable content head (default: OFF)\n"
            "  --n-args N       Arg heads count, 0-6 (default: 6)\n"
            "  --no-draft       Skip DRAFT mode\n"
            "  --no-seq         Skip SEQUENTIAL mode\n",
            argv[0]);
        return 1;
    }
    const char * model_path = argv[1];

    int  n_ctx       = 2048;
    bool use_content = false;
    int  n_args      = 6;
    bool run_draft   = true;
    bool run_seq     = true;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--content")  == 0) use_content = true;
        if (strcmp(argv[i], "--no-draft") == 0) run_draft   = false;
        if (strcmp(argv[i], "--no-seq")   == 0) run_seq     = false;
        if (strcmp(argv[i], "--n-args")   == 0 && i + 1 < argc) {
            n_args = atoi(argv[++i]);
            if (n_args < 0) n_args = 0;
            if (n_args > 6) n_args = 6;
        }
        if (strcmp(argv[i], "--n-ctx")    == 0 && i + 1 < argc) {
            n_ctx = atoi(argv[++i]);
            if (n_ctx < 256)  n_ctx = 256;
            if (n_ctx > 32768) n_ctx = 32768;
        }
    }

    // 活跃 head 列表
    std::vector<int> active_head_indices;
    if (use_content) active_head_indices.push_back(0);
    active_head_indices.push_back(1);
    for (int i = 0; i < n_args; i++) active_head_indices.push_back(2 + i);
    int n_active = (int)active_head_indices.size();

    // --- 加载模型 ---
    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99;
    auto * model = llama_model_load_from_file(model_path, mparams);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }

    const auto * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    // --- 模型 & 内存信息 ---
    uint64_t model_size   = llama_model_size(model);
    uint64_t model_params = llama_model_n_params(model);

    // KV cache 估算: 对 Qwen3-4B (36L, d=2560, 20 KV heads, d_head=128, GQA=4:1)
    // KV per layer = 2 * n_ctx * n_kv_heads * d_head * sizeof(f16)
    // 精确计算需要模型元数据, 这里用 model_size 粗略推
    // 实际 KV cache 大小由 llama.cpp 在初始化时报告

    printf("\n=== Model Info ===\n\n");
    printf("  Path   : %s\n", model_path);
    printf("  Params : %.2fB\n", model_params / 1e9);
    printf("  Size   : %.1f MB (on disk / VRAM)\n", model_size / (1024.0 * 1024.0));
    printf("  Vocab  : %d tokens\n", n_vocab);

    // --- Context ---
    auto cparams = llama_context_default_params();
    cparams.n_ctx     = n_ctx;
    cparams.n_batch   = n_ctx;
    cparams.n_ubatch  = 512;
    cparams.n_seq_max = MAX_HEADS;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cparams.kv_unified = true;

    auto * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "Failed to create context\n"); return 1; }

    auto mem = llama_get_memory(ctx);

    printf("\n=== Context Config ===\n\n");
    printf("  n_ctx     : %d\n", n_ctx);
    printf("  n_batch   : %d\n", n_ctx);
    printf("  n_ubatch  : 512\n");
    printf("  n_seq_max : %d\n", MAX_HEADS);
    printf("  flash_attn: enabled\n");
    printf("  kv_unified: true\n");

    // --- 特殊 token ---
    llama_token null_id   = find_single_token(vocab, "<|null|>");
    llama_token im_end_id = find_single_token(vocab, "<|im_end|>");

    printf("\n=== Special Tokens ===\n\n");
    printf("  %-20s %s\n", "<|null|>",
           null_id >= 0 ? (std::to_string(null_id) + " OK").c_str() : "MISSING");
    printf("  %-20s %s\n", "<|im_end|>",
           im_end_id >= 0 ? (std::to_string(im_end_id) + " OK").c_str() : "MISSING");

    bool all_found = (null_id >= 0 && im_end_id >= 0);
    for (int h = 0; h < MAX_HEADS; h++) {
        ALL_HEADS[h].open_id  = find_single_token(vocab, ALL_HEADS[h].open_tag);
        ALL_HEADS[h].close_id = find_single_token(vocab, ALL_HEADS[h].close_tag);
        bool ok = (ALL_HEADS[h].open_id >= 0 && ALL_HEADS[h].close_id >= 0);
        if (!ok) all_found = false;
        printf("  %-20s open=%-6d close=%-6d %s\n",
               ALL_HEADS[h].name, ALL_HEADS[h].open_id, ALL_HEADS[h].close_id,
               ok ? "OK" : "MISSING");
    }
    if (!all_found) {
        printf("\n  WARNING: Some special tokens missing.\n");
    }

    // --- 活跃 Head ---
    printf("\n=== Active Heads ===\n\n");
    printf("  content: %s | function: always | args: %d\n",
           use_content ? "ON" : "OFF", n_args);
    printf("  active (%d):", n_active);
    for (int hi : active_head_indices) {
        printf(" %s[%d]", ALL_HEADS[hi].name, hi);
    }
    printf("\n");

    // ============================================================
    // Cold Start
    // ============================================================

    std::string cold_str = build_cold_prompt();
    auto cold_tokens = tokenize(vocab, cold_str, false, true);
    int n_cold = (int)cold_tokens.size();

    printf("\n=== Cold Start ===\n\n");
    auto t_cold = Clock::now();
    {
        auto batch = llama_batch_init(n_cold, 0, 1);
        batch.n_tokens = 0;
        for (int i = 0; i < n_cold; i++) {
            int idx = batch.n_tokens;
            batch.token[idx]     = cold_tokens[i];
            batch.pos[idx]       = i;
            batch.n_seq_id[idx]  = 1;
            batch.seq_id[idx][0] = 0;
            batch.logits[idx]    = 0;
            batch.n_tokens++;
        }
        llama_decode(ctx, batch);
        llama_synchronize(ctx);
        llama_batch_free(batch);
    }
    double dt_cold = ms_since(t_cold);
    double cold_tps = (n_cold > 0 && dt_cold > 0) ? n_cold / (dt_cold / 1000.0) : 0;

    printf("  Tokens : %d\n", n_cold);
    printf("  Time   : %.1fms\n", dt_cold);
    printf("  Speed  : %.0f tok/s\n", cold_tps);
    printf("  KV used: %d / %d positions (%.1f%%)\n",
           n_cold, n_ctx, 100.0 * n_cold / n_ctx);

    // ============================================================
    // Test Execution
    // ============================================================

    int n_cases = sizeof(CASES) / sizeof(CASES[0]);

    printf("\n=== SimpleTool Voice Commander Test ===\n");
    printf("  Cases: %d | Draft: %s | Seq: %s\n\n",
           n_cases, run_draft ? "ON" : "OFF", run_seq ? "ON" : "OFF");

    // --- PARALLEL FULL ---
    std::vector<CasePerf> full_perfs;
    {
        printf("======================================================\n");
        printf("  PARALLEL - FULL MODEL\n");
        printf("======================================================\n\n");

        for (int i = 0; i < n_cases; i++) {
            auto & tc = CASES[i];
            auto hot_tokens = tokenize(vocab, build_hot_prompt(tc.environment, tc.history, tc.query), false, true);
            auto perf = run_parallel_inference(ctx, mem, vocab, null_id, im_end_id,
                                               active_head_indices, hot_tokens, n_cold,
                                               tc, n_vocab, false);
            print_case_results("[FULL]", perf);
            full_perfs.push_back(perf);
        }
        print_summary("[FULL] PARALLEL", full_perfs, dt_cold, n_cold);
    }

    // --- PARALLEL DRAFT ---
    std::vector<CasePerf> draft_perfs;
    if (run_draft) {
        printf("======================================================\n");
        printf("  PARALLEL - DRAFT MODE\n");
        printf("======================================================\n\n");

        for (int i = 0; i < n_cases; i++) {
            auto & tc = CASES[i];
            auto hot_tokens = tokenize(vocab, build_hot_prompt(tc.environment, tc.history, tc.query), false, true);
            auto perf = run_parallel_inference(ctx, mem, vocab, null_id, im_end_id,
                                               active_head_indices, hot_tokens, n_cold,
                                               tc, n_vocab, true);
            print_case_results("[DRAFT]", perf);
            draft_perfs.push_back(perf);
        }
        print_summary("[DRAFT] PARALLEL", draft_perfs, dt_cold, n_cold);
    }

    // --- SEQUENTIAL ---
    std::vector<CasePerf> seq_perfs;
    if (run_seq) {
        printf("======================================================\n");
        printf("  SEQUENTIAL - Reference Baseline\n");
        printf("======================================================\n\n");

        for (int i = 0; i < n_cases; i++) {
            auto & tc = CASES[i];
            auto hot_tokens = tokenize(vocab, build_hot_prompt(tc.environment, tc.history, tc.query), false, true);
            auto perf = run_sequential_inference(ctx, mem, vocab, null_id, im_end_id,
                                                  active_head_indices, hot_tokens, n_cold,
                                                  tc, n_vocab);
            print_case_results("[SEQ]", perf);
            seq_perfs.push_back(perf);
        }
        print_summary("[SEQ] SEQUENTIAL", seq_perfs, dt_cold, n_cold);
    }

    // ============================================================
    // 最终汇总
    // ============================================================

    printf("======================================================\n");
    printf("  FINAL SUMMARY\n");
    printf("======================================================\n\n");

    printf("  Model: %s (%.2fB params, %.1f MB)\n",
           model_path, model_params / 1e9, model_size / (1024.0 * 1024.0));
    printf("  Config: n_ctx=%d, heads=%d (content=%s, args=%d)\n",
           n_ctx, n_active, use_content ? "on" : "off", n_args);
    printf("  Cold start: %.1fms (%d tok, %.0f tok/s)\n\n", dt_cold, n_cold, cold_tps);

    // 对比表
    printf("  %-12s %-8s %-10s %-10s %-10s %-10s\n",
           "Mode", "Pass", "HotPF(ms)", "Signal", "Decode", "E2E(ms)");
    printf("  ---------------------------------------------------------------\n");

    auto print_row = [](const char * label, const std::vector<CasePerf> & ps) {
        if (ps.empty()) return;
        int pass = 0;
        double sh = 0, ss = 0, sd = 0, se = 0;
        for (const auto & p : ps) {
            if (p.passed) pass++;
            sh += p.dt_hot_prefill_ms;
            ss += p.dt_signal_ms;
            sd += p.dt_decode_ms;
            se += p.dt_e2e_ms;
        }
        int n = (int)ps.size();
        printf("  %-12s %d/%-5d %-10.1f %-10.1f %-10.1f %-10.1f\n",
               label, pass, n, sh/n, ss/n, sd/n, se/n);
    };

    print_row("FULL", full_perfs);
    print_row("DRAFT", draft_perfs);
    print_row("SEQ", seq_perfs);

    printf("\n  Memory:\n");
    printf("    Model weights : %.1f MB\n", model_size / (1024.0 * 1024.0));
    printf("    KV cache (est): n_ctx=%d, n_seq=%d\n", n_ctx, MAX_HEADS);
    printf("    KV utilization: ~%d / %d positions per request\n",
           n_cold + 80 + 10, n_ctx);  // rough estimate

    printf("\n======================================================\n");

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}