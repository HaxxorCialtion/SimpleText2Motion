// cpp/config.h - shared config loader for fused_server / t2m_infer / bench_t2m
// Reads config.toml (or path given via --config). All paths inside the toml
// are resolved relative to the toml file's own directory, NOT cwd.
#pragma once

#include <filesystem>
#include <string>
#include <vector>
#include <stdexcept>

#define TOML_HEADER_ONLY 1
#include <toml++/toml.hpp>

namespace fs = std::filesystem;

struct PathsCfg {
    std::string llama_model;
    std::string t2m_onnx_dir;
    std::string logs_dir = "logs";
};

struct FusedCfg {
    std::string binary          = "llama.cpp/fused_server";
    int  port_hidden            = 8421;
    int  port_simpletool        = 8422;
    int  hidden_layer           = 5;
    int  n_ctx                  = 2048;
    int  n_gpu_layers           = 999;
    int  max_cold_tokens        = -1;
    bool debug_cache            = false;
};

struct RunCfg {
    int port = 8765;
};

struct T2mCfg {
    std::string binary          = "build/t2m_infer";
    int         server_port     = 8423;
    std::string ep              = "auto";   // auto|cuda|cpu
    int         num_steps       = 4;
    std::string fused_host      = "127.0.0.1";
    // fused_port is derived from fused.port_hidden at use time
    std::string motion_to_vrma_bin = ""; 
};

struct BenchCfg {
    int         warmup_runs     = 3;
    int         timed_runs      = 20;
    float       duration        = 3.0f;
    unsigned    seed            = 42;
    std::vector<std::string> prompts = {
        "A person walks forward.",
        "A person waves their right hand.",
        "A person jumps in place.",
        "A person turns left and looks around..",
        "A person sits down slowly.",
    };
};

struct AppConfig {
    PathsCfg paths;
    FusedCfg fused;
    T2mCfg   t2m;
    BenchCfg bench;
    RunCfg   run;
    std::string config_dir;   // dir containing the toml file, "" if defaults

    static AppConfig defaults() { return AppConfig{}; }

    // Resolve a relative path against config_dir; absolute paths pass through.
    std::string resolve(const std::string& p) const {
        if (p.empty()) return p;
        fs::path pp(p);
        if (pp.is_absolute()) return pp.string();
        if (config_dir.empty()) return pp.string();   // relative to cwd
        return (fs::path(config_dir) / pp).lexically_normal().string();
    }

    static AppConfig from_toml(const std::string& path) {
        AppConfig c;
        c.config_dir = fs::absolute(path).parent_path().string();

        toml::table t;
        try {
            t = toml::parse_file(path);
        } catch (const toml::parse_error& e) {
            throw std::runtime_error("config parse failed: " + std::string(e.what()));
        }

        if (auto p = t["paths"].as_table()) {
            c.paths.llama_model  = (*p)["llama_model"].value_or(c.paths.llama_model);
            c.paths.t2m_onnx_dir = (*p)["t2m_onnx_dir"].value_or(c.paths.t2m_onnx_dir);
            c.paths.logs_dir     = (*p)["logs_dir"].value_or(c.paths.logs_dir);
            // resolve to absolute
            c.paths.llama_model  = c.resolve(c.paths.llama_model);
            c.paths.t2m_onnx_dir = c.resolve(c.paths.t2m_onnx_dir);
            c.paths.logs_dir     = c.resolve(c.paths.logs_dir);
        }
        if (auto f = t["fused_server"].as_table()) {
            c.fused.binary          = (*f)["binary"].value_or(c.fused.binary);
            c.fused.binary          = c.resolve(c.fused.binary);
            c.fused.port_hidden     = (int)(*f)["port_hidden"].value_or((int64_t)c.fused.port_hidden);
            c.fused.port_simpletool = (int)(*f)["port_simpletool"].value_or((int64_t)c.fused.port_simpletool);
            c.fused.hidden_layer    = (int)(*f)["hidden_layer"].value_or((int64_t)c.fused.hidden_layer);
            c.fused.n_ctx           = (int)(*f)["n_ctx"].value_or((int64_t)c.fused.n_ctx);
            c.fused.n_gpu_layers    = (int)(*f)["n_gpu_layers"].value_or((int64_t)c.fused.n_gpu_layers);
            c.fused.max_cold_tokens = (int)(*f)["max_cold_tokens"].value_or((int64_t)c.fused.max_cold_tokens);
            c.fused.debug_cache     = (*f)["debug_cache"].value_or(c.fused.debug_cache);
        }
        if (auto t2 = t["t2m_infer"].as_table()) {
            c.t2m.motion_to_vrma_bin = (*t2)["motion_to_vrma_bin"].value_or(c.t2m.motion_to_vrma_bin);
            if (!c.t2m.motion_to_vrma_bin.empty())
                c.t2m.motion_to_vrma_bin = c.resolve(c.t2m.motion_to_vrma_bin);
            c.t2m.binary       = (*t2)["binary"].value_or(c.t2m.binary);
            c.t2m.binary       = c.resolve(c.t2m.binary);
            c.t2m.server_port  = (int)(*t2)["server_port"].value_or((int64_t)c.t2m.server_port);
            c.t2m.ep           = (*t2)["ep"].value_or(c.t2m.ep);
            c.t2m.num_steps    = (int)(*t2)["num_steps"].value_or((int64_t)c.t2m.num_steps);
            c.t2m.fused_host   = (*t2)["fused_host"].value_or(c.t2m.fused_host);
        }
        if (auto b = t["bench"].as_table()) {
            c.bench.warmup_runs = (int)(*b)["warmup_runs"].value_or((int64_t)c.bench.warmup_runs);
            c.bench.timed_runs  = (int)(*b)["timed_runs"].value_or((int64_t)c.bench.timed_runs);
            c.bench.duration    = (float)(*b)["duration"].value_or((double)c.bench.duration);
            c.bench.seed        = (unsigned)(*b)["seed"].value_or((int64_t)c.bench.seed);
            if (auto arr = (*b)["prompts"].as_array()) {
                c.bench.prompts.clear();
                for (auto& el : *arr) if (auto s = el.value<std::string>()) c.bench.prompts.push_back(*s);
            }
        }
        if (auto r = t["run_server"].as_table()) {
            c.run.port = (int)(*r)["port"].value_or((int64_t)c.run.port);
        }
        return c;
    }
};