# SimpleT2M

Real-time text-to-motion generation. End-to-end **13–17 ms** per request on a single GPU, from text prompt to streamable VRM animation. Pure C++17, zero Python at runtime.

| GPU | client wall mean | dit_total | EP |
|---|---|---|---|
| NVIDIA H100 | **13.0 ms** | 8.5 ms | CUDA |
| NVIDIA RTX 4090 | **16.9 ms** | 12.3 ms | CUDA |

(20-run timed bench, 4-step Euler DiT, 3-second motions @ 30 FPS, no thermal throttling observed.)

---

## What is this?

SimpleT2M is a fused inference stack for text-conditioned human motion generation:

1. **`fused_server`** — a llama.cpp-based server that exposes (a) a hidden-state extraction port and (b) a multi-head parallel SimpleTool decode port. Built on Qwen3-4B (q4_k_m).
2. **`t2m_infer`** — an ONNX Runtime server that takes hidden states from `fused_server`, runs a refiner + 4-step Euler DiT, and returns a 135-dim SMPL-H rot6d motion buffer (T frames).
3. **`motion_to_vrma`** — converts the 135-dim motion to a `.vrma` (VRM animation, glTF binary) ready for any three-vrm based viewer.
4. **`run_server`** — an HTTP server that bridges a browser front-end (drag-drop VRM viewer, prompt input) to the above pipeline.
5. **`bench_t2m`** — benchmark client for regression testing.

All five are native binaries. No Python at runtime; no Node, no npm, no Docker.

---

## Build (Linux, x86-64, CUDA)

Tested on Ubuntu 22.04 / 24.04 with H100 (sm_90) and RTX 4090 (sm_89). See `BUILD_WINDOWS.md` for the Windows path; macOS works similarly via the Apple Silicon ONNX Runtime + Metal-backed llama.cpp.

### System dependencies (Linux)

These have to be installed via your package manager / CUDA installer:

| Dep | Version | Why |
|---|---|---|
| **CUDA Toolkit** (with dev headers) | ≥ 12.x | `nvcc`, `cublas_v2.h`. Driver-only install will fail. |
| **CMake** | ≥ 3.20 | Root project + llama.cpp build |
| **g++** | C++17 | Tested with 11+ |
| **patchelf** | any | Linux-only: fixes hard-coded rpath in llama.cpp's shared libs after build |

```bash
sudo apt install cmake g++ patchelf
# (install CUDA toolkit per NVIDIA's instructions for your distro)
```

### Vendored dependencies (already in repo or one-time download)

Everything else is vendored under `third_party/` to avoid runtime/build-time network requirements. After cloning, your `third_party/` directory should look like this:

```
third_party/
├── cpp-httplib/                        # HTTP server (single-header)
│   └── httplib.h
├── tomlplusplus/                       # TOML parser (single-header)
│   └── include/toml++/toml.hpp
├── onnxruntime-linux-x64-1.18.1/       # CPU build of ONNX Runtime
├── onnxruntime-linux-x64-gpu-1.26.0/   # GPU (CUDA EP) build
└── onnxruntime-osx-arm64-1.18.1/       # Mac
```

**One-time setup**: if you cloned without the single-header files, fetch them:

```bash
mkdir -p third_party/tomlplusplus/include/toml++ third_party/cpp-httplib
curl -L -o third_party/tomlplusplus/include/toml++/toml.hpp \
     https://raw.githubusercontent.com/marzer/tomlplusplus/master/toml.hpp
curl -L -o third_party/cpp-httplib/httplib.h \
     https://raw.githubusercontent.com/yhirose/cpp-httplib/master/httplib.h
```

The ONNX Runtime tarballs are larger; download from [microsoft/onnxruntime releases](https://github.com/microsoft/onnxruntime/releases) and extract under `third_party/`. The root `CMakeLists.txt` auto-discovers any `onnxruntime-<platform>[-gpu]-<version>/` directory and picks the GPU build by default.

### Model weights

Place under `models/` (paths configurable in `config.toml`):

```
models/
├── qwen3-4b-v2-trim6-q4_k_m.gguf      # LLM for fused_server
└── SimpleT2M/
    ├── birefiner.onnx                 # text refiner
    ├── dit_step.onnx                  # one-step DiT
    └── small_weights.npz              # timestep/duration/cond MLPs
```

### Build commands

See `BUILD_LINUX.md` for the full step-by-step. TL;DR:

```bash
# 1. llama.cpp (CUDA shared libs)
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DBUILD_SHARED_LIBS=ON
cmake --build build --config Release -j

# 2. fused_server (manual g++ — links llama.cpp libs)
g++ -std=c++17 -O2 \
    -I include -I ggml/include -I .. \
    -I ../cpp -I ../third_party/tomlplusplus/include \
    fused_server.cpp \
    -L build/bin -lllama -lggml -lggml-base -lggml-cpu -lpthread \
    -Wl,-rpath,'$ORIGIN/build/bin' \
    -o fused_server
cd ..

# 3. everything else (root CMake)
cmake -B build && cmake --build build -j
```

Replace `89` with your GPU's compute capability (75=T4, 80=A100, 86=A10/RTX30, 89=RTX40, 90=H100).

---

## Run

```bash
./scripts/start_servers.sh     # terminal A: brings up fused_server + t2m_infer
./build/run_server             # terminal B: HTTP demo on :8765, opens browser
./build/bench_t2m              # terminal B: 20-run benchmark
```

Ctrl-C in terminal A stops both backend servers cleanly.

All paths, ports, and bench prompts live in `config.toml`. All five binaries accept `--config <path>` to point at an alternate file; `start_servers.sh` reads the same config so it stays in sync.

---

## Repository layout

```
SimpleT2M/
├── CMakeLists.txt
├── config.toml                  # unified config for all binaries
├── BUILD_LINUX.md
├── BUILD_WINDOWS.md
├── index.html                   # three-vrm front-end for run_server
├── cpp/
│   ├── platform.h               # cross-platform socket abstraction
│   ├── config.h                 # TOML loader (uses toml++)
│   ├── npz_reader.h             # .npz / .npy reader
│   ├── t2m_infer.cpp            # ONNX inference server + one-shot
│   ├── motion_to_vrma.cpp       # 135-dim → .vrma offline tool
│   ├── bench_t2m.cpp            # benchmark client
│   └── run_server.cpp           # HTTP demo server
├── llama.cpp/
│   └── fused_server.cpp         # hidden-state + SimpleTool server
├── scripts/
│   ├── start_servers.sh         # Linux/macOS launcher
│   └── start_servers.ps1        # Windows launcher
├── assets/                      # .vrm + idle/sample .vrma for the demo
├── models/                      # weights (not in git)
└── third_party/                 # vendored deps (see above)
```

---

## Architecture notes

The pipeline is intentionally split across two ports rather than fused into one process:

```
prompt
  │
  ├──TCP 8421──► fused_server: extract layer-5 hidden state (Qwen3-4B)
  │                  │
  │                  └──► (n_tokens, 2560) fp32
  │
  └─────────► t2m_infer: refiner → 4-step Euler DiT
                     │
                     └──► (T, 135) fp32 motion
                            │
                            └──► motion_to_vrma → .vrma → browser
```

Why the split? The llama.cpp side benefits from being a long-lived CUDA process with warm KV cache (cold-prefix cache hit detection saves ~100ms per request). The ONNX side benefits from ORT's separate kernel scheduling. Tying them together would force one execution provider on both halves; keeping them apart lets you mix CPU/GPU per stage (e.g. Apple Silicon: Metal-backed llama.cpp + CPU EP for the small DiT).

The hidden-state interface is the contract between them: both can be replaced independently as long as the protocol on port 8421 holds.

---

## Status

- [x] Linux (H100, RTX 4090): bench passes, demo runs end-to-end
- [ ] Windows (CUDA): builds, perf TBD
- [ ] macOS (Apple Silicon, Metal/CPU EP): builds, perf TBD

---

## License

See `LICENSE` and `NOTICE`. Third-party components retain their own licenses:

- llama.cpp — MIT
- ONNX Runtime — MIT
- toml++ — MIT
- cpp-httplib — MIT
- zlib — zlib license

Qwen3-4B weights are subject to the Tongyi Qianwen license; users must accept its terms separately.
