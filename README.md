<!-- Language: English | [中文](README_ZH.md) -->

# SimpleText2Motion

> Real-time text-to-motion: type an English prompt, get a VRMA animation streamed to a VRM character in the browser.

[中文 README](README_ZH.md) · [License](#license) · [ModelScope mirror](#model-download)

<table>
<tr>
<td width="50%">
  <img src="./assets/tmp.png" alt="demo" />
</td>
<td width="50%">

https://github.com/user-attachments/assets/31aa4018-7225-46eb-a469-6097b5e64802

</td>
</tr>
</table>
---

## Overview

SimpleText2Motion is a **lightweight, cross-platform, Python-free** real-time
text-to-motion framework. Type a prompt like
`"A person walks forward and waves their right hand."` and it generates the
corresponding human motion in real time, driving a VRM character in the browser
via VRMA.

The entire inference pipeline is implemented in C++ and needs **no Python
runtime**: LLM inference runs on [llama.cpp](https://github.com/ggml-org/llama.cpp),
and the motion model is exported to ONNX and executed by ONNX Runtime. This lets
it compile into standalone executables, easy to ship on Linux / macOS / Windows.

### The model

The motion generator is a lightweight **conditional DiT + Flow Matching** model:

- **Text encoder (Bi-Refiner)**: 2 bidirectional self-attention layers on top of
  a frozen SimpleTool-4B intermediate hidden state (dim 2560), producing
  token-level and pooled global text features.
- **Motion decoder (DiT)**: 8 Transformer blocks with cross-attention to the
  text tokens; timestep / text / duration are injected via adaLN-Zero.
- **Objective**: Conditional Flow Matching (velocity prediction).
- **Output**: `(T, 135)` SMPL-H rot6d (3 translation + 6 root rotation +
  21×6 body rotation) at 30 fps.

Trained on ~**1M English text–motion pairs**, so instruction following is
currently English-oriented.

---

## Inference framework & data flow

The framework itself is a core part of this project: **all C++, cross-platform,
no Python dependency**, with each stage decoupled over a local TCP port for easy
extension and integration.

```
"A person walks forward and waves their right hand."
                     │
                     ▼
             ┌───────────────┐         ┌──────────────────┐
             │  fused_server │ ──────▶ │  SimpleTool-4B   │
             │   port 8421   │ hidden  │  via llama.cpp   │
             └───────┬───────┘         └──────────────────┘
                     │ (N, 2560) float32
                     ▼
             ┌───────────────┐         ┌──────────────────┐
             │   t2m_infer   │ ──────▶ │   birefiner +    │
             │   port 8423   │  ONNX   │  dit_step (ORT)  │
             └───────┬───────┘         └──────────────────┘
                     │ (T, 135) float32  (SMPL-H rot6d, 30 fps)
                     ▼
             motion.bin ─▶ motion_to_vrma ─▶ motion.vrma
```

Each stage is a standalone service on a fixed local port:

- `fused_server` (:8421) — extracts SimpleTool-4B hidden states via llama.cpp.
- `t2m_infer` (:8423) — runs birefiner + DiT on ONNX Runtime, yielding the motion tensor.
- `motion_to_vrma` — converts the motion tensor into a playable VRMA.
- `run_server` (:8765) — HTTP server chaining the pipeline and streaming results to the browser.

**The ports are cleanly separated — contributions and integration are welcome.**
You can tap a single stage (e.g. call `t2m_infer` directly for the motion
tensor) or embed the whole pipeline into your own app.

---

## Performance

End-to-end latency for a 3-second clip (`bench_t2m`):

| Hardware                     | client_wall | hidden  | dit_total |
|------------------------------|------------:|--------:|----------:|
| H100 (CUDA)                  |    15.7 ms  |  2.4 ms |   11.1 ms |
| Apple M4 Pro (Metal)         |    71.3 ms  | 21.5 ms |   48.2 ms |
| RTX 3060 + i5-12600KF (CUDA) |   117.5 ms  | 11.8 ms |  102.1 ms |

---

## Per-platform backends

The LLM stage (llama.cpp) always uses GPU acceleration for speed; the motion DiT
is tiny, so whether ONNX uses the GPU has limited impact on latency.

| Platform | llama.cpp backend | ONNX Runtime backend |
|----------|-------------------|----------------------|
| Linux    | CUDA              | CPU or CUDA          |
| macOS    | Metal (automatic) | CPU                  |
| Windows  | CUDA              | **CPU**              |

> **Windows note**: the Windows build runs **llama.cpp on CUDA and ONNX Runtime
> on CPU** by default. This keeps GPU acceleration for the LLM stage while
> avoiding the large cuDNN / CUDA-ONNX dependency; the DiT model is small, so
> CPU inference adds only tens of ms — fine for real-time use.

---

## Installation

Prerequisites: Git, CMake ≥ 3.18, a C++17 compiler; GPU builds also need the
platform CUDA Toolkit. macOS needs `xcode-select --install`; Windows needs
Visual Studio 2022 (Desktop development with C++).

### 1. Clone and download models

```bash
git clone https://github.com/HaxxorCialtion/SimpleText2Motion.git
cd SimpleText2Motion

mkdir -p models/SimpleT2M
BASE=https://huggingface.co/Cialtion/SimpleLove/resolve/main
wget -q -P ./models           $BASE/SimpleTool-4B-trim6-q4.gguf &
wget -q -P ./models/SimpleT2M $BASE/SimpleT2M/birefiner.onnx &
wget -q -P ./models/SimpleT2M $BASE/SimpleT2M/config.json &
wget -q -P ./models/SimpleT2M $BASE/SimpleT2M/dit_step.onnx &
wget -q -P ./models/SimpleT2M $BASE/SimpleT2M/small_weights.bin &
wget -q -P ./models/SimpleT2M $BASE/SimpleT2M/small_weights.npz &
wait
```

<details>
<summary>Windows (PowerShell) model download</summary>

```powershell
mkdir models\SimpleT2M
$BASE = "https://huggingface.co/Cialtion/SimpleLove/resolve/main"
curl.exe -L -o models\SimpleTool-4B-trim6-q4.gguf  $BASE/SimpleTool-4B-trim6-q4.gguf
curl.exe -L -o models\SimpleT2M\birefiner.onnx     $BASE/SimpleT2M/birefiner.onnx
curl.exe -L -o models\SimpleT2M\config.json        $BASE/SimpleT2M/config.json
curl.exe -L -o models\SimpleT2M\dit_step.onnx      $BASE/SimpleT2M/dit_step.onnx
curl.exe -L -o models\SimpleT2M\small_weights.bin  $BASE/SimpleT2M/small_weights.bin
curl.exe -L -o models\SimpleT2M\small_weights.npz  $BASE/SimpleT2M/small_weights.npz
```

China mirror (ModelScope): replace `$BASE` with
`https://www.modelscope.cn/models/cialtion/SImpleLove/resolve/master`
</details>

### 2. Third-party dependencies

```bash
mkdir -p third_party && cd third_party
git clone --depth 1 https://github.com/yhirose/cpp-httplib.git
git clone --depth 1 https://github.com/marzer/tomlplusplus.git
# git clone --depth 1 https://github.com/madler/zlib.git   # Windows only
```

Download ONNX Runtime for your OS, then make an `onnxruntime` symlink/rename so
the build always references `third_party/onnxruntime`:

```bash
# Linux (CPU package recommended)
ORT_PKG=onnxruntime-linux-x64-1.20.1
curl -L -O https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/$ORT_PKG.tgz
tar -xzf $ORT_PKG.tgz && rm $ORT_PKG.tgz && ln -sfn $ORT_PKG onnxruntime

# macOS (arm64)
# ORT_PKG=onnxruntime-osx-arm64-1.18.1
# curl -L -O https://github.com/microsoft/onnxruntime/releases/download/v1.18.1/$ORT_PKG.tgz
# tar -xzf $ORT_PKG.tgz && rm $ORT_PKG.tgz && ln -sfn $ORT_PKG onnxruntime
cd ..
```

<details>
<summary>Windows (PowerShell) ONNX Runtime</summary>

```powershell
cd third_party
git clone --depth 1 https://github.com/madler/zlib.git    # required on Windows
$ORT_PKG = "onnxruntime-win-x64-1.20.1"                    # CPU (default on Windows)
# $ORT_PKG = "onnxruntime-win-x64-gpu-1.20.1"              # GPU (CUDA 12)
curl.exe -L -o "$ORT_PKG.zip" "https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/$ORT_PKG.zip"
Expand-Archive "$ORT_PKG.zip" -DestinationPath .
Remove-Item "$ORT_PKG.zip"
Rename-Item $ORT_PKG onnxruntime
cd ..
```
</details>

### 3. Build

```bash
# llama.cpp — Linux uses CUDA; on macOS drop GGML_CUDA (Metal is automatic)
cd llama.cpp && rm -rf build
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native -DBUILD_SHARED_LIBS=ON
cmake --build build --config Release -j

# fused_server — on macOS replace $ORIGIN with @loader_path
g++ -std=c++17 -O2 -I include -I ggml/include -I .. -I ../cpp \
    -I ../third_party/tomlplusplus/include -I ../third_party/onnxruntime/include \
    fused_server.cpp -L build/bin -lllama -lggml -lggml-base -lggml-cpu \
    -L ../third_party/onnxruntime/lib -lonnxruntime -lpthread \
    -Wl,-rpath,'$ORIGIN/build/bin' \
    -Wl,-rpath,'$ORIGIN/../third_party/onnxruntime/lib' -o fused_server
cd ..

# everything else
rm -rf build && cmake -B build && cmake --build build -j
```

<details>
<summary>Windows (PowerShell) build</summary>

```powershell
# llama.cpp (CUDA)
cd llama.cpp
if (Test-Path build) { Remove-Item build -Recurse -Force }
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native -DBUILD_SHARED_LIBS=ON
cmake --build build --config Release -j
cd ..

# fused_server is built by the CMake step below (no manual compile on Windows)
if (Test-Path build) { Remove-Item build -Recurse -Force }
cmake -B build
cmake --build build --config Release -j
```
</details>

### 4. Run

```bash
# Linux / macOS
bash ./scripts/start_servers.sh   # terminal A: fused_server + t2m_infer
./build/run_server                # terminal B: demo -> http://localhost:8765

# Windows (PowerShell)
# ./scripts/start_servers.ps1
# .\build\Release\run_server.exe
```

---

## Model download

Weights are hosted on Hugging Face, with a ModelScope mirror for China.

- Hugging Face: `https://huggingface.co/Cialtion/SimpleLove`
- ModelScope (China mirror): `https://www.modelscope.cn/models/cialtion/SImpleLove`

Files: `SimpleTool-4B-trim6-q4.gguf` plus, under `SimpleT2M/`,
`birefiner.onnx`, `dit_step.onnx`, `config.json`, `small_weights.bin`,
`small_weights.npz`.

---

## Troubleshooting

**`t2m_infer failed to open port 8423 within 30s` (GPU ONNX only).**
First GPU warmup runs cuDNN algorithm search + kernel compilation (30–60 s).
Increase the timeout in `scripts/start_servers.sh`:

```bash
sed -i 's/\(wait_for_port "\$T2M_PORT" "t2m_infer"\) 30/\1 90/' scripts/start_servers.sh
```

**`CUDA EP unavailable ... falling back to CPU` / `libcublasLt.so.11`.**
The ONNX-GPU package doesn't match your CUDA major version (`.so.11` = CUDA 11,
but you have CUDA 12). Since 1.19 the CUDA-12 Linux package is
`onnxruntime-linux-x64-gpu-<ver>` (no `cuda12` suffix), and it needs **cuDNN 9**.
If you don't need GPU ONNX, just use the CPU package — latency is still real-time.

---

## Roadmap

- [ ] Prebuilt, **compile-free distributions** for each platform.
- [ ] Better **instruction following** for text-to-motion (complex / compositional / multi-action prompts).

---

## Contributing & integration

Stage ports are fixed and the protocol is simple — extensions and integrations
are welcome. Open an issue for questions or collaboration.

---

## Acknowledgements

- [llama.cpp](https://github.com/ggml-org/llama.cpp) — LLM inference backend.
- The motion generator (Bi-Refiner + DiT + Flow Matching) is this project's own lightweight implementation.

---

## License

This project is released under the **MIT License** — **commercial use**,
modification, and redistribution are fully permitted. See [LICENSE](LICENSE).

> Note: third-party components referenced here (e.g. llama.cpp) remain under their own licenses.