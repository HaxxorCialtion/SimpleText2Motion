# SimpleT2M (macOS)

Real-time text-to-motion engine for AI-native NPCs. Apple Silicon edition.
~90 ms end-to-end on an M-series Mac, CPU only, zero GPU vendor lock-in.

```
"A person walks forward and waves their right hand."
                    │
                    ▼
            ┌───────────────┐         ┌──────────────────┐
            │  fused_server │ ──────▶ │ Qwen3-4B (q8_0)  │
            │   port 8421   │ hidden  │  via llama.cpp   │
            └───────┬───────┘         └──────────────────┘
                    │ (N, 2560) float32
                    ▼
            ┌───────────────┐         ┌──────────────────┐
            │   t2m_infer   │ ──────▶ │  birefiner +     │
            │   port 8423   │  ONNX   │  dit_step (ORT)  │
            └───────┬───────┘         └──────────────────┘
                    │ (T, 135) float32  (SMPL-H rot6d, 30 fps)
                    ▼
            motion.bin ─▶ motion_to_vrma ─▶ motion.vrma
```

This repo ships the macOS port of the inference stack. The application layer
(NPC director, VRM rendering, dialogue) lives elsewhere.

---

## Requirements

- Apple Silicon Mac (M1 / M2 / M3 / M4)
- macOS 12 Monterey or later
- Xcode Command Line Tools (`xcode-select --install`)
- CMake ≥ 3.20 (`brew install cmake` if missing)
- Python ≥ 3.9 with `numpy` (only for the example clients)

That's it. No CUDA, no Metal SDK, no Rosetta, no `brew install onnxruntime`.
ONNX Runtime is vendored under `third_party/`.

---

## Quick start

This assumes you have the model files already in place under `models/`
(see [Models](#models) below if not).

```bash
# 1. Build the C++ binaries (~30s)
cmake -B build
cmake --build build -j

# 2. Start both servers
./scripts/start_servers.sh
```

When you see `all servers ready`, leave that terminal running and open a new one:

```bash
# 3. Generate a motion file and convert it to .vrma
python client/t2m_client.py "A person waves their right hand."
./build/motion_to_vrma --in motion.bin --out hello.vrma
```

Drop `hello.vrma` into any [VRM Animation viewer](https://vrm.dev/en/vrm_animation/)
to play it back. That's the full pipeline.

To stop the servers, press `Ctrl-C` in the first terminal.

---

## What's in this repo

```
.
├── cpp/                            C++ sources (4 files, pure stdlib + ORT)
│   ├── platform.h                  Win/POSIX shim (sockets, paths)
│   ├── npz_reader.h                Minimal .npz parser (zlib only)
│   ├── t2m_infer.cpp               Inference server + one-shot CLI
│   └── motion_to_vrma.cpp          motion.bin → .vrma offline tool
│
├── llama.cpp/                      LLM backend
│   ├── fused_server.cpp            Custom server: hidden states + simpletool
│   ├── platform_net.h              Cross-platform socket shim
│   └── ...                         (llama.cpp upstream sources)
│
├── third_party/
│   └── onnxruntime-osx-arm64-1.18.1/  Vendored ORT (~12 MB dylib)
│
├── models/                         Model files (see below — not in git)
│   ├── qwen3-4b-q8_0.gguf
│   └── SimpleT2M/
│       ├── birefiner.onnx
│       ├── dit_step.onnx
│       ├── small_weights.npz
│       └── config.json
│
├── scripts/
│   ├── start_servers.sh            One-command launcher
│   └── stop_servers.sh             Standalone cleanup
│
├── client/                         Python SDK + demos
│   ├── t2m_client.py               Minimal client for the t2m service
│   ├── simpletool_client.py        Minimal client for the simpletool service
│   ├── demo_client.py              End-to-end demo: query → tool → motion → vrma
│   ├── bench_t2m.py                Benchmark with warmup + stats
│   └── prompts.txt                 Default prompt set for bench
│
├── CMakeLists.txt
└── README.md                       This file
```

---

## Models

Model files are **not** included in the git repo (too large; they live on
Hugging Face). If you cloned this repo, place them under `models/` like this:

```
models/
├── qwen3-4b-q8_0.gguf              ~4 GB
└── SimpleT2M/
    ├── birefiner.onnx              30 MB
    ├── dit_step.onnx              180 MB
    ├── small_weights.npz          3.6 MB
    └── config.json
```

> Hugging Face URLs will be linked here once published.
> For now, ask the author or copy from your existing checkout.

---

## Three clients, three jobs

### `t2m_client.py` — just motion
Lowest-level wrapper. One function call, returns `(T, 135)` numpy array.

```python
from t2m_client import call_t2m
motion, timing = call_t2m("127.0.0.1", 8423,
                          "A person walks forward.",
                          duration=3.0, seed=42)
# motion: (90, 135) float32  — see "Motion format" below
```

```bash
python client/t2m_client.py "A person walks forward."
# Writes motion.bin in the current directory.
```

### `simpletool_client.py` — just LLM tool calling
Multi-head parallel decode for structured function calling. No motion involved.

```bash
python client/simpletool_client.py                 # 4 built-in scenarios
python client/simpletool_client.py --query "Walk to the chair, slowly."
python client/simpletool_client.py --cache-bench   # show cold-cache speedup
```

Each call returns parsed `function`, `content`, and `arg1..arg6` heads decoded
in parallel by [SimpleTool](https://arxiv.org/abs/2603.00030). The server keeps
a byte-identical cold-prefix cache; subsequent calls with the same system
prompt + tools skip prefill entirely (~700 ms saved per call).

### `demo_client.py` — the full pipeline
Wires everything together: user query → simpletool decides what to do → motion
prompt is generated → t2m produces `(T, 135)` → `motion_to_vrma` converts to
`.vrma`. Falls back gracefully if any stage is unavailable.

```bash
python client/demo_client.py                                  # interactive REPL
python client/demo_client.py "Walk over to the chair slowly."
python client/demo_client.py --no-simpletool "Wave."          # skip LLM step
```

Each run produces a timestamped directory under `out/`:

```
out/20260515_120759/
├── motion.bin         raw (T, 135) float32
├── motion.npy         same, numpy-friendly
├── motion.vrma        ready for VRM viewers
└── meta.txt           prompt + simpletool decision + stats
```

---

## Performance

Measured on M-series MacBook (Apple Silicon, CPU only). Run
`python client/bench_t2m.py` to reproduce.

| Stage                    | Mean    | p95     |
|--------------------------|--------:|--------:|
| **client_wall (e2e)**    |  89 ms  | 102 ms  |
| hidden (Qwen3-4B q8)     |  40 ms  |  52 ms  |
| refiner (birefiner.onnx) |  0.5 ms |  0.6 ms |
| DiT total (4 steps)      |  48 ms  |  48 ms  |
| DiT step 0               |  13 ms  |  14 ms  |
| DiT steps 1–3 avg        |  12 ms  |  12 ms  |

For context: same model on an Intel i5-12600KF + ORT CPU EP runs at 114 ms
wall / 97 ms DiT total. Apple Silicon is ~30% faster on this workload.

> n=20, duration=3.0s, after 3 warmup runs. Single physical request at a time;
> multi-NPC scenes are served sequentially — GPU/CPU is the bottleneck, not
> the protocol.

---

## Architecture notes

A few design choices worth knowing if you're going to modify this:

**Why CPU only?** The DiT is ~50 M parameters with `hidden_dim=512` and short
sequences (T<200). It's launch-bound on consumer GPUs — per-op kernel launch
overhead dominates compute. On Windows we measured DirectML at 2× **slower**
than CPU EP on the same machine. Apple's CoreML EP would lose for the same
reason. Apple Silicon CPU + Accelerate.framework BLAS is the right fit here.

**Why LLM hidden states instead of a text encoder?** birefiner conditions on
Qwen3 layer-5 hidden states directly, replacing what would otherwise be a
CLIP-class text encoder. Saves loading a second model when the product already
runs an LLM.

**Why service mode instead of spawn-per-request?** Loading ORT sessions and
running ORT's first-pass mem-pattern planning costs ~80 ms. We pay it once at
startup, not per request.

**Protocol** Both services speak a tiny binary protocol over loopback TCP:

| Service     | Port | Endianness | Job                                              |
|-------------|------|------------|--------------------------------------------------|
| hidden      | 8421 | big        | prompt → Qwen3 hidden states (N, 2560)           |
| simpletool  | 8422 | big        | system+tools+query → parallel tool-call heads    |
| t2m_infer   | 8423 | little     | prompt + duration → motion (T, 135)              |

Yes, the endianness is inconsistent across the two services. This is a known
historical wart inherited from independent development of the two codebases.
The Python clients hide it; if you write a native client, mind the bytes.

Full wire format for `t2m_infer`:

```
request:
  [u32 LE]  prompt_len_bytes
  [bytes]   prompt utf-8
  [f32 LE]  duration_seconds
  [u32 LE]  seed

response (ok, status==0):
  [u32 LE]  status (0)
  [u32 LE]  T  (frames)
  [u32 LE]  D  (= 135)
  [bytes]   motion fp32 LE  (T * D * 4 bytes)
  [f32 LE]  hidden_ms
  [f32 LE]  refiner_ms
  [f32 LE]  dit_total_ms
  [f32 LE]  dit_step0_ms
  [f32 LE]  dit_step1_3_avg_ms
```

For `simpletool`, see comments at the top of `client/simpletool_client.py`.

---

## Motion format

The `(T, 135)` output is **SMPL-H rot6d**, not directly playable quaternions:

```
indices  meaning
[0:3]    root translation (xyz, meters, world space)
[3:9]    pelvis 6D rotation
[9:15]   spine_1 6D rotation
...      (22 SMPL-H joints × 6D rotation each = 132)
[129:135] right_hand 6D rotation
```

Frame rate is fixed at **30 fps**, so `T = duration * 30`.

To convert to standard skeleton/quaternion form: use
`./build/motion_to_vrma` (offline) or implement the Gram-Schmidt 6D → rotation
matrix → quaternion path yourself. The C++ reference is short and well-commented
in `cpp/motion_to_vrma.cpp`.

The motion_to_vrma tool also handles SMPL-H → VRM bone-name remapping (e.g.
`pelvis → hips`, `spine_1 → spine`, etc.), hemisphere alignment for quaternion
continuity, and Gaussian smoothing (σ=2.0 by default).

---

## Manual build (if `scripts/start_servers.sh` fails)

The repo ships with a precompiled `llama.cpp/fused_server` binary plus its
required dylibs under `llama.cpp/build_mac/bin/`. If that binary doesn't work
on your machine (different macOS major version, code-signing rejection, etc.),
rebuild from source.

### Build fused_server from source

```bash
cd llama.cpp

# Build llama.cpp itself first (produces the dylibs)
cmake -B build_mac -DGGML_METAL=ON -DBUILD_SHARED_LIBS=ON
cmake --build build_mac --config Release -j

# Then build our fused_server
clang++ -std=c++17 -O2 \
    -I include -I ggml/include -I .. \
    fused_server.cpp \
    -L build_mac/bin \
    -lllama -lggml -lggml-base -lggml-cpu \
    -lpthread \
    -Wl,-rpath,@loader_path/build_mac/bin \
    -o fused_server

cd ..
```

After this, `./scripts/start_servers.sh` should work.

### macOS Gatekeeper

If you get *"Apple cannot verify ... is free of malware"*, strip the
quarantine attribute from the vendored binaries:

```bash
xattr -dr com.apple.quarantine third_party/
xattr -dr com.apple.quarantine llama.cpp/build_mac/
```

This is the same as clicking "Open Anyway" in System Settings, but for all
files at once. The dylibs come from the official Microsoft and ggml-org GitHub
releases.

---

## FAQ

### Why is my first inference call slow?
The Metal backend in llama.cpp JIT-compiles GPU kernels on first use. Expect
the first 1-2 calls to spend ~50 ms extra in the `hidden` stage compiling
pipelines like `kernel_mul_mv_ext_q8_0_f32`. After that they're cached. The
bench script's warmup phase handles this automatically.

### Same prompt, same result?
With the same `seed`, yes — bit-exact reproducible across runs. With a
different seed you get a stylistically similar but slightly different motion.

### How long can a motion be?
Up to ~10 seconds, but DiT cost grows linearly with `T`. Sweet spot is 2-5
seconds. Beyond that, generate in chunks and blend at the application layer.

### Multi-character scenes?
The current servers handle requests **serially**. For 5-10 concurrent NPCs,
that's fine if each generation is sub-100 ms. For higher throughput you'd need
to add batched inference to both servers, which we haven't done yet.

### Non-English prompts?
Qwen3 is multilingual so prompts in Chinese/Japanese will tokenize fine, but
the t2m model was trained primarily on English motion descriptions. For best
quality, route foreign-language input through an LLM translation step first.

### Can I run this on Intel Mac?
The C++ code should compile (it's `-std=c++17` and pure stdlib). But the
vendored ORT dylib is arm64-only — replace `third_party/onnxruntime-osx-arm64-1.18.1/`
with the corresponding `onnxruntime-osx-x86_64` release from
[microsoft/onnxruntime](https://github.com/microsoft/onnxruntime/releases),
update the path in `CMakeLists.txt`, and rebuild. Untested, but should work.

### Why two endiannesses in the wire protocol?
Historical. The two services were developed independently and the choice of
byte order leaked into each one. Fixing it would break existing clients in
SimpleLove (the parent project). The Python SDK shields you from it.

---

## Related projects

- **[SimpleTool](https://arxiv.org/abs/2603.00030)** — parallel decoding for
  real-time LLM tool calling (the `simpletool` head on port 8422).
- **SimpleLove / NPC.exe** — application layer that consumes both services
  to drive VRM characters.

---

## License

TBD. Place a `LICENSE` file at the repo root before publishing.

---

## Acknowledgements

- [llama.cpp](https://github.com/ggerganov/llama.cpp) — LLM inference backend
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) — DiT inference
- [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) — base language model
- [three-vrm-animation](https://github.com/pixiv/three-vrm) — VRMA format reference