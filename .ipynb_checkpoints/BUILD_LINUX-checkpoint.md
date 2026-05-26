# Build SimpleText2Motion on Linux (NVIDIA GPU)

> 实测环境：H100 80GB + CUDA 12.9 + cuDNN 9.10.2，onnxruntime 1.20.1。
> 全程无需 root（除非系统缺 cuDNN 9，见末尾排错）。

## 1. Clone

```bash
git clone https://github.com/HaxxorCialtion/SimpleText2Motion.git
cd SimpleText2Motion
```

## 2. 下载模型权重

```bash
mkdir -p models/SimpleT2M
BASE=https://huggingface.co/Cialtion/SimpleLove/resolve/main

wget -q -P ./models           $BASE/qwen3-4b-v2-trim6-q4_k_m.gguf &
wget -q -P ./models/SimpleT2M $BASE/SimpleT2M/birefiner.onnx &
wget -q -P ./models/SimpleT2M $BASE/SimpleT2M/config.json &
wget -q -P ./models/SimpleT2M $BASE/SimpleT2M/dit_step.onnx &
wget -q -P ./models/SimpleT2M $BASE/SimpleT2M/small_weights.bin &
wget -q -P ./models/SimpleT2M $BASE/SimpleT2M/small_weights.npz &
wait
echo "全部下载完成"
```

## 3. 下载第三方依赖

```bash
mkdir -p third_party
cd third_party

# --- header-only 库 ---
git clone --depth 1 https://github.com/yhirose/cpp-httplib.git
git clone --depth 1 https://github.com/marzer/tomlplusplus.git

# --- onnxruntime：按你的硬件选一个，取消对应行的注释 ---
ORT_VER=1.20.1
# 选项 1：NVIDIA GPU + CUDA 12.x（推荐：H100/A100/40系/30系等较新卡）
ORT_PKG=onnxruntime-linux-x64-gpu-${ORT_VER}
# 选项 2：NVIDIA GPU + CUDA 11.x（较老工具链；如需更稳可用 1.18.1：onnxruntime-linux-x64-gpu-1.18.1）
# ORT_PKG=onnxruntime-linux-x64-gpu-1.18.1
# 选项 3：纯 CPU（无 N 卡 / GPU 跑不通时的兜底，一定能跑，只是慢）
# ORT_PKG=onnxruntime-linux-x64-${ORT_VER}

curl -L -O https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VER}/${ORT_PKG}.tgz
ls -l ${ORT_PKG}.tgz   # 检查：应为上百 MB；若只有几字节说明 404，请核对文件名
tar -xzf ${ORT_PKG}.tgz && rm ${ORT_PKG}.tgz
ln -sfn ${ORT_PKG} onnxruntime   # 统一软链接，后续编译只引用 ./third_party/onnxruntime
cd ..
```

> 不确定 CUDA 版本：`nvcc --version` 或 `nvidia-smi` 看一眼。
> 注意：从 ort 1.19 起 CUDA 12 是默认，包名**不带** `cuda12` 后缀；
> 带 `cuda12` 后缀的命名只在 1.18.x 时代用过。

## 4. 编译 llama.cpp（CUDA）

```bash
cd llama.cpp
rm -rf build
cmake -B build -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=native \
    -DBUILD_SHARED_LIBS=ON
cmake --build build --config Release -j
```

> `native` 自动探测当前机器 GPU 架构。**仅本机用**；
> 若要打包分发给不同显卡的用户，改成显式多架构，例如
> `-DCMAKE_CUDA_ARCHITECTURES="80;86;89;90"`。

## 5. 编译 fused_server

```bash
# 仍在 llama.cpp 目录下执行
g++ -std=c++17 -O2 \
    -I include -I ggml/include -I .. \
    -I ../cpp \
    -I ../third_party/tomlplusplus/include \
    -I ../third_party/onnxruntime/include \
    fused_server.cpp \
    -L build/bin \
    -lllama -lggml -lggml-base -lggml-cpu \
    -L ../third_party/onnxruntime/lib \
    -lonnxruntime \
    -lpthread \
    -Wl,-rpath,'$ORIGIN/build/bin' \
    -Wl,-rpath,'$ORIGIN/../third_party/onnxruntime/lib' \
    -o fused_server
cd ..
```

> 全部用 `third_party/onnxruntime` 软链接名，不写死版本目录，
> 这样换 ort 版本时本步无需改动。

## 6. 编译其余模块

```bash
rm -rf build
cmake -B build
cmake --build build -j
```

## 7. 运行

```bash
bash ./scripts/start_servers.sh         # 终端 A：fused_server + t2m_infer
./build/bench_t2m                       # 终端 B：benchmark
./build/run_server                      # 终端 B：demo（浏览器打开 :8765）
```

启动后查看日志确认 ONNX 是否走 GPU：

```
[server] EP in use: cuda          ← 这样才是 GPU 生效
```

---

## 排错

### GPU 模式下 t2m_infer 启动超时（首次 warmup 慢）

CPU 模式 warmup 只要几秒，但 **GPU 首次启动**要做 cuDNN 算法搜索 + kernel 编译，
H100 上可能需要 30~60 秒，会触发 `scripts/start_servers.sh` 的等待超时：

```
[start_servers] ✗ t2m_infer failed to open port 8423 within 30s
```

**解决**：把脚本里 `wait_for_port ... 30` 的超时调大（实测 60 足够）：

```bash
# 找到那一行
grep -n 'wait_for_port "$T2M_PORT"' scripts/start_servers.sh
# 改超时（例：30 -> 90，留足余量）
sed -i 's/\(wait_for_port "\$T2M_PORT" "t2m_infer"\) 30/\1 90/' scripts/start_servers.sh
```

首次 warmup 后 cuDNN 会缓存搜索结果，后续启动会快很多。

### 日志报 `libcublasLt.so.11` / `CUDA EP unavailable ... falling back to CPU`

ort 包与本机 CUDA 大版本不匹配（`.so.11` = 包要 CUDA 11，但你装的是 CUDA 12）。
按第 3 步换成 CUDA 12 的 ort 包（选项 1），并重编 fused_server / 其余模块。

### 确认 cuDNN

ort 1.20.x 需要 **cuDNN 9**。检查：

```bash
find / -name 'libcudnn.so.9*' 2>/dev/null
```

无输出说明缺 cuDNN 9，可在 conda/pip 环境免 root 补齐：

```bash
pip install nvidia-cudnn-cu12
python -c "import nvidia.cudnn, os; print(os.path.dirname(nvidia.cudnn.__file__)+'/lib')"
# 把打印出的路径加进 LD_LIBRARY_PATH 后重启服务
```