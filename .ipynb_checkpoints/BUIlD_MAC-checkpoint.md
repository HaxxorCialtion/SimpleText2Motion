# 1 
git clone https://github.com/HaxxorCialtion/SimpleText2Motion.git
cd  SimpleText2Motion
# 2 download models
mkdir -p  models
mkdir -p ./models/SimpleT2M
mkdir -p  third_party

BASE=https://huggingface.co/Cialtion/SimpleLove/resolve/main
wget -q -P ./models $BASE/qwen3-4b-v2-trim6-q4_k_m.gguf &
wget -q -P ./models/SimpleT2M $BASE/SimpleT2M/birefiner.onnx &
wget -q -P ./models/SimpleT2M $BASE/SimpleT2M/config.json &
wget -q -P ./models/SimpleT2M $BASE/SimpleT2M/dit_step.onnx &
wget -q -P ./models/SimpleT2M $BASE/SimpleT2M/small_weights.bin &
wget -q -P ./models/SimpleT2M $BASE/SimpleT2M/small_weights.npz &
wait
echo "全部下载完成"

# 3. download third_party
mkdir -p ./third_party
cd ./third_party

# --- cpp-httplib (header-only，单头文件) ---
git clone --depth 1 https://github.com/yhirose/cpp-httplib.git

# --- tomlplusplus (header-only) ---
git clone --depth 1 https://github.com/marzer/tomlplusplus.git

# --- onnxruntime (macOS arm64，无 CUDA 烦恼) ---
ORT_VER=1.18.1
ORT_PKG=onnxruntime-osx-arm64-${ORT_VER}
curl -L -O https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VER}/${ORT_PKG}.tgz
ls -l ${ORT_PKG}.tgz                  # 检查：上百 MB 才对，几字节是 404
tar -xzf ${ORT_PKG}.tgz && rm ${ORT_PKG}.tgz
ln -sfn ${ORT_PKG} onnxruntime        # 统一软链接，编译只引用 ./third_party/onnxruntime
cd ..

# 4. build llama.cpp
cd llama.cpp
rm -rf build
cmake -B build -DBUILD_SHARED_LIBS=ON
cmake --build build --config Release -j

# 5. Build fused_server
g++ -std=c++17 -O2 \
    -I include -I ggml/include -I .. \
    -I ../cpp \
    -I ../third_party/tomlplusplus/include \
    -I ../third_party/onnxruntime-osx-arm64-1.18.1/include \
    fused_server.cpp \
    -L build/bin \
    -lllama -lggml -lggml-base -lggml-cpu \
    -L ../third_party/onnxruntime-osx-arm64-1.18.1/lib \
    -lonnxruntime \
    -lpthread \
    -Wl,-rpath,'@loader_path/build/bin' \
    -Wl,-rpath,'@loader_path/../third_party/onnxruntime/lib' \
    -o fused_server
cd ..

# 6. Build everything else
rm -rf build
cmake -B build
cmake --build build -j

# 7. Run

```bash
bash ./scripts/start_servers.sh         # terminal A: fused_server + t2m_infer
./build/bench_t2m                       # terminal B: benchmark
./build/run_server                      # terminal B: demo (opens browser to :8765)
```