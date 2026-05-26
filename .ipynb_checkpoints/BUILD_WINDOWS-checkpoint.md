# ============ Windows (PowerShell) 参考流程 ============
# 前置：装好 Visual Studio 2022（含 "Desktop development with C++"）、Git、CMake
#       GPU 版还需 CUDA Toolkit 12.x + cuDNN 9（装在系统里，加进 PATH）

# 1. Clone
git clone https://github.com/HaxxorCialtion/SimpleText2Motion.git
cd SimpleText2Motion

# 2. 下载模型（PowerShell 没有 wget，用 curl.exe；Win10+ 自带）
mkdir models\SimpleT2M
$BASE = "https://huggingface.co/Cialtion/SimpleLove/resolve/main"
curl.exe -L -o models\qwen3-4b-v2-trim6-q4_k_m.gguf $BASE/qwen3-4b-v2-trim6-q4_k_m.gguf
curl.exe -L -o models\SimpleT2M\birefiner.onnx     $BASE/SimpleT2M/birefiner.onnx
curl.exe -L -o models\SimpleT2M\config.json        $BASE/SimpleT2M/config.json
curl.exe -L -o models\SimpleT2M\dit_step.onnx      $BASE/SimpleT2M/dit_step.onnx
curl.exe -L -o models\SimpleT2M\small_weights.bin  $BASE/SimpleT2M/small_weights.bin
curl.exe -L -o models\SimpleT2M\small_weights.npz  $BASE/SimpleT2M/small_weights.npz

# 3. 第三方依赖
mkdir third_party
cd third_party
git clone --depth 1 https://github.com/yhirose/cpp-httplib.git
git clone --depth 1 https://github.com/marzer/tomlplusplus.git

# onnxruntime（Windows 是 .zip，区分 CPU/GPU）
$ORT_VER = "1.20.1"
# 选项 1：NVIDIA GPU + CUDA 12.x（推荐）
$ORT_PKG = "onnxruntime-win-x64-gpu-$ORT_VER"
# 选项 2：纯 CPU 兜底
# $ORT_PKG = "onnxruntime-win-x64-$ORT_VER"
curl.exe -L -o "$ORT_PKG.zip" "https://github.com/microsoft/onnxruntime/releases/download/v$ORT_VER/$ORT_PKG.zip"
Expand-Archive "$ORT_PKG.zip" -DestinationPath .
Remove-Item "$ORT_PKG.zip"
# Windows 没有软链接习惯，直接重命名成 onnxruntime
Rename-Item $ORT_PKG onnxruntime
cd ..

# 4. 编译 llama.cpp（CUDA）
cd llama.cpp
if (Test-Path build) { Remove-Item build -Recurse -Force }
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native -DBUILD_SHARED_LIBS=ON
cmake --build build --config Release -j
cd ..

# 5. 编译 fused_server —— 用 CMake，别手敲 cl.exe
#    （前提：CMakeLists 里有 fused_server 这个 target，且引用 third_party/onnxruntime）
#    如果项目还没把 fused_server 纳入 CMake，需要先加（见下方说明）

# 6. 编译其余模块
if (Test-Path build) { Remove-Item build -Recurse -Force }
cmake -B build
cmake --build build --config Release -j

# 7. 关键：把依赖 DLL 拷到 exe 同目录（Windows 没有 rpath）
#    onnxruntime.dll、onnxruntime_providers_cuda.dll、llama 的 dll 等
copy third_party\onnxruntime\lib\*.dll build\Release\
copy llama.cpp\build\bin\Release\*.dll build\Release\

# 8. 运行
.\build\Release\run_server.exe