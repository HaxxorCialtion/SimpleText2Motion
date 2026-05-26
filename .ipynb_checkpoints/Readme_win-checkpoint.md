# 创建目录(-Force 让已存在的目录不报错,相当于 mkdir -p)
New-Item -ItemType Directory -Force -Path "third_party\tomlplusplus\include\toml++" | Out-Null
New-Item -ItemType Directory -Force -Path "third_party\cpp-httplib" | Out-Null

# 下载文件
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/marzer/tomlplusplus/master/toml.hpp" `
                  -OutFile "third_party\tomlplusplus\include\toml++\toml.hpp"

Invoke-WebRequest -Uri "https://raw.githubusercontent.com/yhirose/cpp-httplib/master/httplib.h" `
                  -OutFile "third_party\cpp-httplib\httplib.h"

# llama.cpp                  
cmake -B build -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=86 \
    -DBUILD_SHARED_LIBS=ON
cmake --build build --config Release -j
Pick the arch for your GPU: 75=T4, 80=A100, 86=A10/RTX30, 89=RTX40/L40, 90=H100. Portable multi-arch: `"75-real;80-real;86-real;89-real;90-real;90-virtual"`.

# others
cmake -B build "-DORT_DIR=onnxruntime-win-x64-gpu-1.26.0"                                    
>> cmake --build build --config Release -j      

# start
```bash
./scripts/start_servers.ps1             # terminal A: fused_server + t2m_infer
.\build\Release\bench_t2m.exe                      # terminal B: benchmark
.\build\Release\run_server.exe                    # terminal B: demo (opens browser to :8765)