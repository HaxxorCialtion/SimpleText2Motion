cmake_minimum_required(VERSION 3.20)
project(SimpleT2M CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES arm64)
    set(CMAKE_OSX_DEPLOYMENT_TARGET 12.0)
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
if(MSVC)
    add_compile_options(/utf-8)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
endif()

# ============================================================
# ONNX Runtime (vendored, only needed by t2m_infer)
#
# Auto-discovers any onnxruntime-* directory under third_party/.
# Multiple packages can coexist (e.g. CPU 1.18 + GPU 1.26);
# selection rules below pick a sensible default, and the user
# can override with -DORT_DIR=<exact dir name>.
#
# Naming convention assumed (matches Microsoft's release tarballs):
#   onnxruntime-<os>-<arch>[-gpu]-<version>
#   e.g.  onnxruntime-linux-x64-1.18.1
#         onnxruntime-linux-x64-gpu-1.26.0
#         onnxruntime-osx-arm64-1.18.1
#         onnxruntime-win-x64-gpu-1.26.0
# ============================================================

# --- 1. Resolve platform tag used inside ORT package names. ---
# Runtime patterns = what must ship next to the exe / be installed.
# Primary lib      = what the linker needs (import lib on Windows).
if(APPLE)
    set(_ORT_PLATFORM_TAG     "osx-arm64")
    set(_ORT_RUNTIME_PATTERNS "libonnxruntime*.dylib")
    set(_ORT_PRIMARY_LIB      "libonnxruntime.dylib")
elseif(WIN32)
    set(_ORT_PLATFORM_TAG     "win-x64")
    set(_ORT_RUNTIME_PATTERNS "onnxruntime*.dll")   # runtime: only .dll
    set(_ORT_PRIMARY_LIB      "onnxruntime.lib")    # import lib, link-time only
elseif(UNIX)
    set(_ORT_PLATFORM_TAG     "linux-x64")
    set(_ORT_RUNTIME_PATTERNS "libonnxruntime*.so*")
    set(_ORT_PRIMARY_LIB      "libonnxruntime.so")
else()
    message(FATAL_ERROR "Unsupported platform.")
endif()

# --- 2. Discover candidate ORT packages. ---
set(_ORT_THIRDPARTY "${CMAKE_SOURCE_DIR}/third_party")
file(GLOB _ORT_CANDIDATES RELATIVE "${_ORT_THIRDPARTY}"
     "${_ORT_THIRDPARTY}/onnxruntime-${_ORT_PLATFORM_TAG}-*"
     "${_ORT_THIRDPARTY}/onnxruntime-${_ORT_PLATFORM_TAG}"
     "${_ORT_THIRDPARTY}/onnxruntime")

# Filter to actual directories with a real include/ inside (skip half-extracted ones).
set(_ORT_VALID "")
foreach(_cand ${_ORT_CANDIDATES})
    if(IS_DIRECTORY "${_ORT_THIRDPARTY}/${_cand}"
       AND EXISTS "${_ORT_THIRDPARTY}/${_cand}/include/onnxruntime_cxx_api.h")
        list(APPEND _ORT_VALID "${_cand}")
    endif()
endforeach()

# --- 3. Pick one: user override > GPU build > highest version. ---
# Default selection: prefer GPU variant, then lexicographically largest name
# (which for "x.y.z" version suffixes equals highest version).
if(ORT_DIR)
    get_filename_component(_ORT_PICK "${ORT_DIR}" NAME)
elseif(_ORT_VALID)
    # If a bare "onnxruntime" symlink/dir exists, it wins — user-curated default.
    list(FIND _ORT_VALID "onnxruntime" _ORT_BARE_IDX)
    if(NOT _ORT_BARE_IDX EQUAL -1)
        set(_ORT_PICK "onnxruntime")
    else()
        set(_ORT_GPU_HITS "")
        set(_ORT_CPU_HITS "")
        foreach(_cand ${_ORT_VALID})
            if(_cand MATCHES "-gpu-")
                list(APPEND _ORT_GPU_HITS "${_cand}")
            else()
                list(APPEND _ORT_CPU_HITS "${_cand}")
            endif()
        endforeach()
        list(SORT _ORT_GPU_HITS)
        list(SORT _ORT_CPU_HITS)
        if(_ORT_GPU_HITS)
            list(GET _ORT_GPU_HITS -1 _ORT_PICK)
        else()
            list(GET _ORT_CPU_HITS -1 _ORT_PICK)
        endif()
    endif()
else()
    message(FATAL_ERROR
        "No ONNX Runtime package found under ${_ORT_THIRDPARTY}.\n"
        "Expected a directory like onnxruntime-${_ORT_PLATFORM_TAG}[-gpu]-<version>\n"
        "containing include/onnxruntime_cxx_api.h.\n"
        "Extract a release tarball from https://github.com/microsoft/onnxruntime/releases")
endif()

set(ORT_ROOT "${_ORT_THIRDPARTY}/${_ORT_PICK}" CACHE PATH "Selected ONNX Runtime root")
set(ORT_LIB  "${ORT_ROOT}/lib/${_ORT_PRIMARY_LIB}")

if(NOT EXISTS "${ORT_ROOT}/include/onnxruntime_cxx_api.h")
    message(FATAL_ERROR "ORT not usable: missing ${ORT_ROOT}/include/onnxruntime_cxx_api.h")
endif()
if(NOT EXISTS "${ORT_LIB}")
    message(FATAL_ERROR "ORT not usable: missing ${ORT_LIB}")
endif()

# Resolve symlink (if any) to check the real package name for the -gpu- tag.
get_filename_component(_ORT_PICK_REAL
    "${_ORT_THIRDPARTY}/${_ORT_PICK}" REALPATH)
get_filename_component(_ORT_PICK_REAL_NAME "${_ORT_PICK_REAL}" NAME)

if(_ORT_PICK MATCHES "-gpu-" OR _ORT_PICK MATCHES "-gpu$"
   OR _ORT_PICK_REAL_NAME MATCHES "-gpu-" OR _ORT_PICK_REAL_NAME MATCHES "-gpu$")
    set(ORT_HAS_CUDA TRUE)
else()
    set(ORT_HAS_CUDA FALSE)
endif()

add_library(ort SHARED IMPORTED GLOBAL)
set_target_properties(ort PROPERTIES
    IMPORTED_LOCATION "${ORT_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${ORT_ROOT}/include")
if(WIN32)
    # On Windows the .lib is the import library; the runtime .dll lives in lib/ too.
    set_target_properties(ort PROPERTIES
        IMPORTED_IMPLIB "${ORT_LIB}"
        IMPORTED_LOCATION "${ORT_ROOT}/lib/onnxruntime.dll")
endif()

# Collect every runtime artifact in the ORT lib dir for install + (on Windows)
# post-build copy. Glob each pattern separately and accumulate — file(GLOB) with
# a multi-element pattern variable is fragile across CMake versions. After the
# split above each platform has a single runtime pattern, but the loop keeps it
# robust if a platform ever needs more than one.
set(_ORT_RUNTIME_FILES "")
foreach(_pat ${_ORT_RUNTIME_PATTERNS})
    file(GLOB _hits "${ORT_ROOT}/lib/${_pat}")
    list(APPEND _ORT_RUNTIME_FILES ${_hits})
endforeach()

# ============================================================
# zlib
#   Windows: build from vendored source (no system zlib, and
#            avoids needing zlib1.dll next to the exe).
#   macOS/Linux: use system zlib as before.
# ============================================================
if(WIN32)
    set(ZLIB_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(SKIP_INSTALL_ALL ON CACHE BOOL "" FORCE)
    add_subdirectory(third_party/zlib EXCLUDE_FROM_ALL)

    # zlibstatic doesn't export an include dir; zconf.h is generated
    # into the build dir. Wrap both into one INTERFACE target.
    add_library(zlib_local INTERFACE)
    target_link_libraries(zlib_local INTERFACE zlibstatic)
    target_include_directories(zlib_local INTERFACE
        "${CMAKE_SOURCE_DIR}/third_party/zlib"
        "${CMAKE_BINARY_DIR}/third_party/zlib")
else()
    find_package(ZLIB REQUIRED)
    add_library(zlib_local INTERFACE)
    target_link_libraries(zlib_local INTERFACE ZLIB::ZLIB)
endif()

# ============================================================
# toml++ (vendored, header-only)
# Single-header at third_party/tomlplusplus/include/toml++/toml.hpp
# Used by fused_server / t2m_infer / bench_t2m for config.toml parsing.
# ============================================================
set(_TOMLPP_INC "${CMAKE_SOURCE_DIR}/third_party/tomlplusplus/include")
if(NOT EXISTS "${_TOMLPP_INC}/toml++/toml.hpp")
    message(FATAL_ERROR
        "missing ${_TOMLPP_INC}/toml++/toml.hpp\n"
        "Get it from https://github.com/marzer/tomlplusplus (header-only release):\n"
        "  mkdir -p third_party/tomlplusplus/include/toml++\n"
        "  curl -L -o third_party/tomlplusplus/include/toml++/toml.hpp \\\n"
        "       https://raw.githubusercontent.com/marzer/tomlplusplus/master/toml.hpp")
endif()
add_library(tomlpp INTERFACE)
target_include_directories(tomlpp INTERFACE "${_TOMLPP_INC}")
target_compile_features(tomlpp INTERFACE cxx_std_17)

# ============================================================
# cpp-httplib (vendored, header-only)
# Used only by run_server. Single ~9000-line header.
# ============================================================
set(_HTTPLIB_INC "${CMAKE_SOURCE_DIR}/third_party/cpp-httplib")
if(NOT EXISTS "${_HTTPLIB_INC}/httplib.h")
    message(FATAL_ERROR
        "missing ${_HTTPLIB_INC}/httplib.h\n"
        "Get it from https://github.com/yhirose/cpp-httplib:\n"
        "  mkdir -p third_party/cpp-httplib\n"
        "  curl -L -o third_party/cpp-httplib/httplib.h \\\n"
        "       https://raw.githubusercontent.com/yhirose/cpp-httplib/master/httplib.h")
endif()
add_library(httplib INTERFACE)
target_include_directories(httplib INTERFACE "${_HTTPLIB_INC}")
target_compile_features(httplib INTERFACE cxx_std_17)

# ============================================================
# Target: t2m_infer  (server + one-shot inference)
# ============================================================
add_executable(t2m_infer cpp/t2m_infer.cpp)
target_include_directories(t2m_infer PRIVATE cpp)
target_link_libraries(t2m_infer PRIVATE ort zlib_local tomlpp)

if(ORT_HAS_CUDA)
    target_compile_definitions(t2m_infer PRIVATE ORT_USE_CUDA=1)
endif()

if(APPLE)
    set_target_properties(t2m_infer PROPERTIES
        BUILD_RPATH   "${ORT_ROOT}/lib"
        INSTALL_RPATH "@loader_path")
elseif(WIN32)
    # Windows has no rpath. Copy ORT runtime DLLs next to the exe so it just runs.
    if(_ORT_RUNTIME_FILES)
        add_custom_command(TARGET t2m_infer POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${_ORT_RUNTIME_FILES} $<TARGET_FILE_DIR:t2m_infer>)
    else()
        message(WARNING "No ORT runtime files found to copy next to t2m_infer.exe")
    endif()
elseif(UNIX)
    # $ORIGIN is the Linux analogue of @loader_path. Quote it so CMake passes
    # the literal token to the linker rather than expanding it.
    # Note: cuDNN/cuBLAS are system libs, resolved via ldconfig — not bundled.
    set_target_properties(t2m_infer PROPERTIES
        BUILD_RPATH   "${ORT_ROOT}/lib"
        INSTALL_RPATH "$ORIGIN")
endif()

# ============================================================
# Target: motion_to_vrma  (motion.bin -> .vrma offline tool)
# Pure stdlib C++17, zero deps. No ORT, no zlib.
# ============================================================
add_executable(motion_to_vrma cpp/motion_to_vrma.cpp)
target_include_directories(motion_to_vrma PRIVATE cpp)

# ============================================================
# Target: bench_t2m  (benchmark client for t2m_infer --server)
# Pure stdlib C++17 + platform.h, no ORT/llama deps. Reads config.toml.
# ============================================================
add_executable(bench_t2m cpp/bench_t2m.cpp)
target_include_directories(bench_t2m PRIVATE cpp)
target_link_libraries(bench_t2m PRIVATE tomlpp)
if(WIN32)
    target_link_libraries(bench_t2m PRIVATE Ws2_32)
endif()

# ============================================================
# Target: run_server (HTTP demo server, replaces run.py)
# Pure stdlib C++17 + cpp-httplib + toml++. Bridges browser <-> t2m_infer.
# ============================================================
add_executable(run_server cpp/run_server.cpp)
target_include_directories(run_server PRIVATE cpp)
target_link_libraries(run_server PRIVATE httplib tomlpp)
if(WIN32)
    target_link_libraries(run_server PRIVATE Ws2_32 Shell32)
else()
    find_package(Threads REQUIRED)
    target_link_libraries(run_server PRIVATE Threads::Threads)
endif()

# ============================================================
# Target: fused_server  (WINDOWS ONLY)
#
# Links the prebuilt CPU llama.cpp shared libs. On macOS/Linux this
# target is intentionally NOT defined — fused_server is built there by
# its own path (outside this CMakeLists), so those platforms see no
# change whatsoever. Everything below lives strictly inside if(WIN32).
#
# Prerequisite: llama.cpp must already be built (see build notes):
#   cd llama.cpp
#   cmake -B build -DGGML_CUDA=OFF -DBUILD_SHARED_LIBS=ON
#   cmake --build build --config Release -j
# which produces (paths are MSVC multi-config layout):
#   import libs : llama.cpp/build/src/Release/llama.lib
#                 llama.cpp/build/ggml/src/Release/ggml.lib
#   runtime dll : llama.cpp/build/bin/Release/*.dll
#                 (llama.dll ggml.dll ggml-base.dll ggml-cpu.dll ...)
# ============================================================
if(WIN32)
    set(_LLAMA_ROOT    "${CMAKE_SOURCE_DIR}/llama.cpp")
    set(_LLAMA_BUILD   "${_LLAMA_ROOT}/build")
    set(_LLAMA_LIB     "${_LLAMA_BUILD}/src/Release/llama.lib")
    set(_GGML_LIB      "${_LLAMA_BUILD}/ggml/src/Release/ggml.lib")
    set(_LLAMA_DLL_DIR "${_LLAMA_BUILD}/bin/Release")

    # Fail early with a clear message if llama.cpp wasn't built first,
    # rather than emitting a cryptic linker error later.
    if(NOT EXISTS "${_LLAMA_LIB}")
        message(FATAL_ERROR
            "fused_server: missing ${_LLAMA_LIB}\n"
            "Build llama.cpp first:\n"
            "  cd llama.cpp\n"
            "  cmake -B build -DGGML_CUDA=OFF -DBUILD_SHARED_LIBS=ON\n"
            "  cmake --build build --config Release -j")
    endif()
    if(NOT EXISTS "${_GGML_LIB}")
        message(FATAL_ERROR "fused_server: missing ${_GGML_LIB} (build llama.cpp first)")
    endif()

    add_executable(fused_server llama.cpp/fused_server.cpp)

    # Mirrors the bare `cl` recipe: /I include /I ggml\include /I ..\cpp
    target_include_directories(fused_server PRIVATE
        "${_LLAMA_ROOT}/include"
        "${_LLAMA_ROOT}/ggml/include"
        cpp)

    # Mirrors `cl /link build\src\Release\llama.lib build\ggml\src\Release\ggml.lib`.
    # Ws2_32 is pulled in automatically by a #pragma comment(lib,...) in the source.
    target_link_libraries(fused_server PRIVATE
        "${_LLAMA_LIB}"
        "${_GGML_LIB}"
        tomlpp)

    # Windows has no rpath. Copy the llama.cpp runtime DLLs next to the exe so
    # it just runs — same treatment t2m_infer gets for the ORT DLLs above.
    # This replaces the manual `Copy-Item build\bin\Release\*.dll .` step.
    file(GLOB _LLAMA_RUNTIME_FILES "${_LLAMA_DLL_DIR}/*.dll")
    if(_LLAMA_RUNTIME_FILES)
        add_custom_command(TARGET fused_server POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${_LLAMA_RUNTIME_FILES} $<TARGET_FILE_DIR:fused_server>)
    else()
        message(WARNING
            "No llama.cpp runtime DLLs found in ${_LLAMA_DLL_DIR} to copy next "
            "to fused_server.exe — it will fail to start with 0xC0000135.")
    endif()
endif()

# ============================================================
# Install (used by `cmake --install build --prefix dist`)
# ============================================================
install(TARGETS t2m_infer motion_to_vrma bench_t2m run_server
        RUNTIME DESTINATION bin)
# Install every ORT runtime file we discovered — works for CPU packages
# (one .so/.dylib) and GPU packages (plus providers_cuda etc.) without
# hardcoding any version number. The split above keeps link-only import
# libs (.lib on Windows) out of the distribution.
install(FILES ${_ORT_RUNTIME_FILES} DESTINATION bin)

# Windows-only: install fused_server and its llama.cpp runtime DLLs alongside
# the rest, so `dist/bin` is a self-contained drop for non-developer users.
if(WIN32)
    install(TARGETS fused_server RUNTIME DESTINATION bin)
    if(_LLAMA_RUNTIME_FILES)
        install(FILES ${_LLAMA_RUNTIME_FILES} DESTINATION bin)
    endif()
endif()

# ============================================================
# Summary
# ============================================================
message(STATUS "")
message(STATUS "SimpleT2M configuration:")
message(STATUS "  CXX:           ${CMAKE_CXX_COMPILER}")
message(STATUS "  Build type:    ${CMAKE_BUILD_TYPE}")
if(APPLE)
    message(STATUS "  Platform:      macOS (${CMAKE_OSX_ARCHITECTURES})")
    message(STATUS "  Deploy target: ${CMAKE_OSX_DEPLOYMENT_TARGET}")
elseif(WIN32)
    message(STATUS "  Platform:      Windows x86_64")
else()
    message(STATUS "  Platform:      Linux x86_64")
endif()
message(STATUS "  ORT package:   ${_ORT_PICK}")
message(STATUS "  ORT root:      ${ORT_ROOT}")
message(STATUS "  ORT lib:       ${ORT_LIB}")
message(STATUS "  ORT CUDA EP:   ${ORT_HAS_CUDA}")
if(_ORT_VALID)
    list(LENGTH _ORT_VALID _ORT_N)
    if(_ORT_N GREATER 1)
        message(STATUS "  (other candidates available: ${_ORT_VALID} — override with -DORT_DIR=<name>)")
    endif()
endif()
message(STATUS "  zlib:          ${ZLIB_VERSION_STRING}")
message(STATUS "  toml++:        ${_TOMLPP_INC}/toml++/toml.hpp")
message(STATUS "  cpp-httplib:   ${_HTTPLIB_INC}/httplib.h")
if(WIN32)
    message(STATUS "  Targets:       t2m_infer, motion_to_vrma, bench_t2m, run_server, fused_server")
    message(STATUS "  llama libs:    ${_LLAMA_LIB}")
    message(STATUS "                 ${_GGML_LIB}")
    message(STATUS "  llama dll dir: ${_LLAMA_DLL_DIR}")
else()
    message(STATUS "  Targets:       t2m_infer, motion_to_vrma, bench_t2m, run_server")
endif()
message(STATUS "")