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
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# ============================================================
# ONNX Runtime (vendored, only needed by t2m_infer)
# Platform-selected: osx-arm64 on macOS, linux-x64 on Linux.
# CPU build on both — the DiT is launch-bound; CPU EP is the
# intended execution provider (see README architecture notes).
# ============================================================
if(APPLE)
    set(ORT_DIRNAME "onnxruntime-osx-arm64-1.18.1")
    set(ORT_SHARED_EXT "dylib")
    # macOS: libonnxruntime.dylib  +  libonnxruntime.1.18.1.dylib
    set(ORT_SONAME_FULL "libonnxruntime.1.18.1.dylib")
elseif(UNIX)
    set(ORT_DIRNAME "onnxruntime-linux-x64-1.18.1")
    set(ORT_SHARED_EXT "so")
    # Linux: libonnxruntime.so  +  libonnxruntime.so.1.18.1
    set(ORT_SONAME_FULL "libonnxruntime.so.1.18.1")
else()
    message(FATAL_ERROR "Unsupported platform (only macOS arm64 / Linux x64).")
endif()

set(ORT_ROOT ${CMAKE_SOURCE_DIR}/third_party/${ORT_DIRNAME}
    CACHE PATH "Path to vendored onnxruntime")

set(ORT_LIB "${ORT_ROOT}/lib/libonnxruntime.${ORT_SHARED_EXT}")

if(NOT EXISTS "${ORT_ROOT}/include/onnxruntime_cxx_api.h")
    message(FATAL_ERROR
        "ORT not found at ${ORT_ROOT}. "
        "Expected onnxruntime_cxx_api.h in include/. "
        "(Did you extract third_party/${ORT_DIRNAME}.tgz ?)")
endif()
if(NOT EXISTS "${ORT_LIB}")
    message(FATAL_ERROR "ORT shared lib not found at ${ORT_LIB}")
endif()

add_library(ort SHARED IMPORTED GLOBAL)
set_target_properties(ort PROPERTIES
    IMPORTED_LOCATION "${ORT_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${ORT_ROOT}/include")

# ============================================================
# zlib (system)
# ============================================================
find_package(ZLIB REQUIRED)

# ============================================================
# Target: t2m_infer  (server + one-shot inference)
# ============================================================
add_executable(t2m_infer cpp/t2m_infer.cpp)
target_include_directories(t2m_infer PRIVATE cpp)
target_link_libraries(t2m_infer PRIVATE ort ZLIB::ZLIB)

if(APPLE)
    set_target_properties(t2m_infer PROPERTIES
        BUILD_RPATH   "${ORT_ROOT}/lib"
        INSTALL_RPATH "@loader_path")
elseif(UNIX)
    # $ORIGIN is the Linux analogue of @loader_path. Quote it so CMake
    # passes the literal token to the linker rather than expanding it.
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
# Install (used by `cmake --install build --prefix dist`)
# ============================================================
install(TARGETS t2m_infer motion_to_vrma
        RUNTIME DESTINATION bin)
install(FILES "${ORT_ROOT}/lib/${ORT_SONAME_FULL}"
        DESTINATION bin)
install(FILES "${ORT_LIB}"
        DESTINATION bin)

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
else()
    message(STATUS "  Platform:      Linux x86_64")
endif()
message(STATUS "  ORT root:      ${ORT_ROOT}")
message(STATUS "  ORT lib:       ${ORT_LIB}")
message(STATUS "  zlib:          ${ZLIB_VERSION_STRING}")
message(STATUS "  Targets:       t2m_infer, motion_to_vrma")
message(STATUS "")
