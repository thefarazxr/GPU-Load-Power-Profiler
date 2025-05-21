cmake_minimum_required(VERSION 3.20)
project(gpu_profiler LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# ── CUDA ──────────────────────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)

# Override via -DCUDA_ARCH=<sm>:
#   89  = Ada Lovelace  (RTX 4000 series)
#   100 = Blackwell     (RTX 5000 series, requires CUDA 12.8+)
if(NOT DEFINED CUDA_ARCH)
    set(CUDA_ARCH 89)
endif()
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})

# ── Vulkan ────────────────────────────────────────────────────────────────────
find_package(Vulkan REQUIRED)

# ── NVML ──────────────────────────────────────────────────────────────────────
# NVML ships with the CUDA toolkit; fall back to system path if needed
find_library(NVML_LIB nvidia-ml
    HINTS ${CUDAToolkit_LIBRARY_DIR} /usr/lib/x86_64-linux-gnu
    REQUIRED)

# ── Sources ───────────────────────────────────────────────────────────────────
add_executable(gpu_profiler
    src/main.cpp
    src/config.cpp
    src/scheduler.cpp
    telemetry/nvml_monitor.cpp
    telemetry/logger.cpp
    vulkan/vulkan_runner.cpp
    cuda/heavy_hash.cu
)

target_include_directories(gpu_profiler PRIVATE
    include
    ${CUDAToolkit_INCLUDE_DIRS}
    ${Vulkan_INCLUDE_DIRS}
)

target_link_libraries(gpu_profiler PRIVATE
    CUDA::cudart
    ${NVML_LIB}
    Vulkan::Vulkan
    pthread
)

# Aggressive CUDA optimisation flags
target_compile_options(gpu_profiler PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -O3
        --expt-relaxed-constexpr
    >
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

# ── SPIR-V compilation (requires glslc from Vulkan SDK) ───────────────────────
find_program(GLSLC glslc HINTS $ENV{VULKAN_SDK}/bin)
if(GLSLC)
    add_custom_command(
        OUTPUT  ${CMAKE_SOURCE_DIR}/vulkan/sha256.spv
        COMMAND ${GLSLC} -o ${CMAKE_SOURCE_DIR}/vulkan/sha256.spv
                         ${CMAKE_SOURCE_DIR}/vulkan/sha256.comp
        DEPENDS ${CMAKE_SOURCE_DIR}/vulkan/sha256.comp
        COMMENT "Compiling sha256.comp -> sha256.spv"
    )
    add_custom_target(compile_shaders ALL
        DEPENDS ${CMAKE_SOURCE_DIR}/vulkan/sha256.spv)
    add_dependencies(gpu_profiler compile_shaders)
else()
    message(WARNING "glslc not found — compile vulkan/sha256.comp manually:\n"
                    "  glslc -o vulkan/sha256.spv vulkan/sha256.comp")
endif()
