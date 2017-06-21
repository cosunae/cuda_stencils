project(cuda_stencils)
cmake_minimum_required(VERSION 3.0)
set (CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories ( "${CMAKE_CURRENT_SOURCE_DIR}/libjson" )
add_definitions(-DJSON_ISO_STRICT)
add_subdirectory( libjson )

find_package(CUDA REQUIRED)
set(CUDA_PROPAGATE_HOST_FLAGS ON)
set( CUDA_ARCH "sm_60" CACHE STRING "Compute capability for CUDA" )
set( ENABLE_GPU "ON" CACHE BOOL "Enable GPU compilation" )

if( ENABLE_GPU )
  add_definitions(-DENABLE_GPU)
endif()

# adding the additional nvcc flags
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "-arch=${CUDA_ARCH}" "-Xcudafe" "--diag_suppress=dupl_calling_convention")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "-Xcudafe" "--diag_suppress=code_is_unreachable" "-Xcudafe")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "--diag_suppress=implicit_return_from_non_void_function" "-Xcudafe")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "--diag_suppress=calling_convention_not_allowed" "-Xcudafe")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "--diag_suppress=conflicting_calling_conventions")

set(HEADERS stencil_kernels.h tools.h defs.h)
cuda_add_executable (stencil_bench main.cu ${HEADERS})
target_link_libraries(stencil_bench json)

add_subdirectory( unstructured )

