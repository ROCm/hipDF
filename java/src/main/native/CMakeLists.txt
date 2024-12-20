# =============================================================================
# Copyright (c) 2019-2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================
# MIT License
#
# Modifications Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================
cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

# NOTE(HIP/AMD): make *.cu files HIP lang files
set(CMAKE_USER_MAKE_RULES_OVERRIDE "${CMAKE_CURRENT_LIST_DIR}/../../../../overrides.cmake")

include(../../../../fetch_rapids.cmake)
include(rapids-cmake)
include(rapids-hip)
include(rapids-find)
include(rapids-cpm)
rapids_cpm_init()

# Use GPU_ARCHS if it is defined
if(DEFINED GPU_ARCHS)
  set(CMAKE_HIP_ARCHITECTURES "${GPU_ARCHS}")
endif()
rapids_hip_init_architectures(CUDF_JNI)

project(
  CUDF_JNI
  VERSION 23.10.00
  LANGUAGES C CXX HIP
)

# ##################################################################################################
# * build options ---------------------------------------------------------------------------------

option(USE_NVTX "Build with NVTX support" OFF)  # TODO(HIP/AMD): this option is currently not supported
option(BUILD_SHARED_LIBS "Build cuDF JNI shared libraries" ON)
option(BUILD_TESTS "Configure CMake to build tests" ON)
option(CUDF_USE_PER_THREAD_DEFAULT_STREAM "Build with per-thread default stream" OFF)
option(CUDA_STATIC_RUNTIME "Statically link the CUDA runtime" OFF)
option(USE_GDS "Build with GPUDirect Storage (GDS)/cuFile support" OFF)
option(CUDF_JNI_LIBCUDF_STATIC "Link with libcudf.a" OFF)
option(CUDF_JNI_ENABLE_PROFILING "Build with profiling support" OFF)  # TODO(HIP/AMD): this option is currently not supported
option(CUDF_USE_WARPSIZE_32 "Uses warp size 32 in cuDF. Only to be used for gfx1100." OFF)


message(VERBOSE "CUDF_JNI: Build with NVTX support: ${USE_NVTX}")
message(VERBOSE "CUDF_JNI: Build cuDF JNI shared libraries: ${BUILD_SHARED_LIBS}")
message(VERBOSE "CUDF_JNI: Configure CMake to build tests: ${BUILD_TESTS}")
message(VERBOSE
        "CUDF_JNI: Build with per-thread default stream: ${CUDF_USE_PER_THREAD_DEFAULT_STREAM}"
)
message(VERBOSE "CUDF_JNI: Statically link the CUDA runtime: ${CUDA_STATIC_RUNTIME}")
message(VERBOSE "CUDF_JNI: Build with GPUDirect Storage support: ${USE_GDS}")
message(VERBOSE "CUDF_JNI: Link with libcudf statically: ${CUDF_JNI_LIBCUDF_STATIC}")

set(CUDF_SOURCE_DIR "${PROJECT_SOURCE_DIR}/../../../../cpp")
if(NOT DEFINED CUDF_CPP_BUILD_DIR OR CUDF_CPP_BUILD_DIR STREQUAL "")
  if(DEFINED ENV{CUDF_CPP_BUILD_DIR})
    set(CUDF_CPP_BUILD_DIR "$ENV{CUDF_CPP_BUILD_DIR}")
  else()
    set(CUDF_CPP_BUILD_DIR "${CUDF_SOURCE_DIR}/build")
  endif()
endif()

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/"
                      "${CUDF_SOURCE_DIR}/cmake/Modules/" ${CMAKE_MODULE_PATH}
)

# ##################################################################################################
# * compiler options ------------------------------------------------------------------------------

set(CUDF_CXX_FLAGS "")
set(CUDF_CUDA_FLAGS "")
set(CUDF_CXX_DEFINITIONS "")
set(CUDF_CUDA_DEFINITIONS "")
set(CUDF_CUDA_FLAGS "-Wno-c++11-narrowing-const-reference")

add_compile_definitions(HIP_ENABLE_WARP_SYNC_BUILTINS)

rapids_find_package(HIP REQUIRED)
include(ConfigureCUDA) # set other CUDA compilation flags

if(CUDF_USE_PER_THREAD_DEFAULT_STREAM)
  message(STATUS "Using per-thread default stream")
  add_compile_definitions(CUDA_API_PER_THREAD_DEFAULT_STREAM __HIP_API_PER_THREAD_DEFAULT_STREAM__ CUDF_USE_PER_THREAD_DEFAULT_STREAM)
endif()

# ##################################################################################################
# * build type ------------------------------------------------------------------------------------
#   Set a default build type if none was specified
rapids_cmake_build_type("Release")

# ##################################################################################################
# * CUDF ------------------------------------------------------------------------------------------

set(cudf_ROOT "${CUDF_CPP_BUILD_DIR}")
rapids_find_package(cudf REQUIRED)

# ##################################################################################################
# * nvcomp------------------------------------------------------------------------------------------

if(NOT DEFINED hipcomp_DIR)
  set(hipcomp_DIR "${CUDF_CPP_BUILD_DIR}/_deps/hipcomp-build")
endif()
rapids_find_package(hipcomp REQUIRED)
if(TARGET hipcomp AND CUDF_USE_WARPSIZE_32)
  target_compile_definitions(hipcomp PRIVATE USE_WARPSIZE_32)
endif()


# ##################################################################################################
# * find JNI -------------------------------------------------------------------------------------

set(JAVA_AWT_INCLUDE_PATH "$ENV{JAVA_HOME}/include")

find_package(JNI REQUIRED)
if(JNI_FOUND)
  message(STATUS "JDK with JNI in ${JNI_INCLUDE_DIRS}")
else()
  message(FATAL_ERROR "JDK with JNI not found, please check your settings.")
endif()

# ##################################################################################################
# * GDS/cufile ------------------------------------------------------------------------------------

if(USE_GDS)
  message(STATUS "Building with GPUDirect Storage (GDS)/cuFile support")
  find_package(cuFile REQUIRED)
endif()

# ##################################################################################################
# * library targets -------------------------------------------------------------------------------

add_library(
  cudfjni
  src/Aggregation128UtilsJni.cpp
  src/AggregationJni.cpp
  src/ChunkedPackJni.cpp
  src/ChunkedReaderJni.cpp
  src/CudfJni.cpp
  src/CudaJni.cpp
  src/ColumnVectorJni.cpp
  src/ColumnViewJni.cpp
  src/ColumnViewJni.cu
  src/CompiledExpression.cpp
  src/ContiguousTableJni.cpp
  src/HashJoinJni.cpp
  src/HostMemoryBufferNativeUtilsJni.cpp
  src/NvcompJni.cpp
  # src/NvtxRangeJni.cpp # TODO(HIP/AMD): Enable these tests when NVTX/rocTX is supported
  # src/NvtxUniqueRangeJni.cpp # TODO(HIP/AMD): Enable these tests when NVTX/rocTX is supported
  src/PackedColumnMetadataJni.cpp
  src/RmmJni.cpp
  src/ScalarJni.cpp
  src/TableJni.cpp
  src/aggregation128_utils.cu
  src/maps_column_view.cu
  src/row_conversion.cu
  src/check_nvcomp_output_sizes.cu
)

# Disable NVTX if necessary
if(NOT USE_NVTX)
  target_compile_definitions(cudfjni PUBLIC NVTX_DISABLE)
endif()

if(CUDF_JNI_ENABLE_PROFILING)
  target_compile_definitions(cudfjni PRIVATE CUDF_JNI_ENABLE_PROFILING)
endif()

if(CUDF_JNI_LIBCUDF_STATIC AND BUILD_SHARED_LIBS)
  # When linking against libcudf.a, the JNI library will include the old libcudf.so. For
  # backwards-compatibility for software that expects to find libcudf.so in the JVM environment
  # after cudf has loaded, the JNI code and libcudf.a will be combined into libcudf.so. A stub
  # library will be created for libcudfjni.so that will simply require libcudf.so for backwards
  # compatibility with software that expects to find libcudfjni.so at runtime.
  set_target_properties(cudfjni PROPERTIES OUTPUT_NAME "cudf")
  add_library(cudfjnistub SHARED src/emptyfile.cpp)
  set_target_properties(cudfjnistub PROPERTIES OUTPUT_NAME "cudfjni")
  target_link_libraries(cudfjnistub -Wl,--no-as-needed cudfjni -Wl,--as-needed)
endif()

# ##################################################################################################
# * include paths ---------------------------------------------------------------------------------

target_include_directories(
  cudfjni PUBLIC "${CMAKE_BINARY_DIR}/include" "${CMAKE_SOURCE_DIR}/include"
                 "${CMAKE_SOURCE_DIR}/src" "${JNI_INCLUDE_DIRS}"
)

# ##################################################################################################
# * compile options
# ---------------------------------------------------------------------------------

# Override RPATH for cudfjni
set_target_properties(
  cudfjni
  PROPERTIES BUILD_RPATH "\$ORIGIN"
             INSTALL_RPATH "\$ORIGIN"
             # set target compile options
             CXX_STANDARD 17
             CXX_STANDARD_REQUIRED ON
             CUDA_STANDARD 17
             CUDA_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
             INTERFACE_POSITION_INDEPENDENT_CODE ON
)

target_compile_options(
  cudfjni PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUDF_CXX_FLAGS}>"
                  "$<$<COMPILE_LANGUAGE:HIP>:${CUDF_CUDA_FLAGS}>"
)

target_compile_definitions(
  cudfjni PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${CUDF_CXX_DEFINITIONS}>"
                 "$<$<COMPILE_LANGUAGE:HIP>:${CUDF_CUDA_DEFINITIONS}>"
)

if(USE_GDS)
  add_library(cufilejni src/CuFileJni.cpp)
  set_target_properties(
    cufilejni
    PROPERTIES BUILD_RPATH "\$ORIGIN"
               INSTALL_RPATH "\$ORIGIN"
               # set target compile options
               CXX_STANDARD 17
               CXX_STANDARD_REQUIRED ON
               POSITION_INDEPENDENT_CODE ON
               INTERFACE_POSITION_INDEPENDENT_CODE ON
  )
  target_include_directories(cufilejni PRIVATE "${cuFile_INCLUDE_DIRS}")
  target_link_libraries(cufilejni PRIVATE cudfjni "${cuFile_LIBRARIES}")
endif()

# ##################################################################################################
# * link libraries --------------------------------------------------------------------------------

set(CUDF_LINK PUBLIC cudf::cudf)
if(CUDF_JNI_LIBCUDF_STATIC)
  # Whole-link libcudf.a into the shared library but not its dependencies
  set(CUDF_LINK PRIVATE -Wl,--whole-archive cudf::cudf -Wl,--no-whole-archive PUBLIC cudf::cudf)
endif()

# When nvcomp is installed we need to use nvcomp::nvcomp but from the cudf build directory it will
# just be nvcomp.
target_link_libraries(
  cudfjni ${CUDF_LINK} PRIVATE $<TARGET_NAME_IF_EXISTS:hipcomp>
                               $<TARGET_NAME_IF_EXISTS:hipcomp::hipcomp>
)

# ##################################################################################################
# * cudart options --------------------------------------------------------------------------------
#   cudart can be statically linked or dynamically linked. The python ecosystem wants dynamic
#   linking

if(CUDA_STATIC_RUNTIME)
  # Tell CMake what CUDA language runtime to use
  set_target_properties(cudfjni PROPERTIES CUDA_RUNTIME_LIBRARY Static)
else()
  # Tell CMake what CUDA language runtime to use
  set_target_properties(cudfjni PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
endif()

# ##################################################################################################
# * install shared libraries ----------------------------------------------------------------------
if(TARGET hipcomp::hipcomp)
  add_custom_command(
    TARGET cudfjni
    PRE_LINK
    COMMAND
      ${CMAKE_COMMAND} -E copy $<TARGET_FILE:hipcomp::hipcomp>
      "${PROJECT_BINARY_DIR}"
    COMMENT "Copying hipcomp libraries to ${PROJECT_BINARY_DIR}"
  )
endif()
