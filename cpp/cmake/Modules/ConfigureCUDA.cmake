# =============================================================================
# Copyright (c) 2018-2022, NVIDIA CORPORATION.
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

if(CMAKE_COMPILER_IS_GNUCXX)
  list(APPEND CUDF_CXX_FLAGS -Wall -Werror -Wno-unknown-pragmas -Wno-error=deprecated-declarations)
endif()

list(APPEND CUDF_GPU_FLAGS --expt-extended-lambda --expt-relaxed-constexpr)

# set warnings as errors
if(GPU_WARNINGS_AS_ERRORS)
  list(APPEND CUDF_GPU_FLAGS -Werror=all-warnings)
else()
  list(APPEND CUDF_GPU_FLAGS -Werror=cross-execution-space-call)
endif()
list(APPEND CUDF_GPU_FLAGS -Xcompiler=-Wall,-Werror,-Wno-error=deprecated-declarations)

if(DISABLE_DEPRECATION_WARNINGS)
  list(APPEND CUDF_CXX_FLAGS -Wno-deprecated-declarations)
  list(APPEND CUDF_GPU_FLAGS -Xcompiler=-Wno-deprecated-declarations)
endif()

# make sure we produce smallest binary size
list(APPEND CUDF_GPU_FLAGS -Xfatbin=-compress-all)

# Option to enable line info in CUDA device compilation to allow introspection when profiling /
# memchecking
if(GPU_ENABLE_LINEINFO)
  list(APPEND CUDF_GPU_FLAGS -lineinfo)
endif()

# Debug options
if(CMAKE_BUILD_TYPE MATCHES Debug)
  message(VERBOSE "CUDF: Building with debugging flags")
  list(APPEND CUDF_GPU_FLAGS -Xcompiler=-rdynamic)
endif()

macro(set_cudf_target_properties)
  set_target_properties(
    cudf
    PROPERTIES BUILD_RPATH "\$ORIGIN"
               INSTALL_RPATH "\$ORIGIN"
               # set target compile options
               CXX_STANDARD 17
               CXX_STANDARD_REQUIRED ON
               # For std:: support of __int128_t. Can be removed once using hip::std
               CXX_EXTENSIONS ON
               CUDA_STANDARD 17
               CUDA_STANDARD_REQUIRED ON
               POSITION_INDEPENDENT_CODE ON
               INTERFACE_POSITION_INDEPENDENT_CODE ON
  )
  
  target_compile_options(
    cudf PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUDF_CXX_FLAGS}>"
                  "$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_GPU_FLAGS}>"
  )
  
  if(CUDF_BUILD_STACKTRACE_DEBUG)
    # Remove any optimization level to avoid nvcc warning "incompatible redefinition for option
    # 'optimize'".
    string(REGEX REPLACE "(\-O[0123])" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
    string(REGEX REPLACE "(\-O[0123])" "" CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE}")
    string(REGEX REPLACE "(\-O[0123])" "" CMAKE_CUDA_FLAGS_MINSIZEREL
                         "${CMAKE_CUDA_FLAGS_MINSIZEREL}"
    )
    string(REGEX REPLACE "(\-O[0123])" "" CMAKE_CUDA_FLAGS_RELWITHDEBINFO
                         "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO}"
    )
  
    add_library(cudf_backtrace INTERFACE)
    target_compile_definitions(cudf_backtrace INTERFACE _BUILD_STACKTRACE_DEBUG)
    target_compile_options(
      cudf_backtrace INTERFACE "$<$<COMPILE_LANGUAGE:CXX>:-Og>"
                               "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Og>"
    )
    target_link_options(
      cudf_backtrace INTERFACE "$<$<LINK_LANGUAGE:CXX>:-rdynamic>"
      "$<$<LINK_LANGUAGE:CUDA>:-Xlinker=-rdynamic>"
    )
    target_link_libraries(cudf PRIVATE cudf_backtrace)
  endif()

  target_compile_definitions(
    cudf PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${CUDF_CXX_DEFINITIONS}>"
                 "$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_GPU_DEFINITIONS}>>"
  )

  if(GPU_STATIC_RUNTIME)
    # Tell CMake what CUDA language runtime to use
    set_target_properties(cudf PROPERTIES CUDA_RUNTIME_LIBRARY Static)
    # Make sure to export to consumers what runtime we used
    target_link_libraries(cudf PUBLIC CUDA::cudart_static)
  else()
    # Tell CMake what CUDA language runtime to use
    set_target_properties(cudf PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
    # Make sure to export to consumers what runtime we used
    target_link_libraries(cudf PUBLIC CUDA::cudart)
  endif()
endmacro()

macro(set_hidftest_default_stream_target)
  set_target_properties(
    cudftest_default_stream
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
endmacro()

macro(set_cudftestutil_target)
  set_target_properties(
    cudftestutil
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
endmacro()
