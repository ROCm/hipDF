# =============================================================================
# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

# Create `jitify_preprocess` executable
add_executable(jitify_preprocess "${JITIFY_INCLUDE_DIR}/jitify2_preprocess.cpp")

target_compile_definitions(jitify_preprocess PRIVATE "_FILE_OFFSET_BITS=64")
target_link_libraries(jitify_preprocess hip::host ${CMAKE_DL_LIBS})

# Take a list of files to JIT-compile and run them through jitify_preprocess.
function(jit_preprocess_files)
  cmake_parse_arguments(ARG "" "SOURCE_DIRECTORY" "FILES" ${ARGN})

  foreach(inc IN LISTS libhipcxx_raw_includes)
    list(APPEND libhipcxx_includes "-I${inc}")
  endforeach()
  foreach(ARG_FILE ${ARG_FILES})
    set(ARG_OUTPUT ${CUDF_GENERATED_INCLUDE_DIR}/include/jit_preprocessed_files/${ARG_FILE}.jit.hpp)
    get_filename_component(jit_output_directory "${ARG_OUTPUT}" DIRECTORY)
    list(APPEND JIT_PREPROCESSED_FILES "${ARG_OUTPUT}")

    set(CUDF_JITIFY_EXTRA_PREPROCESSING_FLAGS "")
    if(CUDF_USE_WARPSIZE_32)
      set(CUDF_JITIFY_EXTRA_PREPROCESSING_FLAGS "-DCUDF_USE_WARPSIZE_32")
    endif()

    # Note: need to pass _FILE_OFFSET_BITS=64 in COMMAND due to a limitation in how conda builds
    # glibc
    add_custom_command(
      OUTPUT ${ARG_OUTPUT}
      DEPENDS jitify_preprocess "${ARG_SOURCE_DIRECTORY}/${ARG_FILE}"
      WORKING_DIRECTORY ${ARG_SOURCE_DIRECTORY}
      VERBATIM
      COMMAND ${CMAKE_COMMAND} -E make_directory "${jit_output_directory}"
      COMMAND
        "${CMAKE_COMMAND}" -E env LD_LIBRARY_PATH=${HIP_LIB_INSTALL_DIR}
        $<TARGET_FILE:jitify_preprocess> ${ARG_FILE} -o
	${CUDF_GENERATED_INCLUDE_DIR}/include/jit_preprocessed_files -i -m -std=c++17
	-D_FILE_OFFSET_BITS=64 -D__HIPCC_RTC__ ${CUDF_JITIFY_EXTRA_PREPROCESSING_FLAGS} -I${CUDF_SOURCE_DIR}/include
	-I${CUDF_SOURCE_DIR}/src ${libhipcxx_includes} -I${_libhipcxx_INCLUDE_DIR} -I${HIP_INCLUDE_DIRS}
        --no-replace-pragma-once
      COMMENT "Custom command to JIT-compile files."
    )
  endforeach()
  set(JIT_PREPROCESSED_FILES
      "${JIT_PREPROCESSED_FILES}"
      PARENT_SCOPE
  )
endfunction()

jit_preprocess_files(
  SOURCE_DIRECTORY ${CUDF_SOURCE_DIR}/src FILES binaryop/jit/kernel.cu transform/jit/kernel.cu
  rolling/jit/kernel.cu
)

add_custom_target(
  jitify_preprocess_run
  DEPENDS ${JIT_PREPROCESSED_FILES}
  COMMENT "Target representing jitified files."
)

