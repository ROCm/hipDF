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

# This function finds hipcomp and sets any additional necessary environment variables.
function(find_and_configure_hipcomp)

  include(${rapids-cmake-dir}/cpm/hipcomp.cmake)
  rapids_cpm_hipcomp(
    BUILD_EXPORT_SET cudf-exports
    INSTALL_EXPORT_SET cudf-exports
    USE_PROPRIETARY_BINARY ${CUDF_USE_PROPRIETARY_HIPCOMP}
    BUILD_STATIC OFF
  )

  # Per-thread default stream
  if(TARGET hipcomp AND CUDF_USE_PER_THREAD_DEFAULT_STREAM)
    target_compile_definitions(hipcomp PRIVATE CUDA_API_PER_THREAD_DEFAULT_STREAM HIP_API_PER_THREAD_DEFAULT_STREAM __HIP_API_PER_THREAD_DEFAULT_STREAM__)
  endif()

  # Wave size 32
  if(TARGET hipcomp AND CUDF_USE_WARPSIZE_32)
    target_compile_definitions(hipcomp PRIVATE USE_WARPSIZE_32)
  endif()
endfunction()

find_and_configure_hipcomp()
