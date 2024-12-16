/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// MIT License
//
// Modifications Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <rolling/detail/rolling_jit.hpp>
#include <rolling/jit/operation.hpp>

#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>

namespace cudf {
namespace rolling {
namespace jit {

template <typename WindowType>
__device__  cudf::size_type get_window(WindowType window, cudf::thread_index_type index)
{
  return window[index];
}

template <>
__device__  cudf::size_type get_window(cudf::size_type window, cudf::thread_index_type index)
{
  return window;
}

template <typename InType,
          typename OutType,
          class agg_op,
          typename PrecedingWindowType,
          typename FollowingWindowType>
__global__ void gpu_rolling_new(cudf::size_type nrows,
                                InType const* const __restrict__ in_col,
                                cudf::bitmask_type const* const __restrict__ in_col_valid,
                                OutType* __restrict__ out_col,
                                cudf::bitmask_type* __restrict__ out_col_valid,
                                cudf::size_type* __restrict__ output_valid_count,
                                PrecedingWindowType preceding_window_begin,
                                FollowingWindowType following_window_begin,
                                cudf::size_type min_periods)
{
  cudf::thread_index_type i            = blockIdx.x * blockDim.x + threadIdx.x;
  cudf::thread_index_type const stride = blockDim.x * gridDim.x;

  cudf::size_type warp_valid_count{0};

  //auto active_threads = __ballot_sync(0xffff'ffffu, i < nrows);
  //TODO(HIP/AMD): is this WAR for missing __ballot_sync correct?
  auto active_threads = __ballot(i < nrows);
  while (i < nrows) {
    int64_t const preceding_window = get_window(preceding_window_begin, i);
    int64_t const following_window = get_window(following_window_begin, i);

    // compute bounds
    auto const start = static_cast<cudf::size_type>(
      min(static_cast<int64_t>(nrows), max(int64_t{0}, i - preceding_window + 1)));
    auto const end = static_cast<cudf::size_type>(
      min(static_cast<int64_t>(nrows), max(int64_t{0}, i + following_window + 1)));
    auto const start_index = min(start, end);
    auto const end_index   = max(start, end);

    // aggregate
    // TODO: We should explore using shared memory to avoid redundant loads.
    //       This might require separating the kernel into a special version
    //       for dynamic and static sizes.
    cudf::size_type count = end_index - start_index;
    OutType val           = agg_op::template operate<OutType, InType>(in_col, start_index, count);

    // check if we have enough input samples
    bool const output_is_valid = (count >= min_periods);

    // set the mask
    bitmask_type const result_mask = __ballot(output_is_valid) & active_threads;

    // store the output value, one per thread
    if (output_is_valid) { out_col[i] = val; }

    // only one thread writes the mask
    if (0 == cudf::intra_word_index(i)) {
      out_col_valid[cudf::word_index(i)] = result_mask;
      warp_valid_count += __POPC(result_mask);
    }

    // process next element
    i += stride;
    //TODO(HIP/AMD): is this WAR for missing __ballot_sync correct?
    active_threads = __ballot(i < nrows) & active_threads;
  }

  // TODO: likely faster to do a single_lane_block_reduce and a single
  // atomic per block but that requires jitifying single_lane_block_reduce...
  if (0 == cudf::intra_word_index(threadIdx.x)) { atomicAdd(output_valid_count, warp_valid_count); }
}

}  // namespace jit
}  // namespace rolling
}  // namespace cudf
//TODO(HIP/AMD): This new line appears to be needed as an "f" would otherwise be appended into the source code passed to jitify/cudf with minification, see internal jitify issue #31
