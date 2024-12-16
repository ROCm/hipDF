/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

#pragma once

#include <hipcub/block/block_scan.hpp>
#include <hip_extensions/hipcub_ext/hipcub_ext.cuh>

#include <hip/atomic>

namespace cudf {
namespace io {
namespace text {
namespace detail {

enum class scan_tile_status : uint8_t {
  oob,
  invalid,
  partial,
  inclusive,
};

template <typename T>
struct scan_tile_state_view {
  uint64_t num_tiles;
  hip::atomic<scan_tile_status, hip::thread_scope_device>* tile_status;
  T* tile_partial;
  T* tile_inclusive;

  __device__ inline void set_status(cudf::size_type tile_idx, scan_tile_status status)
  {
    auto const offset = (tile_idx + num_tiles) % num_tiles;
    tile_status[offset].store(status, hip::memory_order_relaxed);
  }

  __device__ inline void set_partial_prefix(cudf::size_type tile_idx, T value)
  {
    auto const offset = (tile_idx + num_tiles) % num_tiles;
    hipcub::ThreadStore<hipcub::STORE_CG>(tile_partial + offset, value);
    tile_status[offset].store(scan_tile_status::partial);
  }

  __device__ inline void set_inclusive_prefix(cudf::size_type tile_idx, T value)
  {
    auto const offset = (tile_idx + num_tiles) % num_tiles;
    hipcub::ThreadStore<hipcub::STORE_CG>(tile_inclusive + offset, value);
    tile_status[offset].store(scan_tile_status::inclusive);
  }

  //: TODO(HIP/AMD): we use hipcub_extensions here to fix the hipcub error "no viable conversion from 'int' to 'cudf::io::text::detail::multistate'"
  __device__ inline T get_prefix(cudf::size_type tile_idx, scan_tile_status& status)
  {
    auto const offset = (tile_idx + num_tiles) % num_tiles;

    while ((status = tile_status[offset].load(hip::memory_order_relaxed)) ==
           scan_tile_status::invalid) {}

    //: NOTE(HIP/AMD): This threadfence is necessary, as the subsequent ThreadLoad
    // otherwise appears re-ordered before the loading of the atomic flag in line 68,
    // thus resulting in the return of an invalid prefix. 
    // See: internal issue 71
    __threadfence();

    if (status == scan_tile_status::partial) {
      return hipcub_extensions::ThreadLoad<hipcub::LOAD_CG>(tile_partial + offset);
    } else {
      return hipcub_extensions::ThreadLoad<hipcub::LOAD_CG>(tile_inclusive + offset);
    }
  }
};

template <typename T>
struct scan_tile_state {
  rmm::device_uvector<hip::atomic<scan_tile_status, hip::thread_scope_device>> tile_status;
  rmm::device_uvector<T> tile_state_partial;
  rmm::device_uvector<T> tile_state_inclusive;

  scan_tile_state(cudf::size_type num_tiles,
                  rmm::cuda_stream_view stream,
                  rmm::mr::device_memory_resource* mr)
    : tile_status(rmm::device_uvector<hip::atomic<scan_tile_status, hip::thread_scope_device>>(
        num_tiles, stream, mr)),
      tile_state_partial(rmm::device_uvector<T>(num_tiles, stream, mr)),
      tile_state_inclusive(rmm::device_uvector<T>(num_tiles, stream, mr))
  {
  }

  operator scan_tile_state_view<T>()
  {
    return scan_tile_state_view<T>{tile_status.size(),
                                   tile_status.data(),
                                   tile_state_partial.data(),
                                   tile_state_inclusive.data()};
  }

  inline T get_inclusive_prefix(cudf::size_type tile_idx, rmm::cuda_stream_view stream) const
  {
    auto const offset = (tile_idx + tile_status.size()) % tile_status.size();
    return tile_state_inclusive.element(offset, stream);
  }
};

template <typename T>
struct scan_tile_state_callback {
  __device__ inline scan_tile_state_callback(scan_tile_state_view<T>& tile_state,
                                             cudf::size_type tile_idx)
    : _tile_state(tile_state), _tile_idx(tile_idx)
  {
  }

  __host__ __device__ inline T operator()(T const& block_aggregate)
  {
    T exclusive_prefix;

    if (threadIdx.x == 0) {
      _tile_state.set_partial_prefix(_tile_idx, block_aggregate);

      auto predecessor_idx    = _tile_idx - 1;
      auto predecessor_status = scan_tile_status::invalid;

      // scan partials to form prefix

      //: TODO(HIP/AMD): we use hipcub_extensions here to fix the hipcub error "no viable conversion from 'int' to 'cudf::io::text::detail::multistate'"
      auto window_partial = _tile_state.get_prefix(predecessor_idx, predecessor_status);
      while (predecessor_status != scan_tile_status::inclusive) {
        predecessor_idx--;
        auto predecessor_prefix = _tile_state.get_prefix(predecessor_idx, predecessor_status);
        window_partial          = predecessor_prefix + window_partial;
      }
      exclusive_prefix = window_partial;
      //: TODO(HIP/AMD): we use hipcub_extensions here to fix the hipcub error "no viable conversion from 'int' to 'cudf::io::text::detail::multistate'"

      _tile_state.set_inclusive_prefix(_tile_idx, exclusive_prefix + block_aggregate);
    }

    return exclusive_prefix;
  }

  scan_tile_state_view<T>& _tile_state;
  cudf::size_type _tile_idx;
};

}  // namespace detail
}  // namespace text
}  // namespace io
}  // namespace cudf
