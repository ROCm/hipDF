/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/hashing/detail/default_hash.cuh>
#include <cudf/hashing/detail/helper_functions.cuh>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>

#include <hip/atomic>

namespace cudf {
namespace detail {
/*
 *  Device view of the unordered multiset
 */
template <typename Element,
          typename Hasher   = cudf::hashing::detail::default_hash<Element>,
          typename Equality = equal_to<Element>>
class unordered_multiset_device_view {
 public:
  unordered_multiset_device_view(size_type hash_size,
                                 size_type const* hash_begin,
                                 Element const* hash_data)
    : hash_size{hash_size}, hash_begin{hash_begin}, hash_data{hash_data}, hasher(), equals()
  {
  }

  bool __device__ contains(Element e) const
  {
    size_type loc = hasher(e) % (2 * hash_size);

    for (size_type i = hash_begin[loc]; i < hash_begin[loc + 1]; ++i) {
      if (equals(hash_data[i], e)) return true;
    }

    return false;
  }

 private:
  Hasher hasher;
  Equality equals;
  size_type hash_size;
  size_type const* hash_begin;
  Element const* hash_data;
};

/*
 * Fixed size set on a device.
 */
template <typename Element,
          typename Hasher   = cudf::hashing::detail::default_hash<Element>,
          typename Equality = equal_to<Element>>
class unordered_multiset {
 public:
  /**
   * @brief Factory to construct a new unordered_multiset
   */
  static unordered_multiset<Element> create(column_view const& col, rmm::cuda_stream_view stream)
  {
    auto d_column = column_device_view::create(col, stream);
    auto d_col    = *d_column;

    auto hash_bins_start = cudf::detail::make_zeroed_device_uvector_async<size_type>(
      2 * d_col.size() + 1, stream, rmm::mr::get_current_device_resource());
    auto hash_bins_end = cudf::detail::make_zeroed_device_uvector_async<size_type>(
      2 * d_col.size() + 1, stream, rmm::mr::get_current_device_resource());
    auto hash_data = rmm::device_uvector<Element>(d_col.size(), stream);

    Hasher hasher;
    size_type* d_hash_bins_start = hash_bins_start.data();
    size_type* d_hash_bins_end   = hash_bins_end.data();
    Element* d_hash_data         = hash_data.data();

    thrust::for_each(
      rmm::exec_policy(stream),
      thrust::make_counting_iterator<size_type>(0),
      thrust::make_counting_iterator<size_type>(col.size()),
      [d_hash_bins_start, d_col, hasher] __device__(size_t idx) {
        if (!d_col.is_null(idx)) {
          Element e     = d_col.element<Element>(idx);
          size_type tmp = hasher(e) % (2 * d_col.size());
          hip::atomic_ref<size_type, hip::thread_scope_device> ref{*(d_hash_bins_start + tmp)};
          ref.fetch_add(1, hip::std::memory_order_relaxed);
        }
      });

    thrust::exclusive_scan(rmm::exec_policy(stream),
                           hash_bins_start.begin(),
                           hash_bins_start.end(),
                           hash_bins_end.begin());

    thrust::copy(rmm::exec_policy(stream),
                 hash_bins_end.begin(),
                 hash_bins_end.end(),
                 hash_bins_start.begin());

    thrust::for_each(
      rmm::exec_policy(stream),
      thrust::make_counting_iterator<size_type>(0),
      thrust::make_counting_iterator<size_type>(col.size()),
      [d_hash_bins_end, d_hash_data, d_col, hasher] __device__(size_t idx) {
        if (!d_col.is_null(idx)) {
          Element e     = d_col.element<Element>(idx);
          size_type tmp = hasher(e) % (2 * d_col.size());
          hip::atomic_ref<size_type, hip::thread_scope_device> ref{*(d_hash_bins_end + tmp)};
          size_type offset    = ref.fetch_add(1, hip::std::memory_order_relaxed);
          d_hash_data[offset] = e;
        }
      });

    return unordered_multiset(d_col.size(), std::move(hash_bins_start), std::move(hash_data));
  }

  unordered_multiset_device_view<Element, Hasher, Equality> to_device() const
  {
    return unordered_multiset_device_view<Element, Hasher, Equality>(
      size, hash_bins.data(), hash_data.data());
  }

 private:
  unordered_multiset(size_type size,
                     rmm::device_uvector<size_type>&& hash_bins,
                     rmm::device_uvector<Element>&& hash_data)
    : size{size}, hash_bins{std::move(hash_bins)}, hash_data{std::move(hash_data)}
  {
  }

  size_type size;
  rmm::device_uvector<size_type> hash_bins;
  rmm::device_uvector<Element> hash_data;
};

}  // namespace detail
}  // namespace cudf
