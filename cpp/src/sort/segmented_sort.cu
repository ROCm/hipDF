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

#include "segmented_sort_impl.cuh"

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/sorting.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/iterator/counting_iterator.h>

namespace cudf {
namespace detail {

rmm::device_uvector<size_type> get_segment_indices(size_type num_rows,
                                                   column_view const& offsets,
                                                   rmm::cuda_stream_view stream)
{
  rmm::device_uvector<size_type> segment_ids(num_rows, stream);

  auto offset_begin  = offsets.begin<size_type>();
  auto offset_end    = offsets.end<size_type>();
  auto counting_iter = thrust::make_counting_iterator<size_type>(0);
  thrust::transform(rmm::exec_policy(stream),
                    counting_iter,
                    counting_iter + segment_ids.size(),
                    segment_ids.begin(),
                    [offset_begin, offset_end] __device__(auto idx) {
                      if (offset_begin == offset_end || idx < *offset_begin) { return idx; }
                      if (idx >= *(offset_end - 1)) { return idx + 1; }
                      return static_cast<size_type>(
                        *thrust::upper_bound(thrust::seq, offset_begin, offset_end, idx));
                    });
  return segment_ids;
}

std::unique_ptr<column> segmented_sorted_order(table_view const& keys,
                                               column_view const& segment_offsets,
                                               std::vector<order> const& column_order,
                                               std::vector<null_order> const& null_precedence,
                                               rmm::cuda_stream_view stream,
                                               rmm::mr::device_memory_resource* mr)
{
  return segmented_sorted_order_common<sort_method::UNSTABLE>(
    keys, segment_offsets, column_order, null_precedence, stream, mr);
}

std::unique_ptr<table> segmented_sort_by_key(table_view const& values,
                                             table_view const& keys,
                                             column_view const& segment_offsets,
                                             std::vector<order> const& column_order,
                                             std::vector<null_order> const& null_precedence,
                                             rmm::cuda_stream_view stream,
                                             rmm::mr::device_memory_resource* mr)
{
  return segmented_sort_by_key_common<sort_method::UNSTABLE>(
    values, keys, segment_offsets, column_order, null_precedence, stream, mr);
}

}  // namespace detail

std::unique_ptr<column> segmented_sorted_order(table_view const& keys,
                                               column_view const& segment_offsets,
                                               std::vector<order> const& column_order,
                                               std::vector<null_order> const& null_precedence,
                                               rmm::cuda_stream_view stream,
                                               rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::segmented_sorted_order(
    keys, segment_offsets, column_order, null_precedence, stream, mr);
}

std::unique_ptr<table> segmented_sort_by_key(table_view const& values,
                                             table_view const& keys,
                                             column_view const& segment_offsets,
                                             std::vector<order> const& column_order,
                                             std::vector<null_order> const& null_precedence,
                                             rmm::cuda_stream_view stream,
                                             rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::segmented_sort_by_key(
    values, keys, segment_offsets, column_order, null_precedence, stream, mr);
}

}  // namespace cudf
