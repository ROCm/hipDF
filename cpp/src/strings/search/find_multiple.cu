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

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sequence.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/find_multiple.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

namespace cudf {
namespace strings {
namespace detail {
std::unique_ptr<column> find_multiple(strings_column_view const& input,
                                      strings_column_view const& targets,
                                      rmm::cuda_stream_view stream,
                                      rmm::mr::device_memory_resource* mr)
{
  auto const strings_count = input.size();
  auto const targets_count = targets.size();
  CUDF_EXPECTS(targets_count > 0, "Must include at least one search target");
  CUDF_EXPECTS(!targets.has_nulls(), "Search targets cannot contain null strings");

  auto strings_column = column_device_view::create(input.parent(), stream);
  auto d_strings      = *strings_column;
  auto targets_column = column_device_view::create(targets.parent(), stream);
  auto d_targets      = *targets_column;

  auto const total_count = strings_count * targets_count;

  // create output column
  auto results = make_numeric_column(
    data_type{type_id::INT32}, total_count, rmm::device_buffer{0, stream, mr}, 0, stream, mr);

  // fill output column with position values
  thrust::transform(rmm::exec_policy(stream),
                    thrust::make_counting_iterator<size_type>(0),
                    thrust::make_counting_iterator<size_type>(total_count),
                    results->mutable_view().begin<int32_t>(),
                    [d_strings, d_targets, targets_count] __device__(size_type idx) {
                      size_type str_idx = idx / targets_count;
                      if (d_strings.is_null(str_idx)) return -1;
                      string_view d_str = d_strings.element<string_view>(str_idx);
                      string_view d_tgt = d_targets.element<string_view>(idx % targets_count);
                      return d_str.find(d_tgt);
                    });
  results->set_null_count(0);

  auto offsets = cudf::detail::sequence(strings_count + 1,
                                        numeric_scalar<size_type>(0, true, stream),
                                        numeric_scalar<size_type>(targets_count, true, stream),
                                        stream,
                                        mr);
  return make_lists_column(strings_count,
                           std::move(offsets),
                           std::move(results),
                           0,
                           rmm::device_buffer{0, stream, mr},
                           stream,
                           mr);
}

}  // namespace detail

// external API
std::unique_ptr<column> find_multiple(strings_column_view const& input,
                                      strings_column_view const& targets,
                                      rmm::cuda_stream_view stream,
                                      rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::find_multiple(input, targets, stream, mr);
}

}  // namespace strings
}  // namespace cudf
