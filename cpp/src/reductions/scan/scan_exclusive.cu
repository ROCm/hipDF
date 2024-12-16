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

#include "scan.cuh"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/null_mask.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/scan.h>

namespace cudf {
namespace detail {
namespace {

/**
 * @brief Dispatcher for running a scan operation on an input column
 *
 * @tparam Op device binary operator (e.g. min, max, sum)
 */
template <typename Op>
struct scan_dispatcher {
 public:
  /**
   * @brief Creates a new column from input column by applying exclusive scan operation
   *
   * @tparam T type of input column
   *
   * @param input  Input column view
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param mr Device memory resource used to allocate the returned column's device memory
   * @return Output column with scan results
   */
  template <typename T, std::enable_if_t<hip::std::is_arithmetic_v<T>>* = nullptr>
  std::unique_ptr<column> operator()(column_view const& input,
                                     bitmask_type const*,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    auto output_column =
      detail::allocate_like(input, input.size(), mask_allocation_policy::NEVER, stream, mr);
    mutable_column_view output = output_column->mutable_view();

    auto d_input  = column_device_view::create(input, stream);
    auto identity = Op::template identity<T>();

    auto begin = make_null_replacement_iterator(*d_input, identity, input.has_nulls());
    thrust::exclusive_scan(
      rmm::exec_policy(stream), begin, begin + input.size(), output.data<T>(), identity, Op{});

    CUDF_CHECK_CUDA(stream.value());
    return output_column;
  }

  template <typename T, typename... Args>
  std::enable_if_t<not hip::std::is_arithmetic_v<T>, std::unique_ptr<column>> operator()(Args&&...)
  {
    CUDF_FAIL("Non-arithmetic types not supported for exclusive scan");
  }
};

}  // namespace

std::unique_ptr<column> scan_exclusive(column_view const& input,
                                       scan_aggregation const& agg,
                                       null_policy null_handling,
                                       rmm::cuda_stream_view stream,
                                       rmm::mr::device_memory_resource* mr)
{
  auto [mask, null_count] = [&] {
    if (null_handling == null_policy::EXCLUDE) {
      return std::make_pair(std::move(detail::copy_bitmask(input, stream, mr)), input.null_count());
    } else if (input.nullable()) {
      return mask_scan(input, scan_type::EXCLUSIVE, stream, mr);
    }
    return std::make_pair(rmm::device_buffer{}, size_type{0});
  }();

  auto output = scan_agg_dispatch<scan_dispatcher>(
    input, agg, static_cast<bitmask_type*>(mask.data()), stream, mr);
  output->set_null_mask(mask, null_count);

  return output;
}

}  // namespace detail

}  // namespace cudf