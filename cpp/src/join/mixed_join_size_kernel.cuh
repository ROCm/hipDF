/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include "join_common_utils.cuh"
#include "join_common_utils.hpp"
#include "mixed_join_common_utils.cuh"

#include <cudf/ast/detail/expression_evaluator.cuh>
#include <cudf/ast/detail/expression_parser.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/span.hpp>

#include <hip/hip_cooperative_groups.h>
#include <hipcub/hipcub.hpp>
#include <thrust/iterator/discard_iterator.h>

namespace cudf {
namespace detail {
namespace cg = cooperative_groups;

template <int block_size, bool has_nulls>
__launch_bounds__(block_size) __global__ void compute_mixed_join_output_size(
  table_device_view left_table,
  table_device_view right_table,
  table_device_view probe,
  table_device_view build,
  row_hash const hash_probe,
  row_equality const equality_probe,
  join_kind const join_type,
  cudf::detail::mixed_multimap_type::device_view hash_table_view,
  ast::detail::expression_device_view device_expression_data,
  bool const swap_tables,
  std::size_t* output_size,
  cudf::device_span<cudf::size_type> matches_per_row)
{
  // The (required) extern storage of the shared memory array leads to
  // conflicting declarations between different templates. The easiest
  // workaround is to declare an arbitrary (here char) array type then cast it
  // after the fact to the appropriate type.
  extern __shared__ char raw_intermediate_storage[];
  cudf::ast::detail::IntermediateDataType<has_nulls>* intermediate_storage =
    reinterpret_cast<cudf::ast::detail::IntermediateDataType<has_nulls>*>(raw_intermediate_storage);
  auto thread_intermediate_storage =
    intermediate_storage + (threadIdx.x * device_expression_data.num_intermediates);

  std::size_t thread_counter{0};
  cudf::size_type const start_idx      = threadIdx.x + blockIdx.x * block_size;
  cudf::size_type const stride         = block_size * gridDim.x;
  cudf::size_type const left_num_rows  = left_table.num_rows();
  cudf::size_type const right_num_rows = right_table.num_rows();
  auto const outer_num_rows            = (swap_tables ? right_num_rows : left_num_rows);

  auto evaluator = cudf::ast::detail::expression_evaluator<has_nulls>(
    left_table, right_table, device_expression_data);

  auto const empty_key_sentinel = hash_table_view.get_empty_key_sentinel();
  make_pair_function pair_func{hash_probe, empty_key_sentinel};

  // Figure out the number of elements for this key.
  cg::thread_block_tile<1> this_thread = cg::tiled_partition<1>(cg::this_thread_block());
  // TODO: Address asymmetry in operator.
  auto count_equality = pair_expression_equality<has_nulls>{
    evaluator, thread_intermediate_storage, swap_tables, equality_probe};

  for (cudf::size_type outer_row_index = start_idx; outer_row_index < outer_num_rows;
       outer_row_index += stride) {
    auto query_pair = pair_func(outer_row_index);
    if (join_type == join_kind::LEFT_JOIN || join_type == join_kind::FULL_JOIN) {
      matches_per_row[outer_row_index] =
        hash_table_view.pair_count_outer(this_thread, query_pair, count_equality);
    } else {
      matches_per_row[outer_row_index] =
        hash_table_view.pair_count(this_thread, query_pair, count_equality);
    }
    thread_counter += matches_per_row[outer_row_index];
  }

  using BlockReduce = hipcub::BlockReduce<cudf::size_type, block_size>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  std::size_t block_counter = BlockReduce(temp_storage).Sum(thread_counter);

  // Add block counter to global counter
  if (threadIdx.x == 0) {
    hip::atomic_ref<std::size_t, hip::thread_scope_device> ref{*output_size};
    ref.fetch_add(block_counter, hip::std::memory_order_relaxed);
  }
}

}  // namespace detail
}  // namespace cudf
