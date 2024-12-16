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

/**
 * @file typed_statistics_chunk.cuh
 * @brief Templated wrapper to generalize statistics chunk reduction and aggregation
 * across different leaf column types
 */

#pragma once

#include "byte_array_view.cuh"
#include "statistics.cuh"
#include "statistics_type_identification.cuh"
#include "temp_storage_wrapper.cuh"

#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <hip/hip_math_constants.h>

#include <thrust/extrema.h>

namespace cudf {
namespace io {

/**
 * @brief Class used to get reference to members of unions related to statistics calculations
 */
class union_member {
  template <typename U, typename V>
  using reference_type = std::conditional_t<std::is_const_v<U>, V const&, V&>;

 public:
  template <typename T, typename U>
  using type = std::conditional_t<
    std::is_same_v<std::remove_cv_t<T>, string_view>,
    reference_type<U, string_stats>,
    std::conditional_t<std::is_same_v<std::remove_cv_t<T>, statistics::byte_array_view>,
                       reference_type<U, byte_array_stats>,
                       reference_type<U, T>>>;

  template <typename T, typename U>
  __device__ static std::enable_if_t<std::is_integral_v<T> and std::is_unsigned_v<T>, type<T, U>>
  get(U& val)
  {
    return val.u_val;
  }

  // NOTE(HIP/AMD): The check for !std::is_same_v<T, __int128_t> was added to resolve compiler error (ambiguity when selecing correct template)
  template <typename T, typename U>
  __device__ static std::enable_if_t<!std::is_same_v<T, __int128_t> and std::is_integral_v<T> and std::is_signed_v<T>, type<T, U>> get(
    U& val)
  {
    return val.i_val;
  }

  template <typename T, typename U>
  __device__ static std::enable_if_t<std::is_same_v<T, __int128_t>, type<T, U>> get(U& val)
  {
    return val.d128_val;
  }

  template <typename T, typename U>
  __device__ static std::enable_if_t<std::is_floating_point_v<T>, type<T, U>> get(U& val)
  {
    return val.fp_val;
  }

  template <typename T, typename U>
  __device__ static std::enable_if_t<std::is_same_v<T, string_view>, type<T, U>> get(U& val)
  {
    return val.str_val;
  }

  template <typename T, typename U>
  __device__ static std::enable_if_t<std::is_same_v<T, statistics::byte_array_view>, type<T, U>>
  get(U& val)
  {
    return val.byte_val;
  }
};

/**
 * @brief Templated structure used for merging and gathering of statistics chunks
 *
 * This uses the reduce function to compute the minimum, maximum and aggregate
 * values simultaneously.
 *
 * @tparam T The input type associated with the chunk
 * @tparam is_aggregation_supported Set to true if input type is meant to be aggregated
 */
template <typename T, bool is_aggregation_supported>
struct typed_statistics_chunk {};

template <typename T>
struct typed_statistics_chunk<T, true> {
  using E = typename detail::extrema_type<T>::type;
  using A = typename detail::aggregation_type<T>::type;

  uint32_t non_nulls{0};   //!< number of non-null values in chunk
  uint32_t null_count{0};  //!< number of null values in chunk

  E minimum_value;
  E maximum_value;
  A aggregate;

  uint8_t has_minmax{false};  //!< Nonzero if min_value and max_values are valid
  uint8_t has_sum{false};     //!< Nonzero if sum is valid

  __device__ typed_statistics_chunk()
    : minimum_value(detail::minimum_identity<E>()),
      maximum_value(detail::maximum_identity<E>()),
      aggregate(0)
  {
  }

  __device__ void reduce(T const& elem)
  {
    non_nulls++;
    minimum_value = thrust::min<E>(minimum_value, detail::extrema_type<T>::convert(elem));
    maximum_value = thrust::max<E>(maximum_value, detail::extrema_type<T>::convert(elem));
    aggregate += detail::aggregation_type<T>::convert(elem);
    has_minmax = true;
  }

  __device__ void reduce(statistics_chunk const& chunk)
  {
    if (chunk.has_minmax) {
      minimum_value = thrust::min<E>(minimum_value, union_member::get<E>(chunk.min_value));
      maximum_value = thrust::max<E>(maximum_value, union_member::get<E>(chunk.max_value));
    }
    if (chunk.has_sum) { aggregate += union_member::get<A>(chunk.sum); }
    non_nulls += chunk.non_nulls;
    null_count += chunk.null_count;
  }
};

template <typename T>
struct typed_statistics_chunk<T, false> {
  using E = typename detail::extrema_type<T>::type;

  uint32_t non_nulls{0};   //!< number of non-null values in chunk
  uint32_t null_count{0};  //!< number of null values in chunk

  E minimum_value;
  E maximum_value;

  uint8_t has_minmax{false};  //!< Nonzero if min_value and max_values are valid
  uint8_t has_sum{false};     //!< Nonzero if sum is valid

  __device__ typed_statistics_chunk()
    : minimum_value(detail::minimum_identity<E>()), maximum_value(detail::maximum_identity<E>())
  {
  }

  __device__ void reduce(T const& elem)
  {
    non_nulls++;
    minimum_value = thrust::min<E>(minimum_value, detail::extrema_type<T>::convert(elem));
    maximum_value = thrust::max<E>(maximum_value, detail::extrema_type<T>::convert(elem));
    has_minmax    = true;
  }

  __device__ void reduce(statistics_chunk const& chunk)
  {
    if (chunk.has_minmax) {
      minimum_value = thrust::min<E>(minimum_value, union_member::get<E>(chunk.min_value));
      maximum_value = thrust::max<E>(maximum_value, union_member::get<E>(chunk.max_value));
    }
    non_nulls += chunk.non_nulls;
    null_count += chunk.null_count;
  }
};

/**
 * @brief Function to reduce members of a typed_statistics_chunk across a thread block
 *
 * @tparam T Type associated with typed_statistics_chunk
 * @tparam block_size Dimension of the thread block
 * @param chunk The input typed_statistics_chunk
 * @param storage Temporary storage to be used by cub calls
 */
template <typename T, bool include_aggregate, int block_size>
__inline__ __device__ typed_statistics_chunk<T, include_aggregate> block_reduce(
  typed_statistics_chunk<T, include_aggregate>& chunk, detail::storage_wrapper<block_size>& storage)
{
  typed_statistics_chunk<T, include_aggregate> output_chunk = chunk;

  using E              = typename detail::extrema_type<T>::type;
  using extrema_reduce = hipcub::BlockReduce<E, block_size>;
  using count_reduce   = hipcub::BlockReduce<uint32_t, block_size>;
  output_chunk.minimum_value =
    extrema_reduce(storage.template get<E>()).Reduce(output_chunk.minimum_value, hipcub::Min());
  __syncthreads();
  output_chunk.maximum_value =
    extrema_reduce(storage.template get<E>()).Reduce(output_chunk.maximum_value, hipcub::Max());
  __syncthreads();
  output_chunk.non_nulls =
    count_reduce(storage.template get<uint32_t>()).Sum(output_chunk.non_nulls);
  __syncthreads();
  output_chunk.null_count =
    count_reduce(storage.template get<uint32_t>()).Sum(output_chunk.null_count);
  __syncthreads();
  output_chunk.has_minmax = __syncthreads_or(output_chunk.has_minmax);

  // FIXME : Is another syncthreads needed here?
  if constexpr (include_aggregate) {
    if (output_chunk.has_minmax) {
      using A                = typename detail::aggregation_type<T>::type;
      using aggregate_reduce = hipcub::BlockReduce<A, block_size>;
      output_chunk.aggregate =
        aggregate_reduce(storage.template get<A>()).Sum(output_chunk.aggregate);
    }
  }
  return output_chunk;
}

/**
 * @brief Function to convert typed_statistics_chunk into statistics_chunk
 *
 * @tparam T Type associated with typed_statistics_chunk
 * @param chunk The input typed_statistics_chunk
 */
template <typename T, bool include_aggregate>
__inline__ __device__ statistics_chunk
get_untyped_chunk(typed_statistics_chunk<T, include_aggregate> const& chunk)
{
  using E = typename detail::extrema_type<T>::type;
  statistics_chunk stat{};
  stat.non_nulls  = chunk.non_nulls;
  stat.null_count = chunk.null_count;
  stat.has_minmax = chunk.has_minmax;
  stat.has_sum    = [&]() {
    // invalidate the sum if overflow or underflow is possible
    if constexpr (std::is_floating_point_v<E> or std::is_integral_v<E>) {
      if (!chunk.has_minmax) { return true; }
      return std::numeric_limits<E>::max() / chunk.non_nulls >=
               static_cast<E>(chunk.maximum_value) and
             std::numeric_limits<E>::lowest() / chunk.non_nulls <=
               static_cast<E>(chunk.minimum_value);
    }
    return true;
  }();
  if (chunk.has_minmax) {
    if constexpr (std::is_floating_point_v<E>) {
      union_member::get<E>(stat.min_value) =
        (chunk.minimum_value != 0.0) ? chunk.minimum_value : HIP_NEG_ZERO;
      union_member::get<E>(stat.max_value) =
        (chunk.maximum_value != 0.0) ? chunk.maximum_value : HIP_ZERO;
    } else {
      union_member::get<E>(stat.min_value) = chunk.minimum_value;
      union_member::get<E>(stat.max_value) = chunk.maximum_value;
    }
    if constexpr (include_aggregate) {
      using A                        = typename detail::aggregation_type<T>::type;
      union_member::get<A>(stat.sum) = chunk.aggregate;
    }
  }
  return stat;
}

}  // namespace io
}  // namespace cudf
