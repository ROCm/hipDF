/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
 * @file temp_storage_wrapper.cuh
 * @brief Temporary storage for cub calls and helper wrapper class
 */

#pragma once

#include "byte_array_view.cuh"
#include "statistics.cuh"

#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/wrappers/durations.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <hipcub/hipcub.hpp>

namespace cudf {
namespace io {
namespace detail {

template <typename T, int block_size>
using cub_temp_storage = typename hipcub::BlockReduce<T, block_size>::TempStorage;
using statistics::byte_array_view;

#define MEMBER_NAME(TYPE) TYPE##_stats

#define DECLARE_MEMBER(TYPE) cub_temp_storage<TYPE, block_size> MEMBER_NAME(TYPE);

/**
 * @brief Templated union to hold temporary storage to be used by cub reduce
 * calls
 *
 * @tparam block_size Dimension of the block
 */
template <int block_size>
union block_reduce_storage {
  DECLARE_MEMBER(bool)
  DECLARE_MEMBER(int8_t)
  DECLARE_MEMBER(int16_t)
  DECLARE_MEMBER(int32_t)
  DECLARE_MEMBER(int64_t)
  DECLARE_MEMBER(__int128_t)
  DECLARE_MEMBER(uint8_t)
  DECLARE_MEMBER(uint16_t)
  DECLARE_MEMBER(uint32_t)
  DECLARE_MEMBER(uint64_t)
  DECLARE_MEMBER(float)
  DECLARE_MEMBER(double)
  DECLARE_MEMBER(string_view)
  DECLARE_MEMBER(byte_array_view)
};

#define STORAGE_WRAPPER_GET(TYPE)                                                                 \
  template <typename T>                                                                           \
  __device__ std::enable_if_t<std::is_same_v<T, TYPE>, cub_temp_storage<TYPE, block_size>&> get() \
  {                                                                                               \
    return storage.MEMBER_NAME(TYPE);                                                             \
  }

/**
 * @brief Templated wrapper for block_reduce_storage to return member reference based on requested
 * type
 *
 * @tparam block_size Dimension of the block
 */
template <int block_size>
struct storage_wrapper {
  block_reduce_storage<block_size>& storage;
  __device__ storage_wrapper(block_reduce_storage<block_size>& _temp_storage)
    : storage(_temp_storage)
  {
  }

  STORAGE_WRAPPER_GET(bool);
  STORAGE_WRAPPER_GET(int8_t);
  STORAGE_WRAPPER_GET(int16_t);
  STORAGE_WRAPPER_GET(int32_t);
  STORAGE_WRAPPER_GET(int64_t);
  STORAGE_WRAPPER_GET(__int128_t);
  STORAGE_WRAPPER_GET(uint8_t);
  STORAGE_WRAPPER_GET(uint16_t);
  STORAGE_WRAPPER_GET(uint32_t);
  STORAGE_WRAPPER_GET(uint64_t);
  STORAGE_WRAPPER_GET(float);
  STORAGE_WRAPPER_GET(double);
  STORAGE_WRAPPER_GET(string_view);
  STORAGE_WRAPPER_GET(byte_array_view);
};

#undef DECLARE_MEMBER
#undef MEMBER_NAME

}  // namespace detail
}  // namespace io
}  // namespace cudf
