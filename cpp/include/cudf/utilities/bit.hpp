/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <cassert>
#include <hip/std/climits>
#include <cudf/types.hpp>

/**
 * @file bit.hpp
 * @brief Utilities for bit and bitmask operations.
 */

namespace cudf {
namespace detail {
// @cond
// Work around a bug in NVRTC that fails to compile assert() in constexpr
// functions (fixed after CUDA 11.0)
#if defined __GNUC__
#define LIKELY(EXPR) __builtin_expect(!!(EXPR), 1)
#else
#define LIKELY(EXPR) (!!(EXPR))
#endif

#ifdef NDEBUG
#define constexpr_assert(CHECK) static_cast<void>(0)
#else
#define constexpr_assert(CHECK) (LIKELY(CHECK) ? void(0) : [] { assert(!#CHECK); }())
#endif
// @endcond

/**
 * @brief Returns the number of bits the given type can hold.
 *
 * @tparam T The type to query
 * @return `sizeof(T)` in bits
 */
template <typename T>
constexpr CUDF_HOST_DEVICE inline std::size_t size_in_bits()
{
  static_assert(CHAR_BIT == 8, "Size of a byte must be 8 bits.");
  return sizeof(T) * CHAR_BIT;
}
}  // namespace detail

/**
 * @addtogroup utility_bitmask
 * @{
 * @file
 */

/**
 * @brief Returns the index of the word containing the specified bit.
 *
 * @param bit_index The index of the bit to query
 * @return The index of the word containing the specified bit
 */
constexpr CUDF_HOST_DEVICE inline size_type word_index(size_type bit_index)
{
  return bit_index / detail::size_in_bits<bitmask_type>();
}

/**
 * @brief Returns the position within a word of the specified bit.
 *
 * @param bit_index The index of the bit to query
 * @return The position within a word of the specified bit
 */
constexpr CUDF_HOST_DEVICE inline size_type intra_word_index(size_type bit_index)
{
  return bit_index % detail::size_in_bits<bitmask_type>();
}

/**
 * @brief Sets the specified bit to `1`
 *
 * This function is not thread-safe, i.e., attempting to update bits within the
 * same word concurrently from multiple threads results in undefined behavior.
 *
 * @param bitmask The bitmask containing the bit to set
 * @param bit_index Index of the bit to set
 */
CUDF_HOST_DEVICE inline void set_bit_unsafe(bitmask_type* bitmask, size_type bit_index)
{
  assert(nullptr != bitmask);
  bitmask[word_index(bit_index)] |= (bitmask_type{1} << intra_word_index(bit_index));
}

/**
 * @brief Sets the specified bit to `0`
 *
 * This function is not thread-safe, i.e., attempting to update bits within the
 * same word concurrently from multiple threads results in undefined behavior.
 *
 * @param bitmask The bitmask containing the bit to clear
 * @param bit_index The index of the bit to clear
 */
CUDF_HOST_DEVICE inline void clear_bit_unsafe(bitmask_type* bitmask, size_type bit_index)
{
  assert(nullptr != bitmask);
  bitmask[word_index(bit_index)] &= ~(bitmask_type{1} << intra_word_index(bit_index));
}

/**
 * @brief Indicates whether the specified bit is set to `1`
 *
 * @param bitmask The bitmask containing the bit to clear
 * @param bit_index Index of the bit to test
 * @return true The specified bit is `1`
 * @return false  The specified bit is `0`
 */
CUDF_HOST_DEVICE inline bool bit_is_set(bitmask_type const* bitmask, size_type bit_index)
{
  assert(nullptr != bitmask);
  return bitmask[word_index(bit_index)] & (bitmask_type{1} << intra_word_index(bit_index));
}

/**
 * @brief optional-like interface to check if a specified bit of a bitmask is set.
 *
 * @param bitmask The bitmask containing the bit to clear
 * @param bit_index Index of the bit to test
 * @param default_value Value to return if `bitmask` is nullptr
 * @return true The specified bit is `1`
 * @return false  The specified bit is `0`
 * @return `default_value` if `bitmask` is nullptr
 */
CUDF_HOST_DEVICE inline bool bit_value_or(bitmask_type const* bitmask,
                                          size_type bit_index,
                                          bool default_value)
{
  return bitmask != nullptr ? bit_is_set(bitmask, bit_index) : default_value;
}

/**
 * @brief Returns a bitmask word with the `n` least significant bits set.
 *
 * Behavior is undefined if `n < 0` or if `n >= size_in_bits<bitmask_type>()`
 *
 * @param n The number of least significant bits to set
 * @return A bitmask word with `n` least significant bits set
 */
constexpr CUDF_HOST_DEVICE inline bitmask_type set_least_significant_bits(size_type n)
{
  constexpr_assert(0 <= n && n < static_cast<size_type>(detail::size_in_bits<bitmask_type>()));
  return ((bitmask_type{1} << n) - 1);
}

/**
 * @brief Returns a bitmask word with the `n` most significant bits set.
 *
 * Behavior is undefined if `n < 0` or if `n >= size_in_bits<bitmask_type>()`
 *
 * @param n The number of most significant bits to set
 * @return A bitmask word with `n` most significant bits set
 */
constexpr CUDF_HOST_DEVICE inline bitmask_type set_most_significant_bits(size_type n)
{
  constexpr size_type word_size{detail::size_in_bits<bitmask_type>()};
  constexpr_assert(0 <= n && n < word_size);
  // TODO(HIP/AMD): warning: shift count >= width of type
  // Check the returned value to not exceed UINT32_MAX.
  return ~((bitmask_type{1} << (word_size - n)) - 1);
}

// HIP: This method is the 32 bit version of set_most_significant_bits which is defined at line 171.
// This method is only used in cudf/cpp/src/copying/contiguous_split.hip in copy_buffer method.
// We added this implementation to avoid overwritting copy_buffer.
/**
 * @brief Returns a bitmask word with the `n` most significant bits set.
 *
 * Behavior is undefined if `n < 0` or if `n >= size_in_bits<bitmask_type>()`
 *
 * @param n The number of most significant bits to set
 * @return A bitmask word with `n` most significant bits set
 */
constexpr CUDF_HOST_DEVICE inline uint32_t set_most_significant_bits_32(size_type n)
{
  constexpr size_type word_size{detail::size_in_bits<uint32_t>()};
  constexpr_assert(0 <= n && n < word_size);
  // TODO(HIP/AMD): warning: shift count >= width of type
  // To fix the issue, we should check if the returned value does not exceed UINT32_MAX.
  // NOTE(HIP/AMD): treat UB arising for n=0 separately here (issue 175).
  return (n==0) ? 0 : ~((uint32_t{1} << (word_size - n)) - 1);
}

#if defined(__HIPCC__) || defined(__CUDACC__) 

/**
 * @brief Sets the specified bit to `1`
 *
 * @note This operation requires a global atomic operation. Therefore, it is
 * not recommended to use this function in performance critical regions. When
 * possible, it is more efficient to compute and update an entire word at
 * once using `set_word`.
 *
 * This function is thread-safe.
 *
 * @param bitmask The bitmask containing the bit to set
 * @param bit_index  Index of the bit to set
 */
__device__ inline void set_bit(bitmask_type* bitmask, size_type bit_index)
{
  assert(nullptr != bitmask);
  atomicOr(&bitmask[word_index(bit_index)], (bitmask_type{1} << intra_word_index(bit_index)));
}

/**
 * @brief Sets the specified bit to `0`
 *
 * @note This operation requires a global atomic operation. Therefore, it is
 * not recommended to use this function in performance critical regions. When
 * possible, it is more efficient to compute and update an entire element at
 * once using `set_element`.

 * This function is thread-safe.
 *
 * @param bit_index  Index of the bit to clear
 */
__device__ inline void clear_bit(bitmask_type* bitmask, size_type bit_index)
{
  assert(nullptr != bitmask);
  atomicAnd(&bitmask[word_index(bit_index)], ~(bitmask_type{1} << intra_word_index(bit_index)));
}
#endif
/** @} */  // end of group
}  // namespace cudf
