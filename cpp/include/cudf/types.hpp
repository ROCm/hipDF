/*
 * Copyright (c) 2018-2023, NVIDIA CORPORATION.
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

//: TODO(HIP/AMD) is __HIP_PLATFORM_AMD__ good replacement of  __CUDACC__ ?
#ifdef __HIP_PLATFORM_AMD__
#define CUDF_HOST_DEVICE __host__ __device__
#include "cudf/cuda_runtime.h" //: including "hip/device_functions.h" causes errors


//: TODO(HIP/AMD): ROCm does not provide __syncwarp, likely for a good reason:
// due to the lockstepping, these sync operations may simply be noopts.
// For now, we adopt the approach from the previously used warp 
// primitive extensions. More testing is required to see if we can 
// make those __syncwarps noops without breaking unit tests.
__device__ inline void __syncwarp()
{
  /* sync/barrier all threads in a warp */
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
  __builtin_amdgcn_wave_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}

__device__ inline void __syncwarp(uint64_t activemask)
{
  /* sync/barrier all threads in a warp */
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
  __builtin_amdgcn_wave_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}
#else
#define CUDF_HOST_DEVICE
#endif

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>

// NOTE(HIP/AMD): including thrust/optional.h leads to symbol conflicts when JIT-compiling
#ifndef __HIPCC_RTC__
#include <thrust/optional.h>
#endif

/**
 * @file
 * @brief Type declarations for libcudf.
 */

// Forward declarations
/// @cond
namespace rmm {
class device_buffer;
/// @endcond

}  // namespace rmm

namespace cudf {

// NOTE(HIP/AMD): This is a WAR for thrust::optional::value() not being supported in device code (internal issue 6). 
#ifdef __HIP_PLATFORM_AMD__
// NOTE(HIP/AMD): including thrust/optional.h leads to symbol conflicts when JIT-compiling
#ifndef __HIPCC_RTC__
template<typename T>
CUDF_HOST_DEVICE T inline THRUST_OPTIONAL_VALUE(thrust::optional<T> opt)
{
    return opt.value_or(T{});
}
#else
#define THRUST_OPTIONAL_VALUE(opt) opt.value()
#endif
#endif

// Forward declaration
class column;
class column_view;
class mutable_column_view;
class string_view;
class list_view;
class struct_view;

class scalar;

// clang-format off
class list_scalar;
class struct_scalar;
class string_scalar;
template <typename T> class numeric_scalar;
template <typename T> class fixed_point_scalar;
template <typename T> class timestamp_scalar;
template <typename T> class duration_scalar;

class string_scalar_device_view;
template <typename T> class numeric_scalar_device_view;
template <typename T> class fixed_point_scalar_device_view;
template <typename T> class timestamp_scalar_device_view;
template <typename T> class duration_scalar_device_view;
// clang-format on

class table;
class table_view;
class mutable_table_view;

/**
 * @addtogroup utility_types
 * @{
 * @file
 */

#ifdef __HIP_PLATFORM_AMD__  
  constexpr bool HIP_PLATFORM_AMD = true;
#else
  constexpr bool HIP_PLATFORM_AMD = false;
#endif

using size_type         = int32_t;   ///< Row index type for columns and tables
#ifdef CUDF_USE_WARPSIZE_32
using bitmask_type      = uint32_t;  ///< Bitmask type stored as 32-bit unsigned integer
#else
using bitmask_type      = uint64_t;  ///< Bitmask type stored as 64-bit unsigned integer
#endif
using valid_type        = uint8_t;   ///< Valid type in host memory
using thread_index_type = int64_t;   ///< Thread index type in kernels

constexpr unsigned bitmask_size_in_bits = sizeof(bitmask_type)*8;

#ifdef __HIP_PLATFORM_AMD__
#ifdef CUDF_USE_WARPSIZE_32
constexpr unsigned LOG2_WARPSIZE = 5;  ///< Logarithm to base 2 of wavefront size 32
#else
constexpr unsigned LOG2_WARPSIZE = 6;  ///< Logarithm to base 2 of wavefront size 64
#endif
#else
constexpr unsigned LOG2_WARPSIZE = 5;  ///< Logarithm to base 2 of warp size 32
#endif

constexpr bitmask_type LANE_MASK_ONE = 0b1;
constexpr bitmask_type LANE_MASK_TWO = 0b10;
/**
 * \return a sequence of 1s with length (t+1), followed by 0s.
 *
 * Returns the bit sequence \f$\{1\}_{0<=i<=t}, \{0\}_{t<i<n}\f$,
 * where \f$n\$f is the the number of bits of the return type.
 */
constexpr bitmask_type LANE_MASK_ALL_UNTIL_INCL(unsigned t) { return (LANE_MASK_TWO << t) - 1; }
/**
 * \return a sequence of 1's with length t, followed by 0s.
 *
 * Returns the bit sequence \f$\{1\}_{0<=i<t}, \{0\}_{t<=i<n}\f$,
 * where \f$n\$f is the the number of bits of the return type.
 */
constexpr bitmask_type LANE_MASK_ALL_UNTIL_EXCL(unsigned t) { return (LANE_MASK_ONE << t) - 1; }
/**
 * \return the full mask: all bits are set to 1.
 */
constexpr bitmask_type LANE_MASK_ALL = ~0;

/**
 * \brief Count Leading Zeros
 * \return the number of consecutive bits of highest significance that contain zeros.
 * \note Return value type matches that of the underlying device builtin.
 */
template <typename T>
__device__ inline int __CLZ(T v);

template <>
__device__ inline int __CLZ<int>(int v) {
  return __clz(v);
}

template <>
__device__ inline int __CLZ<int64_t>(int64_t v) {
  return __clzll(v);
}

template <>
__device__ inline int __CLZ<uint32_t>(uint32_t v) {
  return __clz(v);
}

template <>
__device__ inline int __CLZ<uint64_t>(uint64_t v) {
  return __clzll(v);
}

/**
 * \brief Find First Set
 * \return index of first set bit of lowest significance.
 * \note Return value type matches that of the underlying device builtin.
 * \note While `uint64_t` is defined as `unsigned long int` on x86_64,
 *       the HIP `__ffsll` device function provides `__ffsll` with `unsigned long long int`
 *       argument, which is also an 64-bit integer type on x86_64.
 *       However, the compilers typically see both as different types.
 *       We work with `uint64t` and `uint32t` here, so explicit instantiations
 *       for both are added here.
 */
template <typename T>
__device__ inline int __FFS(T v);

template <>
__device__ inline int __FFS<int32_t>(int32_t v) {
  return __ffs(v);
}

template <>
__device__ inline int __FFS<int64_t>(int64_t v) {
  return __ffsll(static_cast<unsigned long long int>(v));
}

template <>
__device__ inline int __FFS<uint32_t>(uint32_t v) {
  return __ffs(v);
}

template <>
__device__ inline int __FFS<uint64_t>(uint64_t v) {
  return __ffsll(static_cast<unsigned long long int>(v));
}

// FIXME(HIP/AMD): For HIPRTC, enabling this code yields a duplicate symbol definition,
// as unsigned long long int and uint64_t seem to be the same type.
#ifndef __HIPCC_RTC__
template <>
__device__ inline int __FFS<unsigned long long int>(unsigned long long int v) {
  return __ffsll(v);
}
#endif

/**
 * \return Number of bits set to 1.
 * \note Return value type matches that of the underlying device builtin.
 */
template <typename T>
__device__ inline int __POPC(T v);


template <>
__device__ inline int __POPC<int32_t>(int32_t v) {
  return __popc(v);
}

template <>
__device__ inline int __POPC<int64_t>(int64_t v) {
  return __popcll(v);
}

template <>
__device__ inline int __POPC<uint32_t>(uint32_t v) {
  return __popc(v);
}

template <>
__device__ inline int __POPC<uint64_t>(uint64_t v) {
  return __popcll(v);
}

//With hiprtc/jitify, uint64_t == unsigned long long int, so this would give a re-definition error.
//TODO/FIXME(HIP): use type_traits to not provide template specialization when uint64_t == unsigned long long int 
#ifndef __HIPCC_RTC__
 template <> //: On x86_64, uint64_t == unsigned long int != unsigned long long int, both have 64 bit
 __device__ inline int __POPC<unsigned long long int>(unsigned long long int v) {
   return __popcll(v);
 }
#endif

/**
 * @brief Similar to `std::distance` but returns `cudf::size_type` and performs `static_cast`
 *
 * @tparam T Iterator type
 * @param f "first" iterator
 * @param l "last" iterator
 * @return The distance between first and last
 */
template <typename T>
size_type distance(T f, T l)
{
  //TODO(HIP/AMD): HIPRTC doesn't define std::distance, while NVRTC does (even if iterator is not included!)
  // investigate further workarounds if required
  // We could e.g. add an sample implementation to the jitsafe iterator
  // header in jitify (header <iterator> is supposed to provide std::distance)
  #ifndef __HIPCC_RTC__
  return static_cast<size_type>(std::distance(f, l));
  #endif
}

/**
 * @brief Indicates the order in which elements should be sorted.
 */
enum class order : bool {
  ASCENDING,  ///< Elements ordered from small to large
  DESCENDING  ///< Elements ordered from large to small
};

/**
 * @brief Enum to specify whether to include nulls or exclude nulls
 */
enum class null_policy : bool {
  EXCLUDE,  ///< exclude null elements
  INCLUDE   ///< include null elements
};

/**
 * @brief Enum to treat NaN floating point value as null or non-null element
 */
enum class nan_policy : bool {
  NAN_IS_NULL,  ///< treat nans as null elements
  NAN_IS_VALID  ///< treat nans as valid elements (non-null)
};

/**
 * @brief Enum to consider different elements (of floating point types) holding NaN value as equal
 * or unequal
 */
enum class nan_equality /*unspecified*/ {
  ALL_EQUAL,  ///< All NaNs compare equal, regardless of sign
  UNEQUAL     ///< All NaNs compare unequal (IEEE754 behavior)
};

/**
 * @brief Enum to consider two nulls as equal or unequal
 */
enum class null_equality : bool {
  EQUAL,   ///< nulls compare equal
  UNEQUAL  ///< nulls compare unequal
};

/**
 * @brief Indicates how null values compare against all other values.
 */
enum class null_order : bool {
  AFTER,  ///< NULL values ordered *after* all other values
  BEFORE  ///< NULL values ordered *before* all other values
};

/**
 * @brief Indicates whether a collection of values is known to be sorted.
 */
enum class sorted : bool { NO, YES };

/**
 * @brief Indicates how a collection of values has been ordered.
 */
struct order_info {
  sorted is_sorted;          ///< Indicates whether the collection is sorted
  order ordering;            ///< Indicates the order in which the values are sorted
  null_order null_ordering;  ///< Indicates how null values compare against all other values
};

/**
 * @brief Controls the allocation/initialization of a null mask.
 */
enum class mask_state : int32_t {
  UNALLOCATED,    ///< Null mask not allocated, (all elements are valid)
  UNINITIALIZED,  ///< Null mask allocated, but not initialized
  ALL_VALID,      ///< Null mask allocated, initialized to all elements valid
  ALL_NULL        ///< Null mask allocated, initialized to all elements NULL
};

/**
 * @brief Interpolation method to use when the desired quantile lies between
 * two data points i and j
 */
enum class interpolation : int32_t {
  LINEAR,    ///< Linear interpolation between i and j
  LOWER,     ///< Lower data point (i)
  HIGHER,    ///< Higher data point (j)
  MIDPOINT,  ///< (i + j)/2
  NEAREST    ///< i or j, whichever is nearest
};

/**
 * @brief Identifies a column's logical element type
 */
enum class type_id : int32_t {
  EMPTY,                   ///< Always null with no underlying data
  INT8,                    ///< 1 byte signed integer
  INT16,                   ///< 2 byte signed integer
  INT32,                   ///< 4 byte signed integer
  INT64,                   ///< 8 byte signed integer
  UINT8,                   ///< 1 byte unsigned integer
  UINT16,                  ///< 2 byte unsigned integer
  UINT32,                  ///< 4 byte unsigned integer
  UINT64,                  ///< 8 byte unsigned integer
  FLOAT32,                 ///< 4 byte floating point
  FLOAT64,                 ///< 8 byte floating point
  BOOL8,                   ///< Boolean using one byte per value, 0 == false, else true
  TIMESTAMP_DAYS,          ///< point in time in days since Unix Epoch in int32
  TIMESTAMP_SECONDS,       ///< point in time in seconds since Unix Epoch in int64
  TIMESTAMP_MILLISECONDS,  ///< point in time in milliseconds since Unix Epoch in int64
  TIMESTAMP_MICROSECONDS,  ///< point in time in microseconds since Unix Epoch in int64
  TIMESTAMP_NANOSECONDS,   ///< point in time in nanoseconds since Unix Epoch in int64
  DURATION_DAYS,           ///< time interval of days in int32
  DURATION_SECONDS,        ///< time interval of seconds in int64
  DURATION_MILLISECONDS,   ///< time interval of milliseconds in int64
  DURATION_MICROSECONDS,   ///< time interval of microseconds in int64
  DURATION_NANOSECONDS,    ///< time interval of nanoseconds in int64
  DICTIONARY32,            ///< Dictionary type using int32 indices
  STRING,                  ///< String elements
  LIST,                    ///< List elements
  DECIMAL32,               ///< Fixed-point type with int32_t
  DECIMAL64,               ///< Fixed-point type with int64_t
  DECIMAL128,              ///< Fixed-point type with __int128_t
  STRUCT,                  ///< Struct elements
  // `NUM_TYPE_IDS` must be last!
  NUM_TYPE_IDS  ///< Total number of type ids
};

/**
 * @brief Indicator for the logical data type of an element in a column.
 *
 * Simple types can be entirely described by their `id()`, but some types
 * require additional metadata to fully describe elements of that type.
 */
class data_type {
 public:
  data_type()                 = default;
  ~data_type()                = default;
  data_type(data_type const&) = default;  ///< Copy constructor
  data_type(data_type&&)      = default;  ///< Move constructor

  /**
   * @brief Copy assignment operator for data_type
   *
   * @return Reference to this object
   */
  data_type& operator=(data_type const&) = default;

  /**
   * @brief Move assignment operator for data_type
   *
   * @return Reference to this object
   */
  data_type& operator=(data_type&&) = default;

  /**
   * @brief Construct a new `data_type` object
   *
   * @param id The type's identifier
   */
  explicit constexpr data_type(type_id id) : _id{id} {}

  /**
   * @brief Construct a new `data_type` object for `numeric::fixed_point`
   *
   * @param id The `fixed_point`'s identifier
   * @param scale The `fixed_point`'s scale (see `fixed_point::_scale`)
   */
  explicit data_type(type_id id, int32_t scale) : _id{id}, _fixed_point_scale{scale}
  {
    assert(id == type_id::DECIMAL32 || id == type_id::DECIMAL64 || id == type_id::DECIMAL128);
  }

  /**
   * @brief Returns the type identifier
   *
   * @return The type identifier
   */
  [[nodiscard]] constexpr type_id id() const noexcept { return _id; }

  /**
   * @brief Returns the scale (for fixed_point types)
   *
   * @return The scale
   */
  [[nodiscard]] constexpr int32_t scale() const noexcept { return _fixed_point_scale; }

 private:
  type_id _id{type_id::EMPTY};

  // Below is additional type specific metadata. Currently, only _fixed_point_scale is stored.

  int32_t _fixed_point_scale{};  // numeric::scale_type not available here, use int32_t
};

/**
 * @brief Compares two `data_type` objects for equality.
 *
 * // TODO Define exactly what it means for two `data_type`s to be equal. e.g.,
 * are two timestamps with different resolutions equal? How about decimals with
 * different scale/precision?
 *
 * @param lhs The first `data_type` to compare
 * @param rhs The second `data_type` to compare
 * @return true `lhs` is equal to `rhs`
 * @return false `lhs` is not equal to `rhs`
 */
constexpr bool operator==(data_type const& lhs, data_type const& rhs)
{
  // use std::tie in the future, breaks JITIFY currently
  return lhs.id() == rhs.id() && lhs.scale() == rhs.scale();
}

/**
 * @brief Compares two `data_type` objects for inequality.
 *
 * // TODO Define exactly what it means for two `data_type`s to be equal. e.g.,
 * are two timestamps with different resolutions equal? How about decimals with
 * different scale/precision?
 *
 * @param lhs The first `data_type` to compare
 * @param rhs The second `data_type` to compare
 * @return true `lhs` is not equal to `rhs`
 * @return false `lhs` is equal to `rhs`
 */
inline bool operator!=(data_type const& lhs, data_type const& rhs) { return !(lhs == rhs); }

/**
 * @brief Returns the size in bytes of elements of the specified `data_type`
 *
 * @note Only fixed-width types are supported
 *
 * @throws cudf::logic_error if `is_fixed_width(element_type) == false`
 *
 * @param t The `data_type` to get the size of
 * @return Size in bytes of an element of the specified `data_type`
 */
std::size_t size_of(data_type t);

/** @} */
}  // namespace cudf
