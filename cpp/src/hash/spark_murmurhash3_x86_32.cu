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
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/hashing/detail/hash_functions.cuh>
#include <cudf/hashing/detail/hashing.hpp>
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/table/table_device_view.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/tabulate.h>

namespace cudf {
namespace hashing {
namespace detail {

namespace {

using spark_hash_value_type = int32_t;

template <typename Key> //TODO(HIP/AMD): re-enable: CUDF_ENABLE_IF(not cudf::is_nested<Key>())>
                        //TODO(HIP/AMD): template instantiation of  "DeviceRowHasher<hash_function, Nullate> device_hasher()" appears to fail for hash function Spark_MurmurHash3_x86_32 if it uses the CUDF_ENABLE_IF
struct Spark_MurmurHash3_x86_32 {
  using result_type = spark_hash_value_type;

  constexpr Spark_MurmurHash3_x86_32() = default;
  constexpr Spark_MurmurHash3_x86_32(uint32_t seed) : m_seed(seed) {}

  [[nodiscard]] __device__ inline uint32_t fmix32(uint32_t h) const
  {
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
  }

  [[nodiscard]] __device__ inline uint32_t getblock32(std::byte const* data,
                                                      cudf::size_type offset) const
  {
    // Read a 4-byte value from the data pointer as individual bytes for safe
    // unaligned access (very likely for string types).
    auto block = reinterpret_cast<uint8_t const*>(data + offset);
    return block[0] | (block[1] << 8) | (block[2] << 16) | (block[3] << 24);
  }

  [[nodiscard]] result_type __device__ inline operator()(Key const& key) const
  {
    return compute(key);
  }

  template <typename T>
  result_type __device__ inline compute(T const& key) const
  {
    return compute_bytes(reinterpret_cast<std::byte const*>(&key), sizeof(T));
  }

  result_type __device__ inline compute_remaining_bytes(std::byte const* data,
                                                        cudf::size_type len,
                                                        cudf::size_type tail_offset,
                                                        result_type h) const
  {
    // Process remaining bytes that do not fill a four-byte chunk using Spark's approach
    // (does not conform to normal MurmurHash3).
    for (auto i = tail_offset; i < len; i++) {
      // We require a two-step cast to get the k1 value from the byte. First,
      // we must cast to a signed int8_t. Then, the sign bit is preserved when
      // casting to uint32_t under 2's complement. Java preserves the sign when
      // casting byte-to-int, but C++ does not.
      uint32_t k1 = static_cast<uint32_t>(std::to_integer<int8_t>(data[i]));
      k1 *= c1;
      k1 = rotate_bits_left(k1, rot_c1);
      k1 *= c2;
      h ^= k1;
      h = rotate_bits_left(static_cast<uint32_t>(h), rot_c2);
      h = h * 5 + c3;
    }
    return h;
  }

  result_type __device__ compute_bytes(std::byte const* data, cudf::size_type const len) const
  {
    constexpr cudf::size_type BLOCK_SIZE = 4;
    cudf::size_type const nblocks        = len / BLOCK_SIZE;
    cudf::size_type const tail_offset    = nblocks * BLOCK_SIZE;
    result_type h                        = m_seed;

    // Process all four-byte chunks.
    for (cudf::size_type i = 0; i < nblocks; i++) {
      uint32_t k1 = getblock32(data, i * BLOCK_SIZE);
      k1 *= c1;
      k1 = rotate_bits_left(k1, rot_c1);
      k1 *= c2;
      h ^= k1;
      h = rotate_bits_left(static_cast<uint32_t>(h), rot_c2);
      h = h * 5 + c3;
    }

    h = compute_remaining_bytes(data, len, tail_offset, h);

    // Finalize hash.
    h ^= len;
    h = fmix32(h);
    return h;
  }

 private:
  uint32_t m_seed{cudf::DEFAULT_HASH_SEED};
  static constexpr uint32_t c1     = 0xcc9e2d51;
  static constexpr uint32_t c2     = 0x1b873593;
  static constexpr uint32_t c3     = 0xe6546b64;
  static constexpr uint32_t rot_c1 = 15;
  static constexpr uint32_t rot_c2 = 13;
};

template <>
spark_hash_value_type __device__ inline Spark_MurmurHash3_x86_32<bool>::operator()(
  bool const& key) const
{
  return compute<uint32_t>(key);
}

template <>
spark_hash_value_type __device__ inline Spark_MurmurHash3_x86_32<int8_t>::operator()(
  int8_t const& key) const
{
  return compute<uint32_t>(key);
}

template <>
spark_hash_value_type __device__ inline Spark_MurmurHash3_x86_32<uint8_t>::operator()(
  uint8_t const& key) const
{
  return compute<uint32_t>(key);
}

template <>
spark_hash_value_type __device__ inline Spark_MurmurHash3_x86_32<int16_t>::operator()(
  int16_t const& key) const
{
  return compute<uint32_t>(key);
}

template <>
spark_hash_value_type __device__ inline Spark_MurmurHash3_x86_32<uint16_t>::operator()(
  uint16_t const& key) const
{
  return compute<uint32_t>(key);
}

template <>
spark_hash_value_type __device__ inline Spark_MurmurHash3_x86_32<float>::operator()(
  float const& key) const
{
  return compute<float>(normalize_nans(key));
}

template <>
spark_hash_value_type __device__ inline Spark_MurmurHash3_x86_32<double>::operator()(
  double const& key) const
{
  return compute<double>(normalize_nans(key));
}

template <>
spark_hash_value_type __device__ inline Spark_MurmurHash3_x86_32<cudf::string_view>::operator()(
  cudf::string_view const& key) const
{
  auto const data = reinterpret_cast<std::byte const*>(key.data());
  auto const len  = key.size_bytes();
  return compute_bytes(data, len);
}

template <>
spark_hash_value_type __device__ inline Spark_MurmurHash3_x86_32<numeric::decimal32>::operator()(
  numeric::decimal32 const& key) const
{
  return compute<uint64_t>(key.value());
}

template <>
spark_hash_value_type __device__ inline Spark_MurmurHash3_x86_32<numeric::decimal64>::operator()(
  numeric::decimal64 const& key) const
{
  return compute<uint64_t>(key.value());
}

template <>
spark_hash_value_type __device__ inline Spark_MurmurHash3_x86_32<numeric::decimal128>::operator()(
  numeric::decimal128 const& key) const
{
  // Generates the Spark MurmurHash3 hash value, mimicking the conversion:
  // java.math.BigDecimal.valueOf(unscaled_value, _scale).unscaledValue().toByteArray()
  // https://github.com/apache/spark/blob/master/sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/expressions/hash.scala#L381
  __int128_t const val               = key.value();
  constexpr cudf::size_type key_size = sizeof(__int128_t);
  std::byte const* data              = reinterpret_cast<std::byte const*>(&val);

  // Small negative values start with 0xff..., small positive values start with 0x00...
  bool const is_negative     = val < 0;
  std::byte const zero_value = is_negative ? std::byte{0xff} : std::byte{0x00};

  // If the value can be represented with a shorter than 16-byte integer, the
  // leading bytes of the little-endian value are truncated and are not hashed.
  auto const reverse_begin = thrust::reverse_iterator(data + key_size);
  auto const reverse_end   = thrust::reverse_iterator(data);
  auto const first_nonzero_byte =
    thrust::find_if_not(thrust::seq, reverse_begin, reverse_end, [zero_value](std::byte const& v) {
      return v == zero_value;
    }).base();
  // Max handles special case of 0 and -1 which would shorten to 0 length otherwise
  cudf::size_type length =
    std::max(1, static_cast<cudf::size_type>(thrust::distance(data, first_nonzero_byte)));

  // Preserve the 2's complement sign bit by adding a byte back on if necessary.
  // e.g. 0x0000ff would shorten to 0x00ff. The 0x00 byte is retained to
  // preserve the sign bit, rather than leaving an "f" at the front which would
  // change the sign bit. However, 0x00007f would shorten to 0x7f. No extra byte
  // is needed because the leftmost bit matches the sign bit. Similarly for
  // negative values: 0xffff00 --> 0xff00 and 0xffff80 --> 0x80.
  if ((length < key_size) && (is_negative ^ bool(data[length - 1] & std::byte{0x80}))) { ++length; }

  // Convert to big endian by reversing the range of nonzero bytes. Only those bytes are hashed.
  __int128_t big_endian_value = 0;
  auto big_endian_data        = reinterpret_cast<std::byte*>(&big_endian_value);
  thrust::reverse_copy(thrust::seq, data, data + length, big_endian_data);
  return compute_bytes(big_endian_data, length);
}

/**
 * @brief Computes the hash value of a row in the given table.
 *
 * This functor uses Spark conventions for Murmur hashing, which differs from
 * the Murmur implementation used in the rest of libcudf. These differences
 * include:
 * - Serially using the output hash as an input seed for the next item
 * - Ignorance of null values
 *
 * The serial use of hashes as seeds means that data of different nested types
 * can exhibit hash collisions. For example, a row of an integer column
 * containing a 1 will have the same hash as a lists column of integers
 * containing a list of [1] and a struct column of a single integer column
 * containing a struct of {1}.
 *
 * As a consequence of ignoring null values, inputs like [1], [1, null], and
 * [null, 1] have the same hash (an expected hash collision). This kind of
 * collision can also occur across a table of nullable columns and with nulls
 * in structs ({1, null} and {null, 1} have the same hash). The seed value (the
 * previous element's hash value) is returned as the hash if an element is
 * null.
 *
 * For additional differences such as special tail processing and decimal type
 * handling, refer to the Spark_MurmurHash3_x86_32 functor.
 *
 * @tparam hash_function Hash functor to use for hashing elements. Must be Spark_MurmurHash3_x86_32.
 * @tparam Nullate A cudf::nullate type describing whether to check for nulls.
 */
template <template <typename> class hash_function, typename Nullate>
class spark_murmur_device_row_hasher {
  friend class cudf::experimental::row::hash::row_hasher;  ///< Allow row_hasher to access private
                                                           ///< members.

 public:
  /**
   * @brief Return the hash value of a row in the given table.
   *
   * @param row_index The row index to compute the hash value of
   * @return The hash value of the row
   */
  __device__ auto operator()(size_type row_index) const noexcept
  {
    return cudf::detail::accumulate(
      _table.begin(),
      _table.end(),
      _seed,
      [row_index, nulls = this->_check_nulls] __device__(auto hash, auto column) {
        return cudf::type_dispatcher(
          column.type(), element_hasher_adapter<hash_function>{nulls, hash}, column, row_index);
      });
  }

 private:
  /**
   * @brief Computes the hash value of an element in the given column.
   *
   * When the column is non-nested, this is a simple wrapper around the element_hasher.
   * When the column is nested, this uses a seed value to serially compute each
   * nested element, with the output hash becoming the seed for the next value.
   * This requires constructing a new hash functor for each nested element,
   * using the new seed from the previous element's hash. The hash of a null
   * element is the input seed (the previous element's hash).
   */
  template <template <typename> class hash_fn>
  class element_hasher_adapter {
   public:
    __device__ element_hasher_adapter(Nullate check_nulls, uint32_t seed) noexcept
      : _check_nulls(check_nulls), _seed(seed)
    {
    }

    using hash_functor = cudf::experimental::row::hash::element_hasher<hash_fn, Nullate>;

    template <typename T, CUDF_ENABLE_IF(not cudf::is_nested<T>())>
    __device__ spark_hash_value_type operator()(column_device_view const& col,
                                                size_type row_index) const noexcept
    {
      auto const hasher = hash_functor{_check_nulls, _seed, _seed};
      return hasher.template operator()<T>(col, row_index);
    }

    template <typename T, CUDF_ENABLE_IF(cudf::is_nested<T>())>
    __device__ spark_hash_value_type operator()(column_device_view const& col,
                                                size_type row_index) const noexcept
    {
      column_device_view curr_col = col.slice(row_index, 1);
      while (curr_col.type().id() == type_id::STRUCT || curr_col.type().id() == type_id::LIST) {
        if (curr_col.type().id() == type_id::STRUCT) {
          if (curr_col.num_child_columns() == 0) { return _seed; }
          // Non-empty structs are assumed to be decomposed and contain only one child
          curr_col = cudf::detail::structs_column_device_view(curr_col).get_sliced_child(0);
        } else if (curr_col.type().id() == type_id::LIST) {
          curr_col = cudf::detail::lists_column_device_view(curr_col).get_sliced_child();
        }
      }

      return cudf::detail::accumulate(
        thrust::counting_iterator(0),
        thrust::counting_iterator(curr_col.size()),
        _seed,
        [curr_col, nulls = this->_check_nulls] __device__(auto hash, auto element_index) {
          auto const hasher = hash_functor{nulls, hash, hash};
          return cudf::type_dispatcher<cudf::experimental::dispatch_void_if_nested>(
            curr_col.type(), hasher, curr_col, element_index);
        });
    }

    Nullate const _check_nulls;  ///< Whether to check for nulls
    uint32_t const _seed;        ///< The seed to use for hashing, also returned for null elements
  };

  CUDF_HOST_DEVICE spark_murmur_device_row_hasher(Nullate check_nulls,
                                                  table_device_view t,
                                                  uint32_t seed = DEFAULT_HASH_SEED) noexcept
    : _check_nulls{check_nulls}, _table{t}, _seed(seed)
  {
    // Error out if passed an unsupported hash_function
    static_assert(
      std::is_base_of_v<Spark_MurmurHash3_x86_32<int>, hash_function<int>>,
      "spark_murmur_device_row_hasher only supports the Spark_MurmurHash3_x86_32 hash function");
  }

  Nullate const _check_nulls;
  table_device_view const _table;
  uint32_t const _seed;
};

void check_hash_compatibility(table_view const& input)
{
  using column_checker_fn_t = std::function<void(column_view const&)>;

  column_checker_fn_t check_column = [&](column_view const& c) {
    if (c.type().id() == type_id::LIST) {
      auto const& list_col = lists_column_view(c);
      CUDF_EXPECTS(list_col.child().type().id() != type_id::STRUCT,
                   "Cannot compute hash of a table with a LIST of STRUCT columns.");
      check_column(list_col.child());
    } else if (c.type().id() == type_id::STRUCT) {
      for (auto child = c.child_begin(); child != c.child_end(); ++child) {
        check_column(*child);
      }
    }
  };

  for (column_view const& c : input) {
    check_column(c);
  }
}

}  // namespace

std::unique_ptr<column> spark_murmurhash3_x86_32(table_view const& input,
                                                 uint32_t seed,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::mr::device_memory_resource* mr)
{
  auto output = make_numeric_column(data_type(type_to_id<spark_hash_value_type>()),
                                    input.num_rows(),
                                    mask_state::UNALLOCATED,
                                    stream,
                                    mr);

  // Return early if there's nothing to hash
  if (input.num_columns() == 0 || input.num_rows() == 0) { return output; }

  // Lists of structs are not supported
  check_hash_compatibility(input);

  bool const nullable   = has_nested_nulls(input);
  auto const row_hasher = cudf::experimental::row::hash::row_hasher(input, stream);
  auto output_view      = output->mutable_view();

  // Compute the hash value for each row
  thrust::tabulate(
    rmm::exec_policy(stream),
    output_view.begin<spark_hash_value_type>(),
    output_view.end<spark_hash_value_type>(),
    row_hasher.device_hasher<Spark_MurmurHash3_x86_32, spark_murmur_device_row_hasher>(nullable,
                                                                                       seed));

  return output;
}

}  // namespace detail

std::unique_ptr<column> spark_murmurhash3_x86_32(table_view const& input,
                                                 uint32_t seed,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::spark_murmurhash3_x86_32(input, seed, stream, mr);
}

}  // namespace hashing
}  // namespace cudf
