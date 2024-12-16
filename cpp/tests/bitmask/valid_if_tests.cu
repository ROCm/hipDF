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

#include <cudf/detail/iterator.cuh>
#include <cudf/detail/valid_if.cuh>
#include <cudf/types.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>

#include <thrust/iterator/counting_iterator.h>

struct ValidIfTest : public cudf::test::BaseFixture {};

struct odds_valid {
  __host__ __device__ bool operator()(cudf::size_type i) { return i % 2; }
};
struct all_valid {
  __host__ __device__ bool operator()(cudf::size_type i) { return true; }
};
struct all_null {
  __host__ __device__ bool operator()(cudf::size_type i) { return false; }
};

TEST_F(ValidIfTest, EmptyRange)
{
  auto actual        = cudf::detail::valid_if(thrust::make_counting_iterator(0),
                                       thrust::make_counting_iterator(0),
                                       odds_valid{},
                                       cudf::get_default_stream(),
                                       rmm::mr::get_current_device_resource());
  auto const& buffer = actual.first;
  EXPECT_EQ(0u, buffer.size());
  EXPECT_EQ(nullptr, buffer.data());
  EXPECT_EQ(0, actual.second);
}

TEST_F(ValidIfTest, InvalidRange)
{
  EXPECT_THROW(cudf::detail::valid_if(thrust::make_counting_iterator(1),
                                      thrust::make_counting_iterator(0),
                                      odds_valid{},
                                      cudf::get_default_stream(),
                                      rmm::mr::get_current_device_resource()),
               cudf::logic_error);
}
// NOTE(HIP/AMD): This test has been modified to avoid using memset to zeros
TEST_F(ValidIfTest, OddsValid)
{
  auto iter     = cudf::detail::make_counting_transform_iterator(0, odds_valid{});
  auto expected = cudf::test::detail::make_null_mask(iter, iter + 10240);
  auto actual   = cudf::detail::valid_if(thrust::make_counting_iterator(0),
                                       thrust::make_counting_iterator(10240),
                                       odds_valid{},
                                       cudf::get_default_stream(),
                                       rmm::mr::get_current_device_resource());
  CUDF_TEST_EXPECT_EQUAL_BUFFERS(expected.first.data(), actual.first.data(), expected.first.size());
  EXPECT_EQ(5120, actual.second);
  EXPECT_EQ(expected.second, actual.second);
}
// NOTE(HIP/AMD): This test has been modified to avoid using memset to zeors
TEST_F(ValidIfTest, AllValid)
{
  auto iter     = cudf::detail::make_counting_transform_iterator(0, all_valid{});
  auto expected = cudf::test::detail::make_null_mask(iter, iter + 10240);
  auto actual   = cudf::detail::valid_if(thrust::make_counting_iterator(0),
                                       thrust::make_counting_iterator(10240),
                                       all_valid{},
                                       cudf::get_default_stream(),
                                       rmm::mr::get_current_device_resource());
  CUDF_TEST_EXPECT_EQUAL_BUFFERS(expected.first.data(), actual.first.data(), expected.first.size());
  EXPECT_EQ(0, actual.second);
  EXPECT_EQ(expected.second, actual.second);
}
// NOTE(HIP/AMD): This test has been modified to avoid using memset to zeors
TEST_F(ValidIfTest, AllNull)
{
  auto iter     = cudf::detail::make_counting_transform_iterator(0, all_null{});
  auto expected = cudf::test::detail::make_null_mask(iter, iter + 10240);
  auto actual   = cudf::detail::valid_if(thrust::make_counting_iterator(0),
                                       thrust::make_counting_iterator(10240),
                                       all_null{},
                                       cudf::get_default_stream(),
                                       rmm::mr::get_current_device_resource());
  CUDF_TEST_EXPECT_EQUAL_BUFFERS(expected.first.data(), actual.first.data(), expected.first.size());
  EXPECT_EQ(10240, actual.second);
  EXPECT_EQ(expected.second, actual.second);
}
