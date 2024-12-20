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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/lists/combine.hpp>

#include <stdexcept>

using namespace cudf::test::iterators;

namespace {
using StrListsCol = cudf::test::lists_column_wrapper<cudf::string_view>;
using IntListsCol = cudf::test::lists_column_wrapper<int32_t>;
using IntCol      = cudf::test::fixed_width_column_wrapper<int32_t>;

constexpr cudf::test::debug_output_level verbosity{cudf::test::debug_output_level::FIRST_ERROR};
constexpr int32_t null{0};

template <class T, class... Ts>
auto build_lists_col(T& list, Ts&... lists)
{
  return T(std::initializer_list<T>{std::move(list), std::move(lists)...});
}

}  // namespace

struct ConcatenateListElementsTest : public cudf::test::BaseFixture {};

TEST_F(ConcatenateListElementsTest, InvalidInput)
{
  // Input lists is not a 2-level depth lists column.
  {
    auto const col = IntCol{};
    EXPECT_THROW(cudf::lists::concatenate_list_elements(col), std::invalid_argument);
  }

  // Input lists is not at least 2-level depth lists column.
  {
    auto const col = IntListsCol{1, 2, 3};
    EXPECT_THROW(cudf::lists::concatenate_list_elements(col), std::invalid_argument);
  }
}

template <typename T>
struct ConcatenateListElementsTypedTest : public cudf::test::BaseFixture {};

using TypesForTest = cudf::test::Concat<cudf::test::IntegralTypesNotBool,
                                        cudf::test::FloatingPointTypes,
                                        cudf::test::FixedPointTypes>;
TYPED_TEST_SUITE(ConcatenateListElementsTypedTest, TypesForTest);

TYPED_TEST(ConcatenateListElementsTypedTest, SimpleInputNoNull)
{
  using ListsCol = cudf::test::lists_column_wrapper<TypeParam>;

  auto row0           = ListsCol{{1, 2}, {3}, {4, 5, 6}};
  auto row1           = ListsCol{ListsCol{}};
  auto row2           = ListsCol{{7, 8}, {9, 10}};
  auto const col      = build_lists_col(row0, row1, row2);
  auto const results  = cudf::lists::concatenate_list_elements(col);
  auto const expected = ListsCol{{1, 2, 3, 4, 5, 6}, {}, {7, 8, 9, 10}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, verbosity);
}

TYPED_TEST(ConcatenateListElementsTypedTest, SimpleInputNestedManyLevelsNoNull)
{
  using ListsCol = cudf::test::lists_column_wrapper<TypeParam>;

  auto row00 = ListsCol{{1, 2}, {3}, {4, 5, 6}};
  // TODO(HIP/AMD): HIP Workaround for nested Empty List
  auto row01 = ListsCol{{ListsCol{}}};
  auto row02 = ListsCol{{7, 8}, {9, 10}};
  auto row0  = build_lists_col(row00, row01, row02);


  auto row10 = ListsCol{{1, 2}, {3}, {4, 5, 6}};
  // TODO(HIP/AMD): HIP Workaround for nested Empty List
  auto row11 = ListsCol{{ListsCol{}}};
  auto row12 = ListsCol{{7, 8}, {9, 10}};
  auto row1  = build_lists_col(row10, row11, row12);

  auto row20 = ListsCol{{1, 2}, {3}, {4, 5, 6}};
  // TODO(HIP/AMD): HIP Workaround for nested Empty List
  auto row21 = ListsCol{{ListsCol{}}};
  auto row22 = ListsCol{{7, 8}, {9, 10}};
  auto row2  = build_lists_col(row20, row21, row22);

  auto const col      = build_lists_col(row0, row1, row2);
  auto const results  = cudf::lists::concatenate_list_elements(col);
  auto const expected = ListsCol{ListsCol{{1, 2}, {3}, {4, 5, 6}, {}, {7, 8}, {9, 10}},
                                 ListsCol{{1, 2}, {3}, {4, 5, 6}, {}, {7, 8}, {9, 10}},
                                 ListsCol{{1, 2}, {3}, {4, 5, 6}, {}, {7, 8}, {9, 10}}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, verbosity);
}

TEST_F(ConcatenateListElementsTest, SimpleInputStringsColumnNoNull)
{
  auto row0 = StrListsCol{StrListsCol{"Tomato", "Apple"}, StrListsCol{"Orange"}};
  auto row1 = StrListsCol{StrListsCol{"Banana", "Kiwi", "Cherry"}, StrListsCol{"Lemon", "Peach"}};
  auto row2 = StrListsCol{StrListsCol{"Coconut"}, StrListsCol{}};
  auto const col      = build_lists_col(row0, row1, row2);
  auto const results  = cudf::lists::concatenate_list_elements(col);
  auto const expected = StrListsCol{StrListsCol{"Tomato", "Apple", "Orange"},
                                    StrListsCol{"Banana", "Kiwi", "Cherry", "Lemon", "Peach"},
                                    StrListsCol{"Coconut"}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, verbosity);
}

TYPED_TEST(ConcatenateListElementsTypedTest, SimpleInputWithNulls)
{
  using ListsCol = cudf::test::lists_column_wrapper<TypeParam>;
  auto row0      = ListsCol{{ListsCol{{1, null, 3, 4}, null_at(1)},
                             ListsCol{{10, 11, 12, null}, null_at(3)},
                             ListsCol{} /*NULL*/},
                       null_at(2)};
  auto row1      = ListsCol{ListsCol{{null, 2, 3, 4}, null_at(0)},
                       ListsCol{{13, 14, 15, 16, 17, null}, null_at(5)},
                       ListsCol{{20, null}, null_at(1)}};
  auto row2      = ListsCol{{ListsCol{{null, 2, 3, 4}, null_at(0)},
                             ListsCol{} /*NULL*/,
                             ListsCol{{null, 21, null, null}, nulls_at({0, 2, 3})}},
                       null_at(1)};
  auto row3      = ListsCol{{ListsCol{} /*NULL*/, ListsCol{{null, 18}, null_at(0)}}, null_at(0)};
  auto row4      = ListsCol{ListsCol{{1, 2, null, 4}, null_at(2)},
                       ListsCol{{19, 20, null}, null_at(2)},
                       ListsCol{22, 23, 24, 25}};
  auto row5      = ListsCol{ListsCol{{1, 2, 3, null}, null_at(3)},
                       ListsCol{{null}, null_at(0)},
                       ListsCol{{null, null, null, null, null}, all_nulls()}};
  auto row6 =
    ListsCol{{ListsCol{} /*NULL*/, ListsCol{} /*NULL*/, ListsCol{} /*NULL*/}, all_nulls()};
  auto const col = build_lists_col(row0, row1, row2, row3, row4, row5, row6);

  // Ignore null list elements.
  {
    auto const results = cudf::lists::concatenate_list_elements(col);
    auto const expected =
      ListsCol{{ListsCol{{1, null, 3, 4, 10, 11, 12, null}, nulls_at({1, 7})},
                ListsCol{{null, 2, 3, 4, 13, 14, 15, 16, 17, null, 20, null}, nulls_at({0, 9, 11})},
                ListsCol{{null, 2, 3, 4, null, 21, null, null}, nulls_at({0, 4, 6, 7})},
                ListsCol{{null, 18}, null_at(0)},
                ListsCol{{1, 2, null, 4, 19, 20, null, 22, 23, 24, 25}, nulls_at({2, 6})},
                ListsCol{{1, 2, 3, null, null, null, null, null, null, null},
                         nulls_at({3, 4, 5, 6, 7, 8, 9})},
                ListsCol{} /*NULL*/},
               null_at(6)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, verbosity);
  }

  // Null lists result in null rows.
  {
    auto const results = cudf::lists::concatenate_list_elements(
      col, cudf::lists::concatenate_null_policy::NULLIFY_OUTPUT_ROW);
    auto const expected =
      ListsCol{{ListsCol{} /*NULL*/,
                ListsCol{{null, 2, 3, 4, 13, 14, 15, 16, 17, null, 20, null}, nulls_at({0, 9, 11})},
                ListsCol{} /*NULL*/,
                ListsCol{} /*NULL*/,
                ListsCol{{1, 2, null, 4, 19, 20, null, 22, 23, 24, 25}, nulls_at({2, 6})},
                ListsCol{{1, 2, 3, null, null, null, null, null, null, null},
                         nulls_at({3, 4, 5, 6, 7, 8, 9})},
                ListsCol{} /*NULL*/},
               nulls_at({0, 2, 3, 6})};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, verbosity);
  }
}

TYPED_TEST(ConcatenateListElementsTypedTest, SimpleInputNestedManyLevelsWithNulls)
{
  using ListsCol = cudf::test::lists_column_wrapper<TypeParam>;

  auto row00 = ListsCol{{1, 2}, {3}, {4, 5, 6}};
  // TODO(HIP/AMD): HIP Workaround for nested Empty List
  auto row01 = ListsCol{{ListsCol{}}}; /*NULL*/
  auto row02 = ListsCol{{7, 8}, {9, 10}};
  auto row0  = ListsCol{{std::move(row00), std::move(row01), std::move(row02)}, null_at(1)};

  auto row10 = ListsCol{{{1, 2}, {3}, {4, 5, 6} /*NULL*/}, null_at(2)};
  // TODO(HIP/AMD): HIP Workaround for nested Empty List
  auto row11 = ListsCol{{ListsCol{}}};
  auto row12 = ListsCol{{7, 8}, {9, 10}};
  auto row1  = build_lists_col(row10, row11, row12);

  auto row20 = ListsCol{{1, 2}, {3}, {4, 5, 6}};
  // TODO(HIP/AMD): HIP Workaround for nested Empty List
  auto row21 = ListsCol{{ListsCol{}}};
  auto row22 = ListsCol{ListsCol{{null, 8}, null_at(0)}, {9, 10}};
  auto row2  = build_lists_col(row20, row21, row22);

  auto const col = build_lists_col(row0, row1, row2);

  // Ignore null list elements.
  {
    auto const results = cudf::lists::concatenate_list_elements(col);
    auto const expected =
      ListsCol{ListsCol{{1, 2}, {3}, {4, 5, 6}, {7, 8}, {9, 10}},
               ListsCol{{{1, 2}, {3}, {} /*NULL*/, {}, {7, 8}, {9, 10}}, null_at(2)},
               ListsCol{{1, 2}, {3}, {4, 5, 6}, {}, ListsCol{{null, 8}, null_at(0)}, {9, 10}}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, verbosity);
  }

  // Null lists result in null rows.
  {
    auto const results = cudf::lists::concatenate_list_elements(
      col, cudf::lists::concatenate_null_policy::NULLIFY_OUTPUT_ROW);
    auto const expected =
      ListsCol{{ListsCol{ListsCol{}}, /*NULL*/
                ListsCol{{{1, 2}, {3}, {} /*NULL*/, {}, {7, 8}, {9, 10}}, null_at(2)},
                ListsCol{{1, 2}, {3}, {4, 5, 6}, {}, ListsCol{{null, 8}, null_at(0)}, {9, 10}}},
               null_at(0)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, verbosity);
  }
}

TEST_F(ConcatenateListElementsTest, SimpleInputStringsColumnWithNulls)
{
  auto row0 = StrListsCol{
    StrListsCol{{"Tomato", "Bear" /*NULL*/, "Apple"}, null_at(1)},
    StrListsCol{{"Orange", "Dog" /*NULL*/, "Fox" /*NULL*/, "Duck" /*NULL*/}, nulls_at({1, 2, 3})}};
  auto row1 = StrListsCol{
    StrListsCol{{"Banana", "Pig" /*NULL*/, "Kiwi", "Cherry", "Whale" /*NULL*/}, nulls_at({1, 4})},
    StrListsCol{"Lemon", "Peach"}};
  auto row2      = StrListsCol{{StrListsCol{"Coconut"}, StrListsCol{} /*NULL*/}, null_at(1)};
  auto const col = build_lists_col(row0, row1, row2);

  // Ignore null list elements.
  {
    auto const results  = cudf::lists::concatenate_list_elements(col);
    auto const expected = StrListsCol{
      StrListsCol{{"Tomato", "" /*NULL*/, "Apple", "Orange", "" /*NULL*/, "" /*NULL*/, ""
                   /*NULL*/},
                  nulls_at({1, 4, 5, 6})},
      StrListsCol{{"Banana", "" /*NULL*/, "Kiwi", "Cherry", "" /*NULL*/, "Lemon", "Peach"},
                  nulls_at({1, 4})},
      StrListsCol{"Coconut"}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, verbosity);
  }

  // Null lists result in null rows.
  {
    auto const results = cudf::lists::concatenate_list_elements(
      col, cudf::lists::concatenate_null_policy::NULLIFY_OUTPUT_ROW);
    auto const expected = StrListsCol{
      {StrListsCol{
         {"Tomato", "" /*NULL*/, "Apple", "Orange", "" /*NULL*/, "" /*NULL*/, "" /*NULL*/},
         nulls_at({1, 4, 5, 6})},
       StrListsCol{{"Banana", "" /*NULL*/, "Kiwi", "Cherry", "" /*NULL*/, "Lemon", "Peach"},
                   nulls_at({1, 4})},
       StrListsCol{} /*NULL*/},
      null_at(2)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, verbosity);
  }
}
TEST_F(ConcatenateListElementsTest, SimpleInputStringsColumnWithEmptyStringsAndNulls)
{
  auto row0 = StrListsCol{
    StrListsCol{"", "", ""},
    StrListsCol{{"Orange", "" /*NULL*/, "" /*NULL*/, "" /*NULL*/}, nulls_at({1, 2, 3})}};
  auto row1 = StrListsCol{
    StrListsCol{{"Banana", "" /*NULL*/, "Kiwi", "Cherry", "" /*NULL*/}, nulls_at({1, 4})},
    StrListsCol{""}};
  auto row2      = StrListsCol{{StrListsCol{"Coconut"}, StrListsCol{} /*NULL*/}, null_at(1)};
  auto const col = build_lists_col(row0, row1, row2);

  // Ignore null list elements.
  {
    auto const results  = cudf::lists::concatenate_list_elements(col);
    auto const expected = StrListsCol{
      StrListsCol{{"", "", "", "Orange", "" /*NULL*/, "" /*NULL*/, "" /*NULL*/},
                  nulls_at({4, 5, 6})},
      StrListsCol{{"Banana", "" /*NULL*/, "Kiwi", "Cherry", "" /*NULL*/, ""}, nulls_at({1, 4})},
      StrListsCol{"Coconut"}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, verbosity);
  }

  // Null lists result in null rows.
  {
    auto const results = cudf::lists::concatenate_list_elements(
      col, cudf::lists::concatenate_null_policy::NULLIFY_OUTPUT_ROW);
    auto const expected = StrListsCol{
      {StrListsCol{{"", "", "", "Orange", "" /*NULL*/, "" /*NULL*/, "" /*NULL*/},
                   nulls_at({4, 5, 6})},
       StrListsCol{{"Banana", "" /*NULL*/, "Kiwi", "Cherry", "" /*NULL*/, ""}, nulls_at({1, 4})},
       StrListsCol{} /*NULL*/},
      null_at(2)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, verbosity);
  }
}

TYPED_TEST(ConcatenateListElementsTypedTest, SlicedColumnsInputNoNull)
{
  using ListsCol = cudf::test::lists_column_wrapper<TypeParam>;

  auto const col_original = ListsCol{ListsCol{{1, 2, 3}, {2, 3}},
                                     ListsCol{{3, 4, 5, 6}, {5, 6}, {}, {7}},
                                     ListsCol{{7, 7, 7}, {7, 8, 1, 0}, {1}},
                                     ListsCol{{9, 10, 11}},
                                     ListsCol{},
                                     ListsCol{{12, 13, 14, 15}, {16}, {17}}};

  {
    auto const col     = cudf::slice(col_original, {0, 3})[0];
    auto const results = cudf::lists::concatenate_list_elements(col);
    auto const expected =
      ListsCol{{1, 2, 3, 2, 3}, {3, 4, 5, 6, 5, 6, 7}, {7, 7, 7, 7, 8, 1, 0, 1}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, verbosity);
  }
  {
    auto const col      = cudf::slice(col_original, {1, 4})[0];
    auto const results  = cudf::lists::concatenate_list_elements(col);
    auto const expected = ListsCol{{3, 4, 5, 6, 5, 6, 7}, {7, 7, 7, 7, 8, 1, 0, 1}, {9, 10, 11}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, verbosity);
  }
  {
    auto const col      = cudf::slice(col_original, {2, 5})[0];
    auto const results  = cudf::lists::concatenate_list_elements(col);
    auto const expected = ListsCol{{7, 7, 7, 7, 8, 1, 0, 1}, {9, 10, 11}, {}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, verbosity);
  }
  {
    auto const col      = cudf::slice(col_original, {3, 6})[0];
    auto const results  = cudf::lists::concatenate_list_elements(col);
    auto const expected = ListsCol{{9, 10, 11}, {}, {12, 13, 14, 15, 16, 17}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, verbosity);
  }
}

TYPED_TEST(ConcatenateListElementsTypedTest, SlicedColumnsInputWithNulls)
{
  using ListsCol = cudf::test::lists_column_wrapper<TypeParam>;

  auto row0 = ListsCol{ListsCol{{null, 2, 3}, null_at(0)}, ListsCol{2, 3}};
  auto row1 = ListsCol{ListsCol{{3, null, null, 6}, nulls_at({1, 2})},
                       ListsCol{{5, 6, null}, null_at(2)},
                       {ListsCol{}}, // TODO(HIP/AMD): Workaround for HIP
                       ListsCol{{7, null}, null_at(1)}};
  auto row2 = ListsCol{ListsCol{7, 7, 7}, ListsCol{{7, 8, null, 0}, null_at(2)}, ListsCol{1}};
  // TODO(HIP/AMD): Workaround for HIP
  auto row3 = ListsCol{{ListsCol{9, 10, 11}}};
  // TODO(HIP/AMD): Workaround for HIP
  auto row4 = ListsCol{{ListsCol{}}};
  auto row5 = ListsCol{ListsCol{{12, null, 14, 15}, null_at(1)}, ListsCol{16}, ListsCol{17}};
  auto const col_original = build_lists_col(row0, row1, row2, row3, row4, row5);

 {
    auto const col     = cudf::slice(col_original, {0, 3})[0];
    auto const results = cudf::lists::concatenate_list_elements(col);
    auto const expected =
      ListsCol{ListsCol{{null, 2, 3, 2, 3}, null_at(0)},
               ListsCol{{3, null, null, 6, 5, 6, null, 7, null}, nulls_at({1, 2, 6, 8})},
               ListsCol{{7, 7, 7, 7, 8, null, 0, 1}, null_at(5)}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, verbosity);
  }
  {
    auto const col     = cudf::slice(col_original, {1, 4})[0];
    auto const results = cudf::lists::concatenate_list_elements(col);
    auto const expected =
      ListsCol{ListsCol{{3, null, null, 6, 5, 6, null, 7, null}, nulls_at({1, 2, 6, 8})},
               ListsCol{{7, 7, 7, 7, 8, null, 0, 1}, null_at(5)},
               ListsCol{9, 10, 11}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, verbosity);
  }
  {
    auto const col     = cudf::slice(col_original, {2, 5})[0];
    auto const results = cudf::lists::concatenate_list_elements(col);
    auto const expected =
      ListsCol{ListsCol{{7, 7, 7, 7, 8, null, 0, 1}, null_at(5)}, ListsCol{9, 10, 11}, ListsCol{}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, verbosity);
  }
  {
    auto const col     = cudf::slice(col_original, {3, 6})[0];
    auto const results = cudf::lists::concatenate_list_elements(col);
    auto const expected =
      ListsCol{ListsCol{9, 10, 11}, ListsCol{}, ListsCol{{12, null, 14, 15, 16, 17}, null_at(1)}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, verbosity);
  }
}

TEST_F(ConcatenateListElementsTest, SlicedStringsColumnsInputWithNulls)
{
  auto row0 = StrListsCol{
    StrListsCol{{"Tomato", "Bear" /*NULL*/, "Apple"}, null_at(1)},
    StrListsCol{{"Banana", "Pig" /*NULL*/, "Kiwi", "Cherry", "Whale" /*NULL*/}, nulls_at({1, 4})},
    StrListsCol{"Coconut"}};
  auto row1 = StrListsCol{
    StrListsCol{{"Banana", "Pig" /*NULL*/, "Kiwi", "Cherry", "Whale" /*NULL*/}, nulls_at({1, 4})},
    StrListsCol{"Coconut"},
    StrListsCol{{"Orange", "Dog" /*NULL*/, "Fox" /*NULL*/, "Duck" /*NULL*/}, nulls_at({1, 2, 3})}};
  auto row2 = StrListsCol{
    StrListsCol{"Coconut"},
    StrListsCol{{"Orange", "Dog" /*NULL*/, "Fox" /*NULL*/, "Duck" /*NULL*/}, nulls_at({1, 2, 3})},
    StrListsCol{"Lemon", "Peach"}};
  auto row3 = StrListsCol{
    {StrListsCol{{"Orange", "Dog" /*NULL*/, "Fox" /*NULL*/, "Duck" /*NULL*/}, nulls_at({1, 2, 3})},
     StrListsCol{"Lemon", "Peach"},
     StrListsCol{} /*NULL*/},
    null_at(2)};
  auto const col_original = build_lists_col(row0, row1, row2, row3);

  {
    auto const col      = cudf::slice(col_original, {0, 2})[0];
    auto const results  = cudf::lists::concatenate_list_elements(col);
    auto const expected = StrListsCol{StrListsCol{{"Tomato",
                                                   "" /*NULL*/,
                                                   "Apple",
                                                   "Banana",
                                                   "" /*NULL*/,
                                                   "Kiwi",
                                                   "Cherry",
                                                   "" /*NULL*/,
                                                   "Coconut"},
                                                  nulls_at({1, 4, 7})},
                                      StrListsCol{{"Banana",
                                                   "" /*NULL*/,
                                                   "Kiwi",
                                                   "Cherry",
                                                   "" /*NULL*/,
                                                   "Coconut",
                                                   "Orange",
                                                   "" /*NULL*/,
                                                   "" /*NULL*/,
                                                   "" /*NULL*/},
                                                  nulls_at({1, 4, 7, 8, 9})}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, verbosity);
  }
  {
    auto const col      = cudf::slice(col_original, {1, 3})[0];
    auto const results  = cudf::lists::concatenate_list_elements(col);
    auto const expected = StrListsCol{StrListsCol{{"Banana",
                                                   "" /*NULL*/,
                                                   "Kiwi",
                                                   "Cherry",
                                                   "" /*NULL*/,
                                                   "Coconut",
                                                   "Orange",
                                                   "" /*NULL*/,
                                                   "" /*NULL*/,
                                                   "" /*NULL*/},
                                                  nulls_at({1, 4, 7, 8, 9})},
                                      StrListsCol{{"Coconut",
                                                   "Orange",
                                                   "" /*NULL*/,
                                                   "" /*NULL*/,
                                                   "", /*NULL*/
                                                   "Lemon",
                                                   "Peach"},
                                                  nulls_at({2, 3, 4})}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, verbosity);
  }
  {
    auto const col      = cudf::slice(col_original, {2, 4})[0];
    auto const results  = cudf::lists::concatenate_list_elements(col);
    auto const expected = StrListsCol{StrListsCol{{"Coconut",
                                                   "Orange",
                                                   "" /*NULL*/,
                                                   "" /*NULL*/,
                                                   "", /*NULL*/
                                                   "Lemon",
                                                   "Peach"},
                                                  nulls_at({2, 3, 4})},
                                      StrListsCol{{"Orange",
                                                   "" /*NULL*/,
                                                   "" /*NULL*/,
                                                   "", /*NULL*/
                                                   "Lemon",
                                                   "Peach"},
                                                  nulls_at({1, 2, 3})}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, verbosity);
  }
  {
    auto const col     = cudf::slice(col_original, {2, 4})[0];
    auto const results = cudf::lists::concatenate_list_elements(
      col, cudf::lists::concatenate_null_policy::NULLIFY_OUTPUT_ROW);
    auto const expected = StrListsCol{{StrListsCol{{"Coconut",
                                                    "Orange",
                                                    "" /*NULL*/,
                                                    "" /*NULL*/,
                                                    "", /*NULL*/
                                                    "Lemon",
                                                    "Peach"},
                                                   nulls_at({2, 3, 4})},
                                       StrListsCol{} /*NULL*/},
                                      null_at(1)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *results, verbosity);
  }
}

TEST_F(ConcatenateListElementsTest, ListsOfListsOfStructsNoNull)
{
  using structs_col = cudf::test::structs_column_wrapper;
  using int32s_col  = cudf::test::fixed_width_column_wrapper<int32_t>;
  using strings_col = cudf::test::strings_column_wrapper;

  // Input:
  // [ [{1, 11, "1"}, {2, 12, "2"}], [{3, 13, "3"}], [{4, 14, "4"}, {5, 15, "5"}, {6, 16, "6"}] ]
  // [ [] ]
  // [ [{7, 17, "7"}, {8, 18, "8"}], [{9, 19, "9"}, {10, 110, "10"}] ]
  auto const input = [] {
    auto child = [] {
      auto child1  = int32s_col{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
      auto child2  = int32s_col{11, 12, 13, 14, 15, 16, 17, 18, 19, 110};
      auto child3  = strings_col{"1", "2", "3", "4", "5", "6", "7", "8", "9", "10"};
      auto structs = structs_col{{child1, child2, child3}};
      auto offsets = int32s_col{0, 2, 3, 6, 6, 8, 10};
      return cudf::make_lists_column(6, offsets.release(), structs.release(), 0, {});
    }();

    auto offsets = int32s_col{0, 3, 4, 6};
    return cudf::make_lists_column(3, offsets.release(), std::move(child), 0, {});
  }();

  // Output:
  // [{1, 11, "1"}, {2, 12, "2"}, {3, 13, "3"}, {4, 14, "4"}, {5, 15, "5"}, {6, 16, "6"}]
  // []
  // [{7, 17, "7"}, {8, 18, "8"}, {9, 19, "9"}, {10, 110, "10"}]
  auto const expected = [] {
    auto child1  = int32s_col{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    auto child2  = int32s_col{11, 12, 13, 14, 15, 16, 17, 18, 19, 110};
    auto child3  = strings_col{"1", "2", "3", "4", "5", "6", "7", "8", "9", "10"};
    auto structs = structs_col{{child1, child2, child3}};
    auto offsets = int32s_col{0, 6, 6, 10};
    return cudf::make_lists_column(3, offsets.release(), structs.release(), 0, {});
  }();

  auto const results = cudf::lists::concatenate_list_elements(*input);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected, *results, verbosity);
}

TEST_F(ConcatenateListElementsTest, ListsOfListsOfStructsWithNull)
{
  using structs_col = cudf::test::structs_column_wrapper;
  using int32s_col  = cudf::test::fixed_width_column_wrapper<int32_t>;
  using strings_col = cudf::test::strings_column_wrapper;

  // Input:
  // [ [{1, 11, "1"}, {null, null, null}], [{3, 13, "3"}], NULL ]
  // [ [{4, 14, "4"}, {5, 15, "5"}, {null, null, null}] ]
  // [ [{7, 17, "7"}, {null, null, null}], [{9, 19, "9"}, {10, 110, "10"}] ]
  auto const input = [] {
    auto child = [] {
      auto child1                  = int32s_col{1, null, 3, 4, 5, null, 7, null, 9, 10};
      auto child2                  = int32s_col{11, null, 13, 14, 15, null, 17, null, 19, 110};
      auto child3                  = strings_col{"1", "", "3", "4", "5", "", "7", "", "9", "10"};
      auto structs                 = structs_col{{child1, child2, child3}, nulls_at({1, 5, 7})};
      auto offsets                 = int32s_col{0, 2, 3, 3, 6, 8, 10};
      auto const null_it           = null_at(2);  // null list
      auto [null_mask, null_count] = cudf::test::detail::make_null_mask(null_it, null_it + 6);
      return cudf::make_lists_column(
        6, offsets.release(), structs.release(), null_count, std::move(null_mask));
    }();

    auto offsets = int32s_col{0, 3, 4, 6};
    return cudf::make_lists_column(3, offsets.release(), std::move(child), 0, {});
  }();

  // Concatenate with ignoring null lists.
  {
    // Output:
    // [{1, 11, "1"}, {null, null, null}, {3, 13, "3"}]
    // [{4, 14, "4"}, {5, 15, "5"}, {null, null, null}]
    // [{7, 17, "7"}, {null, null, null}, {9, 19, "9"}, {10, 110, "10"}]
    auto const expected = [] {
      auto child1  = int32s_col{1, null, 3, 4, 5, null, 7, null, 9, 10};
      auto child2  = int32s_col{11, null, 13, 14, 15, null, 17, null, 19, 110};
      auto child3  = strings_col{"1", "", "3", "4", "5", "", "7", "", "9", "10"};
      auto structs = structs_col{{child1, child2, child3}, nulls_at({1, 5, 7})};
      auto offsets = int32s_col{0, 3, 6, 10};
      return cudf::make_lists_column(3, offsets.release(), structs.release(), 0, {});
    }();

    auto const results = cudf::lists::concatenate_list_elements(*input);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected, *results, verbosity);
  }

  // Concatenate with ignoring null lists and sliced input.
  {
    // Output:
    // [{4, 14, "4"}, {5, 15, "5"}, {null, null, null}]
    auto const expected = [] {
      auto child1  = int32s_col{4, 5, null};
      auto child2  = int32s_col{14, 15, null};
      auto child3  = strings_col{"4", "5", ""};
      auto structs = structs_col{{child1, child2, child3}, null_at(2)};
      auto offsets = int32s_col{0, 3};
      return cudf::make_lists_column(1, offsets.release(), structs.release(), 0, {});
    }();

    auto const sliced_input = cudf::slice(*input, {1, 2})[0];
    auto const results      = cudf::lists::concatenate_list_elements(sliced_input);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected, *results, verbosity);
  }

  // Concatenate with `concatenate_null_policy::NULLIFY_OUTPUT_ROW`.
  {
    // Output:
    // NULL
    // [{4, 14, "4"}, {5, 15, "5"}, {null, null, null}]
    // [{7, 17, "7"}, {null, null, null}, {9, 19, "9"}, {10, 110, "10"}]
    auto const expected = [] {
      auto child1                  = int32s_col{4, 5, null, 7, null, 9, 10};
      auto child2                  = int32s_col{14, 15, null, 17, null, 19, 110};
      auto child3                  = strings_col{"4", "5", "", "7", "", "9", "10"};
      auto structs                 = structs_col{{child1, child2, child3}, nulls_at({2, 4})};
      auto offsets                 = int32s_col{0, 0, 3, 7};
      auto const null_it           = null_at(0);  // null row
      auto [null_mask, null_count] = cudf::test::detail::make_null_mask(null_it, null_it + 3);
      return cudf::make_lists_column(
        3, offsets.release(), structs.release(), null_count, std::move(null_mask));
    }();

    auto const results = cudf::lists::concatenate_list_elements(
      *input, cudf::lists::concatenate_null_policy::NULLIFY_OUTPUT_ROW);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected, *results, verbosity);
  }

  // Concatenate with `concatenate_null_policy::NULLIFY_OUTPUT_ROW` and sliced input.
  {
    // Output:
    // NULL
    // [{4, 14, "4"}, {5, 15, "5"}, {null, null, null}]
    auto const expected = [] {
      auto child1                  = int32s_col{4, 5, null};
      auto child2                  = int32s_col{14, 15, null};
      auto child3                  = strings_col{"4", "5", ""};
      auto structs                 = structs_col{{child1, child2, child3}, null_at(2)};
      auto offsets                 = int32s_col{0, 0, 3};
      auto const null_it           = null_at(0);  // null row
      auto [null_mask, null_count] = cudf::test::detail::make_null_mask(null_it, null_it + 2);
      return cudf::make_lists_column(
        2, offsets.release(), structs.release(), null_count, std::move(null_mask));
    }();

    auto const sliced_input = cudf::slice(*input, {0, 2})[0];
    auto const results      = cudf::lists::concatenate_list_elements(
      sliced_input, cudf::lists::concatenate_null_policy::NULLIFY_OUTPUT_ROW);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected, *results, verbosity);
  }
}

TEST_F(ConcatenateListElementsTest, ListsOfListsOfStructsHavingListsNoNull)
{
  using structs_col = cudf::test::structs_column_wrapper;
  using int32s_col  = cudf::test::fixed_width_column_wrapper<int32_t>;
  using lists_col   = cudf::test::lists_column_wrapper<int32_t>;

  // clang-format off
  // Input:
  // [ [{1, 11, [1, 1]}, {2, 12, [2]}], [{3, 13, [3, 3]}], [{4, 14, []}, {5, 15, [5, 5, 5]}, {6, 16, [6, 6]}] ]
  // [ [] ]
  // [ [{7, 17, [7]}, {8, 18, [8]}], [{9, 19, [9, 9]}, {10, 110, [10, 10, 10, 10]}] ]
  // clang-format on
  auto const input = [] {
    auto child = [] {
      auto child1 = int32s_col{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
      auto child2 = int32s_col{11, 12, 13, 14, 15, 16, 17, 18, 19, 110};
      auto child3 =
        lists_col{{1, 1}, {2}, {3, 3}, {}, {5, 5, 5}, {6, 6}, {7}, {8}, {9, 9}, {10, 10, 10, 10}};
      auto structs = structs_col{{child1, child2, child3}};
      auto offsets = int32s_col{0, 2, 3, 6, 6, 8, 10};
      return cudf::make_lists_column(6, offsets.release(), structs.release(), 0, {});
    }();

    auto offsets = int32s_col{0, 3, 4, 6};
    return cudf::make_lists_column(3, offsets.release(), std::move(child), 0, {});
  }();

  // clang-format off
  // Output:
  // [{1, 11, [1, 1]}, {2, 12, [2]}, {3, 13, [3, 3]}, {4, 14, []}, {5, 15, [5, 5, 5]}, {6, 16, [6, 6]}]
  // []
  // [{7, 17, [7]}, {8, 18, [8]}, {9, 19, [9, 9]}, {10, 110, [10, 10, 10, 10]}]
  // clang-format on
  auto const expected = [] {
    auto child1 = int32s_col{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    auto child2 = int32s_col{11, 12, 13, 14, 15, 16, 17, 18, 19, 110};
    auto child3 =
      lists_col{{1, 1}, {2}, {3, 3}, {}, {5, 5, 5}, {6, 6}, {7}, {8}, {9, 9}, {10, 10, 10, 10}};
    auto structs = structs_col{{child1, child2, child3}};
    auto offsets = int32s_col{0, 6, 6, 10};
    return cudf::make_lists_column(3, offsets.release(), structs.release(), 0, {});
  }();

  auto const results = cudf::lists::concatenate_list_elements(*input);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected, *results, verbosity);
}

TEST_F(ConcatenateListElementsTest, ListsOfListsOfStructsHavingListsWithNulls)
{
  using structs_col = cudf::test::structs_column_wrapper;
  using int32s_col  = cudf::test::fixed_width_column_wrapper<int32_t>;
  using lists_col   = cudf::test::lists_column_wrapper<int32_t>;

  // Input:
  // [ [{1, 11, [1, 1]}, {2, 12, [2]}], [{3, 13, [3, 3]}] ]
  // [ [{4, 14, null}, {5, 15, [5, 5, 5]}, {6, 16, [6, 6]}], NULL ]
  // [ [{7, 17, [7]}, {8, 18, [8]}], [{9, 19, [9, 9]}, {10, 110, [10, 10, 10, 10]}] ]
  auto const input = [] {
    auto child = [] {
      auto child1 = int32s_col{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
      auto child2 = int32s_col{11, 12, 13, 14, 15, 16, 17, 18, 19, 110};
      auto child3 =
        lists_col{{{1, 1}, {2}, {3, 3}, {}, {5, 5, 5}, {6, 6}, {7}, {8}, {9, 9}, {10, 10, 10, 10}},
                  null_at(3)};
      auto structs                 = structs_col{{child1, child2, child3}};
      auto offsets                 = int32s_col{0, 2, 3, 6, 6, 8, 10};
      auto const null_it           = null_at(3);  // null list
      auto [null_mask, null_count] = cudf::test::detail::make_null_mask(null_it, null_it + 6);
      return cudf::make_lists_column(
        6, offsets.release(), structs.release(), null_count, std::move(null_mask));
    }();

    auto offsets = int32s_col{0, 2, 4, 6};
    return cudf::make_lists_column(3, offsets.release(), std::move(child), 0, {});
  }();

  // Concatenate with ignoring null lists.
  {
    // Output:
    // [{1, 11, [1, 1]}, {2, 12, [2]}, {3, 13, [3, 3]}]
    // [{4, 14, null}, {5, 15, [5, 5, 5]}, {6, 16, [6, 6]}]
    // [{7, 17, [7]}, {8, 18, [8]}, {9, 19, [9, 9]}, {10, 110, [10, 10, 10, 10]}]
    auto const expected = [] {
      auto child1 = int32s_col{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
      auto child2 = int32s_col{11, 12, 13, 14, 15, 16, 17, 18, 19, 110};
      auto child3 =
        lists_col{{{1, 1}, {2}, {3, 3}, {}, {5, 5, 5}, {6, 6}, {7}, {8}, {9, 9}, {10, 10, 10, 10}},
                  null_at(3)};
      auto structs = structs_col{{child1, child2, child3}};
      auto offsets = int32s_col{0, 3, 6, 10};
      return cudf::make_lists_column(3, offsets.release(), structs.release(), 0, {});
    }();

    auto const results = cudf::lists::concatenate_list_elements(*input);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected, *results, verbosity);
  }

  // Concatenate with ignoring null lists and sliced input.
  {
    // Output:
    // [{4, 14, null}, {5, 15, [5, 5, 5]}, {6, 16, [6, 6]}]
    auto const expected = [] {
      auto child1  = int32s_col{4, 5, 6};
      auto child2  = int32s_col{14, 15, 16};
      auto child3  = lists_col{{{}, {5, 5, 5}, {6, 6}}, null_at(0)};
      auto structs = structs_col{{child1, child2, child3}};
      auto offsets = int32s_col{0, 3};
      return cudf::make_lists_column(1, offsets.release(), structs.release(), 0, {});
    }();

    auto const sliced_input = cudf::slice(*input, {1, 2})[0];
    auto const results      = cudf::lists::concatenate_list_elements(sliced_input);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected, *results, verbosity);
  }

  // Concatenate with `concatenate_null_policy::NULLIFY_OUTPUT_ROW`.
  {
    // Output:
    // [{1, 11, [1, 1]}, {2, 12, [2]}, {3, 13, [3, 3]}]
    // NULL
    // [{7, 17, [7]}, {8, 18, [8]}, {9, 19, [9, 9]}, {10, 110, [10, 10, 10, 10]}]
    auto const expected = [] {
      auto child1 = int32s_col{1, 2, 3, 7, 8, 9, 10};
      auto child2 = int32s_col{11, 12, 13, 17, 18, 19, 110};
      auto child3 =
        lists_col{{{1, 1}, {2}, {3, 3}, {7}, {8}, {9, 9}, {10, 10, 10, 10}}, no_nulls()};
      auto structs                 = structs_col{{child1, child2, child3}};
      auto offsets                 = int32s_col{0, 3, 3, 7};
      auto const null_it           = null_at(1);  // null row
      auto [null_mask, null_count] = cudf::test::detail::make_null_mask(null_it, null_it + 3);
      return cudf::make_lists_column(
        3, offsets.release(), structs.release(), null_count, std::move(null_mask));
    }();

    auto const results = cudf::lists::concatenate_list_elements(
      *input, cudf::lists::concatenate_null_policy::NULLIFY_OUTPUT_ROW);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected, *results, verbosity);
  }

  // Concatenate with `concatenate_null_policy::NULLIFY_OUTPUT_ROW` and sliced input.
  {
    // Output:
    // NULL
    // [{7, 17, [7]}, {8, 18, [8]}, {9, 19, [9, 9]}, {10, 110, [10, 10, 10, 10]}]
    auto const expected = [] {
      auto child1                  = int32s_col{7, 8, 9, 10};
      auto child2                  = int32s_col{17, 18, 19, 110};
      auto child3                  = lists_col{{{7}, {8}, {9, 9}, {10, 10, 10, 10}}, no_nulls()};
      auto structs                 = structs_col{{child1, child2, child3}};
      auto offsets                 = int32s_col{0, 0, 4};
      auto const null_it           = null_at(0);  // null row
      auto [null_mask, null_count] = cudf::test::detail::make_null_mask(null_it, null_it + 2);
      return cudf::make_lists_column(
        2, offsets.release(), structs.release(), null_count, std::move(null_mask));
    }();

    auto const sliced_input = cudf::slice(*input, {1, 3})[0];
    auto const results      = cudf::lists::concatenate_list_elements(
      sliced_input, cudf::lists::concatenate_null_policy::NULLIFY_OUTPUT_ROW);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected, *results, verbosity);
  }
}
