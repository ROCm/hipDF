/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/char_types/char_types.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <rmm/mr/device/per_device_resource.hpp>

namespace cudf {
namespace strings {
/**
 * @addtogroup strings_case
 * @{
 * @file
 */

/**
 * @brief Returns a column of capitalized strings.
 *
 * If the `delimiters` is an empty string, then only the first character of each
 * row is capitalized. Otherwise, a non-delimiter character is capitalized after
 * any delimiter character is found.
 *
 * @code{.pseudo}
 * Example:
 * input = ["tesT1", "a Test", "Another Test", "a\tb"];
 * output = capitalize(input)
 * output is ["Test1", "A test", "Another test", "A\tb"]
 * output = capitalize(input, " ")
 * output is ["Test1", "A Test", "Another Test", "A\tb"]
 * output = capitalize(input, " \t")
 * output is ["Test1", "A Test", "Another Test", "A\tB"]
 * @endcode
 *
 * Any null string entries return corresponding null output column entries.
 *
 * @throw cudf::logic_error if `delimiter.is_valid()` is `false`.
 *
 * @param input String column
 * @param delimiters Characters for identifying words to capitalize
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Column of strings capitalized from the input column
 */
std::unique_ptr<column> capitalize(
  strings_column_view const& input,
  string_scalar const& delimiters     = string_scalar("", true, cudf::get_default_stream()),
  rmm::cuda_stream_view stream        = cudf::get_default_stream(),
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Modifies first character of each word to upper-case and lower-cases the rest.
 *
 * A word here is a sequence of characters of `sequence_type` delimited by
 * any characters not part of the `sequence_type` character set.
 *
 * This function returns a column of strings where, for each string row in the input,
 * the first character of each word is converted to upper-case,
 * while all the remaining characters in a word are converted to lower-case.
 *
 * @code{.pseudo}
 * Example:
 * input = ["   teST1", "a Test", " Another test ", "n2vidia"];
 * output = title(input)
 * output is ["   Test1", "A Test", " Another Test ", "N2Vidia"]
 * output = title(input,ALPHANUM)
 * output is ["   Test1", "A Test", " Another Test ", "N2vidia"]
 * @endcode
 *
 * Any null string entries return corresponding null output column entries.
 *
 * @param input String column
 * @param sequence_type The character type that is used when identifying words
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Column of titled strings
 */
std::unique_ptr<column> title(
  strings_column_view const& input,
  string_character_types sequence_type = string_character_types::ALPHA,
  rmm::cuda_stream_view stream         = cudf::get_default_stream(),
  rmm::mr::device_memory_resource* mr  = rmm::mr::get_current_device_resource());

/**
 * @brief Checks if the strings in the input column are title formatted.
 *
 * The first character of each word should be upper-case while all other
 * characters should be lower-case. A word is a sequence of upper-case
 * and lower-case characters.
 *
 * This function returns a column of booleans indicating true if the string in
 * the input row is in title format and false if not.
 *
 * @code{.pseudo}
 * Example:
 * input = ["   Test1", "A Test", " Another test ", "N2Vidia Corp", "!Abc"];
 * output = is_title(input)
 * output is [true, true, false, true, true]
 * @endcode
 *
 * Any null string entries result in corresponding null output column entries.
 *
 * @param input String column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Column of type BOOL8
 */
std::unique_ptr<column> is_title(
  strings_column_view const& input,
  rmm::cuda_stream_view stream        = cudf::get_default_stream(),
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace cudf
