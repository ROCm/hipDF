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
#include <cudf/strings/strings_column_view.hpp>

namespace nvtext {
/**
 * @addtogroup nvtext_ngrams
 * @{
 * @file
 */

/**
 * @brief Returns a single column of strings by tokenizing the input strings
 * column and then producing ngrams of each string.
 *
 * An ngram is a grouping of 2 or more tokens with a separator. For example,
 * generating bigrams groups all adjacent pairs of tokens for a string.
 *
 * ```
 * ["a bb ccc"] can be tokenized to ["a", "bb", "ccc"]
 * bigrams would generate ["a_bb", "bb_ccc"] and trigrams would generate ["a_bb_ccc"]
 * ```
 *
 * The `delimiter` is used for tokenizing and may be zero or more characters.
 * If the `delimiter` is empty, whitespace (character code-point <= ' ') is used
 * for identifying tokens.
 *
 * Once tokens are identified, ngrams are produced by joining the tokens
 * with the specified separator. The generated ngrams use the tokens for each
 * string and not across strings in adjacent rows.
 * Any input string that contains fewer tokens than the specified ngrams value is
 * skipped and will not contribute to the output. Therefore, a bigram of a single
 * token is ignored as well as a trigram of 2 or less tokens.
 *
 * Tokens are found by locating delimiter(s) starting at the beginning of each string.
 * As each string is tokenized, the ngrams are generated using input column row order
 * to build the output column. That is, ngrams created in input row[i] will be placed in
 * the output column directly before ngrams created in input row[i+1].
 *
 * The size of the output column will be the total number of ngrams generated from
 * the input strings column.
 *
 * @code{.pseudo}
 * Example:
 * s = ["a b c", "d e", "f g h i", "j"]
 * t = ngrams_tokenize(s, 2, " ", "_")
 * t is now ["a_b", "b_c", "d_e", "f_g", "g_h", "h_i"]
 * @endcode
 *
 * All null row entries are ignored and the output contains all valid rows.
 *
 * @param input Strings column to tokenize and produce ngrams from
 * @param ngrams The ngram number to generate
 * @param delimiter UTF-8 characters used to separate each string into tokens.
 *                  An empty string will separate tokens using whitespace.
 * @param separator The string to use for separating ngram tokens
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New strings columns of tokens
 */
std::unique_ptr<cudf::column> ngrams_tokenize(
  cudf::strings_column_view const& input,
  cudf::size_type ngrams,
  cudf::string_scalar const& delimiter,
  cudf::string_scalar const& separator,
  rmm::cuda_stream_view stream        = cudf::get_default_stream(),
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/** @} */  // end of group
}  // namespace nvtext
