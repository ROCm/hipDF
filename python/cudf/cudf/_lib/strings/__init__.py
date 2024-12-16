# Copyright (c) 2020-2023, NVIDIA CORPORATION.

# MIT License
#
# Modifications Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from cudf._lib.nvtext.edit_distance import edit_distance, edit_distance_matrix
from cudf._lib.nvtext.generate_ngrams import (
    generate_character_ngrams,
    generate_ngrams,
    hash_character_ngrams,
)
from cudf._lib.nvtext.jaccard import jaccard_index
from cudf._lib.nvtext.minhash import minhash, minhash64
from cudf._lib.nvtext.ngrams_tokenize import ngrams_tokenize
from cudf._lib.nvtext.normalize import normalize_characters, normalize_spaces
from cudf._lib.nvtext.replace import filter_tokens, replace_tokens
from cudf._lib.nvtext.stemmer import (
    LetterType,
    is_letter,
    is_letter_multi,
    porter_stemmer_measure,
)
from cudf._lib.nvtext.tokenize import (
    _count_tokens_column,
    _count_tokens_scalar,
    _tokenize_column,
    _tokenize_scalar,
    character_tokenize,
    detokenize,
    tokenize_with_vocabulary,
)
from cudf._lib.strings.attributes import (
    code_points,
    count_bytes,
    count_characters,
)
from cudf._lib.strings.capitalize import capitalize, is_title, title
from cudf._lib.strings.case import swapcase, to_lower, to_upper
from cudf._lib.strings.char_types import (
    filter_alphanum,
    is_alnum,
    is_alpha,
    is_decimal,
    is_digit,
    is_lower,
    is_numeric,
    is_space,
    is_upper,
)
from cudf._lib.strings.combine import (
    concatenate,
    join,
    join_lists_with_column,
    join_lists_with_scalar,
)
from cudf._lib.strings.contains import contains_re, count_re, like, match_re
from cudf._lib.strings.convert.convert_fixed_point import to_decimal
from cudf._lib.strings.convert.convert_floats import is_float
from cudf._lib.strings.convert.convert_integers import is_integer
from cudf._lib.strings.convert.convert_urls import url_decode, url_encode
from cudf._lib.strings.extract import extract
from cudf._lib.strings.find import (
    contains,
    contains_multiple,
    endswith,
    endswith_multiple,
    find,
    rfind,
    startswith,
    startswith_multiple,
)
from cudf._lib.strings.find_multiple import find_multiple
from cudf._lib.strings.findall import findall
from cudf._lib.strings.json import GetJsonObjectOptions, get_json_object
from cudf._lib.strings.padding import (
    SideType,
    center,
    ljust,
    pad,
    rjust,
    zfill,
)
from cudf._lib.strings.repeat import repeat_scalar, repeat_sequence
from cudf._lib.strings.replace import (
    insert,
    replace,
    replace_multi,
    slice_replace,
)
from cudf._lib.strings.replace_re import (
    replace_multi_re,
    replace_re,
    replace_with_backrefs,
)
from cudf._lib.strings.split.partition import partition, rpartition
from cudf._lib.strings.split.split import (
    rsplit,
    rsplit_re,
    rsplit_record,
    rsplit_record_re,
    split,
    split_re,
    split_record,
    split_record_re,
)
from cudf._lib.strings.strip import lstrip, rstrip, strip
from cudf._lib.strings.substring import get, slice_from, slice_strings
from cudf._lib.strings.translate import filter_characters, translate
from cudf._lib.strings.wrap import wrap
