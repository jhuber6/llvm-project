//===--------------- Printf format parsing for the GPU --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/arg_list.h"
#include "src/__support/ctype_utils.h"
#include "src/string/string_utils.h"

namespace LIBC_NAMESPACE {

namespace gpu {

// These sizes need to be compatible to simplify parsing the lengths.
static_assert(sizeof(uintptr_t) == sizeof(long) &&
                  sizeof(uintptr_t) == sizeof(long long) &&
                  sizeof(uintptr_t) == sizeof(intmax_t) &&
                  sizeof(uintptr_t) == sizeof(size_t) &&
                  sizeof(uintptr_t) == sizeof(ptrdiff_t),
              "Invalid lengths for target");

enum class LengthModifier { none = 0, l = 1 };

enum class SizeArgument { finished = 0, width = 1, precision = 2 };

struct Specifier {
  uintptr_t raw_value;
  bool is_string;
  bool is_empty;
};

template <typename ArgProvider> struct MicroParser {
  LIBC_INLINE MicroParser(const char *format, ArgProvider args)
      : format(format), args(args) {}

  LIBC_INLINE static constexpr bool is_flag(char c) {
    switch (c) {
    case ' ':
    case '-':
    case '+':
    case '#':
    case '0':
      return true;
    default:
      return false;
    }
  }

  LIBC_INLINE Specifier get_next_specifier() {
    Specifier specifier{};
    // Skip any characters until we reach a control character or the end.
    while (format[cur_pos] != '\0' && format[cur_pos] != '%' &&
           size_pos == SizeArgument::finished)
      ++cur_pos;

    if (format[cur_pos] != '\0')
      cur_pos++;

    // Skip all characters that aren't related to the length or type.
    if (size_pos == SizeArgument::finished) {
      while (format[cur_pos] != '\0' && is_flag(format[cur_pos]))
        ++cur_pos;

      if (format[cur_pos] == '*') {
        specifier.raw_value =
            static_cast<uintptr_t>(args.template next_var<uint32_t>());
        size_pos = SizeArgument::width;
        return specifier;
      }

      while (format[cur_pos] != '\0' && internal::isdigit(format[cur_pos]))
        ++cur_pos;
    }

    if (format[cur_pos] == '.' && size_pos != SizeArgument::precision) {
      ++cur_pos;

      if (format[cur_pos] == '*') {
        specifier.raw_value =
            static_cast<uintptr_t>(args.template next_var<uint32_t>());
        size_pos = SizeArgument::precision;
        return specifier;
      }

      while (format[cur_pos] != '\0' && internal::isdigit(format[cur_pos]))
        ++cur_pos;
    }

    LengthModifier lm = parse_length_modifier();

    // We use the type and length modifier to access the variadic argument
    // appropriately. All arguments are promoted to a simple integer.
    switch (format[cur_pos]) {
    case 'c':
      specifier.raw_value =
          static_cast<uintptr_t>(args.template next_var<uint32_t>());
      break;
    case 'd':
    case 'i':
    case 'o':
    case 'x':
    case 'X':
    case 'u':
      if (lm == LengthModifier::none)
        specifier.raw_value =
            static_cast<uintptr_t>(args.template next_var<uint32_t>());
      else
        specifier.raw_value =
            static_cast<uintptr_t>(args.template next_var<uint64_t>());
      break;
    case 'f':
    case 'F':
    case 'e':
    case 'E':
    case 'a':
    case 'A':
    case 'g':
    case 'G': {
      specifier.raw_value =
          cpp::bit_cast<uintptr_t>(args.template next_var<double>());
      break;
    }
    case 'p':
      specifier.raw_value =
          reinterpret_cast<uintptr_t>(args.template next_var<void *>());
      break;
    case 's':
      // Strings require special handling as they cannot simply be promoted.
      specifier.raw_value =
          reinterpret_cast<uintptr_t>(args.template next_var<void *>());
      specifier.is_string = true;
      break;
    default:
      // This was a malformed input or a '%' literal.
      specifier.is_empty = true;
      break;
    }

    size_pos = SizeArgument::finished;
    return specifier;
  }

  LIBC_INLINE bool end(const Specifier &cur) const {
    return format[cur_pos] == '\0' && cur.is_empty;
  }

  LIBC_INLINE size_t get_size(const Specifier &cur) const {
    if (cur.is_empty)
      return 0;
    else if (cur.is_string)
      return internal::string_length(reinterpret_cast<char *>(cur.raw_value)) +
             1;
    else
      return sizeof(uintptr_t);
  }

  LIBC_INLINE const void *get_pointer(const Specifier &cur) const {
    if (cur.is_string)
      return reinterpret_cast<void *>(cur.raw_value);
    else
      return &cur.raw_value;
  }

private:
  LIBC_INLINE LengthModifier parse_length_modifier() {
    // We are only concerned with whether or not the length specifier is larger
    // than a regular integer.
    switch (format[cur_pos]) {
    case 'l': {
      if (format[cur_pos + 1] == 'l')
        ++cur_pos;
      [[fallthrough]];
    case 't':
    case 'j':
    case 'z':
      ++cur_pos;
      return LengthModifier::l;
    }
    case 'h': {
      if (format[cur_pos + 1] == 'h')
        ++cur_pos;
      [[fallthrough]];
    case 'q':
    case 'L':
      ++cur_pos;
      return LengthModifier::none;
    }
    default:
      return LengthModifier::none;
    };
    return LengthModifier::none;
  }

  const char *__restrict const format;
  ArgProvider args;
  uint32_t cur_pos = 0;
  SizeArgument size_pos = SizeArgument::finished;
};

} // namespace gpu

} // namespace LIBC_NAMESPACE
