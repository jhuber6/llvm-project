//===-- RPC test to check args to printf ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/GPU/utils.h"
#include "src/stdio/fopen.h"
#include "src/stdio/fprintf.h"
#include "test/IntegrationTest/test.h"
#include <stdint.h>

using namespace LIBC_NAMESPACE;

FILE *file = LIBC_NAMESPACE::fopen("testdata/test_data.txt", "w");

// NVPTX requires that the constructor contains the full signature.
[[gnu::constructor]] void init(int, char **, char **) {
  file = LIBC_NAMESPACE::fopen("testdata/test_data.txt", "w");
}

TEST_MAIN(int argc, char **argv, char **envp) {
  ASSERT_TRUE(file && "failed to open file");

  int written = 0;
  written = LIBC_NAMESPACE::fprintf(file, "A simple string\n");
  ASSERT_EQ(written, 16);

  written = LIBC_NAMESPACE::fprintf(file, "%s", "A simple string\n");
  ASSERT_EQ(written, 16);

  // Check printing a different value with each thread.
  written = LIBC_NAMESPACE::fprintf(file, "%8ld\n", gpu::get_thread_id());
  ASSERT_EQ(written, 9);

  // Check a literal '%' with a bunch of stuff inbetween.
  written = LIBC_NAMESPACE::fprintf(file, "%00000%%c", 'c');
  ASSERT_EQ(written, 2);

  // Check floating point precision and printing.
  written = LIBC_NAMESPACE::fprintf(file, "%d%c%.1f\n", 1, 'c', 1.0);
  ASSERT_EQ(written, 6);

  // Check various length modifiers.
  written = LIBC_NAMESPACE::fprintf(file, "%hhd%hd%d%ld%lld%jd%zd%td%.f", '\x1', 1,
                                 1, 1l, 1ll, 1ll, 1ll, 1ll, 1.1);
  ASSERT_EQ(written, 9);

  // Check that the server properly handles a divergent number of arguments.
  const char *str = gpu::get_thread_id() % 2 ? "%s" : "%20ld\n";
  written = LIBC_NAMESPACE::fprintf(file, str, "string\n");
  ASSERT_EQ(written, gpu::get_thread_id() % 2 ? 7 : 21);

  const char *arg = gpu::get_thread_id() % 2 ? "string\n" : "%s";
  written = LIBC_NAMESPACE::fprintf(file, arg, "string\n");
  ASSERT_EQ(written, 7);

  // Check that we correctly ignore some malformed input.
  written = LIBC_NAMESPACE::fprintf(file, "%()d %d %d %d", 1, 2, 3);
  ASSERT_EQ(written, 10);
  written = LIBC_NAMESPACE::fprintf(file, "%1.1.d %d", 1);
  ASSERT_EQ(written, 8);

  // Check that we handle variable widths correctly.
  written = LIBC_NAMESPACE::fprintf(file, "%*d%*.*f", 5, 100, 10, 5);
  ASSERT_EQ(written, 15);

  // Check the values of some sign modifiers.
  written = LIBC_NAMESPACE::fprintf(file, "%-10f%++10f% 10f", 1.0, 1.0, 1.0);
  ASSERT_EQ(written, 30);

  // Check for extremely abused variable width arguments
  written = LIBC_NAMESPACE::fprintf(file, "%**d", 1, 2);
  ASSERT_EQ(written, 4);
  written = LIBC_NAMESPACE::fprintf(file, "%**d%6d", 1, 1);
  ASSERT_EQ(written, 10);
  written = LIBC_NAMESPACE::fprintf(file, "%**.**f", 1, 1, 1.0);
  ASSERT_EQ(written, 7);

  return 0;
}
