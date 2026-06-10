// RUN: %clang_csan %s -o %t && %run --threads 64 --blocks 64 %t 2>&1 | FileCheck %s --allow-empty

#include <gpuintrin.h>

static int data[64 * 64];

// CHECK-NOT: ConcurrencySanitizer
int main(void) {
  unsigned id = __gpu_num_threads(0) * __gpu_block_id(0) + __gpu_thread_id(0);
  for (int i = 0; i < 1024; ++i)
    data[id % (64 * 64)] += i;
  return 0;
}
