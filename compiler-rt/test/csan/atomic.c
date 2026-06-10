// RUN: %clang_csan %s -o %t && %run --threads 64 --blocks 64 %t 2>&1 | FileCheck %s --allow-empty

int global;

// CHECK-NOT: ConcurrencySanitizer
int main(void) {
  for (int i = 0; i < 1024; ++i)
    __atomic_fetch_add(&global, 1, __ATOMIC_RELAXED);
  return 0;
}
