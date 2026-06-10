// RUN: %clang_csan %s -o %t && %run --threads 1 --blocks 1 %t 2>&1 | FileCheck %s --allow-empty

int global;

int main(void) {
  for (int i = 0; i < 1024; ++i)
    global++;
  return 0;
}

// CHECK-NOT: ConcurrencySanitizer
