// RUN: %clang_dasan_omp %s -o %t
// RUN: %clang_dasan_omp_O1 %s -o %t.opt
// RUN: %run %t 2>&1 | FileCheck %s
// RUN: %run %t.opt 2>&1 | FileCheck %s

// A device runtime that allocates for the kernel it is running has the host
// write entries mid dispatch, so what publishes one has to hold against a
// kernel already in flight. Dropping what the device cached before the write
// has landed does worse than nothing: the invalidation races ahead of a store
// that is still posted, and the refill that follows pulls the value it was
// meant to replace back out of memory. The check then fails on an entry that
// reads as though nothing were allocated there, which is a report against a
// perfectly good access.
//
// Allocating from many threads at once is what makes the allocator grow during
// the dispatch rather than settle on its first slab.

#include <cstdio>
#include <cstdlib>

int main() {
  const int Teams = 64, Threads = 64, Rounds = 4, N = 256;
  int Bad = 0;

#pragma omp target teams distribute parallel for num_teams(Teams)              \
    num_threads(Threads) map(tofrom : Bad) reduction(+ : Bad)
  for (int T = 0; T < Teams * Threads; ++T) {
    for (int R = 0; R < Rounds; ++R) {
      int *P = (int *)malloc(N * sizeof(int));
      if (!P)
        continue;
      for (int I = 0; I < N; ++I)
        P[I] = T + R;
      if (P[N - 1] != T + R)
        ++Bad;
      free(P);
    }
  }

  if (Bad) {
    printf("%d bad allocations\n", Bad);
    return 2;
  }

  printf("allocator survived\n");
  return 0;
}

// CHECK-NOT: ERROR: DeviceAddressSanitizer
// CHECK: allocator survived
