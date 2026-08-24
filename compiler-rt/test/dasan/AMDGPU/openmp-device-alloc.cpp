// RUN: %clang_omp_offload_dasan %s -o %t
// RUN: %run %t
// RUN: %clang_omp_offload_dasan -O1 %s -o %t
// RUN: %run %t

// The device allocator recovers a slab by masking a pointer to a granule, so
// placement must keep granule alignment. Allocating from many threads grows it
// mid-dispatch, which the host must publish without disturbing live entries.

#include <cstdlib>

int main() {
  int Bad = 0;

#pragma omp target map(tofrom : Bad)
  {
    int *P = (int *)malloc(256 * sizeof(int));
    if (!P)
      Bad = 1;
    else {
      for (int I = 0; I < 256; ++I)
        P[I] = I;
      for (int I = 0; I < 256; ++I)
        if (P[I] != I)
          Bad = 1;
      free(P);
    }
  }
  if (Bad)
    return 1;

  const int Teams = 64, Threads = 64, Rounds = 4, N = 256;
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
  return Bad != 0;
}
