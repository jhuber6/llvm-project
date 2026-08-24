// RUN: %clang_hip_dasan %s -o %t %hip_libs
// RUN: %env_dasan_opts=halt_on_error=0 %run %t 2>&1 \
// RUN:   | FileCheck %s --implicit-check-not=SUMMARY
// RUN: %clang_hip_dasan -O1 %s -o %t %hip_libs
// RUN: %env_dasan_opts=halt_on_error=0 %run %t 2>&1 \
// RUN:   | FileCheck %s --implicit-check-not=SUMMARY

// The last byte and the one past it for a size below the leading redzone, one
// sharing its chunk with both redzones, a page, and one backed a chunk at a
// time. The extent is the requested size, not the chunk's.
__global__ void touch(char *P, long I) { P[I] = 1; }

static const long Sizes[] = {16, 1000, 4096, 100000};

int main() {
  for (long N : Sizes) {
    char *P = nullptr;
    CHECK_HIP(hipMalloc((void **)&P, N));
    if ((unsigned long)P % 256)
      return 1;
    touch<<<1, 1>>>(P, N - 1);
    touch<<<1, 1>>>(P, N);
  }
  CHECK_HIP(hipDeviceSynchronize());
  return 0;
}

// CHECK: #0 {{0x[0-9a-f]+}} in touch{{.*}}size-classes.hip
// CHECK: is located 0 bytes after 16-byte region
// CHECK: SUMMARY: DeviceAddressSanitizer: heap-buffer-overflow
// CHECK: #0 {{0x[0-9a-f]+}} in touch{{.*}}size-classes.hip
// CHECK: is located 0 bytes after 1000-byte region
// CHECK: SUMMARY: DeviceAddressSanitizer: heap-buffer-overflow
// CHECK: #0 {{0x[0-9a-f]+}} in touch{{.*}}size-classes.hip
// CHECK: is located 0 bytes after 4096-byte region
// CHECK: SUMMARY: DeviceAddressSanitizer: heap-buffer-overflow
// CHECK: #0 {{0x[0-9a-f]+}} in touch{{.*}}size-classes.hip
// CHECK: is located 0 bytes after 100000-byte region
// CHECK: SUMMARY: DeviceAddressSanitizer: heap-buffer-overflow
