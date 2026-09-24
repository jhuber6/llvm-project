// REQUIRES: asan-hip
// RUN: %clang_asan_hip %s -o %t %hip_libs
// RUN: not %run %t 2>&1 | FileCheck %s

__global__ void store(int *P) { P[4] = 1; }

int main() {
  int *P = nullptr;
  CHECK_HIP(hipMallocManaged((void **)&P, 4 * sizeof(int)));
  store<<<1, 1>>>(P);
  hipDeviceSynchronize();
  return 0;
}

// CHECK: ERROR: AddressSanitizer: heap-buffer-overflow
// CHECK: WRITE of size 4 at [[ADDR:0x[0-9a-f]+]] by device
// CHECK: #0 {{.*}} in store(int*) {{.*}}managed-oob.hip:5
// CHECK: [[ADDR]] is located 0 bytes after 16-byte region
// CHECK: allocated by host thread here:
// CHECK-NEXT: #0 {{.*}} in hsa_amd_vmem_address_reserve_align
