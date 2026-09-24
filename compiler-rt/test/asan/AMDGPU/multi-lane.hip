// REQUIRES: asan-hip
// RUN: %clang_asan_hip %s -o %t %hip_libs
// RUN: not %run %t 2>&1 | FileCheck %s

__global__ void oob(int *P) { P[__builtin_amdgcn_workitem_id_x()] = 0; }

int main() {
  int *P = nullptr;
  CHECK_HIP(hipMalloc((void **)&P, 2 * sizeof(int)));
  oob<<<1, 4>>>(P);
  hipDeviceSynchronize();
  return 0;
}

// CHECK: ERROR: AddressSanitizer: heap-buffer-overflow
// CHECK: WRITE of size 4 at [[ADDR:0x[0-9a-f]+]] by device thread (2,0,0) block (0,0,0) lane 2 on GPU {{[0-9]+}}:
// CHECK: #0 {{.*}}oob{{.*}}multi-lane.hip:5
// CHECK: Lanes and accessed addresses:
// CHECK-NEXT: 02 : [[ADDR]]  03 : 0x
// CHECK: is located 0 bytes after 8-byte region
