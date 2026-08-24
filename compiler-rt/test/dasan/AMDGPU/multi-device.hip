// REQUIRES: amdgpu-multi
// RUN: %clang_dasan_hip %s -o %t %hip_libs -pthread
// RUN: %clang_dasan_hip_opt %s -o %t.opt %hip_libs -pthread
// RUN: %run %t clean 2>&1 | FileCheck %s --check-prefix=CLEAN
// RUN: %run %t.opt clean 2>&1 | FileCheck %s --check-prefix=CLEAN
// RUN: not %run %t heap0 2>&1 | FileCheck %s --check-prefix=HEAP
// RUN: not %run %t.opt heap0 2>&1 | FileCheck %s --check-prefix=HEAP
// RUN: not %run %t heap1 2>&1 | FileCheck %s --check-prefix=HEAP
// RUN: not %run %t.opt heap1 2>&1 | FileCheck %s --check-prefix=HEAP
// RUN: not %run %t global0 2>&1 | FileCheck %s --check-prefix=GLOBAL
// RUN: not %run %t.opt global0 2>&1 | FileCheck %s --check-prefix=GLOBAL
// RUN: not %run %t global1 2>&1 | FileCheck %s --check-prefix=GLOBAL
// RUN: not %run %t.opt global1 2>&1 | FileCheck %s --check-prefix=GLOBAL

#include "hip.h"

#ifndef __HIP_DEVICE_COMPILE__
#include <pthread.h>
#endif

// Two GPUs, one reserved window. Each agent has its own heap slot, its own
// alias of the image, and its own report channel. Host threads that call into
// the runtime at once still have to come out, and a report has to be able to
// come from the second card, not only the first.
//
// Both cards must be able to run the image this was built for; mixed-arch
// machines that amdgpu-arch listed as two GPUs but that cannot both consume
// --offload-arch will fail hipSetDevice or the launch and print setup failed.

__device__ int Table[16];

__global__ void fill(int *P, int N, int Tag) {
  unsigned T = threadIdxX();
  if ((int)T < N)
    P[T] = Tag + (int)T;
}

__global__ void poke_heap(int *P, int I) { P[I] = 1; }

__global__ void poke_global(int I) { Table[I] = 1; }

#ifndef __HIP_DEVICE_COMPILE__

static const int N = 64;

struct Job {
  int Device;
  int Tag;
  int *Live;
  int Overflow; // 0 clean, 1 heap, 2 global
  int Rc;
};

static int OnDevice(int Device) {
  if (hipSetDevice(Device))
    return 2;
  return 0;
}

static int *Place(int Device) {
  if (OnDevice(Device))
    return nullptr;
  int *P = nullptr;
  if (hipMalloc((void **)&P, N * sizeof(int)))
    return nullptr;
  return P;
}

static int RunKernel(int Device, int *P, int Tag) {
  if (OnDevice(Device))
    return 2;
  fill<<<1, N>>>(P, N, Tag);
  return hipDeviceSynchronize();
}

static void *Work(void *Arg) {
  Job *J = (Job *)Arg;
  if (J->Overflow == 2) {
    if (OnDevice(J->Device)) {
      J->Rc = 2;
      return nullptr;
    }
    poke_global<<<1, 1>>>(N / 4);
    hipDeviceSynchronize();
    J->Rc = 3;
    return nullptr;
  }

  int *P = Place(J->Device);
  if (!P) {
    J->Rc = 2;
    return nullptr;
  }
  J->Live = P;
  if (J->Overflow == 1) {
    poke_heap<<<1, 1>>>(P, N);
    hipDeviceSynchronize();
    J->Rc = 3;
    return nullptr;
  }
  J->Rc = RunKernel(J->Device, P, J->Tag);
  return nullptr;
}

static int Join(pthread_t T, Job *J) {
  if (pthread_join(T, nullptr))
    return 2;
  return J->Rc;
}

int main(int argc, char **argv) {
  int Devices = 0;
  CHECK_HIP(hipGetDeviceCount(&Devices));
  if (Devices < 2) {
    printf("setup failed: hipGetDeviceCount\n");
    return 2;
  }

  const char *Mode = argc > 1 ? argv[1] : "clean";

  if (Mode[0] == 'h' || Mode[0] == 'g') {
    const char *End = Mode;
    while (End[1])
      ++End;
    const int Dev = *End == '1' ? 1 : 0;
    Job J = {Dev, Dev + 1, nullptr, Mode[0] == 'h' ? 1 : 2, 0};
    Work(&J);
    if (J.Live) {
      OnDevice(Dev);
      hipFree(J.Live);
    }
    if (J.Rc == 2) {
      printf("setup failed: overflow device %d\n", Dev);
      return 2;
    }
    printf("survived\n");
    return 3;
  }

  // Sequential: each device places, runs, and frees on its own.
  for (int D = 0; D < 2; ++D) {
    Job J = {D, D + 1, nullptr, 0, 0};
    Work(&J);
    if (J.Rc || !J.Live) {
      printf("setup failed: sequential device %d\n", D);
      return 2;
    }
    if (OnDevice(D) || hipFree(J.Live)) {
      printf("setup failed: free device %d\n", D);
      return 2;
    }
  }

  // Both heaps live in the same size class, then a kernel on each.
  int *A = Place(0);
  int *B = Place(1);
  if (!A || !B || RunKernel(0, A, 10) || RunKernel(1, B, 20)) {
    printf("setup failed: both live\n");
    return 2;
  }
  if (OnDevice(0) || hipFree(A) || OnDevice(1) || hipFree(B)) {
    printf("setup failed: free both\n");
    return 2;
  }

  // Two host threads, one per device, through the interceptors at once.
  Job J0 = {0, 30, nullptr, 0, 0};
  Job J1 = {1, 40, nullptr, 0, 0};
  pthread_t T0, T1;
  if (pthread_create(&T0, nullptr, Work, &J0) ||
      pthread_create(&T1, nullptr, Work, &J1)) {
    printf("setup failed: pthread_create\n");
    return 2;
  }
  const int R0 = Join(T0, &J0);
  const int R1 = Join(T1, &J1);
  if (J0.Live) {
    OnDevice(0);
    hipFree(J0.Live);
  }
  if (J1.Live) {
    OnDevice(1);
    hipFree(J1.Live);
  }
  if (R0 || R1) {
    printf("setup failed: concurrent %d %d\n", R0, R1);
    return 2;
  }

  printf("survived\n");
  return 0;
}

#endif

// CLEAN-NOT: ERROR: DeviceAddressSanitizer
// CLEAN: survived

// HEAP: ERROR: DeviceAddressSanitizer: heap-buffer-overflow
// HEAP: WRITE of size 4
// HEAP: is located 0 bytes after 256-byte region
// HEAP-NOT: survived

// GLOBAL: ERROR: DeviceAddressSanitizer: global-buffer-overflow
// GLOBAL: in poke_global{{.*}}multi-device.hip
// GLOBAL: is located 0 bytes after global variable 'Table' {{.*}} of size 64
// GLOBAL-NOT: survived
