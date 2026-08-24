// RUN: %clang_dasan_hip --offload-device-only %s -o %t.co
// RUN: %clang_dasan -x c++ -DHOST %s -o %t %hip_libs
// RUN: %run %t %t.co 2>&1 | FileCheck %s

#ifdef HOST

extern "C" {
typedef struct ihipModule_t *hipModule_t;
typedef struct ihipModuleSymbol_t *hipFunction_t;
typedef struct ihipStream_t *hipStream_t;
int hipModuleLoad(hipModule_t *Module, const char *File);
int hipModuleUnload(hipModule_t Module);
int hipModuleGetFunction(hipFunction_t *Function, hipModule_t Module,
                         const char *Name);
int hipModuleLaunchKernel(hipFunction_t Function, unsigned GridX,
                          unsigned GridY, unsigned GridZ, unsigned BlockX,
                          unsigned BlockY, unsigned BlockZ,
                          unsigned SharedMem, hipStream_t Stream,
                          void **Params, void **Extra);
int hipDeviceSynchronize(void);
int printf(const char *, ...);
}

// Load, run, and unload the same instrumented image twice. Class zero, the
// described globals, and the symbolizer's copy of the ELF all have to go with
// the executable, or the second load claims a new span instead of the old one.
int main(int argc, char **argv) {
  for (int N = 0; N < 2; ++N) {
    hipModule_t Module = nullptr;
    hipFunction_t Fn = nullptr;
    if (hipModuleLoad(&Module, argv[1]) ||
        hipModuleGetFunction(&Fn, Module, "touch") ||
        hipModuleLaunchKernel(Fn, 1, 1, 1, 1, 1, 1, 0, 0, nullptr, nullptr) ||
        hipDeviceSynchronize() || hipModuleUnload(Module)) {
      printf("setup failed\n");
      return 2;
    }
  }
  printf("survived\n");
  return 0;
}

#else
#include "hip.h"
extern "C" __global__ void touch() {}
#endif

// CHECK: aliased {{0x[0-9a-f]+}} (+{{[0-9]+}}) at [[ALIAS:0x[0-9a-f]+]]
// CHECK: released {{0x[0-9a-f]+}} (+{{[0-9]+}}) at [[ALIAS]]
// CHECK: aliased {{0x[0-9a-f]+}} (+{{[0-9]+}}) at [[ALIAS]]
// CHECK-NOT: ERROR: DeviceAddressSanitizer
// CHECK: survived
