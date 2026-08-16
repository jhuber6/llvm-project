// REQUIRES: x86-registered-target, amdgpu-registered-target

// The sanitizer has a runtime on each side and enabling it has to be enough to
// get both. Neither is named on the link line: each side asks for the library
// its own triple has, the device the static archive holding the reporting path
// and the host the shared library that places allocations. The host half has to
// be shared, because the vendor runtime activates it by walking the shared
// objects loaded into the process and never looks at the executable. Its
// run-time path is left to -frtlib-add-rpath like any other shared runtime,
// rather than an rpath forced into every program that enables the sanitizer.

// RUN: %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu \
// RUN:   -fopenmp=libomp --offload-arch=gfx908 -fsanitize=gpuasan \
// RUN:   -resource-dir=%S/Inputs/resource_dir_with_amdgpu_per_target_subdir \
// RUN:   --rocm-path=%S/Inputs/rocm -nogpuinc %s 2>&1 \
// RUN: | FileCheck %s

// The device link happens inside the offload wrapper, which runs the driver
// again for the device triple, so forwarding the flag is what gets the device
// its archive.
//
// CHECK-DAG: "--device-compiler=amdgcn-amd-amdhsa=-fsanitize=gpuasan"
// CHECK-DAG: "{{[^"]*}}resource_dir_with_amdgpu_per_target_subdir{{[^"]*}}libclang_rt.gpuasan.so"
// CHECK-NOT: "-rpath"

// RUN: %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu \
// RUN:   -fopenmp=libomp --offload-arch=gfx908 -fsanitize=gpuasan \
// RUN:   -frtlib-add-rpath \
// RUN:   -resource-dir=%S/Inputs/resource_dir_with_amdgpu_per_target_subdir \
// RUN:   --rocm-path=%S/Inputs/rocm -nogpuinc %s 2>&1 \
// RUN: | FileCheck --check-prefix=RPATH %s

// RPATH: "-rpath" "{{[^"]*}}resource_dir_with_amdgpu_per_target_subdir{{[^"]*}}"

// Neither runtime is handed to the device linker by name: the host half is an
// x86 object, and the device half is found by the device link itself.

// RUN: %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu \
// RUN:   -fopenmp=libomp --offload-arch=gfx908 -fsanitize=gpuasan \
// RUN:   -resource-dir=%S/Inputs/resource_dir_with_amdgpu_per_target_subdir \
// RUN:   --rocm-path=%S/Inputs/rocm -nogpuinc %s 2>&1 \
// RUN: | FileCheck --check-prefix=DEVICE %s

// DEVICE-NOT: "--device-linker=amdgcn-amd-amdhsa={{[^"]*}}libclang_rt.gpuasan.{{a|so}}"
// DEVICE-NOT: "--device-linker=amdgcn-amd-amdhsa=-lstdc++"

// A device-only compile is the same lookup with nothing on the other side: the
// static archive comes from the resource directory for the target being linked
// for, with no offload wrapper and no host runtime involved.

// RUN: %clang -no-canonical-prefixes -### --target=amdgpu-amd-amdhsa -mcpu=gfx908 \
// RUN:   -fsanitize=gpuasan \
// RUN:   -resource-dir=%S/Inputs/resource_dir_with_amdgpu_per_target_subdir %s 2>&1 \
// RUN: | FileCheck --check-prefix=STANDALONE %s

// STANDALONE: "-fsanitize=gpuasan"
// STANDALONE: "--whole-archive" "{{[^"]*}}amdgpu-amd-amdhsa{{[/\\]}}libclang_rt.gpuasan.a" "--no-whole-archive"
// STANDALONE-NOT: libclang_rt.gpuasan.so

// A toolchain with no device runtime is not told about the sanitizer at all: the
// flag would reach the device compilation and then fail to link.

// RUN: %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu \
// RUN:   -fopenmp=libomp --offload-arch=gfx908 -fsanitize=gpuasan \
// RUN:   -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir \
// RUN:   --rocm-path=%S/Inputs/rocm -nogpuinc %s 2>&1 \
// RUN: | FileCheck --check-prefix=MISSING %s

// MISSING-NOT: "--device-compiler=amdgcn-amd-amdhsa=-fsanitize=gpuasan"
