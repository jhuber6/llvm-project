// RUN: %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu \
// RUN:     -fopenmp=libomp --offload-arch=gfx908 -fsanitize=daddress -nogpuinc \
// RUN:     --rocm-path=%S/Inputs/rocm \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_amdgpu_per_target_subdir %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-OPENMP
// CHECK-OPENMP-DAG: "--device-compiler=amdgcn-amd-amdhsa=-fsanitize=daddress"
// CHECK-OPENMP-DAG: "--whole-archive" "{{[^"]*}}x86_64-unknown-linux-gnu{{/|\\\\}}libclang_rt.dasan.a" "--no-whole-archive"

// RUN: %clang -no-canonical-prefixes -### --target=amdgcn-amd-amdhsa -mcpu=gfx908 \
// RUN:     -nogpulib -fsanitize=daddress \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_amdgpu_per_target_subdir %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-DEVICE
//      CHECK-DEVICE: ld.lld
// CHECK-DEVICE-SAME: "--whole-archive" "{{[^"]*}}amdgpu-amd-amdhsa{{/|\\\\}}libclang_rt.dasan.a" "--no-whole-archive"

// RUN: %clang -no-canonical-prefixes -### -x hip --offload-arch=gfx908 \
// RUN:     --offload-device-only -fsanitize=daddress -nogpuinc -nogpulib \
// RUN:     --rocm-path=%S/Inputs/rocm \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_amdgpu_per_target_subdir %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-HIP
//      CHECK-HIP: lld
// CHECK-HIP-SAME: "--whole-archive"
// CHECK-HIP-SAME: "{{[^"]*}}amdgpu-amd-amdhsa{{/|\\\\}}libclang_rt.dasan.a"

// RUN: %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu \
// RUN:     -fopenmp=libomp --offload-arch=gfx908 -fsanitize=daddress -nogpuinc \
// RUN:     --rocm-path=%S/Inputs/rocm \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-MISSING
// CHECK-MISSING-NOT: "--device-compiler=amdgcn-amd-amdhsa=-fsanitize=daddress"

int main(void) { return 0; }
