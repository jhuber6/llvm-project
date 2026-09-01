// RUN: %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu \
// RUN:     -fopenmp=libomp --offload-arch=gfx908 -fsanitize=concurrency -nogpuinc \
// RUN:     --rocm-path=%S/Inputs/rocm \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_amdgpu_per_target_subdir %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-OPENMP
// CHECK-OPENMP-DAG: "--device-compiler=amdgpu-amd-amdhsa=-fsanitize=concurrency"
// CHECK-OPENMP-DAG: "-u" "__csan_offload_init"
// CHECK-OPENMP-DAG: "{{[^"]*}}x86_64-unknown-linux-gnu{{/|\\\\}}libclang_rt.csan_offload.a"
// CHECK-OPENMP-DAG: "{{[^"]*}}x86_64-unknown-linux-gnu{{/|\\\\}}libclang_rt.csan.a"

// RUN: %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu \
// RUN:     -x hip --offload-arch=gfx908 -fsanitize=concurrency -nogpuinc -nogpulib \
// RUN:     --rocm-path=%S/Inputs/rocm \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_amdgpu_per_target_subdir %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-HIP
// CHECK-HIP-DAG: "-u" "__csan_offload_init"
// CHECK-HIP-DAG: "{{[^"]*}}x86_64-unknown-linux-gnu{{/|\\\\}}libclang_rt.csan_offload.a"
// CHECK-HIP-DAG: "{{[^"]*}}x86_64-unknown-linux-gnu{{/|\\\\}}libclang_rt.csan.a"

// RUN: %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu \
// RUN:     -x hip --offload-arch=gfx908 -Xarch_device -fsanitize=concurrency \
// RUN:     -nogpuinc -nogpulib --rocm-path=%S/Inputs/rocm \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_amdgpu_per_target_subdir %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-XARCH-DEV
// CHECK-XARCH-DEV-DAG: "-u" "__csan_offload_init"
// CHECK-XARCH-DEV-DAG: "{{[^"]*}}x86_64-unknown-linux-gnu{{/|\\\\}}libclang_rt.csan_offload.a"
// CHECK-XARCH-DEV-DAG: "{{[^"]*}}x86_64-unknown-linux-gnu{{/|\\\\}}libclang_rt.csan.a"

// RUN: not %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu \
// RUN:     -x hip --offload-arch=gfx908 -Xarch_host -fsanitize=concurrency \
// RUN:     -nogpuinc -nogpulib --rocm-path=%S/Inputs/rocm \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_amdgpu_per_target_subdir %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-XARCH-HOST
// CHECK-XARCH-HOST: error: unsupported option '-fsanitize=concurrency' for target 'x86_64-unknown-linux-gnu'

// RUN: %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu \
// RUN:     -fsanitize=undefined \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_amdgpu_per_target_subdir %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-HOST
// CHECK-HOST-NOT: csan_offload
// CHECK-HOST-NOT: __csan_offload_init

// RUN: %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu \
// RUN:     -x hip --offload-arch=gfx908 -Xarch_device -fsanitize=concurrency \
// RUN:     -fPIC -shared -nogpuinc -nogpulib --rocm-path=%S/Inputs/rocm \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_amdgpu_per_target_subdir %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-SHARED-DEV
// CHECK-SHARED-DEV-DAG: "-u" "__csan_offload_init"
// CHECK-SHARED-DEV-DAG: "{{[^"]*}}x86_64-unknown-linux-gnu{{/|\\\\}}libclang_rt.csan_offload.a"
// CHECK-SHARED-DEV-DAG: "{{[^"]*}}x86_64-unknown-linux-gnu{{/|\\\\}}libclang_rt.csan.a"

// RUN: %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu \
// RUN:     -x hip --offload-arch=gfx908 -fsanitize=concurrency \
// RUN:     -fPIC -shared -nogpuinc -nogpulib --rocm-path=%S/Inputs/rocm \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_amdgpu_per_target_subdir %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-SHARED
// CHECK-SHARED-DAG: "-u" "__csan_offload_init"
// CHECK-SHARED-DAG: "{{[^"]*}}x86_64-unknown-linux-gnu{{/|\\\\}}libclang_rt.csan_offload.a"

// RUN: %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu \
// RUN:     -x hip --offload-arch=gfx908 -Xarch_gfx908 -fsanitize=concurrency \
// RUN:     -nogpuinc -nogpulib --rocm-path=%S/Inputs/rocm \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_amdgpu_per_target_subdir %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-XARCH-GPU
// CHECK-XARCH-GPU-DAG: "-u" "__csan_offload_init"
// CHECK-XARCH-GPU-DAG: "{{[^"]*}}x86_64-unknown-linux-gnu{{/|\\\\}}libclang_rt.csan_offload.a"

// RUN: %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu \
// RUN:     -x hip --offload-arch=gfx908 -Xarch_gfx90a -fsanitize=concurrency \
// RUN:     -nogpuinc -nogpulib --rocm-path=%S/Inputs/rocm \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_amdgpu_per_target_subdir %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-XARCH-OTHER
// CHECK-XARCH-OTHER-NOT: csan_offload
// CHECK-XARCH-OTHER-NOT: __csan_offload_init

// RUN: %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu \
// RUN:     -fopenmp=libomp -fopenmp-targets=x86_64-unknown-linux-gnu \
// RUN:     -fsanitize=concurrency \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_amdgpu_per_target_subdir %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-OMP-CPU
// CHECK-OMP-CPU-NOT: csan_offload
// CHECK-OMP-CPU-NOT: __csan_offload_init

int main(void) { return 0; }
