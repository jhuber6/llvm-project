// RUN: %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu \
// RUN:     -fopenmp=libomp --offload-arch=gfx908:xnack+ -fsanitize=address -nogpuinc \
// RUN:     --rocm-path=%S/Inputs/rocm \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_amdgpu_per_target_subdir %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-OPENMP
// CHECK-OPENMP-DAG: "--device-compiler=amdgpu-amd-amdhsa=-fsanitize=address"
// CHECK-OPENMP-DAG: "-u" "__asan_offload_init"
// CHECK-OPENMP-DAG: "{{[^"]*}}x86_64-unknown-linux-gnu{{/|\\\\}}libclang_rt.asan_offload.a"
// CHECK-OPENMP-DAG: "{{[^"]*}}x86_64-unknown-linux-gnu{{/|\\\\}}libclang_rt.asan.a"

// RUN: %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu \
// RUN:     -x hip --offload-arch=gfx908:xnack+ -fsanitize=address -nogpuinc -nogpulib \
// RUN:     --rocm-path=%S/Inputs/rocm \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_amdgpu_per_target_subdir %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-HIP
// CHECK-HIP-NOT: asanrtl.bc
// CHECK-HIP-DAG: "--device-compiler=amdgpu-amd-amdhsa=-fsanitize=address"
// CHECK-HIP-DAG: "-u" "__asan_offload_init"
// CHECK-HIP-DAG: "{{[^"]*}}x86_64-unknown-linux-gnu{{/|\\\\}}libclang_rt.asan_offload.a"

// RUN: %clang -no-canonical-prefixes -### --target=amdgcn-amd-amdhsa \
// RUN:     -mcpu=gfx908:xnack+ -fsanitize=address -nogpulib \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_amdgpu_per_target_subdir %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-DEVICE
// CHECK-DEVICE: "--whole-archive" "{{[^"]*}}amdgpu-amd-amdhsa{{/|\\\\}}libclang_rt.asan.a" "--no-whole-archive"
// CHECK-DEVICE-NOT: asan_offload
// CHECK-DEVICE-NOT: asan_static
// CHECK-DEVICE-NOT: asan_cxx

// RUN: %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu \
// RUN:     -x hip --offload-arch=gfx908:xnack+ -fsanitize=address -nogpuinc -nogpulib \
// RUN:     --offload-device-only --no-gpu-bundle-output \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_amdgpu_per_target_subdir %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-DEVICE-ONLY
// CHECK-DEVICE-ONLY: "{{[^"]*}}lld{{[^"]*}}" {{.*}}"--whole-archive" "{{[^"]*}}amdgpu-amd-amdhsa{{/|\\\\}}libclang_rt.asan.a" "--no-whole-archive"

// RUN: %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu \
// RUN:     -x hip --offload-arch=gfx908:xnack+ -Xarch_device -fsanitize=address \
// RUN:     -nogpuinc -nogpulib --rocm-path=%S/Inputs/rocm \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_amdgpu_per_target_subdir %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-XARCH-DEV
// CHECK-XARCH-DEV-DAG: "-u" "__asan_offload_init"
// CHECK-XARCH-DEV-DAG: "{{[^"]*}}x86_64-unknown-linux-gnu{{/|\\\\}}libclang_rt.asan_offload.a"
// CHECK-XARCH-DEV-DAG: "{{[^"]*}}x86_64-unknown-linux-gnu{{/|\\\\}}libclang_rt.asan.a"

// RUN: %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu \
// RUN:     -x hip --offload-arch=gfx908:xnack+ -Xarch_host -fsanitize=address \
// RUN:     -nogpuinc -nogpulib --rocm-path=%S/Inputs/rocm \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_amdgpu_per_target_subdir %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-XARCH-HOST
// CHECK-XARCH-HOST-NOT: asan_offload
// CHECK-XARCH-HOST-NOT: __asan_offload_init

// RUN: %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu \
// RUN:     -fsanitize=address \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_amdgpu_per_target_subdir %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-HOST
// CHECK-HOST-NOT: asan_offload
// CHECK-HOST-NOT: __asan_offload_init

// RUN: %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu \
// RUN:     -x hip --offload-arch=gfx908:xnack+ -fsanitize=address \
// RUN:     -fPIC -shared -nogpuinc -nogpulib --rocm-path=%S/Inputs/rocm \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_amdgpu_per_target_subdir %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-SHARED
// CHECK-SHARED-NOT: asan_offload
// CHECK-SHARED-NOT: __asan_offload_init

// RUN: not %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu \
// RUN:     -x hip --offload-arch=gfx908:xnack+ -fsanitize=address -shared-libsan \
// RUN:     -nogpuinc -nogpulib --rocm-path=%S/Inputs/rocm \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_amdgpu_per_target_subdir %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-SHARED-LIBSAN
// CHECK-SHARED-LIBSAN: error: invalid argument '-shared-libsan' not allowed with '-fsanitize=address for AMDGPU offloading'

// RUN: %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu \
// RUN:     -x hip --offload-arch=gfx908:xnack+ -Xarch_gfx908:xnack+ -fsanitize=address \
// RUN:     -nogpuinc -nogpulib --rocm-path=%S/Inputs/rocm \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_amdgpu_per_target_subdir %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-XARCH-GPU
// CHECK-XARCH-GPU-DAG: "-u" "__asan_offload_init"
// CHECK-XARCH-GPU-DAG: "{{[^"]*}}x86_64-unknown-linux-gnu{{/|\\\\}}libclang_rt.asan_offload.a"

// RUN: %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu \
// RUN:     -x hip --offload-arch=gfx908:xnack+ -Xarch_gfx90a -fsanitize=address \
// RUN:     -nogpuinc -nogpulib --rocm-path=%S/Inputs/rocm \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_amdgpu_per_target_subdir %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-XARCH-OTHER
// CHECK-XARCH-OTHER-NOT: asan_offload
// CHECK-XARCH-OTHER-NOT: __asan_offload_init

// RUN: %clang -no-canonical-prefixes -### --target=x86_64-unknown-linux-gnu \
// RUN:     -fopenmp=libomp -fopenmp-targets=x86_64-unknown-linux-gnu \
// RUN:     -fsanitize=address \
// RUN:     -resource-dir=%S/Inputs/resource_dir_with_amdgpu_per_target_subdir %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-OMP-CPU
// CHECK-OMP-CPU-NOT: asan_offload
// CHECK-OMP-CPU-NOT: __asan_offload_init

int main(void) { return 0; }
