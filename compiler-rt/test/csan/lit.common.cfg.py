# -*- Python -*-

import os


def get_required_attr(config, attr_name):
    attr_value = getattr(config, attr_name, None)
    if attr_value is None:
        lit_config.fatal("No attribute %r in test configuration!" % attr_name)
    return attr_value


config.name = "CSan-" + config.name_suffix
config.test_source_root = os.path.dirname(__file__)
config.suffixes = [".c", ".cpp"]


def build_invocation(compile_flags):
    return " " + " ".join([config.clang] + compile_flags) + " "


target_cflags = [get_required_attr(config, "target_cflags")]
config.substitutions.append(("%clang ", build_invocation(target_cflags)))
config.substitutions.append(
    ("%clangxx ", build_invocation(config.cxx_mode_flags + target_cflags))
)

if config.target_os not in ["Linux"]:
    config.unsupported = True

_amdgpu_csan_rt = any(
    os.path.isfile(
        os.path.join(
            config.compiler_rt_output_dir, "lib", triple, "libclang_rt.csan.a"
        )
    )
    for triple in ("amdgpu-amd-amdhsa", "amdgcn-amd-amdhsa")
)
if config.csan_can_run_hip and _amdgpu_csan_rt:
    config.available_features.add("csan-hip")
    hip_common = [
        "-xhip",
        "--offload-arch=" + config.csan_gpu_arch,
        "-nogpuinc",
        "-nogpulib",
        "-g",
        "-isystem",
        os.path.join(config.compiler_rt_src_root, "test", "ubsan", "Inputs"),
        "-include",
        "hip.h",
        "-fsanitize=concurrency",
    ]
    hip_libs = [
        "-L" + config.csan_hip_lib_dir,
        "-lamdhip64",
        "-Wl,-rpath," + config.csan_hip_lib_dir,
    ]
    config.substitutions.append(("%clang_csan_hip ", build_invocation(hip_common) + " "))
    config.substitutions.append(("%hip_libs", " ".join(hip_libs)))
if config.csan_can_run_openmp_offload and _amdgpu_csan_rt:
    config.available_features.add("csan-openmp-offload")
    omp_common = [
        "-fopenmp",
        "--offload-arch=" + config.csan_gpu_arch,
        "-g",
        "-frtlib-add-rpath",
        "-fno-exceptions",
        "-fsanitize=concurrency",
    ]
    config.substitutions.append(("%clang_csan_omp ", build_invocation(omp_common) + " "))
if config.csan_can_run_hip or config.csan_can_run_openmp_offload:
    lit_config.parallelism_groups["gpu"] = 1
