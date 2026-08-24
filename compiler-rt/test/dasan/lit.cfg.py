import os
import subprocess

config.name = "DASAN" + config.name_suffix

config.test_source_root = os.path.dirname(__file__)

default_dasan_opts = "verbosity=1"
config.environment["DASAN_OPTIONS"] = default_dasan_opts
config.substitutions.append(
    ("%env_dasan_opts=", "env DASAN_OPTIONS=" + default_dasan_opts + ":")
)


def build_invocation(compile_flags):
    return " " + " ".join([config.clang] + compile_flags) + " "


clang_dasan_cflags = ["-fsanitize=daddress", config.target_cflags]

config.substitutions.append(("%clang ", build_invocation([config.target_cflags])))
config.substitutions.append(("%clang_dasan ", build_invocation(clang_dasan_cflags)))

config.suffixes = [".c", ".cpp", ".hip"]

# Sources a test compiles alongside its own, which are not tests themselves.
config.excludes = ["Inputs"]


# A device test needs a GPU on the machine running the suite, which need not be
# the one that configured the build, so this is asked here rather than by CMake.
def find_amdgpu_archs():
    tool = os.path.join(config.llvm_tools_dir, "amdgpu-arch")
    if not os.path.exists(tool):
        return []
    try:
        out = subprocess.check_output([tool], stderr=subprocess.DEVNULL).decode()
    except (subprocess.CalledProcessError, OSError):
        return []
    return out.split()


# Only the runtime library, never the headers: the tests declare what they use.
def find_rocm():
    for root in [os.environ.get("ROCM_PATH"), "/opt/rocm"]:
        if root and os.path.exists(os.path.join(root, "lib", "libamdhip64.so")):
            return root
    return None


amdgpu_archs = find_amdgpu_archs()
amdgpu_arch = amdgpu_archs[0] if amdgpu_archs else None
rocm_path = find_rocm()

if amdgpu_arch and rocm_path:
    config.available_features.add("amdgpu")
    if len(amdgpu_archs) >= 2:
        config.available_features.add("amdgpu-multi")
    # -O0 is the default: allocas and generic decay are still there, which is
    # when the instrumentation has to work. -O1 is the optimized counterpart
    # (%clang_dasan_hip_opt / %clang_dasan_omp_O1). Higher levels can delete a
    # dead out-of-bounds access or promote a private array into registers,
    # leaving nothing to catch.
    hip_common = [
        "-xhip",
        "--offload-arch=" + amdgpu_arch,
        "-nogpuinc",
        "-nogpulib",
        "-g",
        "-I" + os.path.join(config.test_source_root, "Inputs"),
        "-fsanitize=daddress",
    ]
    hip_libs = [
        "-L" + os.path.join(rocm_path, "lib"),
        "-lamdhip64",
        "-Wl,-rpath," + os.path.join(rocm_path, "lib"),
    ]
    config.substitutions.append(
        ("%clang_dasan_hip_opt ",
         build_invocation(hip_common + ["-O1"]) + " ")
    )
    config.substitutions.append(
        ("%clang_dasan_hip ", build_invocation(hip_common + ["-O0"]) + " ")
    )
    config.substitutions.append(("%hip_libs", " ".join(hip_libs)))
    config.substitutions.append(("%amdgpu_arch", amdgpu_arch))
    config.substitutions.append(
        (
            "%offload_bundler",
            os.path.join(config.llvm_tools_dir, "clang-offload-bundler"),
        )
    )

    # The arch is named rather than left to `native` because a machine with a
    # second vendor's device would otherwise build for both and run on whichever
    # the offload runtime picked first.
    omp_common = [
        "-fopenmp",
        "--offload-arch=" + amdgpu_arch,
        "-g",
        "-fsanitize=daddress",
    ]
    config.substitutions.append(
        ("%clang_dasan_omp_O1 ",
         build_invocation(omp_common + ["-O1"]) + " ")
    )
    config.substitutions.append(
        ("%clang_dasan_omp ", build_invocation(omp_common + ["-O0"]) + " ")
    )

    # HIP and OpenMP tests share one GPU.
    lit_config.parallelism_groups["gpu"] = 1

if config.target_os != "Linux" or config.target_arch not in [
    "x86_64",
    "aarch64",
    "powerpc64le",
]:
    config.unsupported = True
