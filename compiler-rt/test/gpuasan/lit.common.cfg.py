# -*- Python -*-

import os

config.test_source_root = os.path.dirname(__file__)
config.name = "GPUSanitizer" + config.name_suffix


def build_invocation(compile_flags):
    return " " + " ".join([config.clang] + compile_flags) + " "


target_cflags = [getattr(config, "target_cflags", "")]
clang_gpuasan_cflags = ["-fsanitize=gpuasan"] + target_cflags

config.substitutions.append(("%clang_gpuasan ", build_invocation(clang_gpuasan_cflags)))

config.suffixes = [".c", ".cpp"]

# The runtime only exists for offload targets.
if not config.target_arch.startswith(("amdgcn", "amdgpu", "nvptx")):
    config.unsupported = True
