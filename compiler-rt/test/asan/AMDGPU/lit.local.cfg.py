# These tests require a functioning XNACK-capable GPU and runtime.
if (
    "asan-hip" not in config.available_features
    and "asan-openmp-offload" not in config.available_features
):
    config.unsupported = True
else:
    config.parallelism_group = "gpu"
    config.suffixes = [".cpp", ".hip"]
