# These tests require a functioning GPU device and runtime on the system.
if "csan-hip" not in config.available_features and (
    "csan-openmp-offload" not in config.available_features
):
    config.unsupported = True
else:
    config.parallelism_group = "gpu"
    config.suffixes = [".c", ".cpp", ".hip"]
