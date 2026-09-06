# These tests require a functioning GPU device and runtime on the system.
if "asan-hip" not in config.available_features:
    config.unsupported = True
else:
    config.parallelism_group = "gpu"
    config.suffixes = [".c", ".cpp", ".hip"]
