# These tests require a functioning GPU device and runtime on the system.
if "asan-hip" not in config.available_features:
    config.unsupported = True
else:
    config.parallelism_group = "gpu"
    config.suffixes = [".c", ".cpp", ".hip"]
    # The device reads the host shadow, which only works with page migration.
    config.environment["HSA_XNACK"] = "1"
