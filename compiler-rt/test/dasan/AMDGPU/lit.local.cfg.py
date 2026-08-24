# These run a kernel, so they need a device present, not merely a compiler that
# can target one.
if "amdgpu" not in config.available_features:
    config.unsupported = True
else:
    # One GPU; HIP and OpenMP tests cannot share it.
    config.parallelism_group = "gpu"
