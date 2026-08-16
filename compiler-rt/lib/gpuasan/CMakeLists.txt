# The GPU address sanitizer is built twice from here. The device half is the
# cold reporting path the instrumentation calls into; the host half places
# allocations and serves the reports. Which one this is depends on who is
# configuring us, so there is no combined build.

# Both halves speak the same RPC transport, so both take its headers from the
# libc project rather than carrying a copy.
include(FindLibcCommonUtils)
get_target_property(GPUASAN_LIBC_INCLUDE
  llvm-libc-common-utilities INTERFACE_INCLUDE_DIRECTORIES)

add_compiler_rt_component(gpuasan)

if (COMPILER_RT_GPU_BUILD)
  include_directories(..)
  include_directories(../../include)

  set(GPUASAN_CFLAGS
    ${SANITIZER_COMMON_CFLAGS}
    -DSANITIZER_COMMON_NO_REDEFINE_BUILTINS
    -isystem ${GPUASAN_LIBC_INCLUDE})
  append_rtti_flag(OFF GPUASAN_CFLAGS)

  add_compiler_rt_object_libraries(RTGpuasan
    OS ${SANITIZER_COMMON_SUPPORTED_OS}
    ARCHS ${GPUASAN_SUPPORTED_ARCH}
    SOURCES gpuasan.cpp CFLAGS ${GPUASAN_CFLAGS})

  add_compiler_rt_runtime(clang_rt.gpuasan
    STATIC
    ARCHS ${GPUASAN_SUPPORTED_ARCH}
    OBJECT_LIBS RTGpuasan
    CFLAGS ${GPUASAN_CFLAGS}
    PARENT_TARGET gpuasan)
else()
  include_directories(../../include)

  # Hidden by default: the only symbols that need to be found are OnLoad,
  # OnUnload and the tool priority, which are marked individually. The hooks
  # themselves are reached through the API tables, not by name, so exporting
  # them would only create the chance of interposing something by accident.
  set(GPUASAN_CFLAGS
    -std=c++17 -fPIC -fno-rtti -fvisibility=hidden
    -isystem ${GPUASAN_LIBC_INCLUDE})

  add_compiler_rt_object_libraries(RTGpuasan
    ARCHS ${GPUASAN_SUPPORTED_ARCH}
    SOURCES gpuasan_host.cpp CFLAGS ${GPUASAN_CFLAGS})

  # Shared, and it has to be: ROCr finds a tool by walking the objects loaded
  # into the process with dl_iterate_phdr, looking for HSA_AMD_TOOL_PRIORITY in
  # each one's dynamic string table, and it skips the entry whose name is empty
  # -- which is the executable. Code linked into the program itself is therefore
  # never found, however forcefully it was linked in.
  add_compiler_rt_runtime(clang_rt.gpuasan
    SHARED
    ARCHS ${GPUASAN_SUPPORTED_ARCH}
    OBJECT_LIBS RTGpuasan
    LINK_LIBS ${CMAKE_DL_LIBS} pthread
    CFLAGS ${GPUASAN_CFLAGS}
    PARENT_TARGET gpuasan)
endif()
