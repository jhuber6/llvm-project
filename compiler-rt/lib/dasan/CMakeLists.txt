include_directories(..)

# Both halves speak the RPC transport libc already runs between a GPU and its
# host, so both take its headers from the libc project rather than copying them.
include(FindLibcCommonUtils)
get_target_property(DASAN_LIBC_INCLUDE
  llvm-libc-common-utilities INTERFACE_INCLUDE_DIRECTORIES)

# This runtime is built for both the offloading host and the GPU device.
if (COMPILER_RT_GPU_BUILD)
  set(DASAN_DEVICE_CFLAGS
    ${SANITIZER_COMMON_CFLAGS}
    -DSANITIZER_COMMON_NO_REDEFINE_BUILTINS
    -isystem ${DASAN_LIBC_INCLUDE})
  append_rtti_flag(OFF DASAN_DEVICE_CFLAGS)

  add_compiler_rt_component(dasan)

  add_compiler_rt_object_libraries(RTDasan
    ARCHS ${DASAN_SUPPORTED_ARCH}
    SOURCES dasan_device.cpp
    CFLAGS ${DASAN_DEVICE_CFLAGS})

  add_compiler_rt_runtime(clang_rt.dasan
    STATIC
    ARCHS ${DASAN_SUPPORTED_ARCH}
    OBJECT_LIBS RTDasan
    CFLAGS ${DASAN_DEVICE_CFLAGS}
    PARENT_TARGET dasan)

  return()
endif()

# The runtime proper talks in VA and mmap. HSA is the one device binding.
set(DASAN_CXX_SOURCES
  dasan.cpp
  dasan_allocator.cpp
  dasan_image.cpp
  dasan_report.cpp
  dasan_symbolize.cpp
  )

# The binding to one device runtime; a second would be another pair beside
# these.
list(APPEND DASAN_CXX_SOURCES
  dasan_hsa.cpp
  dasan_hsa_interceptors.cpp
  dasan_rpc.cpp
  )

set(DASAN_HEADERS
  dasan.h
  dasan_allocator.h
  dasan_flags.h
  dasan_flags.inc
  dasan_hsa.h
  dasan_image.h
  dasan_mapping.h
  dasan_platform.h
  dasan_report.h
  dasan_rpc.h
  dasan_symbolize.h
  hsa.h
  )

set(DASAN_CFLAGS ${SANITIZER_COMMON_CFLAGS} -isystem ${DASAN_LIBC_INCLUDE})
set(DASAN_LINK_FLAGS ${SANITIZER_COMMON_LINK_FLAGS})

append_rtti_flag(OFF DASAN_CFLAGS)

add_compiler_rt_object_libraries(RTDasan
  ARCHS ${DASAN_SUPPORTED_ARCH}
  SOURCES ${DASAN_CXX_SOURCES}
  ADDITIONAL_HEADERS ${DASAN_HEADERS}
  CFLAGS ${DASAN_CFLAGS})

set(DASAN_COMMON_RUNTIME_OBJECT_LIBS
  RTInterception
  RTSanitizerCommon
  RTSanitizerCommonLibc
  RTSanitizerCommonCoverage
  RTSanitizerCommonSymbolizer)

add_compiler_rt_component(dasan)

# Static linking is required to intercept HSA calls early enough for all users.
add_compiler_rt_runtime(clang_rt.dasan
  STATIC
  ARCHS ${DASAN_SUPPORTED_ARCH}
  OBJECT_LIBS RTDasan
              ${DASAN_COMMON_RUNTIME_OBJECT_LIBS}
  LINK_FLAGS ${DASAN_LINK_FLAGS}
  CFLAGS ${DASAN_CFLAGS}
  PARENT_TARGET dasan)

# The list of symbols the driver hands the linker as --dynamic-list.
if (SANITIZER_USE_SYMBOLS)
  foreach(arch ${DASAN_SUPPORTED_ARCH})
    add_sanitizer_rt_symbols(clang_rt.dasan
      ARCHS ${arch}
      EXTRA dasan.syms.extra)
    add_dependencies(dasan clang_rt.dasan-${arch}-symbols)
  endforeach()
endif()
