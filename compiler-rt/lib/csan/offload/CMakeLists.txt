include(FindLibcCommonUtils)
if(NOT TARGET llvm-libc-common-utilities)
  return()
endif()

set(CSAN_OFFLOAD_SOURCES
  csan_offload_hsa.cpp
  csan_offload_hsa_interceptors.cpp
  csan_offload_rpc.cpp
  csan_offload_report.cpp
  csan_offload_symbolize.cpp
  )

set(CSAN_OFFLOAD_HEADERS
  hsa.h
  csan_offload.h
  csan_offload_hsa.h
  csan_offload_rpc.h
  csan_offload_symbolize.h
  ../csan_defs.h
  ../csan_offload_packet.h
  )

# No OBJECT_LIBS: sanitizer_common already comes from clang_rt.csan.
add_compiler_rt_runtime(clang_rt.csan_offload
  STATIC
  ARCHS ${UBSAN_SUPPORTED_ARCH}
  SOURCES ${CSAN_OFFLOAD_SOURCES}
  ADDITIONAL_HEADERS ${CSAN_OFFLOAD_HEADERS}
  CFLAGS ${CSAN_CFLAGS}
  LINK_LIBS llvm-libc-common-utilities
  PARENT_TARGET csan)
