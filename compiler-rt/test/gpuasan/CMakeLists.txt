set(GPUASAN_LIT_TESTS_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(GPUASAN_TESTSUITES)
set(GPUASAN_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS})
list(APPEND GPUASAN_TEST_DEPS gpuasan)

foreach(arch ${GPUASAN_SUPPORTED_ARCH})
  set(GPUASAN_TEST_TARGET_ARCH ${arch})
  string(TOLOWER "-${arch}-${OS_NAME}" GPUASAN_TEST_CONFIG_SUFFIX)
  get_test_cc_for_arch(${arch} GPUASAN_TEST_TARGET_CC GPUASAN_TEST_TARGET_CFLAGS)

  if(COMPILER_RT_TARGET_AMDGPU)
    list(APPEND GPUASAN_TEST_TARGET_CFLAGS
         -Wno-multi-gpu -flto -mcpu=native -nogpulib -startfiles -stdlib)
  elseif("${arch}" MATCHES "nvptx")
    list(APPEND GPUASAN_TEST_TARGET_CFLAGS
         -Wno-multi-gpu -flto -march=native -nogpulib -startfiles -stdlib)
  endif()
  list(APPEND GPUASAN_TEST_TARGET_CFLAGS --target=${COMPILER_RT_DEFAULT_TARGET_TRIPLE})
  string(REPLACE ";" " " GPUASAN_TEST_TARGET_CFLAGS "${GPUASAN_TEST_TARGET_CFLAGS}")

  set(CONFIG_NAME ${arch})
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg.py)
  list(APPEND GPUASAN_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME})
endforeach()

add_lit_testsuite(check-gpuasan "Running GPUSanitizer tests"
  ${GPUASAN_TESTSUITES}
  DEPENDS ${GPUASAN_TEST_DEPS})
