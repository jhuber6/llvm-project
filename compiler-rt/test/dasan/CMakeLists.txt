set(DASAN_LIT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(DASAN_TESTSUITES)
set(DASAN_TEST_DEPS ${SANITIZER_COMMON_LIT_TEST_DEPS})
# Only the host half. A device test also links the copy of this runtime built
# for the GPU, which another runtimes build produces and which nothing here can
# depend on, so the two are kept in step by building the runtimes together.
list(APPEND DASAN_TEST_DEPS dasan)
set(DASAN_TEST_ARCH ${DASAN_SUPPORTED_ARCH})

foreach(arch ${DASAN_TEST_ARCH})
  set(DASAN_TEST_TARGET_ARCH ${arch})
  string(TOLOWER "-${arch}-${OS_NAME}" DASAN_TEST_CONFIG_SUFFIX)
  get_test_cc_for_arch(${arch} DASAN_TEST_TARGET_CC DASAN_TEST_TARGET_CFLAGS)
  string(TOUPPER ${arch} ARCH_UPPER_CASE)
  set(CONFIG_NAME ${ARCH_UPPER_CASE}${OS_NAME}Config)

  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME}/lit.site.cfg.py)
  list(APPEND DASAN_TESTSUITES ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_NAME})
endforeach()

add_lit_testsuite(check-dasan "Running the DeviceAddressSanitizer tests"
  ${DASAN_TESTSUITES}
  DEPENDS ${DASAN_TEST_DEPS})
set_target_properties(check-dasan PROPERTIES FOLDER "compiler-rt Misc")
