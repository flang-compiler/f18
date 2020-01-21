# Test runner infrastructure for Flang. This configures the Flang test trees
# for use by Lit, and delegates to LLVM's lit test handlers.

# f18/flang tools path for lit
set(FLANG_TOOLS_DIR  ${FLANG_BINARY_DIR}/tools/f18/bin)
# bbc/tco tools path for lit
if (FLANG_BUILT_STANDALONE)
 set(FLANG_LLVM_TOOLS_DIR  ${FLANG_BINARY_DIR}/bin)
else()
 set(FLANG_LLVM_TOOLS_DIR  ${LLVM_BINARY_DIR}/bin)
endif()

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(FLANG_TEST_PARAMS
  flang_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py)

set(FLANG_TEST_DEPENDS
  flang
  f18
  llvm-lit
  FileCheck
  count
  not
  bbc
  tco
)

add_lit_testsuite(check-flang "Running the Flang regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  PARAMS ${FLANG_TEST_PARAMS}
  DEPENDS ${FLANG_TEST_DEPENS}
)

set_target_properties(check-flang PROPERTIES FOLDER "Tests")
add_lit_testsuites(FLANG ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${FLANG_TEST_DEPS}
)
