
######## f18 ##########

add_executable( f18
  f18.cc
)
target_link_libraries( f18
  FortranParser
  FlangSemantics
)

######## test-sema ##########

add_executable( test-sema  
  test-sema.cc
)

find_library(clang_basic NAMES clangBasic PATHS ${LIBCLANG_LIBRARY_DIR} )

target_link_libraries( test-sema
  FortranParser
  FlangSemantics
  ${llvm_common}
  ${clang_basic}
)

