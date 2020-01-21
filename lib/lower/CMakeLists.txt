set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error")

add_library(FortranLower
  ASTBuilder.cpp
  ConvertType.cpp
)

target_link_libraries(FortranLower
  FIRDialect
  MLIRAffineOps
  MLIRLLVMIR
  MLIRStandardOps
)

install (TARGETS FortranLower
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
