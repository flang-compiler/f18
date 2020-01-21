set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error")

add_subdirectory(Transforms)

add_llvm_library(FortranOptimizer
  CodeGen.cpp
  InternalNames.cpp
  IteratedDominanceFrontier.cpp
  KindMapping.cpp
  StdConverter.cpp
)

target_link_libraries(FortranOptimizer
  FIRDialect
)

install (TARGETS FortranOptimizer
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)
