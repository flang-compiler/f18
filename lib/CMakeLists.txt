add_subdirectory(Common)
add_subdirectory(Evaluate)
add_subdirectory(Decimal)
add_subdirectory(Lower)
add_subdirectory(Parser)
add_subdirectory(Semantics)

if(LINK_WITH_FIR)
  add_subdirectory(Optimizer)
endif()

add_library(FortranEvaluateSemantics
  $<TARGET_OBJECTS:EvaluateObjects>
  $<TARGET_OBJECTS:SemanticsObjects>
)

target_link_libraries(FortranEvaluateSemantics
  FortranCommon
  FortranDecimal
  FortranParser
  m
)

if (LIBPGMATH_DIR)
  # If pgmath library is found, it can be used for constant folding.
  find_library(LIBPGMATH pgmath PATHS ${LIBPGMATH_DIR})
  if(LIBPGMATH)
    add_compile_definitions(LINK_WITH_LIBPGMATH)
    target_link_libraries(FortranEvaluateSemantics ${LIBPGMATH})
    message(STATUS "Found libpgmath: ${LIBPGMATH}")
  else()
    message(STATUS "Libpgmath not found in: ${LIBPGMATH_DIR}")
  endif()
endif()

install(TARGETS FortranEvaluateSemantics
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
