
add_library(FortranDecimal
  binary-to-decimal.cpp
  decimal-to-binary.cpp
)

install (TARGETS FortranDecimal
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
