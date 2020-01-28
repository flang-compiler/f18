
add_subdirectory(common)
add_subdirectory(evaluate)
add_subdirectory(decimal)
add_subdirectory(lower)
add_subdirectory(parser)
add_subdirectory(semantics)

# Install header files to mirror a directory structure that follows
# aspects of the LLVM/Clang layout (for full, not toolchain,
# installs).
if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
  install(DIRECTORY common evaluate decimal parser semantics 
    DESTINATION include/flang
    COMPONENT flang-headers
    FILES_MATCHING
    PATTERN "*.def"
    PATTERN "*.h"
    PATTERN "config.h" EXCLUDE
    PATTERN ".git"     EXCLUDE
    PATTERN ".svn"     EXCLUDE)
endif()
