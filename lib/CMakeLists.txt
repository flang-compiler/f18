add_subdirectory(common)
add_subdirectory(evaluate)
add_subdirectory(decimal)
add_subdirectory(lower)
add_subdirectory(parser)
add_subdirectory(semantics)

<<<<<<< HEAD
# Install header files to mirror the directory structure that follows
# aspects of the LLVM/Clang layout (for full, not toolchain,
# installs).
=======
# FIXME: Added install of headers to give a directory structure 
# (upon install) that follows aspects of the Clang layout.  If 
# we decide to move header files out of lib this install target
# will need to be moved elsewhere... 
>>>>>>> CMake update:
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
