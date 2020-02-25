cmake_minimum_required(VERSION 3.4.3) 

# RPATH settings on macOS do not affect INSTALL_NAME.
if (POLICY CMP0068) 
  cmake_policy(SET CMP0068 NEW)
  set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif()

# Include file check macros honor CMAKE_REQUIRED_LIBRARIES.
if(POLICY CMP0075)   
  cmake_policy(SET CMP0075 NEW)
endif()

# option() honors normal variables.
if (POLICY CMP0077) 
  cmake_policy(SET CMP0077 NEW)
endif()

# Flang requires C++17. 
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(FLANG_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(FLANG_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR AND NOT MSVC_IDE)
  message(FATAL_ERROR "In-source builds are not allowed. \
          Please create a directory and run cmake from there,\
          passing the path to this source directory as the last argument.\
          This process created the file `CMakeCache.txt' and the directory\
          `CMakeFiles'. Please delete them.")
endif()

# Check for a standalone build and configure as appropriate from
# there. 
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  message("Building Flang as a standalone project.")
  project(Flang)

  if (NOT MSVC_IDE)
    set(LLVM_ENABLE_ASSERTIONS ${ENABLE_ASSERTIONS}
      CACHE BOOL "Enable assertions")
    # Assertions follow llvm's configuration.
    mark_as_advanced(LLVM_ENABLE_ASSERTIONS)
  endif()

  # We need a pre-built/installed version of LLVM. 
  find_package(LLVM REQUIRED HINTS "${LLVM_CMAKE_PATH}")
  list(APPEND CMAKE_MODULE_PATH ${LLVM_DIR})  

  find_program(MLIR_TABLEGEN_EXE "mlir-tblgen" ${LLVM_TOOLS_BINARY_DIR}
               NO_DEFAULT_PATH)

  option(LLVM_ENABLE_WARNINGS "Enable compiler warnings." ON)
  option(LLVM_INSTALL_TOOLCHAIN_ONLY
    "Only include toolchain files in the 'install' target." OFF)
  option(LLVM_FORCE_USE_OLD_HOST_TOOLCHAIN
    "Set to ON to force using an old, unsupported host toolchain." OFF)

  include(AddLLVM)    
  include(HandleLLVMOptions)  
  include(VersionFromVCS)
  #include(LLVMDistributionSupport)
  #include(AddMLIR)  # TODO: Uncomment for FIR support. 

  # Add LLVM include files as if they were SYSTEM because there are complex unused
  # parameter issues that may or may not appear depending on the environments and
  # compilers (ifdefs are involved). This allows warnings from LLVM headers to be
  # ignored while keeping -Wunused-parameter a fatal error inside f18 code base.
  # This may have to be fine-tuned if flang headers are consider part of this
  # LLVM_INCLUDE_DIRS when merging in the monorepo (Warning from flang headers
  # should not be suppressed).
  include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
  add_definitions(${LLVM_DEFINITIONS})
  
  # LLVM's cmake configuration files currently sneak in a c++11 flag.
  # We look for it here and remove it from Flang's compile flags to 
  # avoid some mixed compilation flangs (e.g. -std=c++11 ... -std=c++17). 
  if (DEFINED LLVM_CXX_STD) 
    message("LLVM configuration set a C++ standard: ${LLVM_CXX_STD}")
    if (NOT LLVM_CXX_STD EQUAL "c++17")
      message("Flang: Overriding LLVM's 'cxx_std' setting...")
      message("    removing '-std=${LLVM_CXX_STD}'")
      message("    CMAKE_CXX_FLAGS='${CMAKE_CXX_FLAGS}'")
      string(REPLACE " -std=${LLVM_CXX_STD}" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
      message("    [NEW] CMAKE_CXX_FLAGS='${CMAKE_CXX_FLAGS}'")
    endif()
  endif()

  set(PACKAGE_VERSION "${LLVM_PACKAGE_VERSION}")
    
  link_directories("${LLVM_LIBRARY_DIR}")
  set(MLIR_BINARY_DIR ${LLVM_BINARY_DIR})

  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY 
    ${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX})
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY 
    ${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX})
  
  set(BACKEND_PACKAGE_STRING "LLVM ${LLVM_PACKAGE_VERSION}")
else()
  set(BACKEND_PACKAGE_STRING "${PACKAGE_STRING}")
  set(MLIR_BINARY_DIR ${LLVM_BINARY_DIR}/tools/mlir)
  set(MLIR_SOURCE_DIR ${LLVM_SOURCE_DIR}/../mlir)
endif()

# Add Flang-centric modules to cmake path. 
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

if (MLIR_SOURCE_DIR) 
  include_directories(BEFORE
    ${FLANG_BINARY_DIR}/include 
    ${FLANG_SOURCE_DIR}/include 
    ${MLIR_BINARY_DIR}/include 
    ${MLIR_SOURCE_DIR}/include)
  set(MLIR_MAIN_SRC_DIR ${MLIR_SOURCE_DIR})
  set(MLIR_INCLUDE_DIR ${MLIR_SOURCE_DIR}/include)
else ()
  include_directories(BEFORE 
    ${FLANG_BINARY_DIR}/include 
    ${FLANG_SOURCE_DIR}/include 
    ${MLIR_BINARY_DIR}/include)
  set(MLIR_MAIN_SRC_DIR ${MLIR_BINARY_DIR})
  set(MLIR_INCLUDE_DIR ${MLIR_BINARY_DIR}/include)
endif()

set(MLIR_TABLEGEN_EXE mlir-tblgen)

if (NOT DEFAULT_SYSROOT)  # FIXME: Need to doublecheck that this actually behaves correctly. 
  set(DEFAULT_SYSROOT "" CACHE PATH
    "The <path> to use for the system root for all compiler invocations (--sysroot=<path>).")
endif()

if (NOT ENABLE_LINKER_BUILD_ID)
  set(ENABLE_LINKER_BUILD_ID OFF CACHE BOOL "pass --build-id to ld")
endif()

set(FLANG_DEFAULT_LINKER "" CACHE STRING
  "Default linker to use (linker name or absolute path, empty for platform default)")

set(FLANG_DEFAULT_RTLIB "" CACHE STRING 
   "Default Fortran runtime library to use (\"libFortranRuntime\"), leave empty for platform default.")

if (NOT(FLANG_DEFAULT_RTLIB STREQUAL ""))
  message(WARNING "Resetting Flang's default runtime library to use platform default.")
  set(FLANG_DEFAULT_RTLIB "" CACHE STRING
      "Default runtime library to use (empty for platform default)" FORCE)
endif()


if(CMAKE_COMPILER_IS_GNUCXX OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
   if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      if(BUILD_WITH_CLANG_LIBRARIES)
         set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -nostdinc++ -I${BUILD_WITH_CLANG_LIBRARIES}/include/c++/v1 -DCLANG_LIBRARIES")
         set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++ -Wl,-rpath,${BUILD_WITH_CLANG_LIBRARIES}/lib -L${BUILD_WITH_CLANG_LIBRARIES}/lib")
      else()
         set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lstdc++")
      endif()
      if(GCC)
         set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --gcc-toolchain=${GCC}")
      endif()
   endif()

   set(CMAKE_CXX_FLAGS_RELEASE    "-O2")
   set(CMAKE_CXX_FLAGS_MINSIZEREL "-O2 '-DCHECK=(void)'")
   set(CMAKE_CXX_FLAGS_DEBUG      "-g -DDEBUGF18")

   # Building shared libraries is death on performance with GCC by default
   # due to the need to preserve the right to override external entry points
   # at dynamic link time.  -fno-semantic-interposition waives that right and
   # recovers a little bit of that performance.
   if (BUILD_SHARED_LIBS AND NOT (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
     set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fno-semantic-interposition")
   endif()
endif()


set(FLANG_VENDOR ${PACKAGE_VENDOR} CACHE STRING
  "Vendor-specific Flang version information.")

  if (FLANG_VENDOR)
  add_definitions(-DFLANG_VENDOR="${FLANG_VENDOR} ")
endif()

set(FLANG_REPOSITORY_STRING "" CACHE STRING
  "Vendor-specific text for showing the repository the source is taken from.")

if (FLANG_REPOSITORY_STRING)
  add_definitions(-DFLANG_REPOSITORY_STRING="${FLANG_REPOSITORY_STRING}")
endif()

set(FLANG_VENDOR_UTI "org.llvm.flang" CACHE STRING
  "Vendor-specific uti.")

# The libdir suffix must exactly match whatever LLVM's configuration used.
set(FLANG_LIBDIR_SUFFIX "${LLVM_LIBDIR_SUFFIX}")


# Override LLVM versioning for now... 
set(FLANG_VERSION_MAJOR      "0")
set(FLANG_VERSION_MINOR      "1")
set(FLANG_VERSION_PATCHLEVEL "0")

if (NOT DEFINED FLANG_VERSION_MAJOR)
  set(FLANG_VERSION_MAJOR ${LLVM_VERSION_MAJOR})
endif()

if (NOT DEFINED FLANG_VERSION_MINOR)
  set(FLANG_VERSION_MINOR ${LLVM_VERSION_MINOR})
endif()

if (NOT DEFINED FLANG_VERSION_PATCHLEVEL)
  set(FLANG_VERSION_PATCHLEVEL ${LLVM_VERSION_PATCH})
endif()

# Unlike PACKAGE_VERSION, FLANG_VERSION does not include LLVM_VERSION_SUFFIX.
set(FLANG_VERSION "${FLANG_VERSION_MAJOR}.${FLANG_VERSION_MINOR}.${FLANG_VERSION_PATCHLEVEL}")
message(STATUS "Flang version: ${FLANG_VERSION}")
# Flang executable version information
set(FLANG_EXECUTABLE_VERSION
    "${FLANG_VERSION_MAJOR}" CACHE STRING
    "Major version number to appended to the flang executable name.")
set(LIBFLANG_LIBRARY_VERSION
    "${FLANG_VERSION_MAJOR}" CACHE STRING
    "Major version number to appended to the libflang library.")

mark_as_advanced(FLANG_EXECUTABLE_VERSION LIBFLANG_LIBRARY_VERSION)

# Configure Flang's Version.inc file.
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/flang/Version.inc.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/flang/Version.inc)

# Add appropriate flags for GCC
if (LLVM_COMPILER_IS_GCC_COMPATIBLE)
  if (NOT "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-strict-aliasing -fno-semantic-interposition")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-command-line-argument -Wstring-conversion \
          -Wcovered-switch-default")

    # The following works around warnings in the f18 sources
    check_cxx_compiler_flag("-Werror -Wdeprecated-copy" CXX_SUPPORTS_NO_DEPRECATED_COPY_FLAG)
    if (CXX_SUPPORTS_NO_DEPRECATED_COPY_FLAG)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-copy")
    endif()

    check_cxx_compiler_flag("-Werror -Wstring-conversion" CXX_SUPPORTS_NO_STRING_CONVERSION_FLAG)

    if (CXX_SUPPORTS_NO_STRING_CONVERSION_FLAG)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-string-conversion")
    endif()
  endif()

  # Enable -pedantic for Flang even if it's not enabled for LLVM.
  if (NOT LLVM_ENABLE_PEDANTIC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic")
  endif ()

  check_cxx_compiler_flag("-Werror -Wnested-anon-types" CXX_SUPPORTS_NO_NESTED_ANON_TYPES_FLAG)
  if (CXX_SUPPORTS_NO_NESTED_ANON_TYPES_FLAG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-nested-anon-types")
  endif()

  if (GCC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --gcc-toolchain=${GCC}")
  endif()

  set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} -DCHECK=(void)")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUGF18")

  # Building shared libraries is death on performance with GCC by default
  # due to the need to preserve the right to override external entry points
  if (BUILD_SHARED_LIBS AND NOT (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
   set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fno-semantic-interposition")
  endif()
  
  list(REMOVE_DUPLICATES CMAKE_CXX_FLAGS)

endif()


# Determine HOST_LINK_VERSION on Darwin.
set(HOST_LINK_VERSION)
if (APPLE)
  set(LD_V_OUTPUT)
  execute_process(
    COMMAND sh -c "${CMAKE_LINKER} -v 2>&1 | head -1"
    RESULT_VARIABLE HAD_ERROR
    OUTPUT_VARIABLE LD_V_OUTPUT)
  if (NOT HAD_ERROR)
    if ("${LD_V_OUTPUT}" MATCHES ".*ld64-([0-9.]+).*")
      string(REGEX REPLACE ".*ld64-([0-9.]+).*" "\\1" HOST_LINK_VERSION ${LD_V_OUTPUT})
    elseif ("${LD_V_OUTPUT}" MATCHES "[^0-9]*([0-9.]+).*")
      string(REGEX REPLACE "[^0-9]*([0-9.]+).*" "\\1" HOST_LINK_VERSION ${LD_V_OUTPUT})
    endif()
  else()
    message(FATAL_ERROR "${CMAKE_LINKER} failed with status ${HAD_ERROR}")
  endif()
endif()

include(CMakeParseArguments)
include(AddFlang)

if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)

  install(DIRECTORY include/flang
    DESTINATION include
    COMPONENT flang-headers
    FILES_MATCHING
    PATTERN "*.def"
    PATTERN "*.h"
    PATTERN "config.h" EXCLUDE
    PATTERN ".git"     EXCLUDE
    PATTERN ".svn"     EXCLUDE)

  install(DIRECTORY ${FLANG_BINARY_DIR}/include/flang
    DESTINATION include
    COMPONENT flang-headers
    FILES_MATCHING
    PATTERN "CMakeFiles" EXCLUDE
    PATTERN "*.inc"
    PATTERN "*.h")

endif()

add_subdirectory(include)
add_subdirectory(lib)

option(FLANG_BUILD_TOOLS
  "Build the Flang tools. If OFF, just generate build targets." ON)
if (FLANG_BUILD_TOOLS)
  add_subdirectory(tools)
endif()

add_subdirectory(runtime)

option(FLANG_INCLUDE_TESTS
       "Generate build targets for the Flang unit tests."
       ${LLVM_INCLUDE_TESTS})

if (FLANG_INCLUDE_TESTS)
  enable_testing()
  #add_subdirectory(test)
  #add_subdirectory(test-lit)
endif()

# TODO: Add doxygen support.
#option(FLANG_INCLUDE_DOCS "Generate build targets for the Flang docs."
#       ${LLVM_INCLUDE_DOCS})
#if (FLANG_INCLUDE_DOCS)
#  add_subdirectory(documentation)
#endif()

# Custom target to install Flang libraries.
add_custom_target(flang-libraries)
set_target_properties(flang-libraries PROPERTIES FOLDER "Misc")

if (NOT LLVM_ENABLE_IDE)
  add_llvm_install_targets(install-flang-libraries
    DEPENDS flang-libraries
   COMPONENT flang-libraries)
endif()

get_property(FLANG_LIBS GLOBAL PROPERTY FLANG_LIBS)
if (FLANG_LIBS)
  list(REMOVE_DUPLICATES FLANG_LIBS)
  foreach(lib ${FLANG_LIBS})
    add_dependencies(flang-libraries ${lib})
    if (NOT LLVM_ENABLE_IDE)
      add_dependencies(install-flang-libraries install-${lib})
    endif()
  endforeach()
endif()

add_subdirectory(cmake/modules)

configure_file(
  ${FLANG_SOURCE_DIR}/include/flang/Config/config.h.cmake
  ${FLANG_BINARY_DIR}/include/flang/Config/config.h)
  
