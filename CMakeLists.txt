#===-- CMakeLists.txt ------------------------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#

# FIXME (?) : downgraded requirement here to match the rest of LLVM.  There 
# were no apparent issues but perhaps future PRs will change this? 
cmake_minimum_required(VERSION 3.4.3) 

# RPATH settings on macOS do not affect INSTALL_NAME.
if (POLICY CMP0068) 
  cmake_policy(SET CMP0068 NEW)
  set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif()

# From Clang: Include file check macros honor CMAKE_REQUIRED_LIBRARIES.
if(POLICY CMP0075)   
  cmake_policy(SET CMP0075 NEW)
endif()

# option() honors normal variables.
if (POLICY CMP0077) 
  cmake_policy(SET CMP0077 NEW)
endif()

# Force all of Flang into C++17 mode... 
# FIXME (?) : I've had no issues with building across numerous platforms with 
# the base components of LLVM build in C++14 and Flang with C++17 but 
# this might bite someone as we start tighter integration (e.g. w/ MLIR). 
set(CMAKE_CXX_STANDARD 17)

# Check to see if are part of an LLVM build, if not we need 
# to configure as a standalone project (use LLVM as an external 
# library).
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(Flang)

  set(CONFIG_OUTPUT)
  if (LLVM_CONFIG)

    set(LLVM_CONFIG_FOUND 1)
    message(STATUS "Found llvm_config as ${LLVM_CONFIG}.")
    message(DEPRECATION "Using llvm-config to detect the LLVM instalation is \
            deprecated.  The installed cmake files should be used instead. \
            CMake should be able to detect your LLVM install automatically, \
            but you can also use LLVM_DIR to specify the path to the directory \
            containing LLVMConfig.cmake.")

    set(CONFIG_COMMAND ${LLVM_CONFIG}
      "--assertion-mode"
      "--bindir"
      "--libdir"
      "--includedir"
      "--prefix"
      "--src-root"
      "--cmakedir")

    execute_process(
      COMMAND ${CONFIG_COMMAND}
      RESULT_VARIABLE HAD_ERROR
      OUTPUT_VARIABLE CONFIG_OUTPUT)

    if (NOT HAD_ERROR) 
      string(REGEX REPLACE
        "[ \t]*[\r\n]+[ \t]*" ";"
        CONFIG_OUTPUT ${CONFIG_OUTPUT}) 
    else()
      string(REPLACE ";" " " CONFIG_COMMAND_STR "${CONFIG_COMMAND}")
      message(STATUS "${CONFIG_COMMAND_STR}")
      message(FATAL_ERROR "llvm-config failed with status ${HAD_ERROR}")
    endif()

    list(GET CONFIG_OUTPUT 0 ENABLE_ASSERTIONS)
    list(GET CONFIG_OUTPUT 1 TOOLS_BINARY_DIR)
    list(GET CONFIG_OUTPUT 2 LIBRARY_DIR)
    list(GET CONFIG_OUTPUT 3 INCLUDE_DIR)
    list(GET CONFIG_OUTPUT 4 LLVM_OBJ_ROOT)
    list(GET CONFIG_OUTPUT 5 MAIN_SRC_DIR)
    list(GET CONFIG_OUTPUT 6 LLVM_CONFIG_CMAKE_PATH)

    # Normalize the LLVM_CMAKE_PATH.  --cmakedir might contain 
    # backslahses and CMake assumes slashes as a path... 
    file(TO_CMAKE_PATH ${LLVM_CONFIG_CMAKE_PATH} LLVM_CMAKE_PATH)
  endif()

  if (NOT MSVC_IDE) 
    set(LLVM_ENABLE_ASSERTIONS ${ENABLE_ASSERTIONS}
      CACHE BOOL "Enable assertions.")
    # Assertion should follow llvm-config's. 
    mark_as_advanced(LLVM_ENABLE_ASSERTIONS)
  endif()

  find_package(LLVM REQUIRED HINTS "${LLVM_CMAKE_PATH}")
  list(APPEND CMAKE_MODULE_PATH ${LLVM_DIR})

  # We can't check LLVM_CONFIG here because find_package(LLVM ...) also sets
  # LLVM_CONFIG... 
  if (NOT LLVM_CONFIG_FOUND)
    # Pull in values from LLVMConfig.cmake.  We can drop this once llvm-config 
    # path is removed. 
    set(TOOLS_BINARY_DIR ${LLVM_TOOLS_BINARY_DIR})
    set(LIBRARY_DIR      ${LLVM_LIBRARY_DIR})
    set(INCLUDE_DIR      ${LLVM_INCLUDE_DIR})
    set(LLVM_OBJ_DIR     ${LLVM_BINARY_DIR})

    # The LLVM_CMAKE_PATH variable is set when doing a non-standalone build
    # and used in this project, so we need to make sure set this value.  
    # FIXME: LLVM_CMAKE_DIR comes from LLVMConfig.cmake.  We should rename
    # LLVM_CMAKE_PATH to LLVM_CMAKE_DIR throughout the project. 
    set(LLVM_CMAKE_PATH ${LLVM_CMAKE_DIR})
  endif()

  set(LLVM_TOOLS_BINARY_DIR ${TOOLS_BINARY_DIR} CACHE PATH "Path to llvm/bin.")
  set(LLVM_LIBRARY_DIR      ${LIBARY_DIR} CACHE PATH "Path to llvm/lib.")
  set(LLVM_MAIN_INCLUDE_DIR ${INCLUDE_DIR} CACHE PATH "Path to the llvm/include tree.")
  set(LLVM_BINARY_DIR       ${LLVM_OBJ_ROOT} CACHE PATH "Path to the LLVM build tree.")
  set(LLVM_MAIN_SRC_DIR     ${LLVM_SRC_DIR} CACHE PATH "Path to the LLVM source tree.")

  find_program(LLVM_TABLEGET_EXE "llvm-tblgen" ${LLVM_TOOLS_BINARY_DIR}
    NO_DEFAULT_PATH)
  find_program(MLIR_TABLEGEN_EXE "mlir-tblgen" ${LLVM_TOOLS_BINARY_DIR}
    NO_DEFAULT_PATH)

  set(LLVM_RUNTIME_OUTPUT_INTDIR 
    ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR 
    ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib${LLVM_LIBDIR_SUFFIX})

  if (WIN32 OR CYGWIN)
    # DLL platform -- place DLLs into bin directory.
    set(LLVM_SHLIB_OUTPUT_INTDIR ${LLVM_RUNTIME_OUTPUT_INTDIR})
  else()
    set(LLVM_SHLIB_OUTPUT_INTDIR ${LLVM_LIBRARY_OUTPUT_INTDIR})
  endif()
  
  option(LLVM_ENABLE_WARNINGS "Enable compiler warnings." ON)
  option(LLVM_INSTALL_TOOLCHAIN_ONLY
    "Only include toolchain files in the 'install' target." OFF)

  option(LLVM_FORCE_USE_OLD_HOST_TOOLCHAIN
    "Set to ON to force using an old, unsupported host toolchain." OFF)
  option(CLANG_ENABLE_BOOTSTRAP "Generate the clang bootstrap target" OFF)
  option(LLVM_ENABLE_LIBXML2 "Use libxml2 if available." ON)

  include(AddLLVM)
  include(TableGen)
  include(HandleLLVMOptions)
  include(VersionFromVCS)
# FIXME: uncomment once FIR code is upstreamed. 
#  include(AddMLIR)

  set(PACKAGE_VERSION "${LLVM_PACKAGE_VERSION}")

  if (NOT DEFINED LLVM_INCLUDE_TESTS) 
    set(LLVM_INCLUDE_TESTS ON)
  endif()

  include_directories("${LLVM_BINARY_DIR}/include" "${LLVM_MAIN_INCLUDE_DIR}")
  link_directories("${LLVM_LIBRARY_DIR}")
  set(MLIR_BINARY_DIR ${LLVM_BINARY_DIR})

  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY 
    ${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX})
  set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY 
    ${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX})
  
  if (LLVM_INCLUDE_TESTS)
    include(FindPythonInterp)
    if (NOT PYTHONINTERP_FOUND)
      message(FATAL_ERROR
        "Unable to find Python interpreter, required for builds and testing. \
        Please install Python or specify the PYTHON_EXECUTABLE CMake variable.")
    endif()

    if (${PYTHON_VERSION_STRING} VERSION_LESS 2.7)
      message(FATAL_ERROR "Python 2.7 or newer is required.")
    endif()
  
    # Check prebuilt llvm/utils.
    if (EXISTS ${LLVM_TOOLS_BINARY_DIR}/FileCheck${CMAKE_EXECUTABLE_SUFFIX}
        AND EXISTS ${LLVM_TOOLS_BINARY_DIR}/count${CMAKE_EXECUTABLE_SUFFIX}
        AND EXISTS ${LLVM_TOOLS_BINARY_DIR}/not${CMAKE_EXECUTABLE_SUFFIX})
      set(LLVM_UTILS_PROVIDED ON)
    endif()

    if (EXISTS ${LLVM_MAIN_SRC_DIR}/utils/lit/lit.py)
      # Note: the path is not really used, except for checking if lit was found
      set(LLVM_LIT ${LLVM_MAIN_SRC_DIR}/utils/lit/lit.py)
      if(EXISTS ${LLVM_MAIN_SRC_DIR}/utils/llvm-lit)
        add_subdirectory(${LLVM_MAIN_SRC_DIR}/utils/llvm-lit utils/llvm-lit)
      endif()

    if(NOT LLVM_UTILS_PROVIDED)
      add_subdirectory(${LLVM_MAIN_SRC_DIR}/utils/FileCheck utils/FileCheck)
      add_subdirectory(${LLVM_MAIN_SRC_DIR}/utils/count utils/count)
      add_subdirectory(${LLVM_MAIN_SRC_DIR}/utils/not utils/not)
      set(LLVM_UTILS_PROVIDED ON)
      set(FLANG_TEST_DEPS FileCheck count not)
    endif()

    set(UNITTEST_DIR ${LLVM_MAIN_SRC_DIR}/utils/unittest)
    
    if (EXISTS ${UNITTEST_DIR}/googletest/include/gtest/gtest.h
        AND NOT EXISTS 
          ${LLVM_LIBRARY_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}gtest${CMAKE_STATIC_LIBRARY_SUFFIX}
        AND EXISTS ${UNITTEST_DIR}/CMakeLists.txt)
          add_subdirectory(${UNITTEST_DIR} utils/unittest)
      endif()
    else()
      # Search for an installed lit.
      find_program(LLVM_LIT
                   NAMES llvm-lit lit.py lit
                   PATHS "${LLVM_MAIN_SRC_DIR}/utils/lit"
                   DOC "Path to lit.py")
    endif()

    if (LLVM_LIT)
      # Define the default arguments to use with 'lit', and provide an option for the user
      # to override the arguments.
      set(LIT_ARGS_DEFAULT "-sv")

      if (MSVC OR XCODE)
        set(LIT_ARGS_DEFAULT "${LIT_ARGS_DEFAULT} --no-progress-bar")
      endif()
      
      set(LLVM_LIT_ARGS "${LIT_ARGS_DEFAULT}" CACHE STRING "Default options for lit")

      # On Win32 hosts, provide an option to specify the path to the GnuWin32 tools.
      if( WIN32 AND NOT CYGWIN )
        set(LLVM_LIT_TOOLS_DIR "" CACHE PATH "Path to GnuWin32 tools")
      endif()

    else()
      set(LLVM_INCLUDE_TESTS OFF)
    endif()

  endif()

  set(FLANG_BUILT_STANDALONE 1)
  set(BACKEND_PACKAGE_STRING "LLVM ${LLVM_PACKAGE_VERSION}")
else()
  set(BACKEND_PACKAGE_STRING "${PACKAGE_STRING}")
  set(MLIR_BINARY_DIR ${LLVM_BINARY_DIR}/tools/mlir)
  set(MLIR_SOURCE_DIR ${LLVM_SOURCE_DIR}/../mlir)
endif()

# Make sure that the flang source directory is on the current cmake module path so that 
# we can include flang-centric cmake files from within the project. 
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

if(LLVM_ENABLE_LIBXML2)
  # Don't look for libxml if we're using MSan, since uninstrumented third party
  # code may call MSan interceptors like strlen, leading to false positives.
  if(NOT LLVM_USE_SANITIZER MATCHES "Memory.*")
    set (LIBXML2_FOUND 0)
    find_package(LibXml2 2.5.3 QUIET)
    if (LIBXML2_FOUND)
      set(FLANG_HAVE_LIBXML 1)
    endif()
  endif()
endif()

option(LINK_WITH_FIR "Link driver with FIR and LLVM" ON)
if (LINK_WITH_FIR)
  message(STATUS "Linking driver with FIR and LLVM")
  llvm_map_components_to_libnames(LLVM_COMMON_LIBS support)
  message(STATUS "LLVM libraries: ${LLVM_COMMON_LIBS}")
endif()

set(FLANG_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(FLANG_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

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

set(FLANG_C_INCLUDE_DIRS "" CACHE STRING
  "Colon separated list of directories Flang will use to search for header files.")

if (NOT GCC_INSTALL_PREFIX)   # FIXME: Need to doublecheck that this actually behaves correctly. 
  set(GCC_INSTALL_PREFIX "" CACHE PATH
    "The <path> to directory where GCC is installed.")
endif()

if (NOT DEFAULT_SYSROOT)  # FIXME: Need to doublecheck that this actually behaves correctly. 
  set(DEFAULT_SYSROOT "" CACHE PATH
    "The <path> to use for the system root for all compiler invocations (--sysroot=<path>).")
endif()

if (NOT ENABLE_LINLDER_BUILD_ID)
  set(ENABLE_LINKER_BUILD_ID OFF CACHE BOOL "pass --build-id to ld")
endif()

set(FLANG_DEFAULT_LINKER "" CACHE STRING
  "Default linker to use (linker name or absolute path, empty for platform default)")

set(FLANG_DEFAULT_RTLIB "" CACHE STRING 
   "Default Fortran runtime library to use (\"libFortranRuntime\"), leave empty for platform default.")


if (NOT(FLANG_DEFAULT_RTLIB STREQUAL ""))
  message(WARNING "Resetting Flang's default runtime library to use platform default.")
  set(FLANG_DEFAULT_RTLIB "" CACHE STRING
      "Default runtime library to use (\"libgcc\" or \"compiler-rt\", empty for platform default)" FORCE)
endif()

set(FLANG_VENDOR ${PACKAGE_VENDOR} CACHE STRING
  "Vendor-specific text for showing with version information.")

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

# FIXME: Redundant w/ above... 
#set(FLANG_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
#set(FLANG_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR AND NOT MSVC_IDE)
  message(FATAL_ERROR "In-source builds are not allowed. \
          Please create a directory and run cmake from there,\
          passing the path to this source directory as the last argument.\
          This process created the file `CMakeCache.txt' and the directory\
          `CMakeFiles'. Please delete them.")
endif()


# Override LLVM versioning for now... 
# 
# FIXME: Once merged We need to determine how to we want to version early 
# Flang relative to LLVM as a whole.  
set(FLANG_VERSION_MAJOR      "0")
set(FLANG_VERSION_MINOR      "1")
set(FLANG_VERSION_PATCHLEVEL "0")

if (NOT DEFINED FLANG_VERSION_MAJOR)
  set(FLANG_VERSION_MAJOR ${LLVM_VERSION_MAJOR})
endif ()

if (NOT DEFINED FLANG_VERSION_MINOR)
  set(FLANG_VERSION_MINOR ${LLVM_VERSION_MINOR})
endif ()

if (NOT DEFINED FLANG_VERSION_PATCHLEVEL)
  set(FLANG_VERSION_PATCHLEVEL ${LLVM_VERSION_PATCH})
endif ()

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

if (FLANG_LINK_WITH_FIR)
  message(STATUS "Linking Flang driver with FIR and LLVM")
  llvm_map_components_to_libnames(LLVM_COMMON_LIBS support)
  message(STATUS "Flang using LLVM libraries: ${LLVM_COMMON_LIBS}")
endif()

# Add appropriate flags for GCC
if (LLVM_COMPILER_IS_GCC_COMPATIBLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fno-rtti -fno-exceptions \
        -Wall -Wextra -Werror -Wcast-qual -Wimplicit-fallthrough -Wdelete-non-virtual-dtor")

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

# FIXME: Couldn't find this is used in the current master branch.  Not
# sure why we need it but left it in under the assumption it was necessary 
# for the upcoming PRs (FIR/MLIR, etc.). 
add_definitions(-D_GNU_SOURCE)

option(FLANG_INCLUDE_TESTS
       "Generate build targets for the Flang unit tests."
       ${LLVM_INCLUDE_TESTS})

add_subdirectory(include)
add_subdirectory(lib)

option(FLANG_BUILD_TOOLS
  "Build the Flang tools. If OFF, just generate build targets." ON)
if (FLANG_BUILD_TOOLS)
  add_subdirectory(tools)
endif()

add_subdirectory(runtime)

# FIXME: Does not look like we use anything here that needs CMake support.
# Once again leaving it here in case it is going to be needed at a later 
# date... 
#option(FLANG_INCLUDE_DOCS "Generate build targets for the Flang docs."
#  ${LLVM_INCLUDE_DOCS})
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
  
