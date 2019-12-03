# The flang cmake configuration is mostly based off of Clang with a
# few tweaks and deletions (e.g. tlbgen is currently not used in Flang).
# This should hopefully make it easier to keep the build process aligned
# across the LLVM infrastructure but will require we keep an eye on it
# such that they don't significantly diverge... 
#
# Given that flang requires C++17, each of the Flang libraries, tools,
# etc. all call that out with a CMake-centric compiler feature: 
# 
#   target_compile_features(libFlangCommon PUBLIC cxx_std_17)
#
#   FIXME: Need to verify this is actually working as documented.
#
# FIXME?? : Note that the requirement of C++17, combined with piggybacking
# on the LLVM/Clang cmake structure currently results in both C++11
# and C++17 command line arguments appearing.  E.g., when building
# flang you'll noticed both --std=c++11 and --std=c++17 on the
# compile lines.  
#
# FIXME: Windows platforms have not yet been tested -- nor the MSVC IDE
# pieces.
# 
cmake_minimum_required(VERSION 3.9.0)

if (POLICY CMP0068)
  cmake_policy(SET CMP0068 NEW)
  set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif()

if (POLICY CMP0077)
  # FIXME??: Not yet certain this is the behavior we want... 
  cmake_policy(SET CMP0077 NEW)
endif()

# If we are not building as part of LLVM (e.g., within the mono-repo),
# build Flang as a standalone project.  This assumes the use of an
# existing LLVM install w/ external libraries.
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)

   project(Flang)

   # FIXME: We currently rely on llvm-config to find all the
   # details for us.  This is deprecated.  The cmake files installed
   # with LLVM should be able to detect the LLVM install automatically.
   # If not, you can use the LLMV_DIR to specify the path containing
   # LLVMConfig.cmake.
   set(CONFIG_OUTPUT)
   if (LLVM_CONFIG)
      set(LLVM_CONFIG_FOUND 1)
      message(STATUS "Found LLVM_CONFIG as ${LLVM_CONFIG}...")
      message(DEPRECATION "Using llvm-config is deprecated.  Use the \
                           LLVM installed CMake files instead. \
			   This should allow CMake to automatically find \
			   find your install.  You can also use LLVM_DIR \
			   to specify the path to LLVMConfig.cmake.")
      set(CONFIG_COMMAND ${LLVM_CONFIG}
          "--assertion-mode"
	  "--bindir"
	  "--libdir"
	  "--includedir"
	  "--prefix"
	  "--src-root"
	  "--cmakedir")
      execute_process(
         COMMAND   ${CONFIG_COMMAND}
	 RESULT_VARIABLE HAD_ERROR
	 OUTPUT_VARIABLE CONFIG_OUTPUT
      )

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

    # Normalize LLVM_CMAKE_PATH. --cmakedir might contain backslashes.
    # CMake assumes slashes as PATH.
    file(TO_CMAKE_PATH ${LLVM_CONFIG_CMAKE_PATH} LLVM_CMAKE_PATH)
  endif()

  if (NOT MSVC_IDE)
    set(LLVM_ENABLE_ASSERTIONS ${ENABLE_ASSERTIONS}
        CACHE BOOL "Enable assertions")
    # Assertions should follow llvm-config's.
    mark_as_advanced(LLVM_ENABLE_ASSERTIONS)
  endif()

  find_package(LLVM REQUIRED HINTS "${LLVM_CMAKE_PATH}")
  list(APPEND CMAKE_MODULE_PATH ${LLVM_DIR})

  # We can't check LLVM_CONFIG here, because find_package(LLVM ...) also sets
  # LLVM_CONFIG.
  if (NOT LLVM_CONFIG_FOUND)
    # Pull values from LLVMConfig.cmake.  We can drop this once the llvm-config
    # path is removed.
    set(TOOLS_BINARY_DIR ${LLVM_TOOLS_BINARY_DIR})
    set(LIBRARY_DIR ${LLVM_LIBRARY_DIR})
    set(INCLUDE_DIR ${LLVM_INCLUDE_DIR})
    set(LLVM_OBJ_DIR ${LLVM_BINARY_DIR})
    # The LLVM_CMAKE_PATH variable is set when doing non-standalone builds and
    # used in this project, so we need to make sure we set this value.
    # FIXME: LLVM_CMAKE_DIR comes from LLVMConfig.cmake.  We should rename
    # LLVM_CMAKE_PATH to LLVM_CMAKE_DIR throughout the project.
    set(LLVM_CMAKE_PATH ${LLVM_CMAKE_DIR})
  endif()

  set(LLVM_TOOLS_BINARY_DIR ${TOOLS_BINARY_DIR} CACHE PATH "Path to llvm/bin")
  set(LLVM_LIBRARY_DIR ${LIBRARY_DIR} CACHE PATH "Path to llvm/lib")
  set(LLVM_MAIN_INCLUDE_DIR ${INCLUDE_DIR} CACHE PATH "Path to llvm/include")
  set(LLVM_BINARY_DIR ${LLVM_OBJ_ROOT} CACHE PATH "Path to LLVM build tree")
  set(LLVM_MAIN_SRC_DIR ${MAIN_SRC_DIR} CACHE PATH "Path to LLVM source tree")

  # FIXME?? -- skipped tablegen stuff for now... 

  option(LLVM_ENABLE_WARNINGS "Enable compiler warnings." ON)
  option(LLVM_INSTALL_TOOLCHAIN_ONLY
    "Only include toolchain files in the 'install' target." OFF)

  include(AddLLVM)
  include(HandleLLVMOptions)
  include(VersionFromVCS)

  set(PACKAGE_VERSION "${LLVM_PACKAGE_VERSION}")

  if (NOT DEFINED LLVM_INCLUDE_TESTS)
    set(LLVM_INCLUDE_TESTS ON)
  endif()

  include_directories("${LLVM_BINARY_DIR}/include" "${LLVM_MAIN_INCLUDE_DIR}")
  link_directories("${LLVM_LIBRARY_DIR}")

  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX})
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX})
  
  if (LLVM_INCLUDE_TESTS)
    set(Python_ADDITIONAL_VERSIONS 2.7)
    include(FindPythonInterp)
    if (NOT PYTHONINTERP_FOUND)
      message(FATAL_ERROR
              "Unable to find Python interpreter, required for builds and testing.
               Please install Python or specify the PYTHON_EXECUTABLE CMake variable.")
    endif()

    if (${PYTHON_VERSION_STRING} VERSION_LESS 2.7)
      message(FATAL_ERROR "Python 2.7 or newer is required")
    endif()


    # Check prebuilt llvm/utils.
    if (EXISTS ${LLVM_TOOLS_BINARY_DIR}/FileCheck${CMAKE_EXECUTABLE_SUFFIX}
        AND EXISTS ${LLVM_TOOLS_BINARY_DIR}/count${CMAKE_EXECUTABLE_SUFFIX}
        AND EXISTS ${LLVM_TOOLS_BINARY_DIR}/not${CMAKE_EXECUTABLE_SUFFIX})
      set(LLVM_UTILS_PROVIDED ON)
    endif()

    if (EXISTS ${LLVM_MAIN_SRC_DIR}/utils/lit/lit.py)
      # Note: path not really used, except for checking if lit was found
      set(LLVM_LIT ${LLVM_MAIN_SRC_DIR}/utils/lit/lit.py)
      if (EXISTS ${LLVM_MAIN_SRC_DIR}/utils/llvm-lit)
        add_subdirectory(${LLVM_MAIN_SRC_DIR}/utils/llvm-lit utils/llvm-lit)
      endif()
      if (NOT LLVM_UTILS_PROVIDED)
        add_subdirectory(${LLVM_MAIN_SRC_DIR}/utils/FileCheck utils/FileCheck)
        add_subdirectory(${LLVM_MAIN_SRC_DIR}/utils/count utils/count)
        add_subdirectory(${LLVM_MAIN_SRC_DIR}/utils/not utils/not)
        set(LLVM_UTILS_PROVIDED ON)
        set(FLANG_TEST_DEPS FileCheck count not)
      endif()

      set(UNITTEST_DIR ${LLVM_MAIN_SRC_DIR}/utils/unittest)
      if (EXISTS ${UNITTEST_DIR}/googletest/include/gtest/gtest.h
          AND NOT EXISTS ${LLVM_LIBRARY_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}gtest${CMAKE_STATIC_LIBRARY_SUFFIX}
          AND EXISTS ${UNITTEST_DIR}/CMakeLists.txt)
        add_subdirectory(${UNITTEST_DIR} utils/unittest)
      endif()
    else()
      # Seek installed Lit.
      find_program(LLVM_LIT
                   NAMES llvm-lit lit.py lit
                   PATHS "${LLVM_MAIN_SRC_DIR}/utils/lit"
                   DOC "Path to lit.py")
    endif()

    if (LLVM_LIT)
      # Define the default arguments to use with 'lit', and an option for the user to override.
      set(LIT_ARGS_DEFAULT "-sv")
      if (MSVC OR XCODE)
        set(LIT_ARGS_DEFAULT "${LIT_ARGS_DEFAULT} --no-progress-bar")
      endif()
      set(LLVM_LIT_ARGS "${LIT_ARGS_DEFAULT}" CACHE STRING "Default options for lit")

      # On Win32 hosts, provide an option to specify the path to the GnuWin32 tools.
      if (WIN32 AND NOT CYGWIN)
        set(LLVM_LIT_TOOLS_DIR "" CACHE PATH "Path to GnuWin32 tools")
      endif()
    else()
      set(LLVM_INCLUDE_TESTS OFF)
    endif()
  endif()

  set(FLANG_BUILD_STANDALONE 1)
  set(BACKEND_PACKAGE_STRING "LLVM ${LLVM_PACKAGE_VERSION}")
else()
  set(BACKEND_PACKAGE_STRING "${PACKAGE_STRING}")
endif()

# Make sure that our source directory is on the current cmake module path so that
# we can include cmake files from this directory.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

option(FLANG_LINK_WITH_FIR "Link Flang driver with FIR and LLVM" ON)
if (LINK_WITH_FIR)
  message(STATUS "Linking driver with FIR and LLVM")
  llvm_map_components_to_libnames(LLVM_COMMON_LIBS support)
  message(STATUS "LLVM libraries: ${LLVM_COMMON_LIBS}")
endif()

set(FLANG_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(FLANG_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(MLIR_SOURCE_DIR ${LLVM_SOURCE_DIR}/projects/mlir)
set(MLIR_BINARY_DIR ${LLVM_BINARY_DIR}/projects/mlir)

include_directories(BEFORE
  ${FLANG_BINARY_DIR}/include
  ${FLANG_SOURCE_DIR}/include
  ${MLIR_BINARY_DIR}/include
  ${MLIR_SOURCE_DIR}/include
  )

set(MLIR_MAIN_SRC_DIR ${MLIR_SOURCE_DIR})
set(MLIR_INCLUDE_DIR ${MLIR_SOURCE_DIR}/include)
set(MLIR_TABLEGEN_EXE mlir-tblgen)

set(FLANG_C_INCLUDE_DIRS "" CACHE STRING
  "Colon separated list of directories flang will search for headers.")

set(GCC_INSTALL_PREFIX "" CACHE PATH "Directory where gcc is installed.")
set(DEFAULT_SYSROOT "" CACHE PATH
  "Default <path> to all compiler invocations for --sysroot=<path>.")

set(ENABLE_LINKER_BUILD_ID OFF CACHE BOOL "pass --build-id to ld")

set(FLANG_DEFAULT_LINKER "" CACHE STRING
  "Default linker to use (linker name or absolute path, empty for platform default)")

set(FLANG_DEFAULT_RTLIB "" CACHE STRING "Default Fortran runtime library to use (\"libFortranRuntime\", empty for platform default)")
if (NOT(FLANG_DEFAULT_RTLIB STREQUAL ""))
  message(WARNING "Resetting Flang's default runtime library to use platform default")
  set(FLANG_DEFAULT_RTLIB "" CACHE STRING
      "Default runtime library to use (\"libgcc\" or \"compiler-rt\", empty for platform default)" FORCE)
endif()

# FIXME: Not handling OpenMP details here...


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

# FIXME?? No Python bindings...

# The libdir suffix must exactly match whatever LLVM's configuration used.
set(FLANG_LIBDIR_SUFFIX "${LLVM_LIBDIR_SUFFIX}")

set(FLANG_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(FLANG_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR AND NOT MSVC_IDE)
  message(FATAL_ERROR "In-source builds are not allowed. "
"Please create a directory and run cmake "
"from there, passing the path to this source directory as the last argument. "
"This process created the file `CMakeCache.txt' and the directory "
"`CMakeFiles'. Please delete them.")
endif()

# Override version information to reflect early stages of flang vs.
# following LLVM's versioning info...
set(FLANG_VERSION_MAJOR      "0")
set(FLANG_VERSION_MINOR      "1")
set(FLANG_VERSION_PATCHLEVEL "0")

# If FLANG_VERSION_* is specified, use it, if not use LLVM_VERSION_*.
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

# Configure the Version.inc file.
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/flang/Version.inc.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/flang/Version.inc)

if (FLANG_LINK_WITH_FIR)
  message(STATUS "Linking Flang driver with FIR and LLVM")
  llvm_map_components_to_libnames(LLVM_COMMON_LIBS support)
  message(STATUS "Flang using LLVM libraries: ${LLVM_COMMON_LIBS}")
endif()

#set(FLANG_CXX_FLAGS "-std=c++17")

# Add appropriate flags for GCC
if (LLVM_COMPILER_IS_GCC_COMPATIBLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fno-rtti -fno-exceptions -Wall")
  if (NOT "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-strict-aliasing -fno-semantic-interposition")
  endif ()

  # Enable -pedantic for Flang even if it's not enabled for LLVM.
  if (NOT LLVM_ENABLE_PEDANTIC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic")
  endif ()

  if ((CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
    if (APPLE)
      # don't add --gcc-toolchain
    else()
      #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --gcc-toolchain=${CMAKE_CXX_COMPILER}")
    endif()
  endif()
  
  check_cxx_compiler_flag("-Werror -Wnested-anon-types" CXX_SUPPORTS_NO_NESTED_ANON_TYPES_FLAG)
  if (CXX_SUPPORTS_NO_NESTED_ANON_TYPES_FLAG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-nested-anon-types")
  endif()

  if (GCC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --gcc-toolchain=${GCC}")
  endif ()

  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUGF18")

   # Building shared libraries is death on performance with GCC by default
   # due to the need to preserve the right to override external entry points
   # at dynamic link time.  -fno-semantic-interposition waives that right and
   # recovers a little bit of that performance.
   if (BUILD_SHARED_LIBS AND NOT (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
     set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fno-semantic-interposition")
   endif()
  
endif ()

# Determine HOST_LINK_VERSION on Darwin.
set(HOST_LINK_VERSION)
if (APPLE)
  set(LD_V_OUTPUT)
  execute_process(
    COMMAND sh -c "${CMAKE_LINKER} -v 2>&1 | head -1"
    RESULT_VARIABLE HAD_ERROR
    OUTPUT_VARIABLE LD_V_OUTPUT
  )
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

# set(CMAKE_INCLUDE_CURRENT_DIR ON)

#include_directories(BEFORE
#  ${CMAKE_CURRENT_BINARY_DIR}/include
#  ${CMAKE_CURRENT_SOURCE_DIR}/include
#  )

if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
  install(DIRECTORY include/flang
    DESTINATION include
    COMPONENT flang-headers
    FILES_MATCHING
    PATTERN "*.def"
    PATTERN "*.h"
    PATTERN "config.h" EXCLUDE
    PATTERN ".svn" EXCLUDE
    )

  install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/flang
    DESTINATION include
    COMPONENT flang-headers
    FILES_MATCHING
    PATTERN "CMakeFiles" EXCLUDE
    PATTERN "*.inc"
    PATTERN "*.h"
    )

#  install(PROGRAMS utils/bash-autocomplete.sh
#    DESTINATION share/flang
#    )

endif()

add_definitions(-D_GNU_SOURCE)

option(FLANG_BUILD_TOOLS
  "Build the Flang tools. If OFF, just generate build targets." ON)

# Flang version information
set(FLANG_EXECUTABLE_VERSION
    "${FLANG_VERSION_MAJOR}" CACHE STRING
    "Major version number that will be appended to the flang executable name")

set(LIBFLANG_LIBRARY_VERSION
    "${FLANG_VERSION_MAJOR}" CACHE STRING
    "Major version number that will be appended to the libflang library")

mark_as_advanced(FLANG_EXECUTABLE_VERSION LIBFLANG_LIBRARY_VERSION)

option(FLANG_INCLUDE_TESTS
       "Generate build targets for the Flang unit tests."
       ${LLVM_INCLUDE_TESTS})

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(runtime)

#option(FLANG_BUILD_EXAMPLES "Build Flang example programs by default." OFF)
#add_subdirectory(examples)

# FIXME??  Do we need an order file like Clang (for Darwin)?

#if (FLANG_INCLUDE_TESTS)
#  if (EXISTS ${LLVM_MAIN_SRC_DIR}/utils/unittest/googletest/include/gtest/gtest.h)
#    add_subdirectory(unittests)
#    list(APPEND FLANG_TEST_DEPS FlangUnitTests)
#    list(APPEND FLANG_TEST_PARAMS
#      flang_unit_site_config=${CMAKE_CURRENT_BINARY_DIR}/test/Unit/lit.site.cfg
#      )
#  endif()

#  add_subdirectory(test)
#
#  if (FLANG_BUILT_STANDALONE)
#   # Add a global check rule now that all subdirectories have been traversed
#    # and we know the total set of lit testsuites.
#    get_property(LLVM_LIT_TESTSUITES GLOBAL PROPERTY LLVM_LIT_TESTSUITES)
#    get_property(LLVM_LIT_PARAMS GLOBAL PROPERTY LLVM_LIT_PARAMS)
#    get_property(LLVM_LIT_DEPENDS GLOBAL PROPERTY LLVM_LIT_DEPENDS)
#    get_property(LLVM_LIT_EXTRA_ARGS GLOBAL PROPERTY LLVM_LIT_EXTRA_ARGS)
#    get_property(LLVM_ADDITIONAL_TEST_TARGETS
#                 GLOBAL PROPERTY LLVM_ADDITIONAL_TEST_TARGETS)
#
#    add_lit_target(check-all
#      "Running all regression tests"
#      ${LLVM_LIT_TESTSUITES}
#      PARAMS ${LLVM_LIT_PARAMS}
#      DEPENDS ${LLVM_LIT_DEPENDS} ${LLVM_ADDITIONAL_TEST_TARGETS}
#      ARGS ${LLVM_LIT_EXTRA_ARGS}
#      )
#  endif()
#endif()

option(FLANG_INCLUDE_DOCS "Generate build targets for the Flang docs."
  ${LLVM_INCLUDE_DOCS})

if (FLANG_INCLUDE_DOCS)
  add_subdirectory(docs)
endif()

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

