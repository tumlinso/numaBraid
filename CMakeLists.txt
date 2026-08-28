cmake_minimum_required(VERSION 3.24)

project(
  numaBraid
  VERSION 0.0.0
  DESCRIPTION "Narrow GPU forwarding for a dual-NUMA NVLink X topology"
  LANGUAGES CXX
)

include(CTest)
include(CheckLanguage)

option(NUMABRAID_BUILD_TESTS "Build numaBraid non-hardware tests" "${BUILD_TESTING}")
option(NUMABRAID_BUILD_BENCHMARKS "Build hardware-dependent benchmarks" OFF)
option(NUMABRAID_BUILD_MODULES "Build the optional C++20 public module" OFF)
option(NUMABRAID_ENABLE_CUDA "Enable CUDA language support for CUDA implementation targets" OFF)
option(NUMABRAID_ENABLE_NCCL "Discover NCCL for implementation targets" OFF)
option(NUMABRAID_DEVELOPER_WARNINGS "Enable warnings on numaBraid-owned compiled targets" ON)

add_library(numabraid INTERFACE)
add_library(numabraid::numabraid ALIAS numabraid)
target_compile_features(numabraid INTERFACE cxx_std_20)
target_include_directories(
  numabraid
  INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
)

function(numabraid_enable_warnings target)
  if(NOT NUMABRAID_DEVELOPER_WARNINGS)
    return()
  endif()

  if(MSVC)
    target_compile_options(${target} PRIVATE /W4)
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
  endif()
endfunction()

if(NUMABRAID_ENABLE_CUDA)
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "NUMABRAID_ENABLE_CUDA=ON, but no CUDA compiler was found")
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)
endif()

if(NUMABRAID_ENABLE_NCCL)
  find_path(NCCL_INCLUDE_DIR NAMES nccl.h HINTS ENV NCCL_ROOT PATH_SUFFIXES include)
  find_library(NCCL_LIBRARY NAMES nccl HINTS ENV NCCL_ROOT PATH_SUFFIXES lib lib64)
  if(NOT NCCL_INCLUDE_DIR OR NOT NCCL_LIBRARY)
    message(FATAL_ERROR "NUMABRAID_ENABLE_NCCL=ON, but NCCL headers and library were not found; set NCCL_ROOT")
  endif()

  add_library(numabraid_nccl INTERFACE)
  add_library(numabraid::nccl ALIAS numabraid_nccl)
  target_include_directories(numabraid_nccl INTERFACE "${NCCL_INCLUDE_DIR}")
  target_link_libraries(numabraid_nccl INTERFACE "${NCCL_LIBRARY}")
endif()

if(NUMABRAID_BUILD_MODULES)
  if(CMAKE_VERSION VERSION_LESS 3.28)
    message(FATAL_ERROR "NUMABRAID_BUILD_MODULES requires CMake 3.28 or newer")
  endif()

  add_library(numabraid_module)
  add_library(numabraid::module ALIAS numabraid_module)
  target_compile_features(numabraid_module PUBLIC cxx_std_20)
  target_sources(
    numabraid_module
    PUBLIC
      FILE_SET CXX_MODULES
      FILES modules/numabraid.cppm
  )
  numabraid_enable_warnings(numabraid_module)
endif()

if(NUMABRAID_BUILD_TESTS)
  if(NOT BUILD_TESTING)
    message(FATAL_ERROR "NUMABRAID_BUILD_TESTS=ON requires BUILD_TESTING=ON")
  endif()
  add_subdirectory(tests)
endif()

if(NUMABRAID_BUILD_BENCHMARKS)
  message(STATUS "Benchmark infrastructure is reserved for NB-02; no benchmark targets exist yet")
endif()
