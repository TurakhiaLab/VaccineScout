cmake_minimum_required(VERSION 3.16)
project(min_dist LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Prefer the active conda env (Arrow / Parquet). Many named envs lack tbb-devel; base
# installs often provide lib/cmake/tbb/TBBConfig.cmake — append base after env so
# Arrow/Parquet still resolve from the env first.
if(DEFINED ENV{CONDA_PREFIX} AND NOT "$ENV{CONDA_PREFIX}" STREQUAL "")
  list(PREPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
  if("$ENV{CONDA_PREFIX}" MATCHES "/envs/[^/]+$")
    get_filename_component(_conda_inst_root "$ENV{CONDA_PREFIX}" DIRECTORY)
    get_filename_component(_conda_inst_root "${_conda_inst_root}" DIRECTORY)
    if(EXISTS "${_conda_inst_root}/lib/cmake/tbb/TBBConfig.cmake")
      list(APPEND CMAKE_PREFIX_PATH "${_conda_inst_root}")
    endif()
  endif()
endif()

find_package(TBB REQUIRED CONFIG)
find_package(Arrow REQUIRED CONFIG)
find_package(Parquet REQUIRED CONFIG)

add_executable(min_dist min_dist.cpp)

set_target_properties(min_dist PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(min_dist PRIVATE
  TBB::tbb
  Arrow::arrow_shared
  Parquet::parquet_shared
)

if(MSVC)
  target_compile_options(min_dist PRIVATE /W4)
else()
  target_compile_options(min_dist PRIVATE -Wall -Wextra -O3)
endif()
