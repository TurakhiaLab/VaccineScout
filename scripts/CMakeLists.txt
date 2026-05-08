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

# ---------------------------------------------------------------------------
# get_mutations: dump cumulative nt mutations per node from a UShER MAT (C++),
# Parquet columns node_name, nt_mutations — consumed by
# get_amino_acid_mutations_of_sequences.py (invokes this binary automatically).
#
# Requires UShER sources (parsimony.proto at repo root). Default: scripts/usher,
# populated by scripts/fetch_usher.sh. Override with -DUSHER_ROOT=/other/path.
#
#   bash scripts/fetch_usher.sh
#   cmake -S scripts -B scripts/build -DBUILD_GET_MUTATIONS=ON
#   cmake --build scripts/build --target get_mutations
# ---------------------------------------------------------------------------
option(BUILD_GET_MUTATIONS "Build get_mutations MAT exporter (needs USHER_ROOT)" OFF)
set(_usher_default "${CMAKE_CURRENT_SOURCE_DIR}/usher")
# Quote ${USHER_ROOT}: bare `if(USHER_ROOT STREQUAL "")` is FALSE when unset (CMake quirk).
if("${USHER_ROOT}" STREQUAL "")
  set(USHER_ROOT "${_usher_default}" CACHE PATH
      "Path to yatisht/usher repository root (run scripts/fetch_usher.sh for scripts/usher)" FORCE)
endif()

if(BUILD_GET_MUTATIONS)
  if(NOT EXISTS "${USHER_ROOT}/parsimony.proto")
    message(FATAL_ERROR
      "No parsimony.proto under USHER_ROOT (${USHER_ROOT}). "
      "Run: bash ${CMAKE_CURRENT_SOURCE_DIR}/fetch_usher.sh")
  endif()
  if(NOT EXISTS "${USHER_ROOT}/src/mutation_annotated_tree.cpp")
    message(FATAL_ERROR "USHER_ROOT/src/mutation_annotated_tree.cpp not found under ${USHER_ROOT}")
  endif()

  # UShER uses Intel TBB 2020 API (tbb/rw_mutex.h). Do not use ENV{CPATH}: conda
  # often has a different TBB and breaks UShER (e.g. flow_graph lambda signatures).
  find_path(USHER_TBB_INCLUDE_DIR NAMES tbb/rw_mutex.h
    PATHS /usr/include /usr/local/include "${CMAKE_SYSROOT}/usr/include")
  find_library(USHER_TBB_LIBRARY NAMES tbb
    PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu /usr/lib64)
  if(NOT USHER_TBB_INCLUDE_DIR OR NOT USHER_TBB_LIBRARY)
    message(FATAL_ERROR
      "Need system TBB 2020 headers (tbb/rw_mutex.h) and libtbb for UShER. "
      "On Debian/Ubuntu: sudo apt install libtbb-dev. Conda oneTBB alone is not sufficient.")
  endif()

  find_package(Protobuf REQUIRED)
  find_package(Boost REQUIRED COMPONENTS iostreams filesystem)
  find_package(ZLIB REQUIRED)
  if(Protobuf_VERSION VERSION_GREATER_EQUAL "4.0.0")
    find_package(absl REQUIRED)
  endif()

  set(PROTOBUF_IMPORT_DIRS "${USHER_ROOT}")
  protobuf_generate_cpp(PARSIMONY_PB_SRCS PARSIMONY_PB_HDRS "${USHER_ROOT}/parsimony.proto")

  # usher_common.cpp includes "version.hpp"; upstream CMake generates it from version.hpp.in.
  set(usher_VERSION "FutureCoV-bundled")
  configure_file(
    "${USHER_ROOT}/src/version.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/version.hpp"
    @ONLY
  )

  add_executable(get_mutations
    get_mutations.cpp
    "${USHER_ROOT}/src/mutation_annotated_tree.cpp"
    "${USHER_ROOT}/src/usher_mapper.cpp"
    "${USHER_ROOT}/src/usher_common.cpp"
    ${PARSIMONY_PB_SRCS}
  )

  target_include_directories(get_mutations PRIVATE
    "${USHER_ROOT}/src"
    ${CMAKE_CURRENT_BINARY_DIR}
    "${USHER_TBB_INCLUDE_DIR}"
  )
  target_compile_definitions(get_mutations PRIVATE SAVE_PROFILE=0)
  set_target_properties(get_mutations PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  target_link_libraries(get_mutations PRIVATE
    Arrow::arrow_shared
    Parquet::parquet_shared
    ${Boost_LIBRARIES}
    ${Protobuf_LIBRARIES}
    ZLIB::ZLIB
    ${USHER_TBB_LIBRARY}
  )
  if(Protobuf_VERSION VERSION_GREATER_EQUAL "4.0.0")
    target_link_libraries(get_mutations PRIVATE absl::log absl::log_internal_check_impl)
  endif()
  if(MSVC)
    target_compile_options(get_mutations PRIVATE /W4)
  else()
    target_compile_options(get_mutations PRIVATE -Wall -Wextra -O3)
  endif()
endif()
