cmake_minimum_required(VERSION 3.20)
project(qp_engine LANGUAGES CXX)

# This module is built two ways: standalone during development, and as a
# subdirectory of the parent solver project. Anything that would reach outside
# this directory -- forcing a build type, enabling tests, adding extra
# executables, tuning for the local CPU -- is gated on which of the two is
# happening, so add_subdirectory() here has no side effects on the parent.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(QP_STANDALONE TRUE)
else()
    set(QP_STANDALONE FALSE)
endif()

option(QP_BUILD_TESTS "Build the QP test suite" ${QP_STANDALONE})
option(QP_BUILD_BENCH "Build the QP benchmark harness" ${QP_STANDALONE})
option(QP_ENABLE_LTO  "Enable link-time optimisation for Release builds" ON)

# -march=native produces binaries that fault on a different CPU, so it is only
# a default when this module is the whole build. In the parent build the team
# gets portable output unless it explicitly asks otherwise.
option(QP_NATIVE_ARCH "Tune for the building machine's CPU" ${QP_STANDALONE})

# Only meaningful standalone: setting this in a subdirectory would overwrite the
# parent project's cached build type.
if(QP_STANDALONE AND NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    message(STATUS "CMAKE_BUILD_TYPE not set; defaulting to Release")
endif()

add_library(qp_engine
    src/qp_types.cpp
    src/qp_model.cpp
    src/scaling.cpp
    src/kkt_solver.cpp
    src/admm_solver.cpp
    src/polishing.cpp
    src/qp_adapter.cpp
    src/convexity.cpp
    src/qp_solver.cpp
    src/parallel.cpp
)

target_include_directories(qp_engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

# Consumers get C++17 without the parent having to set anything.
target_compile_features(qp_engine PUBLIC cxx_std_17)

find_package(Threads REQUIRED)
target_link_libraries(qp_engine PUBLIC Threads::Threads)

include(CheckCXXCompilerFlag)

if(MSVC)
    target_compile_options(qp_engine PRIVATE /W4 /permissive- /fp:precise)
    target_compile_options(qp_engine PRIVATE $<$<CONFIG:Release>:/O2>)
else()
    target_compile_options(qp_engine PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(qp_engine PRIVATE $<$<CONFIG:Release>:-O3>)
endif()

if(QP_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT QP_IPO_SUPPORTED OUTPUT QP_IPO_MESSAGE)
    if(QP_IPO_SUPPORTED)
        set_property(TARGET qp_engine
            PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    else()
        message(STATUS "LTO unavailable: ${QP_IPO_MESSAGE}")
    endif()
endif()

if(QP_BUILD_BENCH)
    add_executable(qp_bench bench/bench_qp.cpp)
    target_link_libraries(qp_bench PRIVATE qp_engine)
endif()

if(QP_BUILD_TESTS)
    if(QP_STANDALONE)
        enable_testing()
    endif()
    add_executable(qp_tests tests/test_qp_solver.cpp)
    target_link_libraries(qp_tests PRIVATE qp_engine)
    add_test(NAME qp_tests COMMAND qp_tests)
endif()
