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

# Optional hybrid CUDA backend for the ADMM iteration (KKT stays on the CPU).
# OFF by default; the parent project's OPTIMSOLVER_ENABLE_CUDA, when set,
# supplies the default.
if(DEFINED OPTIMSOLVER_ENABLE_CUDA)
    set(QP_CUDA_DEFAULT ${OPTIMSOLVER_ENABLE_CUDA})
else()
    set(QP_CUDA_DEFAULT OFF)
endif()
option(QP_ENABLE_CUDA "Build the hybrid CUDA backend for the ADMM iteration" ${QP_CUDA_DEFAULT})

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
    src/admm_backend.cpp
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

if(QP_ENABLE_CUDA)
    include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/OptimsolverCuda.cmake)
    optimsolver_enable_cuda()

    # Device code in its own target: nvcc never sees the engine's host flags,
    # and nothing CUDA-specific reaches consumers beyond the runtime libraries.
    add_library(qp_cuda STATIC
        src/cuda/admm_cuda_backend.cu
    )
    target_include_directories(qp_cuda PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../cuda_support/include
    )
    target_compile_definitions(qp_cuda PRIVATE CUDA_SUPPORT_WITH_CUSPARSE)
    optimsolver_configure_cuda_target(qp_cuda)
    target_link_libraries(qp_cuda PUBLIC CUDA::cudart_static CUDA::cusparse)
    target_link_libraries(qp_engine PRIVATE qp_cuda)
else()
    target_sources(qp_engine PRIVATE src/cuda/cuda_unavailable.cpp)
endif()

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

# CUDA validation must not disappear when the ordinary engine tests are off.
option(OPTIMSOLVER_REQUIRE_CUDA_TEST_DEVICE "Fail CUDA tests instead of skipping when no GPU is usable" OFF)
if(QP_STANDALONE AND OPTIMSOLVER_REQUIRE_CUDA_TEST_DEVICE AND NOT QP_ENABLE_CUDA)
    message(FATAL_ERROR "Strict CUDA validation requires a CUDA backend to be enabled")
endif()
if(QP_ENABLE_CUDA)
    if(QP_STANDALONE)
        enable_testing()
    endif()
    add_executable(qp_cuda_tests tests/test_qp_cuda.cpp)
    target_link_libraries(qp_cuda_tests PRIVATE qp_engine CUDA::cudart_static)
    target_include_directories(qp_cuda_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../cuda_support/include)
    add_test(NAME qp_cuda_tests COMMAND qp_cuda_tests)
    set_tests_properties(qp_cuda_tests PROPERTIES SKIP_RETURN_CODE 77 LABELS "cuda" TIMEOUT 300)
    if(OPTIMSOLVER_REQUIRE_CUDA_TEST_DEVICE)
        set_tests_properties(qp_cuda_tests PROPERTIES ENVIRONMENT "OPTIMSOLVER_REQUIRE_CUDA_TEST_DEVICE=1")
    endif()
endif()
