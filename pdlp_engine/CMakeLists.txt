cmake_minimum_required(VERSION 3.20)
project(pdlp_engine LANGUAGES CXX)

# This module is built two ways: standalone during development, and as a
# subdirectory of the parent solver project. Anything that would reach outside
# this directory -- forcing a build type, enabling tests, adding extra
# executables, tuning for the local CPU -- is gated on which of the two is
# happening, so add_subdirectory() here has no side effects on the parent.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(PDLP_STANDALONE TRUE)
else()
    set(PDLP_STANDALONE FALSE)
endif()

option(PDLP_BUILD_TESTS "Build the PDLP test suite" ${PDLP_STANDALONE})
option(PDLP_BUILD_TOOLS "Build the benchmark, validation and ablation harnesses" ${PDLP_STANDALONE})
option(PDLP_ENABLE_LTO  "Enable link-time optimisation for Release builds" ON)

# -march=native produces binaries that fault on a different CPU, so it is only
# a default when this module is the whole build. In the parent build the team
# gets portable output unless it explicitly asks otherwise.
option(PDLP_NATIVE_ARCH "Tune for the building machine's CPU" ${PDLP_STANDALONE})

# Optional CUDA backend for the PDHG iteration. OFF by default: without it no
# CUDA toolkit, driver or GPU is needed to configure, build, test or run. The
# parent project's OPTIMSOLVER_ENABLE_CUDA, when set, supplies the default.
if(DEFINED OPTIMSOLVER_ENABLE_CUDA)
    set(PDLP_CUDA_DEFAULT ${OPTIMSOLVER_ENABLE_CUDA})
else()
    set(PDLP_CUDA_DEFAULT OFF)
endif()
option(PDLP_ENABLE_CUDA "Build the CUDA backend for the PDHG iteration" ${PDLP_CUDA_DEFAULT})

# Only meaningful standalone: setting this in a subdirectory would overwrite the
# parent project's cached build type.
if(PDLP_STANDALONE AND NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    message(STATUS "CMAKE_BUILD_TYPE not set; defaulting to Release")
endif()

add_library(pdlp_engine
    src/sparse_matrix.cpp
    src/parallel.cpp
    src/scaling.cpp
    src/infeasibility.cpp
    src/preconditioner.cpp
    src/step_controller.cpp
    src/iterate_average.cpp
    src/restart_controller.cpp
    src/termination.cpp
    src/pdhg_kernel.cpp
    src/iteration_backend.cpp
    src/feasibility_polishing.cpp
    src/pdlp_solver.cpp
)

target_include_directories(pdlp_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Consumers get C++17 without the parent having to set anything.
target_compile_features(pdlp_engine PUBLIC cxx_std_17)

find_package(Threads REQUIRED)
target_link_libraries(pdlp_engine PUBLIC Threads::Threads)

include(CheckCXXCompilerFlag)

if(PDLP_ENABLE_CUDA)
    include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/OptimsolverCuda.cmake)
    optimsolver_enable_cuda()

    # The device code is its own target so that nvcc never sees the engine's
    # host-compiler flags (-march=native, -Wpedantic, LTO), and those flags never
    # see nvcc. Nothing CUDA-specific reaches consumers of pdlp_engine except
    # the runtime library it links.
    add_library(pdlp_cuda STATIC
        src/cuda/cuda_iteration_backend.cu
    )
    target_include_directories(pdlp_cuda PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../cuda_support/include
    )
    optimsolver_configure_cuda_target(pdlp_cuda)
    target_link_libraries(pdlp_cuda PUBLIC CUDA::cudart_static)
    target_link_libraries(pdlp_engine PRIVATE pdlp_cuda)
else()
    target_sources(pdlp_engine PRIVATE src/cuda/cuda_unavailable.cpp)
endif()

if(MSVC)
    target_compile_options(pdlp_engine PRIVATE /W4 /permissive- /fp:precise)
    target_compile_options(pdlp_engine PRIVATE $<$<CONFIG:Release>:/O2>)
else()
    target_compile_options(pdlp_engine PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(pdlp_engine PRIVATE $<$<CONFIG:Release>:-O3>)

    # Contraction into FMA is safe here and shortens the sparse dot products.
    # Full -ffast-math is deliberately NOT used: it permits reassociating the
    # residual reductions and would let the compiler assume no infinities, which
    # this code relies on for one-sided row bounds.
    check_cxx_compiler_flag("-ffp-contract=fast" PDLP_HAS_FP_CONTRACT)
    if(PDLP_HAS_FP_CONTRACT)
        target_compile_options(pdlp_engine PRIVATE -ffp-contract=fast)
    endif()

    # PRIVATE, not PUBLIC: this tunes how the engine itself is compiled and must
    # not be forced onto anything that links it.
    if(PDLP_NATIVE_ARCH)
        check_cxx_compiler_flag("-march=native" PDLP_HAS_MARCH_NATIVE)
        if(PDLP_HAS_MARCH_NATIVE)
            target_compile_options(pdlp_engine PRIVATE -march=native)
        else()
            check_cxx_compiler_flag("-mcpu=native" PDLP_HAS_MCPU_NATIVE)
            if(PDLP_HAS_MCPU_NATIVE)
                target_compile_options(pdlp_engine PRIVATE -mcpu=native)
            endif()
        endif()
    endif()
endif()

if(PDLP_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT PDLP_IPO_SUPPORTED OUTPUT PDLP_IPO_MESSAGE)
    if(PDLP_IPO_SUPPORTED)
        set_property(TARGET pdlp_engine
            PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    else()
        message(STATUS "LTO unavailable: ${PDLP_IPO_MESSAGE}")
    endif()
endif()

# Development harnesses. Off in the parent build: they are how the engine's
# performance and accuracy claims are reproduced, not part of the library.
#   pdlp_bench    time per iteration on generated sparse LPs
#   pdlp_validate emit instances plus answers for cross-checking against HiGHS
if(PDLP_BUILD_TOOLS)
    add_executable(pdlp_example examples/solve_example.cpp)
    target_link_libraries(pdlp_example PRIVATE pdlp_engine)

    add_executable(pdlp_bench bench/bench_pdlp.cpp)
    target_link_libraries(pdlp_bench PRIVATE pdlp_engine)

    add_executable(pdlp_validate tools/validate_against_reference.cpp)
    target_link_libraries(pdlp_validate PRIVATE pdlp_engine)
endif()

if(PDLP_BUILD_TESTS)
    if(PDLP_STANDALONE)
        enable_testing()
    endif()
    add_executable(pdlp_tests tests/test_pdlp.cpp)
    target_link_libraries(pdlp_tests PRIVATE pdlp_engine)
    add_test(NAME pdlp_tests COMMAND pdlp_tests)

    # CPU-vs-CUDA equivalence. Built only with CUDA; exits 77 (reported by
    # CTest as skipped, not failed) when the machine has no usable GPU.
    if(PDLP_ENABLE_CUDA)
        add_executable(pdlp_cuda_tests tests/test_pdlp_cuda.cpp)
        target_link_libraries(pdlp_cuda_tests PRIVATE pdlp_engine)
        add_test(NAME pdlp_cuda_tests COMMAND pdlp_cuda_tests)
        set_tests_properties(pdlp_cuda_tests PROPERTIES SKIP_RETURN_CODE 77)
    endif()
endif()
