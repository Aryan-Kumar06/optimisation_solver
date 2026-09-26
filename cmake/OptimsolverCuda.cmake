# CUDA build support shared by pdlp_engine and qp_engine.
#
# Included ONLY when a CUDA backend has been requested (OPTIMSOLVER_ENABLE_CUDA,
# PDLP_ENABLE_CUDA or QP_ENABLE_CUDA). A default configure never reads this file,
# never looks for a CUDA compiler and never calls find_package(CUDAToolkit).

include_guard(GLOBAL)

# Enables the CUDA language and locates the toolkit. A macro, not a function:
# enable_language() must execute at directory scope.
#
# In the parent build this is invoked from the top-level CMakeLists before the
# engines are added. It has to be: CMake requires a language to be enabled in
# the highest directory common to every target that compiles OR LINKS it, and
# the engines' device code reaches the orchestrator, the CLI and the tests
# through their link dependencies.
macro(optimsolver_enable_cuda)
    if(NOT CMAKE_CUDA_COMPILER_LOADED)
        include(CheckLanguage)
        check_language(CUDA)
        if(NOT CMAKE_CUDA_COMPILER)
            message(FATAL_ERROR
                "A CUDA backend was requested but no CUDA compiler was found.\n"
                "Install the CUDA Toolkit (and, on Windows, the MSVC toolchain nvcc "
                "requires), or point CMAKE_CUDA_COMPILER at nvcc, or configure "
                "without -DOPTIMSOLVER_ENABLE_CUDA / -DPDLP_ENABLE_CUDA / -DQP_ENABLE_CUDA "
                "for a CPU-only build.")
        endif()

        # Never hard-coded. An explicit -DCMAKE_CUDA_ARCHITECTURES always wins;
        # otherwise build for the GPU(s) in this machine, which is what a local
        # build wants. Machines without a GPU (CI, packaging) must pass an
        # explicit list, e.g. -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90".
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
            if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
                set(CMAKE_CUDA_ARCHITECTURES native)
                message(STATUS "CMAKE_CUDA_ARCHITECTURES not set; using 'native'")
            endif()
        endif()

        enable_language(CUDA)
    endif()

    # IEEE semantics are load-bearing: bounds are represented by infinities,
    # and non-finite detection relies on NaN propagating. Fast math breaks both.
    if(CMAKE_CUDA_FLAGS MATCHES "use_fast_math|ftz=true|prec-div=false|prec-sqrt=false")
        message(FATAL_ERROR
            "CMAKE_CUDA_FLAGS contains a fast-math flag (${CMAKE_CUDA_FLAGS}). "
            "The solver relies on IEEE infinities and NaN propagation; remove it.")
    endif()

    find_package(CUDAToolkit REQUIRED)
endmacro()

# Compile settings for a target holding CUDA sources. Applied to the CUDA
# targets only, so nothing here reaches ordinary C++ targets.
function(optimsolver_configure_cuda_target target)
    set_target_properties(${target} PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_EXTENSIONS OFF
        POSITION_INDEPENDENT_CODE ON
    )
    target_compile_features(${target} PUBLIC cxx_std_17)
    target_compile_options(${target} PRIVATE
        # Explicitly the IEEE-conforming defaults, so the intent is visible and
        # a stray global flag cannot silently change them. FMA contraction
        # (-fmad=true) stays on: it is exact per operation and matches the CPU
        # build's -ffp-contract=fast.
        $<$<COMPILE_LANGUAGE:CUDA>:--prec-div=true>
        $<$<COMPILE_LANGUAGE:CUDA>:--prec-sqrt=true>
        $<$<COMPILE_LANGUAGE:CUDA>:--ftz=false>
        # Source-line attribution for compute-sanitizer and Nsight Compute; it
        # does not change generated code.
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    )
endfunction()
