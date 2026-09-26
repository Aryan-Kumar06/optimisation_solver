cmake_minimum_required(VERSION 3.20)

project(Optimisation_solver)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(bench
    src/main.cpp
    src/model/model.cpp
    src/benchmark/benchmarkresult.cpp
    src/presolve/presolver.cpp
)

target_include_directories(bench PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)


# ---------------------------------------------------------------------------
# assert() and Release builds
#
# CMake gives every Release build -DNDEBUG, which compiles assert() out. Many
# test files here use assert() as their ONLY check, so under Release they ran
# their setup, verified nothing, and reported success unconditionally -- a green
# ctest run in Release was not evidence those assertions held. Undefining NDEBUG
# for TEST targets only (never the libraries under test, which stay optimised)
# makes their assertions fire in every build type.
# ---------------------------------------------------------------------------
function(optimsolver_enable_assertions target)
    if(TARGET ${target})
        if(MSVC)
            target_compile_options(${target} PRIVATE /UNDEBUG)
        else()
            target_compile_options(${target} PRIVATE -UNDEBUG)
        endif()
    endif()
endfunction()

enable_testing()

add_executable(test_batch2
    tests/presolve/test_batch2.cpp
    src/model/model.cpp
    src/presolve/presolver.cpp
)
target_include_directories(test_batch2 PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)
add_test(NAME PresolveBatch2 COMMAND test_batch2)

add_executable(test_presolve_correctness
    tests/presolve/test_presolve_correctness.cpp
    src/model/model.cpp
    src/presolve/presolver.cpp
)
target_include_directories(test_presolve_correctness PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)
add_test(NAME PresolveCorrectness COMMAND test_presolve_correctness)

add_executable(test_transformation_metadata
    tests/presolve/test_transformation_metadata.cpp
    src/model/model.cpp
    src/presolve/presolver.cpp
)
target_include_directories(test_transformation_metadata PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)
add_test(NAME PresolveTransformationMetadata COMMAND test_transformation_metadata)

add_executable(test_pipeline_cascades
    tests/presolve/test_pipeline_cascades.cpp
    src/model/model.cpp
    src/presolve/presolver.cpp
)
target_include_directories(test_pipeline_cascades PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)
add_test(NAME PresolvePipelineCascades COMMAND test_pipeline_cascades)

add_executable(test_adversarial
    tests/presolve/test_adversarial.cpp
    src/model/model.cpp
    src/presolve/presolver.cpp
)
target_include_directories(test_adversarial PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)
add_test(NAME PresolveAdversarial COMMAND test_adversarial)

# Postsolve: reconstructs the original-model solution from the
# PresolveResult/PostsolveMetadata produced above.
add_library(postsolver
    src/postsolve/postsolver.cpp
)
target_include_directories(postsolver PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_compile_features(postsolver PUBLIC cxx_std_17)
if(NOT MSVC)
    target_compile_options(postsolver PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(test_postsolve
    tests/postsolve/test_postsolver.cpp
    src/model/model.cpp
    src/presolve/presolver.cpp
)
target_include_directories(test_postsolve PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(test_postsolve PRIVATE postsolver)
add_test(NAME PostsolveTests COMMAND test_postsolve)

# Exercise reconstruction directly, without wiring duals into the pipeline.
add_executable(test_postsolve_duals
    tests/postsolve/test_dual_reconstruction.cpp
    src/model/model.cpp
    src/presolve/presolver.cpp
)
target_include_directories(test_postsolve_duals PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(test_postsolve_duals PRIVATE postsolver)
foreach(case IN ITEMS
    known_shadow_price derived_bound singleton_lower singleton_upper
    singleton_equality singleton_ranged inactive_singleton reduced_costs
    equality_dual lower_dual upper_dual fixed_variable empty_duals
    wrong_dual_size invalid_constraint_mapping residual_gate nonfinite_duals nonfinite_reconstruction
    quadratic_gradient quadratic_reconstruction primal_only missing_bound_provenance
    invalid_bound_indices invalid_bound_direction invalid_source_coefficient
    invalid_singleton_provenance provenance_after_elimination provenance_bound_history
    derived_bound_directions)
    add_test(NAME PostsolveDual_${case} COMMAND test_postsolve_duals ${case})
endforeach()

add_executable(test_presolve_metadata
    tests/postsolve/test_presolve_metadata.cpp
    src/model/model.cpp
    src/presolve/presolver.cpp
)
target_include_directories(test_presolve_metadata PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/tests/postsolve
)
target_link_libraries(test_presolve_metadata PRIVATE postsolver)
add_test(NAME test_presolve_metadata COMMAND test_presolve_metadata)

# Linear programming engine. Self-contained and C++17; builds the library only.
# Its tests and benchmark harnesses stay off here and are enabled by configuring
# pdlp_engine/ standalone, or with -DPDLP_BUILD_TESTS=ON -DPDLP_BUILD_TOOLS=ON.
add_subdirectory(pdlp_engine)

# Branch-and-cut over a dual simplex, and ADMM for convex QP. Both build their
# own tests, so `ctest` from the root covers all three engines.
add_subdirectory(milp_engine)
add_subdirectory(qp_engine)

# Classifier and dispatcher. Classification runs before presolve (its reductions
# differ by problem class); engine selection runs after (presolve changes size,
# can eliminate all integrality, and can prove infeasibility outright).
add_library(solver_dispatch
    src/solver/classifier.cpp
    src/solver/dispatcher.cpp
    src/model/model.cpp
    src/presolve/presolver.cpp
)
target_include_directories(solver_dispatch PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_compile_features(solver_dispatch PUBLIC cxx_std_17)
if(NOT MSVC)
    target_compile_options(solver_dispatch PRIVATE -Wall -Wextra -Wpedantic)
endif()

enable_testing()
add_executable(test_classifier_dispatcher tests/solver/test_classifier_dispatcher.cpp)
target_link_libraries(test_classifier_dispatcher PRIVATE solver_dispatch)
add_test(NAME test_classifier_dispatcher COMMAND test_classifier_dispatcher)

# The only place model::Model and pdlp::CompiledLp meet. Links pdlp_engine for
# CompiledLp and PdlpResult; everything else in the pipeline speaks model::Model.
add_library(pdlp_adapter
    src/adapter/pdlp_adapter.cpp
    src/model/model.cpp
)
target_include_directories(pdlp_adapter PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(pdlp_adapter PUBLIC pdlp_engine)
target_compile_features(pdlp_adapter PUBLIC cxx_std_17)
if(NOT MSVC)
    target_compile_options(pdlp_adapter PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(test_pdlp_adapter tests/adapter/test_pdlp_adapter.cpp)
target_link_libraries(test_pdlp_adapter PRIVATE pdlp_adapter)
add_test(NAME test_pdlp_adapter COMMAND test_pdlp_adapter)

# The orchestrator is the one entry point a caller needs: it classifies, runs
# presolve, dispatches to whichever engine the decision names, and normalises
# every engine's result into a single SolveResult. It is therefore the only
# target that links all three engines at once.
add_library(solver_orchestrator
    src/solver/orchestrator.cpp
)
target_include_directories(solver_orchestrator PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(solver_orchestrator PUBLIC
    solver_dispatch
    pdlp_adapter
    milp_engine
    qp_engine
    postsolver
)
target_compile_features(solver_orchestrator PUBLIC cxx_std_17)
if(NOT MSVC)
    target_compile_options(solver_orchestrator PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Parser for mathematical programming system (.mps) input format.
add_library(mps_reader
    src/mps/mps_reader.cpp
)
target_include_directories(mps_reader PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_compile_features(mps_reader PUBLIC cxx_std_17)
if(NOT MSVC)
    target_compile_options(mps_reader PRIVATE -Wall -Wextra -Wpedantic)
endif()

# MPS parser suite. Fixtures are referenced by paths relative to the project
# source directory, so the test runs from there rather than from the build tree.
# model.cpp comes in directly: the mps_reader library deliberately does not
# carry it (it only produces a Model, never validates one), but the test asserts
# that a parsed model passes Model::validate().
add_executable(test_mps_reader
    tests/mps/mps_reader_tests.cpp
    src/model/model.cpp
)
target_link_libraries(test_mps_reader PRIVATE mps_reader)
add_test(NAME test_mps_reader
    COMMAND test_mps_reader
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)

# End-to-end: model in, solution out, through every layer in the real order.
add_executable(test_end_to_end tests/pipeline/test_end_to_end.cpp)
target_link_libraries(test_end_to_end PRIVATE solver_orchestrator)
add_test(NAME test_end_to_end COMMAND test_end_to_end)

add_executable(test_postsolve_integration tests/pipeline/test_postsolve_integration.cpp)
target_link_libraries(test_postsolve_integration PRIVATE solver_orchestrator)
foreach(case IN ITEMS lp_engines qp_signs all_eliminated bound_only integer_models
    unavailable_duals failure_statuses limit_no_solution limit_feasible_solution reduced_api)
    add_test(NAME PostsolveIntegration_${case} COMMAND test_postsolve_integration ${case})
endforeach()

# Pins the Model quadratic convention through the public solve API: a
# diagonal q_ii must reach the engine as P_ii = 2*q_ii and an off-diagonal
# q_ij as P_ij = P_ji = q_ij, with maximization negating both P and q and the
# objective offset restored exactly once. Every expected value is derived by
# hand in the test, independently of the code under test.
add_executable(test_qp_convention tests/qp/test_qp_convention.cpp)
target_link_libraries(test_qp_convention PRIVATE solver_orchestrator)
add_test(NAME test_qp_convention COMMAND test_qp_convention)

# Ground truth by exhaustive enumeration: every small binary MILP is solved
# by brute force over all 2^n subsets and compared against the pipeline. No
# reference solver required, so this stays runnable anywhere.
add_executable(test_milp_enumeration tests/milp/test_milp_enumeration.cpp)
target_link_libraries(test_milp_enumeration PRIVATE solver_orchestrator)
add_test(NAME test_milp_enumeration COMMAND test_milp_enumeration)

# Command-line interface application layer
add_library(optimsolver_cli
    cli/argument_parser.cpp
    cli/cli.cpp
    cli/mascot.cpp
    cli/json_report.cpp
)
target_include_directories(optimsolver_cli PUBLIC
    ${PROJECT_SOURCE_DIR}/cli
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(optimsolver_cli PUBLIC
    solver_orchestrator
    mps_reader
    postsolver
)
execute_process(
    COMMAND git rev-parse HEAD
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_VARIABLE OPTIMSOLVER_GIT_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(NOT OPTIMSOLVER_GIT_COMMIT)
    set(OPTIMSOLVER_GIT_COMMIT "unknown")
endif()
target_compile_definitions(optimsolver_cli PRIVATE
    OPTIMSOLVER_GIT_COMMIT="${OPTIMSOLVER_GIT_COMMIT}"
    OPTIMSOLVER_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
)

target_compile_features(optimsolver_cli PUBLIC cxx_std_17)
if(NOT MSVC)
    target_compile_options(optimsolver_cli PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(optimsolver
    cli/main.cpp
)
target_link_libraries(optimsolver PRIVATE optimsolver_cli)
install(TARGETS optimsolver RUNTIME DESTINATION bin)

add_executable(test_cli
    tests/cli/test_cli.cpp
)
target_link_libraries(test_cli PRIVATE optimsolver_cli)
target_compile_definitions(test_cli PRIVATE
    OPTIMSOLVER_BIN_PATH="$<TARGET_FILE:optimsolver>"
    TEST_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)
add_test(NAME test_cli COMMAND test_cli)

# ---------------------------------------------------------------------------
# Benchmark harness. POSIX only (see benchmarks/harness/process_manager.h), so
# it is guarded rather than failing to build on Windows. It is the promoted form
# of benchmark_model/benchmarkcode.cpp, which no target ever referenced.
# ---------------------------------------------------------------------------
if(NOT WIN32)
    add_library(bench_harness benchmarks/harness/process_manager.cpp)
    target_include_directories(bench_harness PUBLIC
        ${PROJECT_SOURCE_DIR}/benchmarks/harness
        ${PROJECT_SOURCE_DIR}/include
    )
    target_compile_features(bench_harness PUBLIC cxx_std_17)
    if(NOT MSVC)
        target_compile_options(bench_harness PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_executable(bench_runner benchmarks/harness/bench_runner.cpp)
    target_link_libraries(bench_runner PRIVATE bench_harness)

    add_executable(test_process_manager benchmarks/harness/test_process_manager.cpp)
    target_link_libraries(test_process_manager PRIVATE bench_harness)
    add_test(NAME test_process_manager COMMAND test_process_manager)

    # Proves the checker REJECTS wrong answers. A checker that never fails would
    # make every benchmark run look green.
    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        add_test(NAME test_benchmark_pipeline
            COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/benchmarks/test_pipeline.py
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        )
        set_tests_properties(test_benchmark_pipeline PROPERTIES
            TIMEOUT 300
            ENVIRONMENT "OPTIMSOLVER_BINARY=$<TARGET_FILE:optimsolver>")
    endif()
endif()

# Applied last so every test target, including those in engine subdirectories,
# already exists.
foreach(optimsolver_test_target IN ITEMS
    bench test_batch2 test_presolve_correctness test_transformation_metadata
    test_pipeline_cascades test_adversarial test_postsolve test_postsolve_duals
    test_presolve_metadata test_classifier_dispatcher test_pdlp_adapter
    test_mps_reader test_end_to_end test_qp_convention test_milp_enumeration
    test_postsolve_integration test_cli test_process_manager
    test_dual_simplex test_branch_and_bound test_gomory_cuts test_heuristics
    test_branching pdlp_tests qp_tests)
    optimsolver_enable_assertions(${optimsolver_test_target})
endforeach()
