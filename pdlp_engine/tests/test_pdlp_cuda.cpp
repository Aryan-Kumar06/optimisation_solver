// CPU-vs-CUDA equivalence for the PDHG backend. Built only when
// PDLP_ENABLE_CUDA is on; exits 77 (CTest: skipped) without a usable GPU, so a
// CUDA build on a GPU-less CI machine stays green without pretending to test.

#include "backend_contract.h"

#include "pdlp/compute_backend.h"
#include "pdlp/iteration_backend.h"
#include "pdlp/pdlp_solver.h"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

int failures = 0;

void run(const std::string& name, const std::function<void()>& test) {
    try {
        test();
        std::cout << "  pass  " << name << '\n';
    } catch (const std::exception& error) {
        std::cout << "  FAIL  " << name << ": " << error.what() << '\n';
        ++failures;
    }
}

contract::BackendFactory cpuFactory() {
    return [](const pdlp::CompiledLp& problem, const pdlp::DiagonalPreconditioner& preconditioner) {
        return pdlp::makeCpuIterationBackend(problem, preconditioner);
    };
}

contract::BackendFactory cudaFactory(int groupSize, std::int64_t heavyNonzeros) {
    return [=](const pdlp::CompiledLp& problem, const pdlp::DiagonalPreconditioner& preconditioner) {
        pdlp::CudaBackendConfig config;
        config.forcedGroupSize = groupSize;
        config.heavyLineNonzeros = heavyNonzeros;
        std::string error;
        auto backend = pdlp::makeCudaIterationBackend(problem, preconditioner, config, error);
        if (!backend) {
            throw std::runtime_error("CUDA backend creation failed: " + error);
        }
        return backend;
    };
}

// Summation order is the only permitted difference; 1e-10 relative is loose
// enough for the reordered dot products of these sizes and far tighter than
// any arithmetic mistake.
constexpr double kKernelTolerance = 1e-10;

void kernelEquivalence() {
    // Every group-size specialisation, with the heavy-line kernel both idle
    // (default threshold) and forced to take most lines (threshold 2).
    const int groups[] = {0, 1, 2, 4, 8, 16, 32};
    const std::int64_t heavy[] = {4096, 2};
    for (const contract::Fixture& fixture : contract::kernelFixtures()) {
        for (int group : groups) {
            for (std::int64_t threshold : heavy) {
                const std::string name = fixture.name + " G=" + std::to_string(group) +
                                         " heavy>" + std::to_string(threshold);
                run("kernel " + name, [&] {
                    contract::Fixture labelled{name, fixture.problem};
                    contract::compareBackends(labelled, cpuFactory(), cudaFactory(group, threshold),
                                              kKernelTolerance);
                });
            }
        }
    }
}

pdlp::PdlpResult solveWith(const pdlp::CompiledLp& problem, pdlp::ComputeBackend backend,
                           void (*configure)(pdlp::PdlpOptions&)) {
    pdlp::PdlpOptions options;
    options.iterationLimit = 50000;
    options.primalTolerance = 1e-7;
    options.dualTolerance = 1e-7;
    options.gapTolerance = 1e-7;
    options.threadCount = 1;
    options.backend = backend;
    configure(options);
    return pdlp::PdlpSolver{}.solve(problem, options);
}

pdlp::CompiledLp feasibleLp(unsigned seed, int rows, int columns) {
    // Feasible by construction around a known point, bounded by boxes.
    std::mt19937 generator(seed);
    std::uniform_real_distribution<double> value(-1.0, 1.0);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::uniform_int_distribution<int> column(0, columns - 1);
    std::vector<pdlp::MatrixTriplet> triplets;
    for (int row = 0; row < rows; ++row) {
        for (int k = 0; k < 5; ++k) {
            triplets.push_back({row, column(generator), value(generator)});
        }
    }
    pdlp::CompiledLp problem;
    problem.matrix = pdlp::SparseMatrix::fromTriplets(rows, columns, std::move(triplets));
    problem.objective.resize(static_cast<std::size_t>(columns));
    problem.variableLower.assign(static_cast<std::size_t>(columns), 0.0);
    problem.variableUpper.assign(static_cast<std::size_t>(columns), 10.0);
    std::vector<double> point(static_cast<std::size_t>(columns));
    for (int j = 0; j < columns; ++j) {
        problem.objective[static_cast<std::size_t>(j)] = value(generator);
        point[static_cast<std::size_t>(j)] = 10.0 * unit(generator);
    }
    std::vector<double> activity;
    problem.matrix.multiply(point, activity);
    problem.rowLower.resize(static_cast<std::size_t>(rows));
    problem.rowUpper.resize(static_cast<std::size_t>(rows));
    for (int i = 0; i < rows; ++i) {
        const auto k = static_cast<std::size_t>(i);
        switch (i % 3) {
            case 0: problem.rowLower[k] = activity[k]; problem.rowUpper[k] = activity[k]; break;
            case 1: problem.rowLower[k] = -contract::kInf; problem.rowUpper[k] = activity[k] + 1.0; break;
            default: problem.rowLower[k] = activity[k] - 1.0; problem.rowUpper[k] = contract::kInf; break;
        }
    }
    return problem;
}

pdlp::CompiledLp infeasibleLp() {
    pdlp::CompiledLp problem;
    problem.matrix = pdlp::SparseMatrix::fromTriplets(2, 2, {{0, 0, 1}, {0, 1, 1}, {1, 0, 1}, {1, 1, 1}});
    problem.objective = {1.0, 1.0};
    problem.variableLower = {0.0, 0.0};
    problem.variableUpper = {contract::kInf, contract::kInf};
    problem.rowLower = {-contract::kInf, 3.0};
    problem.rowUpper = {1.0, contract::kInf};
    return problem;
}

pdlp::CompiledLp unboundedLp() {
    pdlp::CompiledLp problem;
    problem.matrix = pdlp::SparseMatrix::fromTriplets(1, 2, {{0, 0, 1}, {0, 1, -1}});
    problem.objective = {-1.0, -1.0};
    problem.variableLower = {0.0, 0.0};
    problem.variableUpper = {contract::kInf, contract::kInf};
    problem.rowLower = {-contract::kInf};
    problem.rowUpper = {1.0};
    return problem;
}

void solveEquivalence() {
    struct Variant {
        const char* name;
        void (*configure)(pdlp::PdlpOptions&);
    };
    const Variant variants[] = {
        {"default", [](pdlp::PdlpOptions&) {}},
        {"noRuiz", [](pdlp::PdlpOptions& o) { o.useRuizScaling = false; }},
        {"noAveraging", [](pdlp::PdlpOptions& o) { o.useAveraging = false; }},
        {"noRestarts", [](pdlp::PdlpOptions& o) { o.useRestarts = false; }},
        {"fixedStep", [](pdlp::PdlpOptions& o) { o.useAdaptiveLinesearch = false; }},
        {"noPreconditioner", [](pdlp::PdlpOptions& o) { o.useDiagonalPreconditioning = false; }},
        {"noPolishing", [](pdlp::PdlpOptions& o) { o.useFeasibilityPolishing = false; }},
        {"iterationLimit", [](pdlp::PdlpOptions& o) { o.iterationLimit = 300; }},
    };
    struct Case {
        const char* name;
        pdlp::CompiledLp problem;
    };
    const Case cases[] = {
        {"small", feasibleLp(1u, 80, 120)},
        {"medium", feasibleLp(2u, 1500, 2500)},
        {"infeasible", infeasibleLp()},
        {"unbounded", unboundedLp()},
    };

    for (const Case& c : cases) {
        for (const Variant& v : variants) {
            run(std::string("solve ") + c.name + "/" + v.name, [&] {
                const pdlp::PdlpResult cpu = solveWith(c.problem, pdlp::ComputeBackend::Cpu, v.configure);
                const pdlp::PdlpResult gpu = solveWith(c.problem, pdlp::ComputeBackend::Cuda, v.configure);
                contract::check(gpu.executedBackend == pdlp::ComputeBackend::Cuda,
                                "the CUDA solve must report the CUDA backend: " + gpu.backendMessage);
                contract::check(cpu.status == gpu.status,
                                std::string("status differs: cpu ") + pdlp::toString(cpu.status) +
                                    ", cuda " + pdlp::toString(gpu.status));
                if (cpu.status == pdlp::PdlpStatus::Optimal) {
                    contract::requireClose(cpu.primalObjective, gpu.primalObjective, 1e-6, "objective");
                    contract::check(gpu.primalResidual <= 1e-7 && gpu.dualResidual <= 1e-7 &&
                                        gpu.relativeGap <= 1e-7,
                                    "CUDA optimum must meet the tolerances");
                }
                // Iteration paths may diverge by rounding; report, do not assert equality.
                std::cout << "        iterations cpu " << cpu.iterations << " cuda " << gpu.iterations
                          << ", cuda setup " << gpu.backendProfile.setupSeconds << " s, D2H "
                          << gpu.backendProfile.deviceToHostBytes << " B\n";
            });
        }
    }
}

// Auto must pick CUDA above the threshold when a device is usable.
void autoSelection() {
    run("auto selects cuda above threshold", [] {
        const pdlp::CompiledLp problem = feasibleLp(3u, 200, 300);
        pdlp::PdlpOptions options;
        options.cudaNonzeroThreshold = 0;
        const pdlp::PdlpResult result = pdlp::PdlpSolver{}.solve(problem, options);
        contract::check(result.executedBackend == pdlp::ComputeBackend::Cuda,
                        "Auto with a usable device and threshold 0 must run on CUDA: " +
                            result.backendMessage);
    });
    run("auto keeps small problems on cpu", [] {
        const pdlp::PdlpResult result = pdlp::PdlpSolver{}.solve(feasibleLp(4u, 20, 30), {});
        contract::check(result.executedBackend == pdlp::ComputeBackend::Cpu,
                        "Auto below the threshold must run on the CPU");
    });
    run("invalid device index is refused", [] {
        pdlp::PdlpOptions options;
        options.backend = pdlp::ComputeBackend::Cuda;
        options.cudaDevice = 4096;
        const pdlp::PdlpResult result = pdlp::PdlpSolver{}.solve(feasibleLp(5u, 20, 30), options);
        contract::check(result.status == pdlp::PdlpStatus::InvalidProblem,
                        "a nonexistent device must be refused, not replaced by the CPU");
    });
}

}  // namespace

int main() {
    const pdlp::CudaAvailability availability = pdlp::cudaAvailability(0);
    if (!availability.usable) {
        std::cout << "SKIPPED: no usable CUDA device (" << availability.reason << ")\n";
        return 77;
    }
    std::cout << "CUDA device 0: " << availability.deviceName << '\n';

    kernelEquivalence();
    solveEquivalence();
    autoSelection();

    if (failures == 0) {
        std::cout << "All PDLP CUDA tests passed\n";
        return EXIT_SUCCESS;
    }
    std::cout << failures << " test(s) failed\n";
    return EXIT_FAILURE;
}
