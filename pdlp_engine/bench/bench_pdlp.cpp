// Benchmark harness: generates large, feasible, sparse LPs and reports
// wall-clock throughput of the PDLP engine.
//
//   ./pdlp_bench [rows] [cols] [nnzPerRow] [iterationLimit] [threads] [backend] [repeats]
//
// backend is cpu (default), cuda or auto. One untimed warm-up solve precedes
// `repeats` timed solves (default 3), and the median is reported, so one-time
// costs -- CUDA context creation, first-touch page faults -- are visible as the
// gap between the warm-up and the timed runs rather than hidden in them.
//
// Timings reported per solve:
//   end-to-end   wall clock around PdlpSolver::solve: scaling, preconditioner,
//                device setup and problem upload, iterations, polishing
//   setup        device allocation and problem upload (0 on the CPU)
//   core         end-to-end minus setup: what an iteration loop costs
//   host checks  termination/certificate evaluation on the host, including
//                the iterate downloads a device backend needs for them

#include "pdlp/pdlp_solver.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <thread>
#include <vector>

namespace {

constexpr double kInfinity = std::numeric_limits<double>::infinity();

// Builds a feasible LP with a known interior point x0, so that every generated
// instance is guaranteed solvable and the reported residuals are meaningful.
pdlp::CompiledLp makeInstance(int rows, int columns, int nnzPerRow, unsigned seed) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<double> coefficient(-1.0, 1.0);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::uniform_int_distribution<int> column(0, columns - 1);

    std::vector<pdlp::MatrixTriplet> triplets;
    triplets.reserve(static_cast<std::size_t>(rows) * nnzPerRow);
    for (int row = 0; row < rows; ++row) {
        for (int k = 0; k < nnzPerRow; ++k) {
            double value = coefficient(generator);
            if (value == 0.0) {
                value = 1.0;
            }
            triplets.push_back({row, column(generator), value});
        }
    }

    pdlp::CompiledLp problem;
    problem.matrix = pdlp::SparseMatrix::fromTriplets(rows, columns, std::move(triplets));

    problem.objective.resize(columns);
    problem.variableLower.assign(columns, 0.0);
    problem.variableUpper.assign(columns, 10.0);

    std::vector<double> reference(columns);
    for (int j = 0; j < columns; ++j) {
        problem.objective[j] = coefficient(generator);
        reference[j] = 10.0 * unit(generator);
    }

    std::vector<double> activity;
    problem.matrix.multiply(reference, activity);

    problem.rowLower.resize(rows);
    problem.rowUpper.resize(rows);
    for (int i = 0; i < rows; ++i) {
        // Mix equalities, one-sided rows and ranged rows.
        switch (i % 3) {
            case 0:
                problem.rowLower[i] = activity[i];
                problem.rowUpper[i] = activity[i];
                break;
            case 1:
                problem.rowLower[i] = -kInfinity;
                problem.rowUpper[i] = activity[i] + 1.0;
                break;
            default:
                problem.rowLower[i] = activity[i] - 1.0;
                problem.rowUpper[i] = activity[i] + 1.0;
                break;
        }
    }
    return problem;
}

int argOr(int argc, char** argv, int index, int fallback) {
    return index < argc ? std::atoi(argv[index]) : fallback;
}

pdlp::ComputeBackend backendArg(int argc, char** argv, int index) {
    if (index >= argc || std::strcmp(argv[index], "cpu") == 0) {
        return pdlp::ComputeBackend::Cpu;
    }
    if (std::strcmp(argv[index], "cuda") == 0) {
        return pdlp::ComputeBackend::Cuda;
    }
    if (std::strcmp(argv[index], "auto") == 0) {
        return pdlp::ComputeBackend::Auto;
    }
    std::cerr << "unknown backend '" << argv[index] << "' (cpu|cuda|auto)\n";
    std::exit(2);
}

struct Timed {
    pdlp::PdlpResult result;
    double endToEndSeconds = 0.0;
};

Timed timedSolve(const pdlp::CompiledLp& problem, const pdlp::PdlpOptions& options) {
    const auto start = std::chrono::steady_clock::now();
    Timed timed;
    timed.result = pdlp::PdlpSolver{}.solve(problem, options);
    timed.endToEndSeconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return timed;
}

}  // namespace

int main(int argc, char** argv) {
    const int rows = argOr(argc, argv, 1, 20000);
    const int columns = argOr(argc, argv, 2, 40000);
    const int nnzPerRow = argOr(argc, argv, 3, 10);
    const int iterationLimit = argOr(argc, argv, 4, 2000);
    const int threads = argOr(argc, argv, 5, 0);
    const pdlp::ComputeBackend backend = backendArg(argc, argv, 6);
    const int repeats = std::max(argOr(argc, argv, 7, 3), 1);

    const auto buildStart = std::chrono::steady_clock::now();
    const pdlp::CompiledLp problem = makeInstance(rows, columns, nnzPerRow, 12345u);
    const double buildSeconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - buildStart).count();

    pdlp::PdlpOptions options;
    options.iterationLimit = iterationLimit;
    options.terminationCheckFrequency = 200;
    options.useFeasibilityPolishing = false;
    options.threadCount = threads;
    options.backend = backend;

    const pdlp::CudaAvailability cuda = pdlp::cudaAvailability(0);
    std::cout << "hardware threads " << std::thread::hardware_concurrency() << '\n'
              << "requested threads " << threads << " (0 = all)\n"
              << "cuda compiled    " << (cuda.compiled ? "yes" : "no") << '\n'
              << "cuda device      " << (cuda.usable ? cuda.deviceName : "none (" + cuda.reason + ")") << '\n'
              << "cuda runtime     " << cuda.runtimeVersion << " driver " << cuda.driverVersion << '\n';

    const Timed warmUp = timedSolve(problem, options);
    if (warmUp.result.status == pdlp::PdlpStatus::InvalidProblem) {
        std::cout << "solve refused: " << warmUp.result.statusMessage << '\n';
        return 1;
    }

    std::vector<Timed> runs;
    for (int r = 0; r < repeats; ++r) {
        runs.push_back(timedSolve(problem, options));
    }
    std::sort(runs.begin(), runs.end(),
              [](const Timed& a, const Timed& b) { return a.endToEndSeconds < b.endToEndSeconds; });
    const Timed& median = runs[runs.size() / 2];
    const pdlp::PdlpResult& result = median.result;
    const pdlp::BackendProfile& profile = result.backendProfile;
    const double coreSeconds = median.endToEndSeconds - profile.setupSeconds;

    const double nnz = static_cast<double>(problem.matrix.nonzeros());
    const double perIteration = result.iterations > 0
        ? coreSeconds / static_cast<double>(result.iterations)
        : 0.0;

    std::cout << "rows            " << rows << '\n'
              << "columns         " << columns << '\n'
              << "nonzeros        " << nnz << '\n'
              << "build seconds   " << buildSeconds << '\n'
              << "backend         " << pdlp::toString(result.executedBackend)
              << " (" << result.backendMessage << ")\n"
              << "warm-up e2e s   " << warmUp.endToEndSeconds << '\n'
              << "timed runs      " << repeats << " (median reported)\n"
              << "status          " << pdlp::toString(result.status) << '\n'
              << "iterations      " << result.iterations << '\n'
              << "step trials     " << result.stepTrials << '\n'
              << "e2e seconds     " << median.endToEndSeconds << '\n'
              << "setup seconds   " << profile.setupSeconds << '\n'
              << "core seconds    " << coreSeconds << '\n'
              << "host check s    " << result.hostCheckSeconds
              << " (snapshots " << profile.snapshotSeconds << ")\n"
              << "H2D / D2H bytes " << profile.hostToDeviceBytes << " / " << profile.deviceToHostBytes << '\n'
              << "host syncs      " << profile.synchronisations << '\n'
              << "solve seconds   " << result.solveTimeSeconds << '\n'
              << "us / iteration  " << perIteration * 1e6 << '\n'
              << "Mnnz/s (2 SpMV) " << (perIteration > 0 ? (2.0 * nnz / perIteration) / 1e6 : 0.0) << '\n'
              << "primal residual " << result.primalResidual << '\n'
              << "dual residual   " << result.dualResidual << '\n'
              << "relative gap    " << result.relativeGap << '\n'
              << "objective       " << result.primalObjective << '\n';
    return 0;
}
