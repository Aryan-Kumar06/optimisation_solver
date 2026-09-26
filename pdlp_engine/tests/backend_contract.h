#pragma once

// Backend equivalence checks, shared by two executables:
//
//   pdlp_tests       reference = CPU, candidate = CPU. Exercises this harness and
//                    the CPU backend's contract on every build, so the checks the
//                    GPU is held to are themselves known to compile and pass.
//   pdlp_cuda_tests  reference = CPU, candidate = CUDA (CUDA builds only).
//
// Every comparison allows for summation-order differences and nothing else:
// the per-coordinate arithmetic is shared code (pdhg_math.h), so any larger
// gap is a bug, not rounding.

#include "pdlp/iteration_backend.h"
#include "pdlp/preconditioner.h"

#include <cmath>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace contract {

constexpr double kInf = std::numeric_limits<double>::infinity();

struct Fixture {
    std::string name;
    pdlp::CompiledLp problem;
};

inline void check(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

// |a - b| <= tolerance * (1 + max(|a|, |b|)), and identical non-finite values.
inline bool close(double a, double b, double tolerance) {
    if (!std::isfinite(a) || !std::isfinite(b)) {
        return (std::isnan(a) && std::isnan(b)) || a == b;
    }
    return std::abs(a - b) <= tolerance * (1.0 + std::max(std::abs(a), std::abs(b)));
}

inline void requireClose(double a, double b, double tolerance, const std::string& what) {
    check(close(a, b, tolerance),
          what + ": reference " + std::to_string(a) + " vs candidate " + std::to_string(b));
}

inline void requireClose(const std::vector<double>& a, const std::vector<double>& b,
                         double tolerance, const std::string& what) {
    check(a.size() == b.size(), what + ": size " + std::to_string(a.size()) + " vs " +
                                    std::to_string(b.size()));
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (!close(a[i], b[i], tolerance)) {
            throw std::runtime_error(what + "[" + std::to_string(i) + "]: reference " +
                                     std::to_string(a[i]) + " vs candidate " + std::to_string(b[i]));
        }
    }
}

// Row and column bounds cycling through every kind the solver distinguishes:
// equality, one-sided either way, ranged, free.
inline void assignBounds(pdlp::CompiledLp& problem, std::mt19937& generator) {
    std::uniform_real_distribution<double> value(-1.0, 1.0);
    const int columns = problem.numColumns();
    const int rows = problem.numRows();
    problem.objective.resize(static_cast<std::size_t>(columns));
    problem.variableLower.resize(static_cast<std::size_t>(columns));
    problem.variableUpper.resize(static_cast<std::size_t>(columns));
    for (int j = 0; j < columns; ++j) {
        const auto k = static_cast<std::size_t>(j);
        problem.objective[k] = value(generator);
        switch (j % 5) {
            case 0: problem.variableLower[k] = 0.0;   problem.variableUpper[k] = 10.0; break;
            case 1: problem.variableLower[k] = 0.0;   problem.variableUpper[k] = kInf; break;
            case 2: problem.variableLower[k] = -kInf; problem.variableUpper[k] = 5.0;  break;
            case 3: problem.variableLower[k] = -kInf; problem.variableUpper[k] = kInf; break;
            default: problem.variableLower[k] = 2.0;  problem.variableUpper[k] = 2.0;  break;  // fixed
        }
    }
    problem.rowLower.resize(static_cast<std::size_t>(rows));
    problem.rowUpper.resize(static_cast<std::size_t>(rows));
    for (int i = 0; i < rows; ++i) {
        const auto k = static_cast<std::size_t>(i);
        switch (i % 4) {
            case 0: problem.rowLower[k] = 1.0;   problem.rowUpper[k] = 1.0;  break;
            case 1: problem.rowLower[k] = -kInf; problem.rowUpper[k] = 3.0;  break;
            case 2: problem.rowLower[k] = -2.0;  problem.rowUpper[k] = kInf; break;
            default: problem.rowLower[k] = -1.0; problem.rowUpper[k] = 4.0;  break;
        }
    }
}

inline Fixture randomFixture(const std::string& name, int rows, int columns, int perRow, unsigned seed,
                             double valueScale = 1.0) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<double> value(-1.0, 1.0);
    std::uniform_int_distribution<int> column(0, std::max(columns - 1, 0));
    std::vector<pdlp::MatrixTriplet> triplets;
    if (columns > 0) {
        for (int row = 0; row < rows; ++row) {
            for (int k = 0; k < perRow; ++k) {
                triplets.push_back({row, column(generator), valueScale * value(generator)});
            }
        }
    }
    Fixture fixture{name, {}};
    fixture.problem.matrix = pdlp::SparseMatrix::fromTriplets(rows, columns, std::move(triplets));
    assignBounds(fixture.problem, generator);
    return fixture;
}

// Matrices chosen to reach every kernel path and every degenerate shape.
inline std::vector<Fixture> kernelFixtures() {
    std::vector<Fixture> fixtures;
    fixtures.push_back(randomFixture("random", 300, 420, 6, 11u));
    fixtures.push_back(randomFixture("tall", 900, 60, 3, 12u));
    fixtures.push_back(randomFixture("wide", 40, 1500, 20, 13u));
    fixtures.push_back(randomFixture("veryShortLines", 500, 500, 1, 14u));
    fixtures.push_back(randomFixture("tinyValues", 120, 150, 5, 15u, 1e-9));
    fixtures.push_back(randomFixture("hugeValues", 120, 150, 5, 16u, 1e9));
    fixtures.push_back(randomFixture("noConstraints", 0, 25, 0, 17u));
    fixtures.push_back(randomFixture("noNonzeros", 30, 40, 0, 18u));

    {
        // One nonzero, plus empty rows and columns around it.
        Fixture fixture{"oneNonzero", {}};
        fixture.problem.matrix = pdlp::SparseMatrix::fromTriplets(5, 7, {{2, 3, 1.5}});
        std::mt19937 generator(19u);
        assignBounds(fixture.problem, generator);
        fixtures.push_back(std::move(fixture));
    }
    {
        // Skewed: one dense row and one dense column in an otherwise sparse
        // matrix, the shape the heavy-line kernel exists for.
        std::mt19937 generator(20u);
        std::uniform_real_distribution<double> value(-1.0, 1.0);
        const int rows = 600;
        const int columns = 700;
        std::vector<pdlp::MatrixTriplet> triplets;
        for (int j = 0; j < columns; ++j) {
            triplets.push_back({0, j, value(generator)});
        }
        for (int i = 0; i < rows; ++i) {
            triplets.push_back({i, 5, value(generator)});
            triplets.push_back({i, (i * 7) % columns, value(generator)});
        }
        Fixture fixture{"skewed", {}};
        fixture.problem.matrix = pdlp::SparseMatrix::fromTriplets(rows, columns, std::move(triplets));
        assignBounds(fixture.problem, generator);
        fixtures.push_back(std::move(fixture));
    }
    return fixtures;
}

// A starting point that is finite everywhere but not necessarily feasible.
inline void startingPoint(const pdlp::CompiledLp& problem, unsigned seed,
                          std::vector<double>& primal, std::vector<double>& dual) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<double> value(-3.0, 3.0);
    primal.resize(static_cast<std::size_t>(problem.numColumns()));
    dual.resize(static_cast<std::size_t>(problem.numRows()));
    for (double& x : primal) {
        x = value(generator);
    }
    for (double& y : dual) {
        y = value(generator);
    }
}

using BackendFactory = std::function<std::unique_ptr<pdlp::IterationBackend>(
    const pdlp::CompiledLp&, const pdlp::DiagonalPreconditioner&)>;

// Drives two backends through the same sequence -- trials accepted and
// rejected, commits, averaging, a restart from the average, and a non-finite
// start -- and requires identical results up to `tolerance`.
inline void compareBackends(const Fixture& fixture, const BackendFactory& makeReference,
                            const BackendFactory& makeCandidate, double tolerance) {
    const pdlp::CompiledLp& problem = fixture.problem;
    const std::string& name = fixture.name;
    const pdlp::DiagonalPreconditioner preconditioner = pdlp::Preconditioner::compute(problem, true);
    auto reference = makeReference(problem, preconditioner);
    auto candidate = makeCandidate(problem, preconditioner);
    check(reference != nullptr && candidate != nullptr, name + ": backend creation failed");

    std::vector<double> primal;
    std::vector<double> dual;
    startingPoint(problem, 101u, primal, dual);
    reference->reset(primal, dual);
    candidate->reset(primal, dual);
    check(candidate->iteration() == 0 && candidate->averageEmpty(), name + ": reset state");
    requireClose(reference->primal(), candidate->primal(), 0.0, name + " reset primal");
    requireClose(reference->dual(), candidate->dual(), 0.0, name + " reset dual");

    const auto compareTrial = [&](const pdlp::StepParameters& steps, const std::string& label) {
        const pdlp::KernelTrialResult a = reference->trial(steps);
        const pdlp::KernelTrialResult b = candidate->trial(steps);
        requireClose(a.primalMovementWeighted, b.primalMovementWeighted, tolerance, name + " " + label + " primal movement");
        requireClose(a.dualMovementWeighted, b.dualMovementWeighted, tolerance, name + " " + label + " dual movement");
        requireClose(a.interaction, b.interaction, tolerance, name + " " + label + " interaction");
        check(a.finite == b.finite, name + " " + label + ": finite flags differ");
    };

    pdlp::StepParameters large;
    large.globalStep = 0.9;
    large.primalWeight = 1.7;
    pdlp::StepParameters small;
    small.globalStep = 0.15;
    small.primalWeight = 0.6;

    // A rejected trial must leave the iterate untouched: repeating it gives the
    // same answer, and a different step from the same state matches.
    compareTrial(large, "trial");
    compareTrial(large, "repeated trial");
    compareTrial(small, "retry after rejection");
    reference->commit();
    candidate->commit();
    check(candidate->iteration() == 1, name + ": commit must advance the iteration");
    requireClose(reference->primal(), candidate->primal(), tolerance, name + " committed primal");
    requireClose(reference->dual(), candidate->dual(), tolerance, name + " committed dual");

    reference->addToAverage(0.5);
    candidate->addToAverage(0.5);
    compareTrial(large, "second trial");
    reference->commit();
    candidate->commit();
    reference->addToAverage(0.25);
    candidate->addToAverage(0.25);
    reference->addToAverage(0.0);   // ignored by contract
    candidate->addToAverage(-1.0);  // ignored by contract
    check(!candidate->averageEmpty(), name + ": average must be non-empty");
    requireClose(reference->average().primal, candidate->average().primal, tolerance, name + " average primal");
    requireClose(reference->average().dual, candidate->average().dual, tolerance, name + " average dual");

    // Restarting from the average replaces the iterate and recomputes A*x; the
    // next trial reads that activity, so it checks the refresh too.
    reference->restartFromAverage();
    candidate->restartFromAverage();
    requireClose(reference->primal(), candidate->primal(), tolerance, name + " restarted primal");
    requireClose(reference->dual(), candidate->dual(), tolerance, name + " restarted dual");
    compareTrial(small, "trial after restart");
    reference->resetAverage();
    candidate->resetAverage();
    check(candidate->averageEmpty(), name + ": resetAverage must empty the average");

    // A NaN in the iterate must surface as a non-finite reduction on both.
    if (!primal.empty() && problem.matrix.nonzeros() > 0) {
        primal[0] = std::numeric_limits<double>::quiet_NaN();
        reference->reset(primal, dual);
        candidate->reset(primal, dual);
        const pdlp::KernelTrialResult a = reference->trial(large);
        const pdlp::KernelTrialResult b = candidate->trial(large);
        check(!a.finite && !b.finite, name + ": a NaN iterate must be reported as non-finite");
    }
}

}  // namespace contract
