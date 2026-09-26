#include "backend_contract.h"

#include "pdlp/iteration_backend.h"
#include "pdlp/parallel.h"
#include "pdlp/pdhg_math.h"
#include "pdlp/pdlp_solver.h"
#include "pdlp/scaling.h"
#include "pdlp/sparse_matrix.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kInfinity = std::numeric_limits<double>::infinity();

// Tests index vectors with int loop counters throughout. This keeps that
// readable without a static_cast at every subscript, and keeps the suite clean
// under -Wsign-conversion.
template <typename T>
T& at(std::vector<T>& values, int index) {
    return values[static_cast<std::size_t>(index)];
}

template <typename T>
const T& at(const std::vector<T>& values, int index) {
    return values[static_cast<std::size_t>(index)];
}

int failures = 0;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void requireNear(double actual, double expected, double tolerance, const std::string& name) {
    if (!(std::abs(actual - expected) <= tolerance)) {
        throw std::runtime_error(
            name + " expected " + std::to_string(expected) +
            " but received " + std::to_string(actual)
        );
    }
}

pdlp::PdlpOptions strictOptions() {
    pdlp::PdlpOptions options;
    options.iterationLimit = 100000;
    options.primalTolerance = 2e-6;
    options.dualTolerance = 2e-6;
    options.gapTolerance = 2e-6;
    options.terminationCheckFrequency = 50;
    options.polishingIterations = 2000;
    return options;
}

// ---------------------------------------------------------------------------
// Sparse matrix
// ---------------------------------------------------------------------------

void testDuplicateTriplets() {
    const auto matrix = pdlp::SparseMatrix::fromTriplets(
        1, 1, {{0, 0, 2.0}, {0, 0, -1.0}});
    std::vector<double> product;
    matrix.multiply({4.0}, product);
    requireNear(product[0], 4.0, 1e-12, "merged matrix coefficient");
    require(matrix.nonzeros() == 1, "duplicates must merge into one entry");
}

void testCancellingDuplicatesAreDropped() {
    const auto matrix = pdlp::SparseMatrix::fromTriplets(
        1, 2, {{0, 0, 2.0}, {0, 0, -2.0}, {0, 1, 3.0}});
    require(matrix.nonzeros() == 1, "entries cancelling to zero must be dropped");
    require(matrix.validate(), "matrix must validate after cancellation");
}

void testUnsortedTripletsAndEmptyLines() {
    // Reversed input order, an empty row (1) and an empty column (1).
    const auto matrix = pdlp::SparseMatrix::fromTriplets(
        3, 3, {{2, 2, 5.0}, {0, 2, 1.0}, {2, 0, 4.0}, {0, 0, 2.0}});
    require(matrix.validate(), "matrix built from unsorted triplets must validate");

    std::vector<double> product;
    matrix.multiply({1.0, 1.0, 1.0}, product);
    requireNear(product[0], 3.0, 1e-12, "row 0 product");
    requireNear(product[1], 0.0, 1e-12, "empty row product");
    requireNear(product[2], 9.0, 1e-12, "row 2 product");

    std::vector<double> transposed;
    matrix.transposeMultiply({1.0, 1.0, 1.0}, transposed);
    requireNear(transposed[0], 6.0, 1e-12, "column 0 transpose product");
    requireNear(transposed[1], 0.0, 1e-12, "empty column transpose product");
    requireNear(transposed[2], 6.0, 1e-12, "column 2 transpose product");
}

// CSR and CSC must describe the same matrix; a transpose bug here is invisible
// in A*x but corrupts every dual step.
void testCsrCscAgree() {
    std::mt19937 generator(7);
    std::uniform_real_distribution<double> value(-2.0, 2.0);
    std::uniform_int_distribution<int> pick(0, 39);

    std::vector<pdlp::MatrixTriplet> triplets;
    for (int k = 0; k < 400; ++k) {
        triplets.push_back({pick(generator) % 25, pick(generator), value(generator)});
    }
    const auto matrix = pdlp::SparseMatrix::fromTriplets(25, 40, triplets);
    require(matrix.validate(), "random matrix must validate");

    std::vector<double> x(40);
    std::vector<double> y(25);
    for (double& entry : x) { entry = value(generator); }
    for (double& entry : y) { entry = value(generator); }

    std::vector<double> ax;
    std::vector<double> aty;
    matrix.multiply(x, ax);
    matrix.transposeMultiply(y, aty);

    // <A x, y> == <x, A^T y> holds only if both representations agree.
    double left = 0.0;
    for (int i = 0; i < 25; ++i) { left += at(ax, i) * at(y, i); }
    double right = 0.0;
    for (int j = 0; j < 40; ++j) { right += at(x, j) * at(aty, j); }
    requireNear(left, right, 1e-9 * (1.0 + std::abs(left)), "CSR/CSC adjoint identity");
}

void testMatrixScaling() {
    const auto matrix = pdlp::SparseMatrix::fromTriplets(
        2, 2, {{0, 0, 1.0}, {0, 1, 2.0}, {1, 0, 3.0}, {1, 1, 4.0}});
    const auto scaled = matrix.scaled({10.0, 100.0}, {1.0, 0.5});

    std::vector<double> product;
    scaled.multiply({1.0, 0.0}, product);
    requireNear(product[0], 10.0, 1e-12, "scaled a00");
    requireNear(product[1], 300.0, 1e-12, "scaled a10");
    scaled.multiply({0.0, 1.0}, product);
    requireNear(product[0], 10.0, 1e-12, "scaled a01");
    requireNear(product[1], 200.0, 1e-12, "scaled a11");

    // The CSC copy must carry the same scaling as the CSR copy.
    std::vector<double> transposed;
    scaled.transposeMultiply({1.0, 0.0}, transposed);
    requireNear(transposed[0], 10.0, 1e-12, "scaled CSC a00");
    requireNear(transposed[1], 10.0, 1e-12, "scaled CSC a01");
}

// Chunked parallel products must be bitwise identical to the serial ones: each
// output entry is produced by exactly one chunk, in the same order.
void testParallelProductMatchesSerial() {
    std::mt19937 generator(11);
    std::uniform_real_distribution<double> value(-1.0, 1.0);
    std::uniform_int_distribution<int> pick(0, 1999);

    std::vector<pdlp::MatrixTriplet> triplets;
    for (int row = 0; row < 3000; ++row) {
        for (int k = 0; k < 12; ++k) {
            triplets.push_back({row, pick(generator), value(generator)});
        }
    }
    const auto matrix = pdlp::SparseMatrix::fromTriplets(3000, 2000, triplets);

    std::vector<double> x(2000);
    std::vector<double> y(3000);
    for (double& entry : x) { entry = value(generator); }
    for (double& entry : y) { entry = value(generator); }

    std::vector<double> serialAx;
    std::vector<double> serialAty;
    matrix.multiply(x, serialAx);
    matrix.transposeMultiply(y, serialAty);

    pdlp::Executor executor(4);
    const auto plan = pdlp::SpmvPlan::build(matrix, executor.threadCount());
    require(plan.rowChunkCount() >= executor.threadCount(),
            "plan must produce at least one chunk per worker");

    std::vector<double> parallelAx;
    std::vector<double> parallelAty;
    matrix.multiply(x, parallelAx, &executor, &plan);
    matrix.transposeMultiply(y, parallelAty, &executor, &plan);

    for (std::size_t i = 0; i < serialAx.size(); ++i) {
        require(serialAx[i] == parallelAx[i], "parallel A*x must match serial bitwise");
    }
    for (std::size_t j = 0; j < serialAty.size(); ++j) {
        require(serialAty[j] == parallelAty[j], "parallel A^T*y must match serial bitwise");
    }
}

void testExecutorCoversEveryPart() {
    pdlp::Executor executor(4);
    std::vector<int> visits(static_cast<std::size_t>(executor.threadCount()), 0);
    for (int round = 0; round < 50; ++round) {
        executor.run([&](int part) { ++visits[static_cast<std::size_t>(part)]; });
    }
    for (int visited : visits) {
        require(visited == 50, "every worker must run in every dispatch");
    }
}

// ---------------------------------------------------------------------------
// Solver behaviour
// ---------------------------------------------------------------------------

void testEqualityRow() {
    pdlp::CompiledLp problem;
    problem.matrix = pdlp::SparseMatrix::fromTriplets(1, 2, {{0, 0, 1.0}, {0, 1, 1.0}});
    problem.objective = {1.0, 2.0};
    problem.rowLower = {1.0};
    problem.rowUpper = {1.0};
    problem.variableLower = {0.0, 0.0};
    problem.variableUpper = {kInfinity, kInfinity};

    const auto result = pdlp::PdlpSolver{}.solve(problem, strictOptions());
    require(result.status == pdlp::PdlpStatus::Optimal, "Equality LP did not solve");
    requireNear(result.primal[0], 1.0, 2e-4, "equality x");
    requireNear(result.primal[1], 0.0, 2e-4, "equality y");
    requireNear(result.primalObjective, 1.0, 2e-4, "equality objective");
}

void testUpperBoundRow() {
    pdlp::CompiledLp problem;
    problem.matrix = pdlp::SparseMatrix::fromTriplets(1, 1, {{0, 0, 1.0}});
    problem.objective = {-1.0};
    problem.rowLower = {-kInfinity};
    problem.rowUpper = {3.0};
    problem.variableLower = {0.0};
    problem.variableUpper = {10.0};

    const auto result = pdlp::PdlpSolver{}.solve(problem, strictOptions());
    require(result.status == pdlp::PdlpStatus::Optimal, "Upper-bound LP did not solve");
    requireNear(result.primal[0], 3.0, 2e-4, "upper-bound x");
    requireNear(result.primalObjective, -3.0, 2e-4, "upper-bound objective");
}

void testLowerBoundRow() {
    pdlp::CompiledLp problem;
    problem.matrix = pdlp::SparseMatrix::fromTriplets(1, 1, {{0, 0, 1.0}});
    problem.objective = {1.0};
    problem.rowLower = {2.0};
    problem.rowUpper = {kInfinity};
    problem.variableLower = {-10.0};
    problem.variableUpper = {10.0};

    const auto result = pdlp::PdlpSolver{}.solve(problem, strictOptions());
    require(result.status == pdlp::PdlpStatus::Optimal, "Lower-bound LP did not solve");
    requireNear(result.primal[0], 2.0, 2e-4, "lower-bound x");
    requireNear(result.primalObjective, 2.0, 2e-4, "lower-bound objective");
}

void testRangedRow() {
    pdlp::CompiledLp problem;
    problem.matrix = pdlp::SparseMatrix::fromTriplets(1, 1, {{0, 0, 2.0}});
    problem.objective = {-1.0};
    problem.rowLower = {2.0};
    problem.rowUpper = {6.0};
    problem.variableLower = {0.0};
    problem.variableUpper = {10.0};

    const auto result = pdlp::PdlpSolver{}.solve(problem, strictOptions());
    require(result.status == pdlp::PdlpStatus::Optimal, "Ranged-row LP did not solve");
    requireNear(result.primal[0], 3.0, 2e-4, "ranged-row x");
    requireNear(result.primalObjective, -3.0, 2e-4, "ranged-row objective");
}

// The dual prox is written so that a strictly inactive row yields exactly 0.0,
// not the rounding residue of v - step*(v/step). Non-zero noise on slack rows
// leaks into the dual objective and the reported gap.
void testInactiveRowDualIsExactlyZero() {
    pdlp::CompiledLp problem;
    problem.matrix = pdlp::SparseMatrix::fromTriplets(
        2, 1, {{0, 0, 1.0}, {1, 0, 1.0}});
    problem.objective = {1.0};
    problem.rowLower = {2.0, -kInfinity};
    problem.rowUpper = {kInfinity, 1000.0};   // row 1 is far from active
    problem.variableLower = {0.0};
    problem.variableUpper = {10.0};

    const auto result = pdlp::PdlpSolver{}.solve(problem, strictOptions());
    require(result.status == pdlp::PdlpStatus::Optimal, "Slack-row LP did not solve");
    require(result.rowDual[1] == 0.0,
            "dual of a strictly inactive row must be exactly zero, got " +
            std::to_string(result.rowDual[1]));
}

// Entries spanning eight orders of magnitude. Without equilibration the large
// rows inflate the residual normaliser until the small rows' violations vanish
// below tolerance, and the solver reports success on an infeasible point.
void testIllConditionedProblem() {
    pdlp::CompiledLp problem;
    problem.matrix = pdlp::SparseMatrix::fromTriplets(
        2, 2,
        {{0, 0, 1e-4}, {0, 1, 1e-4}, {1, 0, 1e4}}
    );
    problem.objective = {-1.0, -1.0};
    problem.rowLower = {-kInfinity, -kInfinity};
    problem.rowUpper = {3e-4, 2e4};          // x0 + x1 <= 3, x0 <= 2
    problem.variableLower = {0.0, 0.0};
    problem.variableUpper = {10.0, 10.0};

    auto options = strictOptions();
    options.primalTolerance = 1e-8;
    options.dualTolerance = 1e-8;
    options.gapTolerance = 1e-8;

    const auto result = pdlp::PdlpSolver{}.solve(problem, options);
    require(result.status == pdlp::PdlpStatus::Optimal,
            "Ill-conditioned LP did not solve");
    requireNear(result.primalObjective, -3.0, 1e-5, "ill-conditioned objective");

    // Absolute feasibility, not the normalised residual: the normaliser is what
    // hid this failure before.
    const double activity0 = 1e-4 * (result.primal[0] + result.primal[1]);
    const double activity1 = 1e4 * result.primal[0];
    require(activity0 <= 3e-4 + 1e-9,
            "row 0 absolute violation too large: " + std::to_string(activity0 - 3e-4));
    require(activity1 <= 2e4 + 1e-3,
            "row 1 absolute violation too large: " + std::to_string(activity1 - 2e4));
}

void testRuizEquilibrationReducesSpread() {
    pdlp::CompiledLp problem;
    problem.matrix = pdlp::SparseMatrix::fromTriplets(
        3, 3,
        {{0, 0, 1e-5}, {0, 1, 2e-5}, {1, 1, 1e3}, {1, 2, 5e2}, {2, 0, 7.0}, {2, 2, 3.0}}
    );
    problem.objective = {1.0, 1.0, 1.0};
    problem.rowLower.assign(3, 0.0);
    problem.rowUpper.assign(3, 1.0);
    problem.variableLower.assign(3, 0.0);
    problem.variableUpper.assign(3, 1.0);

    const double before = pdlp::RuizScaler::conditionSpread(problem.matrix);
    const auto scaling = pdlp::RuizScaler::equilibrate(problem, 10);
    const double after = pdlp::RuizScaler::conditionSpread(scaling.problem.matrix);

    require(before > 1e6, "test matrix should start badly scaled");
    require(after < 10.0,
            "equilibration should bring row norms within an order of magnitude, got " +
            std::to_string(after));
}

// Degenerate: many constraints active at the optimum, and a redundant copy of
// each. Degeneracy is the case the problem statement calls out explicitly.
void testDegenerateProblem() {
    std::vector<pdlp::MatrixTriplet> triplets;
    const int n = 12;
    // x_i + x_{i+1} <= 1 for every i, each row duplicated to force degeneracy.
    int row = 0;
    for (int i = 0; i + 1 < n; ++i) {
        triplets.push_back({row, i, 1.0});
        triplets.push_back({row, i + 1, 1.0});
        ++row;
        triplets.push_back({row, i, 1.0});
        triplets.push_back({row, i + 1, 1.0});
        ++row;
    }

    pdlp::CompiledLp problem;
    problem.matrix = pdlp::SparseMatrix::fromTriplets(row, n, triplets);
    problem.objective.assign(n, -1.0);
    problem.rowLower.assign(static_cast<std::size_t>(row), -kInfinity);
    problem.rowUpper.assign(static_cast<std::size_t>(row), 1.0);
    problem.variableLower.assign(n, 0.0);
    problem.variableUpper.assign(n, 1.0);

    const auto result = pdlp::PdlpSolver{}.solve(problem, strictOptions());
    require(result.status == pdlp::PdlpStatus::Optimal, "Degenerate LP did not solve");
    // Optimum takes alternate variables at 1: ceil(n/2) = 6.
    requireNear(result.primalObjective, -6.0, 1e-3, "degenerate objective");
}

// The engine has no unboundedness certificate. What it must never do is claim
// optimality on an unbounded problem.
void testUnboundedIsNotReportedOptimal() {
    pdlp::CompiledLp problem;
    problem.matrix = pdlp::SparseMatrix::fromTriplets(1, 1, {{0, 0, 1.0}});
    problem.objective = {-1.0};
    problem.rowLower = {-kInfinity};
    problem.rowUpper = {kInfinity};
    problem.variableLower = {0.0};
    problem.variableUpper = {kInfinity};

    auto options = strictOptions();
    options.iterationLimit = 2000;
    const auto result = pdlp::PdlpSolver{}.solve(problem, options);
    require(result.status != pdlp::PdlpStatus::Optimal,
            "unbounded LP must not be reported optimal");
}

void testInvalidProblemIsRejected() {
    pdlp::CompiledLp problem;
    problem.matrix = pdlp::SparseMatrix::fromTriplets(1, 1, {{0, 0, 1.0}});
    problem.objective = {1.0};
    problem.rowLower = {5.0};
    problem.rowUpper = {1.0};   // crossed row bounds
    problem.variableLower = {0.0};
    problem.variableUpper = {1.0};

    const auto result = pdlp::PdlpSolver{}.solve(problem, strictOptions());
    require(result.status == pdlp::PdlpStatus::InvalidProblem,
            "crossed row bounds must be rejected");

    problem.rowLower = {0.0};
    problem.variableLower = {2.0};
    problem.variableUpper = {1.0};   // crossed variable bounds
    const auto second = pdlp::PdlpSolver{}.solve(problem, strictOptions());
    require(second.status == pdlp::PdlpStatus::InvalidProblem,
            "crossed variable bounds must be rejected");
}

void testTimeLimitIsHonoured() {
    std::mt19937 generator(3);
    std::uniform_real_distribution<double> value(-1.0, 1.0);
    std::uniform_int_distribution<int> pick(0, 3999);

    std::vector<pdlp::MatrixTriplet> triplets;
    for (int row = 0; row < 3000; ++row) {
        for (int k = 0; k < 12; ++k) {
            triplets.push_back({row, pick(generator), value(generator)});
        }
    }

    pdlp::CompiledLp problem;
    problem.matrix = pdlp::SparseMatrix::fromTriplets(3000, 4000, triplets);
    problem.objective.assign(4000, 1.0);
    problem.rowLower.assign(3000, -1.0);
    problem.rowUpper.assign(3000, 1.0);
    problem.variableLower.assign(4000, -1.0);
    problem.variableUpper.assign(4000, 1.0);

    pdlp::PdlpOptions options;
    options.iterationLimit = 100000000;
    // A relative gap of exactly zero requires the primal and dual objectives to
    // agree bit for bit, so the solve cannot terminate early and the time limit
    // is the only way out. Asserting against a merely tiny tolerance made this
    // test flaky: the solver became fast enough to sometimes reach it first.
    options.primalTolerance = 0.0;
    options.dualTolerance = 0.0;
    options.gapTolerance = 0.0;
    options.timeLimitSeconds = 0.25;

    const auto result = pdlp::PdlpSolver{}.solve(problem, options);
    require(result.status == pdlp::PdlpStatus::TimeLimit,
            "solver should stop at the time limit, got " +
            std::string(pdlp::toString(result.status)));
    // Generous ceiling: polishing must also respect the remaining budget.
    require(result.solveTimeSeconds < 3.0,
            "time limit overshot: " + std::to_string(result.solveTimeSeconds));
}

// Thread count must not change the answer beyond reduction-order rounding.
void testThreadCountDoesNotChangeTheAnswer() {
    std::mt19937 generator(5);
    std::uniform_real_distribution<double> value(-1.0, 1.0);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::uniform_int_distribution<int> pick(0, 2999);

    const int rows = 2500;
    const int columns = 3000;
    std::vector<pdlp::MatrixTriplet> triplets;
    for (int row = 0; row < rows; ++row) {
        for (int k = 0; k < 10; ++k) {
            triplets.push_back({row, pick(generator), value(generator)});
        }
    }

    pdlp::CompiledLp problem;
    problem.matrix = pdlp::SparseMatrix::fromTriplets(rows, columns, triplets);
    problem.objective.resize(columns);
    problem.variableLower.assign(columns, 0.0);
    problem.variableUpper.assign(columns, 5.0);

    std::vector<double> reference(columns);
    for (int j = 0; j < columns; ++j) {
        at(problem.objective, j) = value(generator);
        at(reference, j) = 5.0 * unit(generator);
    }
    std::vector<double> activity;
    problem.matrix.multiply(reference, activity);
    problem.rowLower.resize(rows);
    problem.rowUpper.resize(rows);
    for (int i = 0; i < rows; ++i) {
        at(problem.rowLower, i) = -kInfinity;
        at(problem.rowUpper, i) = at(activity, i) + 0.5;
    }

    pdlp::PdlpOptions options;
    options.iterationLimit = 4000;
    options.terminationCheckFrequency = 200;
    options.useFeasibilityPolishing = false;

    options.threadCount = 1;
    const auto serial = pdlp::PdlpSolver{}.solve(problem, options);
    options.threadCount = 4;
    const auto parallel = pdlp::PdlpSolver{}.solve(problem, options);

    const double scale = 1.0 + std::abs(serial.primalObjective);
    requireNear(parallel.primalObjective, serial.primalObjective, 1e-6 * scale,
                "objective must not depend on thread count");
}


// A moderately sized random LP with a known feasible interior point, used by the
// step-policy tests below. Every policy must land on the same optimum.
pdlp::CompiledLp randomFeasibleProblem(int rows, int columns, unsigned seed) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<double> value(-1.0, 1.0);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::uniform_int_distribution<int> pick(0, columns - 1);

    std::vector<pdlp::MatrixTriplet> triplets;
    for (int row = 0; row < rows; ++row) {
        for (int k = 0; k < 6; ++k) {
            triplets.push_back({row, pick(generator), value(generator)});
        }
    }

    pdlp::CompiledLp problem;
    problem.matrix = pdlp::SparseMatrix::fromTriplets(rows, columns, triplets);
    problem.objective.resize(static_cast<std::size_t>(columns));
    problem.variableLower.assign(static_cast<std::size_t>(columns), 0.0);
    problem.variableUpper.assign(static_cast<std::size_t>(columns), 10.0);

    std::vector<double> reference(static_cast<std::size_t>(columns));
    for (int j = 0; j < columns; ++j) {
        problem.objective[static_cast<std::size_t>(j)] = value(generator);
        reference[static_cast<std::size_t>(j)] = 10.0 * unit(generator);
    }
    std::vector<double> activity;
    problem.matrix.multiply(reference, activity);

    problem.rowLower.resize(static_cast<std::size_t>(rows));
    problem.rowUpper.resize(static_cast<std::size_t>(rows));
    for (int i = 0; i < rows; ++i) {
        const auto r = static_cast<std::size_t>(i);
        if (i % 3 == 0) {
            problem.rowLower[r] = activity[r];
            problem.rowUpper[r] = activity[r];
        } else {
            problem.rowLower[r] = -kInfinity;
            problem.rowUpper[r] = activity[r] + 1.0;
        }
    }
    return problem;
}

pdlp::PdlpOptions convergenceOptions() {
    pdlp::PdlpOptions options;
    options.iterationLimit = 60000;
    options.primalTolerance = 1e-8;
    options.dualTolerance = 1e-8;
    options.gapTolerance = 1e-8;
    options.terminationCheckFrequency = 40;
    options.useFeasibilityPolishing = false;
    return options;
}

// The point of the linesearch is that it is free to exceed the static
// Pock-Chambolle bound. If this ever stops holding, the linesearch has silently
// degenerated into the fixed-step policy.
void testLinesearchExceedsStaticBound() {
    const auto problem = randomFeasibleProblem(300, 420, 21u);
    auto options = convergenceOptions();
    options.useAdaptiveLinesearch = true;

    const auto result = pdlp::PdlpSolver{}.solve(problem, options);
    require(result.staticStepBound > 0.0, "static step bound should be reported");
    require(result.finalStepSize > result.staticStepBound,
            "linesearch should exceed the static bound, got ratio " +
            std::to_string(result.finalStepSize / result.staticStepBound));
    require(result.stepTrials >= result.iterations,
            "every iteration costs at least one trial");
}

// Step policy is a performance choice, never a correctness one: all of them must
// agree on the objective.
void testStepPoliciesAgreeOnTheOptimum() {
    const auto problem = randomFeasibleProblem(260, 360, 33u);

    auto linesearch = convergenceOptions();
    linesearch.useAdaptiveLinesearch = true;
    const auto adaptive = pdlp::PdlpSolver{}.solve(problem, linesearch);
    require(adaptive.status == pdlp::PdlpStatus::Optimal,
            "linesearch should solve the reference problem");

    auto fixed = convergenceOptions();
    fixed.useAdaptiveLinesearch = false;
    const auto staticStep = pdlp::PdlpSolver{}.solve(problem, fixed);
    require(staticStep.status == pdlp::PdlpStatus::Optimal,
            "fixed step should solve the reference problem");

    const double scale = 1.0 + std::abs(adaptive.primalObjective);
    requireNear(staticStep.primalObjective, adaptive.primalObjective, 1e-6 * scale,
                "fixed and adaptive step objectives");
}


// ---------------------------------------------------------------------------
// Infeasibility and unboundedness
// ---------------------------------------------------------------------------

// Contradictory rows: x + y >= 2 and x + y <= 1 cannot both hold.
void testDetectsInfeasibleRows() {
    pdlp::CompiledLp problem;
    problem.matrix = pdlp::SparseMatrix::fromTriplets(
        2, 2, {{0, 0, 1.0}, {0, 1, 1.0}, {1, 0, 1.0}, {1, 1, 1.0}});
    problem.objective = {1.0, 1.0};
    problem.rowLower = {2.0, -kInfinity};
    problem.rowUpper = {kInfinity, 1.0};
    problem.variableLower = {0.0, 0.0};
    problem.variableUpper = {10.0, 10.0};

    auto options = strictOptions();
    options.iterationLimit = 20000;
    const auto result = pdlp::PdlpSolver{}.solve(problem, options);
    require(result.status == pdlp::PdlpStatus::Infeasible,
            "contradictory rows should be proven infeasible, got " +
            std::string(pdlp::toString(result.status)));
    require(result.dualRay.size() == 2, "a Farkas ray should be returned");

    // Verify the returned ray independently of the detector that produced it:
    // inf over the box of (A'y)'x minus sup over the row bounds of y'z must be
    // strictly positive.
    const std::vector<double>& y = result.dualRay;
    const double reduced0 = y[0] + y[1];
    const double reduced1 = y[0] + y[1];
    const double boxInfimum =
        (reduced0 > 0.0 ? reduced0 * 0.0 : reduced0 * 10.0) +
        (reduced1 > 0.0 ? reduced1 * 0.0 : reduced1 * 10.0);
    const double support =
        (y[0] > 0.0 ? 0.0 : y[0] * 2.0) + (y[1] > 0.0 ? y[1] * 1.0 : 0.0);
    require(boxInfimum - support > 1e-9,
            "returned ray does not certify infeasibility: value " +
            std::to_string(boxInfimum - support));
}

// Two equalities on the same row vector with different right-hand sides.
void testDetectsInfeasibleEqualities() {
    pdlp::CompiledLp problem;
    problem.matrix = pdlp::SparseMatrix::fromTriplets(
        2, 2, {{0, 0, 1.0}, {0, 1, 1.0}, {1, 0, 1.0}, {1, 1, 1.0}});
    problem.objective = {0.0, 0.0};
    problem.rowLower = {1.0, 2.0};
    problem.rowUpper = {1.0, 2.0};
    problem.variableLower = {-kInfinity, -kInfinity};
    problem.variableUpper = {kInfinity, kInfinity};

    auto options = strictOptions();
    options.iterationLimit = 20000;
    const auto result = pdlp::PdlpSolver{}.solve(problem, options);
    require(result.status == pdlp::PdlpStatus::Infeasible,
            "inconsistent equalities should be proven infeasible, got " +
            std::string(pdlp::toString(result.status)));
}

// min -x with x unbounded above and no row constraining it from above.
void testDetectsUnbounded() {
    pdlp::CompiledLp problem;
    problem.matrix = pdlp::SparseMatrix::fromTriplets(1, 2, {{0, 1, 1.0}});
    problem.objective = {-1.0, 0.0};
    problem.rowLower = {-kInfinity};
    problem.rowUpper = {5.0};
    problem.variableLower = {0.0, 0.0};
    problem.variableUpper = {kInfinity, 10.0};

    auto options = strictOptions();
    options.iterationLimit = 20000;
    const auto result = pdlp::PdlpSolver{}.solve(problem, options);
    require(result.status == pdlp::PdlpStatus::Unbounded,
            "unbounded LP should be proven unbounded, got " +
            std::string(pdlp::toString(result.status)));
    require(result.primalRay.size() == 2, "an improving ray should be returned");
    require(result.primalRay[0] > 0.0,
            "the ray must grow the unbounded variable");
}

// The safety property that matters most: a feasible, bounded problem must never
// be reported infeasible or unbounded.
void testFeasibleProblemsAreNotFlagged() {
    std::mt19937 generator(17);
    std::uniform_real_distribution<double> value(-1.0, 1.0);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::uniform_int_distribution<int> pick(0, 599);

    for (int instance = 0; instance < 6; ++instance) {
        const int rows = 300 + 40 * instance;
        const int columns = 600;
        std::vector<pdlp::MatrixTriplet> triplets;
        for (int row = 0; row < rows; ++row) {
            for (int k = 0; k < 8; ++k) {
                triplets.push_back({row, pick(generator), value(generator)});
            }
        }

        pdlp::CompiledLp problem;
        problem.matrix = pdlp::SparseMatrix::fromTriplets(rows, columns, triplets);
        problem.objective.resize(columns);
        problem.variableLower.assign(columns, 0.0);
        problem.variableUpper.assign(columns, 4.0);

        std::vector<double> reference(columns);
        for (int j = 0; j < columns; ++j) {
            at(problem.objective, j) = value(generator);
            at(reference, j) = 4.0 * unit(generator);
        }
        std::vector<double> activity;
        problem.matrix.multiply(reference, activity);
        problem.rowLower.resize(static_cast<std::size_t>(rows));
        problem.rowUpper.resize(static_cast<std::size_t>(rows));
        for (int i = 0; i < rows; ++i) {
            at(problem.rowLower, i) = at(activity, i) - 0.5;
            at(problem.rowUpper, i) = at(activity, i) + 0.5;
        }

        auto options = strictOptions();
        options.iterationLimit = 6000;
        const auto result = pdlp::PdlpSolver{}.solve(problem, options);
        require(result.status != pdlp::PdlpStatus::Infeasible &&
                    result.status != pdlp::PdlpStatus::Unbounded,
                "feasible bounded instance was wrongly flagged as " +
                std::string(pdlp::toString(result.status)));
    }
}

// ---------------------------------------------------------------------------
// Compute backends
// ---------------------------------------------------------------------------

// The shared per-coordinate arithmetic must keep std::min/std::max semantics
// (the CUDA kernels compile the same functions): a NaN reaching the dual prox
// has to survive into the reductions, where the solver detects it.
void testPdhgMathNanAndInfinitySemantics() {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    require(std::isnan(pdlp::math::maxOf(nan, 0.0)), "maxOf(NaN, 0) must be NaN (std::max order)");
    require(std::isnan(pdlp::math::minOf(nan, 0.0)), "minOf(NaN, 0) must be NaN (std::min order)");
    require(pdlp::math::maxOf(1.0, nan) == 1.0, "maxOf(1, NaN) keeps std::max's answer");

    const pdlp::math::DualUpdate nanStep = pdlp::math::dualUpdate(nan, 1.0, 0.0, 1.0, 1.0, -1.0, 1.0);
    require(std::isnan(nanStep.value), "a NaN dual must propagate through the dual prox");

    // Infinite bounds: no inf - inf, no 0 * inf, and an inactive one-sided row
    // gives exactly zero.
    const pdlp::math::DualUpdate free = pdlp::math::dualUpdate(0.0, 2.0, 1.0, 1.0, 1.0, -kInfinity, kInfinity);
    require(free.value == 0.0 && free.deltaActivity == 1.0, "free row dual must be exactly 0");
    const pdlp::math::DualUpdate underflow = pdlp::math::dualUpdate(0.5, 2.0, 1.0, 0.0, 1.0, -kInfinity, 3.0);
    require(std::isfinite(underflow.value), "a zero step must not evaluate 0 * infinity");
    require(pdlp::math::primalUpdate(1.0, 1.0, 1.0, -5.0, 0.0, kInfinity) == 6.0, "unbounded above");
    require(pdlp::math::primalUpdate(1.0, 1.0, 1.0, 5.0, -kInfinity, kInfinity) == -4.0, "free variable");
    require(pdlp::math::primalUpdate(1.0, 1.0, 1.0, 5.0, 0.0, 10.0) == 0.0, "clipped at the lower bound");
    require(pdlp::math::blended(2.0, 4.0, 0.25) == 2.5, "average blend");
}

// The backend contract the CUDA backend is held to (backend_contract.h), run
// CPU against CPU so the harness itself is exercised on every build.
void testCpuBackendContract() {
    const contract::BackendFactory cpu = [](const pdlp::CompiledLp& problem,
                                            const pdlp::DiagonalPreconditioner& preconditioner) {
        return pdlp::makeCpuIterationBackend(problem, preconditioner);
    };
    for (const contract::Fixture& fixture : contract::kernelFixtures()) {
        contract::compareBackends(fixture, cpu, cpu, 0.0);
    }
}

pdlp::CompiledLp smallBoxedLp() {
    pdlp::CompiledLp problem;
    problem.matrix = pdlp::SparseMatrix::fromTriplets(
        2, 3, {{0, 0, 1.0}, {0, 1, 2.0}, {1, 1, 1.0}, {1, 2, 3.0}});
    problem.objective = {-1.0, -1.0, -2.0};
    problem.rowLower = {-kInfinity, 1.0};
    problem.rowUpper = {4.0, 6.0};
    problem.variableLower = {0.0, 0.0, 0.0};
    problem.variableUpper = {3.0, kInfinity, 2.0};
    return problem;
}

// An explicit CUDA request is honoured or refused -- never quietly run on the
// CPU. On a machine without a usable device (always, in a CPU-only build) the
// solve must fail and say why.
void testExplicitCudaRequestNeverFallsBack() {
    const pdlp::CudaAvailability availability = pdlp::cudaAvailability(0);
    if (availability.usable) {
        return;  // covered by pdlp_cuda_tests
    }
    require(!availability.reason.empty(), "an unusable CUDA backend must say why");
    auto options = strictOptions();
    options.backend = pdlp::ComputeBackend::Cuda;
    const auto result = pdlp::PdlpSolver{}.solve(smallBoxedLp(), options);
    require(result.status == pdlp::PdlpStatus::InvalidProblem,
            "CUDA requested without a usable device must not solve, got " +
            std::string(pdlp::toString(result.status)));
    require(result.statusMessage.find("CUDA backend requested but unavailable") != std::string::npos,
            "the failure must name the CUDA request: " + result.statusMessage);
    require(result.primal.empty(), "no solution may be reported for a refused backend");
}

void testAutoBackendResolution() {
    // Below the threshold, Auto is the CPU whatever the build.
    auto options = strictOptions();
    options.backend = pdlp::ComputeBackend::Auto;
    auto result = pdlp::PdlpSolver{}.solve(smallBoxedLp(), options);
    require(result.executedBackend == pdlp::ComputeBackend::Cpu, "small problem must run on the CPU");
    require(result.backendMessage.find("below cudaNonzeroThreshold") != std::string::npos,
            "Auto must say why it chose the CPU: " + result.backendMessage);
    require(result.backendProfile.hostToDeviceBytes == 0, "the CPU backend transfers nothing");

    // Threshold removed: Auto uses CUDA only if it is actually usable.
    options.cudaNonzeroThreshold = 0;
    result = pdlp::PdlpSolver{}.solve(smallBoxedLp(), options);
    require(result.status == pdlp::PdlpStatus::Optimal, "Auto must still solve");
    if (!pdlp::cudaAvailability(0).usable) {
        require(result.executedBackend == pdlp::ComputeBackend::Cpu, "no device: Auto must run on the CPU");
        require(result.backendMessage.rfind("auto -> cpu", 0) == 0,
                "Auto must record the fallback reason: " + result.backendMessage);
    }
}

// Forcing the CPU must be exactly the default solve, bit for bit, whenever the
// default resolves to the CPU.
void testForcedCpuMatchesDefault() {
    const pdlp::CompiledLp problem = smallBoxedLp();
    auto automatic = strictOptions();
    auto forced = strictOptions();
    forced.backend = pdlp::ComputeBackend::Cpu;
    const auto a = pdlp::PdlpSolver{}.solve(problem, automatic);
    const auto b = pdlp::PdlpSolver{}.solve(problem, forced);
    require(a.executedBackend == pdlp::ComputeBackend::Cpu, "default must resolve to the CPU here");
    require(a.status == b.status && a.iterations == b.iterations && a.stepTrials == b.stepTrials,
            "forced CPU must follow the identical iteration path");
    require(a.primal == b.primal && a.rowDual == b.rowDual, "forced CPU must return identical vectors");
    require(a.primalObjective == b.primalObjective, "forced CPU must return the identical objective");
    require(b.backendMessage == "cpu: requested", "forced CPU message: " + b.backendMessage);
}

void testInvalidCudaDeviceIsRejected() {
    auto options = strictOptions();
    options.cudaDevice = -1;
    const auto result = pdlp::PdlpSolver{}.solve(smallBoxedLp(), options);
    require(result.status == pdlp::PdlpStatus::InvalidProblem, "a negative CUDA device must be rejected");
}

void run(const char* name, void (*test)()) {
    try {
        test();
        std::cout << "  pass  " << name << '\n';
    } catch (const std::exception& error) {
        std::cout << "  FAIL  " << name << ": " << error.what() << '\n';
        ++failures;
    }
}

}  // namespace

int main() {
    run("duplicateTriplets", testDuplicateTriplets);
    run("cancellingDuplicatesAreDropped", testCancellingDuplicatesAreDropped);
    run("unsortedTripletsAndEmptyLines", testUnsortedTripletsAndEmptyLines);
    run("csrCscAgree", testCsrCscAgree);
    run("matrixScaling", testMatrixScaling);
    run("parallelProductMatchesSerial", testParallelProductMatchesSerial);
    run("executorCoversEveryPart", testExecutorCoversEveryPart);

    run("equalityRow", testEqualityRow);
    run("upperBoundRow", testUpperBoundRow);
    run("lowerBoundRow", testLowerBoundRow);
    run("rangedRow", testRangedRow);
    run("inactiveRowDualIsExactlyZero", testInactiveRowDualIsExactlyZero);
    run("illConditionedProblem", testIllConditionedProblem);
    run("ruizEquilibrationReducesSpread", testRuizEquilibrationReducesSpread);
    run("degenerateProblem", testDegenerateProblem);
    run("unboundedIsNotReportedOptimal", testUnboundedIsNotReportedOptimal);
    run("invalidProblemIsRejected", testInvalidProblemIsRejected);
    run("timeLimitIsHonoured", testTimeLimitIsHonoured);
    run("threadCountDoesNotChangeTheAnswer", testThreadCountDoesNotChangeTheAnswer);

    run("detectsInfeasibleRows", testDetectsInfeasibleRows);
    run("detectsInfeasibleEqualities", testDetectsInfeasibleEqualities);
    run("detectsUnbounded", testDetectsUnbounded);
    run("feasibleProblemsAreNotFlagged", testFeasibleProblemsAreNotFlagged);

    run("linesearchExceedsStaticBound", testLinesearchExceedsStaticBound);
    run("stepPoliciesAgreeOnTheOptimum", testStepPoliciesAgreeOnTheOptimum);

    run("pdhgMathNanAndInfinitySemantics", testPdhgMathNanAndInfinitySemantics);
    run("cpuBackendContract", testCpuBackendContract);
    run("explicitCudaRequestNeverFallsBack", testExplicitCudaRequestNeverFallsBack);
    run("autoBackendResolution", testAutoBackendResolution);
    run("forcedCpuMatchesDefault", testForcedCpuMatchesDefault);
    run("invalidCudaDeviceIsRejected", testInvalidCudaDeviceIsRejected);

    if (failures == 0) {
        std::cout << "All PDLP tests passed\n";
        return EXIT_SUCCESS;
    }
    std::cout << failures << " test(s) failed\n";
    return EXIT_FAILURE;
}
