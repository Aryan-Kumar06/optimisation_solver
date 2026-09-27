#pragma once
#include "nlp/model.h"
#include <functional>

namespace nlp {
enum class Status { FirstOrderStationary, IterationLimit, TimeLimit, UserStopped,
    InvalidProblem, EvaluationFailure, SubproblemFailure, NoProgress, NumericalFailure };
const char* toString(Status status) noexcept;
struct Iteration {
    int iteration = 0;
    double objective = 0, primalResidual = infinity, dualResidual = infinity;
    double complementarity = infinity, penalty = 0, stepLength = 0;
};
struct Options {
    int iterationLimit = 500, qpIterationLimit = 20000, lineSearchLimit = 40;
    double tolerance = 1e-6, timeLimitSeconds = 0;
    double initialPenalty = 10, maximumPenalty = 1e8;
    double regularization = 1e-8;
    // Dense damped BFGS only below this dimension; diagonal spectral curvature
    // above it preserves sparse QPs. Explicit bounded memory, no dense Jacobian.
    int denseBfgsLimit = 256;
    // Positive original-coordinate units; empty means all ones.
    std::vector<double> variableScale;
    bool scaleConstraints = true;
    // Called at each evaluated iterate; false stops. Exceptions become failures.
    std::function<bool(const Iteration&)> callback;
};
struct Result : Iteration {
    Status status = Status::InvalidProblem;
    std::string message;
    std::vector<double> primal, constraintMultipliers, boundMultipliers;
    // Multipliers use grad f + J^T lambda + z = 0. Upper positive, lower
    // negative. These are KKT multipliers, NOT the LP API's shadow prices.
    bool hasPrimal = false, feasible = false;
    int evaluations = 0, rejectedTrials = 0, elasticSubproblems = 0;
    long long qpIterations = 0;
    double solveSeconds = 0;
};
class Solver {
public:
    Result solve(const Problem& problem, const std::vector<double>& initial,
                 const Options& options = {}) const;
};
} // namespace nlp
