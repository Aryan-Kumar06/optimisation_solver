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
    // Dense damped BFGS is used only while n <= denseBfgsLimit; it stores the
    // full n x n approximation, so O(n^2) memory. ABOVE this threshold there is
    // no BFGS at all: curvature becomes a single scalar spectral diagonal with
    // NO off-diagonal information. That is strictly weaker, and on strongly
    // coupled nonlinear problems convergence can be substantially slower or hit
    // the iteration limit. This is NOT a limited-memory BFGS: no (s,y) history
    // is kept. Changing this value changes the algorithm, not just a budget.
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
// Local first-order elastic SQP. Curvature is a BFGS secant approximation built
// from FIRST derivatives only: there is no exact Hessian callback, no
// Hessian-vector product callback, and no second-order optimality certificate.
// FirstOrderStationary therefore does not imply a local minimum and does not
// imply a global minimum; a stationary saddle point or maximum is reported with
// the same status. It means only that the returned point satisfies the
// implemented original-unit KKT residual checks: it does not imply LICQ, MFCQ or
// any other constraint qualification, and the multipliers are not guaranteed to
// be unique. No status is an infeasibility certificate.
//
// `initial` is PROJECTED onto the variable bounds before the first evaluation.
// Nonlinear constraint violations are not repaired at initialization, and no
// replacement start is invented. If the projected point lies outside the
// evaluation domain the result is EvaluationFailure, even when the supplied
// point was inside it.
class Solver {
public:
    Result solve(const Problem& problem, const std::vector<double>& initial,
                 const Options& options = {}) const;
};
} // namespace nlp
