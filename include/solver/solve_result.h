#pragma once

#include "solver/dispatcher.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace solver {

// One status for every engine.
//
// PDLP reports PdlpStatus, branch-and-cut reports MilpStatus, and the two
// disagree on both spelling and meaning -- MilpStatus::NodeLimit and
// PdlpStatus::IterationLimit are the same idea under different names, and only
// one of the enums has Unbounded. Postsolve should not have to know which
// engine ran, so both are normalised here.
enum class SolveStatus {
    Optimal,
    Infeasible,
    Unbounded,

    // Ran out of budget with no proof either way. `variableValues` holds the
    // best point found, if any.
    LimitReached,

    NumericalFailure,
    InvalidModel,

    // No engine in this build can solve the problem as classified.
    Unsupported
};

[[nodiscard]] const char* toString(SolveStatus value) noexcept;

// One result contract for every engine. Solution vectors use the indices, units
// and objective sense of the model passed to the API: ORIGINAL coordinates for
// solve(), and the supplied reduced model's coordinates for solveReduced().
// reducedVariableCount/reducedConstraintCount describe presolve statistics,
// not the dimensions of the reconstructed vectors returned by solve().
struct SolveResult {
    SolveStatus status = SolveStatus::InvalidModel;
    std::string message;

    // Which engine the DISPATCHER chose, and why. Kept because "which engine
    // ran" is the first question asked when a solve looks wrong.
    Engine engine = Engine::Unsupported;
    std::string engineReason;

    // Which engine ACTUALLY ran, recorded by the code path that invokes it
    // rather than by the dispatcher that selected it.
    //
    // These two are separate fields on purpose. They disagreed once already:
    // the dispatcher named Engine::DualSimplex for requireVertexSolution
    // while the orchestrator's switch routed that case into PDLP, so the
    // reported engine was a decision nobody had executed and the vertex
    // promise was silently broken. A decision label cannot detect that; only
    // a value written by the path that did the work can.
    //
    // Engine::Unsupported here means no engine was invoked at all -- the
    // solve was answered by presolve (Infeasible) or refused before dispatch.
    // A bound-walk result is recorded as Engine::Trivial.
    Engine executedEngine = Engine::Unsupported;

    // True when a complete, finite point passes primal postsolve validation.
    // A valid zero-variable solution is empty with hasPrimal=true. An explicitly
    // forced continuous relaxation can still have integralityRespected=false.
    bool hasPrimal = false;

    // Length of the input model's variable list when hasPrimal=true.
    std::vector<double> variableValues;

    // Shadow prices, d(objective)/d(right-hand side), length
    // model.constraints.size(). Published only after optimality validation.
    // Integer models and non-optimal results have no sensitivity multipliers.
    // Check hasDuals: a valid zero-constraint dual vector is empty.
    std::vector<double> constraintDuals;
    bool hasDuals = false;
    // grad f(x) - A^T y, in the same input-model variable order.
    std::vector<double> reducedCosts;
    std::string dualsUnavailableReason = "No dual solution is available.";
    double maxDualResidual = 0.0;

    double objectiveValue = 0.0;

    // For a model with integer variables, how far the returned point is from
    // integral. Zero for a pure LP. A non-zero value on a MILP means the caller
    // was handed a relaxation, not a solution.
    double maxIntegralityViolation = 0.0;
    bool integralityRespected = true;

    // Work done. Whichever is meaningful for the engine that ran; the other
    // stays zero.
    std::int64_t iterations = 0;
    std::int64_t nodeCount = 0;
    double solveSeconds = 0.0;

    // Dimensions solved by the engine, retained as reduction statistics.
    std::size_t reducedVariableCount = 0;
    std::size_t reducedConstraintCount = 0;
};

}  // namespace solver
