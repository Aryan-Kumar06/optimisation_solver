#pragma once
#include "nlp/solver.h"
#include "solver/dispatcher.h"

namespace solver {
// Preserve local termination semantics while exposing common pipeline metadata.
struct NlpSolveResult : nlp::Result {
    Classification classification;
    Engine engine = Engine::Nlp;
    Engine executedEngine = Engine::Unsupported;
    std::string engineReason;
};

[[nodiscard]] Classification classify(const nlp::Problem& problem);
[[nodiscard]] DispatchDecision dispatch(const nlp::Problem& problem,
    std::optional<Engine> requestedEngine = {});

// Nonlinear models never enter affine presolve/postsolve. The reported point,
// multipliers and residuals remain in the supplied problem's coordinates.
[[nodiscard]] NlpSolveResult solve(const nlp::Problem& problem,
    const std::vector<double>& initial, const nlp::Options& options = {},
    std::optional<Engine> requestedEngine = {});
} // namespace solver
