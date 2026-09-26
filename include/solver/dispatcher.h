#pragma once

#include "model/model.h"
#include "presolve/presolve_result.h"
#include "solver/classifier.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace solver {

enum class Engine {
    // First-order PDHG. Large sparse LPs; returns no basis and no vertex.
    Pdlp,

    // Dual simplex. Exact vertex solutions, warm-startable from a basis.
    DualSimplex,

    // Branch-and-cut over the dual simplex. The only engine handling integrality.
    BranchAndCut,

    // ADMM engine for convex quadratic objectives. Continuous variables only.
    Qp,

    // Presolve proved infeasibility; no engine runs.
    Infeasible,

    // Presolve reduced the problem away entirely; the answer is already known.
    Trivial,

    // No engine in this repository can solve the problem as classified.
    Unsupported
};

[[nodiscard]] const char* toString(Engine value) noexcept;
[[nodiscard]] std::optional<Engine> parseEngine(std::string_view name) noexcept;

// Caller-facing knobs the dispatcher must respect.
struct SolverOptions {
    // Always honoured, ahead of every structural rule.
    std::optional<Engine> forceEngine;

    // A basis, a vertex, or duals usable for warm starting. PDLP provides none
    // of these, so this forces the simplex path regardless of size.
    bool requireVertexSolution = false;

    // Above either threshold the dual simplex's dense m x m basis inverse stops
    // being affordable and PDLP takes over. Defaults are deliberately
    // conservative and belong to the dispatcher, not to either engine.
    std::size_t dualSimplexMaxRows = 2000;
    std::int64_t dualSimplexMaxNonzeros = 500000;

    // Passed through to whichever engine runs. PDLP's duals in particular are
    // only as accurate as the tolerance it converged to, so a caller that needs
    // shadow prices should tighten this.
    double tolerance = 1e-8;
    double timeLimitSeconds = 0.0;

    // Worker threads for whichever engine runs. 0 selects hardware_concurrency,
    // 1 forces serial. Exposed because a benchmark whose thread count varies
    // with the host is not reproducible. PDLP and the QP engine still run
    // serially below their own nonzero thresholds regardless of this value, so
    // on small models it changes nothing.
    int threadCount = 0;
};

struct DispatchDecision {
    Engine engine = Engine::Pdlp;

    // Why this engine was chosen, in words. Logged, and shown to the user when
    // a solve goes badly -- "which engine ran and why" is the first question.
    std::string reason;
};

// Chooses an engine for the REDUCED model.
//
// This runs AFTER presolve, and the distinction matters:
//   * presolve can fix or bound-tighten away every integer variable, so a model
//     that arrives as a MILP can leave as a pure LP -- dispatching earlier would
//     start branch-and-cut on a tree with one node;
//   * presolve routinely changes row and nonzero counts by tens of percent, and
//     the PDLP/simplex crossover is a function of exactly those numbers;
//   * presolve can prove infeasibility outright, in which case no engine runs.
//
// `classification` is from classify() on the ORIGINAL model: integrality that
// presolve eliminated is not relevant to the engine choice, but the structural
// hints still describe the problem the user posed.
[[nodiscard]] DispatchDecision dispatch(
    const model::Model& reduced,
    const Classification& classification,
    const presolve::PresolveResult& presolveResult,
    const SolverOptions& options = {}
);

}  // namespace solver
