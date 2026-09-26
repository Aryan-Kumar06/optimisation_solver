#include "solver/dispatcher.h"

#include <cmath>
#include <cstddef>

namespace solver {
namespace {

constexpr double kEps = 1e-9;

std::int64_t countNonzeros(const model::Model& model) {
    std::int64_t total = 0;
    for (const auto& constraint : model.constraints) {
        for (const auto& term : constraint.linearTerms) {
            if (std::abs(term.value) > kEps) {
                ++total;
            }
        }
    }
    return total;
}

bool hasIntegrality(const model::Model& model) {
    for (const auto& variable : model.variables) {
        if (variable.type != model::VariableType::Continuous) {
            return true;
        }
    }
    return false;
}

}  // namespace

const char* toString(Engine value) noexcept {
    switch (value) {
        case Engine::Pdlp:         return "pdlp";
        case Engine::DualSimplex:  return "dual_simplex";
        case Engine::BranchAndCut: return "branch_and_cut";
        case Engine::Qp:           return "qp";
        case Engine::Infeasible:   return "infeasible";
        case Engine::Trivial:      return "trivial";
        case Engine::Unsupported:  return "unsupported";
    }
    return "unknown";
}

std::optional<Engine> parseEngine(std::string_view name) noexcept {
    if (name == "pdlp") {
        return Engine::Pdlp;
    }
    if (name == "dual_simplex") {
        return Engine::DualSimplex;
    }
    if (name == "branch_and_cut") {
        return Engine::BranchAndCut;
    }
    if (name == "qp") {
        return Engine::Qp;
    }
    return std::nullopt;
}

DispatchDecision dispatch(
    const model::Model& reduced,
    const Classification& classification,
    const presolve::PresolveResult& presolveResult,
    const SolverOptions& options
) {
    DispatchDecision decision;

    // Presolve's verdict outranks everything: there is nothing left to solve.
    if (presolveResult.infeasible) {
        decision.engine = Engine::Infeasible;
        decision.reason = "presolve proved the model infeasible";
        return decision;
    }

    // An explicit request is honoured ahead of every structural rule, including
    // the integrality rule -- forcing an LP engine onto a model with integer
    // variables solves the relaxation, which is a legitimate thing to ask for.
    if (options.forceEngine.has_value()) {
        decision.engine = *options.forceEngine;
        decision.reason = "engine forced by the caller";
        return decision;
    }

    if (reduced.variables.empty()) {
        decision.engine = Engine::Trivial;
        decision.reason = "presolve eliminated every variable";
        return decision;
    }

    // A quadratic objective goes to the ADMM engine, but only without
    // integrality: nothing here does branch-and-cut over a quadratic
    // relaxation. Saying so plainly beats dropping the quadratic terms and
    // returning a confident answer to a different problem.
    const bool quadratic =
        classification.problemClass == ProblemClass::QP ||
        classification.problemClass == ProblemClass::MIQP ||
        classification.problemClass == ProblemClass::QCQP;

    if (quadratic) {
        if (classification.problemClass == ProblemClass::QCQP) {
            decision.engine = Engine::Unsupported;
            decision.reason =
                "quadratic constraints are not representable in model::Model";
            return decision;
        }
        if (hasIntegrality(reduced)) {
            decision.engine = Engine::Unsupported;
            decision.reason =
                "MIQP: no engine does branch-and-cut over a quadratic relaxation";
            return decision;
        }
        decision.engine = Engine::Qp;
        decision.reason = "convex quadratic objective with continuous variables";
        return decision;
    }

    // Integrality decides before size does: branch-and-cut is the only engine
    // that enforces it. Note this tests the REDUCED model -- presolve may have
    // fixed every integer variable, in which case this is now a pure LP.
    if (hasIntegrality(reduced)) {
        decision.engine = Engine::BranchAndCut;
        decision.reason = "integer variables survive presolve";
        return decision;
    }

    if (options.requireVertexSolution) {
        // The dual simplex is the only vertex-capable engine here, and it
        // maintains a DENSE m x m basis inverse -- 8*m^2 bytes, allocated up
        // front, before a single pivot. That is 32 MB at the 2000-row
        // threshold but 20 GB at 50000 rows, so honouring this option at any
        // size would mean attempting an allocation that cannot succeed.
        //
        // Checking the same size limits used for automatic routing keeps the
        // option honest: it is satisfied where it can be, and refused with a
        // reason where no engine can satisfy it, rather than silently handing
        // back PDLP's non-vertex iterate (which is what this did before) or
        // grinding against a budget it cannot honour.
        const std::size_t vertexRows = reduced.constraints.size();
        const std::int64_t vertexNonzeros = countNonzeros(reduced);
        if (vertexRows >= options.dualSimplexMaxRows ||
            vertexNonzeros > options.dualSimplexMaxNonzeros) {
            decision.engine = Engine::Unsupported;
            decision.reason =
                "a vertex solution was requested, but the only vertex-capable "
                "engine (dual simplex, dense basis inverse) is limited to " +
                std::to_string(options.dualSimplexMaxRows) + " rows and " +
                std::to_string(options.dualSimplexMaxNonzeros) +
                " nonzeros; this model has " + std::to_string(vertexRows) +
                " rows and " + std::to_string(vertexNonzeros) + " nonzeros";
            return decision;
        }
        decision.engine = Engine::DualSimplex;
        decision.reason =
            "a basis or vertex solution was requested; the dual simplex "
            "terminates at one and PDLP does not";
        return decision;
    }

    if (reduced.constraints.empty()) {
        decision.engine = Engine::Trivial;
        decision.reason =
            "no constraints remain; the optimum follows from the bounds alone";
        return decision;
    }

    const std::size_t rows = reduced.constraints.size();
    const std::int64_t nonzeros = countNonzeros(reduced);

    if (rows < options.dualSimplexMaxRows &&
        nonzeros <= options.dualSimplexMaxNonzeros) {
        decision.engine = Engine::DualSimplex;
        decision.reason = "small enough for the dual simplex (" +
            std::to_string(rows) + " rows, " + std::to_string(nonzeros) +
            " nonzeros); it terminates at an exact vertex";
        return decision;
    }

    decision.engine = Engine::Pdlp;
    decision.reason = "too large for a dense basis inverse (" +
        std::to_string(rows) + " rows, " + std::to_string(nonzeros) +
        " nonzeros); a first-order method avoids factorization entirely";
    return decision;
}

}  // namespace solver
