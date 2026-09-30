// The QP engine's result contract as seen through the public pipeline.
//
// An engine status is not the end of the story: the orchestrator normalises it
// and postsolve validates the point. These tests pin the combined contract --
// a non-converged or numerically failed QP never reaches the caller as a valid
// solution, and a converged one arrives with validated primal and duals.
#include "solver/orchestrator.h"

#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
constexpr double inf = std::numeric_limits<double>::infinity();

void check(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(what);
}

model::Model boxedQp() {
    // min x0^2 + x1^2 + x2^2 + 0.5 x0 x1 - x0 - 2 x1 - 3 x2,  0 <= x <= 10,  x0 + x1 + x2 <= 4
    model::Model m;
    for (int j = 0; j < 3; ++j) {
        model::Variable v; v.name = "x" + std::to_string(j); v.lowerBound = 0; v.upperBound = 10;
        m.variables.push_back(v);
    }
    m.objective.linearTerms = {{0, -1}, {1, -2}, {2, -3}};
    m.objective.quadraticTerms = {{0, 0, 1}, {1, 1, 1}, {2, 2, 1}, {0, 1, 0.5}};
    model::Constraint c; c.name = "budget"; c.lowerBound = -inf; c.upperBound = 4;
    c.linearTerms = {{0, 1}, {1, 1}, {2, 1}};
    m.constraints = {c};
    return m;
}
}  // namespace

int main() {
    try {
        solver::SolverOptions forcedQp;
        forcedQp.forceEngine = solver::Engine::Qp;

        // A converged QP: Optimal, with validated primal and multipliers.
        {
            const auto r = solver::solve(boxedQp(), forcedQp);
            check(r.status == solver::SolveStatus::Optimal, std::string("reference QP: ") + solver::toString(r.status));
            check(r.executedEngine == solver::Engine::Qp, "reference QP did not run on the QP engine");
            check(r.hasPrimal && r.hasDuals, "converged QP missing validated primal or duals");
        }

        // Stopped by an exhausted time limit: never Optimal, never duals.
        {
            solver::SolverOptions limited = forcedQp;
            limited.timeLimitSeconds = 1e-9;
            const auto r = solver::solve(boxedQp(), limited);
            check(r.status == solver::SolveStatus::LimitReached,
                  std::string("time-limited QP reported ") + solver::toString(r.status));
            check(!r.hasDuals, "a non-converged QP published multipliers");
        }

        // A numerical breakdown stays a failure end to end. The iterates of
        // this finite problem overflow; the engine used to burn its whole budget
        // and report an iteration limit (and on a sibling case, a false
        // Unbounded). Now it is NumericalFailure, and no point is published.
        {
            model::Model m;
            model::Variable x; x.name = "x"; x.lowerBound = -inf; x.upperBound = inf;
            m.variables = {x};
            m.objective.linearTerms = {{0, 1.7e308}};
            m.objective.quadraticTerms = {{0, 0, 5e-301}};
            model::Constraint c; c.name = "r"; c.lowerBound = -1e300; c.upperBound = 1e300;
            c.linearTerms = {{0, 1.0}};
            m.constraints = {c};
            const auto r = solver::solve(m, forcedQp);
            std::printf("  overflowing QP: %s | %s\n", solver::toString(r.status), r.message.c_str());
            check(r.status == solver::SolveStatus::NumericalFailure,
                  std::string("numerical breakdown reported as ") + solver::toString(r.status));
            check(!r.hasPrimal && !r.hasDuals, "a failed solve published a solution");
        }

        std::printf("QP result contract tests passed\n");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAILED: %s\n", error.what());
        return 1;
    }
}
