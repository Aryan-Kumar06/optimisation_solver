#include "qp/qp_solver.h"

#include "qp/admm_solver.h"
#include "qp/polishing.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace qp {
namespace {

// Worst violation of l <= A*x <= u.
//
// Two bugs used to live here, both letting a corrupted (inf/nan) x pass as
// "no violation". First: gating each bound's check on std::isfinite(bound)
// means a row with one infinite side (the common one-sided case, e.g.
// x >= 0) never checks Ax[i] against that side at all -- so if Ax[i] itself
// went to +inf on the unchecked side, nothing here noticed. Second:
// std::max(worst, v) with v == NaN returns worst unchanged (NaN loses every
// comparison, including the one std::max makes internally), so a NaN
// violation was silently dropped from the "worst" accumulator instead of
// dominating it. Together these let KktPolisher's corrupted output (see its
// own history) score as a perfect, zero-violation "improvement".
double primalViolation(const QpModel& model, const std::vector<double>& x) {
    for (double v : x) {
        if (!std::isfinite(v)) return std::numeric_limits<double>::infinity();
    }
    if (!model.hasConstraints()) {
        return 0.0;
    }
    std::vector<double> Ax;
    model.A.multiply(x, Ax);
    double worst = 0.0;
    for (std::size_t i = 0; i < Ax.size(); ++i) {
        if (!std::isfinite(Ax[i])) return std::numeric_limits<double>::infinity();
        if (std::isfinite(model.u[i]) && Ax[i] - model.u[i] > worst) worst = Ax[i] - model.u[i];
        if (std::isfinite(model.l[i]) && model.l[i] - Ax[i] > worst) worst = model.l[i] - Ax[i];
    }
    return worst;
}

// || P*x + q + A^T*y ||_inf, the stationarity residual. Same two hazards as
// primalViolation above, guarded the same way.
double dualViolation(const QpModel& model, const std::vector<double>& x,
                     const std::vector<double>& y) {
    for (double v : x) if (!std::isfinite(v)) return std::numeric_limits<double>::infinity();
    for (double v : y) if (!std::isfinite(v)) return std::numeric_limits<double>::infinity();

    std::vector<double> g;
    model.P.multiply(x, g);
    for (std::size_t j = 0; j < g.size(); ++j) {
        g[j] += model.q[j];
    }
    if (model.hasConstraints() && y.size() == static_cast<std::size_t>(model.numConstraints())) {
        std::vector<double> Aty;
        model.A.transposeMultiply(y, Aty);
        for (std::size_t j = 0; j < g.size(); ++j) {
            g[j] += Aty[j];
        }
    }
    double worst = 0.0;
    for (double v : g) {
        if (!std::isfinite(v)) return std::numeric_limits<double>::infinity();
        if (std::abs(v) > worst) worst = std::abs(v);
    }
    return worst;
}

double objectiveAt(const QpModel& model, const std::vector<double>& x) {
    std::vector<double> Px;
    model.P.multiply(x, Px);
    double value = 0.0;
    for (std::size_t j = 0; j < x.size(); ++j) {
        value += 0.5 * x[j] * Px[j] + model.q[j] * x[j];
    }
    return value;
}

}  // namespace

AdmmResult QpSolver::solve(const QpModel& problem, const AdmmOptions& options) const {
    // Validate the model first; reject obviously bad problems immediately.
    try {
        problem.validate();
    } catch (const std::exception& e) {
        AdmmResult r;
        r.status = QpStatus::InvalidProblem;
        r.statusMessage = e.what();
        return r;
    }

    AdmmResult result = AdmmSolver(problem, options).solve();

    if (result.status == QpStatus::Optimal && options.usePolishing) {
        // Active-set KKT polishing solves the equality-constrained system implied
        // by the constraints it believes are active. When that guess is wrong the
        // refined point violates the constraints it assumed inactive, so it is
        // accepted only if it is measurably better on BOTH residuals.
        //
        // An earlier version applied polish() -- which rewrites primal and dual in
        // place -- and then neither validated the result nor recomputed anything.
        // The reported objective and residuals therefore described the point
        // BEFORE polishing while `primal` held the point after it, so the engine
        // could return an infeasible vector alongside a zero primal residual and
        // a status of Optimal.
        const std::vector<double> primalBefore = result.primal;
        const std::vector<double> dualBefore = result.constraintDual;
        const double primalResidualBefore = primalViolation(problem, primalBefore);
        const double dualResidualBefore = dualViolation(problem, primalBefore, dualBefore);

        KktPolisher::Options pol;
        pol.maxIterations = options.polishingIterations;
        double remaining = 0.0;
        if (options.timeLimitSeconds > 0.0) {
            remaining = options.timeLimitSeconds - result.solveTimeSeconds;
            pol.timeLimitSeconds = std::max(remaining, 1e-9);
        }
        if (options.timeLimitSeconds > 0.0 && remaining <= 0.0) {
            result.statusMessage += " (polishing skipped: no time left)";
        } else if (KktPolisher::polish(problem, result.primal, result.constraintDual, pol)) {
            const double primalResidualAfter = primalViolation(problem, result.primal);
            const double dualResidualAfter =
                dualViolation(problem, result.primal, result.constraintDual);

            const bool improved =
                primalResidualAfter <= std::max(primalResidualBefore, options.primalTolerance) &&
                dualResidualAfter <= dualResidualBefore;

            if (improved) {
                result.primalObjective = objectiveAt(problem, result.primal);
                result.primalResidual = primalResidualAfter;
                result.dualResidual = dualResidualAfter;
            } else {
                result.primal = primalBefore;
                result.constraintDual = dualBefore;
                result.primalResidual = primalResidualBefore;
                result.dualResidual = dualResidualBefore;
                result.statusMessage += " (polishing rejected: no improvement)";
            }
        } else {
            // polish() leaves primal and dual untouched when it returns false.
            result.statusMessage += " (polishing not applied)";
        }
    }

    // Final gate. Optimal is a claim about the returned vectors, so it is
    // re-checked on exactly those vectors, in the caller's units, after any
    // polishing -- with the same test the ADMM loop terminates on. A result
    // that fails it is a numerical failure, never an optimum.
    if (result.status == QpStatus::Optimal) {
        // Same defaulting as AdmmSolver's constructor, so the gate and the
        // loop always apply the identical test.
        const double primalTolerance = options.primalTolerance > 0.0 ? options.primalTolerance : 1e-6;
        const double dualTolerance = options.dualTolerance > 0.0 ? options.dualTolerance : 1e-6;
        const KktCheck check = checkKkt(problem, result.primal, result.constraintDual,
                                        primalTolerance, dualTolerance);
        if (!check.met()) {
            result.status = QpStatus::NumericalFailure;
            result.statusMessage = "converged iterate fails the original-units optimality check "
                                   "(primal violation " + std::to_string(check.primalViolation) +
                                   ", dual residual " + std::to_string(check.dualResidual) + ")";
        }
    }
    return result;
}

}  // namespace qp
