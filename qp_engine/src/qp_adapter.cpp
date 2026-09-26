#include "qp/qp_adapter.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace qp {

QpModel fromModel(const model::Model& model) {
    QpTranslation unused;
    return fromModel(model, unused);
}

QpModel fromModel(const model::Model& model, QpTranslation& translation) {
    const int n = static_cast<int>(model.variables.size());
    const int m = static_cast<int>(model.constraints.size());

    const bool maximise = (model.objective.sense == model::ObjectiveSense::Maximize);
    translation.objectiveNegated = maximise;
    translation.objectiveOffset = model.objective.offset;

    QpModel qp;
    qp.q.assign(static_cast<std::size_t>(n), 0.0);

    // Build P (Hessian) and q (linear objective).
    std::vector<double> pRows;
    std::vector<double> pCols;
    std::vector<double> pVals;
    pRows.reserve(model.objective.quadraticTerms.size() + n);
    pCols.reserve(model.objective.quadraticTerms.size() + n);
    pVals.reserve(model.objective.quadraticTerms.size() + n);

    // P is passed through exactly as the user gave it. It is NOT regularised
    // here.
    //
    // This used to add 1e-12 to every diagonal entry so the ADMM x-step's
    // factorisation would stay definite for an LP (P = 0). That made the engine
    // solve a different problem than the one it was handed, and nothing
    // downstream subtracted the perturbation back off:
    //
    //   * the reported objective was wrong by 0.5*eps*||x||^2 -- 5e-8 relative
    //     at ||x|| = 1e5, past the 1e-8 tolerance, and 50.0 absolute at 1e7;
    //   * unbounded LPs came back Optimal, because min -x + 0.5*eps*x^2 has the
    //     finite minimiser x = 1/eps. "min -x, x >= 0" returned x = 1e12.
    //
    // The factorisation still needs its regularisation -- it now lives in the
    // KKT system as sigma*I, where the x-update's matching +sigma*x_prev makes
    // it cancel at the fixed point. See KktSolver::kSigma.
    // max f(x) = max(0.5 x'Px + q'x) is solved as min(-f(x)) = min(0.5 x'(-P)x
    // + (-q)'x): P and q take the SAME sign flip, or the ADMM engine minimises
    // a saddle of the intended surface rather than its negation. sign is that
    // one shared flip, applied to both.
    const double sign = maximise ? -1.0 : 1.0;

    for (const auto& qt : model.objective.quadraticTerms) {
        const int i1 = qt.variableIndex1;
        const int i2 = qt.variableIndex2;
        if (i1 < 0 || i1 >= n || i2 < 0 || i2 >= n) {
            throw std::invalid_argument(
                "qp::fromModel: quadratic term index out of range");
        }
        // Symmetrize: only one (i1, i2) and (i2, i1) gets half the value.
        // We accumulate P[i1,i2] += qt.value and P[i2,i1] += qt.value.
        // For upper-triangle storage (P[i, j] for i <= j), we add to both
        // (min, max) entries.
        const int lo = std::min(i1, i2);
        const int hi = std::max(i1, i2);
        // Add full value to (lo, hi) entry.  SparseMatrix symmetrises on
        // multiplication, so storing once is enough if we pass symmetric
        // structure via "value" applied at (lo, hi) and (hi, lo).
        // The simplest correct approach: add the value to BOTH (lo,hi) and
        // (hi,lo).  This means P = full symmetric Hessian.
        // For the KKT step (which uses P + rho*A^T*A), this is correct.
        pRows.push_back(static_cast<double>(lo));
        pCols.push_back(static_cast<double>(hi));
        pVals.push_back(sign * qt.value);
        pRows.push_back(static_cast<double>(hi));
        pCols.push_back(static_cast<double>(lo));
        pVals.push_back(sign * qt.value);
    }

    qp.P = SparseMatrix::fromTriplets(n, n, pRows, pCols, pVals);

    // Build q (linear objective). Same sign as P, above.
    for (const auto& lt : model.objective.linearTerms) {
        if (lt.variableIndex < 0 || lt.variableIndex >= n) {
            throw std::invalid_argument(
                "qp::fromModel: linear term index out of range");
        }
        qp.q[static_cast<std::size_t>(lt.variableIndex)] += sign * lt.value;
    }

    // Build A, l, u.
    //
    // QpModel expresses EVERY restriction as a row of `l <= A x <= u`; it has no
    // separate variable-bound vectors. So each variable with a finite bound
    // needs its own identity row. Omitting them -- as an earlier version did --
    // silently solved a relaxation: the bounds in model::Model were read, then
    // dropped, and the engine reported Optimal for a point that could violate
    // every one of them.
    std::vector<int> boundedVariables;
    boundedVariables.reserve(static_cast<std::size_t>(n));
    for (int j = 0; j < n; ++j) {
        const auto& variable = model.variables[static_cast<std::size_t>(j)];
        if (std::isfinite(variable.lowerBound) || std::isfinite(variable.upperBound)) {
            boundedVariables.push_back(j);
        }
    }

    const int boundRows = static_cast<int>(boundedVariables.size());
    const int totalRows = m + boundRows;

    qp.l.assign(static_cast<std::size_t>(totalRows), 0.0);
    qp.u.assign(static_cast<std::size_t>(totalRows), 0.0);

    std::vector<double> aRows;
    std::vector<double> aCols;
    std::vector<double> aVals;
    std::size_t reserve = static_cast<std::size_t>(boundRows);
    for (const auto& c : model.constraints) reserve += c.linearTerms.size();
    aRows.reserve(reserve);
    aCols.reserve(reserve);
    aVals.reserve(reserve);

    // Constraint rows first, so row i of the model is row i here. Negating the
    // objective for a maximisation does not touch the constraints, so the bounds
    // copy across unchanged.
    for (int i = 0; i < m; ++i) {
        const auto& constraint = model.constraints[static_cast<std::size_t>(i)];
        qp.l[static_cast<std::size_t>(i)] = constraint.lowerBound;
        qp.u[static_cast<std::size_t>(i)] = constraint.upperBound;
        for (const auto& lt : constraint.linearTerms) {
            if (lt.variableIndex < 0 || lt.variableIndex >= n) {
                throw std::invalid_argument(
                    "qp::fromModel: constraint term index out of range");
            }
            aRows.push_back(static_cast<double>(i));
            aCols.push_back(static_cast<double>(lt.variableIndex));
            aVals.push_back(lt.value);
        }
    }

    // Then one identity row per bounded variable. An infinite side stays
    // infinite: QpModel tests bounds with std::isfinite, so a sentinel would be
    // read as a real bound.
    for (int b = 0; b < boundRows; ++b) {
        const int j = boundedVariables[static_cast<std::size_t>(b)];
        const auto& variable = model.variables[static_cast<std::size_t>(j)];
        const int row = m + b;
        qp.l[static_cast<std::size_t>(row)] = variable.lowerBound;
        qp.u[static_cast<std::size_t>(row)] = variable.upperBound;
        aRows.push_back(static_cast<double>(row));
        aCols.push_back(static_cast<double>(j));
        aVals.push_back(1.0);
    }

    qp.A = SparseMatrix::fromTriplets(totalRows, n, aRows, aCols, aVals);

    return qp;
}


void toModelDuals(const QpTranslation& translation, std::vector<double>& duals) {
    // See the header for why this is a negation and not a copy, and why the
    // maximisation case is the opposite sign rather than the same one.
    const double sign = translation.objectiveNegated ? 1.0 : -1.0;
    for (double& value : duals) {
        value *= sign;
    }
}

}  // namespace qp
