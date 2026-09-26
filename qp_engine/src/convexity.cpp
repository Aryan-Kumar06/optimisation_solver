#include "qp/convexity.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <sstream>
#include <vector>

namespace qp {

ConvexityCheck checkConvexity(const model::Model& model, double relativeTolerance) {
    ConvexityCheck out;
    const std::size_t n = model.variables.size();
    if (model.objective.quadraticTerms.empty() || n == 0) {
        out.convexForObjectiveSense = true;
        out.reason = "objective has no quadratic curvature";
        return out;
    }

    // Build the actual Hessian H of the Model convention
    //   f(x) = offset + c'x + sum q_ij x_i x_j.
    // Therefore diagonal q_ii contributes 2*q_ii to H_ii and an off-diagonal
    // q_ij contributes q_ij to H_ij and H_ji. For maximization, validate -H
    // because the QP engine minimizes the sign-negated objective.
    std::vector<std::vector<double>> a(n, std::vector<double>(n, 0.0));
    const double senseSign =
        model.objective.sense == model::ObjectiveSense::Maximize ? -1.0 : 1.0;
    double scale = 1.0;

    for (const auto& term : model.objective.quadraticTerms) {
        if (term.variableIndex1 < 0 || term.variableIndex2 < 0 ||
            static_cast<std::size_t>(term.variableIndex1) >= n ||
            static_cast<std::size_t>(term.variableIndex2) >= n ||
            !std::isfinite(term.value)) {
            out.reason = "quadratic objective contains an invalid term";
            return out;
        }
        const std::size_t i = static_cast<std::size_t>(term.variableIndex1);
        const std::size_t j = static_cast<std::size_t>(term.variableIndex2);
        if (i == j) {
            a[i][i] += senseSign * 2.0 * term.value;
            scale = std::max(scale, std::abs(2.0 * term.value));
        } else {
            a[i][j] += senseSign * term.value;
            a[j][i] += senseSign * term.value;
            scale = std::max(scale, std::abs(term.value));
        }
    }

    // Semidefinite LDL^T test. A PSD matrix admits a factorisation with
    // non-negative D. If a pivot is numerically zero, its remaining column
    // must also be zero; otherwise the corresponding 2x2 principal minor is
    // negative and the matrix is indefinite.
    const double tol = std::max(1e-14, relativeTolerance * scale);
    std::vector<std::vector<double>> l(n, std::vector<double>(n, 0.0));
    std::vector<double> d(n, 0.0);
    out.mostNegativePivot = 0.0;

    for (std::size_t k = 0; k < n; ++k) {
        double pivot = a[k][k];
        for (std::size_t s = 0; s < k; ++s) {
            pivot -= l[k][s] * l[k][s] * d[s];
        }
        out.mostNegativePivot = std::min(out.mostNegativePivot, pivot);
        if (pivot < -tol) {
            std::ostringstream message;
            message << "quadratic objective is non-convex for the requested sense: "
                    << "negative LDL pivot " << pivot << " at index " << k;
            out.reason = message.str();
            return out;
        }

        if (std::abs(pivot) <= tol) {
            d[k] = 0.0;
            for (std::size_t i = k + 1; i < n; ++i) {
                double residual = a[i][k];
                for (std::size_t s = 0; s < k; ++s) {
                    residual -= l[i][s] * l[k][s] * d[s];
                }
                if (std::abs(residual) > tol) {
                    out.reason =
                        "quadratic objective is indefinite: a zero curvature "
                        "pivot has non-zero coupling";
                    return out;
                }
                l[i][k] = 0.0;
            }
            continue;
        }

        d[k] = pivot;
        l[k][k] = 1.0;
        for (std::size_t i = k + 1; i < n; ++i) {
            double value = a[i][k];
            for (std::size_t s = 0; s < k; ++s) {
                value -= l[i][s] * l[k][s] * d[s];
            }
            l[i][k] = value / d[k];
        }
    }

    out.convexForObjectiveSense = true;
    out.reason = model.objective.sense == model::ObjectiveSense::Maximize
        ? "concave quadratic maximization (negated Hessian is PSD)"
        : "convex quadratic minimization (Hessian is PSD)";
    return out;
}

}  // namespace qp
