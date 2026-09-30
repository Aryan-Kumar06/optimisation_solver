#pragma once

// Independent original-model checks. Do not trust the solver's reported
// residuals or use the other backend's answer as the correctness oracle.
#include "qp/admm_solver.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace qpcheck {
inline void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
inline void optimal(const qp::QpModel& model, const qp::AdmmResult& result,
                    double tolerance = 1e-5) {
    require(result.status == qp::QpStatus::Optimal, "expected an optimal result");
    const auto& x = result.primal;
    const auto& y = result.constraintDual;
    require(x.size() == model.q.size() && y.size() == model.l.size(), "solution dimension mismatch");
    for (double v : x) require(std::isfinite(v), "nonfinite primal");
    for (double v : y) require(std::isfinite(v), "nonfinite multiplier");
    std::vector<double> gradient = model.q;
    std::vector<double> scale(x.size(), 1.0);
    double objective = 0;
    for (int i = 0; i < model.numVariables(); ++i) {
        double px = 0;
        for (auto k = model.P.csrRowStart()[i]; k < model.P.csrRowStart()[i+1]; ++k) {
            const double term = model.P.csrValues()[k] * x[model.P.csrColumnIndex()[k]];
            px += term;
            scale[i] += std::abs(term);
        }
        gradient[i] += px;
        scale[i] += std::abs(model.q[i]);
        objective += x[i] * (0.5 * px + model.q[i]);
    }
    for (int i = 0; i < model.numConstraints(); ++i) {
        double ax = 0, rowScale = 1;
        for (auto k = model.A.csrRowStart()[i]; k < model.A.csrRowStart()[i+1]; ++k) {
            const int j = model.A.csrColumnIndex()[k];
            const double a = model.A.csrValues()[k];
            ax += a * x[j];
            rowScale += std::abs(a * x[j]);
            gradient[j] += a * y[i];
            scale[j] += std::abs(a * y[i]);
        }
        if (std::isfinite(model.l[i])) rowScale = std::max(rowScale, std::abs(model.l[i]));
        if (std::isfinite(model.u[i])) rowScale = std::max(rowScale, std::abs(model.u[i]));
        require(std::isfinite(ax) && std::isfinite(rowScale), "nonfinite row activity or scale");
        require(ax >= model.l[i] - tolerance * rowScale && ax <= model.u[i] + tolerance * rowScale,
                "original-model primal feasibility failed");
        // Positive multipliers belong to upper bounds, negative to lower bounds.
        const double bound = y[i] > 0 ? model.u[i] : model.l[i];
        if (!std::isfinite(bound)) {
            require(std::abs(y[i]) <= tolerance, "multiplier has invalid sign for an infinite bound");
        } else {
            const double product = std::abs(y[i] * (ax - bound));
            require(std::isfinite(product) && product <= tolerance * (1 + std::abs(y[i]) * rowScale),
                    "original-model complementarity failed");
        }
    }
    for (std::size_t j = 0; j < x.size(); ++j)
        require(std::isfinite(gradient[j]) && std::isfinite(scale[j]) &&
                std::abs(gradient[j]) <= tolerance * scale[j], "original-model stationarity failed");
    require(std::isfinite(objective) && std::isfinite(result.primalObjective) &&
            std::abs(objective - result.primalObjective) <= tolerance * (1 + std::abs(objective)),
            "reported objective disagrees with original model");
}
} // namespace qpcheck
