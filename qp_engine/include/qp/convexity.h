#pragma once

#include "model/model.h"

#include <string>

namespace qp {

struct ConvexityCheck {
    bool convexForObjectiveSense = false;
    double mostNegativePivot = 0.0;
    std::string reason;
};

// Validates the quadratic objective for use by the convex QP engine.
// For minimization the Hessian must be positive semidefinite. For
// maximization the Hessian must be negative semidefinite, which is equivalent
// to the Hessian of the sign-negated minimization form being PSD.
[[nodiscard]] ConvexityCheck checkConvexity(
    const model::Model& model,
    double relativeTolerance = 1e-10);

}  // namespace qp
