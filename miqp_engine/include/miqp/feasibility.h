#pragma once

#include "model/model.h"

#include <string>
#include <vector>

namespace miqp {

struct FeasibilityCheck {
    bool feasible = false;
    std::string reason;
};

[[nodiscard]] FeasibilityCheck checkFeasibility(
    const model::Model& model,
    const std::vector<double>& x,
    double feasibilityTolerance,
    double integralityTolerance);

[[nodiscard]] FeasibilityCheck checkRoundingSafety(
    const model::Model& model,
    const std::vector<double>& beforeRounding,
    const std::vector<double>& rounded,
    double crossingTolerance = 1e-8);

}  // namespace miqp