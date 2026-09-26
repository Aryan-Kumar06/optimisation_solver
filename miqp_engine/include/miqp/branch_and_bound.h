#pragma once

#include "miqp/miqp_types.h"
#include "model/model.h"

namespace miqp {

// Branch-and-bound for convex MIQP with linear constraints. Every node is a
// convex continuous QP relaxation solved by qp_engine. This class deliberately
// does not claim global support for non-convex MIQP.
class BranchAndBoundSolver {
public:
    [[nodiscard]] MiqpResult solve(
        const model::Model& model,
        const MiqpOptions& options = {}) const;
};

}  // namespace miqp
