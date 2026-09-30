#pragma once

#include "qp/qp_model.h"
#include "qp/qp_types.h"

#include <vector>

namespace qp {

// ============================================================================
// Active-set KKT polishing.
//
// After the ADMM inner loop, the iterate is approximately optimal. A
// small number of additional KKT refinement steps using a fixed active set
// can improve the result to high precision.
//
// We perform the following:
//   1. Identify the active set: constraints for which
//        A x ≈ l  or  A x ≈ u  (within an active-set tolerance)
//   2. Build the reduced KKT system with the active constraints enforced
//      as equalities:
//        [ P  A_a^T ] [x]   [-q]
//        [ A_a  0   ] [y_a] = [b_a]   where b_a = A_a * x
//   3. Solve the reduced system and update (x, y_a).
//   4. Repeat for a bounded number of iterations or until convergence.
//
// The polishing pass only works for convex QPs (P SPD). For non-convex
// problems it is skipped and the ADMM result is returned unchanged.
// ============================================================================

class KktPolisher {
public:
    struct Options {
        int maxIterations = 200;
        double tolerance = 1e-8;
        double activeSetTolerance = 1e-6;
        // Polishing is optional refinement of an already converged point, so
        // it must never cost more than the solve it refines. Each pass factors
        // a DENSE (n + active rows) square system, so its cost is cubic.
        // Measured on Maros-Meszaros qship08s after presolve (1,632 variables
        // plus a comparable number of active rows): ADMM converged in 8.2 s,
        // then polishing ran for more than 300 s, past a 55 s time limit that
        // it never checked. Above maximumDenseDimension polishing is skipped;
        // timeLimitSeconds (0 = none) is the budget left by the solver.
        int maximumDenseDimension = 2000;
        double timeLimitSeconds = 0.0;
    };

    // Polishes the ADMM iterate in place.  Returns true on success (P SPD),
    // false on numerical failure or if the problem is not convex enough.
    // Pass an Options{} for default behaviour.
    static bool polish(
        const QpModel& model,
        std::vector<double>& x,
        std::vector<double>& constraintDual,
        const Options& options
    );

    // Convenience overload with default options.
    static bool polish(
        const QpModel& model,
        std::vector<double>& x,
        std::vector<double>& constraintDual
    );
};

}  // namespace qp
