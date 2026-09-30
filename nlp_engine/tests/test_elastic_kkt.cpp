// Elastic restoration, initial-point projection, multiplier signs, and the
// independent residual validation of the constructed QP.
//
// Every success case is re-verified INDEPENDENTLY of Result's own fields: the
// original nonlinear problem is re-evaluated at the returned point and the KKT
// residuals are recomputed from scratch. A solver that reported converged
// residuals while returning a point that does not satisfy them fails here.
#include "nlp/solver.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
using namespace nlp;

namespace {
void check(bool p, const std::string& what) { if (!p) throw std::runtime_error(what); }
void near(double a, double b, double tol, const std::string& what) {
    if (!(std::isfinite(a) && std::abs(a - b) <= tol))
        throw std::runtime_error(what + ": expected " + std::to_string(b) + ", got " + std::to_string(a));
}

// Recomputed from the ORIGINAL problem at the returned point, in original
// units, using the documented convention grad f + J' lambda + z = 0 with
// upper-side multipliers positive and lower-side multipliers negative.
struct Kkt {
    double feasibility = 0, stationarity = 0, complementarity = 0;
    bool signsValid = true;
    std::string detail;
};
Kkt verifyIndependently(const Problem& problem, const Result& r, double tolerance) {
    Kkt k;
    const auto& vb = problem.variableBounds();
    const auto& cb = problem.constraintBounds();
    check(r.primal.size() == vb.size(), "primal size");
    check(r.boundMultipliers.size() == vb.size(), "bound multiplier size");
    check(r.constraintMultipliers.size() == cb.size(), "constraint multiplier size");
    auto e = problem.evaluate(r.primal);
    std::vector<double> stationarity;
    e.jacobian.transposeMultiply(r.constraintMultipliers, stationarity);
    for (size_t j = 0; j < vb.size(); ++j)
        k.stationarity = std::max(k.stationarity,
            std::abs(stationarity[j] + e.gradient[j] + r.boundMultipliers[j]));
    auto side = [&](double value, Bounds b, double multiplier, const std::string& what) {
        k.feasibility = std::max(k.feasibility, std::max({0.0, b.lower - value, value - b.upper}));
        if (multiplier > tolerance) {
            // A positive multiplier claims the UPPER side is active.
            if (!std::isfinite(b.upper)) { k.signsValid = false; k.detail = what + ": positive multiplier with no upper bound"; return; }
            k.complementarity = std::max(k.complementarity, std::abs(multiplier * (value - b.upper)));
        } else if (multiplier < -tolerance) {
            if (!std::isfinite(b.lower)) { k.signsValid = false; k.detail = what + ": negative multiplier with no lower bound"; return; }
            k.complementarity = std::max(k.complementarity, std::abs(multiplier * (value - b.lower)));
        }
    };
    for (size_t j = 0; j < vb.size(); ++j) side(r.primal[j], vb[j], r.boundMultipliers[j], "x[" + std::to_string(j) + "]");
    for (size_t i = 0; i < cb.size(); ++i) side(e.constraints[i], cb[i], r.constraintMultipliers[i], "c[" + std::to_string(i) + "]");
    return k;
}

// Solve, require first-order stationarity, then re-derive the KKT tests here.
Result stationary(const std::string& name, const Problem& problem,
                  std::vector<double> initial, Options o = {}) {
    auto r = Solver().solve(problem, initial, o);
    std::cout << "  " << name << ": " << toString(r.status) << " f=" << r.objective
              << " elastic=" << r.elasticSubproblems << " it=" << r.iteration << "\n";
    check(r.status == Status::FirstOrderStationary, name + ": " + toString(r.status) + " (" + r.message + ")");
    check(r.hasPrimal && r.feasible, name + ": no feasible reported point");
    auto k = verifyIndependently(problem, r, o.tolerance);
    check(k.signsValid, name + ": multiplier sign invalid: " + k.detail);
    check(k.feasibility <= o.tolerance, name + ": independent feasibility " + std::to_string(k.feasibility));
    check(k.stationarity <= o.tolerance, name + ": independent stationarity " + std::to_string(k.stationarity));
    check(k.complementarity <= o.tolerance, name + ": independent complementarity " + std::to_string(k.complementarity));
    return r;
}

// Per-iteration constraint-violation trace, to show restoration reduces the
// ORIGINAL nonlinear violation rather than only the linearized one.
struct Trace {
    std::vector<double> violation;
    Options options() {
        Options o;
        o.callback = [this](const Iteration& it) { violation.push_back(it.primalResidual); return true; };
        return o;
    }
};
}

int main() {
    try {
        auto x = variable(0), y = variable(1);
        const double tol = 1e-6;

        // -------------------------------------------------------------------
        // 1. Elastic restoration.
        //
        // Each start below makes the HARD linearization inconsistent: the step
        // needed to satisfy the linearized row exceeds the variable box, so
        // attempt 0 fails the residual gate and the elastic subproblem runs.
        // The row is  lower-c <= J*D*d + e_plus - e_minus <= upper-c,  so a
        // lower-side violation is repaired by e_plus and an upper-side
        // violation by e_minus. Both columns must exist and be usable.
        // -------------------------------------------------------------------
        std::cout << "elastic restoration\n";

        // Lower-side violation of a nonlinear inequality: x^2 >= 1 from x=0.1.
        // Needs the +1 elastic column. Terminates on the lower side: negative.
        {
            Trace trace; auto o = trace.options();
            auto r = stationary("lower-side inequality",
                Model({{-2, 2}}, square(x), {square(x)}, {{1, infinity}}), {0.1}, o);
            check(r.elasticSubproblems > 0, "lower-side: restoration not exercised");
            near(r.primal[0], 1, 1e-5, "lower-side x");
            near(r.constraintMultipliers[0], -1, 1e-5, "lower-side multiplier");
            // The reported start is the supplied point with its violation
            // intact; initialization does not repair nonlinear constraints.
            near(trace.violation.front(), 0.99, 1e-12, "lower-side initial violation");
            check(trace.violation.back() <= tol, "lower-side final violation");
            check(trace.violation.size() > 1 && trace.violation[1] < trace.violation[0],
                  "lower-side: first restoration step did not reduce the violation");
        }

        // Upper-side violation: -x^2 <= -1 from x=0.1. Same geometry as above
        // with the row sign flipped, so the -1 elastic column is required and
        // the terminating multiplier is positive. If only a +1 column existed
        // this case could not be repaired.
        {
            Trace trace; auto o = trace.options();
            auto r = stationary("upper-side inequality",
                Model({{-2, 2}}, square(x), {-square(x)}, {{-infinity, -1}}), {0.1}, o);
            check(r.elasticSubproblems > 0, "upper-side: restoration not exercised");
            near(r.primal[0], 1, 1e-5, "upper-side x");
            near(r.constraintMultipliers[0], 1, 1e-5, "upper-side multiplier");
            near(trace.violation.front(), 0.99, 1e-12, "upper-side initial violation");
            check(trace.violation.back() <= tol, "upper-side final violation");
            check(trace.violation.size() > 1 && trace.violation[1] < trace.violation[0],
                  "upper-side: first restoration step did not reduce the violation");
        }

        // Ranged row, LOWER side violated at the start, terminating on the
        // UPPER side: 1 <= x^2 <= 4 from x=0.1 minimizing (x-3)^2 gives x=2.
        // The multiplier sign follows the ACTIVE side, not the violated one.
        {
            auto r = stationary("ranged, lower violated -> upper active",
                Model({{-3, 3}}, square(x - 3), {square(x)}, {{1, 4}}), {0.1});
            check(r.elasticSubproblems > 0, "ranged lower: restoration not exercised");
            near(r.primal[0], 2, 1e-5, "ranged lower x");
            near(r.constraintMultipliers[0], 0.5, 1e-5, "ranged lower multiplier");
        }

        // Ranged row, UPPER side violated at the start, terminating on the
        // LOWER side: -4 <= -x^2 <= -1 from x=0.1 minimizing (x-3)^2 gives
        // x=2, where -x^2 = -4 is the range's lower side: negative multiplier.
        {
            auto r = stationary("ranged, upper violated -> lower active",
                Model({{-3, 3}}, square(x - 3), {-square(x)}, {{-4, -1}}), {0.1});
            check(r.elasticSubproblems > 0, "ranged upper: restoration not exercised");
            near(r.primal[0], 2, 1e-5, "ranged upper x");
            near(r.constraintMultipliers[0], -0.5, 1e-5, "ranged upper multiplier");
        }

        // Equality restoration, both multiplier signs. x^2 = 1 from x=0.1 is
        // an inconsistent linearization; the objective decides the sign.
        {
            auto negative = stationary("equality restoration, negative multiplier",
                Model({{-2, 2}}, square(x), {square(x)}, {{1, 1}}), {0.1});
            check(negative.elasticSubproblems > 0, "equality negative: restoration not exercised");
            near(negative.primal[0], 1, 1e-5, "equality negative x");
            near(negative.constraintMultipliers[0], -1, 1e-5, "equality negative multiplier");
            auto positive = stationary("equality restoration, positive multiplier",
                Model({{-2, 2}}, square(x - 2), {square(x)}, {{1, 1}}), {0.1});
            check(positive.elasticSubproblems > 0, "equality positive: restoration not exercised");
            near(positive.primal[0], 1, 1e-5, "equality positive x");
            near(positive.constraintMultipliers[0], 1, 1e-5, "equality positive multiplier");
        }

        // Restoration is engaged only when the hard linearization fails. From
        // x=0.9 the same model's linearized step fits inside the box, so no
        // elastic subproblem runs. Without this contrast the tests above would
        // also pass if restoration ran unconditionally.
        {
            auto r = stationary("consistent linearization uses no elastic step",
                Model({{-3, 3}}, square(x), {square(x)}, {{1, infinity}}), {0.9});
            check(r.elasticSubproblems == 0, "elastic step used on a consistent linearization");
            near(r.primal[0], 1, 1e-5, "consistent x");
            near(r.constraintMultipliers[0], -1, 1e-5, "consistent multiplier");
        }

        // -------------------------------------------------------------------
        // 3. Initial point / bound projection.
        // -------------------------------------------------------------------
        std::cout << "initial point projection\n";

        // A start outside a variable bound is projected onto the bound and the
        // PROJECTED point is what gets evaluated and reported.
        {
            Trace trace; auto o = trace.options();
            auto r = stationary("start outside bounds is projected",
                Model({{1, 2}}, square(x - 1.5)), {100}, o);
            near(r.primal[0], 1.5, 1e-5, "projected start solution");
            // Objective at the projected start x=2 is 0.25; at the supplied
            // start x=100 it would be 9760.5.
            near(trace.violation.front(), 0, 1e-12, "projected start is bound-feasible");
        }

        // Projection happens BEFORE the first evaluation. Here the supplied
        // start x=3 is inside log's domain but violates the bounds; the
        // projected point x=-1 is not, and the result is EvaluationFailure.
        // A solver that evaluated the supplied point first would succeed.
        {
            auto r = Solver().solve(Model({{-5, -1}}, log(x)), {3});
            check(r.status == Status::EvaluationFailure,
                  std::string("projected start outside the domain must be EvaluationFailure, got ") + toString(r.status));
        }

        // Variable-bound violations are projected; nonlinear constraint
        // violations are NOT repaired at initialization. The first reported
        // iterate carries the exact violation of the PROJECTED point: x=-10
        // projects to 0.1, where x^2 = 0.01 leaves the row short by 0.99.
        // Had the supplied point been evaluated instead, the first reported
        // violation would have been the bound violation 10.1.
        {
            Trace trace; auto o = trace.options();
            auto r = stationary("projection does not repair nonlinear rows",
                Model({{0.1, 3}}, square(x), {square(x)}, {{1, infinity}}), {-10}, o);
            near(trace.violation.front(), 0.99, 1e-12, "violation at the projected start");
            check(r.elasticSubproblems > 0, "projected start: restoration not exercised");
            near(r.primal[0], 1, 1e-5, "projected start with active row");
            near(r.constraintMultipliers[0], -1, 1e-5, "projected start multiplier");
            near(r.boundMultipliers[0], 0, 1e-6, "projected start bound multiplier");
        }

        // A local merit-based method cannot cross a worsening-violation barrier.
        // From the projected start x=0.5 the feasible set of x^2 >= 1 inside
        // [-2, 0.5] is only reachable by first INCREASING the violation, so the
        // solver stalls. It reports NoProgress and must not claim infeasibility.
        {
            auto r = Solver().solve(Model({{-2, 0.5}}, square(x), {square(x)}, {{1, infinity}}), {10});
            check(r.status == Status::NoProgress,
                  std::string("unreachable feasible set should stall, got ") + toString(r.status));
            check(!r.feasible && r.hasPrimal, "stalled run must retain an infeasible iterate");
            check(r.elasticSubproblems > 0, "restoration should have been attempted");
        }

        // -------------------------------------------------------------------
        // 4. Multiplier semantics. Signs, not only primal feasibility.
        // -------------------------------------------------------------------
        std::cout << "multiplier semantics\n";

        // Equality rows: unrestricted sign, decided by the objective.
        near(stationary("positive equality multiplier",
            Model({{}}, square(x - 2), {x}, {{1, 1}}), {0}).constraintMultipliers[0], 2, 1e-5, "positive equality");
        near(stationary("negative equality multiplier",
            Model({{}}, square(x + 2), {x}, {{1, 1}}), {0}).constraintMultipliers[0], -6, 1e-5, "negative equality");

        // Dependent equalities: x+y=2 and 2(x+y)=4 have a singular Jacobian
        // and NON-UNIQUE multipliers, so only the stationarity combination is
        // checkable. verifyIndependently does exactly that. LICQ fails here and
        // FirstOrderStationary is still reported: the status does not imply a
        // constraint qualification or unique multipliers.
        {
            auto r = stationary("dependent equalities",
                Model({{}, {}}, square(x) + square(y), {x + y, 2 * (x + y)}, {{2, 2}, {4, 4}}), {0, 0});
            near(r.primal[0], 1, 1e-5, "dependent x");
            near(r.primal[1], 1, 1e-5, "dependent y");
        }

        // Fixed variable combined with an equality row. x is fixed at 1 and
        // x+y=3 forces y=2. Stationarity in y gives lambda=-4; stationarity in
        // x then gives the fixed variable's multiplier z=+4.
        {
            auto r = stationary("fixed variable with equality row",
                Model({{1, 1}, {}}, square(y), {x + y}, {{3, 3}}), {0, 0});
            near(r.primal[0], 1, 1e-9, "fixed x");
            near(r.primal[1], 2, 1e-5, "fixed-case y");
            near(r.constraintMultipliers[0], -4, 1e-5, "fixed-case row multiplier");
            near(r.boundMultipliers[0], 4, 1e-5, "fixed variable multiplier");
            near(r.boundMultipliers[1], 0, 1e-6, "free variable multiplier");
        }

        // Active variable bounds: lower side negative, upper side positive.
        near(stationary("active lower bound",
            Model({{0, 1}}, square(x + 3)), {0.5}).boundMultipliers[0], -6, 1e-5, "active lower bound");
        near(stationary("active upper bound",
            Model({{0, 1}}, square(x - 3)), {0.5}).boundMultipliers[0], 4, 1e-5, "active upper bound");

        // Inactive constraint and inactive bounds produce zero multipliers.
        {
            auto r = stationary("inactive row and bounds",
                Model({{-10, 10}}, square(x - 3), {x}, {{-5, 5}}), {0});
            near(r.primal[0], 3, 1e-5, "inactive solution");
            near(r.constraintMultipliers[0], 0, 1e-6, "inactive row multiplier");
            near(r.boundMultipliers[0], 0, 1e-6, "inactive bound multiplier");
        }

        // Active ranged row on each side, with no competing active bound.
        near(stationary("ranged row, upper side active",
            Model({{-10, 10}}, square(x - 3), {x}, {{1, 2}}), {0}).constraintMultipliers[0], 2, 1e-5, "ranged upper active");
        near(stationary("ranged row, lower side active",
            Model({{-10, 10}}, square(x + 3), {x}, {{-2, -1}}), {0}).constraintMultipliers[0], -2, 1e-5, "ranged lower active");

        // -------------------------------------------------------------------
        // 6. Independent residual validation of the constructed QP.
        //
        // A successful inner QP status must not by itself produce a successful
        // NLP result. ADMM's own termination test is RELATIVE and measured on
        // its Ruiz-equilibrated system; the gate recomputes ABSOLUTE residuals
        // of the QP the solver constructed, in unscaled SQP coordinates. It does
        // not rebuild the SQP model from the original nonlinear problem. With
        // row scaling disabled, the badly conditioned linearization below
        // satisfies ADMM's test -- it returns Optimal -- while its absolute
        // primal residual is ~1.6e-5, far above the gate's threshold. The gate
        // rejects it, and no FirstOrderStationary result is produced.
        // -------------------------------------------------------------------
        std::cout << "independent residual validation of the constructed QP\n";
        {
            Options o; o.scaleConstraints = false;
            Model illConditioned({{}, {}}, square(x) + square(y),
                {1e6 * x + y / 1e6}, {{1e6, 1e6}});
            auto r = Solver().solve(illConditioned, {0.3, 0.7}, o);
            std::cout << "  " << toString(r.status) << ": " << r.message << "\n";
            check(r.status == Status::SubproblemFailure,
                  std::string("inaccurate QP must be rejected, got ") + toString(r.status));
            check(r.status != Status::FirstOrderStationary, "false stationary result");
            // "validation: Optimal" is the signature that matters: the inner
            // solver reported success and the residual gate overruled it.
            check(r.message.find("independent residual validation of the constructed QP: Optimal") != std::string::npos,
                  "rejection did not come from the residual gate on an Optimal QP: " + r.message);
            check(r.message.find("not an NLP infeasibility certificate") != std::string::npos,
                  "subproblem failure must not read as an infeasibility proof");
            check(r.hasPrimal, "last evaluated iterate must be retained");
        }
        // A starved QP budget is likewise rejected rather than accepted.
        for (int budget : {1, 2, 5}) {
            Options o; o.qpIterationLimit = budget;
            auto r = Solver().solve(Model({{}}, square(x), {x}, {{2, infinity}}), {0}, o);
            check(r.status == Status::SubproblemFailure,
                  "starved QP budget " + std::to_string(budget) + " gave " + toString(r.status));
            check(r.message.find("independent residual validation of the constructed QP") != std::string::npos,
                  "starved QP not rejected by the gate");
        }

        std::cout << "NLP elastic/KKT tests passed\n";
    } catch (const std::exception& e) {
        std::cerr << "FAILED: " << e.what() << "\n";
        return 1;
    }
}
