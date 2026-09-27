#include "miqp/branch_and_bound.h"
#include "miqp/feasibility.h"

#include "qp/convexity.h"
#include "qp/qp_adapter.h"
#include "qp/qp_solver.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

namespace miqp {
namespace {

using Clock = std::chrono::steady_clock;

struct Node {
    std::vector<double> lower;
    std::vector<double> upper;
    int depth = 0;
};

bool better(double candidate,
            double incumbent,
            bool maximize,
            double tolerance) {
    return maximize
        ? candidate > incumbent + tolerance
        : candidate < incumbent - tolerance;
}

int chooseBranchVariable(const model::Model& model,
                         const std::vector<double>& x,
                         double tolerance) {
    int best = -1;
    double bestViolation = tolerance;

    for (std::size_t j = 0;
         j < model.variables.size() && j < x.size();
         ++j) {

        if (model.variables[j].type ==
            model::VariableType::Continuous) {
            continue;
        }

        const double violation =
            std::abs(x[j] - std::round(x[j]));

        if (violation > bestViolation) {
            bestViolation = violation;
            best = static_cast<int>(j);
        }
    }

    return best;
}

bool hasUnsupportedUnboundedIntegerDomain(
    const model::Model& model,
    std::size_t& offendingIndex) {

    for (std::size_t j = 0;
         j < model.variables.size();
         ++j) {

        const auto& variable =
            model.variables[j];

        if (variable.type ==
            model::VariableType::Continuous) {
            continue;
        }

        if (!std::isfinite(variable.lowerBound) ||
            !std::isfinite(variable.upperBound)) {

            offendingIndex = j;
            return true;
        }
    }

    return false;
}

bool nodeRowsDefinitelyInfeasible(const model::Model& model,
                                  const Node& node,
                                  double tolerance) {
    for (const auto& constraint :
         model.constraints) {

        double minActivity = 0.0;
        double maxActivity = 0.0;

        for (const auto& term :
             constraint.linearTerms) {

            if (term.value == 0.0) {
                continue;
            }

            const std::size_t j =
                static_cast<std::size_t>(
                    term.variableIndex);

            if (j >= node.lower.size()) {
                return true;
            }

            if (term.value > 0.0) {
                minActivity +=
                    term.value * node.lower[j];

                maxActivity +=
                    term.value * node.upper[j];
            } else {
                minActivity +=
                    term.value * node.upper[j];

                maxActivity +=
                    term.value * node.lower[j];
            }
        }

        if (std::isfinite(
                constraint.lowerBound) &&
            maxActivity <
                constraint.lowerBound -
                    tolerance) {
            return true;
        }

        if (std::isfinite(
                constraint.upperBound) &&
            minActivity >
                constraint.upperBound +
                    tolerance) {
            return true;
        }
    }

    return false;
}

double modelObjective(
    const model::Model& model,
    const std::vector<double>& x) {

    double value =
        model.objective.offset;

    for (const auto& term :
         model.objective.linearTerms) {

        const std::size_t j =
            static_cast<std::size_t>(
                term.variableIndex);

        if (j >= x.size()) {
            return std::numeric_limits<
                double>::quiet_NaN();
        }

        value +=
            term.value * x[j];
    }

    for (const auto& term :
         model.objective.quadraticTerms) {

        const std::size_t i =
            static_cast<std::size_t>(
                term.variableIndex1);

        const std::size_t j =
            static_cast<std::size_t>(
                term.variableIndex2);

        if (i >= x.size() ||
            j >= x.size()) {
            return std::numeric_limits<
                double>::quiet_NaN();
        }

        value +=
            term.value * x[i] * x[j];
    }

    return value;
}

}  // namespace

MiqpResult BranchAndBoundSolver::solve(
    const model::Model& model,
    const MiqpOptions& options) const {

    MiqpResult result;

    // ---------------------------------------------------------------
    // 1. Structural model validation
    // ---------------------------------------------------------------

    if (!model.validate()) {
        result.status =
            MiqpStatus::RelaxationFailure;

        result.message =
            "MIQP model failed structural validation";

        return result;
    }

    // ---------------------------------------------------------------
    // 2. Integer-domain validation
    //
    // This B&B implementation creates children using floor()/ceil()
    // and stores explicit finite node bounds.
    //
    // Unbounded integer domains are therefore not supported yet.
    // Reject them explicitly instead of relying on undefined/implicit
    // behaviour involving +/- infinity.
    // ---------------------------------------------------------------

    std::size_t unboundedIntegerIndex = 0;

    if (hasUnsupportedUnboundedIntegerDomain(
            model,
            unboundedIntegerIndex)) {

        result.status =
            MiqpStatus::RelaxationFailure;

        result.message =
            "MIQP currently requires finite lower and upper bounds "
            "for every integer/binary variable; variable " +
            std::to_string(
                unboundedIntegerIndex) +
            " has an unbounded integer domain";

        return result;
    }

    // ---------------------------------------------------------------
    // 3. Convexity validation
    //
    // Supported:
    //
    //   minimize: Hessian PSD
    //   maximize: Hessian NSD
    //
    // Non-convex MIQP is intentionally rejected.
    // ---------------------------------------------------------------

    const qp::ConvexityCheck convexity =
        qp::checkConvexity(model);

    if (!convexity.convexForObjectiveSense) {
        result.status =
            MiqpStatus::RelaxationFailure;

        result.message =
            convexity.reason;

        return result;
    }

    const bool maximize =
        model.objective.sense ==
        model::ObjectiveSense::Maximize;

    // ---------------------------------------------------------------
    // 4. Timing
    // ---------------------------------------------------------------

    const auto start =
        Clock::now();

    const auto elapsed = [&]() {
        return std::chrono::duration<double>(
            Clock::now() - start).count();
    };

    // ---------------------------------------------------------------
    // 5. Root node
    // ---------------------------------------------------------------

    Node root;

    root.lower.reserve(
        model.variables.size());

    root.upper.reserve(
        model.variables.size());

    for (const auto& variable :
         model.variables) {

        root.lower.push_back(
            variable.lowerBound);

        root.upper.push_back(
            variable.upperBound);
    }

    std::vector<Node> stack;
    stack.push_back(
        std::move(root));

    // ---------------------------------------------------------------
    // 6. Incumbent
    // ---------------------------------------------------------------

    bool hasIncumbent = false;

    double incumbent =
        maximize
            ? -std::numeric_limits<
                  double>::infinity()
            : std::numeric_limits<
                  double>::infinity();

    std::vector<double> incumbentX;

    // ---------------------------------------------------------------
    // CURRENT B&B BOUND LIMITATION
    //
    // qp_engine currently returns a numerical ADMM primal solution
    // and primal objective.
    //
    // That primal objective is NOT a mathematically certified
    // branch-and-bound node bound.
    //
    // For minimization B&B requires a valid LOWER bound.
    // A primal feasible QP objective is generally an UPPER bound on
    // the true continuous-relaxation optimum.
    //
    // For maximization B&B requires a valid UPPER bound, and the
    // analogous problem occurs.
    //
    // Therefore relaxationResult.primalObjective is intentionally
    // NOT used for node pruning.
    //
    // Correctness is preserved by exhausting the B&B tree and using
    // only logically justified pruning such as proven infeasibility.
    //
    // Consequence:
    // this implementation may explore substantially more nodes than
    // an MIQP solver with certified QP dual bounds.
    //
    // Future work:
    // expose a mathematically valid QP lower bound for minimization
    // / upper bound for maximization before objective pruning is
    // reintroduced.
    // ---------------------------------------------------------------

    while (!stack.empty()) {

        // -----------------------------------------------------------
        // 7. Time limit
        // -----------------------------------------------------------

        if (options.timeLimitSeconds > 0.0 &&
            elapsed() >=
                options.timeLimitSeconds) {

            result.status =
                MiqpStatus::TimeLimit;

            result.message =
                "MIQP time limit reached";

            result.primal =
                incumbentX;

            if (hasIncumbent) {
                result.objectiveValue =
                    incumbent;
            }

            return result;
        }

        // -----------------------------------------------------------
        // 8. Node limit
        // -----------------------------------------------------------

        if (options.nodeLimit > 0 &&
            result.nodeCount >=
                options.nodeLimit) {

            result.status =
                MiqpStatus::NodeLimit;

            result.message =
                "MIQP node limit reached";

            result.primal =
                incumbentX;

            if (hasIncumbent) {
                result.objectiveValue =
                    incumbent;
            }

            return result;
        }

        // -----------------------------------------------------------
        // 9. Pop node
        // -----------------------------------------------------------

        Node node =
            std::move(stack.back());

        stack.pop_back();

        ++result.nodeCount;

        // -----------------------------------------------------------
        // 10. Build continuous relaxation
        // -----------------------------------------------------------

        model::Model relaxation =
            model;

        bool inconsistent = false;

        for (std::size_t j = 0;
             j <
             relaxation.variables.size();
             ++j) {

            relaxation.variables[j].type =
                model::VariableType::Continuous;

            relaxation.variables[j].
                lowerBound =
                    node.lower[j];

            relaxation.variables[j].
                upperBound =
                    node.upper[j];

            if (node.lower[j] >
                node.upper[j] +
                    options.
                        integralityTolerance) {

                inconsistent = true;
                break;
            }
        }

        if (inconsistent) {
            continue;
        }

        // -----------------------------------------------------------
        // 11. Exact row/bound interval infeasibility check
        //
        // This is safe pruning: a row is impossible under the
        // current node bounds.
        // -----------------------------------------------------------

        if (nodeRowsDefinitelyInfeasible(
                model,
                node,
                1e-12)) {
            continue;
        }

        // -----------------------------------------------------------
        // 12. Translate QP
        // -----------------------------------------------------------

        qp::QpTranslation translation;
        qp::QpModel qpModel;

        try {
            qpModel =
                qp::fromModel(
                    relaxation,
                    translation);
        } catch (
            const std::exception& error) {

            result.status =
                MiqpStatus::
                    RelaxationFailure;

            result.message =
                error.what();

            return result;
        }

        // -----------------------------------------------------------
        // 13. QP options
        // -----------------------------------------------------------

        qp::AdmmOptions qpOptions;

        const double qpTolerance =
            std::min(
                options.
                    integralityTolerance *
                    0.1,
                options.
                    feasibilityTolerance *
                    0.1);

        qpOptions.primalTolerance =
            qpTolerance;

        qpOptions.dualTolerance =
            qpTolerance;

        qpOptions.threadCount =
            options.threadCount;

        if (options.timeLimitSeconds >
            0.0) {

            qpOptions.timeLimitSeconds =
                std::max(
                    0.0,
                    options.
                        timeLimitSeconds -
                        elapsed());

            if (qpOptions.
                    timeLimitSeconds <=
                0.0) {

                result.status =
                    MiqpStatus::TimeLimit;

                result.message =
                    "MIQP time limit reached";

                result.primal =
                    incumbentX;

                if (hasIncumbent) {
                    result.objectiveValue =
                        incumbent;
                }

                return result;
            }
        }

        // -----------------------------------------------------------
        // 14. Solve continuous QP relaxation
        // -----------------------------------------------------------

        const qp::AdmmResult
            relaxationResult =
                qp::QpSolver{}.solve(
                    qpModel,
                    qpOptions);

        result.qpIterations +=
            relaxationResult.iterations;

        // -----------------------------------------------------------
        // 15. Relaxation status handling
        // -----------------------------------------------------------

        if (relaxationResult.status ==
            qp::QpStatus::Infeasible) {

            continue;
        }

        if (relaxationResult.status ==
            qp::QpStatus::TimeLimit) {

            result.status =
                MiqpStatus::TimeLimit;

            result.message =
                "QP relaxation reached "
                "the MIQP time limit";

            result.primal =
                incumbentX;

            if (hasIncumbent) {
                result.objectiveValue =
                    incumbent;
            }

            return result;
        }

        if (relaxationResult.status ==
            qp::QpStatus::Unbounded) {

            // An unbounded continuous relaxation
            // does NOT constitute a certificate that
            // the mixed-integer problem is unbounded.

            result.status =
                MiqpStatus::
                    RelaxationFailure;

            result.message =
                "QP relaxation is unbounded; "
                "no valid mixed-integer "
                "unboundedness certificate "
                "is available";

            return result;
        }

        if (relaxationResult.status !=
            qp::QpStatus::Optimal) {

            result.status =
                MiqpStatus::
                    RelaxationFailure;

            result.message =
                "QP relaxation failed: " +
                relaxationResult.
                    statusMessage;

            return result;
        }

        if (relaxationResult.
                primal.size() !=
            model.variables.size()) {

            result.status =
                MiqpStatus::
                    RelaxationFailure;

            result.message =
                "QP relaxation returned "
                "an incomplete primal solution";

            return result;
        }

        // -----------------------------------------------------------
        // 16. IMPORTANT:
        // No objective-bound pruning occurs here.
        //
        // relaxationResult.primalObjective is intentionally ignored
        // as a B&B pruning bound.
        // -----------------------------------------------------------

        // -----------------------------------------------------------
        // 17. Look for a genuinely fractional integer variable
        // -----------------------------------------------------------

        int branch =
            chooseBranchVariable(
                model,
                relaxationResult.primal,
                options.
                    integralityTolerance);

        // -----------------------------------------------------------
        // 18. Solution appears integer within normal tolerance
        // -----------------------------------------------------------

        if (branch < 0) {

            std::vector<double>
                roundedCandidate =
                    relaxationResult.primal;

            for (std::size_t j = 0;
                 j <
                 roundedCandidate.size();
                 ++j) {

                if (model.variables[j].
                        type !=
                    model::VariableType::
                        Continuous) {

                    roundedCandidate[j] =
                        std::round(
                            roundedCandidate[j]);
                }
            }

            // -------------------------------------------------------
            // 19. Validate rounded candidate
            // -------------------------------------------------------

            const FeasibilityCheck
                feasibility =
                    checkFeasibility(
                        model,
                        roundedCandidate,
                        options.
                            feasibilityTolerance,
                        options.
                            integralityTolerance);

            const FeasibilityCheck
                roundingSafety =
                    checkRoundingSafety(
                        model,
                        relaxationResult.primal,
                        roundedCandidate);

            // -------------------------------------------------------
            // 20. Feasible integer candidate
            // -------------------------------------------------------

            if (feasibility.feasible &&
                roundingSafety.feasible) {

                const double candidate =
                    modelObjective(
                        model,
                        roundedCandidate);

                if (!std::isfinite(
                        candidate)) {

                    result.status =
                        MiqpStatus::
                            RelaxationFailure;

                    result.message =
                        "failed to evaluate "
                        "MIQP incumbent "
                        "objective";

                    return result;
                }

                if (!hasIncumbent ||
                    better(
                        candidate,
                        incumbent,
                        maximize,
                        options.
                            objectiveTolerance)) {

                    hasIncumbent = true;

                    incumbent =
                        candidate;

                    incumbentX =
                        std::move(
                            roundedCandidate);
                }

                continue;
            }

            // -------------------------------------------------------
            // 21. Candidate only appeared integral because of
            // numerical tolerance.
            //
            // Retry using essentially machine precision so that a
            // tiny but real fractional component can still create a
            // valid B&B branch.
            // -------------------------------------------------------

            branch =
                chooseBranchVariable(
                    model,
                    relaxationResult.primal,
                    std::numeric_limits<
                        double>::epsilon());

            // -------------------------------------------------------
            // CRITICAL NUMERICAL SAFETY:
            //
            // If no fractional integer variable remains, we have:
            //
            //   - no valid incumbent,
            //   - no infeasibility certificate,
            //   - and no legal branch to continue from.
            //
            // Silently continuing here would incorrectly treat the
            // node as exhausted and could lead to false Optimal or
            // Infeasible status.
            //
            // Therefore terminate with RelaxationFailure instead.
            // -------------------------------------------------------

            if (branch < 0) {

                result.status =
                    MiqpStatus::
                        RelaxationFailure;

                result.message =
                    "QP relaxation produced an "
                    "approximately integral point, "
                    "but the rounded candidate failed "
                    "final feasibility validation and "
                    "no fractional integer variable "
                    "remains to branch on";

                return result;
            }
        }

        // -----------------------------------------------------------
        // 22. Create B&B children
        // -----------------------------------------------------------

        const std::size_t j =
            static_cast<std::size_t>(
                branch);

        const double value =
            relaxationResult.primal[j];

        if (!std::isfinite(value)) {

            result.status =
                MiqpStatus::
                    RelaxationFailure;

            result.message =
                "QP relaxation returned a "
                "non-finite value for the "
                "branching variable";

            return result;
        }

        const double down =
            std::floor(value);

        const double up =
            std::ceil(value);

        Node lowerChild =
            node;

        lowerChild.depth =
            node.depth + 1;

        lowerChild.upper[j] =
            std::min(
                lowerChild.upper[j],
                down);

        Node upperChild =
            node;

        upperChild.depth =
            node.depth + 1;

        upperChild.lower[j] =
            std::max(
                upperChild.lower[j],
                up);

        // -----------------------------------------------------------
        // 23. Do not create a child identical to its parent
        // -----------------------------------------------------------

        const bool lowerTightens =
            lowerChild.upper[j] <
            node.upper[j];

        const bool upperTightens =
            upperChild.lower[j] >
            node.lower[j];

        // Push upper first because the stack is LIFO.
        // This causes the down/lower child to be processed first.

        if (upperTightens &&
            upperChild.lower[j] <=
                upperChild.upper[j] +
                    options.
                        integralityTolerance) {

            stack.push_back(
                std::move(
                    upperChild));
        }

        if (lowerTightens &&
            lowerChild.lower[j] <=
                lowerChild.upper[j] +
                    options.
                        integralityTolerance) {

            stack.push_back(
                std::move(
                    lowerChild));
        }
    }

    // ---------------------------------------------------------------
    // 24. Entire B&B tree exhausted
    // ---------------------------------------------------------------

    if (!hasIncumbent) {

        result.status =
            MiqpStatus::Infeasible;

        result.message =
            "all branch-and-bound nodes "
            "were exhausted without a "
            "feasible integer solution";

        return result;
    }

    // Since every B&B node has been exhausted and no uncertified
    // objective pruning was used, the best incumbent found is the
    // global optimum over the represented finite integer domain.

    result.status =
        MiqpStatus::Optimal;

    result.message =
        "convex MIQP branch-and-bound completed";

    result.primal =
        std::move(incumbentX);

    result.objectiveValue =
        incumbent;

    return result;
}

}  // namespace miqp