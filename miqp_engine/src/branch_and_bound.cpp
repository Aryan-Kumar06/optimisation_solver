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

bool nodeRowsDefinitelyInfeasible(const model::Model& model,
                                  const Node& node,
                                  double tolerance) {
    for (const auto& constraint : model.constraints) {
        double minActivity = 0.0;
        double maxActivity = 0.0;

        for (const auto& term : constraint.linearTerms) {
            if (term.value == 0.0) {
                continue;
            }

            const std::size_t j =
                static_cast<std::size_t>(term.variableIndex);

            if (j >= node.lower.size()) {
                return true;
            }

            if (term.value > 0.0) {
                minActivity += term.value * node.lower[j];
                maxActivity += term.value * node.upper[j];
            } else {
                minActivity += term.value * node.upper[j];
                maxActivity += term.value * node.lower[j];
            }
        }

        if (std::isfinite(constraint.lowerBound) &&
            maxActivity <
                constraint.lowerBound - tolerance) {
            return true;
        }

        if (std::isfinite(constraint.upperBound) &&
            minActivity >
                constraint.upperBound + tolerance) {
            return true;
        }
    }

    return false;
}

double modelObjective(const model::Model& model,
                      const std::vector<double>& x) {
    double value = model.objective.offset;

    for (const auto& term :
         model.objective.linearTerms) {
        const std::size_t j =
            static_cast<std::size_t>(
                term.variableIndex);

        value += term.value * x[j];
    }

    for (const auto& term :
         model.objective.quadraticTerms) {
        const std::size_t i =
            static_cast<std::size_t>(
                term.variableIndex1);

        const std::size_t j =
            static_cast<std::size_t>(
                term.variableIndex2);

        value += term.value * x[i] * x[j];
    }

    return value;
}

}  // namespace

MiqpResult BranchAndBoundSolver::solve(
    const model::Model& model,
    const MiqpOptions& options) const {

    MiqpResult result;

    // ---------------------------------------------------------------------
    // 1. Structural validation
    // ---------------------------------------------------------------------

    if (!model.validate()) {
        result.status =
            MiqpStatus::RelaxationFailure;

        result.message =
            "MIQP model failed structural validation";

        return result;
    }

    // ---------------------------------------------------------------------
    // 2. Convexity validation
    //
    // The current MIQP implementation only supports:
    //
    //   minimize convex quadratic objective
    //   maximize concave quadratic objective
    //
    // Non-convex MIQP must never be passed to the convex QP engine.
    // ---------------------------------------------------------------------

    const qp::ConvexityCheck convexity =
        qp::checkConvexity(model);

    if (!convexity.convexForObjectiveSense) {
        result.status =
            MiqpStatus::RelaxationFailure;

        result.message = convexity.reason;

        return result;
    }

    const bool maximize =
        model.objective.sense ==
        model::ObjectiveSense::Maximize;

    // ---------------------------------------------------------------------
    // 3. Timing
    // ---------------------------------------------------------------------

    const auto start = Clock::now();

    const auto elapsed = [&]() {
        return std::chrono::duration<double>(
            Clock::now() - start).count();
    };

    // ---------------------------------------------------------------------
    // 4. Build the root node from the original variable bounds
    // ---------------------------------------------------------------------

    Node root;

    root.lower.reserve(model.variables.size());
    root.upper.reserve(model.variables.size());

    for (const auto& variable :
         model.variables) {

        root.lower.push_back(
            variable.lowerBound);

        root.upper.push_back(
            variable.upperBound);
    }

    std::vector<Node> stack;
    stack.push_back(std::move(root));

    // ---------------------------------------------------------------------
    // 5. Incumbent storage
    // ---------------------------------------------------------------------

    bool hasIncumbent = false;

    double incumbent =
        maximize
            ? -std::numeric_limits<double>::infinity()
            :  std::numeric_limits<double>::infinity();

    std::vector<double> incumbentX;

    // ---------------------------------------------------------------------
    // IMPORTANT NUMERICAL NOTE
    //
    // The QP engine uses ADMM and currently returns a numerically obtained
    // primal objective.
    //
    // That primal objective is NOT a mathematically certified branch-and-
    // bound node bound.
    //
    // For minimization, B&B requires a certified LOWER bound.
    // A primal feasible QP objective is generally an UPPER bound on the
    // continuous relaxation optimum.
    //
    // For maximization, the analogous reverse issue applies.
    //
    // Therefore:
    //
    //     relaxationResult.primalObjective
    //
    // MUST NOT be used to prune branch-and-bound nodes.
    //
    // Until the QP engine exposes a certified dual bound or some rigorously
    // justified numerical bound, optimality is established by exhausting
    // the branch-and-bound tree rather than objective-bound pruning.
    // ---------------------------------------------------------------------

    while (!stack.empty()) {

        // -----------------------------------------------------------------
        // 6. Global time limit
        // -----------------------------------------------------------------

        if (options.timeLimitSeconds > 0.0 &&
            elapsed() >=
                options.timeLimitSeconds) {

            result.status =
                MiqpStatus::TimeLimit;

            result.message =
                "MIQP time limit reached";

            result.primal = incumbentX;

            if (hasIncumbent) {
                result.objectiveValue =
                    incumbent;
            }

            return result;
        }

        // -----------------------------------------------------------------
        // 7. Global node limit
        // -----------------------------------------------------------------

        if (options.nodeLimit > 0 &&
            result.nodeCount >=
                options.nodeLimit) {

            result.status =
                MiqpStatus::NodeLimit;

            result.message =
                "MIQP node limit reached";

            result.primal = incumbentX;

            if (hasIncumbent) {
                result.objectiveValue =
                    incumbent;
            }

            return result;
        }

        // -----------------------------------------------------------------
        // 8. Pop one branch-and-bound node
        // -----------------------------------------------------------------

        Node node =
            std::move(stack.back());

        stack.pop_back();

        ++result.nodeCount;

        // -----------------------------------------------------------------
        // 9. Build the continuous QP relaxation
        // -----------------------------------------------------------------

        model::Model relaxation = model;

        bool inconsistent = false;

        for (std::size_t j = 0;
             j < relaxation.variables.size();
             ++j) {

            relaxation.variables[j].type =
                model::VariableType::Continuous;

            relaxation.variables[j].lowerBound =
                node.lower[j];

            relaxation.variables[j].upperBound =
                node.upper[j];

            if (node.lower[j] >
                node.upper[j] +
                    options.integralityTolerance) {

                inconsistent = true;
                break;
            }
        }

        if (inconsistent) {
            continue;
        }

        // -----------------------------------------------------------------
        // 10. Cheap exact row/bound infeasibility check
        //
        // This is safe pruning because it proves that no point satisfying
        // the current node bounds can satisfy one of the rows.
        // -----------------------------------------------------------------

        if (nodeRowsDefinitelyInfeasible(
                model,
                node,
                1e-12)) {
            continue;
        }

        // -----------------------------------------------------------------
        // 11. Translate to qp_engine model
        // -----------------------------------------------------------------

        qp::QpTranslation translation;
        qp::QpModel qpModel;

        try {
            qpModel =
                qp::fromModel(
                    relaxation,
                    translation);

        } catch (const std::exception& error) {

            result.status =
                MiqpStatus::RelaxationFailure;

            result.message =
                error.what();

            return result;
        }

        // -----------------------------------------------------------------
        // 12. Configure the numerical QP solve
        // -----------------------------------------------------------------

        qp::AdmmOptions qpOptions;

        const double qpTolerance =
            std::min(
                options.integralityTolerance *
                    0.1,
                options.feasibilityTolerance *
                    0.1);

        qpOptions.primalTolerance =
            qpTolerance;

        qpOptions.dualTolerance =
            qpTolerance;

        qpOptions.threadCount =
            options.threadCount;

        if (options.timeLimitSeconds > 0.0) {

            qpOptions.timeLimitSeconds =
                std::max(
                    0.0,
                    options.timeLimitSeconds -
                        elapsed());

            if (qpOptions.timeLimitSeconds <=
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

        // -----------------------------------------------------------------
        // 13. Solve the continuous relaxation
        // -----------------------------------------------------------------

        const qp::AdmmResult relaxationResult =
            qp::QpSolver{}.solve(
                qpModel,
                qpOptions);

        result.qpIterations +=
            relaxationResult.iterations;

        // -----------------------------------------------------------------
        // 14. Handle relaxation statuses
        // -----------------------------------------------------------------

        if (relaxationResult.status ==
            qp::QpStatus::Infeasible) {

            // This node has no feasible
            // continuous relaxation.
            continue;
        }

        if (relaxationResult.status ==
            qp::QpStatus::TimeLimit) {

            result.status =
                MiqpStatus::TimeLimit;

            result.message =
                "QP relaxation reached the MIQP time limit";

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

            // An unbounded continuous relaxation does
            // NOT prove that the original mixed-integer
            // problem is unbounded.
            //
            // We currently do not have a valid
            // mixed-integer unboundedness certificate.
            result.status =
                MiqpStatus::RelaxationFailure;

            result.message =
                "QP relaxation is unbounded; "
                "no mixed-integer unboundedness "
                "certificate is available";

            return result;
        }

        if (relaxationResult.status !=
            qp::QpStatus::Optimal) {

            result.status =
                MiqpStatus::RelaxationFailure;

            result.message =
                "QP relaxation failed: " +
                relaxationResult.statusMessage;

            return result;
        }

        if (relaxationResult.primal.size() !=
            model.variables.size()) {

            result.status =
                MiqpStatus::RelaxationFailure;

            result.message =
                "QP relaxation returned an incomplete "
                "primal solution";

            return result;
        }

        // -----------------------------------------------------------------
        // 15. DO NOT prune using primalObjective
        //
        // There used to be code here similar to:
        //
        //     if (hasIncumbent &&
        //         prunable(relaxationObjective,
        //                  incumbent, ...)) {
        //         continue;
        //     }
        //
        // That was mathematically unsafe because the ADMM primal objective
        // is not a certified relaxation lower/upper bound.
        //
        // No objective-based node pruning is performed here.
        // -----------------------------------------------------------------

        // -----------------------------------------------------------------
        // 16. Check integrality
        // -----------------------------------------------------------------

        int branch =
            chooseBranchVariable(
                model,
                relaxationResult.primal,
                options.integralityTolerance);

        // -----------------------------------------------------------------
        // 17. Near-integral candidate
        // -----------------------------------------------------------------

        if (branch < 0) {

            std::vector<double>
                roundedCandidate =
                    relaxationResult.primal;

            // Round only integer/binary variables.
            for (std::size_t j = 0;
                 j < roundedCandidate.size();
                 ++j) {

                if (model.variables[j].type !=
                    model::VariableType::Continuous) {

                    roundedCandidate[j] =
                        std::round(
                            roundedCandidate[j]);
                }
            }

            // -------------------------------------------------------------
            // 18. Revalidate rounded candidate
            // -------------------------------------------------------------

            const FeasibilityCheck feasibility =
                checkFeasibility(
                    model,
                    roundedCandidate,
                    options.feasibilityTolerance,
                    options.integralityTolerance);

            const FeasibilityCheck roundingSafety =
                checkRoundingSafety(
                    model,
                    relaxationResult.primal,
                    roundedCandidate);

            // -------------------------------------------------------------
            // 19. Candidate is truly feasible
            // -------------------------------------------------------------

            if (feasibility.feasible &&
                roundingSafety.feasible) {

                const double candidate =
                    modelObjective(
                        model,
                        roundedCandidate);

                if (!hasIncumbent ||
                    better(
                        candidate,
                        incumbent,
                        maximize,
                        options.objectiveTolerance)) {

                    hasIncumbent = true;
                    incumbent = candidate;

                    incumbentX =
                        std::move(
                            roundedCandidate);
                }

                // This node has produced a feasible
                // integer solution and needs no further
                // branching.
                continue;
            }

            // -------------------------------------------------------------
            // 20. Candidate only looked integral because of tolerance
            //
            // Rounding changed feasibility.
            //
            // We must NOT accept it.
            //
            // Instead branch on the tiny remaining
            // fractional component.
            // -------------------------------------------------------------

            branch =
                chooseBranchVariable(
                    model,
                    relaxationResult.primal,
                    std::numeric_limits<
                        double>::epsilon());

            if (branch < 0) {

                // Nothing meaningful remains to branch on.
                //
                // Reject the invalid rounded candidate
                // rather than making it an incumbent.
                continue;
            }
        }

        // -----------------------------------------------------------------
        // 21. Branch on selected integer variable
        // -----------------------------------------------------------------

        const std::size_t j =
            static_cast<std::size_t>(
                branch);

        const double value =
            relaxationResult.primal[j];

        const double down =
            std::floor(value);

        const double up =
            std::ceil(value);

        // Down branch:
        //
        // x_j <= floor(value)
        Node lowerChild = node;

        lowerChild.depth =
            node.depth + 1;

        lowerChild.upper[j] =
            std::min(
                lowerChild.upper[j],
                down);

        // Up branch:
        //
        // x_j >= ceil(value)
        Node upperChild = node;

        upperChild.depth =
            node.depth + 1;

        upperChild.lower[j] =
            std::max(
                upperChild.lower[j],
                up);

        // -----------------------------------------------------------------
        // 22. Ensure children actually tighten the current node
        //
        // This prevents a numerically near-integral point from creating a
        // child identical to its parent.
        // -----------------------------------------------------------------

        const bool upperTightens =
            upperChild.lower[j] >
            node.lower[j];

        const bool lowerTightens =
            lowerChild.upper[j] <
            node.upper[j];

        // LIFO:
        // push upper first so lower/down branch
        // is explored first.

        if (upperTightens &&
            upperChild.lower[j] <=
                upperChild.upper[j] +
                    options.integralityTolerance) {

            stack.push_back(
                std::move(upperChild));
        }

        if (lowerTightens &&
            lowerChild.lower[j] <=
                lowerChild.upper[j] +
                    options.integralityTolerance) {

            stack.push_back(
                std::move(lowerChild));
        }
    }

    // ---------------------------------------------------------------------
    // 23. Tree exhausted
    // ---------------------------------------------------------------------

    if (!hasIncumbent) {

        result.status =
            MiqpStatus::Infeasible;

        result.message =
            "all convex QP relaxations were "
            "infeasible or exhausted";

        return result;
    }

    // Because there are no unexplored nodes left, the best feasible
    // integer incumbent is globally optimal over the explored MIQP tree.
    //
    // This proof does NOT depend on the ADMM primal objective being a
    // certified bound.
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