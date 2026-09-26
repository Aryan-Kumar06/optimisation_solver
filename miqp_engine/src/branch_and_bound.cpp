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
            maxActivity < constraint.lowerBound - tolerance) {
            return true;
        }

        if (std::isfinite(constraint.upperBound) &&
            minActivity > constraint.upperBound + tolerance) {
            return true;
        }
    }

    return false;
}

bool better(double candidate, double incumbent, bool maximize, double tol) {
    return maximize ? candidate > incumbent + tol : candidate < incumbent - tol;
}

bool prunable(double bound, double incumbent, bool maximize, double tol) {
    return maximize ? bound <= incumbent - tol
                : bound >= incumbent + tol;
}

int chooseBranchVariable(const model::Model& model,
                         const std::vector<double>& x,
                         double tolerance) {
    int best = -1;
    double bestViolation = tolerance;
    for (std::size_t j = 0; j < model.variables.size() && j < x.size(); ++j) {
        if (model.variables[j].type == model::VariableType::Continuous) continue;
        const double violation = std::abs(x[j] - std::round(x[j]));
        if (violation > bestViolation) {
            bestViolation = violation;
            best = static_cast<int>(j);
        }
    }
    return best;
}

double modelObjective(const model::Model& model, const std::vector<double>& x) {
    double value = model.objective.offset;
    for (const auto& term : model.objective.linearTerms) {
        value += term.value * x[static_cast<std::size_t>(term.variableIndex)];
    }
    for (const auto& term : model.objective.quadraticTerms) {
        value += term.value * x[static_cast<std::size_t>(term.variableIndex1)] *
                 x[static_cast<std::size_t>(term.variableIndex2)];
    }
    return value;
}

}  // namespace

MiqpResult BranchAndBoundSolver::solve(const model::Model& model,
                                       const MiqpOptions& options) const {
    MiqpResult result;

    if (!model.validate()) {
        result.message = "MIQP model failed structural validation";
        return result;
    }
    const qp::ConvexityCheck convexity = qp::checkConvexity(model);
    if (!convexity.convexForObjectiveSense) {
        result.message = convexity.reason;
        return result;
    }

    const bool maximize = model.objective.sense == model::ObjectiveSense::Maximize;
    const auto start = Clock::now();
    const auto elapsed = [&]() {
        return std::chrono::duration<double>(Clock::now() - start).count();
    };

    Node root;
    root.lower.reserve(model.variables.size());
    root.upper.reserve(model.variables.size());
    for (const auto& variable : model.variables) {
        root.lower.push_back(variable.lowerBound);
        root.upper.push_back(variable.upperBound);
    }

    std::vector<Node> stack;
    stack.push_back(std::move(root));
    bool hasIncumbent = false;
    double incumbent = maximize ? -std::numeric_limits<double>::infinity()
                                :  std::numeric_limits<double>::infinity();
    std::vector<double> incumbentX;

    while (!stack.empty()) {
        if (options.timeLimitSeconds > 0.0 && elapsed() >= options.timeLimitSeconds) {
            result.status = MiqpStatus::TimeLimit;
            result.message = "MIQP time limit reached";
            result.primal = incumbentX;
            if (hasIncumbent) result.objectiveValue = incumbent;
            return result;
        }
        if (options.nodeLimit > 0 && result.nodeCount >= options.nodeLimit) {
            result.status = MiqpStatus::NodeLimit;
            result.message = "MIQP node limit reached";
            result.primal = incumbentX;
            if (hasIncumbent) result.objectiveValue = incumbent;
            return result;
        }

        Node node = std::move(stack.back());
        stack.pop_back();
        ++result.nodeCount;

        model::Model relaxation = model;
        bool inconsistent = false;
        for (std::size_t j = 0; j < relaxation.variables.size(); ++j) {
            relaxation.variables[j].type = model::VariableType::Continuous;
            relaxation.variables[j].lowerBound = node.lower[j];
            relaxation.variables[j].upperBound = node.upper[j];
            if (node.lower[j] > node.upper[j] + options.integralityTolerance) {
                inconsistent = true;
                break;
            }
        }
        if (inconsistent) continue;

        if (nodeRowsDefinitelyInfeasible(model, node, 1e-12)) {
    continue;
}

        qp::QpTranslation translation;
        qp::QpModel qpModel;
        try {
            qpModel = qp::fromModel(relaxation, translation);
        } catch (const std::exception& error) {
            result.status = MiqpStatus::RelaxationFailure;
            result.message = error.what();
            return result;
        }

        qp::AdmmOptions qpOptions;
        const double qpTolerance =
        std::min(options.integralityTolerance * 0.1,
                options.feasibilityTolerance * 0.1);

        qpOptions.primalTolerance = qpTolerance;
        qpOptions.dualTolerance = qpTolerance;
        qpOptions.threadCount = options.threadCount;
        if (options.timeLimitSeconds > 0.0) {
            qpOptions.timeLimitSeconds = std::max(0.0, options.timeLimitSeconds - elapsed());
            if (qpOptions.timeLimitSeconds <= 0.0) {
                result.status = MiqpStatus::TimeLimit;
                result.message = "MIQP time limit reached";
                result.primal = incumbentX;
                if (hasIncumbent) result.objectiveValue = incumbent;
                return result;
            }
        }

        const qp::AdmmResult relaxationResult = qp::QpSolver{}.solve(qpModel, qpOptions);
        result.qpIterations += relaxationResult.iterations;

        if (relaxationResult.status == qp::QpStatus::Infeasible) {
            continue;
        }
        if (relaxationResult.status == qp::QpStatus::TimeLimit) {
            result.status = MiqpStatus::TimeLimit;
            result.message = "QP relaxation reached the MIQP time limit";
            result.primal = incumbentX;
            if (hasIncumbent) result.objectiveValue = incumbent;
            return result;
        }
        if (relaxationResult.status == qp::QpStatus::Unbounded) {
            // An unbounded continuous relaxation is NOT, by itself, a valid
            // certificate that the mixed-integer model is unbounded. Stop with
            // an indeterminate relaxation failure rather than making a false
            // global claim.
            result.status = MiqpStatus::RelaxationFailure;
            result.message =
                "QP relaxation is unbounded; no mixed-integer unboundedness "
                "certificate is available";
            return result;
        }
        if (relaxationResult.status != qp::QpStatus::Optimal) {
            result.status = MiqpStatus::RelaxationFailure;
            result.message = "QP relaxation failed: " + relaxationResult.statusMessage;
            return result;
        }
        if (relaxationResult.primal.size() != model.variables.size()) {
            result.status = MiqpStatus::RelaxationFailure;
            result.message = "QP relaxation returned an incomplete primal solution";
            return result;
        }

        const double bound =
            (translation.objectiveNegated ? -relaxationResult.primalObjective
                                          : relaxationResult.primalObjective) +
            translation.objectiveOffset;
        const double pruningTolerance =
    std::max(options.objectiveTolerance,
             10.0 * qpOptions.primalTolerance);

if (hasIncumbent &&
    prunable(bound,
             incumbent,
             maximize,
             pruningTolerance)) {
    continue;
}

int branch = chooseBranchVariable(
    model,
    relaxationResult.primal,
    options.integralityTolerance);

if (branch < 0) {
    std::vector<double> roundedCandidate =
        relaxationResult.primal;

    for (std::size_t j = 0;
         j < roundedCandidate.size();
         ++j) {
        if (model.variables[j].type !=
            model::VariableType::Continuous) {
            roundedCandidate[j] =
                std::round(roundedCandidate[j]);
        }
    }

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

    if (feasibility.feasible &&
        roundingSafety.feasible) {

        const double candidate =
            modelObjective(model, roundedCandidate);

        if (!hasIncumbent ||
            better(candidate,
                   incumbent,
                   maximize,
                   options.objectiveTolerance)) {

            hasIncumbent = true;
            incumbent = candidate;
            incumbentX = std::move(roundedCandidate);
        }

        continue;
    }

    branch = chooseBranchVariable(
        model,
        relaxationResult.primal,
        std::numeric_limits<double>::epsilon());

    if (branch < 0) {
        continue;
    }
}

        const std::size_t j = static_cast<std::size_t>(branch);
        const double value = relaxationResult.primal[j];
        const double down = std::floor(value);
        const double up = std::ceil(value);

        Node lowerChild = node;
        lowerChild.depth = node.depth + 1;
        lowerChild.upper[j] = std::min(lowerChild.upper[j], down);

        Node upperChild = node;
        upperChild.depth = node.depth + 1;
        upperChild.lower[j] = std::max(upperChild.lower[j], up);

        // LIFO order: push upper first so the down branch is explored first.
        const bool upperTightens =
    upperChild.lower[j] > node.lower[j];

const bool lowerTightens =
    lowerChild.upper[j] < node.upper[j];

if (upperTightens &&
    upperChild.lower[j] <=
        upperChild.upper[j] +
            options.integralityTolerance) {
    stack.push_back(std::move(upperChild));
}

if (lowerTightens &&
    lowerChild.lower[j] <=
        lowerChild.upper[j] +
            options.integralityTolerance) {
    stack.push_back(std::move(lowerChild));
}
    }

    if (!hasIncumbent) {
        result.status = MiqpStatus::Infeasible;
        result.message = "all convex QP relaxations were infeasible or exhausted";
        return result;
    }

    result.status = MiqpStatus::Optimal;
    result.message = "convex MIQP branch-and-bound completed";
    result.primal = std::move(incumbentX);
    result.objectiveValue = incumbent;
    return result;
}

}  // namespace miqp
