#include "miqp/branch_and_bound.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

constexpr double INF = std::numeric_limits<double>::infinity();
int failures = 0;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

model::Variable var(const char* name, model::VariableType type, double lo, double hi) {
    model::Variable v;
    v.name = name; v.type = type; v.lowerBound = lo; v.upperBound = hi;
    return v;
}

model::Constraint row(const char* name, double lo, double hi,
                      std::initializer_list<model::LinearTerm> terms) {
    model::Constraint c;
    c.name = name; c.lowerBound = lo; c.upperBound = hi; c.linearTerms = terms;
    return c;
}

// Relaxation optimum is (x,y)=(1.5,0.5), so x is genuinely fractional and
// branch-and-bound must branch. Rows exercise equality, >= and <= senses.
model::Model branchingModel() {
    model::Model m;
    m.name = "mixed_branching";
    m.variables = {
        var("x", model::VariableType::Integer, 0.0, 3.0),
        var("y", model::VariableType::Continuous, 0.0, 2.0)
    };
    m.constraints = {
        row("eq", 1.0, 1.0, {{0, 1.0}, {1, -1.0}}),
        row("ge", 2.0, INF, {{0, 1.0}, {1, 1.0}}),
        row("le", -INF, 1.0, {{1, 1.0}})
    };
    m.objective.sense = model::ObjectiveSense::Minimize;
    m.objective.offset = 2.5;
    m.objective.linearTerms = {{0, -3.0}, {1, -1.0}};
    m.objective.quadraticTerms = {{0, 0, 1.0}, {1, 1, 1.0}};
    return m;
}

void testMixedConstrainedMinimizationRequiresBranching() {
    const auto r = miqp::BranchAndBoundSolver{}.solve(branchingModel());
    require(r.status == miqp::MiqpStatus::Optimal, "mixed MIQP should solve");
    require(r.nodeCount >= 2, "fractional root must require branching");
    require(r.primal.size() == 2, "complete primal expected");
    require(std::abs(r.primal[0] - std::round(r.primal[0])) < 1e-7,
            "integer variable must be integral");
    require(std::abs(r.objectiveValue - 0.5) < 2e-4,
            "known optimum is 0.5");
}

void testConcaveMaximization() {
    model::Model m;
    m.variables = {var("x", model::VariableType::Integer, 0.0, 3.0)};
    m.objective.sense = model::ObjectiveSense::Maximize;
    // -(x-1.5)^2 = -x^2 + 3x - 2.25
    m.objective.offset = -2.25;
    m.objective.linearTerms = {{0, 3.0}};
    m.objective.quadraticTerms = {{0, 0, -1.0}};
    const auto r = miqp::BranchAndBoundSolver{}.solve(m);
    require(r.status == miqp::MiqpStatus::Optimal, "concave maximization accepted");
    require(std::abs(r.objectiveValue + 0.25) < 2e-4,
            "concave max known optimum is -0.25");
}

void testNonConvexRejectedInsideEngine() {
    model::Model m;
    m.variables = {var("x", model::VariableType::Integer, -2.0, 2.0)};
    m.objective.sense = model::ObjectiveSense::Minimize;
    m.objective.quadraticTerms = {{0, 0, -1.0}};
    const auto r = miqp::BranchAndBoundSolver{}.solve(m);
    require(r.status == miqp::MiqpStatus::RelaxationFailure,
            "non-convex minimization must be rejected");
}

void testInfeasible() {
    model::Model m;
    m.variables = {
        var("x", model::VariableType::Integer, 0.0, 2.0),
        var("y", model::VariableType::Continuous, 0.0, 2.0)
    };
    m.constraints = {
        row("lo", 3.0, INF, {{0, 1.0}, {1, 1.0}}),
        row("hi", -INF, 1.0, {{0, 1.0}, {1, 1.0}})
    };
    m.objective.quadraticTerms = {{0, 0, 1.0}, {1, 1, 1.0}};
    const auto r = miqp::BranchAndBoundSolver{}.solve(m);
    require(r.status == miqp::MiqpStatus::Infeasible, "infeasible MIQP reported");
}

void testNodeLimit() {
    miqp::MiqpOptions o;
    o.nodeLimit = 1;
    const auto r = miqp::BranchAndBoundSolver{}.solve(branchingModel(), o);
    require(r.status == miqp::MiqpStatus::NodeLimit, "node limit must stop tree");
    require(r.nodeCount == 1, "exactly root node processed at limit 1");
}

void testTimeLimit() {
    miqp::MiqpOptions o;
    o.timeLimitSeconds = 1e-12;
    const auto r = miqp::BranchAndBoundSolver{}.solve(branchingModel(), o);
    require(r.status == miqp::MiqpStatus::TimeLimit, "time limit must stop tree");
}

void testNearIntegralRoundedCandidateIsRevalidatedAndBranched() {
    model::Model m;
    m.name = "near_integral_rounding";

    m.variables = {
        var("x", model::VariableType::Integer, 0.0, 2.0)
    };

    m.constraints = {
        row("strict_upper",
            -INF,
            0.99999995,
            {{0, 1.0}})
    };

    m.objective.sense =
        model::ObjectiveSense::Minimize;

    m.objective.offset =
        0.9999998800000036;

    m.objective.linearTerms = {
        {0, -1.99999988}
    };

    m.objective.quadraticTerms = {
        {0, 0, 1.0}
    };

    const auto r =
        miqp::BranchAndBoundSolver{}.solve(m);

    require(
        r.status == miqp::MiqpStatus::Optimal,
        "near-integral MIQP should continue after invalid rounding");

    require(
        r.nodeCount >= 2,
        "invalid rounded root candidate must cause branching");

    require(
        r.primal.size() == 1,
        "complete primal expected");

    require(
        std::abs(r.primal[0]) < 1e-9,
        "x=1 is infeasible, so valid optimum must be x=0");
}

void testBinaryVariablePath() {
    model::Model m;
    m.name = "binary_regression";

    m.variables = {
        var("b", model::VariableType::Binary, 0.0, 1.0)
    };

    m.objective.sense =
        model::ObjectiveSense::Minimize;

    m.objective.offset = 0.64;

    m.objective.linearTerms = {
        {0, -1.6}
    };

    m.objective.quadraticTerms = {
        {0, 0, 1.0}
    };

    const auto r =
        miqp::BranchAndBoundSolver{}.solve(m);

    require(
        r.status == miqp::MiqpStatus::Optimal,
        "binary MIQP should solve");

    require(
        r.nodeCount >= 2,
        "binary relaxation should require branching");

    require(
        r.primal.size() == 1,
        "binary primal expected");

    require(
        std::abs(r.primal[0] - 1.0) < 1e-9,
        "binary optimum should be b=1");

    require(
        std::abs(r.objectiveValue - 0.04) < 2e-4,
        "binary objective should match known optimum");
}

void testUnboundedRelaxationDoesNotClaimMiqpUnbounded() {
    model::Model m;
    m.variables = {
        var("x", model::VariableType::Integer, 0.0, 1.0),
        var("y", model::VariableType::Continuous, 0.0, INF)
    };
    m.objective.linearTerms = {{1, -1.0}};
    m.objective.quadraticTerms = {{0, 0, 1.0}};
    const auto r = miqp::BranchAndBoundSolver{}.solve(m);
    require(r.status == miqp::MiqpStatus::RelaxationFailure,
            "unbounded QP relaxation must not become a global MIQP unbounded claim");
    require(r.message.find("certificate") != std::string::npos,
            "failure should explain missing MI certificate");
}

void run(const char* name, void (*test)()) {
    try { test(); std::cout << "  pass  " << name << '\n'; }
    catch (const std::exception& e) {
        ++failures; std::cout << "  FAIL  " << name << ": " << e.what() << '\n';
    }
}

void testNoUnsafePrimalObjectivePruningMinimization() {
    model::Model m;
    m.name = "no_unsafe_pruning_min";

    m.variables = {
        var("x", model::VariableType::Integer, 0.0, 3.0)
    };

    // (x - 1.6)^2
    //
    // = x^2 - 3.2x + 2.56
    //
    // Continuous relaxation optimum:
    //     x = 1.6
    //
    // Branches:
    //     x <= 1  -> best integer x = 1, objective = 0.36
    //     x >= 2  -> best integer x = 2, objective = 0.16
    //
    // The down branch is explored first and gives an incumbent of 0.36.
    // The up branch contains the true global optimum and MUST NOT be
    // discarded using an uncertified QP primal objective.

    m.objective.sense =
        model::ObjectiveSense::Minimize;

    m.objective.offset = 2.56;

    m.objective.linearTerms = {
        {0, -3.2}
    };

    m.objective.quadraticTerms = {
        {0, 0, 1.0}
    };

    const auto r =
        miqp::BranchAndBoundSolver{}.solve(m);

    require(
        r.status == miqp::MiqpStatus::Optimal,
        "minimization pruning regression should solve");

    require(
        r.primal.size() == 1,
        "minimization regression should return one variable");

    require(
        std::abs(r.primal[0] - 2.0) < 1e-8,
        "global minimization optimum must be x=2");

    require(
        std::abs(r.objectiveValue - 0.16) < 2e-4,
        "minimization objective must equal 0.16");

    require(
        r.nodeCount >= 3,
        "solver must explore the branch containing the better integer solution");
}

void testNoUnsafePrimalObjectivePruningMaximization() {
    model::Model m;
    m.name = "no_unsafe_pruning_max";

    m.variables = {
        var("x", model::VariableType::Integer, 0.0, 3.0)
    };

    // -(x - 1.6)^2
    //
    // = -x^2 + 3.2x - 2.56
    //
    // This is concave, so it is a valid quadratic maximization.
    //
    // Continuous relaxation optimum:
    //     x = 1.6
    //
    // Integer branches:
    //     x <= 1 -> x = 1, objective = -0.36
    //     x >= 2 -> x = 2, objective = -0.16
    //
    // For maximization, -0.16 is better than -0.36.
    // Therefore the second branch again contains the global optimum.

    m.objective.sense =
        model::ObjectiveSense::Maximize;

    m.objective.offset = -2.56;

    m.objective.linearTerms = {
        {0, 3.2}
    };

    m.objective.quadraticTerms = {
        {0, 0, -1.0}
    };

    const auto r =
        miqp::BranchAndBoundSolver{}.solve(m);

    require(
        r.status == miqp::MiqpStatus::Optimal,
        "maximization pruning regression should solve");

    require(
        r.primal.size() == 1,
        "maximization regression should return one variable");

    require(
        std::abs(r.primal[0] - 2.0) < 1e-8,
        "global maximization optimum must be x=2");

    require(
        std::abs(r.objectiveValue - (-0.16)) < 2e-4,
        "maximization objective must equal -0.16");

    require(
        r.nodeCount >= 3,
        "solver must explore the branch containing the better maximization solution");
}

void testBetterSolutionInLaterMixedBranchIsNotPruned() {
    model::Model m;
    m.name = "mixed_later_branch_optimum";

    m.variables = {
        var("x", model::VariableType::Integer, 0.0, 3.0),
        var("y", model::VariableType::Continuous, 0.0, 3.0)
    };

    // Force y = x.
    m.constraints = {
        row("link",
            0.0,
            0.0,
            {
                {0, -1.0},
                {1,  1.0}
            })
    };

    // (x - 1.6)^2 + (y - 1.6)^2
    //
    // = x^2 + y^2
    //   - 3.2x - 3.2y
    //   + 5.12
    //
    // Since y = x, the continuous relaxation optimum is:
    //
    //     x = y = 1.6
    //
    // Integer x <= 1:
    //     x = y = 1
    //     objective = 0.72
    //
    // Integer x >= 2:
    //     x = y = 2
    //     objective = 0.32
    //
    // So once again the later branch contains the true optimum.

    m.objective.sense =
        model::ObjectiveSense::Minimize;

    m.objective.offset = 5.12;

    m.objective.linearTerms = {
        {0, -3.2},
        {1, -3.2}
    };

    m.objective.quadraticTerms = {
        {0, 0, 1.0},
        {1, 1, 1.0}
    };

    const auto r =
        miqp::BranchAndBoundSolver{}.solve(m);

    require(
        r.status == miqp::MiqpStatus::Optimal,
        "mixed later-branch regression should solve");

    require(
        r.primal.size() == 2,
        "mixed regression should return two variables");

    require(
        std::abs(r.primal[0] - 2.0) < 1e-8,
        "mixed regression integer optimum must be x=2");

    require(
        std::abs(r.primal[1] - 2.0) < 2e-4,
        "equality constraint should give y=2");

    require(
        std::abs(r.objectiveValue - 0.32) < 4e-4,
        "mixed regression objective must equal 0.32");

    require(
        r.nodeCount >= 3,
        "later mixed branch containing global optimum must be explored");
}

void testUnboundedIntegerDomainRejected() {
    model::Model m;
    m.name = "unbounded_integer_domain";

    m.variables = {
        var(
            "x",
            model::VariableType::Integer,
            0.0,
            INF)
    };

    m.objective.sense =
        model::ObjectiveSense::Minimize;

    m.objective.linearTerms = {
        {0, 1.0}
    };

    const auto r =
        miqp::BranchAndBoundSolver{}.solve(m);

    require(
        r.status ==
            miqp::MiqpStatus::RelaxationFailure,
        "MIQP with unbounded integer domain "
        "must be rejected explicitly");

    require(
        r.message.find(
            "finite lower and upper bounds") !=
            std::string::npos,
        "rejection should explain finite "
        "integer-bound requirement");
}

}  // namespace

int main() {
    run("mixed constrained minimization requires branching",
        testMixedConstrainedMinimizationRequiresBranching);

    run("concave maximization",
        testConcaveMaximization);

    run("non-convex rejected inside engine",
        testNonConvexRejectedInsideEngine);

    run("infeasible",
        testInfeasible);

    run("node limit",
        testNodeLimit);

    run("time limit",
        testTimeLimit);

    run("unbounded relaxation is not MIQP certificate",
        testUnboundedRelaxationDoesNotClaimMiqpUnbounded);

    run("near-integral rounded candidate is revalidated and branched",
        testNearIntegralRoundedCandidateIsRevalidatedAndBranched);

    run("binary variable path",
        testBinaryVariablePath);

    run("no unsafe primal objective pruning - minimization",
        testNoUnsafePrimalObjectivePruningMinimization);

    run("no unsafe primal objective pruning - maximization",
        testNoUnsafePrimalObjectivePruningMaximization);

    run("better solution in later mixed branch is not pruned",
        testBetterSolutionInLaterMixedBranchIsNotPruned);

    run("unbounded integer domain rejected",
        testUnboundedIntegerDomainRejected);

    return failures == 0
        ? EXIT_SUCCESS
        : EXIT_FAILURE;
}
