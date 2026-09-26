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

}  // namespace

int main() {
    run("mixed constrained minimization requires branching",
        testMixedConstrainedMinimizationRequiresBranching);
    run("concave maximization", testConcaveMaximization);
    run("non-convex rejected inside engine", testNonConvexRejectedInsideEngine);
    run("infeasible", testInfeasible);
    run("node limit", testNodeLimit);
    run("time limit", testTimeLimit);
    run("unbounded relaxation is not MIQP certificate",
        testUnboundedRelaxationDoesNotClaimMiqpUnbounded);
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
