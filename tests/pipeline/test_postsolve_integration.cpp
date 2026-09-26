#include "solver/orchestrator.h"
#include "presolve/presolver.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
constexpr double INF = std::numeric_limits<double>::infinity();

void require(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}
void near(double actual, double expected) {
    require(std::isfinite(actual) && std::abs(actual - expected) < 1e-5,
            "expected " + std::to_string(expected) + ", got " + std::to_string(actual));
}
void vectorNear(const std::vector<double>& actual, const std::vector<double>& expected) {
    require(actual.size() == expected.size(), "wrong vector size");
    for (std::size_t i = 0; i < actual.size(); ++i) near(actual[i], expected[i]);
}
model::Variable var(const char* name, double lower = 0, double upper = INF) {
    return {name, model::VariableType::Continuous, lower, upper};
}
model::Constraint row(const char* name, double lower, double upper,
                      std::vector<model::LinearTerm> terms) {
    return {name, lower, upper, std::move(terms)};
}
void optimal(const solver::SolveResult& result) {
    require(result.status == solver::SolveStatus::Optimal, result.message);
    require(result.hasPrimal, "optimal result must publish a complete primal");
}
void noDuals(const solver::SolveResult& result, const std::string& reason) {
    require(!result.hasDuals && result.constraintDuals.empty() && result.reducedCosts.empty(),
            "unavailable duals must have empty vectors");
    require(result.dualsUnavailableReason.find(reason) != std::string::npos,
            "missing dual diagnostic: " + result.dualsUnavailableReason);
}

void lp_engines() {
    // A fixed variable and an empty row precede surviving indices. The public
    // result must restore them and move y's bound multiplier onto the source row.
    model::Model m;
    m.variables = {var("fixed", 2, 2), var("x"), var("y")};
    m.objective.sense = model::ObjectiveSense::Maximize;
    m.objective.offset = 7;
    m.objective.linearTerms = {{0, 5}, {1, 3}, {2, 4}};
    m.constraints = {row("empty", -INF, 1, {}),
                     row("machine", -INF, 28, {{1, 4}, {2, 5}}),
                     row("slack", -INF, 10, {{1, 1}})};
    for (auto engine : {solver::Engine::DualSimplex, solver::Engine::Pdlp}) {
        solver::SolverOptions options;
        options.forceEngine = engine;
        auto result = solver::solve(m, options);
        optimal(result);
        require(result.executedEngine == engine, "requested engine must actually run");
        vectorNear(result.variableValues, {2, 0, 5.6});
        near(result.objectiveValue, 39.4);
        require(result.hasDuals, result.dualsUnavailableReason);
        vectorNear(result.constraintDuals, {0, 0.8, 0});
        vectorNear(result.reducedCosts, {5, -0.2, 0});
        require(result.dualsUnavailableReason.empty(), "available duals have no diagnostic");
        require(result.maxDualResidual < 1e-5, "validated original multipliers");
        require(result.reducedVariableCount == 2 && result.reducedConstraintCount == 1,
                "reduction statistics remain reduced dimensions");

        const auto p = presolve::Presolver().run(m);
        const auto reduced = solver::solveReduced(p.model, solver::classify(m), options);
        optimal(reduced);
        require(reduced.variableValues.size() == 2 && reduced.constraintDuals.size() == 1 &&
                reduced.reducedCosts.size() == 2, "solveReduced keeps its input model's coordinates");
        near(reduced.objectiveValue, result.objectiveValue);
        if (engine == solver::Engine::DualSimplex) {
            // In supplied coordinates the derived upper bound keeps its price.
            // Only solve(original) may transfer it to the originating row.
            vectorNear(reduced.constraintDuals, {0.75});
            vectorNear(reduced.reducedCosts, {0, 0.25});
        }
    }
}

void qp_signs() {
    for (double sense : {1.0, -1.0}) {
        for (int side : {0, 1, 2}) { // lower, upper, equality
            const double c = side == 1 ? -6 : 0;
            model::Model m;
            m.variables = {var("x", 0, 10), var("y", 0, 10)};
            m.objective.sense = sense > 0 ? model::ObjectiveSense::Minimize
                                          : model::ObjectiveSense::Maximize;
            m.objective.offset = 9;
            m.objective.linearTerms = {{0, sense*c}, {1, sense*c}};
            m.objective.quadraticTerms = {{0, 0, sense}, {1, 1, sense}};
            m.constraints = {row("sum", side == 1 ? -INF : 4,
                                 side == 0 ? INF : 4, {{0, 1}, {1, 1}})};
            for (const auto& result : {solver::solve(m),
                                      solver::solveReduced(m, solver::classify(m))}) {
                optimal(result);
                require(result.executedEngine == solver::Engine::Qp, "QP engine must run");
                vectorNear(result.variableValues, {2, 2});
                near(result.objectiveValue, 9 + sense*(8 + 4*c));
                require(result.hasDuals, result.dualsUnavailableReason);
                vectorNear(result.constraintDuals, {sense*(4 + c)});
                vectorNear(result.reducedCosts, {0, 0});
            }
        }
    }
}

void all_eliminated() {
    // f=7+5x+2y^2+xy at x=3,y=2 is 36. grad=(7,11), row 2y=4
    // has price 5.5. The engine input is empty, yet both originals must return.
    model::Model m;
    m.variables = {var("x", 3, 3), var("y", 0, 10)};
    m.objective.offset = 7;
    m.objective.linearTerms = {{0, 5}};
    m.objective.quadraticTerms = {{1, 1, 2}, {0, 1, 1}};
    m.constraints = {row("fix_y", 4, 4, {{1, 2}})};
    const auto result = solver::solve(m);
    optimal(result);
    require(result.executedEngine == solver::Engine::Trivial, "empty model uses trivial path");
    require(result.reducedVariableCount == 0 && result.reducedConstraintCount == 0,
            "presolve eliminated everything");
    vectorNear(result.variableValues, {3, 2});
    near(result.objectiveValue, 36);
    require(result.hasDuals, result.dualsUnavailableReason);
    vectorNear(result.constraintDuals, {5.5});
    vectorNear(result.reducedCosts, {7, 0});
}

void bound_only() {
    model::Model m;
    m.variables = {var("x", 0, 10), var("y", 0, 5)};
    m.objective.linearTerms = {{0, 2}, {1, -3}};
    const auto result = solver::solve(m);
    optimal(result);
    require(result.hasDuals && result.constraintDuals.empty(), "empty dual vector is valid");
    vectorNear(result.variableValues, {0, 5});
    vectorNear(result.reducedCosts, {2, -3});
    model::Model empty;
    empty.objective.offset = 4;
    const auto nothing = solver::solve(empty);
    optimal(nothing);
    require(nothing.hasDuals && nothing.variableValues.empty(), "valid empty solution is explicit");
    near(nothing.objectiveValue, 4);
}

void integer_models() {
    model::Model m;
    m.variables = {var("x", 0, 1), var("y", 0, 1)};
    for (auto& v : m.variables) v.type = model::VariableType::Binary;
    m.objective.sense = model::ObjectiveSense::Maximize;
    m.objective.linearTerms = {{0, 1}, {1, 1}};
    m.constraints = {row("capacity", -INF, 3, {{0, 2}, {1, 2}})};
    auto result = solver::solve(m);
    optimal(result);
    near(result.objectiveValue, 1);
    require(result.integralityRespected, "integer point is restored");
    noDuals(result, "integer models");

    solver::SolverOptions relaxed;
    relaxed.forceEngine = solver::Engine::DualSimplex;
    result = solver::solve(m, relaxed);
    optimal(result);
    near(result.objectiveValue, 1.5);
    require(!result.integralityRespected && result.maxIntegralityViolation > 0.4,
            "forced relaxation remains visibly fractional");
    noDuals(result, "integer models");

    m.variables = {var("fixed", 3, 3)};
    m.variables[0].type = model::VariableType::Integer;
    m.constraints.clear();
    m.objective.linearTerms = {{0, 2}};
    result = solver::solve(m);
    optimal(result);
    vectorNear(result.variableValues, {3});
    require(result.reducedVariableCount == 0, "integer variable was eliminated");
    noDuals(result, "integer models");
}

void unavailable_duals() {
    // The singleton coincides with an ORIGINAL bound; it emitted no tightening.
    // The metadata gate must withhold the sensitivity without losing the primal.
    model::Model m;
    m.variables = {var("x", 0, 3)};
    m.objective.sense = model::ObjectiveSense::Maximize;
    m.objective.linearTerms = {{0, 6}};
    m.constraints = {row("same_bound", -INF, 6, {{0, 2}})};
    const auto result = solver::solve(m);
    optimal(result);
    vectorNear(result.variableValues, {3});
    near(result.objectiveValue, 18);
    noDuals(result, "provenance");
}

void failure_statuses() {
    model::Model m;
    m.variables = {var("x")};
    m.objective.linearTerms = {{0, -1}};
    auto result = solver::solve(m);
    require(result.status == solver::SolveStatus::Unbounded && !result.hasPrimal &&
            result.variableValues.empty(), "unbounded result must not expose a partial point");
    noDuals(result, "optimal");
    m.constraints = {row("impossible", -INF, -1, {{0, 1}})};
    result = solver::solve(m);
    require(result.status == solver::SolveStatus::Infeasible && !result.hasPrimal &&
            !result.hasDuals && result.variableValues.empty(), "presolve infeasibility has no point");
    m.constraints.clear();
    m.objective.linearTerms = {{4, 1}};
    result = solver::solve(m);
    require(result.status == solver::SolveStatus::InvalidModel && !result.hasPrimal,
            "invalid model has no point");
}

void limit_no_solution() {
    model::Model m;
    m.variables = {var("x", 0, 10), var("y", 0, 10)};
    m.objective.linearTerms = {{0, 2}, {1, 3}};
    m.constraints = {row("demand", 4, INF, {{0, 1}, {1, 1}})};
    solver::SolverOptions options;
    options.forceEngine = solver::Engine::Pdlp;
    // Expire before the first iteration; the zero starting point is infeasible.
    options.timeLimitSeconds = 1e-30;
    const auto result = solver::solve(m, options);
    require(result.status == solver::SolveStatus::LimitReached, "time limit must remain a limit");
    require(!result.hasPrimal && result.variableValues.empty(), "infeasible iterate is withheld");
    noDuals(result, "primal");
}

void limit_feasible_solution() {
    model::Model m;
    m.variables = {var("fixed", 3, 3), var("x", 0, 10), var("y", 0, 10)};
    m.objective.linearTerms = {{0, 4}, {1, -1}, {2, -2}};
    m.constraints = {row("capacity", -INF, 5, {{1, 1}, {2, 1}})};
    solver::SolverOptions options;
    options.forceEngine = solver::Engine::Pdlp;
    options.timeLimitSeconds = 1e-30;
    const auto result = solver::solve(m, options);
    require(result.status == solver::SolveStatus::LimitReached && result.hasPrimal,
            "feasible starting point must be reconstructed on a limit");
    vectorNear(result.variableValues, {3, 0, 0});
    near(result.objectiveValue, 12);
    noDuals(result, "optimal solution");
}
void reduced_api() {
    // No presolve: fixed variables retain their supplied indices and costs.
    model::Model m;
    m.variables = {var("fixed", 3, 3), var("lower", 0, 10), var("upper", 0, 5)};
    m.objective.offset = 7;
    m.objective.linearTerms = {{0, 4}, {1, 2}, {2, -3}};
    auto result = solver::solveReduced(m, solver::classify(m));
    optimal(result);
    require(result.hasDuals && result.constraintDuals.empty(), "bound-only dual input is complete");
    vectorNear(result.variableValues, {3, 0, 5});
    vectorNear(result.reducedCosts, {4, 2, -3});
    near(result.objectiveValue, 4);
    require(result.reducedVariableCount == 3, "solveReduced must not eliminate fixed variables");

    model::Model empty;
    empty.objective.offset = 9;
    result = solver::solveReduced(empty, solver::classify(empty));
    optimal(result);
    require(result.hasDuals && result.variableValues.empty() && result.reducedCosts.empty(),
            "empty supplied model has a complete solution");
    near(result.objectiveValue, 9);

    m.variables = {var("x", 0, 1), var("y", 0, 1)};
    for (auto& v : m.variables) v.type = model::VariableType::Binary;
    m.objective.offset = 0;
    m.objective.sense = model::ObjectiveSense::Maximize;
    m.objective.linearTerms = {{0, 1}, {1, 1}};
    m.constraints = {row("capacity", -INF, 3, {{0, 2}, {1, 2}})};
    result = solver::solveReduced(m, solver::classify(m));
    optimal(result);
    near(result.objectiveValue, 1);
    require(result.integralityRespected, "direct integer solve respects integrality");
    noDuals(result, "integer models");
    solver::SolverOptions options;
    options.forceEngine = solver::Engine::DualSimplex;
    result = solver::solveReduced(m, solver::classify(m), options);
    optimal(result);
    near(result.objectiveValue, 1.5);
    require(!result.integralityRespected, "direct forced relaxation reports fractional point");
    noDuals(result, "integer models");

    for (auto& v : m.variables) v.type = model::VariableType::Continuous;
    options.forceEngine = solver::Engine::Pdlp;
    options.timeLimitSeconds = 1e-30;
    result = solver::solveReduced(m, solver::classify(m), options);
    require(result.status == solver::SolveStatus::LimitReached && result.hasPrimal,
            "direct limit keeps a feasible iterate");
    vectorNear(result.variableValues, {0, 0});
    noDuals(result, "optimal solution");
    m.constraints = {row("demand", 1, INF, {{0, 1}, {1, 1}})};
    result = solver::solveReduced(m, solver::classify(m), options);
    require(result.status == solver::SolveStatus::LimitReached && !result.hasPrimal &&
            result.variableValues.empty(), "direct limit withholds an infeasible iterate");
    noDuals(result, "primal");

    m.variables = {var("x")};
    m.constraints.clear();
    m.objective.sense = model::ObjectiveSense::Minimize;
    m.objective.linearTerms = {{0, -1}};
    result = solver::solveReduced(m, solver::classify(m));
    require(result.status == solver::SolveStatus::Unbounded && !result.hasPrimal &&
            result.variableValues.empty(), "direct unbounded solve withholds partial point");
    noDuals(result, "optimal");
}
}  // namespace

int main(int argc, char** argv) {
    struct Test { const char* name; void (*run)(); };
    const Test tests[] = {{"lp_engines", lp_engines}, {"qp_signs", qp_signs},
        {"all_eliminated", all_eliminated}, {"bound_only", bound_only},
        {"integer_models", integer_models}, {"unavailable_duals", unavailable_duals},
        {"failure_statuses", failure_statuses}, {"limit_no_solution", limit_no_solution},
        {"limit_feasible_solution", limit_feasible_solution}, {"reduced_api", reduced_api}};
    int count = 0, failures = 0;
    for (const auto& test : tests) {
        if (argc > 1 && std::string(argv[1]) != test.name) continue;
        ++count;
        try {
            test.run();
            std::cout << "[PASSED] " << test.name << '\n';
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "[FAILED] " << test.name << ": " << error.what() << '\n';
        }
    }
    return count > 0 && failures == 0 ? 0 : 1;
}
