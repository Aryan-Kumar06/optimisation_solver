#include "presolve/presolver.h"
#include "solver/classifier.h"
#include "solver/dispatcher.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
int failures = 0;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

model::Variable variable(const std::string& name, model::VariableType type,
                         double lower, double upper) {
    model::Variable v;
    v.name = name;
    v.type = type;
    v.lowerBound = lower;
    v.upperBound = upper;
    return v;
}

model::Constraint constraint(const std::string& name, double lower, double upper,
                             std::vector<model::LinearTerm> terms) {
    model::Constraint c;
    c.name = name;
    c.lowerBound = lower;
    c.upperBound = upper;
    c.linearTerms = std::move(terms);
    return c;
}

// ---------------------------------------------------------------- classifier

void testClassifiesLp() {
    model::Model m;
    m.variables = {variable("x", model::VariableType::Continuous, 0, 10),
                   variable("y", model::VariableType::Continuous, 0, 10)};
    m.constraints = {constraint("c0", -kInf, 5.0, {{0, 1.0}, {1, 1.0}})};
    const auto c = solver::classify(m);
    require(c.problemClass == solver::ProblemClass::LP, "pure continuous is LP");
    require(c.hints.numContinuous == 2, "continuous count");
    require(c.hints.numNonzeros == 2, "nonzero count");
}

void testClassifiesMilpAndBinary() {
    model::Model m;
    m.variables = {variable("b", model::VariableType::Binary, 0, 1),
                   variable("x", model::VariableType::Continuous, 0, 10)};
    m.constraints = {constraint("c0", -kInf, 5.0, {{0, 1.0}, {1, 1.0}})};
    const auto c = solver::classify(m);
    require(c.problemClass == solver::ProblemClass::MILP, "binary makes it MILP");
    require(c.hints.numBinary == 1, "binary counted");
}

// An Integer variable bounded to [0,1] is a binary however it was declared.
void testIntegerBoundedToUnitIsCountedBinary() {
    model::Model m;
    m.variables = {variable("z", model::VariableType::Integer, 0, 1),
                   variable("x", model::VariableType::Continuous, 0, 10)};
    m.constraints = {constraint("c0", -kInf, 5.0, {{0, 1.0}, {1, 1.0}})};
    const auto c = solver::classify(m);
    require(c.hints.numBinary == 1, "0/1 integer counted as binary");
    require(c.hints.numInteger == 0, "and not double counted as general integer");
}

void testClassifiesQpAndMiqp() {
    model::Model m;
    m.variables = {variable("x", model::VariableType::Continuous, 0, 10)};
    m.constraints = {constraint("c0", -kInf, 5.0, {{0, 1.0}})};
    m.objective.quadraticTerms.push_back({0, 0, 2.0});
    require(solver::classify(m).problemClass == solver::ProblemClass::QP,
            "quadratic objective is QP");

    m.variables.push_back(variable("b", model::VariableType::Binary, 0, 1));
    require(solver::classify(m).problemClass == solver::ProblemClass::MIQP,
            "quadratic objective plus integrality is MIQP");
}

// Node-arc incidence: every column is exactly one +1 and one -1.
void testDetectsNetworkStructure() {
    model::Model m;
    m.variables = {variable("a", model::VariableType::Continuous, 0, 10),
                   variable("b", model::VariableType::Continuous, 0, 10)};
    m.constraints = {
        constraint("n0", 0.0, 0.0, {{0, 1.0}, {1, 1.0}}),
        constraint("n1", 0.0, 0.0, {{0, -1.0}, {1, -1.0}})};
    require(solver::classify(m).hints.hasNetworkStructure,
            "incidence matrix should be flagged as a network");

    m.constraints[1].linearTerms[0].value = -2.0;
    require(!solver::classify(m).hints.hasNetworkStructure,
            "a -2 coefficient is not an incidence matrix");
}

void testDetectsBigM() {
    model::Model m;
    m.variables = {variable("x", model::VariableType::Continuous, 0, 1000),
                   variable("b", model::VariableType::Binary, 0, 1)};
    // x - 1000 b <= 0
    m.constraints = {constraint("indicator", -kInf, 0.0, {{0, 1.0}, {1, -1000.0}})};
    const auto hints = solver::classify(m).hints;
    require(hints.hasBigM, "indicator row should be flagged big-M");
    require(std::abs(hints.maxBigM - 1000.0) < 1e-9, "big-M magnitude reported");
}

void testDetectsSetPartitioningAndSymmetry() {
    model::Model m;
    for (int j = 0; j < 3; ++j) {
        m.variables.push_back(
            variable("b" + std::to_string(j), model::VariableType::Binary, 0, 1));
    }
    m.constraints = {constraint("p0", 1.0, 1.0, {{0, 1.0}, {1, 1.0}, {2, 1.0}})};
    const auto hints = solver::classify(m).hints;
    require(hints.hasSetPartitioning, "unit equality over binaries is set partitioning");
    // All three columns are structurally identical, so they form one group.
    require(hints.symmetricGroups == 1,
            "identical columns form a symmetric group, got " +
            std::to_string(hints.symmetricGroups));
}

void testCoefficientRangeRatio() {
    model::Model m;
    m.variables = {variable("x", model::VariableType::Continuous, 0, 1),
                   variable("y", model::VariableType::Continuous, 0, 1)};
    m.constraints = {constraint("c0", -kInf, 1.0, {{0, 1e-3}, {1, 1e3}})};
    const double ratio = solver::classify(m).hints.coefRangeRatio;
    require(std::abs(ratio - 1e6) < 1.0, "range ratio should be 1e6, got " +
            std::to_string(ratio));
}

// ---------------------------------------------------------------- dispatcher

presolve::PresolveResult feasibleResult() {
    presolve::PresolveResult r;
    r.infeasible = false;
    return r;
}

model::Model lpOfSize(std::size_t rows, std::size_t columns) {
    model::Model m;
    for (std::size_t j = 0; j < columns; ++j) {
        m.variables.push_back(
            variable("x" + std::to_string(j), model::VariableType::Continuous, 0, 10));
    }
    for (std::size_t i = 0; i < rows; ++i) {
        m.constraints.push_back(constraint(
            "c" + std::to_string(i), -kInf, 5.0,
            {{static_cast<int>(i % columns), 1.0}}));
    }
    return m;
}

void testInfeasibleShortCircuits() {
    presolve::PresolveResult r;
    r.infeasible = true;
    const auto d = solver::dispatch(lpOfSize(3, 3), solver::classify(lpOfSize(3, 3)), r);
    require(d.engine == solver::Engine::Infeasible, "presolve infeasibility wins");
}

void testForcedEngineIsHonoured() {
    solver::SolverOptions o;
    o.forceEngine = solver::Engine::Pdlp;
    const model::Model m = lpOfSize(3, 3);
    const auto d = solver::dispatch(m, solver::classify(m), feasibleResult(), o);
    require(d.engine == solver::Engine::Pdlp, "explicit override honoured");
}

// A forced override must beat even the integrality rule: solving the relaxation
// is a legitimate request.
void testForcedEngineBeatsIntegrality() {
    model::Model m = lpOfSize(3, 3);
    m.variables[0].type = model::VariableType::Integer;
    solver::SolverOptions o;
    o.forceEngine = solver::Engine::DualSimplex;
    const auto d = solver::dispatch(m, solver::classify(m), feasibleResult(), o);
    require(d.engine == solver::Engine::DualSimplex,
            "override outranks the integrality rule");
}

void testIntegralityRoutesToBranchAndCut() {
    model::Model m = lpOfSize(5, 5);
    m.variables[2].type = model::VariableType::Binary;
    m.variables[2].upperBound = 1.0;
    const auto d = solver::dispatch(m, solver::classify(m), feasibleResult());
    require(d.engine == solver::Engine::BranchAndCut, "integrality routes to B&C");
}

// The whole point of dispatching after presolve: a MILP whose integer variables
// were all eliminated must not start a tree search.
void testMilpReducedToLpRoutesToLpEngine() {
    model::Model original = lpOfSize(5, 5);
    original.variables[2].type = model::VariableType::Binary;
    original.variables[2].upperBound = 1.0;
    const auto classification = solver::classify(original);
    require(classification.problemClass == solver::ProblemClass::MILP,
            "original is a MILP");

    model::Model reduced = original;
    reduced.variables[2].type = model::VariableType::Continuous;  // presolve fixed it

    const auto d = solver::dispatch(reduced, classification, feasibleResult());
    require(d.engine != solver::Engine::BranchAndCut,
            "a MILP reduced to an LP must not run branch-and-cut");
}

// requireVertexSolution is a hard contract: PDLP's iterate is not a vertex,
// so the dispatcher must name the dual simplex and the orchestrator must
// actually run it (see runDualSimplex). It previously selected
// Engine::DualSimplex while the orchestrator quietly routed it into PDLP,
// breaking the promise silently.
//
// The contract is bounded by what the engine can physically do. The dual
// simplex carries a DENSE m x m basis inverse allocated up front: 32 MB at
// the 2000-row threshold, but 20 GB at 50000 rows. So the request is honoured
// within those limits and refused with a reason beyond them -- never
// "honoured" by handing back a non-vertex point, and never by attempting an
// allocation that cannot succeed.
void testVertexRequestForcesSimplexWhenItFits() {
    solver::SolverOptions o;
    o.requireVertexSolution = true;
    const model::Model m = lpOfSize(50, 50);  // well inside the limits
    const auto d = solver::dispatch(m, solver::classify(m), feasibleResult(), o);
    require(d.engine == solver::Engine::DualSimplex,
            "a vertex request that fits routes to the dual simplex");
}

void testVertexRequestBeyondSimplexLimitsIsRefused() {
    solver::SolverOptions o;
    o.requireVertexSolution = true;
    const model::Model m = lpOfSize(50000, 50000);  // 20 GB of dense inverse
    const auto d = solver::dispatch(m, solver::classify(m), feasibleResult(), o);
    require(d.engine == solver::Engine::Unsupported,
            "a vertex request too large for the only vertex-capable engine is "
            "refused, not silently answered with a non-vertex point");
    require(d.reason.find("vertex") != std::string::npos,
            "and the refusal explains itself");
}

void testSizeCrossover() {
    solver::SolverOptions o;
    o.dualSimplexMaxRows = 2000;

    const model::Model small = lpOfSize(100, 100);
    require(solver::dispatch(small, solver::classify(small), feasibleResult(), o)
                .engine == solver::Engine::DualSimplex,
            "small LP routes to dual simplex");

    const model::Model large = lpOfSize(5000, 5000);
    require(solver::dispatch(large, solver::classify(large), feasibleResult(), o)
                .engine == solver::Engine::Pdlp,
            "large LP routes to PDLP");
}

void testNonzeroThresholdAlsoTriggersPdlp() {
    solver::SolverOptions o;
    o.dualSimplexMaxRows = 100000;   // rows would not trigger it
    o.dualSimplexMaxNonzeros = 10;   // nonzeros will
    const model::Model m = lpOfSize(500, 500);
    require(solver::dispatch(m, solver::classify(m), feasibleResult(), o)
                .engine == solver::Engine::Pdlp,
            "nonzero count alone can force PDLP");
}

void testQuadraticRoutesToQpEngine() {
    model::Model m = lpOfSize(3, 3);
    m.objective.quadraticTerms.push_back({0, 0, 1.0});
    const auto d = solver::dispatch(m, solver::classify(m), feasibleResult());
    require(d.engine == solver::Engine::Qp,
            std::string("a convex QP routes to the QP engine, got ") +
            solver::toString(d.engine));
    require(!d.reason.empty(), "and explains why");
}

void testConvexMiqpRoutesToMiqp() {
    model::Model m = lpOfSize(3, 3);
    m.variables[0].type = model::VariableType::Binary;
    m.variables[0].lowerBound = 0.0;
    m.variables[0].upperBound = 1.0;
    m.objective.quadraticTerms.push_back({1, 1, 1.0});
    const auto d = solver::dispatch(m, solver::classify(m), feasibleResult());
    require(d.engine == solver::Engine::Miqp,
            "convex MIQP routes to the MIQP engine");
}

void testNonConvexMinimizationRejected() {
    model::Model m = lpOfSize(3, 3);
    m.variables[0].type = model::VariableType::Integer;
    m.objective.sense = model::ObjectiveSense::Minimize;
    m.objective.quadraticTerms.push_back({1, 1, -1.0});
    const auto d = solver::dispatch(m, solver::classify(m), feasibleResult());
    require(d.engine == solver::Engine::Unsupported,
            "concave quadratic minimization is non-convex and must be rejected");
}

void testConcaveMaximizationAccepted() {
    model::Model m = lpOfSize(3, 3);
    m.variables[0].type = model::VariableType::Integer;
    m.objective.sense = model::ObjectiveSense::Maximize;
    m.objective.quadraticTerms.push_back({1, 1, -1.0});
    const auto d = solver::dispatch(m, solver::classify(m), feasibleResult());
    require(d.engine == solver::Engine::Miqp,
            "concave quadratic maximization is valid convex-MIQP form");
}

void testConvexMaximizationRejected() {
    model::Model m = lpOfSize(3, 3);
    m.variables[0].type = model::VariableType::Integer;
    m.objective.sense = model::ObjectiveSense::Maximize;
    m.objective.quadraticTerms.push_back({1, 1, 1.0});
    const auto d = solver::dispatch(m, solver::classify(m), feasibleResult());
    require(d.engine == solver::Engine::Unsupported,
            "convex quadratic maximization must be rejected");
}


void testTrivialWhenNothingRemains() {
    model::Model m;
    const auto d = solver::dispatch(m, solver::classify(m), feasibleResult());
    require(d.engine == solver::Engine::Trivial, "no variables is trivial");
}

void testEveryDecisionExplainsItself() {
    const model::Model m = lpOfSize(10, 10);
    const auto d = solver::dispatch(m, solver::classify(m), feasibleResult());
    require(!d.reason.empty(), "every decision carries a reason");
}

// End-to-end against the real presolver.
void testWithRealPresolve() {
    model::Model m;
    m.variables = {variable("x", model::VariableType::Continuous, 0, 10),
                   variable("fixed", model::VariableType::Integer, 3, 3)};
    m.constraints = {constraint("c0", -kInf, 20.0, {{0, 1.0}, {1, 1.0}})};
    m.objective.linearTerms = {{0, 1.0}, {1, 1.0}};

    const auto classification = solver::classify(m);
    require(classification.problemClass == solver::ProblemClass::MILP,
            "model with an integer variable classifies as MILP");

    presolve::Presolver presolver;
    const presolve::PresolveResult result = presolver.run(m);
    const auto d = solver::dispatch(result.model, classification, result);

    // presolve fixes the integer variable (lower == upper), so no tree is needed.
    require(d.engine != solver::Engine::BranchAndCut,
            "presolve fixed the only integer variable; got " +
            std::string(solver::toString(d.engine)) + " (" + d.reason + ")");
}

void run(const char* name, void (*test)()) {
    try {
        test();
        std::cout << "  pass  " << name << '\n';
    } catch (const std::exception& error) {
        std::cout << "  FAIL  " << name << ": " << error.what() << '\n';
        ++failures;
    }
}

}  // namespace

int main() {
    run("classifiesLp", testClassifiesLp);
    run("classifiesMilpAndBinary", testClassifiesMilpAndBinary);
    run("integerBoundedToUnitIsCountedBinary", testIntegerBoundedToUnitIsCountedBinary);
    run("classifiesQpAndMiqp", testClassifiesQpAndMiqp);
    run("detectsNetworkStructure", testDetectsNetworkStructure);
    run("detectsBigM", testDetectsBigM);
    run("detectsSetPartitioningAndSymmetry", testDetectsSetPartitioningAndSymmetry);
    run("coefficientRangeRatio", testCoefficientRangeRatio);

    run("infeasibleShortCircuits", testInfeasibleShortCircuits);
    run("forcedEngineIsHonoured", testForcedEngineIsHonoured);
    run("forcedEngineBeatsIntegrality", testForcedEngineBeatsIntegrality);
    run("integralityRoutesToBranchAndCut", testIntegralityRoutesToBranchAndCut);
    run("milpReducedToLpRoutesToLpEngine", testMilpReducedToLpRoutesToLpEngine);
    run("vertexRequestForcesSimplexWhenItFits", testVertexRequestForcesSimplexWhenItFits);
    run("vertexRequestBeyondSimplexLimitsIsRefused", testVertexRequestBeyondSimplexLimitsIsRefused);
    run("sizeCrossover", testSizeCrossover);
    run("nonzeroThresholdAlsoTriggersPdlp", testNonzeroThresholdAlsoTriggersPdlp);
    run("quadraticRoutesToQpEngine", testQuadraticRoutesToQpEngine);
    run("convexMiqpRoutesToMiqp", testConvexMiqpRoutesToMiqp);
    run("nonConvexMinimizationRejected", testNonConvexMinimizationRejected);
    run("concaveMaximizationAccepted", testConcaveMaximizationAccepted);
    run("convexMaximizationRejected", testConvexMaximizationRejected);
    run("trivialWhenNothingRemains", testTrivialWhenNothingRemains);
    run("everyDecisionExplainsItself", testEveryDecisionExplainsItself);
    run("withRealPresolve", testWithRealPresolve);

    if (failures == 0) {
        std::cout << "All classifier/dispatcher tests passed\n";
        return EXIT_SUCCESS;
    }
    std::cout << failures << " test(s) failed\n";
    return EXIT_FAILURE;
}
