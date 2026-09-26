// End-to-end pipeline tests on classical problems with published optima.
//
// Every case runs the full chain -- classify, presolve, dispatch, engine,
// normalised result -- and checks the class, the engine, the objective and the
// solution itself. Textbook instances are used because their answers are known
// independently of anything in this repository.

#include "solver/orchestrator.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

namespace {

const double INF = std::numeric_limits<double>::infinity();
int checks = 0;
int failures = 0;

void ck(bool ok, const std::string& what) {
    ++checks;
    if (!ok) {
        ++failures;
        std::printf("    FAIL  %s\n", what.c_str());
    }
}

void near(double got, double want, double tol, const std::string& what) {
    ++checks;
    if (!(std::abs(got - want) <= tol)) {
        ++failures;
        std::printf("    FAIL  %s: got %.9g want %.9g\n", what.c_str(), got, want);
    }
}

struct Builder {
    model::Model m;
    void var(const char* name, model::VariableType type, double lo, double hi) {
        model::Variable v;
        v.name = name; v.type = type; v.lowerBound = lo; v.upperBound = hi;
        m.variables.push_back(v);
    }
    void row(const char* name, double lo, double hi,
             std::vector<model::LinearTerm> terms) {
        model::Constraint c;
        c.name = name; c.lowerBound = lo; c.upperBound = hi;
        c.linearTerms = std::move(terms);
        m.constraints.push_back(c);
    }
    void obj(model::ObjectiveSense sense, std::vector<model::LinearTerm> terms,
             double offset = 0.0) {
        m.objective.sense = sense;
        m.objective.linearTerms = std::move(terms);
        m.objective.offset = offset;
    }
};

void report(const char* name, const solver::SolveResult& r) {
    std::printf("  %-34s %-14s %-14s obj=%14.6f  %s\n",
                name, solver::toString(r.status), solver::toString(r.engine),
                r.objectiveValue, r.hasDuals ? "duals" : "no duals");
}

// ---------------------------------------------------------------------------
// LP cases
// ---------------------------------------------------------------------------

// Classic product-mix LP. Optimum 22.4 at (0, 5.6).
void testProductMix() {
    Builder b;
    b.var("x", model::VariableType::Continuous, 0, INF);
    b.var("y", model::VariableType::Continuous, 0, INF);
    b.row("labour",  -INF, 14.0, {{0, 2.0}, {1, 1.0}});
    b.row("machine", -INF, 28.0, {{0, 4.0}, {1, 5.0}});
    b.row("material",-INF, 30.0, {{0, 2.0}, {1, 5.0}});
    b.obj(model::ObjectiveSense::Maximize, {{0, 3.0}, {1, 4.0}});

    const auto r = solver::solve(b.m);
    report("product mix (max)", r);
    ck(r.status == solver::SolveStatus::Optimal, "product mix solves");
    near(r.objectiveValue, 22.4, 1e-4, "product mix objective");
    near(r.variableValues[0], 0.0, 1e-4, "product mix x");
    near(r.variableValues[1], 5.6, 1e-4, "product mix y");
    ck(r.hasDuals, "an LP route reports duals");
    ck(r.constraintDuals.size() == b.m.constraints.size(), "one dual per original row");
    near(r.constraintDuals[1], 0.8, 1e-6, "original machine shadow price");
    ck(r.reducedCosts.size() == b.m.variables.size(), "original reduced-cost dimensions");
    near(r.reducedCosts[0], -0.2, 1e-6, "original x reduced cost");
    near(r.reducedCosts[1], 0, 1e-6, "original y is interior");
    for (std::size_t i = 0; i < r.constraintDuals.size(); ++i) {
        ck(r.constraintDuals[i] >= -1e-6,
           "dual " + std::to_string(i) + " is correctly signed for a <= row");
    }
    ck(r.constraintDuals[1] > 1e-6, "the binding 'machine' row carries a price");
    ck(std::abs(r.constraintDuals[0]) < 1e-6 && std::abs(r.constraintDuals[2]) < 1e-6,
       "slack rows are priced at zero");
}

// Diet problem: minimise cost subject to nutrient minima. Optimum 1.0 at (0, 1).
void testDiet() {
    Builder b;
    b.var("f1", model::VariableType::Continuous, 0, INF);
    b.var("f2", model::VariableType::Continuous, 0, INF);
    b.row("protein", 8.0, INF, {{0, 4.0}, {1, 8.0}});
    b.row("calcium", 3.0, INF, {{0, 3.0}, {1, 3.0}});
    b.obj(model::ObjectiveSense::Minimize, {{0, 2.0}, {1, 1.0}});

    const auto r = solver::solve(b.m);
    report("diet (min)", r);
    ck(r.status == solver::SolveStatus::Optimal, "diet solves");
    near(r.objectiveValue, 1.0, 1e-4, "diet objective");
}

// Degenerate: more constraints active at the optimum than variables.
void testDegenerate() {
    Builder b;
    b.var("x", model::VariableType::Continuous, 0, INF);
    b.var("y", model::VariableType::Continuous, 0, INF);
    b.row("r0", -INF, 4.0, {{0, 1.0}, {1, 1.0}});
    b.row("r1", -INF, 4.0, {{0, 1.0}, {1, 1.0}});   // duplicate of r0
    b.row("r2", -INF, 8.0, {{0, 2.0}, {1, 2.0}});   // parallel to r0
    b.row("r3", -INF, 2.0, {{0, 1.0}});
    b.obj(model::ObjectiveSense::Maximize, {{0, 1.0}, {1, 1.0}});

    const auto r = solver::solve(b.m);
    report("degenerate (duplicate rows)", r);
    ck(r.status == solver::SolveStatus::Optimal, "degenerate solves");
    near(r.objectiveValue, 4.0, 1e-4, "degenerate objective");
}

// An objective offset must survive both the sense flip and presolve.
void testObjectiveOffset() {
    Builder b;
    b.var("x", model::VariableType::Continuous, 0, 10);
    b.row("r0", -INF, 5.0, {{0, 1.0}});
    b.obj(model::ObjectiveSense::Maximize, {{0, 2.0}}, 100.0);

    const auto r = solver::solve(b.m);
    report("objective offset (max)", r);
    near(r.objectiveValue, 110.0, 1e-4, "offset carried through");
}

void testInfeasibleLp() {
    Builder b;
    b.var("x", model::VariableType::Continuous, 0, 10);
    b.var("y", model::VariableType::Continuous, 0, 10);
    b.row("lo", 12.0, INF,  {{0, 1.0}, {1, 1.0}});
    b.row("hi", -INF, 3.0,  {{0, 1.0}, {1, 1.0}});
    b.obj(model::ObjectiveSense::Minimize, {{0, 1.0}, {1, 1.0}});

    const auto r = solver::solve(b.m);
    report("infeasible LP", r);
    ck(r.status == solver::SolveStatus::Infeasible,
       std::string("infeasible LP reported as ") + solver::toString(r.status));
}

// requireVertexSolution must actually run the dual simplex, not quietly fall
// through to PDLP. Product mix: max 3x + 5y s.t. x <= 4, 2y <= 12,
// 3x + 2y <= 18. The optimum is the vertex (2, 6) with objective 36 -- the
// intersection of rows c1 and c2, so a genuine basic solution rather than a
// point that merely happens to be near it.
void testRequireVertexSolutionRunsDualSimplex() {
    Builder b;
    b.var("x", model::VariableType::Continuous, 0, INF);
    b.var("y", model::VariableType::Continuous, 0, INF);
    b.row("c0", -INF, 4.0,  {{0, 1.0}});
    b.row("c1", -INF, 12.0, {{1, 2.0}});
    b.row("c2", -INF, 18.0, {{0, 3.0}, {1, 2.0}});
    b.obj(model::ObjectiveSense::Maximize, {{0, 3.0}, {1, 5.0}});

    solver::SolverOptions o;
    o.requireVertexSolution = true;
    const auto r = solver::solve(b.m, o);
    report("requireVertexSolution", r);
    ck(r.engine == solver::Engine::DualSimplex,
       std::string("vertex request runs the dual simplex, got ") +
       solver::toString(r.engine));
    ck(r.status == solver::SolveStatus::Optimal, "vertex request solves");
    near(r.objectiveValue, 36.0, 1e-6, "vertex objective");
    // A vertex, to simplex precision -- not merely close, the way a
    // first-order method's iterate would be.
    near(r.variableValues[0], 2.0, 1e-9, "vertex x is exact");
    near(r.variableValues[1], 6.0, 1e-9, "vertex y is exact");
    // Both rows through this vertex are tight.
    near(r.variableValues[1] * 2.0, 12.0, 1e-9, "row c1 is tight at the vertex");
    near(3.0 * r.variableValues[0] + 2.0 * r.variableValues[1], 18.0, 1e-9,
         "row c2 is tight at the vertex");
    ck(r.hasDuals, "the simplex reports duals");
}

// The same LP without asking for a vertex: it is small, so the dispatcher
// picks the dual simplex on size alone and must likewise really run it.
void testSmallLpAutoRoutesToDualSimplex() {
    Builder b;
    b.var("x", model::VariableType::Continuous, 0, INF);
    b.var("y", model::VariableType::Continuous, 0, INF);
    b.row("c0", -INF, 4.0,  {{0, 1.0}});
    b.row("c1", -INF, 12.0, {{1, 2.0}});
    b.row("c2", -INF, 18.0, {{0, 3.0}, {1, 2.0}});
    b.obj(model::ObjectiveSense::Maximize, {{0, 3.0}, {1, 5.0}});

    const auto r = solver::solve(b.m);
    report("small LP -> dual simplex", r);
    ck(r.engine == solver::Engine::DualSimplex,
       std::string("a small LP routes to the dual simplex, got ") +
       solver::toString(r.engine));
    ck(r.status == solver::SolveStatus::Optimal, "small LP solves");
    near(r.objectiveValue, 36.0, 1e-6, "small LP objective");
    near(r.variableValues[0], 2.0, 1e-9, "small LP x");
    near(r.variableValues[1], 6.0, 1e-9, "small LP y");
}

// An LP with an improving ray must be reported as unbounded, not answered
// with a large finite point. min -x, x >= 0 with no upper bound.
void testUnboundedLp() {
    Builder b;
    b.var("x", model::VariableType::Continuous, 0, INF);
    b.row("free", -INF, INF, {{0, 1.0}});
    b.obj(model::ObjectiveSense::Minimize, {{0, -1.0}});

    const auto r = solver::solve(b.m);
    report("unbounded LP", r);
    ck(r.status == solver::SolveStatus::Unbounded,
       std::string("unbounded LP reported as ") + solver::toString(r.status));
}

// A multiple-optimum LP is what actually tests vertex-ness: max x+y s.t.
// x+y<=1, 0<=x,y<=1. EVERY point on x+y=1 is optimal, so optimality alone
// proves nothing. A basic solution must be (1,0) or (0,1); (0.5,0.5) is
// optimal but is not a vertex.
//
// Verified independently of any solver claim: at a vertex of an
// n-dimensional LP the active constraints (rows AND bounds) have rank n.
// Measured, PDLP returns exactly (0.5,0.5) here -- rank 1 -- so this case
// genuinely discriminates rather than passing by luck.
void testVertexOnMultipleOptimumLp() {
    Builder b;
    b.var("x", model::VariableType::Continuous, 0.0, 1.0);
    b.var("y", model::VariableType::Continuous, 0.0, 1.0);
    b.row("c0", -INF, 1.0, {{0, 1.0}, {1, 1.0}});
    b.obj(model::ObjectiveSense::Maximize, {{0, 1.0}, {1, 1.0}});

    solver::SolverOptions o;
    o.requireVertexSolution = true;
    const auto r = solver::solve(b.m, o);
    report("vertex on multiple-optimum LP", r);
    ck(r.engine == solver::Engine::DualSimplex, "vertex request runs the simplex");
    ck(r.status == solver::SolveStatus::Optimal, "multiple-optimum LP solves");
    near(r.objectiveValue, 1.0, 1e-9, "objective is 1 at every optimum");

    const double x = r.variableValues[0], y = r.variableValues[1];
    // Count independent active constraints. n = 2, so a vertex needs rank 2.
    int activeRows = 0, activeBounds = 0;
    if (std::abs(x + y - 1.0) < 1e-7) ++activeRows;
    if (std::abs(x) < 1e-7 || std::abs(x - 1.0) < 1e-7) ++activeBounds;
    if (std::abs(y) < 1e-7 || std::abs(y - 1.0) < 1e-7) ++activeBounds;
    ck(activeRows + activeBounds >= 2,
       "the returned point has >= 2 active constraints, i.e. it is a vertex "
       "and not an interior point of the optimal face");
    ck((std::abs(x) < 1e-7 && std::abs(y - 1.0) < 1e-7) ||
       (std::abs(x - 1.0) < 1e-7 && std::abs(y) < 1e-7),
       "and it is one of the two actual vertices, (0,1) or (1,0)");
}

// requireVertexSolution beyond what the only vertex-capable engine can hold
// must be refused with a reason, not answered with PDLP's non-vertex point
// and not by attempting a dense m x m inverse that cannot be allocated
// (8*m^2 bytes: 20 GB at 50000 rows).
void testVertexRequestTooLargeIsRefused() {
    Builder b;
    const int n = 40;
    for (int j = 0; j < n; ++j)
        b.var(("x" + std::to_string(j)).c_str(), model::VariableType::Continuous, 0, 10);
    // Rows must span two variables each. A singleton row is turned into a
    // variable bound and deleted by presolve, which would leave nothing for
    // the size guard to measure.
    for (int i = 0; i < n; ++i)
        b.row(("r" + std::to_string(i)).c_str(), -INF, 5.0,
              {{i, 1.0}, {(i + 1) % n, 1.0}});
    b.obj(model::ObjectiveSense::Maximize, {{0, 1.0}});

    solver::SolverOptions o;
    o.requireVertexSolution = true;
    o.dualSimplexMaxRows = 5;   // force the model past the limit
    const auto r = solver::solve(b.m, o);
    report("vertex request beyond limits", r);
    ck(r.status == solver::SolveStatus::Unsupported,
       std::string("oversized vertex request refused, got ") + solver::toString(r.status));
    ck(r.engineReason.find("vertex") != std::string::npos, "and explains why");
}

// A structurally VALID model on which the solver breaks down numerically must
// report NumericalFailure, not InvalidModel -- the latter blames the caller
// for the solver's limits. Duplicate rows are legal and can make the
// cold-start crash basis singular.
void testNumericalFailureIsNotReportedAsInvalidModel() {
    Builder b;
    b.var("x", model::VariableType::Continuous, -6.0, INF);
    b.var("y", model::VariableType::Continuous, -INF, INF);
    b.var("z", model::VariableType::Continuous, -INF, INF);
    // Two rows with identical coefficients (legal, degenerate).
    b.row("r0", -7.63, 7.63, {{1, 2.68}, {2, 1.25}});
    b.row("r1", -INF, 5.03, {{1, 2.68}, {2, 1.25}});
    b.obj(model::ObjectiveSense::Maximize, {{0, -3.35}, {1, -1.44}, {2, 3.49}});

    solver::SolverOptions o;
    o.requireVertexSolution = true;
    const auto r = solver::solve(b.m, o);
    report("valid-but-hard LP status", r);
    ck(r.status != solver::SolveStatus::InvalidModel,
       "a structurally valid model is never reported InvalidModel");
}

// The dispatcher's decision and the engine that actually ran must agree.
//
// This is the direct check for the bug that started this: the dispatcher
// named Engine::DualSimplex while the orchestrator's switch executed PDLP, so
// SolveResult.engine described a decision nobody carried out. Pivot counts
// and residual sizes only HINT at which engine ran; executedEngine is written
// by the code path that actually calls the solver, so a divergence is caught
// directly instead of inferred.
void testDispatchDecisionMatchesExecutedEngine() {
    struct Case { const char* name; model::Model m; solver::SolverOptions o; };
    std::vector<Case> cases;

    {   // small LP -> dual simplex by size
        Builder b;
        b.var("x", model::VariableType::Continuous, 0, INF);
        b.var("y", model::VariableType::Continuous, 0, INF);
        b.row("c0", -INF, 4.0,  {{0, 1.0}});
        b.row("c1", -INF, 12.0, {{1, 2.0}});
        b.row("c2", -INF, 18.0, {{0, 3.0}, {1, 2.0}});
        b.obj(model::ObjectiveSense::Maximize, {{0, 3.0}, {1, 5.0}});
        cases.push_back({"small LP (auto)", b.m, {}});

        solver::SolverOptions vo; vo.requireVertexSolution = true;
        cases.push_back({"requireVertexSolution", b.m, vo});

        solver::SolverOptions po; po.forceEngine = solver::Engine::Pdlp;
        cases.push_back({"forced PDLP", b.m, po});
    }
    {   // QP
        Builder b;
        b.var("x", model::VariableType::Continuous, 0.0, INF);
        b.var("y", model::VariableType::Continuous, 0.0, INF);
        b.row("c0", -INF, 2.0, {{0, 1.0}, {1, 1.0}});
        b.obj(model::ObjectiveSense::Minimize, {{0, -2.0}, {1, -4.0}});
        b.m.objective.quadraticTerms = {{0, 0, 1.0}, {1, 1, 1.0}};
        cases.push_back({"QP", b.m, {}});
    }
    {   // MILP -> branch and cut
        Builder b;
        for (int j = 0; j < 4; ++j)
            b.var(("b" + std::to_string(j)).c_str(), model::VariableType::Binary, 0, 1);
        b.row("cap", -INF, 10.0, {{0, 2.0}, {1, 4.0}, {2, 6.0}, {3, 9.0}});
        b.obj(model::ObjectiveSense::Maximize,
              {{0, 10.0}, {1, 10.0}, {2, 12.0}, {3, 18.0}});
        cases.push_back({"MILP", b.m, {}});
    }

    for (const auto& c : cases) {
        const auto r = solver::solve(c.m, c.o);
        report((std::string("executed: ") + c.name).c_str(), r);
        ck(r.executedEngine == r.engine,
           std::string(c.name) + ": dispatcher chose " + solver::toString(r.engine) +
           " and the orchestrator executed " + solver::toString(r.executedEngine));
    }
}

// ---------------------------------------------------------------------------
// MILP cases
// ---------------------------------------------------------------------------

// 0/1 knapsack, capacity 10, values {10,10,12,18}, weights {2,4,6,9}.
// Brute force over all 16 subsets gives 22, taking items 1 and 2 (weight
// exactly 10). Verified independently in Python rather than assumed.
void testKnapsack() {
    Builder b;
    const double value[]  = {10, 10, 12, 18};
    const double weight[] = { 2,  4,  6,  9};
    for (int j = 0; j < 4; ++j) {
        b.var(("item" + std::to_string(j)).c_str(),
              model::VariableType::Binary, 0, 1);
    }
    b.row("capacity", -INF, 10.0,
          {{0, weight[0]}, {1, weight[1]}, {2, weight[2]}, {3, weight[3]}});
    b.obj(model::ObjectiveSense::Maximize,
          {{0, value[0]}, {1, value[1]}, {2, value[2]}, {3, value[3]}});

    const auto r = solver::solve(b.m);
    report("0/1 knapsack (max)", r);
    ck(r.status == solver::SolveStatus::Optimal, "knapsack solves");
    ck(r.engine == solver::Engine::BranchAndCut, "knapsack routes to branch-and-cut");
    near(r.objectiveValue, 22.0, 1e-6, "knapsack optimum");
    ck(r.integralityRespected, "knapsack answer is integral");
    ck(!r.hasDuals, "branch-and-cut reports no duals");
}

// 3x3 assignment problem. Cost matrix rows {4,1,3},{2,0,5},{3,2,2};
// the optimal assignment costs 5.
void testAssignment() {
    Builder b;
    const double cost[3][3] = {{4, 1, 3}, {2, 0, 5}, {3, 2, 2}};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            b.var(("x" + std::to_string(i) + std::to_string(j)).c_str(),
                  model::VariableType::Binary, 0, 1);
        }
    }
    for (int i = 0; i < 3; ++i) {
        b.row(("row" + std::to_string(i)).c_str(), 1.0, 1.0,
              {{i * 3 + 0, 1.0}, {i * 3 + 1, 1.0}, {i * 3 + 2, 1.0}});
    }
    for (int j = 0; j < 3; ++j) {
        b.row(("col" + std::to_string(j)).c_str(), 1.0, 1.0,
              {{0 * 3 + j, 1.0}, {1 * 3 + j, 1.0}, {2 * 3 + j, 1.0}});
    }
    std::vector<model::LinearTerm> objective;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            objective.push_back({i * 3 + j, cost[i][j]});
        }
    }
    b.obj(model::ObjectiveSense::Minimize, std::move(objective));

    const auto r = solver::solve(b.m);
    report("3x3 assignment (min)", r);
    ck(r.status == solver::SolveStatus::Optimal, "assignment solves");
    near(r.objectiveValue, 5.0, 1e-6, "assignment optimum");
    ck(r.integralityRespected, "assignment answer is integral");
}

// Set covering: 4 sets over 4 elements; the minimum cover uses 2 sets.
void testSetCovering() {
    Builder b;
    for (int j = 0; j < 4; ++j) {
        b.var(("s" + std::to_string(j)).c_str(), model::VariableType::Binary, 0, 1);
    }
    b.row("e0", 1.0, INF, {{0, 1.0}, {1, 1.0}});
    b.row("e1", 1.0, INF, {{1, 1.0}, {2, 1.0}});
    b.row("e2", 1.0, INF, {{2, 1.0}, {3, 1.0}});
    b.row("e3", 1.0, INF, {{0, 1.0}, {3, 1.0}});
    b.obj(model::ObjectiveSense::Minimize,
          {{0, 1.0}, {1, 1.0}, {2, 1.0}, {3, 1.0}});

    const auto r = solver::solve(b.m);
    report("set covering (min)", r);
    ck(r.status == solver::SolveStatus::Optimal, "set covering solves");
    near(r.objectiveValue, 2.0, 1e-6, "set covering optimum");
}

// A MILP whose LP relaxation is fractional: max x+y, 2x+2y<=3, x,y binary.
// Relaxation gives 1.5; the integer optimum is 1.
void testFractionalRelaxation() {
    Builder b;
    b.var("x", model::VariableType::Binary, 0, 1);
    b.var("y", model::VariableType::Binary, 0, 1);
    b.row("c0", -INF, 3.0, {{0, 2.0}, {1, 2.0}});
    b.obj(model::ObjectiveSense::Maximize, {{0, 1.0}, {1, 1.0}});

    const auto r = solver::solve(b.m);
    report("fractional relaxation (max)", r);
    near(r.objectiveValue, 1.0, 1e-6,
         "integer optimum, not the 1.5 relaxation");
    ck(r.integralityRespected, "answer is integral");
}

// Presolve fixes the only integer variable, so no tree should be needed.
void testMilpReducedToLp() {
    Builder b;
    b.var("x", model::VariableType::Continuous, 0, 10);
    b.var("fixed", model::VariableType::Integer, 3, 3);
    b.row("c0", -INF, 20.0, {{0, 1.0}, {1, 1.0}});
    b.obj(model::ObjectiveSense::Maximize, {{0, 1.0}, {1, 1.0}});

    const auto r = solver::solve(b.m);
    report("MILP reduced to LP", r);
    ck(r.status == solver::SolveStatus::Optimal, "reduced MILP solves");
    ck(r.engine != solver::Engine::BranchAndCut,
       std::string("no tree search needed, got ") + solver::toString(r.engine));
    near(r.objectiveValue, 13.0, 1e-4, "reduced MILP objective");
}

// ---------------------------------------------------------------------------
// Contract every engine must honour, so postsolve can be written once
// ---------------------------------------------------------------------------

void testUniformResultContract() {
    std::vector<model::Model> models;
    {   Builder b;
        b.var("x", model::VariableType::Continuous, 0, 10);
        b.row("c", -INF, 5.0, {{0, 1.0}});
        b.obj(model::ObjectiveSense::Maximize, {{0, 1.0}});
        models.push_back(b.m); }
    {   Builder b;
        b.var("b0", model::VariableType::Binary, 0, 1);
        b.var("b1", model::VariableType::Binary, 0, 1);
        b.row("c", -INF, 1.0, {{0, 1.0}, {1, 1.0}});
        b.obj(model::ObjectiveSense::Maximize, {{0, 3.0}, {1, 2.0}});
        models.push_back(b.m); }

    for (std::size_t k = 0; k < models.size(); ++k) {
        const auto r = solver::solve(models[k]);
        const std::string tag = "model " + std::to_string(k) + ": ";
        ck(r.status == solver::SolveStatus::Optimal, tag + "solves");
        ck(r.hasPrimal && r.variableValues.size() == models[k].variables.size(),
           tag + "variableValues is one entry per original variable");
        // Duals are either absent or one per constraint -- never partial.
        ck(!r.hasDuals ||
               r.constraintDuals.size() == models[k].constraints.size(),
           tag + "duals are absent or one per original constraint");
        ck(!r.engineReason.empty(), tag + "engine choice is explained");
        ck(std::isfinite(r.objectiveValue), tag + "objective is finite");
    }
}


// ---------------------------------------------------------------------------
// QP through the pipeline
// ---------------------------------------------------------------------------

// min x^2 + y^2 - 2x - 4y  s.t.  x + y <= 2, x,y >= 0.
// Unconstrained optimum is (1,2); the row is binding, so the solution is
// (0.5, 1.5) with objective 0.25 + 2.25 - 1 - 6 = -4.5.
void testQuadraticRoutesToQpEngine() {
    Builder b;
    b.var("x", model::VariableType::Continuous, 0.0, INF);
    b.var("y", model::VariableType::Continuous, 0.0, INF);
    b.row("c0", -INF, 2.0, {{0, 1.0}, {1, 1.0}});
    b.obj(model::ObjectiveSense::Minimize, {{0, -2.0}, {1, -4.0}});
    // value * xi * xj convention: x^2 + y^2
    b.m.objective.quadraticTerms = {{0, 0, 1.0}, {1, 1, 1.0}};

    const auto r = solver::solve(b.m);
    report("QP (min, binding row)", r);
    ck(r.engine == solver::Engine::Qp,
       std::string("quadratic objective routes to the QP engine, got ") +
       solver::toString(r.engine));
    ck(r.status == solver::SolveStatus::Optimal, "QP solves");
    near(r.objectiveValue, -4.5, 1e-5, "QP objective");
    near(r.variableValues[0], 0.5, 1e-4, "QP x");
    near(r.variableValues[1], 1.5, 1e-4, "QP y");
}

// max -x^2 - y^2 + 2x + 4y  s.t.  x + y <= 2, x,y >= 0.
//
// This is the exact negation of testQuadraticRoutesToQpEngine's problem: f
// here equals -(that problem's objective), so maximising f is the same
// problem as minimising it there, and both must land on the same point,
// (0.5, 1.5), with the objective negated: -(-4.5) = 4.5.
//
// qp::fromModel negates BOTH P and q for Maximize before handing the ADMM
// engine a minimisation problem -- negating q alone (an earlier version of
// this adapter) sends the engine an unrelated saddle-shaped objective, since
// P's curvature is still the ORIGINAL problem's, not this one's negation.
void testMaximizeQuadraticRoutesToQpEngine() {
    Builder b;
    b.var("x", model::VariableType::Continuous, 0.0, INF);
    b.var("y", model::VariableType::Continuous, 0.0, INF);
    b.row("c0", -INF, 2.0, {{0, 1.0}, {1, 1.0}});
    b.obj(model::ObjectiveSense::Maximize, {{0, 2.0}, {1, 4.0}});
    b.m.objective.quadraticTerms = {{0, 0, -1.0}, {1, 1, -1.0}};

    const auto r = solver::solve(b.m);
    report("QP (max, binding row)", r);
    ck(r.engine == solver::Engine::Qp,
       std::string("quadratic objective routes to the QP engine, got ") +
       solver::toString(r.engine));
    ck(r.status == solver::SolveStatus::Optimal, "maximize QP solves");
    near(r.objectiveValue, 4.5, 1e-4, "maximize QP objective");
    near(r.variableValues[0], 0.5, 1e-4, "maximize QP x");
    near(r.variableValues[1], 1.5, 1e-4, "maximize QP y");
}

// Same problem as testQuadraticRoutesToQpEngine, plus a +100 objective
// offset. The optimal point is unchanged (the offset is constant in x, y);
// only the reported objective moves, by exactly 100.
void testQpWithObjectiveOffset() {
    Builder b;
    b.var("x", model::VariableType::Continuous, 0.0, INF);
    b.var("y", model::VariableType::Continuous, 0.0, INF);
    b.row("c0", -INF, 2.0, {{0, 1.0}, {1, 1.0}});
    b.obj(model::ObjectiveSense::Minimize, {{0, -2.0}, {1, -4.0}}, 100.0);
    b.m.objective.quadraticTerms = {{0, 0, 1.0}, {1, 1, 1.0}};

    const auto r = solver::solve(b.m);
    report("QP with offset (min)", r);
    ck(r.status == solver::SolveStatus::Optimal, "QP with offset solves");
    near(r.objectiveValue, 95.5, 1e-4, "offset carried through minimize QP");
    near(r.variableValues[0], 0.5, 1e-4, "offset QP x unchanged");
    near(r.variableValues[1], 1.5, 1e-4, "offset QP y unchanged");
}

// testMaximizeQuadraticRoutesToQpEngine's problem, plus a -30 offset. Exercises
// negation and the offset together: get the order wrong -- e.g. negate
// (raw + offset) instead of (negate raw) + offset -- and this is the case
// that catches it, since the offset's sign would flip too.
void testMaximizeQpWithObjectiveOffset() {
    Builder b;
    b.var("x", model::VariableType::Continuous, 0.0, INF);
    b.var("y", model::VariableType::Continuous, 0.0, INF);
    b.row("c0", -INF, 2.0, {{0, 1.0}, {1, 1.0}});
    b.obj(model::ObjectiveSense::Maximize, {{0, 2.0}, {1, 4.0}}, -30.0);
    b.m.objective.quadraticTerms = {{0, 0, -1.0}, {1, 1, -1.0}};

    const auto r = solver::solve(b.m);
    report("QP with offset (max)", r);
    ck(r.status == solver::SolveStatus::Optimal, "maximize QP with offset solves");
    near(r.objectiveValue, -25.5, 1e-4, "offset carried through maximize QP");
    near(r.variableValues[0], 0.5, 1e-4, "offset maximize QP x unchanged");
    near(r.variableValues[1], 1.5, 1e-4, "offset maximize QP y unchanged");
}

// Model convention (the source of truth, not to be changed to suit the engine):
//   f(x) = offset + sum c_i x_i + sum q_ij x_i x_j
// QuadraticTerm.value is the DIRECT coefficient of x_i*x_j -- no implicit 1/2.
// The engine solves 1/2 x'Px + q'x, so a diagonal term q_ii x_i^2 requires
// P_ii = 2*q_ii to mean the same function.
//
// min 3x^2 - 12x  ->  f' = 6x - 12 = 0  ->  x = 2, f = 3*4 - 24 = -12.
// If P_ii were set to q_ii instead of 2*q_ii the engine would minimise
// 1.5x^2 - 12x, giving x = 4 and -24 -- so this discriminates the convention
// rather than merely checking that something was solved.
void testQpDiagonalQuadraticCoefficient() {
    Builder b;
    b.var("x", model::VariableType::Continuous, -100.0, 100.0);
    b.row("c0", -INF, 100.0, {{0, 1.0}});
    b.obj(model::ObjectiveSense::Minimize, {{0, -12.0}});
    b.m.objective.quadraticTerms = {{0, 0, 3.0}};  // 3x^2

    const auto r = solver::solve(b.m);
    report("QP diagonal coefficient", r);
    ck(r.status == solver::SolveStatus::Optimal, "diagonal QP solves");
    near(r.variableValues[0], 2.0, 1e-4, "diagonal QP x (4.0 would mean P_ii = q_ii)");
    near(r.objectiveValue, -12.0, 1e-4, "diagonal QP objective");
}

// The off-diagonal case takes the OTHER convention: a single stored term
// q_ij x_i x_j (i != j) means P_ij = P_ji = q_ij, NOT 2*q_ij, because the
// matrix form's (i,j) and (j,i) entries together already reproduce the one
// stored coefficient.
//
// min x^2 + y^2 + xy - 3x - 3y
//   df/dx = 2x + y - 3 = 0,  df/dy = x + 2y - 3 = 0  ->  x = y = 1
//   f(1,1) = 1 + 1 + 1 - 3 - 3 = -3
// Doubling the off-diagonal would make the Hessian singular along x+y and
// move the answer, so this pins the convention too.
void testQpOffDiagonalQuadraticCoefficient() {
    Builder b;
    b.var("x", model::VariableType::Continuous, -100.0, 100.0);
    b.var("y", model::VariableType::Continuous, -100.0, 100.0);
    b.row("c0", -INF, 100.0, {{0, 1.0}, {1, 1.0}});
    b.obj(model::ObjectiveSense::Minimize, {{0, -3.0}, {1, -3.0}});
    b.m.objective.quadraticTerms = {{0, 0, 1.0}, {1, 1, 1.0}, {0, 1, 1.0}};

    const auto r = solver::solve(b.m);
    report("QP off-diagonal coefficient", r);
    ck(r.status == solver::SolveStatus::Optimal, "off-diagonal QP solves");
    near(r.variableValues[0], 1.0, 1e-4, "off-diagonal QP x");
    near(r.variableValues[1], 1.0, 1e-4, "off-diagonal QP y");
    near(r.objectiveValue, -3.0, 1e-4, "off-diagonal QP objective");
}

// A quadratic objective with integer variables has no engine here, and must be
// reported rather than silently relaxed.
void testMiqpIsReportedUnsupported() {
    Builder b;
    b.var("b0", model::VariableType::Binary, 0.0, 1.0);
    b.var("x", model::VariableType::Continuous, 0.0, 10.0);
    b.row("c0", -INF, 5.0, {{0, 1.0}, {1, 1.0}});
    b.obj(model::ObjectiveSense::Minimize, {{1, -2.0}});
    b.m.objective.quadraticTerms = {{1, 1, 1.0}};

    const auto r = solver::solve(b.m);
    report("MIQP (no engine)", r);
    ck(r.engine == solver::Engine::Unsupported, "MIQP is unsupported");
    ck(!r.engineReason.empty(), "and says why");
}

// The QP adapter appends identity rows for variable bounds, so its dual vector
// is longer than the model's constraint list. The pipeline must trim it, or
// postsolve would index past the end.
void testQpDualsMatchConstraintCount() {
    Builder b;
    b.var("x", model::VariableType::Continuous, -1.0, 1.0);
    b.var("y", model::VariableType::Continuous, -1.0, 1.0);
    b.row("c0", -INF, 0.5, {{0, 1.0}, {1, 1.0}});
    b.row("c1", -INF, 0.5, {{0, 1.0}});
    b.obj(model::ObjectiveSense::Minimize, {{0, -1.0}, {1, -1.0}});
    b.m.objective.quadraticTerms = {{0, 0, 0.5}, {1, 1, 0.5}};

    const auto r = solver::solve(b.m);
    report("QP dual vector length", r);
    ck(r.status == solver::SolveStatus::Optimal, "QP with bounds solves");
    ck(!r.hasDuals || r.constraintDuals.size() == b.m.constraints.size(),
       "duals are one per original constraint, not one per QP row");
}

}  // namespace

int main() {
    std::printf("%-36s %-14s %-14s %-20s %s\n",
                "  case", "status", "engine", "objective", "duals");
    std::printf("  %s\n", std::string(94, '-').c_str());

    testProductMix();
    testDiet();
    testDegenerate();
    testObjectiveOffset();
    testInfeasibleLp();
    testUnboundedLp();
    testRequireVertexSolutionRunsDualSimplex();
    testSmallLpAutoRoutesToDualSimplex();
    testVertexOnMultipleOptimumLp();
    testVertexRequestTooLargeIsRefused();
    testNumericalFailureIsNotReportedAsInvalidModel();
    testDispatchDecisionMatchesExecutedEngine();
    testKnapsack();
    testAssignment();
    testSetCovering();
    testFractionalRelaxation();
    testMilpReducedToLp();
    testQpDiagonalQuadraticCoefficient();
    testQpOffDiagonalQuadraticCoefficient();
    testQuadraticRoutesToQpEngine();
    testMaximizeQuadraticRoutesToQpEngine();
    testQpWithObjectiveOffset();
    testMaximizeQpWithObjectiveOffset();
    testMiqpIsReportedUnsupported();
    testQpDualsMatchConstraintCount();
    testUniformResultContract();

    std::printf("\n%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
