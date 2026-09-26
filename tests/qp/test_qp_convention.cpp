// Section 3 of the audit brief, verbatim, plus the convention checks.
// Every expected value is derived by hand in the comment above its case,
// independently of the code under test.
#include "solver/orchestrator.h"
#include "qp/qp_adapter.h"
#include "model/model.h"
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

static const double INF = std::numeric_limits<double>::infinity();
static int checks = 0, failures = 0;

static void ck(bool ok, const std::string& what) {
    ++checks;
    if (!ok) { ++failures; std::printf("   FAIL  %s\n", what.c_str()); }
}
static void near(double got, double want, double tol, const std::string& what) {
    ++checks;
    if (!(std::abs(got - want) <= tol)) {
        ++failures;
        std::printf("   FAIL  %s: got %.12g want %.12g (tol %g)\n", what.c_str(), got, want, tol);
    }
}

struct B {
    model::Model m;
    void var(const char* n, double lo, double hi) {
        model::Variable v; v.name=n; v.lowerBound=lo; v.upperBound=hi;
        v.type=model::VariableType::Continuous; m.variables.push_back(v);
    }
    void row(const char* n, double lo, double hi, std::vector<model::LinearTerm> t) {
        model::Constraint c; c.name=n; c.lowerBound=lo; c.upperBound=hi;
        c.linearTerms=std::move(t); m.constraints.push_back(c);
    }
};

// Evaluate the ORIGINAL model's objective directly from its own convention:
//   f(x) = offset + sum c_i x_i + sum q_ij x_i x_j
// This is the independent oracle the reported objective is checked against.
static double evalModel(const model::Model& m, const std::vector<double>& x) {
    double f = m.objective.offset;
    for (const auto& t : m.objective.linearTerms) f += t.value * x[t.variableIndex];
    for (const auto& q : m.objective.quadraticTerms)
        f += q.value * x[q.variableIndex1] * x[q.variableIndex2];
    return f;
}

// Dense P entry, read out of the adapter by probing with unit vectors.
static double Pentry(const model::Model& m, int i, int j) {
    auto qm = qp::fromModel(m);
    const int n = qm.numVariables();
    std::vector<double> e(n, 0.0), y;
    e[j] = 1.0;
    qm.P.multiply(e, y);
    return y[i];
}

static void report(const char* tag, const model::Model& m, const solver::SolveResult& r) {
    std::printf("  %-42s %-12s %-13s obj=%12.8f", tag,
        solver::toString(r.status), solver::toString(r.engine), r.objectiveValue);
    if (!r.variableValues.empty()) {
        std::printf("  x=[");
        for (size_t i=0;i<r.variableValues.size();++i)
            std::printf("%s%.8f", i?", ":"", r.variableValues[i]);
        std::printf("]");
    }
    std::printf("\n");
    (void)m;
}

// Shared: solve, then check status, bounds feasibility, row feasibility, and
// that the REPORTED objective equals a direct evaluation of the ORIGINAL model.
static void checkCase(const char* tag, const model::Model& m,
                      const std::vector<double>& expectX, double expectObj,
                      double tolX, double tolObj) {
    const auto r = solver::solve(m);
    report(tag, m, r);
    ck(r.status == solver::SolveStatus::Optimal, std::string(tag) + ": status Optimal");
    if (r.status != solver::SolveStatus::Optimal) return;
    ck(r.hasPrimal && r.variableValues.size() == m.variables.size(),
       std::string(tag) + ": all original variables are reconstructed");
    if (r.variableValues.size() != m.variables.size()) return;

    for (size_t j = 0; j < m.variables.size(); ++j) {
        ck(r.variableValues[j] >= m.variables[j].lowerBound - 1e-7 &&
           r.variableValues[j] <= m.variables[j].upperBound + 1e-7,
           std::string(tag) + ": x[" + std::to_string(j) + "] within bounds");
    }
    for (const auto& c : m.constraints) {
        double a = 0.0;
        for (const auto& t : c.linearTerms) a += t.value * r.variableValues[t.variableIndex];
        ck(a >= c.lowerBound - 1e-6 && a <= c.upperBound + 1e-6,
           std::string(tag) + ": row " + c.name + " satisfied");
    }
    for (size_t j = 0; j < expectX.size(); ++j)
        near(r.variableValues[j], expectX[j], tolX, std::string(tag) + ": x[" + std::to_string(j) + "]");
    near(r.objectiveValue, expectObj, tolObj, std::string(tag) + ": reported objective");
    // The reported objective must equal the model's own objective at the
    // returned point -- catches a sense flip or offset applied twice.
    near(r.objectiveValue, evalModel(m, r.variableValues), 1e-6,
         std::string(tag) + ": reported objective == direct evaluation of Model");
}


// ---------------------------------------------------------------------------
// QP DUAL CONVENTION
//
// A row dual must be a shadow price, d(objective)/d(right-hand side), in the
// model's OWN sense -- the same convention HiGHS reports and the one postsolve
// requires.
//
// This was wrong and nothing caught it. qp::fromModel had no return half, so
// AdmmResult::constraintDual reached callers unmodified. The ADMM's
// stationarity is P x + q + A^T y = 0 while a Lagrange multiplier satisfies
// A^T lambda = grad f, making y = -lambda. Measured on "min x^2 + y^2 s.t.
// x + y >= 2" the engine returned -2 where the shadow price is +2. The primal
// was exact, so no status looked wrong; postsolve's residual gate quietly
// refused to publish the duals and every QP solve reported optimal with no
// sensitivities at all.
//
// A maximisation negates AGAIN, because fromModel negates P and q to hand the
// engine a minimisation. The two do not cancel:
//     minimisation:  shadow price = -y_admm
//     maximisation:  shadow price = +y_admm
//
// Both rows below use TWO terms on purpose. A single-term row is a bound in
// disguise and presolve removes it, leaving no dual to check -- which is how a
// weaker version of this test passed while the bug was live.
// ---------------------------------------------------------------------------
static void checkDual(const char* tag, const model::Model& m,
                      double wantObjective, double wantDual, double tol) {
    const solver::SolveResult r = solver::solve(m, {});
    ck(r.status == solver::SolveStatus::Optimal,
       std::string(tag) + ": optimal");
    near(r.objectiveValue, wantObjective, tol, std::string(tag) + ": objective");
    ck(r.hasDuals && r.constraintDuals.size() == 1,
       std::string(tag) + ": one row dual is reported");
    if (r.hasDuals && r.constraintDuals.size() == 1) {
        near(r.constraintDuals[0], wantDual, tol, std::string(tag) + ": shadow price");
        // Sign is the part that was wrong, so assert it on its own rather than
        // letting a tolerance on the magnitude hide it.
        ck((r.constraintDuals[0] > 0.0) == (wantDual > 0.0),
           std::string(tag) + ": shadow price has the correct SIGN");
    }
}

static void qpDualConventionCases() {
    // min x^2 + y^2 s.t. x + y >= 2  ->  x=y=1, f=2.
    // grad f = (2,2); A^T lambda = grad f gives lambda = 2. Tightening the row
    // by one unit raises the objective, so the shadow price is POSITIVE.
    {
        B b; b.var("x",-10.0,10.0); b.var("y",-10.0,10.0);
        b.row("c0", 2.0, INF, {{0,1.0},{1,1.0}});
        b.m.objective.sense = model::ObjectiveSense::Minimize;
        b.m.objective.quadraticTerms = {{0,0,1.0},{1,1,1.0}};
        checkDual("min QP, >= row", b.m, 2.0, 2.0, 1e-4);
    }

    // max -x^2 - y^2 + 4x + 4y s.t. x + y <= 3  ->  x=y=1.5, f=7.5.
    // With x=y=u/2, f(u) = -u^2/2 + 4u, so df/du = -u + 4 = +1 at u=3.
    // POSITIVE: relaxing a binding <= row of a maximisation helps. Reporting a
    // negative number here would claim the opposite.
    {
        B b; b.var("x",-10.0,10.0); b.var("y",-10.0,10.0);
        b.row("c0", -INF, 3.0, {{0,1.0},{1,1.0}});
        b.m.objective.sense = model::ObjectiveSense::Maximize;
        b.m.objective.linearTerms = {{0,4.0},{1,4.0}};
        b.m.objective.quadraticTerms = {{0,0,-1.0},{1,1,-1.0}};
        checkDual("max QP, <= row", b.m, 7.5, 1.0, 1e-4);
    }

    // Strong duality must reproduce the objective from the multipliers. This is
    // the check that fails for ANY wrong sign or scale, not just a flip.
    {
        B b; b.var("x",-10.0,10.0); b.var("y",-10.0,10.0);
        b.row("c0", 2.0, INF, {{0,1.0},{1,1.0}});
        b.m.objective.sense = model::ObjectiveSense::Minimize;
        b.m.objective.quadraticTerms = {{0,0,1.0},{1,1,1.0}};
        const solver::SolveResult r = solver::solve(b.m, {});
        if (r.hasDuals && r.constraintDuals.size() == 1 && r.variableValues.size() == 2) {
            // For min 0.5 x'Px + q'x with an active row, the dual objective is
            //   rhs * y  -  (quadratic part evaluated at x*)
            const double quadratic = r.variableValues[0]*r.variableValues[0] +
                                     r.variableValues[1]*r.variableValues[1];
            const double dualObjective = 2.0 * r.constraintDuals[0] - quadratic;
            near(dualObjective, r.objectiveValue, 1e-4,
                 "min QP: strong duality reproduces the objective from the dual");
        }
    }
}

int main() {
    std::printf("=== Section 3: QP convention, sense, and offset ===\n");

    // (a) min 3x^2, 1 <= x <= 2.
    // 3x^2 is increasing on [1,2], so the minimum is the lower bound x=1,
    // f = 3*1 = 3. Adapter must produce P_xx = 6 (diagonal q_ii -> 2*q_ii,
    // since 0.5*P_xx*x^2 must equal q_ii*x^2).
    {
        B b; b.var("x", 1.0, 2.0);
        b.m.objective.sense = model::ObjectiveSense::Minimize;
        b.m.objective.quadraticTerms = {{0,0,3.0}};
        near(Pentry(b.m, 0, 0), 6.0, 1e-12, "(a) adapter P_xx == 6");
        checkCase("(a) min 3x^2, 1<=x<=2", b.m, {1.0}, 3.0, 1e-6, 1e-6);
    }

    // (b) min x^2 + y^2 + xy - 3x - 3y, bounds [-10,10].
    // grad = (2x + y - 3, x + 2y - 3) = 0  =>  x = y = 1 (interior).
    // f(1,1) = 1 + 1 + 1 - 3 - 3 = -3.
    // Off-diagonal convention: P_xy = P_yx = 1 (NOT 2) since a single stored
    // q_xy already means the whole xy coefficient.
    {
        B b; b.var("x", -10.0, 10.0); b.var("y", -10.0, 10.0);
        b.m.objective.sense = model::ObjectiveSense::Minimize;
        b.m.objective.linearTerms = {{0,-3.0},{1,-3.0}};
        b.m.objective.quadraticTerms = {{0,0,1.0},{1,1,1.0},{0,1,1.0}};
        near(Pentry(b.m,0,0), 2.0, 1e-12, "(b) P_xx == 2");
        near(Pentry(b.m,1,1), 2.0, 1e-12, "(b) P_yy == 2");
        near(Pentry(b.m,0,1), 1.0, 1e-12, "(b) P_xy == 1 (undoubled)");
        near(Pentry(b.m,1,0), 1.0, 1e-12, "(b) P_yx == 1 (symmetric)");
        checkCase("(b) min x^2+y^2+xy-3x-3y", b.m, {1.0,1.0}, -3.0, 1e-5, 1e-5);
    }

    // (c) max -x^2 + 4x, bounds [-10,10].
    // f' = -2x + 4 = 0 => x = 2 (interior, concave so it is the max).
    // f(2) = -4 + 8 = 4.
    {
        B b; b.var("x", -10.0, 10.0);
        b.m.objective.sense = model::ObjectiveSense::Maximize;
        b.m.objective.linearTerms = {{0,4.0}};
        b.m.objective.quadraticTerms = {{0,0,-1.0}};
        checkCase("(c) max -x^2+4x", b.m, {2.0}, 4.0, 1e-5, 1e-5);
    }

    // (d) min 100 + x^2, bounds [-10,10]. Minimum at x=0, f = 100.
    {
        B b; b.var("x", -10.0, 10.0);
        b.m.objective.sense = model::ObjectiveSense::Minimize;
        b.m.objective.offset = 100.0;
        b.m.objective.quadraticTerms = {{0,0,1.0}};
        checkCase("(d) min 100+x^2", b.m, {0.0}, 100.0, 1e-5, 1e-5);
    }

    // (e) max 100 - x^2 + 4x, bounds [-10,10].
    // f' = -2x + 4 = 0 => x = 2. f(2) = 100 - 4 + 8 = 104.
    {
        B b; b.var("x", -10.0, 10.0);
        b.m.objective.sense = model::ObjectiveSense::Maximize;
        b.m.objective.offset = 100.0;
        b.m.objective.linearTerms = {{0,4.0}};
        b.m.objective.quadraticTerms = {{0,0,-1.0}};
        checkCase("(e) max 100-x^2+4x", b.m, {2.0}, 104.0, 1e-5, 1e-5);
    }

    std::printf("\n=== Repeated / duplicate quadratic terms accumulate ===\n");
    // Two stored terms on the same diagonal cell must sum: 1x^2 + 2x^2 = 3x^2,
    // so P_xx = 6, and the answer must match case (a) exactly.
    {
        B b; b.var("x", 1.0, 2.0);
        b.m.objective.sense = model::ObjectiveSense::Minimize;
        b.m.objective.quadraticTerms = {{0,0,1.0},{0,0,2.0}};
        near(Pentry(b.m,0,0), 6.0, 1e-12, "dup diagonal: P_xx == 6 (1+2 -> 3 -> 2*3)");
        checkCase("dup diagonal min (1+2)x^2, 1<=x<=2", b.m, {1.0}, 3.0, 1e-6, 1e-6);
    }
    // Duplicate off-diagonal, and the mirrored index form (i,j)+(j,i):
    // 0.5*xy + 0.5*xy stored as (0,1) twice must equal a single xy term.
    {
        B b; b.var("x", -10.0, 10.0); b.var("y", -10.0, 10.0);
        b.m.objective.sense = model::ObjectiveSense::Minimize;
        b.m.objective.linearTerms = {{0,-3.0},{1,-3.0}};
        b.m.objective.quadraticTerms = {{0,0,1.0},{1,1,1.0},{0,1,0.5},{1,0,0.5}};
        near(Pentry(b.m,0,1), 1.0, 1e-12, "dup off-diagonal (0,1)+(1,0) -> P_xy == 1");
        near(Pentry(b.m,1,0), 1.0, 1e-12, "dup off-diagonal -> P_yx == 1");
        checkCase("dup off-diagonal, same as (b)", b.m, {1.0,1.0}, -3.0, 1e-5, 1e-5);
    }

    std::printf("\n=== Zero quadratic terms / degenerate quadratic ===\n");
    // An explicitly zero quadratic term still classifies as QP; the answer is
    // the LP answer. min -x on [0,5] -> x=5, f=-5.
    {
        B b; b.var("x", 0.0, 5.0);
        b.row("c0", -INF, 10.0, {{0,1.0}});
        b.m.objective.sense = model::ObjectiveSense::Minimize;
        b.m.objective.linearTerms = {{0,-1.0}};
        b.m.objective.quadraticTerms = {{0,0,0.0}};
        checkCase("zero quadratic term (LP in QP clothing)", b.m, {5.0}, -5.0, 1e-5, 1e-5);
    }

    std::printf("\n=== Equality rows and fixed variables ===\n");
    // min x^2 + y^2 s.t. x + y == 2, both in [-10,10].
    // Symmetric, so x=y=1 by Lagrange: grad = lambda*(1,1) -> 2x=2y. f = 2.
    {
        B b; b.var("x",-10.0,10.0); b.var("y",-10.0,10.0);
        b.row("eq", 2.0, 2.0, {{0,1.0},{1,1.0}});
        b.m.objective.sense = model::ObjectiveSense::Minimize;
        b.m.objective.quadraticTerms = {{0,0,1.0},{1,1,1.0}};
        checkCase("equality row: min x^2+y^2 s.t. x+y=2", b.m, {1.0,1.0}, 2.0, 1e-4, 1e-4);
    }
    // Fixed variable (lo == hi). min x^2 + y^2 with x fixed at 3, y in [-10,10]
    // -> x=3, y=0, f = 9.
    {
        B b; b.var("x",3.0,3.0); b.var("y",-10.0,10.0);
        b.row("c0", -INF, 100.0, {{0,1.0},{1,1.0}});
        b.m.objective.sense = model::ObjectiveSense::Minimize;
        b.m.objective.quadraticTerms = {{0,0,1.0},{1,1,1.0}};
        checkCase("fixed variable x==3", b.m, {3.0,0.0}, 9.0, 1e-4, 1e-4);
    }

    qpDualConventionCases();

    std::printf("\n%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
