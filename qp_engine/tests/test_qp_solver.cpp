// Comprehensive tests for the QP engine.
// Run: ./qp_tests
#include "qp/admm_solver.h"
#include "qp/qp_adapter.h"
#include "qp/kkt_solver.h"
#include "qp/polishing.h"
#include "qp/qp_model.h"
#include "qp/qp_solver.h"
#include "qp/qp_types.h"
#include "qp/scaling.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

int failures = 0;

void require(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

void requireNear(double a, double b, double tol, const std::string& msg) {
    if (!(std::abs(a - b) <= tol))
        throw std::runtime_error(msg + " (got " + std::to_string(a) + " expected " + std::to_string(b) + ")");
}

qp::SparseMatrix mkMat(int rows, int cols,
    std::initializer_list<std::tuple<int,int,double>> t) {
    std::vector<double> r, c, v;
    for (auto it = t.begin(); it != t.end(); ++it) {
        r.push_back(static_cast<double>(std::get<0>(*it)));
        c.push_back(static_cast<double>(std::get<1>(*it)));
        v.push_back(std::get<2>(*it));
    }
    return qp::SparseMatrix::fromTriplets(rows, cols, r, c, v);
}

qp::SparseMatrix mkMat(int rows, int cols,
    const std::vector<double>& r,
    const std::vector<double>& c,
    const std::vector<double>& v) {
    return qp::SparseMatrix::fromTriplets(rows, cols, r, c, v);
}

qp::AdmmOptions strictOpts(int iters = 5000) {
    qp::AdmmOptions o;
    o.iterationLimit = iters;
    o.primalTolerance = 1e-6;
    o.dualTolerance = 1e-6;
    o.useRuizScaling = false;
    o.useAdaptiveRho = false;
    o.rho = 1.0;
    return o;
}

// ---------------------------------------------------------------------------
// Sparse matrix
// ---------------------------------------------------------------------------

void testSparseDuplicateTriplets() {
    auto M = mkMat(1, 1, {{0,0,2.0}, {0,0,-1.0}});
    require(M.nonzeros() == 1, "duplicates must merge");
    std::vector<double> y;
    M.multiply({4.0}, y);
    requireNear(y[0], 4.0, 1e-12, "merged coefficient");
}

void testSparseCancelling() {
    auto M = mkMat(1, 2, {{0,0,2.0}, {0,0,-2.0}, {0,1,3.0}});
    require(M.nonzeros() == 1, "cancelling entries must be dropped");
    require(M.validate(), "matrix must validate");
}

void testSparseUnsortedAndEmpty() {
    auto M = mkMat(3, 3, {{2,2,5.0}, {0,2,1.0}, {2,0,4.0}, {0,0,2.0}});
    require(M.validate(), "matrix from unsorted triplets must validate");
    std::vector<double> y;
    M.multiply({1.0, 1.0, 1.0}, y);
    requireNear(y[0], 3.0, 1e-12, "row 0 product");
    requireNear(y[1], 0.0, 1e-12, "empty row product");
    requireNear(y[2], 9.0, 1e-12, "row 2 product");
    std::vector<double> z;
    M.transposeMultiply({1.0, 1.0, 1.0}, z);
    requireNear(z[0], 6.0, 1e-12, "col 0 product");
    requireNear(z[1], 0.0, 1e-12, "empty col product");
    requireNear(z[2], 6.0, 1e-12, "col 2 product");
}

void testSparseCsrCscAgree() {
    std::mt19937 g(7);
    std::uniform_real_distribution<double> dist(-2.0, 2.0);
    std::uniform_int_distribution<int> pick(0, 39);
    std::vector<double> rr, cc, vv;
    for (int k = 0; k < 400; ++k) {
        rr.push_back(static_cast<double>(pick(g) % 25));
        cc.push_back(static_cast<double>(pick(g)));
        vv.push_back(dist(g));
    }
    auto M = qp::SparseMatrix::fromTriplets(25, 40, rr, cc, vv);
    require(M.validate(), "random matrix must validate");
    std::vector<double> x(40);
    std::vector<double> y(25);
    for (double& v : x) v = dist(g);
    for (double& v : y) v = dist(g);
    std::vector<double> ax, aty;
    M.multiply(x, ax);
    M.transposeMultiply(y, aty);
    double lhs = 0.0, rhs = 0.0;
    for (int i = 0; i < 25; ++i) lhs += ax[i] * y[i];
    for (int j = 0; j < 40; ++j) rhs += x[j] * aty[j];
    requireNear(lhs, rhs, 1e-9 * (1.0 + std::abs(lhs)), "CSR/CSC adjoint");
}

void testSparseScaling() {
    auto M = mkMat(2, 2, {{0,0,1.0}, {0,1,2.0}, {1,0,3.0}, {1,1,4.0}});
    // Row scale {10, 100}, col scale {1, 0.5}.
    // a00: 1*10*1=10, a01: 2*10*0.5=10, a10: 3*100*1=300, a11: 4*100*0.5=200.
    auto S = M.scaled({10.0, 100.0}, {1.0, 0.5});
    std::vector<double> y;
    S.multiply({1.0, 0.0}, y);
    requireNear(y[0], 10.0, 1e-12, "scaled a00");
    requireNear(y[1], 300.0, 1e-12, "scaled a10");
    S.multiply({0.0, 1.0}, y);
    requireNear(y[0], 10.0, 1e-12, "scaled a01");
    requireNear(y[1], 200.0, 1e-12, "scaled a11");
}

// ---------------------------------------------------------------------------
// KKT solver
// ---------------------------------------------------------------------------

void testKktDenseUnconstrained() {
    qp::QpModel m;
    m.P = mkMat(2, 2, {{0,0,2.0}, {1,1,3.0}});
    m.A = mkMat(0, 2, {});
    m.q = {0.0, 0.0};
    m.l = {}; m.u = {};
    qp::KktSolver kkt(m, 1.0);
    std::vector<double> x = {1.0, 1.0};
    require(kkt.solve(x), "KKT solve");
    // K = P + sigma*I (no constraints), so K = diag(2+sigma, 3+sigma). The
    // sigma is part of the solver's contract -- the x-update cancels it against
    // a matching sigma*x_prev on the right-hand side -- so the expectation
    // carries it explicitly rather than the tolerance being widened to hide it.
    const double s = qp::KktSolver::kSigma;
    requireNear(x[0], 1.0 / (2.0 + s), 1e-12, "KKT x0");
    requireNear(x[1], 1.0 / (3.0 + s), 1e-12, "KKT x1");
}

void testKktWithRho() {
    // P = I, A^T A = 2I (from A = [1 1; 1 -1]).
    // K = I + sigma*I + 2*2*I = (5 + sigma)I.  Solve K x = [10, 5].
    qp::QpModel m;
    m.P = mkMat(2, 2, {{0,0,1.0}, {1,1,1.0}});
    m.A = mkMat(2, 2, {{0,0,1.0},{0,1,1.0},{1,0,1.0},{1,1,-1.0}});
    m.q = {0.0, 0.0};
    m.l = {0.0, 0.0}; m.u = {0.0, 0.0};
    qp::KktSolver kkt(m, 2.0);
    std::vector<double> x = {10.0, 5.0};
    require(kkt.solve(x), "solve");
    const double s2 = qp::KktSolver::kSigma;
    requireNear(x[0], 10.0 / (5.0 + s2), 1e-12, "x0");
    requireNear(x[1],  5.0 / (5.0 + s2), 1e-12, "x1");
}

void testKktRefactor() {
    qp::QpModel m;
    m.P = mkMat(2, 2, {{0,0,1.0}, {1,1,1.0}});
    m.A = mkMat(0, 2, {});
    m.q = {0.0, 0.0};
    m.l = {}; m.u = {};
    qp::KktSolver kkt(m, 1.0);
    std::vector<double> x = {3.0, 4.0};
    require(kkt.refactor(2.0), "refactor");
    require(kkt.solve(x), "solve after refactor");
    // K = P + sigma*I = (1 + sigma)I; rho is irrelevant with no constraints.
    const double s3 = qp::KktSolver::kSigma;
    requireNear(x[0], 3.0 / (1.0 + s3), 1e-12, "x0");
    requireNear(x[1], 4.0 / (1.0 + s3), 1e-12, "x1");
}

// ---------------------------------------------------------------------------
// Scaling
// ---------------------------------------------------------------------------

void testRuizConditioning() {
    qp::QpModel m;
    m.P = mkMat(2, 2, {{0,0,1.0}, {1,1,1.0}});
    m.A = mkMat(2, 2, {{0,0,1e-5}, {0,1,2e-5}, {1,0,1e5}, {1,1,3e5}});
    m.q = {0.0, 0.0};
    m.l = {0.0, 0.0}; m.u = {1.0, 1.0};
    double before = qp::RuizScaler::conditionSpread(m.A);
    auto sc = qp::RuizScaler::equilibrate(m, 10);
    double after = qp::RuizScaler::conditionSpread(sc.scaled.A);
    require(before > 1e6, "test matrix should be badly scaled");
    require(after < 10.0, "equilibration should improve conditioning, got " + std::to_string(after));
}

void testRuizNoConstraints() {
    // m == 0: column scaling of P should still run without error.
    qp::QpModel m;
    m.P = mkMat(2, 2, {{0,0,1e-8}, {1,1,1e8}});
    m.A = mkMat(0, 2, {});
    m.q = {1.0, 1.0};
    m.l = {}; m.u = {};
    auto sc = qp::RuizScaler::equilibrate(m, 5);
    require(sc.scaled.q.size() == 2, "scaled q must have n entries");
}

// ---------------------------------------------------------------------------
// ADMM
// ---------------------------------------------------------------------------

void testUnconstrainedQp() {
    // min 0.5*x^2 + 2x => optimum x=-2, obj=-2
    qp::QpModel m;
    m.P = mkMat(1, 1, {{0,0,1.0}});
    m.A = mkMat(0, 1, {});
    m.q = {2.0};
    m.l = {}; m.u = {};
    auto o = strictOpts();
    o.rho = 1.0;
    auto r = qp::AdmmSolver(m, o).solve();
    require(r.status == qp::QpStatus::Optimal, "unconstrained: status");
    requireNear(r.primal[0], -2.0, 1e-3, "unconstrained: x");
    requireNear(r.primalObjective, -2.0, 1e-3, "unconstrained: obj");
}

void testUnconstrainedQuadratic() {
    // min 0.5*(x0^2 + x1^2) + x0 + 2*x1  =>  optimum x=[-1, -2]
    qp::QpModel m;
    m.P = mkMat(2, 2, {{0,0,1.0}, {1,1,1.0}});
    m.A = mkMat(0, 2, {});
    m.q = {1.0, 2.0};
    m.l = {}; m.u = {};
    auto o = strictOpts();
    o.rho = 1.0;
    auto r = qp::AdmmSolver(m, o).solve();
    require(r.status == qp::QpStatus::Optimal, "quad: status");
    requireNear(r.primal[0], -1.0, 1e-3, "quad: x0");
    requireNear(r.primal[1], -2.0, 1e-3, "quad: x1");
}

void testBoxedEquality() {
    // Same as above but x,y >= -1 (box binding on y).
    qp::QpModel m;
    m.P = mkMat(2, 2, {{0,0,1.0}, {1,1,1.0}});
    m.A = mkMat(2, 2, {{0,0,1.0}, {1,1,1.0}});
    m.q = {1.0, 2.0};
    m.l = {-1e300, -1.0}; m.u = {1e300, 1.0};
    auto o = strictOpts();
    o.rho = 10.0;
    auto r = qp::AdmmSolver(m, o).solve();
    require(r.status == qp::QpStatus::Optimal, "boxed: status");
    requireNear(r.primal[0], -1.0, 5e-3, "boxed: x0");
    requireNear(r.primal[1], -1.0, 5e-3, "boxed: x1");
}

void testRangedRow() {
    // min 0.5*2*x^2 + x s.t. 3 <= 2*x <= 5.
    // Unconstrained optimum is x=-0.5 (below feasible range), so the
    // nearest feasible point is x=1.5 (binding at the lower constraint).
    //   0.5*2*1.5^2 + 1.5 = 2.25 + 1.5 = 3.75.
    qp::QpModel m;
    m.P = mkMat(1, 1, {{0,0,2.0}});
    m.A = mkMat(1, 1, {{0,0,2.0}});
    m.q = {1.0};
    m.l = {3.0}; m.u = {5.0};
    auto o = strictOpts();
    o.rho = 10.0;
    auto r = qp::AdmmSolver(m, o).solve();
    require(r.status == qp::QpStatus::Optimal, "ranged: status");
    requireNear(r.primal[0], 1.5, 1e-2, "ranged: x");
    // Check feasibility: 2*x should be in [3, 5].
    require(r.primalResidual < 0.1, "ranged: feasibility");
}

void testIterationLimit() {
    // The iteration limit must be respected and reported.
    //
    // This previously used a trivially separable problem and asserted the solver
    // COULD NOT converge within 10 iterations. That held only while the ADMM
    // x-update was wrong; once fixed, the same problem converges in 3 iterations
    // to x = 0, so the test was really asserting the presence of a bug. It now
    // uses a badly conditioned Hessian with a tolerance no run can reach in the
    // budget, and checks the contract that matters: stop at the limit, and say so.
    qp::QpModel m;
    const int n = 40;
    std::vector<double> rv, cv, vv;
    for (int j = 0; j < n; ++j) {
        rv.push_back(static_cast<double>(j));
        cv.push_back(static_cast<double>(j));
        // Diagonal, so positive definite by construction, but with a condition
        // number of 1e12. The coupling that makes the iteration slow comes from
        // A rather than from P, which keeps the Hessian factorable.
        vv.push_back(j % 2 == 0 ? 1e-6 : 1e6);
    }
    m.P = mkMat(n, n, rv, cv, vv);

    std::vector<double> ar, ac, av;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            ar.push_back(static_cast<double>(i));
            ac.push_back(static_cast<double>(j));
            av.push_back(((i + j) % 3) - 1.0);
        }
    }
    m.A = mkMat(n, n, ar, ac, av);
    m.q.assign(static_cast<std::size_t>(n), 1.0);
    m.l.assign(static_cast<std::size_t>(n), -1.0);
    m.u.assign(static_cast<std::size_t>(n), 1.0);

    auto o = strictOpts(5);
    o.iterationLimit = 5;
    o.primalTolerance = 1e-14;
    o.dualTolerance = 1e-14;
    o.rho = 1.0;
    o.useAdaptiveRho = false;
    o.terminationCheckFrequency = 1;
    auto r = qp::AdmmSolver(m, o).solve();
    require(r.status == qp::QpStatus::IterationLimit, "limit: status");
    require(r.iterations <= 5, "limit: iterations never exceed the budget");
}


void testInvalidCrossedBounds() {
    qp::QpModel m;
    m.P = mkMat(1, 1, {{0,0,1.0}});
    m.A = mkMat(0, 1, {});
    m.q = {1.0};
    m.l = {5.0}; m.u = {3.0};  // crossed
    auto r = qp::AdmmSolver(m, qp::AdmmOptions{}).solve();
    require(r.status == qp::QpStatus::InvalidProblem, "crossed: status");
}

void testIllConditioned() {
    // Very poorly scaled but solvable.
    qp::QpModel m;
    m.P = mkMat(2, 2, {{0,0,1e-8}, {1,1,1e8}});
    m.A = mkMat(0, 2, {});
    m.q = {1.0, 1.0};
    m.l = {}; m.u = {};
    auto o = strictOpts();
    o.useRuizScaling = true;
    o.ruizIterations = 10;
    auto r = qp::AdmmSolver(m, o).solve();
    require(r.status == qp::QpStatus::Optimal, "ill-conditioned: status");
    require(std::isfinite(r.primalObjective), "ill-conditioned: objective finite");
}

// ---------------------------------------------------------------------------
// Polishing
// ---------------------------------------------------------------------------

void testPolishing() {
    // Simple problem: x ≈ [-1, -1] with tight constraint.
    qp::QpModel m;
    m.P = mkMat(2, 2, {{0,0,1.0}, {1,1,1.0}});
    m.A = mkMat(2, 2, {{0,0,1.0}, {1,1,1.0}});
    m.q = {1.0, 2.0};
    m.l = {-1.0, -1.0}; m.u = {-1.0, -1.0};
    auto o = strictOpts();
    o.usePolishing = false;
    auto r = qp::AdmmSolver(m, o).solve();
    require(r.status == qp::QpStatus::Optimal, "polish: ADMM status");

    // Polish the result.
    qp::KktPolisher::Options pol;
    pol.maxIterations = 50;
    pol.tolerance = 1e-8;
    std::vector<double> x = r.primal;
    std::vector<double> dual = r.constraintDual;
    bool ok = qp::KktPolisher::polish(m, x, dual, pol);
    require(ok, "polish: success");
    requireNear(x[0], -1.0, 1e-4, "polish: x0");
    requireNear(x[1], -1.0, 1e-4, "polish: x1");
}

// solveActiveKKT used to read model.u[row] as the active bound value
// UNCONDITIONALLY, on the assumption every active row is an equality
// (l == u) -- true of testPolishing's rows above, which is exactly why that
// test never caught this. A row active at its LOWER bound with an unbounded
// upper side (l finite, u = +inf) had +infinity read into the polish's
// linear solve instead, and the corrupted inf/nan result passed every
// finiteness-blind check downstream, coming back labelled Optimal.
//
// min 0.5*(2.6973205522523434*x0^2 + 4.24105341096078*x1^2) +
//     2*3.0918426574126867*x0*x1 - 1.1198224425561307*x0 + 1.9012079630444096*x1
// s.t. x0 free, x1 >= -2.0739021321103577 (no upper bound)
//
// True optimum independently verified against both scipy.optimize.minimize
// (L-BFGS-B) and OSQP: x = (2.792401339, -2.073902132), obj = -5.338570905,
// with x1 pinned at its (one-sided) lower bound.
void testPolishingWithOneSidedActiveBound() {
    qp::QpModel m;
    m.P = mkMat(2, 2, {
        {0, 0, 2.6973205522523434},
        {1, 1, 4.24105341096078},
        {0, 1, 3.0918426574126867},
        {1, 0, 3.0918426574126867},
    });
    m.A = mkMat(1, 2, {{0, 1, 1.0}});  // identity row on x1 only
    m.q = {-1.1198224425561307, 1.9012079630444096};
    m.l = {-2.0739021321103577};
    m.u = {std::numeric_limits<double>::infinity()};  // one-sided: no upper bound

    auto o = strictOpts(200);
    o.usePolishing = true;
    o.polishingIterations = 50;
    auto r = qp::QpSolver{}.solve(m, o);

    require(r.status == qp::QpStatus::Optimal, "one-sided polish: status");
    require(std::isfinite(r.primalObjective), "one-sided polish: objective is finite");
    require(std::isfinite(r.primal[0]) && std::isfinite(r.primal[1]),
            "one-sided polish: primal is finite, not corrupted to inf/nan");
    requireNear(r.primalObjective, -5.338570905102255, 1e-6, "one-sided polish: objective");
    requireNear(r.primal[0], 2.792401339, 1e-5, "one-sided polish: x0");
    requireNear(r.primal[1], -2.073902132, 1e-5, "one-sided polish: x1 at its lower bound");
}

// ---------------------------------------------------------------------------
// Integration
// ---------------------------------------------------------------------------

void testQpSolverFacade() {
    qp::QpModel m;
    m.P = mkMat(1, 1, {{0,0,1.0}});
    m.A = mkMat(0, 1, {});
    m.q = {2.0};
    m.l = {}; m.u = {};
    qp::AdmmOptions o;
    o.rho = 1.0;
    auto r = qp::QpSolver{}.solve(m, o);
    require(r.status == qp::QpStatus::Optimal, "facade: status");
    requireNear(r.primal[0], -2.0, 1e-3, "facade: x");
}

// ---------------------------------------------------------------------------
// Run helpers
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Production defaults: Ruiz scaling AND adaptive rho, tolerance 1e-8.
//
// Every ADMM test above runs with both switched off (strictOpts), which is why
// none of them could see the defects these two reproduce: they only occur with
// both on, the configuration the orchestrator actually uses.
// ---------------------------------------------------------------------------

qp::AdmmOptions productionOpts() {
    qp::AdmmOptions o;            // library defaults: Ruiz on, adaptive rho on
    o.primalTolerance = 1e-8;     // what the orchestrator passes (SolverOptions)
    o.dualTolerance = 1e-8;
    o.threadCount = 1;
    return o;
}

// Maros-Meszaros hs51: min (x1-x2)^2 + (x2+x3-2)^2 + (x4-1)^2 + (x5-1)^2
// s.t. x1 + 3x2 = 4, x3 + x4 - 2x5 = 0, x2 - x5 = 0, all variables free.
// Optimum x = (1,1,1,1,1), objective 0.
//
// The old engine doubled rho on almost every iteration, reached 2.8e14 after
// 50, froze z, and so zeroed the proxy dual residual rho * A'(z - zOld) that
// its termination test used: it reported Optimal with a TRUE stationarity
// residual of 6e-2 against a 1e-8 tolerance.
void testDefaultsNoFalseOptimal() {
    model::Model m;
    for (int j = 0; j < 5; ++j) {
        model::Variable v; v.name = "x" + std::to_string(j + 1);
        v.lowerBound = -std::numeric_limits<double>::infinity();
        v.upperBound = std::numeric_limits<double>::infinity();
        m.variables.push_back(v);
    }
    m.objective.offset = 6.0;
    m.objective.linearTerms = {{1, -4.0}, {2, -4.0}, {3, -2.0}, {4, -2.0}};
    // Direct coefficients of x_i x_j, no implicit 1/2.
    m.objective.quadraticTerms = {{0, 0, 1.0}, {1, 1, 2.0}, {2, 2, 1.0}, {3, 3, 1.0},
                                  {4, 4, 1.0}, {0, 1, -2.0}, {1, 2, 2.0}};
    auto row = [&](double b, std::vector<model::LinearTerm> terms) {
        model::Constraint c; c.lowerBound = b; c.upperBound = b; c.linearTerms = std::move(terms);
        m.constraints.push_back(c);
    };
    row(4.0, {{0, 1.0}, {1, 3.0}});
    row(0.0, {{2, 1.0}, {3, 1.0}, {4, -2.0}});
    row(0.0, {{1, 1.0}, {4, -1.0}});

    qp::QpTranslation tr;
    const qp::QpModel problem = qp::fromModel(m, tr);
    const qp::AdmmResult r = qp::QpSolver{}.solve(problem, productionOpts());
    require(r.status == qp::QpStatus::Optimal, "hs51 not optimal: " + std::string(qp::toString(r.status)));
    // The true stationarity residual, computed by the engine on the original
    // problem -- the number the old termination test never looked at.
    require(r.dualResidual <= 1e-6, "hs51 claimed Optimal with stationarity residual " +
                                    std::to_string(r.dualResidual));
    for (int j = 0; j < 5; ++j) requireNear(r.primal[static_cast<std::size_t>(j)], 1.0, 1e-5, "hs51 x");
    require(r.finalRho >= 1e-6 && r.finalRho <= 1e6, "rho left its bounds: " + std::to_string(r.finalRho));
}

// Randomized reference case 219 (seed 20260908), exactly as the engine receives
// it after presolve: presolve derives the valid, inactive bound
// x2 >= -7.8124353217753315 from row 1. With that bound the old per-iteration
// rho rule settled into a limit cycle -- the iterate bit-identical after 5,000
// and 200,000 iterations, primal residual 2.34, x0 below its own lower bound.
// The expected optimum is the one OSQP returns and the engine reaches without
// the (redundant) bound.
void testDefaultsNoLimitCycle() {
    const double inf = std::numeric_limits<double>::infinity();
    const double P[7][7] = {
        {-0.050000000000000003, 0, 0, 0, 0, 0, 0},
        {0, -1.2235653774078807, 0.11592843473670784, -0.074542339656287079, 1.2219131625987414, 1.7413301853870775, -2.7969329735916553},
        {0, 0.11592843473670784, -3.0105603839109971, -4.7308039643839646, 1.5229768255189169, -2.139727764785412, 0.012006663589338096},
        {0, -0.074542339656287079, -4.7308039643839646, -10.874355855587034, 2.0072649152787818, 0.75577778908579629, 1.3744644406946396},
        {0, 1.2219131625987414, 1.5229768255189169, 2.0072649152787818, -2.6881420647418564, -0.12782552818603812, 3.5830518668524407},
        {0, 1.7413301853870775, -2.139727764785412, 0.75577778908579629, -0.12782552818603812, -9.75653263257961, 1.7020885237069117},
        {0, -2.7969329735916553, 0.012006663589338096, 1.3744644406946396, 3.5830518668524407, 1.7020885237069117, -9.1873525412981376}};
    const double q[7] = {-1.0934842802908478, 1.3560397711737082, -0.27557571672372616, -1.0878130275257649,
                         -2.169308730748416, 0.54310728369404471, -0.71304681115942015};
    const double lo[7] = {-2.4641103229631929, -inf, -7.8124353217753315, -inf, -4.867479682689579, -2.7205158280632631, -inf};
    const double hi[7] = {inf, inf, inf, 4.0210548225447376, 1.8179023737718132, 4.7476952415039779, inf};

    model::Model m;
    m.objective.sense = model::ObjectiveSense::Maximize;
    m.objective.offset = -33.096687170239171;
    for (int j = 0; j < 7; ++j) {
        model::Variable v; v.name = "x" + std::to_string(j); v.lowerBound = lo[j]; v.upperBound = hi[j];
        m.variables.push_back(v);
        m.objective.linearTerms.push_back({j, q[j]});
    }
    for (int i = 0; i < 7; ++i)
        for (int j = i; j < 7; ++j) {
            if (i == j) { if (P[i][i] != 0.0) m.objective.quadraticTerms.push_back({i, i, P[i][i] / 2.0}); }
            else if (P[i][j] != 0.0) m.objective.quadraticTerms.push_back({i, j, P[i][j]});
        }
    model::Constraint r0; r0.lowerBound = -inf; r0.upperBound = 4.0548536127102697;
    r0.linearTerms = {{0, -0.84117965716685228}, {1, 0.27127501384177766}, {2, 0.48838322252995492}, {6, 0.18298124766769885}};
    model::Constraint r1; r1.lowerBound = -1.8566185531883268; r1.upperBound = 1.8566185531883268;
    r1.linearTerms = {{0, 1.7147010465843435}, {2, -0.78658744203289477}, {3, -0.015750198822212291}};
    m.constraints = {r0, r1};

    qp::QpTranslation tr;
    const qp::QpModel problem = qp::fromModel(m, tr);
    const qp::AdmmResult r = qp::QpSolver{}.solve(problem, productionOpts());
    require(r.status == qp::QpStatus::Optimal, "case 219 not optimal: " + std::string(qp::toString(r.status)) +
                                               " after " + std::to_string(r.iterations) + " iterations");
    const double expected[7] = {-2.464110, 5.866167, -3.012554, 0.066266, -4.214639, 1.253397, -3.268982};
    for (int j = 0; j < 7; ++j) requireNear(r.primal[static_cast<std::size_t>(j)], expected[j], 1e-4, "case 219 x");
    require(r.dualResidual <= 1e-6 && r.primalResidual <= 1e-6, "case 219 residuals");
}

// The NLP engine's first elastic QP for min x^2 s.t. x^2 >= 1, x in [-2, 0.5],
// at x = 0.5, with the NLP engine's own ADMM settings. Nearly an LP: the two
// elastic variables carry a linear penalty of 10 and curvature 1e-8. Optimum
// d = 0, e+ = 0.75, e- = 0.
//
// An UNDAMPED residual-ratio rule oscillated here: an early, almost exactly zero
// dual residual proposed rho 1 -> 1.8e5, then 3e-4, then 1e6, repeating until
// the 20,000-iteration limit, and the NLP engine then reported SubproblemFailure
// where it should reach its own NoProgress verdict.
void testDefaultsNoRhoOscillation() {
    const double inf = std::numeric_limits<double>::infinity();
    qp::QpModel s;
    s.P = mkMat(3, 3, {0, 1, 2}, {0, 1, 2}, {1 + 1e-8, 1e-8, 1e-8});
    s.q = {1, 10, 10};
    s.A = mkMat(4, 3, {0, 0, 0, 1, 2, 3}, {0, 1, 2, 0, 1, 2}, {1, 1, -1, 1, 1, 1});
    s.l = {0.75, -2.5, 0, 0};
    s.u = {inf, 0, inf, inf};
    qp::AdmmOptions o;
    o.iterationLimit = 20000;
    o.primalTolerance = o.dualTolerance = 5e-11;
    o.terminationCheckFrequency = 10;
    o.usePolishing = false;
    o.threadCount = 1;
    const qp::AdmmResult r = qp::QpSolver{}.solve(s, o);
    require(r.status == qp::QpStatus::Optimal, "elastic QP not optimal: " + std::string(qp::toString(r.status)) +
                                               " after " + std::to_string(r.iterations) + " iterations");
    requireNear(r.primal[0], 0.0, 1e-6, "elastic d");
    requireNear(r.primal[1], 0.75, 1e-6, "elastic e+");
    requireNear(r.primal[2], 0.0, 1e-6, "elastic e-");
}

// Maros-Meszaros hs268: five free variables, five >= rows, dense Hessian with
// condition number ~1.2e6. The optimum x* = (1, 2, -1, 3, -4) is the
// unconstrained minimiser (gradient exactly zero there) and happens to be
// feasible, so the primal residual reaches EXACTLY zero while stationarity is
// still being worked on. The objective constant is 14463, so the optimal value
// 0 appears here as -14463.
//
// An exact zero residual must still drive rho down. When the adaptive rule
// skipped zero residuals, rho stopped at 0.1 after one damped step, and the
// solve crawled to the 5,000-iteration limit; it needs rho near 1e-4.
void testDefaultsZeroResidualStillAdapts() {
    const double inf = std::numeric_limits<double>::infinity();
    const double Q[5][5] = {{20394, -24908, -2026, 3896, 658},
                            {-24908, 41818, -3466, -9828, -372},
                            {-2026, -3466, 3510, 2178, -348},
                            {3896, -9828, 2178, 3030, -44},
                            {658, -372, -348, -44, 54}};
    std::vector<double> r, c, v;
    for (int i = 0; i < 5; ++i)
        for (int j = 0; j < 5; ++j) { r.push_back(i); c.push_back(j); v.push_back(Q[i][j]); }
    const double A[5][5] = {{-1, -1, -1, -1, -1}, {10, 10, -3, 5, 4}, {-8, 1, -2, -5, 3},
                            {8, -1, 2, 5, -3}, {-4, -2, 3, -5, 1}};
    std::vector<double> ar, ac, av;
    for (int i = 0; i < 5; ++i)
        for (int j = 0; j < 5; ++j) { ar.push_back(i); ac.push_back(j); av.push_back(A[i][j]); }
    qp::QpModel s;
    s.P = mkMat(5, 5, r, c, v);
    s.q = {18340, -34198, 4542, 8672, 86};
    s.A = mkMat(5, 5, ar, ac, av);
    s.l = {-5, 20, -40, 11, -30};
    s.u = {inf, inf, inf, inf, inf};
    const qp::AdmmResult res = qp::QpSolver{}.solve(s, productionOpts());
    require(res.status == qp::QpStatus::Optimal, "hs268 not optimal: " + std::string(qp::toString(res.status)) +
                                                 " after " + std::to_string(res.iterations) + " iterations");
    const double expected[5] = {1, 2, -1, 3, -4};
    for (int j = 0; j < 5; ++j) requireNear(res.primal[static_cast<std::size_t>(j)], expected[j], 1e-3, "hs268 x");
    requireNear(res.primalObjective, -14463.0, 1e-3, "hs268 objective");
}

// Polishing is optional refinement and must respect its budget. Each pass
// factors a dense (n + active rows) system; on presolved qship08s that ran for
// more than 300 s past a 55 s time limit it never checked. When either bound
// stops it, polish() must return false and leave the point exactly as it was,
// so the caller keeps the converged, already verified ADMM iterate.
void testPolishingRespectsBudget() {
    qp::QpModel s;
    s.P = mkMat(2, 2, {{0, 0, 2.0}, {1, 1, 2.0}});
    s.q = {-2.0, -5.0};
    s.A = mkMat(3, 2, {{0, 0, 1.0}, {0, 1, 1.0}, {1, 0, 1.0}, {2, 1, 1.0}});
    s.l = {-1e20, 0.0, 0.0};
    s.u = {1.0, 1e20, 1e20};
    const std::vector<double> x0 = {0.1, 0.9}, y0 = {-1.0, 0.0, 0.0};

    std::vector<double> x = x0, y = y0;
    qp::KktPolisher::Options tooBig;
    tooBig.maximumDenseDimension = 1;  // any active system exceeds this
    require(!qp::KktPolisher::polish(s, x, y, tooBig), "dimension cap did not stop polishing");
    require(x == x0 && y == y0, "a skipped polish modified the point");

    x = x0; y = y0;
    qp::KktPolisher::Options noTime;
    noTime.timeLimitSeconds = 1e-12;   // exhausted before the first solve
    require(!qp::KktPolisher::polish(s, x, y, noTime), "time limit did not stop polishing");
    require(x == x0 && y == y0, "a timed-out polish modified the point");

    x = x0; y = y0;
    require(qp::KktPolisher::polish(s, x, y, qp::KktPolisher::Options{}), "unbounded polish failed");
}

void run(const char* name, void (*f)()) {
    try {
        f();
        std::printf("  pass  %s\n", name);
    } catch (const std::exception& e) {
        std::printf("  FAIL  %s: %s\n", name, e.what());
        ++failures;
    }
}

}  // namespace

int main() {
    // Sparse matrix
    run("duplicateTriplets",     testSparseDuplicateTriplets);
    run("cancellingEntries",    testSparseCancelling);
    run("unsortedAndEmpty",      testSparseUnsortedAndEmpty);
    run("csrCscAgree",          testSparseCsrCscAgree);
    run("sparseScaling",        testSparseScaling);

    // KKT
    run("kktDenseUnconstrained", testKktDenseUnconstrained);
    run("kktWithRho",           testKktWithRho);
    run("kktRefactor",           testKktRefactor);

    // Scaling
    run("ruizConditioning",      testRuizConditioning);
    run("ruizNoConstraints",     testRuizNoConstraints);

    // ADMM
    run("unconstrainedQp",       testUnconstrainedQp);
    run("unconstrainedQuadratic", testUnconstrainedQuadratic);
    run("boxedEquality",         testBoxedEquality);
    run("rangedRow",             testRangedRow);
    run("iterationLimit",         testIterationLimit);
    run("invalidCrossedBounds",   testInvalidCrossedBounds);
    run("illConditioned",        testIllConditioned);
    run("defaultsNoFalseOptimal", testDefaultsNoFalseOptimal);
    run("defaultsNoLimitCycle",   testDefaultsNoLimitCycle);
    run("defaultsNoRhoOscillation", testDefaultsNoRhoOscillation);
    run("defaultsZeroResidualStillAdapts", testDefaultsZeroResidualStillAdapts);

    // Polishing
    run("polishing",             testPolishing);
    run("polishingWithOneSidedActiveBound", testPolishingWithOneSidedActiveBound);
    run("polishingRespectsBudget", testPolishingRespectsBudget);

    // Integration
    run("qpSolverFacade",        testQpSolverFacade);

    if (failures == 0) {
        std::printf("\nAll %d QP engine tests passed\n", 26);
        return 0;
    }
    std::printf("\n%d test(s) failed\n", failures);
    return 1;
}
