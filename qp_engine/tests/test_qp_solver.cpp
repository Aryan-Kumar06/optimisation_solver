// Comprehensive tests for the QP engine.
// Run: ./qp_tests
#include "admm_backend_contract.h"

#include "qp/admm_backend.h"
#include "qp/admm_solver.h"
#include "qp/kkt_solver.h"
#include "qp/polishing.h"
#include "qp/qp_model.h"
#include "qp/qp_solver.h"
#include "qp/qp_types.h"
#include "qp/scaling.h"

#include <cmath>
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
// Compute backends
// ---------------------------------------------------------------------------

// The contract the hybrid CUDA backend is held to (admm_backend_contract.h),
// run CPU against CPU so the harness is exercised on every build.
void testCpuBackendContract() {
    const qpcontract::BackendFactory cpu = [](const qp::QpModel& model) {
        return qp::makeCpuAdmmBackend(model, nullptr, nullptr, nullptr);
    };
    for (const qpcontract::Fixture& fixture : qpcontract::fixtures()) {
        qpcontract::compareBackends(fixture, cpu, cpu, 0.0);
    }
}

qp::QpModel backendTestModel() {
    qp::QpModel m;
    m.P = mkMat(2, 2, {{0,0,2.0}, {0,1,0.5}, {1,0,0.5}, {1,1,1.0}});
    m.A = mkMat(2, 2, {{0,0,1.0}, {0,1,1.0}, {1,0,1.0}});
    m.q = {-1.0, -1.0};
    m.l = {-1e300, 0.0};
    m.u = {1.0, 0.75};
    return m;
}

// An explicit CUDA request is honoured or refused, never quietly run on the CPU.
void testExplicitCudaRequestNeverFallsBack() {
    const qp::CudaAvailability availability = qp::cudaAvailability(0);
    if (availability.usable) return;  // covered by qp_cuda_tests
    require(!availability.reason.empty(), "an unusable CUDA backend must say why");
    qp::AdmmOptions o;
    o.backend = qp::ComputeBackend::Cuda;
    const auto r = qp::QpSolver{}.solve(backendTestModel(), o);
    require(r.status == qp::QpStatus::InvalidProblem, "CUDA requested without a device must not solve");
    require(r.statusMessage.find("CUDA backend requested but unavailable") != std::string::npos,
            "the failure must name the CUDA request: " + r.statusMessage);
}

void testAutoBackendResolution() {
    qp::AdmmOptions o;
    auto r = qp::QpSolver{}.solve(backendTestModel(), o);
    require(r.executedBackend == qp::ComputeBackend::Cpu, "Auto defaults to the CPU for the hybrid QP backend");
    require(r.backendMessage.find("below cudaNonzeroThreshold") != std::string::npos,
            "Auto must say why it chose the CPU: " + r.backendMessage);
    o.cudaNonzeroThreshold = 0;
    r = qp::QpSolver{}.solve(backendTestModel(), o);
    require(r.status == qp::QpStatus::Optimal, "Auto must still solve");
    if (!qp::cudaAvailability(0).usable) {
        require(r.executedBackend == qp::ComputeBackend::Cpu, "no device: Auto must run on the CPU");
        require(r.backendMessage.rfind("auto -> cpu", 0) == 0, "fallback reason: " + r.backendMessage);
    }
    require(r.kktSolveSeconds >= 0.0 && r.kktFactorSeconds >= 0.0, "KKT timings are reported");
}

// Forcing the CPU is the default solve, bit for bit, when Auto resolves to the CPU.
void testForcedCpuMatchesDefault() {
    qp::AdmmOptions automatic;
    qp::AdmmOptions forced;
    forced.backend = qp::ComputeBackend::Cpu;
    const auto a = qp::QpSolver{}.solve(backendTestModel(), automatic);
    const auto b = qp::QpSolver{}.solve(backendTestModel(), forced);
    require(a.status == b.status && a.iterations == b.iterations && a.factorizations == b.factorizations,
            "forced CPU must follow the identical iteration path");
    require(a.primal == b.primal && a.constraintDual == b.constraintDual,
            "forced CPU must return identical vectors");
    require(a.primalObjective == b.primalObjective, "forced CPU must return the identical objective");
    require(b.backendMessage == "cpu: requested", "forced CPU message: " + b.backendMessage);
}

void testInvalidCudaDeviceIsRejected() {
    qp::AdmmOptions o;
    o.cudaDevice = -3;
    const auto r = qp::QpSolver{}.solve(backendTestModel(), o);
    require(r.status == qp::QpStatus::InvalidProblem, "a negative CUDA device must be rejected");
}

// ---------------------------------------------------------------------------
// Run helpers
// ---------------------------------------------------------------------------

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

    // Polishing
    run("polishing",             testPolishing);
    run("polishingWithOneSidedActiveBound", testPolishingWithOneSidedActiveBound);

    // Integration
    run("qpSolverFacade",        testQpSolverFacade);

    // Compute backends
    run("cpuBackendContract",    testCpuBackendContract);
    run("explicitCudaRequestNeverFallsBack", testExplicitCudaRequestNeverFallsBack);
    run("autoBackendResolution", testAutoBackendResolution);
    run("forcedCpuMatchesDefault", testForcedCpuMatchesDefault);
    run("invalidCudaDeviceIsRejected", testInvalidCudaDeviceIsRejected);

    if (failures == 0) {
        std::printf("\nAll %d QP engine tests passed\n", 26);
        return 0;
    }
    std::printf("\n%d test(s) failed\n", failures);
    return 1;
}
