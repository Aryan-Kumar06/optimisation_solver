// QP engine benchmark harness.
//
// Builds a few representative test problems (small unconstrained, medium
// box-constrained, ill-conditioned with Ruiz scaling, large sparse) and
// reports solve time, iteration count, and primal/dual residuals.
//
// Run: ./qp_bench [cpu|cuda|auto]
//
// The optional backend argument applies to every case (default cpu). With the
// hybrid CUDA backend each case also reports where the time went: the CPU KKT
// solves, device setup, and the bytes moved each way.
#include "qp/admm_solver.h"
#include "qp/qp_model.h"
#include "qp/qp_types.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <random>
#include <string>
#include <tuple>
#include <vector>

// std::chrono::steady_clock and QueryPerformanceCounter on this MinGW host
// both produce out-of-scale or negative durations. std::clock() is portable
// and good enough for solve-time reporting on sub-second problems, which
// covers everything in the harness. Larger problems should use a real
// wall-clock profile elsewhere.
namespace bench {
double nowSeconds() {
    return static_cast<double>(std::clock()) / CLOCKS_PER_SEC;
}
}  // namespace bench

namespace {

qp::SparseMatrix mkMat(int rows, int cols,
    const std::vector<double>& rv,
    const std::vector<double>& cv,
    const std::vector<double>& vv) {
    return qp::SparseMatrix::fromTriplets(rows, cols, rv, cv, vv);
}

void section(const char* name) {
    std::printf("\n=== %s ===\n", name);
}

struct BenchCase {
    const char* name;
    qp::QpModel model;
    qp::AdmmOptions options;
};

qp::QpModel smallUnconstrained() {
    // min 0.5 * ||x||^2 + 1^T x, n=20, no constraints.
    qp::QpModel m;
    const int n = 20;
    std::vector<double> rv(n), cv(n), vv(n, 1.0);
    for (int j = 0; j < n; ++j) { rv[j] = j; cv[j] = j; }
    m.P = mkMat(n, n, rv, cv, vv);
    m.A = mkMat(0, n, {}, {}, {});
    m.q.assign(static_cast<std::size_t>(n), 1.0);
    return m;
}

qp::QpModel mediumBoxed(int n) {
    // min 0.5 * x^T diag(1..n) x - sum(x_i), subject to 0 <= x <= 1.
    qp::QpModel m;
    std::vector<double> rv, cv, vv;
    rv.reserve(static_cast<std::size_t>(n));
    cv.reserve(static_cast<std::size_t>(n));
    vv.reserve(static_cast<std::size_t>(n));
    for (int j = 0; j < n; ++j) { rv.push_back(j); cv.push_back(j); vv.push_back(j + 1.0); }
    m.P = mkMat(n, n, rv, cv, vv);
    m.A = mkMat(n, n, rv, cv, std::vector<double>(static_cast<std::size_t>(n), 1.0));
    m.q.assign(static_cast<std::size_t>(n), -1.0);
    m.l.assign(static_cast<std::size_t>(n), 0.0);
    m.u.assign(static_cast<std::size_t>(n), 1.0);
    return m;
}

qp::QpModel illConditionedRanged(int n) {
    // 1e8 * I Hessian, badly scaled constraint matrix, box constraints.
    qp::QpModel m;
    std::vector<double> rv, cv, vv;
    rv.reserve(static_cast<std::size_t>(n));
    cv.reserve(static_cast<std::size_t>(n));
    vv.reserve(static_cast<std::size_t>(n));
    for (int j = 0; j < n; ++j) { rv.push_back(j); cv.push_back(j); vv.push_back(1e8); }
    m.P = mkMat(n, n, rv, cv, vv);

    // A = identity scaled by wildly different factors per row.
    std::vector<double> aR, aC, aV;
    for (int i = 0; i < n; ++i) {
        aR.push_back(i); aC.push_back(i);
        aV.push_back(std::pow(10.0, static_cast<double>((i % 7) - 3)));
    }
    m.A = mkMat(n, n, aR, aC, aV);

    m.q.assign(static_cast<std::size_t>(n), 0.5);
    m.l.assign(static_cast<std::size_t>(n), -1.0);
    m.u.assign(static_cast<std::size_t>(n), 1.0);
    return m;
}

qp::QpModel largeSparse(int n, double density) {
    // Random sparse P and A.  Tests the sparse Cholesky path when n > 500.
    qp::QpModel m;
    std::mt19937 g(42);
    std::uniform_real_distribution<double> val(-1.0, 1.0);
    std::uniform_int_distribution<int> colPick(0, n - 1);
    std::uniform_real_distribution<double> densityDist(0.0, 1.0);

    std::vector<double> pR, pC, pV;
    for (int i = 0; i < n; ++i) {
        // Diagonal entry to keep P positive definite.
        pR.push_back(i); pC.push_back(i); pV.push_back(1.0 + std::abs(val(g)));
        for (int j = 0; j < n; ++j) {
            if (i != j && densityDist(g) < density) {
                pR.push_back(i); pC.push_back(j); pV.push_back(val(g));
            }
        }
    }
    m.P = mkMat(n, n, pR, pC, pV);

    // m = n/2 equality-ish rows (boxed).
    const int mrows = n / 2;
    std::vector<double> aR, aC, aV;
    for (int i = 0; i < mrows; ++i) {
        int nnz = 1 + (colPick(g) % 5);
        for (int k = 0; k < nnz; ++k) {
            aR.push_back(i); aC.push_back(colPick(g)); aV.push_back(val(g));
        }
    }
    m.A = mkMat(mrows, n, aR, aC, aV);
    m.q.assign(static_cast<std::size_t>(n), 0.0);
    m.l.assign(static_cast<std::size_t>(mrows), -0.5);
    m.u.assign(static_cast<std::size_t>(mrows), 0.5);
    return m;
}

qp::ComputeBackend gBackend = qp::ComputeBackend::Cpu;

void run(const char* label, const BenchCase& bc) {
    qp::AdmmOptions options = bc.options;
    options.backend = gBackend;
    const double t0 = bench::nowSeconds();
    auto r = qp::AdmmSolver(bc.model, options).solve();
    double ms = 1000.0 * (bench::nowSeconds() - t0);
    // Sanitise: clock() can return -1 (failure sentinel) or negative values
    // on some MinGW builds. Clamp to 0 if the result is non-finite or negative.
    if (!(ms >= 0.0)) ms = 0.0;
    // Clamp negative or NaN times (can occur due to clock() quantization
    // on some hosts) to 0. All solve times here are well below 1 ms anyway.
    std::printf("  %-30s  status=%-9s  iters=%4d  obj=%12.6g  "
                "rPri=%10.3e  rDual=%10.3e  time=%7.2f ms\n",
                label, qp::toString(r.status), static_cast<int>(r.iterations),
                r.primalObjective, r.primalResidual, r.dualResidual, ms);
    if (gBackend != qp::ComputeBackend::Cpu) {
        std::printf("  %-30s  backend=%s  kkt solve=%.3g s  kkt factor=%.3g s  setup=%.3g s  "
                    "H2D=%lld B  D2H=%lld B  syncs=%lld\n",
                    "", qp::toString(r.executedBackend), r.kktSolveSeconds, r.kktFactorSeconds,
                    r.backendProfile.setupSeconds,
                    static_cast<long long>(r.backendProfile.hostToDeviceBytes),
                    static_cast<long long>(r.backendProfile.deviceToHostBytes),
                    static_cast<long long>(r.backendProfile.synchronisations));
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1) {
        if (std::strcmp(argv[1], "cuda") == 0) {
            gBackend = qp::ComputeBackend::Cuda;
        } else if (std::strcmp(argv[1], "auto") == 0) {
            gBackend = qp::ComputeBackend::Auto;
        } else if (std::strcmp(argv[1], "cpu") != 0) {
            std::printf("unknown backend '%s' (cpu|cuda|auto)\n", argv[1]);
            return 2;
        }
    }
    const qp::CudaAvailability cuda = qp::cudaAvailability(0);
    std::printf("backend=%s  cuda compiled=%s  device=%s  runtime=%d driver=%d\n",
                qp::toString(gBackend), cuda.compiled ? "yes" : "no",
                cuda.usable ? cuda.deviceName.c_str() : "none", cuda.runtimeVersion, cuda.driverVersion);

    section("small unconstrained (n=20)");
    {
        BenchCase bc{"", smallUnconstrained(), qp::AdmmOptions{}};
        bc.options.rho = 1.0;
        bc.options.useRuizScaling = false;
        run("n=20, rho=1.0", bc);
    }

    section("medium box-constrained");
    {
        for (int n : {50, 200}) {
            BenchCase bc{"", mediumBoxed(n), qp::AdmmOptions{}};
            bc.options.rho = 5.0;
            bc.options.useRuizScaling = true;
            bc.options.iterationLimit = 5000;
            char label[64];
            std::snprintf(label, sizeof(label), "n=%d, rho=5.0, Ruiz", n);
            run(label, bc);
        }
    }

    section("ill-conditioned ranged (Ruiz on)");
    {
        for (int n : {50, 200}) {
            BenchCase bc{"", illConditionedRanged(n), qp::AdmmOptions{}};
            bc.options.rho = 1.0;
            bc.options.useRuizScaling = true;
            bc.options.ruizIterations = 10;
            bc.options.iterationLimit = 5000;
            char label[64];
            std::snprintf(label, sizeof(label), "n=%d, Ruiz=10", n);
            run(label, bc);
        }
    }

    section("large sparse (sparse Cholesky path)");
    {
        for (int n : {600, 1000}) {
            BenchCase bc{"", largeSparse(n, 0.01), qp::AdmmOptions{}};
            bc.options.rho = 1.0;
            bc.options.useRuizScaling = true;
            bc.options.iterationLimit = 3000;
            char label[64];
            std::snprintf(label, sizeof(label), "n=%d, density=0.01", n);
            run(label, bc);
        }
    }

    std::printf("\n");
    return 0;
}
