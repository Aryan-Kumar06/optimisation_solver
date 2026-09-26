#pragma once

#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ < 7
#pragma GCC diagnostic ignored "-Wattributes"
#endif

#include "qp/admm_backend.h"
#include "qp/compute_backend.h"
#include "qp/parallel.h"
#include "qp/qp_model.h"
#include "qp/scaling.h"
#include "qp/qp_types.h"

#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace qp {

class KktSolver;

// ============================================================================
// Options for the ADMM solver
// ============================================================================

struct AdmmOptions {
    // Iterations.
    std::int64_t iterationLimit = 5000;
    double timeLimitSeconds = 0.0;

    // Convergence tolerances (applied to the scaled problem, so the
    // scaled and original norms are equivalent).
    double primalTolerance = 1e-6;
    double dualTolerance = 1e-6;

    // Frequency (in iterations) at which we check for termination.
    int terminationCheckFrequency = 50;

    // Penalty parameter ρ. 0 means the solver picks a default (1.0).
    double rho = 0.0;

    // Adaptive rho: the scheme from Boyd et al. §3.4.1.
    // If μ > 1, the solver adjusts rho to balance primal and dual residual.
    bool useAdaptiveRho = true;
    double adaptiveRhoMu = 10.0;       // residual ratio threshold
    double adaptiveRhoTau = 2.0;        // rho multiplier when out of balance

    // Hysteresis on adopting a new rho, to trade convergence for fewer
    // factorisations. DEFAULTS DISABLE IT, and that is deliberate.
    //
    // Every rho change costs a numeric refactorisation, so gating the changes
    // looks like an obvious saving. Measured, it is the opposite: rho adaptation
    // is load-bearing for ADMM convergence. With a threshold of 5 and a
    // 25-iteration interval, two instances that converge in 100-150 iterations
    // instead ran 200000 iterations and finished with a primal residual of 1.2.
    // Meanwhile the saving never materialised -- the imbalance test fires rarely
    // enough that these benchmarks refactorised once either way.
    //
    // Raise them only with a benchmark that shows both fewer factorisations and
    // no loss of convergence.
    double adaptiveRhoThreshold = 1.0;   // 1 = adopt every change
    std::int64_t adaptiveRhoInterval = 0;

    // Ruiz equilibration of P and A before iterating. The solver operates on
    // the scaled problem and maps the result back, so tolerances stay in the
    // caller's units.
    bool useRuizScaling = true;
    int ruizIterations = 10;

    // KKT polishing after the main loop: a small number of active-set
    // refinement steps on the KKT system.
    // Worker threads for the sparse products in the hot loop. 0 selects
    // hardware_concurrency; 1 forces serial. Small problems stay serial
    // regardless: below parallelNonzeroThreshold the barrier costs more than
    // the work it distributes.
    int threadCount = 0;
    std::int64_t parallelNonzeroThreshold = 50000;

    bool usePolishing = true;
    int polishingIterations = 200;

    // Where the per-iteration vector and sparse-product work runs; see
    // ComputeBackend and docs/cuda.md. The algorithm and its options are the
    // same on every backend.
    ComputeBackend backend = ComputeBackend::Auto;
    int cudaDevice = 0;

    // Auto selects the hybrid CUDA backend only when nnz(A) + nnz(P) reaches
    // this. The default means Auto never selects it: the KKT solve stays on the
    // CPU and two n-vectors cross the bus every iteration, so the hybrid has to
    // EARN its place with a measured crossover on real hardware before Auto
    // may pick it (docs/cuda.md). backend = Cuda always runs it.
    std::int64_t cudaNonzeroThreshold = std::numeric_limits<std::int64_t>::max();
};

// ============================================================================
// ADMM result
// ============================================================================

struct AdmmResult {
    QpStatus status = QpStatus::InvalidProblem;
    std::string statusMessage;

    std::vector<double> primal;    // solution x

    // Dual variables: one multiplier per constraint row.
    std::vector<double> constraintDual;

    double primalObjective  = std::numeric_limits<double>::quiet_NaN();
    double dualObjective    = -std::numeric_limits<double>::infinity();

    double primalResidual  = std::numeric_limits<double>::infinity();
    double dualResidual   = std::numeric_limits<double>::infinity();

    std::int64_t iterations = 0;
    double solveTimeSeconds = 0.0;

    // Diagnostics.
    double finalRho = 0.0;

    // Numeric refactorisations performed. The dominant cost in ADMM on a sparse
    // problem, so this is the number to watch when tuning the rho policy.
    std::int64_t factorizations = 0;
    double bestObjective = std::numeric_limits<double>::quiet_NaN();

    // Which backend ran the iteration and why; see PdlpResult for the same idea.
    ComputeBackend executedBackend = ComputeBackend::Cpu;
    std::string backendMessage;
    BackendProfile backendProfile;

    // Host time in the KKT factorisation (all refactorisations) and in the
    // per-iteration triangular solves. Measured on every backend: on the
    // hybrid CUDA backend this is the CPU share of each iteration.
    double kktFactorSeconds = 0.0;
    double kktSolveSeconds = 0.0;
};

// ============================================================================
// ADMM solver
//
// Solves
//   min 0.5*x'*P*x + q'*x
//   s.t. l <= A*x <= u
//
// using the Alternating Direction Method of Multipliers on the consensus form.
//
// The augmented Lagrangian for the QP is:
//   L_ρ(x, z, y) = 0.5*x'*P*x + q'*x + y'*(A*x - z) + ρ/2*||A*x - z||^2
//
// ADMM steps:
//   x^{k+1} = argmin_x L_ρ(x, z^k, y^k)          [sparse KKT solve]
//   z^{k+1} = proj_{[l,u]}(A*x^{k+1} + y^k/ρ)   [coordinate projection]
//   y^{k+1} = y^k + ρ*(A*x^{k+1} - z^{k+1})      [dual ascent]
//
// Termination uses the residual-based stopping criterion from Boyd et al.
// ============================================================================

class AdmmSolver {
public:
    explicit AdmmSolver(const QpModel& problem, const AdmmOptions& options = {});

    AdmmResult solve();

    [[nodiscard]] const QpModel& problem() const { return scaled_; }

private:
    // The iteration loop proper, run by solve() once validation has passed.
    AdmmResult iterate();

    // One ADMM iteration.  rhoChanged is set to true if the penalty changed
    // and the caller must refactor the KKT.
    void step(KktSolver& kkt, bool& rhoChanged);

    // Map the best iterate (x_, y_) back to the original coordinates.
    void toOriginal();

    const QpModel& original_;
    QpModel scaled_;
    AdmmOptions options_;

    // State: x, z, y, A*x and their saved copies live in the backend, on
    // whichever device runs the iteration.
    std::unique_ptr<AdmmBackend> backend_;
    std::string backendMessage_;

    double rho_ = 1.0;

    // For adaptive rho diagnostics.
    // The equilibration is kept from construction rather than recomputed when
    // unscaling. It was previously run a second time in toOriginal(), which
    // repeated the whole Ruiz sweep and rebuilt a scaled copy of P and A only to
    // read two diagonal vectors out of it.
    double desiredRho_ = 0.0;
    std::int64_t lastRhoUpdate_ = 0;

    QpScaling scaling_;
    bool scalingValid_ = false;

    std::unique_ptr<Executor> executor_;
    SparseMatrix::Plan planA_;
    SparseMatrix::Plan planP_;
    bool parallel_ = false;

    AdmmResult result_;
};

}  // namespace qp
