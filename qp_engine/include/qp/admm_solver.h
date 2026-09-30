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

    // Convergence tolerances checked on the original problem.
    double primalTolerance = 1e-6;
    double dualTolerance = 1e-6;

    // Frequency (in iterations) at which we check for termination.
    int terminationCheckFrequency = 50;

    // Penalty parameter ρ. 0 means the solver picks a default (1.0).
    double rho = 0.0;

    // Adaptive rho, as in OSQP (Stellato et al. 2020, section 5.2).
    //
    // At each termination check -- never between checks -- the penalty is
    // re-estimated from the NORMALISED residual ratio
    //
    //   rho_new = rho * sqrt( (||Ax - z|| / max(||Ax||, ||z||))
    //                       / (||Px + q + A'y|| / max(||Px||, ||A'y||, ||q||)) )
    //
    // limited to a factor of maximumRhoStep per update, clamped to
    // [rhoMinimum, rhoMaximum], and adopted only when it differs from the
    // current rho by more than adaptiveRhoTolerance. At most
    // maximumRhoUpdates changes are made, so rho is eventually constant: that is
    // the condition under which Boyd et al. (section 3.4.1) prove convergence
    // with a varying penalty.
    //
    // This replaces a per-iteration doubling/halving rule on the raw residual
    // ratio, which had neither property and failed in three distinct ways:
    //   * a limit cycle -- rho and the iterate oscillating forever, the iterate
    //     bit-identical after 5,000 and 200,000 iterations (randomized QP case
    //     219, Maros-Meszaros hs118);
    //   * unbounded growth -- rho = 2.8e14 after 50 iterations on hs51, hs52 and
    //     genhs28, which froze z and so zeroed the proxy dual residual the old
    //     termination test used, producing a false Optimal;
    //   * a failed refactorisation at a large rho that was never recovered from,
    //     leaving x frozen at zero (cvxqp3s).
    //
    // The damping is not in OSQP's rule and is needed here: without it the
    // ratio alone oscillated between extremes on a nearly-LP elastic QP from
    // the NLP engine (see maximumRhoStep's use in the solver).
    bool useAdaptiveRho = true;
    double adaptiveRhoTolerance = 5.0;
    double maximumRhoStep = 10.0;
    int maximumRhoUpdates = 50;
    double rhoMinimum = 1e-6;
    double rhoMaximum = 1e6;

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
// Optimality check in the ORIGINAL problem's units
// ============================================================================
//
// The single definition of "converged" for this engine. The ADMM loop
// terminates on it, and QpSolver re-applies it after polishing before it will
// return Optimal. It is evaluated on the caller's (unscaled) problem because
// that is what every consumer -- postsolve, MIQP node bounds, NLP subproblems --
// actually checks. The loop previously terminated on Ruiz-SCALED residuals with
// sqrt(n + m) factors instead: measured on randomly generated QPs, 31 of 400
// Optimal results then failed this check at the requested tolerance, rising to
// 121 of 375 when coefficients spanned 1e-4 to 1e4, and presolved Maros-Meszaros
// qship08s came back Optimal with a variable bound violated in original units.
//
//   primal: every row i satisfies  dist(A_i x, [l_i, u_i])
//                                    <= primalTolerance * max(1, |A_i x|, |l_i|, |u_i|)
//           (infinite bounds excluded from the scale) -- per row, like postsolve;
//   dual:   ||P x + q + A'y||_inf  <= dualTolerance * max(1, ||Px||, ||A'y||, ||q||)
//   and every number involved finite.
struct KktCheck {
    double primalViolation = std::numeric_limits<double>::infinity();  // worst absolute row violation
    double dualResidual = std::numeric_limits<double>::infinity();     // ||P x + q + A'y||_inf
    bool primalMet = false;
    bool dualMet = false;
    bool finite = false;
    [[nodiscard]] bool met() const noexcept { return finite && primalMet && dualMet; }
};
[[nodiscard]] KktCheck checkKkt(const QpModel& model,
                                const std::vector<double>& x,
                                const std::vector<double>& y,
                                double primalTolerance,
                                double dualTolerance);

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

    // Return false if the KKT factorisation cannot solve the step.
    [[nodiscard]] bool step(KktSolver& kkt);

    // Map the backend iterate back to the original coordinates.
    void toOriginal();

    const QpModel& original_;
    QpModel scaled_;
    AdmmOptions options_;

    // State: x, z, y, A*x and their saved copies live in the backend, on
    // whichever device runs the iteration.
    std::unique_ptr<AdmmBackend> backend_;
    std::string backendMessage_;

    double rho_ = 1.0;

    // Adaptive rho bookkeeping: how many changes have been made, and the
    // interval rho must stay inside after a refactorisation failed at a value.
    int rhoUpdates_ = 0;
    double rhoFloor_ = 0.0;
    double rhoCeiling_ = 0.0;

    // The equilibration is kept from construction rather than recomputed when
    // unscaling. It was previously run a second time in toOriginal(), which
    // repeated the whole Ruiz sweep and rebuilt a scaled copy of P and A only to
    // read two diagonal vectors out of it.

    QpScaling scaling_;
    bool scalingValid_ = false;

    std::unique_ptr<Executor> executor_;
    SparseMatrix::Plan planA_;
    SparseMatrix::Plan planP_;
    bool parallel_ = false;

    AdmmResult result_;
};

}  // namespace qp
