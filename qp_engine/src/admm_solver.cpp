#include "qp/admm_solver.h"

#include "qp/kkt_solver.h"
#include "qp/scaling.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>

namespace qp {

namespace {

using Clock = std::chrono::steady_clock;

double secondsSince(const Clock::time_point& start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

}  // namespace

AdmmSolver::AdmmSolver(const QpModel& problem, const AdmmOptions& options)
    : original_(problem), options_(options) {
    if (options_.rho <= 0.0) options_.rho = 1.0;
    if (options_.primalTolerance <= 0.0) options_.primalTolerance = 1e-6;
    if (options_.dualTolerance <= 0.0) options_.dualTolerance = 1e-6;
    if (options_.terminationCheckFrequency <= 0)
        options_.terminationCheckFrequency = 50;
    if (options_.iterationLimit <= 0)
        options_.iterationLimit = 5000;
    if (options_.ruizIterations < 0)
        options_.ruizIterations = 0;
    if (options_.polishingIterations < 0)
        options_.polishingIterations = 0;
    if (options_.adaptiveRhoMu <= 1.0)
        options_.adaptiveRhoMu = 10.0;
    if (options_.adaptiveRhoTau <= 1.0)
        options_.adaptiveRhoTau = 2.0;
    if (options_.cudaDevice < 0) {
        result_.status = QpStatus::InvalidProblem;
        result_.statusMessage = "AdmmOptions: cudaDevice must be non-negative";
        return;
    }

    // Validate the problem first, before any equilibration.
    try {
        problem.validate();
    } catch (const std::exception& e) {
        result_.status = QpStatus::InvalidProblem;
        result_.statusMessage = e.what();
        return;
    }

    if (options_.useRuizScaling) {
        scaling_ = RuizScaler::equilibrate(problem, options_.ruizIterations);
        scaled_ = scaling_.scaled;
        scalingValid_ = true;
    } else {
        scaled_ = problem;
    }

    try {
        scaled_.validate();
    } catch (const std::exception& e) {
        result_.status = QpStatus::InvalidProblem;
        result_.statusMessage = e.what();
        return;
    }

    const int n = scaled_.numVariables();
    const int m = scaled_.numConstraints();

    // Threading pays only when the products are large enough to amortise the
    // barrier. The KKT triangular solve stays sequential either way: its
    // dependency chain is inherently serial without level scheduling.
    const auto totalNonzeros =
        static_cast<std::int64_t>(scaled_.A.nonzeros() + scaled_.P.nonzeros());
    if (options_.threadCount != 1 && totalNonzeros >= options_.parallelNonzeroThreshold) {
        auto candidate = std::make_unique<Executor>(options_.threadCount);
        if (candidate->threadCount() > 1) {
            planA_ = scaled_.A.buildPlan(candidate->threadCount());
            planP_ = scaled_.P.buildPlan(candidate->threadCount());
            executor_ = std::move(candidate);
            parallel_ = true;
        }
    }

    // Backend selection: deterministic in the options, the build, the device
    // and the size of the scaled problem. An explicit CUDA request that cannot
    // be honoured is an error, never a silent CPU solve.
    bool wantCuda = false;
    switch (options_.backend) {
        case ComputeBackend::Cpu:
            backendMessage_ = "cpu: requested";
            break;
        case ComputeBackend::Cuda:
            wantCuda = true;
            break;
        case ComputeBackend::Auto:
            if (totalNonzeros < options_.cudaNonzeroThreshold) {
                backendMessage_ = "auto -> cpu: " + std::to_string(totalNonzeros) +
                    " nonzeros is below cudaNonzeroThreshold";
            } else {
                const CudaAvailability availability = cudaAvailability(options_.cudaDevice);
                if (availability.usable) {
                    wantCuda = true;
                } else {
                    backendMessage_ = "auto -> cpu: " + availability.reason;
                }
            }
            break;
    }
    if (wantCuda) {
        std::string error;
        if (options_.backend == ComputeBackend::Cuda) {
            const CudaAvailability availability = cudaAvailability(options_.cudaDevice);
            if (!availability.usable) {
                result_.status = QpStatus::InvalidProblem;
                result_.statusMessage =
                    "CUDA backend requested but unavailable: " + availability.reason;
                return;
            }
        }
        backend_ = makeCudaAdmmBackend(scaled_, options_.cudaDevice, error);
        if (!backend_) {
            if (options_.backend == ComputeBackend::Cuda) {
                result_.status = QpStatus::InvalidProblem;
                result_.statusMessage =
                    "CUDA backend requested but could not be started: " + error;
                return;
            }
            backendMessage_ = "auto -> cpu: CUDA backend could not be started: " + error;
        } else {
            backendMessage_ = std::string(options_.backend == ComputeBackend::Cuda
                                              ? "cuda (hybrid, KKT on CPU): requested"
                                              : "auto -> cuda (hybrid, KKT on CPU)") +
                ", device " + std::to_string(options_.cudaDevice);
        }
    }
    if (!backend_) {
        backend_ = makeCpuAdmmBackend(
            scaled_,
            parallel_ ? executor_.get() : nullptr,
            parallel_ ? &planA_ : nullptr,
            parallel_ ? &planP_ : nullptr);
    }

    rho_ = options_.rho;
    desiredRho_ = options_.rho;

    result_.primal.assign(static_cast<std::size_t>(n), 0.0);
    result_.constraintDual.assign(static_cast<std::size_t>(m), 0.0);
    result_.status = QpStatus::IterationLimit;
}

AdmmResult AdmmSolver::solve() {
    if (result_.status == QpStatus::InvalidProblem)
        return result_;

    // A device failure part-way through is reported as a failed solve. CPU
    // exceptions propagate exactly as they always did.
    try {
        return iterate();
    } catch (const std::exception& error) {
        if (!backend_ || backend_->kind() != ComputeBackend::Cuda) {
            throw;
        }
        result_.status = QpStatus::NumericalFailure;
        result_.statusMessage = std::string("CUDA backend error: ") + error.what();
        result_.executedBackend = ComputeBackend::Cuda;
        result_.backendMessage = backendMessage_;
        result_.backendProfile = backend_->profile();
        return result_;
    }
}

AdmmResult AdmmSolver::iterate() {
    AdmmBackend& state = *backend_;
    result_.executedBackend = state.kind();
    result_.backendMessage = backendMessage_;

    const auto tStart = std::chrono::steady_clock::now();
    const int n = scaled_.numVariables();
    const int m = scaled_.numConstraints();

    if (n == 0) {
        result_.status = QpStatus::Optimal;
        result_.statusMessage = "zero variables";
        result_.primalObjective = 0.0;
        result_.iterations = 0;
        result_.primal.clear();
        return result_;
    }

    const Clock::time_point factorStart = Clock::now();
    KktSolver kkt(scaled_, rho_);
    result_.kktFactorSeconds += secondsSince(factorStart);
    result_.factorizations = 1;
    if (!kkt.isFactorValid()) {
        result_.status = QpStatus::NumericalFailure;
        result_.statusMessage = "KKT factorization failed";
        return result_;
    }

    // Infeasibility / unboundedness certificates (OSQP, Banjac et al. 2019).
    //
    // These are not optional refinements. The convergence test is a RELATIVE
    // one -- epsDual scales with ||A^T y|| -- so on a problem with no solution
    // the iterates diverge, the tolerance grows with them, and the run
    // eventually reports "Optimal" for a point that is nothing of the kind. On
    // "min -x-y s.t. x-y <= 1, x,y >= 0" this engine returned Optimal at
    // x = 2.96e7 with a dual residual of 1.0. The relative test is standard and
    // stays; what was missing is the pair of certificates that make it sound.
    //
    // Both are evaluated on the scaled problem. Positive diagonal scaling maps
    // feasible points to feasible points and recession directions to recession
    // directions, so the STATUS it certifies is the original problem's status.
    const double certEps = 1e-9;

    // delta-x certifies unboundedness (dual infeasibility) when it is a
    // direction that costs nothing to move along, strictly improves the
    // objective, and never leaves the feasible region.
    const auto certifiesUnbounded = [&](const std::vector<double>& dx) {
        double dxInf = 0.0;
        for (int j = 0; j < n; ++j) dxInf = std::max(dxInf, std::abs(dx[static_cast<std::size_t>(j)]));
        if (dxInf <= certEps) return false;
        const double tol = certEps * dxInf;

        std::vector<double> Pdx;
        scaled_.P.multiply(dx, Pdx);
        for (int j = 0; j < n; ++j)
            if (std::abs(Pdx[static_cast<std::size_t>(j)]) > tol) return false;  // curvature: not a ray

        double qdx = 0.0;
        for (int j = 0; j < n; ++j)
            qdx += scaled_.q[static_cast<std::size_t>(j)] * dx[static_cast<std::size_t>(j)];
        if (qdx >= -tol) return false;  // does not strictly improve

        if (m > 0) {
            std::vector<double> Adx;
            scaled_.A.multiply(dx, Adx);
            for (int i = 0; i < m; ++i) {
                const double a  = Adx[static_cast<std::size_t>(i)];
                const double lo = scaled_.l[static_cast<std::size_t>(i)];
                const double hi = scaled_.u[static_cast<std::size_t>(i)];
                const bool loFinite = std::isfinite(lo);
                const bool hiFinite = std::isfinite(hi);
                if (loFinite && hiFinite) { if (std::abs(a) > tol) return false; }
                else if (hiFinite)        { if (a >  tol) return false; }
                else if (loFinite)        { if (a < -tol) return false; }
            }
        }
        return true;
    };

    // delta-y certifies primal infeasibility: a nonnegative combination of the
    // rows that cancels in x but whose bound-side support is strictly negative
    // is exactly a Farkas certificate.
    const auto certifiesInfeasible = [&](const std::vector<double>& dy) {
        if (m == 0) return false;
        double dyInf = 0.0;
        for (int i = 0; i < m; ++i) dyInf = std::max(dyInf, std::abs(dy[static_cast<std::size_t>(i)]));
        if (dyInf <= certEps) return false;
        const double tol = certEps * dyInf;

        std::vector<double> Atdy;
        scaled_.A.transposeMultiply(dy, Atdy);
        for (int j = 0; j < n; ++j)
            if (std::abs(Atdy[static_cast<std::size_t>(j)]) > tol) return false;

        // support = u'max(dy,0) + l'min(dy,0). An infinite bound may only be
        // paired with a zero multiplier; anything else is not a certificate.
        double support = 0.0;
        for (int i = 0; i < m; ++i) {
            const double d  = dy[static_cast<std::size_t>(i)];
            const double lo = scaled_.l[static_cast<std::size_t>(i)];
            const double hi = scaled_.u[static_cast<std::size_t>(i)];
            if (d > tol) {
                if (!std::isfinite(hi)) return false;
                support += hi * d;
            } else if (d < -tol) {
                if (!std::isfinite(lo)) return false;
                support += lo * d;
            }
        }
        return support < -tol;
    };

    bool rhoChanged = false;
    double bestObj = std::numeric_limits<double>::infinity();
    // The best iterate (held by the backend) is recorded after the first
    // iteration; it can't be pre-filled with x/y because they start as all-zero,
    // which is generally infeasible and would corrupt the "best so far" tracking.
    bool hasBest = false;
    bool optimal = false;

    for (std::int64_t k = 0; k < options_.iterationLimit; ++k) {
        state.saveIterate();
        step(kkt, rhoChanged);

        // Residuals of the new iterate:
        //   r = A*x - z   (primal)
        //   s = -rho * A^T * (z - zOld)  (dual)
        //   s_alt = P*x + q  (stationarity, m == 0 path)
        const AdmmIterationMetrics metrics = state.metrics(rho_);
        const double rNorm = metrics.primalResidualNorm;
        const double sNorm = metrics.dualResidualNorm;

        const double obj = metrics.objective;
        if (std::isfinite(obj) && obj < bestObj) {
            bestObj = obj;
            state.recordBest();
            hasBest = true;
        }

        // Termination (Boyd et al. §3.3.1).
        const bool doCheck =
            (k + 1) % options_.terminationCheckFrequency == 0 ||
            k + 1 == options_.iterationLimit;
        if (doCheck) {
            const AdmmCheckView view = state.checkView();
            // Host views of the iterate. On a device backend this is where
            // it is downloaded -- once per check, never per iteration.
            const std::vector<double>& x = view.x;
            const std::vector<double>& xOld = view.xOld;
            const std::vector<double>& y = view.y;
            const std::vector<double>& yOld = view.yOld;
            const std::vector<double>& z = view.z;
            const std::vector<double>& Ax = view.Ax;
            const double absTol = options_.primalTolerance;
            const double relTol = options_.dualTolerance;

            double AxNorm = 0.0;
            double zNorm = 0.0;
            for (int i = 0; i < m; ++i) {
                AxNorm += Ax[static_cast<std::size_t>(i)] * Ax[static_cast<std::size_t>(i)];
                zNorm  += z[static_cast<std::size_t>(i)]  * z[static_cast<std::size_t>(i)];
            }
            AxNorm = std::sqrt(AxNorm);
            zNorm  = std::sqrt(zNorm);

            const double epsPri = std::sqrt(static_cast<double>(n + m)) * absTol +
                                  relTol * std::max({AxNorm, zNorm, 1.0});

            double AtzNorm = 0.0;
            if (m > 0) {
                std::vector<double> Atz;
                scaled_.A.transposeMultiply(y, Atz);
                for (int j = 0; j < n; ++j)
                    AtzNorm += Atz[static_cast<std::size_t>(j)] *
                               Atz[static_cast<std::size_t>(j)];
                AtzNorm = std::sqrt(AtzNorm);
            }
            const double epsDual = std::sqrt(static_cast<double>(n)) * absTol +
                                   relTol * (AtzNorm + 1.0);

            const bool primalOK = (m > 0) ? (rNorm <= epsPri) : true;
            const bool dualOK   = (sNorm <= epsDual);

            // The certificates are checked BEFORE optimality, not after.
            //
            // epsDual grows with ||A^T y||, so on an unbounded problem -- where y
            // diverges -- the convergence test is eventually satisfied by a
            // point that is not optimal at all, and it would win the race
            // against a certificate checked afterwards. Measured on
            // "min -x-y s.t. x-y <= 1, x,y >= 0": Optimal at x = 2.96e7 with a
            // dual residual of 1.0 against a tolerance that had inflated to 10.
            //
            // The order is safe in the other direction because a certified ray
            // and optimality are mutually exclusive: certifiesUnbounded requires
            // a strictly improving feasible recession direction, which a
            // bounded problem does not have, and both certificates require
            // ||delta|| > certEps, which fails near convergence where the
            // iterate differences go to zero.
            std::vector<double> dx(static_cast<std::size_t>(n));
            for (int j = 0; j < n; ++j)
                dx[static_cast<std::size_t>(j)] =
                    x[static_cast<std::size_t>(j)] - xOld[static_cast<std::size_t>(j)];
            if (certifiesUnbounded(dx)) {
                result_.status = QpStatus::Unbounded;
                result_.statusMessage = "unbounded: improving ray certified from iterate difference";
                result_.iterations = k + 1;
                break;
            }

            std::vector<double> dy(static_cast<std::size_t>(m));
            for (int i = 0; i < m; ++i)
                dy[static_cast<std::size_t>(i)] =
                    y[static_cast<std::size_t>(i)] - yOld[static_cast<std::size_t>(i)];
            if (certifiesInfeasible(dy)) {
                result_.status = QpStatus::Infeasible;
                result_.statusMessage = "infeasible: Farkas certificate from dual iterate difference";
                result_.iterations = k + 1;
                break;
            }

            if (primalOK && dualOK) {
                result_.status = QpStatus::Optimal;
                result_.statusMessage = "converged";
                result_.iterations = k + 1;
                optimal = true;
                // Use the last iterate (x) rather than the best one, because the
                // "best" objective tracker can pick an infeasible early iterate
                // over a feasible converged one.  The converged x is guaranteed
                // feasible by the termination check.
                break;
            }
        }

        // Adaptive rho.
        //
        // The desired rho is tracked continuously, but adopted only when it has
        // drifted far enough from the factorised value to be worth a numeric
        // refactorisation, and never more often than adaptiveRhoInterval
        // iterations apart. rho and the factor must agree -- the x-update's
        // right-hand side uses rho -- so there is no way to move one without the
        // other; the saving has to come from moving less often.
        if (options_.useAdaptiveRho && m > 0) {
            const double mu = options_.adaptiveRhoMu;
            const double tau = options_.adaptiveRhoTau;
            if (rNorm > mu * sNorm) {
                desiredRho_ *= tau;
            } else if (sNorm > mu * rNorm) {
                desiredRho_ /= tau;
            }

            const double threshold = std::max(options_.adaptiveRhoThreshold, 1.0);
            const bool farEnough = desiredRho_ > threshold * rho_ ||
                                   desiredRho_ * threshold < rho_;
            const bool longEnough =
                (k - lastRhoUpdate_) >= options_.adaptiveRhoInterval;

            if (farEnough && longEnough && desiredRho_ > 0.0) {
                rho_ = desiredRho_;
                const Clock::time_point refactorStart = Clock::now();
                const bool refactored = kkt.refactor(rho_);
                result_.kktFactorSeconds += secondsSince(refactorStart);
                if (refactored) {
                    ++result_.factorizations;
                    lastRhoUpdate_ = k;
                }
            }
            rhoChanged = false;
        }

        if (options_.timeLimitSeconds > 0.0) {
            const double elapsed =
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - tStart).count();
            if (elapsed >= options_.timeLimitSeconds) {
                result_.status = QpStatus::TimeLimit;
                result_.statusMessage = "time limit";
                result_.iterations = k + 1;
                state.restoreBest();
                break;
            }
        }
    }

    if (!optimal && result_.status == QpStatus::IterationLimit) {
        if (hasBest) {
            state.restoreBest();
        }
    }

    toOriginal();

    // result_.primal is always set to x in toOriginal() (for no-scaling case)
    // or to the scaled result (for scaling case). Ensure it's set.
    if (result_.primal.empty() && !state.primal().empty())
        result_.primal = state.primal();

    // Compute the primal objective on the original (un-scaled) problem so
    // the caller sees the value in their own coordinate system.  result_.primal
    // is already in original coordinates (toOriginal ran above).
    result_.primalObjective = detail::objectiveValue(original_, result_.primal);
    result_.dualObjective   = -result_.primalObjective;
    result_.finalRho = rho_;
    result_.bestObjective = bestObj;
    result_.iterations = result_.iterations == 0 ? options_.iterationLimit : result_.iterations;

    // Compute actual residuals on the original problem.
    {
        const int mo = original_.numConstraints();
        result_.primalResidual = 0.0;
        if (mo > 0) {
            std::vector<double> AxOrig;
            original_.A.multiply(result_.primal, AxOrig);
            for (int i = 0; i < mo; ++i) {
                const double v = AxOrig[static_cast<std::size_t>(i)];
                const double lo = original_.l[static_cast<std::size_t>(i)];
                const double hi = original_.u[static_cast<std::size_t>(i)];
                if (v < lo) result_.primalResidual = std::max(result_.primalResidual, lo - v);
                if (v > hi) result_.primalResidual = std::max(result_.primalResidual, v - hi);
            }
        }
        // Dual residual: ||P x + q + A^T y||_inf.
        std::vector<double> PxOrig;
        original_.P.multiply(result_.primal, PxOrig);
        std::vector<double> AtyOrig;
        if (mo > 0) original_.A.transposeMultiply(result_.constraintDual, AtyOrig);
        else AtyOrig.assign(static_cast<std::size_t>(n), 0.0);
        double dualR = 0.0;
        for (int j = 0; j < n; ++j) {
            const double v = PxOrig[static_cast<std::size_t>(j)] +
                             original_.q[static_cast<std::size_t>(j)] +
                             AtyOrig[static_cast<std::size_t>(j)];
            dualR = std::max(dualR, std::abs(v));
        }
        result_.dualResidual = dualR;
    }

    result_.solveTimeSeconds =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - tStart).count();
    result_.backendProfile = state.profile();
    return result_;
}

void AdmmSolver::step(KktSolver& kkt, bool& rhoChanged) {
    // x-update right-hand side, sigma*x_prev - q + rho*A^T(z - y/rho); see
    // AdmmBackend::buildRhs for why each term is there.
    std::vector<double> rhs;
    backend_->buildRhs(rho_, rhs);

    // The KKT solve runs on the host on every backend.
    const Clock::time_point solveStart = Clock::now();
    const bool solved = kkt.solve(rhs);
    result_.kktSolveSeconds += secondsSince(solveStart);
    if (!solved) {
        // The factor went bad; nudge rho and rebuild rather than continue on it.
        rho_ *= 1.5;
        desiredRho_ = rho_;
        rhoChanged = true;
        return;
    }

    // x = rhs, then the z-update (projection onto the box) and the y-update.
    backend_->acceptPrimal(std::move(rhs), rho_);
}

void AdmmSolver::toOriginal() {
    const std::vector<double>& x = backend_->primal();
    const std::vector<double>& y = backend_->dual();
    if (!options_.useRuizScaling || !scalingValid_) {
        result_.primal = x;
        result_.constraintDual = y;
        return;
    }
    scaling_.toOriginal(x, result_.primal);
    scaling_.toOriginalDual(y, result_.constraintDual);
}

}  // namespace qp
