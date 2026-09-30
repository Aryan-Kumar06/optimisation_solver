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

double primalObjective(const QpModel& model, const std::vector<double>& x) {
    const int n = model.numVariables();
    std::vector<double> Px;
    model.P.multiply(x, Px);
    double obj = 0.0;
    for (int j = 0; j < n; ++j)
        obj += model.q[static_cast<std::size_t>(j)] * x[static_cast<std::size_t>(j)];
    for (int j = 0; j < n; ++j)
        obj += 0.5 * x[static_cast<std::size_t>(j)] * Px[static_cast<std::size_t>(j)];
    return obj;
}

}  // namespace

KktCheck checkKkt(const QpModel& model, const std::vector<double>& x,
                  const std::vector<double>& y, double primalTolerance, double dualTolerance) {
    KktCheck out;
    const int n = model.numVariables();
    const int m = model.numConstraints();
    if (x.size() != static_cast<std::size_t>(n) || y.size() != static_cast<std::size_t>(m)) return out;
    for (double v : x) if (!std::isfinite(v)) return out;
    for (double v : y) if (!std::isfinite(v)) return out;

    std::vector<double> ax, px, aty(static_cast<std::size_t>(n), 0.0);
    model.A.multiply(x, ax);
    model.P.multiply(x, px);
    if (m > 0) model.A.transposeMultiply(y, aty);

    out.primalViolation = 0.0;
    out.primalMet = true;
    for (int i = 0; i < m; ++i) {
        const auto ui = static_cast<std::size_t>(i);
        const double v = ax[ui], lo = model.l[ui], hi = model.u[ui];
        const double violation = std::max({0.0, lo - v, v - hi});
        double scale = std::max(1.0, std::abs(v));
        if (std::isfinite(lo)) scale = std::max(scale, std::abs(lo));
        if (std::isfinite(hi)) scale = std::max(scale, std::abs(hi));
        out.primalViolation = std::max(out.primalViolation, violation);
        if (!(violation <= primalTolerance * scale)) out.primalMet = false;
    }
    double dual = 0.0, dualScale = 1.0;
    for (int j = 0; j < n; ++j) {
        const auto uj = static_cast<std::size_t>(j);
        dual = std::max(dual, std::abs(px[uj] + model.q[uj] + aty[uj]));
        dualScale = std::max({dualScale, std::abs(px[uj]), std::abs(aty[uj]), std::abs(model.q[uj])});
    }
    out.dualResidual = dual;
    out.dualMet = dual <= dualTolerance * dualScale;
    out.finite = std::isfinite(out.primalViolation) && std::isfinite(dual) && std::isfinite(dualScale);
    return out;
}

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
    if (!(options_.adaptiveRhoTolerance > 1.0))
        options_.adaptiveRhoTolerance = 5.0;
    if (options_.maximumRhoUpdates < 0)
        options_.maximumRhoUpdates = 0;
    if (!(options_.rhoMinimum > 0.0)) options_.rhoMinimum = 1e-6;
    if (!(options_.rhoMaximum >= options_.rhoMinimum)) options_.rhoMaximum = 1e6;
    if (!(options_.maximumRhoStep > 1.0)) options_.maximumRhoStep = 10.0;

    // Validate the problem first, before any equilibration.
    try {
        problem.validate();
    } catch (const std::exception& e) {
        result_.status = QpStatus::InvalidProblem;
        result_.statusMessage = e.what();
        return;
    }

    // This is a CONVEX-QP engine. A negative diagonal entry proves P is not
    // positive semidefinite, and ADMM then converges happily to a stationary
    // point that need not be a minimum: on min -x^2/2 over [-1, 1] it returned
    // Optimal at x = 0, the MAXIMUM. Full PSD verification is a dense
    // factorisation and belongs to the caller (the public pipeline runs
    // qp::checkConvexity before dispatch); this O(nnz) test is the necessary
    // condition the engine can afford on every call.
    {
        const auto& start = problem.P.csrRowStart();
        const auto& column = problem.P.csrColumnIndex();
        const auto& value = problem.P.csrValues();
        for (int i = 0; i < problem.numVariables(); ++i)
            for (auto k = start[static_cast<std::size_t>(i)]; k < start[static_cast<std::size_t>(i) + 1]; ++k)
                if (column[static_cast<std::size_t>(k)] == i && value[static_cast<std::size_t>(k)] < 0.0) {
                    result_.status = QpStatus::InvalidProblem;
                    result_.statusMessage = "P has a negative diagonal entry, so it is not positive "
                                            "semidefinite; this engine solves convex QPs only";
                    return;
                }
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

    x_.assign(static_cast<std::size_t>(n), 0.0);
    z_.assign(static_cast<std::size_t>(m), 0.0);
    y_.assign(static_cast<std::size_t>(m), 0.0);
    Ax_.assign(static_cast<std::size_t>(m), 0.0);
    rho_ = std::clamp(options_.rho, options_.rhoMinimum, options_.rhoMaximum);
    rhoFloor_ = options_.rhoMinimum;
    rhoCeiling_ = options_.rhoMaximum;

    result_.primal.assign(static_cast<std::size_t>(n), 0.0);
    result_.constraintDual.assign(static_cast<std::size_t>(m), 0.0);
    result_.status = QpStatus::IterationLimit;
}

AdmmResult AdmmSolver::solve() {
    if (result_.status == QpStatus::InvalidProblem)
        return result_;

    const auto tStart = std::chrono::steady_clock::now();
    const int n = scaled_.numVariables();
    const int m = scaled_.numConstraints();

    if (n == 0) {
        // With no variables every row's activity is 0, so each row must admit
        // 0. A row that does not is a complete infeasibility certificate on its
        // own. This used to return Optimal unconditionally -- including for a
        // row requiring 0 in [1, 2], with a reported primal residual of 1.0.
        result_.primal.clear();
        result_.constraintDual.assign(static_cast<std::size_t>(m), 0.0);
        result_.iterations = 0;
        result_.primalObjective = 0.0;
        result_.primalResidual = 0.0;
        result_.dualResidual = 0.0;
        for (int i = 0; i < m; ++i) {
            const double lo = original_.l[static_cast<std::size_t>(i)];
            const double hi = original_.u[static_cast<std::size_t>(i)];
            const double violation = std::max({0.0, lo, -hi});
            result_.primalResidual = std::max(result_.primalResidual, violation);
            if (violation > 0.0) {
                result_.status = QpStatus::Infeasible;
                result_.statusMessage = "infeasible: with no variables, row " + std::to_string(i) +
                                        " requires 0 to lie in [" + std::to_string(lo) + ", " +
                                        std::to_string(hi) + "]";
                return result_;
            }
        }
        result_.status = QpStatus::Optimal;
        result_.statusMessage = "zero variables; every row admits the empty point";
        return result_;
    }

    KktSolver kkt(scaled_, rho_);
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

    std::vector<double> xOld(static_cast<std::size_t>(n), 0.0);
    std::vector<double> yOld(static_cast<std::size_t>(m), 0.0);
    double bestObj = std::numeric_limits<double>::infinity();
    // bestX/bestY are filled in after the first iteration; we can't pre-fill
    // with x_/y_ because they start as all-zero which is generally infeasible
    // and would corrupt the "best so far" tracking.
    std::vector<double> bestX;
    std::vector<double> bestY;
    bool hasBest = false;
    bool optimal = false;

    for (std::int64_t k = 0; k < options_.iterationLimit; ++k) {
        xOld = x_;
        yOld = y_;
        if (!step(kkt)) {
            result_.status = QpStatus::NumericalFailure;
            result_.statusMessage = "KKT solve failed; the factorisation is unusable";
            result_.iterations = k + 1;
            break;
        }

        // Primal residual r = A*x - z. The dual residual is the true
        // stationarity residual, computed at termination checks below.
        double rNorm = 0.0;
        if (m > 0) {
            scaled_.A.multiply(x_, Ax_,
                               parallel_ ? executor_.get() : nullptr,
                               parallel_ ? &planA_ : nullptr);
            for (int i = 0; i < m; ++i) {
                const double r = Ax_[static_cast<std::size_t>(i)] -
                                 z_[static_cast<std::size_t>(i)];
                rNorm += r * r;
            }
            rNorm = std::sqrt(rNorm);
        }

        const double obj = primalObjective(scaled_, x_);
        if (std::isfinite(obj) && obj < bestObj) {
            bestObj = obj;
            bestX = x_;
            bestY = y_;
            hasBest = true;
        }

        // Termination (Boyd et al. §3.3.1).
        const bool doCheck =
            (k + 1) % options_.terminationCheckFrequency == 0 ||
            k + 1 == options_.iterationLimit;
        if (doCheck) {

            double AxNorm = 0.0;
            double zNorm = 0.0;
            for (int i = 0; i < m; ++i) {
                AxNorm += Ax_[static_cast<std::size_t>(i)] * Ax_[static_cast<std::size_t>(i)];
                zNorm  += z_[static_cast<std::size_t>(i)]  * z_[static_cast<std::size_t>(i)];
            }
            AxNorm = std::sqrt(AxNorm);
            zNorm  = std::sqrt(zNorm);


            double AtzNorm = 0.0;
            if (m > 0) {
                std::vector<double> Atz;
                scaled_.A.transposeMultiply(y_, Atz);
                for (int j = 0; j < n; ++j)
                    AtzNorm += Atz[static_cast<std::size_t>(j)] *
                               Atz[static_cast<std::size_t>(j)];
                AtzNorm = std::sqrt(AtzNorm);
            }
            // Dual residual: the TRUE stationarity residual P x + q + A'y, as
            // in OSQP -- not the ADMM proxy rho * A'(z - zOld). The proxy only
            // measures how much z moved. When rho is very large z stops moving,
            // the proxy goes to zero, and the old test declared convergence:
            // measured on Maros-Meszaros hs51, hs52 and genhs28, "Optimal" with
            // a true stationarity residual of 0.04-0.13 against a 1e-8
            // tolerance, and objectives off by up to 4e-4 relative.
            std::vector<double> Px, stationarity(static_cast<std::size_t>(n));
            scaled_.P.multiply(x_, Px);
            double PxNorm = 0.0, qNorm = 0.0, dNorm = 0.0;
            std::vector<double> Aty(static_cast<std::size_t>(n), 0.0);
            if (m > 0) scaled_.A.transposeMultiply(y_, Aty);
            for (int j = 0; j < n; ++j) {
                const auto uj = static_cast<std::size_t>(j);
                const double d = Px[uj] + scaled_.q[uj] + Aty[uj];
                dNorm += d * d;
                PxNorm += Px[uj] * Px[uj];
                qNorm += scaled_.q[uj] * scaled_.q[uj];
            }
            dNorm = std::sqrt(dNorm);
            PxNorm = std::sqrt(PxNorm);
            qNorm = std::sqrt(qNorm);
            // A non-finite iterate is a numerical breakdown. Stop and report
            // it: carrying on only burns the remaining budget and ends as an
            // iteration limit, which says "needs more time" when the truth is
            // "cannot continue".
            bool finiteIterate = std::isfinite(rNorm) && std::isfinite(dNorm);
            for (int j = 0; finiteIterate && j < n; ++j)
                finiteIterate = std::isfinite(x_[static_cast<std::size_t>(j)]);
            for (int i = 0; finiteIterate && i < m; ++i)
                finiteIterate = std::isfinite(y_[static_cast<std::size_t>(i)]);
            if (!finiteIterate) {
                result_.status = QpStatus::NumericalFailure;
                result_.statusMessage = "iterate became non-finite";
                result_.iterations = k + 1;
                if (hasBest) { x_ = bestX; y_ = bestY; }
                break;
            }

            // Convergence is judged in the ORIGINAL problem's units; see
            // checkKkt. The scaled norms above still drive rho adaptation.
            std::vector<double> xOriginal = x_, yOriginal = y_;
            if (options_.useRuizScaling && scalingValid_) {
                scaling_.toOriginal(x_, xOriginal);
                scaling_.toOriginalDual(y_, yOriginal);
            }
            const KktCheck kktCheck = checkKkt(original_, xOriginal, yOriginal,
                                               options_.primalTolerance, options_.dualTolerance);
            const bool primalOK = kktCheck.finite && kktCheck.primalMet;
            const bool dualOK = kktCheck.finite && kktCheck.dualMet;

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
                    x_[static_cast<std::size_t>(j)] - xOld[static_cast<std::size_t>(j)];
            if (certifiesUnbounded(dx)) {
                result_.status = QpStatus::Unbounded;
                result_.statusMessage = "unbounded: improving ray certified from iterate difference";
                result_.iterations = k + 1;
                break;
            }

            std::vector<double> dy(static_cast<std::size_t>(m));
            for (int i = 0; i < m; ++i)
                dy[static_cast<std::size_t>(i)] =
                    y_[static_cast<std::size_t>(i)] - yOld[static_cast<std::size_t>(i)];
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
                // Use the last iterate (x_) rather than bestX, because the
                // "best" objective tracker can pick an infeasible early iterate
                // over a feasible converged one.  The converged x_ is guaranteed
                // feasible by the termination check.
                break;
            }

            // Adaptive rho, at checks only (see AdmmOptions::useAdaptiveRho).
            if (options_.useAdaptiveRho && m > 0 && rhoUpdates_ < options_.maximumRhoUpdates) {
                const double primalScale = std::max({AxNorm, zNorm, 1e-300});
                const double dualScale = std::max({PxNorm, AtzNorm, qNorm, 1e-300});
                // Floored rather than skipped when zero. An exactly zero primal
                // residual is the clearest possible signal that rho should
                // FALL: the iterate is feasible and only stationarity is left.
                // Skipping that case left rho stuck at 0.1 on Maros-Meszaros
                // hs268, which needs rho near 1e-6, and the solve crawled to
                // the iteration limit. With the floor, a zero residual yields
                // the maximum damped step in the right direction.
                const double primalRelative = std::max(rNorm / primalScale, 1e-300);
                const double dualRelative = std::max(dNorm / dualScale, 1e-300);
                const bool bothConverged = rNorm == 0.0 && dNorm == 0.0;
                if (!bothConverged && std::isfinite(primalRelative) && std::isfinite(dualRelative)) {
                    // Damped: at most maximumRhoStep per update. A residual can
                    // be almost exactly zero early on, which makes the undamped
                    // ratio enormous: on the NLP engine's first elastic QP one
                    // update proposed 1 -> 1.8e5, the next 3e-4, then 1e6, and
                    // the three states repeated until the iteration limit.
                    const double step = options_.maximumRhoStep;
                    double proposal = rho_ * std::clamp(std::sqrt(primalRelative / dualRelative),
                                                        1.0 / step, step);
                    proposal = std::clamp(proposal, rhoFloor_, rhoCeiling_);
                    const double tolerance = options_.adaptiveRhoTolerance;
                    if (proposal > tolerance * rho_ || proposal * tolerance < rho_) {
                        const double previous = rho_;
                        if (kkt.refactor(proposal)) {
                            rho_ = proposal;
                            ++rhoUpdates_;
                            ++result_.factorizations;
                        } else {
                            // The factorisation failed at this rho (typically a
                            // large rho making P + sigma I + rho A'A too
                            // ill-conditioned for Cholesky). Restore the last
                            // factor that worked and never propose this far
                            // again. Before, rho kept the failed value with no
                            // valid factor behind it and x never moved again.
                            if (proposal > previous) rhoCeiling_ = std::sqrt(previous * proposal);
                            else rhoFloor_ = std::sqrt(previous * proposal);
                            if (!kkt.refactor(previous)) {
                                result_.status = QpStatus::NumericalFailure;
                                result_.statusMessage = "KKT refactorisation failed and could not be restored";
                                result_.iterations = k + 1;
                                break;
                            }
                            ++result_.factorizations;
                        }
                    }
                }
            }
        }


        if (options_.timeLimitSeconds > 0.0) {
            const double elapsed =
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - tStart).count();
            if (elapsed >= options_.timeLimitSeconds) {
                result_.status = QpStatus::TimeLimit;
                result_.statusMessage = "time limit";
                result_.iterations = k + 1;
                if (hasBest) { x_ = bestX; y_ = bestY; }
                break;
            }
        }
    }

    if (!optimal && result_.status == QpStatus::IterationLimit) {
        if (hasBest) {
            x_ = bestX;
            y_ = bestY;
        }
    }

    toOriginal();

    // result_.primal is always set to x_ in toOriginal() (for no-scaling case)
    // or to the scaled result (for scaling case). Ensure it's set.
    if (result_.primal.empty() && !x_.empty())
        result_.primal = x_;

    // Compute the primal objective on the original (un-scaled) problem so
    // the caller sees the value in their own coordinate system.  result_.primal
    // is already in original coordinates (toOriginal ran above).
    result_.primalObjective = primalObjective(original_, result_.primal);
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
    return result_;
}

bool AdmmSolver::step(KktSolver& kkt) {
    const int n = scaled_.numVariables();
    const int m = scaled_.numConstraints();

    // x-update: solve (P + sigma I + rho A^T A) x = sigma x_prev - q + rho A^T z - A^T y
    //
    // The sigma*x_prev is what makes the factorisation's sigma*I free of charge:
    // at the fixed point x = x_prev the two sigma terms cancel and what is left
    // is (P + rho A^T A) x = -q + rho A^T z - A^T y, the unregularised
    // condition. Dropping this term would turn sigma into a silent perturbation
    // of the problem, which is the bug that folding an epsilon onto P used to
    // cause. See KktSolver::kSigma.
    std::vector<double> rhs(static_cast<std::size_t>(n));
    for (int j = 0; j < n; ++j)
        rhs[static_cast<std::size_t>(j)] =
            KktSolver::kSigma * x_[static_cast<std::size_t>(j)] - scaled_.q[static_cast<std::size_t>(j)];

    if (m > 0) {
        // rho * A^T * (z - y/rho)  IS  rho*A^T*z - A^T*y, the whole term.
        //
        // An earlier version then subtracted A^T*y a second time, so the
        // right-hand side was -q + rho*A^T*z - 2*A^T*y. The iteration still
        // converged, and to a stable fixed point with tiny ADMM residuals -- but
        // of the wrong operator. On a problem whose only constraint was repeated
        // k times it returned x_true/k, and the duplication is exactly what
        // encoding variable bounds as extra rows produces.
        std::vector<double> zMinusYOverRho(static_cast<std::size_t>(m));
        for (int i = 0; i < m; ++i)
            zMinusYOverRho[static_cast<std::size_t>(i)] =
                z_[static_cast<std::size_t>(i)] - y_[static_cast<std::size_t>(i)] / rho_;

        std::vector<double> Atz;
        scaled_.A.transposeMultiply(zMinusYOverRho, Atz,
                                    parallel_ ? executor_.get() : nullptr,
                                    parallel_ ? &planA_ : nullptr);

        for (int j = 0; j < n; ++j) {
            rhs[static_cast<std::size_t>(j)] += rho_ * Atz[static_cast<std::size_t>(j)];
        }
    }

    if (!kkt.solve(rhs)) {
        // Only an invalid factor makes the solve fail, and a refactorisation
        // failure is already recovered from where it happens. Report it. The
        // old code multiplied rho by 1.5 and returned without refactorising or
        // updating x, so rho and the factor disagreed and x stayed frozen.
        return false;
    }
    x_ = std::move(rhs);

    // z-update: projection onto the box.
    if (m > 0) {
        scaled_.A.multiply(x_, Ax_,
                           parallel_ ? executor_.get() : nullptr,
                           parallel_ ? &planA_ : nullptr);
        for (int i = 0; i < m; ++i) {
            const double v = Ax_[static_cast<std::size_t>(i)] +
                             y_[static_cast<std::size_t>(i)] / rho_;
            const double lo = scaled_.l[static_cast<std::size_t>(i)];
            const double hi = scaled_.u[static_cast<std::size_t>(i)];
            z_[static_cast<std::size_t>(i)] = std::min(std::max(v, lo), hi);
        }
    }

    // y-update.
    if (m > 0) {
        for (int i = 0; i < m; ++i) {
            y_[static_cast<std::size_t>(i)] +=
                rho_ * (Ax_[static_cast<std::size_t>(i)] -
                        z_[static_cast<std::size_t>(i)]);
        }
    }
    return true;
}

void AdmmSolver::toOriginal() {
    if (!options_.useRuizScaling || !scalingValid_) {
        result_.primal = x_;
        result_.constraintDual = y_;
        return;
    }
    scaling_.toOriginal(x_, result_.primal);
    scaling_.toOriginalDual(y_, result_.constraintDual);
}

}  // namespace qp
