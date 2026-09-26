#include "qp/admm_backend.h"

#include "qp/kkt_solver.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>

namespace qp {

namespace detail {

double objectiveValue(const QpModel& model, const std::vector<double>& x) {
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

}  // namespace detail

namespace {

// The ADMM arithmetic exactly as AdmmSolver performed it before backends
// existed -- the same products (serial or planned), the same temporaries, the
// same loop order -- so a CPU solve is bit-for-bit what it was.
class CpuAdmmBackend final : public AdmmBackend {
public:
    CpuAdmmBackend(
        const QpModel& model,
        Executor* executor,
        const SparseMatrix::Plan* planA,
        const SparseMatrix::Plan* planP
    )
        : model_(model), executor_(executor), planA_(planA), planP_(planP) {
        const int n = model.numVariables();
        const int m = model.numConstraints();
        x_.assign(static_cast<std::size_t>(n), 0.0);
        z_.assign(static_cast<std::size_t>(m), 0.0);
        y_.assign(static_cast<std::size_t>(m), 0.0);
        Ax_.assign(static_cast<std::size_t>(m), 0.0);
        zOld_.assign(static_cast<std::size_t>(m), 0.0);
        xOld_.assign(static_cast<std::size_t>(n), 0.0);
        yOld_.assign(static_cast<std::size_t>(m), 0.0);
    }

    void saveIterate() override {
        zOld_ = z_;
        xOld_ = x_;
        yOld_ = y_;
    }

    void buildRhs(double rho, std::vector<double>& rhs) override {
        const int n = model_.numVariables();
        const int m = model_.numConstraints();

        // x-update: solve (P + sigma I + rho A^T A) x = sigma x_prev - q + rho A^T z - A^T y
        //
        // The sigma*x_prev is what makes the factorisation's sigma*I free of charge:
        // at the fixed point x = x_prev the two sigma terms cancel and what is left
        // is (P + rho A^T A) x = -q + rho A^T z - A^T y, the unregularised
        // condition. Dropping this term would turn sigma into a silent perturbation
        // of the problem, which is the bug that folding an epsilon onto P used to
        // cause. See KktSolver::kSigma.
        rhs.assign(static_cast<std::size_t>(n), 0.0);
        for (int j = 0; j < n; ++j)
            rhs[static_cast<std::size_t>(j)] =
                KktSolver::kSigma * x_[static_cast<std::size_t>(j)] - model_.q[static_cast<std::size_t>(j)];

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
                    z_[static_cast<std::size_t>(i)] - y_[static_cast<std::size_t>(i)] / rho;

            std::vector<double> Atz;
            model_.A.transposeMultiply(zMinusYOverRho, Atz, executor_, planA_);

            for (int j = 0; j < n; ++j) {
                rhs[static_cast<std::size_t>(j)] += rho * Atz[static_cast<std::size_t>(j)];
            }
        }
    }

    void acceptPrimal(std::vector<double>&& solution, double rho) override {
        const int m = model_.numConstraints();
        x_ = std::move(solution);

        // z-update: projection onto the box.
        if (m > 0) {
            model_.A.multiply(x_, Ax_, executor_, planA_);
            for (int i = 0; i < m; ++i) {
                const double v = Ax_[static_cast<std::size_t>(i)] +
                                 y_[static_cast<std::size_t>(i)] / rho;
                const double lo = model_.l[static_cast<std::size_t>(i)];
                const double hi = model_.u[static_cast<std::size_t>(i)];
                z_[static_cast<std::size_t>(i)] = std::min(std::max(v, lo), hi);
            }
        }

        // y-update.
        if (m > 0) {
            for (int i = 0; i < m; ++i) {
                y_[static_cast<std::size_t>(i)] +=
                    rho * (Ax_[static_cast<std::size_t>(i)] -
                           z_[static_cast<std::size_t>(i)]);
            }
        }
    }

    AdmmIterationMetrics metrics(double rho) override {
        const int n = model_.numVariables();
        const int m = model_.numConstraints();

        // Compute residuals.
        //   r = A*x - z   (primal)
        //   s = -rho * A^T * (z - zOld)  (dual)
        //   s_alt = P*x + q + A^T*y  (stationarity, m == 0 path)
        double rNorm = 0.0;
        double sNorm = 0.0;
        if (m > 0) {
            model_.A.multiply(x_, Ax_, executor_, planA_);
            for (int i = 0; i < m; ++i) {
                const double r = Ax_[static_cast<std::size_t>(i)] -
                                 z_[static_cast<std::size_t>(i)];
                rNorm += r * r;
            }
            rNorm = std::sqrt(rNorm);

            std::vector<double> zDiff(static_cast<std::size_t>(m));
            for (int i = 0; i < m; ++i) {
                zDiff[static_cast<std::size_t>(i)] =
                    z_[static_cast<std::size_t>(i)] - zOld_[static_cast<std::size_t>(i)];
            }
            std::vector<double> sVec;
            model_.A.transposeMultiply(zDiff, sVec);
            const double scale = rho;
            for (int j = 0; j < n; ++j) {
                sNorm += (scale * sVec[static_cast<std::size_t>(j)]) *
                         (scale * sVec[static_cast<std::size_t>(j)]);
            }
            sNorm = std::sqrt(sNorm);
        } else {
            // m == 0: dual residual is the gradient.
            std::vector<double> grad;
            model_.P.multiply(x_, grad, executor_, planP_);
            for (int j = 0; j < n; ++j)
                grad[static_cast<std::size_t>(j)] += model_.q[static_cast<std::size_t>(j)];
            for (int j = 0; j < n; ++j)
                sNorm += grad[static_cast<std::size_t>(j)] * grad[static_cast<std::size_t>(j)];
            sNorm = std::sqrt(sNorm);
        }

        AdmmIterationMetrics result;
        result.primalResidualNorm = rNorm;
        result.dualResidualNorm = sNorm;
        result.objective = detail::objectiveValue(model_, x_);
        return result;
    }

    void recordBest() override {
        bestX_ = x_;
        bestY_ = y_;
    }

    void restoreBest() override {
        x_ = bestX_;
        y_ = bestY_;
    }

    AdmmCheckView checkView() override {
        return AdmmCheckView{x_, xOld_, y_, yOld_, z_, Ax_};
    }

    const std::vector<double>& primal() override { return x_; }
    const std::vector<double>& dual() override { return y_; }

    ComputeBackend kind() const noexcept override { return ComputeBackend::Cpu; }

private:
    const QpModel& model_;
    Executor* executor_;
    const SparseMatrix::Plan* planA_;
    const SparseMatrix::Plan* planP_;

    std::vector<double> x_, z_, y_, Ax_;
    std::vector<double> xOld_, zOld_, yOld_;
    std::vector<double> bestX_, bestY_;
};

}  // namespace

const char* toString(ComputeBackend backend) noexcept {
    switch (backend) {
        case ComputeBackend::Auto: return "auto";
        case ComputeBackend::Cpu:  return "cpu";
        case ComputeBackend::Cuda: return "cuda";
    }
    return "unknown";
}

std::unique_ptr<AdmmBackend> makeCpuAdmmBackend(
    const QpModel& model,
    Executor* executor,
    const SparseMatrix::Plan* planA,
    const SparseMatrix::Plan* planP
) {
    return std::make_unique<CpuAdmmBackend>(model, executor, planA, planP);
}

}  // namespace qp
