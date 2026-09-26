#pragma once

#include "qp/compute_backend.h"
#include "qp/parallel.h"
#include "qp/qp_model.h"

#include <memory>
#include <string>
#include <vector>

namespace qp {

// Norms and objective of the current ADMM iterate; see AdmmSolver::solve.
struct AdmmIterationMetrics {
    double primalResidualNorm = 0.0;  // ||A x - z||, or 0 when m == 0
    double dualResidualNorm = 0.0;    // ||rho A^T (z - z_old)||, or ||P x + q|| when m == 0
    double objective = 0.0;           // 0.5 x'Px + q'x on the scaled problem
};

// Host views of the iterate for the periodic termination and certificate
// checks. References stay valid until the next mutating call.
struct AdmmCheckView {
    const std::vector<double>& x;
    const std::vector<double>& xOld;
    const std::vector<double>& y;
    const std::vector<double>& yOld;
    const std::vector<double>& z;
    const std::vector<double>& Ax;
};

// The ADMM iterate (x, z, y, A*x, the previous iterate and the best one seen)
// and the per-iteration operations on it.
//
// AdmmSolver keeps the algorithm -- the KKT solve, rho adaptation, the
// termination test, the certificates, best-iterate policy -- and calls these
// in the order the iteration defines. The CPU backend performs the arithmetic
// the solver always did, verbatim, so a CPU solve is unchanged.
//
// The x-update is split in two because the KKT solve between the halves always
// runs on the host:
//   buildRhs      rhs = sigma*x - q + rho*A^T(z - y/rho)
//   (KKT solve on the host)
//   acceptPrimal  x = solution; Ax = A x; z = proj(Ax + y/rho); y += rho(Ax - z)
class AdmmBackend {
public:
    virtual ~AdmmBackend() = default;

    // Remembers the iterate the residuals and certificates are measured
    // against: xOld = x, zOld = z, yOld = y.
    virtual void saveIterate() = 0;

    virtual void buildRhs(double rho, std::vector<double>& rhs) = 0;
    virtual void acceptPrimal(std::vector<double>&& solution, double rho) = 0;

    [[nodiscard]] virtual AdmmIterationMetrics metrics(double rho) = 0;

    // bestX = x, bestY = y; and the reverse. restoreBest() before any
    // recordBest() empties x and y, exactly as assigning the never-filled best
    // vectors did.
    virtual void recordBest() = 0;
    virtual void restoreBest() = 0;

    [[nodiscard]] virtual AdmmCheckView checkView() = 0;

    // Final iterate, scaled coordinates.
    [[nodiscard]] virtual const std::vector<double>& primal() = 0;
    [[nodiscard]] virtual const std::vector<double>& dual() = 0;

    [[nodiscard]] virtual ComputeBackend kind() const noexcept = 0;
    [[nodiscard]] virtual BackendProfile profile() const noexcept { return {}; }
};

// `model` (the scaled problem) must outlive the backend. Null executor or
// plans select serial products, as in AdmmSolver.
[[nodiscard]] std::unique_ptr<AdmmBackend> makeCpuAdmmBackend(
    const QpModel& model,
    Executor* executor,
    const SparseMatrix::Plan* planA,
    const SparseMatrix::Plan* planP
);

// The hybrid CUDA backend, or null with `error` set. Never falls back itself.
[[nodiscard]] std::unique_ptr<AdmmBackend> makeCudaAdmmBackend(
    const QpModel& model,
    int device,
    std::string& error
);

namespace detail {

// 0.5 x'Px + q'x with the solver's historical evaluation order.
[[nodiscard]] double objectiveValue(const QpModel& model, const std::vector<double>& x);

}  // namespace detail

}  // namespace qp
