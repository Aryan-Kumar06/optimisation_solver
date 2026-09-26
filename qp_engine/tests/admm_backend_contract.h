#pragma once

// ADMM backend equivalence checks, shared by qp_tests (CPU vs CPU, exercising
// the harness on every build) and qp_cuda_tests (CPU vs the hybrid CUDA
// backend). Both backends are fed the SAME host KKT solution every iteration,
// so any divergence is the backend's own arithmetic, never a drifting iterate.

#include "qp/admm_backend.h"
#include "qp/kkt_solver.h"
#include "qp/qp_model.h"

#include <cmath>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace qpcontract {

constexpr double kInf = std::numeric_limits<double>::infinity();

struct Fixture {
    std::string name;
    qp::QpModel model;
};

inline void check(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

inline bool close(double a, double b, double tolerance) {
    if (!std::isfinite(a) || !std::isfinite(b)) {
        return (std::isnan(a) && std::isnan(b)) || a == b;
    }
    return std::abs(a - b) <= tolerance * (1.0 + std::max(std::abs(a), std::abs(b)));
}

inline void requireClose(double a, double b, double tolerance, const std::string& what) {
    check(close(a, b, tolerance),
          what + ": reference " + std::to_string(a) + " vs candidate " + std::to_string(b));
}

inline void requireClose(const std::vector<double>& a, const std::vector<double>& b,
                         double tolerance, const std::string& what) {
    check(a.size() == b.size(), what + ": size " + std::to_string(a.size()) + " vs " +
                                    std::to_string(b.size()));
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (!close(a[i], b[i], tolerance)) {
            throw std::runtime_error(what + "[" + std::to_string(i) + "]: reference " +
                                     std::to_string(a[i]) + " vs candidate " + std::to_string(b[i]));
        }
    }
}

// Convex QP: diagonally dominant symmetric P in FULL storage (both (i,j) and
// (j,i)), so the off-diagonal convention is exercised; rows of every bound kind.
inline Fixture randomFixture(const std::string& name, int n, int m, int perRow, unsigned seed,
                             bool withHessian) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<double> value(-1.0, 1.0);
    std::uniform_int_distribution<int> column(0, std::max(n - 1, 0));
    std::vector<double> pr, pc, pv;
    if (withHessian) {
        for (int j = 0; j < n; ++j) {
            pr.push_back(j); pc.push_back(j); pv.push_back(2.0 + std::abs(value(generator)));
        }
        for (int j = 0; j + 1 < n; j += 2) {
            const double c = 0.4 * value(generator);
            pr.push_back(j); pc.push_back(j + 1); pv.push_back(c);
            pr.push_back(j + 1); pc.push_back(j); pv.push_back(c);
        }
    }
    std::vector<double> ar, ac, av;
    for (int i = 0; i < m; ++i) {
        for (int k = 0; k < perRow; ++k) {
            ar.push_back(i); ac.push_back(column(generator)); av.push_back(value(generator));
        }
    }
    Fixture fixture{name, {}};
    qp::QpModel& model = fixture.model;
    model.P = qp::SparseMatrix::fromTriplets(n, n, pr, pc, pv);
    model.A = qp::SparseMatrix::fromTriplets(m, n, ar, ac, av);
    model.q.resize(static_cast<std::size_t>(n));
    for (double& q : model.q) {
        q = value(generator);
    }
    model.l.resize(static_cast<std::size_t>(m));
    model.u.resize(static_cast<std::size_t>(m));
    for (int i = 0; i < m; ++i) {
        const auto k = static_cast<std::size_t>(i);
        switch (i % 4) {
            case 0: model.l[k] = 0.5;   model.u[k] = 0.5;  break;
            case 1: model.l[k] = -kInf; model.u[k] = 1.0;  break;
            case 2: model.l[k] = -1.0;  model.u[k] = kInf; break;
            default: model.l[k] = -2.0; model.u[k] = 2.0;  break;
        }
    }
    return fixture;
}

inline std::vector<Fixture> fixtures() {
    std::vector<Fixture> result;
    result.push_back(randomFixture("constrained", 120, 90, 4, 21u, true));
    result.push_back(randomFixture("sparseKkt", 700, 400, 3, 22u, true));
    result.push_back(randomFixture("lpNoHessian", 60, 50, 3, 23u, false));
    result.push_back(randomFixture("unconstrained", 40, 0, 0, 24u, true));
    result.push_back(randomFixture("emptyRowsAndColumns", 50, 40, 1, 25u, true));
    return result;
}

using BackendFactory = std::function<std::unique_ptr<qp::AdmmBackend>(const qp::QpModel&)>;

inline void compareBackends(const Fixture& fixture, const BackendFactory& makeReference,
                            const BackendFactory& makeCandidate, double tolerance) {
    const qp::QpModel& model = fixture.model;
    const std::string& name = fixture.name;
    auto reference = makeReference(model);
    auto candidate = makeCandidate(model);
    check(reference != nullptr && candidate != nullptr, name + ": backend creation failed");

    // restoreBest before any recordBest empties the iterate, like the CPU did.
    {
        auto a = makeReference(model);
        auto b = makeCandidate(model);
        a->restoreBest();
        b->restoreBest();
        check(a->primal().empty() && b->primal().empty() && a->dual().empty() && b->dual().empty(),
              name + ": restoreBest without a recorded best must empty the iterate");
    }

    double rho = 1.0;
    qp::KktSolver kkt(model, rho);
    check(kkt.isFactorValid(), name + ": KKT factorisation failed");

    for (int k = 0; k < 12; ++k) {
        if (k == 6) {
            rho = 4.0;  // exercise a rho change mid-run, as adaptive rho does
            check(kkt.refactor(rho), name + ": refactor failed");
        }
        reference->saveIterate();
        candidate->saveIterate();

        std::vector<double> rhsReference;
        std::vector<double> rhsCandidate;
        reference->buildRhs(rho, rhsReference);
        candidate->buildRhs(rho, rhsCandidate);
        requireClose(rhsReference, rhsCandidate, tolerance, name + " rhs@" + std::to_string(k));

        std::vector<double> solution = rhsReference;
        check(kkt.solve(solution), name + ": KKT solve failed");
        reference->acceptPrimal(std::vector<double>(solution), rho);
        candidate->acceptPrimal(std::vector<double>(solution), rho);

        const qp::AdmmIterationMetrics a = reference->metrics(rho);
        const qp::AdmmIterationMetrics b = candidate->metrics(rho);
        const std::string at = "@" + std::to_string(k);
        requireClose(a.primalResidualNorm, b.primalResidualNorm, tolerance, name + " primal residual" + at);
        requireClose(a.dualResidualNorm, b.dualResidualNorm, tolerance, name + " dual residual" + at);
        requireClose(a.objective, b.objective, tolerance, name + " objective" + at);

        if (k == 3) {
            reference->recordBest();
            candidate->recordBest();
        }
        if (k % 4 == 1) {
            const qp::AdmmCheckView va = reference->checkView();
            const qp::AdmmCheckView vb = candidate->checkView();
            requireClose(va.x, vb.x, 0.0, name + " x" + at);
            requireClose(va.xOld, vb.xOld, 0.0, name + " xOld" + at);
            requireClose(va.y, vb.y, tolerance, name + " y" + at);
            requireClose(va.yOld, vb.yOld, tolerance, name + " yOld" + at);
            requireClose(va.z, vb.z, tolerance, name + " z" + at);
            requireClose(va.Ax, vb.Ax, tolerance, name + " Ax" + at);
        }
    }

    reference->restoreBest();
    candidate->restoreBest();
    requireClose(reference->primal(), candidate->primal(), 0.0, name + " restored best x");
    requireClose(reference->dual(), candidate->dual(), tolerance, name + " restored best y");
}

}  // namespace qpcontract
