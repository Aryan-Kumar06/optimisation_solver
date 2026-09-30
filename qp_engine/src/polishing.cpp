#include "qp/polishing.h"

#include <chrono>

#include "qp/kkt_solver.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>

namespace qp {

namespace {

// Build a reduced KKT system for an active set.
// The active set consists of constraint indices where the bound is active.
// We solve: [P  A_a^T; A_a  0] [x; y_a] = [-q; b_a]
// where A_a is the submatrix of A formed by the active rows,
// b_a = A_a * x (the bound values).
//
// The reduced system is built using dense Cholesky when the active set
// is small, and sparse Cholesky otherwise.
// (row, boundValue): which row is active, and the SIDE it is active on --
// lo for a row pinned at its lower bound, hi for one pinned at its upper
// bound, either (they are ~equal) for a genuine equality row. The value is
// decided by the caller, which already knows which side triggered activity;
// this function must not re-derive it, since the wrong assumption here is
// exactly the bug this replaced (see the comment below).
struct ActiveRow {
    int row;
    double bound;
};

bool solveActiveKKT(
    const QpModel& model,
    const std::vector<ActiveRow>& activeIdx,
    const std::vector<double>& x,
    std::vector<double>& newX,
    std::vector<double>& newY
) {
    const int n = model.numVariables();
    const int na = static_cast<int>(activeIdx.size());
    if (na == 0) {
        // Unconstrained: solve P x = -q directly using a dense KKT solver.
        QpModel reduced;
        reduced.P = model.P;
        reduced.A = SparseMatrix::fromTriplets(0, n, {}, {}, {});
        reduced.q = model.q;
        reduced.l = {}; reduced.u = {};
        KktSolver kkt(reduced, 1.0);
        newX = x;
        if (!kkt.solve(newX)) return false;
        for (double v : newX) if (!std::isfinite(v)) return false;
        return true;
    }

    // Build the dense KKT matrix [P  A_a^T; A_a  0], size (n+na) x (n+na).
    const int N = n + na;
    std::vector<double> K(N * N, 0.0);

    // P into the top-left block.
    {
        const auto& pStart = model.P.csrRowStart();
        const auto& pCols  = model.P.csrColumnIndex();
        const auto& pVals  = model.P.csrValues();
        for (int r = 0; r < n; ++r) {
            const auto begin = static_cast<std::size_t>(pStart[static_cast<std::size_t>(r)]);
            const auto end   = static_cast<std::size_t>(pStart[static_cast<std::size_t>(r) + 1]);
            for (std::size_t k = begin; k < end; ++k) {
                const Index c = pCols[k];
                K[static_cast<std::size_t>(r) + static_cast<std::size_t>(c) * static_cast<std::size_t>(N)] += pVals[k];
                if (r != c) {
                    K[static_cast<std::size_t>(c) + static_cast<std::size_t>(r) * static_cast<std::size_t>(N)] += pVals[k];
                }
            }
        }
    }

    // A_a^T into the top-right block (column n + i has A_a[:,i]^T).
    // A_a[j] = row activeIdx[j].row of A.
    std::vector<double> b(na);
    {
        const auto& aStart = model.A.csrRowStart();
        const auto& aCols  = model.A.csrColumnIndex();
        const auto& aVals  = model.A.csrValues();
        for (int j = 0; j < na; ++j) {
            const int row = activeIdx[static_cast<std::size_t>(j)].row;
            // The bound value for THIS row's side, from the caller -- not
            // re-derived here. An earlier version read model.u[row]
            // unconditionally on the assumption every active row was an
            // equality (l == u). For a one-sided row active at its LOWER
            // bound with an unbounded upper side (the common case: most
            // variable-bound rows are one-sided), that read +infinity
            // straight into this system's right-hand side, and the dense
            // solve below propagated it into inf/nan -- accepted anyway,
            // because none of its pivot checks test for non-finite output.
            const double bound = activeIdx[static_cast<std::size_t>(j)].bound;
            const auto begin = static_cast<std::size_t>(aStart[static_cast<std::size_t>(row)]);
            const auto end   = static_cast<std::size_t>(aStart[static_cast<std::size_t>(row) + 1]);
            for (std::size_t k = begin; k < end; ++k) {
                const Index c = aCols[k];
                const double aik = aVals[k];
                K[static_cast<std::size_t>(c) + static_cast<std::size_t>(n + j) * static_cast<std::size_t>(N)] += aik;
                K[static_cast<std::size_t>(n + j) + static_cast<std::size_t>(c) * static_cast<std::size_t>(N)] += aik;
            }
            b[static_cast<std::size_t>(j)] = bound;
        }
    }

    // Build the RHS: [-q; b - A_a * x] = [-q; 0] (since b = A_a * x for optimality).
    std::vector<double> rhs(N);
    for (int j = 0; j < n; ++j)
        rhs[static_cast<std::size_t>(j)] = -model.q[static_cast<std::size_t>(j)];
    for (int j = 0; j < na; ++j)
        rhs[static_cast<std::size_t>(n + j)] = b[static_cast<std::size_t>(j)];

    // Solve the dense KKT system using Cholesky on the Schur complement.
    // We solve [P  A^T; A  0] [x; y] = [r1; r2]
    // Using block elimination: solve P x + A^T y = r1  =>  x = P^{-1} (r1 - A^T y)
    // Then A x = r2  =>  A P^{-1} (r1 - A^T y) = r2
    //   =>  (A P^{-1} A^T) y = A P^{-1} r1 - r2
    // For P = I (diagonal): P^{-1} is trivial.
    // We use a direct dense LU/Cholesky approach for simplicity.
    // The KKT matrix [P A^T; A 0] is indefinite; we use dense LU.

    // Add a small regularisation to make the KKT matrix better conditioned.
    // We add to the (n,n) block and to the (na,na) block. For the constraint
    // block the Schur complement is P_reg which is PD if P_reg is PD.
    const double reg = 1e-8;
    for (int j = 0; j < n; ++j)
        K[static_cast<std::size_t>(j) + static_cast<std::size_t>(j) * static_cast<std::size_t>(N)] += reg;
    // Regularise the constraint block to make [P A^T; A 0] nonsingular.
    for (int j = 0; j < na; ++j)
        K[static_cast<std::size_t>(n + j) + static_cast<std::size_t>(n + j) * static_cast<std::size_t>(N)] += reg;

    // Dense LU decomposition (Gaussian elimination with partial pivoting for
    // the KKT). The KKT matrix [P A^T; A 0] is indefinite, so we use LU
    // rather than Cholesky.
    bool ok = false;
    {
        std::vector<double> Acopy = K;
        std::vector<double> bcopy = rhs;
        // Forward elimination + back substitution.
        std::vector<double> y(N);
        for (int col = 0; col < N; ++col) {
            // Partial pivoting: find the largest element in this column below diagonal.
            double maxAbs = 0.0;
            int pivotRow = col;
            for (int r = col; r < N; ++r) {
                const double v = std::abs(Acopy[static_cast<std::size_t>(r) + static_cast<std::size_t>(col) * static_cast<std::size_t>(N)]);
                if (v > maxAbs) { maxAbs = v; pivotRow = r; }
            }
            if (maxAbs < 1e-12) {
                // Singular, bail out.
                return false;
            }
            // Swap rows if needed.
            if (pivotRow != col) {
                for (int c = 0; c < N; ++c) {
                    std::swap(Acopy[static_cast<std::size_t>(col) + static_cast<std::size_t>(c) * static_cast<std::size_t>(N)],
                             Acopy[static_cast<std::size_t>(pivotRow) + static_cast<std::size_t>(c) * static_cast<std::size_t>(N)]);
                }
                std::swap(bcopy[static_cast<std::size_t>(col)], bcopy[static_cast<std::size_t>(pivotRow)]);
            }
            const double piv = Acopy[static_cast<std::size_t>(col) + static_cast<std::size_t>(col) * static_cast<std::size_t>(N)];
            if (std::abs(piv) < 1e-12) return false;
            for (int r = col + 1; r < N; ++r) {
                const double factor = Acopy[static_cast<std::size_t>(r) + static_cast<std::size_t>(col) * static_cast<std::size_t>(N)] / piv;
                for (int c = col + 1; c < N; ++c) {
                    Acopy[static_cast<std::size_t>(r) + static_cast<std::size_t>(c) * static_cast<std::size_t>(N)] -=
                        factor * Acopy[static_cast<std::size_t>(col) + static_cast<std::size_t>(c) * static_cast<std::size_t>(N)];
                }
                bcopy[static_cast<std::size_t>(r)] -= factor * bcopy[static_cast<std::size_t>(col)];
                Acopy[static_cast<std::size_t>(r) + static_cast<std::size_t>(col) * static_cast<std::size_t>(N)] = 0.0;
            }
        }

        // Back substitution.
        ok = true;
        for (int i = N - 1; i >= 0; --i) {
            double sum = bcopy[static_cast<std::size_t>(i)];
            for (int j = i + 1; j < N; ++j) {
                sum -= Acopy[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * static_cast<std::size_t>(N)] *
                       y[static_cast<std::size_t>(j)];
            }
            const double diag = Acopy[static_cast<std::size_t>(i) + static_cast<std::size_t>(i) * static_cast<std::size_t>(N)];
            if (std::abs(diag) < 1e-12) { ok = false; break; }
            y[static_cast<std::size_t>(i)] = sum / diag;
        }

        if (ok) {
            newX.resize(static_cast<std::size_t>(n));
            newY.resize(static_cast<std::size_t>(na));
            for (int j = 0; j < n; ++j) newX[static_cast<std::size_t>(j)] = y[static_cast<std::size_t>(j)];
            for (int j = 0; j < na; ++j) newY[static_cast<std::size_t>(j)] = y[static_cast<std::size_t>(n + j)];

            // The pivot checks above only guard against a small denominator;
            // they do not guard against a bad NUMERATOR (an infinite entry in
            // b, from the caller passing the wrong bound -- the exact defect
            // this file used to have). inf/nan divides cleanly by a healthy
            // pivot and comes out the other end still inf/nan, so a solve can
            // report ok=true while newX/newY are garbage. Catch it here
            // rather than trust every caller to re-check: this function
            // returning true must mean the numbers are usable.
            for (double v : newX) if (!std::isfinite(v)) { ok = false; break; }
            if (ok) for (double v : newY) if (!std::isfinite(v)) { ok = false; break; }
        }
    }

    return ok;
}

}  // namespace

bool KktPolisher::polish(
    const QpModel& model,
    std::vector<double>& x,
    std::vector<double>& constraintDual,
    const Options& options
) {
    const int n = model.numVariables();
    const int m = model.numConstraints();

    if (m == 0 || n == 0) return true;  // Nothing to polish.
    if (options.maxIterations <= 0) return true;

    std::vector<double> curX = x;
    std::vector<double> curDual = constraintDual;
    std::vector<int> prevActiveRows;
    const auto started = std::chrono::steady_clock::now();

    for (int iter = 0; iter < options.maxIterations; ++iter) {
        // Identify active constraints, and which side of each is active --
        // the two are not the same row value in general (see ActiveRow).
        std::vector<ActiveRow> activeIdx;
        for (int i = 0; i < m; ++i) {
            // Compute A_i * x.
            double Aix = 0.0;
            const auto& aStart = model.A.csrRowStart();
            const auto& aCols  = model.A.csrColumnIndex();
            const auto& aVals  = model.A.csrValues();
            const auto begin = static_cast<std::size_t>(aStart[static_cast<std::size_t>(i)]);
            const auto end   = static_cast<std::size_t>(aStart[static_cast<std::size_t>(i) + 1]);
            for (std::size_t k = begin; k < end; ++k) {
                Aix += aVals[k] * curX[static_cast<std::size_t>(aCols[k])];
            }
            const double lo = model.l[static_cast<std::size_t>(i)];
            const double hi = model.u[static_cast<std::size_t>(i)];
            const double tol = options.activeSetTolerance;
            const double gap = hi - lo;
            if (gap < tol) {
                // Equality constraint (or a numerically tight box): lo and hi
                // agree to within tol, so either serves as the bound value.
                activeIdx.push_back({i, lo});
            } else if (Aix <= lo + tol) {
                activeIdx.push_back({i, lo});
            } else if (Aix >= hi - tol) {
                activeIdx.push_back({i, hi});
            }
        }

        // Check convergence: same set of ACTIVE ROWS as last iteration. The
        // bound value each carries is a deterministic function of (row,
        // which side fired) and is not part of this comparison, same as the
        // original int-keyed version.
        std::vector<int> activeRows;
        activeRows.reserve(activeIdx.size());
        for (const auto& a : activeIdx) activeRows.push_back(a.row);
        if (activeRows == prevActiveRows) {
            break;  // Active set unchanged, converged.
        }
        prevActiveRows = activeRows;

        // Give up, leaving x untouched, rather than exceed the budget. The
        // caller keeps the converged ADMM point, which was already verified.
        if (n + static_cast<int>(activeIdx.size()) > options.maximumDenseDimension) return false;
        if (options.timeLimitSeconds > 0.0 &&
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() >=
                options.timeLimitSeconds)
            return false;

        // Solve the reduced KKT.
        std::vector<double> newX;
        std::vector<double> newY;
        if (!solveActiveKKT(model, activeIdx, curX, newX, newY)) {
            return false;  // Numerical failure.
        }

        // Check convergence.
        double maxDiff = 0.0;
        for (int j = 0; j < n; ++j) {
            maxDiff = std::max(maxDiff, std::abs(newX[static_cast<std::size_t>(j)] - curX[static_cast<std::size_t>(j)]));
        }
        curX = newX;

        if (maxDiff < options.tolerance) break;
    }

    x = std::move(curX);
    return true;
}

bool KktPolisher::polish(
    const QpModel& model,
    std::vector<double>& x,
    std::vector<double>& constraintDual
) {
    return polish(model, x, constraintDual, Options{});
}

}  // namespace qp
