# QP CPU Engine v1

A self-contained C++17 ADMM engine for continuous convex quadratic programs in
the form

```text
minimize    0.5 x^T P x + q^T x
subject to  l <= A x <= u
```

where `P` is positive (semi-)definite. No third-party dependencies: sparse
storage, Ruiz equilibration, the KKT linear solve, the ADMM iteration kernel,
KKT polishing, and the termination tests are all built here from the
mathematical foundations.

It deliberately does not include an MPS parser. Integration happens through a
separate adapter (`qp_adapter.*`) that converts the repository's
`const model::Model&` into `qp::QpModel`.

## Components

| Component | File | Role |
|---|---|---|
| Types / status | `qp_types.*` | `QpStatus`, `AdmmOptions`, `AdmmResult`, `toString` |
| Model | `qp_model.*` | Immutable CSR sparse matrix + `QpModel` (P, A, q, l, u) |
| Equilibration | `scaling.*` | Ruiz row/column infinity-norm scaling, optional |
| KKT solver | `kkt_solver.*` | Dense Cholesky (n<=500) and left-looking sparse Cholesky (n>500) for `P + rho A^T A` |
| ADMM kernel | `admm_solver.*` | x / z / y updates, adaptive rho, Boyd-style termination |
| Polishing | `polishing.*` | Active-set KKT refinement of the converged iterate |
| Adapter | `qp_adapter.*` | `model::Model` -> `qp::QpModel` (max->min, quadratic terms) |
| Facade | `qp_solver.*` | `qp::QpSolver` is the only type most callers need |

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/qp_bench
```

`CMAKE_BUILD_TYPE` defaults to `Release` if unset, because an unoptimised
build of the sparse KKT factorisation is roughly an order of magnitude
slower and that is easy to mistake for the solver being slow.

Without CMake:

```bash
g++ -std=c++17 -O3 -pthread -I include src/*.cpp tests/test_qp_solver.cpp -o qp_tests
./qp_tests
```

## Algorithm

The engine solves the consensus form via ADMM (Boyd, Parikh, Chu, Peleato,
Eckstein 2011):

```text
x^{k+1} = argmin  0.5 x^T P x + q^T x + (rho/2) ||A x - z^k + y^k||^2
z^{k+1} = proj_{[l,u]} (A x^{k+1} + y^k)
y^{k+1} = y^k + rho (A x^{k+1} - z^{k+1})
```

The x-update requires solving the KKT system `K = P + rho A^T A`, which is
positive definite because `P` is. The solver factors `K` once with dense
Cholesky when `n <= 500` and with left-looking sparse Cholesky otherwise, then
back-substitutes on every iteration. When adaptive rho changes `rho`, the
factorisation is refactored.

Primal and dual residuals follow Boyd et al. 3.3.1:

```text
r   = A x - z                         (primal)
s   = -rho A^T (z - zOld)             (dual)
eps_pri  = sqrt(n + m) eps_abs + eps_rel max(||A x||, ||z||)
eps_dual = sqrt(n)     eps_abs + eps_rel ||A^T y||  (no rho here)
```

The termination check fires every `terminationCheckFrequency` iterations to
amortise the residual computation.

## KKT polishing

After the ADMM termination check passes (or hits the iteration limit), the
polisher forms a reduced KKT system for the active set (equalities plus
near-binding inequalities) and iterates the active set. Each iteration solves
a dense `(n + na) x (n + na)` linear system with partial-pivoting LU, then
re-identifies the active set. Two consecutive iterations with the same active
set indicate local convergence.

Polishing is optional and disabled with `AdmmOptions::usePolishing = false`.

## Known limitations

These are real and should be read before quoting this engine against a
commercial solver.

1. **Inequality-only constraint form.** Pure equalities `A x = b` are
   expressed as `l[i] = u[i] = b[i]` on a single row. There is no separate
   equality-only code path or specialised row format.

2. **Dense/sparse KKT threshold is at n = 500, untested beyond ~n = 1000.**
   The sparse Cholesky path is left-looking, single-threaded, and has not been
   stress-tested on the very large instances (n > 10,000) where fill-in
   matters. The threshold is a switch, not a tuned value, and assumes the
   caller knows their own matrix structure.

3. **No warm-starting.** Every call to `solve()` starts from zero. Reusing
   iterates across related problems (e.g. inside an SQP loop) would need
   explicit state transfer; the public `AdmmResult` does not currently expose
   the internal iterates.

4. **No constraint dual reporting when m = 0.** The unconstrained path
   correctly returns `primal` and skips the dual, but the KKT polishing
   system is not invoked and the result does not synthesise a dummy dual
   vector. The `constraintDual` field is empty in that case.

5. **MinGW `std::chrono` quirk on this host.** `steady_clock` and
   `QueryPerformanceCounter` on the MinGW 6.3 toolchain used to develop this
   engine both produce out-of-scale or negative durations, so the
   `solveTimeSeconds` field and the `qp_bench` harness report zero on
   sub-millisecond solves. On a standard Linux, MSVC, or modern MinGW build
   the timing is correct.

6. **`rho = 0` is rejected.** The KKT solver requires a positive penalty;
   `AdmmOptions::rho <= 0` is silently defaulted to `1.0` rather than
   reported as an error. Callers that want to opt out of rho entirely have no
   way to do so.

7. **Indefinite P is not guaranteed to converge.** If `P` is not positive
   semi-definite, the KKT factorisation can fail silently or produce garbage.
   The solver catches KKT factorisation failure and bumps `rho` as a recovery
   heuristic, but does not detect the indefiniteness itself and does not
   guarantee convergence on indefinite Hessians.

8. **CUDA backend is hybrid and unvalidated.** `QP_ENABLE_CUDA` builds a backend
   that moves the sparse products and vector updates to the GPU but keeps the
   KKT solve on the CPU, so each iteration transfers two n-vectors. `Auto` never
   selects it until a benchmark shows a crossover, and it has not yet been
   compiled or run on a GPU; see [docs/cuda.md](../docs/cuda.md).

## Integration boundary

The model adapter must:

1. convert maximisation to minimisation (flip the sign of `q` and `c0`);
2. expand quadratic objective terms `0.5 x^T Q x + q^T x` into the symmetric
   `P` matrix;
3. copy variable and row bounds;
4. convert `Constraint.linearTerms` into matrix triplets;
5. build `SparseMatrix::fromTriplets()`;
6. preserve original row/column index mappings for later postsolve.

Do not add MPS parsing logic to this module.
