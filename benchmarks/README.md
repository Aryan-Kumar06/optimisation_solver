# Benchmark pipeline

Runs a solver on an instance in an isolated process, then verifies the answer
against an independent model of the problem. The solver is never asked whether
it was right.

## Stages

```
instance.mps
   |
   |-- 1. parse cross-check    our reader (--dump-model) vs benchmarks/lib/mps_model.py
   |                           compared field by field: sense, offset, bounds,
   |                           ranged rows, integrality, quadratic coefficients
   |
   |-- 2. isolated solve       bench_runner: fork, own process group, wall-clock
   |                           watchdog, SIGTERM then SIGKILL, peak RSS from wait4
   |
   |-- 3. independent check    benchmarks/lib/verify.py against the ORIGINAL model,
   |                           before presolve and before any scaling
   |
   `-- 4. record               one JSON row: everything above, nulls for anything
                               genuinely unavailable
```

Every solver goes through the same `bench_runner`, ours and the reference
alike, so wall clock and peak memory are measured by one mechanism rather than
self-reported by each adapter.

## Frozen tolerances

**These are frozen. Do not change them for a scored run.** Changing a tolerance
changes what "verified" means, and a comparison across runs with different
tolerances is not a comparison.

| Quantity | Value |
|---|---|
| feasibility, absolute | `1e-6` |
| feasibility, relative | `1e-8` |
| integrality, absolute | `1e-6` |
| optimality, normalised | `1e-6` |

### Scaling, stated explicitly

"Relative to what" is where these comparisons usually go wrong, so the
denominators are written down rather than left to a library default.

Row *i* is satisfied when its violation is at most

```
tol_i = 1e-6 + 1e-8 * s_i
s_i   = max(1, |l_i|, |u_i|, sum_j |a_ij * x_j|)
```

The last term is the row's own activity magnitude, so a row summing a million
large terms is not held to the same absolute residual as a row of two small
ones.

Variable *j* is satisfied when its bound violation is at most

```
tol_j = 1e-6 + 1e-8 * max(1, |lb_j|, |ub_j|, |x_j|)
```

Optimality uses

```
gap_norm = |p - d| / (1 + |p| + |d|)
```

with *p* the primal objective and *d* the dual objective, both in minimisation
form.

**Absolute residuals are reported alongside every normalised one**, so a reader
can apply a different rule without re-running anything.

## Termination status and checker verdict are separate

They are different questions and are never merged into one "pass".

| Solver status | what the solver claims |
|---|---|
| `optimal`, `infeasible`, `unbounded`, `limit_reached`, … | the solver's own termination |

| Checker verdict | what was independently established |
|---|---|
| `optimal_verified` | feasible **and** KKT/gap closed within tolerance |
| `feasible` | the point satisfies the model; optimality **not** proven |
| `infeasible_point` | the returned point violates the original model |
| `nonfinite` | NaN or infinity in the point |
| `malformed` | wrong length or missing fields |
| `no_point` | nothing to check (crash, timeout, no output) |
| `not_applicable` | solver claimed infeasible/unbounded/unsupported |

Two rules follow from this, and both are enforced in code and asserted in
`test_pipeline.py`:

- **A feasible point is not an optimality proof.** Without duals, the best
  available verdict is `feasible`. `duals_unavailable_reason` records why.
- **Agreement with a best-known objective is not an optimality proof.** It is
  corroboration; the reference value is an external claim, and matching it
  cannot distinguish a true optimum from a coincidence. Agreement is recorded
  and labelled, and never promotes a verdict.

Infeasibility and unboundedness claims are **not** independently verified.
Doing so needs a Farkas ray or an improving ray, which is a separate check this
pipeline does not attempt. Those rows are `not_applicable`, not "correct".

## Missing values are null

A value that does not exist is `null`, never `0.0` and never an empty vector.
"Branch-and-cut produced no duals" and "the duals are all zero" are different
facts and a checker must be able to tell them apart.

## The reference solver is a reference, not a dependency

HiGHS is reached through `scipy.optimize` and lives entirely under
`benchmarks/adapters/`. It is invoked as a separate process. Nothing under
`src/`, `include/`, `cli/` or the engine directories refers to it; delete
`benchmarks/` and the solver builds and runs unchanged.
`test_pipeline.py::test_reference_solver_is_not_a_dependency` asserts this by
scanning for includes and link directives and by checking the built binary's
dynamic libraries, rather than leaving it to convention.

### Verified reference capabilities

| Capability | Available | Note |
|---|---|---|
| LP, dual simplex (`highs-ds`) | yes | with duals and reduced costs |
| LP, interior point (`highs-ipm`) | yes | |
| MILP (`scipy.optimize.milp`) | yes | with `mip_gap` and `mip_dual_bound`; no duals |
| QP | **no** | HiGHS supports QP but scipy exposes no entry point |
| reads MPS directly | **no** | fed from `benchmarks/lib/mps_model.py` |

The last row is useful rather than inconvenient: the reference is driven from
the independent reader, so a disagreement between our solver and HiGHS also
catches a parsing disagreement.

## Independence, and its limit

`benchmarks/lib/mps_model.py` shares no code with `src/mps/mps_reader.cpp` and
differs structurally on purpose. But **both were written by the same author**,
so correlated blind spots are possible. Parse agreement is necessary, not
sufficient. The check that does not share an author is agreement with HiGHS on
the objective value.

## Usage

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release -j8

# self-test first: proves the checker rejects wrong answers
python3 benchmarks/test_pipeline.py

python3 benchmarks/bench.py benchmarks/instances/known/*.mps \
    --solvers auto,highs --timeout 60 \
    --best-known benchmarks/instances/known_answers.json \
    --out benchmarks/results/known.json

python3 benchmarks/bench.py benchmarks/instances/netlib/afiro.mps \
    --solvers dual_simplex,pdlp,highs --timeout 60 \
    --best-known benchmarks/instances/netlib/best_known.json \
    --out benchmarks/results/netlib_afiro.json
```

`benchmarks/results/baseline.json` records the commit, build settings and
platform the reference numbers were taken on.

## Instances

`instances/known/` are hand-written with optima derived by hand, covering
optimal, infeasible, unbounded, degenerate, ranged, maximisation with an
objective constant, integer markers, convex quadratic, free/fixed/MI bounds,
and an unsupported MIQP.

`instances/netlib/` carries provenance and the decompression step Netlib
requires — see `instances/netlib/PROVENANCE.md`. Netlib does not distribute
plain MPS.

## Accuracy review and comparable timing

The MILP result now includes a global dual bound over queued, active and failed
subtrees. Completed searches with an incumbent have zero gap; interrupted
searches retain the frontier bound. Missing bounds and gaps remain null.
`termination.reason` distinguishes time, node and iteration limits from LP
failures. A generic `limit_reached` without a reason is not assumed to be a
timeout. Summary medians use the arithmetic mean of the two middle values for
even-sized samples.

CTest supplies `OPTIMSOLVER_BINARY` to the pipeline self-test so it tests the
binary from that build directory, rather than a potentially stale Release
binary. The no-incumbent regression uses the bundled knapsack with an expired
root deadline; it does not depend on a fetched MIPLIB file or machine speed.

Preserve `*_PREFIX.json` as historical measurements. New accuracy-review runs
use separate filenames. Compare identical instance files, builds, thread counts
and time budgets; do not compare a short smoke-run bound with a longer
full-development-run bound. Reference difference measures solution quality,
not a solver-proven gap. Node counts alone are not a measure of useful search.
