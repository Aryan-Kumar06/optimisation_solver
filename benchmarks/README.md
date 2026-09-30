# Benchmark pipeline

Runs a solver on an instance in an isolated process, then verifies the answer
against an independent model of the problem. The solver is never asked whether
it was right.

## Published validation results — 2026-09-30

Validated on 2026-09-30 in the SIH checkout, based on PR #15 revision
`08c1520ae033e56ad0a86c6794f2ee3484965b46`, with the local test/harness changes.
Production numerical engines were not changed for this coverage work.

### Automated tests

The final Release integration build against upstream `b67250b` ran **72 CTests:
71 passed, 1 failed**, with assertions enabled in test targets. The earlier
PR #15 baseline ran 70 CTests (69 passed, the same one failed). This includes the restored NLP engine, elastic KKT,
public pipeline and CLI tests, NLP/SLSQP comparisons, PDLP and QP engine unit
tests, parser/presolve/postsolve/MILP tests, benchmark harness tests, and the
new reference/integrity tests.

The failing test is `qp_reference`: **299/300** randomized QPs were compared or
had their nonoptimal status corroborated by OSQP. Zero-based case **219**, seed
**20260908**, returned `limit_reached` while OSQP returned `solved`. This is a
convergence limitation, not evidence of an incorrect optimality claim. The test
is deliberately still failing; it is not skipped or marked `WILL_FAIL`.

The new Python suites contain eight offline tests of selection, checksums,
decoding and coverage accounting, plus six reference-adapter tests of sparse
row translation, dual signs, offsets, min/max QPs, infeasibility, unboundedness,
nonconvex refusal and reference-error propagation. Both suites pass. After these additions, the affected harness tests were rerun successfully.

External packages: NumPy 2.3.2, SciPy 1.17.1, OSQP 1.1.3. Platform: macOS 15.3.1,
ARM64. External reference tests are optional, separate from production dependencies.

### Public benchmark smoke run

Frozen selections were run at **five seconds per solver process**, one thread
requested for OptimSolver. Reference startup is included in that budget.
Reference native thread counts are not controlled by this harness; this is a
correctness/coverage smoke run, not a performance comparison. Parsers are
cross-checked independently, and solutions are checked on the original model.

| Dataset | Selected | Objective agrees | Other outcomes |
|---|---:|---:|---|
| Netlib LP | 8 | 6 | 2 feasible but solver did not report optimal |
| MIPLIB 2017 Collection subset | 25 | 3 | 15 reference not optimal, 5 unverified, 2 feasible but solver not optimal |
| Maros–Mészáros QP | 14 | 9 | 3 objective disagreements, 2 unverified |
| Public Mittelmann LP subset | 3 | 0 | 3 unverified due to time limits |

The Netlib and QP selections were rerun after merging upstream `b67250b`;
their outcome counts were unchanged. MIPLIB and Mittelmann counts below remain
the recorded PR #15 baseline measurements.

All 50 selected instances reached the harness. After fixing nested compression
for `Linf_520c`, every input passed the independent parse cross-check. An
`objective_agrees` result requires validated feasible points, optimal statuses
from both solvers, the requested QP engine actually executing, and relative
objective agreement at 1e-6. It is **not** an independent optimality certificate;
JSON retains the separate KKT/gap verdict and residuals. Unverified runs are
never counted as successful comparisons. Infeasible/unbounded claims require
separate validation and are not inferred from timeout.

QP discrepancies, reproducible through `corpus_qp` or `run_suites.py --suite qp`:

| Instance | OptimSolver objective | OSQP objective | Observation |
|---|---:|---:|---|
| genhs28 | 0.928286870219662 | 0.927173693766391 | Solver reports optimal, original-model dual check rejects multipliers |
| hs51 | 0.000155936442925 | approximately 0 | Same |
| hs52 | 5.326683495975829 | 5.326647564469914 | Same |
| cvxqp3s | unavailable | 11943.432201967 | Solver reaches 5000 iterations without a validated feasible point |
| hs118 | unavailable | 664.82045 | Same |

OSQP solutions for these cases pass the independent primal/KKT checks and
agree with the published, rounded Maros–Mészáros values. The equalities-only
mismatch cases expose the existing QP termination path: its ADMM stopping test
uses auxiliary-variable movement rather than full stationarity. This is a
suspected cause requiring a numerical-engine fix and broader regression review;
this test addition does not change solver tolerances or mask these failures.

Netlib `blend` and `share2b` return feasible points with `limit_reached`.
The MIPLIB and Mittelmann five-second runs are insufficient to establish full
solution coverage. All three Mittelmann solver processes hit their watchdog;
HiGHS also reached a time limit or its watchdog. Longer runs can be requested
without changing the frozen selection.

See [coverage and provenance](COVERAGE.md) for dataset sources and scope, and
[reproduction commands](#reproducible-suite-coverage) below.

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

## Reproducible suite coverage

See [COVERAGE.md](COVERAGE.md) for the frozen selections, results and known
failures. Root CTest now includes PDLP and QP engine unit tests by default,
restored NLP tests, and offline integrity/accounting tests. Independent OSQP
and SciPy comparisons are opt-in; they are not production dependencies.

```sh
python3 -m pip install numpy scipy 'osqp>=1.0,<2'
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DQP_REFERENCE_TESTS=ON -DNLP_REFERENCE_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure

python3 benchmarks/fetch_netlib.py --set smoke
python3 benchmarks/fetch_miplib.py FROZEN_DEV25.json
python3 benchmarks/fetch_suites.py --suite qp
python3 benchmarks/fetch_suites.py --suite mittelmann
python3 benchmarks/run_suites.py --suite all --binary build/optimsolver --runner build/bench_runner --timeout 60 --out /tmp/coverage.json
```

`fetch_suites.py --offline` uses only cached inputs. New manifests pin archive
and decoded-file SHA256; corrupt caches fail rather than silently redownload.
`--suite qp --full` fetches all 138 Maros–Mészáros instances; run these with
`run_suites.py --suite qp --full-qp`. No dataset download happens during CTest.
QP and Mittelmann inputs remain ignored/generated files.

The suite runner records each selected instance, parse disagreements, both
solver statuses, independent residuals and objective comparisons. Missing data,
reference errors and timeouts cannot count as passes, and cause a nonzero exit.
Only two validated points with optimal statuses, matching objectives and an
agreed parse earn `objective_agrees`; that label is **not** an optimality proof.
JSON retains the independent checker's separate `optimal_verified` verdict.
Published rounded QP objectives are supplementary comparisons, not certificates.
All solver processes use the existing watchdog, with a separate bounded parse
cross-check. Output includes binary hash, revision, working-tree status,
package versions, platform and time budget. Budgets are per process and include
reference startup; these runs are not published Mittelmann performance results.

To register all four cached suites with CTest, configure with
`-DBENCHMARK_CORPUS_TESTS=ON -DBENCHMARK_CASE_SECONDS=10`, then run
`ctest --test-dir build -L benchmark --output-on-failure`. Reports go under the
ignored build directory. These are strict conformance/coverage checks and
currently expose documented solver limitations; enabling them does not promise
a green run. The regular unit tests need neither these data nor network access.
The `Linf_520c` download is nested bzip2 + Netlib `emps`, not plain MPS after
bzip2 decompression; its pinned decoder is compiled only during explicit fetch.

The suite runner explicitly selects the SciPy HiGHS adapter to preserve the
published reference protocol. Direct `bench.py` runs keep upstream’s default
of preferring native HiGHS; use `--highs-backend scipy` to pin the Python route.
