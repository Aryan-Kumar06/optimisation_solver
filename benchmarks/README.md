# Benchmark pipeline

Runs a solver on an instance in an isolated process, then verifies the answer
against an independent model of the problem. The solver is never asked whether
it was right.

## Published validation results — 2026-09-30

Validated on 2026-09-30 in the SIH checkout, based on PR #15 revision
`08c1520ae033e56ad0a86c6794f2ee3484965b46`, with the local test/harness changes.
The coverage work and two reviews exposed defects in the ADMM QP engine, which
this PR fixes (see *QP engine fixes* below). No other numerical engine was changed.

### Automated tests

The final Release integration build against upstream `b67250b` ran **73 CTests:
73 passed**, with assertions enabled in test targets. This includes the restored NLP engine, elastic KKT,
public pipeline and CLI tests, NLP/SLSQP comparisons, PDLP and QP engine unit
tests, parser/presolve/postsolve/MILP tests, benchmark harness tests, and the
new reference/integrity tests.

`qp_reference` passes with its contract unchanged: all **300/300** randomized
QPs (seed **20260908**) are accounted for. 279 optimal results agree with OSQP on
objective (relative 1e-4) and solution (absolute 1e-3), and 21
infeasible/unbounded statuses are corroborated. Case **219**, which previously
returned `limit_reached`, is among the 279. It failed because of an engine
defect, not a test-contract problem; nothing was skipped or marked `WILL_FAIL`.
OSQP's C layer printed "Polishing not needed" on stdout even with
`verbose=False`, which made the comparator's JSON report unparseable when a seed
was replayed; that output is now suppressed at the file-descriptor level.

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
| Maros–Mészáros QP | 14 | 13 | 1 unverified (cvxqp3s, iteration limit) |
| Public Mittelmann LP subset | 3 | 0 | 3 unverified due to time limits |

The Netlib and QP selections were rerun with the fixed QP engine
(`run_suites.py --suite qp|netlib --timeout 5`). Netlib is unchanged, since the
fix does not touch the LP engines; QP rose from 9 to 13 agreements. MIPLIB and
Mittelmann counts below remain the recorded PR #15 baseline measurements; their
engines were not changed.

All 50 selected instances reached the harness. After fixing nested compression
for `Linf_520c`, every input passed the independent parse cross-check. An
`objective_agrees` result requires validated feasible points, optimal statuses
from both solvers, the requested QP engine actually executing, and relative
objective agreement at 1e-6. It is **not** an independent optimality certificate;
JSON retains the separate KKT/gap verdict and residuals. Unverified runs are
never counted as successful comparisons. Infeasible/unbounded claims require
separate validation and are not inferred from timeout.

QP engine fixes. The five smoke-set discrepancies previously listed here, and
the failing `qp_reference` case 219, traced to four defects in the ADMM engine,
all reproduced on individual instances before being changed:

| Defect | Symptom | Instances |
|---|---|---|
| Adaptive rho re-estimated every iteration, never frozen | rho and the iterate in a limit cycle; the iterate was bit-identical after 5,000 and 200,000 iterations | reference case 219, hs118 |
| rho unbounded | rho = 2.8e14 after 50 iterations | hs51, hs52, genhs28 |
| Termination tested the proxy dual residual `rho * A'(z - zOld)` | once rho froze z, the proxy was zero and the engine reported **Optimal** with a true stationarity residual of 0.04-0.13 against a 1e-8 tolerance | hs51, hs52, genhs28 |
| A failed KKT refactorisation left rho changed but no valid factor | x never moved again | cvxqp3s (old run) |

Case 219 is well posed (a strictly concave maximisation, Hessian condition
~276). Presolve derives a valid, inactive bound for one variable from a row,
and with that bound the old rho rule cycled; without it the same engine
converged. Neither the tolerance, the objective/sign convention nor the OSQP
adapter was at fault.

The engine now follows OSQP (Stellato et al. 2020): termination uses the true
stationarity residual `P x + q + A'y`; rho is re-estimated only at termination
checks from the normalised residual ratio, clamped to [1e-6, 1e6], limited to a
factor of 10 per update, and changed at most 50 times, so it is eventually
constant as Boyd et al. section 3.4.1 requires for convergence. A zero residual
is floored rather than skipped, and a failed refactorisation restores the last
working factor. The damping and the zero-residual handling were each added
because the undamped rule oscillated on the NLP engine's elastic QPs and the
skip left hs268 stuck; both cases are now regression tests. KKT polishing,
which factors a dense system on each pass, now respects the remaining time
limit and skips systems above dimension 2,000: on presolved qship08s it had run
for more than 300 s past a 55 s limit.

Five QP engine regression tests run in the normal suite with production
settings, which the previous ADMM tests never used. Each of the four behavioural
tests was shown to fail against the engine it guards against.

Across all 138 Maros-Meszaros instances (public pipeline, forced ADMM, 60 s
wall clock per instance, objective checked against the published value at
1e-6 normalised), the fixed engine agrees on **45** against **29** before, with
**no regressions** and **no false optimal claims** (two before). Timeouts fell
from 35 to 27, mostly instances that converged and then hung in polishing.

`cvxqp3s` remains `unverified` at the smoke budget. The engine converges to the
published objective given 16,700 iterations, but not within the default 5,000.
The cause is structural: a single scalar rho for every row, where OSQP scales
rho by 1e3 on equality rows, and cvxqp3s has 75 of them. Per-row rho changes the
KKT assembly in both the dense and sparse paths and is left as separate work.

A second review found that several result paths could still report success
without the returned point satisfying the model. Each was reproduced first:

| Defect | Reproduction | Now |
|---|---|---|
| Termination judged on Ruiz-**scaled** residuals, while every consumer (postsolve, MIQP nodes, NLP subproblems) checks the original problem | 31 of 400 randomly generated QPs returned Optimal yet failed an original-units KKT check at the requested tolerance, rising to 121 of 375 with coefficients spanning 1e-4 to 1e4; presolved qship08s returned Optimal with a variable bound violated | the loop terminates on `qp::checkKkt` in original units (per-row relative primal, relative dual), as OSQP does by default; `QpSolver` re-applies it after polishing and returns `NumericalFailure` if it fails |
| Non-finite iterates were not detected | finite data whose iterates overflow ran all 5,000 iterations and returned `IterationLimit` with a NaN vector; on a bounded problem whose optimum (~-1e600) a double cannot hold, the overflowing iterate difference was certified **Unbounded** | stops at the next check with `NumericalFailure`, before the certificates can misread it |
| No variables: `Optimal` returned without checking the rows | a row requiring 0 in [1, 2] came back Optimal with a reported primal residual of 1.0 | each row must admit 0, otherwise `Infeasible` |
| No convexity guard in the engine | `min -x^2/2` over [-1, 1] returned Optimal at x = 0, the maximum | a negative diagonal of P is rejected as `InvalidProblem` (an O(nnz) necessary condition; the pipeline's full `checkConvexity` still runs before dispatch) |
| Time limit could adopt an empty best iterate | defensive | only a recorded iterate is adopted |

`qp_tests` now has nine regression cases for these and the earlier defects, and
the new `qp_contract` CTest pins the pipeline contract: a converged QP arrives
with validated primal and duals, a time-limited one as `limit_reached` without
multipliers, and a numerical breakdown as `numerical_failure` with no point.
Every behavioural engine test was shown to fail against the engine it guards
against. `qp_contract` also passes on the previous engine, because postsolve's
validation already caught those cases end to end; it documents the pipeline
contract, and the engine tests bind the fixes.

One NLP test depended on the old inaccuracy: its badly conditioned QP used to
come back Optimal with an absolute residual of 1.6e-5, and the NLP gate rejected
it. The engine now solves that QP accurately, so the test uses a right-hand side
of 1e9 instead: ADMM converges relatively while the absolute residual (~2e-6)
still exceeds the NLP gate. Removing the gate's residual check still makes that
case report `FirstOrderStationary`.

Pre-existing issue on `main`, not changed here: presolve deletes quadratic
objective terms with an absolute coefficient of at most 1e-9
(`src/presolve/presolver.cpp`), whatever the problem's scale. On
`min 1e-12 x^2 - x` subject to `-1 <= 1e-12 x <= 1`, presolve drops both the
quadratic term and the row as negligible, leaving `min -x` over a free x, which
is reported **Unbounded**. The problem is strictly convex, with optimum
x = 5e11. This belongs in its own presolve change.


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
