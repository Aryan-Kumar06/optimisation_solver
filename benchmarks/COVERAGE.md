# Test and benchmark coverage

Validated on 2026-09-30 in the SIH checkout, based on PR #15 revision
`08c1520ae033e56ad0a86c6794f2ee3484965b46`, with the local test/harness changes.
The coverage work and two reviews exposed defects in the ADMM QP engine, which
this PR fixes (see *QP engine fixes* below). No other numerical engine was changed.

## Automated tests

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

## Public benchmark smoke run

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

## Scope and provenance

- [Netlib LP data](https://www.netlib.org/lp/data/): existing eight-instance selection and checksummed manifest; not the entire library.
- [MIPLIB 2017](https://miplib.zib.de/): existing frozen 25-instance Collection selection, not the full 240-instance Benchmark Set. Instances are never replaced after failure.
- [Maros–Mészáros primary archive](https://www.doc.ic.ac.uk/~im/00README.QP): all **138** inputs have pinned archive/member hashes and published objectives in `suites/qp.json`; all 138 were downloaded and decoded successfully. **Only the fixed 14 smoke cases were solved in this run.** Use `--full-qp` for the whole corpus.
- [Mittelmann QP benchmark](https://plato.asu.edu/ftp/qpbench.html): uses the Maros–Mészáros corpus; these runs do not replicate its hardware/settings.
- [Mittelmann LPopt](https://plato.asu.edu/ftp/lpopt.html): three public instances (`qap15`, `Linf_520c`, `irish-e`) pinned in `suites/mittelmann.json`. This subset is not the full LPopt/LPfeas benchmark; no claim is made about producing optimal basic solutions under the published protocol.

The ordinary CTest suite downloads nothing. Enable `QP_REFERENCE_TESTS` and
`NLP_REFERENCE_TESTS` for numerical reference comparisons. After fetching the
frozen public datasets, enable `BENCHMARK_CORPUS_TESTS` to register four strict
corpus CTests; current solver limitations cause failures. Commands are in
[README.md](README.md). Generated data, binaries and verbose run reports stay
outside source control; durable selections, hashes, tests and this coverage
record remain in the repository.
