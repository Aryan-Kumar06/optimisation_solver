# Test and benchmark coverage

Validated on 2026-09-30 in the SIH checkout, based on PR #15 revision
`08c1520ae033e56ad0a86c6794f2ee3484965b46`, with the local test/harness changes.
Production numerical engines were not changed for this coverage work.

## Automated tests

The Release build, with assertions enabled in test targets, ran **70 CTests:
69 passed, 1 failed**. This includes the restored NLP engine, elastic KKT,
public pipeline and CLI tests, NLP/SLSQP comparisons, PDLP and QP engine unit
tests, parser/presolve/postsolve/MILP tests, benchmark harness tests, and the
new reference/integrity tests.

The failing test is `qp_reference`: **299/300** randomized QPs were compared or
had their nonoptimal status corroborated by OSQP. Zero-based case **219**, seed
**20260908**, returned `limit_reached` while OSQP returned `solved`. This is a
convergence limitation, not evidence of an incorrect optimality claim. The test
is deliberately still failing; it is not skipped or marked `WILL_FAIL`.

The new Python suites contain seven offline tests of selection, checksums,
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
| Maros–Mészáros QP | 14 | 9 | 3 objective disagreements, 2 unverified |
| Public Mittelmann LP subset | 3 | 0 | 3 unverified due to time limits |

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
