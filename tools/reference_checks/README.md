# External reference checks

Reference packages are optional and never linked into the production solver.
To enable independent QP (OSQP) and NLP (SciPy SLSQP) tests:

```sh
python3 -m pip install numpy scipy 'osqp>=1.0,<2'
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DQP_REFERENCE_TESTS=ON -DNLP_REFERENCE_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

QP reference configuration fails if dependencies are missing. The QP test runs
300 cases with seed 20260908 through the public `solver::solve` API. To replay
or try another seed:

```sh
python3 tools/reference_checks/qp_vs_osqp.py --generator build/qp_reference_cases --count 300 --seed 20260908
```

Cases include convex minimization, concave maximization, offsets, full Hessians,
equalities, ranged rows, one-sided rows and free/bounded variables. Original
objectives and primal feasibility are independently recomputed. Strictly convex
cases compare both objective (relative tolerance 1e-4) and solution (absolute
1e-3). Feasibility tolerance is 1.01e-4, retained from the original randomized
checker; the public corpus uses the stricter frozen benchmark tolerances.

Reference exceptions, inaccurate statuses, missing cases, nonfinite values and
unverified outcomes fail the test. Infeasible/unbounded statuses require OSQP
corroboration; this is not certificate verification. Every case is reported.
Bound identity rows are included in the reference model. Separate analytic
adapter tests verify dual signs and objective conventions with exact solutions.

The historical `lp_random_generator.cpp` remains available for manual LP
experiments. Public LP/MILP comparisons use `benchmarks/run_suites.py` with
SciPy/HiGHS; public QPs use the isolated OSQP adapter. See
[benchmark coverage](../../benchmarks/COVERAGE.md) for limitations and results.
