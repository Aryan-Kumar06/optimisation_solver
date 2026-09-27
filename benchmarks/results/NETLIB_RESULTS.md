# Netlib LP benchmark results

Generated 2026-09-27 04:38 UTC from `netlib_lp.json`.

## Provenance

- Commit: `c2e5fd881ad607b8bcdd0acf89db1af7d6a3b03c`
- Build: Release (`-O3 -DNDEBUG`), single-threaded (`--threads 1`)
- Compiler: Apple clang version 15.0.0 (clang-1500.3.9.4)
- Platform: Darwin 24.3.0 arm64
- Instances: Netlib LP, decoded from the source archive with Netlib's own
  `emps` decompressor. Netlib does not distribute plain MPS. Per-file SHA-256
  and dimensions are in `instances/netlib/manifest.json`.
- Reference objectives: the table in <https://www.netlib.org/lp/data/readme>.
  These are EXTERNAL CLAIMS used for corroboration only, never as proof of
  optimality.

## How to read this

`primal_feasible` and `optimality_verified` answer different questions and
are never merged:

| column | meaning |
|---|---|
| `primal_feasible` | the returned point satisfies the ORIGINAL model, checked before presolve and before any scaling |
| `optimality_verified` | a duality gap / KKT check closed independently |

A solver can be feasible with optimality simply not established — that is the
normal outcome when it returns no duals, and it is not a feasibility failure.
Both are recomputed by `benchmarks/lib/verify.py` against a reader written
independently of the solver's own; neither is the solver's self-report, which
is kept in `solver_status`.

## Frozen tolerances

| quantity | value |
|---|---|
| feasibility absolute | `1e-06` |
| feasibility relative | `1e-08` |
| integrality absolute | `1e-06` |
| optimality normalised | `1e-06` |

## Independently verified optimal

| solver | verified optimal | matches published objective | feasible point | infeasible point | peak MB |
|---|---|---|---|---|---|
| `highs:highs-ds` | **8/8** | 8/8 | 8/8 | 0 | 5.1 |
| `optimsolver:dual_simplex` | **6/8** | 6/8 | 8/8 | 0 | 9.1 |
| `optimsolver:pdlp` | **6/8** | 8/8 | 8/8 | 0 | 3.5 |

## Per instance

| instance | rows | cols | reference | `highs:highs-ds` | `optimsolver:dual_simplex` | `optimsolver:pdlp` |
|---|---|---|---|---|---|---|
| adlittle | 56 | 97 | 225494.963 | 225494.963 ✅ | 225494.963 ✅ | 225494.962 ⚠️ |
| afiro | 27 | 32 | -464.753143 | -464.753143 ✅ | -464.753143 ✅ | -464.753143 ✅ |
| blend | 74 | 83 | -30.8121498 | -30.8121498 ✅ | 8.04951084e-15 ❌ 100.0% off | -30.8121493 ✅ |
| degen2 | 444 | 534 | -1435.178 | -1435.178 ✅ | -1435.178 ✅ | -1435.178 ✅ |
| recipe | 91 | 180 | -266.616 | -266.616 ✅ | -266.616 ✅ | -266.616 ✅ |
| sc50a | 50 | 48 | -64.5750771 | -64.5750771 ✅ | -64.5750771 ✅ | -64.5750769 ✅ |
| sc50b | 50 | 48 | -70 | -70 ✅ | -70 ✅ | -70.0000028 ⚠️ |
| share2b | 96 | 79 | -415.732241 | -415.732241 ✅ | -374.522624 ❌ 9.9% off | -415.732243 ✅ |

✅ optimality independently verified · ⚠️ feasible and matches the published objective, optimality not proven · ❌ feasible but materially short of the published objective · — no point returned

## Reproducing

```sh
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release -j8
python3 benchmarks/fetch_netlib.py --set smoke
python3 benchmarks/bench.py benchmarks/instances/netlib/mps/*.mps \
    --solvers dual_simplex,pdlp,highs --timeout 30 --threads 1 \
    --best-known benchmarks/instances/netlib/best_known.json \
    --out benchmarks/results/netlib_lp.json
python3 benchmarks/report_netlib.py benchmarks/results/netlib_lp.json
```

HiGHS is reached through SciPy and is a REFERENCE only: it lives under
`benchmarks/` and nothing in the solver links or includes it.
