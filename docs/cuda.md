# CUDA Backend

Optional GPU execution for the two engines whose iterations are dominated by
sparse products and vector updates: **PDLP** (the whole PDHG iteration runs on
the device) and **QP/ADMM** (a *hybrid*: vector work and sparse products on the
device, the KKT factorisation and solves on the CPU).

> **Status: implemented, not yet built or run on a GPU.** The CUDA sources in
> this tree were written on a machine with no CUDA toolkit, so none of them has
> been compiled, executed, sanitised, profiled or benchmarked. Everything on the
> CPU side -- the backend seam, the unchanged CPU results, the option handling,
> the CPU-only build -- is verified (see [Verification status](#verification-status)).
> Treat the GPU path as a first build to be validated with the commands under
> [Validating on a CUDA machine](#validating-on-a-cuda-machine), not as a
> supported feature.

---

## 1. Scope

| Engine | On the GPU | On the CPU |
| :--- | :--- | :--- |
| PDLP | Trial step (both fused halves), linesearch reductions, commit, iterate averaging, restart-from-average, activity refresh | Ruiz scaling, preconditioner, spectral-norm estimate (once, before iterating); linesearch/restart/step-size decisions (scalars); termination, infeasibility certificates and polishing *scoring* at check boundaries |
| QP (ADMM) | `A x`, `Aᵀ v`, `P x` (cuSPARSE), right-hand-side assembly, box projection + dual ascent, residual norms and objective | KKT factorisation and triangular solves, rho adaptation, termination test, certificates, polishing |
| Dual simplex, branch-and-cut | — | Everything (no CUDA backend; out of scope) |

Presolve, classification, dispatch and postsolve are untouched. A CUDA backend is
a compute device for one engine call on the already-presolved model; it never
presolves, rescales outside the engine, or changes what the engine returns.

## 2. Prerequisites

* NVIDIA GPU with a driver supporting your toolkit.
* CUDA Toolkit providing `nvcc`, the CUDA runtime and cuSPARSE.
* CMake ≥ 3.20 (the project minimum). `CMAKE_CUDA_ARCHITECTURES=native` as a
  default needs CMake ≥ 3.24; with older CMake, pass an explicit architecture list.
* **Windows:** `nvcc` only works with the MSVC host compiler, so the whole
  project must be configured with the Visual Studio toolchain. MinGW/MSYS2 g++
  builds cannot enable CUDA.

### Toolkit version

Derived from the APIs the code uses, not from a tested build:

* C++17 device code (`CUDA_STANDARD 17`): CUDA 11.0 or newer.
* Generic cuSPARSE SpMV with `CUSPARSE_SPMV_CSR_ALG2` (deterministic CSR
  product): CUDA 11.2 or newer.
* A GPU's architecture must be supported by the toolkit; e.g. Blackwell
  (compute capability 12.x, such as the RTX 50 series) requires CUDA 12.8+.

Libraries linked: `CUDA::cudart_static` (both engines) and `CUDA::cusparse`
(QP only). cuBLAS and cuSOLVER are not used.

## 3. Building

CUDA is **off by default**. A default configure never looks for a CUDA compiler.

```bash
# Whole project, both engines
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DOPTIMSOLVER_ENABLE_CUDA=ON
cmake --build build-cuda -j
ctest --test-dir build-cuda --output-on-failure

# A machine without a GPU (CI, packaging) must name the architectures:
cmake -S . -B build-cuda -DOPTIMSOLVER_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90"

# Engines standalone (each builds its own tests, including the CUDA ones)
cmake -S pdlp_engine -B build-pdlp-cuda -DPDLP_ENABLE_CUDA=ON
cmake -S qp_engine   -B build-qp-cuda   -DQP_ENABLE_CUDA=ON
```

| Option | Default | Effect |
| :--- | :--- | :--- |
| `OPTIMSOLVER_ENABLE_CUDA` | `OFF` | Enables CUDA at the top level and defaults both engine options below to `ON`. |
| `PDLP_ENABLE_CUDA` | `OPTIMSOLVER_ENABLE_CUDA`, else `OFF` | Builds `pdlp_cuda` and `pdlp_cuda_tests`. |
| `QP_ENABLE_CUDA` | `OPTIMSOLVER_ENABLE_CUDA`, else `OFF` | Builds `qp_cuda` and `qp_cuda_tests`. |
| `CMAKE_CUDA_ARCHITECTURES` | `native` (CMake ≥ 3.24) | Never hard-coded; any explicit value wins. |

Build-system guarantees:

* With CUDA off, no `.cu` file is compiled and a stub (`src/cuda/cuda_unavailable.cpp`)
  provides the same API, reporting that CUDA was not compiled in.
* Device code lives in separate static targets (`pdlp_cuda`, `qp_cuda`), so the
  engines' host flags (`-march=native`, `-Wpedantic`, LTO) never reach `nvcc`,
  and no CUDA flag reaches ordinary C++ targets.
* IEEE semantics are enforced: `--prec-div=true --prec-sqrt=true --ftz=false`,
  and configuring with `--use_fast_math` (or `ftz=true`, `prec-div=false`,
  `prec-sqrt=false`) in `CMAKE_CUDA_FLAGS` is a configure error. FMA contraction
  stays on, matching the CPU build's `-ffp-contract=fast`.
* All solver arithmetic stays in `double`.

## 4. Selecting a backend

Every layer exposes the same three-valued choice, defaulting to `Auto`:

| Layer | Field / flag |
| :--- | :--- |
| PDLP | `pdlp::PdlpOptions::backend`, `cudaDevice`, `cudaNonzeroThreshold` |
| QP | `qp::AdmmOptions::backend`, `cudaDevice`, `cudaNonzeroThreshold` |
| Pipeline | `solver::SolverOptions::backend`, `cudaDevice` |
| CLI | `--backend auto\|cpu\|cuda`, `--cuda-device <index>` |

| Value | Behaviour |
| :--- | :--- |
| `cpu` | Always the CPU implementation. |
| `cuda` | The CUDA backend or an error. If the build lacks CUDA, no device is usable, the device index is out of range, or setup fails (e.g. insufficient device memory), the solve **fails with the reason** -- `InvalidProblem` from the engine, `Unsupported` from the pipeline, a non-zero exit from the CLI. It is never silently run on the CPU. |
| `auto` | CUDA only when all hold: the build has CUDA, the problem is at least `cudaNonzeroThreshold` nonzeros, the device is usable, and setup succeeds. Otherwise the CPU, with the reason recorded. |

Selection is deterministic: it depends only on the options, the build, the
device, and the size of the (scaled) problem.

What actually ran is always recorded, never inferred from the request:

* `PdlpResult::executedBackend` / `backendMessage`, `AdmmResult::executedBackend` / `backendMessage`
* `SolveResult::executedBackend` / `backendReason`
* The CLI prints `Compute backend: …` when `--backend` was given or a GPU ran
  (the default output is unchanged).

Engines without a CUDA backend (dual simplex, branch-and-cut) run on the CPU
whatever was requested; with `--backend cuda` the result says so
(`"dual_simplex has no CUDA backend; ran on the CPU"`).

### Auto thresholds -- provisional

| Engine | Default `cudaNonzeroThreshold` | Why |
| :--- | :--- | :--- |
| PDLP | 1,000,000 | Conservative guess: below this, per-iteration launch and synchronisation latency (~5 launches and one host sync per linesearch trial) is expected to outweigh the bandwidth advantage. **Not measured.** |
| QP | `INT64_MAX` (Auto never picks CUDA) | The hybrid moves two `n`-vectors across the bus and still runs the serial KKT solve on the CPU every iteration. Auto should select it only once a measured crossover exists. `backend = cuda` always runs it. |

Calibrate on real hardware with the benchmarks in section 9 and change the
defaults only with that evidence.

## 5. PDLP on the device

**Residency.** The scaled problem -- CSR *and* CSC of `A` (64-bit offsets,
32-bit indices, never truncated), bounds, objective, both diagonal
preconditioners -- is uploaded once when the backend is created. The iterate
`x, y`, the carried activity `A x`, the three trial buffers and the running
average stay on the device for the whole solve.

**The step.** The CPU kernel's fused structure is kept:

* primal half over CSC columns: `(Aᵀy)_j → gradient → prox/clip → x'_j`, plus
  `Σ dx²/T`;
* dual half over CSR rows: `(A x')_i → extrapolation → Moreau prox → y'_i, (A x')_i`,
  plus `Σ dy²/Σ` and `Σ dy·(A dx)`.

The kernel boundary between the halves is the one global dependency PDHG has.
The per-coordinate arithmetic is **the same source** on both backends
(`pdlp/pdhg_math.h`, `__host__ __device__`), including `std::min/std::max`
comparison order: it differs from `fmin/fmax` on NaN, and NaN propagation is
how numerical failure is detected.

**Work mapping.** A line (row or column) is reduced by a group of G lanes,
G ∈ {1,2,4,8,16,32} chosen from the matrix's mean line length (largest power of
two not above it). Lines with more than `heavyLineNonzeros` (default 4096)
nonzeros are excluded from the group kernel and reduced by one thread block
each, so a few dense rows or columns cannot stall a warp. Both are heuristics
awaiting profiling (`CudaBackendConfig` overrides them).

**Reductions.** No floating-point atomics. Each block writes one partial per
sum; a single-block pass adds them in index order. Results are therefore
reproducible run-to-run on a given device and launch configuration (unlike the
multi-threaded CPU path, whose dynamic chunk claiming makes its reductions -- and
occasionally its iteration count -- vary between runs).

**Per-iteration traffic.** One host synchronisation per linesearch trial, to
read three doubles (`Σdx²/T`, `Σdy²/Σ`, the interaction). The host needs them to
accept or reject the step. Commit is a buffer swap; averaging is a device
kernel with the blend fraction computed on the host in the same order as the
CPU, so the fractions are bitwise identical.

**Check boundaries (every `terminationCheckFrequency` iterations, default 100).**
The host downloads `x`, `y`, and the average (at most `2(n+m)` doubles each) and
runs the *existing* CPU termination checker, infeasibility detector and restart
policy on them, multi-threaded. This is the correctness-first stage: the checks
are the identical code the CPU path runs, not a port. Its cost is reported as
`PdlpResult::hostCheckSeconds` (of which `backendProfile.snapshotSeconds` is the
download) so a benchmark can decide whether GPU-side termination is worth
building. Restart-from-average is a device-to-device copy plus an activity
refresh; the host-side restart anchor is taken from the snapshot already on the
host.

**Polishing** reuses the same device backend (the problem is already resident).

**Failure handling.** A CUDA error mid-solve returns `NumericalFailure` with
`statusMessage = "CUDA backend error: <API, file:line, status>"`; device memory
is released by RAII. The up-front memory check refuses a problem that would not
fit (`Auto` then uses the CPU; `Cuda` reports it).

## 6. QP on the device (hybrid)

| | Where | Per iteration |
| :--- | :--- | :--- |
| `A`, `Aᵀ` (as its own CSR), `P` (full symmetric CSR, as on the host) | device, uploaded once | — |
| `z, y, A x, z_old, y_old, best y` | device | — |
| `x` | host (KKT output) and device copy | upload `n` doubles |
| x-update right-hand side | built on device | download `n` doubles |
| KKT factorisation + triangular solves | **CPU** (`KktSolver`, unchanged) | — |
| `‖Ax−z‖`, `‖ρAᵀΔz‖` (or `‖Px+q‖`), objective | device reduction | download 3 doubles |
| `y, y_old, z, A x` for the termination/certificate checks | device | download at check boundaries only |

Every sparse product is a non-transposed, deterministic CSR SpMV
(`CUSPARSE_SPMV_CSR_ALG2`); `Aᵀ` is stored explicitly because a transposed CSR
product in cuSPARSE accumulates with atomics. Descriptors and SpMV workspaces
are created once. Offsets and indices are stored 32-bit when `nnz < 2³¹`
(provably safe), otherwise 64-bit; offsets are never truncated.

Rho adaptation, refactorisation, the termination test, both certificates and
polishing run on the host exactly as before. `AdmmResult::kktSolveSeconds` /
`kktFactorSeconds` (measured on every backend) and `backendProfile` quantify the
split. **Expect the KKT solve to bound any speed-up** (Amdahl): the GPU can only
remove the sparse-product and vector share of each iteration. Moving the KKT
solve to the GPU (cuSOLVER dense Cholesky for small dense KKT systems, or a GPU
sparse factorisation) is deliberately not attempted until profiling justifies it.

## 7. Known limitations

* **Never compiled or run.** See the status note at the top.
* PDLP termination, certificates and restart scoring run on the host at check
  boundaries (correctness-first stage; see section 5).
* Ruiz scaling, the diagonal preconditioner and (without preconditioning) the
  power iteration run on the CPU before upload.
* Group size, heavy-line threshold, grid sizing and both `Auto` thresholds are
  unprofiled heuristics.
* One GPU per solve (`cudaDevice`); no multi-GPU partitioning.
* QP is hybrid; its KKT solve is on the CPU and `Auto` never selects it.
* On Windows, CUDA builds require MSVC for the whole project; the MinGW build
  this repository is usually developed with cannot enable CUDA.

## 8. Tests

| Test | Builds | What it checks |
| :--- | :--- | :--- |
| `pdlp_tests` (new cases) | always | shared math NaN/∞ semantics; backend contract CPU-vs-CPU on 10 matrix shapes; explicit CUDA never falls back; Auto resolution; forced CPU is bitwise the default; invalid device rejected |
| `qp_tests` (new cases) | always | the same for the ADMM backend |
| `test_compute_backend` | always | pipeline plumbing, `Unsupported` for an unavailable CUDA request, reasons recorded |
| `test_cli` (new cases) | always | `--backend` / `--cuda-device` parsing, refusal, reporting |
| `pdlp_cuda_tests` | CUDA only | CPU-vs-CUDA kernel contract for every group size, with and without forced heavy lines, on 10 shapes (empty, one-nonzero, `m = 0`, tall, wide, skewed, 1e±9 scaled, all bound kinds); rejected-trial correctness; averaging; restart; NaN detection; full solves over 8 option variants × feasible/infeasible/unbounded; Auto selection; bad device index |
| `qp_cuda_tests` | CUDA only | CPU-vs-hybrid backend contract (lock-stepped on the same KKT solutions, including a rho change); full solves over 5 variants × 7 models |

The CUDA tests exit with code 77, which CTest reports as **skipped**, when no
usable device exists, so a CUDA build on a GPU-less machine is not a failure.

## 9. Benchmark methodology

`pdlp_bench [rows] [cols] [nnzPerRow] [iterationLimit] [threads] [cpu|cuda|auto] [repeats]`
reports: hardware threads, CUDA runtime/driver version and device name,
dimensions and nonzeros, one untimed warm-up solve, the median of `repeats`
timed solves, and for that solve the end-to-end time (everything inside
`solve()`), device setup/upload time, core time (end-to-end minus setup), host
check time and snapshot time, bytes each way, host synchronisations, and
µs/iteration. `qp_bench [cpu|cuda|auto]` adds the KKT/setup/transfer split per
case.

Report speed-up as `CPU time / CUDA time`, per size, for both the core and the
end-to-end time, with the CPU thread count and GPU model. Expect small problems
to be slower on the GPU; the crossover is what `cudaNonzeroThreshold` should be
set from.

### CPU baseline on the development machine

Intel Core i7-13645HX (14 cores / 20 threads), MSYS2 UCRT64 g++ 15.2, standalone
Release build (`-O3 -march=native`, LTO), `iterationLimit = 1000`, polishing off,
3 timed runs after 1 warm-up. The laptop throttles under sustained load:
warm-up and median can differ by 2–3× on the largest size, so both are shown.

| rows × cols | nnz | threads | warm-up (s) | median e2e (s) | µs / iteration (median) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 2,000 × 4,000 | 15,983 | 20 | 0.043 | 0.035 | 34.7 |
| 2,000 × 4,000 | 15,983 | 1 | 0.036 | 0.034 | 33.8 |
| 20,000 × 40,000 | 199,978 | 20 | 0.141 | 0.132 | 131.6 |
| 20,000 × 40,000 | 199,978 | 1 | 0.749 | 0.765 | 765.3 |
| 100,000 × 200,000 | 999,977 | 20 | 1.140 | 3.263 | 3,263.3 |
| 100,000 × 200,000 | 999,977 | 1 | 11.507 | 20.367 | 20,366.8 |

The smallest case is below `parallelNonzeroThreshold`, so it runs serially
either way. No CUDA numbers exist yet.

## 10. Validating on a CUDA machine

```bash
# Linux (or Windows from a "x64 Native Tools" prompt with MSVC)
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DOPTIMSOLVER_ENABLE_CUDA=ON \
      -DPDLP_BUILD_TESTS=ON -DQP_BUILD_TESTS=ON -DPDLP_BUILD_TOOLS=ON -DQP_BUILD_BENCH=ON
cmake --build build-cuda -j
ctest --test-dir build-cuda --output-on-failure            # expect 0 failed, 0 skipped

# Sanitisers on the equivalence tests
compute-sanitizer --tool memcheck  build-cuda/pdlp_engine/pdlp_cuda_tests
compute-sanitizer --tool racecheck build-cuda/pdlp_engine/pdlp_cuda_tests
compute-sanitizer --tool initcheck build-cuda/pdlp_engine/pdlp_cuda_tests
compute-sanitizer --tool memcheck  build-cuda/qp_engine/qp_cuda_tests

# Benchmarks: sweep sizes, CPU vs CUDA
for n in "20000 40000" "100000 200000" "400000 800000"; do
  build-cuda/pdlp_engine/pdlp_bench $n 10 2000 0 cpu 3
  build-cuda/pdlp_engine/pdlp_bench $n 10 2000 0 cuda 3
done
build-cuda/qp_engine/qp_bench cpu && build-cuda/qp_engine/qp_bench cuda

# Profiles
nsys profile -o pdlp build-cuda/pdlp_engine/pdlp_bench 100000 200000 10 2000 0 cuda 1
ncu --set full -k regex:"lineKernel|heavyLineKernel" -c 20 \
    build-cuda/pdlp_engine/pdlp_bench 100000 200000 10 200 0 cuda 1
```

What to look at first: whether `hostCheckSeconds` is a material share of the
PDLP solve (then move termination onto the device); achieved bandwidth and
load balance of `lineKernel` across group sizes; and, for QP, `kktSolveSeconds`
against the total.

## Verification status

Verified on the development machine (no CUDA toolkit, so CPU-only):

* CPU-only configure/build of the root project and both standalone engines, no
  new warnings.
* **Bitwise-unchanged CPU results:** a fingerprint harness solving 96 serial
  LP/QP configurations (every option toggle, feasible/infeasible/unbounded,
  iteration limits) produced byte-identical output -- statuses, iteration and
  trial counts, objectives, residuals, step sizes, and hashes of every solution
  and ray vector -- before and after the backend refactor.
* The full test suite, the new CPU-side backend tests, and the CUDA test
  harnesses compiled against a CPU build (where they correctly skip).

Not verified: anything that requires `nvcc` or a GPU.
