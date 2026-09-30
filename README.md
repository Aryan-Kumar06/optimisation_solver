# Optimisation Solver

An open-source mathematical optimization solver written in modern C++17 for linear programming (LP), mixed-integer linear programming (MILP), and convex quadratic programming (QP).

The solver parses MPS models, builds an internal model representation, validates problem consistency, executes presolve reductions once, routes the reduced model to specialized numerical engines, and restores full primal and dual solutions into original coordinates.

---

## SIH Problem Statement

- **Organization:** Mangalore Refinery and Petrochemicals Limited (MRPL)
- **Problem Statement ID:** SIH26119
- **Problem Statement Title:** Indigenous GPU-Accelerated Optimization Solver (Sovereign Alternative to Express / CPLEX)
- **Category:** Software
- **Theme:** Smart Automation

---

## Project Overview

Mathematical optimization is critical across refinery operations, supply chain scheduling, pipeline routing, and resource allocation. Commercial enterprise solvers such as FICO Xpress and IBM CPLEX are proprietary, closed-source, and subject to high licensing fees and foreign technology dependencies.

The objective of `optimisation_solver` is to establish an open-source, modular optimization engine built around clean software architecture and sound numerical principles. Rather than tightly coupling input parsing, presolve reductions, and numerical execution into a monolithic codebase, this project separates each stage into distinct, testable modules with well-defined mathematical contracts.

*Note: This project is an ongoing engineering initiative focused on establishing a sovereign, modular solver architecture. It is not currently a complete drop-in production replacement for mature commercial solvers.*

---

## Current Capabilities

- **MPS Ingestion:** Reads standard fixed-format and free-format `.mps` files (`ROWS`, `COLUMNS`, `RHS`, `RANGES`, `BOUNDS`, `QUADOBJ`, `QMATRIX`).
- **Model Representation & Validation:** Explicit in-memory Model IR storing bounds, integrality types (`Continuous`, `Integer`, `Binary`), sparse linear constraints, and quadratic objective terms, with strict structural validation.
- **Presolve Pipeline:** Single-pass reduction passes that eliminate fixed variables, remove singleton equality rows, tighten bounds with provenance tracking, and discard redundant constraints.
- **Linear Programming (LP):** Solved via first-order Primal-Dual Hybrid Gradient (PDLP) with Ruiz equilibration, or tableau-based Dual Simplex for basic feasible solutions.
- **Mixed-Integer Linear Programming (MILP):** Branch-and-bound search tree combined with Gomory fractional cut generation from simplex tableau rows and primal heuristics.
- **Convex Quadratic Programming (QP):** Solved via Alternating Direction Method of Multipliers (ADMM) with KKT linear system factorizations.
- **Postsolve Reconstruction:** Reconstructs original variable coordinates and re-evaluates original objective values with strict non-finite value protection (`NaN`, `±Inf`).
- **Dual Reconstruction:** Recovers constraint duals (shadow prices) and variable reduced costs by reversing bound-tightening provenance histories, validated against stationarity and complementary slackness residuals.
- **Dual Interface:** Interactive terminal menu interface alongside a non-interactive command-line tool.
- **Automated Verification:** Comprehensive CTest automated test suite covering unit operations, cascades, and end-to-end solve pipelines.

---

## Architecture

```text
       MPS File
          ↓
       Model IR
          ↓
       Validation
          ↓
       Original Classification
          ↓
       Presolve (Once) ─── logs metadata ───┐
          ↓                                 │
       Reduced Model                        │
          ↓                                 │
       Solver Dispatch                      │
          ↓                                 │
       PDLP / Dual Simplex / Branch & Cut / QP
          ↓                                 │
       Reduced SolveResult                  │
          ↓                                 │
       Postsolve (Once) <───────────────────┘
          ↓
       Original-Space SolveResult
```

### The Single-Presolve Invariant
Presolve is performed exactly **once** on the original problem formulation. The reduction passes generate an explicit transformation audit log (`PresolveResult`). The reduced problem is then solved in reduced coordinates. Postsolve consumes the same transformation metadata to restore original variable coordinates and attribute shadow prices back to source constraints. Double-presolving is strictly forbidden to prevent coordinate divergence.

For complete architectural specifications, see [docs/architecture.md](docs/architecture.md).

---

## Quick Start

### Installation (Recommended)

To install `optimsolver` on macOS or Linux:

```bash
git clone https://github.com/RaghavGupta2910/optimisation_solver.git
cd optimisation_solver
./install.sh
```

The installer:
- Verifies prerequisites (`cmake` and a C++17 compiler).
- Configures and builds the project in Release mode.
- Installs the binary to `~/.local/bin/optimsolver`.
- Configures your shell startup file (`~/.zshrc` on macOS/zsh, or `~/.bashrc`/`~/.bash_profile` on bash) so `~/.local/bin` is in your `PATH` if not already present.

After reloading your shell:
```bash
source ~/.zshrc    # on zsh / macOS, or open a new terminal window
```

You can run `optimsolver` from any directory:

**1. Interactive Terminal Interface:**
```bash
optimsolver
```

**2. Command-Line Batch Solve:**
```bash
optimsolver solve tests/cli/simple_lp.mps
```

---

### Advanced / Developer Build (Manual CMake)

For active development, test suite compilation, or custom compiler flags without installing to `~/.local/bin`:

```bash
# Configure and compile
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Run directly from the build tree:
./build/optimsolver
./build/optimsolver solve tests/cli/simple_lp.mps
```

---

## Documentation

- **[User Guide](USER_GUIDE.md):** Detailed guide covering build instructions, interactive navigation, command-line arguments, complete MPS format specifications, and troubleshooting.
- **[System Architecture](docs/architecture.md):** In-depth engineering notes on pipeline dataflow, coordinate spaces, numerical engine internals, and design contracts.
- **[Presentation Slide Deck](submission/PRESENTATION.md):** Outline and slide deck links for SIH 2026.
- **[Demonstration Video](submission/DEMO.md):** Demonstration video walkthrough and script guide.
- **[Screenshots Plan](assets/screenshots/README.md):** Planned terminal captures and CLI screenshots.

---

## Testing

Run the automated test suite with CTest:

```bash
ctest --test-dir build --output-on-failure
```

### Published validation results

The 2026-09-30 Release validation against upstream `b67250b` ran **72 CTests:
71 passed and one failed**. The failure is a QP iteration-limit case found by
the independent OSQP comparison; it remains visible in the test suite.

| Benchmark selection | Instances run | Objective agreement | Remaining outcomes |
|---|---:|---:|---|
| Netlib LP | 8 | 6 | 2 feasible, not optimal |
| MIPLIB 2017 Collection subset | 25 | 3 | 22 limited or unverified |
| Maros–Mészáros QP | 14 | 9 | 3 objective disagreements, 2 unverified |
| Public Mittelmann LP subset | 3 | 0 | 3 time-limited, unverified |
| **Total** | **50** | **18** | **32 require further work** |

The benchmark table records the PR #15 baseline before integrating newer upstream
MIQP changes. These are fixed-subset smoke runs at **five seconds per solver process**, not
full-library certification or comparable performance rankings. Objective
agreement requires independently validated feasible points and optimal statuses
from both solvers; it is not itself an optimality certificate. All 50 inputs
passed the independent parse cross-check. All 138 QP inputs are available via
the pinned downloader, but only 14 were solved in this run.

See the **[published results, known failures and reproduction commands](benchmarks/README.md#published-validation-results--2026-09-30)**
for the full tables, environment, tolerances, and reference-solver comparisons.

---

## Repository Structure

```text
optimisation_solver/
├── README.md                      # Project landing page
├── USER_GUIDE.md                  # Comprehensive user manual and MPS specification
├── LICENSE                        # MIT License
├── CMakeLists.txt                 # Top-level CMake build configuration
│
├── submission/                    # SIH deliverables
│   ├── PRESENTATION.md            # Slide deck documentation and links
│   └── DEMO.md                    # Demonstration video script and links
│
├── docs/                          # Technical specifications
│   └── architecture.md            # Engine dataflow, contracts, and design decisions
│
├── assets/                        # Screenshots and visuals
│   └── screenshots/
│       └── README.md              # Planned terminal capture catalogue
│
├── include/                       # Public C++ API headers
├── src/                           # Pipeline implementations (MPS, model, presolve, postsolve, solver)
├── cli/                           # Interactive and batch command-line application
├── pdlp_engine/                   # PDHG first-order linear programming engine
├── milp_engine/                   # Dual simplex and branch-and-cut MILP engine
├── qp_engine/                     # ADMM convex quadratic programming engine
├── benchmark_model/               # Sample models for benchmarking
├── tests/                         # Automated unit, integration, and pipeline test suites
└── tools/                         # Helper scripts
```

---

## Current Limitations & Roadmap

- **GPU Acceleration:** Future integration of CUDA / GPU-accelerated linear algebra kernels for matrix-vector multiplication in PDLP and ADMM QP.
- **Additional File Formats:** Ingesting LP format (`.lp`) alongside existing MPS support.
- **Additional Presolve Passes:** Linear variable substitution, duplicate row/column detection, and binary probing.
- **Parallel Branch-and-Bound:** Multi-threaded tree exploration and cut pool management.
- **Barrier Solver:** Infeasible primal-dual interior-point method for large continuous problems.

---

## Team

- **Raghav Gupta** — Team Lead — System Architecture, Solver Orchestration & CLI
- **Harehar Narayan Seth** — Lead Numerical Solver Development & Optimization Algorithms
- **Aryan Kumar** — Solver Engine Development & Numerical Methods
- **Preetish Attray** — MILP / Branch-and-Cut Development
- **Kabir Pahwa** — Postsolve, Dual Reconstruction & Solution Validation
- **Sarisha Jhinghan** — Presolve, Model Processing & Testing

---

## License

This project is licensed under the [MIT License](LICENSE).

## Smooth nonlinear programming

The `nlp_engine` module provides an elastic SQP solver, expression DAG with
reverse automatic differentiation, sparse Jacobians, and a callback API. It
reuses the QP engine and reports verified **first-order stationarity**, not
global optimality. `FirstOrderStationary` means the returned point satisfies the
implemented original-unit KKT residual checks; it does not imply LICQ, MFCQ or
another constraint qualification, and multiplier uniqueness is not guaranteed.
Run `optimsolver solve-nlp model.nlp`; see the
[NLP architecture, API, numerical limits, and examples](nlp_engine/README.md).
