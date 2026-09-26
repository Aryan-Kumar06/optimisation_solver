# Optimisation Solver — User Guide

A practical guide for compiling, running, and using `optimisation_solver` on mathematical programming instances.

---

## 1. Overview

`optimisation_solver` is an open-source mathematical optimization solver written in modern C++17. It supports:
- **Linear Programming (LP):** Continuous variables, linear constraints, linear objective.
- **Mixed-Integer Linear Programming (MILP):** Continuous, integer, and binary variables, linear constraints, linear objective.
- **Convex Quadratic Programming (QP):** Continuous variables, linear constraints, convex quadratic objective terms.

The solver ingests standard MPS format files, validates model structure, applies presolve reductions once, routes the reduced problem to an appropriate numerical engine, and restores the solution back to original problem coordinates with full primal and dual validation.

---

## 2. Building the Project

### Prerequisites
- C++17 compliant compiler (`g++` >= 9.0 or `clang++` >= 10.0)
- CMake >= 3.16
- Make or Ninja build tool

### Installation (Recommended)

The repository provides a self-contained installation script:

```bash
# 1. Clone repository
git clone https://github.com/RaghavGupta2910/optimisation_solver.git
cd optimisation_solver

# 2. Run automated installer
./install.sh
```

The script:
1. Verifies that `cmake` and a C++17 compiler are available on your system.
2. Configures and compiles the project in Release mode.
3. Installs the `optimsolver` executable to `~/.local/bin/optimsolver`.
4. Checks whether `~/.local/bin` is present in your `PATH`. If missing, safely appends the export command to your shell startup file (`~/.zshrc` on macOS / zsh, or `~/.bashrc`/`~/.bash_profile` on bash) without duplicate entries.
5. Informs you how to reload your active terminal session.

After running `./install.sh`, reload your shell environment:

```bash
source ~/.zshrc    # on macOS / zsh, or source ~/.bashrc on bash
```

Once reloaded, `optimsolver` is available globally from any directory:

```bash
optimsolver --help
```

### Advanced / Developer Build (Manual CMake)

For active development, running unit tests, or custom compiler configurations without installing to `~/.local/bin`:

```bash
# 1. Configure build files
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# 2. Compile executable and test suites
cmake --build build -j

# 3. Run directly from the build tree
./build/optimsolver
./build/optimsolver solve tests/cli/simple_lp.mps
```

*(Optional: You can install the compiled binary to a custom directory prefix with `cmake --install build --prefix <path>`.)*

---

## 3. Starting the Solver

The solver supports two operating modes:
1. **Interactive Mode:** Launch the terminal interface by running `optimsolver` with no arguments (or `./build/optimsolver` from the build directory).
2. **Batch / Command-Line Mode:** Execute direct solves from scripts or terminal using `optimsolver solve <model.mps> [options]`.

---

## 4. Interactive CLI

Launch the interactive interface:

```bash
optimsolver
# or from build directory: ./build/optimsolver
```

This presents the startup banner, mascot, and main menu:

```text
        (x)───(y)
         │  ∇  │        OPTIMSOLVER
       ╭─┴─────┴─╮      Mathematical Optimization Engine
       │  ◈   ◈  │
       │ min f(x)│      Problem Families:  LP · QP · MILP
       │ ─────── │      Solver Engines:    PDLP · Dual Simplex · Branch & Cut · QP
       ╰──┬───┬──╯
         ═╧═══╧═

╭──────────────────────────────────────────────────────────────╮
│                                                              │
│                    OPTIMISATION SOLVER                       │
│                                                              │
│        Modular Mathematical Optimization Engine              │
│                    SIH26 • SIH26119                          │
│                                                              │
╰──────────────────────────────────────────────────────────────╯

  MAIN MENU

  [1]  Open MPS Model
  [2]  Solver Settings
  [3]  Model Information
  [4]  Help
  [5]  Exit

  Select an option [1-5]:
```

### Main Menu and Current Model Context

The interactive session maintains a **Current Model** state across operations:
- When no model is loaded, options `[1]–[5]` allow opening a model, configuring settings, checking model info, viewing help, or exiting.
- Once a model is opened, the terminal displays the **CURRENT MODEL** card:

```text
╭──────────────────────────────────────────────────────────────╮
│  CURRENT MODEL                                               │
╰──────────────────────────────────────────────────────────────╯

  File             refinery.mps
  Problem type     MILP
  Variables        1,248
  Constraints        863
  Nonzeros         7,421

  MAIN MENU

  [1]  Solve Current Model
  [2]  Open Another Model
  [3]  Model Information
  [4]  Solver Settings
  [5]  Help
  [6]  Exit

  Select an option [1-6]:
```

### Navigating Interactive Features

- **Open MPS Model:**
  - Prompts for a file path. Supports absolute paths (`/Users/.../model.mps`), relative paths (`models/test.mps`), and home-directory shorthand (`~/models/transport.mps`).
  - Validates the model IR and handles errors gracefully with options to retry or return to the menu.
- **Solve Current Model:**
  - Solves the currently loaded model using configured solver settings and automatic engine dispatch (or selected override).
  - Displays pipeline status with mascot animations on interactive terminals, followed by a comprehensive **SOLVE RESULT** card.
  - Prompts to export the original-space solution vector and shadow prices/reduced costs.
- **Model Information:**
  - Displays detailed IR statistics: filename, detected problem family (LP, QP, MILP), variable counts broken down into continuous, integer, and binary, constraint count, matrix nonzeros, and objective specifications (sense and quadratic terms).
- **Solver Settings:**
  - View and change session options:
    - Change numerical engine: automatic dispatch or manual override (`pdlp`, `dual_simplex`, `branch_and_cut`, `qp`).
    - Change maximum solve time limit in seconds (or 0 for unlimited).
    - Configure default output file destination.
- **Help:**
  - Concise summary of interactive and command-line usage.
- **Exit:**
  - Closes the application cleanly (or enter `exit`, `quit`, or `q` at any prompt).

---

## 5. Command-Line (Batch) Mode

For automated workflows, benchmarking, and shell scripts, invoke the `solve` subcommand:

```bash
optimsolver solve <model.mps> [options]
```

### Supported Command-Line Flags

| Option | Argument | Description |
| :--- | :--- | :--- |
| `--solver` | `<name>` | Force a specific numerical engine: `pdlp`, `dual_simplex`, `branch_and_cut`, `qp`. |
| `--time-limit` | `<seconds>` | Set a maximum solve time budget in seconds (positive floating-point number). |
| `--output` | `<file>` | Write the reconstructed original-space solution vector and duals to a file. |
| `-h`, `--help` | — | Display help information and usage examples for the solve command. |

### Command Examples
```bash
# Basic solve with automatic engine selection
optimsolver solve models/production.mps

# Force the first-order PDLP engine
optimsolver solve models/large_scale.mps --solver pdlp

# Enforce a 30-second time budget and export the solution
optimsolver solve models/schedule.mps --time-limit 30.0 --output /tmp/solution.txt
```

---

## 6. Supported Solver Engines

The dispatcher selects an appropriate engine based on the original model classification, or respects the user's `--solver` override:

1. **PDLP (`pdlp`):**
   - **Algorithm:** First-Order Primal-Dual Hybrid Gradient (PDHG).
   - **Best For:** Large-scale continuous linear programs (LPs).
   - **Features:** Adaptive step sizes (Malitsky-Pock), normalized duality gap restart checks, and Ruiz diagonal matrix preconditioning.
2. **Dual Simplex (`dual_simplex`):**
   - **Algorithm:** Tableau-based dual simplex.
   - **Best For:** Small-to-medium continuous linear programs (LPs) requiring basic solutions (vertex points).
   - **Features:** Exact basis maintenance; used as the LP relaxation solver inside branch-and-cut.
3. **Branch-and-Cut (`branch_and_cut`):**
   - **Algorithm:** Branch-and-bound search with dynamic cutting planes.
   - **Best For:** Mixed-Integer Linear Programs (MILP).
   - **Features:** Gomory fractional cuts generated from simplex tableau rows, most-fractional branching, and primal feasibility heuristics.
4. **QP Engine (`qp`):**
   - **Algorithm:** Alternating Direction Method of Multipliers (ADMM).
   - **Best For:** Convex Quadratic Programs (QP) with linear constraints.
   - **Features:** Augmented KKT linear system factorizations and adaptive penalty updates ($\rho$).

---

## 7. MPS Input Format Specification

The MPS reader in `optimisation_solver` is implemented in `src/mps/mps_reader.cpp`. It supports standard fixed-format and free-format `.mps` files with standard sections:

### Supported Sections

1. **`NAME`**
   - Format: `NAME [problem_name]`
   - Defines the model identifier (defaults to unnamed if omitted).
2. **`OBJSENSE` (or `OBJSENS`)**
   - Specifies objective direction: `MIN`, `MINIMIZE`, `MAX`, or `MAXIMIZE`.
   - May appear on the same line as the header or on the next indented line.
   - Defaults to `Minimize` if omitted.
3. **`ROWS`**
   - Declares constraint rows and objective row:
     - `N`: Free row. The **first** `N` row encountered is treated as the objective function. Any subsequent `N` rows are treated as free rows and ignored with a warning.
     - `L`: Less-than-or-equal inequality ($\sum a_i x_i \le b$).
     - `G`: Greater-than-or-equal inequality ($\sum a_i x_i \ge b$).
     - `E`: Equality constraint ($\sum a_i x_i = b$).
4. **`COLUMNS`**
   - Defines matrix non-zero entries by column:
     - Format: `[col_name] [row_name1] [val1] [row_name2] [val2]`
   - **Integer Markers:**
     - Enclosing columns between `'MARKER'` records flags variables as integer:
       ```text
       MARK0001 'MARKER' 'INTORG'
       VAR1     ROW01    2.0
       MARK0002 'MARKER' 'INTEND'
       ```
     - Single quotes and double quotes around keywords are stripped automatically.
5. **`RHS`**
   - Defines right-hand-side constants for constraint rows.
   - **Vector Selection:** The first RHS vector name encountered is selected; any subsequent vectors with different names are ignored with a warning.
   - **Objective Offset:** An RHS entry naming the objective row is treated as the negated objective constant ($\text{offset} = -b$).
6. **`RANGES`**
   - Defines ranged constraints ($L_i \le \sum a_{ij} x_j \le U_i$).
   - The first RANGES vector name is selected; subsequent vectors are ignored.
   - Adjusts lower and upper row bounds depending on the row sense:
     - Row `G`: $b \le \text{row} \le b + |r|$
     - Row `L`: $b - |r| \le \text{row} \le b$
     - Row `E`: if $r > 0$, $b \le \text{row} \le b + r$; if $r < 0$, $b + r \le \text{row} \le b$
7. **`BOUNDS`**
   - Defines variable bounds. The default for any unspecified continuous variable is $0 \le x < +\infty$.
   - Supported bound types:
     - `LO`: Lower bound ($x \ge v$).
     - `UP`: Upper bound ($x \le v$). If $v < 0$ and no lower bound was explicitly set, lower bound is opened to $-\infty$.
     - `FX`: Fixed variable ($x = v$).
     - `FR`: Free variable ($-\infty < x < +\infty$).
     - `MI`: Minus infinity lower bound ($-\infty < x \le \text{ub}$).
     - `PL`: Plus infinity upper bound ($\text{lb} \le x < +\infty$).
     - `BV`: Binary variable ($x \in \{0, 1\}$).
     - `LI`: Integer lower bound ($x \ge v$, $x \in \mathbb{Z}$). Must be an integral value.
     - `UI`: Integer upper bound ($x \le v$, $x \in \mathbb{Z}$). Must be an integral value.
8. **Quadratic Objective Sections (`QUADOBJ` / `QUADS` / `QMATRIX` / `QUADOBJ2`)**
   - `QUADOBJ` / `QUADS`: Specifies lower triangle coefficients of the quadratic objective matrix.
   - `QMATRIX` / `QUADOBJ2`: Specifies full quadratic objective matrix entries.
   - Values represent direct terms $q_{ij} x_i x_j$ without implicit half factors.
9. **`ENDATA`**
   - Marks the end of the file.

### Unsupported Sections
If the file contains sections that cannot be represented in `model::Model`, the reader halts with an informative error rather than silently ignoring them:
- `SOS`: Special Ordered Sets
- `INDICATORS`: Indicator constraints
- `QCMATRIX` / `QCROWS`: Quadratically constrained rows

---

## 8. Example MPS File

The following small LP instance (`tests/cli/simple_lp.mps`) minimizes $3 x_1 + 2 x_2$ subject to $x_1 + x_2 \ge 4$, with $x_1 \ge 0, x_2 \ge 0$:

```text
NAME          SIMPLE_LP
OBJSENSE
  MIN
ROWS
 N  OBJ
 G  C1
COLUMNS
    X1        OBJ                  3.0   C1                   1.0
    X2        OBJ                  2.0   C1                   1.0
RHS
    RHS1      C1                   4.0
BOUNDS
 LO BND       X1                   0.0
 LO BND       X2                   0.0
ENDATA
```

### Mathematical Interpretation
$$\begin{aligned}
\min \quad & 3 x_1 + 2 x_2 \\
\text{s.t.} \quad & x_1 + x_2 \ge 4 \\
& x_1 \ge 0, \quad x_2 \ge 0
\end{aligned}$$

Optimal solution: $x_1 = 0, x_2 = 4$, with objective value $f^* = 8$.

---

## 9. Preparing Your Own Model

When constructing an MPS file for `optimisation_solver`:
1. **Row Names:** Use unique alphanumeric identifiers for every row.
2. **Objective Row:** Place your objective row under `ROWS` with type `N`. It should be the first `N` row in the file.
3. **Finite Numbers:** Ensure all matrix coefficients, RHS values, and bounds are finite floating-point numbers.
4. **Consistent Bounds:** Verify that variable bounds satisfy $\text{lowerBound} \le \text{upperBound}$.
5. **Standard Sections:** Follow the standard ordering: `NAME` $\to$ `OBJSENSE` $\to$ `ROWS` $\to$ `COLUMNS` $\to$ `RHS` $\to$ `RANGES` $\to$ `BOUNDS` $\to$ `ENDATA`.

---

## 10. Understanding Solver Output

Upon completing a solve, the CLI prints:

```text
╭─ OPTIMSOLVER ────────────────────────────────────────╮
│                                                      │
│  Solving SIMPLE_LP                                   │
│                                                      │
│  Model      2 variables · 1 constraint               │
│  Reduced    2 variables · 1 constraint               │
│  Engine     dual_simplex                             │
│                                                      │
╰──────────────────────────────────────────────────────╯

✓ Optimal

  Objective       8
  Iterations      1
  Solve time      0.0002 s
  Duals available: 1 shadow prices, 2 reduced costs
```

### Output Fields
- **Model / Reduced Dimensions:** Shows original problem size versus reduced size after presolve reductions.
- **Engine:** The solver engine that executed the problem.
- **Status:**
  - `✓ Optimal`: A feasible solution satisfying optimality criteria was found.
  - `✗ Infeasible`: Presolve or the solver engine proved the model has no feasible solution.
  - `✗ Unbounded`: The objective can be improved without bound.
  - `Limit Reached`: Time limit or iteration limit was reached.
  - `Numerical Failure`: Numerical difficulties prevented convergence.
- **Dual Information:**
  - When duals are trustworthy, the CLI reports the count of reconstructed shadow prices and reduced costs.
  - When duals cannot be established (e.g., MILP models, unsupported transformations, non-finite values), the solver fails closed and reports the specific reason rather than emitting corrupt data.

---

## 11. Saving Solutions

When invoking the CLI with `--output <path>`, original-space solution values are written to a plain-text file:

```text
# Solution for SIMPLE_LP
# Status: optimal
# Objective: 8
X1 0
X2 4
# Dual C1 2
# Reduced cost X1 1
# Reduced cost X2 0
```

- Lines starting with `#` are metadata headers, constraint shadow prices, or reduced costs.
- Variable assignments are formatted as `VARIABLE_NAME VALUE`.

---

## 12. Common Errors & Resolutions

- **`Failed to read MPS file: Unable to open file`**
  - Verify that the file path exists and is accessible. When using interactive mode, paths starting with `~` are expanded automatically.
- **`Model failed structural validation`**
  - The model contains conflicting bounds ($lb > ub$) or invalid row indices. Check the `BOUNDS` and `ROWS` sections of the MPS file.
- **`Solver error: Unsupported problem`**
  - The model features a problem class not supported by the requested engine (for example, attempting to solve a MILP with the continuous QP engine). Use automatic engine selection or choose `branch_and_cut`.
- **`Output error: Unable to open output file`**
  - Check write permissions and verify that the target directory exists.

---

## 13. Troubleshooting

1. **Rebuilding Cleanly:**
   ```bash
   rm -rf build && cmake -S . -B build && cmake --build build -j
   ```
2. **Verifying Correctness:**
   Run the test suite to verify that all algorithms function properly on your system:
   ```bash
   ctest --test-dir build --output-on-failure
   ```
3. **Checking Model Feasibility:**
   If a model unexpectedly reports infeasible, inspect whether conflicting bounds were introduced in the `BOUNDS` section or tight equality rows in `ROWS`.

---

## 14. Running the Test Suite

The test suite is managed via CTest:

```bash
ctest --test-dir build --output-on-failure
```

To run a specific test target:
```bash
# Run CLI tests
./build/test_cli

# Run MPS reader tests
./build/test_mps_reader

# Run end-to-end pipeline tests
./build/test_end_to_end
```
