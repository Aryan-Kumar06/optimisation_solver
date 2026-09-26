# Architecture Notes & Pipeline Design

This document details the software architecture, dataflow pipeline, module contracts, and design decisions of `optimisation_solver`.

---

## 1. Pipeline Overview

The solver processes models through a sequential, modular pipeline:

```text
       MPS File
          ↓
       Model IR  (src/mps, src/model)
          ↓
       Validation  (model.validate())
          ↓
       Original Classification  (solver::classify())
          ↓
       Presolve (Once)  (src/presolve) ── logs PresolveResult ──┐
          ↓                                                    │
       Reduced Model                                           │
          ↓                                                    │
       Solver Dispatch & solveReduced()                        │
       ├── PDLP (pdlp_engine/)                                 │
       ├── Dual Simplex (milp_engine/)                         │
       ├── Branch-and-Cut (milp_engine/)                       │
       └── QP (qp_engine/)                                     │
          ↓                                                    │
       Engine Normalization                                    │
          ↓                                                    │
       Reduced-Space SolveResult                               │
          ↓                                                    │
       Postsolve (Once)  (src/postsolve) <─────────────────────┘
          ↓
       Original-Space SolveResult (Primal & Dual Validation)
```

Each module has a clearly defined interface and data ownership boundary.

---

## 2. Pipeline Execution Stages

### Stage 1: MPS Ingestion & Model IR (`src/mps/`, `src/model/`)
- `mps_reader` parses fixed-format and free-format `.mps` files into memory.
- `model::Model` is the internal intermediate representation (IR), storing:
  - Variables with bounds, types (`Continuous`, `Integer`, `Binary`), and names.
  - Linear constraints in sparse representation, with lower and upper row bounds.
  - Objective offset, linear objective terms, and quadratic terms ($q_{ij} x_i x_j$, where $q_{ij}$ is the direct coefficient without an implicit $1/2$ factor).
- `model.validate()` verifies that variable and constraint bounds are structurally sound ($lb \le ub$) and that all sparse row indices point to valid variables.

### Stage 2: Original Model Classification (`src/solver/classifier.cpp`)
- `solver::classify(const model::Model&)` analyzes the **original** problem formulation:
  - Determines Problem Class: `LP` (linear continuous), `QP` (convex quadratic objective with linear constraints), or `MILP` (linear with integer or binary variables).
  - Inspects whether quadratic objective terms exist and checks for integrality requirements.
  - Generates a `solver::Classification` containing problem class flags and the default solver engine.

### Stage 3: Invertible Presolve Pipeline (`src/presolve/`)
- Executes reduction passes on the model to reduce rows and columns before solving:
  - **Fixed variable elimination:** Detects variables where $lb_i = ub_i$, logs their values into `PresolveResult::fixedVariables`, and removes them from the model.
  - **Singleton equality rows:** Identifies rows containing a single non-zero entry $a \cdot x_i = b$, computes $x_i = b/a$, updates bounds, records the transformation, and eliminates the row and variable. Both positive and negative coefficients are handled.
  - **Singleton inequality rows (bound tightening):** Uses single-variable rows to tighten variable bounds and records the provenance (which constraint tightened which bound) in `PresolveResult::boundProvenance` so dual multipliers can be attributed correctly during postsolve.
  - **Redundant constraints:** Computes minimum and maximum constraint activity based on variable bounds; if $[\text{activity}_{\min}, \text{activity}_{\max}] \subseteq [lb, ub]$, the row cannot be violated and is removed.
- **Single-Presolve Invariant:** Presolve is executed exactly once on the original model. Re-running presolve inside solver engines or dispatchers is prohibited because postsolve reconstruction requires the exact transformation mapping from the initial reduction.

### Stage 4: Engine Selection & `solveReduced()` (`src/solver/orchestrator.cpp`, `src/solver/dispatcher.cpp`)
- High-level orchestration provides two APIs:
  - **`solver::solve(model, options)`:** Complete pipeline: validates, classifies the original model, runs presolve once, calls `solveReduced()`, normalizes engine output, runs postsolve once, and returns the original-space solution.
  - **`solver::solveReduced(presolvedModel, originalClassification, options)`:** Dispatches directly to the selected solver engine on the reduced model, using the `Classification` of the original model.
- **Dispatcher Contract:** The dispatcher inspects the **original** classification (to know if quadratic terms were present originally) while inspecting the **reduced** model for runtime integrality (to detect if presolve eliminated all integer variables, turning a MILP into an LP).
- **Engine Selection:**
  - If the user provides `--solver <name>`, the requested engine is used (`pdlp`, `dual_simplex`, `branch_and_cut`, `qp`).
  - Otherwise, the problem is routed based on problem class: pure continuous LPs default to `dual_simplex` or `pdlp`, integer models route to `branch_and_cut`, and quadratic models route to `qp`.

### Stage 5: Solver Result Normalization (`src/solver/orchestrator.cpp`)
- Different solver engines have distinct internal status enums (`PdlpStatus`, `DualSimplexStatus`, `MilpStatus`, `QpStatus`).
- The orchestrator normalizes these into a uniform `solver::SolveStatus` (`Optimal`, `Infeasible`, `Unbounded`, `LimitReached`, `NumericalFailure`, `InvalidModel`).
- Engine results are converted into a `solver::SolveResult` before being passed to postsolve.

### Stage 6: Postsolve Reconstruction & Validation (`src/postsolve/`)
- `postsolve::Postsolver` reverses the transformations recorded in `PresolveResult` to map reduced-space results back into original model coordinates:
  - **Primal Reconstruction:** Maps reduced variable values back to their original variable indices using `presolvedToOriginalVar`, and restores fixed variables from `fixedVariables`.
  - **Non-Finite Detection:** Rejects solutions containing `NaN` or `±Inf`, returning `PostsolveStatus::BoundViolation`.
  - **Objective Re-evaluation:** Computes the objective value directly from reconstructed primal variables using the original objective terms, catching any offset or scaling discrepancy.
  - **Primal Feasibility Verification:** Computes maximum bound residual and maximum constraint residual against original bounds.
  - **Dual Reconstruction:** Recovers original-space constraint duals (shadow prices) and variable reduced costs:
    - Reduced costs are calculated via $d_j = \nabla_j f(x^*) - \sum_i a_{ij} y_i$.
    - Constraint duals from presolved rows are mapped back to their original constraint indices.
    - Multipliers from tightened bounds are attributed back to the source singleton rows using the recorded bound provenance history.
  - **Dual Optimality Checks:** Checks stationarity, dual sign feasibility, and complementary slackness against the original constraints (`maxDualResidual`).
  - **Fail-Closed Dual Behavior:** If reduced-space duals were not supplied by the engine, or if any transformation step cannot be reliably inverted, `dualsAvailable` is set to `false` and a specific `dualsUnavailableReason` is recorded, rather than emitting incorrect or unmapped values.

---

## 3. Implemented Solver Engines

### 1. PDLP Engine (`pdlp_engine/`)
- **Algorithm:** Primal-Dual Hybrid Gradient (PDHG) first-order method for linear programming.
- **Components:**
  - `pdhg_kernel.cpp`: Core iterate updates with operator extrapolation.
  - `step_controller.cpp`: Adaptive step-size adjustments (Malitsky-Pock and line-search checks).
  - `preconditioner.cpp`: Ruiz diagonal equilibration and Pock-Chambolle scaling.
  - `restart_controller.cpp`: Restarts based on normalized duality gap metrics.
  - `termination.cpp`: Primal and dual feasibility and duality gap convergence criteria.

### 2. Dual Simplex Solver (`milp_engine/src/dual_simplex_solver.cpp`)
- **Algorithm:** Tableau-based dual simplex algorithm for continuous linear programs.
- **Functionality:**
  - Maintains dual feasibility while iterating toward primal feasibility.
  - Delivers basic solutions (vertex points).
  - Serves as the subproblem LP relaxation solver at each node of the branch-and-cut tree.

### 3. Branch-and-Cut Engine (`milp_engine/`)
- **Algorithm:** Branch-and-bound search with dynamic cutting plane separation for Mixed-Integer Linear Programs (MILP).
- **Components:**
  - `branch_and_bound.cpp`: Priority queue node selection and tree search management.
  - `gomory_cuts.cpp`: Generates Gomory fractional cuts from fractional simplex tableau rows to separate non-integer LP relaxation points.
  - `branching_rules.cpp`: Variable selection rules (most-fractional branching).
  - `heuristics.cpp`: Primal feasibility pump and rounding heuristics to find integer feasible solutions during tree search.

### 4. QP Engine (`qp_engine/`)
- **Algorithm:** Alternating Direction Method of Multipliers (ADMM) for convex Quadratic Programming.
- **Objective Mathematical Convention:**
  - Model IR stores quadratic objective terms as:
    $$f(x) = \text{offset} + \sum c_i x_i + \sum q_{ij} x_i x_j$$
    where $q_{ij}$ is the direct coefficient of $x_i x_j$ without an implicit $1/2$ factor.
  - The QP adapter (`qp::fromModel`) converts this to standard quadratic form:
    $$\min \frac{1}{2} x^T P x + q^T x + \text{offset}$$
    with diagonal entries $P_{ii} = 2 q_{ii}$ and off-diagonal entries $P_{ij} = P_{ji} = q_{ij}$ (undoubled, since $\frac{1}{2}(P_{ij} + P_{ji}) = q_{ij}$). Maximization negates both $P$ and $q$.
- **Components:**
  - `admm_solver.cpp`: Alternating minimization over primal variables and constraint slacks with dual multiplier updates.
  - `kkt_solver.cpp`: Factorization of augmented KKT linear systems.
  - `scaling.cpp`: Matrix equilibration and adaptive penalty parameter ($\rho$) updating.

---

## 4. Key Architectural Decisions

### Decoupling Presolve and Postsolve
Presolve and postsolve are completely decoupled:
- Presolve only transforms `model::Model` and outputs an audit log (`PresolveResult`).
- Postsolve takes that log and the reduced solution vector to reconstruct original-space coordinates.
This separation allows presolve rules to be tested independently from solver engines, and allows engines to solve the reduced problem without needing any knowledge of how presolve occurred.

### Scaling Location & Preserving Presolve Reductions
A frequent source of numerical error in mathematical programming pipelines is redundant or conflicting matrix scaling.
- **Presolve does model reduction, not matrix scaling.** It eliminates variables and rows, tightens bounds, and preserves integer coefficient properties for MILP cuts.
- **Numerical engines handle their own scaling.** PDLP performs Ruiz equilibration and diagonal preconditioning inside `pdlp_engine/src/preconditioner.cpp`. The QP engine handles its own matrix scaling internally.

### Single-Presolve Invariant
The orchestrator classifies the original model once, runs presolve once, and passes the reduced model into `solveReduced()`. Nesting or repeating presolve passes is strictly forbidden because postsolve metadata relies on the exact index mappings established during the initial presolve.

---

## 5. Current Implementation vs. Planned Work

| Subsystem | Implemented in Current Codebase | Planned Work |
| :--- | :--- | :--- |
| **Input Format** | MPS (Fixed & Free format) | LP format (`.lp`) |
| **Model IR** | Linear, Quadratic terms ($q_{ij} x_i x_j$), Continuous/Integer/Binary types | Conic constraints (SOCP) |
| **Presolve** | Fixed variables, Singleton equality/inequality, Bound tightening with provenance, Redundant rows | Variable substitution, Binary probing, Duplicate row/column detection |
| **Engines** | PDLP (First-Order LP), Dual Simplex, Branch-and-Cut (MILP), ADMM (QP) | Infeasible interior-point barrier solver |
| **Postsolve** | Full primal reconstruction, Objective re-evaluation, Dual reconstruction (shadow prices & reduced costs), Bound provenance validation, Fail-closed error handling | Support for dual reconstruction through variable substitution passes |
| **CLI & UX** | Interactive menu, Batch solve mode, Mascot banner, Solve dashboard, Solution export, TTY detection, CMake install | Real-time solve progression streaming |
