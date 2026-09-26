# SIH 2026 Project Presentation

This document tracks the slide deck outline and reference materials for the SIH 2026 final presentation.

---

## Project & Problem Statement Information

- **Organization:** Mangalore Refinery and Petrochemicals Limited (MRPL)
- **Problem Statement ID:** SIH26119
- **Problem Statement Title:** Indigenous GPU-Accelerated Optimization Solver (Sovereign Alternative to Express / CPLEX)
- **Category:** Software
- **Theme:** Smart Automation
- **Repository:** `https://github.com/RaghavGupta2910/optimisation_solver`

### Team Members
- **Raghav Gupta** — Team Lead — System Architecture, Solver Orchestration & CLI
- **Harehar Narayan Seth** — Lead Numerical Solver Development & Optimization Algorithms
- **Aryan Kumar** — Solver Engine Development & Numerical Methods
- **Preetish Attray** — MILP / Branch-and-Cut Development
- **Kabir Pahwa** — Postsolve, Dual Reconstruction & Solution Validation
- **Sarisha Jhinghan** — Presolve, Model Processing & Testing

---

## Slide Deck Links

- **Cloud Backup Link (Google Drive / OneDrive):**
- https://drive.google.com/file/d/1o24vzDsYfrNa9i11FnNQFsK9G2Me1p-o/view

---

## Slide Structure & Content Outline

1. **Title & Team:** Official project title, MRPL Problem Statement ID (SIH26119), team members, and technical roles.
2. **Problem Context:** Strategic and operational need for indigenous mathematical optimization solvers in refinery scheduling, supply chains, and resource allocation to eliminate dependence on expensive proprietary commercial tools (Xpress, CPLEX).
3. **Architecture & Design Principles:**
   - Decoupled, modular architecture: `MPS Ingestion → Model IR → Validation → Original Classification → Presolve (Once) → Dispatch → Solver Engine → Postsolve (Once) → Feasibility Verification`.
   - The Single-Presolve Invariant: coordinate transformations are logged explicitly in metadata and inverted during postsolve.
4. **Presolve & Postsolve Invertibility:**
   - Reduction passes: fixed variable removal, singleton equality rows, bound tightening, and redundant constraint elimination.
   - Dual postsolve: reconstructing shadow prices and reduced costs by reversing bound-tightening provenance logs, with stationarity and complementary slackness verification.
   - Fail-closed safety: rejecting non-finite values (`NaN`, `±Inf`) and invalid dual reconstructions.
5. **Numerical Engines:**
   - PDLP: First-order PDHG algorithm with Ruiz equilibration and adaptive step sizes for continuous LPs.
   - Dual Simplex: Tableau-based solver providing basic solutions and LP relaxation solves.
   - Branch-and-Cut: Branch-and-bound tree with Gomory fractional cuts and primal rounding heuristics for MILP.
   - QP Engine: ADMM algorithm with augmented KKT factorizations for convex quadratic objectives.
6. **User Experience & CLI:**
   - Interactive terminal menu interface for inspecting models, configuring parameters, and running solves.
   - Direct command-line batch mode (`optimsolver solve <model.mps> [options]`).
   - Standard CMake install target for terminal deployment.
7. **Automated Verification:** Test coverage across presolve cascades, dual postsolve checks, engine conventions, and end-to-end solve pipelines.
8. **Roadmap:** Future GPU/CUDA acceleration for linear algebra kernels, `.lp` file format support, and parallel branch-and-bound search.
