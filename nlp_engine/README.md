# Smooth nonlinear programming

`nlp_engine` solves continuous smooth minimization problems

```
minimize f(x)
subject to lower_c <= c(x) <= upper_c
           lower_x <= x <= upper_x
```

It is a local, first-order elastic SQP implementation.

Scope, stated once and not qualified away elsewhere in this document:

- This is a **local, first-order** NLP solver.
- `FirstOrderStationary` means the returned point and multipliers satisfy the
  original-unit first-order KKT tests. It is **not global optimality**, and not
  a second-order (local minimum) certificate - a saddle point or a maximum
  satisfies the same tests.
- `FirstOrderStationary` means only that the returned point satisfies the
  implemented original-unit KKT residual checks. It does **not** imply LICQ,
  MFCQ or any other constraint qualification, and the multipliers are **not
  guaranteed to be unique** (the dependent-equality test is an example).
- The solver provides **no infeasibility certificate**. `NoProgress` and
  `SubproblemFailure` mean this method stopped making progress, never that the
  problem is proven infeasible.
- **No parity with IPOPT or SNOPT** is claimed, in maturity, performance,
  robustness or problem coverage.
- **Large-scale performance has not been established.** Measured coverage is
  small-to-moderate problems (the largest in the suite is n=300); above
  `denseBfgsLimit` the curvature model changes and is untested at scale.

## Architecture and repository fit

The existing `model::Model` represents linear constraints and polynomial
objectives of degree at most two. Its presolver, classifier and postsolver rely
on that structure; the MPS parser cannot encode general nonlinear expressions.
None of these affine-model paths is repurposed for nonlinear data.

The existing `qp_engine` already has CSR/CSC matrices, ADMM, Ruiz equilibration,
sparse/dense Cholesky selection, and regularization. Its matrix convention is
`0.5*d'P*d + q'd`, with **full symmetric** P and `lower <= A*d <= upper`.
Its dual convention is `P*d + q + A'lambda = 0`. NLP uses that convention
throughout, rather than the affine public API's shadow-price convention.

There is no general sparse indefinite LDL factorization, inertia control or
symmetric ordering interface. A primal-dual interior-point implementation would
require substantial new linear algebra. SQP instead reuses the convex QP engine
and its existing safeguards. The new module does not modify any engine source.
The dual simplex's dense basis machinery is not suitable for general NLP KKT
systems. PDLP's linear objectives cannot represent curvature subproblems.

The integration is an overload of `solver::solve` for `nlp::Problem`, returning
`solver::NlpSolveResult` (an `nlp::Result` with classification and engine metadata).
Shared classification identifies `ProblemClass::NLP`; dispatch selects
`Engine::Nlp` (`nlp_sqp`). The common CLI argument parser routes
`optimsolver solve model.nlp` or `optimsolver solve-nlp file` to this pipeline.
Interactive sessions can load, inspect, solve, and export NLP models, and switch
between MPS and NLP without retaining the wrong model or result.

This preserves the affine API's meaning of `Optimal` and its presolve/postsolve
contracts. `--solver nlp` is supported for nonlinear input; forcing an affine
engine on NLP or the NLP engine on an affine model is explicitly rejected.
MPS is never reinterpreted as a nonlinear expression format. Future algorithms can consume the same `Problem` interface;
MINLP requires a separate discrete model and a relaxation/certification layer.

## C++ API

Link `solver_orchestrator` for the unified API below. Direct `nlp::Solver`
users can link only `nlp_engine`.

```cpp
#include "solver/orchestrator.h"
#include "nlp/derivative_check.h"

using namespace nlp;
auto x = variable(0), y = variable(1);
Model model({{}, {}}, square(x) + square(y), {x + y}, {{2, 2}});
Options options;
options.tolerance = 1e-6;
auto check = checkDerivatives(model, {0.5, 0.5});
auto result = solver::solve(model, {0, 0}, options);
// result.primal = [1,1], constraintMultipliers = [-2].
```

`Bounds{}` means free, unlike the affine model's default nonnegative variable.
Infinity is allowed only as a missing lower/upper side. Fixed variables, ranged
constraints, equality constraints, and zero-variable models are supported.
Minimization only: negate a maximization objective explicitly (including any
offset), and negate its reported value/multipliers when interpreting that model.

Expressions form an immutable DAG, compiled in topological order. Reverse AD
runs per output on its reachable subgraph; Jacobians use the QP sparse matrix.
Shared subexpressions and repeated variable nodes accumulate derivatives.
Evaluation scratch is local to a call. Model evaluation and independent solver
instances are reentrant; user callbacks must provide their own thread safety.

Supported operations: constants, variables, `+ - * /`, unary minus, `exp`, `log`,
`sin`, `cos`, `sqrt`, `square`. Compose integer powers with multiplication.
There is no nonsmooth `abs`, min/max, conditional expression, or arbitrary code
execution. Values and first derivatives must be finite: e.g. `sqrt(0)` fails
because its derivative is singular. Unused expressions are not evaluated.
There is no exact Hessian AD, no exact Hessian callback and no Hessian-vector
product callback; see *No second-order information* below.

For external simulators, derive from `Problem`: provide bound vectors and an
`Evaluation` containing f, gradient, constraints and a valid sparse Jacobian.
Dimensions, sparse structure, and finiteness are validated on every evaluation.
Throw `std::domain_error` for a recoverable domain failure. Other exceptions
produce a diagnostic failure. `checkDerivatives` compares gradients and every
Jacobian column with central differences and identifies the worst entry.
It is opt-in and costs 2*n+1 evaluations; a callback can still supply internally
consistent but mathematically wrong derivatives unless independently checked.

## Numerical method

1. Validate input/options, project the supplied start into variable bounds,
   and evaluate. A start outside the function domain returns `EvaluationFailure`;
   the solver does not guess a replacement start.
2. Solve the convex SQP model with damped BFGS Lagrangian curvature and diagonal
   regularization. Work in `x = x_current + D*d`, where D contains user-supplied
   positive variable units. Constraint scales are fixed from the initial scaled
   Jacobian row norms, bounded below by 1e-8. The QP adds Ruiz equilibration.
3. If the hard linearization cannot be solved to the inner tolerance, retry with
   two nonnegative elastic variables per nonlinear row:
   `lower-c <= J*D*d + e_plus - e_minus <= upper-c`.
   Elastic variables receive a positive L1 penalty and small quadratic
   regularization. Variable bounds remain hard. Rank-deficient Jacobians do not
   require an unregularized equality-KKT inverse.
4. Independent residual validation of the constructed QP: recompute the QP
   primal and stationarity residuals from the QP data the solver assembled, in
   unscaled SQP coordinates. Inner solver status alone is insufficient. This
   checks the inner solve only; it does **not** independently rebuild the SQP
   model from the original nonlinear problem, so an error in constructing the
   QP (for example, wrong gradient or Jacobian scaling) is not detected by this
   gate. Such an error cannot by itself produce `FirstOrderStationary`, which is
   decided only by the original-unit KKT tests on a fresh evaluation of the
   nonlinear problem. Inner tolerances tighten with curvature magnitude. QP time
   limits receive the remaining outer budget; inner factorization/evaluation
   calls are not preemptible.
5. Increase the L1 merit penalty using row-scaled multiplier estimates. Backtrack
   on `f + penalty * sum(scaled constraint violations)` using Armijo decrease
   of the linearized merit model. Reject domain errors/nonfinite evaluations.
   Project trial points to hard bounds to remove QP roundoff.
6. Update curvature using Lagrangian gradients at both points with the **same**
   current multipliers. Powell damping preserves positive curvature for
   nonconvex objectives; unsafe updates reset to the identity. The curvature
   representation depends on the problem size; see *Curvature* below, because
   crossing the threshold changes the algorithm and not merely a budget.

The merit line search is the chosen globalization strategy; there is no filter
or second-order correction. Elastic steps provide feasibility recovery when a
linearization is inconsistent. No-descent cases increase the penalty within a
cap, then report `NoProgress`; stationary infeasibility is not misreported as
proven infeasibility. The method can stall at degenerate starts (e.g. x=0 in
x^2=1), at constraint qualification failures, or from the Maratos effect.

## Curvature: bounded memory, not limited-memory BFGS

The curvature approximation has two regimes, and they are **not equivalent**.

- While `n <= denseBfgsLimit` the solver uses **dense damped BFGS**, storing the
  full `n x n` approximation. This costs **O(n^2) memory** and O(n^2) work per
  update.
- The default threshold is **256**, so dense BFGS holds up to 65,536 doubles.
  The option is capped at 2048.
- **Above the threshold the solver does not use BFGS at all.** It switches to a
  single **diagonal spectral** curvature estimate: one scalar
  `clamp(y'y / s'y, 1e-6, 1e6)` repeated on the diagonal.
- That estimate carries **no off-diagonal information**. For strongly coupled
  nonlinear problems, where the Hessian's cross terms drive the step, this can
  make convergence **substantially slower** - more iterations, more evaluations
  and more QP solves - and a problem that converges below the threshold may
  reach `IterationLimit` above it. Raising `denseBfgsLimit` changes the
  algorithm, not just a limit.
- **This is not a limited-memory BFGS implementation.** No history of `(s, y)`
  pairs is retained above the threshold, and there is no L-BFGS or L-SR1
  two-loop recursion anywhere in this module. Limited-memory curvature is listed
  under remaining work below; it is not implemented.

The large-dimensional path is therefore a deliberate memory bound with a real
convergence cost, not a drop-in substitute for dense BFGS. No claim is made
about its performance on large coupled problems; see *Validation* below for what
has actually been measured.

## No second-order information

- There is **no exact Hessian callback**, and no way to supply one. `Problem`
  requests only f, its gradient, c and the Jacobian.
- There is **no Hessian-vector product callback**.
- All curvature is a secant approximation built from first derivatives only.
- There is **no second-order optimality certificate**. `FirstOrderStationary`
  reports that the original-unit first-order KKT tests hold at the returned
  point. It does **not** imply a local minimum, and it does **not** imply a
  global minimum.
- A first-order stationary **saddle point or maximum satisfies exactly the same
  tests** and is reported with the same status. `Solver().solve(Model({{}},
  -square(x)), {0})` returns `FirstOrderStationary` at the maximum x=0, and the
  test suite asserts that. Nothing in the reported result distinguishes a
  minimum from a saddle, because no curvature test is performed at the solution.

## Initial point

- The supplied start is **projected onto the variable bounds** before the first
  evaluation: `x_j <- clamp(x_j, lower_j, upper_j)`. Every reported point,
  objective and residual refers to the projected start, never the supplied one.
- **Nonlinear constraint violations are not repaired at initialization.** No
  feasibility phase runs before the first iteration. The first reported iterate
  is the projected start carrying whatever row violation it has. Elastic
  restoration runs later, and only when a linearization proves inconsistent.
- If the **projected** point is outside the evaluation domain, the solve returns
  `EvaluationFailure` - including when the supplied point was inside the domain
  and only the projection moved it out. Projection strictly precedes evaluation.
- The solver **does not invent a new nonlinear-feasible starting point.** It
  never searches for an alternative start and never reports one.
- A start whose feasible region is reachable only by first *increasing* the
  constraint violation can stall at `NoProgress`. That is a limitation of a
  local merit-based method, not an infeasibility finding.

## Results and tolerances

At the returned point, success requires all three absolute infinity-norm tests
in **original units**, each at most `Options::tolerance`:

- bound and nonlinear constraint violation;
- `gradient_f + J' * constraintMultipliers + boundMultipliers`;
- multiplier times slack on the corresponding signed side (and rejection of
  multipliers pointing toward a missing bound).

Upper-side multipliers are positive; lower-side multipliers are negative;
equality/fixed-variable multipliers are unrestricted and can be nonunique.
No constraint qualification is checked or assumed, so when LICQ fails (as with
dependent equalities) any multipliers passing the tests may be returned.
These are KKT estimates, not globally valid sensitivity information. Reports
include objective, residuals, iteration/evaluation/QP counts, rejected trials,
elastic subproblems and elapsed time. A callback can inspect each current
iterate and stop by returning false.

`hasPrimal` means a complete finite **evaluated iterate**, which may be
infeasible; check `feasible` separately. Failure retains the last accepted
iterate and its diagnostics. `IterationLimit`, `TimeLimit`, `UserStopped`,
`NoProgress`, `SubproblemFailure`, `EvaluationFailure`, `InvalidProblem`, and
`NumericalFailure` are distinct. There is deliberately no global `Optimal`,
`Unbounded`, or `Infeasible` status. Reaching a first-order stationary saddle
point is possible and is reported with the same explicitly first-order status.

## CLI and format

```
optimsolver solve nlp_engine/examples/rosenbrock.nlp
optimsolver solve model.nlp --solver nlp --output solution.txt
optimsolver solve-nlp model.nlp --tolerance 1e-7 --iterations 1000 \
  --time-limit 30 --json result.json
```

JSON is written to stdout (and optionally with `--json file`) through the shared
reporting API. The explicit `optimsolver.nlp.v1` schema includes input path/hash,
options, classification, requested/executed engines, stage applicability and
original-unit residuals. Cached interactive models omit the file hash because
the file on disk may have changed since loading. Nonfinite/unavailable scalar
fields are JSON null. `--output file` writes a labeled iterate/status listing;
its feasibility flag must be checked before treating it as a feasible solution.
Output files may not alias the input or each other.

Exit codes: 0 stationary, 2 other solver termination, 1 command/input/output
error. Common options may precede or follow the model path; `-h` and `--help`
work for both commands. NLP-specific options are not silently accepted for MPS.
`--threads` and `--dump-model` are explicitly unsupported for NLP. Existing MPS
solve behavior remains unchanged. The MPS benchmark verifier is affine-specific;
use the nonlinear reference runner below to check nonlinear residuals and local
termination without interpreting them as global `Optimal` results.

The versioned whitespace-delimited format is:

```
nlp 1
variables N
lower upper initial          # N records, numeric +/-inf allowed for bounds
nodes K
var variable_index           # node 0; indices are zero-based
const finite_value           # node 1
sub earlier_node earlier_node
square earlier_node          # ... K records
objective node_index
constraints M
lower upper node_index       # M records
```

The comments above explain the format; actual files do **not** support comments.
Binary node names: `add sub mul div`; unary: `neg exp log sin cos sqrt square`.
Node references must name earlier nodes; trailing input and unknown operations
are rejected. N, M and K are each limited to one million on input. The reader
constructs a minimization model and start; semantic bounds/options validation
occurs at solve time. This is a small interchange format, not AMPL `.nl`.

## Validation and remaining work

```
cmake -S . -B build-nlp -DCMAKE_BUILD_TYPE=Release \
  -DQP_BUILD_TESTS=ON -DPDLP_BUILD_TESTS=ON -DNLP_REFERENCE_TESTS=ON
cmake --build build-nlp -j
ctest --test-dir build-nlp --output-on-failure
python3 nlp_engine/benchmarks/reference.py --binary build-nlp/optimsolver
```

SciPy is needed only for the optional reference test. C++ and CLI tests have no
new external dependencies. Release checks use throwing assertions, so NDEBUG
cannot disable them.

`nlp_tests` covers central-difference AD checks, HS71, Rosenbrock, active/ranged
sides, fixed variables, duplicate equalities, domain backtracking, restoration,
nonconvexity, scaling, sparse diagonal curvature, concurrent solves, invalid
callbacks, NaNs, parser failures and budgets.

`nlp_elastic_kkt_tests` covers the elastic formulation, the initial point and
the multiplier convention, and re-derives the KKT tests **independently**: it
re-evaluates the original nonlinear problem at each returned point and
recomputes feasibility, `grad f + J' lambda + z` and signed complementarity from
scratch, so a run that reported converged residuals while returning a point that
does not satisfy them fails. Specifically:

- lower-side and upper-side violations of a nonlinear inequality, each repaired
  through the elastic column of the matching sign;
- ranged rows violated on either side, terminating on the *opposite* side, which
  pins the multiplier sign to the **active** side rather than the violated one;
- equality restoration with both a positive and a negative multiplier;
- a contrast case proving restoration is engaged only when the hard
  linearization is inconsistent, not unconditionally;
- the per-iteration violation trace, showing the original nonlinear violation
  actually falls from the start's value to within tolerance;
- projection of an out-of-bounds start, projection strictly preceding the first
  evaluation, non-repair of nonlinear rows at initialization, and a stall that
  reports `NoProgress` rather than infeasibility;
- positive/negative equality multipliers, dependent equalities, a fixed variable
  combined with an equality row, active lower and upper bounds, both active
  ranged sides, and inactive rows/bounds giving zero multipliers;
- independent residual validation of the constructed QP rejecting an inner
  solve that returned `Optimal`. ADMM's own test is relative and measured on its
  equilibrated system, so a badly conditioned linearization satisfies it while
  its absolute unscaled residual is ~1.6e-5; the gate overrules the inner
  status. The gate validates the returned solution against the QP as built; it
  does not reconstruct that QP from the nonlinear problem. Removing
  the residual half of the gate makes that case report `FirstOrderStationary`,
  which is what the test prevents.

Each of these was confirmed to fail against a deliberately mutated solver:
dropping either elastic column breaks the corresponding side's restoration,
dropping the residual gate produces the false `FirstOrderStationary` above, and
removing the bound projection changes the first reported iterate.
The reference script independently recomputes objectives and feasibility and
compares 15 problems with SLSQP and analytic objective values. Optional JSON
reports record the SciPy version/seed; timings are machine-specific, not a
performance claim.

Priorities before broad production deployment: a much larger diverse benchmark
corpus (CUTEst and application problems), fuzzing and resource stress tests,
stronger restoration and second-order correction, limited-memory curvature,
QP warm starts/symbolic reuse, explicit objective scaling, sparse indefinite
linear algebra with ordering/inertia for an optional interior-point backend,
exact Hessian/Hessian-vector callbacks, and second-order checks. The existing
QP normal-equation Cholesky can amplify conditioning and produce fill; sparse
input does not guarantee low memory usage. No large-scale performance or
infeasibility-certification claim is made.

Reference semantics: [IPOPT output/status documentation](https://coin-or.github.io/Ipopt/OUTPUT.html)
explains local infeasibility versus solver failure; [SciPy SLSQP documentation](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html)
describes the independent comparator and its tolerances. No external solver is
required or called by the NLP engine itself.
