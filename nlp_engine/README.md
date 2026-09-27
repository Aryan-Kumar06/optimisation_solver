# Smooth nonlinear programming

`nlp_engine` solves continuous smooth minimization problems

```
minimize f(x)
subject to lower_c <= c(x) <= upper_c
           lower_x <= x <= upper_x
```

It is a local, first-order elastic SQP implementation. `FirstOrderStationary`
means the reported point and multipliers satisfy the original-unit KKT tests.
It is not a global optimum, second-order minimum, or infeasibility certificate.
This implementation has regression and reference coverage, but is not a claim
of parity with the maturity, performance, or problem coverage of IPOPT/SNOPT.

## Architecture and repository fit

The existing `model::Model` represents linear constraints and polynomial
objectives of degree at most two. Its presolver, classifier and postsolver rely
on that structure; the MPS parser cannot encode general nonlinear expressions.
The similarly named `nlp_frontend` is a **natural-language** LP/MILP frontend,
not a nonlinear evaluator. None of these paths is repurposed for nonlinear data.

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
There is no exact Hessian AD in this version.

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
4. Independently validate the QP primal and stationarity residuals in the SQP
   coordinates. Inner solver status alone is insufficient. Inner tolerances
   tighten with curvature magnitude. QP time limits receive the remaining
   outer budget; inner factorization/evaluation calls are not preemptible.
5. Increase the L1 merit penalty using row-scaled multiplier estimates. Backtrack
   on `f + penalty * sum(scaled constraint violations)` using Armijo decrease
   of the linearized merit model. Reject domain errors/nonfinite evaluations.
   Project trial points to hard bounds to remove QP roundoff.
6. Update curvature using Lagrangian gradients at both points with the **same**
   current multipliers. Powell damping preserves positive curvature for
   nonconvex objectives; unsafe updates reset to the identity. Dense BFGS uses
   O(n^2) storage only through `denseBfgsLimit` (default 256, capped at 2048).
   Larger models use a bounded scalar spectral diagonal; their convergence can
   be much slower on strongly coupled curvature.

The merit line search is the chosen globalization strategy; there is no filter
or second-order correction. Elastic steps provide feasibility recovery when a
linearization is inconsistent. No-descent cases increase the penalty within a
cap, then report `NoProgress`; stationary infeasibility is not misreported as
proven infeasibility. The method can stall at degenerate starts (e.g. x=0 in
x^2=1), at constraint qualification failures, or from the Maratos effect.

## Results and tolerances

At the returned point, success requires all three absolute infinity-norm tests
in **original units**, each at most `Options::tolerance`:

- bound and nonlinear constraint violation;
- `gradient_f + J' * constraintMultipliers + boundMultipliers`;
- multiplier times slack on the corresponding signed side (and rejection of
  multipliers pointing toward a missing bound).

Upper-side multipliers are positive; lower-side multipliers are negative;
equality/fixed-variable multipliers are unrestricted and can be nonunique.
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
cannot disable them. Coverage includes central-difference AD checks, HS71,
Rosenbrock, active/ranged sides, fixed variables, duplicate equalities, domain
backtracking, restoration, nonconvexity, scaling, sparse diagonal curvature,
concurrent solves, invalid callbacks, NaNs, parser failures and budgets.
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
