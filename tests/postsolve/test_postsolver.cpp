// Postsolve correctness tests.
//
// Conventions under test (see model/model.h and presolve/transformation.h):
//   f(x) = objective.offset + sum(c_i x_i) + sum(q_ij x_i x_j)
//   QuadraticTerm.value is the DIRECT coefficient of x_i * x_j -- there is
//   NO implicit 1/2 factor, for diagonal or off-diagonal terms.

#include "model/model.h"
#include "presolve/presolve_result.h"
#include "presolve/presolver.h"
#include "presolve/transformation.h"
#include "postsolve/postsolver.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr double INF = std::numeric_limits<double>::infinity();
constexpr double EPS = 1e-6;

bool approx(double a, double b) { return std::abs(a - b) <= EPS; }

model::Variable makeVar(const std::string& name, double lower = 0.0,
                         double upper = INF,
                         model::VariableType type = model::VariableType::Continuous) {
  model::Variable v;
  v.name = name;
  v.lowerBound = lower;
  v.upperBound = upper;
  v.type = type;
  return v;
}

model::Constraint makeCon(const std::string& name, double lower, double upper) {
  model::Constraint c;
  c.name = name;
  c.lowerBound = lower;
  c.upperBound = upper;
  return c;
}

// Builds a trivial PresolveResult that maps every original variable/constraint
// to itself, with no transformations and nothing eliminated. Useful for
// exercising postsolve behavior (objective evaluation, validation, mapping
// checks) in isolation from the presolve algorithms.
presolve::PresolveResult makeIdentityResult(const model::Model& m) {
  presolve::PresolveResult r;
  r.model = m;
  r.originalVariables = m.variables.size();
  r.originalConstraints = m.constraints.size();
  r.presolvedVariables = m.variables.size();
  r.presolvedConstraints = m.constraints.size();

  r.postsolve.presolvedToOriginalVar.resize(m.variables.size());
  r.postsolve.originalToPresolvedVar.assign(m.variables.size(), -1);
  for (std::size_t i = 0; i < m.variables.size(); ++i) {
    r.postsolve.presolvedToOriginalVar[i] = i;
    r.postsolve.originalToPresolvedVar[i] = static_cast<int>(i);
  }

  r.postsolve.presolvedToOriginalConstraint.resize(m.constraints.size());
  r.postsolve.originalToPresolvedConstraint.assign(m.constraints.size(), -1);
  for (std::size_t i = 0; i < m.constraints.size(); ++i) {
    r.postsolve.presolvedToOriginalConstraint[i] = i;
    r.postsolve.originalToPresolvedConstraint[i] = static_cast<int>(i);
  }

  return r;
}

}  // namespace

// ============================================================
// A. No transformations
// ============================================================
void test_no_transformations() {
  model::Model model;
  model.name = "NoTransform";
  model.variables.push_back(makeVar("x0", 0.0, 10.0));
  model.variables.push_back(makeVar("x1", 0.0, 10.0));

  model.objective.offset = 5.0;
  model.objective.linearTerms.push_back({0, 1.0});
  model.objective.linearTerms.push_back({1, 2.0});

  model::Constraint con = makeCon("c0", -INF, 10.0);
  con.linearTerms.push_back({0, 1.0});
  con.linearTerms.push_back({1, 1.0});
  model.constraints.push_back(con);

  auto presolveRes = makeIdentityResult(model);
  std::vector<double> presolvedSolution = {3.0, 4.0};

  postsolve::Postsolver postsolver;
  auto result = postsolver.process(model, presolveRes, presolvedSolution);

  assert(result.isSuccess());
  assert(result.primalSolution.size() == 2);
  assert(approx(result.primalSolution[0], 3.0));
  assert(approx(result.primalSolution[1], 4.0));
  // 5 + 1*3 + 2*4 = 16
  assert(approx(result.originalObjectiveValue, 16.0));

  std::cout << "[PASSED] test_no_transformations\n";
}

// ============================================================
// B. Fixed continuous variable
// ============================================================
void test_fixed_continuous_variable() {
  model::Model model;
  model.name = "FixedContinuous";
  model.variables.push_back(makeVar("x0", 5.0, 5.0));  // fixed
  model.variables.push_back(makeVar("x1", 0.0, 10.0));

  model.objective.linearTerms.push_back({0, 2.0});
  model.objective.linearTerms.push_back({1, 3.0});

  presolve::Presolver presolver;
  auto presolveRes = presolver.run(model);
  assert(!presolveRes.infeasible);
  assert(presolveRes.postsolve.fixedVariables.size() == 1);

  std::vector<double> presolvedSolution = {4.0};  // x1 = 4

  postsolve::Postsolver postsolver;
  auto result = postsolver.process(model, presolveRes, presolvedSolution);

  assert(result.isSuccess());
  assert(approx(result.primalSolution[0], 5.0));
  assert(approx(result.primalSolution[1], 4.0));
  // 2*5 + 3*4 = 22
  assert(approx(result.originalObjectiveValue, 22.0));

  std::cout << "[PASSED] test_fixed_continuous_variable\n";
}

// ============================================================
// C. Fixed integer variable
// ============================================================
void test_fixed_integer_variable() {
  model::Model model;
  model.name = "FixedInteger";
  model.variables.push_back(
      makeVar("x0", 3.0, 3.0, model::VariableType::Integer));  // fixed
  model.variables.push_back(makeVar("x1", 0.0, 10.0));

  model.objective.linearTerms.push_back({0, 1.0});
  model.objective.linearTerms.push_back({1, 1.0});

  presolve::Presolver presolver;
  auto presolveRes = presolver.run(model);
  assert(!presolveRes.infeasible);

  std::vector<double> presolvedSolution = {2.5};  // x1 = 2.5

  postsolve::Postsolver postsolver;
  auto result = postsolver.process(model, presolveRes, presolvedSolution);

  assert(result.isSuccess());
  assert(approx(result.primalSolution[0], 3.0));
  assert(approx(result.primalSolution[1], 2.5));
  assert(approx(result.originalObjectiveValue, 5.5));

  std::cout << "[PASSED] test_fixed_integer_variable\n";
}

// ============================================================
// D. Fixed binary variable
// ============================================================
void test_fixed_binary_variable() {
  model::Model model;
  model.name = "FixedBinary";
  model.variables.push_back(
      makeVar("x0", 1.0, 1.0, model::VariableType::Binary));  // fixed
  model.variables.push_back(makeVar("x1", 0.0, 5.0));

  model.objective.linearTerms.push_back({0, 10.0});
  model.objective.linearTerms.push_back({1, 1.0});

  presolve::Presolver presolver;
  auto presolveRes = presolver.run(model);
  assert(!presolveRes.infeasible);

  std::vector<double> presolvedSolution = {2.0};

  postsolve::Postsolver postsolver;
  auto result = postsolver.process(model, presolveRes, presolvedSolution);

  assert(result.isSuccess());
  assert(approx(result.primalSolution[0], 1.0));
  assert(approx(result.primalSolution[1], 2.0));
  assert(approx(result.originalObjectiveValue, 12.0));

  std::cout << "[PASSED] test_fixed_binary_variable\n";
}

// ============================================================
// E. Removed variable reconstruction + diagonal quadratic term
//    (end-to-end presolve -> postsolve)
// ============================================================
void test_removed_variable_reconstruction_qp() {
  model::Model model;
  model.name = "RemovedVarQP";

  model.variables.push_back(makeVar("x0", 5.0, 5.0));  // fixed, will be removed
  model.variables.push_back(makeVar("x1", 0.0, 10.0));
  model.variables.push_back(makeVar("x2", 0.0, 5.0, model::VariableType::Integer));

  const int x0 = 0, x1 = 1, x2 = 2;
  model.objective.linearTerms.push_back({x0, 2.0});
  model.objective.linearTerms.push_back({x1, 3.0});
  model.objective.quadraticTerms.push_back({x1, x1, 1.0});  // diagonal term

  presolve::Presolver presolver;
  auto presolveRes = presolver.run(model);

  assert(!presolveRes.infeasible);
  assert(presolveRes.postsolve.presolvedToOriginalVar.size() == 2);
  assert(presolveRes.postsolve.presolvedToOriginalVar[0] == 1);
  assert(presolveRes.postsolve.presolvedToOriginalVar[1] == 2);

  assert(presolveRes.postsolve.originalToPresolvedVar.size() == 3);
  assert(presolveRes.postsolve.originalToPresolvedVar[0] == -1);
  assert(presolveRes.postsolve.originalToPresolvedVar[1] == 0);
  assert(presolveRes.postsolve.originalToPresolvedVar[2] == 1);

  assert(presolveRes.postsolve.fixedVariables.size() == 1);
  assert(presolveRes.postsolve.fixedVariables[0].originalIndex == 0);
  assert(approx(presolveRes.postsolve.fixedVariables[0].fixedValue, 5.0));

  std::vector<double> presolvedSolution = {2.0, 3.0};  // x1 = 2.0, x2 = 3.0

  postsolve::Postsolver postsolver;
  auto result = postsolver.process(model, presolveRes, presolvedSolution);

  assert(result.isSuccess());
  assert(result.primalSolution.size() == 3);
  assert(approx(result.primalSolution[x0], 5.0));
  assert(approx(result.primalSolution[x1], 2.0));
  assert(approx(result.primalSolution[x2], 3.0));

  // No implicit 1/2 factor: 2*(5) + 3*(2) + 1*(2^2) = 10 + 6 + 4 = 20.0
  assert(approx(result.originalObjectiveValue, 20.0));

  std::cout << "[PASSED] test_removed_variable_reconstruction_qp\n";
}

// ============================================================
// F. Invalid mapping: incorrect mapping size
// ============================================================
void test_invalid_mapping_incorrect_size() {
  model::Model model;
  model.variables.push_back(makeVar("x0"));
  model.variables.push_back(makeVar("x1"));

  auto presolveRes = makeIdentityResult(model);
  // Corrupt: claim 2 presolved variables but only provide 1 mapping entry.
  presolveRes.presolvedVariables = 2;
  presolveRes.postsolve.presolvedToOriginalVar = {0};

  std::vector<double> presolvedSolution = {1.0};

  postsolve::Postsolver postsolver;
  auto result = postsolver.process(model, presolveRes, presolvedSolution);

  assert(!result.isSuccess());
  assert(result.status == postsolve::PostsolveStatus::InvalidMapping);

  std::cout << "[PASSED] test_invalid_mapping_incorrect_size\n";
}

// ============================================================
// G. Invalid mapping: duplicate original variable index
// ============================================================
void test_invalid_mapping_duplicate() {
  model::Model model;
  model.variables.push_back(makeVar("x0"));
  model.variables.push_back(makeVar("x1"));

  auto presolveRes = makeIdentityResult(model);
  presolveRes.postsolve.presolvedToOriginalVar = {0, 0};  // duplicate

  std::vector<double> presolvedSolution = {1.0, 2.0};

  postsolve::Postsolver postsolver;
  auto result = postsolver.process(model, presolveRes, presolvedSolution);

  assert(!result.isSuccess());
  assert(result.status == postsolve::PostsolveStatus::InvalidMapping);

  std::cout << "[PASSED] test_invalid_mapping_duplicate\n";
}

// ============================================================
// H. Invalid mapping: out-of-range original variable index
// ============================================================
void test_invalid_mapping_out_of_range() {
  model::Model model;
  model.variables.push_back(makeVar("x0"));
  model.variables.push_back(makeVar("x1"));

  auto presolveRes = makeIdentityResult(model);
  presolveRes.postsolve.presolvedToOriginalVar = {0, 5};  // 5 is out of range

  std::vector<double> presolvedSolution = {1.0, 2.0};

  postsolve::Postsolver postsolver;
  auto result = postsolver.process(model, presolveRes, presolvedSolution);

  assert(!result.isSuccess());
  assert(result.status == postsolve::PostsolveStatus::InvalidMapping);

  std::cout << "[PASSED] test_invalid_mapping_out_of_range\n";
}

// ============================================================
// Additional mapping validation: invalid fixed-variable index
// ============================================================
void test_invalid_fixed_variable_index() {
  model::Model model;
  model.variables.push_back(makeVar("x0"));
  model.variables.push_back(makeVar("x1"));

  auto presolveRes = makeIdentityResult(model);

  presolve::FixedVariableRecord rec;
  rec.originalIndex = 99;  // out of range
  rec.name = "bogus";
  rec.fixedValue = 1.0;
  presolveRes.postsolve.fixedVariables.push_back(rec);

  std::vector<double> presolvedSolution = {1.0, 2.0};

  postsolve::Postsolver postsolver;
  auto result = postsolver.process(model, presolveRes, presolvedSolution);

  assert(!result.isSuccess());
  assert(result.status == postsolve::PostsolveStatus::InvalidMapping);

  std::cout << "[PASSED] test_invalid_fixed_variable_index\n";
}

// ============================================================
// Additional mapping validation: invalid removed-constraint index
// ============================================================
void test_invalid_removed_constraint_index() {
  model::Model model;
  model.variables.push_back(makeVar("x0"));
  model.constraints.push_back(makeCon("c0", -INF, 10.0));

  auto presolveRes = makeIdentityResult(model);

  presolve::RemovedConstraintRecord rec;
  rec.originalIndex = 50;  // out of range
  rec.name = "bogus";
  presolveRes.postsolve.removedConstraints.push_back(rec);

  std::vector<double> presolvedSolution = {1.0};

  postsolve::Postsolver postsolver;
  auto result = postsolver.process(model, presolveRes, presolvedSolution);

  assert(!result.isSuccess());
  assert(result.status == postsolve::PostsolveStatus::InvalidMapping);

  std::cout << "[PASSED] test_invalid_removed_constraint_index\n";
}

// ============================================================
// I. Original constraint violation detection
// ============================================================
void test_original_constraint_violation() {
  model::Model model;
  model.variables.push_back(makeVar("x0", 0.0, 10.0));

  model::Constraint con = makeCon("c0", -INF, 5.0);
  con.linearTerms.push_back({0, 1.0});
  model.constraints.push_back(con);

  auto presolveRes = makeIdentityResult(model);
  std::vector<double> presolvedSolution = {7.0};  // violates c0 <= 5

  postsolve::Postsolver postsolver;
  auto result = postsolver.process(model, presolveRes, presolvedSolution);

  assert(!result.isSuccess());
  assert(result.status == postsolve::PostsolveStatus::ConstraintViolation);
  assert(result.maxConstraintResidual > 0.0);

  std::cout << "[PASSED] test_original_constraint_violation\n";
}

// ============================================================
// J. Original bound violation detection
// ============================================================
void test_original_bound_violation() {
  model::Model model;
  model.variables.push_back(makeVar("x0", 0.0, 5.0));

  auto presolveRes = makeIdentityResult(model);
  std::vector<double> presolvedSolution = {7.0};  // violates upper bound

  postsolve::Postsolver postsolver;
  auto result = postsolver.process(model, presolveRes, presolvedSolution);

  assert(!result.isSuccess());
  assert(result.status == postsolve::PostsolveStatus::BoundViolation);
  assert(result.maxBoundResidual > 0.0);

  std::cout << "[PASSED] test_original_bound_violation\n";
}

// ============================================================
// K. Integrality violation detection
// ============================================================
void test_integrality_violation() {
  model::Model model;
  model.variables.push_back(
      makeVar("x0", 0.0, 10.0, model::VariableType::Integer));

  auto presolveRes = makeIdentityResult(model);
  std::vector<double> presolvedSolution = {2.5};  // not integral

  postsolve::Postsolver postsolver;
  auto result = postsolver.process(model, presolveRes, presolvedSolution);

  assert(!result.isSuccess());
  assert(result.status == postsolve::PostsolveStatus::IntegralityViolation);

  std::cout << "[PASSED] test_integrality_violation\n";
}

// ============================================================
// L. Objective offset
// ============================================================
void test_objective_offset() {
  model::Model model;
  model.variables.push_back(makeVar("x0", 0.0, 10.0));
  model.objective.offset = 7.5;

  auto presolveRes = makeIdentityResult(model);
  std::vector<double> presolvedSolution = {3.0};

  postsolve::Postsolver postsolver;
  auto result = postsolver.process(model, presolveRes, presolvedSolution);

  assert(result.isSuccess());
  assert(approx(result.originalObjectiveValue, 7.5));

  std::cout << "[PASSED] test_objective_offset\n";
}

// ============================================================
// M. Linear + quadratic objective
// ============================================================
void test_linear_plus_quadratic_objective() {
  model::Model model;
  model.variables.push_back(makeVar("x0", -INF, INF));
  model.variables.push_back(makeVar("x1", -INF, INF));
  model.objective.offset = 1.0;
  model.objective.linearTerms.push_back({0, 2.0});
  model.objective.quadraticTerms.push_back({0, 1, 3.0});  // off-diagonal

  auto presolveRes = makeIdentityResult(model);
  std::vector<double> presolvedSolution = {2.0, 3.0};

  postsolve::Postsolver postsolver;
  auto result = postsolver.process(model, presolveRes, presolvedSolution);

  assert(result.isSuccess());
  // 1 + 2*2 + 3*(2*3) = 1 + 4 + 18 = 23
  assert(approx(result.originalObjectiveValue, 23.0));

  std::cout << "[PASSED] test_linear_plus_quadratic_objective\n";
}

// ============================================================
// N. Diagonal quadratic objective coefficient (no implicit 1/2 factor)
// ============================================================
void test_diagonal_quadratic_no_half_factor() {
  model::Model model;
  model.variables.push_back(makeVar("x0", -INF, INF));
  model.objective.quadraticTerms.push_back({0, 0, 4.0});  // diagonal

  auto presolveRes = makeIdentityResult(model);
  std::vector<double> presolvedSolution = {3.0};

  postsolve::Postsolver postsolver;
  auto result = postsolver.process(model, presolveRes, presolvedSolution);

  assert(result.isSuccess());
  // 4 * 3 * 3 = 36 (NOT 0.5 * 4 * 9 = 18)
  assert(approx(result.originalObjectiveValue, 36.0));

  std::cout << "[PASSED] test_diagonal_quadratic_no_half_factor\n";
}

// ============================================================
// Objective sense is not accidentally negated
// ============================================================
void test_objective_sense_not_negated() {
  model::Model model;
  model.objective.sense = model::ObjectiveSense::Maximize;
  model.variables.push_back(makeVar("x0", -INF, INF));
  model.objective.quadraticTerms.push_back({0, 0, 2.0});

  auto presolveRes = makeIdentityResult(model);
  std::vector<double> presolvedSolution = {3.0};

  postsolve::Postsolver postsolver;
  auto result = postsolver.process(model, presolveRes, presolvedSolution);

  assert(result.isSuccess());
  // 2 * 3 * 3 = 18, regardless of Maximize/Minimize sense
  assert(approx(result.originalObjectiveValue, 18.0));

  std::cout << "[PASSED] test_objective_sense_not_negated\n";
}

// ============================================================
// O. Empty presolved model (every variable fixed away)
// ============================================================
void test_empty_presolved_model() {
  model::Model model;
  model.variables.push_back(makeVar("x0", 4.0, 4.0));  // fixed
  model.objective.linearTerms.push_back({0, 2.0});
  model.objective.offset = 1.0;

  presolve::Presolver presolver;
  auto presolveRes = presolver.run(model);

  assert(!presolveRes.infeasible);
  assert(presolveRes.presolvedVariables == 0);
  assert(presolveRes.postsolve.presolvedToOriginalVar.empty());

  std::vector<double> presolvedSolution;  // empty

  postsolve::Postsolver postsolver;
  auto result = postsolver.process(model, presolveRes, presolvedSolution);

  assert(result.isSuccess());
  assert(result.primalSolution.size() == 1);
  assert(approx(result.primalSolution[0], 4.0));
  // 1 + 2*4 = 9
  assert(approx(result.originalObjectiveValue, 9.0));

  std::cout << "[PASSED] test_empty_presolved_model\n";
}

// ============================================================
// P. Invalid mapping: incomplete variable coverage (neither mapped nor fixed)
// ============================================================
void test_invalid_mapping_incomplete_coverage() {
  model::Model model;
  model.name = "IncompleteCoverage";
  model.variables.push_back(makeVar("x0", 0.0, 10.0));
  model.variables.push_back(makeVar("x1", 0.0, 10.0));
  model.variables.push_back(makeVar("x2", 0.0, 10.0));

  // Presolve maps only x0 and x1. Variable x2 is neither in
  // presolvedToOriginalVar nor in fixedVariables.
  presolve::PresolveResult presolveRes;
  presolveRes.originalVariables = 3;
  presolveRes.presolvedVariables = 2;
  presolveRes.postsolve.presolvedToOriginalVar = {0, 1};
  presolveRes.postsolve.originalToPresolvedVar = {0, 1, -1};

  std::vector<double> presolvedSolution = {1.0, 2.0};

  postsolve::Postsolver postsolver;
  auto result = postsolver.process(model, presolveRes, presolvedSolution);

  assert(!result.isSuccess());
  assert(result.status == postsolve::PostsolveStatus::InvalidMapping);

  std::cout << "[PASSED] test_invalid_mapping_incomplete_coverage\n";
}

// ============================================================
// Q. Reject non-finite primal solution: NaN
// ============================================================
void test_reject_nan_primal_solution() {
  model::Model model;
  model.variables.push_back(makeVar("x0", 0.0, 10.0));

  auto presolveRes = makeIdentityResult(model);
  std::vector<double> presolvedSolution = {std::numeric_limits<double>::quiet_NaN()};

  postsolve::Postsolver postsolver;
  auto result = postsolver.process(model, presolveRes, presolvedSolution);

  assert(!result.isSuccess());
  assert(result.status == postsolve::PostsolveStatus::BoundViolation);

  std::cout << "[PASSED] test_reject_nan_primal_solution\n";
}

// ============================================================
// R. Reject non-finite primal solution: +Inf
// ============================================================
void test_reject_pos_inf_primal_solution() {
  model::Model model;
  // Free variable with [-INF, +INF] bounds: +Inf must still be rejected.
  model.variables.push_back(makeVar("x0", -INF, INF));

  auto presolveRes = makeIdentityResult(model);
  std::vector<double> presolvedSolution = {INF};

  postsolve::Postsolver postsolver;
  auto result = postsolver.process(model, presolveRes, presolvedSolution);

  assert(!result.isSuccess());
  assert(result.status == postsolve::PostsolveStatus::BoundViolation);

  std::cout << "[PASSED] test_reject_pos_inf_primal_solution\n";
}

// ============================================================
// S. Reject non-finite primal solution: -Inf
// ============================================================
void test_reject_neg_inf_primal_solution() {
  model::Model model;
  // Free variable with [-INF, +INF] bounds: -Inf must still be rejected.
  model.variables.push_back(makeVar("x0", -INF, INF));

  auto presolveRes = makeIdentityResult(model);
  std::vector<double> presolvedSolution = {-INF};

  postsolve::Postsolver postsolver;
  auto result = postsolver.process(model, presolveRes, presolvedSolution);

  assert(!result.isSuccess());
  assert(result.status == postsolve::PostsolveStatus::BoundViolation);

  std::cout << "[PASSED] test_reject_neg_inf_primal_solution\n";
}


// ============================================================
// Scale-aware feasibility gate
//
// The gate was purely absolute, so its verdict depended on the model's UNITS
// rather than the answer's quality. Measured on Netlib adlittle, PDLP
// converged to its requested RELATIVE tolerance and produced a row violation
// of 5.35e-06 on rows whose own activity is in the hundreds -- 2.5e-08
// relative -- and postsolve refused it, so a correctly solved LP returned an
// error instead of an answer.
// ============================================================
void test_relative_feasibility_accepts_large_scale_row() {
  model::Model model;
  model.variables.push_back(makeVar("x0", 0.0, 1e9));
  model::Constraint row;
  row.name = "big";
  row.lowerBound = -std::numeric_limits<double>::infinity();
  row.upperBound = 1e6;
  row.linearTerms.push_back({0, 1.0});
  model.constraints.push_back(row);

  auto presolveRes = makeIdentityResult(model);
  // 1e-3 over a bound of 1e6 is 1e-9 relative: inside the 1e-8 relative term,
  // far outside the 1e-6 absolute one.
  std::vector<double> presolvedSolution = {1e6 + 1e-3};

  postsolve::Postsolver postsolver;
  auto result = postsolver.process(model, presolveRes, presolvedSolution);

  assert(result.isSuccess());
  // The absolute residual is still REPORTED even though the point passed:
  // reporting 0.0 here would call a point exact that merely passed the gate.
  assert(std::abs(result.maxConstraintResidual - 1e-3) < 1e-9);
  assert(result.maxConstraintResidualScaled < 1e-8);

  std::cout << "[PASSED] test_relative_feasibility_accepts_large_scale_row\n";
}

void test_relative_feasibility_still_rejects_real_violation() {
  model::Model model;
  model.variables.push_back(makeVar("x0", 0.0, 1e9));
  model::Constraint row;
  row.name = "big";
  row.lowerBound = -std::numeric_limits<double>::infinity();
  row.upperBound = 1e6;
  row.linearTerms.push_back({0, 1.0});
  model.constraints.push_back(row);

  auto presolveRes = makeIdentityResult(model);
  // 1.0 over 1e6 is 1e-6 relative -- two orders past the relative term, so it
  // must still be refused. The gate is scale-aware, not simply looser.
  std::vector<double> presolvedSolution = {1e6 + 1.0};

  postsolve::Postsolver postsolver;
  auto result = postsolver.process(model, presolveRes, presolvedSolution);

  assert(!result.isSuccess());
  assert(result.status == postsolve::PostsolveStatus::ConstraintViolation);

  std::cout << "[PASSED] test_relative_feasibility_still_rejects_real_violation\n";
}

void test_small_scale_gate_is_unchanged() {
  // On a unit-magnitude row the relative term contributes ~1e-8, so behaviour
  // must be indistinguishable from the old absolute 1e-6 rule.
  model::Model model;
  model.variables.push_back(makeVar("x0", 0.0, 10.0));
  model::Constraint row;
  row.name = "small";
  row.lowerBound = -std::numeric_limits<double>::infinity();
  row.upperBound = 1.0;
  row.linearTerms.push_back({0, 1.0});
  model.constraints.push_back(row);

  auto presolveRes = makeIdentityResult(model);

  postsolve::Postsolver postsolver;
  auto inside = postsolver.process(model, presolveRes, {1.0 + 1e-9});
  assert(inside.isSuccess());
  auto outside = postsolver.process(model, presolveRes, {1.0 + 1e-3});
  assert(!outside.isSuccess());

  std::cout << "[PASSED] test_small_scale_gate_is_unchanged\n";
}

void test_integrality_is_not_scaled() {
  // Integrality must stay ABSOLUTE. Being 0.4 away from an integer is 0.4 away
  // whatever the variable's magnitude; scaling it would let a large integer
  // variable drift arbitrarily far from integral and still pass.
  model::Model model;
  model.variables.push_back(
      makeVar("x0", 0.0, 1e9, model::VariableType::Integer));

  auto presolveRes = makeIdentityResult(model);
  std::vector<double> presolvedSolution = {1e6 + 0.4};

  postsolve::Postsolver postsolver;
  auto result = postsolver.process(model, presolveRes, presolvedSolution);

  assert(!result.isSuccess());
  assert(result.status == postsolve::PostsolveStatus::IntegralityViolation);

  std::cout << "[PASSED] test_integrality_is_not_scaled\n";
}

int main() {
  test_no_transformations();
  test_fixed_continuous_variable();
  test_fixed_integer_variable();
  test_fixed_binary_variable();
  test_removed_variable_reconstruction_qp();
  test_invalid_mapping_incorrect_size();
  test_invalid_mapping_duplicate();
  test_invalid_mapping_out_of_range();
  test_invalid_fixed_variable_index();
  test_invalid_removed_constraint_index();
  test_original_constraint_violation();
  test_original_bound_violation();
  test_integrality_violation();
  test_objective_offset();
  test_linear_plus_quadratic_objective();
  test_diagonal_quadratic_no_half_factor();
  test_objective_sense_not_negated();
  test_empty_presolved_model();
  test_invalid_mapping_incomplete_coverage();
  test_reject_nan_primal_solution();
  test_reject_pos_inf_primal_solution();
  test_reject_neg_inf_primal_solution();
  test_relative_feasibility_accepts_large_scale_row();
  test_relative_feasibility_still_rejects_real_violation();
  test_small_scale_gate_is_unchanged();
  test_integrality_is_not_scaled();

  std::cout << "All postsolve tests passed successfully!\n";
  return 0;
}
