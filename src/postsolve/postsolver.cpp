#include "postsolve/postsolver.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <string>
#include <unordered_set>

namespace postsolve {

PostsolveResult Postsolver::process(
    const model::Model& originalModel,
    const presolve::PresolveResult& presolveResult,
    const std::vector<double>& presolvedPrimalSolution) {
  PostsolveResult result;

  if (presolveResult.infeasible) {
    result.status = PostsolveStatus::InfeasiblePresolve;
    result.errorMessage = "Presolve marked the problem as infeasible.";
    return result;
  }

  // Validate the presolve metadata/mapping before dereferencing any of it.
  if (!validateMapping(originalModel, presolveResult, presolvedPrimalSolution, result)) {
    return result;
  }

  const auto& meta = presolveResult.postsolve;
  std::size_t nOrigVars = originalModel.variables.size();

  // 1. Initialize full primal solution vector for original variables
  std::vector<double> x(nOrigVars, 0.0);

  // Map presolved variable values back to their original positions
  for (std::size_t pIdx = 0; pIdx < presolvedPrimalSolution.size(); ++pIdx) {
  const std::size_t origIdx = meta.presolvedToOriginalVar[pIdx];
  x[origIdx] = presolvedPrimalSolution[pIdx];
}

  // 2. Map fixed variables back to their fixed values
  for (const auto& fixed : meta.fixedVariables) {
  x[fixed.originalIndex] = fixed.fixedValue;
}

  // 3. Apply transformation log in reverse order (for bound modifications / substitutions)
for (auto it = presolveResult.transformations.rbegin();
     it != presolveResult.transformations.rend(); ++it) {
  const auto& trans = *it;

  switch (trans.type) {
    case presolve::TransformationType::FixVariable:
      // Fixed variables are already restored from
      // PostsolveMetadata::fixedVariables above.
      break;

    case presolve::TransformationType::TightenLowerBound:
    case presolve::TransformationType::TightenUpperBound:
      // Bound tightening changes the presolved model but does not
      // require reconstructing a variable value during postsolve.
      break;

    case presolve::TransformationType::RemoveConstraint:
      // Removed constraints do not change the primal variable vector.
      break;

    case presolve::TransformationType::RemoveVariable:
      // Eliminated variables are restored through
      // PostsolveMetadata.
      break;

    case presolve::TransformationType::SubstituteVariable:
      // The current presolver does not generate general substitution
      // transformations, so no substitution reconstruction is needed.
      break;
  }
}
  // 4. Calculate original objective value
  result.originalObjectiveValue = evaluateObjective(originalModel, x);
  result.primalSolution = x;

  // 5. Validate final primal solution against original model bounds and constraints
  if (!validateSolution(originalModel, x, result)) {
    return result;
  }

  result.status = PostsolveStatus::Success;
  return result;
}

double Postsolver::evaluateObjective(
    const model::Model& model,
    const std::vector<double>& x) const {
  double objVal = model.objective.offset;

  // Linear terms
  for (const auto& term : model.objective.linearTerms) {
    if (term.variableIndex >= 0 && static_cast<std::size_t>(term.variableIndex) < x.size()) {
      objVal += term.value * x[term.variableIndex];
    }
  }

  // Quadratic terms
  for (const auto& term : model.objective.quadraticTerms) {
    if (term.variableIndex1 >= 0 && static_cast<std::size_t>(term.variableIndex1) < x.size() &&
        term.variableIndex2 >= 0 && static_cast<std::size_t>(term.variableIndex2) < x.size()) {
      // Model convention: f(x) = offset + sum(c_i x_i) + sum(q_ij x_i x_j),
      // where QuadraticTerm.value is the DIRECT coefficient of x_i * x_j.
      // There is no implicit 1/2 factor, for diagonal or off-diagonal terms.
      objVal += term.value * x[term.variableIndex1] * x[term.variableIndex2];
    }
  }

  return objVal;
}

bool Postsolver::validateMapping(
    const model::Model& originalModel,
    const presolve::PresolveResult& presolveResult,
    const std::vector<double>& presolvedPrimalSolution,
    PostsolveResult& result) const {
  const auto& meta = presolveResult.postsolve;
  const std::size_t nOrigVars = originalModel.variables.size();
  const std::size_t nOrigCons = originalModel.constraints.size();

  auto invalid = [&](const std::string& message) {
    result.status = PostsolveStatus::InvalidMapping;
    result.errorMessage = message;
    return false;
  };

  // presolvedToOriginalVar.size() must match the presolved variable count.
  if (meta.presolvedToOriginalVar.size() != presolveResult.presolvedVariables) {
    return invalid(
        "Invalid mapping: presolvedToOriginalVar size does not match the presolved variable count.");
  }

  // The presolved solution vector must line up with the presolved variable mapping.
  if (presolvedPrimalSolution.size() != meta.presolvedToOriginalVar.size()) {
    return invalid(
        "Invalid mapping: presolved solution size does not match presolvedToOriginalVar size.");
  }

  // Every original variable index referenced by the mapping must be in range,
  // and no original variable may be claimed by more than one presolved variable.
  // Track coverage across all original variables to ensure every original variable
  // is represented exactly once (either mapped or fixed, never both, never missing).
  std::vector<bool> covered(nOrigVars, false);

  for (std::size_t origIdx : meta.presolvedToOriginalVar) {
    if (origIdx >= nOrigVars) {
      return invalid(
          "Invalid mapping: presolvedToOriginalVar contains an out-of-range original variable index.");
    }
    if (covered[origIdx]) {
      return invalid(
          "Invalid mapping: presolvedToOriginalVar contains a duplicate original variable index.");
    }
    covered[origIdx] = true;
  }

  // Fixed-variable metadata must reference valid original variables, including
  // any bilinear cross-contribution indices that will later be dereferenced.
  std::unordered_set<std::size_t> fixedOrigVars;
  fixedOrigVars.reserve(meta.fixedVariables.size());
  for (const auto& fixed : meta.fixedVariables) {
    if (fixed.originalIndex >= nOrigVars) {
      return invalid(
          "Invalid mapping: fixed-variable record has an out-of-range original variable index.");
    }
    // A variable that is fixed by presolve cannot also survive into the
    // presolved model under the same original index.
    if (covered[fixed.originalIndex]) {
      return invalid(
          "Invalid mapping: fixed-variable original index also appears in presolvedToOriginalVar.");
    }
    if (!fixedOrigVars.insert(fixed.originalIndex).second) {
      return invalid(
          "Invalid mapping: duplicate fixed-variable record for original variable index.");
    }
    covered[fixed.originalIndex] = true;

    for (const auto& [otherOrigIdx, contribution] : fixed.quadraticCrossContributions) {
      (void)contribution;
      if (otherOrigIdx >= nOrigVars) {
        return invalid(
            "Invalid mapping: fixed-variable cross-contribution references an out-of-range "
            "original variable index.");
      }
    }
  }

  // Verify complete original-variable coverage:
  // mapped original variables + fixed variables = every original variable exactly once.
  for (std::size_t i = 0; i < nOrigVars; ++i) {
    if (!covered[i]) {
      return invalid(
          "Invalid mapping: original variable " + std::to_string(i) +
          " is neither mapped nor fixed (incomplete variable coverage).");
    }
  }

  // Removed-constraint metadata must reference valid original constraints/variables.
  for (const auto& removed : meta.removedConstraints) {
    if (removed.originalIndex >= nOrigCons) {
      return invalid(
          "Invalid mapping: removed-constraint record has an out-of-range original constraint index.");
    }
    if (removed.wasSingleton && removed.singletonOriginalVarIndex >= nOrigVars) {
      return invalid(
          "Invalid mapping: removed-constraint singleton record has an out-of-range original "
          "variable index.");
    }
    if (removed.wasDuplicate && removed.duplicateOfOriginalIndex >= nOrigCons) {
      return invalid(
          "Invalid mapping: removed-constraint duplicate record has an out-of-range original "
          "constraint index.");
    }
  }

  return true;
}

bool Postsolver::validateSolution(
    const model::Model& originalModel,
    const std::vector<double>& x,
    PostsolveResult& result) const {
  result.maxBoundResidual = 0.0;
  result.maxConstraintResidual = 0.0;
  result.maxBoundResidualScaled = 0.0;
  result.maxConstraintResidualScaled = 0.0;

  // Size guard from origin/main: every loop below indexes x by variable, so a
  // mismatched vector must be refused before any of them run.
  if (x.size() != originalModel.variables.size()) {
    result.status = PostsolveStatus::InvalidMapping;
    result.errorMessage = "Primal vector size does not match the supplied model.";
    return false;
  }

  // Feasibility is judged as  violation <= tolerance_ + relativeTolerance_*scale,
  // where `scale` is the entity's OWN magnitude. A purely absolute gate holds a
  // row summing values in the millions to the same slack as a row of two small
  // terms, which is a statement about units rather than about correctness.
  // Integrality is deliberately NOT scaled below: being 0.4 away from an
  // integer is 0.4 away whatever the variable's magnitude.
  const auto feasTolerance = [this](double scale) {
    return tolerance_ + relativeTolerance_ * scale;
  };

  // 1. Explicitly reject all non-finite primal values (NaN, +Inf, -Inf)
  // before ordinary bound, integrality, or constraint validation.
  for (std::size_t i = 0; i < originalModel.variables.size(); ++i) {
    double val = x[i];
    if (!std::isfinite(val)) {
      result.status = PostsolveStatus::BoundViolation;
      result.errorMessage = "Non-finite primal value on variable " + originalModel.variables[i].name;
      result.maxBoundResidual = std::numeric_limits<double>::infinity();
      return false;
    }
  }

  // 2. Validate Variable Bounds & Integrality
  for (std::size_t i = 0; i < originalModel.variables.size(); ++i) {
    const auto& var = originalModel.variables[i];
    double val = x[i];

    const double boundScale = std::max({
        1.0,
        std::isfinite(var.lowerBound) ? std::abs(var.lowerBound) : 0.0,
        std::isfinite(var.upperBound) ? std::abs(var.upperBound) : 0.0,
        std::abs(val)});
    const double boundTolerance = feasTolerance(boundScale);

    // Residuals are recorded for EVERY variable, not only for the ones that
    // breach the gate. Recording them only inside the violation branch made a
    // passing solve report 0.0 when its true worst violation was 5.35e-06 --
    // the reported number then said "exact" for a point that merely passed.
    {
      const double below = std::isfinite(var.lowerBound)
                               ? std::max(0.0, var.lowerBound - val) : 0.0;
      const double above = std::isfinite(var.upperBound)
                               ? std::max(0.0, val - var.upperBound) : 0.0;
      const double worst = std::max(below, above);
      result.maxBoundResidual = std::max(result.maxBoundResidual, worst);
      result.maxBoundResidualScaled =
          std::max(result.maxBoundResidualScaled, worst / boundScale);
    }

    // Check Lower Bound
    if (val < var.lowerBound - boundTolerance) {
      result.status = PostsolveStatus::BoundViolation;
      result.errorMessage = "Lower bound violation on variable " + var.name;
    }

    // Check Upper Bound
    if (val > var.upperBound + boundTolerance) {
      result.status = PostsolveStatus::BoundViolation;
      result.errorMessage = "Upper bound violation on variable " + var.name;
    }

    // Check Integrality
    if (var.type == model::VariableType::Integer || var.type == model::VariableType::Binary) {
      double rounded = std::round(val);
      if (std::abs(val - rounded) > tolerance_) {
        result.status = PostsolveStatus::IntegralityViolation;
        result.errorMessage = "Integrality violation on variable " + var.name;
      }
    }
  }

  if (result.status == PostsolveStatus::BoundViolation ||
      result.status == PostsolveStatus::IntegralityViolation) {
    return false;
  }

  // Validate Constraints
  for (const auto& constraint : originalModel.constraints) {
    double activity = 0.0;

    for (const auto& term : constraint.linearTerms) {
  if (term.variableIndex < 0 ||
      static_cast<std::size_t>(term.variableIndex) >= x.size()) {
    result.status = PostsolveStatus::InvalidMapping;
    result.errorMessage =
        "Invalid mapping: constraint contains an out-of-range variable index.";
    return false;
  }

  activity += term.value * x[term.variableIndex];
}

    // Check non-finite activity
    if (!std::isfinite(activity)) {
      result.status = PostsolveStatus::ConstraintViolation;
      result.errorMessage = "Non-finite activity on constraint " + constraint.name;
      result.maxConstraintResidual = std::numeric_limits<double>::infinity();
      return false;
    }

    // The row's own magnitude: the size of the numbers actually being added
    // up, not just its right-hand side.
    double rowMagnitude = 0.0;
    for (const auto& term : constraint.linearTerms) {
      rowMagnitude += std::abs(term.value * x[term.variableIndex]);
    }
    const double rowScale = std::max({
        1.0,
        std::isfinite(constraint.lowerBound) ? std::abs(constraint.lowerBound) : 0.0,
        std::isfinite(constraint.upperBound) ? std::abs(constraint.upperBound) : 0.0,
        rowMagnitude});
    const double rowTolerance = feasTolerance(rowScale);

    {
      const double below = std::isfinite(constraint.lowerBound)
                               ? std::max(0.0, constraint.lowerBound - activity) : 0.0;
      const double above = std::isfinite(constraint.upperBound)
                               ? std::max(0.0, activity - constraint.upperBound) : 0.0;
      const double worst = std::max(below, above);
      result.maxConstraintResidual = std::max(result.maxConstraintResidual, worst);
      result.maxConstraintResidualScaled =
          std::max(result.maxConstraintResidualScaled, worst / rowScale);
    }

    // Check Lower Bound violation
    if (activity < constraint.lowerBound - rowTolerance) {
      result.status = PostsolveStatus::ConstraintViolation;
      result.errorMessage = "Constraint lower bound violation on " + constraint.name;
    }

    // Check Upper Bound violation
    if (activity > constraint.upperBound + rowTolerance) {
      result.status = PostsolveStatus::ConstraintViolation;
      result.errorMessage = "Constraint upper bound violation on " + constraint.name;
    }
  }

  if (result.status == PostsolveStatus::ConstraintViolation) {
    return false;
  }

  return true;
}

}  // namespace postsolve
namespace postsolve {

// ============================================================================
// Dual and reduced-cost reconstruction
// ============================================================================
//
// CONVENTION. Everything below is expressed in the model's OWN objective
// sense, so no sense flip happens here -- the engines already report shadow
// prices that way, and applying a second correction is how a sign error gets
// introduced.
//
//   y_i  = d(objective) / d(rhs of row i)
//   d_j  = grad_j f(x*) - sum_i a_ij * y_i
//
// At an optimum d_j is zero for a variable strictly inside its bounds, and is
// the bound multiplier otherwise. For MINIMIZE, d_j >= 0 at a lower bound and
// <= 0 at an upper bound; for MAXIMIZE the signs reverse. Row multipliers
// obey the matching rule against the side that is tight.
//
// THE PROBLEM THIS SOLVES. Presolve can derive a variable bound FROM a row --
// from "4x + 5y <= 28" with x >= 0 it derives y <= 5.6. The reduced model
// then carries that restriction as a bound, and the engine naturally parks
// part of the row's price on it. Copying reduced row duals into original
// positions therefore under-reports the row: measured on that model the row
// price comes back 0.75 where the original model's shadow price is 0.8, with
// the missing 0.05 sitting on y's derived bound.
//
// THE TRANSFER. Undoing such a bound means moving its multiplier onto the row
// it came from. With coefficient a_ik of variable k in row i:
//
//     delta = d_k / a_ik ,   y_i += delta ,   d_j -= delta * a_ij  for all j in row i
//
// after which d_k is exactly zero -- the bound no longer exists, so it can
// carry no multiplier -- and the row absorbs the price. On the model above
// delta = 0.25/5 = 0.05 and the row goes to 0.8 exactly.
//
// WHY THE OTHER VARIABLES IN THE ROW CAN ABSORB IT. The transfer perturbs
// d_j for every other j in row i, which would be invalid if any of them were
// strictly inside its bounds, since stationarity forces d_j = 0 there. That
// cannot happen. A derived bound u_k = (U_i - otherMin)/a_ik is only ACTIVE
// when row i is tight AND every other variable in it sits exactly at the
// bound that attains otherMin -- otherwise the row still has slack and x_k
// could move further. Those variables are therefore all at bounds, where a
// nonzero d_j is permitted. The perturbation also moves them in the safe
// direction: a variable at the otherMin-attaining bound has a_ij > 0 exactly
// when it sits at its lower bound, so -delta*a_ij pushes d_j the way that
// bound's sign condition already allows.
//
// NON-UNIQUENESS. Degenerate problems admit many valid multiplier vectors.
// This returns one that satisfies the original model's conditions rather than
// attempting to match any particular reference solver's choice.

namespace {
constexpr double kDualEps = 1e-9;

// A row that is not tight at the solution cannot carry a nonzero price --
// complementary slackness. This has to be checked before moving any
// multiplier onto a row: a bound the solution sits on is not evidence that
// the ROW it came from is binding. Measured on a model whose singleton row
// 2x <= 6 became x <= 3 while the optimum had x = 0, transferring
// unconditionally moved x's OWN lower-bound multiplier onto that slack row
// and broke stationarity by 0.1.
bool rowIsTight(const model::Constraint& row, const std::vector<double>& x, double tol) {
  double act = 0.0;
  for (const auto& t : row.linearTerms) {
    if (t.variableIndex >= 0 && static_cast<std::size_t>(t.variableIndex) < x.size()) {
      act += t.value * x[static_cast<std::size_t>(t.variableIndex)];
    }
  }
  return (std::isfinite(row.lowerBound) && std::abs(act - row.lowerBound) <= tol) ||
         (std::isfinite(row.upperBound) && std::abs(act - row.upperBound) <= tol);
}

double coefficientOf(const model::Constraint& row, std::size_t variableIndex) {
  double a = 0.0;
  for (const auto& t : row.linearTerms) {
    if (t.variableIndex >= 0 &&
        static_cast<std::size_t>(t.variableIndex) == variableIndex) {
      a += t.value;  // duplicate entries on one variable accumulate
    }
  }
  return a;
}

bool sameBound(double a, double b, double tol) {
  return a == b || (std::isfinite(a) && std::isfinite(b) &&
      std::abs(a - b) <= tol * std::max({1.0, std::abs(a), std::abs(b)}));
}

// Verify causality from the forward log, independently of the solution or duals.
// The six tightening emitters in Presolver already record stable row/variable
// indices and old/new bounds. Replay only those bounds to check that the named
// original row actually implies each recorded bound, with the recorded direction.
// Fixed-variable elimination is accounted for by equal bounds in this history;
// no primal reconstruction or substitution is performed here.
std::string validateBoundProvenance(const model::Model& original,
                                   const presolve::PresolveResult& presolved,
                                   double tol) {
  std::vector<double> lower, upper;
  for (const auto& v : original.variables) {
    lower.push_back(v.lowerBound);
    upper.push_back(v.upperBound);
  }
  std::unordered_set<std::size_t> removedRows;
  for (std::size_t pos = 0; pos < presolved.transformations.size(); ++pos) {
    const auto& tr = presolved.transformations[pos];
    if (tr.type == presolve::TransformationType::RemoveConstraint) {
      removedRows.insert(tr.originalConstraintIndex);
      continue;
    }
    const bool tightenLower = tr.type == presolve::TransformationType::TightenLowerBound;
    if (!tightenLower && tr.type != presolve::TransformationType::TightenUpperBound) continue;
    const auto invalid = [&](const std::string& detail) {
      return "bound provenance at transformation " + std::to_string(pos) + ": " + detail;
    };
    const auto k = tr.originalVariableIndex;
    const auto i = tr.originalConstraintIndex;
    if (i >= original.constraints.size())
      return invalid("originating original constraint index is missing or out of range");
    if (k >= original.variables.size())
      return invalid("affected original variable index is missing or out of range");
    if (removedRows.count(i))
      return invalid("source row was already removed; expected a preceding tightening");

    const auto& row = original.constraints[i];
    std::map<std::size_t, double> coefficients;
    for (const auto& term : row.linearTerms) {
      if (term.variableIndex < 0 ||
          static_cast<std::size_t>(term.variableIndex) >= lower.size())
        return invalid("source row contains an invalid original variable index");
      if (!std::isfinite(term.value)) return invalid("source coefficient is not finite");
      auto& sum = coefficients[static_cast<std::size_t>(term.variableIndex)];
      sum += term.value;
      if (!std::isfinite(sum)) return invalid("source coefficient is not finite");
    }
    const auto found = coefficients.find(k);
    if (found == coefficients.end()) return invalid("affected variable is absent from source row");
    const double a = found->second;
    if (std::abs(a) <= kDualEps) return invalid("source coefficient is zero or too small");
    if (!std::isfinite(tr.newValue) || std::isnan(tr.oldValue))
      return invalid("recorded bound value is not finite");
    const double oldBound = tightenLower ? lower[k] : upper[k];
    if (!sameBound(tr.oldValue, oldBound, tol))
      return invalid("old bound does not match the preceding bound history");
    if (tightenLower ? tr.newValue <= tr.oldValue : tr.newValue >= tr.oldValue)
      return invalid("recorded bound does not tighten in the declared direction");

    // For a>0, a lower variable bound comes from the row's lower side;
    // for a<0 it comes from the upper side. Upper variable bounds reverse this.
    const bool fromLowerRow = tightenLower == (a > 0.0);
    const double rhs = fromLowerRow ? row.lowerBound : row.upperBound;
    if (!std::isfinite(rhs))
      return invalid("bound direction requires a finite source constraint side");
    double otherActivity = 0.0;
    for (const auto& [j, coefficient] : coefficients) {
      if (j == k || coefficient == 0.0) continue;
      // Lower rows use the maximum other activity; upper rows use the minimum.
      const double bound = fromLowerRow == (coefficient > 0.0) ? upper[j] : lower[j];
      if (!std::isfinite(bound))
        return invalid("preceding bounds do not establish a finite source-row implication");
      otherActivity += coefficient * bound;
      if (!std::isfinite(otherActivity)) return invalid("source-row implication overflows");
    }
    double implied = (rhs - otherActivity) / a;
    if (!std::isfinite(implied)) return invalid("source-row implication is not finite");
    if (original.variables[k].type != model::VariableType::Continuous) {
      // Match the presolver's existing integer-bound rounding convention.
      implied = tightenLower ? std::ceil(implied - kDualEps) : std::floor(implied + kDualEps);
    }
    if (!sameBound(tr.newValue, implied, tol))
      return invalid("new bound is not implied by the named original constraint and preceding bounds");
    if (tightenLower) lower[k] = tr.newValue;
    else upper[k] = tr.newValue;
  }
  return {};
}
}  // namespace

std::vector<double> Postsolver::objectiveGradient(
    const model::Model& model, const std::vector<double>& x) const {
  std::vector<double> g(model.variables.size(), 0.0);
  for (const auto& t : model.objective.linearTerms) {
    if (t.variableIndex >= 0 &&
        static_cast<std::size_t>(t.variableIndex) < g.size()) {
      g[static_cast<std::size_t>(t.variableIndex)] += t.value;
    }
  }
  for (const auto& q : model.objective.quadraticTerms) {
    const auto i = static_cast<std::size_t>(q.variableIndex1);
    const auto j = static_cast<std::size_t>(q.variableIndex2);
    if (i >= g.size() || j >= g.size()) continue;
    if (i == j) {
      // value * x_i^2  ->  d/dx_i = 2 * value * x_i
      g[i] += 2.0 * q.value * x[i];
    } else {
      // value * x_i * x_j, stored once
      g[i] += q.value * x[j];
      g[j] += q.value * x[i];
    }
  }
  return g;
}

void Postsolver::reconstructDuals(
    const model::Model& originalModel,
    const presolve::PresolveResult& presolveResult,
    const std::vector<double>& presolvedConstraintDuals,
    PostsolveResult& result) const {
  const auto& meta = presolveResult.postsolve;
  const std::size_t n = originalModel.variables.size();
  const std::size_t m = originalModel.constraints.size();
  const std::vector<double>& x = result.primalSolution;

  const auto failProvenance = [&](const std::string& reason) {
    result.dualsAvailable = false;
    result.constraintDuals.clear();
    result.reducedCosts.clear();
    result.dualsUnavailableReason = reason;
  };
  // Validate before either reverse-log handler can consume a multiplier. In
  // particular, singleton removal must not hide a malformed earlier tightening.
  const auto provenanceError = validateBoundProvenance(originalModel, presolveResult, tolerance_);
  if (!provenanceError.empty()) {
    failProvenance(provenanceError);
    return;
  }

  if (presolvedConstraintDuals.size() != meta.presolvedToOriginalConstraint.size()) {
    result.dualsAvailable = false;
    result.dualsUnavailableReason =
        "reduced dual vector has " + std::to_string(presolvedConstraintDuals.size()) +
        " entries but presolve left " +
        std::to_string(meta.presolvedToOriginalConstraint.size()) + " constraints";
    return;
  }

  for (double dual : presolvedConstraintDuals) {
    if (!std::isfinite(dual)) {
      result.dualsUnavailableReason = "reduced dual vector contains a non-finite multiplier";
      return;
    }
  }

  // Seed: rows that survived presolve keep the engine's price; rows presolve
  // removed start at zero and are filled in as the log is reversed.
  std::vector<double> y(m, 0.0);
  for (std::size_t r = 0; r < presolvedConstraintDuals.size(); ++r) {
    const std::size_t orig = meta.presolvedToOriginalConstraint[r];
    if (orig >= m) {
      result.dualsAvailable = false;
      result.dualsUnavailableReason = "constraint mapping points outside the original model";
      return;
    }
    y[orig] = presolvedConstraintDuals[r];
  }

  // d_j = grad_j - sum_i a_ij y_i, kept consistent with y throughout.
  std::vector<double> d = objectiveGradient(originalModel, x);
  const auto applyRow = [&](std::size_t row, double delta) {
    y[row] += delta;
    for (const auto& t : originalModel.constraints[row].linearTerms) {
      if (t.variableIndex >= 0 && static_cast<std::size_t>(t.variableIndex) < n) {
        d[static_cast<std::size_t>(t.variableIndex)] -= delta * t.value;
      }
    }
  };
  for (std::size_t i = 0; i < m; ++i) {
    if (y[i] != 0.0) {
      for (const auto& t : originalModel.constraints[i].linearTerms) {
        if (t.variableIndex >= 0 && static_cast<std::size_t>(t.variableIndex) < n) {
          d[static_cast<std::size_t>(t.variableIndex)] -= y[i] * t.value;
        }
      }
    }
  }

  // Index the removed-constraint records so a RemoveConstraint transformation
  // can find the details of what it removed.
  std::map<std::size_t, const presolve::RemovedConstraintRecord*> removed;
  for (const auto& rec : meta.removedConstraints) removed[rec.originalIndex] = &rec;

  const double tol = tolerance_;

  // Reverse order: the last transformation applied is the first undone.
  // A singleton logs tightening before removal, so its removal transfers first.
  // Both handlers use the CURRENT d[k], which applyRow consumes, preventing the
  // earlier tightening record from transferring the same multiplier again.
  for (auto it = presolveResult.transformations.rbegin();
       it != presolveResult.transformations.rend(); ++it) {
    const auto& tr = *it;
    switch (tr.type) {
      case presolve::TransformationType::TightenLowerBound:
      case presolve::TransformationType::TightenUpperBound: {
        const std::size_t k = tr.originalVariableIndex;
        const std::size_t i = tr.originalConstraintIndex;
        // Indices, coefficient, direction and row implication were checked
        // against the forward metadata before entering the reverse walk.
        // Only a bound that the solution actually sits on carries a
        // multiplier; an inactive derived bound has none to move.
        if (std::abs(x[k] - tr.newValue) > tol) break;
        // If the ORIGINAL bound is equally tight the restriction was not
        // really introduced by the row, so the multiplier stays on the bound.
        if (std::abs(tr.oldValue - tr.newValue) <= tol) break;
        // The row must itself be binding, or it can hold no price.
        if (!rowIsTight(originalModel.constraints[i], x, tol)) break;
        const double a = coefficientOf(originalModel.constraints[i], k);
        if (std::abs(a) <= kDualEps) break;
        if (std::abs(d[k]) <= kDualEps) break;  // no price parked on it
        applyRow(i, d[k] / a);                  // leaves d[k] == 0
        break;
      }
      case presolve::TransformationType::RemoveConstraint: {
        const std::size_t i = tr.originalConstraintIndex;
        if (i >= m) {
          failProvenance("singleton provenance: original constraint index is missing or out of range");
          return;
        }
        auto found = removed.find(i);
        if (found == removed.end()) {
          failProvenance("removal provenance: no metadata for the referenced original constraint");
          return;
        }
        const auto* rec = found->second;
        if (rec->wasSingleton) {
          // The row was turned into a bound on one variable. Its price is
          // whatever ended up on that bound.
          const std::size_t k = rec->singletonOriginalVarIndex;
          if (k >= n || tr.originalVariableIndex != k) {
            failProvenance("singleton provenance: affected original variable indices do not agree");
            return;
          }
          const double a = coefficientOf(originalModel.constraints[i], k);
          if (!std::isfinite(a) || std::abs(a) <= kDualEps ||
              !std::isfinite(rec->singletonCoefficient) ||
              !sameBound(a, rec->singletonCoefficient, tol)) {
            failProvenance("singleton provenance: coefficient does not match the original source row");
            return;
          }
          // Same gate: x sitting on a bound does not mean the row that
          // produced that bound is tight. Here the variable may simply be at
          // its own original bound, whose multiplier belongs to the bound and
          // not to this row.
          if (!rowIsTight(originalModel.constraints[i], x, tol)) break;
          if (std::abs(d[k]) <= kDualEps) break;
          // wasSingleton identifies a removed row, but only an earlier,
          // validated tightening establishes that THIS active bound came from it.
          const bool hasSourceBound = std::any_of(
              presolveResult.transformations.begin(), it.base() - 1,
              [&](const presolve::Transformation& bound) {
                return (bound.type == presolve::TransformationType::TightenLowerBound ||
                        bound.type == presolve::TransformationType::TightenUpperBound) &&
                       bound.originalConstraintIndex == i && bound.originalVariableIndex == k &&
                       std::abs(bound.newValue - x[k]) <= tol;
              });
          if (!hasSourceBound) {
            failProvenance("singleton provenance: no preceding tightening identifies the active source bound");
            return;
          }
          applyRow(i, d[k] / a);
        }
        // Redundant, duplicate, parallel and empty rows can be assigned zero;
        // a duplicate's price stays wholly on the surviving row. Binding does
        // not require a nonzero multiplier when the restriction is redundant.
        break;
      }
      case presolve::TransformationType::FixVariable:
        // A fixed variable has lower == upper, so any reduced cost satisfies
        // its bound conditions. d[k] as computed is already correct.
        break;
      default:
        break;
    }
  }

  result.constraintDuals = std::move(y);
  result.reducedCosts = std::move(d);
  result.dualsAvailable = true;
  result.dualsUnavailableReason.clear();
}

}  // namespace postsolve

namespace postsolve {

double Postsolver::dualResidual(const model::Model& originalModel,
                                const PostsolveResult& result) const {
  const std::size_t n = originalModel.variables.size();
  const std::size_t m = originalModel.constraints.size();
  if (result.primalSolution.size() != n ||
      result.constraintDuals.size() != m || result.reducedCosts.size() != n) {
    return std::numeric_limits<double>::infinity();
  }
  const std::vector<double>& x = result.primalSolution;
  const std::vector<double>& y = result.constraintDuals;
  const std::vector<double>& d = result.reducedCosts;
  const bool maximise =
      originalModel.objective.sense == model::ObjectiveSense::Maximize;

  // Scale-aware tolerance: an absolute floor plus a term proportional to the
  // size of the numbers involved, so a model whose coefficients are 1e6 is
  // not judged against the same absolute slack as one whose numbers are 1.
  double scale = 1.0;
  const std::vector<double> g = objectiveGradient(originalModel, x);
  // NaN comparisons can silently leave std::max's accumulator at zero.
  // Check explicitly, including arithmetic overflow from otherwise finite
  // inputs, before any residual or fixed-variable/equality exemption.
  const auto allFinite = [](const std::vector<double>& values) {
    return std::all_of(values.begin(), values.end(),
                       [](double v) { return std::isfinite(v); });
  };
  if (!allFinite(g) || !allFinite(y) || !allFinite(d)) {
    return std::numeric_limits<double>::infinity();
  }
  for (double v : g) scale = std::max(scale, std::abs(v));
  for (double v : y) scale = std::max(scale, std::abs(v));
  const double tol = tolerance_ * scale;

  double worst = 0.0;

  // 1. Stationarity: d_j must equal grad_j - sum_i a_ij y_i by construction.
  {
    std::vector<double> lhs = g;
    for (std::size_t i = 0; i < m; ++i) {
      if (y[i] == 0.0) continue;
      for (const auto& t : originalModel.constraints[i].linearTerms) {
        if (t.variableIndex >= 0 && static_cast<std::size_t>(t.variableIndex) < n) {
          lhs[static_cast<std::size_t>(t.variableIndex)] -= y[i] * t.value;
        }
      }
    }
    if (!allFinite(lhs)) return std::numeric_limits<double>::infinity();
    for (std::size_t j = 0; j < n; ++j) worst = std::max(worst, std::abs(lhs[j] - d[j]));
  }

  // 2. Reduced-cost sign conditions against the ORIGINAL bounds, and
  //    complementary slackness on those bounds.
  for (std::size_t j = 0; j < n; ++j) {
    const auto& v = originalModel.variables[j];
    const bool atLower = std::isfinite(v.lowerBound) && std::abs(x[j] - v.lowerBound) <= tol;
    const bool atUpper = std::isfinite(v.upperBound) && std::abs(x[j] - v.upperBound) <= tol;
    const bool fixed = atLower && atUpper;
    if (fixed) continue;                       // any sign is valid on a fixed variable
    if (!atLower && !atUpper) {
      worst = std::max(worst, std::abs(d[j])); // strictly inside: must be zero
    } else if (atLower) {
      worst = std::max(worst, maximise ? std::max(0.0, d[j]) : std::max(0.0, -d[j]));
    } else {
      worst = std::max(worst, maximise ? std::max(0.0, -d[j]) : std::max(0.0, d[j]));
    }
  }

  // 3. Row sign conditions and complementary slackness.
  for (std::size_t i = 0; i < m; ++i) {
    const auto& c = originalModel.constraints[i];
    double act = 0.0;
    for (const auto& t : c.linearTerms) {
      if (t.variableIndex >= 0 && static_cast<std::size_t>(t.variableIndex) < n) {
        act += t.value * x[static_cast<std::size_t>(t.variableIndex)];
      }
    }
    const bool atLower = std::isfinite(c.lowerBound) && std::abs(act - c.lowerBound) <= tol;
    const bool atUpper = std::isfinite(c.upperBound) && std::abs(act - c.upperBound) <= tol;
    if (atLower && atUpper) continue;          // equality row: either sign valid
    if (!atLower && !atUpper) {
      worst = std::max(worst, std::abs(y[i])); // slack row must be priced at zero
    } else if (atUpper) {
      worst = std::max(worst, maximise ? std::max(0.0, -y[i]) : std::max(0.0, y[i]));
    } else {
      worst = std::max(worst, maximise ? std::max(0.0, y[i]) : std::max(0.0, -y[i]));
    }
  }
  return worst;
}

PostsolveResult Postsolver::process(
    const model::Model& originalModel,
    const presolve::PresolveResult& presolveResult,
    const std::vector<double>& presolvedPrimalSolution,
    const std::vector<double>& presolvedConstraintDuals) {
  PostsolveResult result =
      process(originalModel, presolveResult, presolvedPrimalSolution);
  if (!result.isSuccess()) return result;
  // An empty dual vector is the CORRECT input when presolve left no
  // constraints -- the model was solved by presolve plus the bound walk, and
  // every original row's price is then recovered from the transformation log
  // alone. Only demand duals when the reduced model actually has rows.
  const std::size_t reducedRows =
      presolveResult.postsolve.presolvedToOriginalConstraint.size();
  if (presolvedConstraintDuals.empty() && reducedRows > 0) {
    result.dualsAvailable = false;
    result.dualsUnavailableReason = "no reduced-space duals were supplied";
    return result;
  }

  reconstructDuals(originalModel, presolveResult, presolvedConstraintDuals, result);
  if (!result.dualsAvailable) return result;

  // Publish only what actually satisfies the original model's conditions.
  // Reporting multipliers that fail them would be worse than reporting none,
  // because a caller has no way to tell the difference.
  result.maxDualResidual = dualResidual(originalModel, result);
  double scale = 1.0;
  for (double v : result.constraintDuals) scale = std::max(scale, std::abs(v));
  if (!std::isfinite(result.maxDualResidual) ||
      !(result.maxDualResidual <= tolerance_ * 100.0 * scale)) {
    result.dualsUnavailableReason =
        "reconstructed multipliers violate the original model's optimality "
        "conditions by " + std::to_string(result.maxDualResidual);
    result.dualsAvailable = false;
    result.constraintDuals.clear();
    result.reducedCosts.clear();
  }
  return result;
}

}  // namespace postsolve
