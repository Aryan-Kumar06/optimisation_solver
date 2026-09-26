// Hand-derived shadow prices and reduced costs, in the original objective sense.
// No engine or external reference solver is needed. Checks remain active in Release.
#include "postsolve/postsolver.h"
#include "presolve/presolver.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
constexpr double INF = std::numeric_limits<double>::infinity();
constexpr double EPS = 1e-8;
using Type = presolve::TransformationType;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void near(double actual, double expected, const std::string& message) {
  require(std::isfinite(actual) && std::abs(actual - expected) <= EPS,
          message + ": expected " + std::to_string(expected) +
          ", got " + std::to_string(actual));
}

void vectorNear(const std::vector<double>& actual,
                const std::vector<double>& expected, const std::string& message) {
  require(actual.size() == expected.size(), message + ": wrong size");
  for (std::size_t i = 0; i < actual.size(); ++i)
    near(actual[i], expected[i], message + "[" + std::to_string(i) + "]");
}

model::Variable variable(const std::string& name, double lower = 0, double upper = INF) {
  return {name, model::VariableType::Continuous, lower, upper};
}

model::Constraint row(double lower, double upper,
                      std::vector<model::LinearTerm> terms) {
  return {"row", lower, upper, std::move(terms)};
}

presolve::PresolveResult identity(const model::Model& m) {
  presolve::PresolveResult p;
  p.model = m;
  p.originalVariables = p.presolvedVariables = m.variables.size();
  p.originalConstraints = p.presolvedConstraints = m.constraints.size();
  for (std::size_t i = 0; i < m.variables.size(); ++i) {
    p.postsolve.presolvedToOriginalVar.push_back(i);
    p.postsolve.originalToPresolvedVar.push_back(static_cast<int>(i));
  }
  for (std::size_t i = 0; i < m.constraints.size(); ++i) {
    p.postsolve.presolvedToOriginalConstraint.push_back(i);
    p.postsolve.originalToPresolvedConstraint.push_back(static_cast<int>(i));
  }
  return p;
}

std::vector<double> reducedPrimal(const presolve::PresolveResult& p,
                                  const std::vector<double>& original) {
  require(!p.infeasible && p.converged, "presolve must succeed and converge");
  std::vector<double> x;
  for (auto i : p.postsolve.presolvedToOriginalVar) x.push_back(original.at(i));
  return x;
}

void available(const postsolve::PostsolveResult& r,
               const std::vector<double>& duals, const std::vector<double>& costs) {
  require(r.isSuccess(), "primal reconstruction: " + r.errorMessage);
  require(r.dualsAvailable, "duals unavailable: " + r.dualsUnavailableReason);
  require(r.dualsUnavailableReason.empty(), "published duals must have no failure reason");
  vectorNear(r.constraintDuals, duals, "row duals");
  vectorNear(r.reducedCosts, costs, "reduced costs");
  near(r.maxDualResidual, 0, "dual residual");
}

void unavailable(const postsolve::PostsolveResult& r, const std::string& reason) {
  require(r.isSuccess(), "dual failure must preserve a valid primal");
  require(!r.dualsAvailable, "invalid duals must not be published");
  require(r.constraintDuals.empty() && r.reducedCosts.empty(),
          "unavailable duals and costs must be empty");
  require(r.dualsUnavailableReason.find(reason) != std::string::npos,
          "missing diagnostic: " + reason + "; got " + r.dualsUnavailableReason);
}

model::Model basicLp() {
  // min 3x + 5y, x+y >= 4, x,y >= 0. Optimum (4,0), price 3, costs (0,2).
  model::Model m;
  m.variables = {variable("x"), variable("y")};
  m.objective.linearTerms = {{0, 3}, {1, 5}};
  m.constraints = {row(4, INF, {{0, 1}, {1, 1}})};
  return m;
}

void known_shadow_price() {
  const auto m = basicLp();
  auto r = postsolve::Postsolver().process(m, identity(m), {4, 0}, {3});
  available(r, {3}, {0, 2});
  vectorNear(r.primalSolution, {4, 0}, "primal");
  near(r.originalObjectiveValue, 12, "objective");
}

void derived_bound() {
  // max 3x+4y, 4x+5y <= 28. Presolve gives y<=5.6. Reduced price .75
  // leaves .25 on that bound; original y is interior, so 4=5*price -> .8.
  model::Model m;
  m.variables = {variable("x"), variable("y")};
  m.objective.sense = model::ObjectiveSense::Maximize;
  m.objective.linearTerms = {{0, 3}, {1, 4}};
  m.constraints = {row(-INF, 28, {{0, 4}, {1, 5}})};
  auto p = presolve::Presolver().run(m);
  require(p.postsolve.presolvedToOriginalConstraint == std::vector<std::size_t>{0},
          "product-mix row must survive presolve");
  require(std::any_of(p.transformations.begin(), p.transformations.end(), [](const auto& t) {
    return t.type == Type::TightenUpperBound && t.originalVariableIndex == 1 &&
           t.originalConstraintIndex == 0 && std::abs(t.newValue - 5.6) < EPS;
  }), "presolve must log y's derived upper bound");
  auto x = reducedPrimal(p, {0, 5.6});
  auto r = postsolve::Postsolver().process(m, p, x, {0.75});
  available(r, {0.8}, {-0.2, 0});
  near(r.originalObjectiveValue, 22.4, "product-mix objective");

  // Losing the log leaves the reduced-space .75 invalid in original space.
  p.transformations.clear();
  unavailable(postsolve::Postsolver().process(m, p, x, {0.75}), "optimality");
}

void missing_bound_provenance() {
  // Both rows are tight at x=3. Either can support a valid KKT multiplier,
  // but that does not tell us WHICH row caused the derived upper bound.
  model::Model m;
  m.variables = {variable("x", 0, 10)};
  m.objective.sense = model::ObjectiveSense::Maximize;
  m.objective.linearTerms = {{0, 6}};
  m.constraints = {row(-INF, 6, {{0, 2}}), row(-INF, 12, {{0, 4}})};
  auto p = identity(m);
  p.model.variables[0].upperBound = 3;
  presolve::Transformation t;
  t.type = Type::TightenUpperBound;
  t.oldValue = 10;
  t.newValue = 3;
  // Deliberately leave the original source indices unset. Local index zero,
  // a tight row, and an active bound must not substitute for provenance.
  p.transformations = {t};
  unavailable(postsolve::Postsolver().process(m, p, {3}, {0, 0}), "provenance");

  // Explicitly identify row 1; it must receive the price, not tight row 0.
  t.originalVariableIndex = 0;
  t.originalConstraintIndex = 1;
  p.transformations = {t};
  available(postsolve::Postsolver().process(m, p, {3}, {0, 0}), {0, 1.5}, {0});
}

// Verify the actual singleton log, then compare the combined log with tightening
// alone. Removal alone has lost the causal bound record and must fail closed.
// Negative coefficients reverse bound sides.
void singleton(double lower, double upper, double optimum, double minCost) {
  for (double coefficient : {2.0, -2.0}) {
    for (double sense : {1.0, -1.0}) {
      model::Model m;
      m.variables = {variable("x", 0, 10)};
      m.objective.sense = sense > 0 ? model::ObjectiveSense::Minimize
                                    : model::ObjectiveSense::Maximize;
      m.objective.linearTerms = {{0, sense * minCost}};
      m.constraints = {coefficient > 0
          ? row(coefficient * lower, coefficient * upper, {{0, coefficient}})
          : row(coefficient * upper, coefficient * lower, {{0, coefficient}})};
      auto p = presolve::Presolver().run(m);
      auto x = reducedPrimal(p, {optimum});
      require(p.presolvedConstraints == 0 && p.postsolve.presolvedToOriginalConstraint.empty(),
              "singleton must leave zero reduced constraints");
      require(p.postsolve.removedConstraints.size() == 1, "one removed row record");
      const auto& rec = p.postsolve.removedConstraints.front();
      require(rec.wasSingleton && rec.originalIndex == 0 && rec.singletonOriginalVarIndex == 0,
              "singleton metadata must use original row/variable indices");
      near(rec.singletonCoefficient, coefficient, "singleton coefficient");
      int lowerCount = 0, upperCount = 0, removeCount = 0;
      bool removed = false;
      for (const auto& t : p.transformations) {
        if (t.type == Type::TightenLowerBound || t.type == Type::TightenUpperBound) {
          require(!removed, "tightening must precede singleton removal in the forward log");
          require(t.originalVariableIndex == 0 && t.originalConstraintIndex == 0,
                  "tightening must reference the singleton's original indices");
          if (t.type == Type::TightenLowerBound) {
            ++lowerCount;
            near(t.oldValue, 0, "old lower bound");
            near(t.newValue, lower, "derived lower bound");
          } else {
            ++upperCount;
            near(t.oldValue, 10, "old upper bound");
            near(t.newValue, upper, "derived upper bound");
          }
        } else if (t.type == Type::RemoveConstraint) {
          require(t.originalConstraintIndex == 0, "removed original row index");
          removed = true;
          ++removeCount;
        }
      }
      require(lowerCount == (std::isfinite(lower) ? 1 : 0) &&
              upperCount == (std::isfinite(upper) ? 1 : 0) && removeCount == 1,
              "expected tightening records followed by exactly one removal");
      const double price = sense * minCost / coefficient;
      auto r = postsolve::Postsolver().process(m, p, x, {});
      available(r, {price}, {0});
      vectorNear(r.primalSolution, {optimum}, "singleton primal");
      if (lower == upper)
        require(x.empty() && p.presolvedVariables == 0,
                "equality singleton must reconstruct an entirely eliminated model");

      for (bool retainRemoval : {false, true}) {
        auto singlePath = p;
        auto& log = singlePath.transformations;
        log.erase(std::remove_if(log.begin(), log.end(), [=](const auto& t) {
          return retainRemoval
              ? t.type == Type::TightenLowerBound || t.type == Type::TightenUpperBound
              : t.type == Type::RemoveConstraint;
        }), log.end());
        auto singleResult = postsolve::Postsolver().process(m, singlePath, x, {});
        if (retainRemoval) unavailable(singleResult, "provenance");
        else available(singleResult, {price}, {0});
      }
    }
  }
}

void singleton_lower() { singleton(3, INF, 3, 6); }
void singleton_upper() { singleton(-INF, 3, 3, -6); }
void singleton_equality() {
  singleton(3, 3, 3, 6);
  singleton(3, 3, 3, -6);
}
void singleton_ranged() {
  singleton(2, 4, 2, 6);
  singleton(2, 4, 4, -6);
}

void invalid_bound_indices() {
  model::Model m;
  m.variables = {variable("x", 0, 10), variable("absent", 0, 10)};
  m.objective.sense = model::ObjectiveSense::Maximize;
  m.objective.linearTerms = {{0, 6}};
  m.constraints = {row(-INF, 6, {{0, 2}})};
  presolve::Transformation valid;
  valid.type = Type::TightenUpperBound;
  valid.originalVariableIndex = 0;
  valid.originalConstraintIndex = 0;
  valid.oldValue = 10;
  valid.newValue = 3;
  auto p = identity(m);
  const auto check = [&](const presolve::Transformation& t, const std::string& reason) {
    auto bad = p;
    bad.transformations = {t};
    unavailable(postsolve::Postsolver().process(m, bad, {3, 3}, {0}), reason);
  };
  for (auto index : {presolve::INVALID_ORIGINAL_INDEX, m.constraints.size()}) {
    auto t = valid;
    t.originalConstraintIndex = index;
    check(t, "original constraint index");
  }
  for (auto index : {presolve::INVALID_ORIGINAL_INDEX, m.variables.size()}) {
    auto t = valid;
    t.originalVariableIndex = index;
    check(t, "original variable index");
  }
  auto t = valid;
  t.originalVariableIndex = 1;
  check(t, "absent from source row");
}

void invalid_bound_direction() {
  // All supplied primal points are feasible, all candidate rows are tight.
  // A valid source index still does not excuse an inconsistent bound record.
  for (double coefficient : {2.0, -2.0}) {
    model::Model m;
    m.variables = {variable("x", 0, 10)};
    m.objective.linearTerms = {{0, 0}};
    m.constraints = {coefficient > 0 ? row(-INF, 6, {{0, coefficient}})
                                    : row(-6, INF, {{0, coefficient}})};
    presolve::Transformation t;
    t.type = Type::TightenLowerBound;  // row actually implies an UPPER bound
    t.originalVariableIndex = t.originalConstraintIndex = 0;
    t.oldValue = 0;
    t.newValue = 3;
    auto p = identity(m);
    p.transformations = {t};
    unavailable(postsolve::Postsolver().process(m, p, {3}, {0}), "bound direction");
    t.type = Type::TightenUpperBound;
    t.oldValue = 10;
    t.newValue = 2;  // not the bound implied by this row
    p.transformations = {t};
    unavailable(postsolve::Postsolver().process(m, p, {3}, {0}), "not implied");
    t.newValue = 3;
    t.oldValue = 9;  // not the preceding upper bound
    p.transformations = {t};
    unavailable(postsolve::Postsolver().process(m, p, {3}, {0}), "bound history");
    t.oldValue = 10;
    t.newValue = 11;
    p.transformations = {t};
    unavailable(postsolve::Postsolver().process(m, p, {3}, {0}), "declared direction");
    for (double value : {INF, -INF, std::numeric_limits<double>::quiet_NaN()}) {
      t.newValue = value;
      p.transformations = {t};
      unavailable(postsolve::Postsolver().process(m, p, {3}, {0}), "bound value");
    }
  }
}

void invalid_source_coefficient() {
  // Zero/cancelling or overflowing sums of finite coefficients must fail even
  // though activity at x=0 remains finite, so the primal itself is valid.
  for (const auto& terms : std::vector<std::vector<model::LinearTerm>>{
           {{0, 0}}, {{0, 2}, {0, -2}},
           {{0, std::numeric_limits<double>::max()}, {0, std::numeric_limits<double>::max()}}}) {
    model::Model m;
    m.variables = {variable("x", 0, 10)};
    m.constraints = {row(-INF, 0, terms)};
    presolve::Transformation t;
    t.type = Type::TightenUpperBound;
    t.originalVariableIndex = t.originalConstraintIndex = 0;
    t.oldValue = 10;
    t.newValue = 0;
    auto p = identity(m);
    p.transformations = {t};
    unavailable(postsolve::Postsolver().process(m, p, {0}, {0}), "source coefficient");
  }
}

void invalid_singleton_provenance() {
  model::Model m;
  m.variables = {variable("x", 0, 10)};
  m.objective.sense = model::ObjectiveSense::Maximize;
  m.objective.linearTerms = {{0, 6}};
  m.constraints = {row(-INF, 6, {{0, 2}})};
  const auto p = presolve::Presolver().run(m);
  const auto x = reducedPrimal(p, {3});
  for (double a : {0.0, -2.0, 4.0, INF, std::numeric_limits<double>::quiet_NaN()}) {
    auto bad = p;
    bad.postsolve.removedConstraints.front().singletonCoefficient = a;
    unavailable(postsolve::Postsolver().process(m, bad, x, {}), "singleton provenance");
  }
  // The later removal must not consume d[k] and mask a corrupt tightening.
  auto bad = p;
  bad.transformations.front().originalConstraintIndex = presolve::INVALID_ORIGINAL_INDEX;
  unavailable(postsolve::Postsolver().process(m, bad, x, {}), "bound provenance");
  bad = p;
  std::reverse(bad.transformations.begin(), bad.transformations.end());
  unavailable(postsolve::Postsolver().process(m, bad, x, {}), "preceding tightening");
  bad = p;
  bad.transformations.back().originalVariableIndex = presolve::INVALID_ORIGINAL_INDEX;
  unavailable(postsolve::Postsolver().process(m, bad, x, {}), "indices do not agree");
}

void provenance_after_elimination() {
  // Original variable 0 and row 0 disappear. The source must still be named
  // using original indices (row 1, variable 1), with the fixed contribution
  // 2*1 accounted for in 2*fixed + 4*x <= 14 -> x<=3.
  model::Model m;
  m.variables = {variable("fixed", 1, 1), variable("x", 0, 10)};
  m.objective.sense = model::ObjectiveSense::Maximize;
  m.objective.linearTerms = {{1, 8}};
  m.constraints = {row(-INF, 1, {}), row(-INF, 14, {{0, 2}, {1, 4}})};
  const auto p = presolve::Presolver().run(m);
  require(p.postsolve.presolvedToOriginalVar == std::vector<std::size_t>{1},
          "original variable zero must be eliminated");
  require(std::any_of(p.transformations.begin(), p.transformations.end(), [](const auto& t) {
    return t.type == Type::TightenUpperBound && t.originalVariableIndex == 1 &&
           t.originalConstraintIndex == 1 && t.index == 0;
  }), "source provenance must survive index compaction");
  auto r = postsolve::Postsolver().process(m, p, reducedPrimal(p, {1, 3}), {});
  available(r, {0, 2}, {-4, 0});
  vectorNear(r.primalSolution, {1, 3}, "primal after elimination");
}

void provenance_bound_history() {
  // z>=1 is tightened first. Only that earlier bound allows x+z<=4 to
  // derive x<=3; using the optimum z=1 to invent this history is forbidden.
  model::Model m;
  m.variables = {variable("x", 0, 10), variable("z", 0, 10)};
  m.objective.sense = model::ObjectiveSense::Maximize;
  m.objective.linearTerms = {{0, 2}};
  m.constraints = {row(-INF, 4, {{0, 1}, {1, 1}}), row(1, INF, {{1, 1}})};
  const auto p = presolve::Presolver().run(m);
  const auto x = reducedPrimal(p, {3, 1});
  available(postsolve::Postsolver().process(m, p, x, {0}), {2, -2}, {0, 0});
  auto bad = p;
  auto& log = bad.transformations;
  log.erase(std::remove_if(log.begin(), log.end(), [](const auto& t) {
    return t.type == Type::TightenLowerBound && t.originalConstraintIndex == 1;
  }), log.end());
  unavailable(postsolve::Postsolver().process(m, bad, x, {0}), "not implied");
}

void derived_bound_directions() {
  // Exercise all four general (non-singleton) bound-tightening emitters:
  // lower/upper row side crossed with positive/negative target coefficient.
  for (bool lowerRow : {false, true}) {
    for (double a : {2.0, -2.0}) {
      const bool lowerVariable = lowerRow == (a > 0);
      const double z = lowerRow ? 1 : 0;
      const double cost = lowerVariable ? 6 : -6;
      const double rhs = a*3 + z;
      model::Model m;
      m.variables = {variable("x", 0, 10), variable("z", 0, 1)};
      m.objective.linearTerms = {{0, cost}};
      m.constraints = {lowerRow ? row(rhs, INF, {{0, a}, {1, 1}})
                                : row(-INF, rhs, {{0, a}, {1, 1}})};
      const auto p = presolve::Presolver().run(m);
      require(p.postsolve.presolvedToOriginalConstraint == std::vector<std::size_t>{0},
              "source row must survive");
      require(std::any_of(p.transformations.begin(), p.transformations.end(), [&](const auto& t) {
        return t.type == (lowerVariable ? Type::TightenLowerBound : Type::TightenUpperBound) &&
               t.originalVariableIndex == 0 && t.originalConstraintIndex == 0 &&
               std::abs(t.newValue - 3) < EPS;
      }), "presolve must emit the expected source and bound direction");
      available(postsolve::Postsolver().process(m, p, reducedPrimal(p, {3, z}), {0}),
                {cost/a}, {0, -cost/a});
    }
  }
}

void inactive_singleton() {
  for (bool upperRow : {false, true}) {
    model::Model m;
    m.variables = {variable("x", 0, 10)};
    m.objective.linearTerms = {{0, upperRow ? 0.2 : -0.2}};
    m.constraints = {upperRow ? row(-INF, 6, {{0, 2}}) : row(6, INF, {{0, 2}})};
    auto p = presolve::Presolver().run(m);
    require(p.postsolve.removedConstraints.size() == 1 &&
            p.postsolve.removedConstraints.front().wasSingleton,
            "inactive row must be removed as a singleton");
    auto x = reducedPrimal(p, {upperRow ? 0.0 : 10.0});
    // x is at its ORIGINAL bound; the row is slack and must receive no price.
    available(postsolve::Postsolver().process(m, p, x, {}),
              {0}, {upperRow ? 0.2 : -0.2});
  }
}

void reduced_costs() {
  for (double sense : {1.0, -1.0}) {
    model::Model m;
    m.variables = {variable("interior", 0, 10), variable("lower", 0, 10),
                   variable("upper", 0, 5)};
    m.objective.sense = sense > 0 ? model::ObjectiveSense::Minimize
                                  : model::ObjectiveSense::Maximize;
    m.objective.linearTerms = {{0, 2*sense}, {1, 5*sense}, {2, -2*sense}};
    m.constraints = {row(7, 7, {{0, 1}, {1, 1}, {2, 1}})};
    // grad - A^T y = (2,5,-2) - (2,2,2) = (0,3,-4) for minimization.
    available(postsolve::Postsolver().process(m, identity(m), {2, 0, 5}, {2*sense}),
              {2*sense}, {0, 3*sense, -4*sense});
  }
}

void row_dual(double lower, double upper, double cost) {
  for (double sense : {1.0, -1.0}) {
    model::Model m;
    m.variables = {variable("x", -INF, INF)};
    m.objective.sense = sense > 0 ? model::ObjectiveSense::Minimize
                                  : model::ObjectiveSense::Maximize;
    m.objective.linearTerms = {{0, sense * cost}};
    m.constraints = {row(lower, upper, {{0, 2}})};
    const double price = sense * cost / 2;
    available(postsolve::Postsolver().process(m, identity(m), {3}, {price}), {price}, {0});
  }
}
void equality_dual() { row_dual(6, 6, 6); row_dual(6, 6, -6); }
void lower_dual() { row_dual(6, INF, 6); }
void upper_dual() { row_dual(-INF, 6, -6); }

void fixed_variable() {
  for (double cost : {-3.0, 3.0}) {
    model::Model m;
    m.variables = {variable("fixed", 2, 2), variable("interior", 0, 10)};
    m.objective.linearTerms = {{0, cost}, {1, -2}};
    m.objective.quadraticTerms = {{0, 1, 1}};
    auto p = presolve::Presolver().run(m);
    require(p.postsolve.fixedVariables.size() == 1 &&
            p.postsolve.presolvedToOriginalVar == std::vector<std::size_t>{1},
            "fixed variable must be eliminated, leaving original variable 1");
    auto r = postsolve::Postsolver().process(m, p, reducedPrimal(p, {2, 1}), {});
    // f = cost*x - 2y + xy; original gradient at (2,1) is (cost+1, 0).
    available(r, {}, {cost + 1, 0});
    vectorNear(r.primalSolution, {2, 1}, "fixed-variable primal");
    near(r.originalObjectiveValue, 2*cost, "fixed-variable objective");
  }
}

void empty_duals() {
  model::Model m;
  m.variables = {variable("x", 0, 10), variable("y", 0, 5)};
  m.objective.linearTerms = {{0, 2}, {1, -3}};
  available(postsolve::Postsolver().process(m, identity(m), {0, 5}, {}), {}, {2, -3});
  // Also cover the genuinely empty original model.
  model::Model empty;
  available(postsolve::Postsolver().process(empty, identity(empty), {}, {}), {}, {});
}

void wrong_dual_size() {
  const auto m = basicLp();
  unavailable(postsolve::Postsolver().process(m, identity(m), {4, 0}, {}), "no reduced-space");
  unavailable(postsolve::Postsolver().process(m, identity(m), {4, 0}, {3, 0}), "entries");
  auto twoRows = m;
  twoRows.constraints.push_back(row(-INF, 10, {{0, 1}}));
  unavailable(postsolve::Postsolver().process(twoRows, identity(twoRows), {4, 0}, {3}), "entries");
  model::Model empty;
  unavailable(postsolve::Postsolver().process(empty, identity(empty), {}, {1}), "entries");
}

void invalid_constraint_mapping() {
  const auto m = basicLp();
  auto p = identity(m);
  p.postsolve.presolvedToOriginalConstraint[0] = m.constraints.size();
  unavailable(postsolve::Postsolver().process(m, p, {4, 0}, {3}), "mapping");
}

void residual_gate() {
  const auto m = basicLp();
  auto r = postsolve::Postsolver().process(m, identity(m), {4, 0}, {2});
  unavailable(r, "optimality");
  near(r.maxDualResidual, 1, "interior stationarity failure");
  vectorNear(r.primalSolution, {4, 0}, "primal retained after dual failure");
  near(r.originalObjectiveValue, 12, "objective retained after dual failure");

  // Isolate row sign, row slackness, and bound-sign failures while keeping
  // stationarity exact. A feasibility-only check would accept each of these.
  model::Model one;
  one.variables = {variable("x", 0, 10)};
  one.objective.linearTerms = {{0, -1}};
  one.constraints = {row(1, INF, {{0, 1}})};
  unavailable(postsolve::Postsolver().process(one, identity(one), {1}, {-1}), "optimality");
  one.objective.linearTerms = {{0, 1}};
  one.constraints = {row(-INF, 1, {{0, 1}})};
  unavailable(postsolve::Postsolver().process(one, identity(one), {1}, {1}), "optimality");
  one.constraints = {row(0, INF, {{0, 1}})};
  unavailable(postsolve::Postsolver().process(one, identity(one), {1}, {1}), "optimality");
  one.constraints.clear();
  unavailable(postsolve::Postsolver().process(one, identity(one), {10}, {}), "optimality");
  one.objective.linearTerms = {{0, -1}};
  unavailable(postsolve::Postsolver().process(one, identity(one), {0}, {}), "optimality");
}

void nonfinite_duals() {
  const auto m = basicLp();
  for (double dual : {std::numeric_limits<double>::quiet_NaN(), INF, -INF})
    unavailable(postsolve::Postsolver().process(m, identity(m), {4, 0}, {dual}), "non-finite");
}

void nonfinite_reconstruction() {
  const double largest = std::numeric_limits<double>::max();
  // All inputs and the primal objective are finite, but 2*q*x overflows.
  // Being fixed must not exempt a variable from finite reduced-cost checks.
  model::Model m;
  m.variables = {variable("fixed", 1, 1)};
  m.objective.quadraticTerms = {{0, 0, largest}};
  auto r = postsolve::Postsolver().process(m, identity(m), {1}, {});
  unavailable(r, "optimality");
  require(std::isinf(r.maxDualResidual), "non-finite gradient must fail the residual gate");

  // A finite engine dual can overflow A^T*y even with a zero objective.
  m.objective.quadraticTerms.clear();
  m.constraints = {row(2, 2, {{0, 2}})};
  r = postsolve::Postsolver().process(m, identity(m), {1}, {largest});
  unavailable(r, "optimality");
  require(std::isinf(r.maxDualResidual), "non-finite cost must fail the residual gate");
}

void quadratic_gradient() {
  // f = 7 + 3x - 4y + 2x^2 + 5xy + 3y^2.
  // At (2,3): grad = (3+8+15, -4+10+18) = (26,24); f=66.
  model::Model m;
  m.variables = {variable("x"), variable("y")};
  m.objective.offset = 7;
  m.objective.linearTerms = {{0, 1}, {0, 2}, {1, -4}};
  m.objective.quadraticTerms = {{0, 0, 2}, {0, 1, 5}, {1, 1, 3}};
  for (auto sense : {model::ObjectiveSense::Minimize, model::ObjectiveSense::Maximize}) {
    m.objective.sense = sense;
    vectorNear(postsolve::Postsolver().objectiveGradient(m, {2, 3}), {26, 24}, "gradient");
    near(postsolve::Postsolver().evaluateObjective(m, {2, 3}), 66, "direct quadratic objective");
  }
}

void quadratic_reconstruction() {
  // min 2x^2 subject to x>=3. Price = 4x = 12, not 6.
  model::Model m;
  m.variables = {variable("x", 0, 10)};
  m.objective.quadraticTerms = {{0, 0, 2}};
  m.constraints = {row(3, INF, {{0, 1}})};
  auto p = presolve::Presolver().run(m);
  auto r = postsolve::Postsolver().process(m, p, reducedPrimal(p, {3}), {});
  available(r, {12}, {0});
  near(r.originalObjectiveValue, 18, "quadratic objective");
  unavailable(postsolve::Postsolver().process(m, identity(m), {3}, {6}), "optimality");

  // Interior minimizer of 2x^2-8x is x=2; correct gradient makes cost zero.
  m.constraints.clear();
  m.objective.linearTerms = {{0, -8}};
  available(postsolve::Postsolver().process(m, identity(m), {2}, {}), {}, {0});
}

void primal_only() {
  const auto m = basicLp();
  auto r = postsolve::Postsolver().process(m, identity(m), {4, 0});
  require(r.isSuccess() && !r.dualsAvailable, "three-argument overload stays primal-only");
  require(r.constraintDuals.empty() && r.reducedCosts.empty(), "primal-only dual vectors empty");
  auto invalid = postsolve::Postsolver().process(m, identity(m), {-1, 0}, {3});
  require(!invalid.isSuccess() && !invalid.dualsAvailable && invalid.constraintDuals.empty() &&
          invalid.reducedCosts.empty(), "invalid primal must never publish duals");
}

struct Test { const char* name; void (*run)(); };
const Test tests[] = {
    {"missing_bound_provenance", missing_bound_provenance},
    {"invalid_bound_indices", invalid_bound_indices},
    {"invalid_bound_direction", invalid_bound_direction},
    {"invalid_source_coefficient", invalid_source_coefficient},
    {"invalid_singleton_provenance", invalid_singleton_provenance},
    {"provenance_after_elimination", provenance_after_elimination},
    {"provenance_bound_history", provenance_bound_history},
    {"derived_bound_directions", derived_bound_directions},
    {"known_shadow_price", known_shadow_price}, {"derived_bound", derived_bound},
    {"singleton_lower", singleton_lower}, {"singleton_upper", singleton_upper},
    {"singleton_equality", singleton_equality}, {"singleton_ranged", singleton_ranged},
    {"inactive_singleton", inactive_singleton}, {"reduced_costs", reduced_costs},
    {"equality_dual", equality_dual}, {"lower_dual", lower_dual}, {"upper_dual", upper_dual},
    {"fixed_variable", fixed_variable}, {"empty_duals", empty_duals},
    {"wrong_dual_size", wrong_dual_size}, {"invalid_constraint_mapping", invalid_constraint_mapping},
    {"residual_gate", residual_gate}, {"nonfinite_duals", nonfinite_duals},
    {"nonfinite_reconstruction", nonfinite_reconstruction},
    {"quadratic_gradient", quadratic_gradient}, {"quadratic_reconstruction", quadratic_reconstruction},
    {"primal_only", primal_only}};
}  // namespace

int main(int argc, char** argv) {
  int run = 0, failed = 0;
  for (const auto& test : tests) {
    if (argc > 1 && std::string(argv[1]) != test.name) continue;
    ++run;
    try {
      test.run();
      std::cout << "[PASSED] " << test.name << '\n';
    } catch (const std::exception& error) {
      ++failed;
      std::cerr << "[FAILED] " << test.name << ": " << error.what() << '\n';
    }
  }
  if (run == 0) std::cerr << "Unknown test case\n";
  return run > 0 && failed == 0 ? 0 : 1;
}
