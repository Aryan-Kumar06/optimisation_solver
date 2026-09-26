#pragma once

#include "model/model.h"

#include <cstddef>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace presolve {

constexpr std::size_t INVALID_ORIGINAL_INDEX = std::numeric_limits<std::size_t>::max();

enum class TransformationType {
  RemoveVariable,
  RemoveConstraint,
  FixVariable,
  SubstituteVariable,
  TightenLowerBound,
  TightenUpperBound
};

struct Transformation {
  TransformationType type = TransformationType::RemoveVariable;

  // Local index at the moment the transformation occurred (retained for backward compatibility)
  std::size_t index = 0;

  // Stable original indices that remain valid after erasures. Bound tightening
  // must explicitly name BOTH the affected variable and its originating row.
  // An unset index must never be mistaken for original variable/constraint zero.
  std::size_t originalVariableIndex = INVALID_ORIGINAL_INDEX;
  std::size_t originalConstraintIndex = INVALID_ORIGINAL_INDEX;

  double oldValue = 0.0;
  double newValue = 0.0;

  std::string reason;

  // Additional metadata
  std::string entityName;
  double offsetChange = 0.0;
};

// ============================================================
// STRUCTURED POSTSOLVE METADATA
// ============================================================

struct FixedVariableRecord {
  std::size_t originalIndex = 0;
  std::string name;
  double fixedValue = 0.0;
  model::VariableType type = model::VariableType::Continuous;

  // Objective contributions from fixing this variable
  double linearObjectiveContribution = 0.0;   // c_k * v added to offset
  double quadraticDiagonalContribution = 0.0; // q_kk * v^2 added to offset

  // Bilinear cross contributions: originalIndex of other variable -> (q_kj * v)
  std::map<std::size_t, double> quadraticCrossContributions;
};

struct RemovedConstraintRecord {
  std::size_t originalIndex = 0;
  std::string name;
  double lowerBound = 0.0;
  double upperBound = 0.0;
  std::string reason;

  // Singleton row details
  bool wasSingleton = false;
  std::size_t singletonOriginalVarIndex = 0;
  double singletonCoefficient = 0.0;

  // Duplicate / parallel row details
  bool wasDuplicate = false;
  std::size_t duplicateOfOriginalIndex = 0;
  double scale = 1.0;
};

struct PostsolveMetadata {
  // Mapping from original variable index to presolved variable index (-1 if eliminated)
  std::vector<int> originalToPresolvedVar;
  // Mapping from presolved variable index to original variable index
  std::vector<std::size_t> presolvedToOriginalVar;

  // Mapping from original constraint index to presolved constraint index (-1 if eliminated)
  std::vector<int> originalToPresolvedConstraint;
  // Mapping from presolved constraint index to original constraint index
  std::vector<std::size_t> presolvedToOriginalConstraint;

  // Objective offsets
  double initialObjectiveOffset = 0.0;
  double presolvedObjectiveOffset = 0.0;

  // Structured records of eliminated entities
  std::vector<FixedVariableRecord> fixedVariables;
  std::vector<RemovedConstraintRecord> removedConstraints;
};

} // namespace presolve
