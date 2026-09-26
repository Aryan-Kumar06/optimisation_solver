#pragma once

#include "model/model.h"
#include "solver/classifier.h"
#include "solver/dispatcher.h"
#include "solver/solve_result.h"

namespace solver {

// The whole pipeline behind one call.
//
//   classify (before presolve)  ->  presolve  ->  dispatch (after presolve)
//   ->  the chosen engine  ->  postsolve  ->  original-model SolveResult
//
// Classification happens before presolve because presolve's reductions depend
// on the problem class; dispatch happens after because presolve changes the
// size, can eliminate every integer variable, and can settle infeasibility
// outright.
//
// Primal values, shadow prices and reduced costs are reconstructed and validated
// in the original model's coordinates. Presolve runs once, and its same metadata
// is used for reconstruction. Check hasPrimal/hasDuals before consuming vectors.
[[nodiscard]] SolveResult solve(
    const model::Model& model,
    const SolverOptions& options = {}
);

// Solves a model that has ALREADY been reduced/presolved.
//
// Dispatches directly to the selected engine on `presolvedModel` using the
// `classification` of the original model, and normalises the result into a
// SolveResult in the coordinates of `presolvedModel`.
// Unlike solve(), this does NOT run presolve or expand into another model's
// coordinates or invoke postsolve reconstruction. It validates primal/dual
// results directly against presolvedModel itself.
[[nodiscard]] SolveResult solveReduced(
    const model::Model& presolvedModel,
    const Classification& classification,
    const SolverOptions& options = {}
);

}  // namespace solver
