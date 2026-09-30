#pragma once

#include "model/model.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace solver {

// What kind of problem this is, decided from the model's own structure.
//
// Classification runs BEFORE presolve, because presolve's reductions differ by
// class: coefficient tightening on big-M rows, probing and clique merging are
// only valid, and only worth doing, when the presolver knows which variables are
// integer. A presolver that does not know it is looking at a MILP leaves its
// most valuable reductions on the table.
enum class ProblemClass {
    LP,
    MILP,
    QP,
    MIQP,
    NLP, // Smooth nonlinear Problem interface; never inferred from affine Model.

    // Quadratically constrained problems cannot currently be expressed:
    // model::Constraint carries only linearTerms. Kept so the enum does not
    // have to change when the IR gains quadratic constraints; never returned
    // by classify() today.
    QCQP
};

[[nodiscard]] const char* toString(ProblemClass value) noexcept;

// Structural properties worth knowing before choosing reductions or an engine.
// All are cheap: one pass to build a column view, then linear or n log n work.
struct StructureHints {
    // Every column has exactly one +1 and one -1: a node-arc incidence matrix.
    // Network LPs are totally unimodular, so their LP relaxation is integral.
    bool hasNetworkStructure = false;

    // Rows pairing a continuous variable against a binary with a much larger
    // coefficient -- the classic `continuous <= M * binary` indicator pattern.
    bool hasBigM = false;
    double maxBigM = 0.0;

    // Equality rows over binaries with all coefficients 1 and right-hand side 1.
    bool hasSetPartitioning = false;

    // Groups of two or more structurally identical columns. Symmetry is what
    // makes branch-and-bound explore equivalent subtrees repeatedly.
    int symmetricGroups = 0;

    // max|a_ij| / min|a_ij| over the nonzeros. A large ratio predicts the
    // ill-conditioning that equilibration exists to fix.
    double coefRangeRatio = 0.0;

    int numBinary = 0;
    int numInteger = 0;      // integer but not binary
    int numContinuous = 0;

    std::size_t numRows = 0;
    std::size_t numColumns = 0;
    // -1 when the nonlinear Jacobian pattern is unavailable before evaluation.
    std::int64_t numNonzeros = 0;
};

struct Classification {
    ProblemClass problemClass = ProblemClass::LP;
    StructureHints hints;
};

// Structural, cheap and deterministic. Does not modify the model.
[[nodiscard]] Classification classify(const model::Model& model);

}  // namespace solver
