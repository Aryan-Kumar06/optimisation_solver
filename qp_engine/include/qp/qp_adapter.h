#pragma once

#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ < 7
#pragma GCC diagnostic ignored "-Wattributes"
#endif

#include "model/model.h"
#include "qp/qp_model.h"

#include <vector>

namespace qp {

// Converts a model::Model (which may contain quadratic terms) into a qp::QpModel
// in standard form.
//
// Conversion rules:
//   - Maximisation → minimisation (negate objective)
//   - Equality constraints (l == u) are preserved as-is
//   - One-sided constraints (l == -inf or u == +inf) are kept
//   - Quadratic objective terms are accumulated into the P matrix
//     (upper-triangular, so P_ij += term.value for i <= j)
//
// For the convex QP requirement: if P is detected as not SPD, the adapter
// sets a flag and the solver will fall back to non-convex handling.
//
// QpModel has no notion of "the original problem" -- like pdlp::CompiledLp,
// it is purely the engine-facing minimisation form. Everything needed to map
// an AdmmResult back to the original problem's objective goes here, in the
// side channel, mirroring adapter::PdlpTranslation.
struct QpTranslation {
    // True when the original model was Maximize, meaning fromModel negated
    // both P and q to produce an equivalent minimisation problem. The caller
    // must negate AdmmResult::primalObjective back before adding the offset:
    // original_objective = (objectiveNegated ? -raw : raw) + objectiveOffset.
    bool objectiveNegated = false;

    // model.objective.offset, carried through unchanged -- the offset is a
    // constant shift and is sign-independent of Maximize/Minimize, unlike P
    // and q. Added AFTER un-negating primalObjective, never before.
    double objectiveOffset = 0.0;
};

[[nodiscard]] QpModel fromModel(const model::Model& model, QpTranslation& translation);

// THE RETURN HALF. Required, and it was missing.
//
// fromModel() existed without a counterpart, so AdmmResult::constraintDual was
// copied straight out to callers. It is not a shadow price. The ADMM's
// stationarity condition is
//
//     P x + q + A^T y = 0
//
// while the usual Lagrange multiplier lambda satisfies A^T lambda = grad f, so
// y = -lambda. Exactly the same convention mismatch pdlp_adapter documents and
// corrects; this engine simply had nowhere to correct it.
//
// A maximisation adds a second negation, because fromModel() negated P and q to
// hand the engine an equivalent minimisation. The two compose rather than
// cancel:
//
//     minimisation:  shadow price = -y_admm
//     maximisation:  shadow price = +y_admm
//
// Measured before this existed: on "min x^2 + y^2 s.t. x + y >= 2" the engine
// returned -2 where the shadow price is +2. The primal was exact, so nothing
// downstream looked wrong; postsolve's residual gate refused to publish the
// duals with a violation of 4.0 and the solve reported optimal with no
// sensitivity information at all.
//
// Modifies `duals` in place. The QP adapter appends one identity row per
// bounded variable, so entries beyond the model's row count are bound
// multipliers rather than row duals; they take the same sign convention and are
// converted too, so a caller that keeps them gets consistent values.
void toModelDuals(const QpTranslation& translation, std::vector<double>& duals);

// Convenience overload for callers that don't need the translation (existing
// tests, ad hoc use). Prefer the two-argument form when reporting objective
// values back to a caller of the original model.
[[nodiscard]] QpModel fromModel(const model::Model& model);

}  // namespace qp
