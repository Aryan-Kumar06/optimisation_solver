"""Independent verification of a returned point against the ORIGINAL model.

Nothing here trusts the solver. The objective is recomputed from the model as
parsed by mps_model.py (a separate reader), residuals are recomputed from the
original rows and bounds before any presolve or scaling, and the verdict is
formed only from those recomputations.

THE VERDICT IS NOT THE SOLVER'S STATUS. They are reported side by side and
never merged. A solver saying "optimal" and a checker saying "feasible" is a
meaningful, reportable disagreement -- and a feasible point is not a proof of
optimality no matter how well it matches a best-known objective.

FROZEN TOLERANCES (see benchmarks/README.md; do not change for scored runs):

    feasibility     absolute 1e-6, relative 1e-8
    integrality     absolute 1e-6
    optimality      normalised 1e-6

SCALING, stated explicitly because "relative to what" is where these
comparisons usually go wrong:

    row i          tol_i = 1e-6 + 1e-8 * s_i
                   s_i   = max(1, |l_i|, |u_i|, sum_j |a_ij * x_j|)
                   The last term is the row's own activity magnitude, so a row
                   that sums a million large terms is not held to the same
                   absolute residual as a row of two small ones.

    variable j     tol_j = 1e-6 + 1e-8 * max(1, |lb_j|, |ub_j|, |x_j|)

    optimality     gap_norm = |p - d| / (1 + |p| + |d|)
                   with p the primal objective and d the dual objective.

Absolute residuals are reported alongside every normalised one, so a reader can
apply a different rule without re-running anything.
"""

from __future__ import annotations

import math

FEAS_ATOL = 1e-6
FEAS_RTOL = 1e-8
INT_ATOL = 1e-6
OPT_NORM_TOL = 1e-6

TOLERANCES = {
    "feasibility_absolute": FEAS_ATOL,
    "feasibility_relative": FEAS_RTOL,
    "integrality_absolute": INT_ATOL,
    "optimality_normalised": OPT_NORM_TOL,
    "frozen": True,
}

INF = math.inf


class Verdict:
    """Checker outcomes. Deliberately distinct vocabulary from SolveStatus."""
    OPTIMAL_VERIFIED = "optimal_verified"      # feasible AND gap/KKT proven
    FEASIBLE = "feasible"                      # feasible, optimality NOT proven
    INFEASIBLE_POINT = "infeasible_point"      # returned point violates the model
    NONFINITE = "nonfinite"                    # NaN or infinity in the point
    NO_POINT = "no_point"                      # nothing to check
    MALFORMED = "malformed"                    # wrong length, missing fields
    NOT_APPLICABLE = "not_applicable"          # solver claimed infeasible/unbounded


def _violation(value, lower, upper):
    """How far outside [lower, upper] a value sits. Zero when inside."""
    below = (lower - value) if lower > -INF else 0.0
    above = (value - upper) if upper < INF else 0.0
    return max(0.0, below, above)


def check_solution(model, x, duals=None, reduced_costs=None,
                   solver_status=None, best_known=None):
    """Verify a point. Returns a dict; every unavailable number is None."""
    report = {
        "verdict": None,
        "solver_status": solver_status,
        "tolerances": dict(TOLERANCES),
        "objective_recomputed": None,
        "nonfinite": {"count": 0, "variables": []},
        "primal": {
            "max_row_violation_abs": None,
            "max_row_violation_scaled": None,
            "worst_row": None,
            "max_bound_violation_abs": None,
            "max_bound_violation_scaled": None,
            "worst_variable": None,
            "rows_violating": 0,
            "variables_violating": 0,
        },
        "integrality": {
            "checked": False,
            "max_violation_abs": None,
            "worst_variable": None,
            "variables_violating": 0,
        },
        "dual": {
            "available": False,
            "reason": None,
            "max_stationarity_abs": None,
            "max_sign_violation_abs": None,
            "max_complementarity_abs": None,
            "dual_objective": None,
            "gap_abs": None,
            "gap_normalised": None,
        },
        "best_known": {
            "value": best_known,
            "abs_difference": None,
            "normalised_difference": None,
            "agrees": None,
        },
        "notes": [],
    }

    if x is None:
        report["verdict"] = Verdict.NO_POINT
        report["notes"].append("solver returned no primal point")
        return report

    if len(x) != model.n:
        report["verdict"] = Verdict.MALFORMED
        report["notes"].append(
            f"primal has {len(x)} entries, model has {model.n} variables")
        return report

    # ---- non-finite values ------------------------------------------------
    bad = [j for j, value in enumerate(x) if not math.isfinite(value)]
    if bad:
        report["nonfinite"]["count"] = len(bad)
        report["nonfinite"]["variables"] = [model.var_names[j] for j in bad[:10]]
        report["verdict"] = Verdict.NONFINITE
        report["notes"].append("point contains NaN or infinity; nothing else checked")
        return report

    # ---- objective, recomputed from the original model --------------------
    objective = model.objective(x)
    report["objective_recomputed"] = objective if math.isfinite(objective) else None
    if not math.isfinite(objective):
        report["verdict"] = Verdict.NONFINITE
        report["notes"].append("objective evaluated to a non-finite value")
        return report

    # ---- variable bounds --------------------------------------------------
    worst_bound_abs, worst_bound_scaled, worst_var = 0.0, 0.0, None
    variables_violating = 0
    for j, value in enumerate(x):
        lower, upper = model.var_lower[j], model.var_upper[j]
        raw = _violation(value, lower, upper)
        scale = max(1.0,
                    abs(lower) if lower > -INF else 0.0,
                    abs(upper) if upper < INF else 0.0,
                    abs(value))
        tolerance = FEAS_ATOL + FEAS_RTOL * scale
        if raw > tolerance:
            variables_violating += 1
        scaled = raw / scale
        if raw > worst_bound_abs:
            worst_bound_abs, worst_var = raw, model.var_names[j]
        worst_bound_scaled = max(worst_bound_scaled, scaled)

    # ---- rows -------------------------------------------------------------
    activities = model.row_activity(x)
    worst_row_abs, worst_row_scaled, worst_row = 0.0, 0.0, None
    rows_violating = 0
    for i, activity in enumerate(activities):
        lower, upper = model.row_lower[i], model.row_upper[i]
        raw = _violation(activity, lower, upper)
        magnitude = sum(abs(coefficient * x[column])
                        for column, coefficient in model.rows[i].items())
        scale = max(1.0,
                    abs(lower) if lower > -INF else 0.0,
                    abs(upper) if upper < INF else 0.0,
                    magnitude)
        tolerance = FEAS_ATOL + FEAS_RTOL * scale
        if raw > tolerance:
            rows_violating += 1
        if raw > worst_row_abs:
            worst_row_abs, worst_row = raw, model.row_names[i]
        worst_row_scaled = max(worst_row_scaled, raw / scale)

    report["primal"].update({
        "max_row_violation_abs": worst_row_abs,
        "max_row_violation_scaled": worst_row_scaled,
        "worst_row": worst_row,
        "max_bound_violation_abs": worst_bound_abs,
        "max_bound_violation_scaled": worst_bound_scaled,
        "worst_variable": worst_var,
        "rows_violating": rows_violating,
        "variables_violating": variables_violating,
    })

    # ---- integrality ------------------------------------------------------
    integral_columns = [j for j in range(model.n) if model.var_type[j] != "C"]
    if integral_columns:
        report["integrality"]["checked"] = True
        worst_int, worst_int_var, count = 0.0, None, 0
        for j in integral_columns:
            violation = abs(x[j] - round(x[j]))
            if violation > INT_ATOL:
                count += 1
            if violation > worst_int:
                worst_int, worst_int_var = violation, model.var_names[j]
        report["integrality"].update({
            "max_violation_abs": worst_int,
            "worst_variable": worst_int_var,
            "variables_violating": count,
        })

    primal_ok = (rows_violating == 0 and variables_violating == 0
                 and report["integrality"]["variables_violating"] == 0)

    # ---- duals / KKT ------------------------------------------------------
    if duals is None:
        report["dual"]["reason"] = (
            "solver produced no duals; optimality is NOT independently verified")
    else:
        _check_duals(model, x, duals, reduced_costs, activities, objective, report)

    # ---- best known -------------------------------------------------------
    if best_known is not None:
        difference = abs(objective - best_known)
        report["best_known"]["abs_difference"] = difference
        normalised = difference / (1.0 + abs(objective) + abs(best_known))
        report["best_known"]["normalised_difference"] = normalised
        report["best_known"]["agrees"] = normalised <= OPT_NORM_TOL

    # ---- verdict ----------------------------------------------------------
    if not primal_ok:
        report["verdict"] = Verdict.INFEASIBLE_POINT
    elif report["dual"]["available"] and report["dual"]["gap_normalised"] is not None \
            and report["dual"]["gap_normalised"] <= OPT_NORM_TOL \
            and (report["dual"]["max_sign_violation_abs"] or 0.0) <= FEAS_ATOL \
            and (report["dual"]["max_stationarity_abs"] or 0.0) <= 1e-5:
        report["verdict"] = Verdict.OPTIMAL_VERIFIED
    else:
        report["verdict"] = Verdict.FEASIBLE
        if report["dual"]["available"]:
            report["notes"].append(
                "duals present but KKT/gap did not close within tolerance; "
                "optimality not proven")
        # Matching a best-known objective is evidence, never a proof: it cannot
        # distinguish a true optimum from a coincidence, and the reference value
        # itself is an external claim.
        if report["best_known"]["agrees"]:
            report["notes"].append(
                "objective agrees with the best-known value, which is corroboration, "
                "not an optimality proof")

    return report


def _check_duals(model, x, duals, reduced_costs, activities, objective, report):
    """KKT checks in minimisation form.

    A maximisation is converted by negating the objective, and the multipliers
    are negated with it: d(-f)/db = -df/db. Doing one without the other is the
    classic way to get shadow prices that claim relaxing a binding constraint
    hurts.
    """
    if len(duals) != model.m:
        report["dual"]["reason"] = (
            f"dual vector has {len(duals)} entries, model has {model.m} rows")
        return

    flip = -1.0 if model.sense == "max" else 1.0
    minimisation = model.to_minimization()
    y = [flip * value for value in duals]

    if any(not math.isfinite(value) for value in y):
        report["dual"]["reason"] = "dual vector contains non-finite values"
        return

    gradient = minimisation.gradient(x)

    # Reduced costs: use the solver's if supplied, else derive them. Deriving
    # makes stationarity hold by construction, so it is reported as such rather
    # than presented as an independent check.
    derived = False
    if reduced_costs is not None and len(reduced_costs) == model.n:
        d = [flip * value for value in reduced_costs]
        if any(not math.isfinite(value) for value in d):
            report["dual"]["reason"] = "reduced costs contain non-finite values"
            return
    else:
        derived = True
        d = [0.0] * model.n
        for j in range(model.n):
            d[j] = gradient[j]
        for i, row in enumerate(minimisation.rows):
            for column, coefficient in row.items():
                d[column] -= coefficient * y[i]

    report["dual"]["available"] = True
    if derived:
        report["notes"].append(
            "reduced costs were derived from the duals, so stationarity holds by "
            "construction and is not an independent check")

    # ---- stationarity: grad f - A'y - d = 0 -------------------------------
    residual = list(gradient)
    for i, row in enumerate(minimisation.rows):
        for column, coefficient in row.items():
            residual[column] -= coefficient * y[i]
    for j in range(model.n):
        residual[j] -= d[j]
    report["dual"]["max_stationarity_abs"] = max((abs(v) for v in residual), default=0.0)

    # ---- dual sign feasibility and complementary slackness ----------------
    #
    # For a minimisation with y_i = df/db_i:
    #   row active at its LOWER bound  -> y_i >= 0
    #   row active at its UPPER bound  -> y_i <= 0
    #   row strictly inside its bounds -> y_i == 0
    worst_sign, worst_complementarity = 0.0, 0.0

    for i, activity in enumerate(activities):
        lower, upper = model.row_lower[i], model.row_upper[i]
        slack_lower = (activity - lower) if lower > -INF else INF
        slack_upper = (upper - activity) if upper < INF else INF
        at_lower = slack_lower <= FEAS_ATOL
        at_upper = slack_upper <= FEAS_ATOL

        if not at_lower and not at_upper:
            worst_sign = max(worst_sign, abs(y[i]))       # must be zero
        elif at_lower and not at_upper:
            worst_sign = max(worst_sign, max(0.0, -y[i]))  # must be >= 0
        elif at_upper and not at_lower:
            worst_sign = max(worst_sign, max(0.0, y[i]))   # must be <= 0

        if y[i] > 0.0:
            worst_complementarity = max(worst_complementarity,
                                        abs(y[i]) * min(slack_lower, 1e30))
        elif y[i] < 0.0:
            worst_complementarity = max(worst_complementarity,
                                        abs(y[i]) * min(slack_upper, 1e30))

    for j in range(model.n):
        lower, upper = model.var_lower[j], model.var_upper[j]
        slack_lower = (x[j] - lower) if lower > -INF else INF
        slack_upper = (upper - x[j]) if upper < INF else INF
        at_lower = slack_lower <= FEAS_ATOL
        at_upper = slack_upper <= FEAS_ATOL

        if not at_lower and not at_upper:
            worst_sign = max(worst_sign, abs(d[j]))
        elif at_lower and not at_upper:
            worst_sign = max(worst_sign, max(0.0, -d[j]))
        elif at_upper and not at_lower:
            worst_sign = max(worst_sign, max(0.0, d[j]))

        if d[j] > 0.0:
            worst_complementarity = max(worst_complementarity,
                                        abs(d[j]) * min(slack_lower, 1e30))
        elif d[j] < 0.0:
            worst_complementarity = max(worst_complementarity,
                                        abs(d[j]) * min(slack_upper, 1e30))

    report["dual"]["max_sign_violation_abs"] = worst_sign
    report["dual"]["max_complementarity_abs"] = worst_complementarity

    # ---- dual objective and gap -------------------------------------------
    #
    #   g(y, d) = sum_i [ y_i >= 0 ? l_i y_i : u_i y_i ]
    #           + sum_j [ d_j >= 0 ? lb_j d_j : ub_j d_j ]
    #           + offset            ( - 0.5 x'Px for a quadratic objective )
    #
    # A multiplier pushing against an infinite bound makes the dual unbounded
    # below, which means these duals are not dual-feasible and no gap exists.
    # A multiplier is treated as zero below the feasibility tolerance. Testing
    # against exact 0.0 made this bail on afiro, where two reduced costs came
    # back at -5.6e-17 and -8.2e-18 -- floating-point noise on columns with no
    # upper bound. Read literally that is a dual-unbounded direction; read
    # numerically it is zero. Anything ABOVE the tolerance still bails, so a
    # genuine dual infeasibility is not silently absorbed.
    dual_objective = minimisation.offset
    finite = True
    negligible_against_infinite = 0

    for i in range(model.m):
        if abs(y[i]) <= FEAS_ATOL:
            continue
        bound = model.row_lower[i] if y[i] > 0.0 else model.row_upper[i]
        if not math.isfinite(bound):
            finite = False
            break
        dual_objective += bound * y[i]

    if finite:
        for j in range(model.n):
            if abs(d[j]) <= FEAS_ATOL:
                bound = model.var_lower[j] if d[j] > 0.0 else model.var_upper[j]
                if d[j] != 0.0 and not math.isfinite(bound):
                    negligible_against_infinite += 1
                continue
            bound = model.var_lower[j] if d[j] > 0.0 else model.var_upper[j]
            if not math.isfinite(bound):
                finite = False
                break
            dual_objective += bound * d[j]

    if negligible_against_infinite:
        report["notes"].append(
            f"{negligible_against_infinite} multiplier(s) pointed at an infinite "
            f"bound with magnitude below {FEAS_ATOL:g} and were treated as zero")

    if finite and minimisation.obj_quad:
        # For 0.5 x'Px + q'x the Lagrangian dual carries -0.5 x'Px at a
        # stationary x. In direct-coefficient terms that is -(sum q_ij x_i x_j).
        quadratic_value = sum(coefficient * x[i] * x[j]
                              for (i, j), coefficient in minimisation.obj_quad.items())
        dual_objective -= quadratic_value

    if not finite:
        report["dual"]["reason"] = (
            "a multiplier is active against an infinite bound, so the dual "
            "objective is unbounded and no gap can be formed")
        return

    primal_min = minimisation.objective(x)
    gap = abs(primal_min - dual_objective)
    report["dual"]["dual_objective"] = flip * dual_objective
    report["dual"]["gap_abs"] = gap
    report["dual"]["gap_normalised"] = \
        gap / (1.0 + abs(primal_min) + abs(dual_objective))
