#!/usr/bin/env python3
"""HiGHS reference adapter.

HiGHS is a REFERENCE, not a fallback. It lives entirely under benchmarks/,
is invoked as a separate process by bench_runner, and nothing under src/,
include/ or the engine directories refers to it. If this file is deleted the
solver builds and runs unchanged. benchmarks/test_pipeline.py asserts that
separation rather than leaving it to convention.

HiGHS is reached through scipy.optimize, which cannot read MPS. That is useful
rather than inconvenient: the reference is fed from mps_model.py, the reader
written independently of the C++ one, so a disagreement between our solver and
HiGHS also catches a parsing disagreement.

Emits the same JSON schema as `optimsolver --json`, so downstream stages treat
both solvers identically.

Usage:
    highs_ref.py <model.mps> --method highs-ds|highs-ipm|milp --out <file>
                 [--time-limit <seconds>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))

from mps_model import read_mps, MpsError, INF  # noqa: E402


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_scipy_form(model):
    """Translate to what linprog accepts, recording how to invert it.

    linprog takes A_ub x <= b_ub and A_eq x == b_eq only, so a ranged row has to
    become two inequalities and a >= row has to be negated. `row_map` records
    which scipy rows a model row became, and with what sign, so the marginals
    can be folded back into one shadow price per ORIGINAL row.
    """
    import numpy as np

    n = model.n
    A_ub, b_ub, A_eq, b_eq = [], [], [], []
    row_map = []  # per model row: ("eq", k) | ("ub", k_up, k_lo) with None for absent

    for i in range(model.m):
        dense = np.zeros(n)
        for column, coefficient in model.rows[i].items():
            dense[column] = coefficient
        lower, upper = model.row_lower[i], model.row_upper[i]

        if lower == upper and math.isfinite(lower):
            A_eq.append(dense)
            b_eq.append(lower)
            row_map.append(("eq", len(b_eq) - 1, None))
            continue

        index_upper = index_lower = None
        if math.isfinite(upper):
            A_ub.append(dense)
            b_ub.append(upper)
            index_upper = len(b_ub) - 1
        if math.isfinite(lower):
            # a'x >= l  becomes  -a'x <= -l
            A_ub.append(-dense)
            b_ub.append(-lower)
            index_lower = len(b_ub) - 1
        row_map.append(("ub", index_upper, index_lower))

    bounds = [
        (None if model.var_lower[j] == -INF else model.var_lower[j],
         None if model.var_upper[j] == INF else model.var_upper[j])
        for j in range(n)
    ]

    c = np.zeros(n)
    for column, coefficient in model.obj_linear.items():
        c[column] = coefficient

    return (
        c,
        np.array(A_ub) if A_ub else None,
        np.array(b_ub) if b_ub else None,
        np.array(A_eq) if A_eq else None,
        np.array(b_eq) if b_eq else None,
        bounds,
        row_map,
    )


def solve_lp(model, method, time_limit):
    """LP through HiGHS, with duals folded back to one per original row."""
    import numpy as np
    from scipy.optimize import linprog

    minimisation = model.to_minimization()
    c, A_ub, b_ub, A_eq, b_eq, bounds, row_map = build_scipy_form(minimisation)

    options = {}
    if time_limit and time_limit > 0:
        options["time_limit"] = float(time_limit)

    result = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method=method, options=options or None)

    status_map = {
        0: "optimal", 1: "limit_reached", 2: "infeasible",
        3: "unbounded", 4: "numerical_failure",
    }
    status = status_map.get(result.status, "numerical_failure")

    # Sense conversion: the model was negated to minimise, so the objective and
    # every multiplier negate back together.
    flip = -1.0 if model.sense == "max" else 1.0

    primal = list(result.x) if result.x is not None else None
    objective = None
    if primal is not None:
        objective = model.objective(primal)   # recomputed on the ORIGINAL model

    duals = reduced_costs = None
    reason = None
    if status == "optimal" and getattr(result, "ineqlin", None) is not None:
        m_ub = list(result.ineqlin.marginals) if result.ineqlin.marginals is not None else []
        m_eq = list(result.eqlin.marginals) if getattr(result, "eqlin", None) is not None \
            and result.eqlin.marginals is not None else []
        duals = []
        for kind, a, b in row_map:
            if kind == "eq":
                duals.append(flip * m_eq[a])
            else:
                # df/du from the <= part, and df/dl = -marginal from the
                # negated >= part. At most one binds, so the sum is the row's
                # single shadow price.
                value = 0.0
                if a is not None:
                    value += m_ub[a]
                if b is not None:
                    value -= m_ub[b]
                duals.append(flip * value)
        lower_marginals = list(result.lower.marginals) if getattr(result, "lower", None) is not None else [0.0] * model.n
        upper_marginals = list(result.upper.marginals) if getattr(result, "upper", None) is not None else [0.0] * model.n
        reduced_costs = [flip * (lower_marginals[j] + upper_marginals[j])
                         for j in range(model.n)]
    else:
        reason = "HiGHS reported no duals for this termination status"

    return {
        "status": status, "message": str(result.message),
        "primal": primal, "objective": objective,
        "duals": duals, "reduced_costs": reduced_costs,
        "duals_unavailable_reason": reason,
        "iterations": int(result.nit) if getattr(result, "nit", None) is not None else None,
        "nodes": None, "dual_bound": None, "mip_gap": None,
    }


def solve_milp(model, time_limit):
    """MILP through HiGHS. No duals exist for the integer problem."""
    import numpy as np
    from scipy.optimize import milp, LinearConstraint, Bounds

    minimisation = model.to_minimization()
    n = minimisation.n

    c = np.zeros(n)
    for column, coefficient in minimisation.obj_linear.items():
        c[column] = coefficient

    constraints = []
    if minimisation.m:
        A = np.zeros((minimisation.m, n))
        for i, row in enumerate(minimisation.rows):
            for column, coefficient in row.items():
                A[i, column] = coefficient
        constraints.append(LinearConstraint(
            A,
            [v if v > -INF else -np.inf for v in minimisation.row_lower],
            [v if v < INF else np.inf for v in minimisation.row_upper]))

    bounds = Bounds(
        [v if v > -INF else -np.inf for v in minimisation.var_lower],
        [v if v < INF else np.inf for v in minimisation.var_upper])
    integrality = [0 if t == "C" else 1 for t in minimisation.var_type]

    options = {}
    if time_limit and time_limit > 0:
        options["time_limit"] = float(time_limit)

    result = milp(c=c, constraints=constraints, bounds=bounds,
                  integrality=integrality, options=options or None)

    status_map = {
        0: "optimal", 1: "limit_reached", 2: "infeasible",
        3: "unbounded", 4: "numerical_failure",
    }
    status = status_map.get(result.status, "numerical_failure")
    primal = list(result.x) if result.x is not None else None
    objective = model.objective(primal) if primal is not None else None

    flip = -1.0 if model.sense == "max" else 1.0
    dual_bound = getattr(result, "mip_dual_bound", None)
    if dual_bound is not None:
        # mip_dual_bound is in the minimisation form scipy solved, and excludes
        # the model's objective constant.
        dual_bound = flip * (dual_bound + minimisation.offset)

    return {
        "status": status, "message": str(result.message),
        "primal": primal, "objective": objective,
        "duals": None, "reduced_costs": None,
        "duals_unavailable_reason":
            "branch-and-bound produces no duals for the integer problem",
        "iterations": None,
        "nodes": int(result.mip_node_count) if getattr(result, "mip_node_count", None) is not None else None,
        "dual_bound": dual_bound,
        "mip_gap": float(result.mip_gap) if getattr(result, "mip_gap", None) is not None else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("instance")
    parser.add_argument("--method", default="highs-ds",
                        choices=["highs-ds", "highs-ipm", "milp", "auto"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--time-limit", type=float, default=0.0)
    args = parser.parse_args()

    import scipy

    record = {
        "schema": "optimsolver.solve.v1",
        "instance": {"path": args.instance, "sha256": None,
                     "variables": None, "constraints": None},
        "solver": {"name": "highs", "commit": None,
                   "build_type": f"scipy {scipy.__version__}"},
        "settings": {"requested_engine": args.method,
                     "time_limit_seconds": args.time_limit or None,
                     "tolerance": None},
        "classification": None, "presolve": None,
        "termination": {"status": "invalid_model", "message": "",
                        "dispatched_engine": None, "executed_engine": None,
                        "engine_reason": None},
        "objective": None, "dual_bound": None, "mip_gap": None,
        "primal": None, "duals": None, "reduced_costs": None,
        "duals_unavailable_reason": None,
        "self_reported": {"max_bound_residual": None,
                          "max_constraint_residual": None,
                          "max_dual_residual": None,
                          "max_integrality_violation": None},
        "work": {"iterations": None, "nodes": None, "solve_seconds": None},
        "variable_names": None, "constraint_names": None,
    }

    started = time.perf_counter()
    try:
        record["instance"]["sha256"] = sha256_file(args.instance)
        model = read_mps(args.instance)
        record["instance"]["variables"] = model.n
        record["instance"]["constraints"] = model.m
        record["variable_names"] = model.var_names
        record["constraint_names"] = model.row_names
        record["classification"] = {"problem_class": model.problem_class()}

        method = args.method
        if method == "auto":
            method = "milp" if model.is_integer_model() else "highs-ds"

        if model.is_quadratic():
            # HiGHS does support QP, but scipy exposes no entry point for it.
            # Saying so is the only honest option: silently dropping the
            # quadratic terms would make the reference solve a different
            # problem and then be used to judge ours.
            record["termination"].update({
                "status": "unsupported",
                "message": "scipy exposes no HiGHS QP entry point; "
                           "quadratic objectives cannot be referenced",
                "executed_engine": None})
        elif method == "milp" or model.is_integer_model():
            outcome = solve_milp(model, args.time_limit)
            _apply(record, outcome, "milp")
        else:
            outcome = solve_lp(model, method, args.time_limit)
            _apply(record, outcome, method)

    except MpsError as error:
        record["termination"].update(
            {"status": "invalid_model", "message": f"parse error: {error}"})
    except Exception as error:  # noqa: BLE001 - reported, never swallowed
        record["termination"].update(
            {"status": "numerical_failure", "message": f"{type(error).__name__}: {error}"})

    record["work"]["solve_seconds"] = time.perf_counter() - started

    with open(args.out, "w") as handle:
        json.dump(record, handle, indent=2, allow_nan=False, default=_json_safe)
    return 0


def _apply(record, outcome, engine):
    message = outcome.get("message", "").lower()
    reason = ("time_limit" if "time limit" in message else
              "iteration_limit" if "iteration limit" in message else
              "node_limit" if "node limit" in message else "unspecified")
    record["termination"]["reason"] = reason
    record["termination"].update({
        "status": outcome["status"], "message": outcome["message"],
        "dispatched_engine": engine, "executed_engine": engine,
        "engine_reason": "reference solver, engine selected by --method"})
    record["objective"] = outcome["objective"]
    record["primal"] = outcome["primal"]
    record["duals"] = outcome["duals"]
    record["reduced_costs"] = outcome["reduced_costs"]
    record["duals_unavailable_reason"] = outcome["duals_unavailable_reason"]
    record["dual_bound"] = outcome["dual_bound"]
    record["mip_gap"] = outcome["mip_gap"]
    record["work"]["iterations"] = outcome["iterations"]
    record["work"]["nodes"] = outcome["nodes"]


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"not JSON serialisable: {type(value)}")


if __name__ == "__main__":
    sys.exit(main())
