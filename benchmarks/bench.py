#!/usr/bin/env python3
"""Benchmark orchestrator.

Pipeline per (instance, solver):

    1. parse cross-check   our reader vs the independent one, field by field
    2. isolated solve      through bench_runner: watchdog, tree kill, peak RSS
    3. independent check   the returned point against the ORIGINAL model
    4. record              one JSON row carrying all three, with nulls for
                           anything genuinely unavailable

The solver's own termination status and the checker's verdict are recorded as
separate fields and are never combined into a single "did it work". A solver
that says optimal and a checker that can only say feasible is not a failure of
either -- it is the normal outcome when no duals were produced -- and collapsing
the two would hide exactly the cases worth looking at.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, "lib"))

from mps_model import read_mps, MpsError, INF  # noqa: E402
import verify  # noqa: E402

DEFAULT_BINARY = os.path.join(ROOT, "build-release", "optimsolver")
DEFAULT_RUNNER = os.path.join(ROOT, "build-release", "bench_runner")
HIGHS_ADAPTER = os.path.join(HERE, "adapters", "highs_ref.py")


# ---------------------------------------------------------------------------
# Parse cross-check
# ---------------------------------------------------------------------------

def _num(value):
    """Model dumps write a non-finite bound as null; restore it as infinity."""
    return value


def compare_parses(dump, model, atol=1e-9):
    """Diff our reader's dump against the independent reader's model.

    Reports every disagreement rather than the first, because a sense flip and
    a dropped offset are different bugs and finding one should not hide the
    other.
    """
    problems = []

    if dump["sense"] != model.sense:
        problems.append(f"sense: ours={dump['sense']} independent={model.sense}")

    dump_offset = dump["offset"] if dump["offset"] is not None else 0.0
    if abs(dump_offset - model.offset) > atol:
        problems.append(
            f"objective offset: ours={dump_offset} independent={model.offset}")

    if len(dump["variables"]) != model.n:
        problems.append(
            f"variable count: ours={len(dump['variables'])} independent={model.n}")
    else:
        for j, entry in enumerate(dump["variables"]):
            if entry["name"] != model.var_names[j]:
                problems.append(
                    f"variable {j} name: ours={entry['name']} "
                    f"independent={model.var_names[j]}")
                continue
            ours_lower = -INF if entry["lower"] is None else entry["lower"]
            ours_upper = INF if entry["upper"] is None else entry["upper"]
            if not _close(ours_lower, model.var_lower[j], atol):
                problems.append(
                    f"variable {entry['name']} lower: ours={ours_lower} "
                    f"independent={model.var_lower[j]}")
            if not _close(ours_upper, model.var_upper[j], atol):
                problems.append(
                    f"variable {entry['name']} upper: ours={ours_upper} "
                    f"independent={model.var_upper[j]}")
            # Integer-vs-binary is a labelling difference when the bounds are
            # already [0,1]; a continuous-vs-integral difference is not.
            ours_integral = entry["type"] != "C"
            theirs_integral = model.var_type[j] != "C"
            if ours_integral != theirs_integral:
                problems.append(
                    f"variable {entry['name']} integrality: ours={entry['type']} "
                    f"independent={model.var_type[j]}")

    ours_linear = {int(k): v for k, v in dump["objective_linear"].items()}
    for column in set(ours_linear) | set(model.obj_linear):
        a = ours_linear.get(column, 0.0)
        b = model.obj_linear.get(column, 0.0)
        if not _close(a, b, atol):
            name = model.var_names[column] if column < model.n else str(column)
            problems.append(f"objective coefficient on {name}: ours={a} independent={b}")

    ours_quad = {tuple(int(p) for p in k.split(",")): v
                 for k, v in dump["objective_quadratic"].items()}
    for key in set(ours_quad) | set(model.obj_quad):
        a = ours_quad.get(key, 0.0)
        b = model.obj_quad.get(key, 0.0)
        if not _close(a, b, atol):
            problems.append(f"quadratic coefficient {key}: ours={a} independent={b}")

    if len(dump["constraints"]) != model.m:
        problems.append(
            f"row count: ours={len(dump['constraints'])} independent={model.m}")
    else:
        for i, entry in enumerate(dump["constraints"]):
            ours_lower = -INF if entry["lower"] is None else entry["lower"]
            ours_upper = INF if entry["upper"] is None else entry["upper"]
            if not _close(ours_lower, model.row_lower[i], atol):
                problems.append(
                    f"row {entry['name']} lower: ours={ours_lower} "
                    f"independent={model.row_lower[i]}")
            if not _close(ours_upper, model.row_upper[i], atol):
                problems.append(
                    f"row {entry['name']} upper: ours={ours_upper} "
                    f"independent={model.row_upper[i]}")
            ours_terms = {int(k): v for k, v in entry["terms"].items()}
            for column in set(ours_terms) | set(model.rows[i]):
                a = ours_terms.get(column, 0.0)
                b = model.rows[i].get(column, 0.0)
                if not _close(a, b, atol):
                    problems.append(
                        f"row {entry['name']} coefficient on column {column}: "
                        f"ours={a} independent={b}")

    return problems


def _close(a, b, atol):
    if a == b:
        return True
    if math.isinf(a) or math.isinf(b):
        return False
    return abs(a - b) <= atol + 1e-9 * max(abs(a), abs(b))


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def run_optimsolver(runner, binary, instance, engine, timeout, workdir, threads=1):
    solve_json = os.path.join(workdir, "solve.json")
    record_json = os.path.join(workdir, "record.json")
    command = [binary, "solve", instance, "--json", solve_json,
               "--threads", str(threads)]
    if engine:
        command += ["--solver", engine]
    if timeout > 0:
        # The solver gets a slightly tighter budget than the watchdog, so a
        # well-behaved solve reports its own limit status instead of being
        # killed and losing its output.
        command += ["--time-limit", str(max(0.1, timeout - 1.0))]
    return _invoke(runner, f"optimsolver:{engine or 'auto'}", instance,
                   command, solve_json, record_json, timeout)


def run_highs(runner, instance, method, timeout, workdir):
    solve_json = os.path.join(workdir, "solve.json")
    record_json = os.path.join(workdir, "record.json")
    command = [sys.executable, HIGHS_ADAPTER, instance,
               "--method", method, "--out", solve_json]
    if timeout > 0:
        command += ["--time-limit", str(max(0.1, timeout - 1.0))]
    return _invoke(runner, f"highs:{method}", instance,
                   command, solve_json, record_json, timeout)


def _invoke(runner, name, instance, command, solve_json, record_json, timeout):
    subprocess.run(
        [runner, "--solver", name, "--instance", instance,
         "--timeout", str(timeout), "--out", record_json,
         "--solve-json", solve_json, "--"] + command,
        capture_output=True, text=True)
    if not os.path.exists(record_json):
        return {"solver": name, "instance": instance,
                "process": {"run_status": "LAUNCH_FAILED"},
                "solve": None, "logs": {"stdout": "", "stderr": ""}}
    with open(record_json) as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# Scoring one row
# ---------------------------------------------------------------------------

def classify_engine_use(record, requested):
    """Did the engine the caller asked for actually run?

    A run where the dispatcher or the orchestrator routed elsewhere is a
    FALLBACK and cannot be reported as the requested engine's result. The two
    fields already exist for this reason -- `dispatched_engine` is a decision
    and `executed_engine` is written by the code path that did the work -- so
    the harness compares them rather than trusting either alone.

    `trivial` is called out separately: it means presolve answered the model
    outright and no engine ran at all, which is a legitimate outcome but is not
    a measurement of the requested engine.
    """
    solve = record.get("solve") or {}
    termination = solve.get("termination") or {}
    dispatched = termination.get("dispatched_engine")
    executed = termination.get("executed_engine")

    info = {
        "requested": requested,
        "dispatched": dispatched,
        "executed": executed,
        "fallback": False,
        "counts_for_requested": True,
        "reason": None,
    }
    if requested is None:
        return info
    if executed is None:
        info["fallback"] = True
        info["counts_for_requested"] = False
        info["reason"] = ("no engine executed; the solve was answered before "
                          "dispatch or refused")
        return info
    if executed == "trivial":
        info["fallback"] = True
        info["counts_for_requested"] = False
        info["reason"] = ("presolve reduced the model away and the bound-walk "
                          "answered it; the requested engine never ran")
        return info
    if executed != requested:
        info["fallback"] = True
        info["counts_for_requested"] = False
        info["reason"] = (f"requested {requested} but {executed} executed "
                          f"(dispatcher said {dispatched})")
    return info


# ---------------------------------------------------------------------------
# MILP-specific assessment
# ---------------------------------------------------------------------------

COMMON_GAP_TARGET = 1e-4


def common_gap(incumbent, bound):
    """|incumbent - bound| / max(1, |incumbent|, |bound|). None if either is missing."""
    if incumbent is None or bound is None:
        return None
    if not (math.isfinite(incumbent) and math.isfinite(bound)):
        return None
    return abs(incumbent - bound) / max(1.0, abs(incumbent), abs(bound))


def bound_ordering(incumbent, bound, sense):
    """Is the dual bound on the correct SIDE of the incumbent?

    For a minimisation the bound must not exceed the incumbent; for a
    maximisation it must not fall below it. A bound on the wrong side is not a
    small numerical issue -- it means the reported gap is meaningless, so it is
    checked rather than assumed.
    """
    if incumbent is None or bound is None:
        return {"checked": False, "valid": None, "violation": None}
    slack = 1e-6 * max(1.0, abs(incumbent), abs(bound))
    if sense == "min":
        violation = bound - incumbent
    else:
        violation = incumbent - bound
    return {"checked": True, "valid": violation <= slack,
            "violation": max(0.0, violation)}


def milp_assessment(run, model, reference, reference_kind):
    """Outcome category, gaps, and bound validity for one MILP run.

    Categories are kept distinct on purpose. "Solver said optimal" and "we
    verified the point" are different claims, and a timeout that produced an
    incumbent is a different outcome from one that produced nothing. None of
    them is allowed to collapse into a bare pass/fail.
    """
    solve = run.get("_solve") or {}
    termination = run.get("termination") or {}
    check = run.get("check") or {}
    process = run.get("process") or {}
    status = termination.get("status")
    verdict = check.get("verdict")

    incumbent = run.get("solver_objective")
    bound = solve.get("dual_bound")
    native_gap = solve.get("mip_gap")

    out = {
        "incumbent": incumbent,
        "dual_bound": bound,
        "native_gap": native_gap,
        "native_gap_source": "solver-reported" if native_gap is not None else None,
        "common_gap": common_gap(incumbent, bound),
        "common_gap_target": COMMON_GAP_TARGET,
        "common_gap_within_target": None,
        "bound_ordering": bound_ordering(incumbent, bound, model.sense),
        "bound_unavailable_reason": None,
        "reference_objective": reference,
        "reference_kind": reference_kind,
        "reference_relative_difference": None,
        "matches_reference": None,
        "time_to_first_feasible": solve.get("time_to_first_feasible"),
        "nodes": (solve.get("work") or {}).get("nodes"),
        "outcome": None,
        "notes": [],
    }
    if out["common_gap"] is not None:
        out["common_gap_within_target"] = out["common_gap"] <= COMMON_GAP_TARGET
    if bound is None:
        out["bound_unavailable_reason"] = (
            "this run supplied no finite global dual bound; no gap can be formed")

    # Quality against the reference. This is NOT an optimality proof: it is a
    # comparison with an external claim, and it is reported separately from any
    # gap the solver itself can justify.
    if reference is not None and incumbent is not None and math.isfinite(incumbent):
        difference = abs(incumbent - reference)
        out["reference_relative_difference"] = \
            difference / max(1.0, abs(incumbent), abs(reference))
        out["matches_reference"] = \
            out["reference_relative_difference"] <= COMMON_GAP_TARGET

    have_point = verdict in ("optimal_verified", "feasible")
    point_rejected = verdict in ("infeasible_point", "nonfinite", "malformed")
    termination_reason = termination.get("reason")
    timed_out = bool(process.get("timed_out")) or termination_reason == "time_limit"
    limited = status == "limit_reached"
    out["termination_reason"] = termination_reason
    out["discarded_nodes"] = solve.get("discarded_nodes")
    out["root_lp_bound"] = solve.get("root_lp_bound")
    out["root_cut_bound"] = solve.get("root_cut_bound")
    out["root_cut_passes"] = solve.get("root_cut_passes")

    if out["bound_ordering"]["valid"] is False:
        out["outcome"] = "incorrect_result"
        out["notes"].append("reported dual bound has invalid sense-aware ordering")
    elif point_rejected:
        out["outcome"] = "incorrect_result"
        out["notes"].append(
            f"independent check rejected the returned point ({verdict})")
    elif status in ("infeasible", "unbounded"):
        # Never accepted at face value: a timeout or a numerical breakdown must
        # not be reported as infeasibility.
        if reference is not None:
            out["outcome"] = "incorrect_result"
            out["notes"].append(
                f"solver claimed {status}, but the reference records a feasible "
                f"optimum of {reference}; the claim is contradicted")
        else:
            out["outcome"] = f"{status}_claim_uncorroborated"
            out["notes"].append(
                "no reference available to corroborate the claim; no certificate "
                "was validated")
    elif status in ("numerical_failure", "invalid_model", "unsupported"):
        out["outcome"] = "solver_failure"
        out["notes"].append(f"solver reported {status}; NOT infeasibility")
    elif not have_point:
        out["outcome"] = ("timeout_without_incumbent" if timed_out else
                          "limit_without_incumbent" if limited else "no_result")
    elif status == "optimal":
        if out["matches_reference"] is False:
            out["outcome"] = "incorrect_result"
            out["notes"].append(
                f"claimed optimal at {incumbent} but the reference proven optimum "
                f"is {reference}")
        else:
            out["outcome"] = "reported_optimal_within_tolerance"
            if verdict != "optimal_verified":
                out["notes"].append(
                    "feasibility and integrality independently validated; the "
                    "optimality claim is the solver's and is not independently "
                    "proven (no dual bound was supplied)")
    elif timed_out:
        out["outcome"] = "timeout_with_incumbent"
    elif limited:
        out["outcome"] = "limit_with_incumbent"
    else:
        out["outcome"] = "validated_feasible"

    return out


def evaluate(record, model, best_known=None):
    solve = record.get("solve")
    solver_status = None
    if solve:
        solver_status = solve.get("termination", {}).get("status")

    process = record.get("process", {})

    # A structured record is authoritative when one exists, even on a non-zero
    # exit. Our CLI exits 1 when it refuses a model it cannot solve, which is a
    # clean refusal and not a crash; treating the exit code as the primary
    # signal reported that refusal as NO_POINT while the reference solver's
    # identical refusal scored NOT_APPLICABLE. Two names for one outcome, decided
    # by an exit-code convention rather than by what happened.
    if solve is None:
        return {
            "verdict": verify.Verdict.NO_POINT,
            "solver_status": solver_status,
            "notes": [f"process ended as {process.get('run_status')} and wrote no "
                      f"structured result; nothing to check"],
            "tolerances": dict(verify.TOLERANCES),
        }

    if process.get("run_status") in ("TIMED_OUT", "LAUNCH_FAILED"):
        return {
            "verdict": verify.Verdict.NO_POINT,
            "solver_status": solver_status,
            "notes": [f"process ended as {process.get('run_status')}; any partial "
                      f"record is not trustworthy"],
            "tolerances": dict(verify.TOLERANCES),
        }

    # A claim of infeasible or unbounded is not a point, so there is nothing to
    # verify here. Independently PROVING infeasibility needs a Farkas
    # certificate, which is a separate check and is not attempted.
    if (solver_status in ("infeasible", "unbounded", "unsupported", "invalid_model") or
        (solver_status == "numerical_failure" and solve.get("primal") is None)):
        return {
            "verdict": verify.Verdict.NOT_APPLICABLE,
            "solver_status": solver_status,
            "notes": [f"solver reported '{solver_status}'; no point to check. "
                      f"This pipeline does not independently verify "
                      f"infeasibility or unboundedness certificates -- doing so "
                      f"needs a Farkas ray or an improving ray, which is a "
                      f"separate check."],
            "tolerances": dict(verify.TOLERANCES),
        }

    return verify.check_solution(
        model,
        solve.get("primal"),
        duals=solve.get("duals"),
        reduced_costs=solve.get("reduced_costs"),
        solver_status=solver_status,
        best_known=best_known,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("instances", nargs="+")
    parser.add_argument("--binary", default=DEFAULT_BINARY)
    parser.add_argument("--runner", default=DEFAULT_RUNNER)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--out", default=os.path.join(HERE, "results", "results.json"))
    parser.add_argument("--solvers", default="dual_simplex,pdlp,highs",
                        help="comma separated: auto, dual_simplex, pdlp, "
                             "branch_and_cut, qp, highs")
    parser.add_argument("--threads", type=int, default=1,
                        help="worker threads for our engines; 1 = serial")
    parser.add_argument("--best-known", default=None,
                        help="JSON file mapping instance basename to objective")
    args = parser.parse_args()

    best_known_kind = {}
    kind_path = os.path.join(os.path.dirname(args.best_known or ""), "manifest.json") \
        if args.best_known else None
    if kind_path and os.path.exists(kind_path):
        try:
            man = json.load(open(kind_path))
            best_known_kind = {i["name"] + ".mps": i.get("reference_kind")
                               for i in man.get("instances", [])}
        except Exception:
            best_known_kind = {}

    best_known_map = {}
    if args.best_known and os.path.exists(args.best_known):
        with open(args.best_known) as handle:
            best_known_map = json.load(handle)

    rows = []
    for instance in args.instances:
        base = os.path.basename(instance)
        best_known = best_known_map.get(base)

        entry = {
            "instance": instance,
            "instance_name": base,
            "best_known_objective": best_known,
            "parse_check": None,
            "runs": [],
        }

        # ---- stage 1: parse cross-check ----------------------------------
        try:
            model = read_mps(instance)
        except MpsError as error:
            entry["parse_check"] = {
                "status": "independent_reader_failed", "detail": str(error)}
            rows.append(entry)
            continue

        with tempfile.TemporaryDirectory() as workdir:
            dump_path = os.path.join(workdir, "dump.json")
            dumped = subprocess.run(
                [args.binary, "solve", instance, "--dump-model", dump_path],
                capture_output=True, text=True)
            if dumped.returncode != 0 or not os.path.exists(dump_path):
                entry["parse_check"] = {
                    "status": "our_reader_failed",
                    "detail": (dumped.stderr or dumped.stdout).strip()[:500]}
            else:
                with open(dump_path) as handle:
                    dump = json.load(handle)
                problems = compare_parses(dump, model)
                entry["parse_check"] = {
                    "status": "agree" if not problems else "disagree",
                    "problem_class": model.problem_class(),
                    "variables": model.n, "constraints": model.m,
                    "nonzeros": sum(len(r) for r in model.rows),
                    "differences": problems[:25],
                    "difference_count": len(problems),
                    "independent_reader_warnings": model.warnings[:10],
                }

        # ---- stages 2-4: solve, isolate, check ---------------------------
        for solver in [s.strip() for s in args.solvers.split(",") if s.strip()]:
            with tempfile.TemporaryDirectory() as workdir:
                started = time.perf_counter()
                if solver == "highs":
                    method = "milp" if model.is_integer_model() else "highs-ds"
                    record = run_highs(args.runner, instance, method,
                                       args.timeout, workdir)
                else:
                    engine = None if solver == "auto" else solver
                    record = run_optimsolver(args.runner, args.binary, instance,
                                             engine, args.timeout, workdir,
                                             threads=args.threads)
                wall = time.perf_counter() - started

            check = evaluate(record, model, best_known)
            solve = record.get("solve") or {}
            process = record.get("process", {})

            requested = None
            if solver not in ("auto", "highs"):
                requested = solver
            engine_use = classify_engine_use(record, requested)

            row = {
                "solver": record.get("solver", solver),
                "engine_use": engine_use,
                "process": process,
                "stage_seconds": solve.get("stage_seconds"),
                "orchestrator_wall_seconds": wall,
                "termination": solve.get("termination"),
                "settings": solve.get("settings"),
                "_solve": solve,
                "solver_objective": solve.get("objective"),
                "dual_bound": solve.get("dual_bound"),
                "mip_gap": solve.get("mip_gap"),
                "work": solve.get("work"),
                "self_reported": solve.get("self_reported"),
                "duals_unavailable_reason": solve.get("duals_unavailable_reason"),
                "check": check,
                "logs": {
                    "stdout_tail": (record.get("logs", {}).get("stdout") or "")[-2000:],
                    "stderr_tail": (record.get("logs", {}).get("stderr") or "")[-2000:],
                },
            }
            if model.is_integer_model():
                row["milp"] = milp_assessment(
                    row, model, best_known,
                    (best_known_kind or {}).get(base))
            row.pop("_solve", None)
            entry["runs"].append(row)

        rows.append(entry)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump({"tolerances": dict(verify.TOLERANCES), "results": rows},
                  handle, indent=2, allow_nan=False)

    _print_table(rows)
    print(f"\nwrote {args.out}")
    return 0


def _print_table(rows):
    header = (f"{'instance':<12}{'solver':<20}{'status':<13}"
              f"{'feas':<6}{'opt':<6}{'objective':>15}"
              f"{'e2e':>7}{'eng':>7}{'iters':>8}  notes")
    print(header)
    print("-" * len(header))
    for entry in rows:
        parse = entry.get("parse_check") or {}
        mark = {"agree": "parse ok", "disagree": "PARSE DISAGREE"}.get(
            parse.get("status"), parse.get("status") or "?")
        print(f"{entry['instance_name']:<12}[{mark}]")
        for run in entry["runs"]:
            termination = run.get("termination") or {}
            check = run.get("check") or {}
            verdict = check.get("verdict")
            objective = run.get("solver_objective")
            process = run.get("process") or {}
            stages = run.get("stage_seconds") or {}
            work = run.get("work") or {}

            # Primal feasibility and optimality are answered separately: a point
            # can be feasible with optimality simply not established, which is
            # not a failure of feasibility.
            feasible = {
                "optimal_verified": "yes", "feasible": "yes",
                "infeasible_point": "NO", "nonfinite": "NO",
            }.get(verdict, "-")
            optimal = {
                "optimal_verified": "yes", "feasible": "no",
            }.get(verdict, "-")

            notes = []
            use = run.get("engine_use") or {}
            if use.get("fallback"):
                notes.append(f"FALLBACK->{use.get('executed')}")
            if verdict == "not_applicable":
                notes.append(str(termination.get("status")))

            engine_seconds = stages.get("solve")
            print(
                f"{'':<12}"
                f"{run['solver']:<20}"
                f"{str(termination.get('status')):<13}"
                f"{feasible:<6}{optimal:<6}"
                f"{('null' if objective is None else f'{objective:.9g}'):>15}"
                f"{process.get('wall_seconds', 0.0):>7.2f}"
                f"{('-' if engine_seconds is None else f'{engine_seconds:.2f}'):>7}"
                f"{str(work.get('iterations') or '-'):>8}"
                f"  {' '.join(notes)}")


if __name__ == "__main__":
    sys.exit(main())
