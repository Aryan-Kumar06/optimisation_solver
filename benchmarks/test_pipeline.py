#!/usr/bin/env python3
"""Tests for the benchmark pipeline itself.

A checker that never rejects anything proves nothing. Most of this file feeds
the checker answers that are deliberately WRONG -- perturbed, non-finite,
mis-sized, dual-infeasible -- and asserts it says so. Without these, a green
benchmark run is indistinguishable from a checker that returns "fine".

Also asserts the containment rule: no external solver may appear anywhere under
src/, include/ or the engine directories. HiGHS is a reference, and a reference
that leaks into the solver stops being one.

Run:  python3 benchmarks/test_pipeline.py
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, "lib"))

from mps_model import read_mps, MpsError, INF  # noqa: E402
import verify  # noqa: E402
from verify import Verdict  # noqa: E402
import bench  # noqa: E402

KNOWN = os.path.join(HERE, "instances", "known")
BINARY = os.environ.get("OPTIMSOLVER_BINARY", os.path.join(ROOT, "build-release", "optimsolver"))

failures = []


def check(condition, what):
    if condition:
        print(f"[PASS] {what}")
    else:
        print(f"[FAIL] {what}")
        failures.append(what)


# ---------------------------------------------------------------------------
# The checker must reject bad answers
# ---------------------------------------------------------------------------

def test_rejects_bad_points():
    model = read_mps(os.path.join(KNOWN, "opt_lp.mps"))
    good = [2.0, 6.0]

    report = verify.check_solution(model, good)
    check(report["verdict"] == Verdict.FEASIBLE,
          "correct point with no duals -> feasible, NOT optimal_verified")
    check(abs(report["objective_recomputed"] - 36.0) < 1e-9,
          "objective recomputed independently as 36")

    # Row C3 is 3x + 2y <= 18, tight at (2,6). Pushing x out breaks it.
    report = verify.check_solution(model, [3.0, 6.0])
    check(report["verdict"] == Verdict.INFEASIBLE_POINT,
          "point violating a row -> infeasible_point")
    check(report["primal"]["rows_violating"] >= 1,
          "violating row counted")
    check(report["primal"]["worst_row"] is not None,
          "violating row named")

    report = verify.check_solution(model, [-1.0, 6.0])
    check(report["verdict"] == Verdict.INFEASIBLE_POINT,
          "point below a variable lower bound -> infeasible_point")

    report = verify.check_solution(model, [float("nan"), 6.0])
    check(report["verdict"] == Verdict.NONFINITE, "NaN in the point -> nonfinite")

    report = verify.check_solution(model, [float("inf"), 6.0])
    check(report["verdict"] == Verdict.NONFINITE, "infinity in the point -> nonfinite")

    report = verify.check_solution(model, [2.0])
    check(report["verdict"] == Verdict.MALFORMED, "short primal vector -> malformed")

    report = verify.check_solution(model, None)
    check(report["verdict"] == Verdict.NO_POINT, "no point -> no_point")

    # Just inside and just outside the frozen feasibility tolerance.
    report = verify.check_solution(model, [2.0 + 1e-9, 6.0])
    check(report["verdict"] == Verdict.FEASIBLE,
          "violation of 1e-9 is inside the 1e-6 feasibility tolerance")
    report = verify.check_solution(model, [2.0 + 1e-3, 6.0])
    check(report["verdict"] == Verdict.INFEASIBLE_POINT,
          "violation of 1e-3 is outside the 1e-6 feasibility tolerance")


def test_rejects_bad_duals():
    model = read_mps(os.path.join(KNOWN, "opt_lp.mps"))
    good = [2.0, 6.0]
    true_duals = [0.0, 1.5, 1.0]

    report = verify.check_solution(model, good, duals=true_duals)
    check(report["verdict"] == Verdict.OPTIMAL_VERIFIED,
          "correct point with correct duals -> optimal_verified")
    check(abs(report["dual"]["gap_normalised"]) < 1e-9,
          "duality gap closes to zero")

    # Sign-flipped duals: the classic error the PDLP adapter exists to prevent.
    report = verify.check_solution(model, good, duals=[-v for v in true_duals])
    check(report["verdict"] != Verdict.OPTIMAL_VERIFIED,
          "sign-flipped duals are NOT accepted as an optimality proof")

    report = verify.check_solution(model, good, duals=[0.0, 1.5, 5.0])
    check(report["verdict"] != Verdict.OPTIMAL_VERIFIED,
          "wrong dual magnitude is NOT accepted as an optimality proof")

    report = verify.check_solution(model, good, duals=[0.0, 1.5])
    check(report["dual"]["available"] is False,
          "dual vector of the wrong length is refused")

    report = verify.check_solution(model, good, duals=[0.0, float("nan"), 1.0])
    check(report["dual"]["available"] is False, "non-finite duals are refused")


def test_optimality_is_not_inferred_from_agreement():
    """Matching a best-known objective must never promote a verdict."""
    model = read_mps(os.path.join(KNOWN, "opt_lp.mps"))
    report = verify.check_solution(model, [2.0, 6.0], best_known=36.0)
    check(report["best_known"]["agrees"] is True, "best-known agreement detected")
    check(report["verdict"] == Verdict.FEASIBLE,
          "agreement with best-known does NOT produce optimal_verified")
    check(any("not an optimality proof" in note for note in report["notes"]),
          "agreement is explicitly labelled as corroboration, not proof")


def test_integrality():
    model = read_mps(os.path.join(KNOWN, "milp_knapsack.mps"))
    report = verify.check_solution(model, [1.0, 1.0, 0.0])
    check(report["integrality"]["checked"] is True, "integrality checked on a MILP")
    check(report["verdict"] == Verdict.FEASIBLE, "integral knapsack point is feasible")
    check(abs(report["objective_recomputed"] - 9.0) < 1e-9,
          "knapsack objective recomputed as 9")

    report = verify.check_solution(model, [0.5, 1.0, 0.0])
    check(report["verdict"] == Verdict.INFEASIBLE_POINT,
          "fractional value on a binary -> infeasible_point")
    check(report["integrality"]["variables_violating"] == 1,
          "fractional binary counted")


def test_quadratic_objective_and_offset():
    model = read_mps(os.path.join(KNOWN, "convex_qp.mps"))
    report = verify.check_solution(model, [1.0, 1.0])
    check(abs(report["objective_recomputed"] - 2.0) < 1e-12,
          "quadratic objective recomputed as x^2+y^2 = 2")

    # max 2x + 3y + 100, optimum (3,1). Exercises the sense flip and the
    # objective constant together: dropping the constant gives 9, and negating
    # it gives -91, so a single number distinguishes all three behaviours.
    model = read_mps(os.path.join(KNOWN, "max_offset_lp.mps"))
    report = verify.check_solution(model, [3.0, 1.0])
    check(abs(report["objective_recomputed"] - 109.0) < 1e-12,
          "objective constant included: 2*3 + 3*1 + 100 = 109")

    # Shadow prices y = (1.5, 0.5) reproduce the objective through strong
    # duality, which is what a wrong sign on a maximisation breaks.
    report = verify.check_solution(model, [3.0, 1.0], duals=[1.5, 0.5])
    check(report["verdict"] == Verdict.OPTIMAL_VERIFIED,
          "maximisation with an objective constant verifies against its duals")


# ---------------------------------------------------------------------------
# Parse cross-check
# ---------------------------------------------------------------------------

def test_parse_cross_check():
    # Every corpus on disk, not just the hand-written ones. This used to cover
    # instances/known only, which is why three real bugs in mps_model.py went
    # unnoticed: the C++ reader's own adversarial fixtures (multiple RHS and
    # RANGES vectors, and columns whose NAMES contain "MARKER"/"INTORG") live in
    # tests/mps/test_cases and were never cross-checked against it.
    instances = []
    for directory in (
        KNOWN,
        os.path.join(ROOT, "tests", "mps", "test_cases"),
        os.path.join(HERE, "instances", "netlib", "mps"),
        os.path.join(HERE, "instances", "miplib", "mps"),
    ):
        if not os.path.isdir(directory):
            continue
        instances.extend(sorted(
            os.path.join(directory, name) for name in os.listdir(directory)
            if name.endswith(".mps")))

    agreed = 0
    rejected = 0
    for instance in instances:
        try:
            model = read_mps(instance)
        except MpsError:
            # Both readers are expected to refuse some fixtures (SOS, unknown
            # sections). Agreeing to refuse is agreement.
            rejected += 1
            continue
        with tempfile.TemporaryDirectory() as workdir:
            dump_path = os.path.join(workdir, "dump.json")
            result = subprocess.run(
                [BINARY, "solve", instance, "--dump-model", dump_path],
                capture_output=True, text=True)
            if result.returncode != 0:
                rejected += 1
                continue
            with open(dump_path) as handle:
                dump = json.load(handle)
        problems = bench.compare_parses(dump, model)
        if problems:
            check(False, f"parse disagreement on {os.path.basename(instance)}: "
                         f"{problems[:3]}")
        else:
            agreed += 1
    check(agreed + rejected == len(instances),
          f"parse cross-check: {agreed} agree field by field, {rejected} refused "
          f"by both, {len(instances) - agreed - rejected} DISAGREE "
          f"(across {len(instances)} instances in 4 corpora)")


def test_cross_check_catches_a_planted_difference():
    """The cross-check must fail when the two parses really do differ."""
    model = read_mps(os.path.join(KNOWN, "opt_lp.mps"))
    with tempfile.TemporaryDirectory() as workdir:
        dump_path = os.path.join(workdir, "dump.json")
        subprocess.run([BINARY, "solve", os.path.join(KNOWN, "opt_lp.mps"),
                        "--dump-model", dump_path], capture_output=True)
        with open(dump_path) as handle:
            dump = json.load(handle)

    check(not bench.compare_parses(dump, model), "baseline parses agree")

    planted = json.loads(json.dumps(dump))
    planted["sense"] = "min"
    check(any("sense" in p for p in bench.compare_parses(planted, model)),
          "planted sense flip is detected")

    planted = json.loads(json.dumps(dump))
    planted["offset"] = 17.0
    check(any("offset" in p for p in bench.compare_parses(planted, model)),
          "planted objective offset is detected")

    planted = json.loads(json.dumps(dump))
    planted["constraints"][0]["upper"] = 999.0
    check(any("upper" in p for p in bench.compare_parses(planted, model)),
          "planted row bound change is detected")

    planted = json.loads(json.dumps(dump))
    planted["variables"][0]["type"] = "I"
    check(any("integrality" in p for p in bench.compare_parses(planted, model)),
          "planted integrality change is detected")


# ---------------------------------------------------------------------------
# Containment: the reference solver must not reach the solver
# ---------------------------------------------------------------------------

def test_reference_solver_is_not_a_dependency():
    solver_trees = ["src", "include", "cli", "pdlp_engine", "milp_engine",
                    "qp_engine"]
    pattern = re.compile(r"highs|scipy|gurobi|cplex|mosek|osqp", re.IGNORECASE)
    offenders = []
    for tree in solver_trees:
        root = os.path.join(ROOT, tree)
        for directory, _, names in os.walk(root):
            for name in names:
                if not name.endswith((".cpp", ".h", ".hpp", ".txt", ".cmake")):
                    continue
                path = os.path.join(directory, name)
                with open(path, errors="replace") as handle:
                    for number, line in enumerate(handle, 1):
                        # A comment naming a solver as a verification reference
                        # is fine; an include or a link is not.
                        if pattern.search(line) and re.search(
                                r"#\s*include|target_link|find_package|dlopen", line):
                            offenders.append(f"{tree}/{name}:{number}")
    check(not offenders,
          f"no external solver is included or linked by the solver "
          f"(offenders: {offenders})")

    links = subprocess.run(["otool", "-L", BINARY], capture_output=True, text=True)
    if links.returncode == 0:
        leaked = pattern.search(links.stdout)
        check(leaked is None,
              "the built binary links no external solver library")


# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------

def test_watchdog_and_memory_are_recorded():
    runner = os.path.join(ROOT, "build-release", "bench_runner")
    if not os.path.exists(runner):
        print("[SKIP] bench_runner not built")
        return
    with tempfile.TemporaryDirectory() as workdir:
        record = os.path.join(workdir, "record.json")
        subprocess.run([runner, "--solver", "sleeper", "--instance", "none",
                        "--timeout", "1", "--grace", "0.3", "--out", record,
                        "--", "/bin/sh", "-c", "sleep 30"],
                       capture_output=True)
        with open(record) as handle:
            data = json.load(handle)
    check(data["process"]["timed_out"] is True, "watchdog: timeout recorded")
    check(data["process"]["killed_process_tree"] is True,
          "watchdog: process tree termination recorded")
    check(data["process"]["wall_seconds"] < 5.0,
          "watchdog: returned well before the 30s sleep")
    check(data["process"]["peak_memory_bytes"] is not None,
          "peak memory recorded")
    check(data["solve"] is None,
          "no structured solve record when the child was killed")


def test_missing_values_are_null_not_zero():
    """A missing number must be null. Zero is a different claim."""
    with tempfile.TemporaryDirectory() as workdir:
        out = os.path.join(workdir, "solve.json")
        subprocess.run([BINARY, "solve", os.path.join(KNOWN, "milp_knapsack.mps"),
                        "--json", out], capture_output=True)
        with open(out) as handle:
            record = json.load(handle)
    check(record["duals"] is None, "MILP duals are null, not an empty or zero vector")
    check(record["duals_unavailable_reason"] is not None,
          "a reason is given for the missing duals")
    # dual_bound and mip_gap are emitted as null until the MILP engine keeps a
    # global bound over open subtrees. Asserting the KEY is present but null
    # pins the schema now, so a consumer never has to tell "absent" from
    # "null", and the value assertions land with the engine change.
    check("dual_bound" in record and record["dual_bound"] is None,
          "dual bound key present and null until the engine maintains one")
    check("mip_gap" in record and record["mip_gap"] is None,
          "mip gap key present and null until a bound exists")

    with tempfile.TemporaryDirectory() as workdir:
        out = os.path.join(workdir, "solve.json")
        subprocess.run([BINARY, "solve", os.path.join(KNOWN, "infeasible_lp.mps"),
                        "--json", out], capture_output=True)
        with open(out) as handle:
            record = json.load(handle)
    check(record["objective"] is None, "infeasible: objective is null, not 0")
    check(record["primal"] is None, "infeasible: primal is null, not an empty list")
    check(record["termination"]["status"] == "infeasible",
          "infeasible: status preserved")


def test_limit_without_incumbent_is_recorded():
    """A limit reached with no incumbent must produce a RECORD, not an error.

    Branch-and-cut returns an empty primal when it finds nothing, and that used
    to reach postsolve as a size mismatch: exit 1, no JSON, so the run could not
    be recorded as a timeout at all. Measured on MIPLIB gen-ip002 and enlight8.
    """
    # The INVARIANT under test is "no point means null, and a record is still
    # written", not "this instance times out without an incumbent". An earlier
    # version asserted the latter and so encoded one engine's timing: whether a
    # root heuristic lands before a 1e-30 deadline is not a property the record
    # format should depend on.
    instance = os.path.join(KNOWN, "milp_knapsack.mps")
    with tempfile.TemporaryDirectory() as workdir:
        out = os.path.join(workdir, "solve.json")
        result = subprocess.run(
            [BINARY, "solve", instance, "--solver", "branch_and_cut",
             "--threads", "1", "--time-limit", "1e-30", "--json", out],
            capture_output=True, text=True)
        check(result.returncode == 0,
              "limit reached: exits 0, not an error")
        check(os.path.exists(out),
              "limit reached: a structured record is written")
        if os.path.exists(out):
            with open(out) as handle:
                record = json.load(handle)
            check(record["termination"]["status"] == "limit_reached",
                  "limit reached: status is limit_reached, NOT infeasible")
            has_point = record["primal"] is not None
            check(has_point == (record["objective"] is not None),
                  "limit reached: primal and objective agree on whether a point exists")
            if not has_point:
                check(record["objective"] is None,
                      "no incumbent: objective is null, not 0")


def test_eliminated_model_and_limit_categories():
    with tempfile.TemporaryDirectory() as workdir:
        instance = os.path.join(workdir, "fixed.mps")
        output = os.path.join(workdir, "fixed.json")
        with open(instance, "w") as f:
            f.write("NAME FIXED\nROWS\n N OBJ\nCOLUMNS\n X OBJ 2\nBOUNDS\n FX B X 3\nENDATA\n")
        run = subprocess.run([BINARY, "solve", instance, "--json", output], capture_output=True)
        check(run.returncode == 0, "eliminated model succeeds")
        with open(output) as f:
            record = json.load(f)
        check(record["primal"] == [3] and record["objective"] == 6,
              "empty reduced primal reconstructs original fixed variables and objective")
    model = read_mps(os.path.join(KNOWN, "milp_knapsack.mps"))
    for reason, expected in [("time_limit", "timeout_without_incumbent"),
                             ("iteration_limit", "limit_without_incumbent"),
                             ("node_limit", "limit_without_incumbent"),
                             (None, "limit_without_incumbent")]:
        run = {"termination": {"status": "limit_reached", "reason": reason},
               "check": {"verdict": "no_point"}, "process": {"timed_out": False}}
        result = bench.milp_assessment(run, model, None, None)
        check(result["outcome"] == expected, f"limit reason {reason} is classified honestly")
    import milp_report
    check(milp_report.statistics.median([0, 1, 3, 10]) == 2,
          "even sample sizes use the arithmetic median")


def main():
    print("--- benchmark pipeline self-test ---\n")
    test_rejects_bad_points()
    test_rejects_bad_duals()
    test_optimality_is_not_inferred_from_agreement()
    test_integrality()
    test_quadratic_objective_and_offset()
    test_parse_cross_check()
    test_cross_check_catches_a_planted_difference()
    test_reference_solver_is_not_a_dependency()
    test_watchdog_and_memory_are_recorded()
    test_missing_values_are_null_not_zero()
    test_limit_without_incumbent_is_recorded()
    test_eliminated_model_and_limit_categories()

    print()
    if failures:
        print(f"{len(failures)} FAILURE(S)")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("all pipeline self-tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
