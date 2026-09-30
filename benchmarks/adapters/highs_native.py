#!/usr/bin/env python3
"""HiGHS reference adapter using the NATIVE `highs` executable.

This replaces the SciPy route for benchmarking, because SciPy made the memory
comparison meaningless. Measured: on instances SciPy refused outright (a
quadratic objective it has no entry point for), the process still peaked at
29.5-30.9 MB having solved nothing -- that is Python, NumPy and SciPy loading.
On afiro it peaked at 67.9 MB against our own 1.6 MB, and most of that gap was
the interpreter, not the solver. Reporting "23x less memory than HiGHS" off
those numbers would have been wrong.

The native binary is a fair peer: one process, no interpreter, and it reads MPS
itself rather than being fed dense arrays built by a Python adapter.

It is still a REFERENCE, not a dependency. It is located on PATH at run time and
invoked as a separate process; nothing in the solver links, includes or requires
it, and benchmarks/test_pipeline.py asserts that separation.

Emits the same JSON schema as `optimsolver --json`, so downstream stages treat
both solvers identically.

Usage:
    highs_native.py <model.mps> --out <file> [--time-limit <seconds>]
                    [--solver simplex|ipm] [--threads <n>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))

from mps_model import read_mps, MpsError  # noqa: E402

# HiGHS model statuses mapped onto the shared vocabulary. Anything not listed is
# reported as a numerical failure rather than guessed at.
STATUS_MAP = {
    "optimal": "optimal",
    "infeasible": "infeasible",
    "primal infeasible": "infeasible",
    "unbounded": "unbounded",
    "primal unbounded": "unbounded",
    "unbounded or infeasible": "unbounded",
    "time limit reached": "limit_reached",
    "iteration limit reached": "limit_reached",
    "interrupt": "limit_reached",
    "model empty": "optimal",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_solution(text, model):
    """Read HiGHS's solution file.

    Values are keyed by NAME and mapped back through the model's own ordering,
    never by position: HiGHS writes its own column order, and assuming it
    matches ours would silently permute the solution.
    """
    sections, current = {}, None
    primal_feasible = dual_feasible = None
    objective = None

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            heading = line[1:].strip().lower()
            if heading.startswith("primal solution"):
                current = ("primal", None)
            elif heading.startswith("dual solution"):
                current = ("dual", None)
            elif heading.startswith("basis"):
                current = None
            elif heading.startswith("columns") and current:
                current = (current[0], "columns")
            elif heading.startswith("rows") and current:
                current = (current[0], "rows")
            continue
        if line.startswith("Objective"):
            try:
                objective = float(line.split()[-1])
            except ValueError:
                pass
            continue
        if line in ("Feasible", "Infeasible", "Unknown"):
            if current and current[0] == "primal":
                primal_feasible = (line == "Feasible")
            elif current and current[0] == "dual":
                dual_feasible = (line == "Feasible")
            continue
        if current is None or current[1] is None:
            continue
        parts = line.rsplit(None, 1)
        if len(parts) != 2:
            continue
        name, value = parts
        try:
            sections.setdefault(current, {})[name] = float(value)
        except ValueError:
            continue

    def vector(key, names):
        values = sections.get(key)
        if not values:
            return None
        if not all(n in values for n in names):
            return None
        return [values[n] for n in names]

    return {
        "primal": vector(("primal", "columns"), model.var_names),
        "row_values": vector(("primal", "rows"), model.row_names),
        "reduced_costs": vector(("dual", "columns"), model.var_names),
        "duals": vector(("dual", "rows"), model.row_names),
        "objective": objective,
        "primal_feasible": primal_feasible,
        "dual_feasible": dual_feasible,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("instance")
    parser.add_argument("--out", required=True)
    parser.add_argument("--time-limit", type=float, default=0.0)
    parser.add_argument("--solver", default="simplex",
                        choices=["simplex", "ipm", "choose"])
    parser.add_argument("--threads", type=int, default=1)
    # Post-hoc mode: convert an ALREADY-COMPLETED run into the JSON record,
    # without launching the solver.
    #
    # This exists so the measured process can be the `highs` binary and nothing
    # else. Running it from inside this script put a Python interpreter in the
    # measurement, and bare python3 startup alone is 11.0 MB against HiGHS's own
    # 3.7 MB on afiro -- so the reported memory was mostly the wrapper. Peak
    # memory is only comparable when each process contains one solver.
    parser.add_argument("--parse-only", action="store_true")
    parser.add_argument("--solution")
    parser.add_argument("--log")
    args = parser.parse_args()

    binary = shutil.which("highs")
    record = {
        "schema": "optimsolver.solve.v1",
        "instance": {"path": args.instance, "sha256": None,
                     "variables": None, "constraints": None},
        "solver": {"name": "highs", "commit": None, "build_type": None},
        "settings": {"requested_engine": args.solver,
                     "time_limit_seconds": args.time_limit or None,
                     "tolerance": None, "thread_count": args.threads},
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
        "stage_seconds": {"parse": None, "presolve": None,
                          "solve": None, "postsolve": None},
        "work": {"iterations": None, "nodes": None, "solve_seconds": None},
        "variable_names": None, "constraint_names": None,
    }

    started = time.perf_counter()
    try:
        if binary is None and not args.parse_only:
            raise RuntimeError("the `highs` executable is not on PATH")

        if binary is not None:
            version = subprocess.run([binary, "--version"], capture_output=True, text=True)
            record["solver"]["build_type"] = version.stdout.strip().splitlines()[0][:120] \
                if version.stdout else "native"

        record["instance"]["sha256"] = sha256_file(args.instance)
        model = read_mps(args.instance)
        record["instance"]["variables"] = model.n
        record["instance"]["constraints"] = model.m
        record["variable_names"] = model.var_names
        record["constraint_names"] = model.row_names
        record["classification"] = {"problem_class": model.problem_class()}

        with tempfile.TemporaryDirectory() as workdir:
            if args.parse_only:
                solution = args.solution or ""
                log = open(args.log).read() if args.log and os.path.exists(args.log) else ""
            else:
                solution = os.path.join(workdir, "solution.txt")
                command = [binary, args.instance, "--solution_file", solution,
                           "--threads", str(args.threads), "--parallel", "off"]
                if args.solver != "choose":
                    command += ["--solver", args.solver]
                if args.time_limit and args.time_limit > 0:
                    command += ["--time_limit", str(args.time_limit)]
                result = subprocess.run(command, capture_output=True, text=True)
                log = result.stdout or ""

            status = None
            match = re.search(r"Model status\s*:\s*(.+)", log)
            if match:
                status = match.group(1).strip()
            record["termination"]["message"] = (status or "").strip()
            record["termination"]["status"] = STATUS_MAP.get(
                (status or "").lower(), "numerical_failure")
            record["termination"]["dispatched_engine"] = args.solver
            record["termination"]["executed_engine"] = args.solver
            record["termination"]["engine_reason"] = \
                "reference solver, engine selected by --solver"

            iterations = re.search(r"(?:Simplex|IPM)\s+iterations:\s*(\d+)", log)
            if iterations:
                record["work"]["iterations"] = int(iterations.group(1))
            runtime = re.search(r"HiGHS run time\s*:\s*([0-9.eE+-]+)", log)
            if runtime:
                record["stage_seconds"]["solve"] = float(runtime.group(1))

            if os.path.exists(solution):
                parsed = parse_solution(open(solution).read(), model)
                record["primal"] = parsed["primal"]
                if parsed["primal"] is not None:
                    # Recomputed on the ORIGINAL model rather than trusting the
                    # solver's own figure, exactly as the other adapter does.
                    record["objective"] = model.objective(parsed["primal"])
                if model.is_integer_model():
                    record["duals_unavailable_reason"] = (
                        "branch-and-bound produces no duals for the integer problem")
                else:
                    # HiGHS writes row duals as the Lagrange multiplier for a
                    # MINIMISATION. A maximisation is negated once to reach the
                    # model's own sense, matching the shadow-price convention
                    # the checker and postsolve both use.
                    flip = -1.0 if model.sense == "max" else 1.0
                    if parsed["duals"] is not None:
                        record["duals"] = [flip * v for v in parsed["duals"]]
                    if parsed["reduced_costs"] is not None:
                        record["reduced_costs"] = [flip * v for v in parsed["reduced_costs"]]
                    if record["duals"] is None:
                        record["duals_unavailable_reason"] = \
                            "HiGHS wrote no row duals for this termination status"
            elif record["termination"]["status"] not in ("infeasible", "unbounded"):
                record["termination"]["message"] += \
                    "; no solution file was written"

    except (MpsError, RuntimeError, OSError) as error:
        record["termination"].update(
            {"status": "invalid_model", "message": f"{type(error).__name__}: {error}"})
    except Exception as error:  # noqa: BLE001 - reported, never swallowed
        record["termination"].update(
            {"status": "numerical_failure", "message": f"{type(error).__name__}: {error}"})

    record["work"]["solve_seconds"] = time.perf_counter() - started
    with open(args.out, "w") as handle:
        json.dump(record, handle, indent=2, allow_nan=False, default=_json_safe)
    return 0


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"not JSON serialisable: {type(value)}")


if __name__ == "__main__":
    sys.exit(main())
