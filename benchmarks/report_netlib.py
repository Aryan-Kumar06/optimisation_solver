#!/usr/bin/env python3
"""Turn a Netlib benchmark run into a CSV and a readable summary.

Committed as a script rather than pasted output so the published numbers can be
regenerated and checked against the code that produced them:

    python3 benchmarks/bench.py benchmarks/instances/netlib/mps/*.mps \\
        --solvers dual_simplex,pdlp,highs --timeout 30 --threads 1 \\
        --best-known benchmarks/instances/netlib/best_known.json \\
        --out benchmarks/results/netlib_lp.json
    python3 benchmarks/report_netlib.py benchmarks/results/netlib_lp.json

Two columns are kept apart on purpose, and the distinction is the point of the
whole table:

    primal_feasible      the returned point satisfies the ORIGINAL model
    optimality_verified  a duality gap / KKT check closed independently

A solver can be feasible without optimality being verified -- that is the normal
outcome when it produces no duals, and it is NOT a failure of feasibility.
Neither column is the solver's own status, which is reported separately in
`solver_status`; agreement with the published Netlib objective is recorded as
corroboration and never treated as proof.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

COLUMNS = [
    "instance", "rows", "columns", "nonzeros", "reference_objective",
    "solver", "requested_engine", "executed_engine", "solver_status",
    "checker_verdict", "primal_feasible", "optimality_verified",
    "objective", "abs_difference_vs_reference", "relative_difference_vs_reference",
    "max_row_violation_abs", "max_bound_violation_abs", "duality_gap_normalised",
    "iterations", "wall_seconds", "engine_seconds", "peak_memory_mb",
]

FEASIBLE = {"optimal_verified", "feasible"}

# Above this relative difference from the published Netlib objective, a feasible
# point is reported as materially suboptimal rather than merely unproven.
OBJECTIVE_MATCH_TOLERANCE = 1e-4


def shell(command):
    try:
        return subprocess.run(command, shell=True, capture_output=True,
                              text=True, cwd=ROOT).stdout.strip()
    except OSError:
        return ""


def rows_from(results):
    for entry in results:
        parse = entry.get("parse_check") or {}
        reference = entry.get("best_known_objective")
        for run in entry.get("runs", []):
            check = run.get("check") or {}
            primal = check.get("primal") or {}
            dual = check.get("dual") or {}
            use = run.get("engine_use") or {}
            process = run.get("process") or {}
            stages = run.get("stage_seconds") or {}
            work = run.get("work") or {}
            termination = run.get("termination") or {}
            verdict = check.get("verdict")
            objective = run.get("solver_objective")

            absolute = relative = None
            if objective is not None and reference is not None:
                absolute = abs(objective - reference)
                relative = absolute / max(1.0, abs(objective), abs(reference))

            memory = process.get("peak_memory_bytes")
            yield {
                "instance": entry["instance_name"].replace(".mps", ""),
                "rows": parse.get("constraints"),
                "columns": parse.get("variables"),
                "nonzeros": parse.get("nonzeros"),
                "reference_objective": reference,
                "solver": run.get("solver"),
                "requested_engine": use.get("requested"),
                "executed_engine": use.get("executed") or termination.get("executed_engine"),
                "solver_status": termination.get("status"),
                "checker_verdict": verdict,
                # Feasibility and optimality are answered separately. A blank is
                # "not applicable" (nothing to check), never "no".
                "primal_feasible": ("yes" if verdict in FEASIBLE
                                    else ("no" if verdict in
                                          ("infeasible_point", "nonfinite", "malformed")
                                          else "")),
                "optimality_verified": ("yes" if verdict == "optimal_verified"
                                        else ("no" if verdict == "feasible" else "")),
                "objective": objective,
                "abs_difference_vs_reference": absolute,
                "relative_difference_vs_reference": relative,
                "max_row_violation_abs": primal.get("max_row_violation_abs"),
                "max_bound_violation_abs": primal.get("max_bound_violation_abs"),
                "duality_gap_normalised": dual.get("gap_normalised"),
                "iterations": work.get("iterations"),
                "wall_seconds": process.get("wall_seconds"),
                "engine_seconds": stages.get("solve"),
                "peak_memory_mb": (round(memory / 1048576.0, 1)
                                   if memory is not None else None),
            }


def write_csv(path, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path, rows, tolerances, source):
    engines = {}
    for row in rows:
        stats = engines.setdefault(row["solver"], {"n": 0, "verified": 0, "feasible": 0,
                                                   "wrong": 0, "memory": 0.0})
        stats["n"] += 1
        stats["verified"] += row["optimality_verified"] == "yes"
        stats["feasible"] += row["primal_feasible"] == "yes"
        stats["wrong"] += row["primal_feasible"] == "no"
        relative = row["relative_difference_vs_reference"]
        stats["matched"] = stats.get("matched", 0) + int(
            relative is not None and relative <= OBJECTIVE_MATCH_TOLERANCE)
        stats["memory"] = max(stats["memory"], row["peak_memory_mb"] or 0.0)

    instances = sorted({row["instance"] for row in rows})
    solvers = sorted(engines)

    with open(path, "w") as out:
        out.write("# Netlib LP benchmark results\n\n")
        out.write(f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} "
                  f"from `{os.path.basename(source)}`.\n\n")

        out.write("## Provenance\n\n")
        out.write(f"- Commit: `{shell('git rev-parse HEAD')}`\n")
        out.write(f"- Build: Release (`-O3 -DNDEBUG`), single-threaded (`--threads 1`)\n")
        out.write(f"- Compiler: {shell('c++ --version | head -1')}\n")
        out.write(f"- Platform: {platform.system()} {platform.release()} {platform.machine()}\n")
        out.write("- Instances: Netlib LP, decoded from the source archive with Netlib's own\n"
                  "  `emps` decompressor. Netlib does not distribute plain MPS. Per-file SHA-256\n"
                  "  and dimensions are in `instances/netlib/manifest.json`.\n")
        out.write("- Reference objectives: the table in <https://www.netlib.org/lp/data/readme>.\n"
                  "  These are EXTERNAL CLAIMS used for corroboration only, never as proof of\n"
                  "  optimality.\n\n")

        out.write("## How to read this\n\n")
        out.write("`primal_feasible` and `optimality_verified` answer different questions and\n"
                  "are never merged:\n\n")
        out.write("| column | meaning |\n|---|---|\n")
        out.write("| `primal_feasible` | the returned point satisfies the ORIGINAL model, "
                  "checked before presolve and before any scaling |\n")
        out.write("| `optimality_verified` | a duality gap / KKT check closed independently |\n\n")
        out.write("A solver can be feasible with optimality simply not established — that is the\n"
                  "normal outcome when it returns no duals, and it is not a feasibility failure.\n"
                  "Both are recomputed by `benchmarks/lib/verify.py` against a reader written\n"
                  "independently of the solver's own; neither is the solver's self-report, which\n"
                  "is kept in `solver_status`.\n\n")

        out.write("## Frozen tolerances\n\n| quantity | value |\n|---|---|\n")
        for key, value in tolerances.items():
            if key != "frozen":
                out.write(f"| {key.replace('_', ' ')} | `{value}` |\n")
        out.write("\n")

        out.write("## Independently verified optimal\n\n")
        out.write("| solver | verified optimal | matches published objective | "
                  "feasible point | infeasible point | peak MB |\n")
        out.write("|---|---|---|---|---|---|\n")
        for name in solvers:
            s = engines[name]
            out.write(f"| `{name}` | **{s['verified']}/{s['n']}** "
                      f"| {s.get('matched', 0)}/{s['n']} | {s['feasible']}/{s['n']} "
                      f"| {s['wrong']} | {s['memory']:.1f} |\n")
        out.write("\n")

        out.write("## Per instance\n\n")
        out.write("| instance | rows | cols | reference | " +
                  " | ".join(f"`{s}`" for s in solvers) + " |\n")
        out.write("|---" * (4 + len(solvers)) + "|\n")
        by_key = {(r["instance"], r["solver"]): r for r in rows}
        for instance in instances:
            first = next(r for r in rows if r["instance"] == instance)
            cells = []
            for name in solvers:
                row = by_key.get((instance, name))
                if row is None or row["objective"] is None:
                    cells.append("—")
                    continue
                # Three states, not two. "Feasible but optimality unproven" and
                # "feasible and nowhere near the published optimum" are very
                # different results, and collapsing them would let a point that
                # is 100% off the reference read as a minor caveat.
                relative = row["relative_difference_vs_reference"]
                if row["optimality_verified"] == "yes":
                    mark = "✅"
                elif relative is not None and relative > OBJECTIVE_MATCH_TOLERANCE:
                    mark = f"❌ {relative * 100:.1f}% off"
                else:
                    mark = "⚠️"
                cells.append(f"{row['objective']:.9g} {mark}".strip())
            reference = first["reference_objective"]
            out.write(f"| {instance} | {first['rows']} | {first['columns']} | "
                      f"{reference:.9g} | " + " | ".join(cells) + " |\n")
        out.write("\n✅ optimality independently verified · "
                  "⚠️ feasible and matches the published objective, optimality not proven · "
                  "❌ feasible but materially short of the published objective · "
                  "— no point returned\n\n")

        out.write("## Reproducing\n\n```sh\n")
        out.write("cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release\n")
        out.write("cmake --build build-release -j8\n")
        out.write("python3 benchmarks/fetch_netlib.py --set smoke\n")
        out.write("python3 benchmarks/bench.py benchmarks/instances/netlib/mps/*.mps \\\n")
        out.write("    --solvers dual_simplex,pdlp,highs --timeout 30 --threads 1 \\\n")
        out.write("    --best-known benchmarks/instances/netlib/best_known.json \\\n")
        out.write("    --out benchmarks/results/netlib_lp.json\n")
        out.write("python3 benchmarks/report_netlib.py benchmarks/results/netlib_lp.json\n")
        out.write("```\n\nHiGHS is reached through SciPy and is a REFERENCE only: it lives under\n"
                  "`benchmarks/` and nothing in the solver links or includes it.\n")


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        HERE, "results", "netlib_lp.json")
    with open(source) as handle:
        document = json.load(handle)

    rows = list(rows_from(document.get("results", [])))
    if not rows:
        print("no runs found", file=sys.stderr)
        return 1

    results = os.path.join(HERE, "results")
    os.makedirs(results, exist_ok=True)
    csv_path = os.path.join(results, "netlib_lp_benchmark.csv")
    markdown_path = os.path.join(results, "NETLIB_RESULTS.md")

    write_csv(csv_path, rows)
    write_markdown(markdown_path, rows,
                   document.get("tolerances", {}), source)
    print(f"wrote {os.path.relpath(csv_path, ROOT)} ({len(rows)} rows)")
    print(f"wrote {os.path.relpath(markdown_path, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
