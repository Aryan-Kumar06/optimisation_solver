#!/usr/bin/env python3
"""Summarise a MILP benchmark run: per-instance table, engine rollups, gaps."""
from __future__ import annotations
import json, math, sys, statistics

def load(path): return json.load(open(path))

def fmt(v, w=12, p=6):
    if v is None: return "-".rjust(w)
    if isinstance(v, float):
        if not math.isfinite(v): return "-".rjust(w)
        return f"{v:.{p}g}".rjust(w)
    return str(v).rjust(w)

def main(path):
    d = load(path)
    rows = d["results"]
    print(f"{'instance':<22}{'solver':<10}{'outcome':<34}{'incumbent':>14}"
          f"{'bound':>13}{'commongap':>11}{'refdiff':>10}{'nodes':>9}{'s':>6}{'MB':>7}")
    print("-"*136)
    agg = {}
    for e in rows:
        name = e["instance_name"].replace(".mps","")
        ref = e.get("best_known_objective")
        for r in e["runs"]:
            m = r.get("milp") or {}
            p = r.get("process") or {}
            solver = "ours" if r["solver"].startswith("optimsolver") else "highs"
            a = agg.setdefault(solver, {"n":0,"opt":0,"to_inc":0,"to_noinc":0,"limited":0,
                                        "bad":0,"fail":0,"matched":0,
                                        "gaps":[],"refdiffs":[],"nodes":0,"secs":0.0,"mem":0.0})
            a["n"] += 1
            o = m.get("outcome")
            if o == "reported_optimal_within_tolerance": a["opt"] += 1
            elif o == "timeout_with_incumbent": a["to_inc"] += 1
            elif o == "timeout_without_incumbent": a["to_noinc"] += 1
            elif o in ("limit_with_incumbent", "limit_without_incumbent"): a["limited"] += 1
            elif o == "incorrect_result": a["bad"] += 1
            elif o in ("solver_failure","no_result"): a["fail"] += 1
            if m.get("matches_reference"): a["matched"] += 1
            if m.get("common_gap") is not None: a["gaps"].append(m["common_gap"])
            if m.get("reference_relative_difference") is not None:
                a["refdiffs"].append(m["reference_relative_difference"])
            a["nodes"] += (r.get("work") or {}).get("nodes") or 0
            a["secs"] += p.get("wall_seconds") or 0.0
            a["mem"] = max(a["mem"], (p.get("peak_memory_bytes") or 0)/1048576)
            print(f"{name:<22}{solver:<10}{str(o):<34}"
                  f"{fmt(m.get('incumbent'),14)}{fmt(m.get('dual_bound'),13)}"
                  f"{fmt(m.get('common_gap'),11,4)}{fmt(m.get('reference_relative_difference'),10,4)}"
                  f"{fmt((r.get('work') or {}).get('nodes'),9)}"
                  f"{fmt(p.get('wall_seconds'),6,3)}"
                  f"{fmt((p.get('peak_memory_bytes') or 0)/1048576,7,3)}")
    print("\nENGINE SUMMARY")
    print(f"{'engine':<8}{'n':>4}{'opt':>5}{'TO+inc':>8}{'TO-inc':>8}{'wrong':>7}{'fail':>6}"
          f"{'matched ref':>13}{'median gap':>12}{'median refdiff':>16}{'nodes':>12}{'peakMB':>9}")
    for k, a in sorted(agg.items()):
        med = lambda xs: (statistics.median(xs) if xs else None)
        if a["limited"]: print(f"{k}: {a['limited']} non-time limits (excluded from timeout counts)")
        print(f"{k:<8}{a['n']:>4}{a['opt']:>5}{a['to_inc']:>8}{a['to_noinc']:>8}"
              f"{a['bad']:>7}{a['fail']:>6}{a['matched']:>13}"
              f"{fmt(med(a['gaps']),12,4)}{fmt(med(a['refdiffs']),16,4)}"
              f"{a['nodes']:>12}{a['mem']:>9.1f}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv)>1 else "benchmarks/results/miplib_dev25.json")
