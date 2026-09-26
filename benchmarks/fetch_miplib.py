#!/usr/bin/env python3
"""Download the frozen MIPLIB smoke selection, decode, hash, screen compatibility.

MIPLIB ships .mps.gz -- genuinely gzip, unlike Netlib's historical emps format.
The encoding is still detected from the bytes rather than from the extension,
so a source that changes format is noticed instead of silently mis-decoded.

Compatibility screening happens AFTER the selection is frozen and is not a
performance observation: it asks only whether the file can be represented, and
an instance that fails is recorded as incompatible rather than replaced.
"""
from __future__ import annotations
import gzip, hashlib, json, os, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(HERE, "instances", "miplib")
RAW, MPS = os.path.join(DIR, "raw"), os.path.join(DIR, "mps")
sys.path.insert(0, os.path.join(HERE, "lib"))
from mps_model import read_mps, MpsError

def sha(b): return hashlib.sha256(b).hexdigest()

def fetch(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return open(dest, "rb").read()
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    r = subprocess.run([shutil.which("curl"), "-sS", "--fail", "-L",
                        "--max-time", "120", "-o", dest, url],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"curl: {r.stderr.strip()[:160]}")
    return open(dest, "rb").read()

def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "FROZEN_SMOKE.json"
    frozen = json.load(open(os.path.join(DIR, which)))
    os.makedirs(RAW, exist_ok=True); os.makedirs(MPS, exist_ok=True)
    out = []
    for entry in frozen["selection"]:
        name = entry["name"]
        rec = dict(entry)
        try:
            raw = fetch(entry["url"], os.path.join(RAW, name + ".mps.gz"))
            rec["raw_bytes"], rec["raw_sha256"] = len(raw), sha(raw)
            # Encoding from bytes, not from the file name.
            if raw[:2] == b"\x1f\x8b":
                rec["encoding"] = "gzip"; decoded = gzip.decompress(raw)
            else:
                rec["encoding"] = "plain"; decoded = raw
            path = os.path.join(MPS, name + ".mps")
            open(path, "wb").write(decoded)
            rec["mps_path"] = os.path.relpath(path, os.path.dirname(HERE))
            rec["mps_sha256"], rec["mps_bytes"] = sha(decoded), len(decoded)

            m = read_mps(path)
            rec["parsed"] = {
                "rows": m.m, "columns": m.n,
                "nonzeros": sum(len(r) for r in m.rows),
                "binaries": sum(1 for j in range(m.n)
                                if m.var_type[j] != "C"
                                and m.var_lower[j] == 0.0 and m.var_upper[j] == 1.0),
                "integral_columns": sum(1 for t in m.var_type if t != "C"),
                "continuous": sum(1 for t in m.var_type if t == "C"),
                "class": m.problem_class(), "sense": m.sense,
                "warnings": m.warnings[:10],
            }
            # Metadata vs. parse must agree, or the reference objective may not
            # belong to the model we are about to solve.
            mismatches = []
            if m.m != entry["constraints"]:
                mismatches.append(f"rows {m.m} vs metadata {entry['constraints']}")
            if m.n != entry["variables"]:
                mismatches.append(f"cols {m.n} vs metadata {entry['variables']}")
            rec["metadata_agreement"] = "agree" if not mismatches else "; ".join(mismatches)
            rec["compatible"] = (m.problem_class() == "MILP" and not mismatches)
            if m.problem_class() != "MILP":
                rec["incompatibility"] = f"class {m.problem_class()}"
        except (MpsError, RuntimeError, OSError) as e:
            rec["compatible"] = False
            rec["incompatibility"] = f"{type(e).__name__}: {e}"
        out.append(rec)
        p = rec.get("parsed") or {}
        print(f"{name:<12}{str(rec.get('encoding')):<6}"
              f"{str(p.get('rows')):>5} x {str(p.get('columns')):<5} "
              f"nnz={str(p.get('nonzeros')):<6} intcols={str(p.get('integral_columns')):<5} "
              f"{str(p.get('class')):<5} {rec.get('metadata_agreement','')} "
              f"{'OK' if rec.get('compatible') else 'INCOMPATIBLE: '+str(rec.get('incompatibility'))}")
    man = {"source": frozen["source"], "frozen_selection_utc": frozen["frozen_utc"],
           "note": "MIPLIB ships gzip; encoding detected from bytes.",
           "instances": out}
    json.dump(man, open(os.path.join(DIR, "manifest_" + which.replace(".json","") + ".json"), "w"), indent=2)
    json.dump({e["name"] + ".mps": e["reference_objective"] for e in out},
              open(os.path.join(DIR, "best_known_" + which.replace(".json","") + ".json"), "w"), indent=2)
    print("\nmanifest written for", which)

if __name__ == "__main__":
    sys.exit(main())
