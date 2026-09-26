#!/usr/bin/env python3
"""Fetch and decode Netlib LP instances, and write a manifest.

Netlib does NOT distribute plain MPS, and it does not distribute gzip either.
The files are in a historical run-length-compressed format from the mid-1980s
whose decompressor, emps.c, is published in the same directory. A pipeline that
downloads these and feeds them to a reader sees line noise.

The format is detected from the file's own bytes rather than assumed:

    1f 8b ...            gzip            -> gunzip
    "NAME" in the first  plain MPS       -> use as-is
      line and a later
      ROWS/COLUMNS
    otherwise            emps-compressed -> run through emps

Every downloaded byte and every decoded byte is hashed, so a result row can be
tied to an exact input and a silently changed upstream file is detectable.

Usage:
    python3 benchmarks/fetch_netlib.py --set smoke
    python3 benchmarks/fetch_netlib.py --instances afiro adlittle
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
NETLIB_DIR = os.path.join(HERE, "instances", "netlib")
RAW_DIR = os.path.join(NETLIB_DIR, "raw")
MPS_DIR = os.path.join(NETLIB_DIR, "mps")
TOOLS_DIR = os.path.join(NETLIB_DIR, "tools")

BASE_URL = "https://www.netlib.org/lp/data/"
EMPS_URL = BASE_URL + "emps.c"
README_URL = BASE_URL + "readme"

SMOKE = ["afiro", "adlittle", "blend", "sc50a", "sc50b", "share2b",
         "degen2", "recipe"]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download(url: str, destination: str) -> bytes:
    """Cache-first fetch.

    Uses curl rather than urllib: a stock python.org install on macOS ships
    without a CA bundle, so urllib fails CERTIFICATE_VERIFY_FAILED on every
    https URL while curl uses the system trust store. Falling back the other way
    round would make the fetch fail on exactly the machines it needs to work on.
    """
    if os.path.exists(destination) and os.path.getsize(destination) > 0:
        with open(destination, "rb") as handle:
            return handle.read()
    os.makedirs(os.path.dirname(destination), exist_ok=True)

    curl = shutil.which("curl")
    if curl is not None:
        result = subprocess.run(
            [curl, "-sS", "--fail", "--location", "--max-time", "90",
             "-o", destination, url],
            capture_output=True, text=True)
        if result.returncode == 0 and os.path.getsize(destination) > 0:
            with open(destination, "rb") as handle:
                return handle.read()
        raise RuntimeError(f"curl failed for {url}: {result.stderr.strip()[:200]}")

    with urllib.request.urlopen(url, timeout=90) as response:
        data = response.read()
    with open(destination, "wb") as handle:
        handle.write(data)
    return data


def build_emps() -> str:
    """Compile Netlib's own decompressor. Nothing here reimplements it."""
    os.makedirs(TOOLS_DIR, exist_ok=True)
    source = os.path.join(TOOLS_DIR, "emps.c")
    binary = os.path.join(TOOLS_DIR, "emps")
    download(EMPS_URL, source)
    if not os.path.exists(binary) or \
            os.path.getmtime(binary) < os.path.getmtime(source):
        compiler = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
        if compiler is None:
            raise RuntimeError("no C compiler found to build emps")
        subprocess.run([compiler, "-w", "-o", binary, source], check=True)
    return binary


def detect_format(data: bytes) -> str:
    if data[:2] == b"\x1f\x8b":
        return "gzip"
    head = data[:4096].decode("latin-1", errors="replace").upper()
    # A plain MPS file has section headers in column 1. The emps format also
    # opens with NAME, so NAME alone does not settle it -- ROWS is what a
    # decompressed file has and a compressed one does not.
    if "NAME" in head and ("\nROWS" in head or head.startswith("ROWS")):
        return "plain"
    return "emps"


def decode(name: str, raw: bytes, emps_binary: str) -> tuple[bytes, str]:
    kind = detect_format(raw)
    if kind == "plain":
        return raw, kind
    if kind == "gzip":
        import gzip
        return gzip.decompress(raw), kind

    raw_path = os.path.join(RAW_DIR, name)
    result = subprocess.run([emps_binary, raw_path], capture_output=True)
    if result.returncode != 0 or not result.stdout:
        raise RuntimeError(
            f"emps failed on {name}: rc={result.returncode} "
            f"{result.stderr.decode('latin-1', errors='replace')[:200]}")
    return result.stdout, kind


def parse_readme_objectives(text: str) -> dict:
    """Reference optima from the table in Netlib's own readme.

    These are EXTERNAL CLAIMS. The pipeline uses them for corroboration and
    never as an optimality proof; see benchmarks/README.md.
    """
    values = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 6:
            continue
        name = fields[0].lower()
        if not name.isalnum() and "_" not in name:
            continue
        for token in reversed(fields[1:]):
            token = token.replace("*", "")
            if ("E" in token.upper() or "." in token) and any(
                    c.isdigit() for c in token):
                try:
                    values[name] = float(token)
                except ValueError:
                    pass
                break
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", choices=["smoke"], default=None)
    parser.add_argument("--instances", nargs="*", default=None)
    parser.add_argument("--manifest",
                        default=os.path.join(NETLIB_DIR, "manifest.json"))
    args = parser.parse_args()

    names = args.instances if args.instances else (
        SMOKE if args.set == "smoke" else SMOKE)

    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(MPS_DIR, exist_ok=True)

    emps_binary = build_emps()
    readme_path = os.path.join(NETLIB_DIR, "netlib_readme.txt")
    readme = download(README_URL, readme_path).decode("latin-1", errors="replace")
    objectives = parse_readme_objectives(readme)

    sys.path.insert(0, os.path.join(HERE, "lib"))
    from mps_model import read_mps, MpsError

    entries = []
    for name in names:
        url = BASE_URL + name
        raw = download(url, os.path.join(RAW_DIR, name))
        entry = {
            "name": name,
            "source_url": url,
            "raw_bytes": len(raw),
            "raw_sha256": sha256_bytes(raw),
            "encoding": None,
            "decoder": None,
            "mps_path": None,
            "mps_sha256": None,
            "rows": None, "columns": None, "nonzeros": None,
            "problem_class": None, "sense": None,
            "reference_objective": objectives.get(name),
            "reference_provenance":
                "table in https://www.netlib.org/lp/data/readme (external claim, "
                "used for corroboration only, never as an optimality proof)",
            "compatible": None,
            "incompatibility": None,
            "reader_warnings": [],
        }
        try:
            decoded, kind = decode(name, raw, emps_binary)
            entry["encoding"] = kind
            entry["decoder"] = {"emps": "netlib emps.c", "gzip": "gzip",
                                "plain": "none"}[kind]
            mps_path = os.path.join(MPS_DIR, name + ".mps")
            with open(mps_path, "wb") as handle:
                handle.write(decoded)
            entry["mps_path"] = os.path.relpath(mps_path, os.path.dirname(HERE))
            entry["mps_sha256"] = sha256_bytes(decoded)

            model = read_mps(mps_path)
            entry.update({
                "rows": model.m, "columns": model.n,
                "nonzeros": sum(len(r) for r in model.rows),
                "problem_class": model.problem_class(),
                "sense": model.sense,
                "reader_warnings": model.warnings[:20],
            })
            # Compatibility is decided by what the solver can represent, not by
            # whether a solve happens to succeed.
            if model.problem_class() != "LP":
                entry["compatible"] = False
                entry["incompatibility"] = (
                    f"class {model.problem_class()} is outside the LP engines "
                    f"under test")
            else:
                entry["compatible"] = True
        except (MpsError, RuntimeError) as error:
            entry["compatible"] = False
            entry["incompatibility"] = f"{type(error).__name__}: {error}"

        entries.append(entry)
        status = "ok" if entry["compatible"] else "INCOMPATIBLE"
        print(f"{name:<10} {str(entry['encoding']):<6} "
              f"{str(entry['rows']):>5} x {str(entry['columns']):<5} "
              f"nnz={str(entry['nonzeros']):<6} "
              f"ref={entry['reference_objective']!s:<18} {status}")

    manifest = {
        "source": BASE_URL,
        "note": ("Netlib distributes a historical run-length-compressed format, "
                 "not gzip. Files are decoded with Netlib's own emps.c, "
                 "compiled from source at fetch time. Encoding is detected from "
                 "file bytes, never assumed."),
        "emps_source": EMPS_URL,
        "instances": entries,
    }

    # MERGE with whatever the manifest already describes, keyed by name.
    # Overwriting wholesale meant that fetching one instance
    # (`--instances adlittle`) silently reduced an 8-instance manifest to one,
    # discarding the hashes and reference objectives every recorded result is
    # tied to. Re-fetching a subset must refresh those entries, not delete the
    # rest.
    if os.path.exists(args.manifest):
        try:
            with open(args.manifest) as handle:
                previous = json.load(handle)
            merged = {entry["name"]: entry
                      for entry in previous.get("instances", [])}
        except (OSError, ValueError, KeyError):
            merged = {}
        for entry in entries:
            merged[entry["name"]] = entry
        manifest["instances"] = [merged[name] for name in sorted(merged)]
    with open(args.manifest, "w") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"\nmanifest: {args.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
