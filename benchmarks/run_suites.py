#!/usr/bin/env python3
"""Reproducible, offline, bounded suite runs with complete failure accounting."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
from fetch_suites import atomic_write
import platform
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent


def selection(suite, full_qp=False):
    directory = HERE / 'instances' / suite
    if suite in ('qp', 'mittelmann'):
        spec = json.loads((HERE / 'suites' / (suite + '.json')).read_text())
        entries = spec['instances']
        if suite == 'qp' and not full_qp:
            entries = [e for e in entries if e['name'] in spec['smoke']]
    elif suite == 'netlib':
        spec = json.loads((directory / 'manifest.json').read_text())
        entries = spec['instances']
    else:
        spec = json.loads((directory / 'FROZEN_DEV25.json').read_text())
        entries = spec['selection']
        manifest = json.loads((directory / 'manifest_FROZEN_DEV25.json').read_text())
        hashes = {e['name']: e.get('mps_sha256') for e in manifest['instances']}
        entries = [dict(e, sha256=hashes.get(e['name'])) for e in entries]
    return spec, [dict(name=e['name'], sha256=e.get('sha256', e.get('mps_sha256')),
                      published_objective=e.get('published_objective'),
                      path=str(directory / 'mps' / (e['name'] + '.mps'))) for e in entries]


def assess(entry):
    """Comparison only counts two validated points and an optimal reference."""
    if (entry.get('parse_check') or {}).get('status') != 'agree':
        return 'parse_failure'
    runs = entry.get('runs', [])
    if len(runs) != 2:
        return 'missing_result'
    ours, ref = runs
    for run in runs:
        check = run.get('check') or {}
        if check.get('verdict') in ('infeasible_point', 'nonfinite', 'malformed'):
            return 'rejected_point'
        if check.get('verdict') not in ('feasible', 'optimal_verified'):
            return 'unverified'
    if (ref.get('termination') or {}).get('status') != 'optimal':
        return 'reference_not_optimal'
    if (ours.get('termination') or {}).get('status') != 'optimal':
        return 'feasible_not_optimal'
    if not (ours.get('engine_use') or {}).get('counts_for_requested', False):
        return 'requested_engine_not_run'
    a, b = ours['check'].get('objective_recomputed'), ref['check'].get('objective_recomputed')
    if a is None or b is None:
        return 'unverified'
    return 'objective_agrees' if abs(a-b) <= 1e-6 * max(1, abs(a), abs(b)) else 'objective_mismatch'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', choices=['all','netlib','miplib','qp','mittelmann'], default='all')
    p.add_argument('--full-qp', action='store_true')
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--runner', type=Path, required=True)
    p.add_argument('--timeout', type=float, default=10)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.timeout <= 0:
        p.error('--timeout must be positive')
    report = dict(schema='optimsolver.coverage.v1', utc=datetime.now(timezone.utc).isoformat(),
                  platform=platform.platform(), python=sys.version, timeout_seconds=args.timeout,
                  binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                  note='Subset coverage, not equivalent to published Mittelmann runs. Objective agreement is not an optimality certificate.', suites=[])
    report['reference_versions'] = {}
    for package in ('numpy', 'scipy', 'osqp'):
        try:
            report['reference_versions'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report['reference_versions'][package] = None
    for field, command in [('commit', ['git','rev-parse','HEAD']), ('working_tree', ['git','status','--porcelain'])]:
        result = subprocess.run(command,cwd=HERE,capture_output=True,text=True)
        report[field] = result.stdout.strip() if result.returncode == 0 else None
    suites = ['netlib','miplib','qp','mittelmann'] if args.suite == 'all' else [args.suite]
    for suite in suites:
        spec, entries = selection(suite, args.full_qp)
        current = dict(name=suite, source=spec.get('source'), selected=len(entries), instances=[dict(name=e['name'],outcome='not_run') for e in entries])
        report['suites'].append(current)
        for index, entry in enumerate(entries):
            row = dict(name=entry['name'])
            path = Path(entry['path'])
            try:
                if not path.exists():
                    row['outcome'] = 'missing_data'
                elif not entry['sha256'] or hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
                    row['outcome'] = 'checksum_failure'
                else:
                    with tempfile.TemporaryDirectory() as tmp:
                        output = Path(tmp) / 'run.json'
                        solvers = 'qp,osqp' if suite == 'qp' else 'auto,highs'
                        cmd = [sys.executable, str(HERE/'bench.py'), str(path), '--binary', str(args.binary.resolve()),
                               '--runner', str(args.runner.resolve()), '--timeout', str(args.timeout),
                               '--solvers', solvers, '--out', str(output)]
                        known = HERE / 'instances' / suite / ('best_known_FROZEN_DEV25.json' if suite == 'miplib' else 'best_known.json')
                        if entry.get('published_objective') is not None:
                            known = Path(tmp) / 'best_known.json'
                            known.write_text(json.dumps({path.name:entry['published_objective']}))
                        if known.exists():
                            cmd += ['--best-known', str(known)]
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3*args.timeout+60)
                        if result.returncode != 0 or not output.exists():
                            row.update(outcome='harness_failure', detail=result.stderr[-2000:])
                        else:
                            row['record'] = json.loads(output.read_text())['results'][0]
                            row['outcome'] = assess(row['record'])
            except subprocess.TimeoutExpired:
                row['outcome'] = 'harness_timeout'
            except Exception as error:
                row.update(outcome='harness_failure', detail=str(error))
            current['instances'][index] = row
            current['outcomes'] = dict(Counter(r['outcome'] for r in current['instances']))
            args.out.parent.mkdir(parents=True, exist_ok=True)
            atomic_write(args.out, (json.dumps(report, indent=2, allow_nan=False)+'\n').encode())
            print(f"{suite}/{entry['name']}: {row['outcome']}", flush=True)
    # Strict mode is the only mode: unknown, timeout and missing data never pass.
    return int(any(r['outcome'] != 'objective_agrees' for s in report['suites'] for r in s['instances']))


if __name__ == '__main__':
    sys.exit(main())
