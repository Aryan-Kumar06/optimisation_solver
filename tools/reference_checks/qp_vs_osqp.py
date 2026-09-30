#!/usr/bin/env python3
"""Fail-closed randomized convex QP comparison. Every generated case is accounted for."""
import argparse
import contextlib
import io
import json
import os
import subprocess
import sys

import numpy as np
import osqp
import scipy.sparse as sp


def read_cases(stream):
    count = int(stream.readline())
    if count <= 0:
        raise ValueError('No test cases')
    def vector(size):
        v = np.array([float(x) for x in stream.readline().split()])
        if len(v) != size:
            raise ValueError(f'Expected {size} values, got {len(v)}')
        return v
    for _ in range(count):
        n, m, mx, _, offset = stream.readline().split()
        n, m = int(n), int(m)
        P = np.array([vector(n) for _ in range(n)])
        q, bounds = vector(n), vector(2*n).reshape(n, 2)
        rows = np.array([vector(n+2) for _ in range(m)]).reshape(m, n+2)
        status, engine, obj = stream.readline().split()
        x = vector(n)
        if int(stream.readline()):
            vector(m)
        yield dict(P=P, q=q, A=rows[:, :n], lo=bounds[:, 0], hi=bounds[:, 1],
                   lower=rows[:, n], upper=rows[:, n+1], x=x, status=status,
                   engine=engine, objective=float(obj), offset=float(offset), maximize=bool(int(mx)))
    if stream.read().strip():
        raise ValueError('Unexpected trailing records')


@contextlib.contextmanager
def silenced_stdout():
    """Silence file descriptor 1 for the duration of a reference solve.

    OSQP 1.x prints "Polishing not needed - no active set detected at optimal
    point" from its C layer even with verbose=False. That went into this
    script's stdout ahead of the JSON report, so the report could not be
    parsed -- the exit code was still right, but a replayed seed could not be
    inspected. Redirecting sys.stdout does not reach C-level writes, so the
    descriptor itself is redirected.
    """
    sys.stdout.flush()
    saved = os.dup(1)
    try:
        with open(os.devnull, 'w') as null:
            os.dup2(null.fileno(), 1)
            yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)


def compare(case, factory=osqp.OSQP):
    P, q, x = case['P'], case['q'], case['x']
    n = len(q)
    A = sp.vstack([sp.csc_matrix(case['A']), sp.eye(n)], format='csc')
    lo = np.concatenate([case['lower'], case['lo']])
    hi = np.concatenate([case['upper'], case['hi']])
    flip = -1 if case['maximize'] else 1
    ref = factory()
    ref.setup(P=sp.triu(sp.csc_matrix(flip*P), format='csc'), q=flip*q,
              A=A, l=lo, u=hi, verbose=False, eps_abs=1e-9, eps_rel=1e-9,
              max_iter=200000, polishing=True, time_limit=5)
    with silenced_stdout():
        result = ref.solve(raise_error=False)
    reference_status = result.info.status
    status = case['status']
    if status != 'optimal':
        expected = {'infeasible': 'primal infeasible', 'unbounded': 'dual infeasible'}.get(status)
        if expected is None or reference_status != expected:
            raise AssertionError(f'status {status}, reference {reference_status}')
        return 'nonoptimal_corroborated'  # comparison, not certificate verification
    if reference_status != 'solved':
        raise AssertionError(f'optimal result unverified: reference {reference_status}')
    if not np.isfinite(x).all() or not np.isfinite(case['objective']):
        raise AssertionError('nonfinite engine result')
    for point in (x, result.x):
        if point is None or not np.isfinite(point).all():
            raise AssertionError('nonfinite reference result')
        ax = A @ point
        violation = max(0., float(np.max(lo-ax)), float(np.max(ax-hi)))
        if violation > 1.01e-4:
            raise AssertionError(f'primal violation {violation}')
    objective = lambda v: float(.5*v@P@v + q@v + case['offset'])
    ours, reference = objective(x), objective(result.x)
    if abs(ours-case['objective']) / (1+abs(ours)) > 1e-4:
        raise AssertionError('self-reported objective mismatch')
    if abs(ours-reference)/(1+abs(reference)) > 1e-4 or np.max(np.abs(x-result.x)) > 1e-3:
        raise AssertionError(f'OSQP mismatch: engine {ours}, reference {reference}')
    return 'optimal_compared'


def check(stream):
    records = []
    for i, case in enumerate(read_cases(stream)):
        try:
            records.append(dict(case=i, outcome=compare(case)))
        except Exception as error:
            records.append(dict(case=i, outcome='failed', detail=f'{type(error).__name__}: {error}'))
    return dict(selected=len(records), checked=sum(r['outcome'] != 'failed' for r in records),
                failed=sum(r['outcome'] == 'failed' for r in records), cases=records)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', nargs='?')
    p.add_argument('--generator')
    p.add_argument('--count', type=int, default=300)
    p.add_argument('--seed', type=int, default=20260908)
    args = p.parse_args()
    try:
        if args.generator:
            generated = subprocess.run([args.generator, str(args.count), str(args.seed)],
                                       capture_output=True, text=True, check=True, timeout=180)
            report = check(io.StringIO(generated.stdout))
            if report['selected'] != args.count:
                raise ValueError('Generator case count mismatch')
        elif args.input:
            with open(args.input) as stream:
                report = check(stream)
        else:
            p.error('provide an input file or --generator')
        print(json.dumps(report, indent=2))
        return int(report['failed'] != 0)
    except Exception as error:
        print(f'Reference check failed: {type(error).__name__}: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
