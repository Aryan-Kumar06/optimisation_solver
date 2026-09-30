#!/usr/bin/env python3
"""Isolated convex-QP reference; never imported by the production solver."""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
from mps_model import read_mps
from highs_ref import sha256_file


def build_osqp_form(model):
    import numpy as np
    import scipy.sparse as sp
    mini = model.to_minimization()
    data, rows, cols = [], [], []
    for (i, j), value in mini.obj_quad.items():
        rows.append(i)
        cols.append(j)
        data.append(2 * value if i == j else value)
    # Model stores each off-diagonal direct coefficient once; OSQP wants upper P.
    P = sp.csc_matrix((data, (rows, cols)), shape=(model.n, model.n))
    q = np.zeros(model.n)
    for j, value in mini.obj_linear.items():
        q[j] = value
    data, rows, cols = [], [], []
    for i, row in enumerate(model.rows):
        for j, value in row.items():
            rows.append(i)
            cols.append(j)
            data.append(value)
    A = sp.vstack([sp.csc_matrix((data, (rows, cols)), shape=(model.m, model.n)),
                   sp.eye(model.n)], format='csc')
    return P, q, A, np.array(model.row_lower + model.var_lower), np.array(model.row_upper + model.var_upper)


def solve(model, time_limit=0):
    import numpy as np
    import osqp
    if model.is_integer_model():
        return {'termination': {'status': 'unsupported', 'message': 'OSQP reference requires continuous variables'}}
    P, q, A, lower, upper = build_osqp_form(model)
    prob = osqp.OSQP()
    settings = dict(verbose=False, eps_abs=1e-8, eps_rel=1e-8,
                    max_iter=200000, polishing=True)
    if time_limit > 0:
        settings['time_limit'] = time_limit
    prob.setup(P=P, q=q, A=A, l=lower, u=upper, **settings)
    result = prob.solve(raise_error=False)
    status = {1: 'optimal', 2: 'limit_reached', 3: 'infeasible',
              4: 'limit_reached', 5: 'unbounded', 6: 'limit_reached',
              7: 'limit_reached', 8: 'limit_reached'}.get(result.info.status_val, 'numerical_failure')
    have_point = result.info.status_val in (1, 2, 7, 8) and result.x is not None and np.isfinite(result.x).all()
    x = result.x.tolist() if have_point else None
    dual_ok = status == 'optimal' and result.y is not None and np.isfinite(result.y).all()
    flip = 1 if model.sense == 'max' else -1
    y = (flip * result.y).tolist() if dual_ok else None
    return dict(termination=dict(status=status, message=result.info.status,
                executed_engine='osqp', dispatched_engine='osqp',
                reason='time_limit' if result.info.status_val == 8 else 'unspecified'),
                primal=x, objective=model.objective(x) if x is not None else None,
                duals=y[:model.m] if y is not None else None,
                reduced_costs=y[model.m:] if y is not None else None,
                duals_unavailable_reason=None if dual_ok else 'No accurate optimal dual solution',
                work=dict(iterations=int(result.info.iter), nodes=None, solve_seconds=result.info.run_time))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('instance')
    p.add_argument('--out', required=True)
    p.add_argument('--time-limit', type=float, default=0)
    args = p.parse_args()
    record = dict(schema='optimsolver.solve.v1', primal=None, objective=None,
                  duals=None, reduced_costs=None, dual_bound=None, mip_gap=None,
                  settings=dict(requested_engine='osqp', tolerance=1e-8,
                                time_limit_seconds=args.time_limit))
    start = time.perf_counter()
    try:
        import osqp
        record['solver'] = dict(name='osqp', build_type=osqp.__version__, commit=None)
        model = read_mps(args.instance)
        record['instance'] = dict(path=args.instance, sha256=sha256_file(args.instance),
                                  variables=model.n, constraints=model.m)
        record.update(solve(model, args.time_limit))
    except Exception as error:
        record['termination'] = dict(status='numerical_failure', message=f'{type(error).__name__}: {error}')
    record['adapter_wall_seconds'] = time.perf_counter() - start
    with open(args.out, 'w') as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
    return 1 if record['termination']['status'] == 'numerical_failure' else 0


if __name__ == '__main__':
    sys.exit(main())
