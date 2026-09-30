#!/usr/bin/env python3
"""Analytic oracle tests for sparse reference translation and failure handling."""
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
from scipy.sparse import issparse

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE/'adapters'),str(HERE/'lib'),str(HERE.parent/'tools/reference_checks')]
from mps_model import Model
from highs_ref import build_scipy_form, solve_lp, solve_milp
from osqp_ref import build_osqp_form, solve
from verify import check_solution
import qp_vs_osqp


def model():
    return Model(var_names=['x','y'], var_type=['C','C'], var_lower=[-float('inf')]*2,
                 var_upper=[float('inf')]*2, obj_linear={0:-2,1:-4},
                 obj_quad={(0,0):1,(0,1):1,(1,1):2},offset=7)


class ReferenceAdapters(unittest.TestCase):
    def test_quadratic_convention_offset_and_sense(self):
        for sense in ('min','max'):
            m = model()
            if sense == 'max':
                m.sense = sense
                m.obj_quad = {k:-v for k,v in m.obj_quad.items()}
                m.obj_linear = {k:-v for k,v in m.obj_linear.items()}
            P,q,A,l,u = build_osqp_form(m)
            np.testing.assert_allclose(P.toarray(),[[2,1],[0,4]])
            self.assertEqual(A.shape,(2,2))
            r = solve(m)
            self.assertEqual(r['termination']['status'],'optimal')
            np.testing.assert_allclose(r['primal'],[4/7,6/7],atol=1e-7)
            self.assertAlmostEqual(r['objective'],7 + (16/7 if sense=='max' else -16/7),places=6)
            check = check_solution(m,r['primal'],r['duals'],r['reduced_costs'], 'optimal')
            self.assertEqual(check['verdict'],'optimal_verified',check)

    def test_qp_active_bounds_and_equality_dual_signs(self):
        m = model()
        m.row_names=['balance'];m.rows=[{0:1,1:1}];m.row_lower=[2];m.row_upper=[2]
        m.var_upper=[.2,float('inf')]
        r=solve(m)
        np.testing.assert_allclose(r['primal'],[.2,1.8],atol=1e-7)
        self.assertEqual(check_solution(m,r['primal'],r['duals'],r['reduced_costs'],'optimal')['verdict'],'optimal_verified')

    def test_sparse_lp_row_mapping(self):
        m=model();m.obj_quad={}
        m.row_names=['eq','upper','lower','range','free']
        m.rows=[{0:1},{1:2},{0:3},{1:4},{0:5}]
        m.row_lower=[1,-np.inf,3,0,-np.inf];m.row_upper=[1,4,np.inf,8,np.inf]
        _,ub,bub,eq,beq,_,mapping=build_scipy_form(m)
        self.assertTrue(issparse(ub) and issparse(eq))
        np.testing.assert_allclose(ub.toarray(),[[0,2],[-3,0],[0,4],[0,-4]])
        np.testing.assert_allclose(bub,[4,-3,8,0])
        self.assertEqual(mapping,[('eq',0,None),('ub',0,None),('ub',None,1),('ub',2,3),('ub',None,None)])
        r=solve_lp(m,'highs-ds',5)
        np.testing.assert_allclose(r['primal'],[1,2])
        self.assertEqual(check_solution(m,r['primal'],r['duals'],r['reduced_costs'],'optimal')['verdict'],'optimal_verified')
        m.var_type=['I','I']
        np.testing.assert_allclose(solve_milp(m,5)['primal'],[1,2])

    def test_qp_refusal_and_infeasibility(self):
        m=model();m.var_type[0]='I'
        self.assertEqual(solve(m)['termination']['status'],'unsupported')
        m.var_type[0]='C';m.row_names=['a','b'];m.rows=[{0:1},{0:1}]
        m.row_lower=[1,-np.inf];m.row_upper=[np.inf,0]
        r=solve(m)
        self.assertEqual(r['termination']['status'],'infeasible')
        self.assertIsNone(r['primal'])

    def test_qp_semidefinite_unbounded_and_bad_curvature(self):
        m=model();m.obj_quad={};m.obj_linear={0:-1}
        r=solve(m)
        self.assertEqual(r['termination']['status'],'unbounded')
        self.assertIsNone(r['primal'])
        m.obj_quad={(0,0):-1,(1,1):-1}
        # A failed/nonconvex reference must never manufacture an optimum.
        with self.assertRaises(Exception):
            solve(m)

    def test_random_checker_fails_closed(self):
        c=dict(P=np.eye(1),q=np.zeros(1),A=np.zeros((0,1)),lo=np.array([-1.]),hi=np.array([1.]),
               lower=np.array([]),upper=np.array([]),x=np.zeros(1),status='optimal',objective=0.,offset=0.,maximize=False)
        def broken():
            raise RuntimeError('injected reference error')
        with self.assertRaises(RuntimeError):
            qp_vs_osqp.compare(c,broken)
        with patch.object(qp_vs_osqp,'read_cases',return_value=iter([c])), patch.object(qp_vs_osqp,'compare',side_effect=RuntimeError('injected')):
            r=qp_vs_osqp.check(io.StringIO(''))
            self.assertEqual((r['selected'],r['checked'],r['failed']),(1,0,1))
        c['x'][0]=np.nan
        with self.assertRaises(AssertionError):
            qp_vs_osqp.compare(c)
        with self.assertRaises(ValueError):
            list(qp_vs_osqp.read_cases(io.StringIO('0\n')))
        with self.assertRaises(ValueError):
            list(qp_vs_osqp.read_cases(io.StringIO('1\n')))


if __name__ == '__main__':
    unittest.main()
