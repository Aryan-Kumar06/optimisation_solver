#!/usr/bin/env python3
"""Deterministic CLI/reference comparisons; SciPy is an optional test dependency.
Usage: python3 reference.py --binary build/optimsolver [--output report.json]
"""
import argparse
import json
import pathlib
import subprocess
import tempfile
import time
import numpy as np
import scipy
from scipy.optimize import minimize

class Tape:
    def __init__(self):
        self.nodes = []
    def node(self, op, *args):
        self.nodes.append(" ".join(map(str, (op, *args))))
        return len(self.nodes) - 1
    def text(self, bounds, initial, objective, constraints):
        return "\n".join([
            "nlp 1", f"variables {len(initial)}",
            *(f"{lo} {hi} {x}" for (lo, hi), x in zip(bounds, initial)),
            f"nodes {len(self.nodes)}", *self.nodes,
            f"objective {objective}", f"constraints {len(constraints)}",
            *(f"{lo} {hi} {root}" for lo, hi, root in constraints), "",
        ])

def cases():
    t = Tape(); x = t.node("var", 0); y = t.node("var", 1)
    c = t.node("const", 100); one = t.node("const", 1)
    f = t.node("add", t.node("mul", c, t.node("square", t.node("sub", y, t.node("square", x)))), t.node("square", t.node("sub", one, x)))
    yield "rosenbrock", t.text([(-np.inf, np.inf)]*2, [-1.2, 1], f, []), lambda v: 100*(v[1]-v[0]**2)**2+(1-v[0])**2, [-1.2, 1], [(None, None)]*2, [], 0.0
    t = Tape(); x, y = [t.node("var", i) for i in range(2)]
    f = t.node("neg", t.node("add", x, y)); c = t.node("add", t.node("square", x), t.node("square", y))
    yield "circle", t.text([(-np.inf, np.inf)]*2, [.2, .1], f, [(-np.inf, 1, c)]), lambda v: -sum(v), [.2, .1], [(None, None)]*2, [{"type":"ineq", "fun":lambda v:1-np.dot(v,v)}], -np.sqrt(2)
    t = Tape(); x,y,z,w = [t.node("var", i) for i in range(4)]
    f = t.node("add", t.node("mul", t.node("mul", x,w), t.node("add",t.node("add",x,y),z)),z)
    product = t.node("mul", t.node("mul",x,y),t.node("mul",z,w))
    norm = t.node("add",t.node("add",t.node("square",x),t.node("square",y)),t.node("add",t.node("square",z),t.node("square",w)))
    yield "hs71", t.text([(1,5)]*4,[1,5,5,1],f,[(25,np.inf,product),(40,40,norm)]), lambda v:v[0]*v[3]*(v[0]+v[1]+v[2])+v[2], [1,5,5,1], [(1,5)]*4, [{"type":"ineq","fun":lambda v:np.prod(v)-25},{"type":"eq","fun":lambda v:np.dot(v,v)-40}], 17.014017289
    rng=np.random.default_rng(20260927)
    for k in range(12):
        n=2+k%5; target=rng.uniform(.5,2,n); weights=rng.uniform(.5,3,n)
        t=Tape(); f=t.node("const",0)
        for j in range(n):
            x=t.node("var",j); d=t.node("sub",x,t.node("const",target[j])); d2=t.node("square",d)
            f=t.node("add",f,t.node("mul",t.node("const",weights[j]),t.node("add",d2,t.node("square",d2))))
        initial=rng.uniform(-2,3,n).tolist()
        def fun(v, target=target, weights=weights):
            d=np.asarray(v)-target
            return np.dot(weights,d*d+d**4)
        yield f"quartic_{k}",t.text([(-3,3)]*n,initial,f,[]),fun,initial,[(-3,3)]*n,[],0.0

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--binary",required=True);parser.add_argument("--output")
    args=parser.parse_args(); binary=str(pathlib.Path(args.binary).resolve()); results=[]
    with tempfile.TemporaryDirectory() as temp:
        for name,text,fun,x0,bounds,constraints,known in cases():
            path=pathlib.Path(temp)/f"{name}.nlp";path.write_text(text)
            raw=subprocess.run([binary,"solve-nlp",str(path)],capture_output=True,text=True,timeout=60)
            result=json.loads(raw.stdout)
            start=time.perf_counter()
            ref=minimize(fun,x0,method="SLSQP",bounds=bounds,constraints=constraints,options={"ftol":1e-11,"maxiter":1000})
            seconds=time.perf_counter()-start
            assert raw.returncode==0 and result["status"]=="FirstOrderStationary",(name,result,raw.stderr)
            assert ref.success,(name,ref.message)
            evaluated=float(fun(result["primal"]))
            assert abs(evaluated-result["objective"])<1e-8,(name,"objective mismatch")
            assert abs(evaluated-known)<2e-5 and abs(evaluated-ref.fun)<2e-5,(name,evaluated,known,ref.fun)
            for c in constraints:
                value=c["fun"](result["primal"])
                assert abs(value)<1e-6 if c["type"]=="eq" else value>=-1e-6
            for v,(lo,hi) in zip(result["primal"],bounds):
                assert (lo is None or v>=lo-1e-6) and (hi is None or v<=hi+1e-6)
            for key in ("primal_residual","dual_residual","complementarity"):
                assert result[key] is not None and result[key]<=1e-6,(name,key,result[key])
            results.append({"name":name,"nlp":result,"slsqp_objective":float(ref.fun),"slsqp_iterations":int(ref.nit),"slsqp_seconds":seconds})
            print(f"{name}: NLP {evaluated:.10g}, SLSQP {ref.fun:.10g}")
    report={"scipy_version":scipy.__version__,"seed":20260927,"cases":results}
    if args.output:pathlib.Path(args.output).write_text(json.dumps(report,indent=2)+"\n")
    print(f"Passed {len(results)} independent comparisons")
if __name__=="__main__":main()
