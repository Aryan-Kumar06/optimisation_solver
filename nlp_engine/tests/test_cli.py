"""End-to-end input validation and JSON contract. Standard library only."""
import json
import pathlib
import subprocess
import sys
import tempfile

binary=str(pathlib.Path(sys.argv[1]).resolve())
valid="nlp 1 variables 1 -inf inf 0 nodes 4 var 0 const 3 sub 0 1 square 2 objective 3 constraints 0"
with tempfile.TemporaryDirectory() as d:
    path=pathlib.Path(d)/"problem.nlp"
    def run(text=valid,*args):
        path.write_text(text)
        return subprocess.run([binary,"solve-nlp",str(path),*args],capture_output=True,text=True,timeout=20)
    r=run();assert r.returncode==0,r.stderr
    result=json.loads(r.stdout);assert result["status"]=="FirstOrderStationary"
    assert abs(result["primal"][0]-3)<1e-5
    r=run(valid,"--iterations","0");assert r.returncode==2
    assert json.loads(r.stdout)["status"]=="IterationLimit"
    for bad in ("", valid+" junk", valid.replace("sub 0 1","sub 0 99"),valid.replace("var 0","var 1"),
                valid.replace("const 3","const nan"),valid.replace("nlp 1","nlp 2"),valid.replace("square 2","exec 2")):
        r=run(bad);assert r.returncode==1,(bad,r.stdout)
    for args in (("--bad","1"),("--tolerance","1junk"),("--iterations","1.5"),("--json",str(path)),("--json",str(path.parent)+"/./"+path.name)):
        assert run(valid,*args).returncode==1,args
    for args in (("--tolerance","nan"),("--iterations","-1"),("--time-limit","inf")):
        r=run(valid,*args);assert r.returncode==1,args
    output=pathlib.Path(d)/"result.json"
    r=run(valid,"--json",str(output));assert r.returncode==0
    assert json.loads(output.read_text())==json.loads(r.stdout)
    # Ordinary solve command and explicit NLP command share the same pipeline.
    for command in ("solve", "solve-nlp"):
        r=subprocess.run([binary,command,"--tolerance","1e-6",str(path),"--solver","nlp"],capture_output=True,text=True)
        assert r.returncode==0,r.stderr
        record=json.loads(r.stdout)
        assert record["engine"]==record["executed_engine"]=="nlp_sqp"
        assert record["problem_class"]=="NLP" and record["schema"]=="optimsolver.nlp.v1"
        assert record["presolve_applied"] is False and record["postsolve_applied"] is False
        assert record["original_variables"]==1 and record["original_constraints"]==0
        assert len(record["instance_sha256"])==64
        for flags in (("--help",),(str(path),"-h")):
            help_run=subprocess.run([binary,command,*flags],capture_output=True,text=True)
            assert help_run.returncode==0 and "first-order" in help_run.stdout
    for flags in (("--solver","qp"),("--threads","1"),("--dump-model",str(output))):
        assert run(valid,*flags).returncode==1,flags
    solution=pathlib.Path(d)/"solution.txt"
    assert run(valid,"--output",str(solution)).returncode==0
    assert "FirstOrderStationary" in solution.read_text() and "x[0]" in solution.read_text()
    assert run(valid,"--output",str(output),"--json",str(output)).returncode==1
    assert run(valid,"--output",str(path)).returncode==1
    missing=subprocess.run([binary,"solve-nlp"],capture_output=True,text=True)
    assert missing.returncode==1 and "Missing required model path" in missing.stderr
    root=subprocess.run([binary,"--help"],capture_output=True,text=True)
    assert "solve-nlp" in root.stdout
    # Interactive open, inspect, solve, then switch back to an affine model.
    mps=pathlib.Path(d)/"simple.mps"
    mps.write_text("NAME TEST\nROWS\n N OBJ\n G C1\nCOLUMNS\n X OBJ 1 C1 1\nRHS\n RHS1 C1 1\nENDATA\n")
    interaction=f"1\n{path}\n3\n\n1\n\n2\n{mps}\n1\n\n6\n"
    r=subprocess.run([binary],input=interaction,capture_output=True,text=True,timeout=30)
    assert r.returncode==0 and not r.stderr,r.stderr
    assert "FirstOrderStationary" in r.stdout and "OPTIMAL" in r.stdout
    # A failed NLP load retains the previous model; it never installs partial data.
    broken=pathlib.Path(d)/"broken.nlp";broken.write_text("nlp broken")
    interaction=f"1\n{path}\n2\n{broken}\n2\n1\n\n6\n"
    r=subprocess.run([binary],input=interaction,capture_output=True,text=True,timeout=30)
    assert "Could not load model" in r.stdout and "FirstOrderStationary" in r.stdout
print("NLP CLI tests passed")
