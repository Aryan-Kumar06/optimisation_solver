#!/usr/bin/env python3
"""Exercise structured backend provenance through the real CLI, without a GPU."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

binary = sys.argv[1]
root = Path(__file__).resolve().parents[2]
model = root / 'tests/cli/simple_lp.mps'
with tempfile.TemporaryDirectory() as tmp:
    output = Path(tmp) / 'result.json'
    def solve(engine, backend, device=0):
        output.unlink(missing_ok=True)
        process = subprocess.run([binary, 'solve', str(model), '--solver', engine,
            '--backend', backend, '--cuda-device', str(device), '--json', str(output)],
            capture_output=True, text=True, timeout=30)
        return process, json.loads(output.read_text())['compute_backend']

    for requested in ('auto', 'cpu'):
        process, meta = solve('pdlp', requested)
        assert process.returncode == 0, process.stderr
        assert meta['requested'] == requested and meta['executed'] == 'cpu', meta
        assert meta['requested_device'] == 0 and meta['executed_device'] is None, meta
        assert meta['reason'], meta

    process, meta = solve('pdlp', 'cuda', 1024)
    assert process.returncode != 0
    assert meta['requested'] == 'cuda' and meta['requested_device'] == 1024, meta
    assert meta['executed'] is None and meta['executed_device'] is None, meta
    assert 'CUDA' in meta['reason'], meta

    process, meta = solve('dual_simplex', 'cuda')
    assert process.returncode == 0, process.stderr
    assert meta['executed'] == 'cpu' and meta['executed_device'] is None, meta
    assert 'no CUDA backend' in meta['reason'], meta

    nlp = Path(tmp) / 'model.nlp'
    nlp.write_text('nlp 1\nvariables 1\n-inf inf 0\nnodes 2\nvar 0\nsquare 0\nobjective 1\nconstraints 0\n')
    process = subprocess.run([binary, 'solve', str(nlp), '--backend', 'cuda', '--cuda-device', '2'],
        capture_output=True, text=True, timeout=30)
    assert process.returncode == 0, process.stderr
    meta = json.loads(process.stdout)['compute_backend']
    assert meta['requested'] == 'cuda' and meta['requested_device'] == 2, meta
    assert meta['executed'] == 'cpu' and meta['executed_device'] is None, meta
    assert 'no CUDA backend' in meta['reason'], meta
print('Backend JSON provenance tests passed')
