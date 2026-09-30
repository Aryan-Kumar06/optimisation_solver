#!/usr/bin/env python3
"""Fetch pinned public benchmark data. Never fetches during CTest."""
import argparse
import bz2
import hashlib
import io
import json
from pathlib import Path
import tempfile
import shutil
import subprocess
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent
MAX_BYTES = 512 * 1024 * 1024


def checked(data, digest):
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError('SHA256 mismatch; refusing changed/corrupt benchmark data')
    return data


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        tmp = Path(handle.name)
        handle.write(data)
    try:
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def source(path, url, digest, offline):
    if path.exists():
        return checked(path.read_bytes(), digest)
    if offline:
        raise FileNotFoundError(f'Not cached: {path}')
    # macOS system curl uses its native certificate store. Keep TLS verification
    # enabled in both paths; Python installations may lack a system CA bundle.
    if shutil.which('curl'):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / 'download'
            subprocess.run(['curl', '--fail', '--silent', '--show-error', '--location',
                            '--proto', '=https', '--max-time', '60', '--max-filesize',
                            str(MAX_BYTES), '--output', str(dest), url], check=True,
                           capture_output=True, timeout=65)
            data = dest.read_bytes()
    else:
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError('Archive exceeds size limit')
    checked(data, digest)
    atomic_write(path, data)
    return data


def decode(entry, data, emps_binary=None):
    if 'member' in entry:
        # Read a single pinned member, never extract paths supplied by an archive.
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            if archive.getinfo(entry['member']).file_size > MAX_BYTES:
                raise ValueError('Instance exceeds size limit')
            result = archive.read(entry['member'])
    else:
        with bz2.BZ2File(io.BytesIO(data)) as stream:
            result = stream.read(MAX_BYTES + 1)
    if len(result) > MAX_BYTES:
        raise ValueError('Instance exceeds size limit')
    if entry.get('encoding') == 'bz2+emps':
        checked(result, entry['intermediate_sha256'])
        if emps_binary is None:
            raise ValueError('This instance requires the pinned Netlib emps decoder')
        with tempfile.TemporaryDirectory() as tmp:
            inp, out = Path(tmp)/'input', Path(tmp)/'output'
            inp.write_bytes(result)
            with out.open('wb') as handle:
                subprocess.run([str(emps_binary), str(inp)], stdout=handle,
                               stderr=subprocess.PIPE, check=True, timeout=60)
            if out.stat().st_size > MAX_BYTES:
                raise ValueError('Decoded instance exceeds size limit')
            result = out.read_bytes()
    return checked(result, entry['sha256'])


def fetch(suite, full=False, offline=False, root=None):
    spec = json.loads((HERE / 'suites' / (suite + '.json')).read_text())
    root = Path(root) if root else HERE / 'instances' / suite
    archives = {a['file']: a for a in spec.get('archives', [])}
    selected = spec['instances']
    if not full and 'smoke' in spec:
        selected = [i for i in selected if i['name'] in spec['smoke']]
    failures = []
    emps_binary = None
    for entry in selected:
        try:
            dest = root / 'mps' / (entry['name'] + '.mps')
            if dest.exists():
                checked(dest.read_bytes(), entry['sha256'])
                continue
            a = archives[entry['archive']] if 'archive' in entry else dict(
                url=entry['url'], file=entry['url'].rsplit('/', 1)[-1], sha256=entry['compressed_sha256'])
            blob = source(root / 'raw' / a['file'], a['url'], a['sha256'], offline)
            if entry.get('encoding') == 'bz2+emps' and emps_binary is None:
                decoder = spec['decoder']
                decoder_path = root / 'tools' / decoder['file']
                source(decoder_path, decoder['url'], decoder['sha256'], offline)
                compiler = shutil.which('cc')
                if compiler is None:
                    raise RuntimeError('A C compiler is required for the Netlib emps decoder')
                emps_binary = root / 'tools' / 'emps'
                subprocess.run([compiler, '-w', '-o', str(emps_binary), str(decoder_path)],
                               check=True, capture_output=True, timeout=60)
            atomic_write(dest, decode(entry, blob, emps_binary))
        except Exception as error:
            failures.append(dict(name=entry['name'], error=str(error)))
    return dict(suite=suite, selected=len(selected), ready=len(selected)-len(failures), failures=failures)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', required=True, choices=['qp', 'mittelmann'])
    p.add_argument('--full', action='store_true', help='All 138 QPs; Mittelmann remains a three-instance subset')
    p.add_argument('--offline', action='store_true')
    args = p.parse_args()
    report = fetch(args.suite, args.full, args.offline)
    print(json.dumps(report, indent=2))
    return int(bool(report['failures']))


if __name__ == '__main__':
    raise SystemExit(main())
