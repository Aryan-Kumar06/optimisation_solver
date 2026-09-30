#!/usr/bin/env python3
"""Offline tests of suite selection, integrity, and fail-closed accounting."""
import bz2
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import bench
import zipfile
from fetch_suites import checked, decode, source
from run_suites import assess, selection


def point(value=1, status='optimal', verdict='optimal_verified'):
    return dict(termination=dict(status=status), check=dict(verdict=verdict, objective_recomputed=value),
                engine_use=dict(counts_for_requested=True))


class Suites(unittest.TestCase):
    def test_frozen_counts_and_hashes(self):
        for suite, count in [('netlib',8), ('miplib',25), ('qp',14), ('mittelmann',3)]:
            _, entries = selection(suite)
            self.assertEqual(len(entries), count)
            self.assertEqual(len({e['name'] for e in entries}), count)
            self.assertTrue(all(e['sha256'] and len(e['sha256']) == 64 for e in entries))
        self.assertEqual(len(selection('qp', True)[1]),138)

    def test_hash_rejects_corruption(self):
        with self.assertRaises(ValueError):
            checked(b'bad', hashlib.sha256(b'good').hexdigest())

    def test_cached_corruption_not_refetched(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'cache'
            path.write_bytes(b'bad')
            with self.assertRaises(ValueError):
                source(path, 'invalid://must-not-fetch', '0'*64, False)

    def test_offline_missing_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                source(Path(tmp)/'absent', '', '', True)

    def test_decoders_preserve_bytes_and_verify(self):
        data = b'NAME model\r\nENDATA\r\n'
        entry = dict(sha256=hashlib.sha256(data).hexdigest())
        self.assertEqual(decode(entry, bz2.compress(data)), data)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf,'w') as z:
            z.writestr('MODEL.QPS', data)
        self.assertEqual(decode(dict(entry,member='MODEL.QPS'),buf.getvalue()),data)
        with self.assertRaises(KeyError):
            decode(dict(entry,member='missing'),buf.getvalue())
        with self.assertRaises(ValueError):
            decode(dict(entry,sha256='0'*64),bz2.compress(data))
        with self.assertRaises(zipfile.BadZipFile):
            decode(dict(entry,member='MODEL.QPS'),b'not a zip')

    def test_nested_compression_requires_verified_decoder(self):
        data = b'compressed emps payload'
        entry = dict(encoding='bz2+emps', intermediate_sha256=hashlib.sha256(data).hexdigest(),
                     sha256=hashlib.sha256(b'expected decoded mps').hexdigest())
        with self.assertRaisesRegex(ValueError, 'requires.*decoder'):
            decode(entry, bz2.compress(data))
        entry['intermediate_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            decode(entry, bz2.compress(data))

    def test_suite_can_pin_scipy_with_native_highs_installed(self):
        with patch.object(bench.shutil, 'which', return_value='/native/highs'), \
             patch.object(bench, '_invoke', return_value={}) as invoke:
            bench.run_highs('runner', 'model.mps', 'highs-ds', 5, '/tmp', backend='scipy')
            args = invoke.call_args.args
            self.assertEqual(args[1], 'highs-scipy:highs-ds')
            self.assertIn(bench.HIGHS_SCIPY_ADAPTER, args[3])
            self.assertNotIn('/native/highs', args[3])

    def test_comparison_accounting(self):
        entry = dict(parse_check=dict(status='agree'), runs=[point(),point()])
        self.assertEqual(assess(entry),'objective_agrees')
        entry['runs'][0] = point(2)
        self.assertEqual(assess(entry),'objective_mismatch')
        for verdict in ('no_point','not_applicable'):
            entry['runs'][0] = point(verdict=verdict)
            self.assertEqual(assess(entry),'unverified')
        for verdict in ('infeasible_point','nonfinite','malformed'):
            entry['runs'][0] = point(verdict=verdict)
            self.assertEqual(assess(entry),'rejected_point')
        entry['runs'] = [point(),point(status='limit_reached')]
        self.assertEqual(assess(entry),'reference_not_optimal')
        entry['runs'] = [point(status='limit_reached'),point()]
        self.assertEqual(assess(entry),'feasible_not_optimal')
        entry['runs'] = [point(),point()]
        entry['runs'][0]['engine_use']['counts_for_requested'] = False
        self.assertEqual(assess(entry),'requested_engine_not_run')
        entry['parse_check']['status'] = 'disagree'
        self.assertEqual(assess(entry),'parse_failure')
        self.assertEqual(assess(dict(parse_check=dict(status='agree'),runs=[])), 'missing_result')


if __name__ == '__main__':
    unittest.main()
