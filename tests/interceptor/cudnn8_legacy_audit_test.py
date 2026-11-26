#!/usr/bin/env python3
"""CPU-only SDK8 receipt/profile controls; fixture offsets are not vendor evidence."""

import copy
import gzip
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import cudnn8_legacy_audit as audit
import run_rnn_v8_validation as collector

CALL_NAMES = (
    'cudnnOpsInferVersionCheck', 'cudnnOpsTrainVersionCheck', 'cudnnCnnInferVersionCheck',
    'cudnnCnnTrainVersionCheck', 'cudnnAdvInferVersionCheck', 'cudnnAdvTrainVersionCheck',
    'cudnnCreateAlgorithmDescriptor', 'cudnnSetAlgorithmDescriptor', 'cudnnDestroyAlgorithmDescriptor',
    'cudnnSetRNNDescriptor_v6', 'cudnnSetRNNMatrixMathType', 'cudnnGetRNNMatrixMathType',
    'cudnnSetRNNBiasMode', 'cudnnGetRNNBiasMode', 'cudnnRNNSetClip', 'cudnnGetRNNWorkspaceSize',
    'cudnnGetRNNTrainingReserveSize', 'cudnnGetRNNParamsSize', 'cudnnGetRNNLinLayerMatrixParams',
    'cudnnGetRNNLinLayerBiasParams', 'cudnnSetRNNAlgorithmDescriptor',
    'cudnnGetRNNForwardInferenceAlgorithmMaxCount', 'cudnnRNNForwardTraining',
    'cudnnRNNBackwardData', 'cudnnRNNBackwardWeights')


def fixture(providers):
    model = dict(kind='model', layers=1, input=2, hidden=2, steps=3, batch=2, lengths=[3, 2],
                 packed_floats=10, seed=17, dropout=0, algorithm='standard', type='float',
                 bias='double', direction='uni', input_mode='linear', math='default', clip='none',
                 tensor_descriptors=18, central_difference_max_error=1e-12, weight_bytes=192,
                 workspace_bytes=16, reserve_bytes=32, dropout_bytes=64, algorithm_count=1)
    records = [dict(kind='sdk', header_version=8907, cached=False), model,
               dict(kind='providers', paths=list(map(str, providers)))]
    for name in ('weight', 'dweight'):
        for linear in range(8):
            for part, start, count, dims in (('matrix', linear * 24, 4, [1, 2, 2]),
                                             ('bias', linear * 24 + 16, 2, [1, 2, 1])):
                records.append(dict(kind='layout', name=name, linear=linear, part=part, offset=start,
                                    extent=4 * count, dimensions=dims,
                                    logical_offsets=list(range(start, start + count * 4, 4))))
    p, x, hx, cx, dy, dhy, dcy = audit.seeds()

    def pack(values):
        return struct.pack('<' + 'f' * len(values), *values)

    initial = dict(x=pack(x), hx=pack(hx), cx=pack(cx), dy=pack(dy), dhy=pack(dhy), dcy=pack(dcy),
                   weight=pack(p), dweight=pack([v * .25 + .125 for v in p]))

    def buffer(iteration, name, payload):
        records.append(dict(kind='buffer', iteration=iteration, name=name, payload_bytes=len(payload),
                            guard_bytes=256, opaque=False,
                            guarded_hex=(audit.GUARD + payload + audit.GUARD).hex()))

    def opaque(iteration):
        for name, size in (('workspace', 16), ('reserve', 32), ('dropout_state', 64)):
            records.append(dict(kind='buffer', iteration=iteration, name=name, payload_bytes=size,
                                guard_bytes=256, opaque=True, payload_fnv1a64='0' * 16,
                                prefix_hex=audit.GUARD.hex(), suffix_hex=audit.GUARD.hex()))

    for name, payload in initial.items():
        buffer(0, name, payload)
    opaque(0)
    values, gradients = audit.reference()
    for iteration in (1, 2):
        for name in sorted(audit.INPUT_NAMES):
            buffer(iteration, name, initial[name])
        for name, expected in values.items():
            buffer(iteration, name, pack(expected))
        starts = struct.unpack('<' + 'f' * 48, initial['dweight'])
        buffer(iteration, 'dweight', pack([v + iteration * g for v, g in zip(starts, gradients, strict=True)]))
        opaque(iteration)
        records.append(dict(kind='iteration', iteration=iteration, tensor_scalars_checked=36,
                            parameter_gradients_checked=48, guarded_allocations_checked=17,
                            unchanged_inputs_checked=7, complete_oracle_passed=True))
    calls = {name: 1 for name in CALL_NAMES}
    calls.update({name: 2 for name in CALL_NAMES[-3:]})
    calls.update(cudnnGetRNNLinLayerMatrixParams=16, cudnnGetRNNLinLayerBiasParams=16)
    result = dict(kind='legacy_sdk_result', status='passed', iterations=2, mapped_parameters=48,
                  normal_cleanup=True, live_resources=0, legacy_calls=calls)
    result.update({name: False for name in (
        'opaque_padding_mathematics_qualified', 'temporary_mathematics_qualified',
        'full_device_qualified', 'saturation_qualified', 'persistent_qualified',
        'priority_qualified', 'performance_qualified', 'full_goal_qualified')})
    records.append(result)
    return records


class LegacyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.sdk = self.root / 'sdk'
        (self.sdk / 'lib').mkdir(parents=True)
        self.providers = []
        for suffix in ('', '_adv_infer', '_adv_train', '_cnn_infer', '_cnn_train', '_ops_infer', '_ops_train'):
            path = self.sdk / 'lib' / ('libcudnn' + suffix + '.so.8')
            path.write_bytes(b'offline-only')
            self.providers.append(path)
        self.seals = {str(p): 'offline' for p in self.providers}
        self.records = fixture(self.providers)

    def check(self, records=None, remote=False, compressed=True):
        path = self.root / ('receipt.jsonl.gz' if compressed else 'receipt.jsonl')
        opener = gzip.open(path, 'wt') if compressed else path.open('w')
        with opener as stream:
            for record in self.records if records is None else records:
                stream.write(json.dumps(record) + '\n')
        return audit.audit_job(path, False, remote, self.sdk, self.seals)

    def test_full_plain_and_gzip(self):
        for compressed in (False, True):
            self.assertEqual(self.check(compressed=compressed)['complete_tensor_and_gradient_scalars'], 168)

    def test_frozen_contract_corruptions(self):
        for kind, key, bad in (('sdk', 'header_version', 91002), ('sdk', 'cached', True),
                               ('model', 'lengths', [3, 3]), ('layout', 'logical_offsets', [0]),
                               ('legacy_sdk_result', 'normal_cleanup', False),
                               ('legacy_sdk_result', 'full_device_qualified', True)):
            with self.subTest(kind=kind, key=key):
                records = copy.deepcopy(self.records)
                next(r for r in records if r['kind'] == kind)[key] = bad
                with self.assertRaises(ValueError):
                    self.check(records)

    def test_missing_duplicate_record(self):
        for records in (self.records[:-1], self.records + [self.records[-1]]):
            with self.assertRaises(ValueError):
                self.check(records)

    def test_exact_callback_names_and_counts(self):
        for change in ('name', 'count', 'boolean'):
            records = copy.deepcopy(self.records)
            calls = records[-1]['legacy_calls']
            if change == 'name':
                calls['invented_callback'] = calls.pop('cudnnOpsInferVersionCheck')
            else:
                calls['cudnnOpsInferVersionCheck'] = 0 if change == 'count' else True
            with self.assertRaises(ValueError):
                self.check(records)

    def test_numerical_corruption_with_intact_guards(self):
        for name in ('y', 'dweight'):
            records = copy.deepcopy(self.records)
            row = next(r for r in records if r['kind'] == 'buffer' and r['name'] == name and r['iteration'] == 2)
            raw = bytearray.fromhex(row['guarded_hex'])
            struct.pack_into('<f', raw, 256, 999.)
            row['guarded_hex'] = raw.hex()
            with self.assertRaises(ValueError):
                self.check(records)

    def test_guard_and_opaque_capacity(self):
        for opaque in (False, True):
            records = copy.deepcopy(self.records)
            row = next(r for r in records if r['kind'] == 'buffer' and r['opaque'] is opaque)
            row['payload_bytes'] += 1
            with self.assertRaises(ValueError):
                self.check(records)

    def test_provider_gates(self):
        mapped = '\n'.join('1-2 r-xp 0 00:00 1 ' + str(p) for p in self.providers)
        observed = collector.mapped_providers(mapped, self.sdk, self.seals)
        collector.validate_service_providers(observed, 'sdk8-legacy')
        for bad in (set(), set(sorted(observed)[:-1])):
            with self.assertRaises(ValueError):
                collector.validate_service_providers(bad, 'sdk8-legacy')
        with self.assertRaises(ValueError):
            collector.mapped_providers(mapped, self.sdk, {})
        wrong = self.root / 'wrong'
        wrong.mkdir()
        path = wrong / 'libcudnn.so.8'
        path.write_bytes(b'wrong-root')
        with self.assertRaises(ValueError):
            collector.mapped_providers('1-2 r-xp 0 00:00 1 ' + str(path), self.sdk, {str(path): 'offline'})
        records = copy.deepcopy(self.records)
        next(r for r in records if r['kind'] == 'providers')['paths'] = list(map(str, self.providers[:-1]))
        with self.assertRaises(ValueError):
            self.check(records)

    def test_exact_physical_inventory(self):
        lines = []
        for sequence in range(1, 8):
            op = 96 if sequence == 1 else 150
            lines.extend([f'Native admission audit: session=2, sequence={sequence}, accepted=true, operation={op}',
                          f'Native library issue audit: session=2, sequence={sequence}, status=0',
                          f'Native execution audit: session=2, issued={sequence}, retained=0, retired=1, status=0'])
        log = '\n'.join(lines)
        result = collector.physical_receipt(log, 'sdk8-legacy')
        self.assertEqual((result['rnn_computations'], result['dropout_initializations']), (6, 1))
        for bad in ('\n'.join(lines[:-1]), log + '\n' + lines[1], log.replace('retired=1', 'retired=0'),
                    log.replace('operation=96', 'operation=95'), '\n'.join(reversed(lines)),
                    log + '\nNative admission audit: session=2, sequence=8, accepted=true, operation=96'):
            with self.assertRaises(ValueError):
                collector.physical_receipt(bad, 'sdk8-legacy')
        with self.assertRaises(ValueError):
            collector.physical_receipt(log)

    def test_profile_selection(self):
        selected = collector.PROFILES['sdk8-legacy']
        self.assertEqual(selected['source'], 'execution_dnn_legacy_sdk_test.cc')
        self.assertEqual(selected['target'], 'glimmer_execution_dnn_legacy_sdk_test')
        self.assertEqual(selected['soname'], 'libcudnn.so.8')
        self.assertEqual(collector.PROFILES['sdk9-rnn-v8']['operations'], {112: 36})


if __name__ == '__main__':
    unittest.main()
