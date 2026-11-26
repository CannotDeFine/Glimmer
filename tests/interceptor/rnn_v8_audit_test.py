#!/usr/bin/env python3
"""CPU-only positive and corrupted raw-receipt controls; no SDK or GPU."""

import copy
import gzip
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import rnn_v8_audit as audit
import run_rnn_v8_validation as collector


def fixture():
    records = [dict(kind='sdk', runtime_version=91002, header_version=91002, cached=False)]
    for config in range(1, 7):
        dtype, layout = (0 if config % 2 else 1), (config - 1) // 2
        code, width = ('f', 4) if dtype == 0 else ('d', 8)
        lengths = [3, 2] if layout == 1 else [3, 3]
        base = config * 100000

        def emit(kind, **fields):
            records.append(dict(kind=kind, configuration=config, **fields))

        emit('configuration', type=dtype, layout=layout, aux_flags=0 if layout == 1 else 1)
        emit('capacities', weight_bytes=512, workspace_bytes=16, reserve_bytes=32)
        p, inputs = audit.seeds()
        payloads = {}
        for name, values in inputs.items():
            values = audit.sequence(values, layout, lengths) if name in ('x', 'dy') else values
            payloads[name] = struct.pack('<' + code * len(values), *values)
        payloads.update(lengths=struct.pack('<ii', *lengths), workspace=bytes(16), reserve=bytes(32))
        weight, gradient = bytearray(512), bytearray(512)
        for index, value in enumerate(p):
            struct.pack_into('<' + code, weight, index * width, value)
            struct.pack_into('<' + code, gradient, index * width, value * .25 + .125)
        payloads.update(weight=bytes(weight), dweight=bytes(gradient))
        for name in ('y', 'dx', 'hy', 'cy', 'dhx', 'dcx'):
            payloads[name] = bytes(len(payloads['x']) if name in ('y', 'dx') else 4 * width)
        addresses = {name: base + index * 2000 for index, name in enumerate(sorted(payloads))}

        def buffer(name, iteration, action, payload):
            emit('buffer', name=name, iteration=iteration, action=action, base=addresses[name],
                 guard_bytes=256, payload_bytes=len(payload),
                 guarded_hex=(audit.GUARD + payload + audit.GUARD).hex())

        for name, raw in payloads.items():
            buffer(name, 0, 'reset', bytes(len(raw)))
        for name in ('weight', 'dweight', 'x', 'dy', 'hx', 'cx', 'dhy', 'dcy', 'lengths'):
            buffer(name, 0, 'reset', payloads[name])

        def shape(q, descriptor, part, rank=3):
            def array(values):
                return dict(before=0x123456789abcdef0, after=0xfedcba9876543210,
                            values=values + [-8123] * (8 - len(values)))
            emit('shape', query=q, descriptor=descriptor, status=0, type=dtype, rank=rank,
                 dimensions=array([1, 2, 1 if part else 2] if rank else []),
                 strides=array([2, 1, 1] if part else [4, 2, 1]) if rank else array([]))

        for group, name in enumerate(('weight', 'dweight', None)):
            start = addresses[name] + 256 if name else 0
            for linear in range(8):
                q = (config - 1) * 32 + group * 8 + linear + 1
                emit('query', query=q, linear=linear, bytes=512, base=start, status=0,
                     matrix_descriptor=111, bias_descriptor=222, matrix_output=100, bias_output=200,
                     matrix_address=start + linear * 6 * width, bias_address=start + (linear * 6 + 4) * width)
                for part, descriptor in enumerate((111, 222)):
                    shape(q, descriptor, part)
                    for row in range(2):
                        for col in range(1 if part else 2):
                            emit('scalar_location', query=q, linear=linear, part=part, row=row, column=col,
                                 offset=(linear * 6 + (4 + row if part else 2 * row + col)) * width)
        start = addresses['weight'] + 256
        for index in range(8):
            q = (config - 1) * 32 + 25 + index
            m_desc, b_desc = (111 if index != 1 else 0), (222 if index != 0 else 0)
            m_out, b_out = (100 if m_desc else 0), (200 if b_desc else 0)
            m, b = (start if m_desc else 0), (start + 4 * width if b_desc else 0)
            if index == 2:
                b_desc = m_desc
            if index == 3:
                b_out, m = m_out, b
            pointer = start if index < 5 else start + 512 + 256 if index == 5 else 0
            if index >= 4:
                m = b = 0
            if index == 7:
                m_desc = b_desc = m_out = b_out = 0
            emit('query', query=q, linear=0 if index < 4 else 8, bytes=512 if index < 4 else 0,
                 base=pointer, status=0, matrix_descriptor=m_desc, bias_descriptor=b_desc,
                 matrix_output=m_out, bias_output=b_out, matrix_address=m, bias_address=b)
            if index == 2:
                shape(q, m_desc, 1)
            if 4 <= index < 7:
                shape(q, m_desc, 0, 0)
                shape(q, b_desc, 1, 0)
        values, derivatives = audit.reference(lengths)
        for iteration in (1, 2):
            current = dict(payloads)
            for name, wanted in values.items():
                if name in ('y', 'dx'):
                    wanted = audit.sequence(wanted, layout, lengths)
                buffer(name, iteration, 'reset', b'\xc3' * len(current[name]))
                current[name] = struct.pack('<' + code * len(wanted), *wanted)
            result = bytearray(gradient)
            for index, derivative in enumerate(derivatives):
                value = struct.unpack_from('<' + code, gradient, index * width)[0]
                struct.pack_into('<' + code, result, index * width, value + iteration * derivative)
            current['dweight'] = bytes(result)
            for name, raw in current.items():
                buffer(name, iteration, 'read', raw)
            for name in sorted(audit.OUTPUTS):
                buffer(name, iteration, 'read', current[name])
            emit('iteration', iteration=iteration, complete_oracle_passed=True, mapped_parameters=48,
                 cpu_finite_difference_error=0.)
        for name in sorted(payloads):
            emit('device_release', name=name, status=0)
    records.extend(dict(kind='resource_destroy', status=0) for _ in range(81))
    result = dict(kind='rnn_v8_result', status='passed', normal_cleanup=True, parameter_queries=192,
                  numerical_cases=12, forward_calls=12, backward_data_calls=12, backward_weight_calls=12,
                  resources_destroyed=81, device_releases=102, live_libraries=0,
                  logical_parameter_gradient_elements=576, logical_output_and_gradient_elements=1040)
    result.update({key: False for key in ('opaque_padding_gradient_qualified', 'temporary_mathematics_qualified',
                                        'all_rnn_models_qualified', 'full_device_qualified', 'saturation_qualified',
                                        'persistent_qualified', 'priority_qualified', 'performance_qualified',
                                        'full_goal_qualified')})
    records.append(result)
    return records


class ReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = fixture()

    def test_complete(self):
        self.assertEqual(audit.audit_records(self.records, False)['logical_scalars'], 1040)

    def test_lossless_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'application.out.gz'
            with gzip.open(path, 'xt', compresslevel=1) as stream:
                for record in self.records:
                    stream.write(json.dumps(record) + '\n')
            self.assertEqual(audit.audit(path, False)['logical_scalars'], 1040)
            truncated = path.read_bytes()[:-8]
            path.write_bytes(truncated)
            with self.assertRaises(EOFError):
                audit.audit(path, False)

    def test_partial_never_fabricates_full_receipt(self):
        records = [r for r in self.records if r['kind'] != 'rnn_v8_result' and
                   r.get('configuration', 0) <= 3]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'partial.gz'
            with gzip.open(path, 'wt') as stream:
                for r in records:
                    stream.write(json.dumps(r) + '\n')
                stream.write('{"kind":"buffer","guarded_hex":"')
            result = audit.audit_partial(path, False)
            self.assertEqual(result['configurations'], [1, 2, 3])
            self.assertEqual(result['logical_scalars'], 520)
            self.assertFalse(result['full_route_qualified'])
            self.assertFalse(result['normal_route_cleanup'])
            with self.assertRaises((ValueError, json.JSONDecodeError)):
                audit.audit(path, False)

    def test_corruption(self):
        for kind, field, value in [('rnn_v8_result', 'parameter_queries', 191),
                                   ('sdk', 'cached', True), ('configuration', 'layout', 2),
                                   ('configuration', 'aux_flags', 0),
                                   ('scalar_location', 'offset', 512), ('shape', 'rank', 2),
                                   ('query', 'status', 1), ('resource_destroy', 'status', 1),
                                   ('buffer', 'guarded_hex', '00'),
                                   ('rnn_v8_result', 'full_device_qualified', True)]:
            with self.subTest(kind=kind, field=field):
                records = copy.deepcopy(self.records)
                next(r for r in records if r['kind'] == kind)[field] = value
                with self.assertRaises(ValueError):
                    audit.audit_records(records, False)

    def test_numerical_corruption_preserves_guards(self):
        records = copy.deepcopy(self.records)
        for r in records:
            if r['kind'] == 'buffer' and r['name'] == 'y' and r['action'] == 'read':
                raw = bytearray.fromhex(r['guarded_hex'])
                raw[256:264] = b'\x7f' * 8
                r['guarded_hex'] = raw.hex()
        with self.assertRaises(ValueError):
            audit.audit_records(records, False)

    def test_missing_duplicate(self):
        for records in (self.records[:-1], self.records + [self.records[-1]]):
            with self.assertRaises(ValueError):
                audit.audit_records(records, False)

    def test_reference_control(self):
        result = audit.forward([0.] * 48, [0.] * 12, [0.] * 4, [0.] * 4, [3, 2])
        self.assertTrue(all(v == 0 for values in result.values() for v in values))
        values, derivative = audit.reference([3, 2])
        self.assertEqual(values['dx'][-2:], [0., 0.])
        for gate in range(4):
            self.assertAlmostEqual(derivative[6 * gate + 4], derivative[6 * (gate + 4) + 4], places=8)

    def test_foreign_gpu_never_signalled(self):
        with patch.object(collector, 'query', return_value='GPU-test, 99, foreign'), \
                patch.object(collector.os, 'kill') as kill:
            with self.assertRaises(ValueError):
                collector.guard({})
            kill.assert_not_called()

    def test_physical_join(self):
        lines = []
        for sequence in range(1, 37):
            lines.extend([f'Native admission audit: session=4, sequence={sequence}, accepted=true, operation=112',
                          f'Native library issue audit: session=4, sequence={sequence}, status=0',
                          f'Native execution audit: session=4, issued={sequence}, retained=0, retired=1, status=0'])
        log = '\n'.join(lines)
        self.assertEqual(collector.physical_receipt(log)['rnn_computations'], 36)
        for bad in ('\n'.join(lines[:-1]), log + '\n' + lines[1], log.replace('retired=1', 'retired=0'),
                    '\n'.join(reversed(lines)), log.replace('operation=112', 'operation=150')):
            with self.assertRaises(ValueError):
                collector.physical_receipt(bad)

    def test_actual_memory_limits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, value in {'memory.max': str(collector.MEMORY_MAX), 'memory.swap.max': '0',
                                'memory.oom.group': '1', 'memory.current': '4096',
                                'memory.events': 'oom 0\noom_kill 0\n'}.items():
                (root / name).write_text(value)
            with patch.object(collector, 'Path', return_value=root):
                # An absolute suffix would discard the mocked root; use an empty
                # group only for this isolated filesystem fixture.
                self.assertEqual(collector.memory_receipt('')['memory.swap.max'], '0')
                for name, bad in (('memory.max', 'max'), ('memory.swap.max', '4096'),
                                  ('memory.oom.group', '0')):
                    old = (root / name).read_text()
                    (root / name).write_text(bad)
                    with self.assertRaises(ValueError):
                        collector.memory_receipt('')
                    (root / name).write_text(old)

    def test_idle_window_is_bounded(self):
        class Clock:
            value = 0.

            def now(self):
                return self.value

            def sleep(self, seconds):
                self.value += seconds

        for busy, achieved in ((False, True), (True, False)):
            timer = Clock()
            with tempfile.TemporaryDirectory() as temporary, \
                    patch.object(collector, 'query', return_value='foreign' if busy else ''):
                result = collector.wait_for_idle(Path(temporary), timer.now, timer.sleep)
                self.assertIs(result['achieved'], achieved)
                self.assertLessEqual(timer.value, 600)
                self.assertEqual(result.get('sampled_idle_seconds', 0), 60 if achieved else 0)


if __name__ == '__main__':
    unittest.main()
