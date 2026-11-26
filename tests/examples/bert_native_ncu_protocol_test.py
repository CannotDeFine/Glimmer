"""CPU-only fixed native profiling protocol and counter rejection controls."""

import csv
from contextlib import ExitStack, nullcontext
from copy import deepcopy
import io
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import bert_native_ncu_protocol as native
import run_bert_native_ncu_capture as capture


def csv_data(values=((80, 320, 128, 1, 2), (80, 24, 128, 0, 0))):
    stream = io.StringIO()
    writer = csv.writer(stream, quoting=csv.QUOTE_ALL)
    writer.writerow(['ID', 'Process ID', 'Kernel Name', 'Context', 'Stream', *native.METRICS])
    writer.writerow([''] * 5 + ['SM', '', '', 'cycle', 'inst'])
    for index, row in enumerate(values):
        writer.writerow([index, 101 + index, 'volta_sgemm_128x128_nn', 1, 7, *row])
    return stream.getvalue()


def replay_log():
    lines = []
    for serial, pids in enumerate(((201, 202), (101, 102)), 1):
        lines.extend(f'==PROF== Connected to process {pid} (/usr/bin/python3.10)' for pid in pids)
        lines.extend(f'==PROF== Profiling "volta_sgemm_128x128_nn": Application replay pass {serial}'
                     for _ in pids)
        lines.extend(f'==PROF== Disconnected from process {pid}' for pid in pids)
    return '\n'.join(lines)


def target_csv(pid=101, kernel='volta_sgemm_128x32_nn', sm=80, grid=576, block=256,
               cycles=10, ffma=20):
    stream = io.StringIO()
    writer = csv.writer(stream, quoting=csv.QUOTE_ALL)
    writer.writerow(['ID', 'Process ID', 'Kernel Name', 'Context', 'Stream', *native.METRICS])
    writer.writerow([''] * 5 + ['SM', '', '', 'cycle', 'inst'])
    writer.writerow([0, pid, kernel, 1, 7, sm, grid, block, cycles, ffma])
    return stream.getvalue()


def target_shim_receipt(pid=101, tid=102, provider='/fixed/libcublas.so'):
    lines = [f'GLIMMER_NATIVE_SGEMM_RESOLVE path={provider}']
    for ordinal in range(1, 439):
        offset = (ordinal - 1) % 146
        descriptor = native.TARGET_DESCRIPTOR if offset in range(2, 134, 11) else (0, 0, 1, 1, 1, 1, 1, 1)
        ta, tb, m, n, k, lda, ldb, ldc = descriptor
        selected = int(ordinal == native.TARGET_ORDINAL)
        lines.append(f'GLIMMER_NATIVE_SGEMM_BEGIN pid={pid} tid={tid} ordinal={ordinal} '
                     f'ta={ta} tb={tb} m={m} n={n} k={k} lda={lda} ldb={ldb} ldc={ldc} '
                     f'selected={selected}')
        if selected:
            lines.append(f'GLIMMER_NATIVE_SGEMM_RANGE pid={pid} ordinal=149 domain=1 '
                         f'push=-1 pop=-1 name=GlimmerNativeSgemm149-pid-{pid}')
        lines.append(f'GLIMMER_NATIVE_SGEMM_END pid={pid} ordinal={ordinal} status=0')
    return '\n'.join(lines)


class NativeProtocolTests(unittest.TestCase):
    def test_commands_keep_complete_original_native_recipe(self):
        manifest = dict(python=str(native.PYTHON), assets='/assets',
                        startup_reference='/reference', export='/export')
        commands = native.application_commands(manifest, Path('/pass/native'))
        for role, command in commands.items():
            self.assertEqual(command[:2], [str(native.PYTHON), '-B'])
            self.assertNotIn('--profile-cuda', command)
            self.assertIn('--ready-file', command)
            self.assertEqual(command[command.index('--iterations') + 1], '2')
        self.assertIn('--snapshot', commands['low'])
        self.assertIn('--reference', commands['low'])
        with self.assertRaises(ValueError):
            native.application_commands(dict(manifest, python='/bin/sh'), Path('/pass/native'))
        target = native.target_application_commands(manifest, Path('/pass/native'))
        self.assertTrue(all(command.count('--profile-cuda') == 1 for command in target.values()))
        self.assertTrue(all('--profile-cuda' not in command for command in commands.values()))

    def test_target_command_binding_keeps_logical_and_root_recorded_argv_distinct(self):
        manifest = dict(python=str(native.PYTHON), assets='/assets',
                        startup_reference='/reference', export='/export')
        directory = Path('/pass/native')
        logical = native.application_commands(manifest, directory)
        target = native.target_application_commands(manifest, directory)
        identities = {'low': dict(pid=101, start=42), 'high': dict(pid=102, start=43)}
        job = dict(mode='native', application_commands=logical, application_identities=identities)
        records = {role: dict(role=role, identity=identity, command=target[role], released=True, samples=1)
                   for role, identity in identities.items()}
        binding = native.target_command_binding(manifest, directory, job, records)
        self.assertEqual(binding['logical_release_commands'], logical)
        self.assertEqual(binding['root_recorded_target_commands'], target)
        self.assertEqual(binding['identities'], identities)
        self.assertIn('require_authenticated', binding['status'])
        for changed in (dict(records, low=dict(records['low'], command=logical['low'])),
                        dict(records, low=dict(records['low'], identity=identities['high'])),
                        dict(records, low=dict(records['low'], released=False)),
                        dict(records, low=dict(records['low'], samples=0)),
                        {'low': records['low']}):
            with self.subTest(records=changed), self.assertRaises(ValueError):
                native.target_command_binding(manifest, directory, job, changed)
        with self.assertRaises(ValueError):
            native.target_command_binding(manifest, directory, dict(job, application_commands=target), records)
        with self.assertRaises(ValueError):
            native.target_command_binding(manifest, directory, dict(job, mode='equal'), records)

    def test_profiled_onnx_adapter_preserves_original_oracle_and_raw_result(self):
        spec = importlib.util.spec_from_file_location('test_private_onnx_oracle', native.ONNX_VALIDATOR)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        source_sha = native.bridge.seal(native.ONNX_VALIDATOR)
        self.assertTrue(callable(native.instrumented_onnx_validator(source_sha, event_start=False)))
        with self.assertRaisesRegex(ValueError, 'bytes changed'):
            native.instrumented_onnx_validator('0' * 64, event_start=False)
        original_exact = module._exact
        control = json.loads(module.NATIVE_CONTROL.read_text())
        result = json.loads(json.dumps(control))
        result['instrumented'] = True
        validate = native._instrumented_onnx_adapter(module, source_sha, False)
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(module, '_load_contract', return_value=(control, 'a' * 64, 'b' * 64)):
            root = Path(temporary)
            output = root / 'result.json'

            def write(value):
                output.write_text(json.dumps(value, allow_nan=False))

            write(result)
            receipt = validate(root, 'GPU-' + control['environment']['device_uuid'], root,
                               expected_warmup=3, expected_iterations=20)
            self.assertEqual(receipt['result_sha256'], native.bridge.seal(output))
            self.assertIs(receipt['application_instrumented'], True)
            self.assertIs(receipt['profiler_diagnostic_only'], True)
            self.assertIs(receipt['performance_qualified'], False)
            self.assertIs(module._exact, original_exact)
            changes = (dict(instrumented=False), dict(instrumented=1), dict(instrumented='true'),
                       dict(performance_acceptance=True), dict(providers=['CPUExecutionProvider']),
                       dict(validation=dict(result['validation'], checked_requests=19)), dict(unexpected='x'))
            for change in changes:
                with self.subTest(change=change):
                    write(dict(result, **change))
                    with self.assertRaises(ValueError):
                        validate(root, 'GPU-' + control['environment']['device_uuid'], root,
                                 expected_warmup=3, expected_iterations=20)
                    self.assertIs(module._exact, original_exact)
            write(result)
            with self.assertRaisesRegex(ValueError, 'no fresh reference'):
                validate(root, 'GPU-' + control['environment']['device_uuid'], root, reference=root)

    def test_profiled_onnx_adapter_requires_one_exact_flag_check(self):
        module = Mock()
        module._exact = Mock()
        module.validate_onnx_result.return_value = dict(diagnostic_only=True, performance_qualified=False)
        validate = native._instrumented_onnx_adapter(module, 'a' * 64, False)
        with self.assertRaisesRegex(ValueError, 'complete instrumented'):
            validate('/out', 'gpu', '/export')

        def duplicate(*args, **kwargs):
            module._exact(True, False, 'result instrumented')
            module._exact(True, False, 'result instrumented')
            return dict(diagnostic_only=True, performance_qualified=False)

        module.validate_onnx_result.side_effect = duplicate
        with self.assertRaisesRegex(ValueError, 'one explicitly instrumented'):
            validate('/out', 'gpu', '/export')

        def conflicting_receipt(*args, **kwargs):
            module._exact(True, False, 'result instrumented')
            return dict(diagnostic_only=True, performance_qualified=False, application_instrumented=False)

        module.validate_onnx_result.side_effect = conflicting_receipt
        with self.assertRaisesRegex(ValueError, 'complete instrumented'):
            validate('/out', 'gpu', '/export')

    def test_profiled_onnx_adapter_rejects_overlapping_calls_and_recovers(self):
        module = Mock()
        module._exact = Mock()
        entered, release = threading.Event(), threading.Event()
        results = []

        def blocking(*args, **kwargs):
            module._exact(True, False, 'result instrumented')
            entered.set()
            if not release.wait(5):
                raise TimeoutError('test failed to release inference validation')
            return dict(diagnostic_only=True, performance_qualified=False)

        module.validate_onnx_result.side_effect = blocking
        validate = native._instrumented_onnx_adapter(module, 'a' * 64, False)

        def worker():
            try:
                results.append(validate('/out', 'gpu', '/export'))
            except BaseException as error:
                results.append(error)

        thread = threading.Thread(target=worker)
        thread.start()
        try:
            self.assertTrue(entered.wait(5))
            with self.assertRaisesRegex(ValueError, 'already in use'):
                validate('/out', 'gpu', '/export')
        finally:
            release.set()
            thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(results), 1)
        self.assertIs(results[0]['application_instrumented'], True)
        self.assertIs(validate('/out', 'gpu', '/export')['application_instrumented'], True)

    def test_release_is_exact_and_single_use_without_pipe_transfer(self):
        connection = Mock()
        process = native.NativeProcess(connection, 'low', {'pid': 101, 'start': 42}, ['fixed'], {'X': '1'})
        valid = json.dumps(dict(command=['fixed'], environment={'X': '1'})) + '\n'
        for value in (valid.rstrip(), valid + valid, valid.replace('fixed', 'arbitrary'),
                      '{"command":[],"command":[],"environment":{}}\n'):
            with self.assertRaises(ValueError):
                process.stdin.write(value)
        self.assertEqual(process.stdin.write(valid), len(valid))
        process.stdin.flush()
        connection.call.assert_called_once_with('release', role='low')
        with self.assertRaises(ValueError):
            process.stdin.write(valid)
        process.stdin.close()
        with self.assertRaises(ValueError):
            process.stdin.flush()

    def test_status_requires_actual_matching_raw_wait(self):
        connection = Mock()
        connection.call.return_value = {'receipt': None}
        process = native.NativeProcess(connection, 'low', {'pid': 101, 'start': 42}, ['fixed'], {})
        self.assertIsNone(process.poll())
        valid = native.bridge.wait_receipt(101, 42, 0)
        for field, value in (('pid', 102), ('start', 43), ('returncode', 1), ('exit_code_observed', False)):
            connection.call.return_value = {'receipt': dict(valid, **{field: value})}
            with self.assertRaises(ValueError):
                process.poll()
        connection.call.return_value = {'receipt': valid}
        self.assertEqual(process.poll(), 0)

    def test_logs_require_exact_fresh_owned_role_inode(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / 'native').mkdir()
            path = directory / 'native/low.log'
            with path.open('x') as stream:
                native.validate_log(stream.fileno(), directory, 'low', os.getuid())
                with self.assertRaises(ValueError):
                    native.validate_log(stream.fileno(), directory, 'low', os.getuid() + 1)
                stream.write('used')
                stream.flush()
                with self.assertRaises(ValueError):
                    native.validate_log(stream.fileno(), directory, 'low', os.getuid())

    def test_full_inventory_keeps_small_zero_minimum_launches(self):
        rows = native.metric_inventory(csv_data(), {101: 'low', 102: 'high'})
        self.assertEqual(len(rows), 2)
        self.assertEqual([row['all_80_useful'] for row in rows], [True, False])
        rows = native.metric_inventory(csv_data(((80, 320, 128, 0, 1), (80, 320, 128, 1, 0))),
                                       {101: 'low', 102: 'high'})
        self.assertFalse(any(row['all_80_useful'] for row in rows))

    def test_metrics_reject_partial_domain_wrong_role_nan_and_duplicate(self):
        valid = csv_data()
        for value in (valid.replace('"80"', '"79"', 1), valid.replace('"cycle"', '"%"'),
                      valid.replace('"101"', '"999"'), valid.replace('"320"', '"nan"'),
                      valid.replace('volta_sgemm_', 'glimmer_owned_'),
                      valid + valid.splitlines()[2] + '\n',
                      '\n'.join(valid.splitlines()[:3]) + '\n'):
            with self.assertRaises(ValueError):
                native.metric_inventory(value, {101: 'low', 102: 'high'})

    def test_concurrent_replay_export_can_identify_nonfirst_pass(self):
        serial, rows = native.replay_inventory(csv_data(), replay_log(),
            [{201: 'low', 202: 'high'}, {101: 'low', 102: 'high'}])
        self.assertEqual(serial, 2)
        self.assertEqual([row['role'] for row in rows], ['low', 'high'])

    def test_replay_rejects_extra_duplicate_missing_and_mismatched_lifecycle(self):
        valid = replay_log()
        roles = [{201: 'low', 202: 'high'}, {101: 'low', 102: 'high'}]
        for value in (valid + '\n==PROF== Connected to process 999 (/usr/bin/python)',
                      valid.replace('Connected to process 202', 'Connected to process 201'),
                      valid.replace('==PROF== Disconnected from process 202', ''),
                      valid.replace('replay pass 1', 'replay pass 2', 1),
                      valid.replace('volta_sgemm_128x128_nn', 'volta_sgemm_64x64_nn', 1),
                      valid + '\n==ERROR== replay mismatch'):
            with self.subTest(log=value), self.assertRaises(ValueError):
                native.replay_inventory(csv_data(), value, roles)
        for changed in (roles[:1], [roles[1], roles[1]]):
            with self.assertRaises(ValueError):
                native.replay_inventory(csv_data(), valid, changed)

    def test_target_inventory_requires_one_matching_low_full_device_launch(self):
        roles = [{201: 'low', 202: 'high'}, {101: 'low', 102: 'high'}]
        lines = []
        for serial, pids in enumerate(((201, 202), (101, 102)), 1):
            lines.extend(f'==PROF== Connected to process {pid} (/usr/bin/python3)' for pid in pids)
            lines.append(f'==PROF== Profiling "{native.TARGET_KERNEL}": Application replay pass {serial}')
            lines.extend(f'==PROF== Disconnected from process {pid}' for pid in pids)
        log = '\n'.join(lines)
        serial, rows = native.replay_inventory(target_csv(), log, roles, target_only=True)
        self.assertEqual(serial, 2)
        self.assertEqual(rows[0]['process_id'], 101)
        self.assertTrue(rows[0]['all_80_useful'])
        large = native.metric_inventory(target_csv(cycles=9007199254740993), roles[1],
                                        selected_roles={'low'})
        self.assertEqual(large[0]['metrics']['sm__cycles_active.min'], 9007199254740993)
        for bad in (target_csv(pid=102), target_csv(pid=999), target_csv(sm=79),
                    target_csv(grid=575), target_csv(block=128), target_csv(cycles=0),
                    target_csv(cycles='1.5'),
                    target_csv(ffma=0), target_csv(kernel='volta_sgemm_128x32_tn'),
                    target_csv() + target_csv().splitlines()[-1] + '\n'):
            with self.subTest(csv=bad), self.assertRaises(ValueError):
                native.replay_inventory(bad, log, roles, target_only=True)
        with self.assertRaises(ValueError):
            native.replay_inventory(target_csv(), log, roles)
        for bad in (
            log.replace('==PROF== Profiling "volta_sgemm_128x32_nn": Application replay pass 2\n'
                        '==PROF== Disconnected from process 101',
                        '==PROF== Disconnected from process 101\n'
                        '==PROF== Profiling "volta_sgemm_128x32_nn": Application replay pass 2'),
            log.replace('==PROF== Connected to process 101 (/usr/bin/python3)\n'
                        '==PROF== Connected to process 102 (/usr/bin/python3)\n'
                        '==PROF== Profiling "volta_sgemm_128x32_nn": Application replay pass 2',
                        '==PROF== Connected to process 102 (/usr/bin/python3)\n'
                        '==PROF== Profiling "volta_sgemm_128x32_nn": Application replay pass 2\n'
                        '==PROF== Connected to process 101 (/usr/bin/python3)'),
            log + '\n==PROF== Profiling "volta_sgemm_128x32_nn": Application replay pass 2'):
            with self.subTest(log=bad), self.assertRaises(ValueError):
                native.replay_inventory(target_csv(), bad, roles, target_only=True)

    def test_target_shim_receipt_fails_closed_on_identity_and_sequence(self):
        content = target_shim_receipt()
        receipt = native.target_call_receipt(content, 101, '/fixed/libcublas.so')
        self.assertEqual(receipt['ordinal'], 149)
        self.assertFalse(receipt['phase_verified'])
        self.assertFalse(receipt['counter_verified'])
        for bad in (content.replace('ordinal=149', 'ordinal=150', 1),
                    content.replace('ordinal=149 domain=1', 'ordinal=149 domain=0'),
                    content.replace('ordinal=149 status=0', 'ordinal=149 status=1'),
                    content.replace('tid=102', 'tid=103', 1),
                    content.replace('m=3072 n=768 k=768', 'm=1024 n=768 k=768', 1),
                    content.replace('GlimmerNativeSgemm149-pid-101', 'wrong-range'),
                    content.replace('GLIMMER_NATIVE_SGEMM_END pid=101 ordinal=149 status=0\n', ''),
                    content + '\nGLIMMER_NATIVE_SGEMM_UNKNOWN unexpected'):
            with self.subTest(receipt=bad[:100]), self.assertRaises(ValueError):
                native.target_call_receipt(bad, 101, '/fixed/libcublas.so')
        with self.assertRaises(ValueError):
            native.target_call_receipt(content, 102, '/fixed/libcublas.so')
        with self.assertRaises(ValueError):
            native.target_call_receipt(content, 101, '/other/libcublas.so')

    def test_target_evidence_joins_each_pass_and_exact_profiler_recipe(self):
        roles = [{201: 'low', 202: 'high'}, {101: 'low', 102: 'high'}, {301: 'low', 302: 'high'}]
        logs = [target_shim_receipt(pid=pid) for pid in (201, 101, 301)]
        baseline = native.target_call_receipt(logs[0], 201, '/fixed/libcublas.so')['descriptor_sequence']
        oracle = dict(tensors=401, elements=217787909, full_tensor_recomparison=True,
                      source_bound_complete_inference=True, first_command_handoff=False)
        outer, bridge_command = Path('/tmp/fixed-capture'), ['/fixed/authenticated-bridge']
        manifest = dict(python=str(native.PYTHON), assets='/assets', startup_reference='/reference',
                        export='/export')
        directories = [Path('/tmp/pass-1/native'), Path('/tmp/pass-2/native'), Path('/tmp/pass-3/native')]
        application_commands = [native.target_application_commands(manifest, directory)
                                for directory in directories]
        command = native.target_profiler_command(outer, bridge_command)
        lines = []
        for serial, pids in enumerate(((201, 202), (101, 102), (301, 302)), 1):
            lines.extend(f'==PROF== Connected to process {pid} (/usr/bin/python3)' for pid in pids)
            lines.append(f'==PROF== Profiling "{native.TARGET_KERNEL}": Application replay pass {serial}')
            lines.extend(f'==PROF== Disconnected from process {pid}' for pid in pids)
        log = '\n'.join(lines)
        args = dict(shim_logs=logs, provider='/fixed/libcublas.so', reference_descriptors=baseline,
                    oracles=[oracle] * 3, actual_command=command, outer=outer, bridge_command=bridge_command,
                    manifest=manifest, native_directories=directories,
                    application_commands_by_pass=application_commands)
        result = native.target_evidence(target_csv(), log, roles, **args)
        self.assertEqual(result['exported_replay_serial'], 2)
        self.assertEqual(result['target']['process_id'], 101)
        self.assertFalse(result['phase_verified'])
        self.assertFalse(result['counter_verified'])
        self.assertFalse(result['full_device_qualified'])
        variants = [dict(shim_logs=[logs[0], logs[0], logs[2]]),
                    dict(shim_logs=logs[:2]),
                    dict(oracles=[oracle, dict(oracle, elements=1), oracle]),
                    dict(reference_descriptors=[[0] * 8] * 438),
                    dict(actual_command=command + ['--nvtx-exclude', 'anything']),
                    dict(actual_command=[value if value != 'off' else 'on' for value in command]),
                    dict(application_commands_by_pass=[native.application_commands(manifest, directory)
                                                       for directory in directories]),
                    dict(bridge_command=['/other/bridge'])]
        for change in variants:
            with self.subTest(change=change.keys()), self.assertRaises(ValueError):
                native.target_evidence(target_csv(), log, roles, **dict(args, **change))

    def test_adapter_only_releases_original_fixed_wrapper(self):
        manifest = dict(python=str(native.PYTHON), assets='/assets', startup_reference='/reference',
                        export='/export', application_environment={'X': '1'})
        common, helpers, connection = Mock(), Mock(), Mock()
        common.subprocess, common.ROOT, helpers.wrapper = native.subprocess, native.PROJECT, 'fixed-wrapper'
        connection.call.return_value = dict(identity={'pid': 101, 'start': 42})
        old_verify, old_stop = common.verify_process, helpers.stop_owned
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / 'native').mkdir()
            with (directory / 'native/low.log').open('x') as stream:
                kwargs = dict(cwd=native.PROJECT, stdin=native.subprocess.PIPE, stdout=stream,
                    stderr=native.subprocess.STDOUT, text=True, env={'X': '1'}, start_new_session=True)
                with native.native_launches(common, helpers, connection, directory, manifest):
                    with self.assertRaises(ValueError):
                        common.subprocess.Popen(['/bin/sh'], **kwargs)
                    process = common.subprocess.Popen([str(native.PYTHON), '-B', '-c', 'fixed-wrapper'], **kwargs)
                    self.assertEqual(process.pid, 101)
                    with self.assertRaises(ValueError):
                        common.subprocess.Popen([str(native.PYTHON), '-B', '-c', 'fixed-wrapper'], **kwargs)
                self.assertIs(common.subprocess, native.subprocess)
                self.assertIs(common.verify_process, old_verify)
                self.assertIs(helpers.stop_owned, old_stop)

    def test_partial_constructor_failure_retires_actual_direct_child(self):
        process = Mock(pid=101)
        process.stdin.closed = False
        with ExitStack() as stack:
            for name, value in (('cgroup', '/test'), ('target_environment', {}),
                                ('bridge.drop_command', ['fixed']), ('bridge.tracker_descriptors', set()),
                                ('protocol.application_commands', {'low': ['fixed']}),
                                ('subprocess.Popen', process)):
                stack.enter_context(patch('run_bert_native_ncu_capture.' + name, return_value=value))
            stack.enter_context(patch.object(capture.bridge, 'process_identity', side_effect=ProcessLookupError))
            with self.assertRaises(ProcessLookupError):
                capture.NativeTarget(Path('/outer'), {'uid': 1000, 'gid': 1000}, Path('/pass'),
                                     'low', 9, {}, {}, {})
        process.kill.assert_called_once_with()
        process.wait.assert_called_once_with(timeout=5)
        process.stdin.close.assert_called_once_with()

    def test_target_mode_retains_sealed_descriptor_only_for_low_role(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, provider = root / 'libnative_sgemm_range.so', root / 'libcublas.so'
            source.write_bytes(b'exact reviewed shim')
            provider.write_bytes(b'provider')
            expected = capture.bridge.seal(source)
            descriptor = capture.bridge.sealed_snapshot(source, expected)
            try:
                paths = (source, provider)
                inputs = {str(path): dict(target=str(path), bytes=path.stat().st_size,
                                          sha256=capture.bridge.seal(path)) for path in paths}
                frozen = capture.mapping_inputs(inputs)
                source_status = source.stat()
                shim = dict(descriptor=descriptor, expected=expected, provider=str(provider),
                            forbidden_source=dict(path=str(source), device=source_status.st_dev,
                                                  inode=source_status.st_ino))
                manifest = dict(uid=1000, gid=1000)
                common = (root, manifest, root / 'pass', 99, {}, inputs, frozen)
                process = Mock(pid=101)
                process.stdin.closed = False
                with ExitStack() as stack:
                    for name, value in (('cgroup', '/test'), ('target_environment', {'LANG': 'C'}),
                                        ('bridge.drop_command', ['fixed-drop']),
                                        ('protocol.application_commands', {'low': ['old-low'], 'high': ['old-high']}),
                                        ('protocol.target_application_commands',
                                         {'low': ['new-low'], 'high': ['new-high']}),
                                        ('subprocess.Popen', process), ('bridge.process_identity',
                                         {'pid': 101, 'start': 42}), ('os.pidfd_open', 71),
                                        ('NativeTarget.verify_bootstrap', None)):
                        stack.enter_context(patch('run_bert_native_ncu_capture.' + name, return_value=value))
                    tracker = stack.enter_context(patch.object(capture.bridge, 'tracker_descriptors',
                                                                side_effect=lambda *_: {70}))
                    spawn = capture.subprocess.Popen
                    default = capture.NativeTarget(common[0], common[1], common[2], 'low', *common[3:])
                    high = capture.NativeTarget(common[0], common[1], common[2], 'high', *common[3:],
                                                target_mode=True)
                    low = capture.NativeTarget(common[0], common[1], common[2], 'low', *common[3:],
                                               target_mode=True, sealed_shim=shim)
                    self.assertEqual([default.command, high.command, low.command],
                                     [['old-low'], ['new-high'], ['new-low']])
                    self.assertEqual([call.kwargs['pass_fds'] for call in spawn.call_args_list],
                                     [(70,), (70,), tuple(sorted((70, descriptor)))])
                    self.assertNotIn('LD_PRELOAD', high.environment)
                    self.assertEqual(low.environment['LD_PRELOAD'], f'/proc/self/fd/{descriptor}')
                    self.assertEqual(low.environment['GLIMMER_TRACE_CUBLAS_PROVIDER'], str(provider))
                    with self.assertRaises(TypeError):
                        low.sealed_shim['descriptor'] = 100
                    with self.assertRaises(TypeError):
                        low.sealed_shim['forbidden_source']['inode'] = 1
                    for role, mode, supplied in (('low', True, None), ('high', True, shim),
                                                  ('low', False, shim)):
                        with self.subTest(role=role, mode=mode), self.assertRaises(ValueError):
                            capture.NativeTarget(common[0], common[1], common[2], role, *common[3:],
                                                 target_mode=mode, sealed_shim=supplied)
                    for bad in (dict(shim, provider='/unexpected'),
                                dict(shim, forbidden_source=dict(shim['forbidden_source'], inode=1)),
                                dict(shim, expected='0' * 64)):
                        with self.subTest(shim=bad), self.assertRaises(ValueError):
                            capture.NativeTarget(common[0], common[1], common[2], 'low', *common[3:],
                                                 target_mode=True, sealed_shim=bad)
                    tracker.side_effect = lambda *_: {descriptor}
                    with self.assertRaisesRegex(ValueError, 'distinct sealed descriptor'):
                        capture.NativeTarget(common[0], common[1], common[2], 'low', *common[3:],
                                             target_mode=True, sealed_shim=shim)
                    shim['forbidden_source']['inode'] = 1
                    self.assertEqual(low.sealed_shim['forbidden_source']['inode'], source_status.st_ino)
                    self.assertEqual(spawn.call_count, 3)
                    os.fstat(descriptor)  # The bridge, not NativeTarget, owns the borrowed FD.
            finally:
                os.close(descriptor)

    def test_target_shim_owner_reaps_all_before_closing_and_retains_on_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, provider = root / 'libnative_sgemm_range.so', root / 'libcublas.so.12'
            source.write_bytes(b'fixed reviewed shim')
            provider.write_bytes(b'fixed provider')
            inputs = {str(path): dict(target=str(path), bytes=path.stat().st_size, sha256=capture.bridge.seal(path))
                      for path in (source, provider)}
            frozen = capture.mapping_inputs(inputs)
            manifest = dict(target_shim=dict(source=str(source), provider=str(provider)),
                            source_sha256={str(source): frozen[str(source)]['sha256']})

            def target(role, identity, shim, events, fail=False):
                receipt = capture.bridge.wait_receipt(identity['pid'], identity['start'], 0)
                result = Mock(role=role, identity=identity, sealed_shim=shim, receipt=receipt, pidfd=99)
                result.poll.return_value = receipt

                def close():
                    events.append(role)
                    if fail:
                        raise RuntimeError('injected child cleanup failure')

                result.close.side_effect = close
                return result

            owner = capture.TargetShimOwner(manifest, frozen)
            descriptor = owner.descriptor
            events = []
            owner.register('low', target('low', dict(pid=101, start=42), owner.configuration(), events))
            owner.register('high', target('high', dict(pid=102, start=43), None, events))
            original_close = os.close

            def close(fd):
                if fd == descriptor:
                    events.append('sealed-fd')
                return original_close(fd)

            with patch.object(capture.bridge, 'terminal', return_value=True), \
                    patch.object(capture.os, 'close', side_effect=close):
                result = owner.finish(accepted=True)
            self.assertEqual(events, ['low', 'high', 'sealed-fd'])
            self.assertEqual(result, dict(status='retired', cleanup_errors=[], descriptor_closed=True))
            with self.assertRaises(OSError):
                os.fstat(descriptor)
            with self.assertRaises(ValueError):
                owner.finish(accepted=True)

            owner = capture.TargetShimOwner(manifest, frozen)
            events = []
            owner.register('low', target('low', dict(pid=101, start=42), owner.configuration(), events, True))
            owner.register('high', target('high', dict(pid=102, start=43), None, events))
            try:
                with patch.object(capture.bridge, 'terminal', return_value=True):
                    result = owner.finish(accepted=True)
                self.assertEqual(events, ['low', 'high'])
                self.assertEqual(result['status'], 'failed')
                self.assertIs(result['descriptor_closed'], False)
                os.fstat(owner.descriptor)
            finally:
                os.close(owner.descriptor)

            with self.assertRaisesRegex(ValueError, 'reviewed shim SHA'):
                capture.TargetShimOwner(dict(manifest, source_sha256={str(source): '0' * 64}), frozen)
            provider.write_bytes(b'changed provider')
            with self.assertRaisesRegex(ValueError, 'frozen preflight'):
                capture.TargetShimOwner(manifest, frozen)

    def test_target_bridge_publishes_only_after_children_and_shim_are_closed(self):
        for failure in ('none', 'cleanup', 'publication', 'spawn', 'registration', 'reply', 'default'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source, provider = root / 'libnative_sgemm_range.so', root / 'libcublas.so.12'
                source.write_bytes(b'fixed reviewed shim')
                provider.write_bytes(b'fixed provider')
                inputs = {str(path): dict(target=str(path), bytes=path.stat().st_size,
                                          sha256=capture.bridge.seal(path)) for path in (source, provider)}
                frozen = capture.mapping_inputs(inputs)
                manifest = dict(uid=1000, gid=1000, mapping_inputs_sha256='a' * 64,
                                application_environment={}, target_shim=dict(source=str(source),
                                provider=str(provider)), source_sha256={str(source): frozen[str(source)]['sha256']})
                connection = Mock(serial=1, socket=object(), pidfd=999)
                events, reports, sealed = [], [], []
                original_snapshot, original_close = capture.bridge.sealed_snapshot, os.close
                log_fds = [os.open('/dev/null', os.O_RDONLY), os.open('/dev/null', os.O_RDONLY)]
                owned_fds = set(log_fds)
                requests = iter([(dict(op='begin', serial=1, directory=str(root / 'pass')), []),
                                 (dict(op='spawn', serial=1, role='low'), [log_fds[0]]),
                                 (dict(op='spawn', serial=1, role='high'), [log_fds[1]]),
                                 (dict(op='complete', serial=1), [])])

                def snapshot(path, sha):
                    descriptor = original_snapshot(path, sha)
                    sealed.append(descriptor)
                    owned_fds.add(descriptor)
                    return descriptor

                def close(fd):
                    if sealed and fd == sealed[0]:
                        events.append('sealed-fd')
                    result = original_close(fd)
                    owned_fds.discard(fd)
                    return result

                def create_target(*args, **kwargs):
                    role = args[3]
                    if failure == 'spawn' and role == 'low':
                        raise RuntimeError('injected constructor failure')
                    identity = dict(pid=101 if role == 'low' else 102, start=42 if role == 'low' else 43)
                    receipt = capture.bridge.wait_receipt(identity['pid'], identity['start'], 0)
                    target = Mock(role='wrong' if failure == 'registration' and role == 'low' else role,
                                  identity=identity, sealed_shim=kwargs.get('sealed_shim'),
                                  receipt=receipt, released=True, pidfd=99,
                                  record=dict(role=role, samples=1))
                    target.poll.return_value = receipt

                    def retire():
                        events.append(role)
                        if failure == 'cleanup' and role == 'low':
                            raise RuntimeError('injected failure')

                    target.close.side_effect = retire
                    return target

                def publish(path, report):
                    events.append('report')
                    reports.append(json.loads(json.dumps(report)))
                    if failure == 'publication':
                        raise OSError('injected report publication failure')

                def send(sock, reply):
                    if reply['op'] == 'complete':
                        if failure == 'reply':
                            events.append('complete-attempt')
                            raise OSError('injected reply loss')
                        events.append('complete-reply')

                def read(path):
                    return {'manifest.json': manifest, 'inputs-before.json': inputs,
                            'mapping-inputs.json': frozen,
                            'ncu-go.json': {'unit_started_us': int(capture.time.monotonic() * 1e6)}}[path.name]

                try:
                    with ExitStack() as stack:
                        for name, value in (('digest', 'a' * 64), ('read', None),
                                            ('os.getresuid', (0, 0, 0)), ('bridge.profiler_environment', {}),
                                            ('protocol.Connection', connection), ('bridge.process_identity',
                                             dict(pid=201, start=44)), ('bridge.terminal', True),
                                            ('protocol.validate_directory', root / 'pass'),
                                            ('protocol.validate_log', None)):
                            mock = stack.enter_context(patch('run_bert_native_ncu_capture.' + name, return_value=value))
                            if name == 'read':
                                mock.side_effect = read
                        stack.enter_context(patch.object(capture.bridge, 'sealed_snapshot', side_effect=snapshot))
                        stack.enter_context(patch.object(capture.os, 'close', side_effect=close))
                        stack.enter_context(patch.object(capture.select, 'select', return_value=([connection.socket], [], [])))
                        stack.enter_context(patch.object(capture.protocol, 'receive', side_effect=lambda *_: next(requests)))
                        stack.enter_context(patch.object(capture, 'NativeTarget', side_effect=create_target))
                        stack.enter_context(patch.object(capture.ncu, 'publish_json', side_effect=publish))
                        stack.enter_context(patch.object(capture, 'save', side_effect=publish))
                        stack.enter_context(patch.object(capture.bridge, 'send_packet', side_effect=send))
                        if failure in ('publication', 'reply'):
                            with self.assertRaisesRegex(OSError, 'publication failure|reply loss'):
                                capture.bridge_worker(root, 'endpoint', dict(pid=201, start=44),
                                                      target_mode=True, expected_manifest_sha256='a' * 64)
                            status = None
                        elif failure == 'default':
                            status = capture.bridge_worker(root, 'endpoint', dict(pid=201, start=44))
                        else:
                            status = capture.bridge_worker(root, 'endpoint', dict(pid=201, start=44),
                                                           target_mode=True, expected_manifest_sha256='a' * 64)
                    self.assertEqual(len(sealed), 0 if failure == 'default' else 1)
                    self.assertEqual(status, {'none': 0, 'cleanup': 1, 'publication': None, 'spawn': 1,
                                              'registration': 1, 'reply': None, 'default': 0}[failure])
                    self.assertEqual(events, {'none': ['low', 'high', 'sealed-fd', 'report', 'complete-reply'],
                                              'cleanup': ['low', 'high', 'report'],
                                              'publication': ['low', 'high', 'sealed-fd', 'report'],
                                              'spawn': ['report'], 'registration': ['low', 'report'],
                                              'reply': ['low', 'high', 'sealed-fd', 'report', 'complete-attempt'],
                                              'default': ['complete-reply', 'low', 'high', 'report']}[failure])
                    self.assertEqual(reports[0]['status'], 'failed' if failure in ('cleanup', 'spawn', 'registration')
                                     else 'native_bridge_retired')
                    connection.close.assert_called_once_with()
                    if failure == 'default':
                        pass
                    elif failure in ('cleanup', 'spawn', 'registration'):
                        os.fstat(sealed[0])
                    else:
                        with self.assertRaises(OSError):
                            os.fstat(sealed[0])
                finally:
                    for fd in owned_fds:
                        os.close(fd)

    def test_target_bridge_rejects_unbound_manifest_before_opening_shim(self):
        with patch.object(capture, 'digest', return_value='b' * 64), \
                patch.object(capture, 'TargetShimOwner') as owner:
            with self.assertRaisesRegex(ValueError, 'authenticated target manifest'):
                capture.bridge_worker(Path('/unused'), 'endpoint', dict(pid=201, start=44),
                                      target_mode=True, expected_manifest_sha256='a' * 64)
            owner.assert_not_called()
            with self.assertRaisesRegex(ValueError, 'explicit native bridge mode'):
                capture.bridge_worker(Path('/unused'), 'endpoint', dict(pid=201, start=44),
                                      expected_manifest_sha256='a' * 64)

    def test_target_controller_binds_profiled_inference_and_preserves_logical_receipt(self):
        for failure in ('none', 'unreviewed', 'uninstrumented', 'oracle'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / 'passes').mkdir()
                validator = str(native.ONNX_VALIDATOR)
                sha = capture.bridge.seal(native.ONNX_VALIDATOR)
                inputs = {validator: dict(target=validator, bytes=native.ONNX_VALIDATOR.stat().st_size, sha256=sha)}
                frozen = capture.mapping_inputs(inputs)
                (root / 'mapping-inputs.json').write_text(json.dumps(frozen))
                base = {'PATH': '/usr/bin:/bin', 'CUDA_VISIBLE_DEVICES': 'GPU-test'}
                gpu = 'GPU-test, V100, driver, runtime'
                manifest = dict(uid=1000, gid=1000, python=str(native.PYTHON), gpu=gpu,
                    application_environment=base, assets='/assets', startup_reference='/reference',
                    export='/reference/export', mapping_inputs_sha256=capture.digest(root / 'mapping-inputs.json'),
                    source_sha256={} if failure == 'unreviewed' else {validator: sha})
                (root / 'manifest.json').write_text(json.dumps(manifest))
                (root / 'ncu-go.json').write_text(json.dumps(
                    dict(unit_started_us=int(capture.time.monotonic() * 1e6))))
                connection = Mock()
                guard = Mock(receipt=dict(resource_checks=1))
                training = Mock(ASSET_ROOT=Path('/assets'))
                training.clean_environment.return_value = base
                h = Mock(training=training)
                h.query.return_value = gpu
                original_onnx = h.validate_onnx
                common = Mock(STARTUP_INPUTS=Path('/reference'))
                common.application_references.return_value = (Path('/reference'), Path('/reference/export'))
                adapted = Mock(return_value=dict(application_instrumented=failure != 'uninstrumented',
                    profiler_diagnostic_only=True, base_validator_sha256=sha, instrumented_adapter='flag-v1',
                    diagnostic_only=True, performance_qualified=False))

                def control(common_arg, helper, directory, mode, environment, resource_guard):
                    inference = helper.validate_onnx('/out', 'GPU-test', '/reference/export')
                    self.assertIs(inference, adapted.return_value)
                    with self.assertRaisesRegex(ValueError, 'export changed'):
                        helper.validate_onnx('/out', 'GPU-test', '/wrong-export')
                    return dict(mode='native', application_commands=native.application_commands(
                        manifest, directory / 'native'), inference_numerical=inference)

                def reparse(common_arg, helper, directory, mode, job, gpu_arg):
                    self.assertEqual(helper.validate_onnx('/out', 'GPU-test', '/reference/export'),
                                     job['inference_numerical'])
                    return {} if failure == 'oracle' else capture.pair.ORACLE

                environment = dict(base, TMPDIR=str(root / 'ncu-tmp'), CUDA_CACHE_DISABLE='1')
                with ExitStack() as stack:
                    stack.enter_context(patch.dict(capture.os.environ, environment, clear=True))
                    stack.enter_context(patch.object(capture.bridge, 'validate_status'))
                    stack.enter_context(patch.object(capture, 'proc_mapped_paths', return_value={}))
                    stack.enter_context(patch.object(capture.protocol, 'Connection', return_value=connection))
                    stack.enter_context(patch.object(capture.candidate, 'ResourceGuard', return_value=guard))
                    stack.enter_context(patch.object(capture.candidate, 'load_helpers', return_value=(common, h)))
                    adapter = stack.enter_context(patch.object(capture.protocol, 'instrumented_onnx_validator',
                                                                return_value=adapted))
                    stack.enter_context(patch.object(capture.protocol, 'native_launches', return_value=nullcontext()))
                    stack.enter_context(patch.object(capture.candidate, 'guarded_control', side_effect=control))
                    stack.enter_context(patch.object(capture.candidate, 'reparse', side_effect=reparse))
                    stack.enter_context(patch.object(capture.pair, 'validate_job'))
                    stack.enter_context(patch.object(capture.bridge, 'process_identity',
                                                     return_value=dict(pid=101, start=42)))
                    status = capture.controller(root, 'endpoint', dict(pid=201, start=44), 1,
                                                target_mode=True, expected_manifest_sha256=capture.digest(
                                                    root / 'manifest.json'),
                                                expected_mapping_sha256=manifest['mapping_inputs_sha256'])
                self.assertEqual(status, int(failure != 'none'))
                self.assertIs(h.validate_onnx, original_onnx)
                self.assertEqual(connection.call.call_args.kwargs['accepted'], failure == 'none')
                connection.close.assert_called_once_with()
                if failure == 'unreviewed':
                    adapter.assert_not_called()
                else:
                    adapter.assert_called_once_with(sha, event_start=True)
                    report_path = next((root / 'passes').glob('*/pass.json'))
                    report = json.loads(report_path.read_text())
                    self.assertEqual(report['status'], 'native_pass_requires_review' if failure == 'none' else 'failed')
                    if failure == 'none':
                        self.assertTrue(report['inference_event_start'])
                        plan = report['target_command_plan']
                        self.assertFalse(plan['actual_commands_authenticated'])
                        for role in native.ROLES:
                            self.assertEqual(plan['expected_target_commands'][role],
                                             plan['logical_release_commands'][role] + ['--profile-cuda'])

    def test_target_poll_records_real_wait_status_without_zero_synthesis(self):
        for status in (0, 9, 256):
            target = capture.NativeTarget.__new__(capture.NativeTarget)
            target.identity, target.receipt, target.record = {'pid': 101, 'start': 42}, None, {}
            target.process = Mock()
            with patch.object(capture.os, 'waitpid', return_value=(101, status)):
                result = target.poll()
            self.assertEqual(result, native.bridge.wait_receipt(101, 42, status))
            self.assertEqual(target.process.returncode, result['returncode'])

    def test_target_session_binds_root_commands_maps_and_inference(self):
        directory = Path('/outer/passes/pass-0001')
        source, provider, validator = '/fixed/libnative_sgemm_range.so', '/fixed/libcublas.so', str(native.ONNX_VALIDATOR)
        shim_sha, validator_sha = 'a' * 64, 'b' * 64
        frozen = {source: dict(sha256=shim_sha, bytes=100), provider: dict(sha256='c' * 64),
                  validator: dict(sha256=validator_sha)}
        seals = {f'/vendor/{name}': 'd' * 64 for name in capture.NCU_INJECTION_ORDER}
        frozen.update({name: dict(sha256=sha) for name, sha in seals.items()})
        manifest = dict(python=str(native.PYTHON), assets='/assets', startup_reference='/reference', export='/export',
                        uid=1000, gid=1000, target_shim=dict(source=source, provider=provider),
                        ncu_injection_seals=seals, source_sha256={source: shim_sha, validator: validator_sha})
        logical = native.application_commands(manifest, directory / 'native')
        commands = native.target_application_commands(manifest, directory / 'native')
        identities = dict(low=dict(pid=101, start=10), high=dict(pid=102, start=20))
        inference = dict(base_validator_sha256=validator_sha, application_instrumented=True,
                         profiler_diagnostic_only=True, diagnostic_only=True, performance_qualified=False,
                         instrumented_adapter='flag-v1')
        job = dict(mode='native', application_commands=logical, application_identities=identities,
                   inference_numerical=inference)
        report = dict(status='native_pass_requires_review', target_mode=True, profiler_diagnostic_only=True,
                      inference_event_start=True, reparse=capture.pair.ORACLE, inference_validator_sha256=validator_sha,
                      job=job, target_command_plan=dict(logical_release_commands=logical,
                          expected_target_commands=commands, actual_commands_authenticated=False))
        records = {}
        for role in native.ROLES:
            target_identity = identities[role]
            mappings = {name: frozen[name] for name in (provider, *seals)}
            maps = dict(monotonic_ns=1, mappings=mappings)
            environment = {}
            if role == 'low':
                maps['sealed_preload'] = dict(pid=101, start=10, target_descriptor=7, device=1, inode=2,
                    bytes=100, sha256=shim_sha, mode=0o555, seals=capture.bridge.SNAPSHOT_SEALS,
                    mapping_count=1, mapping_permissions=['r-xp'])
                environment = dict(LD_PRELOAD='/proc/self/fd/7', GLIMMER_TRACE_CUBLAS_PROVIDER=provider)
            records[role] = dict(role=role, identity=target_identity, command=commands[role], released=True,
                samples=1, status='retired', wait=capture.bridge.wait_receipt(**target_identity, raw_status=0),
                injection=dict(mapped_ncu_injections=sorted(seals)), status_fields={}, observed_mappings=mappings,
                initial_maps=deepcopy(maps), last_maps=deepcopy(maps),
                initial_environment=environment, last_environment=deepcopy(environment))
        root = dict(status='native_bridge_retired', directory=str(directory), targets=records,
                    shim_cleanup=dict(status='retired', cleanup_errors=[], descriptor_closed=True))
        with patch.object(capture.bridge, 'validate_status'), patch.object(capture.pair, 'validate_job') as check_job:
            receipt = capture.review_target_session(manifest, directory, report, root, frozen, event_start=True)
            check_job.assert_called_once_with({}, job)
            self.assertEqual(receipt['roles'], {101: 'low', 102: 'high'})
            self.assertEqual(receipt['application_commands'], commands)
            self.assertIs(receipt['authenticated'], False)
            self.assertIs(receipt['full_device_qualified'], False)
            self.assertIs(receipt['performance_qualified'], False)
            for case in ('command', 'oracle', 'inference', 'cleanup', 'vendor', 'missing_map',
                         'vendor_seal', 'provider', 'seal', 'high_shim'):
                altered_report, altered_root = deepcopy(report), deepcopy(root)
                altered_frozen = deepcopy(frozen)
                if case == 'command':
                    altered_root['targets']['low']['command'].pop()
                elif case == 'oracle':
                    altered_report['reparse'] = {}
                elif case == 'inference':
                    altered_report['job']['inference_numerical']['application_instrumented'] = False
                elif case == 'cleanup':
                    altered_root['shim_cleanup']['descriptor_closed'] = False
                elif case == 'vendor':
                    altered_root['targets']['high']['injection']['mapped_ncu_injections'].pop()
                elif case == 'missing_map':
                    altered_root['targets']['low']['initial_maps']['mappings'].pop(next(iter(seals)))
                elif case == 'vendor_seal':
                    altered_frozen[next(iter(seals))]['sha256'] = 'f' * 64
                elif case == 'provider':
                    altered_root['targets']['low']['last_environment']['GLIMMER_TRACE_CUBLAS_PROVIDER'] = '/wrong'
                elif case == 'seal':
                    altered_root['targets']['low']['last_maps']['sealed_preload']['sha256'] = 'f' * 64
                else:
                    altered_root['targets']['high']['last_maps']['sealed_preload'] = {}
                with self.subTest(case=case), self.assertRaises(ValueError):
                    capture.review_target_session(manifest, directory, altered_report, altered_root, altered_frozen,
                                                  event_start=True)

    def test_target_supervisor_commands_bind_both_reviewed_inputs(self):
        manifest_sha, mapping_sha = 'a' * 64, 'b' * 64
        outer, supervisor = Path('/outer'), dict(pid=101, start=42)
        ordinary = capture.controller_command(outer, 'endpoint', supervisor, 1)
        command = capture.controller_command(outer, 'endpoint', supervisor, 1,
            target_manifest_sha256=manifest_sha, target_mapping_sha256=mapping_sha)
        self.assertEqual(command, ordinary + ['--target-mode', '--manifest-sha256', manifest_sha,
                                              '--mapping-sha256', mapping_sha])
        with self.assertRaises(ValueError):
            capture.controller_command(outer, 'endpoint', supervisor, 1, target_manifest_sha256=manifest_sha)
        with self.assertRaises(ValueError):
            capture.controller_command(outer, 'endpoint', supervisor, 1,
                target_manifest_sha256=manifest_sha, target_mapping_sha256='invalid')
        capture_instance = capture.NativeCapture.__new__(capture.NativeCapture)
        capture_instance.outer, capture_instance.identity, capture_instance.endpoint = outer, supervisor, 'endpoint'
        capture_instance.code_fd, capture_instance.code_sha = 7, 'c' * 64
        capture_instance.target_mode = False
        ordinary_bridge = capture_instance.target()
        capture_instance.target_mode, capture_instance.target_manifest_sha256 = True, manifest_sha
        target_bridge = capture_instance.target()
        self.assertEqual(target_bridge, ordinary_bridge + ['--target-mode', '--manifest-sha256', manifest_sha])
        self.assertIn('--profile-from-start', native.target_profiler_command(outer, target_bridge))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'manifest.json').write_text('{}')
            (root / 'mapping-inputs.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'supervisor target bindings differ'):
                capture.NativeCapture(root, {}, '/task', 1, None, target_manifest_sha256=manifest_sha,
                                      target_mapping_sha256=mapping_sha)
            actual_mapping_sha = capture.digest(root / 'mapping-inputs.json')
            manifest = dict(mapping_inputs_sha256=actual_mapping_sha)
            (root / 'manifest.json').write_text(json.dumps(manifest))
            actual_manifest_sha = capture.digest(root / 'manifest.json')
            with patch.object(native, 'NCU_BINARY', '/different/ncu'), \
                    self.assertRaisesRegex(ValueError, 'supervisor target bindings differ'):
                capture.NativeCapture(root, manifest, '/task', 1, None,
                    target_manifest_sha256=actual_manifest_sha, target_mapping_sha256=actual_mapping_sha)

    def test_target_supervisor_runs_only_fixed_profiler_recipe(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as resources:
            instance = capture.NativeCapture.__new__(capture.NativeCapture)
            instance.outer, instance.resources = Path(temporary), resources
            instance.target_mode, instance.manifest = True, dict(ncu_supervisor_environment={})
            instance.target = Mock(return_value=['fixed-bridge'])
            instance.sample, instance.listener, instance.sessions, instance.samples = Mock(), Mock(), [{}], []
            process = Mock(returncode=0)
            process.poll.side_effect = [None, 0]
            expected = native.target_profiler_command(instance.outer, ['fixed-bridge'])
            with patch.object(capture.subprocess, 'Popen', return_value=process) as popen, \
                    patch.object(capture.select, 'select', return_value=([], [], [])):
                result = instance.run()
            self.assertEqual(popen.call_args.args[0], expected)
            self.assertEqual(result['command'], expected)

    def test_internal_target_cli_dispatch_rejects_incomplete_bindings(self):
        manifest_sha, mapping_sha = 'a' * 64, 'b' * 64
        with tempfile.TemporaryDirectory() as temporary:
            base = ['capture', '--controller', temporary, '--endpoint', 'endpoint',
                    '--supervisor', '101:42', '--serial', '1']
            with patch.object(sys, 'argv', base + ['--target-mode', '--manifest-sha256', manifest_sha,
                                                   '--mapping-sha256', mapping_sha]), \
                    patch.object(capture, 'controller', return_value=0) as controller:
                self.assertEqual(capture.main(), 0)
                self.assertEqual(controller.call_args.kwargs, dict(target_mode=True,
                    expected_manifest_sha256=manifest_sha, expected_mapping_sha256=mapping_sha))
            for suffix in (['--target-mode'], ['--manifest-sha256', manifest_sha],
                           ['--target-mode', '--manifest-sha256', manifest_sha]):
                with self.subTest(suffix=suffix), patch.object(sys, 'argv', base + suffix), \
                        self.assertRaisesRegex(ValueError, 'internal target mode'):
                    capture.main()
            bridge_args = ['capture', '--bridge', temporary, '--endpoint', 'endpoint',
                           '--supervisor', '101:42', '--target-mode', '--manifest-sha256', manifest_sha]
            with patch.object(sys, 'argv', bridge_args), \
                    patch.object(capture, 'bridge_worker', return_value=0) as worker:
                self.assertEqual(capture.main(), 0)
                self.assertEqual(worker.call_args.kwargs, dict(target_mode=True,
                                  expected_manifest_sha256=manifest_sha))

    def test_target_verification_records_only_checked_environments_and_vendor_maps(self):
        vendor = {f'/vendor/{name}': (1, number + 10)
                  for number, name in enumerate(capture.NCU_INJECTION_ORDER)}
        environment = {'PATH': '/usr/bin:/bin', 'LD_PRELOAD': '/proc/self/fd/7'}
        for role in ('low', 'high', 'default'):
            with self.subTest(role=role), ExitStack() as stack:
                target = capture.NativeTarget.__new__(capture.NativeTarget)
                target.released, target.identity, target.group = True, dict(pid=101, start=42), '/task'
                target.manifest = dict(uid=1000, gid=1000, ncu_injection_seals=dict.fromkeys(vendor, 'a' * 64))
                target.command, target.directory, target.outer = ['fixed'], Path('/pass'), Path('/outer')
                target.role, target.target_mode = 'high' if role == 'default' else role, role != 'default'
                target.sealed_shim = (dict(descriptor=7, expected='b' * 64, provider='/provider',
                                           forbidden_source=dict(path='/source', device=1, inode=2))
                                      if role == 'low' else None)
                target.poll = Mock(return_value=None)
                target.audit = Mock(frozen={'/provider': {}}, observed={})
                target.audit.observe.return_value = dict(monotonic_ns=1, mappings={})
                target.record = dict(samples=0)
                for name, value in (('relay.authenticate_process', None), ('relay.command_line', ['fixed']),
                                    ('relay.parent', 99), ('os.getpid', 99), ('os.getsid', 101),
                                    ('os.getpgid', 101), ('relay.environment', environment),
                                    ('proc_mapped_paths', vendor), ('validate_target_environment', None),
                                    ('validate_instrumented_target_environment', {'verified': True}),
                                    ('verified_preload', {'verified': True})):
                    stack.enter_context(patch('run_bert_native_ncu_capture.' + name, return_value=value))
                stack.enter_context(patch.object(Path, 'resolve', return_value=capture.PROJECT_ROOT))
                stack.enter_context(patch.object(Path, 'samefile', return_value=True))
                stack.enter_context(patch.object(Path, 'iterdir', return_value=[]))
                stack.enter_context(patch.object(Path, 'read_text', return_value='Uid:\t1000 1000 1000 1000\n'))
                target.verify()
                self.assertEqual(target.record['samples'], 1)
                if role == 'default':
                    self.assertNotIn('last_environment', target.record)
                else:
                    self.assertEqual(target.record['initial_environment'], environment)
                    self.assertEqual(target.record['last_environment'], environment)
                if role == 'low':
                    self.assertEqual(target.audit.observe.call_args.kwargs['sealed_preload']['descriptor'], 7)
                    self.assertEqual(target.audit.observe.call_args.kwargs['forbidden_source']['path'], '/source')
                elif role == 'high':
                    capture.proc_mapped_paths.return_value = dict(list(vendor.items())[:3])
                    with self.assertRaises(capture.NcuMappingUnavailable):
                        target.verify()

    def test_target_release_waits_only_for_missing_vendor_mappings(self):
        seals = {f'/vendor/{name}': 'a' * 64 for name in capture.NCU_INJECTION_ORDER}
        partial = dict.fromkeys(list(seals)[:3], (1, 1))
        complete = dict.fromkeys(seals, (1, 1))
        extra = dict(complete, **{'/vendor/other.so': (1, 1)})
        for role in ('low', 'high'):
            for case in ('ready', 'timeout', 'extra'):
                with self.subTest(role=role, case=case), ExitStack() as stack:
                    target = capture.NativeTarget.__new__(capture.NativeTarget)
                    target.role, target.target_mode = role, True
                    target.released, target.record = False, {}
                    target.identity, target.command = dict(pid=101, start=42), ['fixed']
                    target.process = Mock()
                    target.poll = Mock(return_value=None)
                    target.verify_bootstrap = Mock()
                    mappings = iter([extra] if case == 'extra' else
                                    [partial, complete] if case == 'ready' else [partial])
                    target.verify = Mock(side_effect=lambda: capture.require_target_vendor_mappings(
                        seals, next(mappings)))
                    stack.enter_context(patch.object(capture.relay, 'command_line', return_value=['fixed']))
                    stack.enter_context(patch.object(capture.time, 'sleep'))
                    ticks = [0, 0, 11] if case == 'timeout' else [0, 0, 0]
                    stack.enter_context(patch.object(capture.time, 'monotonic', side_effect=ticks))
                    if case == 'ready':
                        target.release()
                        self.assertEqual(target.verify.call_count, 2)
                    elif case == 'timeout':
                        with self.assertRaisesRegex(TimeoutError, 'NCU mappings did not stabilize'):
                            target.release()
                    else:
                        with self.assertRaisesRegex(ValueError, 'invalid target-role'):
                            target.release()
                        self.assertEqual(target.verify.call_count, 1)

    def test_frozen_maps_detect_inode_size_time_and_closure_changes(self):
        with tempfile.NamedTemporaryFile() as stream:
            path = Path(stream.name)
            metadata = path.stat()
            inputs = {str(path): dict(target=str(path), bytes=0, sha256='a' * 64)}
            frozen = capture.mapping_inputs(inputs)
            line = (f'1000-2000 r-xp 0000 {os.major(metadata.st_dev):x}:{os.minor(metadata.st_dev):x} '
                    f'{metadata.st_ino} {path}\n')
            with patch.object(native, 'PYTHON', path), patch.object(Path, 'read_text', return_value=line):
                audit = capture.NativeMaps(inputs, frozen)
                self.assertEqual(audit.observe(101)['mappings'], frozen)
                self.assertEqual(audit.observe(101)['mappings'], frozen)
                for key in ('device', 'inode', 'bytes', 'mtime_ns', 'ctime_ns'):
                    changed = {str(path): dict(frozen[str(path)], **{key: frozen[str(path)][key] + 1})}
                    with self.subTest(field=key), self.assertRaises(ValueError):
                        capture.NativeMaps(inputs, changed).observe(101)
                with self.assertRaises(ValueError):
                    capture.NativeMaps(inputs, {})
                wrong_map = line.replace(str(metadata.st_ino) + ' ', str(metadata.st_ino + 1) + ' ')
                with patch.object(Path, 'read_text', return_value=wrong_map), self.assertRaises(ValueError):
                    audit.observe(101)

    def test_frozen_maps_only_accept_verified_sealed_preload_by_explicit_opt_in(self):
        with tempfile.NamedTemporaryFile() as stream:
            path = Path(stream.name)
            metadata = path.stat()
            inputs = {str(path): dict(target=str(path), bytes=0, sha256='a' * 64)}
            frozen = capture.mapping_inputs(inputs)
            library = capture.bridge.TARGET_PRELOAD_MAP_NAME
            executable = (f'1000-2000 r-xp 0000 {os.major(metadata.st_dev):x}:'
                          f'{os.minor(metadata.st_dev):x} {metadata.st_ino} {path}\n')
            mapped = f'2000-3000 r-xp 0000 00:01 12345 {library}\n'
            preload = dict(descriptor=7, expected='b' * 64, identity=dict(pid=101, start=42),
                           group='/fixture', target_descriptor=7)
            source = dict(path='/fixture/libnative_sgemm_range.so', device=0, inode=1)
            receipt = dict(pid=101, start=42, target_descriptor=7, device=os.makedev(0, 1),
                           inode=12345, bytes=100, sha256='b' * 64, mode=0o555, seals=15,
                           mapping_count=1, mapping_permissions=['r-xp'])
            with patch.object(native, 'PYTHON', path), \
                    patch.object(Path, 'read_text', return_value=executable + mapped), \
                    patch.object(capture.bridge, 'validate_target_preload_mapping', return_value=receipt) as verify:
                audit = capture.NativeMaps(inputs, frozen)
                with self.assertRaisesRegex(ValueError, 'unknown native executable mapping'):
                    audit.observe(101)
                observation = audit.observe(101, sealed_preload=preload, forbidden_source=source)
                self.assertEqual(observation['mappings'], frozen)
                self.assertEqual(observation['sealed_preload'], receipt)
                self.assertEqual(verify.call_count, 2)
                with self.assertRaisesRegex(ValueError, 'PID differs'):
                    audit.observe(102, sealed_preload=preload, forbidden_source=source)
                with self.assertRaises(ValueError):
                    audit.observe(101, sealed_preload=preload)
                with self.assertRaises(ValueError):
                    audit.observe(101, forbidden_source=source)
                for forbidden in (dict(path=str(path), device=0, inode=1),
                                  dict(path='/other/hardlink.so', device=metadata.st_dev,
                                       inode=metadata.st_ino)):
                    with self.subTest(forbidden=forbidden), \
                            self.assertRaisesRegex(ValueError, 'diagnostic source mapping forbidden'):
                        audit.observe(101, sealed_preload=preload, forbidden_source=forbidden)
                for bad in (mapped.replace(' 12345 ', ' 12346 '), mapped.replace('00:01', '00:02'),
                            mapped.replace('r-xp', 'rwxp'), mapped.replace(library, library + 'x'),
                            mapped + mapped.replace(library, '/memfd:other.so (deleted)'), ''):
                    with self.subTest(mapping=bad), patch.object(Path, 'read_text', return_value=executable + bad), \
                            self.assertRaises(ValueError):
                        audit.observe(101, sealed_preload=preload, forbidden_source=source)
                verify.side_effect = [receipt, dict(receipt, inode=12346)]
                with self.assertRaisesRegex(ValueError, 'changed during closure inspection'):
                    audit.observe(101, sealed_preload=preload, forbidden_source=source)

    def test_instrumented_environment_requires_exact_preload_tracker_and_all_vendor_maps(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            provider = root / 'libcublas.so'
            provider.write_bytes(b'provider')
            name = str(provider.resolve())
            frozen = capture.mapping_inputs({name: dict(target=name, bytes=provider.stat().st_size,
                                                        sha256='a' * 64)})
            seals, mapped, paths = {}, {}, []
            for library in capture.NCU_INJECTION_ORDER:
                path = root / library
                path.write_bytes(library.encode())
                canonical = str(path.resolve())
                metadata = path.stat()
                seals[canonical] = capture.bridge.seal(path)
                mapped[canonical] = (metadata.st_dev, metadata.st_ino)
                paths.append(canonical)
            tracker = ('launcher-preload-prepend: ' + ':'.join(paths[2:]) + '\n' +
                       'not-launcher-preload-prepend: ' + paths[0] + '\n' +
                       'preload-update-preload-prepend: ' + paths[1])
            manifest = dict(application_environment={'LANG': 'C'},
                            ncu_supervisor_environment={}, ncu_injection_seals=seals)
            actual = dict(LANG='C', TMPDIR=str(root / 'ncu-tmp'), CUDA_CACHE_DISABLE='1',
                          LD_PRELOAD='/proc/self/fd/7', GLIMMER_TRACE_CUBLAS_PROVIDER=name,
                          **{'NVIDIA-PROCESS-TRACKING-CONFIGURATION': tracker})
            with patch.object(capture.bridge, 'profiler_environment', side_effect=lambda value, *_: value):
                result = capture.validate_instrumented_target_environment(
                    actual, manifest, root, target_descriptor=7, provider=name, frozen=frozen, mapped=mapped)
                self.assertEqual(result['mapped_ncu_injections'], sorted(seals))
                for change in (dict(LD_PRELOAD='/proc/self/fd/8'),
                               dict(LD_PRELOAD='/proc/self/fd/7:' + paths[0]),
                               dict(GLIMMER_TRACE_CUBLAS_PROVIDER=name + '.other'),
                               dict(GLIMMER_UNEXPECTED='1'), dict(LD_AUDIT='/unexpected'),
                               {'NVIDIA-PROCESS-TRACKING-CONFIGURATION': tracker.replace(paths[0], paths[1])},
                               {'NVIDIA-PROCESS-TRACKING-CONFIGURATION':
                                'not-launcher-preload-prepend: ' + ':'.join(paths[:2]) + '\n' +
                                'preload-update-preload-prepend: ' + paths[2] + '\n' +
                                'launcher-preload-prepend: ' + paths[3]},
                               {'NVIDIA-PROCESS-TRACKING-CONFIGURATION': tracker + '\n' + tracker}):
                    with self.subTest(change=change), self.assertRaises(ValueError):
                        capture.validate_instrumented_target_environment(
                            dict(actual, **change), manifest, root, target_descriptor=7, provider=name,
                            frozen=frozen, mapped=mapped)
                with self.assertRaises(capture.NcuMappingUnavailable):
                    capture.validate_instrumented_target_environment(
                        actual, manifest, root, target_descriptor=7, provider=name, frozen=frozen,
                        mapped={key: value for key, value in mapped.items() if key != paths[-1]})
                incorrect = dict(mapped, **{paths[-1]: (mapped[paths[-1]][0], mapped[paths[-1]][1] + 1)})
                with self.assertRaisesRegex(ValueError, 'differs from actual mapping'):
                    capture.validate_instrumented_target_environment(
                        actual, manifest, root, target_descriptor=7, provider=name, frozen=frozen,
                        mapped=incorrect)
                Path(paths[-1]).write_bytes(b'changed injection')
                with self.assertRaisesRegex(ValueError, 'injection bytes changed'):
                    capture.validate_instrumented_target_environment(
                        actual, manifest, root, target_descriptor=7, provider=name, frozen=frozen, mapped=mapped)
                Path(paths[-1]).write_bytes(capture.NCU_INJECTION_ORDER[-1].encode())
                provider.write_bytes(b'changed provider')
                with self.assertRaisesRegex(ValueError, 'provider is not a frozen input'):
                    capture.validate_instrumented_target_environment(
                        actual, manifest, root, target_descriptor=7, provider=name, frozen=frozen, mapped=mapped)

    def test_credential_rejection_closes_received_descriptors(self):
        with patch.object(native.bridge, 'receive_packet', return_value=({}, [9], (101, 1000, 1000))), \
                patch.object(native.os, 'close') as close:
            with self.assertRaises(ValueError):
                native.receive(Mock(), (101, 0, 0))
            close.assert_called_once_with(9)

    def test_elf_metadata_preserves_search_policy_and_rejects_absolute_needed(self):
        raw = ('[Requesting program interpreter: /lib64/ld-linux-x86-64.so.2]\n'
               '(NEEDED) Shared library: [libexample.so.1]\n'
               '(RUNPATH) Library runpath: [$ORIGIN/../lib]\n')
        info = capture.elf_metadata(raw)
        self.assertEqual(info['needed'], ['libexample.so.1'])
        self.assertEqual(info['runpath'], '$ORIGIN/../lib')
        self.assertIsNone(info['rpath'])
        for changed in (raw.replace('[libexample.so.1]', '[/tmp/libexample.so.1]'),
                        raw + '(NEEDED) Shared library: [libexample.so.1]\n',
                        raw + '(RUNPATH) Library runpath: [/other]\n'):
            with self.assertRaises(ValueError):
                capture.elf_metadata(changed)

    def test_loader_uses_each_root_resolution_without_unioning_search_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            first, second = directory / 'first.so', directory / 'second.so'
            first.touch()
            second.touch()
            rows = [f'libexample.so.1 => {path} (0x1234)\n' for path in (first, second)]
            namespaces = [capture.loader_namespace(row) for row in rows]
            self.assertEqual(namespaces, [{'libexample.so.1': str(first)}, {'libexample.so.1': str(second)}])
            for namespace in namespaces:
                selected = namespace['libexample.so.1']
                infos = {'/root.so': {'needed': ['libexample.so.1'], 'soname': None},
                         selected: {'needed': [], 'soname': 'libexample.so.1'}}
                self.assertEqual(capture.dependency_edges(namespace, infos),
                    [{'source': '/root.so', 'needed': 'libexample.so.1', 'target': selected}])
            for value in (''.join(rows), 'libexample.so.1 => not found\n', '',
                          f'/unrelated/interpreter => {first} (0x1234)\n'):
                with self.assertRaises((ValueError, FileNotFoundError)):
                    capture.loader_namespace(value)
            with self.assertRaises(ValueError):
                capture.dependency_edges({}, {'/root.so': {'needed': ['missing.so'], 'soname': None}})
            with self.assertRaises(ValueError):
                capture.dependency_edges(namespaces[0], {str(first): {'needed': ['libexample.so.1'],
                                                                         'soname': 'wrong.so'}})

    def test_loader_empty_trace_requires_no_declared_dependencies(self):
        self.assertEqual(capture.loader_namespace('statically linked\n', no_dependencies=True), {})
        with self.assertRaises(ValueError):
            capture.loader_namespace('statically linked\n')
        with self.assertRaises(ValueError):
            capture.loader_namespace('', no_dependencies=True)

    def test_loader_directory_inventory_detects_added_extension(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / 'first.so').touch()
            receipt = dict(directories={str(directory): capture.directory_identity(directory)},
                layout={'extensions': str(directory)}, extension_roots=[str(directory / 'first.so')])
            capture.check_loader_state(receipt)
            (directory / 'second.so').touch()
            with self.assertRaises(ValueError):
                capture.check_loader_state(receipt)

    def test_package_inventory_includes_unknown_packages_bytecode_and_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            names = ('new_package/bindings/driver.so', 'new_package/source.py',
                     'new_package/__pycache__/source.pyc', 'new_package/data/blob', 'tool-without-suffix')
            for name in names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'fixture')
            receipt = capture.runtime_inventory(root)
            self.assertEqual(set(receipt['files']), {str(root / name) for name in names})
            self.assertEqual(receipt['file_paths'], len(names))
            self.assertEqual(receipt['unique_file_inodes'], len(names))
            self.assertEqual(receipt['path_bytes'], 7 * len(names))
            self.assertEqual(capture.runtime_inventory(root), receipt)
            (root / 'new_package/added.so').touch()
            self.assertNotEqual(capture.runtime_inventory(root), receipt)

    def test_package_inventory_tracks_links_and_rejects_escapes_specials_and_bounds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'packages'
            root.mkdir()
            item = root / 'first.so'
            item.write_bytes(b'file')
            os.link(item, root / 'hardlink.so')
            (root / 'alias.so').symlink_to('first.so')
            receipt = capture.runtime_inventory(root)
            self.assertEqual(receipt['file_paths'], 3)
            self.assertEqual(receipt['unique_file_inodes'], 1)
            self.assertEqual(receipt['unique_file_bytes'], 4)
            self.assertEqual(receipt['path_bytes'], 12)
            self.assertEqual(receipt['entries'][str(root / 'alias.so')]['link'], 'first.so')
            for limits in ({'entry_limit': 3}, {'byte_limit': 11}):
                with self.assertRaises(ValueError):
                    capture.runtime_inventory(root, **limits)
            (root / 'escape').symlink_to(Path(temporary))
            with self.assertRaises(ValueError):
                capture.runtime_inventory(root)
            (root / 'escape').unlink()
            os.mkfifo(root / 'pipe')
            with self.assertRaises(ValueError):
                capture.runtime_inventory(root)

    def test_package_inventory_rejects_nested_mount_even_on_same_device(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mount = f'1 0 0:1 / {root}/nested rw - tmpfs tmpfs rw\n'
            with patch.object(Path, 'read_text', return_value=mount), self.assertRaises(ValueError):
                capture.runtime_inventory(root)

    def test_interpreter_chain_records_every_hop_and_detects_retargeting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'python-real').write_bytes(b'fixture')
            (root / 'python3').symlink_to('python-real')
            (root / 'python').symlink_to('python3')
            receipt = capture.interpreter_chain(root / 'python')
            self.assertEqual(set(receipt), {str(root / name) for name in ('python', 'python3', 'python-real')})
            (root / 'python3').unlink()
            (root / 'python3').symlink_to('python')
            with self.assertRaises(ValueError):
                capture.interpreter_chain(root / 'python')


if __name__ == '__main__':
    unittest.main()
