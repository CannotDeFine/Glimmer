"""CPU-only checks for scoped NCU process-environment verification."""

import hashlib
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from bert_sgemm_ncu_process import (NcuMappingUnavailable, authenticated_profile_processes, proc_environment,
                                   proc_mapped_paths, verified_preload)
from run_bert_sgemm_ncu_capture import (monitored_command, ncu_command, publish_json, root_supervisor,
                                         review_capture, validate_unprivileged_process, loader_dependencies,
                                         supervisor_failure, fresh_capture_service)


class NcuProcess(unittest.TestCase):
    def test_fresh_capture_review_binds_claim_plan_and_mapped_service(self):
        service, claim_sha, manifest_sha = '/fixed/service', 'a' * 64, 'b' * 64
        product = dict(path=service, sha256='c' * 64)
        binding = dict(contract=dict(runtime=dict(service=product)))
        encoded = json.dumps(binding, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        manifest = dict(fresh_runtime=binding, uid=1003, gid=1003, service_environment={'PATH': '/fixed'},
                        bridge_seals={service: product['sha256']})
        supervisor = dict(fresh_claim_receipt_verified=True, fresh_claim_sha256=claim_sha,
                          fresh_manifest_sha256=manifest_sha, fresh_binding_sha256=hashlib.sha256(encoded).hexdigest())
        replay = dict(plan=dict(fresh_claim_sha256=claim_sha, template_sha256=manifest_sha, uid=1003, gid=1003,
                                environment=manifest['service_environment'], seals=manifest['bridge_seals']))
        inputs = {service: dict(target=service, sha256=product['sha256'])}
        claim = dict(sha256=claim_sha)
        self.assertEqual(fresh_capture_service(manifest, supervisor, replay, inputs, binding, claim, manifest_sha),
                         Path(service))
        changed = dict(replay, plan=dict(replay['plan'], fresh_claim_sha256='d' * 64))
        with self.assertRaisesRegex(ValueError, 'authority differs'):
            fresh_capture_service(manifest, supervisor, changed, inputs, binding, claim, manifest_sha)
        with self.assertRaisesRegex(ValueError, 'service seal differs'):
            fresh_capture_service(manifest, supervisor, replay, {service: dict(target=service, sha256='e' * 64)},
                                  binding, claim, manifest_sha)

    def test_inner_failure_is_diagnostic_only_and_bounded(self):
        with tempfile.TemporaryDirectory() as temporary:
            outer = Path(temporary)
            self.assertIsNone(supervisor_failure(outer))
            path = outer / 'ncu-supervisor.json'
            for value in ({'status': 'failed', 'error': 'specific failure'},
                          {'status': 'passed', 'error': 'not a failure'}, [],
                          {'status': 'failed', 'error': 1}, {'status': 'failed', 'error': 'x' * 4097}):
                path.write_text(json.dumps(value))
                self.assertEqual(supervisor_failure(outer),
                                 'specific failure' if value == {'status': 'failed', 'error': 'specific failure'} else None)
            path.write_text('{')
            self.assertIsNone(supervisor_failure(outer))

    def test_proc_environment_recognizes_only_the_sealed_mapped_ncu_scrub(self):
        with tempfile.TemporaryDirectory() as tmp:
            names = ('libcuda-injection.so', 'libTreeLauncherTargetUpdatePreloadInjection.so',
                     'libInterceptorInjectionTarget.so', 'libTreeLauncherTargetInjection.so')
            seals, mapped, paths = {}, {}, []
            for name in names:
                library = Path(tmp) / name
                library.write_bytes(name.encode())
                canonical = str(library.resolve())
                seals[canonical] = hashlib.sha256(library.read_bytes()).hexdigest()
                metadata = library.stat()
                mapped[canonical] = (metadata.st_dev, metadata.st_ino)
                paths.append(os.fsencode(tmp + '/./' + name))
            preload = b':'.join(paths)
            tracker = (b'launcher-preload-prepend: ' + b':'.join(paths[2:]) + b'\n' +
                       b'not-launcher-preload-prepend: ' + paths[0] + b'\n' +
                       b'preload-update-preload-prepend: ' + paths[1])
            expected = {b'PATH': b'/usr/bin:/bin', b'EMPTY': b'', b'VALUE': b'a=b',
                        b'NVTX_INJECTION64_PATH': b'libcuda-injection.so',
                        b'NVIDIA-PROCESS-TRACKING-CONFIGURATION': tracker}
            entries = [b'LD_PRELOAD=', preload[1:]] + [key + b'=' + value for key, value in expected.items()]
            raw = b'\0'.join(entries) + b'\0'
            with patch('pathlib.Path.read_bytes', return_value=raw), \
                    patch('bert_sgemm_ncu_process.proc_mapped_paths', return_value=mapped):
                self.assertEqual(proc_environment(1, ncu_seals=seals), expected)
                with self.assertRaisesRegex(ValueError, 'malformed'):
                    proc_environment(1)
                with self.assertRaises(ValueError):
                    proc_environment(1, ncu_seals={})
            bad_inputs = (
                raw.replace(preload[1:], preload[1:] + b'x', 1),
                raw.replace(preload[1:], preload[1:].replace(b'/./', b'/'), 1),
                raw.replace(b'LD_PRELOAD=\0', b'LD_PRELOAD=other\0', 1),
                raw.replace(b'LD_PRELOAD=\0', b'', 1),
                raw.replace(b'LD_PRELOAD=\0', b'LD_PRELOAD=\0OTHER=1\0', 1),
                raw + b'another-residue\0', raw + b'PATH=/bin\0', raw + b'=empty-key\0',
                raw.replace(tracker, tracker + b'\nlauncher-preload-prepend: ' + b':'.join(paths[2:])),
                raw.replace(b'NVTX_INJECTION64_PATH=libcuda-injection.so', b'NVTX_INJECTION64_PATH=other'),
            )
            for index, bad in enumerate(bad_inputs):
                with self.subTest(case=index), patch('pathlib.Path.read_bytes', return_value=bad), \
                        patch('bert_sgemm_ncu_process.proc_mapped_paths', return_value=mapped), \
                        self.assertRaises(ValueError):
                    proc_environment(1, ncu_seals=seals)
            with patch('pathlib.Path.read_bytes', return_value=raw), \
                    patch('bert_sgemm_ncu_process.proc_mapped_paths', return_value={}):
                with self.assertRaises(ValueError):
                    proc_environment(1, ncu_seals=seals)
            swapped = os.fsdecode(b':'.join([paths[1], paths[0], *paths[2:]]))
            for mappings in (mapped, {}):
                with self.assertRaisesRegex(ValueError, 'order differs') as invalid:
                    verified_preload(swapped, None, seals, mappings)
                self.assertNotIsInstance(invalid.exception, NcuMappingUnavailable)
            (Path(tmp) / names[0]).write_bytes(b'changed injection')
            with patch('pathlib.Path.read_bytes', return_value=raw), \
                    patch('bert_sgemm_ncu_process.proc_mapped_paths', return_value=mapped):
                with self.assertRaisesRegex(ValueError, 'bytes changed'):
                    proc_environment(1, ncu_seals=seals)
            plain = b'PATH=/usr/bin:/bin\0EMPTY=\0VALUE=a=b\0'
            with patch('pathlib.Path.read_bytes', return_value=plain):
                self.assertEqual(proc_environment(1), {b'PATH': b'/usr/bin:/bin', b'EMPTY': b'', b'VALUE': b'a=b'})

    def test_loader_inventory_includes_direct_interpreter_and_arrow_libraries(self):
        output = ('\tlinux-vdso.so.1 (0x00007ffc1970c000)\n'
                  '\tlibc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007fcf9bcdd000)\n'
                  '\t/lib64/ld-linux-x86-64.so.2 (0x00007fcf9ce01000)\n')
        self.assertEqual(loader_dependencies(output), {Path('/lib/x86_64-linux-gnu/libc.so.6'),
                                                       Path('/lib64/ld-linux-x86-64.so.2')})
        self.assertEqual(loader_dependencies(output + output), loader_dependencies(output))

    def test_loader_inventory_rejects_missing_relative_or_unknown_inputs(self):
        for output in ('', 'linux-vdso.so.1 (0x123)\n', 'libunknown.so => not found\n',
                       'libc.so.6 => relative/libc.so.6 (0x123)\n', 'statically linked\n',
                       '/lib/loader.so (bad-address)\n'):
            with self.subTest(output=output), self.assertRaises(ValueError):
                loader_dependencies(output)

    def test_handshake_publication_never_exposes_partial_or_overwrites_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'ready.json'

            def incomplete(data, stream):
                stream.write('{')
                raise OSError('injected write failure')

            with patch('run_bert_sgemm_ncu_capture.json.dump', side_effect=incomplete):
                with self.assertRaises(OSError):
                    publish_json(path, {'pid': 1})
            self.assertFalse(path.exists())
            self.assertEqual(list(Path(tmp).iterdir()), [])
            publish_json(path, {'pid': 1})
            self.assertEqual(path.stat().st_mode & 0o777, 0o644)
            with self.assertRaises(FileExistsError):
                publish_json(path, {'pid': 2})
            self.assertEqual(json.loads(path.read_text()), {'pid': 1})

    def test_range_export_must_identify_the_same_real_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            outer = Path(tmp)
            (outer / 'manifest.json').write_text('{}')
            run = outer / 'passes/pass-one'
            (run / 'equal').mkdir(parents=True)
            (run / 'equal/server.log').write_text('sealed log')
            (outer / 'ncu.log').write_text(
                '==PROF== Profiling "glimmer_owned_sgemm_nn_observed": Application replay pass 1\n')
            raw = ('"ID","Process ID","Kernel Name","Context","Stream","launch__sm_count",'
                   '"launch__grid_size","launch__block_size","sm__cycles_active.min",'
                   '"sm__sass_thread_inst_executed_op_ffma_pred_on.min"\n'
                   '"","","","","","SM","","","cycle","inst"\n'
                   '"1","42","glimmer_owned_sgemm_nn_observed","1","2","80","320","128","123","45"\n')
            ranged = raw.replace('"glimmer_owned_sgemm_nn_observed"',
                                 '"glimmer.sgemm.target.issue.v1/glimmer_owned_sgemm_nn_observed"')
            (outer / 'ncu-metrics.csv').write_text(raw)
            range_path = outer / 'ncu-range-metrics.csv'
            range_path.write_text(ranged.replace('"42"', '"43"'))
            supervisor = dict(status='capture_requires_replay_and_counter_review',
                              capture=dict(returncode=0, samples=[dict(ncu=[dict(pid=5, start=50)])]),
                              export=dict(returncode=0), range_export=dict(returncode=0))
            clients = {role: dict(pid=pid, start=pid * 10, returncode=0, exit_code_observed=True)
                       for role, pid in [('low', 70), ('high', 71)]}
            supervisor['broker'] = dict(status='retired', sessions=[dict(status='complete',
                directory=str(run), clients=clients, retired_clients=clients,
                controller=dict(pid=72, start=720))])
            (outer / 'ncu-supervisor.json').write_text(json.dumps(supervisor))
            descriptor = dict(transa=0, transb=0, m=3072, n=768, k=768, lda=3072, ldb=768,
                              ldc=3072, device_scalars=0, logical_blocks=2304, workers=320)
            selected = dict(descriptor=descriptor)
            job = dict(mode='equal', server_identity=dict(pid=42, start=99),
                       artifact_identities={'server.log': 'sealed'},
                       application_identities={role: dict(pid=value['pid'], start=value['start'])
                                               for role, value in clients.items()},
                       application_returncodes={'low': 0, 'high': 0})
            reparse = dict(tensors=401, elements=217787909, full_tensor_recomparison=True,
                           source_bound_complete_inference=True, first_command_handoff=False)
            receipt = dict(status='pass_requires_profiler_review', pass_directory=str(run),
                           reparse=reparse, job=job, range=selected,
                           broker_clients=clients,
                           controller_identity=dict(pid=72, start=720),
                           ancestors=[dict(pid=5, start=50)])
            (run / 'pass.json').write_text(json.dumps(receipt))
            with patch('run_bert_sgemm_ncu_capture.candidate.normal_cleanup'), \
                    patch('run_bert_sgemm_ncu_capture.digest', return_value='sealed'), \
                    patch('run_bert_sgemm_ncu_capture.parse_semantic_audit'), \
                    patch('run_bert_sgemm_ncu_capture.authenticated_training_sessions', return_value={2}), \
                    patch('run_bert_sgemm_ncu_capture.targeted_issue_range', return_value=selected):
                with self.assertRaisesRegex(ValueError, 'same launch'):
                    review_capture(outer, '2:3127')
                range_path.write_text(ranged)
                self.assertEqual(review_capture(outer, '2:3127')['status'],
                                 'counter_observation_requires_independent_review')
                for wrong in ('glimmer.sgemm.target.issue.v1',
                              'other.range/glimmer_owned_sgemm_nn_observed',
                              'glimmer.sgemm.target.issue.v1/other_kernel'):
                    range_path.write_text(ranged.replace(
                        'glimmer.sgemm.target.issue.v1/glimmer_owned_sgemm_nn_observed', wrong))
                    with self.subTest(kernel=wrong), self.assertRaisesRegex(ValueError, 'kernel identity'):
                        review_capture(outer, '2:3127')

                service = '/fixed/service'
                product = dict(path=service, sha256='c' * 64)
                contract = dict(schema='fresh_runtime_sgemm_v3', specialization='ordinary',
                                runtime=dict(service=product, frontend={}), service_environment={'PATH': '/fixed'})
                binding = dict(schema='fresh_runtime_v3', variant='scalar', contract=contract)
                claim = dict(sha256='a' * 64)
                manifest = dict(fresh_runtime=binding, uid=1003, gid=1003,
                                service_environment=contract['service_environment'],
                                bridge_seals={service: product['sha256']})
                (outer / 'manifest.json').write_text(json.dumps(manifest))
                (outer / 'inputs-before.json').write_text(json.dumps(
                    {service: dict(target=service, sha256=product['sha256'])}))
                encoded = json.dumps(binding, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
                identities = job['application_identities']
                replay = dict(plan=dict(directory=str(run), clients=identities, uid=1003, gid=1003,
                                        fresh_claim_sha256=claim['sha256'], template_sha256='sealed',
                                        environment=manifest['service_environment'], seals=manifest['bridge_seals']),
                              controller_returncode=0, controller=receipt['controller_identity'],
                              service=job['server_identity'], receipt=dict(returncode=0, exit_code_observed=True),
                              complete=dict(report_sha256='sealed'), helper_ancestors=[dict(pid=5, start=50)])
                supervisor.update(topology='clean-ncu-sibling', fresh_claim_receipt_verified=True,
                                  fresh_claim_sha256=claim['sha256'], fresh_manifest_sha256='sealed',
                                  fresh_binding_sha256=hashlib.sha256(encoded).hexdigest())
                supervisor['capture']['sessions'] = [replay]
                (outer / 'ncu-supervisor.json').write_text(json.dumps(supervisor))
                receipt.update(topology='clean-ncu-sibling', ancestors=[], frontend_mappings={}, launch_order={},
                               target_binding=dict(session=2), logical_release=dict(kind='logical_done'))
                (run / 'pass.json').write_text(json.dumps(receipt))
                ordinary_log = '==PROF== Profiling "glimmer_owned_sgemm_nn": Application replay pass 1\n'
                ordinary_raw = raw.replace('glimmer_owned_sgemm_nn_observed', 'glimmer_owned_sgemm_nn')
                ordinary_range = ranged.replace('glimmer_owned_sgemm_nn_observed', 'glimmer_owned_sgemm_nn')
                log_path, raw_path = outer / 'ncu.log', outer / 'ncu-metrics.csv'
                log_path.write_text(ordinary_log)
                raw_path.write_text(ordinary_raw)
                range_path.write_text(ordinary_range)
                records = [SimpleNamespace(value=lambda key: {'pid': 70, 'event': 1, 'role': 0,
                                                              'connection': 1}[key]),
                           SimpleNamespace(value=lambda key: {'pid': 71, 'event': 1, 'role': 0,
                                                              'connection': 2}[key])]
                semantic = SimpleNamespace(family=lambda family: records)
                with patch('run_bert_sgemm_ncu_capture.parse_semantic_audit', return_value=semantic), \
                        patch('run_bert_sgemm_ncu_capture.targeted_training_binding', return_value=dict(session=2)), \
                        patch('bert_ncu_relay.validate_mapping_receipts'), \
                        patch('bert_ncu_service_proxy.validate_frontend_receipts'), \
                        patch('bert_ncu_service_proxy.validate_launch_order'), \
                        patch('bert_ncu_service_proxy.validate_ordinary_release', return_value=dict(kind='logical_done')):
                    observed = review_capture(outer, '2:3127', fresh_binding=binding, fresh_claim=claim)
                    self.assertEqual((observed['specialization'], observed['target_kernel']),
                                     ('ordinary', 'glimmer_owned_sgemm_nn'))
                    for path, wrong in ((log_path, ordinary_log.replace('glimmer_owned_sgemm_nn',
                                                                       'glimmer_owned_sgemm_nn_observed')),
                                        (raw_path, raw), (range_path, ranged)):
                        original = path.read_text()
                        path.write_text(wrong)
                        with self.subTest(path=path.name), self.assertRaises(ValueError):
                            review_capture(outer, '2:3127', fresh_binding=binding, fresh_claim=claim)
                        path.write_text(original)

    def test_gpu_interference_blocks_profiler_before_child_creation(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch('run_bert_sgemm_ncu_capture.memory_receipt',
                      return_value={'memory.events': 'oom 0\noom_kill 0'}), \
                patch('run_bert_sgemm_ncu_capture.check_owned_gpu',
                      side_effect=ValueError('foreign GPU process')), \
                patch('run_bert_sgemm_ncu_capture.subprocess.Popen') as launch:
            with self.assertRaisesRegex(ValueError, 'foreign GPU'):
                monitored_command(['ncu'], {}, Path(tmp), '/task', float('inf'),
                                  Path(tmp) / 'profile.log', 'GPU-test')
            launch.assert_not_called()

    def test_ncu_command_preserves_full_application_and_drops_privileges(self):
        manifest = dict(uid=1003, gid=1003, groups=[], python='/usr/bin/python3')
        command = ncu_command(Path('/tmp/owned-attempt'), manifest)
        self.assertEqual(command[command.index('--replay-mode') + 1], 'application')
        self.assertEqual(command[command.index('--app-replay-mode') + 1], 'strict')
        self.assertEqual(command[command.index('--kill') + 1], 'no')
        self.assertIn('--reuid=1003', command)
        self.assertIn('--clear-groups', command)
        self.assertIn('--no-new-privs', command)
        for bad in (dict(manifest, uid=0), dict(manifest, uid=True),
                    dict(manifest, groups=[27])):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ncu_command(Path('/tmp/owned-attempt'), bad)

    def test_supervisor_cannot_start_ncu_as_an_unprivileged_process(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch('run_bert_sgemm_ncu_capture.os.getresuid', return_value=(1003, 1003, 1003)), \
                patch('run_bert_sgemm_ncu_capture.monitored_command') as launch:
            self.assertEqual(root_supervisor(Path(tmp)), 1)
            launch.assert_not_called()

    def test_failure_after_launch_triggers_only_the_prevalidated_task_containment(self):
        with tempfile.TemporaryDirectory() as tmp:
            outer = Path(tmp)
            (outer / 'ncu-tmp').mkdir()
            manifest = dict(uid=1003, gid=1003, groups=[], python='/usr/bin/python3',
                            gpu='GPU-test,V100', ncu_sha256='sealed', ncu_exec_sha256='sealed')
            (outer / 'manifest.json').write_text(json.dumps(manifest))
            with patch.dict(os.environ, {}, clear=True), \
                    patch('run_bert_sgemm_ncu_capture.os.getresuid', return_value=(0, 0, 0)), \
                    patch('run_bert_sgemm_ncu_capture.resource.setrlimit'), \
                    patch('run_bert_sgemm_ncu_capture.resource.getrlimit',
                          return_value=(1024 ** 3, 1024 ** 3)), \
                    patch('run_bert_sgemm_ncu_capture.cgroup', return_value='/task'), \
                    patch('run_bert_sgemm_ncu_capture.task_unit_paths', return_value=(outer, 12345)), \
                    patch('run_bert_sgemm_ncu_capture.await_outer_handshake', return_value=1000), \
                    patch('run_bert_sgemm_ncu_capture.require_retired_descendants'), \
                    patch('run_bert_sgemm_ncu_capture.os.write') as kill_write, \
                    patch('run_bert_sgemm_ncu_capture.os.close'), \
                    patch('run_bert_sgemm_ncu_capture.memory_receipt'), \
                    patch('run_bert_sgemm_ncu_capture.digest', return_value='sealed'), \
                    patch('run_bert_sgemm_ncu_capture.managed_client_broker',
                          return_value=nullcontext(None)), \
                    patch('run_bert_sgemm_ncu_capture.monitored_command',
                          side_effect=ValueError('foreign GPU process')):
                self.assertEqual(root_supervisor(outer), 1)
            kill_write.assert_called_once_with(12345, b'1')
            receipt = json.loads((outer / 'ncu-supervisor.json').read_text())
            self.assertEqual(receipt['status'], 'failed')
            self.assertIn('foreign GPU process', receipt['error'])

    def test_fresh_supervisor_rejects_manifest_change_before_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            outer = Path(tmp)
            (outer / 'ncu-tmp').mkdir()
            manifest = dict(uid=os.getuid(), gid=os.getgid(), fresh_runtime={}, gpu='GPU-test,V100')
            (outer / 'manifest.json').write_text(json.dumps(manifest))
            with patch.dict(os.environ, {'SUDO_UID': str(os.getuid()), 'SUDO_GID': str(os.getgid())}, clear=True), \
                    patch('run_bert_sgemm_ncu_capture.os.getresuid', return_value=(0, 0, 0)), \
                    patch('run_bert_sgemm_ncu_capture.resource.setrlimit'), \
                    patch('run_bert_sgemm_ncu_capture.resource.getrlimit', return_value=(1024 ** 3, 1024 ** 3)), \
                    patch('run_bert_sgemm_ncu_capture.cgroup', return_value='/task'), \
                    patch('run_bert_sgemm_ncu_capture.task_unit_paths', return_value=(outer, 12345)), \
                    patch('run_bert_sgemm_ncu_capture.await_outer_handshake', return_value=(1000, 'a' * 64)), \
                    patch('run_bert_sgemm_ncu_capture.require_retired_descendants'), \
                    patch('run_bert_sgemm_ncu_capture.memory_receipt'), \
                    patch('run_bert_sgemm_ncu_capture.os.close'), \
                    patch('bert_ncu_replay_exec.verify_fresh_claim',
                          return_value=dict(manifest_sha256='f' * 64, template=manifest)), \
                    patch('run_bert_sgemm_ncu_capture.monitored_command') as launch:
                self.assertEqual(root_supervisor(outer), 1)
            launch.assert_not_called()
            report = json.loads((outer / 'ncu-supervisor.json').read_text())
            self.assertIn('manifest changed during supervisor authentication', report['error'])

    def test_profile_worker_must_fully_drop_privileges_before_any_application(self):
        manifest = dict(uid=1003, gid=1003, groups=[27, 1003])
        good = ('CapInh:\t0000\nCapPrm:\t0000\nCapEff:\t0000\n'
                'CapAmb:\t0000\nNoNewPrivs:\t1\n')
        validate_unprivileged_process(manifest, good, (1003, 1003, 1003),
                                      (1003, 1003, 1003), [1003, 27])
        for status, uids, gids, groups in (
                (good, (1003, 0, 1003), (1003, 1003, 1003), [27, 1003]),
                (good, (1003, 1003, 1003), (1003, 0, 1003), [27, 1003]),
                (good, (1003, 1003, 1003), (1003, 1003, 1003), [1003]),
                (good.replace('CapEff:\t0000', 'CapEff:\t0001'),
                 (1003, 1003, 1003), (1003, 1003, 1003), [27, 1003]),
                (good.replace('NoNewPrivs:\t1', 'NoNewPrivs:\t0'),
                 (1003, 1003, 1003), (1003, 1003, 1003), [27, 1003])):
            with self.subTest(status=status, uids=uids, gids=gids, groups=groups), \
                    self.assertRaises(ValueError):
                validate_unprivileged_process(manifest, status, uids, gids, groups)

    def test_actual_mapping_device_and_inode_must_match_the_sealed_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            library = Path(tmp) / 'libcuda-injection.so'
            library.write_bytes(b'test injection')
            metadata = library.stat()
            line = (f'1000-2000 r-xp 00000000 {os.major(metadata.st_dev):02x}:'
                    f'{os.minor(metadata.st_dev):02x} {metadata.st_ino} {library}\n')
            with patch('pathlib.Path.read_text', return_value=line):
                self.assertEqual(proc_mapped_paths(1),
                                 {str(library.resolve()): (metadata.st_dev, metadata.st_ino)})
            with patch('pathlib.Path.read_text', return_value=line.replace(str(metadata.st_ino), '0')):
                with self.assertRaisesRegex(ValueError, 'identity changed'):
                    proc_mapped_paths(1)

    def test_accepts_only_a_sealed_mapped_injection_and_unchanged_application_preload(self):
        with tempfile.TemporaryDirectory() as tmp:
            library = Path(tmp) / 'libcuda-injection.so'
            library.write_bytes(b'test injection')
            app = Path(tmp) / 'interceptor.so'
            app.write_bytes(b'app interceptor')
            other = Path(tmp) / 'other.so'
            other.write_bytes(b'other library')
            name = str(library.resolve())
            app_name, other_name = str(app.resolve()), str(other.resolve())
            metadata = library.stat()
            mapped = {name: (metadata.st_dev, metadata.st_ino)}
            seals = {name: hashlib.sha256(library.read_bytes()).hexdigest()}
            actual = name + ':' + app_name
            self.assertEqual(verified_preload(actual, app_name, seals, mapped),
                             dict(mapped_ncu_injections=[name], additional_preloads=[name]))
            self.assertEqual(verified_preload(None, None, seals, mapped),
                             dict(mapped_ncu_injections=[name], additional_preloads=[]))
            for value in (actual, None):
                with self.assertRaises(NcuMappingUnavailable) as missing:
                    verified_preload(value, app_name if value else None, seals, {})
                self.assertEqual(missing.exception.expected, [name])
                self.assertEqual(missing.exception.missing, [name])
                self.assertEqual(missing.exception.mapped, [])
            for value in (actual + ':' + name, actual + ':' + other_name):
                with self.assertRaises(ValueError) as invalid:
                    verified_preload(value, app_name, seals, {})
                self.assertNotIsInstance(invalid.exception, NcuMappingUnavailable)
            for value, expected, mappings in (
                    (actual, app_name, {}),
                    (actual, app_name, {name: (metadata.st_dev, metadata.st_ino + 1)}),
                    (actual + ':' + name, app_name, mapped),
                    (actual + ':' + other_name, app_name, mapped),
                    (name + ':' + other_name, app_name, mapped),
                    (name + '\t' + app_name, app_name, mapped)):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    verified_preload(value, expected, seals, mappings)
            library.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'bytes changed'):
                verified_preload(actual, app_name, seals, mapped)

    def test_scoped_hook_preserves_the_original_identity_checks_and_restores_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            library = Path(tmp) / 'libcuda-injection.so'
            library.write_bytes(b'test injection')
            app = Path(tmp) / 'interceptor.so'
            app.write_bytes(b'app interceptor')
            name = str(library.resolve())
            app_name = str(app.resolve())
            metadata = library.stat()
            mapped = {name: (metadata.st_dev, metadata.st_ino)}
            seals = {name: hashlib.sha256(library.read_bytes()).hexdigest()}
            calls = []

            def original(h, process, start, command, environment):
                calls.append((start, command, environment))

            common = SimpleNamespace(verify_process=original)
            process = SimpleNamespace(pid=1)
            process_env = {b'LD_PRELOAD': (name + ':' + app_name).encode(),
                           b'NVTX_INJECTION64_PATH': b'libcuda-injection.so'}
            with patch('bert_sgemm_ncu_process.proc_environment', return_value=process_env), \
                 patch('bert_sgemm_ncu_process.proc_mapped_paths', return_value=mapped):
                with self.assertRaisesRegex(RuntimeError, 'injected'):
                    with authenticated_profile_processes(common, seals):
                        common.verify_process(None, process, 42, ['app'],
                                              {'LD_PRELOAD': app_name, 'CUDA_VISIBLE_DEVICES': '0'})
                        raise RuntimeError('injected')
            self.assertIs(common.verify_process, original)
            self.assertEqual(calls[0][0:2], (42, ['app']))
            self.assertEqual(calls[0][2]['LD_PRELOAD'], name + ':' + app_name)
            self.assertEqual(calls[0][2]['CUDA_VISIBLE_DEVICES'], '0')

    def test_absent_preload_stays_absent_when_ncu_is_mapped(self):
        with tempfile.TemporaryDirectory() as tmp:
            library = Path(tmp) / 'libcuda-injection.so'
            library.write_bytes(b'test injection')
            name = str(library.resolve())
            metadata = library.stat()
            mapped = {name: (metadata.st_dev, metadata.st_ino)}
            seals = {name: hashlib.sha256(library.read_bytes()).hexdigest()}
            calls = []
            common = SimpleNamespace(verify_process=lambda h, p, s, c, e: calls.append(e))
            with patch('bert_sgemm_ncu_process.proc_environment',
                       return_value={b'NVTX_INJECTION64_PATH': b'libcuda-injection.so'}), \
                    patch('bert_sgemm_ncu_process.proc_mapped_paths', return_value=mapped):
                with authenticated_profile_processes(common, seals):
                    common.verify_process(None, SimpleNamespace(pid=1), 42, ['app'], {'CUDA_VISIBLE_DEVICES': '0'})
            self.assertNotIn('LD_PRELOAD', calls[0])


if __name__ == '__main__':
    unittest.main()
