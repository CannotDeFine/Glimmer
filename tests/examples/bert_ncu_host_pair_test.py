"""Cold runtime-selection tests; no CUDA, application or privileged launch."""

from copy import deepcopy
from contextlib import contextmanager, ExitStack
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import bert_ncu_host_pair as host
import bert_ncu_current_sdk as current_sdk
import bert_ncu_replay_exec as bridge
import bert_ncu_relay as relay
import bert_ncu_service_proxy as proxy
import run_bert_sgemm_ncu_capture as capture
from run_bert_sgemm_descriptor_probe import selected_service_target


class HostPairTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.runtime = self.root / 'artifacts/archives/execution-host-pair/attempt-fixture'
        self.runtime.mkdir(parents=True)
        variants = {}
        for variant in ('o0', 'o2'):
            directory = self.runtime / variant
            directory.mkdir()
            products = {}
            for name in ('glimmer_execution_service', 'glimmer_execution_interceptor'):
                path = directory / name
                path.write_text('CPU-only ' + variant + name)
                path.chmod(0o555)
                metadata = path.stat()
                products[name] = dict(path=str(path), mode=0o555, device=metadata.st_dev,
                                      inode=metadata.st_ino, size=metadata.st_size,
                                      ctime_ns=metadata.st_ctime_ns, sha256=bridge.seal(path))
            variants[variant] = products
        self.manifest = self.runtime / 'runtime.json'
        self.manifest.write_text(json.dumps(dict(variants=variants)))
        self.plan = self.root / 'plan.json'
        self.plan.write_text('{}')
        self.report = self.root / 'report.json'
        self.report.write_text(json.dumps(dict(variant_attempts={name: str(self.root / name)
                                                                for name in ('o0', 'o2')})))
        self.review = self.root / 'review.json'
        self.review.write_text(json.dumps(dict(status='fresh_host_pair_numerical_accepted',
                                               report_sha256=bridge.seal(self.report))))
        self.patch = patch.object(bridge, 'PROJECT', self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def binding(self, variant='o0'):
        products = json.loads(self.manifest.read_text())['variants'][variant]
        return dict(variant=variant, server=products['glimmer_execution_service']['path'],
                    frontend=products['glimmer_execution_interceptor']['path'],
                    runtime_manifest=str(self.manifest), runtime_manifest_sha256=bridge.seal(self.manifest),
                    plan=str(self.plan), plan_sha256=bridge.seal(self.plan),
                    numerical_report=str(self.report), numerical_report_sha256=bridge.seal(self.report),
                    numerical_review=str(self.review), numerical_review_sha256=bridge.seal(self.review),
                    native_reference=str(self.root / variant / 'native'))

    def test_fresh_recorded_inputs_preserve_historical_evidence_strength(self):
        hashed, metadata_only = self.root / 'hashed', self.root / 'metadata-only'
        hashed.write_text('authenticated')
        metadata_only.write_text('identity only')
        expected, observed = {}, {}
        for path, has_hash in ((hashed, True), (metadata_only, False)):
            info = path.stat()
            sha = bridge.seal(path)
            expected[str(path)] = dict(identity=[info.st_dev, info.st_ino, info.st_size,
                                                  info.st_mtime_ns, info.st_ctime_ns],
                                       resolved=str(path.resolve()), sha256=sha if has_hash else None)
            observed[str(path)] = dict(target=str(path.resolve()), bytes=info.st_size, sha256=sha)
        summary = current_sdk.validate_fresh_application_inputs(expected, observed)
        self.assertEqual(summary, dict(recorded_files=2, historical_sha256=1, historical_stat_only=1,
                                       complete_python_import_closure=False))
        altered = deepcopy(observed)
        altered[str(hashed)]['sha256'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'bytes changed'):
            current_sdk.validate_fresh_application_inputs(expected, altered)
        altered = deepcopy(observed)
        altered[str(metadata_only)]['target'] = '/different'
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            current_sdk.validate_fresh_application_inputs(expected, altered)

    def test_fresh_inventory_rejects_changed_loader_or_recorded_set(self):
        base = self.root / 'build/tally-full-implementation-1'
        workload = self.root / 'examples/framework_workloads/bert'
        venv = self.root / 'venv'
        training = SimpleNamespace(PYTHON=venv / 'bin/python', PACKAGES=venv / 'site-packages',
            ASSET_ROOT=self.root / 'assets', ORIGINAL=self.root / 'original',
            SERVER=self.root / 'service', PRELOAD=self.root / 'frontend')
        common = SimpleNamespace(STARTUP_INPUTS=self.root / 'startup', OLD_ONNX=self.root / 'onnx-result')
        roots = (training.SERVER, training.PRELOAD, training.PYTHON,
                 training.PACKAGES / 'nvidia/cudnn/lib/libcudnn.so.9',
                 training.PACKAGES / 'nvidia/nccl/lib/libnccl.so.2')
        exact = {workload / name for name in ('bert_contract.py', 'bert_workload.py', 'onnx_bert_workload.py',
                                               'export_bert.py')}
        exact.update(base / name for name in ('collect_application_block_colocation.py',
            'collect_owned_bert_training.py', 'collect_frontend_gpu.py', 'collect_service_gpu.py',
            'collect_dispatcher_gpu.py', 'native_command_audit.py', 'collect_priority_gpu.py',
            'arrival_notification.py', 'owned_child_cleanup.py', 'owned_onnx_result.py'))
        exact.update(self.root / 'scripts' / name for name in ('application_priority_audit.py',
            'execution_priority_audit.py', 'execution_semantic_audit.py', 'execution_application_receipt.py'))
        exact.update((*roots, common.OLD_ONNX, Path('/usr/bin/ldd'), Path('/lib64/ld-linux-x86-64.so.2')))
        package = venv / 'lib/site-package.py'
        output = '\t/lib64/ld-linux-x86-64.so.2 (0x123)\n'
        normalized = output.replace('0x123', '0xADDRESS')
        record = dict(inputs={str(path): dict(resolved=str(path), identity=[1, 2, 3, 4, 5], sha256='a' * 64)
                              for path in exact | {package}},
                      dynamic_libraries={str(path): normalized for path in roots})
        with patch.object(current_sdk.subprocess, 'check_output', return_value=output):
            paths, old, contract = current_sdk.fresh_application_inventory(
                common, SimpleNamespace(training=training), {package}, [record, deepcopy(record)], {})
            self.assertEqual(paths, exact | {package})
            self.assertEqual(old, record['inputs'])
            self.assertEqual(contract['recorded_files'], len(exact) + 1)
            self.assertEqual(len(contract['loader_sha256']), 5)
            audit_only = self.root / 'tests/example_test.py'
            again, _, _ = current_sdk.fresh_application_inventory(
                common, SimpleNamespace(training=training), {package, audit_only}, [record, record], {})
            self.assertEqual(again, paths)
            differing = deepcopy(record)
            differing['inputs'][str(package)]['identity'][1] += 1
            with self.assertRaisesRegex(ValueError, 'numerical application inputs differ'):
                current_sdk.fresh_application_inventory(common, SimpleNamespace(training=training),
                                                       {package}, [record, differing], {})
            missing = deepcopy(record)
            del missing['inputs'][str(base / 'owned_child_cleanup.py')]
            with self.assertRaisesRegex(ValueError, 'inventory differs'):
                current_sdk.fresh_application_inventory(common, SimpleNamespace(training=training),
                                                       {package}, [missing, missing], {})
            with self.assertRaisesRegex(ValueError, 'clean five-binary'):
                current_sdk.fresh_application_inventory(common, SimpleNamespace(training=training),
                                                       {package}, [record, record], {'LD_PRELOAD': '/untrusted'})
            changed = deepcopy(record)
            changed['dynamic_libraries'][str(roots[0])] = 'different'
            with self.assertRaisesRegex(ValueError, 'loader closure differs'):
                current_sdk.fresh_application_inventory(common, SimpleNamespace(training=training),
                                                       {package}, [record, changed], {})
            with self.assertRaisesRegex(ValueError, 'inventory differs'):
                current_sdk.fresh_application_inventory(common, SimpleNamespace(training=training),
                                                       {package, venv / 'lib/extra.py'}, [record, record], {})

    def test_explicit_variants_and_legacy_default(self):
        self.assertEqual(bridge.selected_service({}), bridge.SERVICE)
        for variant in ('o0', 'o2'):
            binding = self.binding(variant)
            selected = bridge.selected_service(dict(uid=os.getuid(), host_pair=binding))
            self.assertEqual(selected, Path(binding['server']))
            host.check_binding(binding)

    def test_reject_cross_variant_runtime_frontend_and_native(self):
        for key in ('server', 'frontend', 'native_reference'):
            binding = self.binding()
            binding[key] = self.binding('o2')[key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                host.check_binding(binding)
        binding = self.binding()
        binding['variant'] = 'o2'
        with self.assertRaises(ValueError):
            bridge.selected_service(dict(uid=os.getuid(), host_pair=binding))

    def test_reject_missing_selection_and_changed_plan(self):
        for key in ('variant', 'server', 'runtime_manifest', 'runtime_manifest_sha256'):
            binding = self.binding()
            del binding[key]
            with self.subTest(key=key), self.assertRaises((ValueError, KeyError)):
                bridge.selected_service(dict(uid=os.getuid(), host_pair=binding))
        binding = self.binding()
        self.plan.write_text('{"changed": true}')
        with self.assertRaises(ValueError):
            host.check_binding(binding)

    def test_reject_hash_path_uid_and_mutable_binary(self):
        for key, value in (('server', str(bridge.SERVICE)), ('runtime_manifest_sha256', '0' * 64),
                           ('variant', 'release')):
            binding = self.binding()
            binding[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                bridge.selected_service(dict(uid=os.getuid(), host_pair=binding))
        binding = self.binding()
        with self.assertRaises(ValueError):
            bridge.selected_service(dict(uid=os.getuid() + 1, host_pair=binding))
        Path(binding['server']).chmod(0o755)
        with self.assertRaises(ValueError):
            bridge.selected_service(dict(uid=os.getuid(), host_pair=binding))

    def test_service_recipe_and_proxy_reject_wrong_variant(self):
        selected = Path(self.binding()['server'])
        request = dict(clients=dict(low=dict(pid=11, start=101), high=dict(pid=12, start=102)),
                       socket='/private/service.sock')
        command = bridge.service_command(request, service=selected)
        self.assertEqual(command[0], str(selected))
        observed = proxy.service_request(command, self.root, service=selected)
        self.assertEqual(observed['clients'], request['clients'])
        for bad in (bridge.SERVICE, Path(self.binding('o2')['server'])):
            changed = deepcopy(command)
            changed[0] = str(bad)
            with self.assertRaises(ValueError):
                proxy.service_request(changed, self.root, service=selected)
        with self.assertRaises(ValueError):
            proxy.service_request(command + ['--unexpected'], self.root, service=selected)

    def test_fresh_signal_recipe_is_explicit_and_preserves_old_recipe(self):
        request = dict(clients=dict(low=dict(pid=11, start=101), high=dict(pid=12, start=102)),
                       socket='/private/service.sock')
        old = bridge.service_command(request)
        scalar = dict(schema='fresh_runtime_v2', variant='scalar')
        sharded = dict(schema='fresh_runtime_v2', variant='sharded')
        flag = '--experimental-sharded-gemm-signal'
        self.assertEqual(bridge.service_command(request, signal_binding=scalar), old)
        self.assertEqual(bridge.service_command(request, signal_binding=sharded), old + [flag])
        self.assertNotIn(flag, old)
        for bad in ({}, {'schema': 'fresh_runtime_v2'}, {'schema': 'current_sdk_v1', 'variant': 'sharded'},
                    {'schema': 'fresh_runtime_v2', 'variant': 'other'},
                    {'schema': 'fresh_runtime_v2', 'variant': True}, 'sharded'):
            with self.subTest(binding=bad), self.assertRaises(ValueError):
                bridge.service_command(request, signal_binding=bad)

    def test_fresh_selector_is_not_a_client_request_field(self):
        request = dict(clients=dict(low=dict(pid=11, start=101), high=dict(pid=12, start=102)),
                       socket='/private/service.sock')
        scalar = dict(schema='fresh_runtime_v2', variant='scalar')
        sharded = dict(schema='fresh_runtime_v2', variant='sharded')
        commands = {variant: bridge.service_command(request, signal_binding=binding)
                    for variant, binding in (('scalar', scalar), ('sharded', sharded))}
        for variant, binding in (('scalar', scalar), ('sharded', sharded)):
            observed = proxy.service_request(commands[variant], self.root, signal_binding=binding)
            self.assertEqual(set(observed), {'directory', 'socket', 'clients'})
            self.assertEqual(observed['clients'], request['clients'])
        with self.assertRaises(ValueError):
            proxy.service_request(commands['scalar'], self.root, signal_binding=sharded)
        with self.assertRaises(ValueError):
            proxy.service_request(commands['sharded'], self.root, signal_binding=scalar)
        with self.assertRaises(ValueError):
            proxy.service_request(commands['sharded'] + ['--experimental-sharded-gemm-signal'],
                                  self.root, signal_binding=sharded)
        with self.assertRaises(ValueError):
            proxy.service_request(commands['sharded'], self.root)

    def test_fresh_selector_requires_one_frozen_runtime_and_is_not_yet_launchable(self):
        binding = dict(schema='fresh_runtime_v2', variant='sharded')
        template = dict(fresh_runtime=binding)
        self.assertEqual(bridge.selected_signal_binding(template), binding)
        self.assertIsNone(bridge.selected_signal_binding({}))
        with self.assertRaisesRegex(ValueError, 'not implemented'):
            bridge.selected_service(template)
        for other in ('host_pair', 'current_sdk'):
            with self.subTest(other=other), self.assertRaises(ValueError):
                bridge.selected_signal_binding(dict(template, **{other: {}}))
        with self.assertRaises(ValueError):
            bridge.selected_signal_binding(dict(fresh_runtime=None))

    def test_fresh_product_rejects_changed_identity_and_bytes(self):
        path = self.root / 'service'
        path.write_bytes(b'fixed service fixture')
        info = path.stat()
        product = dict(path=str(path), device=info.st_dev, inode=info.st_ino, size=info.st_size,
                       mtime_ns=info.st_mtime_ns, ctime_ns=info.st_ctime_ns,
                       mode=info.st_mode & 0o7777, uid=info.st_uid, gid=info.st_gid, sha256=bridge.seal(path))
        self.assertEqual(bridge.validate_runtime_product(product, os.getuid(), os.getgid()), path)
        for key, value in (('device', info.st_dev + 1), ('inode', info.st_ino + 1),
                           ('size', info.st_size + 1), ('mtime_ns', info.st_mtime_ns + 1),
                           ('ctime_ns', info.st_ctime_ns + 1), ('mode', 0),
                           ('sha256', '0' * 64), ('inode', True), ('gid', info.st_gid + 1)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                bridge.validate_runtime_product(dict(product, **{key: value}), os.getuid(), os.getgid())
        with self.assertRaises(ValueError):
            bridge.validate_runtime_product(dict(product, extra=True), os.getuid(), os.getgid())
        with self.assertRaises(ValueError):
            bridge.validate_runtime_product(product, os.getuid() + 1, os.getgid())
        link = self.root / 'link'
        link.symlink_to(path)
        with self.assertRaises(ValueError):
            bridge.validate_runtime_product(dict(product, path=str(link)), os.getuid(), os.getgid())
        with patch.object(bridge.os, 'read', side_effect=lambda descriptor, count: b'x' * count) as reads:
            with self.assertRaisesRegex(ValueError, 'grew during verification'):
                bridge.validate_runtime_product(product, os.getuid(), os.getgid())
            self.assertEqual(reads.call_count, 2)
        with patch.object(bridge.os, 'read', return_value=b''):
            with self.assertRaisesRegex(ValueError, 'ended during verification'):
                bridge.validate_runtime_product(product, os.getuid(), os.getgid())
        original_memfd_create = os.memfd_create
        created = []

        def remember_memfd(*args):
            descriptor = original_memfd_create(*args)
            created.append(descriptor)
            return descriptor

        with patch.object(bridge.os, 'memfd_create', side_effect=remember_memfd), \
                patch.object(bridge.os, 'read', side_effect=OSError('injected read failure')):
            with self.assertRaisesRegex(OSError, 'injected read failure'):
                bridge.snapshot_runtime_product(product, os.getuid(), os.getgid())
        self.assertEqual(len(created), 1)
        with self.assertRaises(OSError):
            os.fstat(created[0])
        created.clear()
        with patch.object(bridge.os, 'memfd_create', side_effect=remember_memfd), \
                patch.object(bridge.fcntl, 'fcntl', side_effect=OSError('injected seal failure')):
            with self.assertRaisesRegex(OSError, 'injected seal failure'):
                bridge.snapshot_runtime_product(product, os.getuid(), os.getgid())
        self.assertEqual(len(created), 1)
        with self.assertRaises(OSError):
            os.fstat(created[0])
        with patch.object(bridge, '_read_runtime_product'):
            with self.assertRaisesRegex(ValueError, 'snapshot bytes differ'):
                bridge.snapshot_runtime_product(product, os.getuid(), os.getgid())
        descriptor = bridge.snapshot_runtime_product(product, os.getuid(), os.getgid())
        try:
            path.write_bytes(b'changed same-length!!')
            self.assertEqual(path.stat().st_size, info.st_size)
            bridge.validate_snapshot(descriptor, product['sha256'])
            with self.assertRaises(ValueError):
                bridge.validate_runtime_product(product, os.getuid(), os.getgid())
            with self.assertRaises(ValueError):
                bridge.snapshot_runtime_product(product, os.getuid(), os.getgid())
        finally:
            os.close(descriptor)
        replacement = self.root / 'replacement'
        replacement.write_bytes(b'fixed service fixture')
        os.replace(replacement, path)
        self.assertNotEqual(path.stat().st_ino, info.st_ino)
        with self.assertRaises(ValueError):
            bridge.snapshot_runtime_product(product, os.getuid(), os.getgid())

    @patch.object(bridge.os, 'geteuid', return_value=0)
    @patch.object(bridge.os, 'getegid', return_value=0)
    def test_service_snapshot_uses_only_the_frozen_fresh_product(self, _gid, _uid):
        service = self.root / 'service'
        service.write_bytes(b'fixed service fixture')
        service.chmod(0o775)
        info = service.stat()
        product = dict(path=str(service), device=info.st_dev, inode=info.st_ino, size=info.st_size,
                       mtime_ns=info.st_mtime_ns, ctime_ns=info.st_ctime_ns, mode=0o775,
                       uid=info.st_uid, gid=info.st_gid, sha256=bridge.seal(service))
        environment = {'PATH': '/fixed'}
        contract = dict(schema='fresh_runtime_sgemm_v2', variant='scalar', target='2:3127', max_seconds=800,
                        deadline_utc='future', runtime=dict(service=product, frontend={}),
                        service_environment=environment, numerical={}, reference={})
        binding = dict(schema='fresh_runtime_v2', variant='scalar', contract=contract, plan='/reviewed-plan',
                       plan_sha256='a' * 64, review='/reviewed-record', review_sha256='b' * 64)
        template = dict(fresh_runtime=binding, sgemm_target='2:3127', service_cap_sys_admin=True,
                        service_environment=environment, bridge_seals={str(service): product['sha256']},
                        uid=os.getuid(), gid=os.getgid())
        authority = dict(claim_sha256='c' * 64, manifest_sha256='d' * 64, binding_sha256='e' * 64,
                         binding=binding, template=template)
        plan = dict(uid=os.getuid(), gid=os.getgid(), environment=environment, seals={str(service): product['sha256']})
        descriptor = bridge.service_executable_snapshot(template, plan, service, fresh_authority=authority)
        try:
            bridge.validate_snapshot(descriptor, product['sha256'])
            self.assertEqual(os.fstat(descriptor).st_size, info.st_size)
            self.assertEqual(os.fstat(descriptor).st_mode & 0o777, 0o555)
        finally:
            os.close(descriptor)
        ordinary_contract = dict(contract, schema='fresh_runtime_sgemm_v3', specialization='ordinary')
        ordinary_binding = dict(binding, schema='fresh_runtime_v3', contract=ordinary_contract)
        ordinary_template = dict(template, fresh_runtime=ordinary_binding)
        ordinary_authority = dict(authority, binding=ordinary_binding, template=ordinary_template)
        descriptor = bridge.service_executable_snapshot(ordinary_template, plan, service,
                                                        fresh_authority=ordinary_authority)
        try:
            bridge.validate_snapshot(descriptor, product['sha256'])
        finally:
            os.close(descriptor)
        for key, value in (('mode', 0o755), ('sha256', '0' * 64), ('uid', info.st_uid + 1),
                           ('path', '/other/service')):
            changed = deepcopy(template)
            changed['fresh_runtime']['contract']['runtime']['service'][key] = value
            with self.subTest(field=key), self.assertRaises(ValueError):
                bridge.service_executable_snapshot(changed, plan, service, fresh_authority=authority)
        for changed in (dict(plan, environment={'PATH': '/different'}),
                        dict(plan, seals={str(service): '0' * 64}), dict(plan, uid=plan['uid'] + 1)):
            with self.assertRaises(ValueError):
                bridge.service_executable_snapshot(template, changed, service, fresh_authority=authority)
        for legacy in ({}, {'host_pair': {}}, {'current_sdk': {}}):
            with self.subTest(legacy=legacy), patch.object(bridge, 'sealed_snapshot', return_value=91) as old:
                self.assertEqual(bridge.service_executable_snapshot(legacy, plan, service), 91)
                old.assert_called_once_with(service, product['sha256'])

    def test_fresh_runtime_must_match_both_numerical_products(self):
        products, inputs = {}, {}
        for name in ('service', 'frontend'):
            path = self.root / name
            path.write_text('numerical fixture ' + name)
            info = path.stat()
            product = dict(path=str(path), device=info.st_dev, inode=info.st_ino, size=info.st_size,
                           mtime_ns=info.st_mtime_ns, ctime_ns=info.st_ctime_ns,
                           mode=info.st_mode & 0o7777, uid=info.st_uid, gid=info.st_gid, sha256=bridge.seal(path))
            products[name] = product
            inputs[str(path)] = dict(resolved=str(path), sha256=product['sha256'],
                                     identity=[info.st_dev, info.st_ino, info.st_size,
                                               info.st_mtime_ns, info.st_ctime_ns])
        selected = bridge.validate_numerical_runtime(products, inputs, os.getuid(), os.getgid())
        self.assertEqual(selected, {name: Path(item['path']) for name, item in products.items()})
        for name in ('service', 'frontend'):
            for key, value in (('resolved', '/wrong'), ('sha256', '0' * 64), ('identity', [])):
                changed = deepcopy(inputs)
                changed[products[name]['path']][key] = value
                with self.subTest(product=name, key=key), self.assertRaises(ValueError):
                    bridge.validate_numerical_runtime(products, changed, os.getuid(), os.getgid())
        with self.assertRaises(ValueError):
            bridge.validate_numerical_runtime(dict(products, unknown={}), inputs, os.getuid(), os.getgid())
        with self.assertRaises(ValueError):
            bridge.validate_numerical_runtime(dict(service=products['service'], frontend=products['service']),
                                              inputs, os.getuid(), os.getgid())

    def test_fresh_numerical_receipt_and_tampering(self):
        stage = self.root / 'artifacts/maintenance/fixture'
        root = stage / 'runs/attempt-one'
        root.mkdir(parents=True)
        names = ('report.json', 'manifest.json', 'batch-publication.json', 'native-worker.json',
                 'equal-worker.json', 'review.md', 'native/attempt.json', 'equal/attempt.json',
                 'equal/server.log', 'native-unit.log', 'equal-unit.log')
        for name in names:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        unit_log = 'Finished with result: success\nMain processes terminated with: code=exited/status=0\n'
        for mode in ('native', 'equal'):
            (root / (mode + '-unit.log')).write_text(unit_log)
        sha = 'a' * 64
        paths = {name: stage / name for name in ('plan', 'launch', 'runner')}
        audit_path = stage / 'official_bert_provider_audit.py'
        claim = root.parent / 'claim.json'
        for path in (*paths.values(), audit_path, claim):
            path.touch()
        provenance = {name: dict(path=str(path), sha256=sha) for name, path in paths.items()}
        provenance['auditor'] = dict(path=str(audit_path), sha256=sha)
        runtime = dict(service=dict(path='/fixed/service'), frontend=dict(path='/fixed/frontend', device=1, inode=2))
        clients = dict(low=dict(pid=11, start=101), high=dict(pid=12, start=102))
        socket = '/private/service.sock'
        command = bridge.service_command(dict(clients=clients, socket=socket), service=Path(runtime['service']['path']),
                                         signal_binding=dict(schema='fresh_runtime_v2', variant='scalar'))
        for option in ('--audit-sgemm-target-client', '--audit-sgemm-range-library'):
            index = command.index(option)
            del command[index:index + 2]
        environment = dict(LD_PRELOAD=runtime['frontend']['path'], GLIMMER_EXECUTION_HOST_RANGE_SEEK='0')
        equal = dict(mode='equal', server_command=command, socket_path=socket, application_identities=clients,
                     server_environment={'PATH': '/fixture'},
                     server_identity=dict(pid=13, start=103),
                     application_environment=environment, application_environments=dict(low=environment, high=environment),
                     service_configuration=dict(scheduler='priority', kernel_mode='preemptive', configured_workers=320,
                                                nccl_enabled=False, sharded_gemm_signal=False, full_device_qualified=False),
                     artifact_identities={'server.log': sha})
        native = dict(mode='native', application_environment={}, application_environments=dict(low={}, high={}),
                      application_identities=clients,
                      artifact_identities=dict.fromkeys(('training/result.json', 'inference/result.json'), sha))
        provider = dict(configured_sessions=[2], completed_sgemm=438, unique_keys=438, workers=320,
                        admission_status={'true': 438}, single_physical_launch_per_sgemm=True)
        maps = {str(item['pid']): dict(pid=item['pid'], start=item['start'], path=runtime['frontend']['path'], device=1,
                                       inode=2, maps=5, seek='0') for item in clients.values()}
        memory = {'memory.events': 'low 0\nhigh 0\nmax 0\noom 0\noom_kill 0',
                  'memory.max': str(16 * 1024**3), 'memory.swap.max': '0', 'memory.oom.group': '1'}
        def resources(mode):
            unit = 'glimmer-bert-' + root.name + '-' + mode + '.service'
            group = '/task/' + unit
            identities = list(clients.values()) + ([equal['server_identity']] if mode == 'equal' else [])
            return dict(initial_memory=memory, last_memory=memory, resource_checks=1, observer={'cgroup': group},
                        participants=[dict(item, cgroup=group) for item in identities],
                        last_task_and_swap={'memory.swap.current': '0', 'pids.events': 'max 0', 'pids.max': '256'})
        workers = {mode: dict(status='passed_requires_review', mode=mode, scoped_publication_valid=False,
                              reparse=current_sdk.ORACLE.copy(), job=job, resources=resources(mode))
                   for mode, job in (('native', native), ('equal', equal))}
        workers['equal'].update(provider=provider, loaded_mappings=maps)
        inputs = {str(path): {'sha256': sha} for path in (*paths.values(), audit_path)}
        manifest = dict(inputs=inputs, gpu='GPU-fixture', plan_sha256=sha, launch_sha256=sha)
        active = dict(LoadState='loaded', ActiveState='active', RuntimeMaxUSec='800s', TimeoutStopUSec='5s',
                      MemoryMax=str(16 * 1024**3), MemorySwapMax='0', TasksMax='256',
                      LimitFSIZE=str(1024**3), NoNewPrivileges='yes', KillMode='control-group')
        terminal = dict(LoadState='not-found', ActiveState='inactive', MainPID='0')
        rows = [dict(mode=mode, unit='glimmer-bert-' + root.name + '-' + mode + '.service',
                     status='passed_requires_review', result=str(root / (mode + '-worker.json')),
                     active_properties=dict(active, MainPID='1', Description='Glimmer fresh numerical ' + root.name + ' ' + mode),
                     terminal_properties=terminal, reparse=current_sdk.ORACLE.copy(),
                     command=['--property=OOMPolicy=kill', 'python', str(paths['runner']), '--worker'])
                for mode in ('native', 'equal')]
        report = dict(status='completed_requires_review', diagnostic_only=True, hardware_counters=False,
                      full_device_qualified=False, performance_qualified=False, priority_qualified=False,
                      saturation_qualified=False, application_qualified=False, final_gpu_processes=[],
                      inputs=inputs, gpu_identity='GPU-fixture', resources_before={}, resources_after={},
                      manifest_sha256=sha, batch_publication_path=str(root / 'batch-publication.json'),
                      claim=str(claim), jobs=rows)
        publication = dict(report_sha256=sha, manifest_sha256=sha, claim_sha256=sha,
                           workers=dict(native=sha, equal=sha))
        records = {str(root / 'report.json'): report, str(root / 'manifest.json'): manifest,
                   str(root / 'batch-publication.json'): publication, str(claim):
                   dict(attempt=str(root), plan_sha256=sha, launch_sha256=sha, order=['native', 'equal']),
                   str(paths['launch']): dict(plan_sha256=sha, runner_sha256=sha)}
        for name in ('training/result.json', 'inference/result.json'):
            path = root / 'native' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            records[str(path)] = dict.fromkeys(current_sdk.RUN_TIMING_FIELDS, 1)
            records[str(path)]['status'] = 'passed'
        for mode in ('native', 'equal'):
            records[str(root / (mode + '-worker.json'))] = workers[mode]
            records[str(root / mode / 'attempt.json')] = workers[mode]['job']
        other_sgemms = ''.join(
            f'Native admission audit: session=2, sequence={i}, accepted=true, operation=30\n'
            f'Native persistent audit: session=2, sequence={i}, total_blocks=1, physical_launches=1, workers=320\n'
            for i in range(1, 438))
        log = ('Execution service configuration: scheduler=priority, kernel_mode=preemptive, '
               'configured_workers=320, nccl_enabled=false, sharded_gemm_signal=false; full_device_qualified=false\n'
               + other_sgemms +
               'Native admission audit: session=2, sequence=3127, accepted=true, operation=30\n'
               'Native persistent audit: session=2, sequence=3127, total_blocks=2304, physical_launches=1, workers=320\n'
               'Native issue audit: session=2, sequence=3127, status=0\n'
               'Native execution audit: session=2, issued=3127, retained=0, retired=1, status=0\n'
               'Priority execution audit: event=low_logical_done, session=2, priority=0, sequence=3127, '
               'completed_blocks=2304, total_blocks=2304, physical_launches=1, status=0\n')
        (root / 'equal/server.log').write_text(log)
        seals = dict.fromkeys(names, sha)
        spec = SimpleNamespace(loader=SimpleNamespace(exec_module=Mock()))
        auditor = SimpleNamespace(audit=Mock(return_value=provider))
        with patch.object(current_sdk, 'read', side_effect=lambda path: records[str(path)]), \
                patch.object(current_sdk, 'digest', return_value=sha), \
                patch.object(current_sdk.candidate, 'normal_cleanup'), \
                patch.object(current_sdk, 'validate_fresh_applications'), \
                patch.object(current_sdk, 'parse_semantic_audit'), \
                patch.object(current_sdk, 'authenticated_training_sessions', return_value={2}), \
                patch.object(current_sdk.importlib.util, 'spec_from_file_location', return_value=spec), \
                patch.object(current_sdk.importlib.util, 'module_from_spec', return_value=auditor), \
                patch.object(bridge, 'validate_numerical_runtime'):
            args = (root, seals, 'scalar', runtime, provenance, os.getuid(), os.getgid())
            self.assertIs(current_sdk.fresh_numerical_evidence(*args)['numerical_only'], True)
            with self.assertRaises(ValueError):
                current_sdk.fresh_numerical_evidence(root, seals, 'sharded', runtime, provenance, os.getuid(), os.getgid())
            for change, restore in ((lambda: maps['11'].update(inode=3), lambda: maps['11'].update(inode=2)),
                                    (lambda: maps['11'].update(maps=1), lambda: maps['11'].update(maps=5)),
                                    (lambda: provider.update(completed_sgemm=437), lambda: provider.update(completed_sgemm=438)),
                                    (lambda: provider.update(single_physical_launch_per_sgemm=False),
                                     lambda: provider.update(single_physical_launch_per_sgemm=True)),
                                    (lambda: publication['workers'].update(equal='0' * 64),
                                     lambda: publication['workers'].update(equal=sha))):
                change()
                with self.assertRaises(ValueError):
                    current_sdk.fresh_numerical_evidence(*args)
                restore()
            (root / 'equal/server.log').write_text(log.replace('sequence=3127', 'sequence=3128'))
            with self.assertRaises(ValueError):
                current_sdk.fresh_numerical_evidence(*args)
            (root / 'equal/server.log').write_text(log.replace('sharded_gemm_signal=false', 'sharded_gemm_signal=true'))
            command.append('--experimental-sharded-gemm-signal')
            equal['service_configuration']['sharded_gemm_signal'] = True
            del provider['single_physical_launch_per_sgemm']
            rows[1]['active_properties'].update(ActiveState='inactive', MainPID='0')
            result = current_sdk.fresh_numerical_evidence(root, seals, 'sharded', runtime, provenance,
                                                          os.getuid(), os.getgid())
            self.assertIs(result['active_samples']['equal'], False)
            workers['equal']['resources']['participants'][0]['cgroup'] = '/foreign/unit'
            with self.assertRaises(ValueError):
                current_sdk.fresh_numerical_evidence(root, seals, 'sharded', runtime, provenance,
                                                     os.getuid(), os.getgid())

    def test_fresh_application_requires_same_run_reference_and_environment(self):
        root = self.root / 'attempt-one'
        project = bridge.PROJECT
        python = str(project / 'examples/framework_workloads/pytorch_smoke/.venv/bin/python')
        workload = project / 'examples/framework_workloads/bert'
        assets = str(project / 'build/bert-intake-assets-3')
        startup = self.root / 'startup'
        manifest = dict(startup_reference_bundle=str(startup), gpu='GPU-fixture,V100')
        base = dict.fromkeys(current_sdk.pair.BASE_ENVIRONMENT_KEYS, 'fixture')
        base.update(CUDA_VISIBLE_DEVICES='GPU-fixture', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                    OPENBLAS_NUM_THREADS='2')
        jobs = {}
        for mode in ('native', 'equal'):
            directory = root / mode
            reference = startup / 'training' if mode == 'native' else root / 'native/training'
            low = [python, '-B', str(workload / 'bert_workload.py'), '--assets', assets,
                   '--output', str(directory / 'training'), '--role', 'training', '--warmup', '1',
                   '--iterations', '2', '--snapshot', '--reference', str(reference),
                   '--ready-file', str(directory / 'low.ready.json'), '--start-file', str(directory / 'low.start')]
            high = [python, '-B', str(workload / 'onnx_bert_workload.py'), '--assets', assets,
                    '--export', str(startup / 'export'), '--output', str(directory / 'inference'),
                    '--warmup', '1', '--iterations', '2', '--ready-file', str(directory / 'high.ready.json'),
                    '--start-file', str(directory / 'high.start')]
            env = dict(base)
            if mode == 'equal':
                env.update(LD_PRELOAD='/fixed/frontend', GLIMMER_EXECUTION_SOCKET='/socket',
                           GLIMMER_EXECUTION_SERVER_PID='7', GLIMMER_EXECUTION_SERVER_START='70',
                           GLIMMER_EXECUTION_PREPARATION_TIMEOUT_MS='240000', GLIMMER_EXECUTION_HOST_RANGE_SEEK='0')
            jobs[mode] = dict(application_commands=dict(low=low, high=high), application_environment=dict(env),
                              application_environments=dict(low=dict(env), high=dict(env)))
        jobs['equal'].update(server_environment=base, socket_path='/socket', server_identity=dict(pid=7, start=70))
        artifacts = ('training/result.json', 'training/snapshot.pt', 'inference/result.json')
        jobs['native']['artifact_identities'] = {name: 'fixture-' + name for name in artifacts}
        references = {str(root / 'native' / name): dict(target=str(root / 'native' / name),
                      sha256=jobs['native']['artifact_identities'][name]) for name in artifacts}
        jobs['equal'].update(reference_before=references, reference_after=deepcopy(references))
        current_sdk.validate_fresh_applications(root, manifest, jobs['native'], jobs['equal'])
        for role, option, value in (('low', '--reference', '/wrong'), ('high', '--iterations', '3')):
            changed = deepcopy(jobs['equal'])
            command = changed['application_commands'][role]
            command[command.index(option) + 1] = value
            with self.subTest(role=role), self.assertRaises(ValueError):
                current_sdk.validate_fresh_applications(root, manifest, jobs['native'], changed)
        changed = deepcopy(jobs['equal'])
        changed['application_environments']['low']['GLIMMER_EXECUTION_HOST_RANGE_SEEK'] = '1'
        with self.assertRaises(ValueError):
            current_sdk.validate_fresh_applications(root, manifest, jobs['native'], changed)
        changed = deepcopy(jobs['equal'])
        changed['reference_before'][str(root / 'native/inference/result.json')]['sha256'] = 'wrong'
        changed['reference_after'] = deepcopy(changed['reference_before'])
        with self.assertRaises(ValueError):
            current_sdk.validate_fresh_applications(root, manifest, jobs['native'], changed)

    def test_fresh_pair_requires_matching_products_and_non_run_inputs(self):
        shared = dict(gpu='GPU-fixture', startup_reference_bundle='/startup',
                      startup_reference_report_sha256='a', dynamic_libraries={'cuda': 'fixture'},
                      inputs={'/model': {'sha256': 'model'}, '/scalar-plan': {}, '/sharded-plan': {}})
        native = dict(artifact_identities={'training/snapshot.pt': 'same-snapshot'})
        projection = {'training/result.json': {'status': 'passed'}}
        scalar = dict(variant='scalar', numerical_only=True, runtime={'service': 'same'}, manifest=deepcopy(shared),
                      service_environment={'PATH': '/fixed'},
                      provenance={'plan': {'path': '/scalar-plan'}}, native_job=deepcopy(native),
                      native_result_projection=deepcopy(projection))
        sharded = dict(variant='sharded', numerical_only=True, runtime={'service': 'same'}, manifest=deepcopy(shared),
                       service_environment={'PATH': '/fixed'},
                       provenance={'plan': {'path': '/sharded-plan'}}, native_job=deepcopy(native),
                       native_result_projection=deepcopy(projection))
        result = current_sdk.match_fresh_numerical_pair(scalar, sharded)
        self.assertIs(result['same_runtime_and_inputs'], True)
        self.assertIs(result['full_device_qualified'], False)
        self.assertEqual(result['service_environment'], {'PATH': '/fixed'})
        result['runtime']['service'] = 'changed'
        self.assertEqual(scalar['runtime']['service'], 'same')
        for section, key, value in (('runtime', 'service', 'different'),
                                    ('manifest', 'gpu', 'other'),
                                    ('manifest', 'dynamic_libraries', {})):
            changed = deepcopy(sharded)
            changed[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                current_sdk.match_fresh_numerical_pair(scalar, changed)
        changed = deepcopy(sharded)
        changed['manifest']['inputs']['/model']['sha256'] = 'different'
        with self.assertRaises(ValueError):
            current_sdk.match_fresh_numerical_pair(scalar, changed)
        changed = deepcopy(sharded)
        changed['service_environment']['PATH'] = '/other'
        with self.assertRaises(ValueError):
            current_sdk.match_fresh_numerical_pair(scalar, changed)
        changed = deepcopy(sharded)
        changed['native_job']['artifact_identities']['training/snapshot.pt'] = 'different-snapshot'
        with self.assertRaises(ValueError):
            current_sdk.match_fresh_numerical_pair(scalar, changed)
        changed = deepcopy(sharded)
        changed['native_result_projection']['training/result.json']['status'] = 'failed'
        with self.assertRaises(ValueError):
            current_sdk.match_fresh_numerical_pair(scalar, changed)

    def test_fresh_plan_is_exact_reviewed_and_bounded(self):
        deadline = datetime.now(timezone.utc) + timedelta(hours=1)
        evidence = dict(root='/evidence', seals={},
                        provenance=dict(plan={}, launch={}, runner={}, auditor={}))
        reference = dict(path='/reference', targets={name: dict(target='/target', sha256='a' * 64) for name in
                         ('training/result.json', 'training/snapshot.pt', 'inference/result.json')})
        contract = dict(schema='fresh_runtime_sgemm_v2', variant='scalar', target='2:3127', max_seconds=800,
                        deadline_utc=deadline.isoformat(), runtime=dict(service={}, frontend={}),
                        service_environment=dict.fromkeys(bridge.ENV_KEYS, 'fixture'),
                        numerical=dict(scalar=deepcopy(evidence), sharded=deepcopy(evidence)), reference=reference)
        plan = dict(contract, review=dict(path='/review', sha256='b' * 64))
        review = dict(status='fresh_runtime_v2_ready', contract=deepcopy(contract), source_sha256={'/source': 'c' * 64})
        self.assertEqual(current_sdk.validate_fresh_contract(plan, review, 'scalar'), contract)
        ordinary = dict(contract, schema='fresh_runtime_sgemm_v3', specialization='ordinary')
        ordinary_plan = dict(ordinary, review=plan['review'])
        ordinary_review = dict(review, status='fresh_runtime_v3_ready', contract=deepcopy(ordinary))
        self.assertEqual(current_sdk.validate_fresh_contract(ordinary_plan, ordinary_review, 'scalar'), ordinary)
        for changed in (dict(ordinary_plan, specialization='observed'),
                        {key: value for key, value in ordinary_plan.items() if key != 'specialization'},
                        dict(ordinary_plan, schema='fresh_runtime_sgemm_v2')):
            with self.assertRaises((KeyError, ValueError)):
                current_sdk.validate_fresh_contract(changed, ordinary_review, 'scalar')
        for change in (lambda x: x.update(variant='sharded'), lambda x: x.update(target='2:3128'),
                       lambda x: x.update(max_seconds=801), lambda x: x.update(unknown=True),
                       lambda x: x.update(deadline_utc=(datetime.now(timezone.utc) + timedelta(seconds=30)).isoformat())):
            changed = deepcopy(plan)
            change(changed)
            with self.assertRaises(ValueError):
                current_sdk.validate_fresh_contract(changed, review, 'scalar')
        changed = deepcopy(review)
        changed['contract']['runtime']['service']['path'] = '/different'
        with self.assertRaises(ValueError):
            current_sdk.validate_fresh_contract(plan, changed, 'scalar')
        short = deepcopy(plan)
        short['deadline_utc'] = (datetime.now(timezone.utc) + timedelta(seconds=10)).isoformat()
        short_review = dict(review, contract={key: short[key] for key in current_sdk.FRESH_CONTRACT_FIELDS})
        self.assertEqual(current_sdk.validate_fresh_contract(short, short_review, 'scalar', minimum_remaining=0),
                         short_review['contract'])
        with self.assertRaises(ValueError):
            current_sdk.validate_fresh_contract(short, short_review, 'scalar')
        short['deadline_utc'] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        short_review['contract']['deadline_utc'] = short['deadline_utc']
        with self.assertRaises(ValueError):
            current_sdk.validate_fresh_contract(short, short_review, 'scalar', minimum_remaining=0)

    def test_fresh_replay_configuration_binds_runtime_and_signal(self):
        service, frontend = Path('/fixed/service'), Path('/fixed/frontend')
        contract = dict(runtime=dict(service={'path': str(service), 'device': 1, 'inode': 2},
                                     frontend={'path': str(frontend), 'device': 1, 'inode': 3}))
        for variant in ('scalar', 'sharded'):
            common = SimpleNamespace(PREFIX=r'[ ]*')
            h = SimpleNamespace(training=SimpleNamespace(SERVER=service, PRELOAD=frontend))
            binding = dict(variant=variant)
            with self.subTest(variant=variant), \
                    patch.object(current_sdk, '_check_fresh_binding_metadata', return_value=(contract, {})) as metadata, \
                    patch.object(bridge, 'validate_runtime_product', side_effect=(service, frontend)), \
                    patch.object(current_sdk.pair.numerical, 'configure', return_value=(common, h)) as configure:
                self.assertEqual(current_sdk.configure_fresh(binding), (common, h))
                metadata.assert_called_once_with(binding, minimum_remaining=0)
                configure.assert_called_once_with(dict(server=str(service), frontend=str(frontend),
                    runtime_environment={'GLIMMER_EXECUTION_HOST_RANGE_SEEK': '0'}))
                signal = str(variant == 'sharded').lower()
                line = ('Execution service configuration: scheduler=priority, kernel_mode=preemptive, '
                        'configured_workers=320, nccl_enabled=false, sharded_gemm_signal=' + signal +
                        '; full_device_qualified=false')
                self.assertIs(common.validate_service_configuration(line)['sharded_gemm_signal'], variant == 'sharded')
                for bad in (line.replace('sharded_gemm_signal=' + signal,
                                         'sharded_gemm_signal=' + str(variant != 'sharded').lower()),
                            line.replace('configured_workers=320', 'configured_workers=80'), line + '\n' + line):
                    with self.assertRaises(ValueError):
                        common.validate_service_configuration(bad)
        status = 'CapInh:\t0\nCapPrm:\t1\nCapEff:\t1\nCapAmb:\t0\n'
        with patch.object(Path, 'read_text', return_value=status):
            with self.assertRaisesRegex(ValueError, 'cannot retain capabilities'):
                current_sdk.configure_fresh({'variant': 'scalar'})

    def test_fresh_preflight_binds_review_pair_and_reference(self):
        stage = self.root / 'artifacts/maintenance/preflight'
        stage.mkdir(parents=True)
        reference = stage / 'reference'
        (reference / 'training').mkdir(parents=True)
        (reference / 'inference').mkdir()
        targets, identities = {}, {}
        for index, name in enumerate(('training/result.json', 'training/snapshot.pt', 'inference/result.json')):
            target = stage / ('source-' + str(index))
            target.write_text('native reference ' + name)
            (reference / name).symlink_to(target)
            sha = bridge.seal(target)
            targets[name], identities[name] = dict(target=str(target), sha256=sha), sha
        runtime = dict(service={'path': str(stage / 'service')}, frontend={'path': str(stage / 'frontend')})
        environment = dict.fromkeys(bridge.ENV_KEYS, 'fixture')
        provenance = {name: dict(path=str(stage / name), sha256='a' * 64)
                      for name in ('plan', 'launch', 'runner', 'auditor')}
        numerical = {}
        for variant in ('scalar', 'sharded'):
            directory = stage / variant
            directory.mkdir()
            (directory / 'report.json').write_text(json.dumps(dict(claim=str(directory / 'claim.json'))))
            numerical[variant] = dict(root=str(directory), seals={}, provenance=deepcopy(provenance))
        contract = dict(schema='fresh_runtime_sgemm_v2', variant='scalar', target='2:3127', max_seconds=800,
                        deadline_utc=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
                        runtime=runtime, service_environment=environment, numerical=numerical,
                        reference=dict(path=str(reference), targets=targets))
        sources = {Path(current_sdk.__file__).resolve(), bridge.SCRIPT, Path(current_sdk.candidate.__file__).resolve()}
        sources.update(Path(current_sdk.__file__).with_name(name).resolve() for name in
                       ('run_bert_sgemm_ncu_capture.py', 'bert_ncu_sibling.py', 'bert_ncu_relay.py',
                        'bert_ncu_service_proxy.py', 'run_bert_sgemm_descriptor_probe.py',
                        'bert_sgemm_coverage_candidates.py', 'bert_sgemm_ncu_process.py'))
        review_path, plan_path = stage / 'review.json', stage / 'capture-plan.json'
        review = dict(status='fresh_runtime_v2_ready', contract=contract,
                      source_sha256={str(path): bridge.seal(path) for path in sources})
        review_path.write_text(json.dumps(review))
        plan = dict(contract, review=dict(path=str(review_path), sha256=bridge.seal(review_path)))
        plan_path.write_text(json.dumps(plan))
        manifest = dict(gpu='GPU-fixture', startup_reference_bundle='/startup',
                        startup_reference_report_sha256='fixture', dynamic_libraries={}, inputs={})

        def evidence(root, seals, variant, products, source, uid, gid):
            return dict(variant=variant, native_job={'artifact_identities': identities}, runtime=products,
                        service_environment=environment, provenance=source, manifest=manifest, numerical_only=True,
                        native_result_projection={'training/result.json': {'status': 'passed'}})

        with patch.object(current_sdk, 'fresh_numerical_evidence', side_effect=evidence) as checked:
            with current_sdk.prepare_fresh(plan_path, 'scalar') as (binding, paths, native, pin, application):
                self.assertEqual(checked.call_count, 2)
                self.assertEqual(binding['contract']['service_environment'], environment)
                self.assertEqual(native['artifact_identities'], identities)
                self.assertIn(reference / 'training/snapshot.pt', paths)
                self.assertEqual(len(application['manifests']), 2)
                pin.verify()
                changed = deepcopy(binding)
                changed['contract']['service_environment']['PATH'] = '/different'
                with self.assertRaises(ValueError):
                    current_sdk._check_fresh_binding_metadata(changed)
            with self.assertRaises(ValueError):
                pin.verify()
            with self.assertRaises(ValueError):
                with current_sdk.prepare_fresh(plan_path, 'sharded'):
                    pass
            def changed_evidence(*args):
                result = evidence(*args)
                if result['variant'] == 'sharded':
                    result['service_environment'] = dict(environment, PATH='/different')
                return result

            with patch.object(current_sdk, 'fresh_numerical_evidence', side_effect=changed_evidence):
                with self.assertRaises(ValueError):
                    with current_sdk.prepare_fresh(plan_path, 'scalar'):
                        pass
            with self.assertRaises(ValueError):
                with current_sdk.prepare_fresh(plan_path, 'scalar'):
                    (stage / 'source-0').write_text('changed native result')
            (stage / 'source-0').write_text('native reference training/result.json')
            link = reference / 'training/result.json'
            link.unlink()
            link.symlink_to(stage / 'source-1')
            with self.assertRaises(ValueError):
                with current_sdk.prepare_fresh(plan_path, 'scalar'):
                    pass

    def test_reference_guard_checks_once_and_closes_on_failure(self):
        target = self.root / 'snapshot'
        target.write_bytes(b'reference data')
        link = self.root / 'snapshot-link'
        link.symlink_to(target)
        expected = bridge.seal(target)
        pin = current_sdk._PinnedReference(link, target, expected)
        with patch.object(current_sdk.os, 'read', wraps=os.read) as reads:
            with pin:
                pin.verify()
                pin.verify()
                self.assertEqual(reads.call_count, 2)
                self.assertFalse(os.get_inheritable(pin.fd))
                self.assertFalse(os.get_inheritable(pin.link_fd))
                with self.assertRaises(ValueError):
                    pin.__enter__()
                with patch.object(current_sdk.os, 'getpid', return_value=pin.pid + 1):
                    with self.assertRaises(ValueError):
                        pin.verify()
                child = os.fork()
                if child == 0:
                    try:
                        pin.verify()
                    except ValueError:
                        os._exit(0)
                    os._exit(1)
                _, status = os.waitpid(child, 0)
                self.assertTrue(os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0)
            self.assertEqual(reads.call_count, 2)
        self.assertIsNone(pin.fd)
        self.assertIsNone(pin.link_fd)
        with self.assertRaises(ValueError):
            pin.verify()
        with self.assertRaises(ValueError):
            pin.__enter__()

        failed = current_sdk._PinnedReference(link, target, expected)
        with patch.object(current_sdk.os, 'read', side_effect=OSError('read failed')):
            with self.assertRaisesRegex(OSError, 'read failed'):
                with failed:
                    pass
        self.assertIsNone(failed.fd)
        self.assertIsNone(failed.link_fd)

        changed = current_sdk._PinnedReference(link, target, expected)
        with self.assertRaisesRegex(current_sdk.ReferenceGuardError, 'postflight.*body RuntimeError: body failed') as raised:
            with changed:
                target.write_bytes(b'changed content')
                raise RuntimeError('body failed')
        self.assertIsInstance(raised.exception.__cause__, RuntimeError)
        self.assertIsNone(changed.fd)
        self.assertIsNone(changed.link_fd)

        target.write_bytes(b'reference data')
        bounded = current_sdk._PinnedReference(link, target, expected)
        with patch.object(current_sdk.os, 'read', side_effect=lambda fd, count: b'x' * count) as reads:
            with self.assertRaisesRegex(ValueError, 'grew'):
                with bounded:
                    pass
            self.assertEqual(reads.call_count, 2)
        self.assertIsNone(bounded.fd)
        self.assertIsNone(bounded.link_fd)

        cleanup = current_sdk._PinnedReference(link, target, expected)
        real_close = os.close

        def close_with_failure(fd):
            real_close(fd)
            if fd == original_fd:
                raise OSError('injected close failure')

        cleanup.__enter__()
        original_fd = cleanup.fd
        with patch.object(current_sdk.os, 'close', side_effect=close_with_failure) as closed:
            with self.assertRaisesRegex(current_sdk.ReferenceGuardError, 'cleanup.*injected close failure'):
                cleanup.__exit__(None, None, None)
        self.assertEqual(closed.call_count, 2)
        self.assertIsNone(cleanup.fd)
        self.assertIsNone(cleanup.link_fd)

        combined = current_sdk._PinnedReference(link, target, expected)
        with patch.object(current_sdk.os, 'close', side_effect=close_with_failure) as closed:
            with self.assertRaisesRegex(current_sdk.ReferenceGuardError,
                                        'postflight.*cleanup.*injected close failure.*body RuntimeError'):
                with combined:
                    original_fd = combined.fd
                    target.write_bytes(b'changed content')
                    raise RuntimeError('body failed')
        self.assertEqual(closed.call_count, 2)
        self.assertIsNone(combined.fd)
        self.assertIsNone(combined.link_fd)

        for action in ('replace', 'unlink', 'link', 'truncate', 'during_read'):
            with self.subTest(action=action):
                directory = self.root / ('case-' + action)
                directory.mkdir()
                source = directory / 'source'
                source.write_bytes(b'reference data')
                reference = directory / 'link'
                reference.symlink_to(source)
                guard = current_sdk._PinnedReference(reference, source, bridge.seal(source))

                def modify():
                    if action == 'replace':
                        alternate = directory / 'alternate'
                        alternate.write_bytes(b'reference data')
                        os.replace(alternate, source)
                    elif action == 'unlink':
                        source.unlink()
                    elif action == 'link':
                        reference.unlink()
                        reference.symlink_to(source)
                    else:
                        source.write_bytes(b'x')

                if action == 'during_read':
                    real_read = os.read

                    def read_after_change(fd, count):
                        modify()
                        return real_read(fd, count)

                    with patch.object(current_sdk.os, 'read', side_effect=read_after_change):
                        with self.assertRaises(ValueError):
                            with guard:
                                pass
                else:
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        with guard:
                            modify()
                self.assertIsNone(guard.fd)
                self.assertIsNone(guard.link_fd)

    def test_service_exec_derives_signal_from_frozen_template(self):
        request = dict(clients=dict(low=dict(pid=11, start=101), high=dict(pid=12, start=102)),
                       socket='/private/service.sock', uid=1003, gid=1003,
                       environment={}, template_sha256='a' * 64)
        flag = '--experimental-sharded-gemm-signal'
        for variant in ('scalar', 'sharded'):
            template = dict(fresh_runtime=dict(schema='fresh_runtime_v2', variant=variant))
            with self.subTest(variant=variant), \
                    patch.object(bridge, 'frozen_template', return_value=template), \
                    patch.object(bridge, 'selected_service', return_value=Path('/fixed/service')), \
                    patch.object(bridge, 'profiler_environment', return_value={}):
                command = bridge.service_exec_command(request, Path('/attempt'), {}, 123, fresh_authority={})
                self.assertEqual(command.count(flag), int(variant == 'sharded'))
                self.assertEqual(command[-1] == flag, variant == 'sharded')

    def test_scoped_signal_uses_binding_and_restores_original(self):
        original = lambda: (['service', '--target', '2:3127', '--observe-block-completion'], ['pin'])
        common = SimpleNamespace(service_command=original)
        flag = '--experimental-sharded-gemm-signal'
        for variant in ('scalar', 'sharded'):
            binding = dict(schema='fresh_runtime_v2', variant=variant)
            with self.subTest(variant=variant), self.assertRaisesRegex(RuntimeError, 'restore'):
                with proxy.service_signal(common, binding):
                    command, pins = common.service_command()
                    self.assertEqual(pins, ['pin'])
                    self.assertEqual(command, original()[0] + ([flag] if variant == 'sharded' else []))
                    raise RuntimeError('restore')
            self.assertIs(common.service_command, original)
        with proxy.service_signal(common, None):
            self.assertIs(common.service_command, original)
        with proxy.service_signal(SimpleNamespace(), None):
            pass
        binding = dict(schema='fresh_runtime_v3', variant='sharded',
                       contract=dict(schema='fresh_runtime_sgemm_v3', specialization='ordinary'))
        with proxy.service_signal(common, binding):
            command, _ = common.service_command()
            self.assertNotIn('--observe-block-completion', command)
            self.assertIn(flag, command)
        missing = SimpleNamespace(service_command=lambda: (['service'], []))
        with proxy.service_signal(missing, binding), self.assertRaises(ValueError):
            missing.service_command()
        common.service_command = lambda: (['service', flag], [])
        with proxy.service_signal(common, dict(schema='fresh_runtime_v2', variant='sharded')):
            with self.assertRaises(ValueError):
                common.service_command()

    def test_capture_specialization_binds_service_and_profiler_without_fallback(self):
        request = dict(socket='/tmp/glimmer-bert-co-x/service.sock',
                       clients=dict(low=dict(pid=7, start=70), high=dict(pid=8, start=80)))
        observed = dict(schema='fresh_runtime_v2', variant='scalar',
                        contract=dict(schema='fresh_runtime_sgemm_v2'))
        ordinary = dict(schema='fresh_runtime_v3', variant='sharded',
                        contract=dict(schema='fresh_runtime_sgemm_v3', specialization='ordinary'))
        for binding, expected, flag in ((observed, 'glimmer_owned_sgemm_nn_observed', True),
                                        (ordinary, 'glimmer_owned_sgemm_nn', False)):
            with self.subTest(binding=binding):
                command = bridge.service_command(request, signal_binding=binding)
                self.assertEqual('--observe-block-completion' in command, flag)
                self.assertEqual(bridge.capture_kernel(binding), expected)
                manifest = dict(uid=1001, gid=1001, groups=[], python=sys.executable, fresh_runtime=binding)
                profiler = capture.ncu_command(Path('/attempt'), manifest)
                self.assertEqual(profiler[profiler.index('--kernel-name') + 1], expected)
                self.assertEqual(profiler[profiler.index('--app-replay-mode') + 1], 'strict')
                self.assertEqual(profiler[profiler.index('--app-replay-match') + 1], 'all')
        for bad in (dict(ordinary, schema='fresh_runtime_v2'),
                    dict(ordinary, contract=dict(schema='fresh_runtime_sgemm_v3')),
                    dict(ordinary, contract=dict(schema='fresh_runtime_sgemm_v3', specialization='observed')),
                    dict(ordinary, schema='unknown')):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                bridge.capture_kernel(bad)

    def test_target_and_signal_scope_matches_proxy_and_root_recipe(self):
        provider = self.root / 'nvtx.so'
        provider.touch()
        request = dict(clients=dict(low=dict(pid=11, start=101), high=dict(pid=12, start=102)),
                       socket='/private/service.sock')
        with patch.object(bridge, 'RANGE_LIBRARY', provider):
            ordinary = bridge.service_command(request)
            common = SimpleNamespace(service_command=lambda: (ordinary[:-4], ['pin']))
            bindings = [dict(schema='fresh_runtime_v2', variant=variant) for variant in ('scalar', 'sharded')]
            bindings += [dict(schema='fresh_runtime_v3', variant=variant,
                              contract=dict(schema='fresh_runtime_sgemm_v3', specialization='ordinary'))
                         for variant in ('scalar', 'sharded')]
            for binding in bindings:
                with self.subTest(binding=binding), \
                        selected_service_target(common, '2:3127', provider, client_bound=True), \
                        proxy.service_signal(common, binding):
                    command, pins = common.service_command()
                    self.assertEqual(pins, ['pin'])
                    self.assertEqual(command, bridge.service_command(request, signal_binding=binding))
                    self.assertEqual(proxy.service_request(command, self.root, signal_binding=binding)['clients'],
                                     request['clients'])

    def test_root_relay_rejects_service_from_other_signal_mode(self):
        plan = dict(clients=dict(low=dict(pid=11, start=101), high=dict(pid=12, start=102)),
                    socket='/private/service.sock')
        flag = '--experimental-sharded-gemm-signal'
        observed_binding = dict(schema='fresh_runtime_v2', variant='scalar')
        ordinary_binding = dict(schema='fresh_runtime_v3', variant='scalar',
                                contract=dict(schema='fresh_runtime_sgemm_v3', specialization='ordinary'))
        pairs = [(dict(schema='fresh_runtime_v2', variant=expected),
                  dict(schema='fresh_runtime_v2', variant=actual))
                 for expected, actual in (('scalar', 'sharded'), ('sharded', 'scalar'))]
        pairs += [(observed_binding, ordinary_binding), (ordinary_binding, observed_binding)]
        for binding, other in pairs:
            instance = relay.ReplayRelay.__new__(relay.ReplayRelay)
            instance.group = '/task'
            instance.service = bridge.SERVICE
            instance.manifest = dict(uid=1003, gid=1003, fresh_runtime=binding)
            instance.current = dict(service=dict(pid=9, start=90), helper=dict(pid=7, start=70), plan=plan)
            command = bridge.service_command(plan, signal_binding=other)
            argv = ['/proc/self/fd/123', *command[1:]]
            with self.subTest(expected=binding, actual=other), ExitStack() as stack:
                stack.enter_context(patch.object(relay, 'authenticate_process'))
                stack.enter_context(patch.object(relay, 'parent', return_value=7))
                stack.enter_context(patch.object(relay.os, 'getsid', return_value=9))
                stack.enter_context(patch.object(relay.os, 'getpgid', return_value=9))
                stack.enter_context(patch.object(relay, 'command_line', return_value=argv))
                with self.assertRaisesRegex(ValueError, 'fixed sealed service command differs'):
                    instance.authenticate_service()
            self.assertEqual(argv.count(flag), int(other['variant'] == 'sharded'))

    def launch_fixture(self, variant, predecessor=None):
        path = self.root / 'launch-review.json'
        value = dict(status='host_pair_ncu_ready', variant=variant)
        if predecessor is not None:
            value['predecessor'] = predecessor
        path.write_text(json.dumps(value))
        return dict(launch_review=str(path), launch_review_sha256=bridge.seal(path),
                    runtime_manifest_sha256=bridge.seal(self.manifest),
                    numerical_report_sha256=bridge.seal(self.report))

    def test_o2_before_o0_cannot_use_o0_review_or_missing_predecessor(self):
        plan = self.launch_fixture('o0')
        self.assertEqual(host.launch_review(plan, 'o0')['variant'], 'o0')
        with self.assertRaises(ValueError):
            host.launch_review(plan, 'o2')
        with self.assertRaises(KeyError):
            host.launch_review(self.launch_fixture('o2'), 'o2')

    def test_o2_requires_successful_and_independently_accepted_o0(self):
        predecessor = self.root / 'predecessor'
        predecessor.mkdir()
        good = dict(status='counter_observation_requires_independent_review', input_identity_unchanged=True,
                    unit=dict(returncode=0, unit_removed=True),
                    observation=dict(metrics=dict(minimums_positive=True, metrics=dict(launch__sm_count=80))),
                    host_pair=dict(variant='o0', runtime_manifest_sha256=bridge.seal(self.manifest),
                                   numerical_report_sha256=bridge.seal(self.report)))
        mutations = [((), None), (('status',), 'failed'),
                     (('observation', 'metrics', 'minimums_positive'), False),
                     (('unit', 'returncode'), 1), (('unit', 'unit_removed'), False),
                     (('input_identity_unchanged',), False)]
        for keys, value in mutations:
            report = deepcopy(good)
            target = report
            for key in keys[:-1]:
                target = target[key]
            if keys:
                target[keys[-1]] = value
            report_path, review_path = predecessor / 'report.json', predecessor / 'review.json'
            report_path.write_text(json.dumps(report))
            review_path.write_text(json.dumps(dict(status='host_pair_replacement_target_80sm_accepted',
                variant='o0', report_sha256=bridge.seal(report_path))))
            bound = dict(attempt=str(predecessor), report_sha256=bridge.seal(report_path),
                         review_sha256=bridge.seal(review_path))
            plan = self.launch_fixture('o2', bound)
            with self.subTest(keys=keys):
                if keys:
                    with self.assertRaises(ValueError):
                        host.launch_review(plan, 'o2')
                else:
                    self.assertEqual(host.launch_review(plan, 'o2')['variant'], 'o2')
        review_path.write_text('{}')
        with self.assertRaises(ValueError):
            host.launch_review(plan, 'o2')


class CurrentSdkNumericalBindingTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / 'attempt-test'
        self.root.mkdir()
        self.plan = self.root / 'plan.json'
        self.plan.write_text(json.dumps(dict(variants=['native', 'scalar', 'sharded'], **current_sdk.FLAGS)))
        for name in ('inputs-before.json', 'inputs-after.json'):
            (self.root / name).write_text('{}')
        rows, workers = [], {}
        for variant in ('native', 'scalar', 'sharded'):
            directory = self.root / f'{variant}-{self.root.name}'
            directory.mkdir()
            mode = 'native' if variant == 'native' else 'equal'
            (directory / mode).mkdir()
            job = dict(mode=mode)
            if variant != 'native':
                log = directory / 'equal/server.log'
                selector = 'GEMM signal diagnostic: sharded=true\n' if variant == 'sharded' else ''
                log.write_text(selector + '\n'.join((
                    'Native admission audit: session=2, sequence=3127, accepted=true, operation=30',
                    'Native persistent audit: session=2, sequence=3127, total_blocks=2304, '
                    'physical_launches=1, workers=320',
                    'Native issue audit: session=2, sequence=3127, status=0',
                    'Native execution audit: session=2, issued=3127, retained=0, retired=1, status=0',
                    'Priority execution audit: event=low_logical_done, session=2, priority=0, '
                    'sequence=3127, completed_blocks=2304, total_blocks=2304, physical_launches=1, status=0',
                )) + '\n')
                job['artifact_identities'] = {'server.log': bridge.seal(log)}
            (directory / mode / 'attempt.json').write_text(json.dumps(job))
            worker = dict(status='passed_requires_review', reparse=current_sdk.ORACLE, job=job,
                          variant=variant, mode=mode, scoped_publication_valid=False, batch_valid=False,
                          **current_sdk.FLAGS)
            path = directory / f'{mode}-worker.json'
            path.write_text(json.dumps(worker))
            workers[variant] = dict(path=str(path), sha256=bridge.seal(path))
            rows.append(dict(variant=variant, unit=dict(returncode=0, unit_removed=True), worker=worker))
        self.report = self.root / 'report.json'
        self.report.write_text(json.dumps(dict(status='fresh_native_scalar_sharded_numerical_requires_review',
            numerical_three_path_passed=True, input_identity_unchanged=True, jobs=rows, **current_sdk.FLAGS)))
        self.publication = self.root / 'batch-publication.json'
        self.publication.write_text(json.dumps(dict(complete=True, qualification='numerical_evidence_only',
            report=str(self.report), report_sha256=bridge.seal(self.report), plan_sha256=bridge.seal(self.plan),
            workers=workers, **current_sdk.FLAGS)))
        paths = [self.report, self.plan, self.publication, self.root / 'inputs-before.json',
                 self.root / 'inputs-after.json']
        for variant in ('native', 'scalar', 'sharded'):
            mode = 'native' if variant == 'native' else 'equal'
            directory = self.root / f'{variant}-{self.root.name}'
            paths.extend((directory / f'{mode}-worker.json', directory / mode / 'attempt.json'))
        self.seals = {str(path): bridge.seal(path) for path in paths}
        for name, replacement in (('normal_cleanup', None),):
            patched = patch.object(current_sdk.candidate, name, return_value=replacement)
            patched.start()
            self.addCleanup(patched.stop)
        for name, replacement in (('parse_semantic_audit', None), ('authenticated_training_sessions', {2})):
            patched = patch.object(current_sdk, name, return_value=replacement)
            patched.start()
            self.addCleanup(patched.stop)

    def check(self, variant='sharded'):
        return current_sdk.numerical_evidence(self.root, self.seals, variant)

    def test_bind_both_variants_and_reject_modified_evidence(self):
        for variant in current_sdk.VARIANTS:
            with self.subTest(variant=variant):
                self.check(variant)
        with self.assertRaises(ValueError):
            current_sdk.numerical_evidence(self.root, {}, 'sharded')
        with self.assertRaises(ValueError):
            self.check('other')
        (self.root / 'inputs-after.json').write_text('{"changed": true}')
        with self.assertRaises(ValueError):
            self.check()

    def test_reject_changed_service_log(self):
        log = self.root / f'sharded-{self.root.name}/equal/server.log'
        log.write_text(log.read_text().replace('workers=320', 'workers=80'))
        with self.assertRaises(ValueError):
            self.check()

    def test_reject_resealed_incomplete_target(self):
        log = self.root / f'sharded-{self.root.name}/equal/server.log'
        worker_path = self.root / f'sharded-{self.root.name}/equal-worker.json'
        raw = log.parent / 'attempt.json'
        for old, new in (('retired=1', 'retired=0'), ('physical_launches=1', 'physical_launches=2'),
                         ('workers=320', 'workers=80')):
            original = log.read_text()
            log.write_text(original.replace(old, new, 1))
            worker, report, publication = (json.loads(path.read_text()) for path in
                                            (worker_path, self.report, self.publication))
            worker['job']['artifact_identities']['server.log'] = bridge.seal(log)
            raw.write_text(json.dumps(worker['job']))
            worker_path.write_text(json.dumps(worker))
            next(row for row in report['jobs'] if row['variant'] == 'sharded')['worker'] = worker
            self.report.write_text(json.dumps(report))
            publication['report_sha256'] = bridge.seal(self.report)
            publication['workers']['sharded']['sha256'] = bridge.seal(worker_path)
            self.publication.write_text(json.dumps(publication))
            seals = {path: bridge.seal(Path(path)) for path in self.seals}
            with self.subTest(change=old), self.assertRaises(ValueError):
                current_sdk.numerical_evidence(self.root, seals, 'sharded')
            log.write_text(original)


class CurrentSdkServiceBindingTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        archive = self.root / 'artifacts/archives/execution-current-sdk/attempt-test'
        products = {}
        for variant in ('scalar', 'sharded'):
            service = archive / variant / 'glimmer_execution_service'
            service.parent.mkdir(parents=True)
            service.write_text(variant)
            service.chmod(0o555)
            metadata = service.stat()
            products[variant] = dict(glimmer_execution_service=dict(
                path=str(service), mode=0o555, device=metadata.st_dev, inode=metadata.st_ino,
                size=metadata.st_size, ctime_ns=metadata.st_ctime_ns, sha256=bridge.seal(service)))
        self.manifest = archive / 'runtime.json'
        self.manifest.write_text(json.dumps(dict(variants=products)))
        patched = patch.object(bridge, 'PROJECT', self.root)
        patched.start()
        self.addCleanup(patched.stop)

    def template(self, variant='sharded'):
        product = json.loads(self.manifest.read_text())['variants'][variant]['glimmer_execution_service']
        return dict(uid=os.getuid(), current_sdk=dict(variant=variant, server=product['path'],
                    runtime_manifest=str(self.manifest), runtime_manifest_sha256=bridge.seal(self.manifest)))

    def test_exact_runtime_and_reject_changed_selection(self):
        for variant in ('scalar', 'sharded'):
            template = self.template(variant)
            self.assertEqual(bridge.selected_service(template), Path(template['current_sdk']['server']))
        for key, value in (('variant', 'other'), ('server', '/tmp/unexpected'),
                           ('runtime_manifest_sha256', '0' * 64)):
            template = self.template()
            template['current_sdk'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                bridge.selected_service(template)
        template = self.template()
        template['host_pair'] = {}
        with self.assertRaises(ValueError):
            bridge.selected_service(template)
        template = self.template()
        Path(template['current_sdk']['server']).chmod(0o755)
        with self.assertRaises(ValueError):
            bridge.selected_service(template)

    def test_counter_job_keeps_existing_full_recipe(self):
        provider = self.root / 'nvtx.so'
        provider.write_text('fixture')
        binding = dict(server='/test/service', frontend='/test/frontend', variant='sharded')
        low, high = dict(pid=101, start=201), dict(pid=102, start=202)
        server = dict(pid=103, start=203)
        socket = '/test/service.sock'
        environment = {name: 'fixture' for name in current_sdk.pair.BASE_ENVIRONMENT_KEYS}
        environment.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
            LD_LIBRARY_PATH=str(current_sdk.pair.numerical.PROJECT_ROOT / 'examples/framework_workloads/pytorch_smoke/.venv/lib/'
                'python3.10/site-packages/nvidia/cudnn/lib') + ':/usr/local/cuda-12.6/lib64')
        app = dict(environment, LD_PRELOAD=binding['frontend'], GLIMMER_EXECUTION_SOCKET=socket,
                   GLIMMER_EXECUTION_SERVER_PID='103', GLIMMER_EXECUTION_SERVER_START='203',
                   GLIMMER_EXECUTION_PREPARATION_TIMEOUT_MS='240000', GLIMMER_EXECUTION_HOST_RANGE_SEEK='1')
        command = [binding['server'], '--socket', socket, '--exit-when-idle', '--persistent-workers', '320',
                   '--kernel-mode', 'preemptive', '--scheduler', 'priority', '--replace-gemm',
                   '--audit-native', '--audit-priority', '--observe-block-completion', '--memory-per-client',
                   '12884901888', '--host-memory-per-client', '536870912', '--priority-idle-us', '2000',
                   '--gemm-tile-side', '32', '--client-priority', '101:201:0:configured',
                   '--client-priority', '102:202:0:original', '--audit-sgemm-target-client', '101:201:3127',
                   '--audit-sgemm-range-library', str(provider)]
        job = dict(mode='equal', application_identities=dict(low=low, high=high), server_identity=server,
                   server_command=command, server_environment=environment, application_environment=app,
                   application_environments=dict(low=app, high=app),
                   application_commands=dict(low=['--warmup', '1', '--iterations', '2'],
                                             high=['--warmup', '1', '--iterations', '2']))
        with patch.object(current_sdk, 'check_binding'), patch.object(current_sdk.candidate, 'normal_cleanup'), \
                patch.object(bridge, 'RANGE_LIBRARY', provider):
            current_sdk.validate_job(binding, job)
            changed = deepcopy(job)
            changed['server_command'][-3] = '101:201:999'
            with self.assertRaises(ValueError):
                current_sdk.validate_job(binding, changed)
            changed = deepcopy(job)
            changed['application_environment']['GLIMMER_EXECUTION_HOST_RANGE_SEEK'] = '0'
            with self.assertRaises(ValueError):
                current_sdk.validate_job(binding, changed)

    def test_prepare_configure_and_one_shot_claim(self):
        frontend = self.root / 'frontend.so'
        frontend.write_text('frontend')
        metadata = frontend.stat()
        runtime = json.loads(self.manifest.read_text())
        runtime['original_frontend'] = str(frontend)
        runtime['frontend'] = dict(path=str(frontend), device=metadata.st_dev, inode=metadata.st_ino,
            size=metadata.st_size, ctime_ns=metadata.st_ctime_ns, sha256=bridge.seal(frontend))
        original_server = str(self.root / 'original-service')
        runtime['variants']['sharded']['original_server'] = original_server
        self.manifest.write_text(json.dumps(runtime))
        source = self.root / 'src/example.cc'
        source.parent.mkdir()
        source.write_text('source')
        numerical = dict(artifact_sha256={original_server:
            runtime['variants']['sharded']['glimmer_execution_service']['sha256'],
            str(frontend): bridge.seal(frontend), str(source): bridge.seal(source)}, runtime_identity={})
        reference = self.root / 'reference'
        targets, artifacts = {}, {}
        for name in ('training/result.json', 'training/snapshot.pt', 'inference/result.json'):
            target = self.root / ('original-' + name.replace('/', '-'))
            target.write_text(name)
            link = reference / name
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(target)
            targets[name] = dict(link=str(target), target=str(target), sha256=bridge.seal(target))
            artifacts[name] = bridge.seal(target)
        sources = [Path(current_sdk.__file__).resolve(), bridge.SCRIPT,
                   Path(current_sdk.__file__).with_name('run_bert_sgemm_ncu_capture.py').resolve(),
                   Path(current_sdk.__file__).with_name('bert_ncu_sibling.py').resolve(),
                   Path(current_sdk.__file__).with_name('bert_ncu_relay.py').resolve()]
        review = self.root / 'review.json'
        review.write_text(json.dumps(dict(status='current_sdk_ncu_ready', variant='sharded',
            numerical_acceptance='fresh_three_path_numerical_accepted', numerical_seals={},
            reference_targets=targets, native_reference=str(reference), target='2:3127', max_seconds=800,
            runtime_manifest_sha256=bridge.seal(self.manifest),
            source_sha256={str(path): bridge.seal(path) for path in sources})))
        plan = self.root / 'plan.json'
        config = dict(schema='current_sdk_sgemm_v1', variant='sharded', target='2:3127',
            deadline_utc=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(), max_seconds=800,
            runtime_environment={'GLIMMER_EXECUTION_HOST_RANGE_SEEK': '1'}, numerical_root='/unused',
            numerical_seals={}, runtime_manifest=str(self.manifest), runtime_manifest_sha256=bridge.seal(self.manifest),
            launch_review=str(review), launch_review_sha256=bridge.seal(review), native_reference=str(reference),
            reference_targets=targets)
        plan.write_text(json.dumps(config))
        with patch.object(current_sdk, 'numerical_evidence', return_value=(numerical,
                          dict(artifact_identities=artifacts))):
            binding, paths, _ = current_sdk.prepare(plan, 'sharded')
            self.assertIn(frontend, paths)
            with patch.object(current_sdk.pair.numerical, 'configure', return_value=('common', 'helpers')) as setup:
                self.assertEqual(current_sdk.configure(binding), ('common', 'helpers'))
                self.assertEqual(setup.call_args.args[0]['runtime_environment'], config['runtime_environment'])
            plan.write_text(json.dumps(dict(config, target='2:999')))
            with self.assertRaises(ValueError):
                current_sdk.claim_launch(binding)
            expired = dict(config, deadline_utc=(datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat())
            plan.write_text(json.dumps(expired))
            with self.assertRaises(ValueError):
                current_sdk.claim_launch(dict(binding, plan_sha256=bridge.seal(plan)))
            self.assertFalse((self.root / 'ncu-sharded-launch.json').exists())
            plan.write_text(json.dumps(config))
            current_sdk.claim_launch(binding)
            with self.assertRaises(FileExistsError):
                current_sdk.claim_launch(binding)
            alias = self.root / 'reference-alias'
            alias.symlink_to(reference, target_is_directory=True)
            changed = dict(config, native_reference=str(alias))
            plan.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                current_sdk.prepare(plan, 'sharded')
            changed = dict(config, deadline_utc=(datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat())
            plan.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                current_sdk.prepare(plan, 'sharded')
            plan.write_text(json.dumps(config))
            frontend.write_text('modified')
            with self.assertRaises(ValueError):
                current_sdk.prepare(plan, 'sharded')

    def test_collector_dispatches_current_variant_only(self):
        h = SimpleNamespace(training=SimpleNamespace(require_idle=lambda: None, manifest=lambda paths: {}),
                            query=lambda *args: 'gpu')
        selection = ({'variant': 'sharded'}, set(), {})
        with patch.object(capture.subprocess, 'run', return_value=SimpleNamespace(returncode=0)), \
                patch.object(current_sdk, 'prepare', return_value=selection) as prepare, \
                patch.object(current_sdk, 'claim_launch') as claim, \
                patch.object(host, 'prepare') as legacy, \
                patch.object(capture, 'resolve_artifact_root', return_value=self.root), \
                patch.object(capture, 'create_attempt', return_value=self.root), \
                patch.object(capture, 'prepare_inputs', return_value=(h, 'gpu', set(), {})) as inputs, \
                patch.object(capture, 'launch_unit', side_effect=RuntimeError('fixture stop')) as launch, \
                patch.object(capture, 'save'), patch('builtins.print'):
            status = capture.collect('2:3127', service_cap_sys_admin=True,
                                     current_sdk_plan=self.root / 'plan.json', variant='sharded')
        self.assertEqual(status, 1)
        prepare.assert_called_once_with(self.root / 'plan.json', 'sharded')
        claim.assert_called_once_with(selection[0])
        legacy.assert_not_called()
        launch.assert_called_once()
        self.assertEqual(inputs.call_args.kwargs['current_selection'], selection)

    def test_fresh_collector_context_controls_publication_and_failures(self):
        for failure in (None, 'enter', 'exit', 'identity'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary, ExitStack() as mocks:
                outer = Path(temporary)
                events, saved = [], {}
                pin = SimpleNamespace(verify=lambda: events.append('pin-verify'))
                selection = ({'variant': 'scalar'}, set(), {}, pin,
                             dict(common=SimpleNamespace(input_paths=lambda h: set()), manifests=[], environment={},
                                  expected={}, summary={}, contract={}))

                @contextmanager
                def prepared(plan, variant):
                    events.append('enter')
                    if failure == 'enter':
                        raise ValueError('preparation failed')
                    yield selection
                    events.append('postflight')
                    if failure == 'exit':
                        raise ValueError('postflight failed')
                    events.append('closed')

                def record(path, value):
                    events.append(path.name)
                    saved[path.name] = deepcopy(value)

                def run_unit(*args, **kwargs):
                    events.append('unit-retired')
                    return dict(returncode=0, unit_removed=True)

                training = SimpleNamespace(require_idle=lambda: None, GPU_FIELDS='fields',
                                           manifest=lambda paths: {'changed': True} if failure == 'identity' else {})
                helpers = SimpleNamespace(training=training, query=lambda *args: 'gpu')
                mocks.enter_context(patch.object(capture.subprocess, 'run', return_value=SimpleNamespace(returncode=0)))
                mocks.enter_context(patch.object(current_sdk, 'prepare_fresh', side_effect=prepared))
                mocks.enter_context(patch.object(current_sdk, 'claim_fresh',
                                                side_effect=lambda *args: events.append('claim') or {'path': '/claim'}))
                mocks.enter_context(patch.object(current_sdk, 'fresh_application_inventory', return_value=(set(), {}, {})))
                mocks.enter_context(patch.object(current_sdk, 'validate_fresh_application_inputs', return_value={}))
                mocks.enter_context(patch.object(capture, 'resolve_artifact_root', return_value=outer))
                mocks.enter_context(patch.object(capture, 'create_attempt', return_value=outer))
                inputs = mocks.enter_context(patch.object(capture, 'prepare_inputs', return_value=(helpers, 'gpu', set(), {})))
                launch = mocks.enter_context(patch.object(capture, 'launch_unit', side_effect=run_unit))
                mocks.enter_context(patch.object(capture, 'review_capture',
                                                return_value={'metrics': {'minimums_positive': True}}))
                mocks.enter_context(patch.object(capture, 'save', side_effect=record))
                mocks.enter_context(patch('builtins.print'))
                status = capture.collect('2:3127', service_cap_sys_admin=True,
                                         fresh_runtime_plan=outer / 'plan.json', variant='scalar')
                report = saved['report.json']
                self.assertEqual(status, 0 if failure is None else 1)
                self.assertEqual(report['status'], 'counter_observation_requires_independent_review'
                                 if failure is None else 'failed')
                if failure == 'enter':
                    launch.assert_not_called()
                    inputs.assert_not_called()
                else:
                    launch.assert_called_once()
                    self.assertLess(events.index('pin-verify'), events.index('claim'))
                    self.assertLess(events.index('claim'), events.index('unit-retired'))
                    self.assertLess(events.index('unit-retired'), events.index('postflight'))
                    self.assertLess(events.index('postflight'), events.index('report.json'))
                    self.assertEqual(inputs.call_args.kwargs['fresh_selection'], selection)
                if failure == 'exit':
                    self.assertIn('postflight failed', report['preparation_error'])
                if failure == 'identity':
                    self.assertIn('frozen input', report['identity_error'])
                if failure is None:
                    self.assertLess(events.index('closed'), events.index('report.json'))

    def test_fresh_collector_rejects_invalid_plan_combinations_before_start(self):
        with patch.object(capture.subprocess, 'run', return_value=SimpleNamespace(returncode=0)), \
                patch.object(capture, 'create_attempt') as create:
            for args in (dict(host_pair_plan=Path('/host'), fresh_runtime_plan=Path('/fresh'), variant='scalar',
                              service_cap_sys_admin=True),
                         dict(fresh_runtime_plan=Path('/fresh'), variant='o0', service_cap_sys_admin=True),
                         dict(fresh_runtime_plan=Path('/fresh'), variant='scalar', service_cap_sys_admin=False)):
                with self.subTest(args=args), self.assertRaises(ValueError):
                    capture.collect('2:3127', **args)
            with self.assertRaises(ValueError):
                capture.collect('2:3128', fresh_runtime_plan=Path('/fresh'), variant='scalar',
                                service_cap_sys_admin=True)
            create.assert_not_called()

    def test_fresh_claim_is_bound_one_shot_and_retains_failed_write(self):
        stage = self.root / 'artifacts/maintenance/claim'
        stage.mkdir(parents=True, mode=0o700)
        outer = self.root / 'artifacts/profiles/ncu/attempt-fixture'
        outer.mkdir(parents=True, mode=0o700)
        service = dict(path='/fixed/service', sha256='a' * 64)
        environment = {'PATH': '/fixed'}
        contract = dict(schema='fresh_runtime_sgemm_v2', target='2:3127', runtime=dict(service=service),
                        service_environment=environment)
        application_contract = dict(schema='recorded_application_inputs_v1')
        review = stage / 'review.json'
        review.write_text('{}')

        def fixture(name):
            plan = stage / name
            plan.write_text('{}')
            binding = dict(schema='fresh_runtime_v2', variant='scalar', plan=str(plan), plan_sha256='b' * 64,
                           review=str(review), review_sha256='c' * 64, contract=contract)
            template = dict(fresh_runtime=binding, application_input_contract=application_contract,
                            service_cap_sys_admin=True, sgemm_target='2:3127',
                            service_environment=environment, bridge_seals={service['path']: service['sha256']},
                            uid=os.getuid(), gid=os.getgid())
            (outer / 'manifest.json').write_bytes(bridge.encode(template))
            (outer / 'inputs-before.json').write_text('{}')
            return binding

        with patch.object(bridge, 'PROJECT', self.root), \
                patch.object(current_sdk, '_check_fresh_binding_metadata', return_value=(contract, {})):
            wrong = fixture('wrong-inventory.json')
            with self.assertRaisesRegex(ValueError, 'manifest differs'):
                current_sdk.claim_fresh(wrong, outer, dict(schema='recorded_application_inputs_v1', changed=True))
            self.assertFalse((stage / 'wrong-inventory.json.launched').exists())
            binding = fixture('plan.json')
            result = current_sdk.claim_fresh(binding, outer, application_contract)
            path = Path(result['path'])
            self.assertEqual(result['sha256'], bridge.seal(path))
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            record = json.loads(path.read_text())
            self.assertEqual(record['attempt'], str(outer))
            self.assertEqual(record['manifest_sha256'], bridge.seal(outer / 'manifest.json'))
            self.assertEqual(record['inputs_sha256'], bridge.seal(outer / 'inputs-before.json'))
            self.assertEqual(record['binding_sha256'], bridge.hashlib.sha256(
                json.dumps(binding, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest())
            with self.assertRaises(current_sdk.ReferenceGuardError):
                current_sdk.claim_fresh(binding, outer, application_contract)

            incomplete = fixture('partial.json')
            with patch.object(current_sdk.os, 'write', return_value=0):
                with self.assertRaisesRegex(current_sdk.ReferenceGuardError, 'short fresh claim write'):
                    current_sdk.claim_fresh(incomplete, outer, application_contract)
            partial = stage / 'partial.json.launched'
            self.assertTrue(partial.is_file())
            with self.assertRaises(current_sdk.ReferenceGuardError):
                current_sdk.claim_fresh(incomplete, outer, application_contract)

            unsynced = fixture('unsynced.json')
            with patch.object(current_sdk.os, 'fsync', side_effect=[OSError('file sync failed'), None]) as sync:
                with self.assertRaisesRegex(current_sdk.ReferenceGuardError, 'file sync failed'):
                    current_sdk.claim_fresh(unsynced, outer, application_contract)
                self.assertEqual(sync.call_count, 2)
            self.assertTrue((stage / 'unsynced.json.launched').is_file())
            with self.assertRaises(current_sdk.ReferenceGuardError):
                current_sdk.claim_fresh(unsynced, outer, application_contract)

            unclosed = fixture('unclosed.json')
            original_open, original_close, claim_fd = os.open, os.close, []

            def track_open(path, *args, **kwargs):
                descriptor = original_open(path, *args, **kwargs)
                if path == 'unclosed.json.launched':
                    claim_fd.append(descriptor)
                return descriptor

            def fail_close(descriptor):
                original_close(descriptor)
                if claim_fd and descriptor == claim_fd[0]:
                    raise OSError('claim close failed')

            with patch.object(current_sdk.os, 'open', side_effect=track_open), \
                    patch.object(current_sdk.os, 'close', side_effect=fail_close):
                with self.assertRaisesRegex(current_sdk.ReferenceGuardError, 'claim close failed'):
                    current_sdk.claim_fresh(unclosed, outer, application_contract)
            self.assertTrue((stage / 'unclosed.json.launched').is_file())
            with self.assertRaises(current_sdk.ReferenceGuardError):
                current_sdk.claim_fresh(unclosed, outer, application_contract)

            linked = fixture('linked.json')
            destination = stage / 'do-not-overwrite'
            destination.write_text('unchanged')
            (stage / 'linked.json.launched').symlink_to(destination)
            with self.assertRaises(current_sdk.ReferenceGuardError):
                current_sdk.claim_fresh(linked, outer, application_contract)
            self.assertEqual(destination.read_text(), 'unchanged')

    def test_fresh_claim_verifier_rejects_changed_inputs(self):
        stage = self.root / 'artifacts/maintenance/verified-claim'
        stage.mkdir(parents=True, mode=0o700)
        outer = self.root / 'artifacts/profiles/ncu/attempt-fixture'
        outer.mkdir(parents=True, mode=0o700)
        service = dict(path='/fixed/service', sha256='a' * 64)
        environment = {'PATH': '/fixed'}
        contract = dict(schema='fresh_runtime_sgemm_v2', variant='scalar', target='2:3127', max_seconds=800,
                        deadline_utc=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
                        runtime=dict(service=service), service_environment=environment, numerical={}, reference={})
        sources = {}
        scripts = self.root / 'scripts'
        scripts.mkdir()
        for name in ('bert_ncu_current_sdk.py', 'bert_ncu_replay_exec.py', 'run_bert_candidate_validation.py',
                     'run_bert_sgemm_ncu_capture.py', 'bert_ncu_sibling.py', 'bert_ncu_relay.py',
                     'bert_ncu_service_proxy.py', 'run_bert_sgemm_descriptor_probe.py',
                     'bert_sgemm_coverage_candidates.py', 'bert_sgemm_ncu_process.py'):
            path = scripts / name
            path.write_text('reviewed source')
            sources[str(path)] = bridge.seal(path)
        review, plan = stage / 'review.json', stage / 'plan.json'
        review.write_text(json.dumps(dict(status='fresh_runtime_v2_ready', contract=contract, source_sha256=sources)))
        plan.write_text(json.dumps(dict(contract, review=dict(path=str(review), sha256=bridge.seal(review)))))
        binding = dict(schema='fresh_runtime_v2', variant='scalar', contract=contract, plan=str(plan),
                       plan_sha256=bridge.seal(plan), review=str(review), review_sha256=bridge.seal(review))
        application = dict(schema='recorded_application_inputs_v1', recorded_files=1, historical_sha256=1,
                           historical_stat_only=0, recorded_digest='a' * 64, historical_sha256_digest='b' * 64,
                           historical_stat_only_digest='c' * 64,
                           loader_sha256={'/loader-' + str(index): 'd' * 64 for index in range(5)},
                           complete_python_import_closure=False)
        template = dict(fresh_runtime=binding, application_input_contract=application,
                        service_cap_sys_admin=True, sgemm_target='2:3127', service_environment=environment,
                        bridge_seals={service['path']: service['sha256']}, uid=os.getuid(), gid=os.getgid())
        (outer / 'manifest.json').write_bytes(bridge.encode(template))
        (outer / 'inputs-before.json').write_text('{}')
        with patch.object(current_sdk, '_check_fresh_binding_metadata', return_value=(contract, {})):
            claim = current_sdk.claim_fresh(binding, outer, application)
        verified = bridge.verify_fresh_claim(outer, claim['sha256'], os.getuid(), os.getgid())
        self.assertEqual(verified['binding'], binding)
        with self.assertRaisesRegex(ValueError, 'claim digest differs'):
            bridge.verify_fresh_claim(outer, '0' * 64, os.getuid(), os.getgid())
        source = scripts / 'bert_ncu_relay.py'
        source.write_text('changed source')
        with self.assertRaisesRegex(ValueError, 'reviewed fresh source differs'):
            bridge.verify_fresh_claim(outer, claim['sha256'], os.getuid(), os.getgid())
        source.write_text('reviewed source')
        (outer / 'inputs-before.json').write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError, 'not bound'):
            bridge.verify_fresh_claim(outer, claim['sha256'], os.getuid(), os.getgid())

        (outer / 'inputs-before.json').write_text('{}')
        ordinary = dict(contract, schema='fresh_runtime_sgemm_v3', specialization='ordinary')
        ordinary_review, ordinary_plan = stage / 'ordinary-review.json', stage / 'ordinary-plan.json'
        ordinary_review.write_text(json.dumps(dict(status='fresh_runtime_v3_ready', contract=ordinary,
                                                   source_sha256=sources)))
        ordinary_plan.write_text(json.dumps(dict(ordinary, review=dict(path=str(ordinary_review),
                                                                        sha256=bridge.seal(ordinary_review)))))
        ordinary_binding = dict(schema='fresh_runtime_v3', variant='scalar', contract=ordinary,
                                plan=str(ordinary_plan), plan_sha256=bridge.seal(ordinary_plan),
                                review=str(ordinary_review), review_sha256=bridge.seal(ordinary_review))
        template['fresh_runtime'] = ordinary_binding
        (outer / 'manifest.json').write_bytes(bridge.encode(template))
        with patch.object(current_sdk, '_check_fresh_binding_metadata', return_value=(ordinary, {})):
            ordinary_claim = current_sdk.claim_fresh(ordinary_binding, outer, application)
        verified = bridge.verify_fresh_claim(outer, ordinary_claim['sha256'], os.getuid(), os.getgid())
        self.assertEqual(verified['binding'], ordinary_binding)
        claim_path = Path(ordinary_claim['path'])
        altered = json.loads(claim_path.read_text())
        altered['specialization'] = 'observed'
        claim_path.write_text(json.dumps(altered))
        with self.assertRaisesRegex(ValueError, 'not bound'):
            bridge.verify_fresh_claim(outer, bridge.seal(claim_path), os.getuid(), os.getgid())

    def test_root_claim_receipt_cannot_be_reused(self):
        outer = self.root / 'attempt'
        outer.mkdir(mode=0o700)
        binding = {'variant': 'scalar'}
        authority = dict(claim_sha256='a' * 64, manifest_sha256='b' * 64, binding_sha256='c' * 64,
                         binding=binding, template=dict(uid=os.getuid(), gid=os.getgid(), fresh_runtime=binding))
        actual_fstat = os.fstat

        def simulated_root_file(descriptor):
            info = actual_fstat(descriptor)
            if stat.S_ISREG(info.st_mode):
                return SimpleNamespace(st_mode=info.st_mode, st_uid=0, st_gid=0, st_nlink=info.st_nlink,
                                       st_size=info.st_size)
            return info

        with patch.object(bridge.os, 'geteuid', return_value=0), patch.object(bridge.os, 'getegid', return_value=0), \
                patch.object(bridge.os, 'fstat', side_effect=simulated_root_file):
            record = bridge.consume_fresh_claim(outer, authority)
            self.assertEqual(json.loads((outer / 'root-fresh-claim.json').read_text()), record)
            with patch.object(bridge, '_owned_bounded_file', return_value=bridge.encode(record)):
                self.assertEqual(bridge.verify_root_claim_receipt(outer, authority), record)
                different = dict(authority, binding_sha256='d' * 64)
                with self.assertRaisesRegex(ValueError, 'root fresh claim receipt differs'):
                    bridge.verify_root_claim_receipt(outer, different)
            with self.assertRaises(FileExistsError):
                bridge.consume_fresh_claim(outer, authority)

    def test_fresh_replay_authenticates_its_own_claim_and_template(self):
        template = dict(fresh_runtime={'variant': 'scalar'})
        authority = dict(manifest_sha256='b' * 64, template=template)
        plan = dict(fresh_claim_sha256='a' * 64, template_sha256='b' * 64, uid=1003, gid=1003)
        with patch.object(bridge, 'verify_fresh_claim', return_value=authority) as claim, \
                patch.object(bridge, 'verify_root_claim_receipt') as receipt:
            self.assertEqual(bridge.authenticate_fresh_replay(plan, self.root, template), authority)
            claim.assert_called_once_with(self.root, 'a' * 64, 1003, 1003)
            receipt.assert_called_once_with(self.root, authority)
            with self.assertRaisesRegex(ValueError, 'authenticated template'):
                bridge.authenticate_fresh_replay(dict(plan, template_sha256='c' * 64), self.root, template)

    def test_fresh_input_preparation_uses_authenticated_unprivileged_service(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as mocks:
            outer = Path(temporary)
            service, frontend = outer / 'service', outer / 'frontend'
            gpu_id = 'GPU-0db675c8-4623-fc5a-9e7b-1c237f625ce2'
            environment = {'CUDA_VISIBLE_DEVICES': gpu_id}
            training = SimpleNamespace(GPU_FIELDS='fields', ORIGINAL=outer, PYTHON=Path('/python'),
                SERVER=service, PRELOAD=frontend, require_idle=Mock(), clean_environment=Mock(return_value=environment),
                read_json=Mock(return_value={'environment': {'device_uuid': gpu_id.removeprefix('GPU-')}}))
            helpers = SimpleNamespace(training=training, query=Mock(return_value=gpu_id + ',V100,driver,runtime'),
                                      wrapper='fixture wrapper')
            common = SimpleNamespace(STARTUP_INPUTS=outer, STARTUP_INPUTS_SHA='startup',
                                     application_references=Mock(return_value=(outer, outer)),
                                     input_paths=Mock(return_value=set()))
            binding = dict(contract=dict(target='2:3127', service_environment=environment,
                runtime=dict(service=dict(path=str(service)), frontend=dict(path=str(frontend)))))
            source_names = ('bert_contract.py', 'bert_workload.py', 'onnx_bert_workload.py', 'export_bert.py')
            native = dict(inference_numerical=dict(export_sha256='sealed', model_sha256='sealed',
                                                   source_sha256=dict.fromkeys(source_names, 'sealed')))
            pin = SimpleNamespace(verify=Mock())
            selection = (binding, set(), native, pin, dict(manifests=[]))
            mocks.enter_context(patch.object(current_sdk, 'configure_fresh', return_value=(common, helpers)))
            inventory = mocks.enter_context(patch.object(current_sdk, 'fresh_application_inventory',
                                                        return_value=(set(), {}, {})))
            selected = mocks.enter_context(patch.object(bridge, 'selected_service'))
            mocks.enter_context(patch.object(bridge, 'protected_system_file',
                                             side_effect=ValueError('CPU stop after service selection')))
            mocks.enter_context(patch.object(capture, 'digest', return_value='sealed'))
            mocks.enter_context(patch.object(capture.shutil, 'disk_usage', return_value=SimpleNamespace(free=32 * 1024**3)))
            with self.assertRaisesRegex(ValueError, 'CPU stop after service selection'):
                capture.prepare_inputs(outer, '2:3127', service_cap_sys_admin=True, fresh_selection=selection)
            pin.verify.assert_called_once()
            inventory.assert_called_once()
            selected.assert_not_called()
            self.assertFalse((outer / 'manifest.json').exists())

    def test_unit_retirement_independently_stops_reaps_and_checks(self):
        unit, description = 'glimmer-bert-attempt-fixture-equal.service', 'owned unit'
        active = dict(LoadState='loaded', ActiveState='active', Description=description,
                      ControlGroup='/owned', MainPID='42', ExecMainStartTimestampMonotonic='1')
        gone = dict(LoadState='not-found', ActiveState='inactive', Description=unit,
                    ControlGroup='', MainPID='0', ExecMainStartTimestampMonotonic='0')
        handshake = dict(group='/owned/' + unit, cgroup_device=1, cgroup_inode=2)
        for failure in (None, 'stop', 'reap', 'foreign', 'replaced', 'cgroup'):
            with self.subTest(failure=failure), ExitStack() as mocks:
                initial = dict(active, Description='foreign') if failure == 'foreign' else active
                final = initial if failure in ('foreign', 'replaced') else gone
                properties = mocks.enter_context(patch.object(capture, 'unit_properties',
                                               side_effect=[initial, initial, final]))
                stop = mocks.enter_context(patch.object(capture, 'command',
                    side_effect=OSError('stop failed') if failure == 'stop' else None))
                mocks.enter_context(patch.object(capture, 'require_owned_unit',
                    side_effect=ValueError('replaced unit') if failure == 'replaced' else
                    ValueError('foreign unit') if failure == 'foreign' else None))
                mocks.enter_context(patch.object(capture, 'task_cgroup_empty',
                                                side_effect=ValueError('cgroup populated') if failure == 'cgroup'
                                                else None, return_value=True))
                process = Mock()
                process.wait.side_effect = TimeoutError('reap failed') if failure == 'reap' else None
                result = capture.retire_unit(unit, description, process, True, handshake)
                self.assertEqual(properties.call_count, 3)
                process.wait.assert_called_once_with(timeout=10)
                if failure in ('foreign', 'replaced'):
                    stop.assert_not_called()
                    self.assertFalse(result['unit_removed'])
                if failure is None:
                    self.assertTrue(result['unit_removed'] and result['cgroup_empty'])
                    self.assertNotIn('cleanup_error', result)
                else:
                    self.assertIn('cleanup_error', result)
        with self.assertRaisesRegex(ValueError, 'active task cgroup changed'):
            capture.require_owned_unit(active, unit, description, True, handshake)
        matched = dict(active, ControlGroup=handshake['group'])
        with patch.object(Path, 'stat', return_value=SimpleNamespace(st_dev=1, st_ino=2)), \
                patch.object(Path, 'is_symlink', return_value=False):
            capture.require_owned_unit(matched, unit, description, True, handshake)

    def test_numerical_runtime_and_source_must_match(self):
        runtime = dict(target='/lib/runtime', bytes=10, sha256='runtime-hash')
        binding = dict(numerical_runtime_identity={'/lib/runtime': runtime},
                       numerical_source_sha256={'/src/source.cc': 'source-hash'})
        inputs = {'/lib/runtime': runtime.copy(), '/src/source.cc': {'sha256': 'source-hash'}}
        current_sdk.validate_numerical_inputs(binding, inputs)
        for name in inputs:
            changed = deepcopy(inputs)
            changed[name]['sha256'] = 'different'
            with self.subTest(path=name), self.assertRaises(ValueError):
                current_sdk.validate_numerical_inputs(binding, changed)


if __name__ == '__main__':
    unittest.main()
