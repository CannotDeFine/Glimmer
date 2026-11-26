"""CPU-only gates for host-pair activity composition and failure retirement."""

from contextlib import ExitStack
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import run_bert_host_pair_activity as activity
import run_bert_host_pair_numerical as pair


def controls():
    return [dict(variant=variant, mode=mode, inference=100, training=10,
                 sm_lower=low, sm_upper=high, minimum_coverage=.99)
            for (variant, mode), (low, high) in zip(activity.ORDER,
                ((.7, .72), (.1, .12), (.2, .22), (.71, .73)))]


class ActivityTests(unittest.TestCase):
    def test_fixed_order_native_gate_and_strict_activity_separation(self):
        rows = controls()
        result = activity.decision(rows)
        self.assertTrue(result['eligible'])
        self.assertEqual(result['verdict'], 'supports_increased_aggregate_activity')
        self.assertTrue(result['counter_collection'])
        for key in ('ordinary_timing', 'performance_qualified', 'full_device_qualified',
                    'native_operation_matched', 'saturation_qualified'):
            self.assertFalse(result[key])
        rows[2]['sm_lower'] = rows[1]['sm_upper']
        self.assertEqual(activity.decision(rows)['verdict'], 'inconclusive')
        rows[2].update(sm_lower=.05, sm_upper=.08)
        self.assertEqual(activity.decision(rows)['verdict'], 'aggregate_activity_decreased')
        rows[3].update(sm_lower=.8, sm_upper=.82)
        self.assertFalse(activity.decision(rows)['eligible'])
        self.assertEqual(activity.decision(rows)['verdict'], 'inconclusive')
        with self.assertRaises(ValueError):
            activity.decision(list(reversed(controls())))

    def test_native_drift_bound_and_invalid_values(self):
        rows = controls()
        rows[3].update(inference=120, training=12)
        self.assertTrue(activity.decision(rows)['eligible'])
        for key, value in (('inference', 121), ('training', 13), ('inference', 0)):
            bad = copy.deepcopy(rows)
            bad[3][key] = value
            self.assertFalse(activity.decision(bad)['eligible'])
        for key, value in (('inference', True), ('training', -1), ('sm_lower', float('nan')),
                           ('sm_upper', float('inf')), ('minimum_coverage', .949999),
                           ('sm_lower', .9)):
            bad = controls()
            bad[2][key] = value
            with self.assertRaises(ValueError):
                activity.decision(bad)

        for index in (1, 2):
            for key in ('inference', 'training'):
                bad = controls()
                bad[index][key] = 0
                self.assertFalse(activity.decision(bad)['eligible'])
                self.assertEqual(activity.decision(bad)['verdict'], 'inconclusive')

    def test_all_fields_require_coverage_and_keep_bounds(self):
        row = dict(interior_coverage_fraction=.95, whole_window_ratio_lower_bound=.1,
                   whole_window_ratio_upper_bound=.15, interior_duration_weighted_ratio=.11)
        data = dict(fields={str(key): dict(row) for key in activity.aggregate.FIELDS})
        with patch.object(activity.timing, 'counts', return_value=dict(inference=2, training=3)):
            result = activity.summarize(data, {})
            self.assertEqual((result['sm_lower'], result['sm_upper'], result['minimum_coverage']), (.1, .15, .95))
            for key in activity.aggregate.FIELDS:
                bad = copy.deepcopy(data)
                bad['fields'][str(key)]['interior_coverage_fraction'] = .949
                with self.assertRaises(ValueError):
                    activity.summarize(bad, {})
            del data['fields']['1005']
            with self.assertRaises(ValueError):
                activity.summarize(data, {})

    def test_accepted_evidence_requires_matching_report_review_and_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            documents = {'report.json': {'status': 'raw'},
                         'review.json': {'status': 'accepted', 'report_sha256': 'report-sha'}}
            binding = dict(attempt=str(root), report_sha256='report-sha', review_sha256='review-sha')
            with patch.object(activity, 'read', side_effect=lambda path: documents[path.name]), \
                 patch.object(activity, 'digest', side_effect=lambda path: path.stem + '-sha'):
                self.assertEqual(activity.accepted(binding, 'accepted')[1]['status'], 'raw')
                for field, value in (('status', 'pending'), ('report_sha256', 'other')):
                    old = documents['review.json'][field]
                    documents['review.json'][field] = value
                    with self.assertRaises(ValueError):
                        activity.accepted(binding, 'accepted')
                    documents['review.json'][field] = old
                with self.assertRaises(ValueError):
                    activity.accepted(dict(binding, review_sha256='other'), 'accepted')

    def test_children_cannot_duplicate_or_escape_sampler_namespace(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = [root / ('attempt-' + str(index)) for index in range(4)]
            for path in paths:
                path.mkdir()
            with patch.object(activity.observer, 'OUTPUT_ROOT', root):
                self.assertEqual(activity.children(dict(children=list(map(str, paths)))), paths)
                for names in ([*paths[:3], paths[0]], paths[:3], [*paths[:3], root]):
                    with self.assertRaises(ValueError):
                        activity.children(dict(children=list(map(str, names))))

    def test_persisted_preflight_rejects_changed_missing_and_stale_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            activity.save(root/'plan.json', {'run_index': 0})
            launch = dict(mode='native', attempt=str(root), plan_sha256=activity.digest(root/'plan.json'),
                          full_preflight_status='complete_input_seals_unchanged', started_ns=10,
                          preflight_verified_ns=10, outer_deadline_ns=820_000_000_010,
                          collector={'pid': 123, 'start_ticks': 456, 'uid': 1000, 'cgroup': '/test'})
            path = root/'native-launch.json'
            activity.save(path, launch)
            binding = activity.preflight_binding(root, 'native', observed_ns=30_000_000_010)
            activity.validate_preflight(root, 'native', binding)
            with self.assertRaises(ValueError):
                activity.preflight_binding(root, 'native', observed_ns=30_000_000_011)
            activity.save(path, dict(launch, collector=dict(launch['collector'], pid=124)))
            with self.assertRaises(ValueError):
                activity.validate_preflight(root, 'native', binding)
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                activity.validate_preflight(root, 'native', binding)
            activity.save(path, launch)
            activity.save(root/'plan.json', {'run_index': 1})
            with self.assertRaises(ValueError):
                activity.validate_preflight(root, 'native', binding)

    def worker_case(self, failure):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            h = SimpleNamespace(query=Mock(return_value='gpu'), training=SimpleNamespace(
                GPU_FIELDS='fields', require_idle=Mock(), clean_environment=Mock(return_value={})))
            common = SimpleNamespace(application_references=Mock(return_value=(root, root)))
            sustained = SimpleNamespace(run_control=Mock(return_value={}))
            resources = SimpleNamespace(resource_snapshot=Mock(return_value={'available_device_sms': 80}))
            guard = Mock(receipt={})
            proof = SimpleNamespace(expected={})
            supervisor = Mock(receipt={'ready': {'ready_ns': 1}}, watch_plan={'owner': {}})
            supervisor.start.return_value = proof
            if failure == 'start':
                supervisor.start.side_effect = ValueError('start-failure')
            if failure == 'application':
                sustained.run_control.side_effect = ValueError('application-failure')
            if failure == 'stop':
                supervisor.stop.side_effect = ValueError('stop-failure')
            if failure == 'close':
                supervisor.close.side_effect = ValueError('close-failure')
            stack.enter_context(patch.object(activity, 'worker_inputs', return_value=(
                root, {'gpu': 'gpu'}, 0, {}, {}, {'numerical_report': str(root/'report.json'),
                                                 'native_reference': str(root/'native')}, {})))
            stack.enter_context(patch.object(activity.timing, 'helpers', return_value=(
                {}, common, h, sustained, {}, resources)))
            stack.enter_context(patch.object(activity, 'read', return_value={}))
            stack.enter_context(patch.object(activity, 'ActivityGuard', return_value=guard))
            stack.enter_context(patch.object(activity.capture, 'SamplerSupervisor', return_value=supervisor))
            stack.enter_context(patch.object(activity.observer, 'scoped_helpers', return_value=ExitStack()))
            stack.enter_context(patch.object(activity.capture, 'audit_application', return_value=(10, 20)))
            stack.enter_context(patch.object(activity.capture, 'audit_guards', return_value={}))
            stack.enter_context(patch.object(activity.timing, 'validate_job'))
            stack.enter_context(patch.object(activity.aggregate, 'read_records', return_value=[]))
            stack.enter_context(patch.object(activity.aggregate, 'analyze_records', return_value={}))
            stack.enter_context(patch.object(activity, 'summarize', return_value={}))
            stack.enter_context(patch.object(activity, 'digest', return_value='raw-sha'))
            status = activity.worker(root, 'native')
            report = json.loads((root/'native-worker.json').read_text())
            self.assertTrue(supervisor.stop.called, report)
            supervisor.stop.assert_called_once()
            supervisor.close.assert_called_once()
            if failure:
                self.assertEqual(status, 1)
                self.assertEqual(report['status'], 'failed')
                self.assertIn(failure + '-failure', json.dumps(report))
                if failure == 'close':
                    self.assertEqual(report['job']['status'], 'failed')
            else:
                self.assertEqual(status, 0)
                self.assertEqual(report['job']['raw_helper_receipt']['path'], 'application/attempt.json')
                self.assertTrue(report['job']['counter_collection'])
                self.assertFalse(report['job']['ordinary_timing'])

    def test_worker_success_preserves_counter_only_provenance(self):
        self.worker_case(None)

    def test_worker_failure_always_retires_sampler_and_preserves_failure(self):
        for failure in ('start', 'application', 'stop', 'close'):
            with self.subTest(failure=failure):
                self.worker_case(failure)

    def test_collector_stops_on_first_failed_unit_and_claims_only_once(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            plan_path = root/'plan.json'
            activity.save(plan_path, {'launch_review': str(root/'review.json')})
            outputs = [root/('attempt-'+str(index)) for index in range(5)]
            for path in outputs:
                path.mkdir()
            h = SimpleNamespace(query=Mock(return_value='gpu'), training=SimpleNamespace(
                GPU_FIELDS='fields', require_idle=Mock(), manifest=Mock(return_value={})))
            stack.enter_context(patch.object(activity, 'references', return_value=(
                {}, {'gpu_identity_after': 'gpu'}, {}, set())))
            stack.enter_context(patch.object(activity, 'reviewed'))
            stack.enter_context(patch.object(activity.numerical, 'authorization'))
            stack.enter_context(patch.object(activity.shutil, 'disk_usage', return_value=SimpleNamespace(free=20*1024**3)))
            stack.enter_context(patch.object(activity.subprocess, 'run', return_value=SimpleNamespace(returncode=0)))
            stack.enter_context(patch.object(activity.timing, 'helpers', return_value=({}, None, h, None, None, None)))
            allocated = stack.enter_context(patch.object(activity, 'create_attempt', side_effect=outputs))
            stack.enter_context(patch.object(activity, 'wait_for_idle', return_value={'achieved': True}))
            stack.enter_context(patch.object(activity.observer, 'process_record', return_value={'pid': 123}))
            stack.enter_context(patch.object(activity, 'budget', return_value=0))
            summary = {key: value for key, value in controls()[0].items() if key not in ('variant', 'mode')}
            launch = stack.enter_context(patch.object(activity.candidate, 'launch_unit', side_effect=[
                ({'returncode': 0}, {'summary': summary}), ValueError('unit-failed')]))
            self.assertEqual(activity.collect(plan_path), 1)
            result = json.loads((outputs[0]/'report.json').read_text())
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['launched_run'], 2)
            self.assertEqual(len(result['jobs']), 1)
            self.assertIn('unit-failed', result['error'])
            self.assertEqual(launch.call_count, 2)
            with self.assertRaises(FileExistsError):
                activity.collect(plan_path)
            self.assertEqual(allocated.call_count, 5)
            self.assertEqual(launch.call_count, 2)


class PairBuildContractTests(unittest.TestCase):
    """Shared numerical prerequisite gates; these fixtures never launch CUDA."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.build = self.root / 'build-evidence'
        self.build.mkdir()
        self.runtime = self.root / 'runtime.json'
        self.plan = dict(build=str(self.build), build_review=str(self.build / 'review.json'),
                         runtime_manifest=str(self.runtime), variant_order=list(pair.VARIANTS),
                         modes=list(pair.MODES),
                         runtime_environment={'GLIMMER_EXECUTION_HOST_RANGE_SEEK': '1'})
        self.names = ('glimmer_execution_service', 'glimmer_execution_interceptor')
        self.source_patch = patch.object(pair, 'PROJECT_ROOT', self.root)
        self.source_patch.start()
        self.addCleanup(self.source_patch.stop)

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return pair.digest(path)

    def bind(self, review, runtime):
        self.plan['build_review_sha256'] = self.write(self.build / 'review.json', review)
        self.plan['runtime_manifest_sha256'] = self.write(self.runtime, runtime)
        return self.plan

    def product(self, directory, name):
        path = self.build / directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(directory + name)
        path.chmod(0o555)
        info = path.stat()
        return dict(path=str(path), mode=0o555, device=info.st_dev, inode=info.st_ino,
                    size=info.st_size, ctime_ns=info.st_ctime_ns, sha256=pair.digest(path))

    def normal(self):
        self.plan['build_contract'] = 'normal_root_v1'
        runtime = dict(build_contract='normal_root_v1', build=str(self.build),
                       variant_options={'o0': 'OFF', 'o2': 'ON'}, variants={}, config_evidence={})
        review = dict(status='normal_execution_host_o2_cpu_integration_accepted',
                      owned_changes={}, evidence_sha256={})
        for name in ('CMakeLists.txt', 'cmake/ExecutionHostOptimization.cmake',
                     'tests/CMakeLists.txt', 'tests/build/execution_host_optimization_test.py',
                     'docs/BUILDING.md'):
            review['owned_changes'][name] = self.write(self.root / name, {'source': name})
        for mode in ('off', 'on'):
            name = mode + '-graph.tar.gz'
            runtime['config_evidence'][name] = self.write(self.build / name, {'graph': mode})
        configuration = dict(status='normal_root_off_on_off_configuration_audit_passed',
                             **{mode + '_graph_sha256': runtime['config_evidence'][mode + '-graph.tar.gz']
                                for mode in ('off', 'on')})
        runtime['config_evidence']['configuration-audit.json'] = self.write(
            self.build / 'configuration-audit.json', configuration)
        review['evidence_sha256']['current-owned-source.tar.gz'] = self.write(
            self.build / 'current-owned-source.tar.gz', {'source': 'frozen'})
        closure = dict(status='complete_normal_pair_link_closure_verified',
                       all_six_archives_seen=True, off_on_extracted_sets_equal=True,
                       all_members_in_ordered_79_member_inventory=True,
                       source_capsule_sha256=review['evidence_sha256']['current-owned-source.tar.gz'],
                       project_members={'off': {}, 'on': {}}, project_member_counts={})
        archives = {str(index): dict(path=str(self.build / f'archive{index}.a'),
                                    members=[f'object{member}.o' for member in range(count)])
                    for index, count in enumerate((14, 13, 13, 13, 13, 13))}
        extracted = [dict(archive=item['path'], member=f'object{index}.o')
                     for index in range(14) for item in archives.values()
                     if f'object{index}.o' in item['members']]
        for variant, mode in (('o0', 'off'), ('o2', 'on')):
            products = {}
            runtime['variants'][variant] = {}
            for name in self.names:
                item = self.product(mode, name)
                item['link_map_sha256'] = self.write(self.build / f'{mode}-{name}.map', {'map': mode + name})
                products[name] = dict(path=item['path'], sha256=item['sha256'],
                                      map_sha256=item['link_map_sha256'])
                runtime['variants'][variant][name] = item
            report = dict(status='normal_variant_cpu_evidence_recorded' if mode == 'off' else
                          'normal_variant_cpu_evidence_recorded_after_independent_predicate_review',
                          variant=mode, compiled_production_cxx_objects=82, tests_passed=45,
                          products=products, archives=archives,
                          objects={str(index): {} for index in range(82)},
                          abi='same', pair_contracts_equal=True,
                          frontend_exports_equal=True, service_strong_exports_equal=True)
            review['evidence_sha256'][mode + '-verification.json'] = self.write(
                self.build / (mode + '-verification.json'), report)
            closure['project_member_counts'][mode] = dict(zip(self.names, (47, 13)))
            closure['project_members'][mode] = dict(zip(self.names, (extracted[:47], extracted[:13])))
        review['evidence_sha256']['pair-link-closure.json'] = self.write(
            self.build / 'pair-link-closure.json', closure)
        return self.bind(review, runtime)

    def legacy(self):
        review = dict(status='consistent_host_pair_cpu_verified', application_validation_ready=True)
        runtime = dict(build=str(self.build), variants={})
        self.write(self.build / 'frozen-inputs.json', {})
        self.write(self.build / 'build-report.json', {})
        for variant in pair.VARIANTS:
            products, members, entries = {}, {}, {}
            capsule = self.build / (variant + '-working.tar.gz')
            seal = dict(sha256=self.write(capsule, {'capsule': variant}), all_members_byte_verified=True)
            self.write(self.build / (variant + '-capsule.json'), seal)
            for name in self.names:
                path = self.root / variant / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(variant + name)
                path.chmod(0o555)
                info = path.stat()
                item = dict(path=str(path), mode=0o555, device=info.st_dev, inode=info.st_ino,
                            size=info.st_size, ctime_ns=info.st_ctime_ns, sha256=pair.digest(path),
                            member='products/' + name, capsule=str(capsule), capsule_sha256=seal['sha256'],
                            link_map_sha256='map-sha')
                entries[name] = item
                products[name] = dict(sha256=item['sha256'], map_sha256='map-sha')
                members[item['member']] = item['sha256']
                members['products/' + name + '.map'] = 'map-sha'
            runtime['variants'][variant] = entries
            self.write(self.build / (variant + '-products.json'), products)
            self.write(self.build / (variant + '-working-identities.json'), members)
        return self.bind(review, runtime)

    def test_legacy_success_and_contract_cannot_be_relabelled(self):
        plan = self.legacy()
        paths = pair.validate_plan(plan)
        self.assertIn(self.root / 'o0' / self.names[0], paths)
        with self.assertRaises(ValueError):
            pair.validate_plan(dict(plan, build_contract='normal_root_v1'))
        self.assertNotIn('build_contract', plan)
        with patch.object(pair, 'validate_legacy_plan', wraps=pair.validate_legacy_plan) as legacy:
            self.assertEqual(pair.validate_plan(plan), paths)
            legacy.assert_called_once_with(plan)

    def test_normal_success_and_explicit_variant_selection(self):
        plan = self.normal()
        paths = pair.validate_plan(plan)
        self.assertEqual(pair.selected_plan(plan, 'o0')['server'], str(self.build / 'off' / self.names[0]))
        self.assertEqual(pair.selected_plan(plan, 'o2')['server'], str(self.build / 'on' / self.names[0]))
        for name in ('review.json', 'off-verification.json', 'on-verification.json',
                     'pair-link-closure.json', 'current-owned-source.tar.gz', 'off-graph.tar.gz'):
            self.assertIn(self.build / name, paths)
        with self.assertRaises(ValueError):
            pair.validate_plan({key: value for key, value in plan.items() if key != 'build_contract'})
        for value in ('', None, 'legacy', 'normal_root_v2'):
            with self.subTest(contract=value), self.assertRaises(ValueError):
                pair.validate_plan(dict(plan, build_contract=value))

    def test_normal_rejects_mutated_evidence_source_map_and_product(self):
        plan = self.normal()
        targets = [self.build / name for name in ('review.json', 'off-verification.json',
                   'pair-link-closure.json', 'current-owned-source.tar.gz', 'off-graph.tar.gz',
                   'off-glimmer_execution_service.map')]
        targets += [self.root / 'CMakeLists.txt']
        for path in targets:
            old = path.read_bytes()
            path.write_bytes(old + b' ')
            with self.subTest(path=path), self.assertRaises(ValueError):
                pair.validate_plan(plan)
            path.write_bytes(old)
        product = self.build / 'off' / self.names[0]
        product.chmod(0o755)
        with self.assertRaises(ValueError):
            pair.validate_plan(plan)
        product.chmod(0o555)
        with self.assertRaises(ValueError):
            pair.validate_plan(plan)  # Restoring mode does not restore ctime.

    def test_normal_rejects_resealed_missing_pins_cross_variant_and_foreign_identity(self):
        plan = self.normal()
        review, runtime = pair.read(plan['build_review']), pair.read(self.runtime)
        changes = []
        missing = copy.deepcopy(review)
        del missing['evidence_sha256']['pair-link-closure.json']
        changes.append((missing, runtime))
        for key in ('build_contract', 'variant_options'):
            changed = copy.deepcopy(runtime)
            changed[key] = 'legacy' if key == 'build_contract' else {'o0': 'ON', 'o2': 'OFF'}
            changes.append((review, changed))
        for key in ('path', 'inode', 'sha256'):
            changed = copy.deepcopy(runtime)
            entry = changed['variants']['o0'][self.names[0]]
            entry[key] = changed['variants']['o2'][self.names[0]][key]
            changes.append((review, changed))
        for bad_review, bad_runtime in changes:
            with self.subTest(review=bad_review['status']), self.assertRaises(ValueError):
                pair.validate_plan(self.bind(bad_review, bad_runtime))
        self.assertTrue(pair.validate_plan(self.bind(review, runtime)))

    def test_timing_requires_its_own_accepted_numerical_prerequisite_for_either_contract(self):
        timing = activity.timing
        for make_plan, label in ((self.legacy, 'legacy'), (self.normal, 'normal')):
            plan = make_plan()
            reference = self.root / ('numerical-' + label)
            reference.mkdir()
            plan['entrypoint_sha256'] = pair.digest(pair.ENTRY)
            plan['qualifier_review'] = str(reference / 'qualifier.json')
            plan['qualifier_review_sha256'] = self.write(Path(plan['qualifier_review']),
                dict(status='numerical_qualifier_ready', entry_sha256=plan['entrypoint_sha256']))
            self.write(reference / 'plan.json', plan)
            self.write(reference / 'manifest.json', {})
            inputs_sha = self.write(reference / 'inputs-before.json', {})
            self.write(reference / 'inputs-after.json', {})
            report = dict(status='fresh_o0_o2_native_equal_numerical_requires_review',
                          input_identity_unchanged=True, variant_attempts={}, jobs=[])
            for variant in pair.VARIANTS:
                child = reference / variant
                report['variant_attempts'][variant] = str(child)
                self.write(child / 'plan.json', pair.selected_plan(plan, variant))
                for mode in pair.MODES:
                    job = dict(mode=mode, artifact_identities={})
                    worker = dict(status='passed_requires_review', reparse=pair.ORACLE, job=job)
                    unit = dict(returncode=0, unit_removed=True)
                    self.write(child / (mode + '-worker.json'), worker)
                    self.write(child / (mode + '-unit.json'), unit)
                    self.write(child / mode / 'attempt.json', job)
                    report['jobs'].append(dict(variant=variant, mode=mode, worker=worker, unit=unit))
            report_sha = self.write(reference / 'report.json', report)
            review = dict(status='fresh_host_pair_numerical_accepted', report_sha256=report_sha,
                          inputs_manifest_sha256=inputs_sha)
            request = dict(order=[list(row) for row in timing.ORDER], point=timing.POINT,
                           runtime_environment=plan['runtime_environment'], numerical_reference=str(reference),
                           numerical_report_sha256=report_sha,
                           numerical_review_sha256=self.write(reference / 'review.json', review),
                           runtime_manifest=plan['runtime_manifest'],
                           runtime_manifest_sha256=plan['runtime_manifest_sha256'])
            # Isolate the receipt-routing boundary; the complete job oracle has its own tests.
            with patch.object(pair, 'validate_job') as jobs:
                self.assertEqual(timing.validate_reference(request)[0], plan)
                self.assertEqual(jobs.call_count, 4)
                review['status'] = 'requires_review'
                request['numerical_review_sha256'] = self.write(reference / 'review.json', review)
                with self.assertRaises(ValueError):
                    timing.validate_reference(request)

    def test_expired_unsigned_plan_cannot_allocate_or_launch(self):
        plan = dict(self.normal(), authorization_deadline_utc='1970-01-01T00:00:00+00:00')
        path = self.root / 'plan-unsigned.json'
        self.write(path, plan)
        with patch.object(pair, 'create_attempt') as allocate, \
                patch.object(pair.candidate, 'launch_unit') as launch, self.assertRaises(ValueError):
            pair.collect(path)
        allocate.assert_not_called()
        launch.assert_not_called()


class PairPrelaunchIdleTests(unittest.TestCase):
    BUSY = 'GPU is not idle; no next job; never signal foreign work: another process'

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.plan, self.before, self.report = {}, {'input': 'unchanged'}, {}
        self.training = SimpleNamespace(require_idle=Mock(), manifest=Mock(return_value=self.before))
        self.helpers = SimpleNamespace(training=self.training)

    def check(self, *, first=False):
        return pair.prelaunch_idle(self.root, 'o0', 'native' if first else 'equal',
                                   self.plan, self.helpers, {'input'}, self.before,
                                   self.report, first=first)

    @staticmethod
    def idle_receipt(directory, achieved=True):
        (directory / 'idle-window.jsonl').write_text('{"sample": true}\n')
        return {'achieved': achieved}

    def test_idle_proceeds_immediately_and_only_later_busy_may_wait(self):
        with patch.object(pair, 'wait_for_idle') as wait:
            self.assertEqual(self.check(first=True)['status'], 'immediate_idle')
            self.assertEqual(self.check()['status'], 'immediate_idle')
            self.training.require_idle.side_effect = RuntimeError(self.BUSY)
            with self.assertRaises(RuntimeError):
                self.check(first=True)
            self.training.require_idle.side_effect = RuntimeError('GPU query failed')
            with self.assertRaisesRegex(RuntimeError, 'GPU query failed'):
                self.check()
            wait.assert_not_called()

    def test_busy_later_task_waits_once_and_rechecks_all_inputs(self):
        self.training.require_idle.side_effect = [RuntimeError(self.BUSY), None]
        with patch.object(pair, 'wait_for_idle', side_effect=self.idle_receipt) as wait, \
                patch.object(pair, 'validate_plan') as validate, \
                patch.object(pair.numerical, 'authorization') as authorization:
            result = self.check()
        self.assertEqual(result['status'], 'idle_after_conditional_wait')
        self.assertEqual(wait.call_count, 1)
        self.assertEqual(self.training.require_idle.call_count, 2)
        self.training.manifest.assert_called_once_with({'input'})
        validate.assert_called_once_with(self.plan)
        self.assertEqual([call.args[1] for call in authorization.call_args_list], [1440, 840])
        self.assertEqual(Path(result['wait_directory']), self.root / 'prelaunch-idle/o0-equal')
        self.assertEqual(result['wait_log_sha256'], pair.digest(Path(result['wait_directory']) / 'idle-window.jsonl'))

    def test_timeout_changed_input_expired_authority_and_second_busy_stop(self):
        for failure in ('timeout', 'input', 'authority', 'second_busy'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                self.root = Path(directory)
                self.report = {}
                self.training.require_idle.reset_mock()
                self.training.require_idle.side_effect = [RuntimeError(self.BUSY),
                    RuntimeError(self.BUSY) if failure == 'second_busy' else None]
                self.training.manifest.return_value = {'changed': True} if failure == 'input' else self.before
                authorization_error = [None, ValueError('authorization expired')] if failure == 'authority' else None
                with patch.object(pair, 'wait_for_idle',
                                  side_effect=lambda path: self.idle_receipt(path, failure != 'timeout')) as wait, \
                        patch.object(pair, 'validate_plan'), \
                        patch.object(pair.numerical, 'authorization', side_effect=authorization_error):
                    with self.assertRaises((ValueError, RuntimeError)):
                        self.check()
                self.assertEqual(wait.call_count, 1)
                self.assertEqual(self.report['prelaunch_idle'][0]['status'], 'failed')

    def test_failed_unit_stops_batch_without_another_wait_or_launch(self):
        with ExitStack() as stack:
            plan_path = self.root / 'plan.json'
            pair.save(plan_path, {'qualifier_review': str(self.root / 'review.json')})
            outputs = [self.root / 'batch', self.root / 'o0', self.root / 'o2']
            for path in outputs:
                path.mkdir()
            self.training.GPU_FIELDS = 'fields'
            self.training.SERVER = self.training.PRELOAD = None
            self.training.require_idle.side_effect = [None, RuntimeError(self.BUSY), None]
            helpers = SimpleNamespace(training=self.training, query=Mock(return_value=
                'GPU-0db675c8-4623-fc5a-9e7b-1c237f625ce2, V100'))
            common = SimpleNamespace(STARTUP_INPUTS=self.root, STARTUP_INPUTS_SHA='reference',
                application_references=Mock(return_value=(self.root, self.root)), input_paths=Mock(return_value=set()))
            stack.enter_context(patch.object(pair, 'require_launch_review'))
            stack.enter_context(patch.object(pair, 'validate_plan', return_value=set()))
            stack.enter_context(patch.object(pair, 'selected_plan', side_effect=lambda plan, variant: dict(
                plan, variant=variant, server=str(self.root/'service'), frontend=str(self.root/'frontend'))))
            authorization = stack.enter_context(patch.object(pair.numerical, 'authorization'))
            stack.enter_context(patch.object(pair.shutil, 'disk_usage',
                                             return_value=SimpleNamespace(free=20*1024**3)))
            stack.enter_context(patch.object(pair.candidate, 'load_helpers', return_value=(common, helpers)))
            stack.enter_context(patch.object(pair, 'resolve_artifact_root', return_value=None))
            stack.enter_context(patch.object(pair, 'create_attempt', side_effect=outputs))
            wait = stack.enter_context(patch.object(pair, 'wait_for_idle', side_effect=self.idle_receipt))
            stack.enter_context(patch.object(pair, 'validate_job'))
            stack.enter_context(patch.object(pair, 'aggregate_budget', return_value=0))
            first = ({'returncode': 0}, {'job': {}, 'reparse': pair.ORACLE})
            launch = stack.enter_context(patch.object(pair.candidate, 'launch_unit',
                                                     side_effect=[first, ValueError('unit failed')]))
            self.assertEqual(pair.collect(plan_path), 1)
        report = json.loads((outputs[0] / 'report.json').read_text())
        self.assertEqual(report['status'], 'failed')
        self.assertIn('unit failed', report['error'])
        self.assertEqual([(row['variant'], row['mode']) for row in report['jobs']], [('o0', 'native')])
        self.assertEqual([event['status'] for event in report['prelaunch_idle']],
                         ['immediate_idle', 'idle_after_conditional_wait'])
        self.assertEqual(wait.call_count, 2)  # Initial window plus one conditional wait.
        self.assertEqual(launch.call_count, 2)  # No relaunch and no later task.
        self.assertEqual(authorization.call_args_list[0].args[1], 6200)


if __name__ == '__main__':
    unittest.main()
