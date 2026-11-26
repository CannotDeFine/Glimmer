"""CPU-only composition and rejection checks for the bounded host-stage pair."""

import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import run_bert_host_pair_stages as collector


def inputs():
    ordinary = dict(jobs=[dict(variant=variant, mode='equal',
        worker=dict(counts=dict(inference=100, training=10)))
        for variant in collector.ORDER for _ in range(2)])
    controls = [dict(variant=variant, counts=dict(inference=80, training=8),
        service_stages=dict(attribution_coverage_gate=True),
        groups=[dict(group='planning', observed_cpu_fraction=.6, cpu_window_fraction=.2)])
        for variant in collector.ORDER]
    return controls, ordinary


class HostStageTests(unittest.TestCase):
    def test_historical_runtime_gate_excludes_fixture_changes_not_runtime_changes(self):
        runtime = str(collector.PROJECT_ROOT / 'src/execution_service/runtime.cc')
        fixture = str(collector.PROJECT_ROOT / 'tests/examples/bert_ncu_relay_test.py')
        original = {runtime: {'sha256': 'runtime'}, fixture: {'sha256': 'old-test'}}
        manifest = Mock(return_value={runtime: {'sha256': 'runtime'}})
        collector.verify_runtime_inputs(original, manifest)
        manifest.assert_called_once_with({Path(runtime)})
        manifest.return_value = {runtime: {'sha256': 'changed-runtime'}}
        with self.assertRaises(ValueError):
            collector.verify_runtime_inputs(original, manifest)
        with self.assertRaises(ValueError):
            collector.verify_runtime_inputs({fixture: original[fixture]}, manifest)

    def test_exact_perturbation_and_boundary_gates(self):
        controls, ordinary = inputs()
        result = collector.decision(controls, ordinary)
        self.assertTrue(result['eligible'])
        self.assertEqual(result['selected_group'], 'planning')
        for key in ('ordinary_timing', 'counter_collection', 'performance_qualified',
                    'full_device_qualified', 'gpu_gap_causality_qualified',
                    'service_frontend_causality_qualified'):
            self.assertFalse(result[key])
        for role, value in (('inference', 79), ('training', 7), ('inference', 0)):
            bad = copy.deepcopy(controls)
            bad[0]['counts'][role] = value
            self.assertFalse(collector.decision(bad, ordinary)['eligible'])
        for key, value in (('observed_cpu_fraction', .59999), ('cpu_window_fraction', .19999)):
            bad = copy.deepcopy(controls)
            bad[1]['groups'][0][key] = value
            self.assertEqual(collector.decision(bad, ordinary)['verdict'], 'mixed_stop')
        controls[1]['groups'].append(dict(group='issue', observed_cpu_fraction=.6, cpu_window_fraction=.2))
        self.assertEqual(collector.decision(controls, ordinary)['verdict'], 'mixed_stop')

    def test_coverage_order_and_invalid_counts(self):
        controls, ordinary = inputs()
        controls[1]['service_stages']['attribution_coverage_gate'] = False
        self.assertIsNone(collector.decision(controls, ordinary)['selected_group'])
        with self.assertRaises(ValueError):
            collector.decision(list(reversed(controls)), ordinary)
        for value in (True, -1, 1.5):
            controls[0]['counts']['training'] = value
            with self.assertRaises(ValueError):
                collector.decision(controls, ordinary)

    def test_disjoint_groups_use_requested_window_and_same_window_completions(self):
        stages = dict(stages=[dict(stage=name, calls=10, exclusive_cpu_ns=100)
                              for name in collector.stage_parser.STAGES],
                      exclusive_observed_cpu_ns=1500, requested_duration_ns=5000)
        groups = collector.groups(stages, dict(inference=5, training=0))
        self.assertEqual(sum(row['exclusive_cpu_ns'] for row in groups), 1500)
        planning = next(row for row in groups if row['group'] == 'planning')
        self.assertEqual(planning['cpu_window_fraction'], .16)
        self.assertEqual(planning['cpu_ns_per_step'], 80)
        self.assertEqual(planning['cpu_ns_per_completion'], dict(inference=160, training=None))
        stages['stages'].pop()
        with self.assertRaises(ValueError):
            collector.groups(stages, {})

    def test_stage_hook_restores_after_exception_and_rejects_duplicate(self):
        original = Mock(return_value=(['service'], ['pins']))
        helper = SimpleNamespace(owner_command=original)
        with self.assertRaises(RuntimeError):
            with collector.stage_option(helper, Path('/tmp/stage-test')):
                self.assertEqual(helper.owner_command()[0],
                    ['service', '--host-stage-output', '/tmp/stage-test/host-stages.json'])
                raise RuntimeError('application failure')
        self.assertIs(helper.owner_command, original)
        original.return_value = (['service', '--host-stage-output', 'other'], [])
        with collector.stage_option(helper, Path('/tmp/stage-test')):
            with self.assertRaises(ValueError):
                helper.owner_command()
        self.assertIs(helper.owner_command, original)

    def test_exact_stage_option_only_then_unchanged_command_validator(self):
        directory = Path('/tmp/stage-test')
        base = ['service', '--socket', 'socket']
        job = dict(mode='equal', server_command=base + ['--host-stage-output', str(directory / 'host-stages.json')])
        with patch.object(collector.timing, 'validate_job') as validate:
            collector.validate_job({}, {}, job, directory)
            self.assertEqual(validate.call_args.args[2]['server_command'], base)
            self.assertEqual(len(job['server_command']), 5)
            for argv in (base, job['server_command'] + ['extra'],
                         base + ['--host-stage-output', '/tmp/foreign'],
                         job['server_command'] + job['server_command'][-2:]):
                with self.assertRaises(ValueError):
                    collector.validate_job({}, {}, dict(job, server_command=argv), directory)
        with patch.object(collector.timing, 'validate_job', side_effect=ValueError('extra option')):
            with self.assertRaises(ValueError):
                collector.validate_job({}, {}, job, directory)

    def test_canonical_envelope_preserves_raw_and_suppresses_timing(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            job = dict(status=collector.capture.SUCCESS, ordinary_timing=True, mode='equal')
            collector.save(directory / 'attempt.json', job)
            original = (directory / 'attempt.json').read_bytes()
            canonical = collector.canonical_job(directory, job)
            self.assertEqual((directory / 'helper-attempt.json').read_bytes(), original)
            self.assertEqual(collector.read(directory / 'attempt.json'), canonical)
            self.assertTrue(job['ordinary_timing'])
            self.assertFalse(canonical['ordinary_timing'])
            self.assertEqual(canonical['raw_helper_receipt']['sha256'],
                             collector.digest(directory / 'helper-attempt.json'))

    def test_worker_preflight_failure_is_recorded_without_launch_or_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            collector.save(directory / 'plan.json', {})
            with patch.object(collector.numerical, 'authorization', side_effect=ValueError('expired')), \
                 patch.object(collector.timing, 'helpers') as helpers:
                self.assertEqual(collector.worker(directory, 'equal'), 1)
                helpers.assert_not_called()
            result = json.loads((directory / 'equal-worker.json').read_text())
            self.assertEqual(result['status'], 'failed')
            self.assertIn('expired', result['error'])
            self.assertFalse(result['ordinary_timing'])

    def test_canonical_flags_remain_compatible_with_normal_application_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            owners = dict(training=dict(pid=11, start=1), inference=dict(pid=12, start=2))
            service = dict(pid=13, start=3)
            identities = dict(owners, service=service)
            schedule = dict(begin_ns=100)
            job = dict(status=collector.capture.SUCCESS, mode='equal', ordinary_timing=True,
                application_identities=owners, server_identity=service, observed_gpu_pids=[13],
                final_returncodes={role: 0 for role in identities}, socket_retained=False,
                authenticated_child_cleanup={role: dict(value, exit_confirmed=True, signals=[])
                                             for role, value in identities.items()},
                artifact_identities={}, application_commands={}, socket_path='/tmp/socket',
                server_command=['service'], client_pins=[], service_configuration={},
                schedule=schedule, metrics=dict(window_begin_ns=100, window_end_ns=30_002_000_100),
                workers_cpu={}, completion='complete', numerical_scope='unchanged')
            collector.save(directory / 'attempt.json', job)
            collector.save(directory / 'inference.start.json', schedule)
            (directory / 'server.log').touch()
            job = collector.canonical_job(directory, job)
            sustained = SimpleNamespace(commands=Mock(return_value={}),
                owner_command=Mock(return_value=(['service'], [])),
                ordinary=SimpleNamespace(ordinary_log=Mock(return_value={})),
                schedule=Mock(return_value=schedule), assess=Mock())
            self.assertEqual(collector.capture.audit_application(sustained, None, None,
                directory, job, collector.timing.POINT), (100, 30_002_000_100))


if __name__ == '__main__':
    unittest.main()
