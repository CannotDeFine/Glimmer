"""No-GPU target binding and cleanup checks for the fixed BERT probe."""

from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from run_bert_sgemm_descriptor_probe import (ncu_environment, parse_target, selected_service_target,
                                            validate_worker_target)


class DescriptorProbe(unittest.TestCase):
    def test_target_requires_canonical_positive_uint64_identity(self):
        self.assertEqual(parse_target('2:3127'), (2, 3127))
        for value in ('', '0:1', '1:0', '-1:1', '1:2:3', '01:2', '1:2\n', '18446744073709551616:1', None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_target(value)

    def test_only_exact_owned_service_command_is_decorated_and_restored(self):
        def command(*args, **kwargs):
            return ['service', '--audit-native'], ['low', 'high']

        common = SimpleNamespace(service_command=command)
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            with selected_service_target(common, '2:3127'):
                self.assertEqual(common.service_command(),
                                 (['service', '--audit-native', '--audit-sgemm-target', '2:3127'],
                                  ['low', 'high']))
                raise RuntimeError('injected')
        self.assertIs(common.service_command, command)

    def test_rejects_missing_or_preexisting_native_and_target_flags(self):
        for command in (['service'], ['service', '--audit-native', '--audit-native'],
                        ['service', '--audit-native', '--audit-sgemm-target', '1:1']):
            common = SimpleNamespace(service_command=lambda: (command, []))
            with selected_service_target(common, '2:3127'):
                with self.assertRaises(ValueError):
                    common.service_command()

    def test_client_target_uses_only_the_configured_training_pin(self):
        command = ['service', '--audit-native', '--client-priority', '8:80:0:original',
                   '--client-priority', '7:70:0:configured']
        common = SimpleNamespace(service_command=lambda: (command, []))
        with selected_service_target(common, '2:3127', client_bound=True):
            self.assertEqual(common.service_command()[0][-2:],
                             ['--audit-sgemm-target-client', '7:70:3127'])
            for changed in (command[:4], command + command[4:],
                            command + ['--audit-sgemm-target-client', '8:80:3127']):
                original = list(command)
                command[:] = changed
                with self.assertRaises(ValueError):
                    common.service_command()
                command[:] = original

    def test_profile_environment_only_preserves_the_observed_ncu_nvtx_value(self):
        base = {'PATH': '/usr/bin', 'CUDA_VISIBLE_DEVICES': 'GPU-example'}
        profiler = {'NVTX_INJECTION64_PATH': 'libcuda-injection.so', 'UNRELATED': 'ignored'}
        selected = ncu_environment(base, profiler)
        self.assertEqual(selected, dict(base, NVTX_INJECTION64_PATH='libcuda-injection.so'))
        self.assertNotIn('NVTX_INJECTION64_PATH', base)
        for changed in ({}, {'NVTX_INJECTION64_PATH': '/tmp/foreign.so'},
                        {'NVTX_INJECTION64_PATH': ''}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                ncu_environment(base, changed)
        with self.assertRaises(ValueError):
            ncu_environment(selected, profiler)

    def test_range_provider_is_explicit_and_ordinary_command_is_unchanged(self):
        common = SimpleNamespace(service_command=lambda: (['service', '--audit-native',
                                                           '--socket', '/tmp/test.sock'], []))
        provider = Path(__file__).resolve()
        with selected_service_target(common, '2:3127', provider):
            self.assertEqual(common.service_command()[0][-2:],
                             ['--audit-sgemm-range-library', str(provider)])
        with selected_service_target(common, '2:3127', Path('relative')):
            with self.assertRaises(ValueError):
                common.service_command()
        with selected_service_target(common, '2:3127'):
            self.assertNotIn('--audit-sgemm-range-library', common.service_command()[0])
        common = SimpleNamespace(service_command=lambda: (['service', '--audit-native', '--socket',
                                                           '/' + 'x' * 108], []))
        with selected_service_target(common, '2:3127', provider):
            with self.assertRaisesRegex(ValueError, 'socket path'):
                common.service_command()

    def test_worker_cannot_replace_the_original_target(self):
        receipt = dict(status='passed_requires_review', descriptor=dict(session=2, sequence=3127))
        validate_worker_target(receipt, '2:3127')
        for changed in (dict(session=2, sequence=3128), dict(session=1, sequence=3127),
                        dict(session=True, sequence=3127)):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validate_worker_target(dict(receipt, descriptor=changed), '2:3127')

    def test_cold_help_does_not_touch_gpu_or_start_unit(self):
        result = subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/run_bert_sgemm_descriptor_probe.py')],
                                capture_output=True, text=True, timeout=10, check=True)
        self.assertIn('--collect', result.stdout)
        self.assertIn('--wait-for-idle', result.stdout)
        self.assertNotIn('SGEMM descriptor evidence:', result.stdout)


if __name__ == '__main__':
    unittest.main()
