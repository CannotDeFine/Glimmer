"""Cold adapter and fail-closed CPU-profile checks; no GPU initialization."""

import contextlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import run_bert_overhead_diagnostic as capture


class Tests(unittest.TestCase):
    def test_help_and_default_are_cold(self):
        for args in (['collector'], ['collector', '--help']):
            with patch.object(sys, 'argv', args), contextlib.redirect_stdout(io.StringIO()), \
                    patch.object(capture, 'helpers', side_effect=AssertionError('not cold')):
                if '--help' in args:
                    with self.assertRaises(SystemExit) as result:
                        capture.main()
                    self.assertEqual(result.exception.code, 0)
                else:
                    self.assertEqual(capture.main(), 0)

    def test_hooks_restore_on_failure_and_check_before_progress(self):
        common = SimpleNamespace(verify_process=Mock(), check_pending=Mock())
        verify, pending = common.verify_process, common.check_pending
        guard = SimpleNamespace(observe=Mock(), check=Mock(side_effect=ValueError('memory event')))
        with self.assertRaisesRegex(ValueError, 'memory event'):
            with capture.resource_hooks(common, guard):
                common.check_pending()
        pending.assert_not_called()
        self.assertIs(common.verify_process, verify)
        self.assertIs(common.check_pending, pending)

    def test_verified_process_observed_and_final_check_required(self):
        common = SimpleNamespace(verify_process=Mock(), check_pending=Mock())
        guard = SimpleNamespace(observe=Mock(), check=Mock())
        with capture.resource_hooks(common, guard):
            common.verify_process('helper', 'process', 37)
        guard.observe.assert_called_once_with('process', 37)
        guard.check.assert_called_once_with(force=True)

    def job(self):
        return dict(server_identity=dict(pid=7, start=8), cpu_samples_profiled=True,
                    cpu_profile_start=dict(pid=7, start=8, pidfd_bound=True, signal=12,
                                           request_ns=10, output_observed_ns=11),
                    ready=dict(low=dict(released_ns=12), high=dict(released_ns=13)),
                    cpu_profile=dict(samples=4), application_environment={})

    def profile(self, root):
        (root / 'service.cpu.prof.0').touch()
        (root / 'server.log').write_text('PROFILE: interrupts/evictions/bytes = 4/0/128\n')
        return SimpleNamespace(TOGGLE=12, parse_profile=Mock(return_value=dict(samples=4)))

    def test_complete_profile(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cpu = self.profile(root)
            self.assertEqual(capture.check_profile(self.job(), root, cpu), dict(samples=4))

    def test_wrong_identity_late_or_unauthenticated_start_rejected(self):
        for changes in (dict(pid=9), dict(start=9), dict(pidfd_bound=False),
                        dict(signal=1), dict(output_observed_ns=14), dict(request_ns=11)):
            job = self.job()
            job['cpu_profile_start'].update(changes)
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                cpu = self.profile(root)
                with self.assertRaisesRegex(ValueError, 'pre-release'):
                    capture.check_profile(job, root, cpu)

    def test_additional_session_count_mismatch_and_frontend_sampling_rejected(self):
        for fault in ('session', 'count', 'raw', 'frontend', 'duplicate_log'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                cpu, job = self.profile(root), self.job()
                if fault == 'session':
                    (root / 'service.cpu.prof.1').touch()
                elif fault == 'count':
                    (root / 'server.log').write_text('PROFILE: interrupts/evictions/bytes = 5/0/128\n')
                elif fault == 'duplicate_log':
                    (root / 'server.log').write_text((root / 'server.log').read_text() * 2)
                elif fault == 'raw':
                    cpu.parse_profile.return_value = dict(samples=5)
                else:
                    job['application_environment']['CPUPROFILE'] = '/unexpected'
                with self.assertRaises(ValueError):
                    capture.check_profile(job, root, cpu)


if __name__ == '__main__':
    unittest.main()
