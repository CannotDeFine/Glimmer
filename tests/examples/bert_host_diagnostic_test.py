"""Small cold and scoped-injection checks; no GPU initialization."""

import contextlib
from datetime import datetime, timezone
import io
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import bert_host_probe as probe
import run_bert_host_diagnostic as runner


class Tests(unittest.TestCase):
    def test_failed_collection_route_is_closed(self):
        with patch.object(runner.recipe, 'read', side_effect=AssertionError('must remain cold')):
            with self.assertRaisesRegex(ValueError, 'collector disabled'):
                runner.collect(Path('/absent'), Path('/absent'))
            with self.assertRaisesRegex(ValueError, 'collector disabled'):
                runner.worker(Path('/absent'), 'equal')

    def test_cold_help(self):
        with patch.object(sys, 'argv', ['runner']), contextlib.redirect_stdout(io.StringIO()), \
                patch.object(runner.capture, 'helpers', side_effect=AssertionError('not cold')):
            self.assertEqual(runner.main(), 0)

    def test_absolute_authorization_budget(self):
        plan = dict(authorization_deadline_utc='2026-10-31T18:02:53Z')
        runner.authorization_budget(plan, 60, datetime(2026, 10, 31, 18, 1, tzinfo=timezone.utc))
        with self.assertRaisesRegex(ValueError, 'insufficient'):
            runner.authorization_budget(plan, 800, datetime(2026, 10, 31, 18, 1, tzinfo=timezone.utc))
        with self.assertRaisesRegex(ValueError, 'absolute'):
            runner.authorization_budget(dict(authorization_deadline_utc='2026-10-31T18:02:53'), 0)

    def test_profiler_environment_has_only_one_library(self):
        clean = dict(PATH='/bin', CUDA_VISIBLE_DEVICES='gpu')
        actual = probe.profile_environment(clean, Path('/tmp/owned'), Path('/lib/profiler'))
        self.assertEqual(actual['LD_PRELOAD'], '/lib/profiler')
        self.assertEqual(actual['CPUPROFILESIGNAL'], str(int(signal.SIGUSR2)))
        self.assertEqual(actual['CPUPROFILE_FREQUENCY'], '500')
        for extra in (dict(LD_PRELOAD='/wait'), dict(LD_AUDIT='/audit'), dict(CPUPROFILE='old')):
            with self.assertRaisesRegex(ValueError, 'clean'):
                probe.profile_environment(dict(clean, **extra), Path('/tmp/owned'), Path('/lib/profiler'))

    def test_stat_handles_spaces_and_parentheses(self):
        fields = ['S'] + [str(i) for i in range(1, 25)]
        parsed = probe.parse_stat('7 (name with ) space) ' + ' '.join(fields))
        self.assertEqual(parsed, dict(state='S', user_ticks=11, system_ticks=12, start_ticks=19))

    def test_stop_window_is_checked_after_signal(self):
        rows = [dict(action=action, pid=17, start_ticks=10, pidfd_bound=True,
                     signal=int(signal.SIGUSR2), low_ns=target, high_ns=target + 1)
                for action, target in (('start', 100), ('stop', 5_000_000_100))]
        probe.validate_toggles(rows, 100, 5_000_000_100)
        rows[1]['high_ns'] += 100_000_000
        with self.assertRaisesRegex(ValueError, 'bounded window'):
            probe.validate_toggles(rows, 100, 5_000_000_100)

    def test_scoped_hooks_inject_only_service_and_restore_on_failure(self):
        clean = dict(PATH='/bin')
        common = SimpleNamespace(verify_process=Mock(), check_pending=Mock())
        sp = SimpleNamespace(Popen=Mock(), PIPE=subprocess.PIPE, STDOUT=subprocess.STDOUT)
        owner = Mock(return_value=(['/service'], []))
        sustained = SimpleNamespace(subprocess=sp, owner_command=owner, ROOT=Path('/repo'),
                                    sys=sys, schedule=Mock(return_value={'begin_ns': 1}))
        h = SimpleNamespace(wrapper='wrapper')
        window = SimpleNamespace(directory=Path('/tmp/owned'), cpu=SimpleNamespace(LIBRARY=Path('/profiler')),
                                 check=Mock(), start=Mock())
        guard = SimpleNamespace(observe=Mock(), check=Mock(), group='/unit')
        original_verify = common.verify_process
        with self.assertRaisesRegex(RuntimeError, 'test failure'):
            with probe.host_hooks(common, h, sustained, guard, 'equal', clean, window):
                args = dict(env=clean, cwd=sustained.ROOT, start_new_session=True)
                sustained.subprocess.Popen([sys.executable, '-B', '-c', 'wrapper'], **args)
                self.assertEqual(sp.Popen.call_args.kwargs['env'], clean)
                sustained.owner_command()
                sustained.subprocess.Popen(['/service'], **args)
                self.assertEqual(sp.Popen.call_args.kwargs['env']['LD_PRELOAD'], '/profiler')
                with self.assertRaisesRegex(ValueError, 'unexpected'):
                    sustained.subprocess.Popen(['/foreign'], **args)
                with self.assertRaisesRegex(ValueError, 'complete host participant'):
                    sustained.schedule()
                raise RuntimeError('test failure')
        self.assertIs(sustained.subprocess, sp)
        self.assertIs(sustained.owner_command, owner)
        self.assertIs(common.verify_process, original_verify)


if __name__ == '__main__':
    unittest.main()
