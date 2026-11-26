"""Cold, scoped-launch and interval-partition checks; no CUDA initialization."""

import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import bert_timeline_audit as audit
import run_bert_timeline_diagnostic as capture
import bert_timeline_protocol_probe as probe


class Tests(unittest.TestCase):
    def test_interactive_probe_separates_session_and_application_options(self):
        start, launch = probe.interactive_commands(Path('/tmp/attempt-test'), Path('/tools/nsys'), '/python')
        self.assertEqual(start[:2], ['/tools/nsys', 'start'])
        self.assertEqual(launch[:2], ['/tools/nsys', 'launch'])
        for flag in ('--capture-range=nvtx', '--capture-range-end=none', '--sample=none',
                     '--cpuctxsw=none', '--gpu-metrics-devices=none', '--export=sqlite'):
            self.assertIn(flag, start)
            self.assertNotIn(flag, launch)
        for flag in ('--wait=all', '--trace=cuda,nvtx,osrt', '--cuda-event-trace=false',
                     '--nvtx-capture=GLIMMER_TIMELINE'):
            self.assertIn(flag, launch)
        self.assertIn('--session-new=glimmer-attempt-test-protocol', start)
        self.assertIn('--session=glimmer-attempt-test-protocol', launch)
        self.assertNotIn('profile', start + launch)
        self.assertFalse(any(arg.startswith('--duration=') for arg in start + launch))
        self.assertEqual(launch[-2:], ['--nsys', '/tools/nsys'])

    def test_probe_profiler_selection_rebases_only_exact_tool_paths(self):
        original = dict(CUDA_INJECTION64_PATH='/old/libToolsInjection64.so',
                        NVTX_INJECTION64_PATH='/old/libToolsInjection64.so',
                        LD_PRELOAD='/old/libToolsInjectionProxy64.so:/old/libLinuxKeyboardInterceptorProxy.so')
        selected = dict(CUDA_INJECTION64_PATH='/new/libToolsInjection64.so',
                        NVTX_INJECTION64_PATH='/new/libToolsInjection64.so',
                        LD_PRELOAD='/new/libToolsInjectionProxy64.so:/new/libLinuxKeyboardInterceptorProxy.so')
        environment = dict(selected, NSYS_PROFILING_SESSION_ID='7', QUADD_AGENT_PID='9')
        timeline = SimpleNamespace(CONSTANTS=original, bootstrap=Mock(side_effect=lambda env: env))
        self.assertEqual(probe.profiler_tools(timeline, environment, Path('/new/nsys')), environment)
        timeline.bootstrap.assert_called_once_with(dict(environment, **original))
        for key in selected:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'injection paths differ'):
                probe.profiler_tools(timeline, dict(environment, **{key: original[key]}), Path('/new/nsys'))
        timeline.bootstrap.side_effect = ValueError('unexpected policy')
        with self.assertRaisesRegex(ValueError, 'unexpected policy'):
            probe.profiler_tools(timeline, environment, Path('/new/nsys'))
        with self.assertRaisesRegex(ValueError, 'absolute executable'):
            probe.collect('explicit-stop', Path('nsys'))
        with patch.object(sys, 'argv', ['probe', '--collect', 'explicit-stop', '--nsys', '/new/nsys']), \
                patch.object(probe, 'collect', return_value=0) as collect:
            self.assertEqual(probe.main(), 0)
            collect.assert_called_once_with('explicit-stop', Path('/new/nsys'))

    def test_protocol_receipt_is_invisible_until_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'receipt.json'
            original = probe.save

            def observed_write(pending, value):
                self.assertFalse(path.exists())
                original(pending, value)
                self.assertFalse(path.exists())
                self.assertEqual(capture.recipe.read(pending), value)

            with patch.object(probe, 'save', side_effect=observed_write):
                probe.publish(path, {'payload': 'x' * 100000})
            self.assertEqual(capture.recipe.read(path), {'payload': 'x' * 100000})
            self.assertFalse(path.with_name(path.name + '.pending').exists())
            with self.assertRaisesRegex(ValueError, 'fresh probe receipt'):
                probe.publish(path, {})

    def test_unqualified_capture_entrypoints_fail_before_side_effects(self):
        with patch.object(capture.recipe, 'read', side_effect=AssertionError('plan read')), \
                patch.object(capture, 'helpers', side_effect=AssertionError('helpers loaded')), \
                patch.object(capture.candidate, 'ResourceGuard', side_effect=AssertionError('unit accessed')):
            for entrypoint, args in ((capture.collect, (Path('/nonexistent'),)),
                                     (capture.worker, (Path('/nonexistent'), 'native')),
                                     (capture.trace_worker, (Path('/nonexistent'), 'equal'))):
                with self.subTest(entrypoint=entrypoint.__name__), \
                        self.assertRaisesRegex(ValueError, 'experimental collector disabled'):
                    entrypoint(*args)
            for flag in ('--collect', '--worker-attempt', '--trace-attempt'):
                argv = ['collector', flag, '/nonexistent', '--worker-mode', 'native']
                with self.subTest(flag=flag), patch.object(sys, 'argv', argv), \
                        self.assertRaisesRegex(ValueError, 'experimental collector disabled'):
                    capture.main()

    def test_help_and_default_do_not_load_helpers(self):
        for argv in (['collector'], ['collector', '--help']):
            with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()), \
                    patch.object(capture, 'helpers', side_effect=AssertionError('not cold')):
                if '--help' in argv:
                    with self.assertRaises(SystemExit) as error:
                        capture.main()
                    self.assertEqual(error.exception.code, 0)
                else:
                    self.assertEqual(capture.main(), 0)

    def test_requested_profiler_flags_disable_killing_and_sampling(self):
        argv = capture.profile_command(Path('/tmp/attempt-test'), 'equal', '/python')
        for flag in ('--capture-range=nvtx', '--capture-range-end=stop', '--kill=none',
                     '--sample=none', '--cpuctxsw=none', '--gpu-metrics-devices=none',
                     '--force-overwrite=false', '--trace=cuda,nvtx,osrt'):
            self.assertIn(flag, argv)
        self.assertNotIn('sudo', argv)
        self.assertIn('--trace-attempt', argv)

    def fixture(self):
        clean = dict(CUDA_VISIBLE_DEVICES='gpu')
        tools = dict(LD_PRELOAD='/proxy', CUDA_INJECTION64_PATH='/inject', NSYS_PROFILING_SESSION_ID='7')
        common = SimpleNamespace(owned_environment=Mock(side_effect=lambda env, *a, **kw:
            dict(env, LD_PRELOAD='/glimmer')), verify_process=Mock(), check_pending=Mock())
        owner = Mock(return_value=(['/service', '--fixed'], ['pin']))
        module = SimpleNamespace(Popen=Mock(return_value='child'))
        sustained = SimpleNamespace(subprocess=module, owner_command=owner,
                                    schedule=Mock(return_value=dict(begin_ns=1)))
        h = SimpleNamespace(wrapper='wrapper')
        guard = SimpleNamespace(check=Mock(), observe=Mock(), group='/test')
        clock = SimpleNamespace(start=Mock(), check=Mock())
        return common, h, sustained, guard, clean, tools, clock

    def test_launch_env_routing_and_full_restoration_on_failure(self):
        common, h, sustained, guard, clean, tools, clock = self.fixture()
        before = dict(vars(sustained)), dict(vars(common))
        original_popen = subprocess.Popen
        with self.assertRaisesRegex(RuntimeError, 'injected failure'):
            with capture.launch_hooks(common, h, sustained, guard, 'equal', clean, tools, clock) as env:
                kwargs = dict(cwd=capture.PROJECT_ROOT, env=env, stdout='log',
                              stderr=subprocess.STDOUT, start_new_session=True)
                sustained.subprocess.Popen([sys.executable, '-B', '-c', h.wrapper],
                                           stdin=subprocess.PIPE, text=True, **kwargs)
                self.assertEqual(before[0]['subprocess'].Popen.call_args.kwargs['env'], clean)
                argv, _ = sustained.owner_command()
                sustained.subprocess.Popen(argv, **kwargs)
                self.assertEqual(before[0]['subprocess'].Popen.call_args.kwargs['env'], env)
                self.assertEqual(common.owned_environment(env), dict(clean, LD_PRELOAD='/glimmer'))
                with self.assertRaisesRegex(ValueError, 'exact frontend/owner environment'):
                    common.verify_process(h, SimpleNamespace(pid=7), 9, ['/application'],
                                          dict(clean, LD_PRELOAD='/glimmer:/proxy'))
                self.assertIs(subprocess.Popen, original_popen)
                raise RuntimeError('injected failure')
        for key, value in before[0].items():
            self.assertIs(getattr(sustained, key), value)
        for key, value in before[1].items():
            self.assertIs(getattr(common, key), value)

    def test_unknown_launch_changed_env_or_changed_kwargs_rejected(self):
        common, h, sustained, guard, clean, tools, clock = self.fixture()
        with capture.launch_hooks(common, h, sustained, guard, 'equal', clean, tools, clock) as env:
            argv = [sys.executable, '-B', '-c', h.wrapper]
            kwargs = dict(cwd=capture.PROJECT_ROOT, env=env, stdout='log', stderr=subprocess.STDOUT,
                          start_new_session=True, stdin=subprocess.PIPE, text=True)
            for changed_argv, changed in ((['/unknown'], kwargs), (argv, dict(kwargs, env=clean)),
                                         (argv, dict(kwargs, start_new_session=False)),
                                         (argv, dict(kwargs, shell=True))):
                with self.assertRaises(ValueError):
                    sustained.subprocess.Popen(changed_argv, **changed)
            with self.assertRaises(ValueError):
                common.owned_environment(dict(env, EXTRA_TOOL='unexpected'))

    def test_schedule_and_pending_preserve_original_progress(self):
        common, h, sustained, guard, clean, tools, clock = self.fixture()
        pending = common.check_pending
        with capture.launch_hooks(common, h, sustained, guard, 'native', clean, tools, clock):
            self.assertEqual(sustained.schedule('measurement'), dict(begin_ns=1))
            clock.start.assert_called_once()
            common.check_pending('processes')
            pending.assert_called_once_with('processes')
            clock.check.side_effect = ValueError('range failed')
            with self.assertRaisesRegex(ValueError, 'range failed'):
                common.check_pending('processes')
            self.assertEqual(pending.call_count, 1)

    def activity(self, start, end, api_end, tid=1):
        return dict(start=start, end=end, api_end=api_end, tid=tid, kind='kernel',
                    name='kernel', api_name='launch', global_pid=1, context_id=1, stream_id=2,
                    grid=[1, 1, 1], block=[128, 1, 1])

    def test_gap_partition_wait_overlap_and_future_launch(self):
        rows = [self.activity(10, 20, 5), self.activity(80, 90, 30)]
        calls = [dict(start=20, end=40, globalTid=1, name='ppoll'),
                 dict(start=0, end=100, globalTid=2, name='futex')]
        result = audit.decompose(rows, calls, 0, 100)
        self.assertEqual(result['gpu_activity_union_ns'], 20)
        self.assertEqual(result['gap_categories_ns'], dict(returned_launch_pending=55,
            no_pending_with_recorded_wait_call=10, no_pending_without_recorded_wait_call=15))
        self.assertEqual(result['primary_gap_category_candidate'], 'returned_launch_pending')
        rows.append(self.activity(120, 130, 95))
        future = audit.decompose(rows, calls, 0, 100)
        self.assertEqual(future['gap_categories_ns']['returned_launch_pending'], 60)
        self.assertEqual(future['gpu_activity_union_ns'], 20)

    def test_union_and_concurrency_do_not_sum_overlapping_gpu_time(self):
        rows = [self.activity(10, 40, 10), self.activity(20, 30, 20)]
        result = audit.decompose(rows, [], 0, 50)
        self.assertEqual(result['gpu_activity_union_ns'], 30)
        self.assertEqual(result['concurrency_depth_duration_ns'], {'0': 20, '1': 20, '2': 10})
        self.assertEqual(sum(result['gap_categories_ns'].values()), 20)
        rows[1]['kind'] = 'copy'
        mixed = audit.decompose(rows, [], 0, 50)
        self.assertEqual(mixed['kernel_concurrency_depth_duration_ns'], {'0': 20, '1': 30})
        self.assertEqual(mixed['concurrency_depth_duration_ns'], {'0': 20, '1': 20, '2': 10})
        self.assertEqual(mixed['kernel_process_context_streams']['1:1:2']['count'], 1)


if __name__ == '__main__':
    unittest.main()
