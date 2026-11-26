"""CPU-only complete-oracle and completion-channel failure regressions."""

from contextlib import ExitStack, nullcontext, redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import bert_ncu_sibling as sibling
import bert_ncu_relay as relay
import bert_ncu_current_sdk as current_sdk
import run_bert_sgemm_ncu_capture as collector


class SiblingTest(unittest.TestCase):
    def test_oracle_and_completion_failures_never_return_success(self):
        for failure in (None, 'oracle', 'completion', 'frontend', 'observe', 'frontend_ok'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
                outer = Path(temporary)
                (outer / 'passes').mkdir()
                provider = outer / 'range.so'
                provider.write_bytes(b'CPU fixture only')
                env = {'CUDA_VISIBLE_DEVICES': 'GPU-test'}
                manifest = dict(service_cap_sys_admin=True, uid=1003, gid=1003, service_environment=env,
                                gpu='GPU-test,V100', range_library=str(provider),
                                range_library_sha256=sibling.bridge.seal(provider), sgemm_target='2:3127',
                                sgemm_selector=dict(kind='client', role='low', sequence=3127))
                fresh = failure in ('frontend', 'observe', 'frontend_ok')
                if fresh:
                    manifest['fresh_runtime'] = dict(schema='fresh_runtime_v2', variant='scalar',
                                                     contract=dict(schema='fresh_runtime_sgemm_v2', runtime=dict(frontend={},
                                                                               service=dict(path=str(sibling.bridge.SERVICE))),
                                                                   service_environment=env,
                                                                   target='2:3127', reference=dict(path='/native')))
                (outer / 'manifest.json').write_text(json.dumps(manifest))
                (outer / 'ncu-go.json').write_text(json.dumps(dict(unit_started_us=1000)))
                connection = Mock()
                if failure == 'completion':
                    connection.request.side_effect = ConnectionError('injected completion failure')
                guard = SimpleNamespace(group='/test', receipt={'bounded': True}, check=Mock())
                training = SimpleNamespace(GPU_FIELDS='fields', SERVER=sibling.bridge.SERVICE,
                                           clean_environment=Mock(return_value=env))
                h = SimpleNamespace(training=training, query=Mock(return_value=manifest['gpu']))
                common = SimpleNamespace(application_references=Mock(return_value=('startup', 'export')))
                clients = dict(low=dict(pid=10, start=100), high=dict(pid=11, start=110))

                def job(*arguments):
                    directory = arguments[2]
                    (directory / 'equal').mkdir()
                    log = directory / 'equal/server.log'
                    log.write_text('complete semantic fixture')
                    return dict(application_identities=clients, socket_path='/private/socket',
                                server_command=['fixed', 'service'],
                                artifact_identities={'server.log': sibling.bridge.seal(log)})

                replacements = [(sibling.bridge, 'validate_status', {}),
                                (sibling.bridge, 'process_identity', {'return_value': dict(pid=8, start=80)}),
                                (sibling.bridge, 'RANGE_LIBRARY', {'new': provider}),
                                (sibling.bridge, 'service_command', {'return_value': ['fixed', 'service']}),
                                (sibling, 'proc_mapped_paths', {'return_value': []}),
                                (sibling, 'RelayConnection', {'return_value': connection}),
                                (sibling.candidate, 'ResourceGuard', {'return_value': guard}),
                                (sibling.candidate, 'load_helpers', {'return_value': (common, h)}),
                                (sibling.candidate, 'guarded_control', {'side_effect': job}),
                                (sibling.candidate, 'reparse',
                                 {'side_effect': ValueError('injected oracle failure')} if failure == 'oracle'
                                 else {'return_value': relay.EXPECTED_REPARSE}),
                                (sibling, 'service_launches', {'side_effect': lambda *args, **kwargs: nullcontext()}),
                                (sibling, 'validate_launch_order', {}),
                                (sibling, 'selected_service_target', {'side_effect': lambda *args, **kwargs: nullcontext()}),
                                (sibling, 'parse_semantic_audit', {'return_value': {}}),
                                (sibling, 'targeted_training_binding', {'return_value': {'session': 3}}),
                                (sibling, 'targeted_issue_range', {'return_value': {'selected': True}}),
                                (collector, 'task_ancestors', {'return_value': []})]
                for module, name, options in replacements:
                    stack.enter_context(patch.object(module, name, **options))
                if fresh:
                    frontend = Mock()
                    frontend.for_job.side_effect = ValueError('missing frontend mapping') if failure == 'frontend' else None
                    frontend.for_job.return_value = {'low': 'verified', 'high': 'verified'}
                    stack.enter_context(patch.object(sibling, 'FrontendMappings', return_value=frontend))
                    stack.enter_context(patch.object(sibling.bridge, 'selected_service', return_value=training.SERVER))
                    stack.enter_context(patch.object(current_sdk, 'configure_fresh', return_value=(common, h)))
                    stack.enter_context(patch.object(sibling, 'service_signal', return_value=nullcontext()))
                    if failure == 'observe':
                        stack.enter_context(patch.object(sibling.candidate, 'guarded_control',
                                                         side_effect=ValueError('frontend observation failed')))
                stack.enter_context(patch.dict(os.environ, dict(env, TMPDIR=str(outer / 'ncu-tmp'),
                                                               CUDA_CACHE_DISABLE='1'), clear=True))
                stack.enter_context(redirect_stdout(io.StringIO()))
                status = sibling.run(outer, 'private-test-endpoint', dict(pid=1, start=1), 1)
                self.assertEqual(status, 0 if failure in (None, 'frontend_ok') else 1)
                connection.close.assert_called_once()
                connection.request.assert_called_once()
                self.assertEqual(connection.request.call_args.args, ('complete',))
                self.assertIs(connection.request.call_args.kwargs['accepted'], failure not in ('oracle', 'frontend', 'observe'))
                paths = list((outer / 'passes').glob('pass-0001-*/pass.json'))
                self.assertEqual(len(paths), 1)
                report = json.loads(paths[0].read_text())
                self.assertEqual(report['status'], 'failed' if failure in ('oracle', 'frontend', 'observe')
                                 else 'pass_requires_profiler_review')
                if failure == 'frontend_ok':
                    self.assertEqual(report['frontend_mappings'], {'low': 'verified', 'high': 'verified'})
                if failure == 'completion':
                    # A published pass file alone must not count: the real
                    # controller exits nonzero and the root relay rejects it.
                    self.assertEqual(status, 1)


if __name__ == '__main__':
    unittest.main()
