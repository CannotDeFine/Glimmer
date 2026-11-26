"""Structural guard for the private scheduling/resource-census boundary.

Behavioral lifecycle and authority checks live in the fake-driver C++ fixtures.
This deliberately narrow source contract catches accidental census coupling;
it is not a performance measurement or a general C++ parser.
"""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]


def body(source, signature):
    start = source.index('{', source.index(signature))
    depth = 1
    end = start + 1
    while depth and end < len(source):
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    if depth:
        raise AssertionError('incomplete function body')
    return source[start + 1:end - 1]


def check_lightweight(test, source):
    light = body(source, 'CudaExecutionSession::scheduling_snapshot()')
    test.assertNotRegex(light, r'(?<![\w])snapshot\s*\(')
    test.assertNotRegex(light, r'\b(for|while|count|count_if)\s*\(')
    test.assertEqual(set(re.findall(r'state\.storage->(\w+)', light)),
                     {'context', 'domain_member'})
    test.assertNotRegex(light, r'state\.(external_contexts|external_commands)\b')
    test.assertIn('if (!state.owns())', light)
    test.assertEqual(light.count('state.executor->scheduling_snapshot()'), 1)
    test.assertIn('state.storage->domain_member.view(true)', light)


class SchedulingBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.session = (ROOT / 'src/backend/cuda/cuda_execution_session.cc').read_text()
        cls.runtime = (ROOT / 'src/execution_service/runtime.cc').read_text()

    def test_lightweight_has_no_resource_census(self):
        check_lightweight(self, self.session)

    def test_boundary_rejects_accidental_recoupling(self):
        marker = 'CudaExecutionSession::scheduling_snapshot() const noexcept {'
        for injected in ('snapshot();', 'state.storage->functions.size();',
                         'std::ranges::count(state.external_commands, true);'):
            with self.subTest(injected=injected), self.assertRaises(AssertionError):
                check_lightweight(self, self.session.replace(marker, marker + injected, 1))

    def test_full_snapshot_preserves_census_before_scheduling(self):
        full = body(self.session, 'CudaExecutionSession::snapshot()')
        projection = full.index('= scheduling_snapshot()')
        for resource in ('persistent_slots', 'persistent_functions',
                         'external_context_leases', 'external_command_leases'):
            self.assertLess(full.index('result.' + resource), projection)

    def test_service_keeps_only_explicit_resource_consumers(self):
        bootstrap = body(self.runtime, 'ExecutionWireReply finish_bootstrap(')
        diagnostic = body(self.runtime, 'case ExecutionOpcode::kSnapshot:')
        self.assertEqual(bootstrap.count('session->snapshot()'), 2)
        self.assertEqual(diagnostic.count('session->snapshot()'), 1)
        self.assertEqual(self.runtime.count('session->snapshot()'), 3)
        self.assertNotRegex(self.runtime,
                            r'\b(selected|peer\.cleanup|path\.source|first|second|owner)'
                            r'->snapshot\(\)')
        self.assertIn('snapshot.private_host_bytes', bootstrap)
        self.assertIn('snapshot.device_bytes', diagnostic)
        self.assertIn('CudaSessionSchedulingSnapshot snapshot()', self.runtime)


if __name__ == '__main__':
    unittest.main()
