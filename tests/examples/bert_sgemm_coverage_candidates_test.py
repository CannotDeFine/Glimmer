"""CPU-only controls for post-log-boundary SGEMM candidate selection."""

from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from bert_sgemm_coverage_candidates import (authenticated_training_sessions, inspect_attempt,
                                             post_boundary_candidates, targeted_descriptor,
                                             targeted_issue_range, targeted_ncu_metrics, targeted_training_binding)


class Record:
    def __init__(self, **values):
        self.values = values

    def value(self, name):
        return self.values[name]


class Semantic:
    def __init__(self, records):
        self.records = records

    def family(self, family):
        assert family == 4
        return self.records


def event(kind, session=2, sequence=7, blocks=2304, workers=320, status=0):
    prefix = '[info] '
    if kind == 'admit':
        return f'{prefix}Native admission audit: session={session}, sequence={sequence}, accepted=true, operation=30\n'
    if kind == 'select':
        return (f'{prefix}Native persistent audit: session={session}, sequence={sequence}, '
                f'total_blocks={blocks}, physical_launches=1, workers={workers}\n')
    if kind == 'issue':
        return f'{prefix}Native issue audit: session={session}, sequence={sequence}, status={status}\n'
    return (f'{prefix}Native execution audit: session={session}, issued={sequence}, '
            f'retained=0, retired=1, status={status}\n')


def command(sequence=7, **kwargs):
    return ''.join(event(kind, sequence=sequence, **kwargs) for kind in ('admit', 'select', 'issue', 'retire'))


def targeted_log():
    start = 'SGEMM diagnostic: v=1, event=start, session=2, sequence=7\n'
    descriptor = ('SGEMM diagnostic: v=1, event=descriptor, session=2, sequence=7, '
                  'transa=0, transb=1, m=3072, n=768, k=768, '
                  'lda=3072, ldb=768, ldc=3072, device_scalars=0, replacement=1\n')
    end = 'SGEMM diagnostic: v=1, event=end, session=2, sequence=7, captured=1\n'
    return (start + event('admit') + descriptor + event('select') + event('issue') +
            event('retire') + end)


def targeted_range_log():
    log = targeted_log()
    first = log.find('\n') + 1
    return (log[:first] + 'SGEMM range diagnostic: v=1, event=start\n' + log[first:] +
            'SGEMM range diagnostic: v=1, event=end, attempts=1, resumes=0, push=0, pop=0, '
            'issued=1, native=0, completion=0, has_library=0, library=-1, saturated=0, supported=1\n')


class Candidates(unittest.TestCase):
    def test_target_binding_resolves_only_authenticated_training_compute_session(self):
        job = dict(mode='equal', effective_selection={'low': [3], 'high': [2]},
                   application_identities={'low': dict(pid=10, start=20),
                                           'high': dict(pid=11, start=21)})
        low = dict(event=1, role=0, connection=3, pid=10, start=20, uid=1003,
                   priority=0, kernel_mode=2, effective=1, retained=1)
        high = dict(event=1, role=0, connection=2, pid=11, start=21, uid=1003,
                    priority=0, kernel_mode=1, effective=0, retained=0)
        semantic = Semantic([Record(**low), Record(**high)])
        binding = 'SGEMM target binding: v=1, pid=10, start=20, uid=1003, session=3, sequence=7\n'
        log = targeted_range_log().replace('session=2', 'session=3')
        self.assertEqual(targeted_training_binding(binding + log, semantic, job, 7, 1003),
                         dict(role='low', pid=10, start=20, uid=1003, session=3, sequence=7))
        targeted_issue_range(binding + log, 3, 7)
        for bad in (log, binding + binding + log, log + binding,
                    binding.replace('pid=10', 'pid=11') + log,
                    binding.replace('start=20', 'start=21') + log,
                    binding.replace('uid=1003', 'uid=1004') + log,
                    binding.replace('session=3', 'session=2') + log,
                    binding.replace('sequence=7', 'sequence=8') + log):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                targeted_training_binding(bad, semantic, job, 7, 1003)
        for changed in (dict(role=1), dict(role=2), dict(role=3), dict(uid=1004)):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                targeted_training_binding(binding + log,
                    Semantic([Record(**dict(low, **changed)), Record(**high)]), job, 7, 1003)

    def test_one_ncu_launch_is_not_coverage_without_application_binding(self):
        header = ('"ID","Process ID","Kernel Name","Context","Stream","launch__sm_count",'
                  '"launch__grid_size","launch__block_size","sm__cycles_active.min",'
                  '"sm__sass_thread_inst_executed_op_ffma_pred_on.min"\n')
        units = '"","","","","","SM","","","cycle","inst"\n'
        row = '"0","42","owned_kernel","1","13","80","320","128","123","45"\n'
        data = '==PROF== Disconnected\n' + header + units + row
        result = targeted_ncu_metrics(data, 'owned_kernel')
        self.assertEqual((result['process_id'], result['context'], result['stream']), (42, 1, 13))
        self.assertTrue(result['minimums_positive'])
        self.assertFalse(result['sm_coverage_verified'])
        self.assertFalse(result['physical_launch_bound'])
        self.assertFalse(targeted_ncu_metrics(data.replace('"123"', '"0"'),
                                               'owned_kernel')['minimums_positive'])
        for bad in (data + row, data.replace('"80"', '"79"'),
                    data.replace('"320"', '"319"'), data.replace('"42"', '"0"'),
                    data.replace('"123"', '"nan"'), data.replace('"cycle"', '"cycles"'),
                    data.replace('"owned_kernel"', '"other"'), data.replace('"80"', '"8,0"'),
                    data.replace('"320"', '"3,,2,0"'), data.removesuffix('"\n')):
            with self.subTest(bad=bad[-100:]), self.assertRaises(ValueError):
                targeted_ncu_metrics(bad, 'owned_kernel')

    def test_range_metadata_rejects_retry_resume_and_unobserved_push(self):
        log = targeted_range_log()
        result = targeted_issue_range(log, 2, 7)
        self.assertIs(result['range_issue_preconditions_verified'], True)
        for key in ('range_marker_observed', 'physical_launch_bound', 'native_operation_matched',
                    'sm_coverage_verified', 'saturation_verified', 'performance_qualified'):
            self.assertIs(result[key], False)
        changes = (('attempts=1', 'attempts=2'), ('resumes=0', 'resumes=1'),
                   ('push=0', 'push=-1'), ('pop=0', 'pop=1'), ('issued=1', 'issued=0'),
                   ('native=0', 'native=1'), ('completion=0', 'completion=1'),
                   ('has_library=0', 'has_library=1'), ('saturated=0', 'saturated=1'),
                   ('supported=1', 'supported=0'), ('event=end, attempts=', 'event=unknown, attempts='))
        for before, after in changes:
            with self.subTest(after=after), self.assertRaises(ValueError):
                targeted_issue_range(log.replace(before, after), 2, 7)
        with self.assertRaises(ValueError):
            targeted_issue_range(log + 'SGEMM range diagnostic: v=1, event=start\n', 2, 7)

    def test_targeted_descriptor_is_partial_and_requires_complete_chain(self):
        log = targeted_log()
        result = targeted_descriptor(log, 2, 7)
        self.assertEqual((result['m'], result['n'], result['transb'], result['logical_blocks']),
                         (3072, 768, 1, 2304))
        for name in ('scalar_equivalence_verified', 'application_identity_verified',
                     'application_phase_verified', 'physical_launch_bound',
                     'native_operation_matched', 'sm_coverage_verified', 'saturation_verified',
                     'performance_qualified'):
            self.assertIs(result[name], False)
        for bad in (log.replace('captured=1', 'captured=0'), log.replace('replacement=1', 'replacement=0'),
                    log.replace(event('issue'), ''), log + log.splitlines(keepends=True)[0],
                    log.replace(event('retire'), event('retire') + event('retire')),
                    log.replace(event('select'), event('issue') + event('select')),
                    log.replace('transb=1', 'transb=3'), log.replace('sequence=7', 'sequence=8', 1),
                    log.replace('event=descriptor', 'event=unknown')):
            with self.subTest(bad=bad[:100]), self.assertRaises(ValueError):
                targeted_descriptor(bad, 2, 7)

    def test_training_role_is_bound_to_authenticated_process_and_configuration(self):
        job = dict(mode='equal', effective_selection={'low': [2], 'high': [4]},
                   application_identities={'low': dict(pid=10, start=20),
                                           'high': dict(pid=11, start=21)})
        low = dict(event=1, role=0, connection=2, pid=10, start=20, priority=0,
                   kernel_mode=2, effective=1, retained=1)
        high = dict(event=1, role=0, connection=4, pid=11, start=21, priority=0,
                    kernel_mode=1, effective=0, retained=0)
        self.assertEqual(authenticated_training_sessions(Semantic([Record(**low), Record(**high)]), job), {2})
        for change in (dict(pid=11), dict(start=21), dict(priority=100), dict(kernel_mode=1),
                       dict(effective=0), dict(retained=0), dict(connection=3)):
            bad = dict(low, **change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                authenticated_training_sessions(Semantic([Record(**bad), Record(**high)]), job)
        job['effective_selection']['high'] = [2]
        with self.assertRaises(ValueError):
            authenticated_training_sessions(Semantic([Record(**low), Record(**high)]), job)

    def test_selects_training_only_after_saved_log_boundary(self):
        before_boundary = command(1)
        content = before_boundary + command(2, session=4) + command(3, blocks=24) + command(4)
        found = post_boundary_candidates(content, len(before_boundary), {2})
        self.assertEqual([(row['session'], row['sequence'], row['logical_blocks']) for row in found],
                         [(2, 4, 2304)])

    def test_rejects_missing_issue_or_retirement_and_duplicate(self):
        prefix = 'boundary\n'
        for content in (event('admit') + event('select') + event('issue'),
                        event('admit') + event('select') + event('retire'),
                        command() + event('admit')):
            with self.subTest(content=content), self.assertRaises(ValueError):
                post_boundary_candidates(prefix + content, len(prefix), {2})

    def test_rejects_failed_or_wrong_geometry_and_order(self):
        prefix = 'boundary\n'
        for content in (command(workers=80), command(status=7),
                        event('admit') + event('issue') + event('select') + event('retire')):
            with self.subTest(content=content), self.assertRaises(ValueError):
                post_boundary_candidates(prefix + content, len(prefix), {2})

    def test_rejects_pre_boundary_only_and_invalid_boundary(self):
        before_boundary = command(1)
        for content, boundary in ((before_boundary + 'end\n', len(before_boundary)),
                                  ('boundary\n' + command(), 2)):
            with self.subTest(boundary=boundary), self.assertRaises(ValueError):
                post_boundary_candidates(content, boundary, {2})

    def test_delayed_record_after_boundary_is_candidate_without_phase_claim(self):
        boundary = 'boundary\n'
        # A prior-phase command may be logged after the saved offset.
        self.assertEqual(post_boundary_candidates(boundary + command(8), len(boundary), {2})[0]['sequence'], 8)

    def test_candidate_inspection_rejects_changed_log_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            attempt = Path(tmp)
            (attempt / 'equal').mkdir()
            (attempt / 'equal/server.log').write_text('changed\n')
            job = dict(status='complete_fixed_application_numerical_diagnostic_passed',
                       artifact_identities={'server.log': '0' * 64})
            (attempt / 'equal/attempt.json').write_text(json.dumps(job))
            reparse = dict(tensors=401, elements=217787909, full_tensor_recomparison=True,
                           source_bound_complete_inference=True, first_command_handoff=False)
            worker = dict(status='passed_requires_review', scoped_publication_valid=True,
                          reparse=reparse, job=job)
            report = dict(status='equal_only_existing_native_requires_review',
                          input_identity_unchanged=True, fresh_native_run=False,
                          jobs=[dict(mode='equal', unit=dict(returncode=0, unit_removed=True),
                                     worker=worker)])
            (attempt / 'report.json').write_text(json.dumps(report))
            with patch('bert_sgemm_coverage_candidates.normal_cleanup'):
                with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                    inspect_attempt(attempt)


if __name__ == '__main__':
    unittest.main()
