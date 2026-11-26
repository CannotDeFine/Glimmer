"""CPU-only current-owner supervision, raw-owner and profiled provenance gates."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import bert_device_capture as capture


def process(pid, start, uid=1000):
    return dict(pid=pid, start_ticks=start, uid=uid, cgroup='/task', state='S', ppid=10)


def job(mode='native'):
    identities = dict(inference=dict(pid=11, start=101))
    if mode != 'exclusive':
        identities['training'] = dict(pid=12, start=102)
    actual = dict(identities)
    if mode in ('equal', 'priority'):
        actual['service'] = dict(pid=13, start=103)
    result = dict(status=capture.SUCCESS, mode=mode, application_identities=identities,
        final_returncodes={role: 0 for role in actual}, socket_retained=False,
        authenticated_child_cleanup={role: dict(exit_confirmed=True, signals=[], **row) for role, row in actual.items()},
        observed_gpu_pids=([13] if mode in ('equal', 'priority') else [row['pid'] for row in identities.values()]))
    if 'service' in actual:
        result['server_identity'] = actual['service']
    return result


def guards(mode='native'):
    actual = job(mode)
    expected = capture.normal_application(actual)
    rows = []
    for stamp in (9, 10, 11, 12):
        pids = expected if stamp in (10, 11) else set()
        rows.append(dict(kind='guard', monotonic_ns=stamp * 1_000_000_000,
            processes=[dict(gpu_uuid=capture.observer.GPU_UUID, pid=pid, identity=process(pid, pid + 90))
                       for pid in sorted(pids)]))
    return rows, actual


def depth_fixture():
    width = 200_000_000
    owners = dict(training=dict(pid=12, start=102), inference=dict(pid=11, start=101))
    pins = ['12:102:0:configured', '11:101:0:original']
    actual = dict(mode='equal', application_identities=owners, server_identity=dict(pid=13, start=103),
        client_pins=pins, server_command=['service', '--client-priority', pins[0], '--client-priority', pins[1],
                                         '--issue-depth-output', '/depth.json'])
    rows = []
    for index in range(161):
        covered = width if index < 160 else 0
        count = 2 if index == 0 else 0
        rows.append(dict(begin_ns=width+index*width, covered_ns=covered,
            admitted_not_submitted_ns=covered//2, submitted_not_retired_ns=covered//4,
            issue_call_ns=covered//10, no_admitted_compute_ns=covered*7//10,
            no_inflight_compute_ns=covered*8//10, admitted=count, submitted=count, retired=count,
            observations=1, ready_compute=1, not_ready_compute=0, other_pending=0,
            no_selected_command=0, unknown=0, ready_without_inflight=1, maximum_observation_gap_ns=5))
    document = dict(schema=1, diagnostic_only=True, valid=True, failure=0, unclosed=0, pid=13,
        origin_ns=width, begin_ns=width, end_ns=161*width, bucket_ns=width,
        readiness_is_instantaneous=True, outstanding_is_gpu_busy=False,
        slots=[dict(connection=index, pid=row['pid'], start_ticks=row['start'], admitted=1, submitted=1, retired=1)
               for index, row in enumerate(owners.values(), 1)], opcodes=[dict(opcode=11, admitted=2)], buckets=rows)
    return document, actual, Path('/depth.json'), width+1_000_000, width+30_003_000_000


def event_fixture(*, generations=False, selections=False):
    from run_bert_delivery_boundary import EVENT_FIELDS, GENERATION_FIELDS, PREPARATION_FIELDS, SELECTION_FIELDS, SELECTION_CLASSES

    document, actual, path, begin, end = depth_fixture()
    document['schema'] = 5 if selections else 3 if generations else 2
    actual['server_command'].append('--issue-depth-events')
    records = [[connection, 100, 11, begin+1, begin+61, begin+71, begin+81,
                begin+91, begin+101, end+1, connection, 1, 1] for connection in (1, 2)]
    if generations or selections:
        records[0].extend((100, 101))
        records[1].extend((0, 2))
    document['events'] = dict(capacity=524288, complete=True, ready_duration=False,
                              gpu_timestamps=False, fields=list(GENERATION_FIELDS if generations else EVENT_FIELDS), records=records)
    if selections:
        actual['server_command'].append('--issue-depth-selections')
        document['events']['fields'] = list(PREPARATION_FIELDS)
        records[0][14] = 103
        for row, slot in zip(records, document['slots']):
            row.extend((0, 0))
            slot['observation_generation'] = row[14]
        for bucket in document['buckets']:
            bucket.update(observations=0, ready_compute=0, ready_without_inflight=0)
        document['buckets'][0].update(observations=105, ready_compute=2, other_pending=1, no_selected_command=102)
        document['selections'] = dict(capacity=524288, complete=True, fields=list(SELECTION_FIELDS),
            classifications=list(SELECTION_CLASSES), records=[
                [1, 101, begin+11, 101, 2], [1, 103, begin+61, 100, 0], [2, 2, begin+61, 100, 0]])
    return document, actual, path, begin, end


class BudgetTests(unittest.TestCase):
    def test_event_receipt_bound_and_raw_identity(self):
        path = Mock()
        path.lstat.return_value = SimpleNamespace(st_mode=0o100600, st_uid=os.getuid(), st_size=9*1024**2)
        path.read_text.return_value = '{}'
        with patch.object(capture, 'digest', return_value='raw'), \
             patch.object(capture, 'validate_hardware_depth', return_value={'complete': True}) as validate:
            result = capture.hardware_depth_receipt(path, {}, 1, 2, expected_events=True)
            self.assertEqual(result['sha256'], 'raw')
            validate.assert_called_once_with({}, {}, path, 1, 2, expected_events=True,
                                              expected_generations=False, expected_selections=False)
            capture.hardware_depth_receipt(path, {}, 1, 2, expected_events=True, expected_generations=True)
            validate.assert_called_with({}, {}, path, 1, 2, expected_events=True,
                                         expected_generations=True, expected_selections=False)
            capture.hardware_depth_receipt(path, {}, 1, 2, expected_events=True,
                                            expected_generations=True, expected_selections=True)
            validate.assert_called_with({}, {}, path, 1, 2, expected_events=True,
                                         expected_generations=True, expected_selections=True)
            with self.assertRaises(ValueError):
                capture.hardware_depth_receipt(path, {}, 1, 2)
            path.lstat.return_value.st_size = 512*1024**2+1
            with self.assertRaises(ValueError):
                capture.hardware_depth_receipt(path, {}, 1, 2, expected_events=True)
            path.lstat.return_value.st_size = 9*1024**2
        with patch.object(capture, 'digest', side_effect=['before', 'changed']), \
             patch.object(capture, 'validate_hardware_depth', return_value={}):
            with self.assertRaises(ValueError):
                capture.hardware_depth_receipt(path, {}, 1, 2, expected_events=True)

    def test_hardware_owner_events_close_without_independent_command_claim(self):
        document, actual, path, begin, end = event_fixture()
        result = capture.validate_hardware_depth(document, actual, path, begin, end, expected_events=True)['events']
        self.assertEqual(result['completed'], 2)
        self.assertEqual(result['complete_command_coverage'], 1)
        self.assertEqual(result['next_boundary'], 'admitted_to_first_observed')
        self.assertEqual(result['phase_latency_fraction']['admitted_to_first_observed'], .6)
        self.assertFalse(result['command_census_independently_observed'])
        self.assertFalse(result['latency_mass_is_wall_time'])
        self.assertEqual(result['connection_sequence_and_opcode_source'], 'service_ledger')
        self.assertNotIn('effective_selection', actual)
        self.assertGreater(result['host_retirement_latency']['p50_ns'], 29_000_000_000)
        with self.assertRaises(ValueError):
            capture.validate_hardware_depth(document, actual, path, begin, end)
        for index, bad in ((0, 99), (1, 0), (2, 70), (3, begin-1_000_001), (4, begin),
                           (9, document['end_ns']+1), (10, 0), (11, 0), (12, 2)):
            changed = copy.deepcopy(document)
            changed['events']['records'][0][index] = bad
            with self.assertRaises(ValueError):
                capture.validate_hardware_depth(changed, actual, path, begin, end, expected_events=True)
        for key, value in (('capacity', 10), ('complete', False), ('ready_duration', True),
                            ('gpu_timestamps', True), ('records', document['events']['records'][:1])):
            changed = copy.deepcopy(document)
            changed['events'][key] = value
            with self.assertRaises(ValueError):
                capture.validate_hardware_depth(changed, actual, path, begin, end, expected_events=True)
        for index in (0, 10):
            changed = copy.deepcopy(document)
            changed['events']['records'][1][index] = changed['events']['records'][0][index]
            with self.assertRaises(ValueError):
                capture.validate_hardware_depth(changed, actual, path, begin, end, expected_events=True)

    def test_owner_event_decision_is_fully_contained_command_latency_mass(self):
        document, _, _, begin, end = event_fixture()
        rows = [list(document['events']['records'][0]) for _ in range(20)]
        rows[0][3] = begin-1
        result = capture.hardware_event_phases(rows, begin, end)
        self.assertEqual((result['included_commands'], result['omitted_edge_commands']), (19, 1))
        self.assertEqual(result['complete_command_coverage'], .95)
        rows[1][8] = end+1
        with self.assertRaises(ValueError):
            capture.hardware_event_phases(rows, begin, end)
        rows[0][3] = begin+1
        self.assertEqual(capture.hardware_event_phases(rows, begin, end)['complete_command_coverage'], .95)
        for phase in range(5):
            row = list(rows[0])
            row[3] = begin+1
            for offset in range(5):
                row[offset+4] = row[offset+3]+(60 if offset == phase else 10)
            result = capture.hardware_event_phases([row], begin, end)
            self.assertEqual(list(result['phase_latency_ns'])[phase], result['next_boundary'])
            self.assertEqual(sum(result['phase_latency_ns'].values()), row[8]-row[3])
        for index in range(3, 9):
            rows[0][index] = begin+index
        self.assertEqual(capture.hardware_event_phases(rows[:1], begin, end)['next_boundary'], 'mixed')
        with self.assertRaises(ValueError):
            capture.hardware_event_phases([], begin, end)

    def test_hardware_generations_require_explicit_exact_schema_and_same_row_order(self):
        document, actual, path, begin, end = event_fixture(generations=True)
        def validate(value, **flags):
            return capture.validate_hardware_depth(value, actual, path, begin, end,
                                                   expected_events=True, **flags)
        result = validate(document, expected_generations=True)['events']['observation_generations']
        self.assertEqual(result['command_counts'], dict(first_observation_positive=1,
                                                        earlier_observations_without_positive=1))
        self.assertEqual(result['next_boundary'], 'mixed')
        self.assertFalse(result['owner_loop_iterations_inferred'])
        self.assertFalse(result['dependency_cause_identified'])
        # Absolute generation values may decrease across different slots; admission0 is valid.
        for kwargs in ({}, {'expected_generations': 1}):
            with self.assertRaises(ValueError):
                validate(document, **kwargs)
        for version in (1, 2, 3.0, True):
            with self.assertRaises(ValueError):
                validate(dict(document, schema=version), expected_generations=True)
        for fields in (document['events']['fields'][:-1], list(reversed(document['events']['fields']))):
            bad = copy.deepcopy(document)
            bad['events']['fields'] = fields
            with self.assertRaises(ValueError):
                validate(bad, expected_generations=True)
        for index, value in ((13, -1), (13, True), (13, 101), (14, 100), (14, 0), (14, 2**64), (14, False)):
            bad = copy.deepcopy(document)
            bad['events']['records'][0][index] = value
            with self.assertRaises(ValueError):
                validate(bad, expected_generations=True)
        for row in (document['events']['records'][0][:-1], document['events']['records'][0]+[1]):
            bad = copy.deepcopy(document)
            bad['events']['records'][0] = row
            with self.assertRaises(ValueError):
                validate(bad, expected_generations=True)
        legacy = event_fixture()[0]
        with self.assertRaises(ValueError):
            validate(legacy, expected_generations=True)
        with self.assertRaises(ValueError):
            capture.validate_hardware_depth(document, actual, path, begin, end, expected_generations=True)
        document, actual, path, begin, end = event_fixture(selections=True)
        flags = dict(expected_events=True, expected_generations=True, expected_selections=True)
        result = capture.validate_hardware_depth(document, actual, path, begin, end, **flags)['events']
        self.assertEqual(result['selections']['records'], 3)
        self.assertEqual(result['selections']['omitted_zero_pending'], 102)
        self.assertEqual(result['earlier_selection_combinations']['next_boundary'], 'mixed')
        self.assertEqual(set(result['earlier_selection_combinations']['combinations']),
                         {'zero_pending', 'zero_pending+scope_external'})
        for key in flags:
            for wrong in (False, 1):
                with self.assertRaises(ValueError):
                    capture.validate_hardware_depth(document, actual, path, begin, end, **dict(flags, **{key: wrong}))
        for version in (1, 2, 3, 4, 5.0):
            with self.assertRaises(ValueError):
                capture.validate_hardware_depth(dict(document, schema=version), actual, path, begin, end, **flags)
        for index, value in ((15, 2), (15, True), (16, begin), (14, 106)):
            bad = copy.deepcopy(document)
            bad['events']['records'][0][index] = value
            with self.assertRaises(ValueError):
                capture.validate_hardware_depth(bad, actual, path, begin, end, **flags)
        prepared = copy.deepcopy(document)
        prepared['events']['records'][0][15:] = [1, begin+5]
        capture.validate_hardware_depth(prepared, actual, path, begin, end, **flags)
        for section, key, value in (('slots', 'observation_generation', 106),
                                    ('buckets', 'no_selected_command', 101)):
            bad = copy.deepcopy(document)
            bad[section][0][key] = value
            with self.assertRaises(ValueError):
                capture.validate_hardware_depth(bad, actual, path, begin, end, **flags)
        for index, value in ((0, 99), (1, 103), (2, begin+62), (3, 100), (4, 3)):
            bad = copy.deepcopy(document)
            bad['selections']['records'][0][index] = value
            with self.assertRaises(ValueError):
                capture.validate_hardware_depth(bad, actual, path, begin, end, **flags)
        for command in (actual['server_command'][:-1], actual['server_command']+['--issue-depth-selections']):
            with self.assertRaises(ValueError):
                capture.validate_hardware_depth(document, dict(actual, server_command=command), path, begin, end, **flags)

    def test_generation_selector_requires_new_first_phase_dominance_and_uses_mass(self):
        document, _, _, begin, end = event_fixture(generations=True)
        rows = copy.deepcopy(document['events']['records'])
        for row in rows:
            row[3:9] = [begin+1, begin+2, begin+2, begin+2, begin+2, begin+2]
        rows = [rows[0]]*9+[rows[1]]
        rows[-1][4:9] = [begin+101]*5
        result = capture.hardware_event_phases(rows, begin, end, expected_generations=True)
        groups = result['observation_generations']
        self.assertEqual(groups['command_count_fraction']['first_observation_positive'], .9)
        self.assertEqual(groups['first_phase_latency_ns'], dict(first_observation_positive=9,
                                                               earlier_observations_without_positive=100))
        self.assertEqual(sum(groups['first_phase_latency_ns'].values()),
                         result['phase_latency_ns']['admitted_to_first_observed'])
        self.assertEqual(groups['next_boundary'], 'earlier_observations_without_positive')
        self.assertFalse(groups['counts_select_direction'])
        # Exact60% selects a group, while50/50 remains mixed.
        pair = copy.deepcopy(document['events']['records'])
        for first_ns, second_ns, expected in ((60, 40, 'first_observation_positive'), (50, 50, 'mixed')):
            for row, duration in zip(pair, (first_ns, second_ns)):
                row[3:9] = [begin+1]+[begin+1+duration]*5
            summary = capture.hardware_event_phases(pair, begin, end, expected_generations=True)
            self.assertEqual(summary['observation_generations']['next_boundary'], expected)
        # First-phase100% subgroup mass cannot select when the five-phase boundary is different/mixed.
        row = list(pair[0])
        for durations, expected in (((10, 60, 10, 10, 10), 'first_to_matched_observed'),
                                     ((20, 20, 20, 20, 20), 'mixed'), ((0, 60, 0, 0, 0), 'first_to_matched_observed')):
            row[3] = begin+1
            for index, duration in enumerate(durations, 4):
                row[index] = row[index-1]+duration
            result = capture.hardware_event_phases([row], begin, end, expected_generations=True)
            self.assertEqual(result['next_boundary'], expected)
            self.assertFalse(result['observation_generations']['first_phase_dominant'])
            self.assertEqual(result['observation_generations']['next_boundary'], 'not_selected')

        def combinations(events, journal):
            phases = capture.hardware_event_phases(events, begin, end,
                expected_generations=True, expected_selections=True)
            return capture.hardware_selection_combinations(dict(events=dict(records=events),
                selections=dict(records=journal)), begin, end, phases['observation_generations'])

        selected, _, _, _, _ = event_fixture(selections=True)
        target = copy.deepcopy(selected['events']['records'][0])
        target[14] = 104
        journal = [[1, 101, begin+11, 101, 2], [1, 102, begin+21, 200, 0],
                   [1, 103, begin+31, 201, 1], [1, 104, begin+61, 100, 0]]
        result = combinations([target], journal)
        self.assertEqual(result['next_boundary'], 'other_tracked+scope_external')
        group = result['combinations'][result['next_boundary']]
        self.assertEqual((group['other_tracked_ready'], group['other_tracked_not_ready']), (1, 1))
        self.assertEqual(group['first_phase_latency_ns'], 60)
        self.assertEqual(group['earlier_observations'], 3)
        self.assertFalse(result['observations_are_durations'])
        self.assertFalse(result['policy_winner_inferred'])
        mixed = [journal[0], [1, 102, begin+21, 100, 1], journal[-1]]
        result = combinations([target], mixed)
        self.assertEqual(result['next_boundary'], 'zero_pending+target_not_ready+scope_external')
        self.assertEqual(result['combinations'][result['next_boundary']]['observation_counts'],
                         dict(zero_pending=1, target_not_ready=1, other_tracked=0, scope_external=1))
        for row in ([1, 102, begin+21, 100, 0], [1, 102, begin, 100, 1],
                    [1, 102, begin+62, 100, 1]):
            with self.assertRaises(ValueError):
                combinations([target], [row, journal[-1]])
        # Generation boundaries are open; same sequence/generation in another slot cannot join.
        bounds = [[1, 100, begin, 999, 2], journal[-1], [2, 102, begin+21, 100, 1]]
        self.assertEqual(combinations([target], bounds)['next_boundary'], 'zero_pending')
        for bad in ([journal[1], journal[0], journal[-1]],
                    [journal[0], journal[0], journal[-1]]):
            with self.assertRaises(ValueError):
                combinations([target], bad)
        # Equal command counts cannot override exact60% latency mass;50/50 remains mixed.
        pair = copy.deepcopy(selected['events']['records'])
        for first_ns, second_ns, expected in ((60, 40, 'zero_pending+scope_external'), (50, 50, 'mixed')):
            for row, duration in zip(pair, (first_ns, second_ns)):
                row[3:9] = [begin+1]+[begin+1+duration]*5
            journal = [[1, 101, begin+2, 101, 2], [1, 103, pair[0][4], 100, 0],
                       [2, 2, pair[1][4], 100, 0]]
            result = combinations(pair, journal)
            self.assertEqual(result['next_boundary'], expected)
            self.assertEqual(sum(row['commands'] for row in result['combinations'].values()), 2)
            self.assertEqual(sum(row['first_phase_latency_ns'] for row in result['combinations'].values()), 100)
        # Neither parent gate may be bypassed even when a combination has100% conditional mass.
        target[3:9] = [begin+1, begin+11, begin+71, begin+81, begin+91, begin+101]
        journal = [[1, 104, begin+11, 100, 0]]
        self.assertEqual(combinations([target], journal)['next_boundary'], 'not_selected')
        fast = copy.deepcopy(pair[1])
        fast[13:15] = [1, 2]
        fast[3:9] = [begin+1]+[begin+101]*5
        target[3:9] = [begin+1]+[begin+11]*5
        self.assertEqual(combinations([target, fast], journal+[[2, 2, begin+101, 100, 0]])['next_boundary'],
                         'not_selected')
        self.assertEqual(combinations([fast], [[2, 2, begin+101, 100, 0]])['next_boundary'], 'not_exercised')
        edge = copy.deepcopy(fast)
        edge[0], edge[1], edge[3], edge[13:15] = 3, 101, begin-1, [100, 104]
        # One omitted edge among20 commands is exactly95%; it does not enter combination mass.
        result = combinations([target]*19+[edge], journal)
        self.assertEqual((result['commands'], result['first_phase_latency_ns']), (19, 190))

    def test_depth_qualification_rejects_consistently_rewritten_raw_proofs(self):
        reference = Path('/reference')
        names = ['report.json', 'inputs-before.json', 'inputs-after.json', 'numerical-scope.json',
                 'equal/issue-depth-receipt.json']
        names += [mode+suffix for mode in ('native', 'equal')
                  for suffix in ('-worker.json', '-unit.json', '/attempt.json')]
        reviewed = {str(reference/name): 'reviewed' for name in names}
        reviewed.update({'/application/plan.json': 'reviewed', '/application/build-seal.json': 'reviewed'})
        current = dict(reviewed, **{'/proof/audit.json': 'audit'})
        audit = dict(status='fresh_numerical_and_depth_evidence_requires_independent_review',
                     reference=str(reference), inputs=reviewed)
        variant = dict(numerical_closure=dict(path='/proof/closure.json'),
                       numerical_proofs={'/proof/audit.json': 'audit'}, application_plan=dict(path='/application/plan.json'))
        with patch.object(capture.recipe, 'read', return_value=audit), \
             patch.object(capture.recipe, 'digest', side_effect=lambda path: current[str(path)]):
            self.assertEqual(capture.recipe.reviewed_depth_inputs(reference, variant), reviewed)
            variant['issue_events'] = True
            with self.assertRaises(ValueError):
                capture.recipe.reviewed_depth_inputs(reference, variant)
            audit['status'] = 'fresh_numerical_and_owner_event_evidence_requires_independent_review'
            self.assertEqual(capture.recipe.reviewed_depth_inputs(reference, variant), reviewed)
            variant['observation_generations'] = True
            with self.assertRaises(ValueError):
                capture.recipe.reviewed_depth_inputs(reference, variant)
            audit['status'] = 'fresh_numerical_and_generation_evidence_requires_independent_review'
            self.assertEqual(capture.recipe.reviewed_depth_inputs(reference, variant), reviewed)
            variant['selected_commands'] = True
            with self.assertRaises(ValueError):
                capture.recipe.reviewed_depth_inputs(reference, variant)
            audit['status'] = 'fresh_numerical_and_selected_command_evidence_requires_independent_review'
            self.assertEqual(capture.recipe.reviewed_depth_inputs(reference, variant), reviewed)
            # Internally consistent rewritten worker/parent bytes still differ from the accepted audit.
            for name in ('report.json', 'equal-worker.json', 'equal/attempt.json'):
                current[str(reference/name)] = 'consistently-rewritten'
            with self.assertRaises(ValueError):
                capture.recipe.reviewed_depth_inputs(reference, variant)
            current.update(reviewed)
            del reviewed['/reference/equal/issue-depth-receipt.json']
            with self.assertRaises(ValueError):
                capture.recipe.reviewed_depth_inputs(reference, variant)

    def test_depth_variant_requires_reviewed_observer_on_numerical_checkpoint(self):
        build = str(capture.PROJECT_ROOT / 'build/candidate')
        batch = dict(status='frozen_issue_depth_hardware_batch', order=['native', 'equal'], build=build,
                     server_sha256='service', frontend_sha256='frontend', deadline_utc='2026-11-01T17:45:00+00:00')
        batch.update({key: False for key in ('full_device_qualified', 'performance_qualified',
                       'saturation_qualified', 'gpu_gap_causality_qualified', 'priority_qualified')})
        application = dict(status='frozen_helper_planning_application', helper_plans=True, issue_depth=True,
            modes=['native', 'equal'], build=build, runtime_environment={'GLIMMER_EXECUTION_HOST_RANGE_SEEK': '1'},
            server_sha256='service', frontend_sha256='frontend')
        specification = dict(status='frozen_issue_depth_hardware_application', reference='/reference', helper_plans=True,
            batch_plan=dict(path='/batch', sha256='sha/batch'), application_plan=dict(path='/application', sha256='sha/application'),
            numerical_closure=dict(path='/closure', sha256='sha/closure'))
        scope = dict(plan=specification['application_plan'], server_sha256='service', frontend_sha256='frontend',
                     server=build+'/bin/glimmer_execution_service', frontend=build+'/lib/libglimmer_execution_interceptor.so')
        closure = dict(status='fresh_numerical_and_depth_independently_reviewed', reference='/reference',
                       service_sha256='service', frontend_sha256='frontend', audit_source_sha256='sha/audit.py',
                       audit_sha256='sha/audit.json', initial_offline_error_sha256='sha/audit-initial-error.json')
        documents = {'/spec': specification, '/batch': batch, '/application': application,
                     '/reference/numerical-scope.json': scope, '/closure': closure}
        with patch.object(capture.recipe, 'read', side_effect=lambda path: documents[str(path)]), \
             patch.object(capture.recipe, 'digest', side_effect=lambda path: 'sha'+str(path)):
            result = capture.recipe.helper_variant(Path('/reference'), Path('/spec'))
            self.assertTrue(result['issue_depth'])
            self.assertEqual(result['numerical_closure'], specification['numerical_closure'])
            for target, key, bad in ((application, 'issue_depth', False), (application, 'helper_plans', False),
                    (closure, 'status', 'unreviewed'), (closure, 'service_sha256', 'old'),
                    (batch, 'order', ['off/native', 'off/equal', 'on/native', 'on/equal'])):
                old = target[key]
                target[key] = bad
                with self.assertRaises(ValueError):
                    capture.recipe.helper_variant(Path('/reference'), Path('/spec'))
                target[key] = old
            for target in (application, specification, batch):
                target['issue_events'] = True

            for target in (application, specification, batch):
                target['observation_generations'] = True
            with self.assertRaises(ValueError):
                capture.recipe.helper_variant(Path('/reference'), Path('/spec'))
            closure['status'] = 'fresh_numerical_and_generations_independently_reviewed'
            result = capture.recipe.helper_variant(Path('/reference'), Path('/spec'))
            self.assertTrue(result['observation_generations'])
            for target in (application, specification, batch):
                target.update(selected_commands=True, preparation_boundary=True,
                              application_purpose='selected_command_observations')
            batch.update(analyzer='/analyze.py', analyzer_sha256='sha/analyze.py')
            with self.assertRaises(ValueError):
                capture.recipe.helper_variant(Path('/reference'), Path('/spec'))
            closure['status'] = 'fresh_numerical_and_selected_commands_independently_reviewed'
            self.assertTrue(capture.recipe.helper_variant(Path('/reference'), Path('/spec'))['selected_commands'])
            batch['analyzer_sha256'] = 'changed'
            with self.assertRaises(ValueError):
                capture.recipe.helper_variant(Path('/reference'), Path('/spec'))
            batch['analyzer_sha256'] = 'sha/analyze.py'
            for target in (application, specification, batch):
                for key, bad in (('selected_commands', False), ('selected_commands', 1),
                                 ('preparation_boundary', False), ('preparation_boundary', 1),
                                 ('application_purpose', 'preparation')):
                    old = target[key]
                    target[key] = bad
                    with self.assertRaises(ValueError):
                        capture.recipe.helper_variant(Path('/reference'), Path('/spec'))
                    target[key] = old
            for target in (application, specification, batch):
                for key in ('selected_commands', 'preparation_boundary', 'application_purpose'):
                    del target[key]
            closure['status'] = 'fresh_numerical_and_generations_independently_reviewed'
            for target in (application, specification, batch):
                for bad in (False, 1):
                    target['observation_generations'] = bad
                    with self.assertRaises(ValueError):
                        capture.recipe.helper_variant(Path('/reference'), Path('/spec'))
                target['observation_generations'] = True
            for target in (application, specification, batch):
                target['issue_events'] = False
            with self.assertRaises(ValueError):
                capture.recipe.helper_variant(Path('/reference'), Path('/spec'))
            for target in (application, specification, batch):
                target['issue_events'] = True
                target['observation_generations'] = False
            closure['status'] = 'fresh_numerical_and_owner_events_independently_reviewed'
            del closure['initial_offline_error_sha256']
            result = capture.recipe.helper_variant(Path('/reference'), Path('/spec'))
            self.assertTrue(result['issue_events'])
            self.assertNotIn('/audit-initial-error.json', result['numerical_proofs'])
            for target in (application, specification, batch):
                target['issue_events'] = False
                with self.assertRaises(ValueError):
                    capture.recipe.helper_variant(Path('/reference'), Path('/spec'))
                target['issue_events'] = True

    def test_depth_hardware_authentication_does_not_invent_connection_receipts(self):
        document, actual, path, begin, end = depth_fixture()
        summary = capture.validate_hardware_depth(document, actual, path, begin, end)
        self.assertFalse(summary['connection_ids_independently_observed'])
        self.assertTrue(summary['pid_start_independently_authenticated'])
        self.assertEqual(summary['connection_binding_source'], 'service_ledger')
        self.assertEqual(summary['roles']['training']['connection'], 1)
        self.assertNotIn('effective_selection', actual)
        for section, key, value in (('slots', 'pid', 999), ('slots', 'start_ticks', 999),
                ('slots', 'connection', 0), ('slots', 'connection', 2), ('slots', 'retired', 0),
                ('opcodes', 'opcode', 70), ('opcodes', 'admitted', 1), ('buckets', 'unknown', 1),
                ('buckets', 'covered_ns', 1), ('buckets', 'no_inflight_compute_ns', 0)):
            bad = copy.deepcopy(document)
            bad[section][0][key] = value
            with self.assertRaises(ValueError):
                capture.validate_hardware_depth(bad, actual, path, begin, end)
        for bad in (dict(actual, client_pins=list(reversed(actual['client_pins']))),
                    dict(actual, server_identity=dict(pid=99)),
                    dict(actual, server_command=actual['server_command'][:-1]+['/foreign.json'])):
            with self.assertRaises(ValueError):
                capture.validate_hardware_depth(document, bad, path, begin, end)
        for key, value in (('valid', False), ('failure', 1), ('unclosed', 1), ('bucket_ns', 1)):
            with self.assertRaises(ValueError):
                capture.validate_hardware_depth(dict(document, **{key: value}), actual, path, begin, end)

    def test_depth_hardware_whole_bucket_partition_and_instantaneous_semantics(self):
        document, actual, path, begin, end = depth_fixture()
        result = capture.validate_hardware_depth(document, actual, path, begin, end)
        self.assertEqual(result['effective_duration_ns'], 29_800_000_000)
        self.assertEqual(result['omitted_edge_ns'], 202_000_000)
        self.assertEqual(result['next_direction'], 'no_admitted')
        self.assertEqual(sum(result['host_state_ns'].values()), result['effective_duration_ns'])
        self.assertEqual(result['bucket_totals']['observations'], 149)
        self.assertTrue(result['readiness_is_instantaneous'])
        self.assertFalse(result['outstanding_is_gpu_busy'])
        for absent, no_flight, expected in ((0, 7, 'admitted_but_no_inflight'),
                                          (0, 0, 'issuing_or_submitted'), (4, 7, 'mixed')):
            changed = copy.deepcopy(document)
            for row in changed['buckets']:
                row['no_admitted_compute_ns'] = row['covered_ns']*absent//10
                row['no_inflight_compute_ns'] = row['covered_ns']*no_flight//10
            self.assertEqual(capture.validate_hardware_depth(changed, actual, path, begin, end)['next_direction'],
                             expected)

    def test_helper_selection_is_scoped_and_restored_on_failure(self):
        for enabled in (False, True):
            owner = Mock(return_value=(['/candidate/bin/glimmer_execution_service'], ['pin']))
            environment = Mock(return_value={'LD_PRELOAD': '/candidate/lib/libglimmer_execution_interceptor.so'})
            common = SimpleNamespace(service_command=owner, owned_environment=environment)
            h = SimpleNamespace(training=SimpleNamespace(SERVER=Path('/old/server'), PRELOAD=Path('/old/preload')))
            variant = dict(build='/candidate', helper_plans=enabled,
                           runtime_environment={'GLIMMER_EXECUTION_HOST_RANGE_SEEK': '1'})
            with self.assertRaisesRegex(ValueError, 'workload failure'):
                with capture.helper_configuration(common, h, variant):
                    argv, pins = common.service_command()
                    self.assertEqual(argv.count('--skip-idle-helper-plans'), int(enabled))
                    self.assertEqual(pins, ['pin'])
                    self.assertEqual(h.training.SERVER, Path('/candidate/bin/glimmer_execution_service'))
                    self.assertEqual(common.owned_environment()['GLIMMER_EXECUTION_HOST_RANGE_SEEK'], '1')
                    owner.return_value = ([*argv, '--skip-idle-helper-plans'], pins)
                    with self.assertRaises(ValueError):
                        common.service_command()
                    raise ValueError('workload failure')
            self.assertIs(common.service_command, owner)
            self.assertIs(common.owned_environment, environment)
            self.assertEqual((h.training.SERVER, h.training.PRELOAD), (Path('/old/server'), Path('/old/preload')))
        owner.return_value = (['service'], ['pin'])
        variant.update(helper_plans=True, issue_depth=True)
        with capture.helper_configuration(common, h, variant, Path('/new/depth.json')):
            self.assertEqual(common.service_command()[0][-2:], ['--issue-depth-output', '/new/depth.json'])
        with capture.helper_configuration(common, h, variant):
            with self.assertRaises(ValueError):
                common.service_command()
        variant['issue_events'] = True
        with capture.helper_configuration(common, h, variant, Path('/new/depth.json')):
            self.assertEqual(common.service_command()[0][-1], '--issue-depth-events')
            owner.return_value = (['service', '--issue-depth-events'], ['pin'])
            with self.assertRaises(ValueError):
                common.service_command()
        owner.return_value = (['service'], ['pin'])
        variant['selected_commands'] = True
        with capture.helper_configuration(common, h, variant, Path('/new/depth.json')):
            self.assertEqual(common.service_command()[0][-1], '--issue-depth-selections')
            variant['issue_events'] = False
            with self.assertRaises(ValueError):
                common.service_command()
            variant['issue_events'] = True
            owner.return_value = (['service', '--issue-depth-selections'], ['pin'])
            with self.assertRaises(ValueError):
                common.service_command()

    def test_helper_actual_observer_and_switch_validation(self):
        variant = dict(build='/candidate', helper_plans=True)
        value = dict(mode='equal', application_environment=dict(
            LD_PRELOAD='/candidate/lib/libglimmer_execution_interceptor.so',
            GLIMMER_EXECUTION_HOST_RANGE_SEEK='1', GLIMMER_EXECUTION_PREPARATION_TIMEOUT_MS='240000'),
            server_command=['/candidate/bin/glimmer_execution_service', '--persistent-workers', '320',
                            '--gemm-tile-side', '32', '--skip-idle-helper-plans'])
        capture.recipe.validate_helper_job(value, variant)
        for flag in ('--skip-idle-helper-plans', '--host-stage-output', '--audit-native',
                     '--observe-block-completion', '--issue-depth-events', '--issue-depth-selections'):
            with self.assertRaises(ValueError):
                capture.recipe.validate_helper_job({**value, 'server_command': value['server_command'] + [flag]}, variant)
        for key in ('GLIMMER_EXECUTION_RPC_OUTPUT_DIR', 'GLIMMER_EXECUTION_HOST_RANGE_DETAIL', 'LD_AUDIT'):
            with self.assertRaises(ValueError):
                capture.recipe.validate_helper_job({**value, 'application_environment':
                    {**value['application_environment'], key: '1'}}, variant)
        capture.recipe.validate_helper_job(dict(mode='native', application_environment={}), variant)
        with self.assertRaises(ValueError):
            capture.recipe.validate_helper_job(dict(value, mode='native'), variant)
        with self.assertRaises(ValueError):
            capture.recipe.validate_helper_job(dict(value, server_command=value['server_command']+
                                                   ['--issue-depth-output', '/depth.json']), variant)
        enabled = dict(variant, issue_depth=True)
        with self.assertRaises(ValueError):
            capture.recipe.validate_helper_job(value, enabled)
        capture.recipe.validate_helper_job(dict(value, server_command=value['server_command']+
                                               ['--issue-depth-output', '/depth.json']), enabled)
        enabled['issue_events'] = True
        with self.assertRaises(ValueError):
            capture.recipe.validate_helper_job(dict(value, server_command=value['server_command']+
                                                   ['--issue-depth-output', '/depth.json']), enabled)
        capture.recipe.validate_helper_job(dict(value, server_command=value['server_command']+
                         ['--issue-depth-output', '/depth.json', '--issue-depth-events']), enabled)
        enabled['selected_commands'] = True
        with self.assertRaises(ValueError):
            capture.recipe.validate_helper_job(dict(value, server_command=value['server_command']+
                             ['--issue-depth-output', '/depth.json', '--issue-depth-events']), enabled)
        capture.recipe.validate_helper_job(dict(value, server_command=value['server_command']+
                         ['--issue-depth-output', '/depth.json', '--issue-depth-events', '--issue-depth-selections']), enabled)

    def test_hardware_comparison_uses_offer_end_and_conservative_bounds(self):
        def worker(mode, activity, completed):
            field = dict(interior_duration_weighted_ratio=activity, whole_window_ratio_lower_bound=activity-.005,
                         whole_window_ratio_upper_bound=activity+.005, interior_coverage_fraction=.98)
            return dict(mode=mode, status='passed_requires_review',
                activity=dict(fields={str(key): dict(field) for key in capture.aggregate.FIELDS}),
                job=dict(metrics=dict(window_begin_ns=1, window_end_ns=30_002_000_001,
                    at_offer_end=dict(completed_requests=completed), completed_requests=999,
                    training_completions_in_window=10)))
        off = [worker('native', .3, 100), worker('equal', .1, 20)]
        on = [worker('native', .3, 100), worker('equal', .15, 19)]
        self.assertTrue(capture.recipe.compare_helper_hardware(off, on)['all_gates_passed'])
        on[1]['job']['metrics']['at_offer_end']['completed_requests'] = 18
        self.assertFalse(capture.recipe.compare_helper_hardware(off, on)['gates']['equal_completions'])
        on[1] = worker('equal', .115, 20)
        self.assertFalse(capture.recipe.compare_helper_hardware(off, on)['gates']['sm_activity_improvement'])
        on[0] = worker('native', .32, 100)
        gates = capture.recipe.compare_helper_hardware(off, on)['gates']
        self.assertTrue(gates['native_sm_drift'])
        self.assertFalse(gates['native_uncertainty_overlap'])

    def test_actual_remaining_unit_and_outer_deadline_not_nominal_constants(self):
        launch = dict(started_ns=1_000_000_000, outer_deadline_ns=821_000_000_000)
        with patch.object(capture, 'command', return_value='RuntimeMaxUSec=13min 20s\nActiveEnterTimestampMonotonic=1000000'), \
             patch.object(capture.recipe, 'read', return_value=launch):
            result = capture.startup_budget(Path('/attempt'), 'native', now_ns=lambda: 50_000_000_000)
            self.assertEqual(result['deadline_ns'], 801_000_000_000)
            with self.assertRaisesRegex(ValueError, 'remaining'):
                capture.startup_budget(Path('/attempt'), 'native', now_ns=lambda: 80_000_000_000)
            launch.update(started_ns=-50_000_000_000, outer_deadline_ns=770_000_000_000)
            with self.assertRaisesRegex(ValueError, 'remaining'):
                capture.startup_budget(Path('/attempt'), 'native', now_ns=lambda: 50_000_000_000)

    def test_changed_actual_unit_bound_rejected(self):
        with patch.object(capture, 'command', return_value='RuntimeMaxUSec=infinity\nActiveEnterTimestampMonotonic=1'), \
             self.assertRaisesRegex(ValueError, 'runtime bound'):
            capture.startup_budget(Path('/attempt'), 'native')

    def test_delayed_preflight_rejected_before_privileged_popen(self):
        with tempfile.TemporaryDirectory() as tmp:
            attempt = Path(tmp)
            (attempt / 'native').mkdir()
            guard = SimpleNamespace(group='/task')
            supervisor = capture.SamplerSupervisor(attempt, 'native', dict(observer_identities={}), guard)
            with patch.object(capture.observer, 'process_record', return_value=process(os.getpid(), 50, os.getuid())), \
                 patch.object(capture, 'startup_budget', side_effect=ValueError('remaining lifetime')), \
                 patch.object(capture.subprocess, 'Popen') as popen, self.assertRaisesRegex(ValueError, 'remaining'):
                supervisor.start()
            popen.assert_not_called()
            supervisor.stop()
            supervisor.close()


class SupervisorTests(unittest.TestCase):
    def fixture(self):
        owner, sampler = process(20, 50), process(42, 77, uid=0)
        value = capture.SamplerSupervisor(Path('/attempt'), 'native', {}, Mock())
        value.directory, value.process, value.proof = Mock(), Mock(), Mock()
        value.directory.exists.return_value = True
        value.process.wait.return_value = 0
        value.proof.expected, value.proof.poll.return_value = sampler, 0
        value.watch_plan = dict(owner=owner)
        sample = dict(bytes=3, sha256='a' * 64, uid=0)
        value.directory.file_identity.return_value = sample
        report = dict(status='external_observation_collected_requires_window_review', sampler=sampler,
            owner=owner, sample_identity=sample, owned_watches_retired=True, embedded_retired=True,
            cleanup_errors=[], counter_collection=True, ordinary_timing=False)
        value.directory.read.return_value = report
        return value, report

    def test_normal_retirement_joins_wrapper_pidfd_and_exact_watch_receipt(self):
        supervisor, _ = self.fixture()
        supervisor.stop()
        self.assertTrue(supervisor.receipt['normal_retirement'])
        self.assertEqual(supervisor.receipt['wrapper_returncode'], 0)
        supervisor.process.wait.assert_called_once_with(timeout=15)
        supervisor.process.kill.assert_not_called()
        supervisor.process.send_signal.assert_not_called()

    def test_failed_start_preserves_root_receipt_separately(self):
        supervisor, _ = self.fixture()
        supervisor.directory.exists.side_effect = lambda name: name == 'observer.json'
        supervisor.directory.read.return_value = dict(status='failed', error='root file limit',
                                                      sdk_entry_attempted=False)
        supervisor.check = Mock(side_effect=ValueError('wrapper exited'))
        with self.assertRaisesRegex(ValueError, 'wrapper exited'):
            supervisor.wait_file('identity.json', 20)
        self.assertEqual(supervisor.receipt['startup_error'], 'ValueError: wrapper exited')
        self.assertEqual(supervisor.receipt['startup_observer']['error'], 'root file limit')
        supervisor.directory.read.assert_called_once_with('observer.json', 0)

    def test_pidfd_terminal_sentinel_cannot_substitute_for_successful_wrapper(self):
        for returned in (124, 137, 1, -15, True):
            with self.subTest(returned=returned):
                supervisor, _ = self.fixture()
                supervisor.process.wait.return_value = returned
                with self.assertRaisesRegex(ValueError, 'wrapper status'):
                    supervisor.stop()
                self.assertFalse(supervisor.receipt['normal_retirement'])

    def test_failed_wrapper_keeps_root_diagnostic_without_qualifying_retirement(self):
        supervisor, report = self.fixture()
        supervisor.process.wait.return_value = 1
        report.update(status='failed', error='invalid profiling sample')
        with self.assertRaisesRegex(ValueError, 'wrapper status'):
            supervisor.stop()
        self.assertEqual(supervisor.receipt['terminal_observer']['error'], 'invalid profiling sample')
        self.assertFalse(supervisor.receipt['normal_retirement'])

    def test_wrapper_zero_without_actual_root_exit_or_proof_cannot_pass(self):
        for missing in ('pidfd', 'proof'):
            with self.subTest(missing=missing):
                supervisor, _ = self.fixture()
                if missing == 'pidfd':
                    supervisor.proof.poll.return_value = None
                else:
                    supervisor.proof = None
                with self.assertRaisesRegex(ValueError, 'actual sampler exit'):
                    supervisor.stop()

    def test_timeout_never_signals_root_and_retains_failure(self):
        supervisor, _ = self.fixture()
        supervisor.process.wait.side_effect = subprocess.TimeoutExpired('owned', 15)
        with self.assertRaisesRegex(RuntimeError, 'independent710'):
            supervisor.stop()
        supervisor.process.kill.assert_not_called()
        supervisor.process.terminate.assert_not_called()
        self.assertFalse(supervisor.receipt['normal_retirement'])

    def test_missing_stale_partial_or_wrong_sample_receipts_fail_closed(self):
        changes = (dict(status='failed'), dict(embedded_retired=False), dict(owned_watches_retired=False),
                   dict(cleanup_errors=['shutdown']), dict(sampler=process(42, 78, uid=0)),
                   dict(owner=process(21, 50)), dict(sample_identity={}), dict(ordinary_timing=True))
        for change in changes:
            with self.subTest(change=change):
                supervisor, report = self.fixture()
                report.update(change)
                with self.assertRaisesRegex(ValueError, 'retirement receipt'):
                    supervisor.stop()
                self.assertFalse(supervisor.receipt['normal_retirement'])
        supervisor, _ = self.fixture()
        supervisor.directory.read.side_effect = FileNotFoundError('root receipt missing')
        with self.assertRaises(FileNotFoundError):
            supervisor.stop()
        self.assertFalse(supervisor.receipt['normal_retirement'])

    def test_failed_close_still_retires_remaining_owned_handles(self):
        supervisor, _ = self.fixture()
        supervisor.log = Mock()
        supervisor.proof.close.side_effect = OSError('pidfd close')
        with self.assertRaisesRegex(ValueError, 'cleanup failed'):
            supervisor.close()
        supervisor.log.close.assert_called_once()
        supervisor.directory.close.assert_called_once()

    def test_startup_failure_invokes_bounded_stop_in_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            attempt = Path(tmp)
            plan = dict(reference='/reference', checkpoint_gpu='gpu', point={}, recipe=capture.recipe.RECIPE)
            h = SimpleNamespace(query=Mock(return_value='gpu'), training=SimpleNamespace(GPU_FIELDS='fields', require_idle=Mock()))
            common = SimpleNamespace(application_references=Mock(return_value=('/reference', '/export')))
            resources = Mock()
            resources.resource_snapshot.return_value = dict(available_device_sms=80)
            sustained = SimpleNamespace(load_resources=Mock(return_value=resources))
            with patch.object(capture.candidate, 'ResourceGuard') as guard, \
                 patch.object(capture.recipe, 'read', side_effect=[plan, {}]), \
                 patch.object(capture.recipe, 'validate_plan'), patch.object(capture, 'helpers', return_value=(common, h, sustained)), \
                 patch.object(capture, 'worker_binding', return_value={}), \
                 patch.object(capture, 'SamplerSupervisor') as supervisor:
                supervisor.return_value.start.side_effect = ValueError('ready failure')
                supervisor.return_value.stop.side_effect = ValueError('retirement failure')
                supervisor.return_value.receipt = dict(status='failed')
                guard.return_value.receipt = {}
                self.assertEqual(capture.worker(attempt, 'native'), 1)
                supervisor.return_value.stop.assert_called_once()
                supervisor.return_value.close.assert_called_once()
            report = json.loads((attempt / 'native-worker.json').read_text())
            self.assertEqual(report['status'], 'failed')
            self.assertEqual(report['error'], 'ValueError: ready failure')
            self.assertEqual(report['operation_error'], report['error'])
            self.assertEqual(report['sampler_cleanup_error'], 'ValueError: retirement failure')


class WorkerCountTests(unittest.TestCase):
    def test_scoped_count_changes_only_command_count_and_checks_actual_log(self):
        argv = ['service', '--persistent-workers', '320', '--gemm-tile-side', '32']
        command = Mock(return_value=(argv, ['low:0:configured', 'high:0:original']))
        original_log = Mock()
        ordinary = SimpleNamespace(ordinary_log=original_log, common=SimpleNamespace(PREFIX='LOG '))
        helper = SimpleNamespace(owner_command=command, ordinary=ordinary)
        line = ('LOG Execution service configuration: scheduler=priority, kernel_mode=preemptive, '
                'configured_workers=640, nccl_enabled=false; full_device_qualified=false')
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            with capture.worker_configuration(helper, 640):
                changed, pins = helper.owner_command()
                self.assertEqual(changed, ['service', '--persistent-workers', '640', '--gemm-tile-side', '32'])
                self.assertEqual(pins, ['low:0:configured', 'high:0:original'])
                self.assertEqual(argv[2], '320')
                self.assertEqual(helper.ordinary.ordinary_log(line)['configured_workers'], 640)
                for invalid in (line.replace('640', '320'), line + '\n' + line,
                                line.replace('preemptive', 'original'), line + '\n[error] failed'):
                    with self.assertRaises(ValueError):
                        helper.ordinary.ordinary_log(invalid)
                raise RuntimeError('injected')
        self.assertIs(helper.owner_command, command)
        self.assertIs(helper.ordinary.ordinary_log, original_log)

    def test_default_is_unchanged_and_unapproved_counts_rejected(self):
        helper = SimpleNamespace()
        with capture.worker_configuration(helper, 320):
            self.assertEqual(vars(helper), {})
        for count in (True, 0, 641, 2560):
            with self.assertRaises(ValueError), capture.worker_configuration(helper, count):
                pass


class OwnerAuditTests(unittest.TestCase):
    def test_application_reparse_requires_exact_raw_trace_and_offered_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            value = job('exclusive')
            schedule = dict(begin_ns=1_000_000_000, trace=dict(duration_ns=30_002_000_000))
            value.update(application_commands={'inference': ['fixed-command']}, schedule=schedule,
                metrics=dict(window_begin_ns=1_000_000_000, window_end_ns=31_002_000_000),
                workers_cpu={}, completion='complete', numerical_scope='unchanged')
            capture.save(directory / 'inference.start.json', schedule)
            value['artifact_identities'] = {'inference.start.json': capture.digest(directory / 'inference.start.json')}
            capture.save(directory / 'attempt.json', value)
            sustained = SimpleNamespace(commands=Mock(return_value=value['application_commands']),
                schedule=Mock(return_value=schedule), assess=Mock())
            self.assertEqual(capture.audit_application(sustained, Mock(), Mock(), directory, value, {}),
                             (1_000_000_000, 31_002_000_000))
            sustained.assess.assert_called_once()
            capture.save(directory / 'inference.start.json', dict(begin_ns=2))
            with self.assertRaisesRegex(ValueError, 'output changed'):
                capture.audit_application(sustained, Mock(), Mock(), directory, value, {})

    def test_scope_and_raw_flags_cannot_mask_a_changed_numerical_reparse(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            value = job('exclusive')
            value.update(application_commands={}, schedule=dict(begin_ns=1), artifact_identities={},
                         metrics={}, workers_cpu={}, completion='complete', numerical_scope='unchanged')
            capture.save(directory / 'attempt.json', value)
            capture.save(directory / 'inference.start.json', value['schedule'])
            def changed(_, __, record):
                record['completion'] = 'changed'
            sustained = SimpleNamespace(commands=Mock(return_value={}), schedule=Mock(return_value=value['schedule']),
                                        assess=changed)
            with self.assertRaisesRegex(ValueError, 'accounting differs'):
                capture.audit_application(sustained, Mock(), Mock(), directory, value, {})

    def test_each_mode_requires_its_actual_owner_set_not_sampler(self):
        for mode in capture.observer.MODES:
            rows, actual = guards(mode)
            with self.subTest(mode=mode):
                result = capture.audit_guards(rows, actual, process(42, 77, uid=0), process(20, 50),
                                              10_000_000_000, 11_000_000_000)
                self.assertEqual(result['inside_window'], 2)
                self.assertNotIn(42, result['required_gpu_owners'])

    def test_sampler_row_is_optional_and_only_exact_root_incarnation_is_allowed(self):
        rows, actual = guards()
        sampler, owner = process(42, 77, uid=0), process(20, 50)
        for row in rows:
            row['processes'].append(dict(gpu_uuid=capture.observer.GPU_UUID, pid=42, identity=dict(sampler)))
        capture.audit_guards(rows, actual, sampler, owner, 10_000_000_000, 11_000_000_000)
        for field, wrong in (('start_ticks', 78), ('uid', 1000), ('cgroup', '/foreign')):
            bad = copy.deepcopy(rows)
            bad[1]['processes'][-1]['identity'][field] = wrong
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'incarnation'):
                capture.audit_guards(bad, actual, sampler, owner, 10_000_000_000, 11_000_000_000)

    def test_guard_gap_unbracketed_wrong_uid_reuse_and_foreign_work_rejected(self):
        for failure in ('missing', 'gap', 'uid', 'reuse', 'foreign', 'sampler-only', 'missing-state', 'bool-uid'):
            rows, actual = guards()
            if failure == 'missing':
                rows.pop(0)
                rows[0]['monotonic_ns'] += 1
            elif failure == 'gap':
                rows[-1]['monotonic_ns'] = 15_000_000_000
            elif failure == 'uid':
                rows[1]['processes'][0]['identity']['uid'] = 0
            elif failure == 'reuse':
                rows[1]['processes'][0]['identity']['start_ticks'] += 1
            elif failure == 'missing-state':
                rows[1]['processes'][0]['identity'].pop('state')
            elif failure == 'bool-uid':
                rows[1]['processes'][0]['identity']['uid'] = True
            elif failure == 'foreign':
                rows[1]['processes'].append(dict(gpu_uuid=capture.observer.GPU_UUID, pid=99))
            elif failure == 'sampler-only':
                rows[1]['processes'] = [dict(gpu_uuid=capture.observer.GPU_UUID, pid=42,
                                             identity=process(42, 77, uid=0))]
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                capture.audit_guards(rows, actual, process(42, 77, uid=0), process(20, 50),
                                     10_000_000_000, 11_000_000_000)

    def test_signalled_failed_or_unobserved_application_cannot_qualify(self):
        for failure in ('signal', 'returncode', 'missing-owner'):
            value = job()
            if failure == 'signal':
                value['authenticated_child_cleanup']['training']['signals'] = [15]
            elif failure == 'returncode':
                value['final_returncodes']['training'] = 1
            else:
                value['observed_gpu_pids'] = [42]
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                capture.normal_application(value)

    def test_profiled_canonical_envelope_overrides_all_qualification_flags(self):
        value = {**job(), **{name: True for name in capture.FLAGS}, 'counter_collection': False}
        result = capture.profiled_envelope(value, dict(status='retired'), 'a' * 64)
        self.assertTrue(value['ordinary_timing'])  # Preserve the original helper input.
        self.assertTrue(result['counter_collection'])
        self.assertEqual(result['status'], capture.PROFILED)
        self.assertTrue(all(result[name] is False for name in capture.FLAGS))
        self.assertEqual(result['raw_helper_receipt']['path'], 'application/helper-attempt.json')


class BindingTests(unittest.TestCase):
    def test_worker_requires_fresh_mode_and_exact_live_parent_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            attempt = Path(tmp)
            capture.save(attempt / 'plan.json', {})
            owner = process(4242, 77, uid=os.getuid())
            launch = dict(mode='native', attempt=str(attempt), full_preflight_status='complete_input_seals_unchanged',
                plan_sha256=capture.digest(attempt / 'plan.json'), preflight_verified_ns=100_000_000_000,
                started_ns=100_000_000_000, collector=owner)
            plan = dict(observer_identities={}, input_identities={})
            capture.save(attempt / 'native-launch.json', launch)
            with patch.object(capture.recipe, 'validate_structure'), patch.object(capture.observer, 'verify_seals'), \
                 patch.object(capture.observer, 'PinnedProcess') as pin:
                result = capture.worker_binding(attempt, 'native', plan, now_ns=lambda: 102_000_000_000)
                self.assertEqual(result['plan_sha256'], launch['plan_sha256'])
                pin.assert_called_once_with(owner)
                for changes in (dict(mode='priority'), dict(plan_sha256='stale'),
                                dict(preflight_verified_ns=0, started_ns=0), dict(full_preflight_status='failed')):
                    capture.save(attempt / 'native-launch.json', {**launch, **changes})
                    with self.subTest(changes=changes), self.assertRaises(ValueError):
                        capture.worker_binding(attempt, 'native', plan, now_ns=lambda: 102_000_000_000)
                capture.save(attempt / 'native-launch.json', launch)
                pin.return_value.__enter__.return_value.check.side_effect = ValueError('parent died')
                with self.assertRaisesRegex(ValueError, 'parent died'):
                    capture.worker_binding(attempt, 'native', plan, now_ns=lambda: 102_000_000_000)

    def test_measured_66_second_full_brackets_are_outside_actual_unit_budget(self):
        self.exercise_bracketed_collect(fail_postflight=False)

    def test_failed_full_postflight_prevents_next_mode(self):
        self.exercise_bracketed_collect(fail_postflight=True)

    def exercise_bracketed_collect(self, fail_postflight):
        with tempfile.TemporaryDirectory() as tmp:
            attempt, source = Path(tmp, 'attempt-test'), Path(tmp, 'source-plan.json')
            attempt.mkdir()
            capture.save(source, dict(recipe=capture.recipe.RECIPE))
            clock, validations, launches = [1_000_000_000], [0], []
            def full_check(_):
                validations[0] += 1
                clock[0] += 66_290_000_000  # The actual measured full-check duration.
                if fail_postflight and validations[0] == 4:
                    raise ValueError('postflight identity changed')
            def unit(folder, mode, **kwargs):
                launches.append(mode)
                launch = capture.recipe.read(folder / (mode + '-launch.json'))
                self.assertEqual(launch['started_ns'], clock[0])
                unit_start_us = (clock[0] + 1_000_000_000) // 1000
                clock[0] += 2_000_000_000
                with patch.object(capture, 'command', return_value=
                        f'RuntimeMaxUSec=13min 20s\nActiveEnterTimestampMonotonic={unit_start_us}'):
                    budget = capture.startup_budget(folder, mode, now_ns=lambda: clock[0])
                self.assertEqual(budget['deadline_ns'] - clock[0], 799_000_000_000)
                clock[0] += 20_000_000_000
                return dict(returncode=0, unit_removed=True), dict(status='passed_requires_review')
            with patch.object(capture.recipe, 'validate_plan', side_effect=full_check), \
                 patch.object(capture, 'create_attempt', return_value=attempt), \
                 patch.object(capture.shutil, 'disk_usage', return_value=SimpleNamespace(free=9 * 1024 ** 3)), \
                 patch.object(capture.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout='', stderr='')), \
                 patch.object(capture, 'wait_for_idle', return_value=dict(achieved=True)), \
                 patch.object(capture.candidate, 'launch_unit', side_effect=unit), \
                 patch.object(capture.observer, 'process_record', return_value=process(4242, 77, uid=os.getuid())), \
                 patch.object(capture.time, 'monotonic_ns', side_effect=lambda: clock[0]):
                self.assertEqual(capture.collect(source), int(fail_postflight))
            self.assertEqual(launches, ['exclusive'] if fail_postflight else list(capture.observer.MODES))
            postflight = capture.recipe.read(attempt / 'exclusive-postflight.json')
            self.assertEqual(postflight['status'], 'failed' if fail_postflight else 'complete_input_seals_unchanged')
            self.assertEqual(postflight['finished_ns'] - postflight['started_ns'], 66_290_000_000)

    def test_explicit_current_helper_binding_required(self):
        common = Mock()
        h = SimpleNamespace(training=SimpleNamespace(SERVER=capture.candidate.BUILD / 'bin/glimmer_execution_service',
            PRELOAD=capture.candidate.BUILD / 'lib/libglimmer_execution_interceptor.so'))
        sustained = SimpleNamespace(ordinary=SimpleNamespace(common=common))
        with patch.object(capture.candidate, 'load_helpers', return_value=(common, h)), \
             patch.object(capture.importlib, 'import_module', return_value=sustained):
            self.assertEqual(capture.helpers(), (common, h, sustained))
            h.training.SERVER = Path('/old/Release')
            with self.assertRaisesRegex(ValueError, 'cannot substitute'):
                capture.helpers()

    def test_privileged_recipe_is_exact_absolute_and_plan_bound(self):
        argv = capture.recipe.PRIVILEGED_COMMAND
        self.assertEqual(argv[:6], ['/usr/bin/sudo', '-n', '/usr/bin/timeout', '--signal=TERM', '--kill-after=5s', '710s'])
        self.assertIn('/usr/bin/python3', argv)
        self.assertIn('/usr/bin/env', argv)
        self.assertEqual(capture.recipe.RECIPE['privileged_command'], argv)
        self.assertEqual(argv[-2:], ['--sampler-folder', '{observer_folder}'])

    def test_collect_fails_auth_before_idle_unit_or_gpu_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            attempt = Path(tmp, 'attempt-test')
            attempt.mkdir()
            source = Path(tmp, 'plan.json')
            source.write_text('{}')
            with patch.object(capture.recipe, 'read', return_value={}), patch.object(capture.recipe, 'validate_plan'), \
                 patch.object(capture, 'create_attempt', return_value=attempt), \
                 patch.object(capture.shutil, 'disk_usage', return_value=SimpleNamespace(free=9 * 1024 ** 3)), \
                 patch.object(capture.subprocess, 'run', return_value=SimpleNamespace(returncode=1, stdout='', stderr='password')), \
                 patch.object(capture, 'wait_for_idle') as idle, patch.object(capture.candidate, 'launch_unit') as unit:
                self.assertEqual(capture.collect(source), 1)
            idle.assert_not_called()
            unit.assert_not_called()
            self.assertEqual(json.loads((attempt / 'report.json').read_text())['launched_modes'], [])


if __name__ == '__main__':
    unittest.main()
