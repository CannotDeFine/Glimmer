"""CPU-only controls for the task wrapper; no historical helper or GPU imports."""

import json
import hashlib
from collections import Counter
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import run_bert_candidate_validation as capture
import run_bert_host_range_optimization as sharing
import run_bert_frontend_compiler as compiler
import analyze_bert_host_range as host_stages
import run_bert_host_range as host_capture
import analyze_bert_host_range_seek as host_seek
import run_bert_intake_boundary as intake_capture
import run_bert_delivery_boundary as delivery_capture
import analyze_bert_poll_boundary as poll_boundary


def memory(**changes):
    return dict({'memory.max': str(capture.MEMORY_MAX), 'memory.swap.max': '0', 'memory.oom.group': '1',
                 'memory.current': '1234', 'memory.events': 'low 0\nhigh 0\nmax 0\noom 0\noom_kill 0'}, **changes)


class GuardTests(unittest.TestCase):
    def test_intake_exit_receipt_is_exact_normal_owned_and_bound(self):
        prefix = '[2026-11-01 00:00:00.000] [info] '
        ready = prefix + ('Execution service ready: pid=42, start=100; '
                          'routing=explicit_opt_in, application_qualified=false\n')
        receipt = prefix + ('Async intake receipt: schema=1, pid=42, start=100, enabled=1, '
                            'successful_second_takes=3\n')
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(delivery_capture.lifecycle.candidate, 'normal_cleanup') as cleanup:
            path = Path(directory) / 'server.log'
            job = dict(mode='equal', server_command=['service', delivery_capture.INTAKE_FLAG],
                       server_identity=dict(pid=42, start=100))

            def parse(text, *, stale=False, changes=None):
                path.write_text(text)
                value = dict(job, artifact_identities={'server.log': 'bad' if stale else
                             delivery_capture.digest(path)}, **(changes or {}))
                return delivery_capture.intake_exit_receipt(path, value)

            self.assertEqual(parse(ready + receipt)['successful_second_takes'], 3)
            self.assertTrue(cleanup.called)
            for text in (ready, receipt, ready + receipt * 2, ready * 2 + receipt, receipt + ready,
                         ready + receipt.replace('schema=1', 'schema=2'),
                         ready + receipt.replace('enabled=1', 'enabled=0'),
                         ready + receipt.replace('pid=42', 'pid=43'),
                         ready + receipt.replace('start=100', 'start=101'),
                         ready + receipt.replace('takes=3', 'takes=03'),
                         ready + receipt.replace('takes=3', 'takes=-1'),
                         ready + receipt.replace('takes=3', 'takes=0'),
                         ready + receipt.replace('takes=3', f'takes={2**64 - 1}'),
                         ready + receipt.replace('takes=3', f'takes={2**64}')):
                with self.assertRaises(ValueError):
                    parse(text)
            with self.assertRaises(ValueError):
                parse(ready + receipt, stale=True)
            for changes in (dict(mode='native'), dict(server_command=['service']),
                            dict(server_command=job['server_command'] + [delivery_capture.INTAKE_FLAG])):
                with self.assertRaises(ValueError):
                    parse(ready + receipt, changes=changes)
            cleanup.side_effect = ValueError('nonzero or signalled exit')
            with self.assertRaises(ValueError):
                parse(ready + receipt)

    def test_intake_purpose_keeps_observers_off_and_preserves_command_shape(self):
        plan = dict(status='frozen_async_intake_application', application_purpose='ordinary_async_intake_exercise',
                    async_intake_burst=True, helper_plans=True, diagnostic_only=True,
                    full_device_qualified=False, performance_qualified=False)
        delivery_capture.validate_intake_plan(plan)
        for key, value in (('status', 'frozen_helper_planning_application'),
                           ('application_purpose', 'selected_command_observations'),
                           ('async_intake_burst', 1), ('helper_plans', False),
                           ('diagnostic_only', False), ('performance_qualified', True),
                           ('issue_depth', True), ('issue_events', True),
                           ('observation_generations', 1), ('preparation_boundary', True),
                           ('selected_commands', True)):
            with self.assertRaises(ValueError):
                delivery_capture.validate_intake_plan(dict(plan, **{key: value}))
        self.assertEqual(delivery_capture.intake_service_command((['service'], ['pin'])),
                         (['service', delivery_capture.HELPER_FLAG, delivery_capture.INTAKE_FLAG], ['pin']))
        with self.assertRaises(ValueError):
            delivery_capture.intake_service_command((['service', delivery_capture.INTAKE_FLAG], []))
        workflow = delivery_capture.lifecycle
        job = dict(mode='equal', application_environment=dict(LD_PRELOAD=str(workflow.PRELOAD),
                   GLIMMER_EXECUTION_HOST_RANGE_SEEK='1'), server_command=[str(workflow.SERVER),
                   delivery_capture.HELPER_FLAG, delivery_capture.INTAKE_FLAG])
        with patch.object(delivery_capture, 'INTAKE_BURST', True), \
                patch.object(delivery_capture, 'HELPER_PLANS', True):
            delivery_capture.validate_environment(job)
            for command in (job['server_command'][:-1], job['server_command'] + [delivery_capture.INTAKE_FLAG],
                            job['server_command'] + [delivery_capture.DEPTH_FLAG, '/output']):
                with self.assertRaises(ValueError):
                    delivery_capture.validate_environment(dict(job, server_command=command))

    def test_session_overlay_is_exact_and_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline, build = root / 'baseline', root / 'candidate'
            baseline.mkdir()
            build.mkdir()
            source = str(delivery_capture.PROJECT_ROOT / 'src/backend/cuda/cuda_execution_session.cc')
            relative = delivery_capture.SESSION + source.lstrip('/') + '.o'
            output = build / relative
            output.parent.mkdir(parents=True)
            output.write_bytes(b'reviewed session object')
            old_output = baseline / 'session.o'
            old_dep, new_dep = Path(str(old_output)+'.d'), Path(str(output)+'.d')
            old_dep.write_text('session.o: ' + source + '\n')
            new_dep.write_text('session.o: ' + source + '\n')
            old = [dict(file=source, directory=str(baseline), command=f'c++ -g -fPIC -o {old_output} -c {source}')]
            new = [dict(file=source, directory=str(build), command=f'c++ -g -fPIC -o {output} -c {source}')]
            backend = str(baseline / 'src/backend/cuda/libglimmer_cuda_backend.a')
            linked = [str(output), backend]
            mapping = relative
            symbols = ('1 T glimmer::backend::cuda::CudaExecutionSession::accept_async_kernel(unsigned long, Kernel const&)\n'
                       '2 T glimmer::backend::cuda::CudaExecutionSession::accept_async_kernel(unsigned long, Kernel const&, bool*)\n')

            def check(part='service', rows=None, args=None, text=None):
                return delivery_capture.session_overlay_identity(part, old, new if rows is None else rows,
                    linked if args is None else args, mapping if text is None else text, build / 'service', build)

            with patch.object(delivery_capture.lifecycle.candidate, 'BUILD', baseline), \
                    patch.object(delivery_capture, 'PREPARATION_BOUNDARY', True), \
                    patch.object(delivery_capture.subprocess, 'check_output', return_value=symbols) as nm:
                selected, paths = check()
                self.assertEqual(selected, str(output))
                self.assertTrue({output, new_dep, old_dep}.issubset(paths))
                for args in ([str(output)+'wrong', backend], [str(output), str(output), backend],
                              [backend, str(output)]):
                    with self.assertRaises(ValueError):
                        check(args=args)
                with self.assertRaises(ValueError):
                    check('frontend')
                self.assertEqual(check('frontend', args=[backend]), (None, set()))
                with patch.object(delivery_capture, 'PREPARATION_BOUNDARY', False), self.assertRaises(ValueError):
                    check()
                with patch.object(delivery_capture, 'PREPARATION_BOUNDARY', False), \
                        patch.object(delivery_capture, 'INTAKE_BURST', True):
                    self.assertEqual(check()[0], str(output))
                    with self.assertRaises(ValueError):
                        check(args=[backend, str(output)])
                for text in ('', mapping + '\nlibglimmer_cuda_backend.a(cuda_execution_session.cc.o)'):
                    with self.assertRaises(ValueError):
                        check(text=text)
                for command in (new[0]['command'].replace('-g', '-O2'),
                                 new[0]['command'].replace(str(output), str(output)+'wrong')):
                    with self.assertRaises(ValueError):
                        check(rows=[dict(new[0], command=command)])
                new_dep.write_text('session.o: ' + source + ' /tmp/unreviewed-header\n')
                with self.assertRaises(ValueError):
                    check()
                new_dep.write_text(old_dep.read_text())
                nm.return_value = symbols.splitlines()[0]
                with self.assertRaises(ValueError):
                    check()
                # The returned object is part of the frozen input envelope.
                bound = delivery_capture.BoundInputs()
                bound.identities[str(output)] = delivery_capture.digest(output)
                bound.verify()
                output.write_bytes(b'changed object')
                with self.assertRaises(ValueError):
                    bound.verify()

    def test_issue_depth_complete_accounting_and_rejection(self):
        width = 200_000_000
        row = dict(begin_ns=width, covered_ns=width, admitted_not_submitted_ns=20,
                   submitted_not_retired_ns=40, issue_call_ns=10, no_admitted_compute_ns=5,
                   no_inflight_compute_ns=6, admitted=2, submitted=2, retired=2,
                   observations=4, ready_compute=1, not_ready_compute=1, other_pending=1,
                   no_selected_command=1, unknown=0, ready_without_inflight=1,
                   maximum_observation_gap_ns=100)
        tail = {key: 0 for key in row}
        tail['begin_ns'] = 2 * width
        slots = [dict(connection=i, pid=10+i, start_ticks=20+i, admitted=1, submitted=1, retired=1)
                 for i in (1, 2)]
        base = dict(schema=1, diagnostic_only=True, valid=True, failure=0, unclosed=0, pid=99,
                    readiness_is_instantaneous=True, outstanding_is_gpu_busy=False,
                    origin_ns=width, begin_ns=width, end_ns=2*width, bucket_ns=width,
                    slots=slots, opcodes=[dict(opcode=11, admitted=1), dict(opcode=42, admitted=1)],
                    buckets=[row, tail])
        job = dict(server_identity=dict(pid=99), effective_selection=dict(low=[1], high=[2]),
                   application_identities=dict(low=dict(pid=11, start=21), high=dict(pid=12, start=22)))
        self.assertEqual(delivery_capture.validate_issue_depth(base, job)['admitted'], 2)
        with_events = deepcopy(base)
        with_events['schema'] = 2
        with_events['events'] = dict(capacity=524288, complete=True, ready_duration=False,
                                     gpu_timestamps=False, fields=list(delivery_capture.EVENT_FIELDS),
                                     records=[[1, 1, 11, width+1, width+2, width+3, width+4,
                                               width+5, width+6, width+7, 1, 2, 1],
                                              [2, 1, 42, width+2, width+3, width+4, width+5,
                                               width+6, width+7, width+8, 2, 1, 1]])
        result = delivery_capture.validate_issue_depth(with_events, job, expected_events=True)
        self.assertEqual(result['events']['completed'], 2)
        generations = deepcopy(with_events)
        generations['schema'] = 3
        generations['events']['fields'] = list(delivery_capture.GENERATION_FIELDS)
        # Generation values are per-slot, not globally ordered or comparable.
        generations['events']['records'][0].extend([100, 101])
        generations['events']['records'][1].extend([0, 2])
        selected = dict(expected_events=True, expected_generations=True)
        result = delivery_capture.validate_issue_depth(generations, job, **selected)
        self.assertEqual(result['events']['observation_generations']['first_observation_positive'], 1)
        self.assertEqual(result['events']['observation_generations']['earlier_observations_without_positive'], 1)
        preparation = deepcopy(generations)
        preparation['schema'] = 4
        preparation['events']['fields'] = list(delivery_capture.PREPARATION_FIELDS)
        preparation['events']['records'][0].extend([1, width+1])
        preparation['events']['records'][1].extend([0, 0])
        prepared_options = dict(selected, expected_preparation=True)
        result = delivery_capture.validate_issue_depth(preparation, job, **prepared_options)
        self.assertEqual(result['events']['preparation_boundary']['deferred_commands'], 1)
        self.assertEqual(result['events']['preparation_boundary']['nondeferred_commands'], 1)
        selections = deepcopy(preparation)
        selections['schema'] = 5
        selections['slots'][0]['observation_generation'] = 4
        selections['slots'][1]['observation_generation'] = 2
        selections['events']['records'][0][13:] = [0, 3, 0, 0]
        selections['buckets'][0].update(observations=6, ready_compute=3)
        selections['selections'] = dict(capacity=524288, complete=True,
            fields=list(delivery_capture.SELECTION_FIELDS),
            classifications=list(delivery_capture.SELECTION_CLASSES),
            records=[[1, 2, width+1, 1, 1], [1, 3, width+2, 1, 0],
                     [2, 1, width+2, 77, 2], [1, 4, width+3, 1, 0], [2, 2, width+3, 1, 0]])
        selection_options = dict(prepared_options, expected_selections=True)
        result = delivery_capture.validate_issue_depth(selections, job, **selection_options)
        self.assertEqual(result['events']['selections']['observations'], 6)
        self.assertEqual(result['events']['selections']['omitted_zero_pending'], 1)
        self.assertEqual(result['events']['selections']['classifications'],
                         dict(tracked_ready=3, tracked_not_ready=1, scope_external=1, unknown=0))
        for value, options in ((preparation, selection_options), (selections, prepared_options),
                (selections, dict(selection_options, expected_preparation=False)),
                (selections, dict(selection_options, expected_selections=1))):
            with self.assertRaises(ValueError):
                delivery_capture.validate_issue_depth(value, job, **options)
        for record, column, value in ((0, 0, 3), (0, 1, 0), (0, 1, True), (0, 1, 2**64),
                (0, 2, width), (0, 3, 77), (0, 4, 0), (0, 4, 3), (0, 4, 4),
                (1, 1, 2), (1, 2, width+1), (2, 3, 1), (2, 4, 0), (3, 1, 5),
                (4, 2, width+9), (4, 4, True)):
            changed = deepcopy(selections)
            changed['selections']['records'][record][column] = value
            with self.assertRaises(ValueError):
                delivery_capture.validate_issue_depth(changed, job, **selection_options)
        for key, value in (('capacity', 524287), ('complete', False), ('records', []),
                ('fields', list(reversed(delivery_capture.SELECTION_FIELDS))),
                ('classifications', list(reversed(delivery_capture.SELECTION_CLASSES)))):
            changed = deepcopy(selections)
            changed['selections'][key] = value
            with self.assertRaises(ValueError):
                delivery_capture.validate_issue_depth(changed, job, **selection_options)
        for change in ('missing', 'duplicate', 'order', 'slot_total', 'slot_bool', 'slot_overflow',
                       'extra_field', 'short_row', 'positive_count', 'first_generation', 'bucket'):
            changed = deepcopy(selections)
            records = changed['selections']['records']
            if change == 'missing':
                records.pop()
            elif change == 'duplicate':
                records.append(records[-1])
            elif change == 'order':
                records[0], records[1] = records[1], records[0]
            elif change.startswith('slot_'):
                changed['slots'][0]['observation_generation'] = {
                    'slot_total': 5, 'slot_bool': True, 'slot_overflow': 2**64}[change]
            elif change == 'extra_field':
                changed['selections']['unexpected'] = 1
            elif change == 'short_row':
                records[0].pop()
            elif change == 'positive_count':
                changed['events']['records'][0][11] = 3
            elif change == 'first_generation':
                changed['events']['records'][0][14] = 4
            else:
                changed['buckets'][0]['other_pending'] = 0
                changed['buckets'][1].update(other_pending=1, observations=1)
                changed['buckets'][0]['observations'] -= 1
            with self.assertRaises(ValueError):
                delivery_capture.validate_issue_depth(changed, job, **selection_options)
        cross_slot = deepcopy(selections)
        cross_slot['events']['records'][0][1] = 77
        for record in cross_slot['selections']['records']:
            if record[0] == 1:
                record[3] = 77
        delivery_capture.validate_issue_depth(cross_slot, job, **selection_options)
        for value, options in ((generations, prepared_options), (preparation, selected),
                               (preparation, dict(expected_events=True, expected_preparation=True)),
                               (preparation, dict(selected, expected_preparation=1))):
            with self.assertRaises(ValueError):
                delivery_capture.validate_issue_depth(value, job, **options)
        for record, column, value in ((0, 15, True), (0, 15, -1), (0, 15, 2),
                                      (0, 16, True), (0, 16, 0), (0, 16, width),
                                      (0, 16, width+3), (0, 16, 2**64), (1, 16, width+2)):
            changed = deepcopy(preparation)
            changed['events']['records'][record][column] = value
            with self.assertRaises(ValueError):
                delivery_capture.validate_issue_depth(changed, job, **prepared_options)
        changed = deepcopy(preparation)
        changed['events']['records'][1][15:] = [1, width+2]
        with self.assertRaises(ValueError):
            delivery_capture.validate_issue_depth(changed, job, **prepared_options)
        for fields in (list(delivery_capture.GENERATION_FIELDS), list(reversed(delivery_capture.PREPARATION_FIELDS))):
            changed = deepcopy(preparation)
            changed['events']['fields'] = fields
            with self.assertRaises(ValueError):
                delivery_capture.validate_issue_depth(changed, job, **prepared_options)
        for record in (preparation['events']['records'][0][:-1], preparation['events']['records'][0]+[0]):
            changed = deepcopy(preparation)
            changed['events']['records'][0] = record
            with self.assertRaises(ValueError):
                delivery_capture.validate_issue_depth(changed, job, **prepared_options)
        for value, options in ((with_events, selected), (generations, dict(expected_events=True)),
                                (generations, dict(expected_generations=True))):
            with self.assertRaises(ValueError):
                delivery_capture.validate_issue_depth(value, job, **options)
        for column, value in ((13, -1), (13, True), (13, 101), (14, 0), (14, 100), (14, 2**64)):
            changed = deepcopy(generations)
            changed['events']['records'][0][column] = value
            with self.assertRaises(ValueError):
                delivery_capture.validate_issue_depth(changed, job, **selected)
        for fields in (list(delivery_capture.EVENT_FIELDS), list(reversed(delivery_capture.GENERATION_FIELDS))):
            changed = deepcopy(generations)
            changed['events']['fields'] = fields
            with self.assertRaises(ValueError):
                delivery_capture.validate_issue_depth(changed, job, **selected)
        for record in (generations['events']['records'][0][:-1], generations['events']['records'][0]+[1]):
            changed = deepcopy(generations)
            changed['events']['records'][0] = record
            with self.assertRaises(ValueError):
                delivery_capture.validate_issue_depth(changed, job, **selected)
        for value, expected in ((base, True), (with_events, False)):
            with self.assertRaises(ValueError):
                delivery_capture.validate_issue_depth(value, job, expected_events=expected)
        for key, value in (('capacity', 524287), ('complete', False), ('ready_duration', True),
                           ('gpu_timestamps', True), ('fields', list(reversed(delivery_capture.EVENT_FIELDS))),
                           ('records', with_events['events']['records'][:1])):
            changed = deepcopy(with_events)
            changed['events'][key] = value
            with self.assertRaises(ValueError):
                delivery_capture.validate_issue_depth(changed, job, expected_events=True)
        for column, value in ((0, 3), (1, 0), (2, 30), (3, width-1), (4, width+4),
                               (6, width+2), (9, 2*width+1), (10, 2), (11, 0), (12, 3),
                               (1, True), (10, 2**64)):
            changed = deepcopy(with_events)
            changed['events']['records'][0][column] = value
            with self.assertRaises(ValueError):
                delivery_capture.validate_issue_depth(changed, job, expected_events=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'issue-depth.json'
            path.write_text(json.dumps(with_events))
            path.chmod(0o600)
            audit = Path(directory) / 'server.log'
            audit.write_text('independent audit fixture')
            audited_job = dict(job, artifact_identities={'server.log': delivery_capture.digest(audit)})
            with patch.object(delivery_capture, 'issue_event_census', return_value={(1, 1, 11), (2, 1, 42)}):
                receipt = delivery_capture.issue_depth_receipt(path, audited_job, expected_events=True)
                self.assertEqual(receipt['summary']['events']['independent_commands'], 2)
                path.write_text(json.dumps(generations))
                receipt = delivery_capture.issue_depth_receipt(path, audited_job, **selected)
                self.assertEqual(receipt['summary']['events']['independent_commands'], 2)
                path.write_text(json.dumps(preparation))
                receipt = delivery_capture.issue_depth_receipt(path, audited_job, **prepared_options)
                self.assertEqual(receipt['summary']['events']['preparation_boundary']['training_deferred_commands'], 1)
                unexercised = deepcopy(preparation)
                unexercised['events']['records'][0][15:] = [0, 0]
                self.assertEqual(delivery_capture.validate_issue_depth(unexercised, job, **prepared_options)
                                 ['events']['preparation_boundary']['deferred_commands'], 0)
                path.write_text(json.dumps(unexercised))
                with self.assertRaisesRegex(ValueError, 'did not exercise'):
                    delivery_capture.issue_depth_receipt(path, audited_job, **prepared_options)
                path.write_text(json.dumps(selections))
                receipt = delivery_capture.issue_depth_receipt(path, audited_job, **selection_options)
                self.assertEqual(receipt['summary']['events']['selections']['training_waiting_commands'], 1)
                no_wait = deepcopy(selections)
                no_wait['events']['records'][0][13] = 2
                no_wait['selections']['records'].pop(0)
                no_wait['buckets'][0].update(not_ready_compute=0, no_selected_command=2)
                delivery_capture.validate_issue_depth(no_wait, job, **selection_options)
                path.write_text(json.dumps(no_wait))
                with self.assertRaisesRegex(ValueError, 'earlier selection observations'):
                    delivery_capture.issue_depth_receipt(path, audited_job, **selection_options)
                path.write_text(json.dumps(with_events))
            with patch.object(delivery_capture, 'issue_event_census', return_value={(1, 1, 11)}):
                with self.assertRaises(ValueError):
                    delivery_capture.issue_depth_receipt(path, audited_job, expected_events=True)
            with patch.object(delivery_capture, 'issue_event_census', return_value={(1, 1, 11), (2, 1, 30)}):
                with self.assertRaises(ValueError):
                    delivery_capture.issue_depth_receipt(path, audited_job, expected_events=True)
            path.chmod(0o644)
            with self.assertRaises(ValueError):
                delivery_capture.issue_depth_receipt(path, audited_job, expected_events=True)
        empty_cohort = deepcopy(base)
        empty_cohort['slots'][0].update(admitted=0, submitted=0, retired=0)
        empty_cohort['opcodes'] = [dict(opcode=42, admitted=1)]
        empty_cohort['buckets'][0].update(admitted=1, submitted=1, retired=1)
        with self.assertRaises(ValueError):
            delivery_capture.validate_issue_depth(empty_cohort, job)
        changes = [('valid', False), ('failure', 1), ('unclosed', 1), ('pid', 98),
                   ('readiness_is_instantaneous', False), ('outstanding_is_gpu_busy', True),
                   ('bucket_ns', width+1), ('end_ns', 2*width+1)]
        for key, value in changes:
            with self.assertRaises(ValueError):
                delivery_capture.validate_issue_depth(dict(base, **{key: value}), job)
        for section, key, value in (('slots', 'connection', 8), ('slots', 'pid', 8),
                ('slots', 'start_ticks', 8), ('slots', 'retired', 0), ('opcodes', 'opcode', 70),
                ('opcodes', 'admitted', 2), ('buckets', 'covered_ns', width-1),
                ('buckets', 'observations', 5), ('buckets', 'unknown', 1),
                ('buckets', 'submitted', 1), ('buckets', 'issue_call_ns', 21)):
            changed = deepcopy(base)
            changed[section][0][key] = value
            with self.assertRaises(ValueError):
                delivery_capture.validate_issue_depth(changed, job)
        for section in ('slots', 'opcodes', 'buckets'):
            changed = deepcopy(base)
            changed[section].append(changed[section][0])
            with self.assertRaises(ValueError):
                delivery_capture.validate_issue_depth(changed, job)

    def test_issue_depth_output_is_explicit_fresh_and_scoped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'issue-depth.json'
            command = (['service'], ['pin'])
            argv, pins = delivery_capture.depth_service_command(command, path)
            self.assertEqual((argv, pins), (['service', delivery_capture.DEPTH_FLAG, str(path)], ['pin']))
            self.assertEqual(command, (['service'], ['pin']))
            selected, selected_pins = delivery_capture.depth_service_command(command, path, events=True)
            self.assertEqual(selected, [*argv, delivery_capture.EVENTS_FLAG])
            self.assertEqual(selected_pins, pins)
            selected, _ = delivery_capture.depth_service_command(command, path, events=True, selections=True)
            self.assertEqual(selected, [*argv, delivery_capture.EVENTS_FLAG, delivery_capture.SELECTIONS_FLAG])
            for options in (dict(selections=True), dict(events=True, selections=1)):
                with self.assertRaises(ValueError):
                    delivery_capture.depth_service_command(command, path, **options)
            with self.assertRaises(ValueError):
                delivery_capture.depth_service_command((['service', delivery_capture.SELECTIONS_FLAG], pins), path)
            with self.assertRaises(ValueError):
                delivery_capture.depth_service_command((["service", delivery_capture.EVENTS_FLAG], pins), path)
            with self.assertRaises(ValueError):
                delivery_capture.depth_service_command((argv, pins), path)
            with self.assertRaises(ValueError):
                delivery_capture.depth_service_command(command, Path('relative.json'))
            path.touch()
            with self.assertRaises(ValueError):
                delivery_capture.depth_service_command(command, path)

    def test_issue_depth_control_restores_bindings_and_preserves_job(self):
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory)
            (attempt / 'equal').mkdir()
            owner = Mock(return_value=(['service'], ['pin']))
            common = SimpleNamespace(service_command=owner)
            job = {}

            def original(*args, **kwargs):
                job['server_command'] = common.service_command()[0]
                return job

            with patch.object(delivery_capture, 'issue_depth_receipt', return_value={'valid': True}), \
                    patch.object(delivery_capture.lifecycle.candidate, 'save') as save:
                result = delivery_capture.depth_numerical_control(original, common, None, attempt, 'equal')
                self.assertIs(result, job)
                self.assertEqual(set(job), {'server_command'})
                save.assert_called_once_with(attempt / 'equal/issue-depth-receipt.json', {'valid': True})
            self.assertIs(common.service_command, owner)
            with patch.object(delivery_capture, 'ISSUE_EVENTS', True), \
                    patch.object(delivery_capture, 'issue_depth_receipt', return_value={'valid': True}) as receipt, \
                    patch.object(delivery_capture.lifecycle.candidate, 'save'):
                result = delivery_capture.depth_numerical_control(original, common, None, attempt, 'equal')
                self.assertEqual(result['server_command'].count(delivery_capture.EVENTS_FLAG), 1)
                receipt.assert_called_once_with(attempt / 'equal/issue-depth.json', job,
                                                expected_events=True, expected_generations=False,
                                                expected_preparation=False, expected_selections=False)
            with patch.object(delivery_capture, 'ISSUE_EVENTS', True), \
                    patch.object(delivery_capture, 'OBSERVATION_GENERATIONS', True), \
                    patch.object(delivery_capture, 'issue_depth_receipt', return_value={'valid': True}) as receipt, \
                    patch.object(delivery_capture.lifecycle.candidate, 'save'):
                delivery_capture.depth_numerical_control(original, common, None, attempt, 'equal')
                receipt.assert_called_once_with(attempt / 'equal/issue-depth.json', job,
                                                expected_events=True, expected_generations=True,
                                                expected_preparation=False, expected_selections=False)
            with patch.object(delivery_capture, 'ISSUE_EVENTS', True), \
                    patch.object(delivery_capture, 'OBSERVATION_GENERATIONS', True), \
                    patch.object(delivery_capture, 'PREPARATION_BOUNDARY', True), \
                    patch.object(delivery_capture, 'issue_depth_receipt', return_value={'valid': True}) as receipt, \
                    patch.object(delivery_capture.lifecycle.candidate, 'save'):
                delivery_capture.depth_numerical_control(original, common, None, attempt, 'equal')
                receipt.assert_called_once_with(attempt / 'equal/issue-depth.json', job,
                                                expected_events=True, expected_generations=True,
                                                expected_preparation=True, expected_selections=False)
            with patch.object(delivery_capture, 'ISSUE_EVENTS', True), \
                    patch.object(delivery_capture, 'OBSERVATION_GENERATIONS', True), \
                    patch.object(delivery_capture, 'PREPARATION_BOUNDARY', True), \
                    patch.object(delivery_capture, 'SELECTED_COMMANDS', True), \
                    patch.object(delivery_capture, 'issue_depth_receipt', return_value={'valid': True}) as receipt, \
                    patch.object(delivery_capture.lifecycle.candidate, 'save'):
                result = delivery_capture.depth_numerical_control(original, common, None, attempt, 'equal')
                self.assertEqual(result['server_command'].count(delivery_capture.SELECTIONS_FLAG), 1)
                receipt.assert_called_once_with(attempt / 'equal/issue-depth.json', job,
                                                expected_events=True, expected_generations=True,
                                                expected_preparation=True, expected_selections=True)
            self.assertIs(common.service_command, owner)
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                delivery_capture.depth_numerical_control(Mock(side_effect=RuntimeError('injected')),
                                                         common, None, attempt, 'equal')
            self.assertIs(common.service_command, owner)
            clean = Mock(return_value='native')
            self.assertEqual(delivery_capture.depth_numerical_control(clean, common, None, attempt, 'native'),
                             'native')
            self.assertIs(common.service_command, owner)

    def test_issue_event_independent_census_rejects_missing_and_wrong_lifecycles(self):
        content = ('Native admission audit: session=1, sequence=1, accepted=true, operation=11\n'
                   'Native issue audit: session=1, sequence=1, status=0\n'
                   'Native execution audit: session=1, issued=1, retained=0, retired=1, status=0\n')
        source = dict(event=1, connection=1, role=0, pid=11, start=21)
        begin = dict(event=1, pid=99)
        semantic = SimpleNamespace(is_complete=True, family=lambda family:
                                   [SimpleNamespace(value=(begin if family == 2 else source).__getitem__)])
        job = dict(server_identity=dict(pid=99), effective_selection=dict(low=[1]),
                   application_identities=dict(low=dict(pid=11, start=21)))
        with patch('execution_priority_audit._parse_audit_model',
                   return_value=({(1, 1): (11, 0)}, [], [], semantic)):
            self.assertEqual(delivery_capture.issue_event_census(content, job), {(1, 1, 11)})
            for changed in (content.split('Native execution audit')[0],
                            content + content.splitlines(True)[-1],
                            content.replace('retired=1', 'retired=0'),
                            content.replace('status=0', 'status=1'),
                            content.replace('issued=1', 'issued=2')):
                with self.assertRaises(ValueError):
                    delivery_capture.issue_event_census(changed, job)
            for key in ('pid', 'start', 'role'):
                source[key] += 1
                with self.assertRaises(ValueError):
                    delivery_capture.issue_event_census(content, job)
                source[key] -= 1

    def test_delivery_split_handles_read_before_return_and_unresolved_delivery(self):
        for returned in range(1, 9):
            for first_read in range(1, 6):
                for frame in range(first_read, 7):
                    for flushed in (True, False):
                        parts = delivery_capture.split_delivery(1, returned, first_read, frame, 7, 8, flushed)
                        self.assertEqual(sum(right-left for left, right, _ in parts), 7)
                        self.assertEqual(parts[0][0], 1)
                        self.assertEqual(parts[-1][1], 8)
                        self.assertTrue(all(a[1] == b[0] for a, b in zip(parts, parts[1:])))
                        actual = [(left, right, 'pre_frame' if phase in delivery_capture.DELIVERY_PHASES else phase)
                                  for left, right, phase in parts]
                        parent = intake_capture.split_outbound(1, returned, frame, 7, 8)
                        actual_buckets, missing = intake_capture.partition([(1, 8)], actual)
                        self.assertEqual((actual_buckets, missing), intake_capture.partition([(1, 8)], parent))
        self.assertEqual(delivery_capture.split_delivery(1, 5, 3, 6, 7, 8, True),
                         [(1, 5, 'during_submit'), (5, 6, 'flushed_read_to_frame'),
                          (6, 7, 'frame_to_owner'), (7, 8, 'owner_to_retained')])
        for args in ((1, 2, 0, 4, 5, 6, True), (1, 2, 5, 4, 5, 6, True), (1, 2, 3, 4, 5, 6, 1)):
            with self.assertRaises(ValueError):
                delivery_capture.split_delivery(*args)

    def test_delivery_mixed_overlap_keeps_parent_denominator(self):
        phase = 'flushed_before_first_read'
        buckets = {(((1, phase), 2),): 60,
                   (((1, phase), 1), ((2, 'frame_to_owner'), 1)): 30,
                   (((1, 'delivery_unresolved'), 1), ((2, 'flushed_read_to_frame'), 1)): 10,
                   (((1, 'during_submit'), 1),): 7}
        result = delivery_capture.summarize(buckets)
        self.assertEqual(result['pre_frame_union_ns'], 100)
        self.assertEqual(result['mixed_pre_frame_overlap_ns'], 40)
        self.assertEqual(result['next_boundary'], phase)
        buckets[(((1, 'delivery_unresolved'), 1),)] = 1
        self.assertEqual(delivery_capture.summarize(buckets)['next_boundary'], 'inconclusive')

    def test_delivery_ledger_requires_complete_exact_boolean_join(self):
        base = dict(schema=6, diagnostic_only=True, valid=True, pid=7, begin_ns=1, end_ns=100,
                    fields=delivery_capture.lifecycle.frontend.FIELDS, capacity=10,
                    records=[[1, 1, 7, 0, 2, 3, 13, 10, 11, 12, 13, 14, 15, 1, 2, 3]],
                    submit_delivery=dict(schema=1, fields=['serial', 'session', 'wire', 'fully_flushed'],
                                         records=[[1, 2, 3, True]]))
        self.assertEqual(delivery_capture.validate_delivery(base), {(2, 3): True})
        base['submit_delivery']['records'][0][3] = False
        self.assertEqual(delivery_capture.validate_delivery(base), {(2, 3): False})
        for records in ([], [[1, 2, 3, 1]], [[2, 2, 3, True]], [[1, 2, 4, True]],
                        [[1, 2, 3, True], [1, 2, 3, False]]):
            changed = deepcopy(base)
            changed['submit_delivery']['records'] = records
            with self.assertRaises(ValueError):
                delivery_capture.validate_delivery(changed)
        service = dict(slots=[[2, 7, 9]], lifecycle=dict(capacity=2, events=[[30, 0, 0, 3, 0, 13]]),
                       intake=dict(schema=2, capacity=2, fields=['frame_received_ns', 'owner_enter_ns',
                           'retained_ns', 'slot', 'wire', 'opcode', 'first_read_ns'], records=[[20, 25, 30, 0, 3, 13, 10]]))
        self.assertEqual(delivery_capture.validate_intake(service), 1)
        for value in (0, 21, True):
            changed = deepcopy(service)
            changed['intake']['records'][0][6] = value
            with self.assertRaises(ValueError):
                delivery_capture.validate_intake(changed)

    def test_delivery_native_and_numerical_environments_are_observer_free(self):
        workflow = delivery_capture.lifecycle
        environment = dict(LD_PRELOAD=str(workflow.PRELOAD), GLIMMER_EXECUTION_HOST_RANGE_SEEK='1')
        job = dict(mode='equal', application_environment=environment, server_command=[str(workflow.SERVER)])
        delivery_capture.validate_environment(job)
        observed = deepcopy(job)
        observed['application_environment'][workflow.OUTPUT_ENV] = '/capture/equal'
        observed['server_command'] += ['--host-stage-output', '/capture/equal/host-stages.json']
        delivery_capture.validate_environment(observed, Path('/capture/equal'))
        for value in (observed, dict(job, application_environments={'low': {**environment, workflow.OUTPUT_ENV: '/bad'}})):
            with self.assertRaises(ValueError):
                delivery_capture.validate_environment(value)
        native = dict(mode='native', application_environment={})
        delivery_capture.validate_environment(native)
        for key in ('LD_PRELOAD', 'LD_AUDIT', 'GLIMMER_EXECUTION_HOST_RANGE_SEEK', workflow.OUTPUT_ENV):
            with self.assertRaises(ValueError):
                delivery_capture.validate_environment(dict(native, application_environment={key: '1'}))
        for enabled in (False, True):
            with patch.object(delivery_capture, 'HELPER_PLANS', enabled):
                selected = deepcopy(job)
                selected['server_command'], pins = delivery_capture.helper_service_command(
                    (selected['server_command'], ['pin']), enabled)
                self.assertEqual(pins, ['pin'])
                delivery_capture.validate_environment(selected)
                selected['server_command'].append(delivery_capture.HELPER_FLAG)
                with self.assertRaises(ValueError):
                    delivery_capture.validate_environment(selected)
                with self.assertRaises(ValueError):
                    delivery_capture.helper_service_command((selected['server_command'], pins), enabled)
        with self.assertRaises(ValueError):
            delivery_capture.validate_environment(dict(native, server_command=['unexpected']))
        with patch.object(delivery_capture, 'ISSUE_DEPTH', True):
            with self.assertRaises(ValueError):
                delivery_capture.validate_environment(job)
            enabled = dict(job, server_command=[*job['server_command'], delivery_capture.DEPTH_FLAG, '/fresh.json'])
            delivery_capture.validate_environment(enabled)
            delivery_capture.validate_environment(native)
            with patch.object(delivery_capture, 'ISSUE_EVENTS', True):
                with self.assertRaises(ValueError):
                    delivery_capture.validate_environment(enabled)
                events = dict(enabled, server_command=[*enabled['server_command'], delivery_capture.EVENTS_FLAG])
                delivery_capture.validate_environment(events)
                delivery_capture.validate_environment(native)
                with self.assertRaises(ValueError):
                    delivery_capture.validate_environment(dict(events, server_command=[*events['server_command'],
                                                                                       delivery_capture.EVENTS_FLAG]))
        with self.assertRaises(ValueError):
            delivery_capture.validate_environment(enabled)

    def test_helper_comparison_preserves_fixed_denominators_and_drift_gate(self):
        off = dict(helper_plans=False, plan_calls_per_step=6, plan_slot_inclusive_cpu_ns_per_step=100,
                   supply_gap_fraction=.4, completions=dict(native=100, equal=21), helper_skip_rate=0)
        on = dict(helper_plans=True, plan_calls_per_step=2, plan_slot_inclusive_cpu_ns_per_step=50,
                  supply_gap_fraction=.3, completions=dict(native=120, equal=20), helper_skip_rate=.99)
        self.assertTrue(delivery_capture.compare_helper_plans(off, on)['all_gates_passed'])
        on['completions']['native'] = 121
        self.assertFalse(delivery_capture.compare_helper_plans(off, on)['gates']['native_control_drift'])
        on['completions']['equal'] = 19
        self.assertFalse(delivery_capture.compare_helper_plans(off, on)['gates']['equal_completions'])
        with self.assertRaises(ValueError):
            delivery_capture.compare_helper_plans(on, off)

    def test_delivery_analysis_rejects_changed_or_missing_direct_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'request-analysis.json'
            path.write_text('{"value": 1}')
            inputs = delivery_capture.BoundInputs()
            self.assertEqual(inputs.read(path), {'value': 1})
            inputs.verify()
            path.write_text('{"value": 2}')
            with self.assertRaises(ValueError):
                inputs.verify()
            with self.assertRaises(ValueError):
                inputs.read(path)
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                inputs.verify()

    def test_intake_split_and_multiset_preserve_edges_and_overlap(self):
        for returned in range(1, 8):
            for frame in range(1, 6):
                for owner in range(frame, 7):
                    parts = intake_capture.split_outbound(1, returned, frame, owner, 7)
                    self.assertEqual(sum(right-left for left, right, _ in parts), 6)
                    self.assertEqual(parts[0][0], 1)
                    self.assertEqual(parts[-1][1], 7)
                    self.assertTrue(all(a[1] == b[0] for a, b in zip(parts, parts[1:])))
                    post = sum(right-left for left, right, phase in parts if phase != 'during_submit')
                    self.assertEqual(post, 7-min(returned, 7))
        for args in ((2, 3, 1, 4, 5), (1, 0, 2, 3, 4), (1, 2, 4, 3, 5)):
            with self.assertRaises(ValueError):
                intake_capture.split_outbound(*args)
        for same in (False, True):
            for first in range(16):
                for second in range(16):
                    keys = ((1,), (1,) if same else (2,))
                    spans = [(i, i+1, key) for mask, key in zip((first, second), keys)
                             for i in range(4) if mask & (1 << i)]
                    actual, uncovered = intake_capture.partition([(0, 4)], spans)
                    expected, empty = {}, 0
                    for i in range(4):
                        active = {}
                        for mask, key in zip((first, second), keys):
                            if mask & (1 << i):
                                active[key] = active.get(key, 0) + 1
                        if active:
                            key = tuple(sorted(active.items()))
                            expected[key] = expected.get(key, 0) + 1
                        else:
                            empty += 1
                    self.assertEqual(actual, expected)
                    self.assertEqual(uncovered, empty)
        mixed = {(((1, 'pre_frame'), 1), ((2, 'during_submit'), 1)): 40,
                 (((1, 'pre_frame'), 1),): 60}
        result = intake_capture.summarize(mixed)
        self.assertEqual(result['post_submit_union_ns'], 100)
        self.assertEqual(result['mixed_post_overlap_ns'], 40)
        self.assertEqual(result['next_boundary'], 'pre_frame')
        mixed[(((1, 'frame_to_owner'), 1),)] = 1
        self.assertEqual(intake_capture.summarize(mixed)['next_boundary'], 'inconclusive')

    def test_intake_requires_exact_complete_receive_join(self):
        document = dict(slots=[[1, 7, 9]], lifecycle=dict(capacity=10,
            events=[[30, 0, 0, 1, 0, 29], [31, 0, 4, 1, 1, 0], [50, 0, 0, 2, 0, 11]]),
            intake=dict(schema=1, capacity=10,
                fields=['frame_received_ns', 'owner_enter_ns', 'retained_ns', 'slot', 'wire', 'opcode'],
                records=[[10, 20, 30, 0, 1, 29], [25, 40, 50, 0, 2, 11]]))
        self.assertEqual(intake_capture.validate_intake(document), 2)
        for index, value in ((0, 0), (0, 21), (1, 31), (2, 29), (3, 1), (4, 0), (5, 30)):
            changed = deepcopy(document)
            changed['intake']['records'][0][index] = value
            with self.assertRaises(ValueError):
                intake_capture.validate_intake(changed)
        for change in ('missing', 'duplicate', 'capacity', 'duplicate_receive'):
            changed = deepcopy(document)
            if change == 'missing':
                changed['intake']['records'].pop()
            elif change == 'duplicate':
                changed['intake']['records'][1] = changed['intake']['records'][0]
            elif change == 'capacity':
                changed['intake']['capacity'] = 1
            else:
                changed['lifecycle']['events'].append(changed['lifecycle']['events'][0])
            with self.assertRaises(ValueError):
                intake_capture.validate_intake(changed)

    def test_seek_comparison_uses_fresh_pair_and_both_mechanism_clocks(self):
        before = dict(helpers=dict(mean_wall_ns=100, mean_cpu_ns=100, attempted=0, hits=0),
                      completions=dict(native=603, equal=20))
        after = dict(helpers=dict(mean_wall_ns=30, mean_cpu_ns=30, hit_fraction=1),
                     completions=dict(native=600, equal=30))
        for count, expected in ((30, 'diagnostic_application_response_supported'),
                                (24, 'inconclusive'), (23, 'seek_optimization_insufficient')):
            changed = dict(after, completions=dict(native=600, equal=count))
            self.assertEqual(host_seek.gates(before, changed)['decision'], expected)
        for changes in (dict(mean_cpu_ns=36), dict(mean_wall_ns=36), dict(hit_fraction=.94)):
            changed = dict(after, helpers=dict(after['helpers'], **changes))
            self.assertEqual(host_seek.gates(before, changed)['decision'], 'mechanism_gate_failed')
        invalid = dict(after, completions=dict(native=400, equal=60))
        self.assertEqual(host_seek.gates(before, invalid)['decision'], 'invalid_environment_or_denominator')

    def test_host_overhead_recomputes_order_counts_pairs_and_threshold(self):
        samples = [dict(index=index, detail=enabled, page_bytes=4096,
                        coarse_helper_ns=110 if enabled else 100)
                   for index, enabled in enumerate(host_capture.ORDER)]
        report = dict(order=host_capture.ORDER, samples=samples, ratios=[1.1]*4, median_ratio=1.1)
        self.assertEqual(host_capture.overhead_ratio(report), 1.1)
        for changes in (dict(median_ratio=1), dict(ratios=[1]*4), dict(order=[True, *host_capture.ORDER[1:]]),
                        dict(samples=samples[:7]), dict(samples=[dict(samples[0], detail=True), *samples[1:]])):
            with self.assertRaises(ValueError):
                host_capture.overhead_ratio(dict(report, **changes))
        excessive = [dict(sample, coarse_helper_ns=121 if sample['detail'] else 100) for sample in samples]
        with self.assertRaises(ValueError):
            host_capture.overhead_ratio(dict(report, samples=excessive, ratios=[1.21]*4, median_ratio=1.21))

    def test_host_stage_partition_and_bidirectional_failure_correspondence(self):
        base = dict(schema=5, diagnostic_only=True, valid=True, pid=7, begin_ns=1, end_ns=100,
                    base_count=2, phase_count=2, host_detail=True, host_count=2)
        lifecycle = host_stages.waits.lifecycle
        boundary = [0, 1, 7, 0, 0, 0, 0, 9, 9, 0, 0, 0, 31, 100, 110, 0]
        for schema in (5, 6):
            document = dict(base, schema=schema, fields=lifecycle.FIELDS,
                            records=[boundary], capacity=2097152)
            self.assertEqual(set(lifecycle.frontend(document, dict(pid=7), 2, 99)[0]), {1})
            with self.assertRaises(ValueError):
                lifecycle.frontend(dict(document, capacity=2097153), dict(pid=7), 2, 99)
        successful = [1, 10, 30, 100, 110, 2, 12, 4, 2, 1, 6, 2, 1, 1, 4096, 1]
        rejected = [2, 40, 41, 200, 201, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        phases = dict(base, host_fields=host_stages.DETAIL_FIELDS, host_capacity=131072,
                      phase_events=[[1, 5, 9, 31], [2, 8, 39, 42]],
                      host_details=[successful, rejected])
        rows = host_stages.details(base, phases)
        self.assertEqual(rows[1]['read_wall'], 12)
        self.assertFalse(rows[2]['success'])
        for index, value in ((0, 2), (1, 8), (2, 32), (6, 13), (10, 7), (15, 0)):
            changed = list(successful)
            changed[index] = value
            with self.assertRaises(ValueError):
                host_stages.details(base, dict(phases, host_details=[changed, rejected]))
        with self.assertRaises(ValueError):
            host_stages.details(dict(base, host_count=1), dict(phases, host_count=1, host_details=[successful]))
        with self.assertRaises(ValueError):
            host_stages.details(dict(base, host_detail=False), dict(phases, host_detail=False), enabled=False)
        base6 = dict(base, schema=6)
        phases6 = dict(phases, schema=6, host_fields=host_stages.SEEK_FIELDS,
                       host_details=[successful + [1, 1], rejected + [0, 0]])
        self.assertEqual(host_stages.details(base6, phases6)[1]['hint_hit'], 1)
        for flags in ([0, 1], [2, 1], [1, 2]):
            with self.assertRaises(ValueError):
                host_stages.details(base6, dict(phases6, host_details=[successful + flags, rejected + [0, 0]]))

    def test_host_stage_cohort_denominator_keeps_clipped_edge_calls(self):
        rows = {index: dict(coarse_begin=left, coarse_end=right, begin=left, end=right,
                           open_wall=0, read_wall=right-left, parse_wall=0, close_other_wall=0,
                           open_cpu=0, read_cpu=right-left, parse_cpu=0, close_other_cpu=0,
                           reads=1, bytes=100) for index, (left, right) in
                enumerate(((0, 4), (4, 6), (8, 12)))}
        result = host_stages.stage_summary(rows, {0, 1, 2}, 2, 10)
        self.assertEqual(result['count'], 1)
        self.assertEqual(result['clipped_all_helper_ns'], 6)
        self.assertEqual(result['cohort_coverage'], 1/3)
        self.assertEqual(result['dominant_stage'], 'inconclusive')
        result = host_stages.stage_summary(rows, {1}, 2, 10)
        self.assertEqual(result['dominant_stage'], 'read')

    def test_frontend_compiler_comparison_uses_fresh_control_and_dual_native_gates(self):
        for values, decision in (
                ((598, 20, 599, 40), 'frontend_compiler_contribution_supported'),
                ((598, 20, 599, 23), 'frontend_optimization_insufficient'),
                ((598, 20, 599, 24), 'inconclusive'),
                ((500, 20, 700, 40), 'invalid_environment_or_denominator'),
                ((598, 0, 599, 40), 'invalid_environment_or_denominator')):
            self.assertEqual(compiler.gates(*values)['decision'], decision)
        with self.assertRaises(ValueError):
            compiler.gates(598, -1, 599, 40)

    def test_frontend_compiler_keeps_other_flags_and_recording_disabled(self):
        compiler.compiler_delta(['-g', '-c'], ['-g', '-O2', '-c'], True)
        compiler.compiler_delta(['-g', '-c'], ['-g', '-c'], False)
        for old, new, cpp in ((['-O0'], ['-O0', '-O2'], True),
                              (['-g'], ['-g', '-O2', '-DNDEBUG'], True),
                              (['-g'], ['-g', '-O2'], False)):
            with self.assertRaises(ValueError):
                compiler.compiler_delta(old, new, cpp)
        environment = dict(LD_PRELOAD='/selected')
        job = dict(mode='equal', application_environment=environment,
                   application_commands={'low': [], 'high': []},
                   application_identities={'low': {}, 'high': {}},
                   application_environments={'low': environment, 'high': environment})
        compiler.environment_matches(job, Path('/selected'))
        with self.assertRaises(ValueError):
            compiler.environment_matches(job, Path('/wrong-profile'))
        environment[compiler.OUTPUT_ENV] = '/unexpected'
        with self.assertRaises(ValueError):
            compiler.environment_matches(job, Path('/selected'))

    def test_frontend_compiler_checks_actual_numerical_and_sustained_receipt_shapes(self):
        environment = dict(LD_PRELOAD='/selected')
        numerical = dict(mode='equal', application_environment=environment,
            application_commands={'low': [], 'high': []},
            application_identities={'low': {}, 'high': {}},
            application_environments={'low': dict(environment), 'high': dict(environment)})
        sustained = dict(mode='equal', application_environment=dict(environment),
            application_identities={'training': {}, 'inference': {}},
            application_commands={'training': ['train'], 'inference': ['infer']})
        compiler.environment_matches(numerical, Path('/selected'))
        compiler.environment_matches(sustained, Path('/selected'), sustained=True)
        with self.assertRaises(ValueError):
            compiler.environment_matches(sustained, Path('/selected'))
        with self.assertRaises(ValueError):
            compiler.environment_matches(numerical, Path('/selected'), sustained=True)
        missing = dict(sustained)
        missing.pop('application_environment')
        with self.assertRaises(KeyError):
            compiler.environment_matches(missing, Path('/selected'), sustained=True)
        for key, value in (('LD_PRELOAD', '/wrong-profile'), (compiler.OUTPUT_ENV, '/unexpected')):
            bad = dict(sustained, application_environment=dict(environment, **{key: value}))
            with self.assertRaises(ValueError):
                compiler.environment_matches(bad, Path('/selected'), sustained=True)
        numerical['application_environments']['high'] = dict(LD_PRELOAD='/wrong-profile')
        with self.assertRaises(ValueError):
            compiler.environment_matches(numerical, Path('/selected'))
        numerical['application_environments']['high'] = dict(environment, EXTRA='different')
        with self.assertRaises(ValueError):
            compiler.environment_matches(numerical, Path('/selected'))
        numerical['application_environments'].pop('high')
        with self.assertRaises(ValueError):
            compiler.environment_matches(numerical, Path('/selected'))
        sustained['application_commands'].pop('training')
        with self.assertRaises(ValueError):
            compiler.environment_matches(sustained, Path('/selected'), sustained=True)

    def test_frontend_compiler_capsule_survives_working_object_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            working, capsule = root / 'working', root / 'capsule'
            relative = Path('frontend/CMakeFiles/glimmer_execution_interceptor.dir')
            for build in (working, capsule):
                (build / relative).mkdir(parents=True)
            old = capsule / relative / 'driver_frontend.cc.o'
            current = working / relative / old.name
            old.write_bytes(b'O0')
            current.write_bytes(b'O2')
            certificate = dict(identities={str(old): dict(target=str(old), bytes=2,
                sha256=hashlib.sha256(b'O0').hexdigest())})
            link = capsule / relative / 'link.txt'
            link.write_text('/usr/bin/c++ ' + str(current))
            self.assertEqual(compiler.link_arguments(capsule, working),
                             [str(Path('/usr/bin/c++').resolve()), str(old)])
            compiler.verify_certificate(certificate)
            current.write_bytes(b'new working object')
            compiler.verify_certificate(certificate)
            old.write_bytes(b'changed capsule')
            with self.assertRaises(ValueError):
                compiler.verify_certificate(certificate)
        baseline, optimized = {Path('/common')}, {Path('/common'), Path('/optimized')}
        compiler.verify_dependencies(baseline, optimized, baseline, optimized, {Path('/optimized')})
        with self.assertRaises(ValueError):
            compiler.verify_dependencies(baseline, optimized | {Path('/unexpected')},
                baseline, optimized, {Path('/optimized')})

    def test_frontend_compiler_rejects_changed_plan_and_unexpected_recording(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            plan, document = directory / 'plan.json', directory / 'worker.json'
            plan.write_text('{}')
            configuration = dict(plan=str(plan), sha256=compiler.digest(plan), profile='o0')
            document.write_text(json.dumps(dict(frontend_compiler_configuration=configuration)))
            self.assertEqual(compiler.worker_configuration(document), configuration)
            plan.write_text('{"changed":true}')
            with self.assertRaises(ValueError):
                compiler.worker_configuration(document)
            compiler.require_no_frontend_output(directory)
            (directory / 'frontend-inference.json').write_text('{}')
            with self.assertRaises(ValueError):
                compiler.require_no_frontend_output(directory)

    def test_equal_only_selects_exactly_one_mode(self):
        self.assertEqual(capture.selected_modes(True), ('equal',))
        self.assertEqual(capture.selected_modes(False), ('native', 'equal', 'priority'))

    def test_equal_run_and_reparse_share_the_existing_native_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            attempt, native = Path(tmp), Path(tmp) / 'existing-native'
            common = SimpleNamespace(verify_process=Mock(), check_pending=Mock(),
                                     run_control=Mock(return_value={}))
            guard = SimpleNamespace(check=Mock(), observe=Mock())
            with patch.object(capture, 'normal_cleanup'):
                capture.guarded_control(common, None, attempt, 'equal', {}, guard, native)
            self.assertEqual(common.run_control.call_args.args[4], native)
            equal = attempt / 'equal'
            equal.mkdir()
            job = dict(artifact_identities={}, training_numerical={'train': 1},
                       inference_numerical={'infer': 1}, priority_audit_start=0,
                       effective_selection={}, actual_execution={'actual': 1}, selection={'sel': 1})
            (equal / 'attempt.json').write_text(json.dumps(job))
            tensor = SimpleNamespace(device=SimpleNamespace(type='cpu'), numel=lambda: 1)
            first = SimpleNamespace(device=SimpleNamespace(type='cpu'), numel=lambda: 217787509)
            tensors = {str(index): first if index == 0 else tensor for index in range(401)}
            torch = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False),
                                    load=Mock(return_value=tensors),
                                    testing=SimpleNamespace(assert_close=Mock()))
            training = SimpleNamespace(validate_result=Mock(return_value={'train': 1}))
            helpers = SimpleNamespace(training=training, export=attempt / 'export',
                                      validate_onnx=Mock(return_value={'infer': 1}))
            common.execution_selection = Mock(return_value=({'actual': 1}, {'sel': 1}))
            with patch.object(capture, 'normal_cleanup'), patch.dict(sys.modules, {'torch': torch}):
                result = capture.reparse(common, helpers, attempt, 'equal', job, 'GPU-test', native)
            self.assertEqual(result['elements'], 217787909)
            self.assertEqual(training.validate_result.call_args.args[1], native / 'training')
            self.assertEqual(helpers.validate_onnx.call_args.kwargs['reference'], native / 'inference')
            self.assertEqual(torch.load.call_args_list[1].args[0], native / 'training/snapshot.pt')

    def test_existing_native_requires_qualified_receipt_and_exact_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = {}
            for name in capture.EXISTING_NATIVE_SHA:
                path = root / 'native' / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(name.encode())
                expected[name] = hashlib.sha256(name.encode()).hexdigest()
            job = dict(status=capture.SUCCESS, mode='native', artifact_identities=expected)
            reparse = dict(tensors=401, elements=217787909, full_tensor_recomparison=True,
                           source_bound_complete_inference=True, first_command_handoff=False)
            worker = dict(status='passed_requires_review', job=job, reparse=reparse)
            report = dict(status='current_candidate_observed_controls_require_review',
                          input_identity_unchanged=True, gpu_identity_after='GPU-test',
                          jobs=[dict(mode='native', unit=dict(returncode=0, unit_removed=True), worker=worker)])
            manifest = dict(gpu='GPU-test', startup_reference_bundle='/sealed',
                            startup_reference_report_sha256='seal')
            for name, value in (('report.json', report), ('native-worker.json', worker),
                                ('native/attempt.json', job), ('manifest.json', manifest)):
                (root / name).write_text(json.dumps(value))
            with patch.object(capture, 'EXISTING_NATIVE_SHA', expected), patch.object(capture, 'normal_cleanup'):
                paths, found = capture.existing_native_reference(root, 'GPU-test', Path('/sealed'), 'seal')
                self.assertEqual(found, job)
                self.assertEqual(len(paths), 7)
                with self.assertRaisesRegex(ValueError, 'qualified'):
                    capture.existing_native_reference(root, 'GPU-other', Path('/sealed'), 'seal')
                (root / 'native/training/snapshot.pt').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'bytes changed'):
                    capture.existing_native_reference(root, 'GPU-test', Path('/sealed'), 'seal')

    def guard(self, attempt, clock=lambda: 0):
        with patch.object(capture, 'cgroup', return_value=f'/tasks/glimmer-bert-{attempt.name}-native.service'), \
             patch.object(capture, 'identity', return_value=71), \
             patch.object(capture.resource, 'getrlimit', return_value=(capture.FILE_MAX,) * 2), \
             patch.object(capture, 'memory_receipt', return_value=memory()):
            return capture.ResourceGuard(attempt, 'native', clock)

    def test_same_group_participant_and_rate_limited_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            now = [0.]
            guard = self.guard(Path(tmp), lambda: now[0])
            child = SimpleNamespace(pid=17, poll=lambda: None)
            with patch.object(capture, 'cgroup', return_value=guard.group), \
                 patch.object(capture, 'identity', return_value=71), \
                 patch.object(capture, 'memory_receipt', return_value=memory()):
                guard.observe(child, 71)
                guard.observe(child, 71)
                now[0] = .1
                guard.check()
                self.assertEqual(guard.receipt['resource_checks'], 1)
                now[0] = .21
                guard.check()
                self.assertEqual(guard.receipt['resource_checks'], 2)
                self.assertEqual(len(guard.receipt['participants']), 1)
                with patch.object(capture, 'cgroup', return_value='/foreign'):
                    with self.assertRaisesRegex(ValueError, 'inherit'):
                        guard.observe(child, 71)
                    with self.assertRaisesRegex(ValueError, 'escaped'):
                        guard.check(force=True)

    def test_wrong_unit_or_file_limit_rejected_before_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            attempt = Path(tmp)
            with patch.object(capture, 'cgroup', return_value='/uncapped'), self.assertRaises(ValueError):
                capture.ResourceGuard(attempt, 'native')
            group = f'/tasks/glimmer-bert-{attempt.name}-native.service'
            with patch.object(capture, 'cgroup', return_value=group), \
                 patch.object(capture.resource, 'getrlimit', return_value=(-1, -1)), self.assertRaises(ValueError):
                capture.ResourceGuard(attempt, 'native')

    def test_memory_event_and_total_output_cannot_qualify(self):
        with tempfile.TemporaryDirectory() as tmp:
            guard = self.guard(Path(tmp))
            for events in ('oom 1', 'max 1', 'oom_kill 1', ''):
                with self.subTest(events=events), patch.object(capture, 'memory_receipt',
                        return_value=memory(**{'memory.events': events})), self.assertRaises(ValueError):
                    guard.check(force=True)
            with patch.object(capture, 'memory_receipt', return_value=memory()), \
                 patch.object(capture, 'check_output', side_effect=ValueError('output cap')), \
                 self.assertRaisesRegex(ValueError, 'output cap'):
                guard.check(force=True)

    def test_existing_success_cannot_override_hook_failure_and_hooks_restore(self):
        common = SimpleNamespace(verify_process=Mock(), check_pending=Mock(), run_control=Mock(return_value={}))
        verify, pending = common.verify_process, common.check_pending
        guard = SimpleNamespace(check=Mock(side_effect=ValueError('resource failure')), observe=Mock())
        with self.assertRaisesRegex(ValueError, 'resource failure'):
            capture.guarded_control(common, None, Path('/attempt'), 'priority', {}, guard)
        self.assertIs(common.verify_process, verify)
        self.assertIs(common.check_pending, pending)
        self.assertEqual(common.run_control.call_args.kwargs, dict(priority_idle_us=2000, gemm_tile_side=32))

    def test_pending_failure_precedes_original_pending_and_preserves_restore(self):
        common = SimpleNamespace(verify_process=Mock(), check_pending=Mock())
        verify, pending = common.verify_process, common.check_pending
        common.run_control = lambda *args, **kwargs: common.check_pending('probe')
        guard = SimpleNamespace(check=Mock(side_effect=ValueError('memory event')), observe=Mock())
        with self.assertRaisesRegex(ValueError, 'memory event'):
            capture.guarded_control(common, None, Path('/attempt'), 'equal', {}, guard)
        pending.assert_not_called()
        self.assertIs(common.verify_process, verify)
        self.assertIs(common.check_pending, pending)

    def test_abnormal_or_signalled_cleanup_rejected(self):
        child = dict(exit_confirmed=True, signals=[], pid=11, start=22)
        job = dict(status=capture.SUCCESS, mode='native', final_returncodes=dict(low=0, high=0),
                   authenticated_child_cleanup=dict(low=dict(child), high=dict(child)),
                   application_identities=dict(low=dict(pid=11, start=22), high=dict(pid=11, start=22)),
                   socket_retained=False, reference_before={'a': 1}, reference_after={'a': 1})
        capture.normal_cleanup(job)
        job['authenticated_child_cleanup']['low']['signals'] = [15]
        with self.assertRaises(ValueError):
            capture.normal_cleanup(job)

    def test_failed_envelope_revokes_all_publication_claims(self):
        job = dict(pre_demand_compute_observed=True, high_release_trigger=dict(useful_compute_proof=True),
                   application_priority=dict(partial_handoff_component_verified=True, pre_demand_compute_observed=True))
        capture.invalidate_publication(job)
        self.assertTrue(job['outer_qualification_invalidated'])
        self.assertFalse(job['pre_demand_compute_observed'])
        self.assertFalse(job['high_release_trigger']['useful_compute_proof'])
        self.assertFalse(job['application_priority']['partial_handoff_component_verified'])
        self.assertFalse(job['application_priority']['pre_demand_compute_observed'])


class UnitTests(unittest.TestCase):
    def test_explicit_worker_script_and_failed_unit_cleanup_are_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            attempt = Path(tmp)
            replies = ['not-found', subprocess.TimeoutExpired('systemctl', 5)]
            with patch.object(capture, 'command', side_effect=replies), \
                 patch.object(capture.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as run, \
                 self.assertRaisesRegex(ValueError, 'bounded task failed'):
                capture.launch_unit(attempt, 'exclusive', worker_script=Path(__file__))
            self.assertIn(str(Path(__file__).resolve()), run.call_args.args[0])
            self.assertIn('--worker-mode', run.call_args.args[0])
            receipt = json.loads((attempt / 'exclusive-unit.json').read_text())
            self.assertFalse(receipt['unit_removed'])
            self.assertIn('TimeoutExpired', receipt['cleanup_error'])

    def test_worker_failure_stops_even_with_zero_unit_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            attempt = Path(tmp)
            (attempt / 'native-worker.json').write_text(json.dumps(dict(status='failed')))
            with patch.object(capture, 'command', return_value='not-found\n'), \
                 patch.object(capture.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as run, \
                 self.assertRaisesRegex(ValueError, 'worker/resource/audit'):
                capture.launch_unit(attempt, 'native')
            args = run.call_args.args[0]
            for option in ('MemoryMax=17179869184', 'MemorySwapMax=0', 'RuntimeMaxSec=800s',
                           'TimeoutStopSec=5s', 'LimitFSIZE=1073741824', 'KillMode=control-group'):
                self.assertIn('--property=' + option, args)
            self.assertEqual(run.call_args.kwargs['timeout'], 820)
            self.assertTrue((attempt / 'native-unit.json').is_file())

    def test_timeout_stops_only_exact_task_and_preserves_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            attempt = Path(tmp)
            unit = f'glimmer-bert-{attempt.name}-equal.service'
            replies = ['not-found', 'LoadState=loaded\nActiveState=failed', 'ActiveState=failed', '', 'not-found']
            # stop and reset-failed each return an empty successful output.
            replies.insert(2, '')
            with patch.object(capture, 'command', side_effect=replies) as command, \
                 patch.object(capture.subprocess, 'run', side_effect=subprocess.TimeoutExpired('unit', 820)), \
                 self.assertRaisesRegex(ValueError, 'bounded task failed'):
                capture.launch_unit(attempt, 'equal')
            calls = [row.args[0] for row in command.call_args_list]
            self.assertIn(['systemctl', '--user', 'stop', unit], calls)
            self.assertIn(['systemctl', '--user', 'reset-failed', unit], calls)
            receipt = json.loads((attempt / 'equal-unit.json').read_text())
            self.assertEqual(receipt['returncode'], -1)
            self.assertTrue(receipt['timeout'])

    def test_help_is_cold(self):
        result = subprocess.run([sys.executable, '-B', str(Path(capture.__file__)), '--help'],
                                capture_output=True, text=True, timeout=10, check=True)
        self.assertIn('--wait-for-idle', result.stdout)
        self.assertIn('--equal-only', result.stdout)
        self.assertNotIn('collect_application_block_colocation', sys.modules)


class PollBoundaryTests(unittest.TestCase):
    def test_poll_purpose_and_capture_hook_preserve_observer_off_paths(self):
        plan = dict(status='frozen_poll_boundary_application', application_purpose='accepted_launch_poll_boundary',
                    decision_role='inference', diagnostic_only=True,
                    full_device_qualified=False, performance_qualified=False,
                    analysis_source=dict(path=str(Path(poll_boundary.__file__).resolve()),
                                         sha256=delivery_capture.digest(Path(poll_boundary.__file__))))
        delivery_capture.validate_poll_plan(plan)
        for name, value in [('status', 'frozen_delivery_boundary_application'),
                            ('decision_role', 'training'), ('application_purpose', 'other'),
                            ('analysis_source', {}),
                            ('diagnostic_only', False), ('performance_qualified', True),
                            *[(key, True) for key in ('helper_plans', 'issue_depth', 'issue_events',
                                'observation_generations', 'preparation_boundary', 'selected_commands',
                                'async_intake_burst')]]:
            with self.assertRaises(ValueError):
                delivery_capture.validate_poll_plan(dict(plan, **{name: value}))
        original = Mock(return_value=(['service'], ['pin']))
        sustained = SimpleNamespace(owner_command=original)
        hooks = delivery_capture.poll_stage_hooks(lambda *_: nullcontext())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(sustained.owner_command()[0], ['service'])
            with hooks(None, sustained, None, root, 'native'):
                self.assertIs(sustained.owner_command, original)
            with self.assertRaisesRegex(RuntimeError, 'fixture'):
                with hooks(None, sustained, None, root, 'equal'):
                    self.assertEqual(sustained.owner_command(),
                        (['service', delivery_capture.POLL_FLAG, str(root / 'poll-journal.bin')], ['pin']))
                    raise RuntimeError('fixture')
            self.assertIs(sustained.owner_command, original)
            self.assertEqual(sustained.owner_command()[0], ['service'])

    def test_poll_analysis_requires_frozen_sources_before_derivation(self):
        workflow = delivery_capture.lifecycle
        modules = (poll_boundary, delivery_capture, delivery_capture.intake, workflow, workflow.frontend, workflow.base)
        identities = {str(Path(module.__file__).resolve()): delivery_capture.digest(Path(module.__file__))
                      for module in modules}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = root / 'plan.json'
            analyzer = str(Path(poll_boundary.__file__).resolve())
            plan.write_text(json.dumps(dict(status='frozen_poll_boundary_application', decision_role='inference',
                analysis_source=dict(path=analyzer, sha256=identities[analyzer]))))
            scope = dict(plan=dict(path=str(plan), sha256=delivery_capture.digest(plan)))
            seal = root / 'build-seal.json'
            seal.write_text(json.dumps(dict(plan_sha256=scope['plan']['sha256'], identities=identities)))
            inputs = delivery_capture.BoundInputs()
            poll_boundary.bind_analyzer(inputs, scope)
            inputs.verify()
            source = str(Path(poll_boundary.__file__).resolve())
            for changed in (dict(identities, **{source: 'stale'}),
                            {name: value for name, value in identities.items() if name != source}):
                seal.write_text(json.dumps(dict(plan_sha256=scope['plan']['sha256'], identities=changed)))
                with self.assertRaisesRegex(ValueError, 'plan/seal binding'):
                    poll_boundary.bind_analyzer(delivery_capture.BoundInputs(), scope)
            with self.assertRaisesRegex(ValueError, 'changed'):
                inputs.verify()
            # A jointly replaced source and sibling seal cannot replace the plan-owned identity.
            replacement = dict(identities, **{analyzer: 'replacement'})
            seal.write_text(json.dumps(dict(plan_sha256=scope['plan']['sha256'], identities=replacement)))
            original_digest = delivery_capture.digest
            with patch('run_rnn_v8_validation.digest', side_effect=lambda path:
                       'replacement' if str(Path(path).resolve()) == analyzer else original_digest(path)), \
                    self.assertRaisesRegex(ValueError, 'direct frozen'):
                poll_boundary.bind_analyzer(delivery_capture.BoundInputs(), scope)

    @staticmethod
    def rows():
        return [[poll_boundary.Cycle(i, base, base + 90, 1, i - 1, 1, i - 1, 1, 1)
                 for i, base in ((1, 100), (2, 200))],
                [poll_boundary.Poll(i, base + 10, base + 30, 1, 0)
                 for i, base in ((1, 100), (2, 200))],
                [poll_boundary.Peer(i, 7, 43, 101, 1, 1, 0, 0, 0, base + 40, 1)
                 for i, base in ((1, 100), (2, 200))],
                [poll_boundary.Read(i, 7, i, base + 50, base + 60, base + 65, 8, 0)
                 for i, base in ((1, 100), (2, 200))],
                [poll_boundary.Frame(i, 7, i, i, 11, base + 65, base + 70, 43, 101)
                 for i, base in ((1, 100), (2, 200))]]

    @staticmethod
    def encoded(rows):
        payload = b''.join(struct.pack(layout, *row) for (_, layout, _), group in
                           zip(poll_boundary.LAYOUTS, rows) for row in group)
        return struct.pack('<16Q', int.from_bytes(b'GLMPOLL1', 'little'), 1, 128, 42, 100,
                           *map(len, rows), len(payload), 0, 0, 0, 0, 0) + payload

    def write(self, path, rows=None, content=None):
        path.write_bytes(self.encoded(self.rows() if rows is None else rows) if content is None else content)
        path.chmod(0o600)

    def test_binary_header_size_owner_and_path_rejections(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ledger'
            content = self.encoded(self.rows())
            self.write(path, content=content)
            with poll_boundary.Journal(path, dict(pid=42, start=100), application=True) as journal:
                self.assertEqual(tuple(journal.counts.values()), (2, 2, 2, 2, 2))
                self.assertEqual(journal.frames[1].wire, 2)
            for index, value in ((0, 0), (1, 2), (2, 127), (3, 0), (4, 0),
                                 (5, 0), (5, 2**64 - 1), (9, 2**64 - 1), (10, 1), (15, 1)):
                with self.subTest(field=index, value=value), self.assertRaises(ValueError):
                    changed = bytearray(content)
                    struct.pack_into('<Q', changed, index * 8, value)
                    self.write(path, content=changed)
                    poll_boundary.Journal(path)
            for changed in (content[:127], content[:-1], content + b'\0'):
                self.write(path, content=changed)
                with self.assertRaises(ValueError):
                    poll_boundary.Journal(path)
            self.write(path, content=content)
            with self.assertRaisesRegex(ValueError, 'identity'):
                poll_boundary.Journal(path, dict(pid=42, start=101))
            path.chmod(0o644)
            with self.assertRaisesRegex(ValueError, 'owned'):
                poll_boundary.Journal(path)
            path.chmod(0o600)
            alias = path.with_name('alias')
            alias.symlink_to(path)
            with self.assertRaises(OSError):
                poll_boundary.Journal(alias)
            alias.unlink()
            alias.hardlink_to(path)
            with self.assertRaisesRegex(ValueError, 'owned'):
                poll_boundary.Journal(path)

    def test_structure_rejects_unproven_associations(self):
        mutations = [(0, 1, dict(id=1)), (0, 1, dict(entry=180)), (0, 1, dict(peer_begin=0)),
                     (0, 1, dict(poll_count=2)), (1, 1, dict(cycle=1)),
                     (1, 1, dict(returned=209)), (1, 1, dict(error=4)),
                     (2, 1, dict(pid=44)), (2, 1, dict(flags=1)), (2, 1, dict(visit=229)),
                     (2, 1, dict(events=0)), (2, 1, dict(revents=0)), (2, 1, dict(alive=0)),
                     (3, 1, dict(token=0)), (3, 1, dict(token=3)), (3, 1, dict(connection=8)),
                     (3, 1, dict(cycle=1)), (3, 1, dict(returned=249)),
                     (3, 1, dict(first_read=259)), (3, 1, dict(token=1, first_read=0)),
                     (4, 1, dict(token=1)), (4, 1, dict(wire=1)), (4, 1, dict(pid=44)),
                     (4, 1, dict(first_read=264)), (4, 1, dict(received=261)),
                     (4, 1, dict(cycle=1)), (4, 0, dict(received=251))]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ledger'
            for group, index, fields in mutations:
                with self.subTest(group=group, fields=fields), self.assertRaises(ValueError):
                    rows = self.rows()
                    rows[group][index] = rows[group][index]._replace(**fields)
                    self.write(path, rows)
                    poll_boundary.Journal(path)
            rows = self.rows()
            rows[-1].pop()
            self.write(path, rows)
            with self.assertRaisesRegex(ValueError, 'dangling'):
                poll_boundary.Journal(path)

    def test_exact_disjoint_partition_and_clipped_boundaries(self):
        expected = [(180, 190, 'unresolved'), (190, 200, 'outside_poll_call'),
                    (200, 210, 'call_setup'), (210, 230, 'ppoll'),
                    (230, 240, 'poll_return_to_target_visit'),
                    (240, 250, 'target_visit_to_recv_entry'), (250, 260, 'recv_syscall'),
                    (260, 265, 'positive_return_to_first_read')]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ledger'
            self.write(path)
            with poll_boundary.Journal(path) as journal:
                frame = journal.frames[1]
                self.assertEqual(journal.partition(frame, 180), expected)
                for begin in (180, 190, 195, 200, 215, 230, 245, 255, 262, 265, 280):
                    clipped = [(max(a, begin), b, phase) for a, b, phase in expected if b > begin]
                    self.assertEqual(journal.partition(frame, begin), clipped)
                self.assertEqual(journal.partition(frame, 180, fully_flushed=False), [(180, 265, 'unresolved')])
                first = journal.partition(journal.frames[0], 90)
                self.assertEqual(first[0], (90, 100, 'unresolved'))
                self.assertEqual(sum(b - a for a, b, _ in first), 75)

    def test_retries_are_separate_and_failed_receives_are_unresolved(self):
        rows = self.rows()
        rows[0][1] = rows[0][1]._replace(poll_count=2)
        rows[1][1:] = [poll_boundary.Poll(2, 210, 215, -1, 4), poll_boundary.Poll(2, 220, 230, 1, 0)]
        rows[3].insert(1, poll_boundary.Read(2, 7, 0, 242, 245, 0, -1, 11))
        rows[3][2] = rows[3][2]._replace(token=3)
        rows[4][1] = rows[4][1]._replace(token=3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ledger'
            self.write(path, rows)
            with poll_boundary.Journal(path) as journal:
                phases = journal.partition(journal.frames[1], 200)
                self.assertIn((215, 220, 'unresolved'), phases)
                self.assertIn((242, 245, 'unresolved'), phases)
                sums = Counter()
                for a, b, phase in phases:
                    sums[phase] += b - a
                self.assertEqual(sums['ppoll'], 15)
                self.assertEqual(sums['target_visit_to_recv_entry'], 7)
                self.assertEqual(sums['unresolved'], 8)
                self.assertEqual(sum(sums.values()), 65)

    def test_partial_frame_uses_first_cycle_not_completion(self):
        rows = self.rows()
        rows[0].append(poll_boundary.Cycle(3, 300, 390, 1, 2, 1, 2, 1, 1))
        rows[1].append(poll_boundary.Poll(3, 310, 330, 1, 0))
        rows[2].append(poll_boundary.Peer(3, 7, 43, 101, 1, 1, 1, 0, 0, 340, 1))
        rows[3].append(poll_boundary.Read(3, 7, 2, 350, 360, 0, 24, 0))
        rows[4][1] = rows[4][1]._replace(cycle=3, received=370)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ledger'
            self.write(path, rows)
            with poll_boundary.Journal(path) as journal:
                phases = journal.partition(journal.frames[1], 200)
                self.assertEqual(sum(b - a for a, b, _ in phases), 65)
                self.assertIn((250, 260, 'recv_syscall'), phases)
                self.assertFalse(any(b > 265 for _, b, _ in phases))

    def test_excluded_missing_nonreadable_and_failed_cycles_remain_unresolved(self):
        for condition in ('ready', 'credit', 'missing', 'nonreadable', 'failed'):
            rows = self.rows()
            rows[3] = [rows[3][1]._replace(token=1)]
            rows[4] = [rows[4][1]._replace(token=1)]
            if condition == 'missing':
                rows[0][0] = rows[0][0]._replace(peer_count=0)
                rows[0][1] = rows[0][1]._replace(peer_begin=0)
                rows[2].pop(0)
            else:
                rows[2][0] = rows[2][0]._replace(revents=0, events=0 if condition in ('ready', 'credit') else 1,
                    flags={'ready': 1, 'credit': 2}.get(condition, 0))
                if condition == 'failed':
                    rows[0][0] = rows[0][0]._replace(success=0)
                    rows[1][0] = rows[1][0]._replace(result=-1, error=5)
                    rows[2][0] = rows[2][0]._replace(visit=0)
            with self.subTest(condition=condition), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'ledger'
                self.write(path, rows)
                with poll_boundary.Journal(path) as journal:
                    phases = journal.partition(journal.frames[0], 105)
                    before = [(a, b, phase) for a, b, phase in phases if a < 190]
                    self.assertEqual(before, [(105, 190 if condition in ('nonreadable', 'failed') else 200, 'unresolved')])

    def test_phase_intersections_preserve_multisets_and_fixed_selector(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ledger'
            self.write(path)
            with poll_boundary.Journal(path) as journal:
                row = dict(wire=2, send_end=180, reply=280, frame=journal.frames[1], fully_flushed=True)
                result = poll_boundary.summarize(journal, [row], [(180, 280)], (180, 280))
                self.assertEqual(result['accepted_to_first_read_gap_union_ns'], 85)
                self.assertEqual(result['launch_wait_gap_union_ns'], 100)
                self.assertEqual(sum(result['exclusive_phase_ns'].values()), 85)
                self.assertEqual(result['next_boundary'], 'mixed_or_inconclusive')
                result = poll_boundary.summarize(journal, [row], [(210, 230)], (180, 280))
                self.assertEqual(result['next_boundary'], 'ppoll')
                self.assertEqual(result['fraction_of_whole_gap'], 1)
                # A second active wire with unresolved evidence must stay mixed, not vote for ppoll.
                second = dict(row, wire=3, fully_flushed=False)
                result = poll_boundary.summarize(journal, [row, second], [(210, 230)], (180, 280))
                self.assertEqual(result['mixed_phase_ns'], 20)
                self.assertEqual(result['weighted_phase_ns']['unresolved'], 20)
                self.assertEqual(result['next_boundary'], 'mixed_or_inconclusive')

    def test_normal_poll_receipt_and_distinct_output_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger, log = root / 'poll-journal.bin', root / 'server.log'
            self.write(ledger)
            ready = ('[2026-11-01 00:00:00.000] [info] Execution service ready: pid=42, start=100; '
                     'routing=explicit_opt_in, application_qualified=false\n')
            receipt = ('Poll journal receipt: schema=1, pid=42, start=100, complete=1, '
                       'cycles=2, polls=2, peers=2, reads=2, frames=2\n')
            job = dict(mode='equal', server_command=['service', delivery_capture.POLL_FLAG, str(ledger)],
                       server_identity=dict(pid=42, start=100), server_returncode=0, socket_retained=False,
                       application_identities={role: dict(pid=pid, start=101) for role, pid in
                                               (('training', 43), ('inference', 44))},
                       final_returncodes=dict(training=0, inference=0, service=0))
            job['authenticated_child_cleanup'] = {role: dict(identity, exit_confirmed=True, signals=[])
                for role, identity in dict(job['application_identities'], service=job['server_identity']).items()}

            def parse(text=ready + receipt, changes=None):
                log.write_text(text)
                return delivery_capture.poll_exit_receipt(log, dict(job, artifact_identities={
                    'server.log': delivery_capture.digest(log)}, **(changes or {})))

            self.assertEqual(parse()['counts']['frames'], 2)
            for text in (ready, receipt, receipt + ready, ready * 2 + receipt, ready + receipt * 2,
                         ready + receipt.replace('pid=42', 'pid=43'), ready + receipt.replace('frames=2', 'frames=3'),
                         ready + receipt.replace('complete=1', 'complete=0'),
                         ready + receipt.replace('cycles=2', 'cycles=02')):
                with self.assertRaises(ValueError):
                    parse(text)
            for changes in (dict(server_returncode=1), dict(socket_retained=True), dict(mode='native'),
                            dict(final_returncodes=dict(training=0, inference=-15, service=0))):
                with self.assertRaises(ValueError):
                    parse(changes=changes)
            for role in job['authenticated_child_cleanup']:
                cleanup = deepcopy(job['authenticated_child_cleanup'])
                cleanup[role]['signals'] = [15]
                with self.assertRaises(ValueError):
                    parse(changes=dict(authenticated_child_cleanup=cleanup))
            argv = job['server_command']
            for flag in ('--host-stage-output', '--issue-depth-output', '--profile-output',
                         '--ptx-cache-output', '--pair-profile-output', '--socket', delivery_capture.POLL_FLAG):
                with self.assertRaisesRegex(ValueError, 'distinct|duplicate'):
                    delivery_capture.validate_poll_paths([*argv, flag, str(ledger)])
            alias = root / 'alias'
            alias.symlink_to(root, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'distinct'):
                delivery_capture.validate_poll_paths([*argv, '--socket', str(alias / ledger.name)])


class CheckpointSharingTests(unittest.TestCase):
    def test_capture_never_shares_and_has_its_own_space_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'plan.json').write_text('{}')
            entry = Path(sharing.wait.lifecycle.collector.__file__)
            launch = Mock(return_value=({}, {}))
            floor = sharing.FLOOR + sharing.CAPTURE_BUDGET + sharing.MARGIN
            with patch.object(sharing, 'share_checkpoint') as share, \
                 patch.object(sharing.shutil, 'disk_usage', return_value=SimpleNamespace(free=floor)):
                sharing.launch_with_retention(launch, root, 'equal', worker_script=entry)
                launch.assert_called_once_with(root, 'equal', worker_script=entry)
                share.assert_not_called()
                with self.assertRaisesRegex(ValueError, 'numerical qualification'):
                    sharing.launch_with_retention(launch, root, 'equal')
            launch.reset_mock()
            with patch.object(sharing.shutil, 'disk_usage', return_value=SimpleNamespace(free=floor - 1)), \
                 self.assertRaisesRegex(ValueError, 'reserved output'):
                sharing.launch_with_retention(launch, root, 'equal', worker_script=entry)
            launch.assert_not_called()

    def test_intervention_uses_contained_calls_and_frozen_gates(self):
        analysis = sharing.optimization
        spans = [dict(caller='get_blas_value', thread=7, success=True, begin=a, end=b)
                 for a, b in ((0, 4), (4, 6), (8, 12))]
        result = analysis.summary(spans, dict(pid=7), 2, 10)
        self.assertEqual(result, dict(count=1, sum_ns=2, mean_ns=2, window_union_ns=6))
        with self.assertRaisesRegex(ValueError, 'nonempty'):
            analysis.summary(spans, dict(pid=8), 2, 10)
        for native, equal, mean, decision in ((598, 32, 5, 'important_host_supply_limitation_supported'),
                (478, 32, 5, 'environment_mismatch_inconclusive'),
                (598, 32, 6, 'mechanism_gate_failed'), (598, 19, 5, 'scan_optimization_insufficient'),
                (598, 20, 5, 'inconclusive')):
            self.assertEqual(analysis.gates(dict(mean_ns=10), dict(mean_ns=mean),
                dict(native=native, equal=equal))['decision'], decision)

    def test_only_fresh_validated_identical_bytes_are_shared(self):
        for failure in (None, 'different', 'unqualified', 'already_linked'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source = root / 'canonical.pt'
                source.write_bytes(b'original')
                source.chmod(0o664)
                target = root / 'native/training/snapshot.pt'
                target.parent.mkdir(parents=True)
                target.write_bytes(b'different' if failure == 'different' else b'original')
                target.chmod(0o664)
                if failure == 'already_linked':
                    (root / 'another.pt').hardlink_to(target)
                checksum = hashlib.sha256(source.read_bytes()).hexdigest()
                original_inode, fresh_inode = source.stat().st_ino, target.stat().st_ino
                allocated = target.stat().st_blocks * 512
                unit = dict(unit_removed=True, returncode=0)
                worker = dict(status='passed_requires_review', reparse=dict(tensors=401,
                    elements=217787909, full_tensor_recomparison=True,
                    source_bound_complete_inference=True, first_command_handoff=False),
                    job=dict(authenticated_child_cleanup={},
                             artifact_identities={'training/snapshot.pt': checksum}))
                if failure == 'unqualified':
                    worker['reparse']['full_tensor_recomparison'] = False
                with patch.object(sharing, 'CANONICAL', {'native': (source, original_inode, 0o664, checksum)}), \
                     patch.object(sharing, 'SNAPSHOT_BYTES', 8), \
                     patch.object(sharing, 'FLOOR', 0), patch.object(sharing, 'MARGIN', 0), \
                     patch.object(capture, 'normal_cleanup'), \
                     patch.object(sharing.shutil, 'disk_usage', side_effect=[
                         SimpleNamespace(free=10**9), SimpleNamespace(free=10**9 + allocated)]):
                    if failure:
                        with self.assertRaises(ValueError):
                            sharing.share_checkpoint(root, 'native', unit, worker)
                        self.assertEqual(target.stat().st_ino, fresh_inode)
                        self.assertFalse((root / 'native-checkpoint-share-intent.json').exists())
                    else:
                        sharing.share_checkpoint(root, 'native', unit, worker)
                        self.assertEqual(source.stat().st_ino, target.stat().st_ino)
                        self.assertEqual(source.stat().st_nlink, 2)
                        self.assertEqual(source.stat().st_mode & 0o777, 0o664)
                        self.assertEqual(target.read_bytes(), b'original')
                        result = json.loads((root / 'native-checkpoint-share-result.json').read_text())
                        self.assertEqual(result['reclaimed_allocation'], allocated)
                        self.assertTrue(result['writable_alias'])

    def test_receipt_publication_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'receipt.json'
            sharing.publish(path, {'first': True})
            with self.assertRaises(FileExistsError):
                sharing.publish(path, {'first': False})
            self.assertEqual(json.loads(path.read_text()), {'first': True})
            self.assertEqual(list(path.parent.iterdir()), [path])


if __name__ == '__main__':
    unittest.main()
