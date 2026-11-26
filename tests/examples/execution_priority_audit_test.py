"""Deterministic typed priority proof positive/negative controls; no GPU calls."""
import pathlib
import sys
import unittest

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]/'scripts'))
from execution_priority_audit import _parse_audit, verify_priority
from semantic_ordering_fixture import SemanticTrace

def admission(session,sequence,operation=11):
    return f'Native admission audit: session={session}, sequence={sequence}, accepted=true, operation={operation}\n'

def event(name,session,sequence,completed=0,total=0,launches=0,status=0):
    return (f'Priority execution audit: event={name}, session={session}, priority={0 if session==1 else 100}, '
        f'sequence={sequence}, completed_blocks={completed}, total_blocks={total}, physical_launches={launches}, status={status}\n')

def fixture():
    prefix=admission(1,1,25)+event('low_issued',1,1)+event('low_logical_done',1,1)
    phase=(admission(1,3)+event('low_issued',1,3,0,65536,1)+event('retreat_requested',1,3,0,65536,1)+
        event('low_physical_stop',1,3,160,65536,1)+event('high_physical_done',2,0)+
        admission(2,3)+event('high_issued',2,3)+admission(2,4)+event('high_issued',2,4)+
        'Native kernel pipeline audit: session=2, first=3, second=4, both_pending=true\n'+
        event('high_physical_done',2,0)+event('low_resumed',1,3,160,65536,2)+
        event('low_logical_done',1,3,65536,65536,2)+admission(2,5,25)+event('high_issued',2,5))
    return prefix+phase,len(prefix)

def lt_fixture(preemptible=True):
    text, offset = fixture()
    if preemptible:
        return (text.replace(admission(1,3), admission(1,3,42))
                    .replace('65536', '4096'), offset)
    prefix = text[:offset]
    phase = (admission(1,3,42) + event('low_issued',1,3) +
             event('nonpreemptible_wait',1,3) + event('low_logical_done',1,3) +
             admission(2,3) + event('high_issued',2,3) +
             admission(2,4) + event('high_issued',2,4) +
             'Native kernel pipeline audit: session=2, first=3, second=4, both_pending=true\n' +
             event('high_physical_done',2,0))
    return prefix + phase, len(prefix)

class TypedPriority(unittest.TestCase):
    def test_generic_copy_is_memory_and_never_a_compute_contract(self):
        text, offset = fixture()
        prefix = text[:offset].replace('operation=25', 'operation=141')
        updated = prefix + text[offset:].replace('operation=25', 'operation=141')
        admissions, _, _ = _parse_audit(updated)
        self.assertEqual(admissions[(1,1)][0], 141)
        proof = verify_priority(updated, len(prefix))
        self.assertEqual(proof['low_kernel_commands'], [(1,3)])
        self.assertEqual(proof['high_kernel_commands'], [(2,3),(2,4)])
        self.assertEqual(proof['compute_operation_classes'],
                         {'persistent_capable': [11,30,31,32,42,56],
                          'original_only': [129,139,140]})
        for preemptible in (False, True):
            with self.subTest(preemptible=preemptible), self.assertRaises(ValueError):
                verify_priority(updated, len(prefix), preemptible=preemptible, low_operation=141)
        for sequence in (3, 4):
            changed = updated.replace(admission(2,sequence), admission(2,sequence,141))
            with self.subTest(sequence=sequence), self.assertRaises(ValueError):
                verify_priority(changed, len(prefix))

    def test_generic_memory_progress_cannot_hide_outside_compute_filter(self):
        text, offset = fixture()
        for name in ('low_issued', 'high_issued', 'nonpreemptible_wait', 'low_logical_done'):
            for progress in ((1,0,0), (0,1,0), (0,0,1)):
                invalid = admission(1,9,141) + event(name,1,9,*progress)
                for before in (True, False):
                    sample = invalid + text if before else text + invalid
                    boundary = offset + len(invalid) if before else offset
                    with self.subTest(name=name,progress=progress,before=before), \
                            self.assertRaisesRegex(ValueError, 'unified copy'):
                        verify_priority(sample,boundary)
        for name in ('retreat_requested', 'low_physical_stop', 'low_resumed'):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'unified copy'):
                _parse_audit(admission(1,9,141) + event(name,1,9))
        with self.assertRaises(ValueError):
            _parse_audit(admission(1,9,142) + event('low_issued',1,9))

    def test_pageable_helper_role_is_not_generic_copy_source_authority(self):
        trace = SemanticTrace(actor=False)
        trace.peer(8, role=1, effective=0)
        trace.append(admission(1,1,141) + event('low_issued',1,1) + event('low_logical_done',1,1))
        admissions, _, _ = _parse_audit(trace.finish())
        self.assertEqual(admissions[(1,1)][0], 141)
        bad = SemanticTrace(actor=False)
        bad.peer(8, role=1, effective=0)
        bad.append(admission(8,1,141))
        with self.assertRaisesRegex(ValueError, 'authenticated source binding'):
            _parse_audit(bad.finish())

    def test_typed_ordering_coexists_with_geam_without_becoming_compute(self):
        trace = SemanticTrace(actor=False)
        trace.append(event('low_bound',1,0) + admission(1,1,129) + event('low_issued',1,1))
        command = trace.ordering(1)
        trace.append(event('event_accepted',1,10))
        trace.ordering(2, command)
        trace.append(event('event_issued',1,10) + event('low_logical_done',1,1))
        trace.ordering(3, command)
        trace.append(event('event_retired',1,10))
        admissions, events, _ = _parse_audit(trace.finish())
        self.assertEqual(admissions[(1,10)][0], 48)
        self.assertEqual(sum(row['event'] == 'event_issued' for _, row in events), 1)
        self.assertFalse(any(row['total'] for _, row in events))

    def test_ordering_is_visible_but_cannot_supply_a_handoff_kernel(self):
        text, offset = fixture()
        trace = SemanticTrace(actor=False)
        trace.peer(2, peer=22, priority=100, kernel_mode=1, effective=0, retained=0)
        trace.append(text[:offset])
        boundary = trace.position
        trace.append(text[offset:])
        command = trace.ordering(1)
        trace.ordering(2, command)
        trace.ordering(3, command)
        proof = verify_priority(trace.finish(), boundary)
        self.assertEqual(proof['low_kernel_commands'], [(1,3)])
        self.assertEqual(proof['high_kernel_commands'], [(2,3),(2,4)])
        self.assertTrue(any(row['event'] == 'event_issued' for row in proof['transitions']))
        self.assertFalse(any(row['event'].startswith('event_')
                             for row in proof['kernel_transitions']))

    def test_legacy_priority_cannot_disagree_with_authenticated_semantic_peer(self):
        text, _ = fixture()
        for changed in (text.replace('session=1, priority=0,', 'session=1, priority=7,'),
                        text.replace('session=2, priority=100,', 'session=2, priority=101,')):
            trace = SemanticTrace(actor=False)
            trace.peer(2, peer=22, priority=100, kernel_mode=1, effective=0, retained=0)
            trace.append(changed)
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                _parse_audit(trace.finish())

    def test_ordering_annotation_requires_exact_typed_identity_and_zero_progress(self):
        with self.assertRaises(ValueError):
            _parse_audit(event('event_accepted',1,10))
        for annotation in (event('event_accepted',1,11),
                           event('event_accepted',1,10,total=1),
                           event('event_accepted',1,10) * 2):
            trace = SemanticTrace(actor=False)
            command = trace.ordering(1)
            trace.append(annotation)
            trace.ordering(2, command)
            trace.ordering(3, command)
            with self.subTest(annotation=annotation), self.assertRaises(ValueError):
                _parse_audit(trace.finish())

    def test_semantic_errors_before_measurement_are_not_filtered(self):
        text, offset = fixture()
        trace = SemanticTrace(actor=False)
        command = trace.ordering(1)
        trace.ordering(2, command, status=1)
        trace.ordering(3, command, status=1)
        prefix = trace.finish()
        with self.assertRaises(ValueError):
            verify_priority(prefix + text, len(prefix) + offset)
        with self.assertRaises(ValueError):
            verify_priority('Execution semantic audit: {"v":1}\n' + text, offset)

    def test_equal_priority_geam_low_can_overlap_persistent_high(self):
        text = (event('low_bound',1,0) + event('native_bound',2,0) +
                admission(1,1,129) + event('low_issued',1,1) +
                admission(2,1,11) + event('high_issued',2,1,0,64,1) +
                event('low_logical_done',1,1) + event('high_physical_done',2,0))
        # Equality is allowed; an audit cohort name cannot reverse actual rank.
        for low_priority, high_priority in ((0,0), (100,100), (100,0)):
            candidate = text.replace('priority=0,', f'priority={low_priority},')
            candidate = candidate.replace('session=2, priority=100,',
                                          f'session=2, priority={high_priority},')
            with self.subTest(low=low_priority, high=high_priority):
                admissions, events, _ = _parse_audit(candidate)
                self.assertEqual(admissions[(1,1)][0], 129)
                self.assertEqual(sum(e['event'] == 'low_logical_done' for _,e in events), 1)

    def test_equal_priority_geam_high_allows_low_issue_and_resume(self):
        text = (event('low_bound',1,0) + event('native_bound',2,0) +
                admission(2,1,129) + event('high_issued',2,1) +
                admission(1,1) + event('low_issued',1,1,0,64,1) +
                event('retreat_requested',1,1,0,64,1) +
                event('low_physical_stop',1,1,16,64,1) +
                event('low_resumed',1,1,16,64,2) +
                event('low_logical_done',1,1,64,64,2) + event('high_physical_done',2,0))
        # This generic parser validates frontiers, not the reason for a retreat.
        # Differentiated-priority qualification remains a separate strict gate.
        for low_priority, high_priority in ((0,0), (100,100), (100,0)):
            candidate = text.replace('priority=0,', f'priority={low_priority},')
            candidate = candidate.replace('session=2, priority=100,',
                                          f'session=2, priority={high_priority},')
            with self.subTest(low=low_priority, high=high_priority):
                _parse_audit(candidate)

    def test_equal_original_issues_in_one_session_remain_exactly_once(self):
        text = (event('low_bound',1,0) + event('native_bound',2,0) +
                admission(1,1,129) + event('low_issued',1,1) +
                admission(2,1) + event('high_issued',2,1) +
                event('low_logical_done',1,1) +
                admission(1,2,129) + event('low_issued',1,2) +
                event('low_logical_done',1,2) + event('high_physical_done',2,0))
        text = text.replace('priority=100,', 'priority=0,')
        _parse_audit(text)
        for name in ('low_issued', 'low_logical_done'):
            for sequence in (1,2):
                record = event(name,1,sequence)
                for candidate in (text.replace(record, ''), text + record):
                    with self.subTest(name=name, sequence=sequence), self.assertRaises(ValueError):
                        _parse_audit(candidate)

    def test_equal_frontier_does_not_hide_a_genuinely_higher_pending_issue(self):
        equal = admission(2,1) + event('high_issued',2,1).replace('priority=100,', 'priority=0,')
        for higher_operation, low_operation in ((129,11), (11,129)):
            higher = admission(3,1,higher_operation) + event('high_issued',3,1)
            low = admission(1,1,low_operation) + event('low_issued',1,1)
            done = event('low_logical_done',1,1)
            with self.subTest(higher=higher_operation), self.assertRaisesRegex(
                    ValueError, 'while high completion unproven'):
                _parse_audit(equal + higher + low + done)
            _parse_audit(equal + higher + event('high_physical_done',3,0) + low + done)
        paused = (admission(1,1) + event('low_issued',1,1,0,64,1) +
                  event('retreat_requested',1,1,0,64,1) +
                  event('low_physical_stop',1,1,16,64,1))
        higher = admission(3,1,129) + event('high_issued',3,1)
        resumed = event('low_resumed',1,1,16,64,2) + event('low_logical_done',1,1,64,64,2)
        with self.assertRaisesRegex(ValueError, 'low resume while high completion unproven'):
            _parse_audit(paused + equal + higher + resumed)
        _parse_audit(paused + equal + higher + event('high_physical_done',3,0) + resumed)
        changed = event('high_physical_done',3,0).replace('priority=100,', 'priority=0,')
        with self.assertRaisesRegex(ValueError, 'original-only session changed priority or role'):
            _parse_audit(paused + equal + higher + changed + resumed)

    def test_equal_frontier_does_not_hide_a_genuinely_lower_active_issue(self):
        equal = admission(3,1) + event('low_issued',3,1)
        for lower_operation, high_operation in ((129,11), (11,129)):
            lower = admission(1,1,lower_operation) + event('low_issued',1,1)
            high = admission(2,1,high_operation) + event('high_issued',2,1)
            done = event('low_logical_done',1,1)
            with self.subTest(lower=lower_operation), self.assertRaisesRegex(
                    ValueError, 'before proven low physical stop'):
                _parse_audit(equal + lower + high + done)
            _parse_audit(equal + lower + done + high + event('high_physical_done',2,0))

    def test_equal_rank_keeps_role_status_progress_and_qualification_checks(self):
        text = (event('low_bound',1,0) + event('native_bound',2,0) +
                admission(1,1,129) + event('low_issued',1,1) +
                admission(2,1) + event('high_issued',2,1) +
                event('low_logical_done',1,1) + event('high_physical_done',2,0))
        text = text.replace('priority=100,', 'priority=0,')
        variants = [text.replace(event('low_issued',1,1), event('high_issued',1,1)),
                    text.replace(event('low_bound',1,0), event('native_bound',1,0)),
                    text.replace(event('low_logical_done',1,1),
                                 event('low_logical_done',1,1).replace('priority=0,', 'priority=1,')),
                    text.replace(event('low_issued',1,1), event('low_issued',1,1,0,64,1))]
        for status in (-1, 1, '0junk'):
            variants.append(text.replace(event('low_issued',1,1), event('low_issued',1,1,status=status)))
        for index, candidate in enumerate(variants):
            with self.subTest(case=index), self.assertRaises(ValueError):
                _parse_audit(candidate)
        proof, offset = fixture()
        proof = proof.replace('priority=100,', 'priority=0,')
        proof += admission(2,9,129) + event('high_issued',2,9).replace('priority=100,', 'priority=0,')
        _parse_audit(proof)
        with self.assertRaisesRegex(ValueError, 'low/high role priorities are not ordered'):
            verify_priority(proof, offset)

    def test_strided_sgemm_requires_its_explicit_type_and_exact_geometry(self):
        text, offset = lt_fixture()
        text = text.replace(admission(1,3,42), admission(1,3,56))
        proof = verify_priority(text, offset, low_operation=56, total_blocks=4096)
        self.assertEqual(proof['low_kernel_commands'], [(1,3)])
        self.assertEqual(proof['compute_operation_classes'],
                         {'persistent_capable': [11,30,31,32,42,56], 'original_only': [129,139,140]})
        self.assertEqual(proof['original_only_compute_commands'], [])
        for kwargs in ({'low_operation':42,'total_blocks':4096},
                       {'low_operation':56,'total_blocks':144}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                verify_priority(text, offset, **kwargs)
        native, offset = lt_fixture(preemptible=False)
        native = native.replace(admission(1,3,42), admission(1,3,56))
        verify_priority(native, offset, preemptible=False, low_operation=56)
        with self.assertRaises(ValueError):
            verify_priority(native, offset, low_operation=56)

    def test_geam_native_control_exposes_original_only_class_without_preemption(self):
        text, offset = lt_fixture(preemptible=False)
        text = text.replace(admission(1,3,42), admission(1,3,129))
        proof = verify_priority(text, offset, preemptible=False, low_operation=129)
        self.assertEqual(proof['original_only_compute_commands'],
                         [dict(session=1, sequence=3, operation=129)])
        self.assertEqual(proof['compute_operation_classes']['original_only'], [129,139,140])
        with self.assertRaisesRegex(ValueError, 'original-only compute cannot qualify preemption'):
            verify_priority(text, offset, low_operation=129)
        persistent, boundary = lt_fixture()
        persistent = persistent.replace(admission(1,3,42), admission(1,3,129))
        with self.assertRaises(ValueError):
            verify_priority(persistent, boundary, preemptible=False, low_operation=129)

    def test_sgemv_and_sgemmex_remain_nonpreemptible_original_compute(self):
        for operation in (139, 140):
            text, offset = lt_fixture(preemptible=False)
            text = text.replace(admission(1,3,42), admission(1,3,operation))
            with self.subTest(operation=operation):
                proof = verify_priority(text, offset, preemptible=False, low_operation=operation)
                self.assertEqual(proof['original_only_compute_commands'],
                                 [dict(session=1, sequence=3, operation=operation)])
                with self.assertRaises(ValueError):
                    verify_priority(text, offset, low_operation=operation)
                for name in ('low_issued', 'nonpreemptible_wait', 'low_logical_done'):
                    invalid = text.replace(event(name,1,3), event(name,1,3,total=1))
                    with self.assertRaises(ValueError):
                        verify_priority(invalid, offset, preemptible=False, low_operation=operation)
                with self.assertRaises(ValueError):
                    verify_priority(text.replace(event('low_logical_done',1,3), ''), offset,
                                    preemptible=False, low_operation=operation)

    def test_geam_zero_progress_is_required_for_every_native_low_event(self):
        text, offset = lt_fixture(preemptible=False)
        text = text.replace(admission(1,3,42), admission(1,3,129))
        for name in ('low_issued', 'nonpreemptible_wait', 'low_logical_done'):
            for progress in ((1,0,0), (0,1,0), (0,0,1), (3,12,2)):
                invalid = text.replace(event(name,1,3), event(name,1,3,*progress))
                with self.subTest(name=name,progress=progress), self.assertRaises(ValueError):
                    verify_priority(invalid, offset, preemptible=False, low_operation=129)
        variants = [text.replace(event('low_logical_done',1,3), ''),
                    text + event('low_logical_done',1,3),
                    text.replace(event('low_issued',1,3), event('low_issued',1,3) * 2),
                    text.replace(event('nonpreemptible_wait',1,3), event('nonpreemptible_wait',1,3) * 2),
                    text.replace(admission(1,3,129), '') + admission(1,3,129),
                    text.replace(event('low_logical_done',1,3), '') + event('low_logical_done',1,3)]
        for index, invalid in enumerate(variants):
            with self.subTest(case=index), self.assertRaises(ValueError):
                verify_priority(invalid, offset, preemptible=False, low_operation=129)

    def test_geam_persistent_transitions_fail_even_outside_selected_phase(self):
        text, offset = fixture()
        for name in ('retreat_requested', 'low_physical_stop', 'low_resumed'):
            for progress in ((0,0,0), (1,12,1)):
                prefix = (admission(1,9,129) + event('low_issued',1,9) +
                          event(name,1,9,*progress) + event('low_logical_done',1,9))
                with self.subTest(name=name,progress=progress), self.assertRaises(ValueError):
                    verify_priority(prefix + text, len(prefix) + offset)

    def test_geam_high_counters_and_unproven_frontiers_cannot_be_filtered(self):
        text, offset = fixture()
        resume = event('low_resumed',1,3,160,65536,2)
        native = admission(2,9,129) + event('high_issued',2,9)
        valid = text.replace(resume, native + event('high_physical_done',2,0) + resume)
        proof = verify_priority(valid, offset)
        self.assertEqual(proof['original_only_compute_commands'],
                         [dict(session=2, sequence=9, operation=129)])
        variants = [text.replace(resume, native + resume),
                    text.replace(resume, native + event('high_physical_done',3,0) + resume),
                    text.replace(event('low_physical_stop',1,3,160,65536,1),
                                 native + event('high_physical_done',2,0) +
                                 event('low_physical_stop',1,3,160,65536,1)),
                    text.replace(admission(2,4), admission(2,4,129))]
        for progress in ((1,0,0), (0,1,0), (0,0,1), (0,12,1)):
            variants.append(valid.replace(event('high_issued',2,9), event('high_issued',2,9,*progress)))
            variants.append(valid.replace(native + event('high_physical_done',2,0),
                                          native + event('high_physical_done',2,0,*progress)))
        for index, invalid in enumerate(variants):
            with self.subTest(case=index), self.assertRaises(ValueError):
                verify_priority(invalid, offset)

    def test_unselected_geam_low_frontier_must_finish_before_any_high_issue(self):
        text, offset = fixture()
        native = admission(1,9,129) + event('low_issued',1,9) + event('nonpreemptible_wait',1,9)
        marker = admission(1,3)
        valid = text.replace(marker, native + event('low_logical_done',1,9) + marker)
        verify_priority(valid, offset)
        invalid = text.replace(marker, native + marker) + event('low_logical_done',1,9)
        with self.assertRaisesRegex(ValueError, 'before proven low physical stop'):
            verify_priority(invalid, offset)

    def test_new_types_do_not_relax_unknown_status_or_syntax_checks(self):
        text, offset = fixture()
        for operation in (55,57,128,130,'129junk','56.0'):
            with self.subTest(operation=operation), self.assertRaises(ValueError):
                verify_priority(text + admission(2,9,operation), offset)
        for status in (1,-1,'0junk'):
            invalid = text + admission(2,9,129) + event('high_issued',2,9,status=status)
            with self.subTest(status=status), self.assertRaises(ValueError):
                verify_priority(invalid, offset)

    def test_native_geam_session_role_and_priority_include_setup_and_wait_done(self):
        text, offset = lt_fixture(preemptible=False)
        text = text.replace(admission(1,3,42), admission(1,3,129))
        for name, sequence in (('low_issued',3), ('nonpreemptible_wait',3),
                               ('low_logical_done',3), ('low_issued',1), ('low_logical_done',1)):
            original = event(name,1,sequence)
            invalid = text.replace(original, original.replace('priority=0', 'priority=100'))
            with self.subTest(name=name,sequence=sequence), self.assertRaisesRegex(
                    ValueError, 'original-only session changed priority or role'):
                verify_priority(invalid, offset, preemptible=False, low_operation=129)
        for name in ('native_bound', 'high_physical_done'):
            prefix = event(name,1,0)
            with self.subTest(name=name), self.assertRaisesRegex(
                    ValueError, 'original-only session changed priority or role'):
                verify_priority(prefix + text, len(prefix) + offset,
                                preemptible=False, low_operation=129)
        prefix = event('low_bound',1,0)
        verify_priority(prefix + text, len(prefix) + offset,
                        preemptible=False, low_operation=129)

    def test_filtered_geam_high_session_never_clears_a_contradictory_completion(self):
        text, offset = fixture()
        resume = event('low_resumed',1,3,160,65536,2)
        native = admission(2,9,129) + event('high_issued',2,9)
        text = text.replace(resume, native + event('high_physical_done',2,0) + resume)
        for name, sequence in (('high_issued',9), ('high_issued',3), ('high_physical_done',0)):
            original = event(name,2,sequence)
            invalid = text.replace(original, original.replace('priority=100', 'priority=0'))
            with self.subTest(name=name,sequence=sequence), self.assertRaisesRegex(
                    ValueError, 'original-only session changed priority or role'):
                verify_priority(invalid, offset)
        for name in ('low_bound', 'low_issued', 'nonpreemptible_wait', 'low_logical_done'):
            prefix = event(name,2,0)
            with self.subTest(name=name), self.assertRaisesRegex(
                    ValueError, 'original-only session changed priority or role'):
                verify_priority(prefix + text, len(prefix) + offset)
        for progress in ((1,0,0), (0,1,0), (0,0,1)):
            prefix = event('native_bound',2,0,*progress)
            with self.subTest(progress=progress), self.assertRaises(ValueError):
                verify_priority(prefix + text, len(prefix) + offset)
        prefix = event('native_bound',2,0)
        verify_priority(prefix + text, len(prefix) + offset)

    def test_lt_requires_exact_operation_and_tile_geometry(self):
        text, offset = lt_fixture()
        proof = verify_priority(text, offset, low_operation=42, total_blocks=4096)
        self.assertEqual(proof['low_kernel_commands'], [(1,3)])
        self.assertEqual(proof['high_kernel_commands'], [(2,3),(2,4)])
        for operation in (11,30,31,32,41,43):
            with self.subTest(operation=operation), self.assertRaises(ValueError):
                verify_priority(text.replace(admission(1,3,42), admission(1,3,operation)),
                                offset, low_operation=42, total_blocks=4096)
        for kwargs in ({}, {'low_operation':42},
                       {'low_operation':42,'total_blocks':144},
                       {'low_operation':43,'total_blocks':4096}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                verify_priority(text, offset, **kwargs)
        with self.assertRaises(ValueError):
            verify_priority(text.replace(admission(2,4), admission(2,4,42)),
                            offset, low_operation=42, total_blocks=4096)

    def test_lt_incomplete_stop_counter_generation_and_physical_high_proof(self):
        text, offset = lt_fixture()
        stop = event('low_physical_stop',1,3,160,4096,1)
        resume = event('low_resumed',1,3,160,4096,2)
        variants = [
            text.replace(stop, ''),
            text.replace(stop, event('low_physical_stop',1,3,0,4096,1)),
            text.replace(stop, event('low_physical_stop',1,3,4096,4096,1)),
            text.replace(stop, '').replace(event('retreat_requested',1,3,0,4096,1),
                                          stop + event('retreat_requested',1,3,0,4096,1)),
            text.replace(resume, '').replace(admission(2,3), resume + admission(2,3)),
            text.replace(resume, event('low_resumed',1,3,0,4096,2)),
            text.replace(resume, event('low_resumed',1,3,160,4096,1)),
            text.replace(event('high_physical_done',2,0), ''),
            text.replace(event('low_logical_done',1,3,4096,4096,2),
                         event('low_logical_done',1,3,4095,4096,2)),
            text + event('low_logical_done',1,3,4096,4096,2),
            text + event('policy_failure',1,3,status=-1),
        ]
        for index, invalid in enumerate(variants):
            with self.subTest(case=index), self.assertRaises(ValueError):
                verify_priority(invalid, offset, low_operation=42, total_blocks=4096)

    def test_lt_native_fallback_control_cannot_claim_preemption(self):
        text, offset = lt_fixture(preemptible=False)
        verify_priority(text, offset, preemptible=False,
                        low_operation=42, total_blocks=4096)
        with self.assertRaises(ValueError):
            verify_priority(text, offset, low_operation=42, total_blocks=4096)
        invalid = text.replace(event('low_logical_done',1,3), '') + event('low_logical_done',1,3)
        with self.assertRaises(ValueError):
            verify_priority(invalid, offset, preemptible=False,
                            low_operation=42, total_blocks=4096)
        persistent, boundary = lt_fixture()
        with self.assertRaises(ValueError):
            verify_priority(persistent, boundary, preemptible=False,
                            low_operation=42, total_blocks=4096)

    def test_negative_and_malformed_memory_status_cannot_disappear(self):
        text, offset = fixture()
        for status in (-1, -2147483648, '0junk', '-1junk', '0, extra=1', ''):
            with self.subTest(status=status), self.assertRaises(ValueError):
                verify_priority(text + event('high_issued',2,5,status=status), offset)

    def test_every_known_marker_is_strict_even_when_not_used_by_the_proof(self):
        text, offset = fixture()
        variants = [
            'Priority execution audit: event=high_issued, session=2\n',
            event('high_issued',2,5).replace('session=2', 'session=-2'),
            event('high_issued',2,5).replace('priority=100', 'priority=100junk'),
            event('high_issued',2,5).replace('status=0', 'status=0, status=0'),
            event('high_issued',2,5).replace('audit:', 'audit'),
            '[Priority execution audit: truncated] [info] ' + event('high_issued',2,5),
            'unrecognized prefix ' + event('high_issued',2,5),
            'Native admission audit: session=broken\n',
            admission(2,6).replace('operation=11', 'operation=11junk'),
            admission(2,6).replace('accepted=true', 'accepted=false'),
            admission(2,6).replace('audit:', 'audit'),
            'Native kernel pipeline audit: session=2\n',
            'Native kernel pipeline audit: session=2, first=3, second=4, both_pending=truejunk\n',
            'Native kernel pipeline audit: session=2, first=3, second=4, both_pending=false\n',
            'Native kernel pipeline audit session=2, first=3, second=4, both_pending=true\n',
        ]
        for index, malformed in enumerate(variants):
            for pipeline in (True, False):
                with self.subTest(case=index,pipeline=pipeline), self.assertRaises(ValueError):
                    verify_priority(text + malformed, offset, require_pipeline=pipeline)

    def test_admission_numeric_prefix_cannot_impersonate_a_valid_operation(self):
        text, offset = fixture()
        for suffix in ('junk', '.0', ', operation=42'):
            invalid = text.replace(admission(2,4), admission(2,4).replace(
                'operation=11', 'operation=11' + suffix))
            with self.subTest(suffix=suffix), self.assertRaises(ValueError):
                verify_priority(invalid, offset)

    def test_setup_errors_are_not_hidden_by_the_post_ready_boundary(self):
        text, offset = fixture()
        for status in (1, -1, '0junk'):
            prefix = event('policy_failure',1,0,status=status)
            with self.subTest(status=status), self.assertRaises(ValueError):
                verify_priority(prefix + text, len(prefix) + offset)
        prefix = 'Native admission audit: truncated\n'
        with self.assertRaises(ValueError):
            verify_priority(prefix + text, len(prefix) + offset)

    def test_actual_spdlog_prefix_and_line_endings_preserve_order(self):
        text, offset = fixture()
        for prefix in ('[2026-10-21 12:34:56.789] [info] ',
                       '[2026-10-21 12:34:56.789] [service] [info] [thread 123] '):
            for ending in ('\n', '\r\n'):
                decorate = lambda part: ''.join(prefix + line + ending for line in part.splitlines())
                before, after = decorate(text[:offset]), decorate(text[offset:])
                for trailing_newline in (True, False):
                    candidate = before + (after if trailing_newline else after.rstrip('\r\n'))
                    with self.subTest(prefix=prefix,ending=ending,trailing=trailing_newline):
                        proof = verify_priority(candidate, len(before))
                        self.assertEqual(proof['high_kernel_commands'], [(2,3),(2,4)])
                malformed = prefix + event('high_issued',2,5,status=-1)
                with self.assertRaises(ValueError):
                    verify_priority(before + after + malformed, len(before))

    def test_library_replacement_requires_explicit_type_and_exact_tile_count(self):
        text,offset=fixture()
        text=text.replace(admission(1,3),admission(1,3,30)).replace('65536','16384')
        verify_priority(text,offset,low_operation=30,total_blocks=16384)
        for kwargs in ({},{'low_operation':30},{'low_operation':11,'total_blocks':16384}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError): verify_priority(text,offset,**kwargs)
        invalid=text.replace(admission(2,4),admission(2,4,30))
        with self.assertRaises(ValueError): verify_priority(invalid,offset,low_operation=30,total_blocks=16384)
        extended=text.replace('operation=30','operation=31')
        verify_priority(extended,offset,low_operation=31,total_blocks=16384)
        with self.assertRaises(ValueError): verify_priority(extended,offset,low_operation=30,total_blocks=16384)
        batched=extended.replace('operation=31','operation=32')
        verify_priority(batched,offset,low_operation=32,total_blocks=16384)
        with self.assertRaises(ValueError): verify_priority(batched,offset,low_operation=31,total_blocks=16384)

    def test_real_kernel_ids_exclude_warmup_and_later_copy(self):
        text,offset=fixture()
        proof=verify_priority(text,offset)
        self.assertEqual(proof['low_kernel_commands'],[(1,3)])
        self.assertEqual(proof['high_kernel_commands'],[(2,3),(2,4)])
        self.assertGreater(len(proof['transitions']),len(proof['kernel_transitions']))

    def test_missing_duplicate_late_or_wrong_type_cannot_supply_a_kernel(self):
        text,offset=fixture()
        variants=[text.replace(admission(2,4),''),text+admission(2,4),
            text.replace(admission(2,4),'')+admission(2,4),text.replace(admission(2,4),admission(2,4,25))]
        for invalid in variants:
            with self.subTest(invalid=invalid),self.assertRaises(ValueError): verify_priority(invalid,offset)

    def test_memory_errors_are_never_filtered_out(self):
        text,offset=fixture()
        with self.assertRaises(ValueError): verify_priority(text+event('high_issued',2,5,status=1),offset)

    def test_wrong_pair_empty_progress_early_resume_and_generation_reset_reject(self):
        text,offset=fixture()
        variants=[text.replace('first=3, second=4','first=1, second=4'),
            text.replace('completed_blocks=160','completed_blocks=0',1),
            text.replace(event('low_resumed',1,3,160,65536,2),'') .replace(admission(2,3),event('low_resumed',1,3,160,65536,2)+admission(2,3)),
            text.replace(event('low_resumed',1,3,160,65536,2),event('low_resumed',1,3,0,65536,2))]
        for invalid in variants:
            with self.subTest(invalid=invalid),self.assertRaises(ValueError): verify_priority(invalid,offset)

    def test_only_existing_instrumented_pipeline_observation_is_optional(self):
        text,offset=fixture()
        text=text.replace('Native kernel pipeline audit: session=2, first=3, second=4, both_pending=true\n','')
        with self.assertRaises(ValueError): verify_priority(text,offset)
        verify_priority(text,offset,require_pipeline=False)
        with self.assertRaises(ValueError): verify_priority(text.replace(event('low_physical_stop',1,3,160,65536,1),''),offset,require_pipeline=False)

    def test_boundary_and_low_count_are_not_inferred_from_existing_success(self):
        text,offset=fixture()
        for boundary in (0,-1,True,len(text)):
            with self.assertRaises(ValueError): verify_priority(text,boundary)
        with self.assertRaises(ValueError): verify_priority(text,offset,low_count=2)

if __name__=='__main__': unittest.main()
