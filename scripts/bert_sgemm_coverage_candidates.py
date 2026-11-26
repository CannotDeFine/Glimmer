"""Identify post-log-boundary BERT SGEMM candidates, never infer SM coverage.

This read-only diagnostic consumes an already completed equal-only application
attempt. It binds the original full service audit and exact log hash, then joins
typed admission, issue, selection and physical retirement for training SGEMMs
after a saved log offset. Async logging means that offset does not prove an
application phase. It does not establish descriptor or profiler-launch identity.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import re
from pathlib import Path

from execution_application_receipt import _RECORDS, execution_application_receipt
from execution_semantic_audit import parse_semantic_audit
from run_bert_candidate_validation import SUCCESS, normal_cleanup


def require(condition, message):
    if not condition:
        raise ValueError(message)


def authenticated_training_sessions(semantic, job):
    require(job['mode'] == 'equal', 'raw application job must be equal priority')
    selection, identities = job['effective_selection'], job['application_identities']
    require(set(selection) == {'low', 'high'}, 'both application roles must be selected')
    expected = {}
    for role in ('low', 'high'):
        sessions = selection[role]
        require(type(sessions) is list and sessions, 'each role needs authenticated sessions')
        identity = identities[role]
        require(all(type(identity[key]) is int and identity[key] > 0 for key in ('pid', 'start')),
                'invalid application process identity')
        for session in sessions:
            require(type(session) is int and session > 0 and session not in expected,
                    'duplicate or invalid selected session')
            expected[session] = role
    require(identities['low']['pid'] != identities['high']['pid'], 'application roles share a PID')
    ordinary = {record.value('connection'): record for record in semantic.family(4)
                if record.value('event') == 1 and record.value('role') == 0}
    require(set(ordinary) == set(expected), 'ordinary connections and selected roles differ')
    for session, role in expected.items():
        record, identity = ordinary[session], identities[role]
        actual = tuple(record.value(name) for name in
                       ('pid', 'start', 'priority', 'kernel_mode', 'effective', 'retained'))
        configured = (2, 1, 1) if role == 'low' else (1, 0, 0)
        require(actual == (identity['pid'], identity['start'], 0, *configured),
                'authenticated role identity or execution configuration differs')
    return set(selection['low'])


def targeted_training_binding(content, semantic, job, sequence, uid):
    """Resolve one run-local compute session from the authenticated training peer."""
    sessions = authenticated_training_sessions(semantic, job)
    pattern = re.compile(r'[ \t]*(?:\[[^\]\r\n]*\][ \t]*)*SGEMM target binding: v=1, '
                         r'pid=(\d+), start=(\d+), uid=(\d+), session=(\d+), sequence=(\d+)')
    lines = [(index, line) for index, line in enumerate(content.splitlines())
             if 'SGEMM target binding:' in line]
    require(len(lines) == 1, 'one unambiguous training target binding required')
    index, line = lines[0]
    match = pattern.fullmatch(line)
    require(match is not None, 'malformed training target binding')
    binding = dict(zip(('pid', 'start', 'uid', 'session', 'sequence'), map(int, match.groups())))
    low = job['application_identities']['low']
    require((binding['pid'], binding['start'], binding['uid'], binding['sequence']) ==
            (low['pid'], low['start'], uid, sequence) and binding['session'] in sessions,
            'target binding differs from the authenticated training process')
    ordinary = [record for record in semantic.family(4) if record.value('event') == 1 and
                record.value('role') == 0 and record.value('connection') == binding['session']]
    require(len(ordinary) == 1 and ordinary[0].value('uid') == uid,
            'target binding UID differs from the authenticated compute peer')
    starts = [position for position, value in enumerate(content.splitlines())
              if 'SGEMM diagnostic:' in value and 'event=start,' in value]
    require(len(starts) == 1 and index < starts[0], 'target binding must precede diagnostic start')
    return dict(role='low', **binding)


def targeted_descriptor(content, session, sequence):
    """Join one bounded service diagnostic; callers still authenticate the full run."""
    require(all(type(value) is int and 0 < value <= 0xffffffffffffffff
                for value in (session, sequence)), 'invalid target identity')
    prefix = r'[ \t]*(?:\[[^\]\r\n]*\][ \t]*)*SGEMM diagnostic: v=1, event='
    start = re.compile(prefix + r'start, session=(\d+), sequence=(\d+)')
    end = re.compile(prefix + r'end, session=(\d+), sequence=(\d+), captured=(\d+)')
    descriptor = re.compile(prefix + r'descriptor, session=(\d+), sequence=(\d+), '
                            r'transa=(\d+), transb=(\d+), m=(\d+), n=(\d+), k=(\d+), '
                            r'lda=(\d+), ldb=(\d+), ldc=(\d+), '
                            r'device_scalars=(\d+), replacement=(\d+)')
    seen = {}
    selected = {}
    position = 0
    for original in content.splitlines(keepends=True):
        line = original.rstrip('\r\n')
        if 'SGEMM diagnostic:' in line:
            matches = [(name, pattern.fullmatch(line)) for name, pattern in
                       (('start', start), ('descriptor', descriptor), ('end', end))]
            matches = [(name, match) for name, match in matches if match is not None]
            require(len(matches) == 1, 'unknown or malformed SGEMM diagnostic')
            name, match = matches[0]
            require(name not in seen, 'duplicate SGEMM diagnostic')
            values = tuple(map(int, match.groups()))
            require(values[:2] == (session, sequence), 'SGEMM diagnostic target differs')
            seen[name] = position, values
        for name in ('Native admission audit', 'Native persistent audit',
                     'Native issue audit', 'Native execution audit'):
            if name not in line:
                continue
            match = _RECORDS[name].fullmatch(line)
            require(match is not None, 'malformed native audit record')
            values = tuple(map(int, match.groups()))
            if values[:2] == (session, sequence):
                require(name not in selected, 'duplicate target native record')
                selected[name] = position, values
        position += len(original)
    require(set(seen) == {'start', 'descriptor', 'end'} and seen['end'][1][2] == 1,
            'target descriptor lifecycle is incomplete')
    require(set(selected) == {'Native admission audit', 'Native persistent audit',
                              'Native issue audit', 'Native execution audit'},
            'target lacks admission, persistent issue or retirement')
    admission, persistent = selected['Native admission audit'], selected['Native persistent audit']
    issue, retirement = selected['Native issue audit'], selected['Native execution audit']
    require(seen['start'][0] < admission[0] < seen['descriptor'][0] < persistent[0] <
            issue[0] < retirement[0] < seen['end'][0], 'target lifecycle order differs')
    require(admission[1][2] == 30 and persistent[1][2] >= 80 and
            persistent[1][3:] == (1, 320) and issue[1][2] == 0 and retirement[1][2:] == (0, 1, 0),
            'target is not a successful large 320-worker SGEMM issue')
    values = seen['descriptor'][1]
    transa, transb, m, n, k, lda, ldb, ldc, device_scalars, replacement = values[2:]
    require(transa <= 2 and transb <= 2 and all(0 < x <= 0x7fffffff for x in (m, n, k, lda, ldb, ldc))
            and device_scalars in (0, 1) and replacement == 1, 'invalid replacement descriptor')
    return dict(session=session, sequence=sequence, transa=transa, transb=transb, m=m, n=n, k=k,
                lda=lda, ldb=ldb, ldc=ldc, device_scalars=device_scalars,
                logical_blocks=persistent[1][2], workers=320, scalar_equivalence_verified=False,
                application_identity_verified=False, application_phase_verified=False,
                physical_launch_bound=False, native_operation_matched=False,
                sm_coverage_verified=False, saturation_verified=False, performance_qualified=False)


def targeted_issue_range(content, session, sequence):
    """Validate owner-side range metadata, not profiler or application evidence."""
    descriptor = targeted_descriptor(content, session, sequence)
    prefix = r'[ \t]*(?:\[[^\]\r\n]*\][ \t]*)*SGEMM range diagnostic: v=1, event='
    start = re.compile(prefix + r'start')
    end = re.compile(prefix + r'end, attempts=(\d+), resumes=(\d+), push=(-?\d+), '
                     r'pop=(-?\d+), issued=(\d+), native=(-?\d+), completion=(-?\d+), '
                     r'has_library=(\d+), library=(-?\d+), saturated=(\d+), supported=(\d+)')
    records = {}
    position = 0
    for original in content.splitlines(keepends=True):
        line = original.rstrip('\r\n')
        if 'SGEMM range diagnostic:' in line:
            matches = [(name, pattern.fullmatch(line)) for name, pattern in (('start', start), ('end', end))]
            matches = [(name, match) for name, match in matches if match is not None]
            require(len(matches) == 1, 'unknown or malformed SGEMM range diagnostic')
            name, match = matches[0]
            require(name not in records, 'duplicate SGEMM range diagnostic')
            records[name] = position, tuple(map(int, match.groups()))
        position += len(original)
    require(set(records) == {'start', 'end'} and records['start'][0] < records['end'][0],
            'missing or unordered SGEMM range lifecycle')
    require(content.find('SGEMM diagnostic: v=1, event=start') < records['start'][0] <
            content.find('SGEMM diagnostic: v=1, event=descriptor') and
            content.find('SGEMM diagnostic: v=1, event=end') < records['end'][0],
            'range lifecycle is outside its exact descriptor lifecycle')
    (attempts, resumes, push, pop, issued, native, completion, has_library,
     library, saturated, supported) = records['end'][1]
    require(attempts == 1 and resumes == 0 and push >= 0 and pop == push and issued == 1 and
            native == 0 and completion == 0 and has_library == 0 and library == -1 and
            saturated == 0 and supported == 1, 'target range is not one supported successful issue')
    return dict(descriptor=descriptor, range_issue_preconditions_verified=True,
                range_marker_observed=False, physical_launch_bound=False,
                native_operation_matched=False, sm_coverage_verified=False,
                saturation_verified=False, performance_qualified=False)


def targeted_ncu_metrics(content, expected_kernel):
    """Read one raw NCU launch; identity, replay and coverage remain unqualified."""
    require(type(content) is str and len(content) <= 4 * 1024 * 1024 and
            type(expected_kernel) is str and expected_kernel, 'bounded NCU input and kernel required')
    lines = content.splitlines()
    starts = [index for index, line in enumerate(lines) if line.startswith('"ID",')]
    require(len(starts) == 1, 'exactly one NCU raw CSV header required')
    reader = csv.DictReader(io.StringIO('\n'.join(lines[starts[0]:])), strict=True)
    fields = reader.fieldnames or []
    metrics = ('launch__sm_count', 'launch__grid_size', 'launch__block_size',
               'sm__cycles_active.min', 'sm__sass_thread_inst_executed_op_ffma_pred_on.min')
    identity = ('ID', 'Process ID', 'Kernel Name', 'Context', 'Stream')
    require(len(fields) == len(set(fields)) and set(identity + metrics) <= set(fields),
            'missing or duplicated NCU launch fields')
    try:
        rows = list(reader)
    except csv.Error as error:
        raise ValueError('malformed NCU CSV') from error
    require(len(rows) == 2 and all(None not in row and None not in row.values() for row in rows),
            'one units row and exactly one complete launch required')
    units, row = rows
    require(all(units[name] == '' for name in identity) and
            [units[name] for name in metrics] == ['SM', '', '', 'cycle', 'inst'],
            'unexpected NCU raw metric units')
    require(all(row[name].isdecimal() for name in ('ID', 'Process ID', 'Context', 'Stream')) and
            int(row['Process ID']) > 0 and row['Kernel Name'] == expected_kernel,
            'invalid process, context, stream or kernel identity')
    values = {}
    for name in metrics:
        require(re.fullmatch(r'(?:0|[1-9][0-9]*|[1-9][0-9]{0,2}(?:,[0-9]{3})+)(?:\.[0-9]+)?',
                             row[name]) is not None, 'invalid NCU metric spelling')
        try:
            number = float(row[name].replace(',', ''))
        except ValueError as error:
            raise ValueError('invalid NCU metric') from error
        require(math.isfinite(number) and number >= 0, 'nonfinite or negative NCU metric')
        values[name] = number
    require(values['launch__sm_count'] == 80 and values['launch__grid_size'] == 320 and
            values['launch__block_size'] > 0 and values['launch__block_size'].is_integer(),
            'unexpected SM domain or launch geometry')
    return dict(process_id=int(row['Process ID']), launch_id=int(row['ID']),
                context=int(row['Context']), stream=int(row['Stream']), kernel=row['Kernel Name'],
                metrics=values,
                minimums_positive=values['sm__cycles_active.min'] > 0 and
                values['sm__sass_thread_inst_executed_op_ffma_pred_on.min'] > 0,
                range_marker_observed=False, physical_launch_bound=False,
                sm_coverage_verified=False, saturation_verified=False, performance_qualified=False)


def post_boundary_candidates(content, offset, low_sessions):
    require(type(offset) is int and 0 < offset < len(content) and content[offset - 1] == '\n',
            'saved log boundary must be an exact complete-line offset')
    require(low_sessions and all(type(value) is int and value > 0 for value in low_sessions),
            'authenticated training sessions are required')
    selected = {name: {} for name in ('Native admission audit', 'Native persistent audit',
                                      'Native issue audit', 'Native execution audit')}
    position = 0
    for original in content.splitlines(keepends=True):
        line = original.rstrip('\r\n')
        markers = [name for name in selected if name in line]
        if markers:
            require(len(markers) == 1, 'ambiguous native record')
            marker = markers[0]
            match = _RECORDS[marker].fullmatch(line)
            require(match is not None, 'malformed native record')
            values = tuple(map(int, match.groups()))
            key = values[:2]
            require(all(key) and key not in selected[marker], 'duplicate or zero command identity')
            selected[marker][key] = (position, values)
        position += len(original)
    candidates = []
    for key, (admission_position, admission) in selected['Native admission audit'].items():
        if admission_position < offset or key[0] not in low_sessions or admission[2] != 30:
            continue
        persistent = selected['Native persistent audit'].get(key)
        if persistent is None or persistent[1][2] < 80:
            continue
        issued = selected['Native issue audit'].get(key)
        retired = selected['Native execution audit'].get(key)
        require(issued is not None and retired is not None, 'selected command lacks issue or retirement')
        require(admission_position < persistent[0] < issued[0] < retired[0],
                'selected command audit order is invalid')
        require(persistent[1][3:] == (1, 320) and issued[1][2] == 0 and
                retired[1][2:] == (0, 1, 0), 'selected command is not a successful 320-worker issue')
        candidates.append(dict(session=key[0], sequence=key[1], operation=30,
                               logical_blocks=persistent[1][2], workers=320,
                               admission_position=admission_position))
    require(candidates, 'no large SGEMM candidate after the saved log boundary')
    return candidates


def inspect_attempt(attempt):
    report_path, log_path = attempt / 'report.json', attempt / 'equal/server.log'
    report = json.loads(report_path.read_text())
    require(report.get('status') == 'equal_only_existing_native_requires_review' and
            report.get('input_identity_unchanged') is True and
            report.get('fresh_native_run') is False, 'attempt is not a sealed equal-only diagnostic')
    jobs = report.get('jobs', [])
    require(len(jobs) == 1 and jobs[0]['mode'] == 'equal' and
            jobs[0]['unit']['returncode'] == 0 and jobs[0]['unit']['unit_removed'] is True,
            'attempt requires one normally removed equal unit')
    worker = jobs[0]['worker']
    require(worker['status'] == 'passed_requires_review' and worker['scoped_publication_valid'] is True,
            'worker result is not qualified for this diagnostic')
    require(worker['reparse'] == dict(tensors=401, elements=217787909,
                                      full_tensor_recomparison=True,
                                      source_bound_complete_inference=True,
                                      first_command_handoff=False), 'full numerical receipt required')
    job = worker['job']
    require(job['status'] == SUCCESS and json.loads((attempt / 'equal/attempt.json').read_text()) == job,
            'embedded and raw application receipts differ')
    normal_cleanup(job)
    require(log_path.stat().st_size <= 64 * 1024 * 1024, 'service log exceeds bounded parser capacity')
    data = log_path.read_bytes()
    require(len(data) <= 64 * 1024 * 1024, 'service log exceeds bounded parser capacity')
    digest = hashlib.sha256(data).hexdigest()
    require(digest == job['artifact_identities']['server.log'], 'service log identity mismatch')
    content = data.decode('utf-8')
    require(execution_application_receipt(content, expected_workers=320, require_geam=True) ==
            job['actual_execution'], 'complete typed service receipt differs')
    semantic = parse_semantic_audit(content, require_complete=True)
    low_sessions = authenticated_training_sessions(semantic, job)
    candidates = post_boundary_candidates(content, job['priority_audit_start'], low_sessions)
    return dict(status='post_log_boundary_sgemm_candidates_only', service_log_sha256=digest,
                saved_log_boundary=job['priority_audit_start'], candidate_count=len(candidates),
                candidates=candidates[:5], descriptor_matched=False, physical_launch_bound=False,
                application_phase_verified=False, native_operation_matched=False, sm_coverage_verified=False,
                saturation_verified=False, performance_qualified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(inspect_attempt(args.attempt.resolve(strict=True)), indent=2))


if __name__ == '__main__':
    main()
