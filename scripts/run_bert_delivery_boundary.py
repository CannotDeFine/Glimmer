#!/usr/bin/env python3
"""Validate and collect explicit native/equal request-delivery diagnostics."""

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import sys

import run_bert_intake_boundary as intake
from artifact_paths import PROJECT_ROOT
from execution_priority_audit import COMPUTE_OPERATIONS
from run_rnn_v8_validation import digest

lifecycle = intake.lifecycle
require = lifecycle.require
ENTRY = Path(__file__).resolve()
DELIVERY_PHASES = ('flushed_before_first_read', 'flushed_read_to_frame', 'delivery_unresolved')
TRANSPORT = 'CMakeFiles/glimmer_execution_transport_diagnostic.dir/'
SESSION = 'CMakeFiles/glimmer_execution_session_diagnostic.dir/'
HELPER_PLANS = None
HELPER_FLAG = '--skip-idle-helper-plans'
ISSUE_DEPTH = False
ISSUE_EVENTS = False
OBSERVATION_GENERATIONS = False
PREPARATION_BOUNDARY = False
SELECTED_COMMANDS = False
INTAKE_BURST = False
POLL_BOUNDARY = False
POLL_FLAG = '--poll-journal-output'
INTAKE_FLAG = '--async-intake-burst'
DEPTH_FLAG = '--issue-depth-output'
EVENTS_FLAG = '--issue-depth-events'
SELECTIONS_FLAG = '--issue-depth-selections'
EVENT_FIELDS = ('connection', 'sequence', 'opcode', 'admitted_ns', 'first_observed_ns',
                'matched_observed_ns', 'authorized_ns', 'issue_begin_ns', 'issue_end_ns',
                'retired_ns', 'plan_serial', 'positive_observations', 'plan_positive_observations')
GENERATION_FIELDS = (*EVENT_FIELDS, 'admission_observation_generation', 'first_positive_observation_generation')
PREPARATION_FIELDS = (*GENERATION_FIELDS, 'preparation_required', 'service_observed_prepared_ns')
SELECTION_FIELDS = ('connection', 'observation_generation', 'observed_ns', 'pending_sequence', 'classification')
SELECTION_CLASSES = ('tracked_ready', 'tracked_not_ready', 'scope_external', 'unknown')
DEPTH_OPCODES = frozenset(COMPUTE_OPERATIONS)
DEPTH_CATEGORIES = ('ready_compute', 'not_ready_compute', 'other_pending',
                    'no_selected_command', 'unknown')


class BoundInputs:
    """Keep every direct analysis input bound across derivation and publication."""

    def __init__(self):
        self.identities = {}

    def read(self, path, expected=None):
        checksum = digest(path)
        require(expected is None or checksum == expected, 'analysis input differs from receipt')
        require(str(path) not in self.identities or self.identities[str(path)] == checksum,
                'analysis input changed between reads')
        self.identities[str(path)] = checksum
        value = json.loads(path.read_text())
        require(digest(path) == checksum, 'analysis input changed during read')
        return value

    def verify(self):
        require(all(digest(Path(path)) == checksum for path, checksum in self.identities.items()),
                'analysis inputs changed during derivation')


def compile_commands(rows, selected, marker=None):
    result = {}
    for row in rows:
        if row['file'] not in selected or marker and marker not in row['command']:
            continue
        args = shlex.split(row['command'])
        index = args.index('-o')
        require(row['file'] not in result, 'duplicate selected compilation')
        result[row['file']] = (args[:index] + args[index + 2:],
            str((Path(row['directory']) / args[index + 1]).resolve()))
    require(set(result) == selected, 'missing selected compilation')
    return result


def session_overlay_identity(part, old_rows, new_rows, linked, mapping, binary, build):
    """Accept one explicitly selected session object, never arbitrary backend replacements."""
    if part != 'service' or not (PREPARATION_BOUNDARY is True or INTAKE_BURST is True or POLL_BOUNDARY is True):
        require(not any(SESSION in arg for arg in linked), 'session overlay was not explicitly selected')
        return None, set()
    source = str(PROJECT_ROOT / 'src/backend/cuda/cuda_execution_session.cc')
    old, new = compile_commands(old_rows, {source}), compile_commands(new_rows, {source})
    relative = SESSION + source.lstrip('/') + '.o'
    output = str(build / relative)
    backend = str(lifecycle.candidate.BUILD / 'src/backend/cuda/libglimmer_cuda_backend.a')
    require(old[source][0] == new[source][0] and new[source][1] == output
            and linked.count(output) == linked.count(backend) == 1
            and linked.index(output) < linked.index(backend), 'exact ordered session overlay required')
    require(relative in mapping and 'libglimmer_cuda_backend.a(cuda_execution_session.cc.o)' not in mapping,
            'link map extracted an unreviewed session member')
    old_dep, new_dep = Path(old[source][1] + '.d'), Path(output + '.d')

    def dependencies(path):
        return {Path(token).resolve() for token in shlex.split(
            path.read_text().replace('\\\n', ' ').split(':', 1)[1])}

    included = dependencies(new_dep)
    require(included == dependencies(old_dep), 'session dependency paths changed')
    symbols = subprocess.check_output(['nm', '-C', '--defined-only', str(binary)], text=True, timeout=10)
    overloads = [line for line in symbols.splitlines()
                 if ' T glimmer::backend::cuda::CudaExecutionSession::accept_async_kernel(' in line]
    require(len(overloads) == 2 and sum('bool*' in line for line in overloads) == 1,
            'session async admission ABI symbols differ')
    return output, included | {old_dep, new_dep, Path(source), Path(output),
        PROJECT_ROOT / 'src/backend/cuda/internal/cuda_execution_session.h'}


def build_identity(part):
    """Permit only the explicitly reviewed transport and optional session overlays."""
    baseline, build = lifecycle.candidate.BUILD, lifecycle.BUILD
    old_rows = json.loads((baseline / 'compile_commands.json').read_text())
    new_rows = json.loads((build / 'compile_commands.json').read_text())
    sources = ({row['file'] for row in old_rows if '/src/execution_service/' in row['file']}
               if part == 'service' else {str(PROJECT_ROOT / 'src/interceptor' / name) for name in
               ('execution/driver_frontend.cc', 'execution/driver_lookup.cc', 'dlsym_entry_x86_64.S')})
    require(len(sources) == (7 if part == 'service' else 3), 'exact inherited sources required')
    target = 'glimmer_execution_service' if part == 'service' else 'glimmer_execution_interceptor'

    marker = None if part == 'service' else target + '.dir/'
    old, new = compile_commands(old_rows, sources, marker), compile_commands(new_rows, sources, marker)
    require(all(old[name][0] == new[name][0] for name in sources), 'inherited flags changed')
    socket = str(PROJECT_ROOT / 'src/control/execution_socket.cc')
    old_socket, new_socket = compile_commands(old_rows, {socket}), compile_commands(new_rows, {socket})
    transport_object = str(build / (TRANSPORT + socket.lstrip('/') + '.o'))
    require(old_socket[socket][0] == new_socket[socket][0]
            and '-fPIC' in new_socket[socket][0] and new_socket[socket][1] == transport_object,
            'exact matching PIC socket compilation required')
    relative = ('bin/glimmer_execution_service' if part == 'service'
                else 'lib/libglimmer_execution_interceptor.so')
    old_binary, new_binary = baseline / relative, build / relative
    old_directory = baseline / ('src/execution_service' if part == 'service' else 'src/interceptor/execution')
    old_link = old_directory / ('CMakeFiles/' + target + '.dir/link.txt')
    new_link = build / part / ('CMakeFiles/' + target + '.dir/link.txt')

    def link(path):
        return [str((path.parents[2] / arg).resolve()) if not arg.startswith('-') else arg
                for arg in shlex.split(path.read_text())]

    before, after = link(old_link), link(new_link)
    substitutions = {old[name][1]: new[name][1] for name in sources}
    substitutions[str(old_binary)] = str(new_binary)
    if part == 'service':
        substitutions[str(old_directory / 'libglimmer_execution_service_runtime.a')] = str(
            build / 'service/libglimmer_execution_service_runtime.a')
    map_path = build / (target + '.map')
    map_option = '-Wl,-Map,' + str(map_path)
    control = str(baseline / 'src/control/libglimmer_control.a')
    mapping = map_path.read_text()
    session_object, session_paths = session_overlay_identity(
        part, old_rows, new_rows, after, mapping, new_binary, build)
    require(after.count(transport_object) == after.count(map_option) == 1
            and after.index(transport_object) < after.index(control)
            and [substitutions.get(arg, arg) for arg in before]
                == [arg for arg in after if arg not in (transport_object, session_object, map_option)],
            'ordered link differs beyond declared objects/output/map')
    require(TRANSPORT + socket.lstrip('/') + '.o' in mapping
            and 'libglimmer_control.a(execution_socket.cc.o)' not in mapping,
            'link map did not select only the current transport')

    def dynamic(path):
        output = subprocess.check_output(['readelf', '-d', str(path)], text=True, timeout=10)
        return [line.split(')', 1)[1].strip() for line in output.splitlines()
                if any('(' + key + ')' in line for key in ('NEEDED', 'RUNPATH', 'RPATH', 'SONAME'))]

    require(dynamic(old_binary) == dynamic(new_binary), 'dynamic dependencies changed')
    if part == 'frontend':
        def exports(path):
            output = subprocess.check_output(['nm', '-D', '--defined-only', str(path)], text=True, timeout=10)
            return sorted(line.split()[1:] for line in output.splitlines())
        require(exports(old_binary) == exports(new_binary), 'frontend public exports changed')
    paths = {Path(arg) for arg in before + after if Path(arg).is_absolute() and Path(arg).is_file()}
    paths.update(session_paths)
    paths.update({old_link, new_link, map_path, baseline / 'compile_commands.json',
                  build / 'compile_commands.json', baseline / 'CMakeCache.txt', build / 'CMakeCache.txt',
                  PROJECT_ROOT / 'cmake/ExecutionFrontendDiagnosticOverlay/CMakeLists.txt',
                  PROJECT_ROOT / 'cmake/ExecutionServiceDiagnosticOverlay/CMakeLists.txt'})
    equivalents = sources | {socket}
    if session_object:
        equivalents.add(str(PROJECT_ROOT / 'src/backend/cuda/cuda_execution_session.cc'))
    return dict(compile_equivalent_files=sorted(equivalents),
                baseline_link=before, candidate_link=after, dynamic_dependencies=dynamic(new_binary),
                identities={str(path): digest(path) for path in sorted(paths)})


def validate_intake(document):
    ledger = document['intake']
    require(ledger['schema'] == 2 and ledger['fields'] == ['frame_received_ns', 'owner_enter_ns',
            'retained_ns', 'slot', 'wire', 'opcode', 'first_read_ns'], 'exact delivery intake schema required')
    for row in ledger['records']:
        require(len(row) == 7 and all(type(value) is int and value >= 0 for value in row)
                and 0 < row[6] <= row[0], 'invalid first-positive-read observation')
    projected = dict(ledger, schema=1, fields=ledger['fields'][:6], records=[row[:6] for row in ledger['records']])
    return intake.validate_intake(dict(document, intake=projected))


def validate_delivery(document):
    ledger = document['submit_delivery']
    require(ledger['schema'] == 1 and ledger['fields'] == ['serial', 'session', 'wire', 'fully_flushed'],
            'exact submit-delivery schema required')
    _, _, wires = lifecycle.frontend.frontend(document, {'pid': document['pid']},
                                             document['begin_ns'], document['end_ns'])
    observations = {}
    for row in ledger['records']:
        require(len(row) == 4 and all(type(value) is int and value > 0 for value in row[:3])
                and type(row[3]) is bool, 'invalid submit-delivery row')
        serial, session, wire, flushed = row
        key = session, wire
        require(key not in observations and key in wires and wires[key]['serial'] == serial,
                'duplicate, foreign or mismatched delivery identity')
        observations[key] = flushed
    require(observations.keys() == wires.keys(), 'every accepted RPC needs a delivery observation')
    return observations


def split_delivery(sent, returned, first_read, frame, owner, retained, flushed):
    require(type(flushed) is bool and sent <= first_read <= frame, 'invalid delivery clock or state')
    result = []
    for left, right, phase in intake.split_outbound(sent, returned, frame, owner, retained):
        if phase != 'pre_frame':
            result.append((left, right, phase))
        elif not flushed:
            result.append((left, right, 'delivery_unresolved'))
        else:
            cut = min(right, max(left, first_read))
            result.extend((a, b, label) for a, b, label in
                          ((left, cut, 'flushed_before_first_read'), (cut, right, 'flushed_read_to_frame'))
                          if a < b)
    return result


def summarize(buckets):
    exclusive, mixed, parent = Counter(), 0, 0
    for key, elapsed in buckets.items():
        phases = {signature[-1] for signature, _ in key}
        require(phases <= {'during_submit', 'frame_to_owner', 'owner_to_retained', *DELIVERY_PHASES},
                'unknown delivery phase')
        if phases.intersection(DELIVERY_PHASES):
            parent += elapsed
            if len(phases) == 1:
                exclusive[next(iter(phases))] += elapsed
            else:
                mixed += elapsed
    require(sum(exclusive.values()) + mixed == parent, 'pre-frame partition lost time')
    dominant = [phase for phase in DELIVERY_PHASES if parent and exclusive[phase] >= .6 * parent]
    return dict(pre_frame_union_ns=parent, mixed_pre_frame_overlap_ns=mixed,
                exclusive_delivery_ns={phase: exclusive[phase] for phase in DELIVERY_PHASES},
                exclusive_fraction_of_pre_frame={phase: exclusive[phase] / parent if parent else None
                                                 for phase in DELIVERY_PHASES},
                next_boundary=dominant[0] if len(dominant) == 1 else 'inconclusive')


def analyze(attempt, scope):
    inputs = BoundInputs()
    worker = inputs.read(attempt / 'equal-worker.json')
    job = inputs.read(attempt / 'equal/helper-attempt.json')
    intake.original_analyze(attempt, scope)
    inputs.verify()
    coarse = inputs.read(attempt / 'frontend-analysis.json')
    request = inputs.read(attempt / 'request-analysis.json')
    identity = job['application_identities']['inference']

    def bound(receipt):
        return inputs.read(Path(receipt['path']), receipt['sha256'])

    receipt = worker['frontend_outputs']['inference']
    require(receipt['application_identity'] == identity, 'frontend inference identity differs')
    base, service = bound(receipt), bound(worker['stage_output'])
    observations = validate_delivery(base)
    validate_intake(service)
    begin, end = coarse['begin_ns'], coarse['end_ns']
    require(end - begin == 5_000_000_000, 'fixed five-second inference window required')
    _, _, wires = lifecycle.frontend.frontend(base, identity, begin, end)
    states = lifecycle.frontend.service(service, begin, end)
    owned = [(i, slot, state) for i, (slot, state) in enumerate(zip(service['slots'], states))
             if tuple(slot[1:]) == (identity['pid'], identity['start']) and state['kind'] == 0]
    require(len(owned) == 1, 'one authenticated inference compute slot required')
    index, slot, state = owned[0]
    coverage = [row for row in request['request_lifecycle']['slots']
                if row['role'] == 'inference' and row['compute_connection']]
    require(len(coverage) == 1 and coverage[0]['slot'] == slot and coverage[0]['coverage_gate']
            and .95 <= coverage[0]['lifecycle_coverage'] <= 1, 'complete lifecycle window required')
    records = {row[4]: row for row in service['intake']['records'] if row[3] == index}
    expected = {wire for wire, call in state['calls'].items() if call['receive'] < end and call['reply'] >= begin}
    matched, detailed, parent, original, pre_frame = set(), [], [], [], []
    counts = Counter()

    def append(target, spans, signature):
        for left, right, phase in spans:
            left, right = max(begin, left), min(end, right)
            if left < right:
                target.append((left, right, (*signature, phase)))

    for (session, wire), row in wires.items():
        if session != slot[0] or row['enter'] >= end or row['finish'] < begin:
            continue
        require(wire in state['calls'] and wire in records, 'missing same-slot compute wire')
        call = state['calls'][wire]
        frame, owner, retained, _, _, opcode, first_read = records[wire]
        require(opcode == row['opcode'] == call['opcode'] and retained == call['receive']
                and retained <= call['reply'] <= row['reply'], 'three-way compute identity differs')
        if wire in expected:
            matched.add(wire)
        signature = (opcode, row['thread'], int(row['boundary'] != 0), row['flags'])
        spans = intake.split_outbound(row['send_begin'], row['send_end'], frame, owner, retained)
        append(parent, spans, signature)
        append(pre_frame, [span for span in spans if span[2] == 'pre_frame'], signature)
        append(detailed, split_delivery(row['send_begin'], row['send_end'], first_read, frame, owner,
                                       retained, observations[session, wire]), signature)
        left, right = max(begin, row['send_begin']), min(end, retained)
        if left < right:
            original.append((left, right, signature))
        counts['fully_flushed' if observations[session, wire] else 'delivery_unresolved'] += 1
    require(matched == expected and matched, 'complete window compute correlation required')
    gaps = intake.union(state['gaps'], begin, end)
    buckets, uncovered = intake.partition(gaps, detailed)
    parents, parent_uncovered = intake.partition(gaps, parent)
    collapsed, unsplit = Counter(), Counter()
    for active, elapsed in buckets.items():
        combined, bare = Counter(), Counter()
        for signature, count in active:
            phase = 'pre_frame' if signature[-1] in DELIVERY_PHASES else signature[-1]
            combined[(*signature[:-1], phase)] += count
            bare[signature[:-1]] += count
        collapsed[tuple(sorted(combined.items()))] += elapsed
        unsplit[tuple(sorted(bare.items()))] += elapsed
    originals, original_uncovered = intake.partition(gaps, original)
    require(collapsed == parents and unsplit == originals and uncovered == parent_uncovered == original_uncovered,
            'delivery split changed parent active multisets')
    prior = next(row for row in coarse['roles'] if row['role'] == 'inference')
    require(intake.duration(gaps) == prior['service_supply_gap_ns'] == coverage[0]['zero_wire_and_zero_unissued_ns']
            and sum(originals.values()) == prior['phase_ns']['outbound_submission'], 'parent union differs')
    result = summarize(buckets)
    pre_buckets, _ = intake.partition(gaps, pre_frame)
    require(result['pre_frame_union_ns'] == sum(pre_buckets.values()), 'independent pre-frame union differs')
    result.update(status='delivery_boundary_requires_independent_review', begin_ns=begin, end_ns=end,
        application_identity=identity, compute_slot=slot, correlated_compute_wires=len(matched),
        touching_window_delivery_counts=dict(counts), supply_gap_ns=intake.duration(gaps),
        outbound_ns=sum(originals.values()), other_gap_ns=uncovered, parent_intake_summary=intake.summarize(parents),
        pre_frame_fraction_of_supply_gap=result['pre_frame_union_ns'] / intake.duration(gaps),
        buckets=[dict(active_multiset=[dict(signature=key, count=count) for key, count in active], duration_ns=elapsed)
                 for active, elapsed in sorted(buckets.items())], raw_inputs=inputs.identities, lifecycle_scope=scope,
        diagnostic_only=True, full_device_qualified=False, performance_qualified=False,
        gpu_gap_causality_qualified=False, priority_qualified=False)
    inputs.verify()
    encoded = json.dumps(result, indent=2, allow_nan=False) + '\n'
    require(len(encoded.encode()) < 32 * 1024 ** 2, 'bounded delivery analysis required')
    with (attempt / 'delivery-analysis.json').open('x') as stream:
        stream.write(encoded)
    inputs.verify()
    print(json.dumps({key: result[key] for key in ('next_boundary', 'exclusive_fraction_of_pre_frame')}))
    if HELPER_PLANS is not None:
        helper_plan_metrics(attempt)
    if POLL_BOUNDARY:
        import analyze_bert_poll_boundary as poll_boundary
        return poll_boundary.analyze(attempt, scope)
    return 0


def helper_plan_metrics(attempt):
    """Use the existing fixed window and expose differently attributed counters."""
    inputs = BoundInputs()
    request = inputs.read(attempt / 'request-analysis.json')
    delivery = inputs.read(attempt / 'delivery-analysis.json')
    worker = inputs.read(attempt / 'equal-worker.json')
    document = inputs.read(Path(worker['stage_output']['path']), worker['stage_output']['sha256'])
    reasons = ['disabled_policy', 'non_helper', 'local_obligation', 'skipped_helper']
    require(document['slot_plan_schema'] == 1 and document['slot_plan_reasons'] == reasons
            and document['slot_plan_attribution'] == 'last_scope_stamp', 'exact helper reason schema required')
    controls = request['controls']
    require([row['mode'] for row in controls] == ['native', 'equal'], 'matched controls required')
    begin, end = delivery['begin_ns'], delivery['end_ns']
    stage_index = document['stages'].index('plan_slot')
    totals, selected = [0] * 4, [0] * 4
    plan_calls = 0
    for bucket in document['buckets']:
        values = bucket['slot_plans']
        require(len(values) == 4 and all(type(value) is int and value >= 0 for value in values),
                'invalid helper reason counters')
        totals = [a + b for a, b in zip(totals, values)]
        plan_calls += bucket['stages'][stage_index][0]
        if (begin <= bucket['begin_ns'] and bucket['begin_ns'] + document['bucket_ns'] <= end
                and begin <= bucket['minimum_span_start_ns'] <= bucket['maximum_span_end_ns'] <= end):
            selected = [a + b for a, b in zip(selected, values)]
    require(sum(totals) == plan_calls + totals[3], 'complete-ledger helper reasons do not close')
    require((selected[0] == 0 if HELPER_PLANS else totals[3] == 0), 'helper switch did not match observation')
    stages = controls[1]['service_stages']
    rows = {row['stage']: row for row in stages['stages']}
    steps = rows['step']['calls']
    require(end - begin == 5_000_000_000 and steps > 0 and stages['attribution_coverage_gate'],
            'complete fixed window with covered service steps required')
    completions = {}
    for row in controls:
        window = row['application_window']
        require(window['end_ns'] - window['begin_ns'] == end - begin, 'matched window duration required')
        completions[row['mode']] = window['end']['completed_requests'] - window['begin']['completed_requests']
    result = dict(helper_plans=HELPER_PLANS, steps=steps, reasons=dict(zip(reasons, selected)),
        complete_ledger_reasons=dict(zip(reasons, totals)), complete_ledger_plan_calls=plan_calls,
        helper_skip_rate=selected[3] / (selected[2] + selected[3]) if selected[2] + selected[3] else None,
        plan_calls_per_step=rows['plan_slot']['calls'] / steps,
        plan_slot_inclusive_cpu_ns_per_step=rows['plan_slot']['cpu_ns'] / steps,
        supply_gap_fraction=delivery['supply_gap_ns'] / (end - begin), completions=completions,
        stage_attribution='span_end_bucket', reason_attribution='last_scope_stamp',
        boundary_work_omitted=True, raw_inputs=inputs.identities,
        diagnostic_only=True, performance_qualified=False, full_device_qualified=False)
    inputs.verify()
    with (attempt / 'helper-plan-analysis.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return result


def compare_helper_plans(off, on):
    """Fixed, pre-registered diagnostic thresholds; never a performance gate."""
    require(off['helper_plans'] is False and on['helper_plans'] is True, 'fixed OFF then ON controls required')
    keys = ('plan_calls_per_step', 'plan_slot_inclusive_cpu_ns_per_step', 'supply_gap_fraction')
    require(all(off[key] > 0 and on[key] >= 0 for key in keys)
            and all(value > 0 for row in (off, on) for value in row['completions'].values()),
            'nonzero diagnostic denominators required')
    reductions = {key: 1 - on[key] / off[key] for key in keys}
    native_off, native_on = off['completions']['native'], on['completions']['native']
    gates = dict(helper_skip_rate=on['helper_skip_rate'] is not None and on['helper_skip_rate'] >= .95,
        plan_calls_per_step=reductions[keys[0]] >= .50,
        plan_slot_inclusive_cpu_per_step=reductions[keys[1]] >= .30,
        supply_gap_fraction=reductions[keys[2]] >= .15,
        equal_completions=on['completions']['equal'] * 100 >= off['completions']['equal'] * 95,
        native_control_drift=abs(native_on - native_off) * 100 <= native_off * 20)
    return dict(status='helper_response_requires_review', gates=gates, all_gates_passed=all(gates.values()),
        relative_reductions=reductions, native_completion_relative_drift=abs(native_on - native_off) / native_off,
        off=off, on=on, diagnostic_only=True, performance_qualified=False, full_device_qualified=False,
        gpu_gap_causality_qualified=False, priority_qualified=False)


def helper_service_command(command, should_skip):
    argv, pins = command
    require(type(should_skip) is bool and HELPER_FLAG not in argv, 'explicit unique helper selection required')
    return ([*argv, HELPER_FLAG] if should_skip else argv), pins


def depth_service_command(command, path, *, events=False, selections=False):
    argv, pins = command
    require(type(events) is bool and type(selections) is bool and (not selections or events)
            and EVENTS_FLAG not in argv and SELECTIONS_FLAG not in argv
            and DEPTH_FLAG not in argv and path.is_absolute() and not path.exists()
            and not path.is_symlink() and path.parent.is_dir(), 'fresh absolute depth output required')
    return [*argv, DEPTH_FLAG, str(path), *([EVENTS_FLAG] if events else []),
            *([SELECTIONS_FLAG] if selections else [])], pins


def validate_issue_depth(document, job, *, expected_events=False, expected_generations=False,
                         expected_preparation=False, expected_selections=False):
    """Validate complete host-observed compute accounting, never GPU busy time."""
    require(type(expected_events) is bool and type(expected_generations) is bool
            and type(expected_preparation) is bool and type(expected_selections) is bool
            and (not expected_selections or expected_preparation)
            and (not expected_preparation or expected_generations)
            and (not expected_generations or expected_events)
            and document['schema'] == (5 if expected_selections else 4 if expected_preparation else 3 if expected_generations else 2 if expected_events else 1)
            and ('selections' in document) == expected_selections
            and ('events' in document) == expected_events and document['diagnostic_only'] is True
            and document['valid'] is True and document['failure'] == 0 and document['unclosed'] == 0
            and document['pid'] == job['server_identity']['pid']
            and document['readiness_is_instantaneous'] is True
            and document['outstanding_is_gpu_busy'] is False, 'invalid or foreign issue-depth ledger')
    origin, begin, end, width = (document[key] for key in ('origin_ns', 'begin_ns', 'end_ns', 'bucket_ns'))
    require(all(type(value) is int for value in (origin, begin, end, width))
            and width == 200_000_000 and origin % width == 0 and origin <= begin < origin + width
            and begin < end < origin + 4096 * width, 'invalid bounded depth clock')
    contracts = {}
    for role, sessions in job['effective_selection'].items():
        owner = job['application_identities'][role]
        for connection in sessions:
            require(type(connection) is int and connection > 0 and connection not in contracts,
                    'duplicate compute connection contract')
            contracts[connection] = (owner['pid'], owner['start'])
    slots, seen, admitted = document['slots'], set(), 0
    require(0 < len(slots) <= 128, 'bounded nonempty compute slots required')
    for row in slots:
        require(all(type(value) is int and value >= 0 for value in row.values())
                and row['connection'] not in seen
                and contracts.get(row['connection']) == (row['pid'], row['start_ticks'])
                and row['admitted'] == row['submitted'] == row['retired'] and row['admitted'] > 0,
                'foreign, duplicate or unclosed depth slot')
        seen.add(row['connection'])
        admitted += row['admitted']
    require(seen == set(contracts) and admitted > 0, 'both complete compute cohorts required')
    census = {}
    for row in document['opcodes']:
        require(type(row['opcode']) is int and row['opcode'] in DEPTH_OPCODES
                and row['opcode'] not in census and type(row['admitted']) is int and row['admitted'] > 0,
                'unsupported or duplicated admission opcode')
        census[row['opcode']] = row['admitted']
    require(sum(census.values()) == admitted, 'opcode admission census differs')
    buckets = document['buckets']
    require(len(buckets) == (end - origin) // width + 1, 'incomplete depth bucket inventory')
    for index, row in enumerate(buckets):
        left = origin + index * width
        require(all(type(value) is int and value >= 0 for value in row.values())
                and row['begin_ns'] == left
                and row['covered_ns'] == max(0, min(left + width, end) - max(left, begin))
                and row['issue_call_ns'] <= row['admitted_not_submitted_ns']
                and row['no_admitted_compute_ns'] <= row['covered_ns']
                and row['no_inflight_compute_ns'] <= row['covered_ns']
                and sum(row[key] for key in DEPTH_CATEGORIES) == row['observations']
                and row['ready_without_inflight'] <= row['ready_compute']
                and row['unknown'] == 0, 'inconsistent depth bucket accounting')
    require(all(sum(row[key] for row in buckets) == admitted
                for key in ('admitted', 'submitted', 'retired')), 'global depth closure differs')
    require(sum(row['covered_ns'] for row in buckets) == end - begin, 'depth coverage is incomplete')
    result = dict(admitted=admitted, slots=len(slots), opcodes=census, begin_ns=begin, end_ns=end,
                  diagnostic_only=True, full_device_qualified=False, performance_qualified=False)
    if expected_events:
        events = document['events']
        fields = PREPARATION_FIELDS if expected_preparation else GENERATION_FIELDS if expected_generations else EVENT_FIELDS
        require(set(events) == {'capacity', 'complete', 'ready_duration', 'gpu_timestamps', 'fields', 'records'}
                and type(events['capacity']) is int and events['capacity'] == 524288
                and events['complete'] is True and events['ready_duration'] is False
                and events['gpu_timestamps'] is False and events['fields'] == list(fields)
                and type(events['records']) is list
                and len(events['records']) == admitted <= events['capacity'], 'incomplete owner-event schema')
        keys, serials, counts, operations = set(), set(), Counter(), Counter()
        for row in events['records']:
            require(type(row) is list and len(row) == len(fields)
                    and all(type(value) is int and (0 if (expected_generations and index == 13)
                                                    or (expected_preparation and index in (15, 16)) else 1)
                            <= value <= 0xffffffffffffffff for index, value in enumerate(row)),
                    'invalid owner-event row or unsigned identity')
            if expected_generations:
                require(row[14] > row[13], 'first positive generation must follow admission within the same slot')
            if expected_preparation:
                require(row[15] in (0, 1)
                        and (row[2] == 11 and row[3] <= row[16] <= row[4] if row[15] else row[16] == 0),
                        'deferred preparation provenance or owner timestamp differs')
            connection, sequence, opcode = row[:3]
            key, serial = (connection, sequence), row[10]
            require(connection in seen and key not in keys and serial not in serials
                    and opcode in census and begin <= row[3] <= row[9] <= end
                    and all(left <= right for left, right in zip(row[3:9], row[4:10]))
                    and row[11] >= row[12] > 0, 'owner-event identity, chronology or plan differs')
            keys.add(key)
            serials.add(serial)
            counts[connection] += 1
            operations[opcode] += 1
        require(counts == Counter({row['connection']: row['retired'] for row in slots})
                and operations == Counter(census), 'owner-event slot or opcode census differs')
        result['events'] = dict(completed=admitted, same_plan_coverage=1.0,
                                continuous_readiness=False, gpu_timestamps=False)
        if expected_generations:
            first = sum(row[14]-row[13] == 1 for row in events['records'])
            result['events']['observation_generations'] = dict(first_observation_positive=first,
                earlier_observations_without_positive=admitted-first,
                owner_loop_iterations_inferred=False, dependency_cause_identified=False)
        if expected_preparation:
            deferred = sum(row[15] for row in events['records'])
            result['events']['preparation_boundary'] = dict(deferred_commands=deferred,
                nondeferred_commands=admitted-deferred, dependency_ready_inferred=False,
                gpu_timestamps=False)
        if expected_selections:
            result['events']['selections'] = validate_issue_selections(document)
    return result


def validate_issue_selections(document):
    """Close sparse owner observations without inferring continuous readiness."""
    journal = document['selections']
    require(set(journal) == {'capacity', 'complete', 'fields', 'classifications', 'records'}
            and type(journal['capacity']) is int and journal['capacity'] == 524288
            and journal['complete'] is True and journal['fields'] == list(SELECTION_FIELDS)
            and journal['classifications'] == list(SELECTION_CLASSES)
            and type(journal['records']) is list and 0 < len(journal['records']) <= journal['capacity'],
            'exact complete bounded selection journal required')
    limits = {row['connection']: row['observation_generation'] for row in document['slots']}
    require(all(type(value) is int and 0 < value <= 0xffffffffffffffff for value in limits.values()),
            'invalid final slot observation generation')
    buckets = document['buckets']
    total = sum(limits.values())
    require(total == sum(row['observations'] for row in buckets)
            and total-len(journal['records']) == sum(row['no_selected_command'] for row in buckets),
            'sparse zero omission or slot generation census differs')
    events = {tuple(row[:2]): row for row in document['events']['records']}
    require(all(row[13] < row[14] <= limits[row[0]] for row in events.values()),
            'event generation exceeds its slot')
    counts, ready, first, last_generation = Counter(), Counter(), {}, {}
    last_stamp = document['begin_ns']
    for row in journal['records']:
        require(type(row) is list and len(row) == 5
                and all(type(value) is int and (0 if index == 4 else 1) <= value <= 0xffffffffffffffff
                        for index, value in enumerate(row)), 'invalid sparse selection row')
        connection, generation, stamp, sequence, classification = row
        require(connection in limits and last_generation.get(connection, 0) < generation <= limits[connection]
                and last_stamp <= stamp <= document['end_ns'] and classification in (0, 1, 2),
                'foreign, repeated, reordered or unknown selection observation')
        last_generation[connection], last_stamp = generation, stamp
        counts[(stamp-document['origin_ns']) // document['bucket_ns'], classification] += 1
        key = connection, sequence
        event = events.get(key)
        if classification == 2:
            require(event is None, 'scope-external selection matches a compute event in the same slot')
            continue
        require(event is not None and event[13] < generation
                and event[3] <= stamp <= event[7], 'tracked selection lacks its admitted unissued event')
        if classification == 0:
            require(generation >= event[14] and stamp >= event[4],
                    'ready selection precedes its recorded first positive')
            first.setdefault(key, (generation, stamp))
            ready[key] += 1
    for index, bucket in enumerate(buckets):
        require(bucket['unknown'] == 0 and all(counts[index, classification] == bucket[name]
                for classification, name in enumerate(('ready_compute', 'not_ready_compute', 'other_pending'))),
                'selection classification and bucket census differ')
    require(all(first.get(key) == (row[14], row[4]) and ready[key] == row[11]
                for key, row in events.items()), 'missing first-positive or positive-count selection correlation')
    return dict(records=len(journal['records']), observations=total,
        omitted_zero_pending=total-len(journal['records']),
        classifications={name: sum(value for (_, kind), value in counts.items() if kind == index)
                         for index, name in enumerate(SELECTION_CLASSES)},
        first_positive_coverage=1.0, continuous_readiness=False, policy_winner_inferred=False)


def issue_event_census(content, job):
    """Derive the numerical cohort independently from authenticated native audit."""
    from execution_application_receipt import _RECORDS
    from execution_priority_audit import _parse_audit_model
    admissions, _, _, semantic = _parse_audit_model(content)
    require(semantic.is_complete, 'complete independent semantic lifecycle required')
    beginning = [record for record in semantic.family(2) if record.value('event') == 1]
    require(len(beginning) == 1 and beginning[0].value('pid') == job['server_identity']['pid'],
            'independent audit service identity differs')
    eligible = {key: (opcode, position) for key, (opcode, position) in admissions.items()
                if opcode in DEPTH_OPCODES}
    expected = {connection: job['application_identities'][role]
                for role, connections in job['effective_selection'].items() for connection in connections}
    sources = {record.value('connection'): record for record in semantic.family(4)
               if record.value('event') == 1}
    require(eligible and {key[0] for key in eligible} == set(expected), 'independent compute cohorts differ')
    for connection, identity in expected.items():
        source = sources.get(connection)
        require(source is not None and source.value('role') == 0
                and (source.value('pid'), source.value('start')) == (identity['pid'], identity['start']),
                'independent audit source differs from authenticated application')
    issues, retired, position = {}, set(), 0
    for original in content.splitlines(keepends=True):
        line = original.rstrip('\r\n')
        for marker in ('Native issue audit', 'Native execution audit'):
            if marker not in line:
                continue
            match = _RECORDS[marker].fullmatch(line)
            require(line.count(marker) == 1 and match is not None, 'malformed independent compute audit')
            values = tuple(map(int, match.groups()))
            key = values[:2]
            if key not in eligible:
                continue
            require(values[-1] == 0 and position > eligible[key][1], 'failed or reordered independent compute')
            if marker == 'Native issue audit':
                require(key not in issues, 'duplicate independent issue')
                issues[key] = position
            else:
                require(values[2:4] == (0, 1) and key in issues and issues[key] < position
                        and key not in retired, 'missing or duplicate independent retirement')
                retired.add(key)
        position += len(original)
    require(set(issues) == retired == set(eligible), 'independent compute lifecycle is incomplete')
    return {(connection, sequence, opcode) for (connection, sequence), (opcode, _) in eligible.items()}


def issue_depth_receipt(path, job, *, expected_events=False, expected_generations=False,
                        expected_preparation=False, expected_selections=False):
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid()
            and stat.S_IMODE(info.st_mode) == 0o600
            and info.st_size > 0
            and (info.st_size <= 512 * 1024 ** 2 if expected_events else info.st_size < 8 * 1024 ** 2),
            'owned bounded depth output required')
    inputs = BoundInputs()
    document = inputs.read(path)
    result = validate_issue_depth(document, job, expected_events=expected_events,
                                  expected_generations=expected_generations,
                                  expected_preparation=expected_preparation,
                                  expected_selections=expected_selections)
    if expected_events:
        audit = path.with_name('server.log')
        checksum = digest(audit)
        require(checksum == job['artifact_identities']['server.log'], 'independent audit identity changed')
        inputs.identities[str(audit)] = checksum
        independent = issue_event_census(audit.read_text(), job)
        require(independent == {tuple(row[:3]) for row in document['events']['records']},
                'owner-event coverage differs from independent admission and retirement census')
        if expected_selections:
            training = set(job['effective_selection']['low'])
            exercised = sum(row[0] in training and row[14]-row[13] > 1
                            for row in document['events']['records'])
            require(exercised > 0, 'training application did not exercise earlier selection observations')
            result['events']['selections']['training_waiting_commands'] = exercised
        elif expected_preparation:
            # The authenticated low-role application in this collector is BERT training.
            training = set(job['effective_selection']['low'])
            exercised = sum(row[0] in training and row[2] == 11 and row[15] == 1
                            for row in document['events']['records'])
            require(exercised > 0, 'training application did not exercise the deferred preparation boundary')
            result['events']['preparation_boundary']['training_deferred_commands'] = exercised
        result['events']['independent_commands'] = len(independent)
        result['events']['independent_audit'] = dict(path=str(audit), sha256=checksum)
    inputs.verify()
    return dict(path=str(path), sha256=inputs.identities[str(path)], summary=result)


def depth_numerical_control(original, common, h, attempt, mode, *args, **kwargs):
    """Keep the normal application oracle intact and bind the extra output."""
    if mode == 'native':
        return original(common, h, attempt, mode, *args, **kwargs)
    require(mode == 'equal', 'issue depth is limited to equal priority')
    path = attempt / mode / 'issue-depth.json'
    command = common.service_command

    def selected(*values, **options):
        return depth_service_command(command(*values, **options), path, events=ISSUE_EVENTS,
                                     selections=SELECTED_COMMANDS)

    common.service_command = selected
    try:
        job = original(common, h, attempt, mode, *args, **kwargs)
        require(job['server_command'].count(DEPTH_FLAG) == 1
                and job['server_command'].count(EVENTS_FLAG) == int(ISSUE_EVENTS)
                and job['server_command'].count(SELECTIONS_FLAG) == int(SELECTED_COMMANDS)
                and job['server_command'][job['server_command'].index(DEPTH_FLAG) + 1] == str(path),
                'actual numerical depth output differs')
        receipt = issue_depth_receipt(path, job, expected_events=ISSUE_EVENTS,
                                      expected_generations=OBSERVATION_GENERATIONS,
                                      expected_preparation=PREPARATION_BOUNDARY,
                                      expected_selections=SELECTED_COMMANDS)
        lifecycle.candidate.save(path.with_name('issue-depth-receipt.json'), receipt)
        return job
    finally:
        common.service_command = command


def intake_exit_receipt(path, job):
    """Bind one normal-exit transfer count to this exact service and raw log."""
    lifecycle.candidate.normal_cleanup(job)
    require(job['mode'] == 'equal' and job['server_command'].count(INTAKE_FLAG) == 1,
            'explicit equal intake service required')
    metadata = path.lstat()
    require(stat.S_ISREG(metadata.st_mode) and metadata.st_uid == os.getuid()
            and metadata.st_size <= 64 * 1024 * 1024, 'bounded owned regular service log required')
    checksum = digest(path)
    require(checksum == job['artifact_identities']['server.log'], 'service log identity differs')
    lines = path.read_text().splitlines()
    prefix = r'\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}\] \[info\] '
    receipts = [line for line in lines if 'Async intake receipt:' in line]
    ready = [line for line in lines if 'Execution service ready:' in line]
    require(len(receipts) == len(ready) == 1, 'exactly one startup and intake receipt required')
    match = re.fullmatch(prefix + r'Async intake receipt: schema=1, pid=([1-9][0-9]*), '
                        r'start=([1-9][0-9]*), enabled=1, successful_second_takes=(0|[1-9][0-9]*)',
                        receipts[0])
    require(match is not None, 'canonical versioned intake receipt required')
    pid, start, count = map(int, match.groups())
    identity = job['server_identity']
    require((pid, start) == (identity['pid'], identity['start'])
            and re.fullmatch(prefix + rf'Execution service ready: pid={pid}, start={start}; '
                             r'routing=explicit_opt_in, application_qualified=false', ready[0])
            and lines.index(ready[0]) < lines.index(receipts[0]), 'intake service identity/order differs')
    require(0 < count < 2**64 - 1, 'application did not exercise an unsaturated second-round take')
    require(digest(path) == checksum, 'service log changed during receipt read')
    return dict(schema=1, service=dict(pid=pid, start=start), enabled=True,
                successful_second_takes=count, path=str(path), sha256=checksum)


def intake_numerical_control(original, common, h, attempt, mode, *args, **kwargs):
    job = original(common, h, attempt, mode, *args, **kwargs)
    if mode == 'native':
        return job
    require(mode == 'equal', 'intake exercise is limited to equal priority')
    receipt = intake_exit_receipt(attempt / mode / 'server.log', job)
    with (attempt / mode / 'async-intake-receipt.json').open('x') as output:
        json.dump(receipt, output, indent=2)
        output.write('\n')
    return job


def validate_intake_plan(plan):
    require(plan['status'] == 'frozen_async_intake_application'
            and plan.get('application_purpose') == 'ordinary_async_intake_exercise'
            and plan.get('async_intake_burst') is True and plan['helper_plans'] is True
            and all(plan.get(key, False) is False for key in
                    ('issue_depth', 'issue_events', 'observation_generations',
                     'preparation_boundary', 'selected_commands'))
            and plan['diagnostic_only'] is True and plan['full_device_qualified'] is False
            and plan['performance_qualified'] is False,
            'intake exercise requires an explicit observer-off diagnostic plan')


def intake_service_command(result):
    argv, pins = helper_service_command(result, True)
    require(INTAKE_FLAG not in argv, 'intake option already supplied')
    return [*argv, INTAKE_FLAG], pins


def validate_poll_paths(argv):
    flags = (POLL_FLAG, '--host-stage-output', '--issue-depth-output', '--profile-output',
             '--ptx-cache-output', '--pair-profile-output', '--socket')
    paths = []
    for flag in flags:
        require(argv.count(flag) <= 1, 'duplicate output option')
        if flag in argv:
            index = argv.index(flag)
            require(index + 1 < len(argv), 'output path missing')
            path = Path(argv[index + 1])
            require(path.is_absolute() and str(path) == argv[index + 1], 'canonical output spelling required')
            paths.append(path.resolve())
    require(len(paths) == len(set(paths)), 'diagnostic and execution outputs must be pairwise distinct')


def validate_poll_plan(plan):
    require(plan['status'] == 'frozen_poll_boundary_application'
            and plan.get('application_purpose') == 'accepted_launch_poll_boundary'
            and plan.get('decision_role') == 'inference'
            and all(plan.get(name, False) is False for name in ('helper_plans', 'issue_depth',
                'issue_events', 'observation_generations', 'preparation_boundary',
                'selected_commands', 'async_intake_burst'))
            and plan['diagnostic_only'] is True and plan['full_device_qualified'] is False
            and plan['performance_qualified'] is False, 'explicit isolated poll diagnostic purpose required')
    source = plan.get('analysis_source', {})
    analyzer = PROJECT_ROOT / 'scripts/analyze_bert_poll_boundary.py'
    require(source.get('path') == str(analyzer) and source.get('sha256') == digest(analyzer),
            'poll analyzer must match the source identity frozen directly in the plan')


def poll_stage_hooks(prior_hooks):
    @contextmanager
    def hooks(common, sustained, guard, directory, mode):
        with prior_hooks(common, sustained, guard, directory, mode):
            original = sustained.owner_command

            def owner_command(*args, **kwargs):
                argv, pins = original(*args, **kwargs)
                require(mode == 'equal' and POLL_FLAG not in argv, 'one explicit observed equal service required')
                argv = [*argv, POLL_FLAG, str(directory / 'poll-journal.bin')]
                validate_poll_paths(argv)
                return argv, pins

            if mode == 'equal':
                sustained.owner_command = owner_command
            try:
                yield
            finally:
                sustained.owner_command = original
    return hooks


def poll_exit_receipt(path, job):
    import analyze_bert_poll_boundary as poll_boundary
    require(job['mode'] == 'equal' and job['server_command'].count(POLL_FLAG) == 1
            and job['server_returncode'] == 0 and job['socket_retained'] is False,
            'normally completed equal diagnostic service required')
    validate_poll_paths(job['server_command'])
    for role in ('training', 'inference', 'service'):
        cleanup = job['authenticated_child_cleanup'][role]
        identity = job['server_identity'] if role == 'service' else job['application_identities'][role]
        require(type(job['final_returncodes'][role]) is int and job['final_returncodes'][role] == 0
                and cleanup['exit_confirmed'] is True and cleanup['signals'] == []
                and (cleanup['pid'], cleanup['start']) == (identity['pid'], identity['start']),
                'normal unsignalled exact diagnostic child cleanup required')
    metadata = path.lstat()
    require(stat.S_ISREG(metadata.st_mode) and metadata.st_uid == os.getuid()
            and metadata.st_size <= 64 * 1024**2, 'bounded owned service log required')
    checksum = digest(path)
    require(checksum == job['artifact_identities']['server.log'], 'service log identity differs')
    lines = path.read_text().splitlines()
    receipts = [line for line in lines if 'Poll journal receipt:' in line]
    ready = [line for line in lines if 'Execution service ready:' in line]
    require(len(receipts) == len(ready) == 1, 'exactly one startup and poll receipt required')
    match = re.fullmatch(r'Poll journal receipt: schema=1, pid=([1-9][0-9]*), start=([1-9][0-9]*), '
        r'complete=1, cycles=([1-9][0-9]*), polls=([1-9][0-9]*), peers=([1-9][0-9]*), '
        r'reads=([1-9][0-9]*), frames=([1-9][0-9]*)', receipts[0])
    require(match is not None, 'malformed poll terminal receipt')
    pid, start, *counts = map(int, match.groups())
    identity = job['server_identity']
    prefix = r'\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}\] \[info\] '
    require((pid, start) == (identity['pid'], identity['start'])
            and re.fullmatch(prefix + rf'Execution service ready: pid={pid}, start={start}; '
                             r'routing=explicit_opt_in, application_qualified=false', ready[0])
            and lines.index(ready[0]) < lines.index(receipts[0]), 'poll service identity/order differs')
    output = Path(job['server_command'][job['server_command'].index(POLL_FLAG) + 1])
    require(output == path.parent / 'poll-journal.bin', 'exact bounded poll output location required')
    output_hash = digest(output)
    with poll_boundary.Journal(output, identity, application=True) as journal:
        require(tuple(journal.counts.values()) == tuple(counts), 'normal receipt row counts differ')
        result = dict(path=str(output), sha256=output_hash, counts=journal.counts, service=identity,
                      log=dict(path=str(path), sha256=checksum), schema=1, complete=True)
    require(digest(output) == output_hash and digest(path) == checksum, 'poll evidence changed during validation')
    return result


def validate_environment(job, directory=None):
    environments = [job['application_environment'], *job.get('application_environments', {}).values()]
    for environment in environments:
        require('LD_AUDIT' not in environment and 'GLIMMER_EXECUTION_HOST_RANGE_DETAIL' not in environment,
                'unexpected loader or detailed observer')
        if job['mode'] == 'native':
            require('server_command' not in job and 'LD_PRELOAD' not in environment
                    and not any(key.startswith('GLIMMER_EXECUTION_') for key in environment),
                    'native observer/runtime contamination')
        else:
            require(job['mode'] == 'equal' and environment.get('LD_PRELOAD') == str(lifecycle.PRELOAD)
                    and environment.get('GLIMMER_EXECUTION_HOST_RANGE_SEEK') == '1', 'equal runtime differs')
            if directory is None:
                require(lifecycle.OUTPUT_ENV not in environment, 'numerical observer must be off')
            else:
                require(environment.get(lifecycle.OUTPUT_ENV) == str(directory), 'wrong observer output')
    if job['mode'] == 'equal':
        argv = job['server_command']
        require(argv[0] == str(lifecycle.SERVER), 'wrong service path')
        require(argv.count(HELPER_FLAG) == int(HELPER_PLANS is True), 'actual helper-plan switch differs')
        require(argv.count(DEPTH_FLAG) == int(ISSUE_DEPTH), 'actual issue-depth selection differs')
        require(argv.count(EVENTS_FLAG) == int(ISSUE_EVENTS), 'actual owner-event selection differs')
        require(argv.count(SELECTIONS_FLAG) == int(SELECTED_COMMANDS), 'actual selected-command journal differs')
        require(argv.count(INTAKE_FLAG) == int(INTAKE_BURST), 'actual async-intake selection differs')
        require(argv.count(POLL_FLAG) == int(POLL_BOUNDARY and directory is not None),
                'poll observation must be explicit and absent from numerical runs')
        if POLL_FLAG in argv:
            require(argv[argv.index(POLL_FLAG) + 1] == str(directory / 'poll-journal.bin'),
                    'wrong poll observer output')
            validate_poll_paths(argv)
        require(not INTAKE_BURST or directory is None, 'intake exercise requires observers off')
        require(not ISSUE_EVENTS or directory is None, 'owner events are numerical-only in this collector')
        if directory is None:
            require('--host-stage-output' not in argv, 'numerical service observer must be off')
        else:
            require(argv.count('--host-stage-output') == 1
                    and argv[argv.index('--host-stage-output') + 1] == str(directory / 'host-stages.json'),
                    'wrong service observer output')


def worker(attempt, mode):
    result = intake.original_worker(attempt, mode)
    if result:
        return result
    path = attempt / (mode + '-worker.json')
    report = lifecycle.collector.read(path)
    try:
        validate_environment(report['job'], attempt / mode)
        if mode == 'equal':
            receipt = report['stage_output']
            require(digest(Path(receipt['path'])) == receipt['sha256'], 'service ledger changed')
            report['intake_records'] = validate_intake(json.loads(Path(receipt['path']).read_text()))
            report['delivery_records'] = {}
            for role, receipt in report['frontend_outputs'].items():
                require(digest(Path(receipt['path'])) == receipt['sha256'], 'frontend ledger changed')
                report['delivery_records'][role] = len(validate_delivery(json.loads(Path(receipt['path']).read_text())))
            if POLL_BOUNDARY:
                report['poll_journal'] = poll_exit_receipt(attempt / mode / 'server.log', report['job'])
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        result = 1
    lifecycle.candidate.save(path, report)
    return result


def configure_plan(path):
    global HELPER_PLANS, ISSUE_DEPTH, ISSUE_EVENTS, OBSERVATION_GENERATIONS, PREPARATION_BOUNDARY, SELECTED_COMMANDS
    global INTAKE_BURST, POLL_BOUNDARY
    plan = lifecycle.collector.read(path)
    helper_plan = plan['status'] == 'frozen_helper_planning_application'
    intake_plan = plan['status'] == 'frozen_async_intake_application'
    poll_plan = plan['status'] == 'frozen_poll_boundary_application'
    require((intake_plan or helper_plan or poll_plan or plan['status'] == 'frozen_delivery_boundary_application')
            and plan['modes'] == ['native', 'equal']
            and plan['build'] == str(PROJECT_ROOT / 'build/candidate')
            and plan['runtime_environment'] == {'GLIMMER_EXECUTION_HOST_RANGE_SEEK': '1'}, 'explicit delivery plan required')
    seal_path = path.with_name('build-seal.json')
    seal = lifecycle.collector.read(seal_path)
    require(seal['plan_sha256'] == digest(path), 'delivery plan changed')
    require(all(digest(Path(name)) == checksum for name, checksum in seal['identities'].items()), 'delivery input changed')
    HELPER_PLANS = None
    ISSUE_DEPTH = plan.get('issue_depth', False)
    ISSUE_EVENTS = plan.get('issue_events', False)
    OBSERVATION_GENERATIONS = plan.get('observation_generations', False)
    PREPARATION_BOUNDARY = plan.get('preparation_boundary', False)
    SELECTED_COMMANDS = plan.get('selected_commands', False)
    INTAKE_BURST = plan.get('async_intake_burst', False)
    POLL_BOUNDARY = poll_plan
    require(type(INTAKE_BURST) is bool and INTAKE_BURST == intake_plan,
            'intake burst requires its explicit application plan')
    require(type(ISSUE_DEPTH) is bool and (not ISSUE_DEPTH or helper_plan and plan['helper_plans'] is True),
            'issue depth requires the explicit reviewed helper-on variant')
    require(type(ISSUE_EVENTS) is bool and (not ISSUE_EVENTS or ISSUE_DEPTH),
            'owner events require the explicit reviewed depth variant')
    require(type(OBSERVATION_GENERATIONS) is bool and (not OBSERVATION_GENERATIONS or ISSUE_EVENTS),
            'observation generations require the explicit reviewed event variant')
    require(type(PREPARATION_BOUNDARY) is bool and (not PREPARATION_BOUNDARY or OBSERVATION_GENERATIONS),
            'preparation boundary requires the explicit reviewed generation variant')
    require(type(SELECTED_COMMANDS) is bool and (not SELECTED_COMMANDS or PREPARATION_BOUNDARY
            and plan.get('application_purpose') == 'selected_command_observations'),
            'selected commands require their explicit reviewed application purpose')
    if poll_plan:
        validate_poll_plan(plan)
        analyzer = PROJECT_ROOT / 'scripts/analyze_bert_poll_boundary.py'
        require(str(analyzer) in seal['identities'] and digest(analyzer) == seal['identities'][str(analyzer)],
                'poll analyzer must be part of the frozen inputs')
        verified_path = Path(plan['cpu_verification'])
        verified = lifecycle.collector.read(verified_path)
        require(digest(verified_path) == plan['cpu_verification_sha256']
                and verified['status'] == 'poll_boundary_integration_cpu_independently_accepted'
                and verified['service_sha256'] == plan['server_sha256']
                and verified['frontend_sha256'] == plan['frontend_sha256'],
                'current independently accepted CPU poll integration required')
        lifecycle.collector.stage_hooks = poll_stage_hooks(lifecycle.collector.stage_hooks)
    elif intake_plan:
        validate_intake_plan(plan)
        closure_path = Path(plan['cpu_closure'])
        closure = lifecycle.collector.read(closure_path)
        require(digest(closure_path) == plan['cpu_closure_sha256']
                and closure['status'] == 'two_round_async_intake_cpu_independently_reviewed',
                'reviewed two-round CPU evidence required')
        HELPER_PLANS = True
        common, _ = lifecycle.stage.BASE_HELPERS()
        original_command = common.service_command

        def command(*args, **kwargs):
            return intake_service_command(original_command(*args, **kwargs))

        common.service_command = command
        original_control = lifecycle.candidate.guarded_control

        def control(*args, **kwargs):
            return intake_numerical_control(original_control, *args, **kwargs)

        lifecycle.candidate.guarded_control = control
    elif helper_plan:
        require(type(plan['helper_plans']) is bool and plan['diagnostic_only'] is True
                and plan['performance_qualified'] is False and plan['full_device_qualified'] is False,
                'explicit diagnostic helper plan required')
        closure_path = Path(plan['cpu_closure'])
        closure = lifecycle.collector.read(closure_path)
        require(digest(closure_path) == plan['cpu_closure_sha256']
                and closure['status'] == ('selected_command_cpu_independently_reviewed' if SELECTED_COMMANDS
                                         else 'preparation_boundary_cpu_independently_reviewed'
                                         if PREPARATION_BOUNDARY else 'observation_generation_cpu_independently_reviewed'
                                         if OBSERVATION_GENERATIONS else 'cpu_verified_and_independently_reviewed')
                and closure['service_sha256'] == plan['server_sha256']
                and closure['frontend_sha256'] == plan['frontend_sha256'],
                'current independently reviewed CPU candidate required')
        HELPER_PLANS = plan['helper_plans']
        common, _ = lifecycle.stage.BASE_HELPERS()
        original_command = common.service_command

        def command(*args, **kwargs):
            return helper_service_command(original_command(*args, **kwargs), HELPER_PLANS)

        common.service_command = command
        if ISSUE_DEPTH:
            original_control = lifecycle.candidate.guarded_control

            def control(*args, **kwargs):
                return depth_numerical_control(original_control, *args, **kwargs)

            lifecycle.candidate.guarded_control = control
    else:
        require('helper_plans' not in plan, 'historical delivery plan cannot select helper optimization')
        overhead_path = Path(plan['overhead_report'])
        overhead = lifecycle.collector.read(overhead_path)
        require(digest(overhead_path) == plan['overhead_sha256']
                and overhead['status'] == 'passed_requires_independent_review'
                and set(overhead['medians']) == {'wall', 'cpu'}
                and all(0 < value <= 1.20 for value in overhead['medians'].values()), 'incremental overhead gate failed')
    lifecycle.ENTRY, lifecycle.PLAN, lifecycle.BUILD = ENTRY, path, Path(plan['build'])
    lifecycle.SERVER = lifecycle.BUILD / 'bin/glimmer_execution_service'
    lifecycle.PRELOAD = lifecycle.BUILD / 'lib/libglimmer_execution_interceptor.so'
    require(digest(lifecycle.SERVER) == plan['server_sha256'] and digest(lifecycle.PRELOAD) == plan['frontend_sha256'],
            'candidate binary changed')
    lifecycle.OUTPUT_SCHEMA, lifecycle.RUNTIME_ENV, lifecycle.EXTRA_ENV = 6, plan['runtime_environment'], {}
    lifecycle.collector.STAGE_PLAN = path
    lifecycle.stage.DEADLINE = datetime.fromisoformat(plan['ordinary_run_deadline_utc'])
    lifecycle.stage.build_identity = lambda: build_identity('service')

    def frontend_identity():
        result = build_identity('frontend')
        result['identities'].update(seal['identities'])
        result['identities'][str(seal_path)] = digest(seal_path)
        return result

    lifecycle.frontend_identity = frontend_identity
    lifecycle.worker = worker
    lifecycle.analyze = analyze
    original_cleanup = lifecycle.candidate.normal_cleanup

    def cleanup(job):
        original_cleanup(job)
        validate_environment(job)

    lifecycle.candidate.normal_cleanup = cleanup


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--check', action='store_true')
    args, remaining = parser.parse_known_args()
    if args.plan is None:
        option = '--worker-attempt' if '--worker-attempt' in remaining else '--analyze'
        require(option in remaining, 'explicit --plan required')
        attempt = Path(remaining[remaining.index(option) + 1])
        document = attempt / 'plan.json'
        scope = (lifecycle.collector.read(document)['lifecycle_scope'] if document.is_file()
                 else lifecycle.collector.read(attempt / 'numerical-scope.json'))
        args.plan = Path(scope['plan']['path'])
        require(digest(args.plan) == scope['plan']['sha256'], 'worker delivery plan changed')
    configure_plan(args.plan.resolve(strict=True))
    if args.check:
        require(not remaining, 'check is CPU-only')
        print(json.dumps(lifecycle.configure(), indent=2))
        return 0
    require(not (ISSUE_DEPTH or INTAKE_BURST)
            or not any(option in remaining for option in ('--collect', '--analyze')),
            'hardware collection requires its separately reviewed device plan')
    sys.argv = [str(ENTRY), *remaining]
    return lifecycle.main()


if __name__ == '__main__':
    raise SystemExit(main())
