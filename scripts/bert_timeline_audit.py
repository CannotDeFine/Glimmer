"""Read-only matched-window CUDA/OSRT decomposition; never performance qualification."""

from collections import Counter, defaultdict
import importlib
from pathlib import Path
import sqlite3
import sys

import run_bert_candidate_validation as candidate
from run_rnn_v8_validation import digest
from run_bert_device_observation import read, require

MASK = -(1 << 24)


def merged(intervals, low, high):
    result = []
    for begin, end in sorted((max(a, low), min(b, high)) for a, b in intervals if a < high and b > low):
        if result and begin <= result[-1][1]:
            result[-1][1] = max(end, result[-1][1])
        else:
            result.append([begin, end])
    return result


def total(intervals):
    return sum(b - a for a, b in intervals)


def intersection(left, right):
    result, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        a, b = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
        if a < b:
            result.append([a, b])
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return result


def complement(intervals, low, high):
    result, cursor = [], low
    for begin, end in intervals:
        if cursor < begin:
            result.append([cursor, begin])
        cursor = max(cursor, end)
    if cursor < high:
        result.append([cursor, high])
    return result


def distribution(values):
    values = sorted(values)
    if not values:
        return dict(count=0, total_ns=0, maximum_ns=0)
    return dict(count=len(values), total_ns=sum(values), maximum_ns=values[-1],
                p50_ns=values[(len(values) - 1) // 2], p95_ns=values[(len(values) * 95 - 1) // 100])


def concurrency(intervals, low, high):
    events = Counter({low: 0, high: 0})
    for a, b in intervals:
        if a < high and b > low:
            events[max(a, low)] += 1
            events[min(b, high)] -= 1
    durations, depth, previous = Counter(), 0, low
    for stamp, change in sorted(events.items()):
        durations[depth] += stamp - previous
        depth += change
        previous = stamp
        require(depth >= 0, 'negative activity concurrency')
    require(depth == 0, 'unclosed activity concurrency')
    return {str(k): v for k, v in sorted(durations.items()) if v}


def decompose(activities, osrt, low, high):
    intervals = [(a['start'], a['end']) for a in activities]
    busy = merged(intervals, low, high)
    gaps = complement(busy, low, high)
    pending = merged([(a['api_end'], a['start']) for a in activities if a['api_end'] < a['start']], low, high)
    pending_gaps = intersection(gaps, pending)
    no_pending = intersection(gaps, complement(pending, low, high))
    issuing = {a['tid'] for a in activities}
    calls = [row for row in osrt if row['globalTid'] in issuing]
    waits = [row for row in calls if any(token in row['name'].lower()
              for token in ('poll', 'futex', 'sem_', 'recv', 'cond_wait', 'cond_timedwait', 'nanosleep'))]
    wait_union = merged([(r['start'], r['end']) for r in waits], low, high)
    wait_gaps = intersection(no_pending, wait_union)
    durations = dict(returned_launch_pending=total(pending_gaps),
                     no_pending_with_recorded_wait_call=total(wait_gaps),
                     no_pending_without_recorded_wait_call=total(no_pending) - total(wait_gaps))
    gap_ns = total(gaps)
    require(sum(durations.values()) == gap_ns, 'gap categories do not partition window')
    primary = max(durations, key=durations.get)
    if not gap_ns or durations[primary] / gap_ns < .6:
        primary = 'mixed_or_ambiguous'
    kinds, names, apis, streams, shapes = {}, {}, {}, {}, {}
    for row in activities:
        elapsed = max(0, min(row['end'], high) - max(row['start'], low))
        if not elapsed:
            continue
        for groups, name in ((kinds, row['kind']), (names, row['name']), (apis, row['api_name'])):
            item = groups.setdefault(name, dict(count=0, clipped_duration_sum_ns=0))
            item['count'] += 1
            item['clipped_duration_sum_ns'] += elapsed
        if row['kind'] == 'kernel':
            stream = f"{row['global_pid']}:{row['context_id']}:{row['stream_id']}"
            shape = f"{row['name']} grid={row['grid']} block={row['block']}"
            for groups, key in ((streams, stream), (shapes, shape)):
                item = groups.setdefault(key, dict(count=0, clipped_duration_sum_ns=0))
                item['count'] += 1
                item['clipped_duration_sum_ns'] += elapsed
    osrt_groups = {}
    for row in calls:
        overlap = total(intersection(no_pending, merged([(row['start'], row['end'])], low, high)))
        if overlap:
            item = osrt_groups.setdefault(row['name'], dict(count=0, overlapping_call_duration_sum_ns=0))
            item['count'] += 1
            item['overlapping_call_duration_sum_ns'] += overlap
    return dict(window_ns=high-low, gpu_activity_union_ns=total(busy), gpu_activity_fraction=total(busy)/(high-low),
                gpu_gaps=distribution([b-a for a, b in gaps]), gap_categories_ns=durations,
                primary_gap_category_candidate=primary, primary_threshold=.6,
                concurrency_depth_duration_ns=concurrency(intervals, low, high), activity_kinds=kinds,
                kernel_concurrency_depth_duration_ns=concurrency(
                    [(a['start'], a['end']) for a in activities if a['kind'] == 'kernel'], low, high),
                kernel_process_context_streams=streams, kernel_shapes=shapes,
                gpu_names=names, activity_by_correlated_api=apis, issuing_thread_ids=sorted(issuing),
                issuing_thread_osrt_calls_during_no_pending_gaps=osrt_groups,
                limitation='OSRT intervals include execution and descheduling; overlapping calls are not CPU blocked time. A primary candidate also needs matched-control direction review.')


def analyze(attempt, mode):
    trace = read(attempt / (mode + '-trace.json'))
    require(trace['status'] == 'passed_requires_review', 'complete traced control required')
    job, clock_receipt = trace['job'], trace['capture_range']
    expected = ({job['server_identity']['pid']} if mode == 'equal' else
                {v['pid'] for v in job['application_identities'].values()})
    require(set(map(int, clock_receipt['owners'])) == expected, 'traced owner set differs')
    sys.path.insert(0, str(candidate.LEGACY))
    legacy = importlib.import_module('analyze_ordinary_timeline')
    database = attempt / (mode + '-timeline.sqlite')
    native_report = attempt / (mode + '-timeline.nsys-rep')
    require(database.is_file() and native_report.is_file(), 'raw profiler report/export missing')
    db = sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    try:
        require(db.execute('pragma integrity_check').fetchone()[0] == 'ok', 'corrupt profiler export')
        tables = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
        require({'PROCESSES', 'NVTX_EVENTS', 'StringIds', 'CUPTI_ACTIVITY_KIND_KERNEL',
                 'DIAGNOSTIC_EVENT', 'OSRT_API'} <= tables, 'incomplete CUDA/NVTX/OSRT schema')
        strings = dict(db.execute('select id,value from StringIds'))
        processes = defaultdict(set)
        for row in db.execute('select pid,globalPid from PROCESSES'):
            processes[row['pid']].add(row['globalPid'])
        controller = clock_receipt['controller_pid']
        require(all(len(processes[p]) == 1 for p in expected | {controller}), 'missing/ambiguous traced PID')
        controller_global = next(iter(processes[controller]))
        owners = {next(iter(processes[p])) for p in expected}
        diagnostics = [dict(row) for row in db.execute('select * from DIAGNOSTIC_EVENT')]
        legacy.validate_diagnostics(diagnostics, controller_global, os_runtime=True)
        marks, ranges = defaultdict(list), []
        for row in db.execute('select start,end,text,textId,globalTid from NVTX_EVENTS'):
            message = row['text'] if row['text'] is not None else strings.get(row['textId'], '')
            if message.startswith('GLIMMER_CLOCK_V1:'):
                require(row['end'] is None and row['globalTid'] & MASK == controller_global,
                        'foreign or noninstantaneous clock marker')
                marks[message].append((row['start'], row['globalTid']))
            if message == clock_receipt['range_name']:
                ranges.append(dict(row))
        require(len(ranges) == 1 and ranges[0]['end'] is not None
                and ranges[0]['globalTid'] & MASK == controller_global
                and ranges[0]['globalTid'] & ((1 << 24)-1) == clock_receipt['range_thread_id'],
                'one same-thread controller capture range required')
        clock = legacy.ClockMap(clock_receipt['clock_brackets'], marks, controller, controller_global)
        begin = clock.bound(clock_receipt['analysis_begin_ns'])
        end = clock.bound(clock_receipt['analysis_end_ns'])
        low, high = begin[1], end[0]
        require(ranges[0]['start'] < begin[0] <= low < high <= end[1] < ranges[0]['end']
                and high-low >= 4_900_000_000, 'capture does not bracket fixed inner window')
        apis = defaultdict(list)
        for table in ('CUPTI_ACTIVITY_KIND_DRIVER', 'CUPTI_ACTIVITY_KIND_RUNTIME'):
            if table not in tables:
                continue
            for row in db.execute(f'select start,end,globalTid,correlationId,nameId from {table}'):
                require(row['end'] >= row['start'], 'invalid API interval')
                gp = row['globalTid'] & MASK
                require(gp in owners, 'CUDA API from unapproved process')
                apis[(gp, row['correlationId'])].append((row['start'], row['end'], row['globalTid'], strings[row['nameId']]))
        for values in apis.values():
            values.sort()
        activity, observed = [], set()
        for table, kind in (('CUPTI_ACTIVITY_KIND_KERNEL', 'kernel'),
                            ('CUPTI_ACTIVITY_KIND_MEMCPY', 'copy'), ('CUPTI_ACTIVITY_KIND_MEMSET', 'memset')):
            if table not in tables:
                continue
            for row in db.execute('select * from ' + table):
                require(row['globalPid'] in owners and row['deviceId'] == 0, 'foreign GPU activity')
                observed.add(row['globalPid'])
                if row['end'] <= low:
                    continue
                require(row['end'] > row['start'], 'invalid activity interval')
                choices = apis[(row['globalPid'], row['correlationId'])]
                eligible = [value for value in choices if value[0] <= row['start']]
                require(eligible, 'in-window activity lacks causal same-process API')
                launch = max(eligible, key=lambda item: item[0])
                if row['start'] >= high and launch[1] >= high:
                    continue
                geometry = (dict(context_id=row['contextId'], stream_id=row['streamId'],
                            grid=[row[k] for k in ('gridX', 'gridY', 'gridZ')],
                            block=[row[k] for k in ('blockX', 'blockY', 'blockZ')]) if kind == 'kernel' else {})
                activity.append(dict(start=row['start'], end=row['end'], kind=kind, global_pid=row['globalPid'],
                    name=strings[row['demangledName']] if kind == 'kernel' else kind,
                    api_begin=launch[0], api_end=launch[1], tid=launch[2], api_name=launch[3], **geometry))
        require(observed == owners and activity, 'missing GPU owner activity')
        osrt = []
        for row in db.execute('select start,end,globalTid,nameId from OSRT_API where start<? and end>?', (high, low)):
            require(row['end'] >= row['start'], 'invalid OSRT interval')
            osrt.append(dict(row, name=strings[row['nameId']]))
        result = decompose(activity, osrt, low, high)
        result.update(status='bounded_timeline_diagnostic_reparsed', mode=mode,
            begin_bounds_ns=begin, end_bounds_ns=end, controller_pid=controller,
            owner_pids=sorted(expected), profiler_diagnostics=diagnostics,
            sqlite_sha256=digest(database), nsys_report_sha256=digest(native_report),
            full_device_qualified=False, saturation_qualified=False, performance_qualified=False,
            ordinary_timing_evidence=False)
        return result
    finally:
        db.close()
