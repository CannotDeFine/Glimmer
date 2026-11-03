#!/usr/bin/env python3
"""Pure device-wide ratio/window arithmetic, never capture or qualification.

This module does not query a GPU or import vendor bindings. Callers must
separately establish source/device/owner identities and application windows.
Field 1005 measures memory-interface activity, not bytes/second. Aggregate
ratios alone cannot establish per-SM coverage or sustained saturation.
Guard records are ignored; this module does not authenticate their contents.
"""

import argparse
import bisect
import json
import math
from pathlib import Path

FIELDS = (1001, 1002, 1003, 1005)
PERIOD_US = 200000
NO_DATA_BLANK = 140737488355328.0
MAX_RECORDS = 1_000_000
MAX_ANCHORS = 250_000
MAX_SAMPLES_PER_FIELD = 200_000


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_object(items):
    result = {}
    for key, value in items:
        require(key not in result, 'duplicate JSON object key')
        result[key] = value
    return result


def reject_constant(_):
    raise ValueError('nonfinite JSON constant')


def read_records(path):
    """Bounded complete JSONL input; importing this module reads no files."""
    with Path(path).open('rb') as stream:
        content = stream.read(16 * 1024 ** 2 + 1)
    require(len(content) <= 16 * 1024 ** 2 and content.endswith(b'\n'), 'bounded complete sample journal required')
    rows = []
    for line in content.splitlines():
        require(0 < len(line) <= 16384, 'bounded nonempty sample line required')
        row = json.loads(line, object_pairs_hook=unique_object, parse_constant=reject_constant)
        require(type(row) is dict, 'sample records must be objects')
        require(len(rows) < MAX_RECORDS, 'too many telemetry records')
        rows.append(row)
    return rows


def validate_row(row):
    require(type(row.get('gpu_id')) is int and row['gpu_id'] == 0
            and type(row.get('field_id')) is int and row['field_id'] in FIELDS
            and type(row.get('status')) is int and row['status'] == 0
            and type(row.get('field_type')) is int and row['field_type'] == ord('d')
            and type(row.get('timestamp_us')) is int and row['timestamp_us'] > 0
            and type(row.get('value')) in (int, float) and math.isfinite(row['value'])
            and 0 <= row['value'] <= 1, 'invalid profiling sample')


def is_no_data(row):
    """DCGM's empty-history response, not a collected zero/blank ratio."""
    return (type(row.get('gpu_id')) is int and row['gpu_id'] == 0
            and type(row.get('field_id')) is int and row['field_id'] in FIELDS
            and type(row.get('field_type')) is int and row['field_type'] == ord('d')
            and type(row.get('status')) is int and row['status'] == -14
            and type(row.get('timestamp_us')) is int and row['timestamp_us'] == 0
            and type(row.get('value')) is float and row['value'] == NO_DATA_BLANK
            and row.get('raw_value_repr') == repr(NO_DATA_BLANK))


def validate_no_data(row):
    require(set(row) == {'kind', 'gpu_id', 'field_id', 'field_type', 'status', 'timestamp_us',
                        'value', 'raw_value_repr', 'observed_monotonic_ns', 'callback_count',
                        'row_index', 'query_index', 'callback_index', 'query_cursor_us', 'next_query_cursor_us'}
            and row['kind'] == 'no_data' and is_no_data(row), 'invalid no_data response')
    require(all(type(row[key]) is int for key in ('observed_monotonic_ns', 'callback_count',
                'row_index', 'query_index', 'callback_index', 'query_cursor_us', 'next_query_cursor_us'))
            and row['observed_monotonic_ns'] > 0 and 0 < row['callback_count'] <= 4096
            and row['query_index'] > 0 and row['callback_index'] > 0
            and 0 <= row['row_index'] < row['callback_count']
            and 0 <= row['query_cursor_us'] <= row['next_query_cursor_us'],
            'invalid no_data callback/cursor provenance')


def clock_series(records):
    anchors = []
    for row in records:
        if row.get('kind') == 'anchor':
            require(len(anchors) < MAX_ANCHORS, 'too many clock anchors')
            anchors.append(row)
    require(len(anchors) >= 2, 'missing clock brackets')
    previous = None
    for row in anchors:
        require(all(type(row.get(key)) is int and row[key] > 0 for key in ('realtime_ns', 'monotonic_ns'))
                and type(row.get('uncertainty_ns')) is int and 0 <= row['uncertainty_ns'] <= 5000000,
                'invalid or excessively uncertain clock bracket')
        if previous is not None:
            elapsed = row['monotonic_ns'] - previous['monotonic_ns']
            wall = row['realtime_ns'] - previous['realtime_ns']
            require(elapsed > 0 and wall > 0 and abs(elapsed - wall) <= max(1000000, elapsed // 1000),
                    'clock step, reversal or invalid alignment')
        previous = row
    return anchors


def align(timestamp_us, anchors):
    require(2 <= len(anchors) <= MAX_ANCHORS, 'invalid clock anchor count')
    return _align(timestamp_us, anchors, [row['realtime_ns'] for row in anchors])


def _align(timestamp_us, anchors, times):
    stamp = timestamp_us * 1000
    index = bisect.bisect_right(times, stamp) - 1
    if index == len(anchors) - 1 and stamp == times[-1]:
        return anchors[-1]['monotonic_ns']
    require(0 <= index < len(anchors) - 1, 'sample lacks bounding clock anchors')
    left, right = anchors[index:index + 2]
    return left['monotonic_ns'] + ((stamp - left['realtime_ns']) *
        (right['monotonic_ns'] - left['monotonic_ns']) // (right['realtime_ns'] - left['realtime_ns']))


def window_summary(stream, begin_ns, end_ns):
    require(type(begin_ns) is int and type(end_ns) is int and 0 < begin_ns < end_ns
            and len(stream) >= 3 and stream[0]['time_ns'] <= begin_ns and stream[-1]['time_ns'] >= end_ns,
            'missing samples bracketing the common window')
    weighted, covered, values, gaps = 0.0, 0, [], []
    for previous, current in zip(stream, stream[1:]):
        delta = current['time_ns'] - previous['time_ns']
        require(delta > 0, 'duplicate or reversed field sampling time')
        if previous['time_ns'] < end_ns and current['time_ns'] > begin_ns:
            gaps.append(delta)
            require(delta <= 3 * PERIOD_US * 1000, 'missing counter updates inside window')
        if begin_ns <= previous['time_ns'] < current['time_ns'] <= end_ns:
            weighted += current['value'] * delta
            covered += delta
            values.append(current['value'])
    duration = end_ns - begin_ns
    missing = duration - covered
    require(covered and values and 0 <= missing <= 6 * PERIOD_US * 1000,
            'insufficient complete interior sampling intervals')
    return dict(complete_interval_samples=len(values), positive_samples=sum(value > 0 for value in values),
        minimum=min(values), maximum=max(values), interior_duration_weighted_ratio=weighted / covered,
        interior_coverage_fraction=covered / duration, unknown_boundary_ns=missing,
        whole_window_ratio_lower_bound=weighted / duration,
        whole_window_ratio_upper_bound=min(1.0, (weighted + missing) / duration),
        maximum_overlapping_gap_ns=max(gaps),
        integration_convention='each recorded ratio applies to the preceding collection interval; edge bins remain unknown')


def analyze_records(records, begin_ns, end_ns, ready_ns):
    require(type(ready_ns) is int and type(begin_ns) is int and 0 < ready_ns <= begin_ns,
            'counter readiness must precede the actual offered window')
    require(type(records) is list and len(records) <= MAX_RECORDS, 'too many telemetry records')
    require(all(type(row) is dict and row.get('kind') in {'anchor', 'guard', 'sample', 'no_data'} for row in records),
            'unknown telemetry event')
    anchors = clock_series(records)
    anchor_times = [row['realtime_ns'] for row in anchors]
    streams, timestamps = {field: [] for field in FIELDS}, {field: set() for field in FIELDS}
    sample_counts = {field: 0 for field in FIELDS}
    no_data_count, previous_absence = 0, None
    for row in records:
        if row.get('kind') == 'no_data':
            validate_no_data(row)
            if previous_absence is not None:
                same_query = row['query_index'] == previous_absence['query_index']
                same_callback = same_query and row['callback_index'] == previous_absence['callback_index']
                order = ('query_index', 'callback_index', 'row_index')
                cursor_ordered = ((row['query_cursor_us'], row['next_query_cursor_us']) ==
                                  (previous_absence['query_cursor_us'], previous_absence['next_query_cursor_us'])
                                  if same_query else row['query_cursor_us'] >= previous_absence['next_query_cursor_us'])
                require(row['observed_monotonic_ns'] >= previous_absence['observed_monotonic_ns']
                        and tuple(row[key] for key in order) > tuple(previous_absence[key] for key in order)
                        and (not same_callback or row['callback_count'] == previous_absence['callback_count'])
                        and cursor_ordered,
                        'duplicate, reversed or backwards-cursor no_data event')
            previous_absence = row
            no_data_count += 1
            continue
        if row.get('kind') != 'sample':
            continue
        field = row.get('field_id')
        if type(field) is int and field in sample_counts:
            sample_counts[field] += 1
            require(sample_counts[field] <= MAX_SAMPLES_PER_FIELD, 'too many field samples')
        require(type(row.get('observed_monotonic_ns')) is int and row['observed_monotonic_ns'] > 0,
                'missing sample observation time')
        try:
            validate_row(row)
        except ValueError:
            require(row['observed_monotonic_ns'] < ready_ns, 'invalid sample after readiness')
            continue
        if row['timestamp_us'] * 1000 < anchors[0]['realtime_ns']:
            require(row['observed_monotonic_ns'] < ready_ns, 'post-readiness sample predates clock brackets')
            continue
        timestamp = _align(row['timestamp_us'], anchors, anchor_times)
        lag = row['observed_monotonic_ns'] - timestamp
        require(-10000000 <= lag <= 3 * PERIOD_US * 1000, 'stale or future profiling sample')
        field = row['field_id']
        require(row['timestamp_us'] not in timestamps[field], 'duplicate field collection timestamp')
        timestamps[field].add(row['timestamp_us'])
        streams[field].append(dict(time_ns=timestamp, value=row['value']))
    summaries = {str(field): window_summary(stream, begin_ns, end_ns) for field, stream in streams.items()}
    require(summaries['1002']['positive_samples'] > 0,
            'no positive window SM signal; cannot infer idle hardware or borrow a solo signal')
    return dict(fields=summaries, clock_anchors=len(anchors), no_data_events=no_data_count,
                maximum_clock_bracket_uncertainty_ns=max(row['uncertainty_ns'] for row in anchors),
                ownership_qualified=False, full_device_qualified=False, saturation_qualified=False,
                performance_qualified=False, current_candidate_qualified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=Path, required=True)
    for name in ('begin-ns', 'end-ns', 'ready-ns'):
        parser.add_argument('--' + name, type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze_records(read_records(args.samples), args.begin_ns, args.end_ns, args.ready_ns),
                     indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
