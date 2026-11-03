"""CPU-only checks for bounded sample parsing and window arithmetic."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import aggregate_sm_audit as audit

WALL = 1000000000000
MONO = 1000000000


def fixture(value=.5):
    records = []
    for index in range(62):
        records.append(dict(kind='anchor', realtime_ns=WALL + index * 100000000,
                            monotonic_ns=MONO + index * 100000000, uncertainty_ns=100))
        if index % 2 == 0:
            for field in audit.FIELDS:
                records.append(dict(kind='sample', gpu_id=0, field_id=field, field_type=ord('d'),
                    status=0, timestamp_us=(WALL + index * 100000000) // 1000, value=value,
                    observed_monotonic_ns=MONO + index * 100000000 + 1000000))
    return records


def analyze(records):
    return audit.analyze_records(records, MONO + 1000000000, MONO + 5000000000, MONO + 50000000)


class AggregateAuditTests(unittest.TestCase):
    @staticmethod
    def no_data(**extra):
        return dict(kind='no_data', gpu_id=0, field_id=1001, field_type=100, status=-14,
                    timestamp_us=0, value=audit.NO_DATA_BLANK, raw_value_repr=repr(audit.NO_DATA_BLANK),
                    observed_monotonic_ns=MONO + 2500000000, callback_count=1, row_index=0,
                    query_index=1, callback_index=1,
                    query_cursor_us=1000000, next_query_cursor_us=1200000, **extra)

    def test_no_data_is_not_a_sample_zero_or_coverage_credit(self):
        records = fixture()
        baseline = analyze(records)
        response = self.no_data()
        result = analyze([*records, response])
        self.assertEqual(result['fields'], baseline['fields'])
        self.assertEqual(result['no_data_events'], 1)
        records = [row for row in records if row['kind'] != 'sample'
                   or not WALL + 2000000000 < row['timestamp_us'] * 1000 < WALL + 4000000000]
        with self.assertRaisesRegex(ValueError, 'missing counter updates'):
            analyze([*records, response])
        # Old raw sample records are not retroactively reclassified or passed.
        raw = {key: value for key, value in response.items() if key not in
               ('callback_count', 'row_index', 'query_index', 'callback_index',
                'query_cursor_us', 'next_query_cursor_us')}
        raw['kind'] = 'sample'
        with self.assertRaisesRegex(ValueError, 'invalid sample after readiness'):
            analyze([*fixture(), raw])

    def test_no_data_schema_mutations_duplicates_and_reversed_cursor_fail_everywhere(self):
        for key, value in (('gpu_id', 1), ('gpu_id', False), ('field_id', 1007), ('field_type', 105),
                           ('status', 0), ('status', -6), ('timestamp_us', 1), ('value', 0.),
                           ('value', audit.NO_DATA_BLANK + 1), ('value', float('nan')),
                           ('raw_value_repr', '0.0'), ('callback_count', 0), ('callback_count', 4097),
                           ('row_index', 1), ('row_index', -1), ('query_cursor_us', -1),
                           ('query_index', 0), ('callback_index', True),
                           ('next_query_cursor_us', 999999), ('observed_monotonic_ns', 0)):
            for at in (MONO + 1, MONO + 2500000000, MONO + 7000000000):
                row = dict(self.no_data(), observed_monotonic_ns=at)
                row[key] = value
                with self.subTest(key=key, at=at), self.assertRaises(ValueError):
                    analyze([*fixture(), row])
        row = self.no_data()
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            analyze([*fixture(), row, row])
        with self.assertRaisesRegex(ValueError, 'backwards-cursor'):
            analyze([*fixture(), row, dict(row, observed_monotonic_ns=MONO + 2600000000,
                                          query_cursor_us=999999)])
        result = analyze([*fixture(), row, dict(row, observed_monotonic_ns=MONO + 2700000000,
                                               query_index=2,
                                               query_cursor_us=1200000, next_query_cursor_us=1400000)])
        self.assertEqual(result['no_data_events'], 2)

    def test_same_timestamp_mixed_callbacks_and_stationary_cursor_are_distinct(self):
        row = dict(self.no_data(), callback_count=3, next_query_cursor_us=1000000)
        mixed = [row, dict(row, row_index=2), dict(row, callback_index=2, callback_count=1),
                 dict(row, query_index=2, callback_index=1, row_index=0)]
        self.assertEqual(analyze([*fixture(), *mixed])['no_data_events'], 4)
        for changed in (dict(row, observed_monotonic_ns=row['observed_monotonic_ns'] + 1),
                        dict(row, row_index=2, callback_count=4),
                        dict(row, query_index=2, query_cursor_us=999999)):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                analyze([*fixture(), row, changed])

    def test_complete_common_window_not_process_average(self):
        records = fixture()
        for row in records:
            if row['kind'] == 'sample' and not WALL + 1000000000 <= row['timestamp_us'] * 1000 <= WALL + 5000000000:
                row['value'] = .1
        result = analyze(records)
        for field in result['fields'].values():
            self.assertEqual(field['interior_duration_weighted_ratio'], .5)
            self.assertEqual(field['interior_coverage_fraction'], 1)
            self.assertEqual(field['whole_window_ratio_lower_bound'], .5)
            self.assertEqual(field['whole_window_ratio_upper_bound'], .5)
        for key in ('ownership_qualified', 'full_device_qualified', 'saturation_qualified',
                    'performance_qualified', 'current_candidate_qualified'):
            self.assertIs(result[key], False)

    def test_partial_edges_are_unknown_not_prorated(self):
        result = audit.analyze_records(fixture(), MONO + 1050000000, MONO + 4950000000, MONO + 50000000)
        for field in result['fields'].values():
            self.assertLess(field['interior_coverage_fraction'], 1)
            self.assertLess(field['whole_window_ratio_lower_bound'], .5)
            self.assertGreater(field['whole_window_ratio_upper_bound'], .5)
            self.assertGreater(field['unknown_boundary_ns'], 0)

    def test_duplicate_reversed_or_missing_internal_update(self):
        records = fixture()
        sample = next(row for row in records if row['kind'] == 'sample' and row['field_id'] == 1002)
        with self.assertRaises(ValueError):
            analyze([*records, copy.deepcopy(sample)])
        records = [row for row in fixture() if row['kind'] != 'sample'
                   or not WALL + 2000000000 < row['timestamp_us'] * 1000 < WALL + 4000000000]
        with self.assertRaises(ValueError):
            analyze(records)

    def test_missing_field_or_bracket(self):
        records = [row for row in fixture() if row['kind'] != 'sample' or row['field_id'] != 1005]
        with self.assertRaises(ValueError):
            analyze(records)
        with self.assertRaises(ValueError):
            audit.analyze_records(fixture(), MONO - 1, MONO + 5000000000, MONO - 2)

    def test_clock_reversal_step_and_uncertainty(self):
        for key, value in (('monotonic_ns', MONO - 1), ('realtime_ns', WALL - 1), ('uncertainty_ns', 5000001)):
            records = fixture()
            [row for row in records if row['kind'] == 'anchor'][20][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                analyze(records)
        records = fixture()
        for row in records:
            if row['kind'] == 'anchor' and row['monotonic_ns'] >= MONO + 2000000000:
                row['realtime_ns'] += 5000000
        with self.assertRaises(ValueError):
            analyze(records)

    def test_integer_clock_mapping_and_last_anchor(self):
        anchors = audit.clock_series(fixture())
        self.assertEqual(audit.align((WALL + 123456000) // 1000, anchors), MONO + 123456000)
        self.assertEqual(audit.align(anchors[-1]['realtime_ns'] // 1000, anchors), anchors[-1]['monotonic_ns'])
        with self.assertRaises(ValueError):
            audit.align((WALL - 1000) // 1000, anchors)

    def test_record_anchor_and_field_limits(self):
        records = fixture()
        with mock.patch.object(audit, 'MAX_RECORDS', len(records) - 1):
            with self.assertRaisesRegex(ValueError, 'too many telemetry records'):
                analyze(records)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'samples.jsonl'
                path.write_text(''.join(json.dumps(row) + '\n' for row in records))
                with self.assertRaisesRegex(ValueError, 'too many telemetry records'):
                    audit.read_records(path)
        with mock.patch.object(audit, 'MAX_ANCHORS', 1):
            with self.assertRaisesRegex(ValueError, 'too many clock anchors'):
                analyze(records)
        with mock.patch.object(audit, 'MAX_SAMPLES_PER_FIELD', 1):
            with self.assertRaisesRegex(ValueError, 'too many field samples'):
                analyze(records)

    def test_status_identity_type_range_stale_and_future(self):
        for key, value in (('status', -1), ('status', False), ('gpu_id', False), ('gpu_id', 1),
                           ('field_type', 105), ('field_id', 1007), ('value', float('nan')),
                           ('value', 1.01), ('value', True), ('timestamp_us', True),
                           ('observed_monotonic_ns', MONO), ('observed_monotonic_ns', MONO + 100000000000)):
            records = fixture()
            row = next(row for row in records if row['kind'] == 'sample' and row['timestamp_us'] * 1000 == WALL + 2000000000)
            row[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                analyze(records)

    def test_startup_invalid_never_earns_credit(self):
        records = fixture()
        next(row for row in records if row['kind'] == 'sample').update(
            status=-6, value=None, observed_monotonic_ns=MONO + 1)
        self.assertEqual(analyze(records)['fields']['1002']['interior_duration_weighted_ratio'], .5)

    def test_no_positive_signal_cannot_borrow_solo_evidence(self):
        with self.assertRaises(ValueError):
            analyze(fixture(0))

    def test_readiness_and_too_short_window_cannot_qualify(self):
        for ready in (True, 0, MONO + 2000000000):
            with self.subTest(ready=ready), self.assertRaises(ValueError):
                audit.analyze_records(fixture(), MONO + 1000000000, MONO + 5000000000, ready)
        with self.assertRaises(ValueError):
            audit.analyze_records(fixture(), MONO + 1000000000, MONO + 1015600000, MONO + 50000000)
        with self.assertRaises(ValueError):
            analyze([*fixture(), dict(kind='unknown')])

    def test_strict_bounded_complete_jsonl_and_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'samples.jsonl'
            for invalid in ('', '{}', '[]\n', '{"kind":"sample","kind":"anchor"}\n',
                            '{"value":NaN}\n', ' ' * 16385 + '\n'):
                path.write_text(invalid)
                with self.subTest(invalid=invalid[:50]), self.assertRaises(ValueError):
                    audit.read_records(path)
            records = fixture()
            path.write_text(''.join(json.dumps(row) + '\n' for row in records))
            self.assertEqual(audit.read_records(path), records)
            result = subprocess.run([sys.executable, '-B', audit.__file__, '--samples', str(path),
                '--begin-ns', str(MONO + 1000000000), '--end-ns', str(MONO + 5000000000),
                '--ready-ns', str(MONO + 50000000)], check=True, capture_output=True, text=True, timeout=10)
            self.assertEqual(json.loads(result.stdout), analyze(records))

    def test_cold_import_without_collectors_or_vendor_modules(self):
        code = ('import sys; sys.path.insert(0,sys.argv[1]); import aggregate_sm_audit; '
                'assert not any(name.startswith(("collect_", "run_bert", "dcgm", "torch", "ctypes")) '
                'for name in sys.modules)')
        subprocess.run([sys.executable, '-B', '-c', code, str(Path(audit.__file__).parent)],
                       check=True, capture_output=True, timeout=10)
        subprocess.run([sys.executable, '-B', audit.__file__, '--help'],
                       check=True, capture_output=True, timeout=10)


if __name__ == '__main__':
    unittest.main()
