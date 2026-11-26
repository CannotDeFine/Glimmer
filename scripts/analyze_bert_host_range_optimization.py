"""Apply the frozen host-range intervention gates to matched diagnostic windows."""

import json
from pathlib import Path

import analyze_bert_host_range as host
from artifact_paths import PROJECT_ROOT
from run_rnn_v8_validation import digest

require = host.require
BASELINE = PROJECT_ROOT / 'artifacts/profiles/bert-service-stages/attempt-bkhrbgfw'
BASELINE_INPUTS = {
    BASELINE / 'equal/frontend-rpc-1831082.json': 'bf4f28fd4bf7e4917ab6afdca8c968ceb8611495176823e69aeacfa1698a9e2c',
    BASELINE / 'equal/frontend-phases-1831082.json': '6f382384f93ef5a5d25d953878f5c606a1bafc88a75433d9e63b8bb915c27824',
    BASELINE / 'equal/host-stages.json': 'e0667e22d4a7f6833e380b12bea760e3da303198ca1a3db37082b9ef3a016d59',
    BASELINE / 'request-analysis.json': 'dd9e3ca5ecf3f59bd52fed60d67a000636d62073d8b60a2e11459e4a62b515a6',
    BASELINE / 'frontend-wait-analysis.json': 'e913fe2f1d353ce132ec4db210483ab61b26c0294deba56bb753f4322e63fd71',
}


def summary(spans, identity, begin, end):
    selected = [row for row in spans if row['caller'] == 'get_blas_value'
                and row['thread'] == identity['pid'] and row['success']]
    complete = [row['end'] - row['begin'] for row in selected
                if begin <= row['begin'] <= row['end'] <= end]
    require(complete and sum(complete) > 0, 'nonempty successful main-thread helper sample required')
    return dict(count=len(complete), sum_ns=sum(complete), mean_ns=sum(complete) / len(complete),
                window_union_ns=host.duration(host.union(
                    ((row['begin'], row['end']) for row in selected), begin, end)))


def completions(report):
    require([row['mode'] for row in report['controls']] == ['native', 'equal'],
            'exact native/equal control pair required')
    result = {}
    for row in report['controls']:
        window = row['application_window']
        require(window['end_ns'] - window['begin_ns'] == 5_000_000_000,
                'fixed five-second completion window required')
        count = window['end']['completed_requests'] - window['begin']['completed_requests']
        require(type(count) is int and count >= 0, 'invalid completion difference')
        result[row['mode']] = count
    return result


def gates(before, after, controls):
    ratio = after['mean_ns'] / before['mean_ns']
    mechanism = ratio <= .5
    environment = 598 * .8 <= controls['native'] <= 598 * 1.2
    uplift = controls['equal'] / 16
    decision = ('environment_mismatch_inconclusive' if not environment else
                'mechanism_gate_failed' if not mechanism else
                'important_host_supply_limitation_supported' if uplift >= 2 else
                'scan_optimization_insufficient' if uplift < 1.2 else 'inconclusive')
    return dict(helper_mean_ratio=ratio, helper_mean_reduction=1 - ratio,
                mechanism_passed=mechanism, native_control_passed=environment,
                equal_completion_ratio=uplift, decision=decision)


def analyze(document, bases, phases, job, begin, end, request, wait_result):
    require(end - begin == 5_000_000_000, 'fixed five-second intervention window required')
    for path, checksum in BASELINE_INPUTS.items():
        require(digest(path) == checksum, 'frozen baseline bytes changed')
    old_request = json.loads((BASELINE / 'request-analysis.json').read_text())
    old_wait = json.loads((BASELINE / 'frontend-wait-analysis.json').read_text())
    old_window = old_request['controls'][1]['application_window']
    old_begin, old_end = old_window['begin_ns'], old_window['end_ns']
    old_identity = dict(pid=1831082, start=52827766)
    old_base = json.loads((BASELINE / 'equal/frontend-rpc-1831082.json').read_text())
    old_phases = json.loads((BASELINE / 'equal/frontend-phases-1831082.json').read_text())
    host.waits.ledger(old_base, old_phases, old_identity, old_begin, old_end)
    old_spans, _ = host.helpers(old_base, old_phases, old_identity, old_begin, old_end)
    before = summary(old_spans, old_identity, old_begin, old_end)
    require(before['count'] == 1528 and before['sum_ns'] == 2527161592
            and completions(old_request) == dict(native=598, equal=16), 'frozen baseline gate differs')
    for result in (old_wait, wait_result):
        for row in result['coarse_lifecycle']['roles']:
            require(row['compute_correlation_fraction'] == row['service_lifecycle_coverage']
                    == row['frontend_recorder_window_coverage'] == 1.0,
                    'complete exact frontend/service correlation required')
    roles = []
    for role, identity in job['application_identities'].items():
        host.waits.ledger(bases[role], phases[role], identity, begin, end)
        spans, outcomes = host.helpers(bases[role], phases[role], identity, begin, end)
        roles.append(dict(role=role, application_identity=identity, helper_outcomes=outcomes,
                          helpers=summary(spans, identity, begin, end) if role == 'inference' else None))
    require({row['role'] for row in roles} == {'training', 'inference'}, 'exact application roles required')
    after = next(row['helpers'] for row in roles if row['role'] == 'inference')
    attempt = Path(wait_result['frontend_outputs']['inference']['path']).parent.parent
    current_path = attempt / 'request-analysis.json'
    require(digest(current_path) == wait_result['request_analysis_sha256'], 'current request evidence changed')
    current = json.loads(current_path.read_text())
    require(current['request_lifecycle'] == request
            and current['controls'][1]['application_window']['begin_ns'] == begin
            and current['controls'][1]['application_window']['end_ns'] == end,
            'intervention window or request join differs')
    controls = completions(current)
    return dict(diagnostic_only=True, begin_ns=begin, end_ns=end, roles=roles,
                baseline_helpers=before, candidate_helpers=after, candidate_completions=controls,
                baseline_inputs={str(path): value for path, value in BASELINE_INPUTS.items()},
                comparison=gates(before, after, controls),
                denominator='Successful main-thread get_blas_value spans fully contained in fixed5s; clipped union reported separately.',
                limitations='Single sequential observed pair; native environmental control, not GPU causality or formal speedup. No host-gap dominance denominator required after optimization.',
                priority_qualified=False, full_device_qualified=False,
                gpu_gap_causality_qualified=False, performance_qualified=False)
