"""Compare fresh seek-off/on host diagnostics, never formal GPU performance."""

import json
from pathlib import Path

import analyze_bert_host_range as host
from run_rnn_v8_validation import digest

require = host.require
details = host.details
PROFILE = None
SEEK_ENV = 'GLIMMER_EXECUTION_HOST_RANGE_SEEK'


def cohort(rows, serials, begin, end):
    selected = [row for serial, row in rows.items() if serial in serials
                and begin <= row['coarse_begin'] <= row['coarse_end'] <= end]
    clipped = host.duration(host.union(((row['coarse_begin'], row['coarse_end'])
                                       for serial, row in rows.items() if serial in serials), begin, end))
    wall = sum(row['coarse_end'] - row['coarse_begin'] for row in selected)
    cpu = sum(row['cpu_end'] - row['cpu_begin'] for row in selected)
    count = len(selected)
    require(count > 0 and wall > 0 and cpu > 0 and clipped > 0 and wall / clipped >= .95,
            'nonempty complete helper cohort with clipped-edge coverage required')
    attempted, hits = (sum(row[key] for row in selected) for key in ('hint_attempted', 'hint_hit'))
    return dict(count=count, wall_ns=wall, cpu_ns=cpu, mean_wall_ns=wall / count,
                mean_cpu_ns=cpu / count, cohort_coverage=wall / clipped,
                attempted=attempted, hits=hits, fallback=count - hits,
                hit_fraction=hits / count)


def gates(before, after):
    b, c = before['helpers'], after['helpers']
    native_b, native_c = before['completions']['native'], after['completions']['native']
    equal_b, equal_c = before['completions']['equal'], after['completions']['equal']
    require(all(type(value) is int and value >= 0 for value in (native_b, native_c, equal_b, equal_c))
            and b['mean_wall_ns'] > 0 and b['mean_cpu_ns'] > 0,
            'nonnegative counts and nonzero helper baseline required')
    valid = (native_b > 0 and equal_b > 0 and 603 * .8 <= native_b <= 603 * 1.2
             and 603 * .8 <= native_c <= 603 * 1.2 and .8 <= native_c / native_b <= 1.2)
    wall, cpu = c['mean_wall_ns'] / b['mean_wall_ns'], c['mean_cpu_ns'] / b['mean_cpu_ns']
    mechanism = b['attempted'] == b['hits'] == 0 and c['hit_fraction'] >= .95 and wall <= .35 and cpu <= .35
    ratio = equal_c / equal_b if equal_b else None
    decision = ('invalid_environment_or_denominator' if not valid else
                'mechanism_gate_failed' if not mechanism else
                'diagnostic_application_response_supported' if ratio >= 1.5 else
                'seek_optimization_insufficient' if ratio < 1.2 else 'inconclusive')
    return dict(decision=decision, native_control_passed=valid, mechanism_passed=mechanism,
                helper_wall_mean_ratio=wall, helper_cpu_mean_ratio=cpu,
                equal_completion_ratio=ratio, full_device_qualified=False,
                performance_qualified=False, gpu_gap_causality_qualified=False, priority_qualified=False)


def analyze(document, bases, phases, job, begin, end, request, wait_result):
    require(PROFILE in ('off', 'on') and end - begin == 5_000_000_000,
            'explicit seek profile and fixed five-second window required')
    require(job['application_environment'][SEEK_ENV] == ('1' if PROFILE == 'on' else '0'),
            'actual seek environment differs from selected profile')
    roles = []
    for role, identity in job['application_identities'].items():
        require(bases[role]['schema'] == 6, 'actual schema6 required for seek comparison')
        host.waits.ledger(bases[role], phases[role], identity, begin, end)
        spans, outcomes = host.helpers(bases[role], phases[role], identity, begin, end)
        rows = details(bases[role], phases[role])
        boundaries, _, _ = host.waits.lifecycle.frontend(bases[role], identity, begin, end)
        if PROFILE == 'off':
            require(all(row['hint_attempted'] == row['hint_hit'] == 0 for row in rows.values()),
                    'seek-off must have no hint attempts throughout the process')
        prior = next(row for row in wait_result['coarse_lifecycle']['roles'] if row['role'] == role)
        require(prior['compute_correlation_fraction'] == prior['service_lifecycle_coverage']
                == prior['frontend_recorder_window_coverage'] == 1.0,
                'complete exact frontend/service correlation required')
        main = {serial for serial, row in boundaries.items() if row['thread'] == identity['pid']}
        summary = cohort(rows, main, begin, end) if role == 'inference' else None
        roles.append(dict(role=role, application_identity=identity, helper_outcomes=outcomes,
                          helpers=summary, observed_helpers=len(spans)))
    require({row['role'] for row in roles} == {'training', 'inference'}, 'exact application roles required')
    attempt = Path(wait_result['frontend_outputs']['inference']['path']).parent.parent
    path = attempt / 'request-analysis.json'
    require(digest(path) == wait_result['request_analysis_sha256'], 'request analysis changed')
    current = json.loads(path.read_text())
    require(current['request_lifecycle'] == request, 'request join differs')
    controls = current['controls']
    require([row['mode'] for row in controls] == ['native', 'equal'], 'exact paired modes required')
    completions = {}
    for row in controls:
        window = row['application_window']
        require(window['end_ns'] - window['begin_ns'] == 5_000_000_000, 'fixed control window required')
        completions[row['mode']] = window['end']['completed_requests'] - window['begin']['completed_requests']
    require(controls[1]['application_window']['begin_ns'] == begin
            and controls[1]['application_window']['end_ns'] == end, 'exact helper/control window required')
    return dict(diagnostic_only=True, profile=PROFILE, begin_ns=begin, end_ns=end, roles=roles,
                helpers=next(row['helpers'] for row in roles if row['role'] == 'inference'),
                completions=completions, request_analysis_sha256=digest(path),
                scope='Complete inference main-thread helpers in the fixed window; no post-optimization dominance denominator.',
                full_device_qualified=False, performance_qualified=False,
                gpu_gap_causality_qualified=False, priority_qualified=False)


def compare(baseline, candidate, plan):
    results, inputs = [], {}
    for attempt, profile in ((baseline, 'off'), (candidate, 'on')):
        path = attempt / 'frontend-host-range-analysis.json'
        result = json.loads(path.read_text())
        scope = result['lifecycle_scope']
        require(result['profile'] == profile and scope['plan'] == dict(path=str(plan), sha256=digest(plan))
                and scope['runtime_environment'] == {SEEK_ENV: '1' if profile == 'on' else '0'},
                'frozen paired plan/profile binding differs')
        report = json.loads((attempt / 'report.json').read_text())
        require(report['status'] == 'service_stage_pair_requires_independent_review',
                'normally completed observed pair required')
        for receipt in result['frontend_outputs'].values():
            for raw in (receipt, receipt['phases']):
                require(digest(Path(raw['path'])) == raw['sha256'], 'raw frontend ledger changed')
                inputs[raw['path']] = raw['sha256']
        for name, expected in (('request-analysis.json', result['request_analysis_sha256']),
                               ('frontend-wait-analysis.json', result['wait_analysis_sha256'])):
            require(digest(attempt / name) == expected, 'joined analysis changed')
            inputs[str(attempt / name)] = expected
        inputs[str(path)] = digest(path)
        inputs[str(attempt / 'report.json')] = digest(attempt / 'report.json')
        results.append(result)
    old, new = (result['lifecycle_scope'] for result in results)
    require({key: value for key, value in old.items() if key != 'runtime_environment'} ==
            {key: value for key, value in new.items() if key != 'runtime_environment'},
            'same-binary source/service/runtime scope required')
    return dict(**gates(*results), baseline=str(baseline), candidate=str(candidate), inputs=inputs,
                limitations='One fresh sequential pair per seek profile; diagnostic application response, not formal speedup or GPU utilization causality.')
