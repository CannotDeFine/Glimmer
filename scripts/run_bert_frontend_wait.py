#!/usr/bin/env python3
"""Qualify and collect one bounded, correlated async frontend wait diagnostic."""

import json
from pathlib import Path

import analyze_bert_frontend_wait as waits
import run_bert_frontend_lifecycle as lifecycle
from artifact_paths import PROJECT_ROOT
from run_rnn_v8_validation import digest

require = lifecycle.require
ENTRY = Path(__file__).resolve()
BUILD = PROJECT_ROOT / 'build/bert-frontend-wait-diagnostic-1'
PLAN = PROJECT_ROOT / 'artifacts/maintenance/bert-sm-diagnosis-20261031/frontend-wait-phase-plan.json'
configure_lifecycle = lifecycle.configure
worker_lifecycle = lifecycle.worker
OUTPUT_SCHEMA = 2
HOST_ANALYZER = None


def configure():
    scope = configure_lifecycle()
    scope['kind'] = 'prospective_native_equal_frontend_wait'
    common, _ = lifecycle.stage.BASE_HELPERS()
    original = common.input_paths

    def inputs(h):
        paths = original(h) | {ENTRY, Path(__file__).resolve(), Path(lifecycle.__file__).resolve(),
                               Path(waits.__file__).resolve()}
        if HOST_ANALYZER:
            paths.add(Path(HOST_ANALYZER.__file__).resolve())
        return paths

    common.input_paths = inputs
    return scope


def worker(attempt, mode):
    result = worker_lifecycle(attempt, mode)
    if result or mode != 'equal':
        return result
    path = attempt / (mode + '-worker.json')
    report = lifecycle.collector.read(path)
    try:
        for role, receipt in report['frontend_outputs'].items():
            identity = receipt['application_identity']
            base = json.loads(Path(receipt['path']).read_text())
            output = attempt / mode / ('frontend-phases-' + str(identity['pid']) + '.json')
            require(output.is_file() and output.stat().st_size <= lifecycle.OUTPUT_LIMIT,
                    'bounded normal phase publication required')
            phases = json.loads(output.read_text())
            require(all(base[key] == phases[key] for key in waits.COMMON)
                    and base['base_count'] == len(base['records'])
                    and 0 < base['base_count'] <= base['capacity'] == 2097152
                    and phases['phase_count'] == len(phases['phase_events'])
                    and 0 < phases['phase_count'] <= phases['phase_capacity'] == 8388608
                    and phases['phase_fields'] == ['serial', 'phase', 'begin', 'end']
                    and phases['phase_names'] == waits.NAMES +
                        (waits.HOST_NAMES if OUTPUT_SCHEMA in (3, 5, 6) else []),
                    'matching complete base and phase ledgers required')
            if OUTPUT_SCHEMA in (5, 6):
                HOST_ANALYZER.details(base, phases)
            receipt['phases'] = dict(path=str(output), sha256=digest(output),
                events=phases['phase_count'], application_identity=identity)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        result = 1
    lifecycle.candidate.save(path, report)
    return result


def analyze(attempt, scope):
    lifecycle.base.analyze(attempt, scope)
    request = lifecycle.collector.read(attempt / 'request-analysis.json')
    report = lifecycle.collector.read(attempt / 'equal-worker.json')
    bases, phases = {}, {}
    for role, receipt in report['frontend_outputs'].items():
        for selected, target in ((receipt, bases), (receipt['phases'], phases)):
            path = Path(selected['path'])
            require(path.parent == attempt / 'equal' and digest(path) == selected['sha256'],
                    'frontend wait evidence changed')
            target[role] = json.loads(path.read_text())
    window = request['controls'][1]['application_window']
    result = waits.analyze(json.loads((attempt / 'equal/host-stages.json').read_text()),
        bases, phases, lifecycle.collector.read(attempt / 'equal/helper-attempt.json'),
        window['begin_ns'], window['end_ns'], request['request_lifecycle'])
    result.update(lifecycle_scope=scope,
                  request_analysis_sha256=digest(attempt / 'request-analysis.json'),
                  frontend_outputs=report['frontend_outputs'])
    with (attempt / 'frontend-wait-analysis.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result['roles'], indent=2))
    if HOST_ANALYZER:
        host = HOST_ANALYZER.analyze(
            json.loads((attempt / 'equal/host-stages.json').read_text()), bases, phases,
            lifecycle.collector.read(attempt / 'equal/helper-attempt.json'),
            window['begin_ns'], window['end_ns'], request['request_lifecycle'], result)
        host.update(lifecycle_scope=scope, frontend_outputs=report['frontend_outputs'],
                    wait_analysis_sha256=digest(attempt / 'frontend-wait-analysis.json'))
        with (attempt / 'frontend-host-range-analysis.json').open('x') as stream:
            json.dump(host, stream, indent=2, allow_nan=False)
            stream.write('\n')
        print(json.dumps(host['roles'], indent=2))
    return 0


def main():
    lifecycle.ENTRY, lifecycle.BUILD, lifecycle.PLAN = ENTRY, BUILD, PLAN
    lifecycle.SERVER = BUILD / 'bin/glimmer_execution_service'
    lifecycle.PRELOAD = BUILD / 'lib/libglimmer_execution_interceptor.so'
    lifecycle.OUTPUT_SCHEMA, lifecycle.OUTPUT_LIMIT = OUTPUT_SCHEMA, 1023 * 1024 ** 2
    lifecycle.configure, lifecycle.worker, lifecycle.analyze = configure, worker, analyze
    return lifecycle.main()


if __name__ == '__main__':
    raise SystemExit(main())
