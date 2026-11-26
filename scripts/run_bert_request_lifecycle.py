#!/usr/bin/env python3
"""Qualify and collect one bounded service-visible BERT request diagnostic."""

import argparse
import json
from pathlib import Path
import sys

import analyze_bert_service_stages as stages
import analyze_bert_request_lifecycle as lifecycle
import run_bert_stage_diagnostic as collector
import run_bert_stage_candidate as stage
from artifact_paths import PROJECT_ROOT
from run_rnn_v8_validation import digest

candidate = stage.candidate
require = candidate.require
ENTRY = Path(__file__).resolve()
PLAN = PROJECT_ROOT / 'artifacts/maintenance/bert-sm-diagnosis-20261031/request-lifecycle-plan.json'
BUILD = PROJECT_ROOT / 'build/bert-request-lifecycle-diagnostic-1'
SERVER = BUILD / 'bin/glimmer_execution_service'


def configure():
    stage.BUILD, stage.SERVER = BUILD, SERVER
    binding = stage.build_identity()
    candidate.MODES = ('native', 'equal')
    candidate.load_helpers = stage.helpers
    scope = dict(kind='prospective_native_equal_request_lifecycle', modes=list(candidate.MODES),
        server=str(SERVER), server_sha256=digest(SERVER),
        plan=dict(path=str(PLAN), sha256=digest(PLAN)),
        priority_qualified=False, triplet_qualified=False, full_device_qualified=False,
        gpu_gap_causality_qualified=False, performance_qualified=False)
    common, _ = stage.BASE_HELPERS()
    original_inputs, original_launch = common.input_paths, candidate.launch_unit
    original_cleanup, original_save = candidate.normal_cleanup, candidate.save
    original_checkpoint, original_collect_save = collector.checkpoint, collector.save

    def inputs(h):
        return original_inputs(h) | {ENTRY, PLAN, Path(stage.__file__).resolve(),
            Path(collector.__file__).resolve(), Path(stages.__file__).resolve(),
            Path(lifecycle.__file__).resolve(),
            PROJECT_ROOT / 'scripts/run_bert_candidate_validation.py',
            PROJECT_ROOT / 'src/execution_service/host_stage_metrics.h',
            PROJECT_ROOT / 'src/execution_service/host_stage_metrics.cc',
            PROJECT_ROOT / 'tests/control/execution_host_stage_metrics_test.cc',
            *(Path(path) for path in binding['identities'])}

    def launch(attempt, mode, *, worker_script=None):
        require(mode in candidate.MODES and worker_script in (None, Path(collector.__file__), ENTRY),
                'explicit native/equal entrypoint required')
        return original_launch(attempt, mode, worker_script=ENTRY)

    def cleanup(job):
        original_cleanup(job)
        if job['mode'] == 'equal':
            require(job['server_command'][0] == str(SERVER)
                    and digest(SERVER) == scope['server_sha256']
                    and '--host-stage-output' not in job['server_command'],
                    'exact uninstrumented numerical candidate required')

    def numerical_save(path, value):
        if path.name == 'manifest.json':
            original_save(path.parent / 'service-build-identity.json', binding)
            original_save(path.parent / 'numerical-scope.json', scope)
        if path.name == 'report.json':
            value = dict(value, qualification_scope=scope)
        return original_save(path, value)

    def checkpoint(reference, contract_path=None):
        require(contract_path is None
                and collector.read(reference / 'numerical-scope.json') == scope
                and collector.read(reference / 'report.json')['qualification_scope'] == scope,
                'exact prospective native/equal checkpoint required')
        common, h, sustained, report, paths = original_checkpoint(reference)
        paths.update({reference / 'numerical-scope.json', ENTRY, PLAN,
                      Path(lifecycle.__file__).resolve()})
        return common, h, sustained, report, paths

    def stage_save(path, value):
        if path.name == 'plan.json':
            value = dict(value, lifecycle_scope=scope)
        return original_collect_save(path, value)

    common.input_paths, candidate.launch_unit = inputs, launch
    candidate.normal_cleanup, candidate.save = cleanup, numerical_save
    collector.checkpoint, collector.save, collector.ANALYZER = checkpoint, stage_save, ENTRY
    return scope


def analyze(attempt, scope):
    plan = collector.read(attempt / 'plan.json')
    require(plan['lifecycle_scope'] == scope
            and collector.read(Path(plan['reference']) / 'numerical-scope.json') == scope,
            'capture numerical scope changed')
    result = stages.analyze(attempt)
    job = collector.read(attempt / 'equal/helper-attempt.json')
    document = json.loads((attempt / 'equal/host-stages.json').read_text())
    window = result['controls'][1]['application_window']
    result['request_lifecycle'] = lifecycle.analyze(document, job, window['begin_ns'], window['end_ns'])
    result['lifecycle_scope'] = scope
    with (attempt / 'request-analysis.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result['request_lifecycle']['summary'], indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--qualify', action='store_true')
    group.add_argument('--collect', type=Path)
    group.add_argument('--analyze', type=Path)
    group.add_argument('--worker-attempt', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--worker-mode', choices=('native', 'equal'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.analyze:
        stage.authorization_budget(2600 if args.qualify else 2300 if args.collect else 800)
    scope = configure()
    if args.analyze:
        return analyze(args.analyze.resolve(strict=True), scope)
    if args.collect:
        return collector.collect(args.collect.resolve(strict=True))
    if args.worker_attempt and (args.worker_attempt / 'plan.json').is_file():
        require(args.worker_mode is not None, 'explicit worker mode required')
        return collector.worker(args.worker_attempt.resolve(strict=True), args.worker_mode)
    sys.argv = [str(ENTRY), '--wait-for-idle'] if args.qualify else [str(ENTRY),
        '--worker-attempt', str(args.worker_attempt), '--worker-mode', str(args.worker_mode)]
    return candidate.main()


if __name__ == '__main__':
    raise SystemExit(main())
