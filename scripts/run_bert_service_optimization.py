#!/usr/bin/env python3
"""Run one native/equal-only service compiler-sensitivity diagnostic.

The preserved stage collector remains authoritative for workload, numerical,
ownership and resource gates. Every subprocess explicitly reenters this file.
"""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

import analyze_bert_service_stages as analysis
import run_bert_stage_diagnostic as collector
import run_bert_stage_candidate as stage
from artifact_paths import PROJECT_ROOT
from run_rnn_v8_validation import digest, save

candidate = stage.candidate
require = candidate.require
BASELINE = stage.BUILD
BUILD = PROJECT_ROOT / 'build/bert-service-o2-diagnostic-1'
SERVER = BUILD / 'bin/glimmer_execution_service'
PLAN = PROJECT_ROOT / 'artifacts/maintenance/bert-sm-diagnosis-20261031/service-o2-plan.json'
BASE_ATTEMPT = PROJECT_ROOT / 'artifacts/profiles/bert-service-stages/attempt-30wex5i7'
ENTRY = Path(__file__).resolve()
SOURCES = {'runtime.cc', 'host_owner.cc', 'profile_archive.cc', 'async_audit_log.cc',
           'nccl_actor.cc', 'nccl_output_effects.cc', 'main.cc'}


def build_identity():
    def commands(build):
        rows = collector.read(build / 'compile_commands.json')
        result = {}
        for row in rows:
            args = shlex.split(row['command'])
            index = args.index('-o')
            key = str((Path(row['directory']) / args[index + 1]).relative_to(build))
            require(key not in result, 'one actual compile command per object required')
            result[key] = (row['file'], args[:index] + args[index + 2:])
        return result

    before, after = commands(BASELINE), commands(BUILD)
    require(before.keys() == after.keys(), 'candidate compile target/object set differs')
    optimized = []
    for key, (source, args) in before.items():
        new_source, new_args = after[key]
        require(source == new_source, 'candidate source path differs')
        if Path(source).name in SOURCES and Path(source).parent == PROJECT_ROOT / 'src/execution_service':
            require(not any(arg.startswith('-O') for arg in args)
                    and new_args.count('-O2') == 1
                    and [arg for arg in new_args if arg != '-O2'] == args,
                    'only one explicit -O2 addition is allowed')
            optimized.append(source)
        else:
            require(new_args == args, 'observer/test compile arguments changed')
    require(len(optimized) == 7 and {Path(path).name for path in optimized} == SOURCES,
            'exact seven-TU optimization required')

    def link(build):
        path = build / 'service/CMakeFiles/glimmer_execution_service.dir/link.txt'
        args = [str((path.parents[2] / arg).resolve()) if not arg.startswith('-') else arg
                for arg in shlex.split(path.read_text())]
        return path, args

    old_link, old = link(BASELINE)
    new_link, new = link(BUILD)
    substitutions = {str(BASELINE / name): str(BUILD / name) for name in (
        'service/CMakeFiles/glimmer_execution_service.dir/main.cc.o',
        'service/libglimmer_execution_service_runtime.a', 'bin/glimmer_execution_service')}
    require([substitutions.get(arg, arg) for arg in old] == new,
            'ordered link contract changed beyond the service candidate')

    def dynamic(server):
        output = subprocess.check_output(['readelf', '-d', str(server)], text=True, timeout=10)
        return [line.split(')', 1)[1].strip() for line in output.splitlines()
                if any('(' + key + ')' in line for key in ('NEEDED', 'RUNPATH', 'RPATH'))]

    dependencies = dynamic(SERVER)
    require(dependencies == dynamic(BASELINE / 'bin/glimmer_execution_service'),
            'dynamic dependencies or loader paths differ')
    paths = {Path(arg) for arg in old + new if Path(arg).is_absolute() and Path(arg).is_file()}
    paths.update({old_link, new_link, ENTRY, PLAN,
        PROJECT_ROOT / 'cmake/ExecutionServiceOptimizationDiagnostic/CMakeLists.txt',
        PROJECT_ROOT / 'cmake/ExecutionServiceDiagnosticOverlay/CMakeLists.txt'})
    paths.update(build / name for build in (BASELINE, BUILD)
                 for name in ('CMakeCache.txt', 'compile_commands.json'))
    return dict(optimized_sources=sorted(optimized), observer_compile_unchanged=True,
        baseline_link=old, candidate_link=new, dynamic_dependencies=dependencies,
        identities={str(path): digest(path) for path in sorted(paths)})


def configure():
    """Bind the exact candidate in both parent and independently spawned workers."""
    binding = build_identity()
    stage.BUILD, stage.SERVER, stage.build_identity = BUILD, SERVER, build_identity
    candidate.MODES = ('native', 'equal')
    candidate.load_helpers = stage.helpers
    scope = dict(kind='prospective_service_o2_native_equal', modes=list(candidate.MODES),
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
            Path(collector.__file__).resolve(), Path(analysis.__file__).resolve(),
            PROJECT_ROOT / 'scripts/run_bert_candidate_validation.py',
            PROJECT_ROOT / 'src/execution_service/host_stage_metrics.h',
            PROJECT_ROOT / 'src/execution_service/host_stage_metrics.cc',
            *(Path(path) for path in binding['identities'])}

    def launch(attempt, mode, *, worker_script=None):
        require(mode in candidate.MODES and worker_script in (None, Path(collector.__file__), ENTRY),
                'only explicit native/equal candidate workers are allowed')
        return original_launch(attempt, mode, worker_script=ENTRY)

    def cleanup(job):
        original_cleanup(job)
        if job['mode'] == 'equal':
            require(job['server_command'][0] == str(SERVER)
                    and digest(SERVER) == scope['server_sha256']
                    and '--host-stage-output' not in job['server_command'],
                    'numerical worker did not execute the exact uninstrumented candidate')

    def numerical_save(path, value):
        if path.name == 'manifest.json':
            original_save(path.parent / 'service-build-identity.json', binding)
            original_save(path.parent / 'numerical-scope.json', scope)
        if path.name == 'report.json':
            value = dict(value, qualification_scope=scope)
        return original_save(path, value)

    def checkpoint(reference, contract_path=None):
        require(contract_path is None, 'prospective two-path scope cannot accept failed-triplet recovery')
        require(collector.read(reference / 'numerical-scope.json') == scope
                and collector.read(reference / 'report.json')['qualification_scope'] == scope,
                'exact prospective native/equal numerical scope required')
        common, h, sustained, report, paths = original_checkpoint(reference)
        paths.update({reference / 'numerical-scope.json', ENTRY, PLAN,
                      BASE_ATTEMPT / 'stage-analysis.json', BASE_ATTEMPT / 'plan.json'})
        return common, h, sustained, report, paths

    def stage_save(path, value):
        if path.name == 'plan.json':
            value = dict(value, optimization_scope=scope,
                baseline_analysis=dict(path=str(BASE_ATTEMPT / 'stage-analysis.json'),
                                       sha256=digest(BASE_ATTEMPT / 'stage-analysis.json')))
        return original_collect_save(path, value)

    common.input_paths, candidate.launch_unit = inputs, launch
    candidate.normal_cleanup, candidate.save = cleanup, numerical_save
    collector.checkpoint, collector.save, collector.ANALYZER = checkpoint, stage_save, ENTRY
    return scope


def analyze(attempt, scope):
    plan = collector.read(attempt / 'plan.json')
    require(plan['optimization_scope'] == scope, 'capture optimization scope differs')
    reference = Path(plan['reference'])
    require(collector.read(reference / 'numerical-scope.json') == scope,
            'capture lacks its prospective numerical scope')
    baseline = plan['baseline_analysis']
    require(baseline['path'] == str(BASE_ATTEMPT / 'stage-analysis.json')
            and digest(Path(baseline['path'])) == baseline['sha256'], 'baseline analysis changed')
    result = analysis.analyze(attempt)
    old = collector.read(Path(baseline['path']))

    def completed(control):
        window = control['application_window']
        return window['end']['completed_requests'] - window['begin']['completed_requests']

    old_native, old_equal = map(completed, old['controls'])
    new_native, new_equal = map(completed, result['controls'])
    require((old_native, old_equal) == (604, 16), 'pre-registered baseline count changed')
    valid = (abs(new_native / old_native - 1) <= .2
             and old['controls'][1]['service_stages']['attribution_coverage_gate']
             and result['controls'][1]['service_stages']['attribution_coverage_gate'])
    ratio = new_equal / old_equal
    verdict = ('inconclusive' if not valid or 1.2 <= ratio < 2 else
               'strong_seven_tu_optimization_sensitivity' if ratio >= 2 else
               'against_seven_tu_optimization_sufficiency')
    result.update(optimization_scope=scope, compiler_sensitivity=dict(
        baseline_native_completions=old_native, candidate_native_completions=new_native,
        baseline_equal_completions=old_equal, candidate_equal_completions=new_equal,
        equal_completion_ratio=ratio, comparison_gates_valid=valid, verdict=verdict,
        gpu_gap_causality_qualified=False, performance_qualified=False))
    output = attempt / 'stage-analysis.json'
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result['compiler_sensitivity'], indent=2))
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
