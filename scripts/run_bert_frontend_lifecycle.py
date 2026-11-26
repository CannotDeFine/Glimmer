#!/usr/bin/env python3
"""Qualify and join one bounded frontend/service BERT diagnostic pair."""

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

import analyze_bert_frontend_lifecycle as frontend
import run_bert_request_lifecycle as base
from artifact_paths import PROJECT_ROOT
from run_rnn_v8_validation import digest

candidate, collector, stage = base.candidate, base.collector, base.stage
require = candidate.require
ENTRY = Path(__file__).resolve()
BUILD = PROJECT_ROOT / 'build/bert-frontend-lifecycle-diagnostic-1'
SERVER = BUILD / 'bin/glimmer_execution_service'
PRELOAD = BUILD / 'lib/libglimmer_execution_interceptor.so'
PLAN = PROJECT_ROOT / 'artifacts/maintenance/bert-sm-diagnosis-20261031/frontend-lifecycle-plan.json'
OUTPUT_ENV = 'GLIMMER_EXECUTION_RPC_OUTPUT_DIR'
OUTPUT_SCHEMA = 1
OUTPUT_LIMIT = 256 * 1024 ** 2
EXTRA_ENV = {}
RUNTIME_ENV = {}


def frontend_identity():
    baseline = candidate.BUILD
    sources = {str(PROJECT_ROOT / 'src/interceptor' / suffix) for suffix in
               ('execution/driver_frontend.cc', 'execution/driver_lookup.cc', 'dlsym_entry_x86_64.S')}

    def commands(build):
        rows = json.loads((build / 'compile_commands.json').read_text())
        result = {}
        for row in rows:
            source = str(Path(row['file']).resolve())
            if source not in sources:
                continue
            args = shlex.split(row['command'])
            if 'glimmer_execution_interceptor.dir' not in row['command']:
                continue
            output = args.index('-o')
            require(source not in result, 'duplicate frontend compilation')
            result[source] = (args[:output] + args[output + 2:],
                              str((Path(row['directory']) / args[output + 1]).resolve()))
        require(set(result) == sources, 'exact frontend translation units required')
        return result

    old, new = commands(baseline), commands(BUILD)
    require(all(old[path][0] == new[path][0] for path in sources),
            'frontend compile flags differ beyond output objects')
    original = baseline / 'lib/libglimmer_execution_interceptor.so'
    old_link = baseline / 'src/interceptor/execution/CMakeFiles/glimmer_execution_interceptor.dir/link.txt'
    new_link = BUILD / 'frontend/CMakeFiles/glimmer_execution_interceptor.dir/link.txt'

    def link(path):
        return [str((path.parents[2] / arg).resolve()) if not arg.startswith('-') else arg
                for arg in shlex.split(path.read_text())]

    before, after = link(old_link), link(new_link)
    substitutions = {old[path][1]: new[path][1] for path in sources}
    substitutions[str(original)] = str(PRELOAD)
    require([substitutions.get(arg, arg) for arg in before] == after,
            'frontend ordered link contract differs beyond objects/output')

    def dynamic(path):
        output = subprocess.check_output(['readelf', '-d', str(path)], text=True, timeout=10)
        return [line.split(')', 1)[1].strip() for line in output.splitlines()
                if any('(' + key + ')' in line for key in ('NEEDED', 'RUNPATH', 'RPATH'))]

    def exports(path):
        output = subprocess.check_output(['nm', '-D', '--defined-only', str(path)], text=True, timeout=10)
        return sorted(' '.join(line.split()[1:]) for line in output.splitlines())

    require(dynamic(original) == dynamic(PRELOAD), 'frontend dynamic dependencies differ')
    require(exports(original) == exports(PRELOAD), 'frontend public exports differ')
    paths = {Path(arg) for arg in before + after if Path(arg).is_absolute() and Path(arg).is_file()}
    paths.update({old_link, new_link, PROJECT_ROOT / 'src/interceptor/execution/exports.map',
                  PROJECT_ROOT / 'cmake/ExecutionFrontendDiagnosticOverlay/CMakeLists.txt'})
    return dict(compile_equivalent_files=sorted(sources), dynamic_dependencies=dynamic(PRELOAD),
                public_exports=exports(PRELOAD), baseline_link=before, candidate_link=after,
                identities={str(path): digest(path) for path in sorted(paths)})


def configure():
    base.BUILD, base.SERVER, base.ENTRY, base.PLAN = BUILD, SERVER, ENTRY, PLAN
    original_identity = stage.build_identity

    def identity():
        result, extra = original_identity(), frontend_identity()
        result['frontend'] = extra
        result['identities'].update(extra['identities'])
        return result

    def helpers():
        common, h = stage.BASE_HELPERS()
        h.training.SERVER, h.training.PRELOAD = SERVER, PRELOAD
        return common, h

    stage.build_identity, stage.helpers = identity, helpers
    scope = base.configure()
    scope.update(kind='prospective_native_equal_frontend_lifecycle',
                 frontend=str(PRELOAD), frontend_sha256=digest(PRELOAD))
    if RUNTIME_ENV:
        scope['runtime_environment'] = dict(RUNTIME_ENV)
    common, _ = stage.BASE_HELPERS()
    original_environment = common.owned_environment

    def runtime_environment(*args, **kwargs):
        result = original_environment(*args, **kwargs)
        require(all(key not in result for key in RUNTIME_ENV), 'unexpected inherited runtime variant')
        return dict(result, **RUNTIME_ENV)

    if RUNTIME_ENV:
        common.owned_environment = runtime_environment
    original_inputs = common.input_paths
    original_cleanup = candidate.normal_cleanup
    original_hooks = collector.stage_hooks

    def inputs(h):
        return original_inputs(h) | {Path(base.__file__).resolve(), Path(frontend.__file__).resolve(),
            PROJECT_ROOT / 'src/interceptor/execution/rpc_metrics.h'}

    def cleanup(job):
        original_cleanup(job)
        environments = [job['application_environment'], *job['application_environments'].values()]
        if job['mode'] == 'equal':
            require(all(env['LD_PRELOAD'] == str(PRELOAD) and OUTPUT_ENV not in env for env in environments)
                    and all(key not in env for env in environments for key in EXTRA_ENV)
                    and all(env.get(key) == value for env in environments for key, value in RUNTIME_ENV.items())
                    and digest(PRELOAD) == scope['frontend_sha256'],
                    'actual exact observer-off numerical frontend required')
        else:
            require(all(key not in env for env in environments for key in RUNTIME_ENV),
                    'native numerical mode must not inherit the frontend runtime variant')

    @contextmanager
    def hooks(common, sustained, guard, directory, mode):
        original_environment = common.owned_environment

        def environment(*args, **kwargs):
            result = original_environment(*args, **kwargs)
            require(mode == 'equal' and result['LD_PRELOAD'] == str(PRELOAD)
                    and OUTPUT_ENV not in result, 'one explicit frontend diagnostic environment required')
            require(all(key not in result for key in EXTRA_ENV), 'unexpected inherited detail observer')
            return dict(result, **{OUTPUT_ENV: str(directory), **EXTRA_ENV})

        if mode == 'equal':
            common.owned_environment = environment
        try:
            with original_hooks(common, sustained, guard, directory, mode):
                yield
        finally:
            common.owned_environment = original_environment

    common.input_paths, candidate.normal_cleanup = inputs, cleanup
    collector.stage_hooks = hooks
    return scope


def worker(attempt, mode):
    result = collector.worker(attempt, mode)
    if result or mode != 'equal':
        return result
    path = attempt / (mode + '-worker.json')
    report = collector.read(path)
    try:
        job = report['job']
        environment = job['application_environment']
        require(set(job['application_identities']) == set(job['application_commands'])
                == {'training', 'inference'}, 'exact two authenticated application roles required')
        require('application_environments' not in job, 'exact sustained shared-environment receipt required')
        environments = [environment]
        require(all(env['LD_PRELOAD'] == str(PRELOAD)
                    and env[OUTPUT_ENV] == str(attempt / mode)
                    and all(env.get(key) == value for key, value in {**EXTRA_ENV, **RUNTIME_ENV}.items())
                    for env in environments)
                and job['server_command'][0] == str(SERVER),
                'actual instrumented frontend/service launch configuration differs')
        require(digest(PRELOAD) == collector.read(attempt / 'plan.json')['lifecycle_scope']['frontend_sha256'],
                'actual instrumented frontend bytes changed')
        outputs = {}
        mapping_sha = hashlib.sha256(json.dumps(environment, sort_keys=True,
                                               separators=(',', ':')).encode()).hexdigest()
        for role, identity in report['job']['application_identities'].items():
            output = attempt / mode / ('frontend-rpc-' + str(identity['pid']) + '.json')
            require(output.is_file() and output.stat().st_size <= OUTPUT_LIMIT,
                    'bounded normally published frontend output required')
            data = json.loads(output.read_text())
            require(data['schema'] == OUTPUT_SCHEMA and data['valid'] is True
                    and data['pid'] == identity['pid'] and data['diagnostic_only'] is True,
                    'valid exact frontend PID observation required')
            outputs[role] = dict(path=str(output), sha256=digest(output), records=len(data['records']),
                application_identity=identity, preload=environment['LD_PRELOAD'],
                output_directory=environment[OUTPUT_ENV],
                shared_launch_environment_sha256=mapping_sha,
                launch_environment_binding='Same frozen selected mapping sent to exact wrapper exec; not direct /proc environment capture.')
        report['frontend_outputs'] = outputs
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        result = 1
    candidate.save(path, report)
    return result


def analyze(attempt, scope):
    base.analyze(attempt, scope)
    request = collector.read(attempt / 'request-analysis.json')
    report = collector.read(attempt / 'equal-worker.json')
    documents = {}
    for role, receipt in report['frontend_outputs'].items():
        path = Path(receipt['path'])
        require(path.parent == attempt / 'equal' and digest(path) == receipt['sha256'],
                'frontend evidence changed')
        documents[role] = json.loads(path.read_text())
    window = request['controls'][1]['application_window']
    result = frontend.analyze(json.loads((attempt / 'equal/host-stages.json').read_text()),
        documents, collector.read(attempt / 'equal/helper-attempt.json'), window['begin_ns'], window['end_ns'],
        request['request_lifecycle'])
    result.update(lifecycle_scope=scope,
                  request_analysis_sha256=digest(attempt / 'request-analysis.json'),
                  frontend_outputs=report['frontend_outputs'])
    with (attempt / 'frontend-analysis.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result['roles'], indent=2))
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
        return worker(args.worker_attempt.resolve(strict=True), args.worker_mode)
    sys.argv = [str(ENTRY), '--wait-for-idle'] if args.qualify else [str(ENTRY),
        '--worker-attempt', str(args.worker_attempt), '--worker-mode', str(args.worker_mode)]
    return candidate.main()


if __name__ == '__main__':
    raise SystemExit(main())
