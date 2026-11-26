#!/usr/bin/env python3
"""Qualify and observe explicit frontend compiler profiles with recording off."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import analyze_bert_service_stages as analysis
import run_bert_request_lifecycle as workflow
from artifact_paths import PROJECT_ROOT
from run_rnn_v8_validation import digest

candidate, collector, stage = workflow.candidate, workflow.collector, workflow.stage
require = candidate.require
ENTRY = Path(__file__).resolve()
SERVICE_IDENTITY = stage.build_identity
OUTPUT_ENV = 'GLIMMER_EXECUTION_RPC_OUTPUT_DIR'
SOURCES = ('execution/driver_frontend.cc', 'execution/driver_lookup.cc', 'dlsym_entry_x86_64.S')


def dependency_paths(build):
    paths = {PROJECT_ROOT / 'src/interceptor/dlsym_entry_x86_64.S'}
    for name in ('driver_frontend.cc', 'driver_lookup.cc'):
        dependency = build / ('frontend/CMakeFiles/glimmer_execution_interceptor.dir/' + name + '.o.d')
        values = shlex.split(dependency.read_text().replace('\\\n', ' ').split(':', 1)[1])
        require(all(Path(value).is_absolute() and Path(value).is_file() for value in values),
                'regular absolute compiler dependencies required')
        paths.update(Path(value) for value in values)
    return paths


def verify_certificate(certificate):
    for name, expected in certificate['identities'].items():
        path = Path(name)
        require(path.is_file() and str(path.resolve(strict=True)) == expected['target']
                and path.stat().st_size == expected['bytes'] and digest(path) == expected['sha256'],
                'sealed compiler dependency changed: ' + name)


def verify_dependencies(baseline, optimized, baseline_expected, optimized_expected, optimized_only):
    require(baseline == baseline_expected and optimized == optimized_expected
            and not baseline - optimized and optimized - baseline == optimized_only,
            'compiler profile dependency sets differ from the sealed plan')


def link_arguments(root, origin):
    path = root / 'frontend/CMakeFiles/glimmer_execution_interceptor.dir/link.txt'
    result = []
    for argument in shlex.split(path.read_text()):
        if argument.startswith('-'):
            result.append(argument)
            continue
        resolved = (path.parents[2] / argument).resolve()
        if resolved.is_relative_to(origin):
            resolved = root / resolved.relative_to(origin)
        result.append(str(resolved))
    return result


def worker_configuration(document):
    configuration = collector.read(document)['frontend_compiler_configuration']
    require(configuration['profile'] in ('o0', 'o2')
            and digest(Path(configuration['plan'])) == configuration['sha256'], 'worker plan changed')
    return configuration


def require_no_frontend_output(directory):
    require(not any(directory.glob('frontend-*.json')), 'unexpected frontend recorder output')


def compiler_delta(old, new, cpp):
    require(not any(arg.startswith('-O') for arg in old), 'unoptimized baseline required')
    require((new.count('-O2') == 1 and [arg for arg in new if arg != '-O2'] == old)
            if cpp else new == old, 'only two C++ optimization flags may change')


def gates(native_o0, equal_o0, native_o2, equal_o2):
    require(all(type(value) is int and value >= 0
                for value in (native_o0, equal_o0, native_o2, equal_o2)), 'natural counts required')
    valid = (598 * .8 <= native_o0 <= 598 * 1.2 and 598 * .8 <= native_o2 <= 598 * 1.2
             and .8 <= native_o2 / native_o0 <= 1.2 and equal_o0 > 0)
    ratio = equal_o2 / equal_o0 if equal_o0 else None
    verdict = ('invalid_environment_or_denominator' if not valid else
               'frontend_compiler_contribution_supported' if ratio >= 2 else
               'frontend_optimization_insufficient' if ratio < 1.2 else 'inconclusive')
    return dict(valid=valid, equal_completion_ratio=ratio, decision=verdict,
                full_device_qualified=False, performance_qualified=False)


def frontend_identity(plan):
    baseline, build = Path(plan['baseline_build']), Path(plan['build'])
    sources = {str(PROJECT_ROOT / 'src/interceptor' / name) for name in SOURCES}

    def commands(root, origin):
        result = {}
        for row in collector.read(root / 'compile_commands.json'):
            if row['file'] not in sources or 'glimmer_execution_interceptor.dir' not in row['command']:
                continue
            args = shlex.split(row['command'])
            index = args.index('-o')
            require(row['file'] not in result, 'duplicate frontend source')
            original = (Path(row['directory']) / args[index + 1]).resolve()
            # The capsule retains the actual compile cwd, but its archived
            # objects are verified at their explicit relocated paths.
            result[row['file']] = (args[:index] + args[index + 2:], str(root / original.relative_to(origin)))
        require(set(result) == sources, 'exact frontend translation units required')
        return result

    old, new = commands(baseline, Path(plan['baseline_origin_build'])), commands(build, build)
    for source in sources:
        compiler_delta(old[source][0], new[source][0], source.endswith('.cc'))
    original, optimized = (root / 'lib/libglimmer_execution_interceptor.so' for root in (baseline, build))
    require(digest(original) == plan['baseline_frontend_sha256'], 'baseline frontend changed')
    require(all(digest(PROJECT_ROOT / name) == value for name, value in plan['source_sha256'].items()),
            'restored frontend source changed')
    manifest_path = Path(plan['frontend_input_certificate'])
    capsule_path = Path(plan['baseline_capsule_certificate'])
    require(digest(manifest_path) == plan['frontend_input_certificate_sha256']
            and digest(capsule_path) == plan['baseline_capsule_certificate_sha256'],
            'frontend input or capsule certificate changed')
    source_seal, capsule = collector.read(manifest_path), collector.read(capsule_path)
    verify_certificate(source_seal)
    verify_certificate(capsule)
    verify_dependencies(dependency_paths(baseline), dependency_paths(build),
        {Path(name) for name in capsule['dependencies']},
        {Path(name) for name in source_seal['dependencies']},
        {Path(name) for name in plan['optimized_only_headers']})
    paths = {manifest_path, capsule_path, original, optimized}
    paths.update(Path(name) for certificate in (source_seal, capsule) for name in certificate['identities'])
    for root in (baseline, build):
        paths.update((root / 'CMakeCache.txt', root / 'compile_commands.json'))
    # Bind the actual dependency files as well as the complete header/toolchain
    # closure. Historical manifests missing SDK headers are not substituted.
    for name in ('driver_frontend.cc', 'driver_lookup.cc'):
        dependency = baseline / ('frontend/CMakeFiles/glimmer_execution_interceptor.dir/' + name + '.o.d')
        paths.add(dependency)

    paths.update(root / 'frontend/CMakeFiles/glimmer_execution_interceptor.dir/link.txt'
                 for root in (baseline, build))
    before = link_arguments(baseline, Path(plan['baseline_origin_build']))
    after = link_arguments(build, build)
    substitutions = {old[source][1]: new[source][1] for source in sources}
    substitutions[str(original)] = str(optimized)
    require([substitutions.get(arg, arg) for arg in before] == after,
            'frontend ordered link inputs differ beyond output objects')
    paths.update(Path(arg) for arg in before + after if Path(arg).is_absolute() and Path(arg).is_file())
    for command in (['readelf', '-d'], ['nm', '-D', '--defined-only']):
        outputs = []
        for path in (original, optimized):
            output = subprocess.check_output([*command, str(path)], text=True, timeout=10)
            outputs.append(sorted(' '.join(line.split()[1:]) for line in output.splitlines())
                if command[0] == 'nm' else [line.split(')', 1)[1].strip() for line in output.splitlines()
                    if any('(' + name + ')' in line for name in ('NEEDED', 'RUNPATH', 'RPATH'))])
        require(outputs[0] == outputs[1], 'frontend exports or dynamic dependencies changed')
    paths.update(PROJECT_ROOT / name for name in plan['source_sha256'])
    return dict(compile_delta='One -O2 per C++ TU; assembly and all other ordered flags unchanged.',
        baseline_link=before, candidate_link=after,
        identities={str(path): digest(path) for path in sorted(paths)})


def environment_matches(job, frontend, *, sustained=False):
    if job['mode'] == 'native':
        return
    environments = [job['application_environment']]
    roles = {'training', 'inference'} if sustained else {'low', 'high'}
    require(set(job['application_commands']) == set(job['application_identities']) == roles,
            'exact application roles required for the selected receipt shape')
    if sustained:
        require('application_environments' not in job, 'exact shared-environment sustained receipt required')
    else:
        require(set(job['application_environments']) == {'low', 'high'},
                'both numerical client environments required')
        environments.extend(job['application_environments'].values())
    require(all(env == environments[0] and env['LD_PRELOAD'] == str(frontend)
                and OUTPUT_ENV not in env for env in environments),
            'exact selected observer-off frontend required')


def configure(path, profile):
    path = path.resolve(strict=True)
    require(path.is_relative_to(PROJECT_ROOT / 'artifacts') and path.stat().st_size <= 65536,
            'bounded explicit artifact plan required')
    plan = collector.read(path)
    require(digest(ENTRY) == plan['runner_sha256'], 'frozen compiler runner changed')
    require(plan['status'] == 'frozen_for_two_observer_off_compiler_pairs'
            and plan['build'] == str(PROJECT_ROOT / 'build/candidate')
            and profile in ('o0', 'o2') and all(plan[key] is False for key in
                ('full_device_qualified', 'performance_qualified', 'priority_qualified')),
            'explicit unqualified compiler profile required')
    deadline = datetime.fromisoformat(plan['ordinary_run_deadline_utc'])
    require(deadline.tzinfo is not None
            and (deadline - datetime.now(timezone.utc)).total_seconds() <= 3 * 3600,
            'bounded timezone-aware deadline required')
    require(not os.environ.get(OUTPUT_ENV), 'frontend observer must be disabled')
    build, server = Path(plan['build']), Path(plan['shared_server'])
    require(server == build / 'bin/glimmer_execution_service'
            and digest(server) == plan['shared_server_sha256'], 'shared service changed')
    frontend = Path(plan['baseline_build'] if profile == 'o0' else plan['build']) / 'lib/libglimmer_execution_interceptor.so'
    require(digest(frontend) == plan['frontend_sha256'][profile], 'selected frozen frontend changed')
    configuration = dict(plan=str(path), sha256=digest(path), profile=profile)
    workflow.ENTRY, workflow.PLAN, workflow.BUILD, workflow.SERVER = ENTRY, path, build, server
    stage.BUILD, stage.SERVER, stage.DEADLINE = build, server, deadline
    collector.STAGE_PLAN = path

    def identity():
        result = SERVICE_IDENTITY()
        certificate = collector.read(Path(plan['shared_service_certificate']))
        require(digest(Path(plan['shared_service_certificate'])) == plan['shared_service_certificate_sha256']
                and all(digest(Path(name)) == value for name, value in certificate['identities'].items()),
                'shared service immutable inputs changed')
        extra = frontend_identity(plan)
        result['frontend'] = extra
        result['identities'].update(extra['identities'])
        result['identities'][plan['shared_service_certificate']] = plan['shared_service_certificate_sha256']
        return result

    def helpers():
        common, h = stage.BASE_HELPERS()
        h.training.SERVER, h.training.PRELOAD = server, frontend
        return common, h

    stage.build_identity, stage.helpers = identity, helpers
    scope = workflow.configure()
    scope.update(kind='prospective_frontend_compiler_native_equal', profile=profile,
        frontend=str(frontend), frontend_sha256=digest(frontend), frontend_recorder=False)
    original_cleanup, original_save = candidate.normal_cleanup, candidate.save
    original_launch, original_collect_save = candidate.launch_unit, collector.save

    def cleanup(job):
        original_cleanup(job)
        environment_matches(job, frontend)

    def numerical_save(output, value):
        if output.name == 'manifest.json':
            value = dict(value, frontend_compiler_configuration=configuration)
        if output.name.endswith('-worker.json'):
            value = dict(value, frontend_compiler_scope=scope)
        return original_save(output, value)

    def stage_save(output, value):
        if output.name == 'plan.json':
            value = dict(value, frontend_compiler_configuration=configuration)
        return original_collect_save(output, value)

    def launch(*args, **kwargs):
        unit, receipt = original_launch(*args, **kwargs)
        require(receipt['frontend_compiler_scope'] == scope, 'worker selected a different compiler profile')
        return unit, receipt

    candidate.normal_cleanup, candidate.save, candidate.launch_unit = cleanup, numerical_save, launch
    collector.save = stage_save
    return scope, configuration, frontend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--profile', choices=('o0', 'o2'))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--qualify', action='store_true')
    group.add_argument('--collect', type=Path)
    group.add_argument('--analyze', type=Path)
    group.add_argument('--worker-attempt', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--worker-mode', choices=('native', 'equal'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_attempt:
        document = args.worker_attempt / ('plan.json' if (args.worker_attempt / 'plan.json').is_file()
                                         else 'manifest.json')
        selected_configuration = worker_configuration(document)
        args.plan, args.profile = Path(selected_configuration['plan']), selected_configuration['profile']
    require(args.plan is not None and args.profile is not None, 'explicit plan and compiler profile required')
    scope, configuration, frontend = configure(args.plan, args.profile)
    if not args.analyze:
        stage.authorization_budget(2600 if args.qualify else 2300 if args.collect else 800)
    if args.analyze:
        attempt = args.analyze.resolve(strict=True)
        require(collector.read(attempt / 'plan.json')['lifecycle_scope'] == scope,
                'capture compiler scope changed')
        result = analysis.analyze(attempt)
        for entry in collector.read(attempt / 'report.json')['jobs']:
            require(entry['worker']['frontend_compiler_scope'] == scope, 'raw worker profile differs')
            environment_matches(entry['worker']['job'], frontend, sustained=True)
        result.update(compiler_scope=scope, report_sha256=digest(attempt / 'report.json'))
        with (attempt / 'frontend-compiler-analysis.json').open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write('\n')
        print(json.dumps(dict(profile=args.profile, completions={row['mode']:
            row['application_window']['end']['completed_requests'] -
            row['application_window']['begin']['completed_requests'] for row in result['controls']})))
        return 0
    if args.collect:
        return collector.collect(args.collect.resolve(strict=True))
    if args.worker_attempt:
        require(args.worker_mode is not None and configuration == selected_configuration,
                'exact worker configuration required')
        attempt = args.worker_attempt.resolve(strict=True)
        if (attempt / 'plan.json').is_file():
            require(collector.read(attempt / 'plan.json')['lifecycle_scope'] == scope,
                    'actual stage worker scope differs')
            result = collector.worker(attempt, args.worker_mode)
            output = attempt / (args.worker_mode + '-worker.json')
            report = collector.read(output)
            if not result:
                try:
                    environment_matches(report['job'], frontend, sustained=True)
                    require_no_frontend_output(attempt / args.worker_mode)
                except BaseException as error:
                    report.update(status='failed', error=f'{type(error).__name__}: {error}')
                    result = 1
            candidate.save(output, report)
            return result
        require(collector.read(attempt / 'numerical-scope.json') == scope,
                'actual numerical worker scope differs')
        return candidate.worker(attempt, args.worker_mode)
    sys.argv = [str(ENTRY), '--wait-for-idle']
    return candidate.main()


if __name__ == '__main__':
    raise SystemExit(main())
