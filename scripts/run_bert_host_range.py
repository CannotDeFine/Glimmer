#!/usr/bin/env python3
"""Qualify and collect bounded host-range diagnostics and seek-profile pairs."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

import analyze_bert_host_range as host_range
import analyze_bert_host_range_seek as seek
import run_bert_frontend_wait as wait
import run_bert_frontend_compiler as compiler
from artifact_paths import PROJECT_ROOT, create_attempt, resolve_artifact_root

DETAIL_ENV = 'GLIMMER_EXECUTION_HOST_RANGE_DETAIL'
require = wait.require
ORDER = [False, True, True, False, False, True, True, False]


def overhead_ratio(report):
    require(report['order'] == ORDER and len(report['samples']) == 8,
            'exact eight-process overhead order required')
    for index, sample in enumerate(report['samples']):
        require(sample['index'] == index and sample['detail'] is ORDER[index]
                and type(sample['coarse_helper_ns']) is int and sample['coarse_helper_ns'] > 0,
                'exact positive paired coarse observations required')
    require(len({sample['page_bytes'] for sample in report['samples']}) == 1, 'fixture page geometry differs')
    ratios = []
    for index in range(0, 8, 2):
        pair = {row['detail']: row['coarse_helper_ns'] for row in report['samples'][index:index + 2]}
        ratios.append(pair[True] / pair[False])
    median = statistics.median(ratios)
    require(report['ratios'] == ratios and report['median_ratio'] == median and median <= 1.2,
            'independently recomputed paired overhead exceeds frozen gate or differs')
    return median


def verify_overhead(receipt, plan_path, addendum_path, profile=None):
    report, addendum = (compiler.collector.read(path) for path in (receipt, addendum_path))
    require(report['status'] == 'passed_requires_review'
            and report['plan'] == str(plan_path) and report['plan_sha256'] == compiler.digest(plan_path)
            and addendum['report_sha256'] == compiler.digest(receipt), 'overhead plan/report binding differs')
    require(report.get('seek_profile') == profile, 'overhead seek profile differs')
    overhead_ratio(report)
    for name, expected in report['identities'].items():
        path = (Path(addendum['overhead_runner_snapshot']) if name == str(Path(__file__).resolve())
                else Path(name))
        require(compiler.digest(path) == expected, 'overhead executable or archived collector changed')
    require(compiler.digest(Path(addendum['overhead_runner_snapshot'])) == addendum['overhead_runner_sha256'],
            'original overhead collector snapshot changed')
    compiler.verify_certificate(addendum)
    first = min((receipt.parent / str(index) / 'process.json').stat().st_ctime_ns for index in range(8))
    require(first == addendum['first_process_receipt_ctime_ns'], 'preflight receipt timestamp changed')
    for name, expected in addendum['identities'].items():
        actual = Path(name).stat()
        require((actual.st_dev, actual.st_ino, actual.st_ctime_ns) ==
                (expected['device'], expected['inode'], expected['ctime_ns'])
                and actual.st_ctime_ns < first, 'fixture inode changed since preflight')
    for index, sample in enumerate(report['samples']):
        directory = receipt.parent / str(index)
        process_path = directory / 'process.json'
        require(compiler.digest(process_path) == sample['process_sha256'], 'raw fixture result changed')
        process = compiler.collector.read(process_path)
        if profile is not None:
            environment = dict(PATH='/usr/bin:/bin', LANG='C', LC_ALL='C',
                GLIMMER_EXECUTION_RPC_OUTPUT_DIR=str(directory), GLIMMER_TEST_HOST_RANGE_OVERHEAD='1',
                **{seek.SEEK_ENV: '1' if profile == 'on' else '0'})
            if ORDER[index]:
                environment[DETAIL_ENV] = '1'
            require(process['environment'] == environment, 'overhead actual environment changed')
        match = re.search(r'begin_ns=(\d+); end_ns=(\d+); thread=(\d+)', process['stdout'])
        require(process['returncode'] == 0 and match is not None
                and '; vma_pages=4096; verified_guards=2; layout_checks=2; warmup=20; measured=200; successes=200' in process['stdout'],
                'complete successful guarded fixture required')
        begin, end, thread = map(int, match.groups())
        publications = {}
        for name, sha in sample['outputs'].items():
            path = Path(name)
            require(path.parent == directory and compiler.digest(path) == sha, 'overhead raw ledger changed')
            publications['base' if path.name.startswith('frontend-rpc-') else 'phases'] = compiler.collector.read(path)
        require(set(publications) == {'base', 'phases'}, 'two overhead ledgers required')
        base, phases = publications['base'], publications['phases']
        boundaries, _, _ = host_range.waits.lifecycle.frontend(base, dict(pid=base['pid']), begin, end)
        host_range.waits.ledger(base, phases, dict(pid=base['pid']), begin, end)
        detail_rows = host_range.details(base, phases, enabled=ORDER[index])
        selected = [(left, right) for serial, kind, left, right in phases['phase_events']
                    if kind == 5 and boundaries[serial]['thread'] == thread and begin <= left <= right <= end]
        require(len(selected) == 200 and sum(right - left for left, right in selected) == sample['coarse_helper_ns'],
                'raw measured coarse cohort differs from reported overhead')
        if profile is not None and ORDER[index]:
            observed = [row for serial, row in detail_rows.items() if boundaries[serial]['thread'] == thread
                        and begin <= row['coarse_begin'] <= row['coarse_end'] <= end]
            expected = 1 if profile == 'on' else 0
            require(len(observed) == 200 and all(row['hint_attempted'] == row['hint_hit'] == expected
                                                for row in observed), 'measured hint profile differs')


def overhead(plan_path, profile=None):
    """Run the frozen same-binary coarse-versus-detailed observer check."""
    plan = compiler.collector.read(plan_path)
    require(plan['status'] == ('frozen_live_host_range_seek_pair' if profile is not None
                              else 'frozen_observer_only_stage_attribution'), 'frozen plan required')
    require(profile in (None, 'off', 'on'), 'explicit overhead profile required')
    build = Path(plan['build'])
    output = create_attempt(resolve_artifact_root('bert-host-range-overhead', category='profiles'))
    binary = build / 'lib/libglimmer_execution_interceptor.so'
    executable = build / 'glimmer_execution_blas_test'
    fixtures = tuple(PROJECT_ROOT / 'build/debug/src/interceptor/execution' / name for name in
                     ('libcuda.so.execution-memory-fixture.so', 'libcublas.so.execution-fixture.so',
                      'libcublasLt.so.execution-fixture.so'))
    identities = {str(path): compiler.digest(path) for path in (binary, executable, Path(__file__), *fixtures)}
    report = dict(plan=str(plan_path), plan_sha256=compiler.digest(plan_path), identities=identities,
                  order=ORDER, samples=[], status='running')
    if profile is not None:
        report['seek_profile'] = profile
    try:
        for index, enabled in enumerate(report['order']):
            directory = output / str(index)
            directory.mkdir()
            env = dict(os.environ) if profile is None else dict(PATH='/usr/bin:/bin', LANG='C', LC_ALL='C')
            for key in ('LD_PRELOAD', 'LD_AUDIT', DETAIL_ENV, seek.SEEK_ENV):
                env.pop(key, None)
            if profile is not None:
                env[seek.SEEK_ENV] = '1' if profile == 'on' else '0'
            env.update(GLIMMER_EXECUTION_RPC_OUTPUT_DIR=str(directory), GLIMMER_TEST_HOST_RANGE_OVERHEAD='1')
            if enabled:
                env[DETAIL_ENV] = '1'
            process = subprocess.run([str(executable),
                str(PROJECT_ROOT / 'build/debug/src/interceptor/execution/libcuda.so.execution-memory-fixture.so'),
                str(PROJECT_ROOT / 'build/debug/src/interceptor/execution/libcublas.so.execution-fixture.so')],
                env=env, capture_output=True, text=True, timeout=30)
            with (directory / 'process.json').open('x') as stream:
                json.dump(dict(returncode=process.returncode, stdout=process.stdout, stderr=process.stderr,
                               environment=env), stream)
            require(process.returncode == 0, 'overhead fixture must complete normally')
            match = re.search(r'Host range overhead: begin_ns=(\d+); end_ns=(\d+); thread=(\d+); page_bytes=(\d+); vma_pages=4096; verified_guards=2; layout_checks=2; warmup=20; measured=200; successes=200', process.stdout)
            require(match is not None, 'exact successful fixture geometry required')
            begin, end, thread, page = map(int, match.groups())
            base_paths, phase_paths = list(directory.glob('frontend-rpc-*.json')), list(directory.glob('frontend-phases-*.json'))
            require(len(base_paths) == len(phase_paths) == 1, 'one exact pair of observer publications required')
            base, phases = (compiler.collector.read(paths[0]) for paths in (base_paths, phase_paths))
            identity = dict(pid=base['pid'])
            boundaries, calls, _ = host_range.waits.lifecycle.frontend(base, identity, begin, end)
            # BLAS CPU fixture has no async-frontier work; validate helper parents
            # directly instead of requiring the application's async poll ledger.
            details = host_range.details(base, phases, enabled=enabled)
            host_range.waits.ledger(base, phases, identity, begin, end)
            selected = []
            for serial, kind, left, right in phases['phase_events']:
                if kind < 5:
                    continue
                require(serial in boundaries and kind in (5, 6, 7, 8), 'fixture helper identity differs')
                parent = boundaries[serial]
                require(parent['locked'] <= left <= right <= parent['finish'], 'fixture helper clock differs')
                if begin <= left <= right <= end:
                    require(kind == 5 and parent['thread'] == thread, 'measured successful getter required')
                    children = [call for call in calls if call['boundary'] == serial]
                    require(len(children) == 1 and children[0]['opcode'] == 29 and children[0]['flags'] == 3
                            and children[0]['thread'] == thread and right <= children[0]['enter'],
                            'measured getter must precede its exact RPC')
                    selected.append((left, right))
                    if profile is not None and enabled:
                        expected = 1 if profile == 'on' else 0
                        require(details[serial]['hint_attempted'] == details[serial]['hint_hit'] == expected,
                                'stable measured fixture must observe selected hint profile')
            require(len(selected) == 200 and all(a[1] <= b[0] for a, b in zip(sorted(selected), sorted(selected)[1:])),
                    'exact disjoint measured helper cohort required')
            report['samples'].append(dict(index=index, detail=enabled, page_bytes=page,
                coarse_helper_ns=sum(right - left for left, right in selected),
                outputs={str(path): compiler.digest(path) for path in (*base_paths, *phase_paths)},
                process_sha256=compiler.digest(directory / 'process.json')))
            require(all(compiler.digest(Path(name)) == sha for name, sha in identities.items()),
                    'overhead binary or runner changed')
        require(len({row['page_bytes'] for row in report['samples']}) == 1, 'fixture geometry changed')
        ratios = []
        for index in range(0, 8, 2):
            pair = {row['detail']: row['coarse_helper_ns'] for row in report['samples'][index:index + 2]}
            ratios.append(pair[True] / pair[False])
        report.update(ratios=ratios, median_ratio=statistics.median(ratios))
        require(report['median_ratio'] <= 1.2, 'observer perturbation exceeds frozen20% gate')
        report['status'] = 'passed_requires_review'
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
    compiler.candidate.save(output / 'report.json', report)
    print(json.dumps(dict(attempt=str(output), **{key: report[key] for key in ('status', 'median_ratio', 'error') if key in report})))
    return 0 if report['status'] == 'passed_requires_review' else 1


def configure_stage_plan(path, profile=None):
    plan = compiler.collector.read(path)
    is_seek = plan['status'] == 'frozen_live_host_range_seek_pair'
    require((is_seek and profile in plan['profiles'] or
             plan['status'] == 'frozen_observer_only_stage_attribution' and profile is None)
            and plan['build'] == str(PROJECT_ROOT / 'build/candidate'), 'frozen bounded stage plan required')
    seal_path = path.with_name('build-seal.json')
    seal = compiler.collector.read(seal_path)
    require(seal['plan_sha256'] == compiler.digest(path), 'build plan changed')
    for name, sha in seal['identities'].items():
        require(compiler.digest(Path(name)) == sha, 'frozen observer input changed: ' + name)
    inputs = compiler.collector.read(Path(seal['input_certificate']))
    compiler.verify_certificate(inputs)
    require(set(map(str, compiler.dependency_paths(Path(plan['build'])))) == set(seal['dependencies']),
            'actual frontend dependency closure changed')
    preflight = compiler.collector.read(path.with_name('preflight.json'))
    require(preflight['status'] == 'passed' and preflight['build_seal_sha256'] == compiler.digest(seal_path),
            'same-build CPU checks and overhead gate required')
    selected = preflight['profiles'][profile] if is_seek else preflight
    receipt = Path(selected['overhead_report'])
    require(compiler.digest(receipt) == selected['overhead_report_sha256'], 'overhead receipt changed')
    addendum_path = Path(selected['overhead_input_addendum'])
    require(compiler.digest(addendum_path) == selected['overhead_input_addendum_sha256'], 'overhead input seal changed')
    verify_overhead(receipt, path, addendum_path, profile)
    wait.ENTRY, wait.BUILD, wait.PLAN = Path(__file__).resolve(), Path(plan['build']), path
    wait.OUTPUT_SCHEMA, wait.HOST_ANALYZER = (6, seek) if is_seek else (5, host_range)
    if is_seek:
        seek.PROFILE = profile
        wait.lifecycle.RUNTIME_ENV = {seek.SEEK_ENV: '1' if profile == 'on' else '0'}
        wait.lifecycle.collector.STAGE_PLAN = path
    wait.lifecycle.EXTRA_ENV = {DETAIL_ENV: '1'}
    wait.lifecycle.stage.DEADLINE = datetime.fromisoformat(plan['ordinary_run_deadline_utc'])
    original = wait.lifecycle.frontend_identity

    def identity():
        result = original()
        result['identities'].update(seal['identities'])
        result['identities'].update({name: value['sha256'] for name, value in inputs['identities'].items()})
        for file in (seal_path, path.with_name('preflight.json'), receipt):
            result['identities'][str(file)] = compiler.digest(file)
        return result

    wait.lifecycle.frontend_identity = identity


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--overhead', action='store_true')
    parser.add_argument('--profile', choices=('off', 'on'))
    parser.add_argument('--compare', nargs=2, type=Path, metavar=('BASELINE', 'CANDIDATE'))
    args, remaining = parser.parse_known_args()
    if '--help' in remaining:
        print('Additional options: --plan PATH; --profile off|on; --overhead (CPU only); --compare BASELINE CANDIDATE.')
        sys.argv = [sys.argv[0], *remaining]
        return wait.main()
    if args.plan is None:
        option = '--worker-attempt' if '--worker-attempt' in remaining else '--analyze'
        require(option in remaining, 'explicit --plan required')
        attempt = Path(remaining[remaining.index(option) + 1])
        document = attempt / 'plan.json'
        scope = (compiler.collector.read(document)['lifecycle_scope'] if document.is_file()
                 else compiler.collector.read(attempt / 'numerical-scope.json'))
        args.plan = Path(scope['plan']['path'])
        if 'runtime_environment' in scope:
            args.profile = 'on' if scope['runtime_environment'][seek.SEEK_ENV] == '1' else 'off'
        require(compiler.digest(args.plan) == scope['plan']['sha256'], 'worker plan changed')
    args.plan = args.plan.resolve(strict=True)
    if args.overhead:
        require(not remaining, 'overhead is a separate CPU-only operation')
        return overhead(args.plan, args.profile)
    if args.compare:
        require(not remaining and args.profile is None, 'comparison is a separate read-only operation')
        result = seek.compare(*(path.resolve(strict=True) for path in args.compare), args.plan)
        output = args.plan.with_name('comparison.json')
        with output.open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write('\n')
        print(json.dumps(result))
        return 0
    configure_stage_plan(args.plan, args.profile)
    sys.argv = [sys.argv[0], *remaining]
    return wait.main()


if __name__ == '__main__':
    raise SystemExit(main())
