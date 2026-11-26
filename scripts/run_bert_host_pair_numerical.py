#!/usr/bin/env python3
"""Validate a reviewed immutable host compiler pair with complete BERT outputs."""

import argparse
import json
import os
from pathlib import Path
import shutil
import stat
import time

from artifact_paths import PROJECT_ROOT, create_attempt, resolve_artifact_root
import run_bert_candidate_validation as candidate
import run_bert_direct_cpu_numerical as numerical
from run_rnn_v8_validation import digest, save, wait_for_idle

ENTRY = Path(__file__).resolve()
require = candidate.require
MODES = ('native', 'equal')
VARIANTS = ('o0', 'o2')
TOTAL_LIMIT = 8 * 1024 ** 3
DISK_FLOOR = 8 * 1024 ** 3
ORACLE = dict(tensors=401, elements=217787909, full_tensor_recomparison=True,
              source_bound_complete_inference=True, first_command_handoff=False)
BASE_ENVIRONMENT_KEYS = {'PATH', 'LANG', 'LC_ALL', 'CUDA_VISIBLE_DEVICES', 'LD_LIBRARY_PATH',
                         'PYTHONNOUSERSITE', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                         'OPENBLAS_NUM_THREADS', 'TOKENIZERS_PARALLELISM',
                         'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'}


def read(path):
    return json.loads(Path(path).read_text())


def require_launch_review(plan):
    require(digest(ENTRY) == plan['entrypoint_sha256'], 'reviewed qualifier entrypoint changed')
    require(digest(plan['qualifier_review']) == plan['qualifier_review_sha256'], 'qualifier review changed')
    review = read(plan['qualifier_review'])
    require(review['status'] == 'numerical_qualifier_ready' and
            review['entry_sha256'] == plan['entrypoint_sha256'], 'qualifier review not accepted')


def validate_plan(plan):
    if 'build_contract' not in plan:
        require('build_contract' not in read(plan['runtime_manifest']),
                'legacy plan cannot select a normal runtime manifest')
        return validate_legacy_plan(plan)
    require(plan['build_contract'] == 'normal_root_v1', 'unknown host pair build contract')
    return validate_normal_plan(plan)


def validate_normal_plan(plan):
    """Bind the reviewed normal build, without depending on mutable build slots."""
    require(plan['variant_order'] == list(VARIANTS) and plan['modes'] == list(MODES)
            and plan['runtime_environment'] == {'GLIMMER_EXECUTION_HOST_RANGE_SEEK': '1'},
            'unchanged four-path normal pair contract required')
    build = Path(plan['build']).resolve(strict=True)
    require(Path(plan['build_review']) == build / 'review.json', 'normal build review binding')
    paths = {Path(plan[key]) for key in ('build_review', 'runtime_manifest')}
    for key in ('build_review', 'runtime_manifest'):
        require(digest(plan[key]) == plan[key + '_sha256'], key + ' changed')
    review, runtime = read(plan['build_review']), read(plan['runtime_manifest'])
    require(review['status'] == 'normal_execution_host_o2_cpu_integration_accepted',
            'independent normal CPU acceptance required')
    require(runtime['build_contract'] == plan['build_contract'] and runtime['build'] == str(build)
            and runtime['variant_options'] == {'o0': 'OFF', 'o2': 'ON'}
            and set(runtime['variants']) == set(VARIANTS), 'explicit normal OFF/ON pair required')
    evidence = {'off-verification.json', 'on-verification.json', 'pair-link-closure.json',
                'current-owned-source.tar.gz'}
    require(set(review['evidence_sha256']) == evidence, 'complete reviewed normal evidence required')
    for name, expected in review['evidence_sha256'].items():
        path = build / name
        require(digest(path) == expected, 'normal build evidence changed: ' + name)
        paths.add(path)
    required_sources = {'CMakeLists.txt', 'cmake/ExecutionHostOptimization.cmake',
                        'tests/CMakeLists.txt', 'tests/build/execution_host_optimization_test.py',
                        'docs/BUILDING.md'}
    require(set(review['owned_changes']) == required_sources, 'normal build source contract changed')
    for name, expected in review['owned_changes'].items():
        path = PROJECT_ROOT / name
        require(digest(path) == expected, 'reviewed normal source changed: ' + name)
        paths.add(path)
    require(set(runtime['config_evidence']) ==
            {'configuration-audit.json', 'off-graph.tar.gz', 'on-graph.tar.gz'},
            'normal configuration evidence required')
    for name, expected in runtime['config_evidence'].items():
        path = build / name
        require(digest(path) == expected, 'normal configuration evidence changed: ' + name)
        paths.add(path)
    configuration = read(build / 'configuration-audit.json')
    require(configuration['status'] == 'normal_root_off_on_off_configuration_audit_passed' and
            all(configuration[mode + '_graph_sha256'] == runtime['config_evidence'][mode + '-graph.tar.gz']
                for mode in ('off', 'on')), 'normal graph binding changed')
    closure = read(build / 'pair-link-closure.json')
    require(closure['status'] == 'complete_normal_pair_link_closure_verified'
            and closure['all_six_archives_seen'] is True
            and closure['off_on_extracted_sets_equal'] is True
            and closure['all_members_in_ordered_79_member_inventory'] is True
            and closure['source_capsule_sha256'] == review['evidence_sha256']['current-owned-source.tar.gz']
            and closure['project_members']['off'] == closure['project_members']['on'],
            'complete normal link closure required')
    names = {'glimmer_execution_service', 'glimmer_execution_interceptor'}
    reports = {mode: read(build / (mode + '-verification.json')) for mode in ('off', 'on')}
    require(reports['off']['status'] == 'normal_variant_cpu_evidence_recorded'
            and reports['on']['status'] ==
            'normal_variant_cpu_evidence_recorded_after_independent_predicate_review'
            and reports['off']['abi'] == reports['on']['abi']
            and reports['on']['pair_contracts_equal'] is True
            and reports['on']['frontend_exports_equal'] is True
            and reports['on']['service_strong_exports_equal'] is True, 'normal CPU pair gates required')
    for variant, mode in (('o0', 'off'), ('o2', 'on')):
        report = reports[mode]
        require(report['variant'] == mode and report['compiled_production_cxx_objects'] == 82
                and len(report['objects']) == 82
                and report['tests_passed'] == 45 and set(report['products']) == names
                and set(runtime['variants'][variant]) == names
                and closure['project_member_counts'][mode] ==
                {'glimmer_execution_service': 47, 'glimmer_execution_interceptor': 13},
                'complete normal variant required')
        archives = report['archives']
        require(len(archives) == 6 and archives.keys() == reports['off']['archives'].keys()
                and sum(len(item['members']) for item in archives.values()) == 79
                and all(item['members'] == reports['off']['archives'][name]['members']
                        for name, item in archives.items()), 'normal ordered archive inventory changed')
        inventory = {item['path']: set(item['members']) for item in archives.values()}
        require(len(inventory) == 6 and set(closure['project_members'][mode]) == names,
                'normal archive or closure inventory incomplete')
        seen = set()
        for name in names:
            members = closure['project_members'][mode][name]
            require(len(members) == closure['project_member_counts'][mode][name] and
                    len({(item['archive'], item['member']) for item in members}) == len(members),
                    'normal extracted member count changed')
            for item in members:
                require(item['archive'] in inventory and item['member'] in inventory[item['archive']],
                        'normal extracted member outside rebuilt archives')
                seen.add(item['archive'])
        require(seen == inventory.keys(), 'normal closure does not cover every archive')
        for name, item in runtime['variants'][variant].items():
            product = report['products'][name]
            path = Path(item['path'])
            require(path == Path(product['path']) and path.resolve(strict=True) == path
                    and path.parent == build / mode, 'normal product variant/path changed')
            info = path.stat()
            require(stat.S_IMODE(info.st_mode) == item['mode'] == 0o555 and
                    (info.st_dev, info.st_ino, info.st_size, info.st_ctime_ns) ==
                    (item['device'], item['inode'], item['size'], item['ctime_ns']),
                    'normal runtime inode/size/mode/ctime changed')
            require(digest(path) == item['sha256'] == product['sha256'], 'normal product changed')
            link_map = build / f'{mode}-{name}.map'
            require(digest(link_map) == product['map_sha256'] == item['link_map_sha256'],
                    'normal product map changed')
            paths.update((path, link_map))
    return paths


def validate_legacy_plan(plan):
    require(plan['variant_order'] == list(VARIANTS) and plan['modes'] == list(MODES),
            'fixed four-path order required')
    require(plan['runtime_environment'] == {'GLIMMER_EXECUTION_HOST_RANGE_SEEK': '1'},
            'unchanged runtime policy required')
    for key in ('build_review', 'runtime_manifest'):
        require(digest(plan[key]) == plan[key + '_sha256'], key + ' changed')
    review = read(plan['build_review'])
    require(review['status'] == 'consistent_host_pair_cpu_verified' and
            review['application_validation_ready'] is True, 'independent CPU pair acceptance required')
    build = Path(plan['build']).resolve(strict=True)
    require(Path(plan['build_review']) == build / 'review.json', 'build review binding')
    runtime = read(plan['runtime_manifest'])
    require(runtime['build'] == str(build) and set(runtime['variants']) == set(VARIANTS),
            'exact immutable runtime pair required')
    paths = {Path(plan['build_review']), Path(plan['runtime_manifest']),
             build / 'frozen-inputs.json', build / 'build-report.json'}
    for name, expected in read(build / 'frozen-inputs.json').items():
        require(Path(name).is_file() and digest(name) == expected, 'frozen build input changed: ' + name)
        paths.add(Path(name))
    for variant in VARIANTS:
        products = read(build / (variant + '-products.json'))
        members = read(build / (variant + '-working-identities.json'))
        capsule = build / (variant + '-working.tar.gz')
        seal = read(build / (variant + '-capsule.json'))
        require(seal['all_members_byte_verified'] is True and digest(capsule) == seal['sha256'],
                'verified capsule changed')
        paths.update((capsule, build / (variant + '-capsule.json'),
                      build / (variant + '-products.json'), build / (variant + '-working-identities.json')))
        for name, item in runtime['variants'][variant].items():
            path = Path(item['path'])
            require(path.resolve(strict=True) == path and path.parent ==
                    Path(plan['runtime_manifest']).parent / variant, 'canonical immutable runtime path')
            info = path.stat()
            require(stat.S_IMODE(info.st_mode) == item['mode'] == 0o555 and
                    (info.st_dev, info.st_ino, info.st_size, info.st_ctime_ns) ==
                    (item['device'], item['inode'], item['size'], item['ctime_ns']),
                    'runtime inode/size/mode/ctime changed')
            require(digest(path) == item['sha256'] == products[name]['sha256'] == members[item['member']]
                    and item['capsule'] == str(capsule) and item['capsule_sha256'] == seal['sha256']
                    and item['link_map_sha256'] == products[name]['map_sha256'] ==
                    members['products/' + name + '.map'], 'runtime capsule/product/map identity changed')
            paths.add(path)
    return paths


def selected_plan(plan, variant):
    require(variant in VARIANTS, 'known compiler variant required')
    products = read(plan['runtime_manifest'])['variants'][variant]
    return dict(plan, variant=variant,
                server=products['glimmer_execution_service']['path'],
                frontend=products['glimmer_execution_interceptor']['path'])


def aggregate_budget(root):
    require(shutil.disk_usage(root).free >= DISK_FLOOR, '8GiB disk reserve required')
    used = sum(path.stat().st_size for path in root.rglob('*') if path.is_file())
    require(used <= TOTAL_LIMIT - 16 * 1024 ** 2, 'whole four-path output limit')
    return used


class PairGuard(candidate.ResourceGuard):
    def __init__(self, attempt, mode):
        self.aggregate = Path(read(attempt / 'plan.json')['aggregate_attempt'])
        require(self.aggregate.resolve(strict=True) == self.aggregate and
                attempt.parent.parent == self.aggregate, 'owned nested output namespace required')
        self.last_budget_check = None
        super().__init__(attempt, mode)

    def check(self, force=False):
        super().check(force)
        now = time.monotonic()
        if not force and self.last_budget_check is not None and now - self.last_budget_check < .2:
            return
        self.last_budget_check = now
        root = Path('/sys/fs/cgroup' + self.group)
        pids = dict(line.split() for line in (root / 'pids.events').read_text().splitlines())
        require((root / 'pids.max').read_text().strip() == '256' and pids and
                all(int(value) == 0 for value in pids.values()), 'task pressure or wrong task limit')
        require((root / 'memory.swap.current').read_text().strip() == '0', 'task used swap')
        self.receipt['aggregate_output_bytes'] = aggregate_budget(self.aggregate)
        self.receipt['pids_events'] = pids
        self.receipt['swap_current'] = 0


def validate_job(plan, job):
    numerical.validate_job(plan, job)
    environments = [job['application_environment'], *job['application_environments'].values()]
    allowed = {'GLIMMER_EXECUTION_SOCKET', 'GLIMMER_EXECUTION_SERVER_PID',
               'GLIMMER_EXECUTION_SERVER_START', 'GLIMMER_EXECUTION_PREPARATION_TIMEOUT_MS',
               'GLIMMER_EXECUTION_HOST_RANGE_SEEK'}
    for environment in environments:
        keys = {key for key in environment if key.startswith('GLIMMER_')}
        require(keys == (allowed if job['mode'] == 'equal' else set()), 'extra application option')
        expected_keys = BASE_ENVIRONMENT_KEYS | (allowed | {'LD_PRELOAD'} if job['mode'] == 'equal' else set())
        require(set(environment) == expected_keys, 'unexpected loader/profiler/application environment')
        require(all(environment[key] == '2' for key in
                    ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')), 'CPU geometry differs')
    for command in job['application_commands'].values():
        for option, value in (('--warmup', '1'), ('--iterations', '2')):
            require(command.count(option) == 1 and command[command.index(option) + 1] == value,
                    'complete fixed application command required')
    if job['mode'] != 'equal':
        return
    command = job['server_command']
    socket = command[command.index('--socket') + 1]
    pins, normalized = [], []
    index = 0
    while index < len(command):
        if command[index] == '--client-priority':
            pins.append(command[index + 1])
            index += 2
        else:
            normalized.append(command[index])
            index += 1
    expected = [plan['server'], '--socket', socket, '--exit-when-idle', '--persistent-workers', '320',
                '--kernel-mode', 'preemptive', '--scheduler', 'priority', '--replace-gemm',
                '--audit-native', '--audit-priority', '--observe-block-completion',
                '--memory-per-client', '12884901888', '--host-memory-per-client', '536870912',
                '--priority-idle-us', '2000', '--gemm-tile-side', '32']
    require(normalized == expected, 'extra or changed service option')
    actual_pins = [value.split(':') for value in pins]
    require(len(actual_pins) == 2 and all(len(value) == 4 and value[2] == '0' for value in actual_pins)
            and {value[3] for value in actual_pins} == {'configured', 'original'} and
            {(int(value[0]), int(value[1])) for value in actual_pins} ==
            {(value['pid'], value['start']) for value in job['application_identities'].values()},
            'exact two equal-priority client identities required')
    require(set(job['server_environment']) == BASE_ENVIRONMENT_KEYS, 'extra service environment option')
    for environment in environments:
        require(all(environment[key] == job['server_environment'][key] for key in BASE_ENVIRONMENT_KEYS),
                'application and service runtime environments differ')
        require(environment['GLIMMER_EXECUTION_SOCKET'] == socket and
                environment['GLIMMER_EXECUTION_SERVER_PID'] == str(job['server_identity']['pid']) and
                environment['GLIMMER_EXECUTION_SERVER_START'] == str(job['server_identity']['start']) and
                environment['GLIMMER_EXECUTION_PREPARATION_TIMEOUT_MS'] == '240000',
                'owned service binding changed')


def worker(attempt, mode):
    report = dict(status='failed', mode=mode, **numerical.FLAGS)
    try:
        plan = read(attempt / 'plan.json')
        numerical.authorization(plan, 800)
        require_launch_review(plan)
        validate_plan(plan)
        selected = selected_plan(plan, plan['variant'])
        require(selected['server'] == plan['server'] and selected['frontend'] == plan['frontend'],
                'worker variant substitution')
        numerical.configure(plan)
        candidate.ResourceGuard = PairGuard
        status = candidate.worker(attempt, mode)
        report = read(attempt / (mode + '-worker.json'))
        require(status == 0, 'existing complete numerical worker failed: ' + str(report.get('error')))
        validate_job(plan, report['job'])
        require(report['reparse'] == ORACLE, 'complete numerical scope required')
        validate_plan(plan)
        aggregate_budget(Path(plan['aggregate_attempt']))
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}',
                      scoped_publication_valid=False)
        if 'job' in report:
            candidate.invalidate_publication(report['job'])
    report.update(numerical.FLAGS)
    save(attempt / (mode + '-worker.json'), report)
    return int(report['status'] != 'passed_requires_review')


def prelaunch_idle(attempt, variant, mode, plan, h, paths, before, report, *, first):
    """Wait once for another user's GPU work only before an unstarted later task."""
    event = dict(variant=variant, mode=mode, status='checking')
    report.setdefault('prelaunch_idle', []).append(event)
    try:
        try:
            h.training.require_idle()
        except RuntimeError as error:
            busy_prefix = 'GPU is not idle; no next job; never signal foreign work:'
            if first or not str(error).startswith(busy_prefix):
                raise
            event.update(status='waiting', initial_busy=str(error))
            numerical.authorization(plan, 1440)
            directory = attempt / 'prelaunch-idle' / (variant + '-' + mode)
            directory.mkdir(parents=True, exist_ok=False)
            event['wait_directory'] = str(directory)
            event['wait'] = wait_for_idle(directory)
            event['wait_log_sha256'] = digest(directory / 'idle-window.jsonl')
            require(event['wait']['achieved'], 'bounded prelaunch idle wait expired')
            numerical.authorization(plan, 840)
            validate_plan(plan)
            require(h.training.manifest(paths) == before, 'frozen application inputs changed during wait')
            h.training.require_idle()
            event['status'] = 'idle_after_conditional_wait'
        else:
            event['status'] = 'immediate_idle'
    except BaseException as error:
        event.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    return event


def collect(plan_path):
    plan = read(plan_path)
    numerical.authorization(plan, 6200)
    require_launch_review(plan)
    frozen_paths = validate_plan(plan)
    require(not os.environ.get('LD_PRELOAD') and not os.environ.get('LD_AUDIT'), 'clean collector')
    require(shutil.disk_usage(PROJECT_ROOT).free >= TOTAL_LIMIT + DISK_FLOOR, 'output plus disk reserve')
    with plan_path.with_name('numerical-launch.json').open('x') as stream:
        json.dump(dict(entry=str(ENTRY), entry_sha256=digest(ENTRY), plan=str(plan_path),
                       plan_sha256=digest(plan_path)), stream)
    attempt = create_attempt(resolve_artifact_root('bert-host-pair-numerical'))
    print('Host pair numerical evidence: ' + str(attempt), flush=True)
    save(attempt / 'plan.json', plan)
    report = dict(status='started', jobs=[], variant_attempts={}, **numerical.FLAGS)
    before = None
    try:
        common, h = candidate.load_helpers()
        gpu = h.query('gpu', h.training.GPU_FIELDS)
        require('GPU-0db675c8-4623-fc5a-9e7b-1c237f625ce2' in gpu and 'V100' in gpu,
                'identified V100 required')
        manifest = dict(gpu=gpu, startup_reference_bundle=str(common.STARTUP_INPUTS),
                        startup_reference_report_sha256=common.STARTUP_INPUTS_SHA)
        h.startup_reference, h.export = common.application_references(h, manifest)
        paths = frozen_paths | {ENTRY, Path(candidate.__file__).resolve(), Path(numerical.__file__).resolve(),
            plan_path, PROJECT_ROOT / 'scripts/artifact_paths.py',
            PROJECT_ROOT / 'scripts/run_rnn_v8_validation.py', Path(plan['qualifier_review'])}
        for variant in VARIANTS:
            selected = selected_plan(plan, variant)
            h.training.SERVER, h.training.PRELOAD = Path(selected['server']), Path(selected['frontend'])
            paths.update(common.input_paths(h))
            child = create_attempt(resolve_artifact_root('bert-host-pair-' + variant,
                                                         output_root=attempt / variant))
            report['variant_attempts'][variant] = str(child)
            save(child / 'plan.json', dict(selected, aggregate_attempt=str(attempt)))
            save(child / 'manifest.json', manifest)
            paths.add(child / 'plan.json')
            paths.add(child / 'manifest.json')
        before = h.training.manifest(paths)
        save(attempt / 'inputs-before.json', before)
        save(attempt / 'manifest.json', manifest)
        report['idle_window'] = wait_for_idle(attempt)
        require(report['idle_window']['achieved'], 'idle window unavailable; no application run')
        for variant in VARIANTS:
            child = Path(report['variant_attempts'][variant])
            selected = read(child / 'plan.json')
            for mode in MODES:
                numerical.authorization(plan, 840)
                validate_plan(plan)
                require(h.training.manifest(paths) == before, 'frozen application inputs changed before launch')
                prelaunch_idle(attempt, variant, mode, plan, h, paths, before, report,
                               first=variant == VARIANTS[0] and mode == MODES[0])
                unit, result = candidate.launch_unit(child, mode, worker_script=ENTRY)
                validate_job(selected, result['job'])
                require(result['reparse'] == ORACLE, 'full tensor/inference receipt required')
                report['jobs'].append(dict(variant=variant, mode=mode, unit=unit, worker=result))
                aggregate_budget(attempt)
                save(attempt / 'report.json', report)
                print(json.dumps(dict(variant=variant, mode=mode, status='numerical_passed')), flush=True)
        report['status'] = 'fresh_o0_o2_native_equal_numerical_requires_review'
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        if before is not None:
            try:
                after = h.training.manifest(paths)
                save(attempt / 'inputs-after.json', after)
                report['input_identity_unchanged'] = before == after
                report['gpu_identity_after'] = h.query('gpu', h.training.GPU_FIELDS)
                require(before == after and report['gpu_identity_after'] == gpu, 'input or GPU identity changed')
                validate_plan(plan)
                report['total_output_bytes'] = aggregate_budget(attempt)
            except BaseException as error:
                report.update(status='failed', identity_error=f'{type(error).__name__}: {error}')
        save(attempt / 'report.json', report)
    print(json.dumps(dict(status=report['status'], error=report.get('error'), attempt=str(attempt))), flush=True)
    return int(report['status'] == 'failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collect', type=Path)
    parser.add_argument('--check', type=Path)
    parser.add_argument('--worker-attempt', type=Path)
    parser.add_argument('--worker-mode', choices=MODES)
    args = parser.parse_args()
    if args.worker_attempt:
        require(args.worker_mode is not None, 'worker mode required')
        return worker(args.worker_attempt.resolve(strict=True), args.worker_mode)
    if args.check:
        plan = read(args.check)
        paths = validate_plan(plan)
        print(json.dumps(dict(status='immutable_pair_inputs_checked', identities=len(paths))))
        return 0
    require(args.collect is not None, 'explicit --collect required')
    return collect(args.collect.resolve(strict=True))


if __name__ == '__main__':
    raise SystemExit(main())
