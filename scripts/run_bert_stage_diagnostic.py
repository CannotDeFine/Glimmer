#!/usr/bin/env python3
"""Collect one bounded native/equal BERT service-stage diagnostic pair.

This is not GPU-gap, full-device, saturation, or performance qualification.
The new service requires its own complete numerical checkpoint first.
"""

import argparse
import copy
from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import shutil
import sys

from artifact_paths import PROJECT_ROOT, create_attempt, resolve_artifact_root
import bert_device_capture as capture
import run_bert_candidate_validation as candidate
import run_bert_stage_candidate as stage
from run_rnn_v8_validation import digest, save, wait_for_idle

require = candidate.require
FLAGS = dict(capture.FLAGS, diagnostic_only=True, counter_collection=False,
             gpu_gap_causality_qualified=False, cpu_profiled=False)
POINT = dict(period_ms=7, requests=4286, slo_ms=12, burst_size=1, repetitions=1, warmup=1)
ANALYZER = PROJECT_ROOT / 'scripts/analyze_bert_service_stages.py'
STAGE_PLAN = PROJECT_ROOT / 'artifacts/maintenance/bert-sm-diagnosis-20261031/stage-plan.json'


def read(path):
    require(path.stat().st_size <= 8 * 1024 ** 2, 'bounded receipt required')
    return json.loads(path.read_text())


def helpers():
    common, h = stage.helpers()
    sustained = importlib.import_module('collect_sustained_application')
    require(sustained.ordinary.common is common, 'one explicit helper binding required')
    return common, h, sustained


def selected_checkpoint_jobs(reference, report, contract_path):
    """Apply an explicit two-path contract without rewriting a failed batch."""
    contract = read(contract_path)
    require(contract['kind'] == 'explicit_native_equal_diagnostic_scope'
            and contract['reference'] == str(reference)
            and contract['modes'] == ['native', 'equal']
            and contract['priority_qualified'] is False
            and contract['triplet_qualified'] is False
            and contract['full_device_qualified'] is False
            and contract['gpu_gap_causality_qualified'] is False
            and contract['performance_qualified'] is False
            and contract['server'] == str(stage.SERVER)
            and contract['server_sha256'] == digest(stage.SERVER)
            and contract['parent_report_sha256'] == digest(reference / 'report.json')
            and contract['stage_plan_sha256'] == digest(STAGE_PLAN),
            'explicit bound two-path scope required')
    require(report['status'] == 'failed'
            and report['error'] == 'ValueError: priority bounded task failed'
            and [row['mode'] for row in report['jobs']] == ['native', 'equal']
            and read(reference / 'priority-worker.json')['status'] == 'failed',
            'original failed priority batch must remain unqualified')
    expected_names = {'report.json', 'manifest.json', 'service-build-identity.json',
                      'inputs-before.json', 'inputs-after.json', 'priority-worker.json'}
    expected_names.update(mode + suffix for mode in ('native', 'equal')
                          for suffix in ('-worker.json', '-unit.json', '/attempt.json'))
    require(set(contract['raw_receipt_sha256']) == expected_names
            and all(digest(reference / name) == checksum
                    for name, checksum in contract['raw_receipt_sha256'].items()),
            'scope-bound original qualification receipts changed')
    require(read(reference / 'manifest.json')['gpu'] == report['gpu_identity_after'] == contract['gpu'],
            'scope-bound GPU identity changed')
    jobs = []
    for row in report['jobs']:
        mode = row['mode']
        worker = read(reference / (mode + '-worker.json'))
        require(worker['scoped_publication_valid'] is True
                and 'outer_qualification_invalidated' not in worker['job'],
                'original passed raw path publication required')
        invalidated = copy.deepcopy(worker)
        invalidated['scoped_publication_valid'] = False
        candidate.invalidate_publication(invalidated['job'])
        require(row['worker'] == invalidated,
                'failed parent differs beyond its original publication invalidation')
        unit = read(reference / (mode + '-unit.json'))
        require(row['unit'] == unit, 'parent/raw unit receipt differs')
        jobs.append(dict(mode=mode, unit=unit, worker=worker))
    return jobs


def checkpoint(reference, contract_path=None):
    report = read(reference / 'report.json')
    before = read(reference / 'inputs-before.json')
    require(report['input_identity_unchanged'] is True
            and before == read(reference / 'inputs-after.json'), 'fresh complete checkpoint required')
    paths = {reference / name for name in ('report.json', 'manifest.json', 'inputs-before.json',
        'inputs-after.json', 'service-build-identity.json')}
    if contract_path is None:
        require(report['status'] == 'current_candidate_observed_controls_require_review'
                and [row['mode'] for row in report['jobs']] == list(candidate.MODES),
                'complete triplet required unless an explicit two-path contract is supplied')
        jobs = report['jobs']
    else:
        jobs = selected_checkpoint_jobs(reference, report, contract_path)
        paths.update({contract_path, reference / 'priority-worker.json'})
    for row in jobs:
        mode, unit, worker = row['mode'], row['unit'], row['worker']
        require(unit['returncode'] == 0 and unit['unit_removed'] is True
                and worker['status'] == 'passed_requires_review', 'normal numerical unit required')
        require(worker['reparse'] == dict(tensors=401, elements=217787909,
            full_tensor_recomparison=True, source_bound_complete_inference=True,
            first_command_handoff=mode == 'priority'), 'complete original numerical oracle required')
        for name, value in ((mode + '-unit.json', unit), (mode + '-worker.json', worker),
                            (mode + '/attempt.json', worker['job'])):
            path = reference / name
            require(read(path) == value, 'raw numerical checkpoint receipt differs')
            paths.add(path)
        candidate.normal_cleanup(worker['job'])
        if mode != 'native':
            require(worker['job']['server_command'][0] == str(stage.SERVER)
                    and '--host-stage-output' not in worker['job']['server_command'],
                    'actual numerical control did not execute the selected uninstrumented service')
        for name, checksum in worker['job']['artifact_identities'].items():
            path = reference / mode / name
            require(path.resolve(strict=True).is_relative_to(reference)
                    and digest(path) == checksum, 'numerical checkpoint bytes changed')
            paths.add(path)
    binding = stage.build_identity()
    require(binding == read(reference / 'service-build-identity.json'), 'candidate build identity changed')
    common, h, sustained = helpers()
    require(h.training.manifest({Path(name) for name in before}) == before,
            'candidate checkpoint source/binary/input identities changed')
    paths.update(Path(name) for name in before)
    sustained.load_measurement()
    sustained.load_resources()
    paths.update(Path(module.__file__).resolve(strict=True) for module in tuple(sys.modules.values())
        if getattr(module, '__file__', None) and
        Path(module.__file__).is_relative_to(PROJECT_ROOT))
    paths.update({Path(__file__).resolve(), PROJECT_ROOT / 'scripts/bert_device_capture.py',
                  PROJECT_ROOT / 'scripts/run_bert_device_observation.py', ANALYZER, STAGE_PLAN})
    paths.update(path for path in (PROJECT_ROOT / 'examples/framework_workloads/bert').glob('*.py'))
    return common, h, sustained, report, paths


@contextmanager
def stage_hooks(common, sustained, guard, directory, mode):
    owner, verify, pending = sustained.owner_command, common.verify_process, common.check_pending

    def owner_command(*args, **kwargs):
        argv, pins = owner(*args, **kwargs)
        require(mode == 'equal' and '--host-stage-output' not in argv, 'one equal service required')
        return [*argv, '--host-stage-output', str(directory / 'host-stages.json')], pins

    def checked_verify(h, process, start, *args, **kwargs):
        verify(h, process, start, *args, **kwargs)
        guard.observe(process, start)

    def checked_pending(*args, **kwargs):
        guard.check()
        return pending(*args, **kwargs)

    sustained.owner_command, common.verify_process = owner_command, checked_verify
    common.check_pending = checked_pending
    try:
        yield
    finally:
        sustained.owner_command, common.verify_process, common.check_pending = owner, verify, pending


def worker(attempt, mode):
    report, guard = dict(status='started', mode=mode, **FLAGS), None
    try:
        stage.authorization_budget(800)
        require(not os.environ.get('LD_PRELOAD') and not os.environ.get('LD_AUDIT'), 'clean collector required')
        common, h, sustained = helpers()
        plan = read(attempt / 'plan.json')
        require(plan['point'] == POINT and plan['server'] == str(stage.SERVER)
                and digest(stage.SERVER) == plan['server_sha256'], 'exact diagnostic candidate required')
        guard = candidate.ResourceGuard(attempt, mode)
        manifest = read(Path(plan['reference']) / 'manifest.json')
        h.startup_reference, h.export = common.application_references(h, manifest)
        require(h.query('gpu', h.training.GPU_FIELDS) == plan['gpu'], 'GPU identity changed')
        resources, measurement = sustained.load_resources(), sustained.load_measurement()
        report['resources_before'] = resources.resource_snapshot()
        require(report['resources_before']['available_device_sms'] == 80, 'full physical device unavailable')
        directory = attempt / mode
        with stage_hooks(common, sustained, guard, directory, mode):
            job = sustained.run_control(h, measurement, directory, mode,
                h.training.clean_environment(plan['gpu'].split(',')[0].strip()), POINT)
            capture.audit_application(sustained, measurement, h, directory, job, POINT)
        if mode == 'equal':
            output = directory / 'host-stages.json'
            require(output.is_file() and output.stat().st_size <= 512 * 1024 ** 2,
                    'bounded normally published stage output required')
            stages = json.loads(output.read_text())
            require(stages['schema'] == 1 and stages['valid'] is True
                    and stages['diagnostic_only'] is True and stages['idle_enabled'] is False
                    and stages['pid'] == job['server_identity']['pid'], 'valid exact-owner stage data required')
            report['stage_output'] = dict(path=str(output), sha256=digest(output),
                buckets=len(stages['buckets']), clock_reads=stages['clock_reads'])
        raw = directory / 'attempt.json'
        raw_hash = digest(raw)
        raw.rename(directory / 'helper-attempt.json')
        report['job'] = copy.deepcopy(job)
        report['job'].update(FLAGS, host_stage_instrumented=mode == 'equal',
            status='service_stage_diagnostic_requires_review',
            raw_helper_receipt=dict(path='helper-attempt.json', sha256=raw_hash))
        save(directory / 'attempt.json', report['job'])
        report['resources_after'] = resources.resource_snapshot()
        require(report['resources_after'] == report['resources_before'], 'physical resource policy changed')
        guard.check(force=True)
        report['status'] = 'passed_requires_review'
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        if guard is not None:
            report['resources'] = guard.receipt
        save(attempt / (mode + '-worker.json'), report)
    return int(report['status'] != 'passed_requires_review')


def collect(reference, contract_path=None):
    stage.authorization_budget(2300)
    common, h, sustained, qualified, paths = checkpoint(reference, contract_path)
    attempt = create_attempt(resolve_artifact_root('bert-service-stages', category='profiles'))
    print('Service-stage diagnostic evidence: ' + str(attempt), flush=True)
    report = dict(status='started', jobs=[], **FLAGS)
    plan = dict(reference=str(reference), gpu=qualified['gpu_identity_after'], point=POINT,
                server=str(stage.SERVER), server_sha256=digest(stage.SERVER),
                analyzer=dict(path=str(ANALYZER), sha256=digest(ANALYZER)),
                stage_plan=dict(path=str(STAGE_PLAN), sha256=digest(STAGE_PLAN)),
                authorization_deadline_utc=stage.DEADLINE.isoformat())
    if contract_path is not None:
        plan['qualification_scope'] = dict(path=str(contract_path), sha256=digest(contract_path),
            modes=['native', 'equal'], parent_batch_status=qualified['status'],
            priority_qualified=False, triplet_qualified=False)
    save(attempt / 'plan.json', plan)
    before = h.training.manifest(paths)
    save(attempt / 'inputs-before.json', before)
    try:
        require(not os.environ.get('LD_PRELOAD') and not os.environ.get('LD_AUDIT'), 'clean outer collector required')
        require(shutil.disk_usage(attempt).free >= 8 * 1024 ** 3, '8GiB free required; no deletion')
        report['idle_window'] = wait_for_idle(attempt)
        require(report['idle_window']['achieved'], 'idle window unavailable; no capture')
        for mode in ('native', 'equal'):
            stage.authorization_budget(840)
            h.training.require_idle()
            require(before == h.training.manifest(paths), 'frozen inputs changed before unit')
            try:
                unit, result = candidate.launch_unit(attempt, mode, worker_script=Path(__file__))
                report['jobs'].append(dict(mode=mode, unit=unit, worker=result))
            finally:
                after = h.training.manifest(paths)
                save(attempt / (mode + '-inputs-after.json'), after)
                unchanged = before == after
                save(attempt / (mode + '-postflight.json'), dict(input_identity_unchanged=unchanged))
                require(unchanged, 'frozen inputs changed after unit')
            save(attempt / 'report.json', report)
        require(h.query('gpu', h.training.GPU_FIELDS) == plan['gpu'], 'GPU identity changed')
        report.update(status='service_stage_pair_requires_independent_review', input_identity_unchanged=True)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        try:
            after = h.training.manifest(paths)
            save(attempt / 'inputs-after.json', after)
            require(before == after, 'full final input manifest differs')
        except BaseException as error:
            report.update(status='failed', identity_error=f'{type(error).__name__}: {error}')
        save(attempt / 'report.json', report)
    print(json.dumps(dict(status=report['status'], error=report.get('error'))), flush=True)
    return int(report['status'] == 'failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--collect', type=Path, help='fresh numerical candidate checkpoint')
    group.add_argument('--worker-attempt', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--native-equal-contract', type=Path,
                        help='explicit bound authorization for native/equal-only qualification')
    parser.add_argument('--worker-mode', choices=('native', 'equal'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_attempt:
        require(args.worker_mode is not None, 'explicit internal mode required')
        return worker(args.worker_attempt.resolve(strict=True), args.worker_mode)
    if args.collect:
        contract = args.native_equal_contract.resolve(strict=True) if args.native_equal_contract else None
        return collect(args.collect.resolve(strict=True), contract)
    parser.print_help()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
