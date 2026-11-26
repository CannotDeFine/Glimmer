#!/usr/bin/env python3
"""Run the fixed observed BERT controls inside authorized task-memory units.

Reuse historical read-only workload, ownership and physical-proof helpers.
This current Debug checkpoint is not ordinary timing or full-device evidence.
Temporary user-systemd 16 GiB / zero-swap units require prior authorization.
"""

import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

from artifact_paths import PROJECT_ROOT, create_attempt, resolve_artifact_root
from run_rnn_v8_validation import cgroup, command, digest, identity, memory_receipt, save, wait_for_idle

LEGACY = PROJECT_ROOT / 'build/tally-full-implementation-1'
BUILD = PROJECT_ROOT / 'build/debug'
MODES = ('native', 'equal', 'priority')
SUCCESS = 'complete_fixed_application_numerical_diagnostic_passed'
FILE_MAX = 1024 ** 3
OUTPUT_MAX = 6 * 1024 ** 3
MEMORY_MAX = 16 * 1024 ** 3
EXISTING_NATIVE = PROJECT_ROOT / 'artifacts/runs/bert-current-candidate/attempt-0yofu28n'
EXISTING_NATIVE_SHA = {
    'training/result.json': 'bd7b6182e629af1f8d231238064c9d50c686edf6b81f305dbd22968308ec089c',
    'training/snapshot.pt': '6ebc766be1716e340b9759549febab78b9567d46fce9bbb8984a57cf2b82677a',
    'inference/result.json': '6ab4bcdfbe4f2e5aa300f5e6cd81dc016eb5db2deb2b29511e3a637c7d499547',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def selected_modes(equal_only):
    return ('equal',) if equal_only else MODES


def load_helpers():
    # Cold --help never imports historical helpers or queries the GPU.
    sys.path.insert(0, str(LEGACY))
    import collect_application_block_colocation as common
    h = common.load_helpers()
    require(common.ROOT == PROJECT_ROOT and h.training.SERVER == BUILD / 'bin/glimmer_execution_service'
            and h.training.PRELOAD == BUILD / 'lib/libglimmer_execution_interceptor.so',
            'only the current explicit Debug candidate is selected')
    return common, h


def check_output(attempt):
    used = sum(p.stat().st_size for p in attempt.rglob('*') if p.is_file())
    require(used <= OUTPUT_MAX - 8 * 1024 ** 2, 'sampled aggregate6GiB output budget exceeded')
    return used


class ResourceGuard:
    """Single collector-thread checks; never change the arrival release parser."""

    def __init__(self, attempt, mode, clock=time.monotonic):
        self.attempt, self.clock, self.last = attempt, clock, None
        self.group = cgroup(os.getpid())
        expected = f'/glimmer-bert-{attempt.name}-{mode}.service'
        require(self.group.endswith(expected), 'worker must run in its unique task-only unit')
        require(resource.getrlimit(resource.RLIMIT_FSIZE) == (FILE_MAX, FILE_MAX),
                'actual inherited1GiB per-file limit required')
        self.participants = {}
        self.receipt = dict(observer=dict(pid=os.getpid(), start=identity(os.getpid()), cgroup=self.group),
                            per_file_limit=FILE_MAX, sampled_aggregate_output_limit=OUTPUT_MAX,
                            participants=[], resource_checks=0)
        self.check(force=True)
        self.receipt['initial_memory'] = self.receipt['last_memory']

    def observe(self, process, start):
        require(identity(process.pid) == start and cgroup(process.pid) == self.group,
                'exact application/service must inherit the capped observer cgroup')
        previous = self.participants.get(process.pid)
        require(previous is None or previous[1] == start, 'participant PID identity changed')
        if previous is None:
            self.participants[process.pid] = process, start
            self.receipt['participants'].append(dict(pid=process.pid, start=start, cgroup=self.group))

    def check(self, force=False):
        now = self.clock()
        if not force and self.last is not None and now - self.last < .2:
            return
        self.last = now
        memory = memory_receipt(self.group)
        events = dict(line.split() for line in memory['memory.events'].splitlines())
        require(events and all(int(value) == 0 for value in events.values()),
                'task memory pressure/OOM event invalidates the control')
        for pid, (process, start) in self.participants.items():
            if process.poll() is not None:
                continue
            try:
                require(identity(pid) == start and cgroup(pid) == self.group, 'live participant escaped its unit')
            except (FileNotFoundError, ProcessLookupError):
                require(process.poll() is not None, 'live participant identity read failed')
        self.receipt.update(last_memory=memory, last_output_bytes=check_output(self.attempt))
        self.receipt['resource_checks'] += 1


def normal_cleanup(job):
    require(job['status'] == SUCCESS, 'original observed control failed: ' + str(job.get('error')))
    for role in ('low', 'high', 'service') if job['mode'] != 'native' else ('low', 'high'):
        cleanup = job['authenticated_child_cleanup'][role]
        expected = job['server_identity'] if role == 'service' else job['application_identities'][role]
        require(type(job['final_returncodes'][role]) is int and job['final_returncodes'][role] == 0
                and cleanup['exit_confirmed'] is True and cleanup['signals'] == []
                and (cleanup['pid'], cleanup['start']) == (expected['pid'], expected['start']),
                'normal unsignalled exact child cleanup required')
    require(job['socket_retained'] is False and job['reference_before'] == job['reference_after'],
            'socket or numerical reference lifecycle failed')


def existing_native_reference(root, gpu, startup_bundle, startup_sha):
    """Require the exact successful prior native result, not a new native run."""
    report_path, worker_path = root / 'report.json', root / 'native-worker.json'
    attempt_path, manifest_path = root / 'native/attempt.json', root / 'manifest.json'
    report, worker = json.loads(report_path.read_text()), json.loads(worker_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    natives = [entry for entry in report['jobs'] if entry['mode'] == 'native']
    require(report['status'] == 'current_candidate_observed_controls_require_review'
            and report['input_identity_unchanged'] is True and report['gpu_identity_after'] == gpu
            and len(natives) == 1 and natives[0]['unit']['returncode'] == 0
            and natives[0]['unit']['unit_removed'] is True and natives[0]['worker'] == worker
            and worker['status'] == 'passed_requires_review', 'prior native control is not qualified')
    require(manifest == dict(gpu=gpu, startup_reference_bundle=str(startup_bundle),
                             startup_reference_report_sha256=startup_sha),
            'prior native input and GPU identity changed')
    job = worker['job']
    require(json.loads(attempt_path.read_text()) == job, 'prior native raw receipt differs')
    normal_cleanup(job)
    require(worker['reparse'] == dict(tensors=401, elements=217787909, full_tensor_recomparison=True,
                                      source_bound_complete_inference=True, first_command_handoff=False),
            'prior native did not retain the full numerical oracle')
    paths = {report_path, worker_path, attempt_path, manifest_path}
    for name, expected in EXISTING_NATIVE_SHA.items():
        path = root / 'native' / name
        require(digest(path) == expected == job['artifact_identities'][name],
                'prior native numerical bytes changed')
        paths.add(path)
    return paths, job


def invalidate_publication(job):
    """Invalidate the derived envelope, preserving the raw attempt on disk."""
    job['outer_qualification_invalidated'] = True
    job['pre_demand_compute_observed'] = False
    if 'high_release_trigger' in job:
        job['high_release_trigger']['useful_compute_proof'] = False
    if 'application_priority' in job:
        job['application_priority']['partial_handoff_component_verified'] = False
        job['application_priority']['pre_demand_compute_observed'] = False


def reparse(common, h, attempt, mode, job, gpu, native_directory=None):
    """Reparse raw receipts and every CPU tensor after the apps have exited."""
    path = attempt / mode
    require(json.loads((path / 'attempt.json').read_text()) == job, 'raw/embedded attempt mismatch')
    normal_cleanup(job)
    for name, value in job['artifact_identities'].items():
        require(digest(path / name) == value, 'raw output identity changed: ' + name)
    native_directory = attempt / 'native' if native_directory is None else native_directory
    reference = native_directory / 'training' if mode != 'native' else h.startup_reference
    training = h.training.validate_result(path / 'training', reference, gpu)
    inference = h.validate_onnx(path / 'inference', gpu, h.export, expected_warmup=1, expected_iterations=2,
                                reference=native_directory / 'inference' if mode != 'native' else None)
    require(training == job['training_numerical'] and inference == job['inference_numerical'],
            'independent numerical receipt differs')
    if mode != 'native':
        actual, selection = common.execution_selection(h, path / 'server.log', job['priority_audit_start'],
            job['effective_selection'], 0 if mode == 'equal' else 100)
        require(actual == job['actual_execution'] and selection == job['selection'], 'raw execution join differs')
        if mode == 'priority':
            text = (path / 'server.log').read_text()
            proof = h.analyze(text, job['priority_audit_start'], demand_position=job['high_demand_position'])
            require(proof == job['application_priority'] and proof['pre_demand_compute_observed'] is True,
                    'original first-generation handoff absent')
            low = job['application_identities']['low']
            trigger = common.arrival_publication(text[:job['high_demand_position']], job['priority_audit_start'],
                                                 low['pid'], low['start'], 2842)
            require(trigger is not None and any(item['low']['sequence'] == 2842
                    and item['low']['operation'] == 42 and item['publication']['position'] == trigger['position']
                    for item in proof['pre_demand_handoff_components']), 'another publication cannot substitute')
    import torch
    require(not torch.cuda.is_initialized(), 'numerical reparse must remain CPU-only')
    actual = torch.load(path / 'training/snapshot.pt', map_location='cpu', weights_only=True)
    original = torch.load(reference / 'snapshot.pt', map_location='cpu', weights_only=True)
    require(type(actual) is dict and set(actual) == set(original) and len(actual) == 401,
            'complete401-tensor inventory required')
    elements = 0
    for name, value in actual.items():
        require(value.device.type == original[name].device.type == 'cpu', 'CPU tensors required')
        torch.testing.assert_close(value, original[name], rtol=2e-4, atol=2e-5, msg=name)
        elements += value.numel()
    require(elements == 217787909 and not torch.cuda.is_initialized(), 'full CPU mathematical scope required')
    return dict(tensors=401, elements=elements, full_tensor_recomparison=True,
                source_bound_complete_inference=True, first_command_handoff=mode == 'priority')


def guarded_control(common, h, attempt, mode, environment, guard, native_directory=None):
    verify, pending = common.verify_process, common.check_pending

    def checked_verify(helpers, process, start, *args, **kwargs):
        verify(helpers, process, start, *args, **kwargs)
        guard.observe(process, start)

    def checked_pending(*args, **kwargs):
        guard.check()
        return pending(*args, **kwargs)

    common.verify_process, common.check_pending = checked_verify, checked_pending
    try:
        native_directory = attempt / 'native' if native_directory is None else native_directory
        job = common.run_control(h, attempt / mode, mode, environment, native_directory,
                                 priority_idle_us=2000, gemm_tile_side=32)
        # A successful historical receipt cannot mask a task-resource failure.
        guard.check(force=True)
        normal_cleanup(job)
        return job
    finally:
        common.verify_process, common.check_pending = verify, pending


def worker(attempt, mode):
    report, guard = dict(status='started', mode=mode), None
    try:
        require(not os.environ.get('LD_PRELOAD') and not os.environ.get('LD_AUDIT'), 'clean observer loader')
        manifest = json.loads((attempt / 'manifest.json').read_text())
        gpu = manifest['gpu'].split(',')[0].strip()
        guard = ResourceGuard(attempt, mode)
        common, h = load_helpers()
        h.startup_reference, h.export = common.application_references(h, manifest)
        native_directory = Path(manifest['native_reference']) if 'native_reference' in manifest else None
        if native_directory is not None:
            require(mode == 'equal' and native_directory == EXISTING_NATIVE / 'native',
                    'only the sealed equal-only reference is supported')
        require(h.query('gpu', h.training.GPU_FIELDS) == manifest['gpu'], 'unchanged GPU/runtime identity')
        environment = h.training.clean_environment(gpu)
        report['job'] = guarded_control(common, h, attempt, mode, environment, guard, native_directory)
        report['reparse'] = reparse(common, h, attempt, mode, report['job'], gpu, native_directory)
        guard.check(force=True)
        report['status'] = 'passed_requires_review'
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        report['scoped_publication_valid'] = report['status'] == 'passed_requires_review'
        if report['status'] == 'failed' and 'job' in report:
            invalidate_publication(report['job'])
        if guard is not None:
            report['resources'] = guard.receipt
        save(attempt / (mode + '-worker.json'), report)
    print(json.dumps(dict(mode=mode, status=report['status'], error=report.get('error'))), flush=True)
    return int(report['status'] != 'passed_requires_review')


def launch_unit(attempt, mode, *, worker_script=None):
    unit = f'glimmer-bert-{attempt.name}-{mode}.service'
    require(command(['systemctl', '--user', 'show', unit, '--property=LoadState', '--value']).strip()
            == 'not-found', 'never reuse an existing unit')
    run = ['systemd-run', '--user', '--unit=' + unit, '--wait', '--pipe',
           '--property=MemoryMax=' + str(MEMORY_MAX), '--property=MemorySwapMax=0',
           '--property=OOMPolicy=kill', '--property=RuntimeMaxSec=800s', '--property=TimeoutStopSec=5s',
           '--property=KillMode=control-group', '--property=TasksMax=256', '--property=LimitFSIZE=' + str(FILE_MAX),
           '/usr/bin/env', '-u', 'LD_PRELOAD', '-u', 'LD_LIBRARY_PATH',
           str(PROJECT_ROOT / 'examples/framework_workloads/pytorch_smoke/.venv/bin/python'), '-B',
           str(Path(__file__).resolve() if worker_script is None else Path(worker_script).resolve(strict=True)),
           '--worker-attempt', str(attempt), '--worker-mode', mode]
    result = dict(mode=mode, unit=unit)
    try:
        with (attempt / (mode + '-unit.log')).open('x') as log:
            result['returncode'] = subprocess.run(run, stdout=log, stderr=subprocess.STDOUT,
                                                  timeout=820, check=False).returncode
    except subprocess.TimeoutExpired:
        result.update(returncode=-1, timeout=True)
    finally:
        result['unit_removed'] = False
        try:
            properties = command(['systemctl', '--user', 'show', unit, '--property=LoadState,ActiveState,'
                                  'SubState,Result,ExecMainCode,ExecMainStatus,MemoryPeak,ControlGroup'])
            result['unit_properties'] = properties
            if 'LoadState=not-found' not in properties:
                command(['systemctl', '--user', 'stop', unit], 10)
                if 'ActiveState=failed' in command(['systemctl', '--user', 'show', unit, '--property=ActiveState']):
                    command(['systemctl', '--user', 'reset-failed', unit], 5)
            result['unit_removed'] = command(['systemctl', '--user', 'show', unit,
                                               '--property=LoadState', '--value']).strip() == 'not-found'
        except BaseException as error:
            result['cleanup_error'] = f'{type(error).__name__}: {error}'
        save(attempt / (mode + '-unit.json'), result)
    require(result['unit_removed'] and result['returncode'] == 0, mode + ' bounded task failed')
    receipt = json.loads((attempt / (mode + '-worker.json')).read_text())
    require(receipt['status'] == 'passed_requires_review', mode + ' worker/resource/audit failure')
    return result, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-for-idle', action='store_true', help='at most10min for60s sampled idle; one batch')
    parser.add_argument('--equal-only', action='store_true',
                        help='run equal only using the sealed prior native numerical reference')
    parser.add_argument('--worker-attempt', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--worker-mode', choices=MODES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_attempt:
        require(args.worker_mode is not None, 'internal worker mode required')
        return worker(args.worker_attempt.resolve(strict=True), args.worker_mode)
    attempt = create_attempt(resolve_artifact_root('bert-current-candidate'))
    print(f'BERT candidate evidence: {attempt}', flush=True)
    report, before, paths = dict(status='started', jobs=[], full_device_qualified=False,
        physical_priority_qualified=False, saturation_qualified=False, performance_qualified=False,
        full_goal_complete=False, ordinary_timing_evidence=False), None, None
    try:
        require(not os.environ.get('LD_PRELOAD') and not os.environ.get('LD_AUDIT'), 'clean collector loader')
        import shutil
        require(shutil.disk_usage(attempt).free >= 8 * 1024 ** 3, '8GiB free required; no deletion')
        common, h = load_helpers()
        gpu = h.query('gpu', h.training.GPU_FIELDS)
        fields = [value.strip() for value in gpu.split(',')]
        require(len(fields) == 4 and '\n' not in gpu and fields[0].startswith('GPU-') and 'V100' in fields[1],
                'one identified V100 required')
        require(fields[0].removeprefix('GPU-') == h.training.read_json(
            h.training.ORIGINAL / 'result.json')['environment']['device_uuid'], 'original GPU identity required')
        manifest = dict(gpu=gpu, startup_reference_bundle=str(common.STARTUP_INPUTS),
                        startup_reference_report_sha256=common.STARTUP_INPUTS_SHA)
        h.startup_reference, h.export = common.application_references(h, manifest)
        paths = common.input_paths(h) | {Path(__file__), PROJECT_ROOT / 'scripts/artifact_paths.py',
            PROJECT_ROOT / 'scripts/run_rnn_v8_validation.py', PROJECT_ROOT / 'scripts/rnn_v8_audit.py',
            PROJECT_ROOT / 'scripts/cudnn8_legacy_audit.py', BUILD / 'CMakeCache.txt', BUILD / 'compile_commands.json'}
        if args.equal_only:
            native_paths, native_job = existing_native_reference(EXISTING_NATIVE, gpu, common.STARTUP_INPUTS,
                                                                 common.STARTUP_INPUTS_SHA)
            inference = native_job['inference_numerical']
            require(digest(h.export / 'export.json') == inference['export_sha256']
                    and digest(h.export / 'bert.onnx') == inference['model_sha256'],
                    'prior native model and export differ from current input')
            example = PROJECT_ROOT / 'examples/framework_workloads/bert'
            expected_sources = {'bert_contract.py', 'bert_workload.py', 'onnx_bert_workload.py', 'export_bert.py'}
            require(set(inference['source_sha256']) == expected_sources
                    and all(digest(example / name) == value for name, value in inference['source_sha256'].items()),
                    'prior native workload source differs from current input')
            manifest['native_reference'] = str(EXISTING_NATIVE / 'native')
            paths.update(native_paths)
            report.update(reference_mode='existing_native_numerical_only',
                          native_reference=manifest['native_reference'], fresh_native_run=False)
        before = h.training.manifest(paths)
        save(attempt / 'inputs-before.json', before)
        save(attempt / 'manifest.json', manifest)
        if args.wait_for_idle:
            report['idle_window'] = wait_for_idle(attempt)
            require(report['idle_window']['achieved'], 'idle window unavailable; no experiment started')
        h.training.require_idle()
        require(before == h.training.manifest(paths), 'frozen input changed before batch')
        for mode in selected_modes(args.equal_only):
            h.training.require_idle()
            unit, receipt = launch_unit(attempt, mode)
            report['jobs'].append(dict(mode=mode, unit=unit, worker=receipt))
            check_output(attempt)
            save(attempt / 'report.json', report)
        h.training.require_idle()
        report['status'] = ('equal_only_existing_native_requires_review' if args.equal_only
                            else 'current_candidate_observed_controls_require_review')
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        if before is not None:
            try:
                after = h.training.manifest(paths)
                save(attempt / 'inputs-after.json', after)
                report['input_identity_unchanged'] = before == after
                report['gpu_identity_after'] = h.query('gpu', h.training.GPU_FIELDS)
                require(before == after and report['gpu_identity_after'] == gpu,
                        'frozen source/binary/provider/input/GPU identity changed')
            except BaseException as error:
                report.update(status='failed', identity_error=str(error))
        if report['status'] == 'failed':
            for embedded in report['jobs']:
                embedded['worker']['scoped_publication_valid'] = False
                invalidate_publication(embedded['worker']['job'])
        save(attempt / 'report.json', report)
    print(json.dumps(dict(status=report['status'], error=report.get('error'), report=str(attempt / 'report.json'))))
    return int(report['status'] == 'failed')


if __name__ == '__main__':
    raise SystemExit(main())
