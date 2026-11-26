#!/usr/bin/env python3
"""One current Debug equal-priority CPU/wait diagnosis, never ordinary timing.

Reuse immutable numerical inputs and installed read-only legacy probe helpers.
No runtime rebuild, sudo, GPU counters or scheduling changes. Cold help/import
does not import vendor libraries or start a process.
"""

import argparse
from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from artifact_paths import PROJECT_ROOT, create_attempt, resolve_artifact_root
import run_bert_candidate_validation as candidate
from run_rnn_v8_validation import digest, save, wait_for_idle

REFERENCE = PROJECT_ROOT / 'artifacts/runs/bert-current-candidate/attempt-0yofu28n'
FLAGS = dict(ordinary_timing_evidence=False, full_device_qualified=False,
             physical_priority_qualified=False, saturation_qualified=False,
             performance_qualified=False, full_goal_complete=False)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def helpers():
    common, h = candidate.load_helpers()
    cpu = importlib.import_module('collect_application_cpu')
    require(cpu.waits.ordinary.common is common, 'one current Debug helper binding required')
    return common, h, cpu


@contextmanager
def resource_hooks(common, guard):
    verify, pending = common.verify_process, common.check_pending

    def checked_verify(h, process, start, *args, **kwargs):
        verify(h, process, start, *args, **kwargs)
        guard.observe(process, start)

    def checked_pending(*args, **kwargs):
        guard.check()
        return pending(*args, **kwargs)

    common.verify_process, common.check_pending = checked_verify, checked_pending
    try:
        yield
        guard.check(force=True)
    finally:
        common.verify_process, common.check_pending = verify, pending


def check_profile(job, directory, cpu):
    owner, start = job['server_identity'], job['cpu_profile_start']
    require(job['cpu_samples_profiled'] is True and start['pidfd_bound'] is True
            and (start['pid'], start['start']) == (owner['pid'], owner['start'])
            and start['signal'] == int(cpu.TOGGLE)
            and start['request_ns'] < start['output_observed_ns']
            <= min(item['released_ns'] for item in job['ready'].values()),
            'exact single pre-release CPU sampler start required')
    require(sorted(p.name for p in directory.glob('service.cpu.prof*')) == ['service.cpu.prof.0'],
            'exactly one CPU session required')
    parsed = cpu.parse_profile(directory / 'service.cpu.prof.0')
    require(parsed == job['cpu_profile'], 'raw CPU profile differs')
    logs = re.findall(r'PROFILE: interrupts/evictions/bytes = (\d+)/(\d+)/(\d+)',
                      (directory / 'server.log').read_text())
    require(len(logs) == 1 and int(logs[0][0]) == parsed['samples'], 'native CPU sample count differs')
    require(not any(key.startswith('CPUPROFILE') for key in job['application_environment']),
            'applications must not receive CPU sampling')
    return parsed


def audit(h, cpu, attempt, job, manifest):
    directory, native = attempt / 'equal', REFERENCE / 'native'
    require(json.loads((directory / 'attempt.json').read_text()) == job, 'raw job differs')
    candidate.normal_cleanup(job)
    argv = job['server_command']
    require(argv[0] == str(h.training.SERVER), 'current Debug service required')
    for flag, value in (('--persistent-workers', '320'), ('--gemm-tile-side', '32'),
                        ('--priority-idle-us', '2000'), ('--kernel-mode', 'preemptive'),
                        ('--scheduler', 'priority')):
        require(argv.count(flag) == 1 and argv[argv.index(flag) + 1] == value,
                'fixed diagnostic geometry/policy required: ' + flag)
    expected_pins = [f"{job['application_identities'][role]['pid']}:"
                     f"{job['application_identities'][role]['start']}:0:{selection}"
                     for role, selection in (('low', 'configured'), ('high', 'original'))]
    require(job['client_pins'] == expected_pins and
            [argv[index + 1] for index, value in enumerate(argv) if value == '--client-priority']
            == expected_pins, 'fixed equal priority and original inference required')
    require(job['mode'] == 'equal' and job['block_observation'] is False
            and job['per_operation_audits'] is False and job['cuda_timeline_profiled'] is False
            and job['wait_calls_profiled'] is True, 'explicit equal CPU/wait-only provenance required')
    for name, value in job['artifact_identities'].items():
        require(digest(directory / name) == value, 'changed raw artifact: ' + name)
    gpu = manifest['gpu'].split(',')[0].strip()
    require(h.training.validate_result(directory / 'training', native / 'training', gpu)
            == job['training_numerical'], 'independent training receipt differs')
    require(h.validate_onnx(directory / 'inference', gpu, h.export, expected_warmup=1,
            expected_iterations=2, reference=native / 'inference') == job['inference_numerical'],
            'independent inference receipt differs')
    import torch
    require(not torch.cuda.is_initialized(), 'math audit must remain CPU-only')
    actual = torch.load(directory / 'training/snapshot.pt', map_location='cpu', weights_only=True)
    original = torch.load(native / 'training/snapshot.pt', map_location='cpu', weights_only=True)
    require(type(actual) is dict and set(actual) == set(original) and len(actual) == 401,
            'complete401-tensor inventory required')
    elements = 0
    for name, value in actual.items():
        torch.testing.assert_close(value, original[name], rtol=2e-4, atol=2e-5, msg=name)
        elements += value.numel()
    require(elements == 217787909 and not torch.cuda.is_initialized(), 'complete CPU math required')
    del actual, original
    profile = check_profile(job, directory, cpu)
    trace_module = importlib.import_module('application_wait_trace')
    owners = dict(job['application_identities'], service=job['server_identity'])
    require(owners == job['wait_probe_owners'], 'exact wait owners required')
    traces, waits = {}, {}
    for role, owner in owners.items():
        trace = trace_module.parse(directory / f"wait-{owner['pid']}.bin", owner['pid'])
        require({k: v for k, v in trace.items() if k != 'rows'} == job['wait_probe_traces'][role],
                'raw wait receipt differs')
        traces[role] = trace
    for source, role in (('training', 'low'), ('inference', 'high')):
        measured = json.loads((directory / source / 'result.json').read_text())
        low, high = measured['measured_begin_ns'], measured['measured_end_ns']
        front = trace_module.summarize(traces[role], low, high)
        service = trace_module.summarize(traces['service'], low, high)
        intervals = [(r['begin_ns'], r['end_ns']) for name in (role, 'service')
                     for r in traces[name]['rows']]
        combined = trace_module.union(intervals, low, high)
        waits[source] = dict(frontend=front, service=service, combined_wait_union_ns=combined,
                             neither_recorded_wait_ns=high-low-combined)
    texts = {}
    for name, flags in (('flat', []), ('cumulative', ['--cum'])):
        proc = subprocess.run(['/usr/bin/google-pprof', '--text', '--nodecount=100', *flags,
                              str(h.training.SERVER), str(directory / 'service.cpu.prof.0')],
                              capture_output=True, text=True, timeout=60, check=True)
        require('Total:' in proc.stdout, 'complete symbolization required')
        texts[name] = dict(stdout=proc.stdout, stderr=proc.stderr)
    return dict(tensors=401, elements=elements, source_bound_complete_inference=True,
                profile=profile, waits=waits, symbolized=texts, **FLAGS)


def worker(attempt, mode):
    report, guard = dict(status='started', **FLAGS), None
    try:
        require(mode == 'equal' and not os.environ.get('LD_PRELOAD') and not os.environ.get('LD_AUDIT'),
                'clean equal-only worker required')
        guard = candidate.ResourceGuard(attempt, mode)
        common, h, cpu = helpers()
        manifest = json.loads((attempt / 'manifest.json').read_text())
        h.startup_reference, h.export = common.application_references(h, manifest)
        require(h.query('gpu', h.training.GPU_FIELDS) == manifest['gpu'], 'GPU identity changed')
        environment = h.training.clean_environment(manifest['gpu'].split(',')[0].strip())
        with resource_hooks(common, guard):
            report['job'] = cpu.waits.ordinary.run_control(h, attempt / mode, mode, environment,
                REFERENCE / 'native', diagnostic=cpu.CpuDiagnostic(h, attempt / mode))
        report['audit'] = audit(h, cpu, attempt, report['job'], manifest)
        guard.check(force=True)
        report['status'] = 'passed_requires_review'
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        if guard is not None:
            report['resources'] = guard.receipt
        save(attempt / (mode + '-worker.json'), report)
    print(json.dumps(dict(status=report['status'], error=report.get('error'))), flush=True)
    return int(report['status'] != 'passed_requires_review')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collect', action='store_true', help='one bounded equal-only capture')
    parser.add_argument('--worker-attempt', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--worker-mode', choices=['equal'], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_attempt:
        require(args.worker_mode == 'equal', 'internal equal mode required')
        return worker(args.worker_attempt.resolve(strict=True), args.worker_mode)
    if not args.collect:
        parser.print_help()
        return 0
    attempt = create_attempt(resolve_artifact_root('bert-current-overhead', category='profiles'))
    print('Overhead evidence: ' + str(attempt), flush=True)
    report, before, paths = dict(status='started', mode='equal', diagnostic_only=True, **FLAGS), None, None
    try:
        require(not os.environ.get('LD_PRELOAD') and not os.environ.get('LD_AUDIT')
                and not any(key.startswith('CPUPROFILE') for key in os.environ), 'clean collector required')
        require(shutil.disk_usage(attempt).free >= 8 * 1024**3, '8GiB free required; no deletion')
        common, h, cpu = helpers()
        from run_bert_device_observation import qualify_checkpoint
        _, selected, checkpoint_paths = qualify_checkpoint(REFERENCE)
        manifest = json.loads((REFERENCE / 'manifest.json').read_text())
        h.startup_reference, h.export = common.application_references(h, manifest)
        require(h.query('gpu', h.training.GPU_FIELDS) == manifest['gpu'], 'checkpoint GPU differs')
        paths = common.input_paths(h) | set(selected) | checkpoint_paths
        paths.update(REFERENCE / 'native' / name for name in
                     ('training/snapshot.pt', 'training/result.json', 'inference/result.json'))
        paths.update(candidate.LEGACY.glob('*.py'))
        paths.update(candidate.LEGACY / name for name in
                     ('application_wait_probe.cc', 'application_wait_probe.inc', 'application_wait_fixture.cc'))
        paths.update((Path(__file__), Path(candidate.__file__), PROJECT_ROOT / 'scripts/artifact_paths.py',
                      PROJECT_ROOT / 'tests/examples/bert_overhead_diagnostic_test.py',
                      PROJECT_ROOT / 'scripts/run_rnn_v8_validation.py',
                      PROJECT_ROOT / 'scripts/run_bert_device_observation.py',
                      candidate.BUILD / 'CMakeCache.txt', candidate.BUILD / 'compile_commands.json',
                      cpu.LIBRARY, cpu.waits.PROBE, Path('/usr/bin/google-pprof'), Path(sys.executable).resolve()))
        paths.add(Path('/lib64/ld-linux-x86-64.so.2'))
        paths.update(Path('/usr/lib/x86_64-linux-gnu') / name for name in
                     ('libunwind.so.8', 'liblzma.so.5', 'libc.so.6', 'libstdc++.so.6', 'libgcc_s.so.1', 'libm.so.6'))
        before = h.training.manifest(paths)
        save(attempt / 'inputs-before.json', before)
        save(attempt / 'manifest.json', manifest)
        report['idle_window'] = wait_for_idle(attempt)
        require(report['idle_window']['achieved'], 'sampled idle window unavailable; no capture')
        require(before == h.training.manifest(paths), 'inputs changed before unit')
        unit, receipt = candidate.launch_unit(attempt, 'equal', worker_script=Path(__file__))
        report.update(unit=unit, worker=receipt, status='current_debug_cpu_wait_diagnostic_requires_review')
        h.training.require_idle()
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        if before is not None:
            try:
                after = h.training.manifest(paths)
                save(attempt / 'inputs-after.json', after)
                report['input_identity_unchanged'] = before == after
                require(before == after and h.query('gpu', h.training.GPU_FIELDS) == manifest['gpu'],
                        'source/tool/input/GPU changed during capture')
            except BaseException as error:
                report.update(status='failed', identity_error=str(error))
        save(attempt / 'report.json', report)
    print(json.dumps(dict(status=report['status'], error=report.get('error'))), flush=True)
    return int(report['status'] == 'failed')


if __name__ == '__main__':
    raise SystemExit(main())
