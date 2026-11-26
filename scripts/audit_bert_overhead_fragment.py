#!/usr/bin/env python3
"""Offline partial CPU/service-wait audit; never repair a failed capture receipt."""

import argparse
import importlib
import json
from pathlib import Path
import re
import subprocess

from artifact_paths import create_attempt, resolve_artifact_root
import run_bert_overhead_diagnostic as capture
from run_rnn_v8_validation import digest, save

require = capture.require


def read(path):
    return json.loads(path.read_text())


def validate_failed_capture(source):
    report = read(source / 'report.json')
    before = read(source / 'inputs-before.json')
    require(report['status'] == 'failed' and report['input_identity_unchanged'] is True
            and before == read(source / 'inputs-after.json'), 'unchanged failed capture required')
    require(read(source / 'equal-unit.json')['unit_removed'] is True, 'capture unit must be removed')
    worker = read(source / 'equal-worker.json')
    job = read(source / 'equal/attempt.json')
    require(worker['job'] == job and job['status'] == 'failed'
            and job.get('cleanup_errors') == ['timeline clock: lost wait records or probe error']
            and not job.get('error') and not job.get('output_identity_error'),
            'only the explicit wait-loss failure may receive a separate component audit')
    require(job['reference_before'] == job['reference_after'] and job['socket_retained'] is False,
            'original reference/socket lifecycle failed')
    for role in ('low', 'high', 'service'):
        owner = job['server_identity'] if role == 'service' else job['application_identities'][role]
        cleanup = job['authenticated_child_cleanup'][role]
        require(type(job['final_returncodes'][role]) is int and job['final_returncodes'][role] == 0
                and cleanup['exit_confirmed'] is True and cleanup['signals'] == []
                and (cleanup['pid'], cleanup['start']) == (owner['pid'], owner['start']),
                'exact normal application/service exits required')
    for name, value in job['artifact_identities'].items():
        require(digest(source / 'equal' / name) == value, 'raw capture bytes changed: ' + name)
    return job, before


def worker(attempt):
    report, guard = dict(status='started', partial_diagnostic=True, **capture.FLAGS), None
    try:
        guard = capture.candidate.ResourceGuard(attempt, 'equal')
        source = Path(read(attempt / 'manifest.json')['source']).resolve(strict=True)
        job, before = validate_failed_capture(source)
        common, h, cpu = capture.helpers()
        directory = source / 'equal'
        metadata = read(source / 'manifest.json')
        h.startup_reference, h.export = common.application_references(h, metadata)
        native = capture.REFERENCE / 'native'
        selected = {h.training.SERVER, h.training.PRELOAD, cpu.LIBRARY, cpu.waits.PROBE,
                    Path('/usr/bin/google-pprof'), Path('/lib64/ld-linux-x86-64.so.2'),
                    native / 'training/snapshot.pt', native / 'training/result.json',
                    native / 'inference/result.json', Path(capture.__file__)}
        selected.update(capture.candidate.LEGACY.glob('*.py'))
        selected.update(Path('/usr/lib/x86_64-linux-gnu') / name for name in
                        ('libunwind.so.8', 'liblzma.so.5', 'libc.so.6', 'libstdc++.so.6', 'libgcc_s.so.1', 'libm.so.6'))
        seals = h.training.manifest(selected)
        require(seals == {str(p): before[str(p)] for p in selected}, 'analysis inputs differ from capture')
        report['analysis_inputs'] = seals
        gpu = metadata['gpu'].split(',')[0].strip()
        require(h.training.validate_result(directory / 'training', native / 'training', gpu)
                == job['training_numerical'], 'training receipt differs')
        require(h.validate_onnx(directory / 'inference', gpu, h.export, expected_warmup=1,
                expected_iterations=2, reference=native / 'inference') == job['inference_numerical'],
                'inference receipt differs')
        import torch
        torch.set_num_threads(2)
        require(not torch.cuda.is_initialized(), 'CPU-only math required')
        actual = torch.load(directory / 'training/snapshot.pt', map_location='cpu', weights_only=True)
        original = torch.load(native / 'training/snapshot.pt', map_location='cpu', weights_only=True)
        require(type(actual) is dict and set(actual) == set(original) and len(actual) == 401,
                'complete401 tensors required')
        elements = 0
        for name, value in actual.items():
            torch.testing.assert_close(value, original[name], rtol=2e-4, atol=2e-5, msg=name)
            elements += value.numel()
            guard.check()
        require(elements == 217787909 and not torch.cuda.is_initialized(), 'full CPU math required')
        report['math'] = dict(tensors=401, elements=elements, source_bound_complete_inference=True)
        del actual, original
        owner, start = job['server_identity'], job['cpu_profile_start']
        require(job['cpu_samples_profiled'] is True and start['pidfd_bound'] is True
                and (start['pid'], start['start']) == (owner['pid'], owner['start'])
                and start['signal'] == int(cpu.TOGGLE)
                and start['request_ns'] < start['output_observed_ns']
                <= min(r['released_ns'] for r in job['ready'].values()), 'exact pre-release sampler required')
        require(start['mapped_inputs'] == h.training.manifest({cpu.LIBRARY, cpu.waits.PROBE, h.training.SERVER}),
                'actual sampler mapping identities differ')
        require(sorted(p.name for p in directory.glob('service.cpu.prof*')) == ['service.cpu.prof.0'],
                'one sampler session required')
        profile = cpu.parse_profile(directory / 'service.cpu.prof.0')
        logs = re.findall(r'PROFILE: interrupts/evictions/bytes = (\d+)/(\d+)/(\d+)',
                          (directory / 'server.log').read_text())
        require(len(logs) == 1 and int(logs[0][0]) == profile['samples'], 'native CPU count mismatch')
        clean = {k: v for k, v in job['server_environment'].items()
                 if k != 'LD_PRELOAD' and not k.startswith('CPUPROFILE')}
        expected = dict(clean, LD_PRELOAD=str(cpu.LIBRARY) + ':' + str(cpu.waits.PROBE),
                        CPUPROFILE=str(directory / 'service.cpu.prof'), CPUPROFILESIGNAL=str(int(cpu.TOGGLE)),
                        CPUPROFILE_FREQUENCY=str(cpu.FREQUENCY))
        require(job['server_environment'] == expected and
                not any(k.startswith('CPUPROFILE') for k in job['application_environment']), 'sampler recipe differs')
        report['cpu_profile'] = profile
        trace_module = importlib.import_module('application_wait_trace')
        trace = trace_module.parse(directory / f"wait-{owner['pid']}.bin", owner['pid'])
        report['service_waits'] = {}
        for name in ('training', 'inference'):
            measured = read(directory / name / 'result.json')
            report['service_waits'][name] = trace_module.summarize(
                trace, measured['measured_begin_ns'], measured['measured_end_ns'])
        report['frontend_waits_excluded'] = True
        report['symbolized'] = {}
        for name, flags in (('flat', []), ('cumulative', ['--cum']),
                            ('snapshot_lines', ['--lines', '--focus=CudaExecutionSession::snapshot'])):
            result = subprocess.run(['/usr/bin/google-pprof', '--text', *flags, str(h.training.SERVER),
                                     str(directory / 'service.cpu.prof.0')],
                                    capture_output=True, text=True, timeout=120, check=True)
            require('Total:' in result.stdout, 'empty symbolization')
            report['symbolized'][name] = dict(stdout=result.stdout, stderr=result.stderr)
            guard.check(force=True)
        require(h.training.manifest(selected) == seals, 'analysis input changed')
        require(read(source / 'report.json')['status'] == 'failed', 'original failure must remain')
        report.update(status='passed_requires_review', source=str(source), capture_status='failed',
                      joint_wait_attribution_qualified=False)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        if guard is not None:
            report['resources'] = guard.receipt
        save(attempt / 'equal-worker.json', report)
    print(json.dumps(dict(status=report['status'], error=report.get('error'))), flush=True)
    return int(report['status'] != 'passed_requires_review')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, nargs='?')
    parser.add_argument('--worker-attempt', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--worker-mode', choices=['equal'], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_attempt:
        require(args.worker_mode == 'equal', 'exact offline worker required')
        return worker(args.worker_attempt.resolve(strict=True))
    require(args.source is not None, 'explicit failed capture required')
    source = args.source.resolve(strict=True)
    validate_failed_capture(source)
    attempt = create_attempt(resolve_artifact_root('bert-overhead-offline-audit', category='profiles'))
    save(attempt / 'manifest.json', dict(source=str(source), audit_script_sha256=digest(Path(__file__))))
    print('Offline partial audit: ' + str(attempt), flush=True)
    report = dict(status='failed', capture_status='failed', **capture.FLAGS)
    try:
        unit, receipt = capture.candidate.launch_unit(attempt, 'equal', worker_script=Path(__file__))
        report.update(status='partial_cpu_service_wait_and_math_reparsed', unit=unit, worker=receipt)
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
    finally:
        save(attempt / 'report.json', report)
    print(json.dumps(dict(status=report['status'], error=report.get('error'))), flush=True)
    return int(report['status'] == 'failed')


if __name__ == '__main__':
    raise SystemExit(main())
