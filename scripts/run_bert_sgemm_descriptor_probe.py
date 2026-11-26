#!/usr/bin/env python3
"""Collect one bounded fixed-BERT SGEMM descriptor, never SM or timing proof."""

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

from artifact_paths import PROJECT_ROOT, create_attempt, resolve_artifact_root
from bert_sgemm_coverage_candidates import (authenticated_training_sessions, targeted_descriptor)
from execution_semantic_audit import parse_semantic_audit
import run_bert_candidate_validation as candidate
from run_rnn_v8_validation import digest, save, wait_for_idle


FLAGS = dict(full_device_qualified=False, physical_priority_qualified=False,
             saturation_qualified=False, performance_qualified=False,
             full_goal_complete=False, ordinary_timing_evidence=False)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parse_target(value):
    require(type(value) is str and re.fullmatch(r'[1-9][0-9]*:[1-9][0-9]*', value),
            'target must be a positive SESSION:SEQUENCE')
    session, sequence = map(int, value.split(':'))
    require(session <= 0xffffffffffffffff and sequence <= 0xffffffffffffffff,
            'target identity exceeds uint64')
    return session, sequence


def validate_worker_target(receipt, target):
    session, sequence = parse_target(target)
    descriptor = receipt['descriptor']
    require(receipt['status'] == 'passed_requires_review' and
            type(descriptor['session']) is int and type(descriptor['sequence']) is int and
            (descriptor['session'], descriptor['sequence']) == (session, sequence),
            'worker descriptor differs from the original target')


def ncu_environment(environment, profiler_environment):
    """Preserve only NCU's observed NVTX injection variable for profiled children.

    Callers must independently authenticate the NCU parent and tool identity.
    This helper alone is not evidence that any CUDA kernel was captured.
    """
    require('NVTX_INJECTION64_PATH' not in environment, 'clean base environment required')
    value = profiler_environment.get('NVTX_INJECTION64_PATH')
    require(value == 'libcuda-injection.so', 'expected NCU NVTX injection is absent or changed')
    return dict(environment, NVTX_INJECTION64_PATH=value)


@contextmanager
def selected_service_target(common, target, range_library=None, *, client_bound=False):
    """Temporarily decorate only the existing owned service command."""
    session, sequence = parse_target(target)
    original = common.service_command

    def selected(*args, **kwargs):
        command, pins = original(*args, **kwargs)
        require(command.count('--audit-native') == 1 and '--audit-sgemm-target' not in command and
                '--audit-sgemm-target-client' not in command,
                'one ordinary native audit and no prior target required')
        if client_bound:
            configured = [command[index + 1] for index, option in enumerate(command[:-1])
                          if option == '--client-priority' and command[index + 1].endswith(':0:configured')]
            require(len(configured) == 1 and
                    re.fullmatch(r'[1-9][0-9]*:[1-9][0-9]*:0:configured', configured[0]) is not None,
                    'one authenticated equal-priority configured training client required')
            pid, start = configured[0].split(':')[:2]
            result = [*command, '--audit-sgemm-target-client', f'{pid}:{start}:{sequence}']
        else:
            result = [*command, '--audit-sgemm-target', f'{session}:{sequence}']
        if range_library is not None:
            require(isinstance(range_library, Path) and range_library.is_absolute() and
                    range_library.is_file() and '--audit-sgemm-range-library' not in command,
                    'one existing absolute range provider required')
            require(command.count('--socket') == 1 and command.index('--socket') + 1 < len(command),
                    'one service socket required for profiler capture')
            socket = command[command.index('--socket') + 1]
            require(type(socket) is str and socket.startswith('/') and '\0' not in socket and
                    len(os.fsencode(socket)) < 108, 'profile service socket path exceeds unix limit')
            result += ['--audit-sgemm-range-library', str(range_library)]
        return result, pins

    common.service_command = selected
    try:
        yield
    finally:
        common.service_command = original


def worker(attempt, mode):
    report, guard = dict(status='started', mode=mode, **FLAGS), None
    try:
        require(mode == 'equal' and not os.environ.get('LD_PRELOAD') and not os.environ.get('LD_AUDIT'),
                'clean equal-only worker required')
        guard = candidate.ResourceGuard(attempt, mode)
        manifest = json.loads((attempt / 'manifest.json').read_text())
        session, sequence = parse_target(manifest['sgemm_target'])
        common, h = candidate.load_helpers()
        h.startup_reference, h.export = common.application_references(h, manifest)
        gpu = manifest['gpu'].split(',')[0].strip()
        require(h.query('gpu', h.training.GPU_FIELDS) == manifest['gpu'], 'GPU identity changed')
        environment = h.training.clean_environment(gpu)
        native = candidate.EXISTING_NATIVE / 'native'
        with selected_service_target(common, manifest['sgemm_target']):
            job = candidate.guarded_control(common, h, attempt, mode, environment, guard, native)
        report['job'] = job
        report['reparse'] = candidate.reparse(common, h, attempt, mode, job, gpu, native)
        argv = job['server_command']
        require(argv.count('--audit-sgemm-target') == 1 and
                argv[argv.index('--audit-sgemm-target') + 1] == manifest['sgemm_target'],
                'exact diagnostic target must belong to the authenticated service command')
        log = attempt / mode / 'server.log'
        require(log.stat().st_size <= 64 * 1024 * 1024, 'service audit exceeds parser capacity')
        data = log.read_bytes()
        require(len(data) <= 64 * 1024 * 1024 and
                hashlib.sha256(data).hexdigest() == job['artifact_identities']['server.log'],
                'bounded original service audit identity required')
        content = data.decode('utf-8')
        semantic = parse_semantic_audit(content, require_complete=True)
        require(session in authenticated_training_sessions(semantic, job),
                'selected command must belong to the authenticated training process')
        report['descriptor'] = targeted_descriptor(content, session, sequence)
        report['descriptor']['application_identity_verified'] = True
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
    parser.add_argument('--collect', action='store_true')
    parser.add_argument('--target', default='2:3127', help='one run-local SESSION:SEQUENCE candidate')
    parser.add_argument('--wait-for-idle', action='store_true', help='wait at most 10 minutes for sampled idle')
    parser.add_argument('--worker-attempt', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--worker-mode', choices=['equal'], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_attempt:
        return worker(args.worker_attempt.resolve(strict=True), args.worker_mode)
    if not args.collect:
        parser.print_help()
        return 0
    parse_target(args.target)
    attempt = create_attempt(resolve_artifact_root('bert-sgemm-descriptor', category='profiles'))
    print('SGEMM descriptor evidence: ' + str(attempt), flush=True)
    report, before, paths = dict(status='started', diagnostic_only=True, target=args.target, **FLAGS), None, None
    try:
        require(not os.environ.get('LD_PRELOAD') and not os.environ.get('LD_AUDIT'), 'clean collector required')
        require(shutil.disk_usage(attempt).free >= 8 * 1024**3, '8 GiB free required; no deletion')
        if args.wait_for_idle:
            report['idle_window'] = wait_for_idle(attempt)
            require(report['idle_window']['achieved'], 'sampled idle window unavailable; no diagnostic started')
        common, h = candidate.load_helpers()
        gpu = h.query('gpu', h.training.GPU_FIELDS)
        fields = [value.strip() for value in gpu.split(',')]
        require(len(fields) == 4 and '\n' not in gpu and fields[0].startswith('GPU-') and 'V100' in fields[1],
                'one identified V100 required')
        require(fields[0].removeprefix('GPU-') == h.training.read_json(
            h.training.ORIGINAL / 'result.json')['environment']['device_uuid'], 'original GPU identity required')
        h.training.require_idle()
        manifest = dict(gpu=gpu, startup_reference_bundle=str(common.STARTUP_INPUTS),
                        startup_reference_report_sha256=common.STARTUP_INPUTS_SHA, sgemm_target=args.target)
        h.startup_reference, h.export = common.application_references(h, manifest)
        native_paths, native_job = candidate.existing_native_reference(
            candidate.EXISTING_NATIVE, gpu, common.STARTUP_INPUTS, common.STARTUP_INPUTS_SHA)
        inference = native_job['inference_numerical']
        require(digest(h.export / 'export.json') == inference['export_sha256'] and
                digest(h.export / 'bert.onnx') == inference['model_sha256'], 'native export identity differs')
        example = PROJECT_ROOT / 'examples/framework_workloads/bert'
        sources = {'bert_contract.py', 'bert_workload.py', 'onnx_bert_workload.py', 'export_bert.py'}
        require(set(inference['source_sha256']) == sources and
                all(digest(example / name) == value for name, value in inference['source_sha256'].items()),
                'native workload source differs')
        paths = common.input_paths(h) | native_paths | {Path(__file__), Path(candidate.__file__),
            PROJECT_ROOT / 'scripts/bert_sgemm_coverage_candidates.py',
            PROJECT_ROOT / 'scripts/artifact_paths.py', PROJECT_ROOT / 'scripts/run_rnn_v8_validation.py',
            PROJECT_ROOT / 'scripts/rnn_v8_audit.py', PROJECT_ROOT / 'scripts/cudnn8_legacy_audit.py',
            attempt / 'manifest.json',
            candidate.BUILD / 'CMakeCache.txt', candidate.BUILD / 'compile_commands.json'}
        save(attempt / 'manifest.json', manifest)
        before = h.training.manifest(paths)
        save(attempt / 'inputs-before.json', before)
        h.training.require_idle()
        require(before == h.training.manifest(paths), 'frozen input changed before diagnostic')
        unit, receipt = candidate.launch_unit(attempt, 'equal', worker_script=Path(__file__))
        validate_worker_target(receipt, args.target)
        report.update(unit=unit, worker=receipt, status='descriptor_requires_review')
        h.training.require_idle()
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        if before is not None:
            try:
                after = h.training.manifest(paths)
                save(attempt / 'inputs-after.json', after)
                report['input_identity_unchanged'] = before == after
                require(before == after and h.query('gpu', h.training.GPU_FIELDS) == gpu,
                        'source/tool/input/GPU changed during diagnostic')
            except BaseException as error:
                report.update(status='failed', identity_error=str(error))
        save(attempt / 'report.json', report)
    print(json.dumps(dict(status=report['status'], error=report.get('error'), report=str(attempt / 'report.json'))))
    return int(report['status'] == 'failed')


if __name__ == '__main__':
    raise SystemExit(main())
