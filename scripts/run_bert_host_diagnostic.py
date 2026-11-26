#!/usr/bin/env python3
"""Disabled host collector: profiler stop authentication did not qualify."""

import argparse
import copy
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from artifact_paths import PROJECT_ROOT, create_attempt, resolve_artifact_root
import bert_device_capture as capture
from bert_host_probe import HostWindow, host_hooks
import run_bert_candidate_validation as candidate
import run_bert_device_observation as recipe
from run_rnn_v8_validation import digest, save, wait_for_idle

require = recipe.require
FLAGS = dict(capture.FLAGS, diagnostic_only=True, counter_collection=False,
             cpu_profiled=True, gpu_gap_causality_qualified=False)
CAPTURE_PROTOCOL_READY = False


def require_capture_ready():
    require(CAPTURE_PROTOCOL_READY, 'host collector disabled: profiler stop authentication is unqualified')


def authorization_budget(authorization, remaining_s, now=None):
    deadline = datetime.fromisoformat(authorization['authorization_deadline_utc'].replace('Z', '+00:00'))
    require(deadline.tzinfo is not None, 'absolute authorization deadline required')
    now = datetime.now(timezone.utc) if now is None else now
    require((deadline - now).total_seconds() >= remaining_s, 'insufficient remaining diagnostic authorization')


def symbolize(directory, cpu, server, window):
    profile = directory / 'service.cpu.prof.0'
    parsed = cpu.parse_profile(profile)
    require({k: v for k, v in parsed.items() if k != 'scope'} ==
            {k: v for k, v in window['profile'].items() if k != 'scope'}, 'profile changed after window stop')
    counts = re.findall(r'PROFILE: interrupts/evictions/bytes = (\d+)/(\d+)/(\d+)',
                        (directory / 'server.log').read_text())
    require(len(counts) == 1 and int(counts[0][0]) == parsed['samples'], 'raw CPU sample counts differ')
    outputs = {}
    for name, options in (('flat', []), ('cumulative', ['--cum'])):
        argv = ['/usr/bin/google-pprof', '--text', '--nodecount=100', *options, str(server), str(profile)]
        result = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, timeout=60, check=False)
        require(result.returncode == 0 and len(result.stdout) <= 2 * 1024 ** 2
                and len(result.stderr) <= 2 * 1024 ** 2, 'bounded symbolization failed')
        path = directory.parent / (directory.name + '-cpu-' + name + '.txt')
        path.write_text(result.stdout)
        outputs[name] = dict(command=argv, path=str(path), sha256=digest(path), stderr=result.stderr)
    return dict(profile_sha256=digest(profile), server_sha256=digest(server), parsed=parsed,
                server_counts=counts, outputs=outputs)


def worker(attempt, mode):
    require_capture_ready()
    report, guard, window = dict(status='failed', mode=mode, **FLAGS), None, None
    report['cpu_profiled'] = mode == 'equal'
    try:
        require(not os.environ.get('LD_PRELOAD') and not os.environ.get('LD_AUDIT'), 'clean controller required')
        authorization_budget(recipe.read(attempt / 'authorization.json'), 800)
        common, h, sustained = capture.helpers()
        cpu = importlib.import_module('collect_application_cpu')
        guard = candidate.ResourceGuard(attempt, mode)
        plan = recipe.read(attempt / 'plan.json')
        require(plan.get('native_equal') is True and plan['recipe'] == recipe.expected_recipe(plan),
                'unchanged matched native/equal recipe required')
        manifest = recipe.read(Path(plan['reference']) / 'manifest.json')
        h.startup_reference, h.export = common.application_references(h, manifest)
        require(h.query('gpu', h.training.GPU_FIELDS) == plan['checkpoint_gpu'], 'GPU identity differs')
        resources = sustained.load_resources()
        report['resources_before'] = resources.resource_snapshot()
        require(report['resources_before']['available_device_sms'] == 80, 'full physical device unavailable')
        directory = attempt / mode
        window = HostWindow(directory, cpu, guard)
        clean = h.training.clean_environment(plan['gpu_uuid'])
        measurement = sustained.load_measurement()
        with host_hooks(common, h, sustained, guard, mode, clean, window):
            job = sustained.run_control(h, measurement, directory, mode, clean, plan['point'])
        window.close()
        capture.audit_application(sustained, measurement, h, directory, job, plan['point'])
        raw = directory / 'attempt.json'
        raw_hash = digest(raw)
        raw.rename(directory / 'helper-attempt.json')
        report['job'] = copy.deepcopy(job)
        report['job'].update(FLAGS, cpu_profiled=mode == 'equal',
            status='current_candidate_host_control_requires_review',
            raw_helper_receipt=dict(path='helper-attempt.json', sha256=raw_hash))
        save(directory / 'attempt.json', report['job'])
        if mode == 'equal':
            report['symbolization'] = symbolize(directory, cpu, h.training.SERVER, window.receipt)
        report['resources_after'] = resources.resource_snapshot()
        require(report['resources_after'] == report['resources_before'], 'physical resource policy changed')
        guard.check(force=True)
        report['status'] = 'passed_requires_review'
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        if window is not None:
            try:
                window.close()
            except BaseException as error:
                report.update(status='failed', observer_error=str(error))
            report['host_window'] = window.receipt
        if guard is not None:
            report['resources'] = guard.receipt
        save(attempt / (mode + '-worker.json'), report)
    return int(report['status'] != 'passed_requires_review')


def collect(plan_path, authorization_path):
    require_capture_ready()
    plan, authorization = recipe.read(plan_path), recipe.read(authorization_path)
    authorization_budget(authorization, 2300)
    require(plan.get('native_equal') is True, 'matched native/equal plan required')
    recipe.validate_plan(plan)
    attempt = create_attempt(resolve_artifact_root('bert-current-host', category='profiles'))
    save(attempt / 'plan.json', plan)
    save(attempt / 'authorization.json', authorization)
    print('Host diagnostic evidence: ' + str(attempt), flush=True)
    report = dict(status='started', jobs=[], plan_source=str(plan_path), **FLAGS)
    paths = {Path(__file__).resolve(), PROJECT_ROOT / 'scripts/bert_host_probe.py',
             Path('/usr/bin/google-pprof'), Path('/usr/lib/x86_64-linux-gnu/libprofiler.so.0').resolve(strict=True),
             Path('/usr/lib/x86_64-linux-gnu/libunwind.so.8').resolve(strict=True)}
    capture.helpers()
    importlib.import_module('collect_application_cpu')
    paths.update(Path(module.__file__).resolve(strict=True) for module in tuple(sys.modules.values())
                 if getattr(module, '__file__', None) and Path(module.__file__).is_relative_to(candidate.LEGACY))
    before = {str(p): digest(p) for p in paths}
    save(attempt / 'host-inputs-before.json', before)
    try:
        require(not os.environ.get('LD_PRELOAD') and not os.environ.get('LD_AUDIT'), 'clean outer collector required')
        require(shutil.disk_usage(attempt).free >= 8 * 1024 ** 3, '8GiB free required; no deletion')
        report['idle_window'] = wait_for_idle(attempt)
        require(report['idle_window']['achieved'], 'idle window unavailable; no capture')
        for mode in ('native', 'equal'):
            authorization_budget(authorization, 840)
            recipe.validate_plan(plan)
            require(before == {str(p): digest(p) for p in paths}, 'host diagnostic inputs changed')
            try:
                unit, result = candidate.launch_unit(attempt, mode, worker_script=Path(__file__))
                report['jobs'].append(dict(mode=mode, unit=unit, worker=result))
            finally:
                post = dict(status='failed')
                try:
                    recipe.validate_plan(plan)
                    require(before == {str(p): digest(p) for p in paths}, 'host diagnostic inputs changed')
                    post['status'] = 'complete_input_seals_unchanged'
                finally:
                    save(attempt / (mode + '-postflight.json'), post)
            save(attempt / 'report.json', report)
        report['status'] = 'current_candidate_host_controls_require_independent_review'
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        save(attempt / 'report.json', report)
    print(json.dumps(dict(status=report['status'], error=report.get('error'))), flush=True)
    return int(report['status'] == 'failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--collect', type=Path)
    group.add_argument('--worker-attempt', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--authorization', type=Path)
    parser.add_argument('--worker-mode', choices=('native', 'equal'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.collect:
        require(args.authorization is not None, 'explicit diagnostic authorization required')
        return collect(args.collect.resolve(strict=True), args.authorization.resolve(strict=True))
    if args.worker_attempt:
        require(args.worker_mode is not None, 'explicit internal mode required')
        return worker(args.worker_attempt.resolve(strict=True), args.worker_mode)
    parser.print_help()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
