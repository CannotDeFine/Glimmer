#!/usr/bin/env python3
"""Opt-in CPU-only real-unit check; never registered in default CTest.

Requires prior authorization for temporary user-systemd resource limits.
The probe has its own source seal and does not modify the frozen capture plan.
"""

import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import bert_device_capture as capture

candidate, recipe, observer = capture.candidate, capture.recipe, capture.observer
save, digest, require = capture.save, capture.digest, capture.require
SCRIPT = Path(__file__).resolve()


def worker(attempt, mode):
    report = dict(status='started', cpu_probe_only=True, counter_collection=False,
                  gpu_work_launched=False, native_sdk_initialized=False, **capture.FLAGS)
    guard = None
    try:
        require(mode == 'native', 'one native-named CPU probe only')
        manifest = recipe.read(attempt / 'probe.json')
        require(manifest['script'] == str(SCRIPT) and manifest['script_sha256'] == digest(SCRIPT),
                'exact independently sealed probe source required')
        report['command'] = list(sys.argv)
        report['script_sha256'] = digest(SCRIPT)
        guard = candidate.ResourceGuard(attempt, mode)
        report['binding'] = capture.worker_binding(attempt, mode, recipe.read(attempt / 'plan.json'))
        report['budget'] = capture.startup_budget(attempt, mode)
        root = Path('/sys/fs/cgroup') / guard.group.lstrip('/')
        report['tasks'] = {name: (root / name).read_text().strip()
                           for name in ('pids.max', 'pids.current')}
        require(report['tasks']['pids.max'] == '256', 'actual TasksMax256 required')
        report['file_limit'] = list(resource.getrlimit(resource.RLIMIT_FSIZE))
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


def probe(plan_path):
    attempt = capture.create_attempt(capture.resolve_artifact_root('bert-unit-probe', category='maintenance'))
    print('CPU unit evidence: ' + str(attempt), flush=True)
    report = dict(status='started', cpu_probe_only=True, counter_collection=False,
                  gpu_work_launched=False, native_sdk_initialized=False, **capture.FLAGS)
    plan = recipe.read(plan_path)
    save(attempt / 'plan.json', plan)
    manifest = dict(script=str(SCRIPT), script_sha256=digest(SCRIPT), command=list(sys.argv),
                    plan_source=str(plan_path), plan_sha256=digest(plan_path))
    save(attempt / 'probe.json', manifest)
    try:
        recipe.validate_plan(plan)
        report['preflight'] = 'complete_input_seals_unchanged'
        started = time.monotonic_ns()
        save(attempt / 'native-launch.json', dict(mode='native', attempt=str(attempt),
            full_preflight_status='complete_input_seals_unchanged', plan_sha256=digest(attempt / 'plan.json'),
            preflight_verified_ns=started, started_ns=started, outer_deadline_ns=started + 820_000_000_000,
            collector=observer.process_record(os.getpid())))
        report['unit'], report['worker'] = candidate.launch_unit(attempt, 'native', worker_script=SCRIPT)
        report['status'] = 'cpu_unit_passed_requires_review'
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        try:
            recipe.validate_plan(plan)
            require(digest(SCRIPT) == manifest['script_sha256'], 'probe source changed')
            report['postflight'] = 'complete_input_and_probe_seals_unchanged'
        except BaseException as error:
            report.update(status='failed', postflight_error=f'{type(error).__name__}: {error}')
        save(attempt / 'report.json', report)
    print(json.dumps(dict(status=report['status'], error=report.get('error'),
                          postflight_error=report.get('postflight_error'))), flush=True)
    return int(report['status'] == 'failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, help='explicit frozen current-device plan')
    parser.add_argument('--worker-attempt', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--worker-mode', choices=('native',), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_attempt:
        return worker(args.worker_attempt.resolve(strict=True), args.worker_mode)
    if args.plan is None:
        parser.error('--plan is required; no default host operation')
    return probe(args.plan.resolve(strict=True))


if __name__ == '__main__':
    raise SystemExit(main())
