#!/usr/bin/env python3
"""Disabled experimental CUDA/OSRT collector for unchanged BERT arrivals.

No runtime rebuild, counters, root privileges or ordinary timing qualification.
Collection requires qualification of controller annotations and owner range termination.
"""

import argparse
from contextlib import contextmanager
import copy
import ctypes
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

from artifact_paths import PROJECT_ROOT, create_attempt, resolve_artifact_root
import bert_device_capture as capture
import run_bert_candidate_validation as candidate
import run_bert_device_observation as recipe
from run_rnn_v8_validation import digest, save, wait_for_idle

NSYS = Path('/opt/nvidia/nsight-systems/2024.5.1/target-linux-x64/nsys')
RANGE = 'GLIMMER_TIMELINE'
FLAGS = dict(capture.FLAGS, counter_collection=False, cuda_timeline_profiled=True,
             diagnostic_only=True)
require = recipe.require
CAPTURE_PROTOCOL_READY = False


def require_capture_ready():
    require(CAPTURE_PROTOCOL_READY,
            'experimental collector disabled: controller environment and '
            'multiprocess range termination are unqualified')


def helpers():
    common, h, sustained = capture.helpers()
    timeline = importlib.import_module('collect_ordinary_timeline')
    require(timeline.ordinary.common is common and timeline.NSYS == NSYS,
            'one current helper and installed profiler binding required')
    return common, h, sustained, timeline


class NvtxAttributes(ctypes.Structure):
    """NVTX v2 event layout from the installed public nvToolsExt.h ABI."""
    _fields_ = [('version', ctypes.c_uint16), ('size', ctypes.c_uint16),
                ('category', ctypes.c_uint32), ('color_type', ctypes.c_int32),
                ('color', ctypes.c_uint32), ('payload_type', ctypes.c_int32),
                ('reserved', ctypes.c_int32), ('payload', ctypes.c_uint64),
                ('message_type', ctypes.c_int32), ('message', ctypes.c_void_p)]


class CaptureRange:
    def __init__(self, timeline):
        self.clock = timeline.ClockBrackets()
        lib = self.clock.library
        lib.nvtxDomainRegisterStringA.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.nvtxDomainRegisterStringA.restype = ctypes.c_void_p
        lib.nvtxDomainRangePushEx.argtypes = [ctypes.c_void_p, ctypes.POINTER(NvtxAttributes)]
        lib.nvtxDomainRangePushEx.restype = ctypes.c_int
        lib.nvtxDomainRangePop.argtypes = [ctypes.c_void_p]
        lib.nvtxDomainRangePop.restype = ctypes.c_int
        require(ctypes.sizeof(NvtxAttributes) == 48, 'installed64-bit NVTX layout required')
        self.attributes = NvtxAttributes(version=2, size=48, message_type=3,
            message=lib.nvtxDomainRegisterStringA(None, RANGE.encode()))
        require(self.attributes.message, 'registered NVTX capture string required')
        self.thread, self.error = None, None
        self.stop = threading.Event()
        self.receipt = dict(controller_pid=os.getpid(), range_name=RANGE)

    def start(self, schedule, authenticate):
        require(self.thread is None, 'one capture range per control required')
        begin = schedule['begin_ns']
        self.receipt.update(offered_begin_ns=begin, analysis_begin_ns=begin + 5_000_000_000,
                            analysis_end_ns=begin + 10_000_000_000)

        def run():
            pushed = False
            try:
                self.receipt['owners'] = authenticate()
                self.receipt['range_thread_id'] = threading.get_native_id()
                require(not self.stop.wait(max(0, (begin + 4_000_000_000 - time.monotonic_ns()) / 1e9)),
                        'capture cancelled before range')
                self.receipt['push_low_ns'] = time.monotonic_ns()
                # The installed injection records complete ranges while returning
                # -1 from these annotation APIs. Preserve the raw result; only
                # the exported unique range and clock brackets establish capture.
                self.receipt['push_return'] = self.clock.library.nvtxDomainRangePushEx(
                    None, ctypes.byref(self.attributes))
                pushed = True
                self.receipt['push_high_ns'] = time.monotonic_ns()
                require(self.receipt['push_high_ns'] < begin + 5_000_000_000,
                        'capture activation missed registered analysis window')
                while True:
                    self.clock.mark('capture_tick')
                    remaining = (begin + 11_000_000_000 - time.monotonic_ns()) / 1e9
                    if remaining <= 0:
                        break
                    require(not self.stop.wait(min(.02, remaining)), 'capture cancelled inside range')
            except BaseException as error:
                self.error = error
            finally:
                if pushed:
                    self.receipt['pop_low_ns'] = time.monotonic_ns()
                    self.receipt['pop_return'] = self.clock.library.nvtxDomainRangePop(None)
                    self.receipt['pop_high_ns'] = time.monotonic_ns()
                self.receipt['clock_brackets'] = list(self.clock.records)

        self.thread = threading.Thread(target=run, name='bounded-timeline-range', daemon=True)
        self.thread.start()

    def check(self):
        if self.error is not None:
            raise ValueError('timeline range failed: ' + str(self.error)) from self.error

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=10)
            require(not self.thread.is_alive(), 'capture thread did not retire')
        self.check()


def profile_command(attempt, mode, python):
    return [str(NSYS), 'profile', '--trace=cuda,nvtx,osrt', '--osrt-threshold=1000',
            '--sample=none', '--cpuctxsw=none', '--gpu-metrics-devices=none',
            '--capture-range=nvtx', '--nvtx-capture=' + RANGE, '--capture-range-end=stop',
            '--kill=none', '--wait=all', '--export=sqlite', '--force-overwrite=false',
            '--session-new=glimmer-' + attempt.name + '-' + mode,
            '--output=' + str(attempt / (mode + '-timeline')), str(python), '-B',
            str(Path(__file__).resolve()), '--trace-attempt', str(attempt), '--worker-mode', mode]


@contextmanager
def launch_hooks(common, h, sustained, guard, mode, clean, tools, clock):
    """Replace only module-local bindings; never mutate global subprocess.Popen."""
    original = dict(subprocess=sustained.subprocess, owner=sustained.owner_command,
                    owned=common.owned_environment, verify=common.verify_process,
                    pending=common.check_pending, schedule=sustained.schedule)
    profiled = dict(clean, **tools)
    service_argv, verified, frontend_environment = [], {}, {}

    def owner(*args, **kwargs):
        argv, pins = original['owner'](*args, **kwargs)
        require(not service_argv and mode == 'equal', 'one equal service launch required')
        service_argv.extend(argv)
        return argv, pins

    def popen(argv, **kwargs):
        wrapper = argv == [sys.executable, '-B', '-c', h.wrapper]
        service = bool(service_argv) and argv == service_argv
        expected = {'cwd', 'env', 'stdout', 'stderr', 'start_new_session'}
        if wrapper:
            expected |= {'stdin', 'text'}
        require((wrapper or service) and set(kwargs) == expected and kwargs['cwd'] == PROJECT_ROOT
                and kwargs['env'] == profiled and kwargs['stderr'] == subprocess.STDOUT
                and kwargs['start_new_session'] is True, 'unexpected profiled launch form')
        if wrapper:
            require(kwargs['stdin'] == subprocess.PIPE and kwargs['text'] is True,
                    'unchanged stdin bootstrap required')
            kwargs = dict(kwargs, env=dict(clean))
        return original['subprocess'].Popen(argv, **kwargs)

    def owned(environment, *args, **kwargs):
        require(mode == 'equal' and environment == profiled, 'exact service environment required')
        stripped = {k: v for k, v in environment.items() if k not in tools}
        require(stripped == clean, 'only exact tool keys may be removed')
        selected = original['owned'](stripped, *args, **kwargs)
        require(not frontend_environment, 'one exact frontend environment required')
        frontend_environment.update(selected)
        return selected

    def verify(helper, process, start, command=None, environment=None):
        original['verify'](helper, process, start, command, environment)
        guard.observe(process, start)
        if command is not None and environment is not None:
            expected = (profiled if mode == 'native' or command == service_argv
                        else frontend_environment)
            require(expected and environment == expected, 'exact frontend/owner environment differs')
            verified[process.pid] = (process, start, list(command), dict(environment))

    def authenticate():
        result = {}
        for pid, (process, start, argv, env) in list(verified.items()):
            if env != profiled:
                require(mode == 'equal' and env == frontend_environment
                        and not (set(tools) - {'LD_PRELOAD'}) & env.keys(),
                        'frontend inherited profiler injection')
                continue
            original['verify'](h, process, start, argv, env)
            expected = capture.observer.process_record(pid)
            require(expected['cgroup'] == guard.group, 'GPU owner escaped task unit')
            with capture.observer.PinnedProcess(expected) as proof:
                mapped = importlib.import_module('collect_ordinary_timeline').maps(h, pid, start, tools)
                proof.check()
            result[str(pid)] = dict(identity=expected, command=argv, environment=env, mapped_inputs=mapped)
        require(len(result) == (2 if mode == 'native' else 1), 'exact traced owner inventory required')
        return result

    def schedule(*args, **kwargs):
        value = original['schedule'](*args, **kwargs)
        clock.start(value, authenticate)
        return value

    def pending(*args, **kwargs):
        guard.check()
        clock.check()
        return original['pending'](*args, **kwargs)

    sustained.subprocess = SimpleNamespace(Popen=popen, PIPE=subprocess.PIPE, STDOUT=subprocess.STDOUT)
    sustained.owner_command, common.owned_environment = owner, owned
    common.verify_process, common.check_pending, sustained.schedule = verify, pending, schedule
    try:
        yield profiled
    finally:
        sustained.subprocess, sustained.owner_command = original['subprocess'], original['owner']
        common.owned_environment, common.verify_process = original['owned'], original['verify']
        common.check_pending, sustained.schedule = original['pending'], original['schedule']
        require(sustained.subprocess is original['subprocess'] and common.verify_process is original['verify']
                and common.check_pending is original['pending'] and sustained.schedule is original['schedule']
                and common.owned_environment is original['owned'] and sustained.owner_command is original['owner'],
                'diagnostic hooks were not restored')


def trace_worker(attempt, mode):
    require_capture_ready()
    report, guard, clock = dict(status='failed', mode=mode, **FLAGS), None, None
    try:
        common, h, sustained, timeline = helpers()
        tools = timeline.bootstrap(os.environ)
        timeline.validate_configs(tools)
        clock = CaptureRange(timeline)
        report['tools'] = tools
        for key in tools:
            os.environ.pop(key, None)
        guard = candidate.ResourceGuard(attempt, mode)
        plan = recipe.read(attempt / 'plan.json')
        require(plan.get('native_equal') is True and plan['recipe'] == recipe.expected_recipe(plan),
                'unchanged native/equal plan required')
        manifest = recipe.read(Path(plan['reference']) / 'manifest.json')
        h.startup_reference, h.export = common.application_references(h, manifest)
        require(h.query('gpu', h.training.GPU_FIELDS) == plan['checkpoint_gpu'], 'GPU identity differs')
        report['resources_before'] = sustained.load_resources().resource_snapshot()
        require(report['resources_before']['available_device_sms'] == 80, 'full physical device unavailable')
        clean = h.training.clean_environment(plan['gpu_uuid'])
        directory = attempt / mode
        with launch_hooks(common, h, sustained, guard, mode, clean, tools, clock) as environment:
            job = sustained.run_control(h, sustained.load_measurement(), directory, mode, environment, plan['point'])
        clock.close()
        capture.audit_application(sustained, sustained.load_measurement(), h, directory, job, plan['point'])
        raw = directory / 'attempt.json'
        raw_hash = digest(raw)
        raw.rename(directory / 'helper-attempt.json')
        report['job'] = copy.deepcopy(job)
        report['job'].update(FLAGS, status='current_candidate_timeline_control_requires_review',
                            raw_helper_receipt=dict(path='helper-attempt.json', sha256=raw_hash))
        save(directory / 'attempt.json', report['job'])
        report['resources_after'] = sustained.load_resources().resource_snapshot()
        require(report['resources_after'] == report['resources_before'], 'physical resource policy changed')
        guard.check(force=True)
        report['status'] = 'passed_requires_review'
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        if clock is not None:
            try:
                clock.close()
            except BaseException as error:
                report.update(status='failed', range_error=str(error))
            report['capture_range'] = clock.receipt
        if guard is not None:
            report['resources'] = guard.receipt
        save(attempt / (mode + '-trace.json'), report)
    return int(report['status'] != 'passed_requires_review')


def worker(attempt, mode):
    require_capture_ready()
    report = dict(status='failed', mode=mode, **FLAGS)
    try:
        require(not os.environ.get('LD_PRELOAD') and not os.environ.get('LD_AUDIT'), 'clean profiler launcher required')
        guard = candidate.ResourceGuard(attempt, mode)
        argv = profile_command(attempt, mode, sys.executable)
        report['command'] = argv
        with (attempt / (mode + '-profiler.log')).open('x') as log:
            report['profiler_returncode'] = subprocess.run(argv, cwd=PROJECT_ROOT, stdout=log,
                stderr=subprocess.STDOUT, timeout=700, check=False, umask=0o077).returncode
        require(report['profiler_returncode'] == 0, 'profiler or traced controller failed')
        report['trace'] = recipe.read(attempt / (mode + '-trace.json'))
        require(report['trace']['status'] == 'passed_requires_review', 'traced control did not pass')
        from bert_timeline_audit import analyze
        report['analysis'] = analyze(attempt, mode)
        guard.check(force=True)
        report.update(status='passed_requires_review', resources=guard.receipt)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
    save(attempt / (mode + '-worker.json'), report)
    return int(report['status'] != 'passed_requires_review')


def collect(plan_path):
    require_capture_ready()
    plan = recipe.read(plan_path)
    require(plan.get('native_equal') is True, 'explicit native/equal checkpoint plan required')
    recipe.validate_plan(plan)
    attempt = create_attempt(resolve_artifact_root('bert-current-timeline', category='profiles'))
    save(attempt / 'plan.json', plan)
    print('Timeline evidence: ' + str(attempt), flush=True)
    report = dict(status='started', jobs=[], plan_source=str(plan_path), **FLAGS)
    paths = {Path(__file__).resolve(), PROJECT_ROOT / 'scripts/bert_timeline_audit.py', NSYS}
    paths.update(NSYS.parent / name for name in ('libToolsInjection64.so', 'libToolsInjectionProxy64.so',
                                               'libLinuxKeyboardInterceptorProxy.so'))
    before = {str(p): digest(p) for p in paths}
    save(attempt / 'timeline-inputs-before.json', before)
    try:
        require(not os.environ.get('LD_PRELOAD') and not os.environ.get('LD_AUDIT'), 'clean outer collector required')
        require(shutil.disk_usage(attempt).free >= 8 * 1024 ** 3, '8GiB free required; no deletion')
        report['idle_window'] = wait_for_idle(attempt)
        require(report['idle_window']['achieved'], 'idle window unavailable; no capture')
        for mode in ('native', 'equal'):
            recipe.validate_plan(plan)
            require(before == {str(p): digest(p) for p in paths}, 'timeline inputs changed')
            try:
                unit, result = candidate.launch_unit(attempt, mode, worker_script=Path(__file__))
                report['jobs'].append(dict(mode=mode, unit=unit, worker=result))
            finally:
                post = dict(status='failed')
                try:
                    recipe.validate_plan(plan)
                    require(before == {str(p): digest(p) for p in paths}, 'timeline inputs changed')
                    post['status'] = 'complete_input_seals_unchanged'
                finally:
                    save(attempt / (mode + '-postflight.json'), post)
            save(attempt / 'report.json', report)
        report['status'] = 'current_candidate_timeline_controls_require_independent_review'
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        save(attempt / 'report.json', report)
    print(json.dumps(dict(status=report['status'], error=report.get('error'))), flush=True)
    return int(report['status'] == 'failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--collect', type=Path, help='one native/equal batch using an unchanged prepared plan')
    group.add_argument('--worker-attempt', type=Path, help=argparse.SUPPRESS)
    group.add_argument('--trace-attempt', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--worker-mode', choices=('native', 'equal'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.collect or args.worker_attempt or args.trace_attempt:
        require_capture_ready()
    if args.collect:
        return collect(args.collect.resolve(strict=True))
    if args.worker_attempt or args.trace_attempt:
        require(args.worker_mode is not None, 'explicit internal mode required')
        folder = (args.worker_attempt or args.trace_attempt).resolve(strict=True)
        return (worker if args.worker_attempt else trace_worker)(folder, args.worker_mode)
    parser.print_help()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
