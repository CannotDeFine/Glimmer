"""One bounded main-owner CPU profile after reviewed two-path numerical checks."""

import argparse
from contextlib import contextmanager
import copy
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import struct
import subprocess
import sys
import time
from types import SimpleNamespace

import analyze_direct_cpu as analysis
from artifact_paths import PROJECT_ROOT, create_attempt, resolve_artifact_root
import bert_device_capture as capture
import qualify_direct_cpu as protocol
import run_bert_candidate_validation as candidate
import run_bert_direct_cpu_numerical as numerical
from run_rnn_v8_validation import digest, save, wait_for_idle

require, read = numerical.require, numerical.read
ENTRY = Path(__file__).resolve()


class DirectWindow:
    """The command pipe is passed only to the exact service child."""

    def __init__(self, directory, plan, guard):
        self.directory, self.plan, self.guard = directory, plan, guard
        directory.mkdir(mode=0o700)
        self.process = None
        self.writer = self.pidfd = -1
        self.begin = self.end = None
        self.maps = []

    def check(self):
        self.guard.check()
        require(self.process is not None and self.process.poll() is None
                and protocol.start_ticks(self.process.pid) == self.start
                and Path(f'/proc/{self.process.pid}/task/{self.process.pid}').is_dir(),
                'main-owner incarnation changed or retired early')
        require(Path(f'/proc/{self.process.pid}/exe').resolve(strict=True) == Path(self.plan['server']),
                'main-owner executable changed')

    def spawn(self, argv, **kwargs):
        require(self.process is None and argv[0] == self.plan['server']
                and kwargs.get('start_new_session') is True and 'pass_fds' not in kwargs,
                'one exact service launch required')
        environment = kwargs['env']
        require(not any(key.startswith(('CPUPROFILE', 'GLIMMER_CPU_')) or
                        key.startswith('LD_') and key != 'LD_LIBRARY_PATH' for key in environment),
                'clean service environment required')
        reader, self.writer = os.pipe2(os.O_CLOEXEC)
        self.reader = reader
        pipe = os.fstat(self.writer)
        self.pipe_identity = dict(device=pipe.st_dev, inode=pipe.st_ino, uid=pipe.st_uid)
        self.environment = dict(environment, LD_PRELOAD=self.plan['observer'], CPUPROFILE_FREQUENCY='500',
                                GLIMMER_CPU_COMMAND_FD=str(reader), GLIMMER_CPU_DIRECTORY=str(self.directory))
        try:
            self.process = subprocess.Popen(argv, **dict(kwargs, env=self.environment, pass_fds=(reader,)))
        finally:
            os.close(reader)
        self.start = protocol.start_ticks(self.process.pid)
        self.pidfd = os.pidfd_open(self.process.pid)
        self.guard.observe(self.process, self.start)
        self.command = list(argv)
        save(self.directory / 'spawn.json', dict(pid=self.process.pid, start_ticks=self.start,
             command=self.command, environment=self.environment, pipe=self.pipe_identity, read_fd=reader))
        return self.process

    def send(self, begin, enabled):
        require(self.begin is None and self.writer >= 0, 'one pipe command only')
        self.check()
        require(time.monotonic_ns() + 100_000_000 <= begin <= time.monotonic_ns() + 300_000_000_000,
                'command lead outside protocol bound')
        endpoint = Path(f'/proc/{self.process.pid}/fd/{self.reader}').stat()
        require(stat.S_ISFIFO(endpoint.st_mode) and
                dict(device=endpoint.st_dev, inode=endpoint.st_ino, uid=endpoint.st_uid) == self.pipe_identity
                and os.get_inheritable(self.writer) is False, 'live owned pipe differs')
        self.begin, self.end = begin, begin + 5_000_000_000
        data = struct.pack('<8Q', protocol.MAGIC, 1, self.process.pid, self.start,
                           self.begin, self.end, int(enabled), 0)
        require(os.write(self.writer, data) == len(data), 'complete atomic pipe command required')
        os.close(self.writer)
        self.writer = -1
        save(self.directory / 'launch.json', dict(command=self.command, pid=self.process.pid,
             start_ticks=self.start, begin_ns=self.begin, end_ns=self.end, enabled=enabled,
             command_hex=data.hex(), observer=self.plan['observer'], pipe=self.pipe_identity))

    def poll(self):
        if self.begin is None or len(self.maps) == 2:
            return
        now = time.monotonic_ns()
        due = self.begin + (1 if not self.maps else 4) * 1_000_000_000
        if now < due:
            return
        self.check()
        require(now < self.end, 'live mapping observation missed sample window')
        name = 'maps-first' if not self.maps else 'maps-last'
        protocol.verify_mappings(self.directory, self.process, self.start,
                                 (Path(self.plan['server']), Path(self.plan['observer']), protocol.LIBPROFILER), name)
        after = time.monotonic_ns()
        require(after < self.end, 'mapping observation crossed stop boundary')
        self.maps.append(dict(path=name, before_ns=now, after_ns=after, pid=self.process.pid,
                              start_ticks=self.start, main_tid=self.process.pid))
        save(self.directory / 'mapping-receipts.json', self.maps)

    def finish(self, enabled, stderr):
        require(self.process.returncode == 0 and not Path(f'/proc/{self.process.pid}').exists()
                and len(self.maps) == 2, 'normal owner retirement and two live mappings required')
        values = protocol.receipt(self.directory, enabled)
        result = dict(normal_retirement=True, receipt=list(values), mappings=self.maps)
        if enabled:
            records, _ = protocol.parse_profile((self.directory / 'owner.cpu.prof').read_bytes())
            counts = re.findall(rb'PROFILE: interrupts/evictions/bytes = (\d+)/(\d+)/(\d+)', stderr)
            total = sum(count for count, _ in records)
            require(len(counts) == 1 and int(counts[0][0]) == total and values[12] <= total,
                    'closed raw profile total differs from profiler receipt')
            result['samples'] = total
        save(self.directory / 'completion.json', result)
        return result

    def close(self):
        for name in ('writer', 'pidfd'):
            descriptor = getattr(self, name)
            if descriptor >= 0:
                os.close(descriptor)
                setattr(self, name, -1)


def preflight(attempt, plan, environment, guard):
    window = DirectWindow(attempt / 'preflight', plan, guard)
    started = time.monotonic()
    try:
        with (window.directory / 'stdout').open('xb') as out, (window.directory / 'stderr').open('xb') as err:
            process = window.spawn([plan['server'], '--help'], env=environment, stdout=out, stderr=err,
                                   start_new_session=True)
            # Popen's exec-error pipe already confirms successful exec; no alternate launcher.
            window.send(time.monotonic_ns() + 1_000_000_000, False)
            while process.poll() is None:
                require(time.monotonic() - started < 15, 'service help preflight wall deadline')
                window.poll()
                time.sleep(.02)
        return window.finish(False, (window.directory / 'stderr').read_bytes())
    finally:
        if window.process is not None and window.process.poll() is None:
            # Only this Popen child, held with a pidfd; cleanup is never evidence of success.
            signal.pidfd_send_signal(window.pidfd, signal.SIGKILL)
            window.process.wait(timeout=5)
        window.close()


@contextmanager
def hooks(common, sustained, window):
    original_subprocess = sustained.subprocess
    verify, pending, schedule = common.verify_process, common.check_pending, sustained.schedule

    def popen(argv, **kwargs):
        if argv[0] == window.plan['server']:
            return window.spawn(argv, **kwargs)
        require('pass_fds' not in kwargs and not any(key.startswith(('CPUPROFILE', 'GLIMMER_CPU_'))
                for key in kwargs['env']), 'sampler leaked to application wrapper')
        return subprocess.Popen(argv, **kwargs)

    def checked_verify(h, process, start, command=None, environment=None):
        if process is window.process:
            require(environment is not None, 'service environment proof required')
            environment = window.environment
        verify(h, process, start, command, environment)
        window.guard.observe(process, start)

    def checked_pending(*args, **kwargs):
        window.guard.check()
        window.poll()
        return pending(*args, **kwargs)

    def checked_schedule(measurement, point, begin):
        value = schedule(measurement, point, begin)
        window.send(begin + 5_000_000_000, True)
        return value

    sustained.subprocess = SimpleNamespace(Popen=popen, PIPE=subprocess.PIPE, STDOUT=subprocess.STDOUT)
    common.verify_process, common.check_pending, sustained.schedule = checked_verify, checked_pending, checked_schedule
    try:
        yield
    finally:
        sustained.subprocess = original_subprocess
        common.verify_process, common.check_pending, sustained.schedule = verify, pending, schedule


def helpers(plan, manifest):
    common, h = numerical.configure(plan)
    h.startup_reference, h.export = common.application_references(h, manifest)
    sustained = importlib.import_module('collect_sustained_application')
    require(sustained.ordinary.common is common, 'one explicit helper binding required')
    measurement, resources = sustained.load_measurement(), sustained.load_resources()
    return common, h, sustained, measurement, resources


def worker(attempt):
    report, guard, window = dict(status='started'), None, None
    try:
        plan, manifest = read(attempt / 'plan.json'), read(attempt / 'manifest.json')
        numerical.authorization(plan, 800)
        numerical.validate_plan(plan)
        common, h, sustained, measurement, resources = helpers(plan, manifest)
        guard = candidate.ResourceGuard(attempt, 'equal')
        require(h.query('gpu', h.training.GPU_FIELDS) == manifest['gpu'], 'GPU identity changed')
        environment = h.training.clean_environment(manifest['gpu'].split(',')[0].strip())
        report['preflight'] = preflight(attempt, plan, environment, guard)
        window = DirectWindow(attempt / 'sampler', plan, guard)
        with hooks(common, sustained, window):
            job = sustained.run_control(h, measurement, attempt / 'application', 'equal', environment,
                                        sustained.frozen_point())
        report['raw_job'] = job
        begin, end = capture.audit_application(sustained, measurement, h, attempt / 'application', job,
                                               sustained.frozen_point())
        # The historical raw receipt predates this observer; retain it byte-for-byte.
        require(window.begin == begin + 5_000_000_000 and window.end < end,
                'profile not inside the offered window')
        actual = copy.deepcopy(job)
        actual.update({**capture.FLAGS, **numerical.FLAGS}, status='cpu_profiled_application_requires_review',
                      ordinary_timing=False, cpu_sampling=True, server_environment=window.environment,
                      raw_helper_sha256=digest(attempt / 'application/attempt.json'))
        require(not any(key.startswith(('CPUPROFILE', 'GLIMMER_CPU_')) for key in job['application_environment'])
                and job['application_environment']['LD_PRELOAD'] == plan['frontend']
                and job['application_environment']['GLIMMER_EXECUTION_HOST_RANGE_SEEK'] == '1',
                'application environment differs')
        report['observation'] = window.finish(True, (attempt / 'application/server.log').read_bytes())
        report['job'] = actual
        report['analysis'] = analysis.analyze(window.directory, plan['server'], read(attempt / 'objects.json'))
        require(report['analysis']['status'] == 'hotspot_capture_requires_review', 'sample/symbol coverage failed')
        guard.check(force=True)
        report['status'] = 'passed_requires_review'
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        if window:
            window.close()
        if guard:
            report['resources'] = guard.receipt
        report.update(numerical.FLAGS)
        save(attempt / 'equal-worker.json', report)
    print(json.dumps(dict(status=report['status'], error=report.get('error'))), flush=True)
    return int(report['status'] != 'passed_requires_review')


def executable_objects(manifest):
    result = {}
    for entry in manifest.values():
        path = Path(entry['target'])
        if str(path) in result:
            continue
        with path.open('rb') as stream:
            elf = stream.read(7)
        if elf != b'\x7fELF\x02\x01\x01':
            continue
        info = path.stat()
        require(info.st_size == entry['bytes'], 'manifest size changed')
        result[str(path)] = dict(path=str(path), bytes=entry['bytes'], sha256=entry['sha256'],
                                 device=info.st_dev, inode=info.st_ino)
    return list(result.values())


def collect(plan_path, numerical_path, review_path):
    plan, prior, review = read(plan_path), read(numerical_path / 'report.json'), read(review_path)
    numerical.authorization(plan, 1800)
    numerical.validate_plan(plan)
    require(prior['status'] == 'fresh_native_equal_numerical_pair_requires_review'
            and prior['input_identity_unchanged'] is True
            and review['status'] == 'approved_for_one_equal_cpu_capture'
            and review['numerical_attempt'] == str(numerical_path)
            and review['numerical_report_sha256'] == digest(numerical_path / 'report.json')
            and review['plan_sha256'] == digest(plan_path), 'independent numerical/capture review required')
    for name, sha in review['capture_source_sha256'].items():
        require(digest(Path(name)) == sha, 'reviewed capture source changed')
    with plan_path.with_name('profile-launch.json').open('x') as stream:
        json.dump(dict(entry=str(ENTRY), plan=str(plan_path), numerical=str(numerical_path)), stream)
    attempt = create_attempt(resolve_artifact_root('bert-direct-cpu', category='profiles'))
    print('CPU capture evidence: ' + str(attempt), flush=True)
    save(attempt / 'plan.json', plan)
    manifest = read(numerical_path / 'manifest.json')
    save(attempt / 'manifest.json', manifest)
    report = dict(status='started', numerical_attempt=str(numerical_path), review=str(review_path), **numerical.FLAGS)
    before = None
    try:
        require(not os.environ.get('LD_PRELOAD') and not os.environ.get('LD_AUDIT'), 'clean collector required')
        require(shutil.disk_usage(attempt).free >= 16 * 1024**3, '8GiB evidence plus8GiB reserve required')
        common, h, sustained, measurement, resources = helpers(plan, manifest)
        original_inputs = read(numerical_path / 'inputs-before.json')
        require(original_inputs == read(numerical_path / 'inputs-after.json'), 'numerical inputs changed')
        paths = {Path(name) for name in original_inputs}
        paths.update((ENTRY, Path(analysis.__file__).resolve(), Path(protocol.__file__).resolve(),
                      Path(capture.__file__).resolve(), analysis.NM, analysis.ADDR2LINE, analysis.CXXFILT,
                      protocol.PPROF, protocol.LIBPROFILER,
                      Path('/usr/lib/x86_64-linux-gnu/libunwind.so.8').resolve(strict=True),
                      plan_path, review_path, numerical_path / 'report.json', sustained.POINT))
        for module in tuple(sys.modules.values()):
            if getattr(module, '__file__', None):
                path = Path(module.__file__).resolve()
                if path.is_relative_to(PROJECT_ROOT) and path.suffix == '.py':
                    paths.add(path)
        before = h.training.manifest(paths)
        require(all(before[name] == value for name, value in original_inputs.items()), 'fresh numerical binding changed')
        save(attempt / 'inputs-before.json', before)
        save(attempt / 'objects.json', executable_objects(before))
        report['resources_before'] = resources.resource_snapshot()
        report['idle_window'] = wait_for_idle(attempt)
        require(report['idle_window']['achieved'], 'idle window unavailable')
        require(h.query('gpu', h.training.GPU_FIELDS) == manifest['gpu'], 'GPU differs from numerical pair')
        report['unit'], report['worker'] = candidate.launch_unit(attempt, 'equal', worker_script=ENTRY)
        report['status'] = 'one_equal_cpu_capture_requires_review'
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        if before is not None:
            try:
                after = h.training.manifest(paths)
                save(attempt / 'inputs-after.json', after)
                report['input_identity_unchanged'] = before == after
                report['resources_after'] = resources.resource_snapshot()
                report['gpu_identity_after'] = h.query('gpu', h.training.GPU_FIELDS)
                require(before == after and report['resources_before'] == report['resources_after']
                        and report['gpu_identity_after'] == manifest['gpu'], 'capture identity/resource changed')
            except BaseException as error:
                report.update(status='failed', identity_error=f'{type(error).__name__}: {error}')
        save(attempt / 'report.json', report)
    print(json.dumps(dict(status=report['status'], error=report.get('error'), attempt=str(attempt))), flush=True)
    return int(report['status'] == 'failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collect', type=Path)
    parser.add_argument('--numerical', type=Path)
    parser.add_argument('--review', type=Path)
    parser.add_argument('--worker-attempt', type=Path)
    parser.add_argument('--worker-mode', choices=('equal',))
    args = parser.parse_args()
    if args.worker_attempt:
        require(args.worker_mode == 'equal', 'only equal CPU capture is in scope')
        return worker(args.worker_attempt.resolve(strict=True))
    if args.collect:
        require(args.numerical is not None and args.review is not None, 'reviewed numerical prerequisite required')
        return collect(args.collect.resolve(strict=True), args.numerical.resolve(strict=True),
                       args.review.resolve(strict=True))
    parser.print_help()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
