#!/usr/bin/env python3
"""CUDA-free, bounded multiprocess qualification of the timeline protocol."""

import argparse
from contextlib import ExitStack
import ctypes
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

import run_bert_timeline_diagnostic as capture
from artifact_paths import create_attempt, resolve_artifact_root
from run_rnn_v8_validation import save

require = capture.require
CLEAN = dict(PATH='/usr/bin:/bin', LANG='C', LC_ALL='C', PYTHONNOUSERSITE='1')


def publish(path, value):
    """Make a complete receipt visible before another process can consume it."""
    pending = path.with_name(path.name + '.pending')
    require(not path.exists() and not pending.exists(), 'fresh probe receipt required')
    save(pending, value)
    with pending.open('rb') as stream:
        os.fsync(stream.fileno())
    os.link(pending, path)
    pending.unlink()


def process_receipt():
    pid = os.getpid()
    environment = dict(item.split('=', 1) for item in
                       Path(f'/proc/{pid}/environ').read_text().split('\0') if item)
    return dict(pid=pid, start=capture.candidate.identity(pid),
                cgroup=capture.candidate.cgroup(pid), environment=environment,
                monotonic_ns=time.monotonic_ns())


def profiler_tools(timeline, environment, nsys):
    """Keep the bootstrap contract, rebasing only exact installed tool paths."""
    selected_paths = dict(
        CUDA_INJECTION64_PATH=str(nsys.parent / 'libToolsInjection64.so'),
        NVTX_INJECTION64_PATH=str(nsys.parent / 'libToolsInjection64.so'),
        LD_PRELOAD=':'.join(str(nsys.parent / name) for name in
                           ('libToolsInjectionProxy64.so', 'libLinuxKeyboardInterceptorProxy.so')))
    require(all(environment.get(key) == value for key, value in selected_paths.items()),
            'selected profiler injection paths differ')
    original_paths = {key: timeline.CONSTANTS[key] for key in selected_paths}
    checked = timeline.bootstrap(dict(environment, **original_paths))
    return dict(checked, **selected_paths)


def profiler_identity(nsys):
    require(nsys.is_absolute() and nsys.name == 'nsys' and os.access(nsys, os.X_OK),
            'absolute executable profiler target required')
    paths = [nsys, *(nsys.parent / name for name in
                    ('libToolsInjection64.so', 'libToolsInjectionProxy64.so',
                     'libLinuxKeyboardInterceptorProxy.so'))]
    require(all(path.is_file() for path in paths), 'complete profiler target tree required')
    version = subprocess.run([str(nsys), '--version'], env=CLEAN, capture_output=True,
                             text=True, timeout=5, check=True).stdout.strip()
    require(version.startswith('NVIDIA Nsight Systems version '), 'profiler version missing')
    return dict(executable=str(nsys), version=version,
                files={str(path): capture.digest(path) for path in paths})


def child(output, until, nsys):
    _, _, _, timeline = capture.helpers()
    tools = profiler_tools(timeline, os.environ, nsys)
    timeline.validate_configs(tools)
    clock = timeline.ClockBrackets()
    receipt = dict(start=process_receipt(), records=[])
    publish(output.with_suffix('.ready.json'), receipt['start'])
    while time.monotonic_ns() < until:
        clock.mark('probe_child')
        time.sleep(.02)
    receipt.update(end=process_receipt(), records=clock.records)
    publish(output, receipt)


def controller(attempt, nsys):
    _, h, _, timeline = capture.helpers()
    tools = profiler_tools(timeline, os.environ, nsys)
    timeline.validate_configs(tools)
    clock = capture.CaptureRange(timeline)
    receipt = dict(start=process_receipt(), tools=tools, children=[])
    processes, logs = [], []
    try:
        helper = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), '--clean-helper'],
                                env=CLEAN, capture_output=True, text=True, timeout=5, check=True)
        receipt['clean_helper'] = json.loads(helper.stdout)
        require(receipt['clean_helper']['environment'] == CLEAN, 'clean helper environment differs')
        until = time.monotonic_ns() + 18_000_000_000
        for index in range(2):
            output = attempt / f'child-{index}.json'
            log = (attempt / f'child-{index}.log').open('x')
            logs.append(log)
            process = subprocess.Popen([sys.executable, '-B', '-c', h.wrapper],
                env=CLEAN, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
                text=True, start_new_session=True)
            processes.append(process)
            argv = [sys.executable, '-B', str(Path(__file__).resolve()), '--child', str(output),
                    '--until', str(until), '--nsys', str(nsys)]
            process.stdin.write(json.dumps(dict(command=argv, environment=dict(CLEAN, **tools))) + '\n')
            process.stdin.close()
        deadline = time.monotonic() + 5
        while not all((attempt / f'child-{i}.ready.json').exists() for i in range(2)):
            require(all(p.poll() is None for p in processes) and time.monotonic() < deadline,
                    'probe children failed readiness')
            time.sleep(.01)
        receipt['children'] = [capture.recipe.read(attempt / f'child-{i}.ready.json') for i in range(2)]
        for process, row in zip(processes, receipt['children']):
            require(row['pid'] == process.pid and row['environment'] == dict(CLEAN, **tools)
                    and row['cgroup'] == receipt['start']['cgroup'], 'probe child identity differs')
        clock.start(dict(begin_ns=time.monotonic_ns() - 3_000_000_000), lambda: {})
        clock.thread.join(timeout=12)
        clock.close()
        receipt['range'] = clock.receipt
        receipt['after_range'] = process_receipt()
        require(all(p.poll() is None for p in processes), 'children stopped with range')
        publish(attempt / 'range-complete.json', dict(schema='timeline_stop_trigger_v1',
            controller=receipt['after_range'], tools=tools,
            pop_low_ns=clock.receipt['pop_low_ns'], pop_high_ns=clock.receipt['pop_high_ns']))
        receipt['returncodes'] = [p.wait(timeout=20) for p in processes]
        require(receipt['returncodes'] == [0, 0], 'children did not retire normally')
        receipt['end'] = process_receipt()
        require(all(receipt[phase]['environment'] == receipt['start']['environment']
                    for phase in ('after_range', 'end')), 'controller environment changed')
        receipt['status'] = 'complete_requires_review'
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
        for log in logs:
            log.close()
        receipt['finished_ns'] = time.monotonic_ns()
        publish(attempt / 'controller.json', receipt)


def analyze(attempt, returned_ns, ending):
    receipt = capture.recipe.read(attempt / 'controller.json')
    require(receipt.get('status') == 'complete_requires_review', 'controller did not finish normally')
    require(receipt['finished_ns'] < returned_ns, 'profiler returned before controller retirement')
    trigger = capture.recipe.read(attempt / 'range-complete.json')
    require(trigger['schema'] == 'timeline_stop_trigger_v1'
            and trigger['controller'] == receipt['after_range'] and trigger['tools'] == receipt['tools']
            and all(trigger[key] == receipt['range'][key] for key in ('pop_low_ns', 'pop_high_ns')),
            'trigger and final controller receipt differ')
    database = attempt / 'protocol-timeline.sqlite'
    db = sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    try:
        require(db.execute('pragma integrity_check').fetchone()[0] == 'ok', 'corrupt export')
        strings = dict(db.execute('select id,value from StringIds'))
        events = [dict(row, message=row['text'] if row['text'] is not None else strings[row['textId']])
                  for row in db.execute('select start,end,text,textId,globalTid from NVTX_EVENTS')]
        ranges = [r for r in events if r['message'] == capture.RANGE]
        require(len(ranges) == 1 and ranges[0]['end'] is not None, 'unique complete range missing')
        range_row = ranges[0]
        low, high = range_row['start'], range_row['end']
        tail_ns = 1_000_000_000 if ending in ('explicit-stop', 'interactive') else 0
        require(6_000_000_000 < high - low < 8_000_000_000, 'range duration differs')
        by_text = {}
        for row in events:
            require(row['message'] not in by_text, 'duplicate marker')
            by_text[row['message']] = row
        marks = receipt['range']['clock_brackets']
        for mark in marks:
            row = by_text.get(mark['text'])
            require(row and row['end'] is None and low <= row['start'] <= high
                    and row['globalTid'] == range_row['globalTid'], 'missing controller clock')
        children = []
        for index in range(2):
            child_receipt = capture.recipe.read(attempt / f'child-{index}.json')
            pid = child_receipt['start']['pid']
            require(child_receipt['end']['monotonic_ns'] > receipt['range']['pop_high_ns'] + 1_000_000_000,
                    'child did not continue after capture')
            observed = [r for r in events if r['message'].startswith(f'GLIMMER_CLOCK_V1:{pid}:')]
            require(observed and all(low <= r['start'] <= high + tail_ns for r in observed),
                    'child events escape bounded capture')
            expected = [r for r in child_receipt['records']
                        if receipt['range']['analysis_begin_ns'] <= r['monotonic_low']
                        and r['monotonic_high'] <= receipt['range']['analysis_end_ns']]
            require(expected and all(r['text'] in by_text for r in expected), 'child inner markers missing')
            thread = by_text[expected[0]['text']]['globalTid']
            calls = [dict(row) for row in db.execute(
                'select start,end,nameId from OSRT_API where globalTid=?', (thread,))
                if strings[row['nameId']] == 'select']
            require(calls and all(low <= row['start'] <= row['end'] <= high + tail_ns for row in calls),
                    'child OSRT events escape bounded capture')
            for left, right in zip(expected, expected[1:]):
                a, b = by_text[left['text']]['start'], by_text[right['text']]['start']
                require(sum(a <= row['start'] <= row['end'] <= b for row in calls) == 1,
                        'inner child heartbeat lacks exactly one recorded sleep')
            children.append(dict(pid=pid, inner_markers=len(expected), recorded_markers=len(observed),
                                 inner_sleep_calls=len(expected)-1))
        diagnostics = [dict(row) for row in db.execute('select * from DIAGNOSTIC_EVENT')]
        allowed = {'No CUDA events collected. Does the process use CUDA?',
                   'CUDA profiling might have not been started correctly.',
                   'Scheduling information is absent. The thread activity is deduced based on OS runtime libraries traces. This is inaccurate and does not take into account asynchronous interrupts and exception faults.'}
        require(all(row['severity'] == 1 or row['text'] in allowed for row in diagnostics),
                'unqualified profiler warnings')
        stopped = [row['timestamp'] for row in diagnostics if row['text'] == 'Profiling has stopped.']
        require(len(stopped) == 1 and high <= stopped[0] <= high + 1_000_000_000,
                'collection did not stop within one second of range end')
        return dict(status='protocol_probe_passed_requires_review', range_ns=high-low,
                    controller_markers=len(marks), children=children, diagnostics=diagnostics)
    finally:
        db.close()


def interactive_commands(attempt, nsys, python):
    session = 'glimmer-' + attempt.name + '-protocol'
    start = [str(nsys), 'start', '--session-new=' + session, '--capture-range=nvtx',
             '--capture-range-end=none', '--sample=none', '--cpuctxsw=none',
             '--gpu-metrics-devices=none', '--export=sqlite', '--force-overwrite=false',
             '--output=' + str(attempt / 'protocol-timeline')]
    launch = [str(nsys), 'launch', '--session=' + session, '--wait=all',
              '--trace=cuda,nvtx,osrt', '--osrt-threshold=1000', '--cuda-event-trace=false',
              '--nvtx-capture=' + capture.RANGE, str(python), '-B',
              str(Path(__file__).resolve()), '--controller', str(attempt), '--nsys', str(nsys)]
    return start, launch


def session_listing(nsys, session):
    listing = subprocess.run([str(nsys), 'sessions', 'list'], env=CLEAN,
                             capture_output=True, text=True, timeout=3, check=True).stdout
    matches = [line.split() for line in listing.splitlines() if line.split()
               and line.split()[-1] == session]
    require(len(matches) <= 1, 'ambiguous named session')
    return listing, matches


def pin_probe_owners(attempt, owner, agent_pid, agent_start, pins):
    rows = [owner, *(capture.recipe.read(attempt / f'child-{index}.ready.json') for index in range(2))]
    for pid in [agent_pid, *(row['pid'] for row in rows)]:
        descriptor = os.pidfd_open(pid)
        pins.callback(os.close, descriptor)
    require(capture.candidate.identity(agent_pid) == agent_start, 'pinned agent identity changed')
    for row in rows:
        pid = row['pid']
        actual = dict(item.split('=', 1) for item in
                      Path(f'/proc/{pid}/environ').read_text().split('\0') if item)
        require(capture.candidate.identity(pid) == row['start']
                and capture.candidate.cgroup(pid) == owner['cgroup']
                and actual == row['environment'], 'pinned probe owner identity differs')
    return [dict(pid=row['pid'], start=row['start'], cgroup=row['cgroup']) for row in rows]


def collect(ending, nsys):
    require(nsys.is_absolute(), 'absolute executable profiler target required')
    nsys = nsys.resolve(strict=True)
    profiler_before = profiler_identity(nsys)
    attempt = create_attempt(resolve_artifact_root('bert-timeline-protocol', category='profiles'))
    print('Protocol evidence: ' + str(attempt), flush=True)
    python = capture.PROJECT_ROOT / 'examples/framework_workloads/pytorch_smoke/.venv/bin/python'
    argv = capture.profile_command(attempt, 'protocol', python)
    argv[0] = str(nsys)
    target = argv.index('--trace-attempt') - 1
    argv[target:] = [str(Path(__file__).resolve()), '--controller', str(attempt), '--nsys', str(nsys)]
    argv[argv.index('--capture-range-end=stop')] = '--capture-range-end=' + (
        'none' if ending == 'explicit-stop' else ending)
    session = 'glimmer-' + attempt.name + '-protocol'
    start_command = None
    if ending == 'interactive':
        start_command, argv = interactive_commands(attempt, nsys, python)
    report = dict(status='failed', command=argv, ending=ending, cuda_initialized=False,
                  profiler_before=profiler_before)
    sources = (Path(__file__).resolve(), Path(capture.__file__).resolve())
    report['source_before'] = {str(p): capture.digest(p) for p in sources}
    try:
        if start_command is not None:
            report['start'] = dict(command=start_command, begin_ns=time.monotonic_ns())
            started = subprocess.run(start_command, env=CLEAN, umask=0o077,
                                     capture_output=True, text=True, timeout=10, check=False)
            report['start'].update(returncode=started.returncode, stdout=started.stdout,
                                   stderr=started.stderr, returned_ns=time.monotonic_ns())
            require(started.returncode == 0, 'interactive session start failed')
            listing, matches = session_listing(nsys, session)
            require(len(matches) == 1, 'new interactive session missing')
            report['start'].update(session_list_after=listing, session_id=matches[0][0])
        with (attempt / 'profiler.log').open('x') as log:
            with subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                  env=CLEAN, umask=0o077) as process:
                start = capture.candidate.identity(process.pid)
                if ending in ('explicit-stop', 'interactive'):
                    deadline = time.monotonic() + 15
                    while not (attempt / 'range-complete.json').exists():
                        require(process.poll() is None and time.monotonic() < deadline,
                                'missing live range completion')
                        time.sleep(.01)
                    trigger = capture.recipe.read(attempt / 'range-complete.json')
                    owner = trigger['controller']
                    agent_pid = int(trigger['tools']['QUADD_AGENT_PID'])
                    agent_start = capture.candidate.identity(agent_pid)
                    require(process.poll() is None and capture.candidate.identity(process.pid) == start
                            and capture.candidate.identity(owner['pid']) == owner['start']
                            and capture.candidate.cgroup(process.pid) == owner['cgroup']
                            == capture.candidate.cgroup(os.getpid())
                            and capture.candidate.cgroup(agent_pid) == owner['cgroup']
                            and Path(f'/proc/{agent_pid}').stat().st_uid == os.getuid()
                            and Path(f'/proc/{agent_pid}/exe').samefile(nsys)
                            and (ending == 'interactive' or agent_pid == process.pid),
                            'stop session process provenance differs')
                    listing, matches = session_listing(nsys, session)
                    require(len(matches) == 1 and matches[0][0] == trigger['tools']['NSYS_PROFILING_SESSION_ID'],
                            'exact live session identity missing')
                    if ending == 'interactive':
                        require(matches[0][0] == report['start']['session_id'], 'interactive session changed')
                    require(capture.candidate.identity(agent_pid) == agent_start, 'agent PID changed')
                    report['stop'] = dict(session_list_before=listing, agent_pid=agent_pid,
                                        agent_start=agent_start, controller=owner['pid'],
                                        range_pop_ns=trigger['pop_high_ns'],
                                        command=[str(nsys), 'stop', '--session=' + session],
                                        begin_ns=time.monotonic_ns())
                    with ExitStack() as pins:
                        report['stop']['pinned_owners'] = pin_probe_owners(
                            attempt, owner, agent_pid, agent_start, pins)
                        stopped = subprocess.run(report['stop']['command'], env=CLEAN, capture_output=True,
                                                 text=True, timeout=15, check=False)
                    report['stop'].update(end_ns=time.monotonic_ns(), returncode=stopped.returncode,
                                          stdout=stopped.stdout, stderr=stopped.stderr)
                    require(stopped.returncode == 0, 'owned session stop failed')
                returncode = process.wait(timeout=50)
        returned_ns = time.monotonic_ns()
        report['returncode'] = returncode
        report['launch_returned_ns' if ending == 'interactive' else 'profiler_returned_ns'] = returned_ns
        # Retain the unit while any early-return controller completes; early return
        # still fails qualification and never authorizes a BERT launch.
        deadline = time.monotonic() + 20
        while not (attempt / 'controller.json').exists() and time.monotonic() < deadline:
            time.sleep(.05)
        require(returncode == 0, 'profiler failed')
        report['analysis'] = analyze(attempt, returned_ns, ending)
        report['source_after'] = {str(p): capture.digest(p) for p in sources}
        require(report['source_before'] == report['source_after'], 'probe source changed')
        if ending != 'interactive':
            report['sessions_after'], matches = session_listing(nsys, session)
            require(not matches, 'probe session still exists')
        report['status'] = 'protocol_probe_passed_requires_review'
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
    if ending == 'interactive' and report.get('start', {}).get('session_id'):
        try:
            listing, matches = session_listing(nsys, session)
            if matches:
                require(matches[0][0] == report['start']['session_id'], 'cleanup session identity changed')
                command = [str(nsys), 'shutdown', '--session=' + session, '--kill=none']
                retired = subprocess.run(command, env=CLEAN, capture_output=True,
                                         text=True, timeout=5, check=False)
                report['shutdown'] = dict(command=command, returncode=retired.returncode,
                                          stdout=retired.stdout, stderr=retired.stderr)
                require(retired.returncode == 0, 'interactive session cleanup failed')
            report['sessions_after'], matches = session_listing(nsys, session)
            require(not matches, 'interactive session still exists')
        except BaseException as error:
            report.update(status='failed', cleanup_error=f'{type(error).__name__}: {error}')
    report['supervisor_handoff_ns'] = time.monotonic_ns()
    try:
        report['profiler_after'] = profiler_identity(nsys)
        require(report['profiler_before'] == report['profiler_after'], 'profiler tree changed')
    except BaseException as error:
        report.update(status='failed', profiler_identity_error=f'{type(error).__name__}: {error}')
    save(attempt / 'report.json', report)
    print(json.dumps(report), flush=True)
    return int(report['status'] == 'failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--collect', choices=('stop', 'stop-shutdown', 'explicit-stop', 'interactive'))
    modes.add_argument('--controller', type=Path)
    modes.add_argument('--child', type=Path)
    modes.add_argument('--clean-helper', action='store_true')
    parser.add_argument('--until', type=int)
    parser.add_argument('--nsys', type=Path, default=capture.NSYS,
                        help='Absolute nsys executable in the selected profiler target directory')
    args = parser.parse_args()
    if args.collect:
        return collect(args.collect, args.nsys)
    if args.controller:
        controller(args.controller, args.nsys)
    elif args.child:
        child(args.child, args.until, args.nsys)
    else:
        print(json.dumps(process_receipt()))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
