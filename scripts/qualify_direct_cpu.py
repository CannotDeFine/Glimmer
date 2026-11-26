"""Bounded standalone qualification of the optional direct CPU sampler."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import signal
import statistics
import struct
import subprocess
import time

MAGIC = 0x474C435055303031
FILE_MAX = 16 * 1024**2
ROOT = Path(__file__).resolve().parents[1]
LIBPROFILER = Path("/usr/lib/x86_64-linux-gnu/libprofiler.so.0").resolve(strict=True)
PPROF = Path("/usr/bin/google-pprof").resolve(strict=True)
NM = Path("/usr/bin/nm").resolve(strict=True)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def identity(path):
    return dict(path=str(path), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def start_ticks(pid):
    return int(Path(f"/proc/{pid}/stat").read_bytes().rsplit(b") ", 1)[1].split()[19])


def verify_mappings(directory, process, owner_start, paths, name):
    require(start_ticks(process.pid) == owner_start, "owner identity changed")
    maps = Path(f"/proc/{process.pid}/maps").read_bytes()
    (directory / name).write_bytes(maps)
    for path in paths:
        info = path.stat()
        found = False
        for line in maps.splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) != 6 or fields[5] != os.fsencode(path):
                continue
            major, minor = (int(value, 16) for value in fields[3].split(b":"))
            require(int(fields[4]) == info.st_ino and os.makedev(major, minor) == info.st_dev,
                    "mapping inode differs")
            found = True
        require(found, f"mapped image missing: {path}")


def parse_profile(data):
    require(64 <= len(data) <= FILE_MAX, "profile size")
    require(struct.unpack_from("<5Q", data) == (0, 3, 0, 2000, 0), "profile header")
    records, offset = [], 40
    while True:
        require(offset + 16 <= len(data), "truncated record")
        count, depth = struct.unpack_from("<2Q", data, offset)
        offset += 16
        require(0 < depth <= 128 and offset + 8 * depth <= len(data), "invalid depth")
        chain = struct.unpack_from(f"<{depth}Q", data, offset)
        offset += depth * 8
        if count == 0:
            require(chain == (0,), "invalid trailer")
            break
        require(0 < count <= 100000 and chain[0] != 0, "invalid sample")
        records.append((count, chain))
    mappings = data[offset:].decode("utf-8", errors="strict")
    require("libprofiler.so" in mappings and "direct_cpu" in mappings, "missing mappings")
    total = sum(item[0] for item in records)
    require(0 < total <= 100000, "sample count")
    return records, mappings


def parser_controls():
    good = struct.pack("<11Q", 0, 3, 0, 2000, 0, 1, 1, 100, 0, 1, 0)
    good += b"libprofiler.so direct_cpu\n"
    require(parse_profile(good)[0] == [(1, (100,))], "positive parser control")
    invalid = [good[:40], good[:72], good.replace(struct.pack("<Q", 2000),
               struct.pack("<Q", 1000), 1), good[:-26]]
    for data in invalid:
        try:
            parse_profile(data)
        except (ValueError, UnicodeError, struct.error):
            continue
        raise ValueError("negative parser control accepted")
    return 5


def child_limits(sanitized):
    resource.setrlimit(resource.RLIMIT_FSIZE, (FILE_MAX, FILE_MAX))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (32, 32))
    if not sanitized:
        resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))


def launch(directory, observer, fixture, enabled, *, mutation=None, sanitized=False):
    directory.mkdir(mode=0o700)
    require(not any(key in ("LD_PRELOAD", "LD_AUDIT") or key.startswith("CPUPROFILE")
                    for key in os.environ), "unclean parent")
    read_fd, write_fd = os.pipe2(os.O_CLOEXEC)
    begin = time.monotonic_ns() + 1000000000
    end = begin + 5000000000
    environment = dict(os.environ, LD_PRELOAD=str(observer), CPUPROFILE_FREQUENCY="500",
                       GLIMMER_CPU_COMMAND_FD=str(read_fd), GLIMMER_CPU_DIRECTORY=str(directory))
    if sanitized:
        asan = subprocess.check_output(["g++", "-print-file-name=libasan.so"], text=True).strip()
        environment["LD_PRELOAD"] = f"{asan}:{observer}"
        environment["ASAN_OPTIONS"] = "detect_leaks=1:halt_on_error=1"
        environment["UBSAN_OPTIONS"] = "halt_on_error=1"
        environment["GLIMMER_FAKE_MODE"] = mutation or "normal"
    command = [str(fixture), str(begin), str(end)] if fixture.name != "true" else [str(fixture)]
    wall_start = time.monotonic_ns()
    usage_start = resource.getrusage(resource.RUSAGE_CHILDREN)
    with (directory / "stdout").open("xb") as stdout, (directory / "stderr").open("xb") as stderr:
        process = subprocess.Popen(command, env=environment, pass_fds=(read_fd,),
                                   start_new_session=True, stdout=stdout, stderr=stderr,
                                   preexec_fn=lambda: child_limits(sanitized))
        os.close(read_fd)
        try:
            owner_start = start_ticks(process.pid)
            limits = Path(f"/proc/{process.pid}/limits").read_bytes()
            (directory / "limits-before-command").write_bytes(limits)
            require(re.search(rb"^Max cpu time\s+32\s+32\s+seconds\s*$", limits, re.M)
                    and start_ticks(process.pid) == owner_start, "owned aggregate CPU limit")
            values = [MAGIC, 1, process.pid, owner_start, begin, end, int(enabled), 0]
            if mutation == "wrong-owner":
                values[3] += 1
            if mutation == "late":
                values[4] -= 2000000000
                values[5] -= 2000000000
            data = struct.pack("<8Q", *values)
            if mutation == "truncated":
                data = data[:-1]
            if mutation == "extra":
                data += b"x"
            # Two writes deliberately exercise partial command delivery.
            require(os.write(write_fd, data[:17]) == 17, "command prefix write")
            time.sleep(0.005)
            require(os.write(write_fd, data[17:]) == len(data) - 17, "command suffix write")
            os.close(write_fd)
            write_fd = -1
            save(directory / "launch.json", dict(command=command, pid=process.pid,
                 start_ticks=owner_start, begin_ns=begin, end_ns=end, enabled=enabled,
                 mutation=mutation, command_hex=data.hex(), observer=str(observer)))
            if not sanitized and mutation is None:
                for delay, name in ((1000000000, "maps-first"), (4000000000, "maps-last")):
                    remaining = (begin + delay - time.monotonic_ns()) / 1e9
                    if remaining > 0:
                        time.sleep(remaining)
                    verify_mappings(directory, process, owner_start,
                                    (observer, LIBPROFILER), name)
            remaining_seconds = 15 - (time.monotonic_ns() - wall_start) / 1e9
            require(remaining_seconds > 0, "cell wall deadline")
            returncode = process.wait(timeout=remaining_seconds)
        finally:
            if write_fd >= 0:
                os.close(write_fd)
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
    usage_end = resource.getrusage(resource.RUSAGE_CHILDREN)
    result = dict(returncode=returncode, whole_wall_ns=time.monotonic_ns() - wall_start,
                  whole_cpu_ns=round(((usage_end.ru_utime + usage_end.ru_stime) -
                      (usage_start.ru_utime + usage_start.ru_stime)) * 1e9),
                  owner_absent=not Path(f"/proc/{process.pid}").exists())
    save(directory / "exit.json", result)
    require(result["owner_absent"], "owner not retired")
    return result


def receipt(directory, enabled):
    values = struct.unpack("<24Q", (directory / "receipt.bin").read_bytes())
    launch_data = json.loads((directory / "launch.json").read_text())
    expected = struct.unpack("<8Q", bytes.fromhex(launch_data["command_hex"]))
    require(values[:8] == expected, "receipt command differs")
    require(values[11] == int(enabled) and values[15] == 0 and not any(values[20:]), "state")
    require(values[8] <= values[4] - 100000000 and values[4] <= values[9] <= values[10]
            <= values[4] + 50000000, "start boundary")
    require(values[5] <= values[13] <= values[14] <= values[5] + 50000000, "stop boundary")
    require(values[19] >= values[14], "finalization order")
    profile = directory / "owner.cpu.prof"
    require(profile.exists() == enabled, "profile presence differs")
    if enabled:
        info = profile.lstat()
        require(not profile.is_symlink() and info.st_uid == os.getuid()
                and (info.st_size, info.st_dev, info.st_ino) == values[16:19], "file identity")
    return values


def symbolize(directory, fixture, observer, records, mappings):
    output = subprocess.run([str(PPROF), "--text", "--cum", "--nodecount=1000",
                             str(fixture), str(directory / "owner.cpu.prof")],
                            capture_output=True, timeout=20, check=False)
    (directory / "pprof.stdout").write_bytes(output.stdout)
    (directory / "pprof.stderr").write_bytes(output.stderr)
    require(output.returncode == 0, "pprof failed")
    # Non-PIE fixture PCs are exact; raw chains are checked before pprof filtering.
    symbols = subprocess.check_output([str(NM), "-S", "--defined-only", str(fixture)], text=True)
    ranges = {}
    for line in symbols.splitlines():
        fields = line.split()
        if len(fields) == 4 and fields[3] in (
                "target_chain", "foreign_chain", "before_chain", "after_chain"):
            ranges[fields[3]] = (int(fields[0], 16), int(fields[0], 16) + int(fields[1], 16))
    require(len(ranges) == 4, "fixture symbol range missing")
    counts = {name: sum(count for count, chain in records
                       if any(low <= pc < high for pc in chain))
              for name, (low, high) in ranges.items()}
    total = sum(count for count, _ in records)
    require(counts["target_chain"] >= total * .95, "target chain sample fraction")
    require(all(counts[name] == 0 for name in (
        "foreign_chain", "before_chain", "after_chain")), "forbidden fixture chain")
    observer_maps = []
    for line in mappings.splitlines():
        fields = line.split()
        if len(fields) >= 6 and Path(fields[-1]) == observer:
            low, high = (int(value, 16) for value in fields[0].split("-"))
            observer_maps.append((low, high))
    require(observer_maps, "observer mapping missing")
    counts["observer"] = sum(count for count, chain in records
        if any(low <= pc < high for pc in chain for low, high in observer_maps))
    require(counts["observer"] == 0, "controller or filter frames sampled")
    return counts


def controls(args):
    report = dict(status="running", parser_controls=parser_controls(), cells=[],
                  inputs=[identity(path) for path in (Path(__file__), args.fake_observer,
                      ROOT / "scripts/direct_cpu_window.cc",
                      ROOT / "tests/support/direct_cpu_fake_profiler.cc")])
    try:
        for name in ("truncated", "extra", "wrong-owner", "late", "start-fail", "stop-fail",
                     "foreign-profiler"):
            cell = args.control_root / ("control-" + name)
            result = launch(cell, args.fake_observer, Path("/usr/bin/true"), True,
                            mutation=name, sanitized=True)
            require(result["returncode"] == 86, f"negative control {name}")
            require(b"ERROR: AddressSanitizer" not in (cell / "stderr").read_bytes()
                    and b"runtime error:" not in (cell / "stderr").read_bytes(), "sanitizer error")
            report["cells"].append(dict(name=name, **result))
        for enabled in (False, True):
            cell = args.control_root / f"control-normal-{int(enabled)}"
            result = launch(cell, args.fake_observer, Path("/usr/bin/true"), enabled,
                            sanitized=True)
            require(result["returncode"] == 0, "fake lifecycle failed")
            receipt(cell, enabled)
            report["cells"].append(dict(name=cell.name, **result))
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save(args.control_root / "controls.json", report)


def collect(args):
    require(json.loads((args.control_root / "controls.json").read_text())["status"] == "passed",
            "CPU controls required")
    require(not (args.attempt / "report.json").exists(), "batch already attempted")
    require(os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize >= 8 * 1024**3, "disk bound")
    inputs = [args.observer, args.fixture, Path(__file__), ROOT / "scripts/direct_cpu_window.cc",
              ROOT / "tests/support/direct_cpu_fixture.cc", args.attempt / "plan.json",
              LIBPROFILER, Path("/usr/include/gperftools/profiler.h"),
              PPROF, NM, Path("/usr/lib/x86_64-linux-gnu/libunwind.so.8").resolve(strict=True),
              args.attempt / "build-receipt.json",
              args.control_root / "controls.json"]
    frozen = [identity(path) for path in inputs]
    save(args.attempt / "frozen-inputs.json", frozen)
    report = dict(status="running", cells=[], application_ready=False,
                  full_device_qualified=False, performance_qualified=False)
    started = time.monotonic()
    try:
        for index, enabled in enumerate((False, True, True, False, False, True)):
            require(time.monotonic() - started < 100, "batch deadline")
            directory = args.attempt / f"cell-{index}-{'on' if enabled else 'off'}"
            result = launch(directory, args.observer, args.fixture, enabled)
            require(result["returncode"] == 0, "fixture did not exit normally")
            state = receipt(directory, enabled)
            work = json.loads((directory / "stdout").read_text())
            require(work["chunks"] > 0 and work["foreign_chunks"] > 0, "missing work")
            result.update(work=work, enabled=enabled,
                          start_uncertainty_ns=state[10] - state[4],
                          stop_uncertainty_ns=state[14] - state[5])
            if enabled:
                records, mappings = parse_profile((directory / "owner.cpu.prof").read_bytes())
                total = sum(count for count, _ in records)
                counters = re.findall(rb"PROFILE: interrupts/evictions/bytes = (\d+)/(\d+)/(\d+)",
                                      (directory / "stderr").read_bytes())
                require(len(counters) == 1 and int(counters[0][0]) == total >= 500
                        and total >= state[12], "sample counts incomplete")
                result.update(samples=total, chains=symbolize(
                    directory, args.fixture, args.observer, records, mappings))
            save(directory / "result.json", result)
            report["cells"].append(result)
        ratios = {"cpu_per_chunk": [], "inverse_throughput": []}
        for offset in (0, 2, 4):
            pair = report["cells"][offset:offset + 2]
            on = next(cell["work"] for cell in pair if cell["enabled"])
            off = next(cell["work"] for cell in pair if not cell["enabled"])
            for key, metric in (("cpu_per_chunk", "cpu_ns"), ("inverse_throughput", "wall_ns")):
                ratios[key].append((on[metric] / on["chunks"]) / (off[metric] / off["chunks"]))
        report["ratios"] = {key: dict(pairs=value, median=statistics.median(value))
                            for key, value in ratios.items()}
        require(all(value["median"] <= 1.20 for value in report["ratios"].values()), "perturbation")
        require(frozen == [identity(path) for path in inputs], "frozen inputs changed")
        report["status"] = "standalone_protocol_passed_pending_review"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        save(args.attempt / "report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--observer", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--fake-observer", type=Path)
    parser.add_argument("--control-root", type=Path)
    parser.add_argument("--controls", action="store_true")
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    args.control_root = args.control_root or args.attempt
    require(args.attempt.is_absolute() and args.attempt.is_dir(), "existing absolute attempt")
    require(args.controls != args.collect, "select controls or collection")
    if args.controls:
        controls(args)
    else:
        collect(args)


if __name__ == "__main__":
    main()
