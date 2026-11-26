#!/usr/bin/env python3
"""Build a source-matched host compiler pair from an explicit link inventory."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shlex
import shutil
import signal
import subprocess
import tarfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
REGION = ROOT / 'build/candidate/CMakeFiles/glimmer_host_closure.dir'
MEMORY = 12 * 1024 ** 3
DISK_FLOOR = 8 * 1024 ** 3
BUILD_LIMIT = 1536 * 1024 ** 2
FILE_LIMIT = 512 * 1024 ** 2


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


class Build:
    def __init__(self, inventory, evidence, inventory_path):
        self.inventory = inventory
        self.evidence = evidence
        self.inventory_path = inventory_path
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.commands = []
        self.started = time.monotonic()
        self.last_check = 0
        self.peak_allocated = 0
        self.resource_checks = 0
        self.sampled_peak_memory = 0
        group = Path('/proc/self/cgroup').read_text().strip().split('0::', 1)[1]
        require(group.endswith(f'/glimmer-host-{evidence.name}.service'), 'wrong task cgroup')
        self.cgroup = Path('/sys/fs/cgroup' + group)
        require((self.cgroup / 'memory.max').read_text().strip() == str(MEMORY), 'memory hard cap')
        require((self.cgroup / 'memory.swap.max').read_text().strip() == '0', 'zero swap required')
        require((self.cgroup / 'pids.max').read_text().strip() == '256', 'task hard cap')
        require(resource.getrlimit(resource.RLIMIT_FSIZE) == (FILE_LIMIT, FILE_LIMIT), 'file hard cap')
        require(not REGION.exists() and REGION.parent.resolve() == REGION.parent, 'fresh fixed target region')
        REGION.mkdir()
        for name in ('objects', 'archives', 'products', 'dependencies', 'tests'):
            (REGION / name).mkdir()
        self.environment = dict(os.environ)
        for key in ('LD_PRELOAD', 'LD_AUDIT', 'LD_LIBRARY_PATH'):
            self.environment.pop(key, None)
        self.environment['PATH'] = '/usr/bin:/bin'
        require(not any(self.environment.get(key) for key in
                        ('CPATH', 'CPLUS_INCLUDE_PATH', 'C_INCLUDE_PATH', 'LIBRARY_PATH',
                         'GCC_EXEC_PREFIX', 'COMPILER_PATH')), 'implicit compiler input overrides')
        self.objects = inventory['objects']
        require(len(self.objects) == 85 and inventory['counts']['archive_members'] == 79,
                'exact complete six-archive closure required')
        self.paths = {}
        for index, (old, record) in enumerate(self.objects.items()):
            directory = REGION / 'objects' / str(index)
            directory.mkdir()
            self.paths[old] = directory / Path(old).name
            require(not record['optimization_flags'], 'unoptimized source command required')
            require(not any('NDEBUG' in value for value in record['macro_flags']), 'assertions required')
            require(str(Path(record['ordered_argv'][0]).resolve()) == record['compiler'],
                    'compiler invocation must resolve to the frozen canonical driver')
        self.archives = {old: REGION / 'archives' / Path(old).name
                         for old in {r['archive'] for r in self.objects.values()} if old}
        require(len(set(self.archives.values())) == len(self.archives), 'distinct archive names')
        self.members = {(str(self.archives[row['archive']]), row['member'])
                        for row in self.objects.values() if row['archive']}
        expected = {(str(self.archives[old]), member)
                    for old, row in inventory['archive_members'].items() for member in row['members']}
        require(self.members == expected and len(self.members) == 79, 'complete archive member contracts')
        self.shadowed = {(str(self.archives[row['archive']]), row['member'])
                         for row in inventory['shadowed_archive_members']}
        require(len(self.shadowed) == 2 and self.shadowed <= self.members,
                'two exact direct-object override contracts required')
        self.prepare_tests()
        self.check(force=True)

    def check(self, force=False):
        with self.lock:
            require(not self.stop.is_set(), 'another command failed')
            now = time.monotonic()
            require(now - self.started < 1800, 'whole build deadline')
            if not force and now - self.last_check < .5:
                return
            self.last_check = now
            require(shutil.disk_usage(ROOT).free >= DISK_FLOOR, 'disk floor')
            allocated = 0
            for path in REGION.rglob('*'):
                try:
                    allocated += path.stat().st_blocks * 512
                except FileNotFoundError:
                    continue  # Compiler temporary rename; the next check observes its destination.
            self.peak_allocated = max(self.peak_allocated, allocated)
            require(allocated <= BUILD_LIMIT, 'working build storage limit')
            self.last_resources = {name: (self.cgroup / name).read_text()
                                   for name in ('memory.current', 'memory.events',
                                                'memory.swap.current', 'pids.current', 'pids.events')}
            peak = self.cgroup / 'memory.peak'
            self.last_resources['memory.peak'] = peak.read_text() if peak.exists() else None
            self.sampled_peak_memory = max(self.sampled_peak_memory,
                                           int(self.last_resources['memory.current']))
            for name in ('memory.events', 'pids.events'):
                events = self.last_resources[name].splitlines()
                require(events and all(int(line.split()[1]) == 0 for line in events), name + ' changed')
            require(self.last_resources['memory.swap.current'].strip() == '0', 'swap used')
            self.resource_checks += 1

    def run(self, argv, label, cwd=ROOT, timeout=1800, environment=None):
        self.check()
        log = self.evidence / (label + '.log')
        record = {'argv': list(map(str, argv)), 'cwd': str(cwd), 'log': str(log)}
        begin = time.monotonic()
        with self.lock:
            self.commands.append(record)
        print(json.dumps({'command': label, 'state': 'started'}), flush=True)
        with log.open('x') as stream:
            process = subprocess.Popen(record['argv'], cwd=cwd, env=environment or self.environment,
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                while process.poll() is None:
                    self.check()
                    require(time.monotonic() - begin < timeout, label + ' timeout')
                    time.sleep(.2)
                record['returncode'] = process.returncode
                record['seconds'] = time.monotonic() - begin
                require(process.returncode == 0, label + ' failed: ' + str(process.returncode))
            except BaseException:
                self.stop.set()
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                record['returncode'] = process.wait()
                raise
        self.check()
        print(json.dumps({'command': label, 'state': 'completed',
                          'returncode': record['returncode']}), flush=True)

    def argv(self, old, variant):
        row = self.objects[old]
        args = row['ordered_argv'].copy()
        args[args.index('-o') + 1] = str(self.paths[old])
        if variant == 'o2' and row['language'] == 'host_cxx':
            args.append('-O2')
        return args

    def dependency_union(self):
        jobs = []
        for index, (old, row) in enumerate(self.objects.items()):
            if row['language'] != 'host_cxx':
                continue
            for variant in ('o0', 'o2'):
                args = self.argv(old, variant)
                args.remove('-c')
                output = args.index('-o')
                del args[output:output + 2]
                dep = REGION / 'dependencies' / f'{index}-{variant}.d'
                args += ['-M', '-MF', str(dep), '-MT', str(self.paths[old])]
                jobs.append((args, f'dep-{index}-{variant}', row['command']['directory']))
        for name, test in self.tests.items():
            args = shlex.split(test['entry']['command'])
            args.remove('-c')
            output = args.index('-o')
            del args[output:output + 2]
            args += ['-M', '-MF', str(REGION / 'dependencies' / ('fixture-' + name + '.d')),
                     '-MT', str(test['object'])]
            jobs.append((args, 'dep-fixture-' + name, test['entry']['directory']))
        for variant in ('o0', 'o2'):
            args, directory = self.abi_argv(variant)
            output = args.index('-o')
            del args[output:output + 2]
            args += ['-M', '-MF', str(REGION / 'dependencies' / ('abi-' + variant + '.d')),
                     '-MT', str(REGION / 'tests/abi')]
            jobs.append((args, 'dep-abi-' + variant, directory))
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda job: self.run(*job), jobs))
        dependencies = set()
        sets = {}
        for dep in sorted((REGION / 'dependencies').glob('*.d')):
            words = shlex.split(dep.read_text().replace('\\\n', ' ').split(':', 1)[1])
            require(all(Path(word).is_absolute() and Path(word).is_file() for word in words),
                    'absolute existing dependency inputs required')
            sets[dep.name] = sorted(set(words))
            dependencies.update(words)
        for index, (old, row) in enumerate(self.objects.items()):
            dependencies.add(row['command']['file'])
            if row['language'] != 'host_cxx':
                continue
            delta = set(sets[f'{index}-o0.d']) ^ set(sets[f'{index}-o2.d'])
            require(all(path.startswith('/usr/include/') for path in delta),
                    'only system-header dependency selection may vary')
        # Link inputs and tooling stay fixed too; retained old object records may be absent.
        dependencies.update(path for path, item in self.inventory['file_identities'].items()
                            if item['exists'])
        dependencies.add(str(Path(__file__).resolve()))
        dependencies.add(str(self.evidence / 'plan.json'))
        dependencies.add(str(self.inventory_path))
        for row in self.objects.values():
            dependencies.update((row['ordered_argv'][0], row['compiler']))
        dependencies.update(['/usr/bin/ar', '/usr/bin/ranlib', '/usr/bin/as', '/usr/bin/ld',
                             '/usr/bin/readelf', '/usr/bin/nm',
                             '/usr/lib/gcc/x86_64-linux-gnu/11/cc1plus',
                             '/usr/lib/gcc/x86_64-linux-gnu/11/collect2'])
        for test in self.tests.values():
            dependencies.add(test['entry']['file'])
            dependencies.add(str(test['link_file']))
        dependencies.add(str(ROOT / 'scripts/execution_host_abi.cc'))
        dependencies.add(str(ROOT / 'build/debug/src/interceptor/execution/libcuda.so.execution-lookup-fixture.so'))
        self.dependency_sets = sets
        self.frozen = {path: digest(path) for path in sorted(dependencies)}
        save(self.evidence / 'frozen-inputs.json', self.frozen)
        save(self.evidence / 'dependency-sets.json', sets)
        source_inputs = {path for values in sets.values() for path in values}
        source_inputs.update(row['command']['file'] for row in self.objects.values())
        capsule = self.evidence / 'source-inputs.tar.gz'
        with tarfile.open(capsule, 'x:gz', compresslevel=1) as archive:
            for path in sorted(source_inputs):
                archive.add(path, arcname=str(Path(path).relative_to('/')), recursive=False)
        self.verify_capsule(capsule, {str(Path(path).relative_to('/')): self.frozen[path]
                                      for path in source_inputs})
        self.verify_inputs()

    def verify_inputs(self):
        for path, expected in self.frozen.items():
            require(digest(path) == expected, 'frozen input changed: ' + path)
        self.check(force=True)

    def verify_capsule(self, path, expected):
        seen = set()
        with tarfile.open(path, 'r:gz') as archive:
            for member in archive:
                require(member.isfile() and member.name in expected, 'unexpected capsule member')
                value = hashlib.sha256()
                with archive.extractfile(member) as stream:
                    while chunk := stream.read(1024 * 1024):
                        value.update(chunk)
                require(value.hexdigest() == expected[member.name], 'capsule byte mismatch')
                seen.add(member.name)
        require(seen == set(expected), 'incomplete capsule')

    def compile(self, variant):
        self.verify_inputs()
        jobs = []
        for index, (old, row) in enumerate(self.objects.items()):
            if row['language'] == 'assembly':
                shutil.copyfile(old, self.paths[old])
                continue
            args = self.argv(old, variant)
            args += ['-MD', '-MF', str(self.paths[old]) + '.d']
            jobs.append((args, f'{variant}-compile-{index}', row['command']['directory']))
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda job: self.run(*job), jobs))
        for index, (old, row) in enumerate(self.objects.items()):
            if row['language'] != 'host_cxx':
                continue
            dep = Path(str(self.paths[old]) + '.d')
            values = shlex.split(dep.read_text().replace('\\\n', ' ').split(':', 1)[1])
            require(sorted(set(values)) == self.dependency_sets[f'{index}-{variant}.d'],
                    'actual compiler dependencies differ from frozen preprocessor closure')
        verified_members = {}
        for index, (old, new) in enumerate(sorted(self.archives.items())):
            shutil.copyfile(old, new)
            members = [self.paths[key] for key, row in self.objects.items() if row['archive'] == old]
            before = subprocess.check_output(['ar', 't', old])
            require(before.decode().splitlines() == self.inventory['archive_members'][old]['members'],
                    'frozen archive order changed')
            self.run(['ar', 'r', new, *members], f'{variant}-archive-{index}')
            self.run(['ranlib', new], f'{variant}-ranlib-{index}')
            require(subprocess.check_output(['ar', 't', new]) == before, 'archive order changed')
            for key, row in self.objects.items():
                if row['archive'] != old:
                    continue
                expected = digest(self.paths[key])
                actual = hashlib.sha256(subprocess.check_output(['ar', 'p', new, row['member']])).hexdigest()
                require(actual == expected, 'archive member bytes differ from rebuilt object')
                verified_members[str(new) + '(' + row['member'] + ')'] = actual
        require(len(verified_members) == 79, 'all archive member contents must be verified')
        save(self.evidence / (variant + '-archive-members.json'), verified_members)
        self.check(force=True)

    def link(self, variant):
        results = {}
        replacements = {**{key: str(value) for key, value in self.paths.items()},
                        **{key: str(value) for key, value in self.archives.items()}}
        for name, row in self.inventory['products'].items():
            directory = Path(row['link_directory'])
            output = REGION / 'products' / ('libglimmer_execution_interceptor.so'
                                          if name.endswith('interceptor') else name)
            linkmap = REGION / 'products' / (name + '.map')
            args = []
            original = row['ordered_link_argv']
            for index, arg in enumerate(original):
                if index and original[index - 1] == '-o':
                    args.append(str(output))
                elif arg.startswith('-Wl,-Map,'):
                    args.append('-Wl,-Map,' + str(linkmap))
                elif not arg.startswith('-'):
                    resolved = str((directory / arg).resolve())
                    args.append(replacements.get(resolved, resolved))
                else:
                    args.append(arg)
            self.run(args, f'{variant}-link-{name}', directory)
            header = linkmap.read_text().split('Merging program properties', 1)[0]
            actual = re.findall(r'^([^\s]+\.a)\(([^()]+\.o)\)$', header, re.M)
            expected = {(str(self.archives[item['archive']]), item['member'])
                        for item in row['all_extracted_members'] if item['archive'] in self.archives}
            project = {(archive, member) for archive, member in actual
                       if 'libglimmer_' in Path(archive).name}
            require(project and project <= self.members and not project & self.shadowed,
                    'old, unknown or shadowed production archive member extracted')
            for old in self.paths:
                require(old not in linkmap.read_text(), 'old direct production object linked')
            results[name] = {'path': str(output), 'sha256': digest(output),
                             'map_sha256': digest(linkmap), 'argv': args,
                             'extracted_project_members': sorted(project),
                             'unextracted_rebuilt_members': sorted(self.members - project),
                             'additional_members_since_original_map': sorted(project - expected)}
            self.run(['readelf', '-d', output], f'{variant}-dynamic-{name}')
            self.run(['nm', '-D', '--defined-only', output], f'{variant}-exports-{name}')
            self.run(['readelf', '--debug-dump=info', '--dwarf-depth=1', output],
                     f'{variant}-dwarf-{name}')
        save(self.evidence / (variant + '-products.json'), results)
        self.verify_inputs()

    def prepare_tests(self):
        candidate = ROOT / 'build/candidate'
        debug = ROOT / 'build/debug'
        specs = {
            'core': (debug, 'tests/core/execution_priority_test.cc',
                     'tests/CMakeFiles/glimmer_core_execution_priority_test.dir/link.txt'),
            'queue': (debug, 'tests/backend/execution_submission_queue_test.cc',
                      'tests/CMakeFiles/glimmer_backend_execution_submission_queue_test.dir/link.txt'),
            'dispatcher': (debug, 'tests/backend/cuda_execution_dispatcher_test.cc',
                           'src/backend/cuda/CMakeFiles/glimmer_backend_cuda_execution_dispatcher_test.dir/link.txt'),
            'service': (candidate, 'tests/control/execution_service_test.cc',
                        'CMakeFiles/glimmer_execution_service_test.dir/link.txt'),
            'priority': (debug, 'tests/control/execution_priority_service_test.cc',
                         'src/execution_service/CMakeFiles/glimmer_execution_priority_service_test.dir/link.txt'),
            'receipt': (debug, 'tests/interceptor/execution_async_receipt_test.cc',
                        'src/interceptor/execution/CMakeFiles/glimmer_execution_async_receipt_test.dir/link.txt'),
        }
        self.tests = {}
        for name, (slot, source, link) in specs.items():
            entries = json.loads((slot / 'compile_commands.json').read_text())
            selected = [row for row in entries if row['file'] == str(ROOT / source)]
            require(len(selected) == 1, 'unique test compile contract: ' + name)
            entry = selected[0]
            args = shlex.split(entry['command'])
            require(not any(value.startswith('-O') for value in args), 'unchanged O0 fixture required')
            old_object = (Path(entry['directory']) / args[args.index('-o') + 1]).resolve()
            self.tests[name] = {'entry': entry, 'old_object': str(old_object),
                                'object': REGION / 'tests' / (name + '.o'),
                                'link_file': slot / link}

    def check_test_map(self, linkmap):
        text = linkmap.read_text()
        header = text.split('Merging program properties', 1)[0]
        members = re.findall(r'^([^\s]+\.a)\(([^()]+\.o)\)$', header, re.M)
        project = {(archive, member) for archive, member in members
                   if 'libglimmer_' in Path(archive).name}
        require(project <= self.members and not project & self.shadowed,
                'old, unknown or shadowed production member in fixture')
        require(not any(old in text for old in self.paths), 'old production object in fixture')

    def test(self, variant):
        replacements = {**{old: str(new) for old, new in self.paths.items()},
                        **{old: str(new) for old, new in self.archives.items()}}
        service_archive = next(new for old, new in self.archives.items()
                               if Path(old).name == 'libglimmer_execution_service_runtime.a')
        replacements[str(ROOT / 'build/debug/src/execution_service/libglimmer_execution_service_runtime.a')] = str(service_archive)
        replacements[str(ROOT / 'build/debug/lib/libglimmer_execution_interceptor.so')] = str(REGION / 'products/libglimmer_execution_interceptor.so')
        transport = next(new for old, new in self.paths.items()
                         if self.objects[old]['archive'] is None and
                         self.objects[old]['command']['file'] == str(ROOT / 'src/control/execution_socket.cc'))
        session = next(new for old, new in self.paths.items()
                       if self.objects[old]['archive'] is None and
                       self.objects[old]['command']['file'] == str(ROOT / 'src/backend/cuda/cuda_execution_session.cc'))
        fixture_environment = dict(self.environment, LD_LIBRARY_PATH=str(REGION / 'products') + ':' +
                                   str(ROOT / 'build/debug/src/interceptor/execution'))
        for name, row in self.tests.items():
            if variant == 'o0':
                args = shlex.split(row['entry']['command'])
                args[args.index('-o') + 1] = str(row['object'])
                self.run(args, 'fixture-compile-' + name, row['entry']['directory'])
            else:
                require(digest(row['object']) == self.o0_identities['tests/' + name + '.o'],
                        'fixture-only object changed between variants')
            link_file = row['link_file']
            directory = link_file.parents[2]
            original = shlex.split(link_file.read_text())
            output = REGION / 'tests' / name
            linkmap = REGION / 'tests' / (name + '.map')
            args = []
            for index, arg in enumerate(original):
                if index and original[index - 1] == '-o':
                    args.append(str(output))
                elif arg.startswith('-Wl,-Map,'):
                    continue
                elif not arg.startswith('-'):
                    absolute = str((directory / arg).resolve())
                    if absolute == row['old_object']:
                        args.append(str(row['object']))
                        if name in ('priority', 'receipt'):
                            args.append(str(transport))
                        if name == 'priority':
                            args.append(str(session))
                    else:
                        args.append(replacements.get(absolute, absolute))
                else:
                    args.append(arg)
            args.append('-Wl,-Map,' + str(linkmap))
            self.run(args, f'{variant}-fixture-link-{name}', directory)
            self.check_test_map(linkmap)
            cases = [[]]
            if name == 'priority':
                cases.append(['--priority-idle'])
            if name == 'receipt':
                source = Path(row['entry']['file']).read_text()
                scope = source.split('constexpr std::array kScenarios{', 1)[1].split('};', 1)[0]
                cases = [[case] for case in re.findall(r'"([a-z-]+)"', scope)]
                require(len(cases) == 38, 'exact existing receipt scenarios required')
            for index, case in enumerate(cases):
                self.run([output, *case], f'{variant}-check-{name}-{index}',
                         timeout=15 if name == 'receipt' else 120, environment=fixture_environment)
        args, directory = self.abi_argv(variant)
        self.run(args, variant + '-abi-build', directory)
        self.run([REGION / 'tests/abi'], variant + '-abi-layout')
        if variant == 'o2':
            require((self.evidence / 'o0-abi-layout.log').read_bytes() ==
                    (self.evidence / 'o2-abi-layout.log').read_bytes(), 'ABI layout mismatch')
            for name in self.inventory['products']:
                def dynamic(mode):
                    text = (self.evidence / f'{mode}-dynamic-{name}.log').read_text()
                    return [line.split(')', 1)[1] for line in text.splitlines()
                            if any('(' + key + ')' in line for key in ('NEEDED', 'RUNPATH', 'RPATH', 'SONAME'))]
                require(dynamic('o0') == dynamic('o2'), 'dynamic link contract mismatch')
                if name.endswith('interceptor'):
                    def symbols(mode):
                        return sorted(' '.join(line.split()[1:]) for line in
                                      (self.evidence / f'{mode}-exports-{name}.log').read_text().splitlines())
                    require(symbols('o0') == symbols('o2'), 'frontend public exports differ')
        self.verify_inputs()

    def abi_argv(self, variant):
        runtime = next(old for old, row in self.objects.items()
                       if row['command']['file'].endswith('/execution_service/runtime.cc'))
        args = self.argv(runtime, variant)
        args[args.index('-o') + 1] = str(REGION / 'tests/abi')
        args[args.index('-c') + 1] = str(ROOT / 'scripts/execution_host_abi.cc')
        args.remove('-c')
        args.append('-I' + str(ROOT / 'src/execution_service'))
        return args, self.objects[runtime]['command']['directory']

    def archive(self, variant):
        files = sorted(path for path in REGION.rglob('*') if path.is_file())
        identities = {str(path.relative_to(REGION)): digest(path) for path in files}
        save(self.evidence / (variant + '-working-identities.json'), identities)
        capsule = self.evidence / (variant + '-working.tar.gz')
        with tarfile.open(capsule, 'x:gz', compresslevel=1) as archive:
            for path in files:
                archive.add(path, arcname=str(path.relative_to(REGION)), recursive=False)
        self.verify_capsule(capsule, identities)
        if variant == 'o0':
            self.o0_identities = identities
        save(self.evidence / (variant + '-capsule.json'),
             {'path': str(capsule), 'sha256': digest(capsule), 'bytes': capsule.stat().st_size,
              'all_members_byte_verified': True})
        self.check(force=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', required=True, type=Path)
    parser.add_argument('--evidence', required=True, type=Path)
    parser.add_argument('--build', action='store_true')
    args = parser.parse_args()
    require(args.build, 'explicit --build required')
    evidence = args.evidence.resolve(strict=True)
    require(evidence.is_relative_to(ROOT / 'artifacts/maintenance') and
            (evidence / 'plan.json').is_file(), 'existing planned evidence directory required')
    report = {'status': 'failed', 'gpu_used': False}
    build = None
    try:
        build = Build(json.loads(args.inventory.read_text()), evidence, args.inventory.resolve())
        build.dependency_union()
        for variant in ('o0', 'o2'):
            build.compile(variant)
            build.link(variant)
            build.test(variant)
            build.archive(variant)
        report['status'] = 'pair_built_cpu_checks_passed_requires_review'
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
    finally:
        if build is not None:
            report.update(commands=build.commands, resource_checks=build.resource_checks,
                          peak_working_allocated_bytes=build.peak_allocated,
                          sampled_peak_memory_bytes=build.sampled_peak_memory,
                          sampled_peak_is_not_kernel_peak=True,
                          last_resources=getattr(build, 'last_resources', {}))
        save(evidence / 'build-report.json', report)
    print(json.dumps({key: value for key, value in report.items()
                      if key in ('status', 'error', 'peak_working_allocated_bytes')}), flush=True)
    return int(report['status'] == 'failed')


if __name__ == '__main__':
    raise SystemExit(main())
