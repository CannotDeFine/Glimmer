#!/usr/bin/env python3
"""One bounded native BERT whole-invocation spatial counter capture.

Root supervises the profiler and fixed launch plumbing only. The original
two-role BERT control and numerical reparse run in a caps0 sibling; only its
two authenticated native application targets retain temporary CAP_SYS_ADMIN.
"""

import argparse
from collections import Counter
from contextlib import ExitStack
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import secrets
import select
import shutil
import socket
import stat
import struct
import subprocess
import sys
import tempfile
import time
from types import MappingProxyType

sys.path.insert(0, str(Path(__file__).resolve().parent))

from artifact_paths import PROJECT_ROOT, create_attempt, resolve_artifact_root
import bert_native_ncu_protocol as protocol
import bert_ncu_replay_exec as bridge
import bert_ncu_relay as relay
from bert_sgemm_ncu_process import NCU_INJECTION_ORDER, NcuMappingUnavailable, proc_mapped_paths, verified_preload
import run_bert_sgemm_ncu_capture as ncu
import run_bert_host_pair_activity as activity
from run_rnn_v8_validation import cgroup, command, digest, identity, memory_receipt, save, wait_for_idle

ENTRY = Path(__file__).resolve()
require, read = bridge.require, activity.read
candidate, pair = activity.candidate, activity.pair
FLAGS = dict(activity.FLAGS, native_spatial_participation_verified=False, application_phase_verified=False)
BOOTSTRAP = ('import os,sys; '
             'v=os.read(0,1); '
             'v==b"1" or sys.exit(97); '
             'os.execve(sys.argv[1],sys.argv[1:],dict(os.environ))')
PYTHON_LAYOUT = '''import json,os,pathlib,sysconfig
fields=dict(line.split(":",1) for line in pathlib.Path("/proc/self/status").read_text().splitlines())
assert os.getuid()>0 and fields["NoNewPrivs"].strip()=="1"
assert all(int(fields[key],16)==0 for key in ("CapEff","CapPrm","CapInh","CapAmb"))
print(json.dumps(dict(stdlib=sysconfig.get_path("stdlib"),
    extensions=sysconfig.get_config_var("DESTSHARED"),suffix=sysconfig.get_config_var("EXT_SUFFIX"))))
'''
PYTHON_ENVIRONMENT = '''import json,os,pathlib,sys,sysconfig
fields=dict(line.split(":",1) for line in pathlib.Path("/proc/self/status").read_text().splitlines())
if os.getuid()==0 or fields["NoNewPrivs"].strip()!="1" or any(
    int(fields[key],16) for key in ("CapEff","CapPrm","CapInh","CapAmb")):
    raise RuntimeError("nonroot NNP caps0 Python environment probe required")
print(json.dumps(dict(prefix=sys.prefix,purelib=sysconfig.get_path("purelib"),
    platlib=sysconfig.get_path("platlib"),search_path=sys.path)))
'''


def inventory_identity(metadata):
    return dict(device=metadata.st_dev, inode=metadata.st_ino, mode=metadata.st_mode,
                uid=metadata.st_uid, gid=metadata.st_gid, bytes=metadata.st_size,
                mtime_ns=metadata.st_mtime_ns, ctime_ns=metadata.st_ctime_ns)


def runtime_inventory(root, *, entry_limit=100000, byte_limit=16 * 1024 ** 3):
    """Inventory one fixed package tree, including bytecode and package data.

    File content hashes belong to the surrounding before/after manifest. This
    inventory additionally freezes names, links, directories and file metadata.
    No suffix or package-name whitelist determines the files included.
    """
    root = Path(root)
    require(root == root.resolve(strict=True) and root.is_dir(), 'canonical package root required')
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        fields = line.split()
        require(len(fields) >= 6, 'complete mount identity required')
        mounted = Path(re.sub(r'\\([0-7]{3})', lambda match: chr(int(match[1], 8)), fields[4]))
        require(not mounted.is_relative_to(root), 'mounted package subtree is unsupported')
    device, deadline = root.stat().st_dev, time.monotonic() + 60
    entries, files, inodes, path_bytes = {}, set(), {}, 0
    pending = [root]
    while pending:
        require(len(entries) < entry_limit and time.monotonic() < deadline,
                'native environment inventory entry/time limit')
        path = pending.pop()
        metadata = path.lstat()
        require(metadata.st_dev == device, 'package inventory crossed a mount boundary')
        row = inventory_identity(metadata)
        target, actual = path, metadata
        if stat.S_ISLNK(metadata.st_mode):
            target = path.resolve(strict=True)
            require(target.is_relative_to(root), 'package symlink escapes the fixed root')
            actual = target.stat()
            require(actual.st_dev == device, 'package symlink crossed a mount boundary')
            row.update(link=os.readlink(path), target=str(target), target_identity=inventory_identity(actual))
        require(stat.S_ISREG(actual.st_mode) or stat.S_ISDIR(actual.st_mode),
                'nonregular package input is unsupported')
        if stat.S_ISREG(actual.st_mode):
            files.add(str(path))
            path_bytes += actual.st_size
            key = (actual.st_dev, actual.st_ino)
            require(key not in inodes or inodes[key] == actual.st_size, 'package inode changed during inventory')
            inodes[key] = actual.st_size
            require(path_bytes <= byte_limit, 'native environment inventory byte limit')
        elif not stat.S_ISLNK(metadata.st_mode):
            children = sorted(path.iterdir())
            row['entries'] = [child.name for child in children]
            pending.extend(reversed(children))
        entries[str(path)] = row
    return dict(root=str(root), entries=entries, files=sorted(files), path_entries=len(entries),
                file_paths=len(files), unique_file_inodes=len(inodes), path_bytes=path_bytes,
                unique_file_bytes=sum(inodes.values()))


def interpreter_chain(path):
    """Retain the invoked venv entry and every interpreter symlink hop."""
    result = {}
    while True:
        path = Path(path)
        require(path.is_absolute() and str(path) not in result and len(result) < 16,
                'invalid or cyclic interpreter chain')
        metadata = path.lstat()
        row = inventory_identity(metadata)
        result[str(path)] = row
        if not stat.S_ISLNK(metadata.st_mode):
            require(stat.S_ISREG(metadata.st_mode), 'regular Python interpreter required')
            break
        link = os.readlink(path)
        row['link'] = link
        path = Path(os.path.normpath(str(path.parent / link)))
    return result


def python_environment_inputs(environment, stdlib, extensions):
    """Freeze the actual fixed venv rather than selected framework packages."""
    argv = [str(bridge.SETPRIV), '--no-new-privs', str(protocol.PYTHON),
            '-I', '-B', '-c', PYTHON_ENVIRONMENT]
    output = subprocess.check_output(argv, env=environment, text=True, timeout=10)
    layout = json.loads(output)
    require(set(layout) == {'prefix', 'purelib', 'platlib', 'search_path'}, 'fixed Python environment layout required')
    prefix = protocol.PYTHON.parent.parent.resolve(strict=True)
    require(Path(layout['prefix']).resolve(strict=True) == prefix, 'target Python left its fixed venv')
    roots = sorted({str(Path(layout[key]).resolve(strict=True)) for key in ('purelib', 'platlib')})
    expected = prefix / 'lib' / stdlib.name / 'site-packages'
    require(roots == [str(expected)] and expected.resolve(strict=True) == expected,
            'unexpected Python package root')
    archive = stdlib.parent / (stdlib.name.replace('.', '') + '.zip')
    allowed = {str(stdlib), str(extensions), str(expected), str(archive)}
    require(type(layout['search_path']) is list and len(layout['search_path']) == len(set(layout['search_path']))
            and set(layout['search_path']) <= allowed
            and {str(stdlib), str(extensions), str(expected)} <= set(layout['search_path'])
            and not archive.exists(), 'unfrozen Python search path or stdlib archive')
    config = prefix / 'pyvenv.cfg'
    require(config.is_file() and config.resolve(strict=True) == config, 'fixed regular venv configuration required')
    trees = {root: runtime_inventory(Path(root)) for root in roots}
    chain = interpreter_chain(protocol.PYTHON)
    paths = {Path(name) for tree in trees.values() for name in tree['files']}
    paths.update(map(Path, chain))
    paths.add(config)
    return paths, dict(status='fixed_venv_inventory_not_application_qualification', layout=layout,
        layout_command=argv, layout_stdout=output, trees=trees, interpreter_chain=chain,
        configuration=str(config), configuration_identity=inventory_identity(config.lstat()),
        absent_search_paths=[str(archive)], historical_mapped_bytes_proven=False,
        ordinary_application_preflight_required=True)


def check_python_environment(receipt):
    require(all(not Path(name).exists() for name in receipt['absent_search_paths']),
            'previously absent Python search path appeared')
    require(interpreter_chain(protocol.PYTHON) == receipt['interpreter_chain'] and
            inventory_identity(Path(receipt['configuration']).lstat()) == receipt['configuration_identity'],
            'venv interpreter or configuration identity changed')
    require({name: runtime_inventory(Path(name)) for name in receipt['trees']} == receipt['trees'],
            'native package inventory changed')


def elf_metadata(output):
    """Read dynamic identities, not an alternate implementation of ld.so."""
    require(type(output) is str and output.strip(), 'ELF metadata required')
    result = {}
    for name in ('NEEDED', 'SONAME', 'RPATH', 'RUNPATH'):
        values = re.findall(r'\(' + name + r'\)[^\n]*\[([^\]]*)\]', output)
        require(len(values) == len(set(values)) and (name == 'NEEDED' or len(values) <= 1),
                'duplicate ELF dynamic identity')
        if name in ('NEEDED', 'SONAME'):
            require(all(value and '/' not in value and not any(char.isspace() for char in value)
                        for value in values), 'non-SONAME dynamic dependency is unsupported')
        result[name.lower()] = values if name == 'NEEDED' else next(iter(values), None)
    interpreters = re.findall(r'\[Requesting program interpreter: ([^\]]+)\]', output)
    require(len(interpreters) <= 1 and all(value.startswith('/') for value in interpreters),
            'one absolute ELF interpreter required')
    result['interpreter'] = next(iter(interpreters), None)
    return result


def loader_namespace(output, interpreter=None, *, no_dependencies=False):
    """Parse the real loader's per-root resolution, rejecting incomplete traces."""
    require(type(output) is str and output.strip() and 'not found' not in output,
            'complete resolved loader trace required')
    if output.strip() == 'statically linked':
        require(no_dependencies, 'loader omitted declared ELF dependencies')
        return {}
    result = {}
    for line in output.splitlines():
        line = line.strip()
        if re.fullmatch(r'linux-vdso\.so\.\d+ \(0x[0-9a-fA-F]+\)', line):
            continue
        match = re.fullmatch(r'(?:(\S+) => )?(/\S+) \(0x[0-9a-fA-F]+\)', line)
        require(match is not None, 'unexpected native loader trace row: ' + line[:256])
        path = Path(match[2]).resolve(strict=True)
        name = match[1] or Path(match[2]).name
        if name.startswith('/'):
            require(interpreter is not None and Path(name).resolve(strict=True) == path == interpreter,
                    'only the exact PT_INTERP alias may be absolute')
            name = Path(name).name
        require('/' not in name and name not in result, 'ambiguous loader SONAME')
        require(path.is_file(), 'resolved loader object must be a file')
        result[name] = str(path)
    require(result, 'empty loader namespace')
    return result


def dependency_edges(namespace, metadata):
    edges = []
    for path, info in metadata.items():
        for name in info['needed']:
            require(name in namespace and namespace[name] in metadata and
                    metadata[namespace[name]]['soname'] == name,
                    'ELF dependency is missing or disagrees with resolved SONAME: ' + name)
            edges.append(dict(source=path, needed=name, target=namespace[name]))
    return edges


def directory_identity(path):
    path = Path(path).resolve(strict=True)
    value = path.stat()
    require(stat.S_ISDIR(value.st_mode), 'loader search directory required')
    return dict(path=str(path), device=value.st_dev, inode=value.st_ino,
                mtime_ns=value.st_mtime_ns, ctime_ns=value.st_ctime_ns,
                entries=sorted(child.name for child in path.iterdir()))


def check_loader_state(receipt):
    require(not Path('/etc/ld.so.preload').exists(), 'system-wide loader injection is not allowed')
    require({name: directory_identity(name) for name in receipt['directories']} == receipt['directories'],
            'native loader search directory changed')
    extensions = Path(receipt['layout']['extensions'])
    require(sorted(str(path.resolve(strict=True)) for path in extensions.glob('*.so')) ==
            receipt['extension_roots'], 'native standard extension inventory changed')
    if 'python_environment' in receipt:
        check_python_environment(receipt['python_environment'])


def native_loader_closure(environment):
    """Freeze stdlib loader dependencies and the complete fixed venv inventory.

    Trace each root once, preserving its RPATH/RUNPATH resolution namespace.
    This augments, and does not replace, the existing vendor/NCU input closure.
    The venv probe executes normal isolated Python startup, without requesting
    any framework workload. Inventory is not a proof of all possible dlopen.
    """
    require(os.getuid() > 0 and 'LD_PRELOAD' not in environment and 'LD_AUDIT' not in environment and
            not any(key.startswith('GLIMMER_') for key in environment), 'clean nonroot loader audit required')
    readelf = Path('/usr/bin/readelf').resolve(strict=True)
    setpriv = bridge.SETPRIV.resolve(strict=True)
    python = protocol.PYTHON.resolve(strict=True)
    paths = {readelf, setpriv, python, Path('/etc/ld.so.cache')}
    for path in (readelf, setpriv, python, Path('/etc/ld.so.cache')):
        bridge.protected_system_file(path)
    recipe = [str(setpriv), '--no-new-privs', str(protocol.PYTHON), '-I', '-S', '-B', '-c', PYTHON_LAYOUT]
    raw_layout = subprocess.check_output(recipe, env=environment, text=True, timeout=10)
    layout = json.loads(raw_layout)
    require(set(layout) == {'stdlib', 'extensions', 'suffix'} and
            type(layout['suffix']) is str and layout['suffix'].startswith('.') and layout['suffix'].endswith('.so'),
            'fixed Python standard-library layout required')
    stdlib, extensions = (Path(layout[name]).resolve(strict=True) for name in ('stdlib', 'extensions'))
    require(stdlib.is_relative_to(Path('/usr/lib')) and extensions == stdlib / 'lib-dynload',
            'only installed system standard extensions are included')
    seeds = sorted(path.resolve(strict=True) for path in extensions.glob('*.so'))
    require(seeds and len(seeds) == len(set(seeds)) and all(path.parent == extensions for path in seeds),
            'complete unique standard extension roots required')
    for path in seeds:
        bridge.protected_system_file(path)
    cache, raw_metadata = {}, {}

    def metadata(path):
        path = str(path)
        if path not in cache:
            raw_metadata[path] = subprocess.check_output([str(readelf), '-W', '-l', '-d', path],
                                                        env=environment, text=True, timeout=10)
            cache[path] = elf_metadata(raw_metadata[path])
        return cache[path]

    selected = metadata(python)['interpreter']
    require(selected is not None, 'native Python must name its actual ELF interpreter')
    loader = Path(selected).resolve(strict=True)
    bridge.protected_system_file(loader)
    paths.add(loader)
    search = environment.get('LD_LIBRARY_PATH', '').split(':')
    require(search and all(value.startswith('/') for value in search), 'explicit absolute target library path required')
    directories = {str(path.resolve(strict=True)) for path in map(Path, search)} | {str(stdlib), str(extensions)}
    receipt = dict(status='native_stdlib_loader_closure_cpu_only', layout=layout, layout_command=recipe,
        layout_stdout=raw_layout, environment=environment, loader=str(loader),
        extension_roots=list(map(str, seeds)), python=str(python), tool_roots=list(map(str, (readelf, setpriv))),
        directories={name: directory_identity(name) for name in sorted(directories)}, roots=[])
    check_loader_state(receipt)
    for root in (python, *seeds, readelf, setpriv):
        root_info = metadata(root)
        if root_info['interpreter']:
            require(Path(root_info['interpreter']).resolve(strict=True) == loader, 'root ELF interpreter differs')
        argv = [str(loader), '--list', str(root)]
        output = subprocess.check_output(argv, env=environment, text=True, stderr=subprocess.STDOUT, timeout=10)
        try:
            namespace = loader_namespace(output, loader,
                no_dependencies=not root_info['needed'] and root_info['interpreter'] is None)
        except ValueError as error:
            raise ValueError(str(root) + ': ' + str(error)) from error
        objects = {str(root), *namespace.values()}
        infos = {name: metadata(name) for name in sorted(objects)}
        for name, path in namespace.items():
            require(infos[path]['soname'] == name, 'loader trace alias disagrees with ELF SONAME')
        edges = dependency_edges(namespace, infos)
        paths.update(map(Path, objects))
        receipt['roots'].append(dict(root=str(root), command=argv, raw_stdout=output,
                                    namespace=namespace, edges=edges, returncode=0))
    receipt.update(elf_metadata=cache, raw_elf_metadata=raw_metadata)
    package_paths, package_receipt = python_environment_inputs(environment, stdlib, extensions)
    paths.update(package_paths)
    receipt['python_environment'] = package_receipt
    check_loader_state(receipt)
    return paths, receipt


def profiler_view(manifest):
    # Reuse the exact vendor environment validator; this alias grants no service.
    return dict(manifest, service_environment=manifest['application_environment'])


def target_environment(inherited, manifest, outer):
    result = bridge.profiler_environment(inherited, profiler_view(manifest), outer)
    result.update(manifest['application_environment'])
    result.update(TMPDIR=str(outer / 'ncu-tmp'), CUDA_CACHE_DISABLE='1')
    return result


def validate_target_environment(actual, manifest, outer):
    baseline = manifest['application_environment']
    require(all(actual.get(key) == value for key, value in baseline.items()) and
            actual.get('TMPDIR') == str(outer / 'ncu-tmp') and actual.get('CUDA_CACHE_DISABLE') == '1' and
            'LD_AUDIT' not in actual and not any(key.startswith(('GLIMMER_', 'PYTHONPATH')) for key in actual),
            'native target environment changed')
    normalized = {key: value for key, value in actual.items() if key not in baseline}
    normalized.update({key: value for key, value in manifest['ncu_supervisor_environment'].items()
                       if key in baseline})
    normalized['LD_LIBRARY_PATH'] = str(bridge.NCU_DIRECTORY)
    bridge.profiler_environment(normalized, profiler_view(manifest), outer)


def file_identity(metadata):
    require(stat.S_ISREG(metadata.st_mode), 'regular frozen input required')
    return dict(device=metadata.st_dev, inode=metadata.st_ino, bytes=metadata.st_size,
                mtime_ns=metadata.st_mtime_ns, ctime_ns=metadata.st_ctime_ns)


def require_target_vendor_mappings(seals, mapped):
    require(type(seals) is dict and len(seals) == len(NCU_INJECTION_ORDER) and
            {Path(path).name for path in seals} == set(NCU_INJECTION_ORDER) and
            type(mapped) is dict and set(mapped) <= set(seals),
            'invalid target-role NCU vendor mappings')
    if set(mapped) != set(seals):
        raise NcuMappingUnavailable(seals, mapped)


def validate_instrumented_target_environment(actual, manifest, outer, *, target_descriptor,
                                             provider, frozen, mapped):
    """Caller authenticates process, FD, complete preflight closure and target maps.

    Provider metadata is compared with that closure; this check does not hash
    the provider independently.
    """
    require(type(target_descriptor) is int and target_descriptor > 2 and type(provider) is str and
            Path(provider).is_absolute() and str(Path(provider).resolve(strict=True)) == provider,
            'exact sealed preload descriptor and canonical provider required')
    require(type(actual) is dict and actual.get('LD_PRELOAD') == f'/proc/self/fd/{target_descriptor}' and
            actual.get('GLIMMER_TRACE_CUBLAS_PROVIDER') == provider,
            'instrumented target preload or provider differs')
    require(type(frozen) is dict and provider in frozen and
            dict(file_identity(Path(provider).stat()), sha256=frozen[provider]['sha256']) == frozen[provider],
            'instrumented provider is not a frozen input')
    clean = dict(actual)
    del clean['LD_PRELOAD']
    del clean['GLIMMER_TRACE_CUBLAS_PROVIDER']
    validate_target_environment(clean, manifest, outer)
    seals = manifest['ncu_injection_seals']
    require(type(seals) is dict and len(seals) == len(NCU_INJECTION_ORDER) and
            {Path(path).name for path in seals} == set(NCU_INJECTION_ORDER) and
            type(mapped) is dict and set(mapped) <= set(seals), 'invalid sealed NCU mappings')
    tracker = actual.get('NVIDIA-PROCESS-TRACKING-CONFIGURATION')
    require(type(tracker) is str, 'target tracker required')
    fields = ('not-launcher-preload-prepend', 'preload-update-preload-prepend',
              'launcher-preload-prepend')
    lines = [line for line in tracker.splitlines() if '-preload-prepend:' in line]
    require(len(lines) == len(fields), 'exact NCU tracker preload fields required')
    ordered = [next(path for path in seals if Path(path).name == name) for name in NCU_INJECTION_ORDER]
    for field, expected_paths in zip(fields, (ordered[:1], ordered[1:2], ordered[2:])):
        entries = [line[len(field) + 2:] for line in lines if line.startswith(field + ': ')]
        require(len(entries) == 1, 'missing or duplicate NCU tracker preload field')
        paths = [str(Path(path).resolve(strict=True)) for path in entries[0].split(':')]
        require(paths == expected_paths, 'NCU tracker preload identity, grouping or order differs')
    injection = verified_preload(None, None, seals, mapped)
    require_target_vendor_mappings(seals, mapped)
    return injection


def mapping_inputs(inputs):
    result = {}
    for value in inputs.values():
        path = Path(value['target'])
        require(path == path.resolve(strict=True), 'canonical mapping input required')
        row = dict(file_identity(path.stat()), sha256=value['sha256'])
        require(row['bytes'] == value['bytes'] and
                (str(path) not in result or result[str(path)] == row), 'ambiguous mapping input identity')
        result[str(path)] = row
    return result


class NativeMaps:
    """Observed executable files must be members of the frozen input closure."""
    def __init__(self, inputs, frozen):
        self.seals, self.observed = {}, {}
        for record in inputs.values():
            target, sha = record['target'], record['sha256']
            require(target not in self.seals or self.seals[target] == sha, 'ambiguous native input seal')
            self.seals[target] = sha
        require(set(frozen) == set(self.seals) and all(
            value['sha256'] == self.seals[name] for name, value in frozen.items()),
            'complete preflight mapped input inventory required')
        self.frozen = frozen

    def observe(self, pid, *, sealed_preload=None, forbidden_source=None):
        # A caller opting in must first authenticate the target's parent,
        # command, credentials and capabilities; this audit does not do so.
        # The caller owns the borrowed descriptor and must keep it alive.
        preload = None
        if sealed_preload is not None:
            bridge.exact_fields(forbidden_source, ('path', 'device', 'inode'))
            source_name = forbidden_source['path']
            require(type(source_name) is str and Path(source_name).is_absolute() and
                    os.path.normpath(source_name) == source_name and
                    type(forbidden_source['device']) is int and forbidden_source['device'] >= 0 and
                    type(forbidden_source['inode']) is int and forbidden_source['inode'] > 0,
                    'exact forbidden source identity required')
            bridge.exact_fields(sealed_preload, ('descriptor', 'expected', 'identity', 'group',
                                                  'target_descriptor'))
            bridge.validate_identity(sealed_preload['identity'])
            require(sealed_preload['identity']['pid'] == pid, 'preload target PID differs')
            preload = bridge.validate_target_preload_mapping(**sealed_preload)
        else:
            require(forbidden_source is None, 'forbidden source requires sealed preload')
        result = {}
        preload_permissions = []
        content = Path(f'/proc/{pid}/maps').read_text()
        if not content:
            raise ProcessLookupError(errno.ESRCH, 'native mappings disappeared', f'/proc/{pid}/maps')
        for line in content.splitlines():
            fields = line.split(maxsplit=5)
            require(len(fields) >= 5, 'malformed native mapping')
            name = fields[5] if len(fields) == 6 else ''
            if 'x' not in fields[1] and '.so' not in name:
                continue
            if name in ('[vdso]', '[vsyscall]'):
                continue
            if preload is not None and name == bridge.TARGET_PRELOAD_MAP_NAME:
                device = fields[3].split(':')
                require(len(device) == 2 and fields[4].isdecimal() and
                        (int(device[0], 16), int(device[1], 16), int(fields[4])) ==
                        (os.major(preload['device']), os.minor(preload['device']), preload['inode']) and
                        not ('w' in fields[1] and 'x' in fields[1]),
                        'native sealed preload mapping differs')
                preload_permissions.append(fields[1])
                continue
            require(name.startswith('/') and not name.startswith('/memfd:') and
                    not name.endswith(' (deleted)'), 'unknown native executable mapping')
            device = fields[3].split(':')
            require(len(device) == 2 and fields[4].isdecimal(), 'invalid native mapping identity')
            if preload is not None:
                require((int(device[0], 16), int(device[1], 16), int(fields[4])) !=
                        (os.major(forbidden_source['device']), os.minor(forbidden_source['device']),
                         forbidden_source['inode']), 'native diagnostic source mapping forbidden')
            path = Path(name).resolve(strict=True)
            metadata = path.stat()
            if preload is not None:
                require(name != source_name and str(path) != source_name and
                        (metadata.st_dev, metadata.st_ino) !=
                        (forbidden_source['device'], forbidden_source['inode']),
                        'native diagnostic source mapping forbidden')
            require((os.major(metadata.st_dev), os.minor(metadata.st_dev), metadata.st_ino) ==
                    (int(device[0], 16), int(device[1], 16), int(fields[4])) and str(path) in self.seals,
                    'native mapped identity is not a frozen input: ' + name)
            value = dict(file_identity(metadata), sha256=self.seals[str(path)])
            require(value == self.frozen[str(path)], 'native mapped file changed since full preflight hash')
            require(str(path) not in self.observed or self.observed[str(path)] == value, 'native map identity drift')
            result[str(path)] = self.observed[str(path)] = value
        require(str(protocol.PYTHON.resolve(strict=True)) in result, 'actual native Python mapping absent')
        observation = dict(monotonic_ns=time.monotonic_ns(), mappings=result)
        if preload is not None:
            require(preload_permissions == preload['mapping_permissions'] and
                    preload == bridge.validate_target_preload_mapping(**sealed_preload),
                    'native sealed preload changed during closure inspection')
            observation['sealed_preload'] = preload
        return observation


class NativeTarget:
    """Actual parent owns a fixed bootstrap, its single release and waitpid."""
    def __init__(self, outer, manifest, directory, role, log_fd, inherited, inputs, frozen, *,
                 target_mode=False, sealed_shim=None):
        require(type(target_mode) is bool and role in protocol.ROLES and
                ((sealed_shim is not None) == (target_mode and role == 'low')),
                'sealed shim is limited to the matched low target')
        self.outer, self.manifest, self.directory, self.role = outer, manifest, directory, role
        self.target_mode = target_mode
        self.group = cgroup(os.getpid())
        commands = protocol.target_application_commands if target_mode else protocol.application_commands
        self.command = commands(manifest, directory / 'native')[role]
        self.environment = dict(target_environment(inherited, manifest, outer))
        self.sealed_shim = None
        retained = bridge.tracker_descriptors(inherited, outer / 'ncu-tmp', 0)
        if sealed_shim is not None:
            bridge.exact_fields(sealed_shim, ('descriptor', 'expected', 'provider', 'forbidden_source'))
            descriptor = sealed_shim['descriptor']
            bridge.validate_snapshot(descriptor, sealed_shim['expected'])
            source = sealed_shim['forbidden_source']
            bridge.exact_fields(source, ('path', 'device', 'inode'))
            source_path = source['path']
            provider = sealed_shim['provider']
            require(descriptor not in retained and descriptor != log_fd and
                    os.readlink(f'/proc/self/fd/{descriptor}') == bridge.TARGET_PRELOAD_MAP_NAME and
                    stat.S_IMODE(os.fstat(descriptor).st_mode) == 0o555 and
                    bridge.fcntl.fcntl(descriptor, bridge.fcntl.F_GET_SEALS) == bridge.SNAPSHOT_SEALS and
                    type(provider) is str and provider in frozen and
                    str(Path(provider).resolve(strict=True)) == provider and
                    dict(file_identity(Path(provider).stat()), sha256=frozen[provider]['sha256']) == frozen[provider] and
                    type(source_path) is str and source_path in frozen and
                    str(Path(source_path).resolve(strict=True)) == source_path and
                    frozen[source_path]['sha256'] == sealed_shim['expected'] and
                    dict(file_identity(Path(source_path).stat()), sha256=sealed_shim['expected']) ==
                    frozen[source_path] and (source['device'], source['inode']) ==
                    (frozen[source_path]['device'], frozen[source_path]['inode']),
                    'distinct sealed descriptor and frozen provider/source required')
            self.sealed_shim = MappingProxyType(dict(sealed_shim, forbidden_source=MappingProxyType(dict(source))))
            self.environment['LD_PRELOAD'] = f'/proc/self/fd/{descriptor}'
            self.environment['GLIMMER_TRACE_CUBLAS_PROVIDER'] = provider
            retained.add(descriptor)
        fixed = [*bridge.drop_command(manifest['uid'], manifest['gid'], service=True), '/usr/bin/env', '-i',
                 *[key + '=' + value for key, value in sorted(self.environment.items())], *self.command]
        self.bootstrap = ['/usr/bin/python3', '-I', '-S', '-B', '-c', BOOTSTRAP, *fixed]
        self.process = subprocess.Popen(self.bootstrap, cwd=PROJECT_ROOT, env=inherited, stdin=subprocess.PIPE,
            stdout=log_fd, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=tuple(sorted(retained)))
        self.identity, self.pidfd = None, None
        self.receipt, self.released = None, False
        try:
            self.identity = bridge.process_identity(self.process.pid)
            self.pidfd = os.pidfd_open(self.process.pid)
            self.audit = NativeMaps(inputs, frozen)
            self.record = dict(identity=self.identity, role=role, bootstrap=self.bootstrap, command=self.command,
                               released=False, status='started', samples=0, observed_mappings={})
            self.verify_bootstrap()
        except BaseException:
            # This is our unreaped direct child, even if initial /proc or pidfd
            # acquisition failed. Popen polls before signalling and reaps it.
            try:
                self.process.kill()
                self.process.wait(timeout=5)
            finally:
                if self.process.stdin and not self.process.stdin.closed:
                    self.process.stdin.close()
                if self.pidfd is not None:
                    os.close(self.pidfd)
            raise

    def poll(self):
        if self.receipt is None:
            pid, status = os.waitpid(self.identity['pid'], os.WNOHANG)
            require(pid in (0, self.identity['pid']), 'unexpected native child waitpid')
            if pid:
                self.receipt = bridge.wait_receipt(pid, self.identity['start'], status)
                self.process.returncode = self.receipt['returncode']
                self.record.update(wait=self.receipt, status='retired')
        return self.receipt

    def verify_bootstrap(self):
        bridge.live_identity(self.identity, self.group)
        pid = self.identity['pid']
        require(relay.parent(pid) == os.getpid() and os.getsid(pid) == os.getpgid(pid) == pid and
                relay.command_line(pid) == self.bootstrap and
                Path(f'/proc/{pid}/exe').samefile('/usr/bin/python3'), 'fixed native bootstrap identity differs')
        fields = dict(line.split(':', 1) for line in Path(f'/proc/{pid}/status').read_text().splitlines())
        require(fields['Uid'].split() == ['0'] * 4, 'bootstrap is not root launch plumbing')

    def release(self):
        require(not self.released and self.poll() is None, 'native target release is single-use')
        self.verify_bootstrap()
        self.process.stdin.write(b'1')
        self.process.stdin.flush()
        self.process.stdin.close()
        self.released = self.record['released'] = True
        until = time.monotonic() + 10
        missing = None
        while time.monotonic() < until:
            require(self.poll() is None, 'native target exited before final exec')
            if relay.command_line(self.identity['pid']) == self.command:
                try:
                    self.verify()
                    return
                except NcuMappingUnavailable as error:
                    if not self.target_mode:
                        raise
                    missing = error
            time.sleep(.01)
        if missing is not None:
            raise TimeoutError(f'native target NCU mappings did not stabilize: {missing}') from missing
        raise TimeoutError('native target did not reach its fixed final exec')

    def verify(self):
        require(self.released, 'released native target required')
        if self.poll() is not None:
            raise ProcessLookupError(errno.ESRCH, 'native target already retired')
        pid = self.identity['pid']
        relay.authenticate_process(self.identity, self.group, self.manifest['uid'], self.manifest['gid'],
            capability=True, executable=protocol.PYTHON)
        actual_command = relay.command_line(pid)
        if not actual_command:
            raise ProcessLookupError(errno.ESRCH, 'native command retired', f'/proc/{pid}/cmdline')
        require(actual_command == self.command, 'native target command changed')
        require(relay.parent(pid) == os.getpid() and os.getsid(pid) == os.getpgid(pid) == pid and
                Path(f'/proc/{pid}/cwd').resolve(strict=True) == PROJECT_ROOT, 'native target owner/cwd changed')
        for name in ('1', '2'):
            require(Path(f'/proc/{pid}/fd/{name}').samefile(self.directory / 'native' / (self.role + '.log')),
                    'native target output descriptor changed')
        for task in Path(f'/proc/{pid}/task').iterdir():
            try:
                children = (task / 'children').read_text()
            except FileNotFoundError:
                # A thread may retire while its process remains live. Only a
                # disappeared exact task entry permits this narrow exception.
                require(not task.exists(), 'live native task children became unreadable')
                bridge.live_identity(self.identity, self.group)
                self.record.setdefault('retired_thread_reads', []).append(str(task))
                continue
            require(not children.strip(), 'native target spawned an unexpected process')
        actual = relay.environment(pid, ncu_seals=self.manifest['ncu_injection_seals'])
        if self.sealed_shim is None:
            validate_target_environment(actual, self.manifest, self.outer)
            vendor_mappings = proc_mapped_paths(pid)
            injection = verified_preload(actual.get('LD_PRELOAD'), None,
                                         self.manifest['ncu_injection_seals'], vendor_mappings)
            if self.target_mode:
                require_target_vendor_mappings(self.manifest['ncu_injection_seals'], vendor_mappings)
            maps = self.audit.observe(pid)
        else:
            shim = self.sealed_shim
            vendor_mappings = proc_mapped_paths(pid)
            injection = validate_instrumented_target_environment(
                actual, self.manifest, self.outer, target_descriptor=shim['descriptor'],
                provider=shim['provider'], frozen=self.audit.frozen, mapped=vendor_mappings)
            maps = self.audit.observe(pid, sealed_preload=dict(
                descriptor=shim['descriptor'], expected=shim['expected'], identity=self.identity,
                group=self.group, target_descriptor=shim['descriptor']), forbidden_source=dict(shim['forbidden_source']))
        self.record.setdefault('initial_maps', maps)
        if self.target_mode:
            self.record.setdefault('initial_environment', actual)
            self.record['last_environment'] = actual
        self.record.update(last_maps=maps, observed_mappings=dict(self.audit.observed), injection=injection,
                           samples=self.record['samples'] + 1)
        self.record['status_fields'] = {key: value.strip() for key, value in
            (line.split(':', 1) for line in Path(f'/proc/{pid}/status').read_text().splitlines())
            if key in ('Uid', 'Gid', 'Groups', 'NoNewPrivs', *bridge.CAP_FIELDS)}

    def sample(self):
        if not self.released or self.poll() is not None:
            return
        try:
            self.verify()
        except (FileNotFoundError, ProcessLookupError, NcuMappingUnavailable) as error:
            poller = select.poll()
            poller.register(self.pidfd, select.POLLIN)
            started = time.monotonic_ns()
            require(poller.poll(250) and bridge.terminal(self.pidfd) and self.poll() is not None,
                    'native verification failed without exact terminal confirmation')
            self.record['terminal_read_race'] = dict(error_type=type(error).__name__,
                error=str(error), started_ns=started, finished_ns=time.monotonic_ns(),
                timeout_ms=250, actual_wait=self.receipt)

    def close(self):
        try:
            if self.poll() is None:
                self.record['failure_cleanup'] = bridge.retire_child(self.identity, self.pidfd)
                self.process.returncode = self.record['failure_cleanup']['returncode']
                self.receipt = self.record['failure_cleanup']
        finally:
            if self.process.stdin and not self.process.stdin.closed:
                self.process.stdin.close()
            os.close(self.pidfd)


class TargetShimOwner:
    """Own the target-only descriptor; caller authenticates manifest and frozen closure."""
    def __init__(self, manifest, frozen):
        spec = manifest['target_shim']
        bridge.exact_fields(spec, ('source', 'provider'))
        source, provider = spec['source'], spec['provider']
        require(type(source) is str and type(provider) is str and source != provider and
                source in frozen and provider in frozen and source in manifest['source_sha256'] and
                Path(source).name == 'libnative_sgemm_range.so' and
                re.fullmatch(r'libcublas\.so(?:\.[0-9]+)*', Path(provider).name) is not None and
                str(Path(source).resolve(strict=True)) == source and
                str(Path(provider).resolve(strict=True)) == provider,
                'reviewed canonical shim and provider required')
        expected = manifest['source_sha256'][source]
        for path in (source, provider):
            require(dict(file_identity(Path(path).stat()), sha256=frozen[path]['sha256']) == frozen[path],
                    'target source or provider differs from frozen preflight')
        require(frozen[source]['sha256'] == expected, 'reviewed shim SHA differs from frozen preflight')
        descriptor = bridge.sealed_snapshot(Path(source), expected)
        try:
            bridge.validate_snapshot(descriptor, expected)
            require(os.readlink(f'/proc/self/fd/{descriptor}') == bridge.TARGET_PRELOAD_MAP_NAME and
                    stat.S_IMODE(os.fstat(descriptor).st_mode) == 0o555 and
                    bridge.fcntl.fcntl(descriptor, bridge.fcntl.F_GET_SEALS) == bridge.SNAPSHOT_SEALS and
                    dict(file_identity(Path(source).stat()), sha256=expected) == frozen[source],
                    'target shim changed during sealing')
        except BaseException:
            os.close(descriptor)
            raise
        self.descriptor, self.expected, self.provider = descriptor, expected, provider
        self.source = dict(path=source, device=frozen[source]['device'], inode=frozen[source]['inode'])
        self.targets, self.finished, self.spawn_in_progress = {}, False, False

    def configuration(self):
        require(not self.finished, 'target shim owner already finalized')
        return dict(descriptor=self.descriptor, expected=self.expected, provider=self.provider,
                    forbidden_source=dict(self.source))

    def register(self, role, target):
        require(not self.finished and role in protocol.ROLES and role not in self.targets,
                'one owned direct child per target role required')
        # Register before further validation so a rejected constructed child is
        # still retired by finish(), including when validation itself raises.
        self.targets[role] = target
        require(target.role == role, 'constructed target role differs')
        bridge.validate_identity(target.identity)
        require(all(other_role == role or other.identity['pid'] != target.identity['pid']
                    for other_role, other in self.targets.items()), 'distinct target process IDs required')
        shim = target.sealed_shim
        require((shim is None and role == 'high') or
                (role == 'low' and shim is not None and shim['descriptor'] == self.descriptor and
                 shim['expected'] == self.expected and shim['provider'] == self.provider and
                 dict(shim['forbidden_source']) == self.source), 'target does not use owned role configuration')

    def spawn(self, role, factory):
        require(not self.finished and not self.spawn_in_progress and role in protocol.ROLES and
                role not in self.targets and callable(factory), 'one target spawn at a time required')
        self.spawn_in_progress = True
        target = factory()
        self.register(role, target)
        self.spawn_in_progress = False
        return target

    def finish(self, *, accepted):
        """Attempt all child cleanup, then close the descriptor only after actual reaps."""
        require(type(accepted) is bool and not self.finished, 'one target-owner finalization required')
        self.finished = True
        errors = []
        if self.spawn_in_progress:
            errors.append('target construction did not complete; descriptor held until bridge exit')
        if accepted and set(self.targets) != set(protocol.ROLES):
            errors.append('successful target completion requires both roles')
        for role, target in self.targets.items():
            try:
                if accepted:
                    receipt = target.poll()
                    require(receipt == bridge.wait_receipt(target.identity['pid'], target.identity['start'], 0) and
                            bridge.terminal(target.pidfd), 'target did not exit successfully before completion')
            except BaseException as error:
                errors.append(role + ' completion: ' + str(error))
            try:
                target.close()
                receipt = target.receipt
                require(receipt is not None and receipt == bridge.wait_receipt(
                    target.identity['pid'], target.identity['start'], receipt['raw_status']),
                    'target cleanup lacks its actual terminal wait receipt')
            except BaseException as error:
                errors.append(role + ' cleanup: ' + str(error))
        if errors:
            return dict(status='failed', cleanup_errors=errors, descriptor_closed=False)
        # A failed close may already have consumed this FD. Do not retry it.
        try:
            os.close(self.descriptor)
        except OSError as error:
            return dict(status='failed', cleanup_errors=['sealed descriptor close: ' + str(error)],
                        descriptor_closed=None)
        return dict(status='retired', cleanup_errors=[], descriptor_closed=True)


def bridge_worker(outer, endpoint, supervisor, *, target_mode=False, expected_manifest_sha256=None):
    """NCU target; target_mode is internal and requires the supervisor's manifest binding."""
    require(type(target_mode) is bool and (target_mode or expected_manifest_sha256 is None),
            'explicit native bridge mode required')
    if target_mode:
        require(type(expected_manifest_sha256) is str and
                re.fullmatch(r'[0-9a-f]{64}', expected_manifest_sha256) is not None and
                digest(outer / 'manifest.json') == expected_manifest_sha256,
                'authenticated target manifest required')
    manifest = read(outer / 'manifest.json')
    if target_mode:
        require(digest(outer / 'manifest.json') == expected_manifest_sha256, 'target manifest changed while reading')
    require(os.getresuid() == (0, 0, 0), 'root native launch bridge required')
    inputs = read(outer / 'inputs-before.json')
    if target_mode:
        require(digest(outer / 'mapping-inputs.json') == manifest['mapping_inputs_sha256'],
                'target bridge frozen closure changed before reading')
    frozen = read(outer / 'mapping-inputs.json')
    require(digest(outer / 'mapping-inputs.json') == manifest['mapping_inputs_sha256'],
            'native bridge preflight mapping identity differs')
    inherited = bridge.profiler_environment(dict(os.environ), profiler_view(manifest), outer)
    deadline = read(outer / 'ncu-go.json')['unit_started_us'] / 1e6 + 780
    connection = protocol.Connection(outer, endpoint, supervisor, 0, deadline, 'bridge')
    targets, directory, owner, pending_reply = {}, None, None, None
    report = dict(status='failed', serial=connection.serial, identity=bridge.process_identity(os.getpid()))
    try:
        if target_mode:
            NativeMaps(inputs, frozen)
            owner = TargetShimOwner(manifest, frozen)
            targets = owner.targets
        while time.monotonic() < deadline:
            for target in targets.values():
                target.sample()
            readable, _, _ = select.select([connection.socket, connection.pidfd], [], [], .2)
            require(connection.pidfd not in readable, 'native root supervisor exited')
            if connection.socket not in readable:
                continue
            request, descriptors = protocol.receive(connection.socket, (supervisor['pid'], 0, 0))
            try:
                require(request.get('serial') == connection.serial, 'native bridge replay serial differs')
                operation = request.get('op')
                reply = dict(op=operation, serial=connection.serial)
                if operation == 'begin':
                    bridge.exact_fields(request, ('op', 'serial', 'directory'))
                    require(directory is None and not descriptors, 'one native replay registration required')
                    directory = protocol.validate_directory(outer, request['directory'], manifest['uid'], connection.serial)
                    report['directory'] = str(directory)
                elif operation == 'spawn':
                    bridge.exact_fields(request, ('op', 'serial', 'role'))
                    role = request['role']
                    require(directory is not None and role in protocol.ROLES and role not in targets and
                            len(descriptors) == 1, 'one exact fresh native target required')
                    protocol.validate_log(descriptors[0], directory, role, manifest['uid'])
                    if target_mode:
                        owner.spawn(role, lambda: NativeTarget(outer, manifest, directory, role, descriptors[0],
                            inherited, inputs, frozen, target_mode=True,
                            sealed_shim=owner.configuration() if role == 'low' else None))
                    else:
                        targets[role] = NativeTarget(outer, manifest, directory, role, descriptors[0], inherited,
                                                     inputs, frozen)
                    reply['identity'] = targets[role].identity
                elif operation in ('release', 'poll', 'verify'):
                    bridge.exact_fields(request, ('op', 'serial', 'role') +
                                        (('released',) if operation == 'verify' else ()))
                    require(not descriptors and request['role'] in targets, 'known native target required')
                    target = targets[request['role']]
                    if operation == 'release':
                        target.release()
                    elif operation == 'poll':
                        reply['receipt'] = target.poll()
                    else:
                        if request['released']:
                            target.verify()
                        elif target.released:
                            target.verify()
                        else:
                            target.verify_bootstrap()
                        reply.update(identity=target.identity, verified=True)
                elif operation == 'complete':
                    bridge.exact_fields(request, ('op', 'serial'))
                    require(not descriptors and set(targets) == set(protocol.ROLES) and all(
                        item.released and item.record['samples'] > 0 and item.poll() is not None and
                        item.receipt['returncode'] == 0 for item in targets.values()), 'both native targets must retire normally')
                    if target_mode:
                        pending_reply = reply
                    else:
                        report['status'] = 'native_bridge_retired'
                        bridge.send_packet(connection.socket, reply)
                    break
                else:
                    raise ValueError('unknown native bridge operation')
                bridge.send_packet(connection.socket, reply)
            finally:
                for fd in descriptors:
                    os.close(fd)
        require(pending_reply is not None if target_mode else report['status'] == 'native_bridge_retired',
                'native bridge deadline exhausted')
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
    finally:
        if target_mode:
            if owner is not None:
                try:
                    cleanup = owner.finish(accepted=pending_reply is not None and 'error' not in report)
                    report['shim_cleanup'] = cleanup
                    if cleanup['status'] == 'retired' and pending_reply is not None and 'error' not in report:
                        report['status'] = 'native_bridge_retired'
                except BaseException as error:
                    report.update(status='failed', cleanup_error=f'{type(error).__name__}: {error}')
            report['targets'] = {role: target.record for role, target in targets.items()}
            try:
                ncu.publish_json(outer / f'bridge-{connection.serial:04d}.json', report)
                if report['status'] == 'native_bridge_retired':
                    bridge.send_packet(connection.socket, pending_reply)
            finally:
                connection.close()
        else:
            errors = []
            for target in targets.values():
                try:
                    target.close()
                except BaseException as error:
                    errors.append(str(error))
            report['targets'] = {role: target.record for role, target in targets.items()}
            if errors:
                report.update(status='failed', cleanup_errors=errors)
            save(outer / f'bridge-{connection.serial:04d}.json', report)
            connection.close()
    return int(report['status'] != 'native_bridge_retired')


def controller(outer, endpoint, supervisor, serial, *, target_mode=False, expected_manifest_sha256=None,
               expected_mapping_sha256=None):
    require(type(target_mode) is bool and
            (target_mode or (expected_manifest_sha256 is None and expected_mapping_sha256 is None)),
            'explicit native controller mode required')
    if target_mode:
        require(type(expected_manifest_sha256) is str and
                re.fullmatch(r'[0-9a-f]{64}', expected_manifest_sha256) is not None and
                type(expected_mapping_sha256) is str and
                re.fullmatch(r'[0-9a-f]{64}', expected_mapping_sha256) is not None and
                digest(outer / 'manifest.json') == expected_manifest_sha256,
                'authenticated target controller manifest required')
    manifest = read(outer / 'manifest.json')
    if target_mode:
        require(digest(outer / 'manifest.json') == expected_manifest_sha256 and
                manifest['mapping_inputs_sha256'] == expected_mapping_sha256,
                'target controller manifest changed while reading')
    bridge.validate_status(Path('/proc/self/status').read_text(), manifest['uid'], manifest['gid'], False)
    require(not proc_mapped_paths(os.getpid()), 'native controller must remain outside NCU ancestry')
    deadline = read(outer / 'ncu-go.json')['unit_started_us'] / 1e6 + 780
    connection = protocol.Connection(outer, endpoint, supervisor, serial, deadline, 'controller')
    report, directory, guard, h, original_onnx = dict(status='failed', serial=serial, **FLAGS), None, None, None, None
    if target_mode:
        report.update(target_mode=True, profiler_diagnostic_only=True)
    try:
        expected = dict(manifest['application_environment'], TMPDIR=str(outer / 'ncu-tmp'), CUDA_CACHE_DISABLE='1')
        require(dict(os.environ) == expected, 'native sibling environment differs')
        guard = candidate.ResourceGuard(outer, 'native')
        common, h = candidate.load_helpers()
        h.startup_reference, h.export = common.application_references(h, manifest)
        require(str(h.startup_reference) == manifest['startup_reference'] and str(h.export) == manifest['export'] and
                str(h.training.ASSET_ROOT) == manifest['assets'], 'native application inputs changed')
        if target_mode:
            require(digest(outer / 'mapping-inputs.json') == manifest['mapping_inputs_sha256'],
                    'target controller frozen closure changed before reading')
            frozen = read(outer / 'mapping-inputs.json')
            require(digest(outer / 'mapping-inputs.json') == manifest['mapping_inputs_sha256'],
                    'target controller frozen closure changed')
            name = str(protocol.ONNX_VALIDATOR)
            require(name in manifest['source_sha256'] and name in frozen and
                    manifest['source_sha256'][name] == frozen[name]['sha256'] and
                    dict(file_identity(Path(name).stat()), sha256=frozen[name]['sha256']) == frozen[name],
                    'reviewed inference validator differs from frozen closure')
            event_start = Path(manifest['export']) == common.STARTUP_INPUTS / 'export'
            profiled_onnx = protocol.instrumented_onnx_validator(frozen[name]['sha256'], event_start=event_start)
            original_onnx = h.validate_onnx

            def validate_target_inference(directory, gpu_uuid, export_root, **options):
                require(Path(export_root) == Path(manifest['export']), 'target inference export changed')
                return profiled_onnx(directory, gpu_uuid, export_root, **options)

            h.validate_onnx = validate_target_inference
            report['inference_validator_sha256'] = frozen[name]['sha256']
            report['inference_event_start'] = event_start
        gpu = manifest['gpu'].split(',')[0].strip()
        require(h.training.clean_environment(gpu) == manifest['application_environment'] and
                h.query('gpu', h.training.GPU_FIELDS) == manifest['gpu'], 'native GPU/environment changed')
        directory = Path(tempfile.mkdtemp(prefix=f'pass-{serial:04d}-', dir=outer / 'passes'))
        report.update(directory=str(directory), controller=bridge.process_identity(os.getpid()))
        with protocol.native_launches(common, h, connection, directory, manifest):
            job = candidate.guarded_control(common, h, directory, 'native', manifest['application_environment'], guard)
        report['job'] = job
        require(job['application_commands'] == protocol.application_commands(manifest, directory / 'native'),
                'raw native application commands changed')
        if target_mode:
            inference = job['inference_numerical']
            require(inference.get('application_instrumented') is True and
                    inference.get('profiler_diagnostic_only') is True and
                    inference.get('diagnostic_only') is True and inference.get('performance_qualified') is False and
                    inference.get('base_validator_sha256') == report['inference_validator_sha256'] and
                    inference.get('instrumented_adapter') == 'flag-v1',
                    'instrumented full inference receipt required')
            report['target_command_plan'] = dict(
                logical_release_commands=job['application_commands'],
                expected_target_commands=protocol.target_application_commands(manifest, directory / 'native'),
                actual_commands_authenticated=False)
        pair.validate_job({}, job)
        report['reparse'] = candidate.reparse(common, h, directory, 'native', job, gpu)
        require(report['reparse'] == pair.ORACLE, 'complete native mathematical oracle required')
        guard.check(force=True)
        report['status'] = 'native_pass_requires_review'
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
    finally:
        if original_onnx is not None:
            h.validate_onnx = original_onnx
        if guard is not None:
            report['resources'] = guard.receipt
        try:
            if directory is not None:
                ncu.publish_json(directory / 'pass.json', report)
            connection.call('complete', directory=str(directory) if directory else None,
                            accepted=report['status'] == 'native_pass_requires_review',
                            report_sha256=digest(directory / 'pass.json') if directory else None)
        finally:
            connection.close()
    return int(report['status'] != 'native_pass_requires_review')


def controller_command(outer, endpoint, supervisor, serial, *, target_manifest_sha256=None,
                       target_mapping_sha256=None):
    command = [str(protocol.PYTHON), '-B', str(ENTRY), '--controller', str(outer), '--endpoint', endpoint,
               '--supervisor', f"{supervisor['pid']}:{supervisor['start']}", '--serial', str(serial)]
    if target_manifest_sha256 is not None or target_mapping_sha256 is not None:
        require(all(type(value) is str and re.fullmatch(r'[0-9a-f]{64}', value) is not None
                    for value in (target_manifest_sha256, target_mapping_sha256)),
                'exact controller target bindings required')
        command += ['--target-mode', '--manifest-sha256', target_manifest_sha256,
                    '--mapping-sha256', target_mapping_sha256]
    return command


class NativeCapture:
    """Root-only relay; each fresh NCU bridge gets one caps0 application controller."""
    def __init__(self, outer, manifest, group, deadline, resources, *, target_manifest_sha256=None,
                 target_mapping_sha256=None):
        self.target_mode = target_manifest_sha256 is not None or target_mapping_sha256 is not None
        self.target_manifest_sha256, self.target_mapping_sha256 = target_manifest_sha256, target_mapping_sha256
        if self.target_mode:
            require(all(type(value) is str and re.fullmatch(r'[0-9a-f]{64}', value) is not None
                        for value in (target_manifest_sha256, target_mapping_sha256)) and
                    digest(outer / 'manifest.json') == target_manifest_sha256 and
                    read(outer / 'manifest.json') == manifest and
                    digest(outer / 'manifest.json') == target_manifest_sha256 and
                    manifest['mapping_inputs_sha256'] == target_mapping_sha256 and
                    digest(outer / 'mapping-inputs.json') == target_mapping_sha256 and
                    protocol.NCU_BINARY == str(ncu.NCU),
                    'supervisor target bindings differ')
        self.outer, self.manifest, self.group, self.deadline = outer, manifest, group, deadline
        self.resources, self.identity = resources, bridge.process_identity(os.getpid())
        self.endpoint = 'glimmer-native-ncu-' + secrets.token_hex(16)
        self.listener = resources.enter_context(socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET))
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
        self.listener.bind('\0' + self.endpoint)
        self.listener.listen(2)
        self.listener.settimeout(1)
        self.code_sha = digest(ENTRY)
        require(self.code_sha == manifest['entrypoint_sha256'], 'native bridge source changed')
        self.code_fd = bridge.sealed_snapshot(ENTRY, self.code_sha)
        resources.callback(os.close, self.code_fd)
        self.sessions, self.samples, self.ncu_identities = [], [], set()
        self.last_sample = 0.

    def target(self):
        command = ['/usr/bin/python3', '-I', '-S', '-B', '-c', bridge.SEALED_LOADER, str(ENTRY),
                   f'/proc/{os.getpid()}/fd/{self.code_fd}', self.code_sha, '--bridge', str(self.outer),
                   '--endpoint', self.endpoint, '--supervisor', f"{self.identity['pid']}:{self.identity['start']}"]
        if self.target_mode:
            command += ['--target-mode', '--manifest-sha256', self.target_manifest_sha256]
        return command

    def sample(self, force=False):
        now = time.monotonic()
        require(now < self.deadline, 'native capture work deadline exceeded')
        if not force and now - self.last_sample < .2:
            return
        self.last_sample = now
        require(shutil.disk_usage(self.outer).free >= pair.DISK_FLOOR, 'native capture disk reserve exhausted')
        memory = memory_receipt(self.group)
        events = dict(line.split() for line in memory['memory.events'].splitlines())
        root = Path('/sys/fs/cgroup' + self.group)
        pids = dict(line.split() for line in (root / 'pids.events').read_text().splitlines())
        require(events and pids and all(int(value) == 0 for value in (*events.values(), *pids.values())) and
                (root / 'pids.max').read_text().strip() == '256' and
                (root / 'memory.swap.current').read_text().strip() == '0', 'native task pressure or swap')
        row = dict(monotonic_ns=time.monotonic_ns(),
            gpu=ncu.check_owned_gpu(self.group, self.manifest['gpu'].split(',')[0].strip()),
            ncu=ncu.observed_ncu_processes(self.group), output_bytes=candidate.check_output(self.outer))
        self.ncu_identities.update((item['pid'], item['start']) for item in row['ncu'])
        if not self.samples or any(row[key] != self.samples[-1][key] for key in ('gpu', 'ncu')):
            self.samples.append(row)

    def receive(self, connection, credentials):
        while True:
            self.sample()
            readable, _, _ = select.select([connection], [], [], .2)
            if readable:
                return protocol.receive(connection, credentials)

    def accept(self):
        while True:
            self.sample()
            if select.select([self.listener], [], [], .2)[0]:
                connection, _ = self.listener.accept()
                self.resources.enter_context(connection)
                connection.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
                connection.settimeout(10)
                peer = struct.unpack('3i', connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
                message, descriptors = self.receive(connection, peer)
                for fd in descriptors:
                    os.close(fd)
                require(not descriptors, 'native hello cannot transfer descriptors')
                return connection, peer, message

    def pass_once(self, connection, peer, hello):
        serial = len(self.sessions) + 1
        helper = bridge.process_identity(peer[0])
        require(peer[1:] == (0, 0) and hello == dict(op='hello', serial=0, kind='bridge', identity=helper) and
                relay.command_line(peer[0]) == self.target() and
                Path(f'/proc/{peer[0]}/exe').samefile('/usr/bin/python3'), 'unexpected native root bridge')
        self.sample(force=True)
        ancestors = relay.ncu_ancestry(helper, self.group, self.ncu_identities)
        bridge.profiler_environment(relay.environment(peer[0], ncu_seals=self.manifest['ncu_injection_seals']),
                                    profiler_view(self.manifest), self.outer)
        helper_fd = os.pidfd_open(peer[0])
        self.resources.callback(os.close, helper_fd)
        bridge.send_packet(connection, dict(op='hello', serial=0, assigned_serial=serial))
        recipe = controller_command(self.outer, self.endpoint, self.identity, serial,
            target_manifest_sha256=self.target_manifest_sha256, target_mapping_sha256=self.target_mapping_sha256)
        environment = dict(self.manifest['application_environment'], TMPDIR=str(self.outer / 'ncu-tmp'),
                           CUDA_CACHE_DISABLE='1')
        log = self.resources.enter_context((self.outer / f'controller-{serial:04d}.log').open('x'))
        process = subprocess.Popen([*bridge.drop_command(self.manifest['uid'], self.manifest['gid'], service=False),
            *recipe], cwd=self.outer, env=environment, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True)
        owner = bridge.process_identity(process.pid)
        controller_socket, controller_peer, greeting = self.accept()
        require(controller_peer == (process.pid, self.manifest['uid'], self.manifest['gid']) and
                greeting == dict(op='hello', serial=serial, kind='controller', identity=owner),
                'unexpected native controller peer')
        relay.authenticate_process(owner, self.group, self.manifest['uid'], self.manifest['gid'],
                                   capability=False, command=recipe, executable=protocol.PYTHON)
        require(relay.parent(process.pid) == os.getpid() and not proc_mapped_paths(process.pid) and
                relay.environment(process.pid) == environment, 'native controller privilege/topology changed')
        bridge.send_packet(controller_socket, dict(op='hello', serial=serial))
        session = dict(serial=serial, helper=helper, helper_ancestors=ancestors, controller=owner,
                       controller_command=recipe, controller_environment=environment, operations=[])
        self.sessions.append(session)
        directory, spawned = None, set()
        while True:
            require(not bridge.terminal(helper_fd), 'native bridge exited before pass acceptance')
            request, fds = self.receive(controller_socket, controller_peer)
            try:
                require(request.get('serial') == serial, 'native controller replay serial changed')
                operation = request.get('op')
                if operation == 'begin':
                    bridge.exact_fields(request, ('op', 'serial', 'directory'))
                    require(directory is None and not fds, 'one native controller pass directory required')
                    directory = protocol.validate_directory(self.outer, request['directory'], self.manifest['uid'], serial)
                elif operation == 'spawn':
                    bridge.exact_fields(request, ('op', 'serial', 'role'))
                    require(directory is not None and request['role'] not in spawned and len(fds) == 1,
                            'one native role launch required')
                    protocol.validate_log(fds[0], directory, request['role'], self.manifest['uid'])
                    spawned.add(request['role'])
                elif operation in ('release', 'poll', 'verify'):
                    bridge.exact_fields(request, ('op', 'serial', 'role') +
                                        (('released',) if operation == 'verify' else ()))
                    require(not fds and request['role'] in spawned and
                            (operation != 'verify' or type(request['released']) is bool), 'unknown native request')
                elif operation == 'complete':
                    bridge.exact_fields(request, ('op', 'serial', 'directory', 'accepted', 'report_sha256'))
                    require(not fds and spawned == set(protocol.ROLES) and request['accepted'] is True and
                            request['directory'] == str(directory) and
                            digest(directory / 'pass.json') == request['report_sha256'], 'native application pass failed')
                    report = read(directory / 'pass.json')
                    require(report['status'] == 'native_pass_requires_review' and report['reparse'] == pair.ORACLE and
                            report['controller'] == owner, 'native complete oracle/controller receipt differs')
                    session.update(directory=str(directory), pass_sha256=request['report_sha256'])
                else:
                    raise ValueError('unapproved native controller operation')
                forwarded = dict(op='complete', serial=serial) if operation == 'complete' else request
                bridge.send_packet(connection, forwarded, fds)
                reply, reply_fds = self.receive(connection, peer)
                for fd in reply_fds:
                    os.close(fd)
                require(not reply_fds and reply.get('op') == operation and reply.get('serial') == serial,
                        'native root bridge reply changed')
                if operation != 'poll':
                    session['operations'].append(dict(request=forwarded, reply=reply))
                bridge.send_packet(controller_socket, reply)
                if operation == 'complete':
                    session['controller_returncode'] = process.wait(timeout=10)
                    require(session['controller_returncode'] == 0, 'native controller did not exit normally')
                    poller = select.poll()
                    poller.register(helper_fd, select.POLLIN)
                    require(poller.poll(10000) and bridge.terminal(helper_fd), 'native root bridge did not retire')
                    path = self.outer / f'bridge-{serial:04d}.json'
                    require(path.stat().st_uid == 0, 'root-owned native bridge report required')
                    receipt = read(path)
                    require(receipt['status'] == 'native_bridge_retired' and receipt['identity'] == helper and
                            receipt['directory'] == str(directory), 'native bridge receipt differs')
                    session.update(bridge_sha256=digest(path), helper_pidfd_exited=True, status='complete')
                    return
            finally:
                for fd in fds:
                    os.close(fd)

    def run(self):
        argv = [str(ncu.NCU), '--config-file', '0', '--target-processes', 'all', '--nvtx',
                '--profile-from-start', 'on', '--kernel-name-base', 'function', '--kernel-name', protocol.KERNEL_FILTER,
                '--replay-mode', 'application', '--app-replay-mode', 'strict', '--app-replay-match', 'all',
                '--kill', 'no', '--clock-control', 'none', '--cache-control', 'none',
                '--metrics', ','.join(protocol.METRICS), '--log-file', str(self.outer / 'ncu.log'),
                '--export', str(self.outer / 'capture'), *self.target()]
        if self.target_mode:
            argv = protocol.target_profiler_command(self.outer, self.target())
        log = self.resources.enter_context((self.outer / 'ncu-supervisor.log').open('x'))
        process = subprocess.Popen(argv, cwd=self.outer, env=self.manifest['ncu_supervisor_environment'],
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            self.sample()
            if select.select([self.listener], [], [], .2)[0]:
                self.pass_once(*self.accept())
        require(process.returncode == 0 and self.sessions, 'native NCU replay failed or missing')
        return dict(returncode=process.returncode, command=argv, sessions=self.sessions, samples=self.samples)


def task_controls(outer):
    group = cgroup(os.getpid())
    require(group.endswith(f'/glimmer-bert-{outer.name}-native.service'), 'exact private native unit required')
    directory = (Path('/sys/fs/cgroup') / group.lstrip('/')).resolve(strict=True)
    require(directory.is_relative_to(Path('/sys/fs/cgroup')) and directory != Path('/sys/fs/cgroup'),
            'native task cannot use cgroup root')
    descriptor = os.open(directory / 'cgroup.kill', os.O_WRONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        actual, expected = os.fstat(descriptor), (directory / 'cgroup.kill').stat()
        require((actual.st_dev, actual.st_ino) == (expected.st_dev, expected.st_ino),
                'native cgroup kill descriptor changed')
    except BaseException:
        os.close(descriptor)
        raise
    return group, directory, descriptor


def root_supervisor(outer):
    report, kill_fd, started = dict(status='failed', **FLAGS), None, False
    previous = os.umask(0o022)
    try:
        require(os.getresuid() == (0, 0, 0) and not os.environ.get('LD_PRELOAD') and
                not os.environ.get('LD_AUDIT'), 'clean root profiler supervisor required')
        resource.setrlimit(resource.RLIMIT_FSIZE, (candidate.FILE_MAX, candidate.FILE_MAX))
        group, directory, kill_fd = task_controls(outer)
        memory_receipt(group)
        ncu.require_retired_descendants(group)
        manifest = read(outer / 'manifest.json')
        report['manifest_sha256'] = digest(outer / 'manifest.json')
        require(manifest['uid'] > 0 and manifest['gid'] >= 0 and manifest['groups'] == [] and
                digest(ncu.NCU) == manifest['ncu_sha256'] and digest(ncu.NCU_EXEC) == manifest['ncu_exec_sha256'],
                'native identity or pinned profiler changed')
        for name, expected in manifest['source_sha256'].items():
            require(digest(name) == expected, 'native reviewed source changed: ' + name)
        deadline = ncu.await_outer_handshake(outer, group, directory)
        started = True
        with ExitStack() as resources:
            capture = NativeCapture(outer, manifest, group, deadline, resources)
            try:
                report['capture'] = capture.run()
            finally:
                report['sessions'] = capture.sessions
                report['samples'] = capture.samples
        ncu.require_retired_descendants(group)
        raw = outer / 'capture.ncu-rep'
        require(raw.is_file() and not raw.is_symlink(), 'native raw counter report missing')
        export = [str(ncu.NCU), '--config-file', '0', '--import', str(raw), '--csv', '--page', 'raw',
                  '--print-units', 'base', '--print-kernel-base', 'function']
        report['export'] = ncu.monitored_command(export, manifest['ncu_supervisor_environment'], outer, group,
            deadline, outer / 'ncu-metrics.csv', manifest['gpu'].split(',')[0].strip())
        ncu.require_retired_descendants(group)
        report['status'] = 'native_counter_capture_requires_review'
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        if started and kill_fd is not None:
            save(outer / 'ncu-supervisor.json', report)
            os.write(kill_fd, b'1')
    finally:
        if kill_fd is not None:
            os.close(kill_fd)
        save(outer / 'ncu-supervisor.json', report)
        os.umask(previous)
    return int(report['status'] != 'native_counter_capture_requires_review')


def launch_unit(outer, gpu):
    unit = f'glimmer-bert-{outer.name}-native.service'
    description = 'Glimmer native spatial capture ' + outer.name
    require(ncu.unit_properties(unit)['LoadState'] == 'not-found', 'never reuse a native task unit')
    argv = ['systemd-run', '--user', '--unit=' + unit, '--description=' + description, '--wait', '--pipe',
            '--property=MemoryMax=' + str(candidate.MEMORY_MAX), '--property=MemorySwapMax=0',
            '--property=OOMPolicy=kill', '--property=RuntimeMaxSec=800s', '--property=TimeoutStopSec=5s',
            '--property=KillMode=control-group', '--property=TasksMax=256',
            '--property=LimitFSIZE=' + str(candidate.FILE_MAX), '/usr/bin/env', '-u', 'LD_PRELOAD',
            '-u', 'LD_AUDIT', '-u', 'LD_LIBRARY_PATH', '/usr/bin/sudo', '-n', '/usr/bin/python3', '-B',
            str(ENTRY), '--root-supervisor', str(outer)]
    result = dict(unit=unit, command=argv, returncode=None, unit_removed=False)
    process, owned = None, False
    try:
        with (outer / 'unit.log').open('x') as log:
            process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
            until = time.monotonic() + 35
            while not (outer / 'ncu-ready.json').exists():
                properties = ncu.unit_properties(unit)
                owned = owned or (properties['LoadState'] != 'not-found' and properties['Description'] == description)
                require(process.poll() is None and time.monotonic() < until, 'native unit startup failed')
                time.sleep(.05)
            ready = read(outer / 'ncu-ready.json')
            bridge.exact_fields(ready, ('pid', 'start', 'group', 'cgroup_device', 'cgroup_inode'))
            require(all(type(ready[key]) is int and ready[key] > 0
                        for key in ('pid', 'start', 'cgroup_device', 'cgroup_inode')), 'native handshake identity invalid')
            properties = ncu.unit_properties(unit)
            owned = properties['LoadState'] != 'not-found' and properties['Description'] == description
            require(owned and properties['ActiveState'] == 'active' and properties['ControlGroup'] == ready['group'] and
                    properties['ExecMainStartTimestampMonotonic'].isdecimal(), 'native systemd identity changed')
            bridge.live_identity({key: ready[key] for key in ('pid', 'start')}, ready['group'])
            status = dict(line.split(':', 1) for line in Path(f"/proc/{ready['pid']}/status").read_text().splitlines())
            require(status['Uid'].split() == ['0'] * 4 and relay.command_line(ready['pid']) ==
                    ['/usr/bin/python3', '-B', str(ENTRY), '--root-supervisor', str(outer)],
                    'unexpected native root supervisor')
            metadata = (Path('/sys/fs/cgroup') / ready['group'].lstrip('/')).stat()
            require((metadata.st_dev, metadata.st_ino) == (ready['cgroup_device'], ready['cgroup_inode']),
                    'native cgroup identity changed')
            memory_receipt(ready['group'])
            require(not ncu.check_owned_gpu(ready['group'], gpu.split(',')[0].strip()), 'native GPU must be idle')
            started = int(properties['ExecMainStartTimestampMonotonic'])
            require(started > 0, 'native unit start required')
            ncu.publish_json(outer / 'ncu-go.json', dict(ready, unit_started_us=started))
            result.update(handshake=ready, unit_started_us=started)
            result['returncode'] = process.wait(timeout=max(1., started / 1e6 + 820 - time.monotonic()))
    except BaseException as error:
        result['error'] = f'{type(error).__name__}: {error}'
    finally:
        try:
            properties = ncu.unit_properties(unit)
            if properties['LoadState'] != 'not-found':
                require(owned and properties['Description'] == description, 'refuse to stop unowned native unit')
                command(['systemctl', '--user', 'stop', unit], timeout=15)
                if ncu.unit_properties(unit)['ActiveState'] == 'failed':
                    command(['systemctl', '--user', 'reset-failed', unit], timeout=5)
            result['unit_removed'] = ncu.unit_properties(unit)['LoadState'] == 'not-found'
            if process is not None and process.poll() is None:
                process.wait(timeout=10)
        except BaseException as error:
            result['cleanup_error'] = f'{type(error).__name__}: {error}'
        save(outer / 'unit.json', result)
    require(result['returncode'] == 0 and result['unit_removed'] and 'error' not in result and
            'cleanup_error' not in result, 'native bounded unit did not retire normally')
    return result


def references(plan):
    root = Path(plan['activity']).resolve(strict=True)
    require(digest(root / 'report.json') == plan['activity_report_sha256'] and
            digest(root / 'review.json') == plan['activity_review_sha256'], 'accepted activity binding changed')
    review = read(root / 'review.json')
    require(review['status'] == 'host_pair_aggregate_activity_increase_accepted' and
            review['report_sha256'] == plan['activity_report_sha256'], 'accepted current activity required')
    previous = read(root / 'plan.json')
    numerical_plan, reference, original, paths = activity.references(previous, artifacts=True)
    require(digest(plan['authorization']) == plan['authorization_sha256'] and
            read(plan['authorization'])['status'] == 'explicit_native_target_authority_granted_plan_required' and
            plan['authorization_deadline_utc'] == read(plan['authorization'])['authorization_deadline_utc'],
            'explicit native target authority required')
    own = Path(reference['variant_attempts']['o0']) / 'native'
    job = read(own / 'attempt.json')
    pair.validate_job({}, job)
    paths.update(root / name for name in ('plan.json', 'report.json', 'review.json', 'analysis.json'))
    paths.update((Path(plan['authorization']), ENTRY, Path(protocol.__file__).resolve()))
    # Retain the accepted installed NCU/driver loader closure without rebuilding.
    counter = Path(previous['targets']['o0']['attempt'])
    paths.update(Path(name) for name in read(counter / 'inputs-before.json'))
    return numerical_plan, reference, original, paths, own, job


def reviewed(plan):
    require(digest(plan['launch_review']) == plan['launch_review_sha256'], 'native launch review changed')
    review = read(plan['launch_review'])
    require(review['status'] == 'native_spatial_capture_ready' and
            review['authorization_sha256'] == plan['authorization_sha256'], 'native launch is not reviewed')
    for name, expected in review['source_sha256'].items():
        require(digest(name) == expected, 'reviewed native source changed: ' + name)
    return review['source_sha256']


def prepare(brief):
    note = read(brief)
    root = Path(note['current_activity']).parent
    plan = dict(status='prepared_requires_launch_review', activity=str(root),
        activity_report_sha256=digest(root / 'report.json'), activity_review_sha256=digest(root / 'review.json'),
        authorization=note['authorization'], authorization_sha256=note['authorization_sha256'],
        authorization_deadline_utc=note['authorization_deadline_utc'], brief=str(brief), brief_sha256=digest(brief),
        scope='whole fixed concurrent invocation including warmup; no measured-window spatial claim', **FLAGS)
    references(plan)
    path = brief.with_name('plan.json')
    with path.open('x') as stream:
        json.dump(plan, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(str(path), flush=True)


def review_target_session(manifest, directory, report, root_report, frozen, *, event_start):
    """Check one target replay; caller must authenticate both reports and all artifacts.

    This is not a capture review or a qualification decision. In particular,
    it does not authenticate the source of the reference descriptor sequence.
    """
    require(type(event_start) is bool and report.get('status') == 'native_pass_requires_review' and
            report.get('target_mode') is True and report.get('profiler_diagnostic_only') is True and
            report.get('inference_event_start') is event_start and report.get('reparse') == pair.ORACLE,
            'complete target controller receipt required')
    job = report['job']
    pair.validate_job({}, job)
    inference = job['inference_numerical']
    validator = str(protocol.ONNX_VALIDATOR)
    require(validator in manifest['source_sha256'] and validator in frozen and
            report.get('inference_validator_sha256') == manifest['source_sha256'][validator] ==
            frozen[validator]['sha256'] == inference.get('base_validator_sha256') and
            inference.get('application_instrumented') is True and
            inference.get('profiler_diagnostic_only') is True and inference.get('diagnostic_only') is True and
            inference.get('performance_qualified') is False and inference.get('instrumented_adapter') == 'flag-v1',
            'complete bound instrumented inference receipt required')
    require(root_report.get('status') == 'native_bridge_retired' and
            root_report.get('directory') == str(directory) and
            root_report.get('shim_cleanup') == dict(status='retired', cleanup_errors=[], descriptor_closed=True),
            'target bridge and sealed descriptor must retire cleanly')
    records = root_report['targets']
    binding = protocol.target_command_binding(manifest, directory / 'native', job, records)
    require(report.get('target_command_plan') == dict(
        logical_release_commands=binding['logical_release_commands'],
        expected_target_commands=binding['root_recorded_target_commands'], actual_commands_authenticated=False),
        'target controller command plan differs from authenticated root record')
    spec = manifest['target_shim']
    bridge.exact_fields(spec, ('source', 'provider'))
    source, provider = spec['source'], spec['provider']
    require(source in frozen and provider in frozen and source in manifest['source_sha256'] and
            manifest['source_sha256'][source] == frozen[source]['sha256'], 'frozen target shim inputs required')
    seals = manifest['ncu_injection_seals']
    require(type(seals) is dict and len(seals) == len(NCU_INJECTION_ORDER) and
            {Path(name).name for name in seals} == set(NCU_INJECTION_ORDER) and
            all(name in frozen and frozen[name]['sha256'] == digest for name, digest in seals.items()),
            'four pinned vendor libraries required')
    roles = {}
    for role in protocol.ROLES:
        record = records[role]
        target_identity = record['identity']
        require(record.get('status') == 'retired' and 'failure_cleanup' not in record and
                record.get('wait') == bridge.wait_receipt(target_identity['pid'], target_identity['start'], 0) and
                record.get('injection', {}).get('mapped_ncu_injections') == sorted(seals),
                'target process, wait or NCU mapping receipt differs')
        status = '\n'.join(key + ':\t' + value for key, value in record['status_fields'].items())
        bridge.validate_status(status, manifest['uid'], manifest['gid'], True)
        observed = record['observed_mappings']
        require(type(observed) is dict and observed and all(frozen.get(name) == value
                for name, value in observed.items()) and source not in observed,
                'target observed mappings differ from frozen closure')
        first, last = record['initial_maps'], record['last_maps']
        require(type(first.get('monotonic_ns')) is int and type(last.get('monotonic_ns')) is int and
                first['monotonic_ns'] <= last['monotonic_ns'] and all(
                    type(sample.get('mappings')) is dict and sample['mappings'] and all(
                        observed.get(name) == value for name, value in sample['mappings'].items())
                    and set(seals) <= set(sample['mappings'])
                    for sample in (first, last)), 'target mapping samples differ from observed closure')
        environments = (record['initial_environment'], record['last_environment'])
        require(all(type(value) is dict for value in environments), 'checked target environments required')
        if role == 'low':
            require(provider in observed and all('sealed_preload' in sample for sample in (first, last)),
                    'low target must map its reviewed provider and sealed preload')
            initial, final = first['sealed_preload'], last['sealed_preload']
            fields = ('pid', 'start', 'target_descriptor', 'device', 'inode', 'bytes', 'sha256', 'mode', 'seals')
            require(all(initial.get(key) == final.get(key) for key in fields) and
                    (initial.get('pid'), initial.get('start')) ==
                    (target_identity['pid'], target_identity['start']) and
                    type(initial.get('target_descriptor')) is int and initial['target_descriptor'] > 2 and
                    type(initial.get('device')) is int and initial['device'] >= 0 and
                    type(initial.get('inode')) is int and initial['inode'] > 0 and
                    initial.get('bytes') == frozen[source]['bytes'] and
                    initial.get('sha256') == frozen[source]['sha256'] and initial.get('mode') == 0o555 and
                    initial.get('seals') == bridge.SNAPSHOT_SEALS,
                    'sealed target preload identity differs')
            for sample in (initial, final):
                permissions = sample.get('mapping_permissions')
                require(type(permissions) is list and permissions and
                        type(sample.get('mapping_count')) is int and sample['mapping_count'] == len(permissions) and
                        all(type(value) is str and re.fullmatch(r'[r-][w-][x-][ps]', value) is not None and
                            not ('w' in value and 'x' in value) for value in permissions) and
                        any('x' in value for value in permissions), 'sealed executable mapping receipt differs')
            require(all(value.get('LD_PRELOAD') == f"/proc/self/fd/{initial['target_descriptor']}" and
                        value.get('GLIMMER_TRACE_CUBLAS_PROVIDER') == provider for value in environments),
                    'low target checked environment differs')
        else:
            require(all('sealed_preload' not in sample for sample in (first, last)) and all(
                'GLIMMER_TRACE_CUBLAS_PROVIDER' not in value and
                '/proc/self/fd/' not in value.get('LD_PRELOAD', '') for value in environments),
                'high target must not inherit the diagnostic shim')
        roles[target_identity['pid']] = role
    require(len(roles) == len(protocol.ROLES), 'target replay roles are not distinct')
    return dict(roles=roles, application_commands=binding['root_recorded_target_commands'],
                status='target_session_requires_authenticated_root_and_artifacts',
                authenticated=False, full_device_qualified=False, performance_qualified=False)


def review_capture(outer):
    manifest, supervisor = read(outer / 'manifest.json'), read(outer / 'ncu-supervisor.json')
    require(supervisor['status'] == 'native_counter_capture_requires_review' and
            supervisor['manifest_sha256'] == digest(outer / 'manifest.json') and
            supervisor['capture']['returncode'] == supervisor['export']['returncode'] == 0,
            'normal complete native profiler and export required')
    sessions = supervisor['capture']['sessions']
    require(sessions and [row['serial'] for row in sessions] == list(range(1, len(sessions) + 1)),
            'native complete ordered replay sessions required')
    common, h = candidate.load_helpers()
    h.startup_reference, h.export = common.application_references(h, manifest)
    inputs = read(outer / 'inputs-before.json')
    loader = read(outer / 'native-loader.json')
    require(digest(outer / 'native-loader.json') == manifest['loader_closure_sha256'] and
            loader['environment'] == manifest['application_environment'], 'native loader closure binding changed')
    check_loader_state(loader)
    frozen = read(outer / 'mapping-inputs.json')
    NativeMaps(inputs, frozen)
    require(digest(outer / 'mapping-inputs.json') == manifest['mapping_inputs_sha256'],
            'native preflight mapping inventory changed')
    directories, roles_by_pass = [], []
    for session in sessions:
        directory = Path(session['directory'])
        protocol.validate_directory(outer, str(directory), manifest['uid'], session['serial'])
        require(directory.parent == outer / 'passes' and digest(directory / 'pass.json') == session['pass_sha256'] and
                session['controller_returncode'] == 0 and session['helper_pidfd_exited'] is True and
                session['status'] == 'complete', 'native replay controller/bridge failed')
        report = read(directory / 'pass.json')
        require(report['status'] == 'native_pass_requires_review' and report['reparse'] == pair.ORACLE and
                report['controller'] == session['controller'], 'native replay oracle or controller differs')
        job = report['job']
        pair.validate_job({}, job)
        require(job['application_commands'] == protocol.application_commands(manifest, directory / 'native') and
                job['application_environment'] == manifest['application_environment'], 'native logical recipe differs')
        # Full raw tensor reparse already ran in every caps0 controller. Repeat
        # output identity and unchanged oracle receipts here, not GPU work.
        require(read(directory / 'native/attempt.json') == job, 'native raw job differs')
        for name, sha in job['artifact_identities'].items():
            require(digest(directory / 'native' / name) == sha, 'native replay artifact changed')
        root_report = outer / f"bridge-{session['serial']:04d}.json"
        require(root_report.stat().st_uid == 0 and digest(root_report) == session['bridge_sha256'],
                'native bridge evidence changed')
        target = read(root_report)
        require(target['status'] == 'native_bridge_retired' and target['identity'] == session['helper'] and
                target['directory'] == str(directory) and
                set(target['targets']) == set(protocol.ROLES), 'native bridge target inventory incomplete')
        spawns = [row['request']['role'] for row in session['operations'] if row['request']['op'] == 'spawn']
        releases = [row['request']['role'] for row in session['operations'] if row['request']['op'] == 'release']
        require(spawns == releases == list(protocol.ROLES), 'fixed native role creation/release order differs')
        roles = {}
        for role, record in target['targets'].items():
            require(record['identity'] == job['application_identities'][role] and record['released'] is True and
                    record['samples'] > 0 and record['command'] == job['application_commands'][role] and
                    record['wait'] == bridge.wait_receipt(record['identity']['pid'], record['identity']['start'], 0) and
                    'failure_cleanup' not in record and record['status'] == 'retired', 'native target lifecycle differs')
            status = '\n'.join(key + ':\t' + value for key, value in record['status_fields'].items())
            bridge.validate_status(status, manifest['uid'], manifest['gid'], True)
            for name, value in record['observed_mappings'].items():
                require(frozen.get(name) == value, 'native observed mapping is not frozen')
            require(record['initial_maps']['monotonic_ns'] <= record['last_maps']['monotonic_ns'] and
                    record['observed_mappings'], 'native mapping sample inventory missing')
            roles[record['identity']['pid']] = role
        require(len(roles) == 2, 'native roles reused a process')
        roles_by_pass.append(roles)
        directories.append(directory)
    require(set((outer / 'passes').iterdir()) == set(directories), 'missing or extra native replay directory')
    raw = (outer / 'ncu-metrics.csv').read_text()
    log = (outer / 'ncu.log').read_text()
    exported, inventory = protocol.replay_inventory(raw, log, roles_by_pass)
    full = [row for row in inventory if row['all_80_useful']]
    require(full, 'native invocation has no authenticated all80 useful-compute launch')
    return dict(status='native_whole_invocation_spatial_observation_requires_review',
                participating_sms=80, native_spatial_participation_verified=True,
                application_phase_verified=False, inventory=inventory, all80_launch_count=len(full),
                role_launch_counts=dict(Counter(row['role'] for row in inventory)),
                replay_count=len(sessions), exported_replay_serial=exported,
                replay_role_binding='authenticated fixed process order with vendor strict/all per-process matching',
                raw_report_sha256=digest(outer / 'capture.ncu-rep'),
                raw_metrics_sha256=digest(outer / 'ncu-metrics.csv'),
                scope='complete selected SGEMM family across fixed concurrent invocation including warmup',
                full_device_qualified=False, saturation_qualified=False, performance_qualified=False)


def collect(plan_path):
    plan = read(plan_path)
    require(not plan_path.with_name('launch.json').exists(), 'native one-shot launch already claimed')
    activity.numerical.authorization(plan, 1100)
    sources = reviewed(plan)
    require(digest(plan['brief']) == plan['brief_sha256'], 'native reviewed brief changed')
    _, reference, original, paths, own, native_job = references(plan)
    require(not os.environ.get('LD_PRELOAD') and not os.environ.get('LD_AUDIT') and os.getuid() > 0,
            'clean nonroot native collector required')
    require(shutil.disk_usage(PROJECT_ROOT).free >= candidate.OUTPUT_MAX + pair.DISK_FLOOR,
            'native output budget and disk reserve required')
    require(subprocess.run(['/usr/bin/sudo', '-n', '/usr/bin/true'], timeout=10, check=False).returncode == 0,
            'native actual-context authorization unavailable')
    common, h = candidate.load_helpers()
    runtime = {name: value for name, value in original.items() if activity.recipe.production_input(Path(name))}
    require(h.training.manifest({Path(name) for name in runtime}) == runtime, 'accepted native runtime inputs changed')
    gpu = h.query('gpu', h.training.GPU_FIELDS)
    require(gpu == reference['gpu_identity_after'], 'native accepted GPU identity changed')
    outer = create_attempt(resolve_artifact_root('bert-native-spatial', category='profiles'))
    print('Native spatial capture: ' + str(outer), flush=True)
    manifest = dict(gpu=gpu, uid=os.getuid(), gid=os.getgid(), groups=[], python=str(protocol.PYTHON),
        startup_reference_bundle=str(common.STARTUP_INPUTS), startup_reference_report_sha256=common.STARTUP_INPUTS_SHA,
        entrypoint_sha256=digest(ENTRY), source_sha256=sources, application_environment=native_job['application_environment'],
        ncu_sha256=digest(ncu.NCU), ncu_exec_sha256=digest(ncu.NCU_EXEC),
        ncu_injection_seals={str((bridge.NCU_DIRECTORY / name).resolve(strict=True)):
            digest(bridge.NCU_DIRECTORY / name) for name in ncu.NCU_INJECTION_NAMES},
        ncu_supervisor_environment=dict(PATH='/usr/bin:/bin', LANG='C.UTF-8', LC_ALL='C.UTF-8',
            TMPDIR=str(outer / 'ncu-tmp'), CUDA_CACHE_DISABLE='1', CUDA_DEVICE_ORDER='PCI_BUS_ID',
            CUDA_VISIBLE_DEVICES=gpu.split(',')[0].strip()), plan_sha256=digest(plan_path))
    h.startup_reference, h.export = common.application_references(h, manifest)
    manifest.update(assets=str(h.training.ASSET_ROOT), startup_reference=str(h.startup_reference), export=str(h.export))
    require(protocol.application_commands(manifest, own) == native_job['application_commands'],
            'native command templates do not match the accepted application')
    loader_paths, loader_receipt = native_loader_closure(manifest['application_environment'])
    save(outer / 'native-loader.json', loader_receipt)
    manifest['loader_closure_sha256'] = digest(outer / 'native-loader.json')
    paths.update(loader_paths | {outer / 'native-loader.json'})
    save(outer / 'manifest.json', manifest)
    save(outer / 'plan.json', plan)
    (outer / 'passes').mkdir()
    (outer / 'ncu-tmp').mkdir()
    paths.update((outer / 'plan.json', plan_path, Path(plan['brief']), Path(plan['launch_review'])))
    paths.update(Path(name) for name in sources)
    paths.update(Path(module.__file__).resolve(strict=True) for module in tuple(sys.modules.values())
                 if getattr(module, '__file__', None) and Path(module.__file__).is_relative_to(PROJECT_ROOT))
    before = h.training.manifest(paths)
    frozen = mapping_inputs(before)
    save(outer / 'mapping-inputs.json', frozen)
    manifest['mapping_inputs_sha256'] = digest(outer / 'mapping-inputs.json')
    save(outer / 'manifest.json', manifest)
    save(outer / 'inputs-before.json', before)
    report = dict(status='failed', manifest_sha256=digest(outer / 'manifest.json'), **FLAGS)
    try:
        report['idle_window'] = wait_for_idle(outer)
        require(report['idle_window']['achieved'], 'native idle window unavailable')
        activity.numerical.authorization(plan, 900)
        require(h.training.manifest(paths) == before, 'native preflight input identity changed')
        require(mapping_inputs(before) == frozen, 'native preflight mapped file identity changed')
        check_loader_state(loader_receipt)
        h.training.require_idle()
        activity.numerical.authorization(plan, 900)
        with plan_path.with_name('launch.json').open('x') as stream:
            json.dump(dict(plan_sha256=digest(plan_path), attempt=str(outer)), stream)
        report['unit'] = launch_unit(outer, gpu)
        report['observation'] = review_capture(outer)
        report['status'] = 'native_spatial_observation_requires_independent_review'
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        report['supervisor_error'] = ncu.supervisor_failure(outer)
    finally:
        try:
            after = h.training.manifest(paths)
            save(outer / 'inputs-after.json', after)
            report['input_identity_unchanged'] = before == after
            require(before == after and h.query('gpu', h.training.GPU_FIELDS) == gpu, 'native final identity changed')
            require(mapping_inputs(after) == frozen and read(outer / 'manifest.json') == manifest,
                    'native final mapped input or manifest identity changed')
            check_loader_state(loader_receipt)
            h.training.require_idle()
            report['output_bytes'] = candidate.check_output(outer)
        except BaseException as error:
            report.update(status='failed', final_error=f'{type(error).__name__}: {error}')
        save(outer / 'report.json', report)
    print(json.dumps(dict(status=report['status'], error=report.get('error'), attempt=str(outer))), flush=True)
    return int(report['status'] == 'failed')


def analyze(outer):
    report, plan, manifest = (read(outer / name) for name in ('report.json', 'plan.json', 'manifest.json'))
    require(report['status'] == 'native_spatial_observation_requires_independent_review' and
            report['input_identity_unchanged'] is True, 'complete unchanged native capture required')
    require(digest(outer / 'manifest.json') == report['manifest_sha256'] ==
            read(outer / 'ncu-supervisor.json')['manifest_sha256'], 'native manifest binding changed')
    sources = reviewed(plan)
    require(sources == manifest['source_sha256'] and digest(plan['brief']) == plan['brief_sha256'],
            'native reviewed source/brief identity changed')
    references(plan)
    before, after = read(outer / 'inputs-before.json'), read(outer / 'inputs-after.json')
    common, h = candidate.load_helpers()
    require(before == after == h.training.manifest({Path(name) for name in before}),
            'native input closure changed after capture')
    require(mapping_inputs(after) == read(outer / 'mapping-inputs.json'), 'native mapped inputs changed')
    unit = read(outer / 'unit.json')
    require(unit == report['unit'] and unit['returncode'] == 0 and unit['unit_removed'] is True and
            'error' not in unit and 'cleanup_error' not in unit, 'native task did not close normally')
    observation = review_capture(outer)
    require(report['observation'] == observation, 'native raw reanalysis differs')
    return observation


def loader_audit(directory, environment_manifest):
    """CPU-only validation; never grants permission or retries a GPU capture."""
    environment = read(environment_manifest)['application_environment']
    paths, receipt = native_loader_closure(environment)
    paths.update((ENTRY, Path(protocol.__file__).resolve(),
                  PROJECT_ROOT / 'tests/examples/bert_native_ncu_protocol_test.py'))

    def identities():
        return {str(path): dict(target=str(path.resolve(strict=True)), bytes=path.stat().st_size,
                                sha256=digest(path)) for path in sorted(paths)}

    before = identities()
    frozen = mapping_inputs(before)
    # Reproduce the standard-library import that exposed the missing DSO. This
    # process imports no GPU framework, creates no CUDA context and gains no cap.
    import lzma
    require(lzma.compress(b'native-loader-audit'), 'CPU standard-library witness failed')
    mappings = NativeMaps(before, frozen).observe(os.getpid())
    after = identities()
    require(before == after and mapping_inputs(after) == frozen, 'CPU loader audit inputs changed')
    check_loader_state(receipt)
    result = dict(status='fixed_python_environment_inventory_checked_cpu_only', closure=receipt,
        input_before=before, input_after=after, frozen_metadata=frozen, cpu_observed_mappings=mappings,
        source_sha256={str(path): digest(path) for path in (ENTRY, Path(protocol.__file__).resolve(),
            PROJECT_ROOT / 'tests/examples/bert_native_ncu_protocol_test.py')},
        gpu_capture_started=False, new_capture_authorized=False, **FLAGS)
    output = directory / 'native-loader-audit.json'
    ncu.publish_json(output, result)
    print(json.dumps(dict(status=result['status'], report=str(output),
                         roots=len(receipt['roots']), files=len(before))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare', 'collect', 'root-supervisor', 'bridge', 'controller', 'analyze', 'loader-audit'):
        mode.add_argument('--' + name, type=Path)
    parser.add_argument('--environment-manifest', type=Path)
    parser.add_argument('--endpoint')
    parser.add_argument('--supervisor')
    parser.add_argument('--serial', type=int)
    parser.add_argument('--target-mode', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--manifest-sha256', help=argparse.SUPPRESS)
    parser.add_argument('--mapping-sha256', help=argparse.SUPPRESS)
    args = parser.parse_args()
    require((args.target_mode and bool(args.bridge or args.controller) and
             type(args.manifest_sha256) is str and
             re.fullmatch(r'[0-9a-f]{64}', args.manifest_sha256) is not None and
             ((args.bridge and args.mapping_sha256 is None) or
              (args.controller and type(args.mapping_sha256) is str and
               re.fullmatch(r'[0-9a-f]{64}', args.mapping_sha256) is not None))) or
            (not args.target_mode and args.manifest_sha256 is None and args.mapping_sha256 is None),
            'internal target mode requires exact bridge or controller bindings')
    if args.loader_audit:
        require(args.environment_manifest is not None, 'CPU audit environment manifest required')
        loader_audit(args.loader_audit.resolve(strict=True), args.environment_manifest.resolve(strict=True))
        return 0
    if args.prepare:
        prepare(args.prepare.resolve(strict=True))
        return 0
    if args.collect:
        return collect(args.collect.resolve(strict=True))
    if args.root_supervisor:
        return root_supervisor(args.root_supervisor.resolve(strict=True))
    if args.analyze:
        outer = args.analyze.resolve(strict=True)
        observation = analyze(outer)
        print(json.dumps(dict(status='native_counter_reanalysis_passed', launches=len(observation['inventory']))))
        return 0
    outer = (args.bridge or args.controller).resolve(strict=True)
    supervisor = bridge.parse_identity(args.supervisor)
    return (bridge_worker(outer, args.endpoint, supervisor, target_mode=args.target_mode,
                          expected_manifest_sha256=args.manifest_sha256) if args.bridge else
            controller(outer, args.endpoint, supervisor, args.serial, target_mode=args.target_mode,
                       expected_manifest_sha256=args.manifest_sha256,
                       expected_mapping_sha256=args.mapping_sha256))


if __name__ == '__main__':
    raise SystemExit(main())
