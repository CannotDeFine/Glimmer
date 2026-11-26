"""Fail-closed process checks for the opt-in BERT NCU capture path.

This module does not start a profiler or establish CUDA/SM coverage. The
collector must independently seal the NCU installation and authenticate its
parent process, cgroup, replay, kernel and complete application results.
"""

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import re


NCU_INJECTION_ORDER = ('libcuda-injection.so', 'libTreeLauncherTargetUpdatePreloadInjection.so',
                       'libInterceptorInjectionTarget.so', 'libTreeLauncherTargetInjection.so')
NCU_INJECTION_NAMES = frozenset(NCU_INJECTION_ORDER)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def split_preload(value):
    if not value:
        return []
    require(type(value) is str and not any(char in value for char in '\x00\t\r\n'),
            'invalid preload environment')
    return [part for part in re.split(r'[: ]+', value) if part]


class NcuMappingUnavailable(ValueError):
    """Validated injection paths absent from one live procfs map observation."""

    def __init__(self, expected, mapped):
        self.expected = sorted(expected)
        self.mapped = sorted(mapped)
        self.missing = sorted(set(expected) - set(mapped))
        super().__init__('NCU injection mapping unavailable: ' +
                         ', '.join(Path(name).name for name in self.missing))


def verified_preload(actual, expected, seals, mapped):
    """Allow only sealed, actually mapped NCU additions to the original preload.

    `seals` maps canonical absolute NCU library paths to lowercase SHA256;
    `mapped` maps observed canonical paths to device/inode pairs. The base
    preload must remain byte-for-byte present and in its original order.
    """
    actual_parts, expected_parts = split_preload(actual), split_preload(expected)
    require(type(seals) is dict and seals and type(mapped) is dict, 'sealed profiler mappings required')
    normalized = {}
    for name, digest in seals.items():
        path = Path(name)
        require(path.is_absolute() and path == path.resolve(strict=True) and
                path.name in NCU_INJECTION_NAMES and type(digest) is str and
                re.fullmatch(r'[0-9a-f]{64}', digest) is not None, 'invalid NCU injection seal')
        normalized[str(path)] = digest
    require(set(mapped) <= set(normalized), 'an actual NCU mapping is not sealed')
    for canonical, identity in mapped.items():
        digest = hashlib.sha256()
        with Path(canonical).open('rb') as stream:
            metadata = os.fstat(stream.fileno())
            require((metadata.st_dev, metadata.st_ino) == identity,
                    'opened NCU library differs from actual mapping')
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
            after = os.fstat(stream.fileno())
            require((after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) ==
                    (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns),
                    'NCU injection changed while hashing')
        require(digest.hexdigest() == normalized[canonical], 'NCU injection bytes changed')
    remaining, injected = [], []
    for item in actual_parts:
        path = Path(item)
        canonical = str(path.resolve(strict=True)) if path.is_absolute() else ''
        if canonical in normalized:
            require(canonical not in injected, 'duplicate NCU preload')
            injected.append(canonical)
        else:
            remaining.append(item)
    require(remaining == expected_parts, 'unexpected or modified non-NCU preload')
    order = [NCU_INJECTION_ORDER.index(Path(path).name) for path in injected]
    require(order == sorted(set(order)), 'NCU preload order differs')
    cuda = [path for path in normalized if Path(path).name == 'libcuda-injection.so']
    require(len(cuda) == 1, 'one sealed CUDA injection required')
    required = set(injected) | set(cuda)
    if not required <= set(mapped):
        raise NcuMappingUnavailable(required, mapped)
    return dict(mapped_ncu_injections=sorted(mapped), additional_preloads=injected)


def proc_environment(pid, *, ncu_seals=None):
    """Read procfs strictly, optionally recognizing the authenticated NCU scrub.

    NCU 2024.3.2 removes its preload from the live environment but replaces
    only the first value byte with NUL in the initial procfs environment.
    Accept that exact residue only when its complete preload is sealed and
    actually mapped. Callers must authenticate the profiled process first.
    """
    raw = Path(f'/proc/{pid}/environ').read_bytes().split(b'\0')
    result, residues = {}, []
    for index, item in enumerate(raw):
        if not item:
            continue
        if b'=' not in item:
            residues.append(index)
            continue
        name, value = item.split(b'=', 1)
        require(name, 'empty process environment key')
        require(name not in result, 'duplicate process environment key')
        result[name] = value
    if residues:
        require(ncu_seals is not None and len(residues) == 1,
                'malformed process environment entry')
        index = residues[0]
        require(index > 0 and raw[index - 1] == b'LD_PRELOAD=' and
                result.get(b'NVTX_INJECTION64_PATH') == b'libcuda-injection.so',
                'unrecognized NCU preload residue')
        tracker = result.get(b'NVIDIA-PROCESS-TRACKING-CONFIGURATION', b'').splitlines()
        fields = (b'not-launcher-preload-prepend', b'preload-update-preload-prepend',
                  b'launcher-preload-prepend')
        values = []
        for field in fields:
            entries = [line[len(field) + 2:] for line in tracker if line.startswith(field + b': ')]
            require(len(entries) == 1, 'missing or duplicate NCU preload tracker field')
            values.append(entries[0])
        preload = b':'.join(values)
        parts = preload.split(b':')
        names = NCU_INJECTION_ORDER
        require(len(parts) == len(names) and
                all(part.startswith(b'/') and Path(os.fsdecode(part)).name == name
                    for part, name in zip(parts, names)) and raw[index] == preload[1:],
                'NCU preload residue differs from the complete tracker preload')
        verified_preload(os.fsdecode(preload), None, ncu_seals, proc_mapped_paths(pid))
        del result[b'LD_PRELOAD']
    return result


def proc_mapped_paths(pid):
    paths = {}
    for line in Path(f'/proc/{pid}/maps').read_text().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) == 6 and parts[5].startswith('/'):
            deleted = parts[5].endswith(' (deleted)')
            path = Path(parts[5][:-10] if deleted else parts[5])
            if path.name in NCU_INJECTION_NAMES:
                require(not deleted, 'deleted NCU process mapping')
                device = parts[3].split(':')
                require(len(device) == 2 and parts[4].isdecimal(), 'invalid mapping identity')
                metadata = path.stat()
                require((int(device[0], 16), int(device[1], 16), int(parts[4])) ==
                        (os.major(metadata.st_dev), os.minor(metadata.st_dev), metadata.st_ino),
                        'mapped NCU library file identity changed')
                canonical = str(path.resolve(strict=True))
                current = (metadata.st_dev, metadata.st_ino)
                require(canonical not in paths or paths[canonical] == current,
                        'ambiguous NCU mapping identity')
                paths[canonical] = current
    return paths


@contextmanager
def authenticated_profile_processes(common, seals, clients=None):
    """Scope injection-aware checks around the original identity verifier."""
    original = common.verify_process

    def verified(h, process, start, command=None, environment=None):
        registered = (None if clients is None else
                      next((child for child in clients.values()
                            if child.pid == process.pid and child.start == start), None))
        if registered is not None:
            require(not proc_mapped_paths(process.pid), 'broker client inherited NCU injection')
            actual = proc_environment(process.pid)
            require(b'NVTX_INJECTION64_PATH' not in actual,
                    'broker client retained profiler NVTX environment')
            return original(h, process, start, command, environment)
        if environment is None:
            return original(h, process, start, command, environment)
        actual = proc_environment(process.pid)
        actual_preload = os.fsdecode(actual.get(b'LD_PRELOAD', b''))
        expected_preload = environment.get('LD_PRELOAD')
        verified_preload(actual_preload, expected_preload, seals, proc_mapped_paths(process.pid))
        require(actual.get(b'NVTX_INJECTION64_PATH') == b'libcuda-injection.so',
                'NCU NVTX injection missing from selected process')
        adjusted = dict(environment)
        if b'LD_PRELOAD' in actual:
            adjusted['LD_PRELOAD'] = actual_preload
        else:
            adjusted.pop('LD_PRELOAD', None)
        return original(h, process, start, command, adjusted)

    common.verify_process = verified
    try:
        yield
    finally:
        common.verify_process = original
