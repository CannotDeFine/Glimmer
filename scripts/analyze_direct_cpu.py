"""Resolve raw CPU samples without converting sample counts into elapsed time."""

from bisect import bisect_right
from collections import Counter
import argparse
import json
import os
from pathlib import Path
import re
import struct
import subprocess

from artifact_paths import PROJECT_ROOT
from qualify_direct_cpu import parse_profile, require
from run_rnn_v8_validation import digest, save

NM = Path('/usr/bin/nm').resolve(strict=True)
ADDR2LINE = Path('/usr/bin/addr2line').resolve(strict=True)
CXXFILT = Path('/usr/bin/c++filt').resolve(strict=True)


def executable_segments(path):
    """File offset, virtual address and file-backed size, not assumed PIE bias."""
    with Path(path).open('rb') as stream:
        header = stream.read(64)
        require(len(header) == 64, 'short ELF header')
        size = os.fstat(stream.fileno()).st_size
        offset, length = program_header_bounds(header, size)
        stream.seek(offset)
        return decode_segments(stream.read(length), size)


def program_header_bounds(header, size):
    require(len(header) == 64, 'short ELF header')
    fields = struct.unpack('<16sHHIQQQIHHHHHH', header)
    require(fields[0][:7] == b'\x7fELF\x02\x01\x01' and fields[1] in (2, 3)
            and fields[2] == 62 and fields[3] == 1 and fields[8] == 64
            and fields[9] == 56 and 0 < fields[10] <= 128, 'unsupported ELF')
    require(64 <= fields[5] and fields[5] + fields[9] * fields[10] <= size,
            'ELF program header bounds')
    return fields[5], fields[9] * fields[10]


def decode_segments(data, size):
    require(data and len(data) % 56 == 0, 'truncated ELF program headers')
    result = []
    for kind, flags, offset, virtual, _, length, memory, alignment in struct.iter_unpack('<IIQQQQQQ', data):
        if kind == 1 and flags & 1:
            aligned = alignment in (0, 1) or (alignment > 1 and alignment & (alignment - 1) == 0
                                             and offset % alignment == virtual % alignment)
            require(0 < length <= memory and offset + length <= size
                    and virtual + memory < 2**64 and aligned, 'invalid executable segment')
            result.append((offset, virtual, length))
    require(result, 'no executable PT_LOAD')
    return result


def parse_maps(content):
    result = []
    for line in content.splitlines():
        if not line or line == 'MAPPED_LIBRARIES:':
            continue
        match = re.fullmatch(r'([0-9a-f]+)-([0-9a-f]+) ([rwxps-]{4}) +([0-9a-f]+) '
                             r'([0-9a-f]+):([0-9a-f]+) +(\d+)(?: +(.*))?', line)
        require(match is not None, 'malformed process mapping')
        low, high, permissions, offset, major, minor, inode, path = match.groups()
        low, high = int(low, 16), int(high, 16)
        require(0 <= low < high < 2**64, 'mapping address bounds')
        if 'x' in permissions:
            result.append(dict(low=low, high=high, permissions=permissions, offset=int(offset, 16), path=path or '',
                               device=os.makedev(int(major, 16), int(minor, 16)), inode=int(inode)))
    require(result, 'no executable mappings')
    return result


def authenticated_mapping(profile, snapshots, frozen):
    # gperftools2.9.1 ProfileData::DumpProcSelfMaps passes dev_t(0) to FormatLine.
    # Obtain the omitted device only from two independently captured live maps.
    require(profile['device'] == 0 and len(snapshots) == 2, 'unexpected profile mapping format')
    keys = ('low', 'high', 'permissions', 'offset', 'inode', 'path')
    matched = []
    for snapshot in snapshots:
        rows = [row for row in snapshot if all(row[key] == profile[key] for key in keys)]
        require(len(rows) == 1, 'missing or ambiguous live mapping')
        matched.append(rows[0])
    require(matched[0] == matched[1], 'live mappings disagree')
    row = matched[0]
    require((row['device'], row['inode'], row['path']) ==
            (frozen['device'], frozen['inode'], frozen['path']), 'prefrozen executable mapping differs')
    return row


def object_address(pc, mapping, segments):
    require(mapping['low'] <= pc < mapping['high'], 'PC outside mapping')
    offset = pc - mapping['low'] + mapping['offset']
    found = [virtual + offset - start for start, virtual, size in segments
             if start <= offset < start + size]
    require(len(found) == 1, 'missing or ambiguous executable segment')
    return found[0]


class Symbols:
    def __init__(self, output):
        rows = set()
        for line in output.splitlines():
            fields = line.split()
            if len(fields) == 4 and fields[2] in ('t', 'T', 'w', 'W'):
                start, size = int(fields[0], 16), int(fields[1], 16)
                if size:
                    require(0 <= start < start + size < 2**64, 'symbol bounds')
                    rows.add((start, start + size, fields[3]))
        require(rows, 'no sized physical functions')
        self.rows = sorted(rows)
        self.starts = [row[0] for row in self.rows]
        self.maximum = []
        self.names = Counter(row[2] for row in self.rows)
        for _, high, _ in self.rows:
            self.maximum.append(max(high, self.maximum[-1] if self.maximum else 0))

    def match(self, pc):
        index = bisect_right(self.starts, pc) - 1
        found = []
        while index >= 0 and self.maximum[index] > pc:
            if self.rows[index][1] > pc:
                found.append(self.rows[index])
            index -= 1
        if len(found) != 1 or self.names[found[0][2]] != 1:
            return None
        return found[0]


def select(counts, total):
    require(total > 0 and all(value >= 0 for value in counts.values())
            and sum(counts.values()) <= total, 'weighted partition bounds')
    qualified = [name for name, count in counts.items() if count >= 100 and count * 4 >= total]
    return qualified[0] if len(qualified) == 1 else None


def declarations(server, symbols, directory):
    rows = sorted(set(symbols))
    query = ''.join(f'{start:x}\n' for start, _, _ in rows)
    output = subprocess.run([str(ADDR2LINE), '-e', str(server)], input=query, text=True,
                            capture_output=True, timeout=30, check=True)
    (directory / 'addr2line.input').write_text(query)
    (directory / 'addr2line.stdout').write_text(output.stdout)
    (directory / 'addr2line.stderr').write_text(output.stderr)
    lines = output.stdout.splitlines()
    require(len(lines) == len(rows), 'declaration symbolization cardinality')
    result = {}
    for row, line in zip(rows, lines):
        match = re.fullmatch(r'(.+):(\d+|\?)(?: \(discriminator \d+\))?', line)
        source = Path(match[1]) if match and match[1] != '??' else None
        if source is None or not source.is_absolute() or match[2] in ('?', '0'):
            owned = None
        else:
            source = source.resolve()
            owned = any(source.is_relative_to(PROJECT_ROOT / name) for name in ('src', 'include'))
        result[row[2]] = dict(location=line, project_owned=owned)
    return result


def analyze(directory, server, objects, *, output=None):
    """All fractions retain the same total; unknown frames cannot increase shares."""
    directory, server = Path(directory), Path(server).resolve(strict=True)
    output = directory if output is None else Path(output)
    records, map_text = parse_profile((directory / 'owner.cpu.prof').read_bytes())
    mappings = parse_maps(map_text)
    live = [parse_maps((directory / name).read_text()) for name in ('maps-first', 'maps-last')]
    frozen = {item['path']: item for item in objects}
    require(len(frozen) == len(objects) and str(server) in frozen, 'unique frozen objects required')
    verified, segments = set(), {}
    raw_nm = subprocess.check_output([str(NM), '-S', '--defined-only', str(server)], timeout=30)
    (output / 'nm.stdout').write_bytes(raw_nm)
    table = Symbols(raw_nm.decode('utf-8'))

    def resolve(pc):
        found = [row for row in mappings if row['low'] <= pc < row['high']]
        service_possible = any(row['path'] == str(server) for row in found)
        if len(found) != 1:
            return None, 'ambiguous_or_missing_mapping', service_possible
        mapping = found[0]
        item = frozen.get(mapping['path'])
        if item is None:
            return None, 'unfrozen_mapping', service_possible
        try:
            mapping = authenticated_mapping(mapping, live, item)
        except ValueError:
            return None, 'mapping_identity', service_possible
        path = Path(item['path'])
        if str(path) not in verified:
            info = path.stat()
            require((info.st_dev, info.st_ino, info.st_size) ==
                    (item['device'], item['inode'], item['bytes']) and digest(path) == item['sha256'],
                    'frozen executable bytes changed')
            verified.add(str(path))
            segments[str(path)] = executable_segments(path)
        try:
            virtual = object_address(pc, mapping, segments[str(path)])
        except ValueError:
            return None, 'ambiguous_or_missing_segment', service_possible
        return (mapping['path'], virtual), None, service_possible

    # Cache each adjusted PC once; the first PC is an instruction, others are returns.
    cache = {pc - int(index > 0): None for _, chain in records for index, pc in enumerate(chain) if pc}
    for pc in cache:
        cache[pc] = resolve(pc)
    hits = {}
    for key, (resolved, _, _) in cache.items():
        if resolved and resolved[0] == str(server):
            hits[key] = table.match(resolved[1])
    ownership = declarations(server, [row for row in hits.values() if row], output)
    counts, unknown, modules = Counter(), Counter(), Counter()
    total = mapped = 0
    for weight, chain in records:
        total += weight
        first, reason, _ = cache[chain[0]]
        if first:
            mapped += weight
            modules[first[0]] += weight
        else:
            modules['unresolved:' + reason] += weight
        assigned, failure = None, 'no_project_frame'
        for index, pc in enumerate(chain):
            if not pc:
                continue
            adjusted = pc - int(index > 0)
            resolved, reason, service_possible = cache[adjusted]
            if not resolved:
                failure = reason
                break
            if resolved[0] != str(server):
                continue
            symbol = hits[adjusted]
            if symbol is None:
                failure = 'ambiguous_or_missing_physical_symbol'
                break
            owned = ownership[symbol[2]]['project_owned']
            if owned is None:
                failure = 'unresolved_declaration'
                break
            if owned:
                assigned = symbol[2]
                break
        if assigned:
            counts[assigned] += weight
        else:
            unknown[failure] += weight
    require(sum(counts.values()) + sum(unknown.values()) == total, 'raw weighted partition')
    names = list(counts)
    labels = (subprocess.run([str(CXXFILT)], input='\n'.join(names) + '\n', text=True,
                            capture_output=True, timeout=15, check=True).stdout.splitlines() if names else [])
    require(len(labels) == len(names), 'display label count')
    demangled = dict(zip(names, labels))
    selected = select(counts, total)
    qualified = total >= 500 and mapped * 100 >= total * 95 and sum(counts.values()) * 100 >= total * 90
    result = dict(status='hotspot_capture_requires_review' if qualified else 'coverage_failed',
        total_samples=total, mapped_sample_pc_count=mapped, mapped_sample_pc_fraction=mapped / total,
        project_frame_samples=sum(counts.values()), project_frame_fraction=sum(counts.values()) / total,
        unresolved=dict(unknown), sampled_modules=dict(modules),
        functions=[dict(object=str(server), object_sha256=frozen[str(server)]['sha256'],
                        symbol=name, label=demangled[name], samples=count, fraction=count / total,
                        **ownership[name]) for name, count in counts.most_common()],
        selected_symbol=selected if qualified else None,
        direction='bounded_source_audit' if qualified and selected else 'mixed_or_unqualified',
        configured_frequency_hz=500, effective_sampling_frequency_known=False,
        device_identity_source='two authenticated live maps joined to prefrozen executable objects',
        profile_device_is_omitted=True,
        sample_counts_are_elapsed_time=False, previous_poll_intervals_joined=False,
        diagnostic_only=True, full_device_qualified=False, performance_qualified=False,
        saturation_qualified=False, gpu_gap_causality_qualified=False)
    save(output / 'analysis.json', result)
    return result


def controls():
    header = struct.pack('<16sHHIQQQIHHHHHH', b'\x7fELF\x02\x01\x01', 3, 62, 1,
                         0, 64, 0, 0, 64, 56, 1, 0, 0, 0)
    segment = struct.pack('<IIQQQQQQ', 1, 5, 0x1000, 0x3000, 0, 0x1000, 0x1000, 0x1000)
    require(program_header_bounds(header, 0x2000) == (64, 56)
            and decode_segments(segment, 0x2000) == [(0x1000, 0x3000, 0x1000)], 'ELF64 DSO fixture')
    invalid = [(program_header_bounds, (header[:50], 0x2000)),
               (program_header_bounds, (b'BAD!' + header[4:], 0x2000)),
               (program_header_bounds, (header, 100)),
               (decode_segments, (segment[:-1], 0x2000)),
               (decode_segments, (segment, 0x1500))]
    for function, args in invalid:
        try:
            function(*args)
        except ValueError:
            pass
        else:
            raise ValueError('malformed ELF accepted')
    mapping = dict(low=0x70001000, high=0x70002000, offset=0x1000)
    require(object_address(0x70001123, mapping, [(0x1000, 0x3000, 0x1000)]) == 0x3123,
            'DSO nonzero load relocation')
    require(object_address(0x70001123, mapping, [(0x1000, 0x1000, 0x1000)]) == 0x1123,
            'PIE relocation')
    for segments in ([], [(0x1000, 0x1000, 0x1000), (0x1000, 0x2000, 0x1000)]):
        try:
            object_address(0x70001123, mapping, segments)
        except ValueError:
            pass
        else:
            raise ValueError('ambiguous/missing segment accepted')
    table = Symbols('100 10 T first\n110 10 T second\n200 20 T alias1\n200 20 T alias2\n'
                    '300 10 t duplicate\n400 10 t duplicate\n')
    require(table.match(0x110)[2] == 'second' and table.match(0x110 - 1)[2] == 'first'
            and table.match(0x208) is None and table.match(0x305) is None
            and table.match(0x250) is None, 'return-PC/aliases/duplicate symbols')
    require(select({'a': 250, 'b': 200}, 1000) == 'a'
            and select({'a': 250, 'b': 250}, 1000) is None
            and select({'a': 99}, 100) is None
            and select({'a': 200}, 1000) is None, 'fixed denominator selector')
    for text in ('invalid', '100-90 r-xp 0000 00:00 1 /bad'):
        try:
            parse_maps(text)
        except ValueError:
            pass
        else:
            raise ValueError('malformed mapping accepted')
    require(len(parse_maps('100-200 r-xp 0000 08:01 1 /object\n')) == 1, 'mapping positive')
    live = parse_maps('100-200 r-xp 0000 08:01 1 /object\n')[0]
    profile = dict(live, device=0)
    frozen = {key: live[key] for key in ('device', 'inode', 'path')}
    require(authenticated_mapping(profile, [[live], [live]], frozen) == live, 'omitted device join')
    invalid = [(dict(profile, device=1), [[live], [live]], frozen),
               (profile, [[], [live]], frozen), (profile, [[live, live], [live]], frozen),
               (profile, [[live], [dict(live, device=2)]], frozen),
               (profile, [[live], [live]], dict(frozen, device=2))]
    for key, value in (('permissions', 'rwxp'), ('low', 0x101), ('high', 0x201),
                       ('offset', 1), ('path', '/other'), ('inode', 2)):
        invalid.append((profile, [[live], [dict(live, **{key: value})]], frozen))
    for args in invalid:
        try:
            authenticated_mapping(*args)
        except ValueError:
            pass
        else:
            raise ValueError('ambiguous or inconsistent mapping accepted')
    return dict(status='passed', scope='relocation_symbols_return_pc_maps_weighted_selector')


def reanalyze(source, output):
    source, output = source.resolve(strict=True), output.resolve(strict=True)
    require(not output.is_relative_to(source) and not (output / 'analysis.json').exists(),
            'independent fresh offline output required')
    plan = json.loads((output / 'plan.json').read_text())
    require(plan['status'] == 'approved_before_offline_successor' and plan['source_attempt'] == str(source),
            'reviewed offline scope required')
    prior = json.loads((source / 'report.json').read_text())
    worker = json.loads((source / 'equal-worker.json').read_text())
    require(prior['status'] == worker['status'] == 'failed' and prior['input_identity_unchanged'] is True
            and worker['error'] == 'ValueError: sample/symbol coverage failed'
            and worker['observation']['normal_retirement'] is True, 'raw capture lifecycle required')
    before_inputs = json.loads((source / 'inputs-before.json').read_text())
    require(before_inputs == json.loads((source / 'inputs-after.json').read_text()), 'capture seals differ')
    for tool in (NM, ADDR2LINE, CXXFILT):
        require(before_inputs[str(tool)]['sha256'] == digest(tool), 'prefrozen symbol tool changed')
    import bert_device_capture as capture
    capture.normal_application(worker['raw_job'])
    unit = json.loads((source / 'equal-unit.json').read_text())
    require(unit['unit_removed'] is True and unit['returncode'] == 1, 'original failed unit must be retired')
    sampler = source / 'sampler'
    from qualify_direct_cpu import receipt
    receipt(sampler, True)
    paths = [path for path in source.rglob('*') if path.is_file()]
    raw_before = {str(path): digest(path) for path in paths}
    save(output / 'source-identities-before.json', raw_before)
    own = {str(path): digest(path) for path in (Path(__file__).resolve(), NM, ADDR2LINE, CXXFILT,
                                               output / 'plan.json')}
    save(output / 'analyzer-identities.json', own)
    save(output / 'controls.json', controls())
    objects = json.loads((source / 'objects.json').read_text())
    targets = {value['target']: value for value in before_inputs.values()}
    require(all(row['path'] in targets and all(row[key] == targets[row['path']][key]
                for key in ('bytes', 'sha256')) for row in objects), 'objects differ from pre-capture manifest')
    server = json.loads((source / 'plan.json').read_text())['server']
    result = analyze(sampler, server, objects, output=output)
    raw_after = {str(path): digest(path) for path in paths}
    require(raw_before == raw_after and all(digest(Path(name)) == sha for name, sha in own.items()),
            'offline input or analyzer changed')
    save(output / 'source-identities-after.json', raw_after)
    save(output / 'report.json', dict(status='offline_reanalysis_requires_review', source=str(source),
         source_unchanged=True, original_capture_status='failed', new_gpu_runs=0,
         analysis=result, diagnostic_only=True, full_device_qualified=False, performance_qualified=False))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reanalyze', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.reanalyze:
        require(args.output is not None, 'explicit offline output required')
        result = reanalyze(args.reanalyze, args.output)
        print(json.dumps(dict(status=result['status'], total=result['total_samples'],
                              selected_symbol=result['selected_symbol'])))
    else:
        print(json.dumps(controls()))
