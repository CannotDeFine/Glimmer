#!/usr/bin/env python3
"""Validate bounded transport journals and partition observed launch wall intervals."""

from array import array
from bisect import bisect_right
from collections import Counter, namedtuple
import json
import mmap
import os
from pathlib import Path
import stat
import struct


def require(value, message):
    if not value:
        raise ValueError(message)


Cycle = namedtuple('Cycle', 'id entry exit timeout peer_begin peer_count poll_begin poll_count success')
Poll = namedtuple('Poll', 'cycle entry returned result error')
Peer = namedtuple('Peer', 'cycle connection pid start events revents pending output flags visit alive')
Read = namedtuple('Read', 'cycle connection token entry returned first_read result error')
Frame = namedtuple('Frame', 'cycle connection token wire opcode first_read received pid start')
LAYOUTS = ((Cycle, '<9Q', 524288), (Poll, '<3QqQ', 1048576),
           (Peer, '<11Q', 4194304), (Read, '<6QqQ', 4194304), (Frame, '<9Q', 1048576))
PHASES = ('outside_poll_call', 'call_setup', 'ppoll', 'poll_return_to_target_visit',
          'target_visit_to_recv_entry', 'recv_syscall', 'positive_return_to_first_read', 'unresolved')
MAX_BYTES = 512 * 1024**2
APP_LIMITS = (524288, 524288, 3145728, 1572864, 524288)


class Rows:
    """Read one immutable row at a time, without copying a large ledger into objects."""

    def __init__(self, content, offset, count, row_type, layout):
        self.content, self.offset, self.count = content, offset, count
        self.row_type, self.layout = row_type, struct.Struct(layout)

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if not isinstance(index, int) or index < 0 or index >= self.count:
            raise IndexError(index)
        return self.row_type(*self.layout.unpack_from(self.content, self.offset + index * self.layout.size))


class Journal:
    """Owner/size/schema checks precede mmap; validity is reconstructed, not trusted."""

    def __init__(self, path, identity=None, *, application=False):
        self.mapping = None
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            info = os.fstat(descriptor)
            require(stat.S_ISREG(info.st_mode) and info.st_uid == os.geteuid()
                    and stat.S_IMODE(info.st_mode) == 0o600 and info.st_nlink == 1
                    and 128 <= info.st_size <= MAX_BYTES, 'bounded owned journal file required')
            header = os.pread(descriptor, 128, 0)
            require(len(header) == 128 and header[:8] == b'GLMPOLL1', 'journal header/magic differs')
            fields = struct.unpack('<16Q', header)
            require(fields[1:3] == (1, 128) and fields[3] > 0 and fields[4] > 0
                    and fields[11:] == (0, 0, 0, 0, 0), 'journal version/owner/reserved fields differ')
            self.identity = dict(pid=fields[3], start=fields[4])
            require(identity is None or identity == self.identity, 'journal service identity differs')
            counts, offset, views = fields[5:10], 128, []
            for index, ((row_type, layout, maximum), count) in enumerate(zip(LAYOUTS, counts)):
                require(0 < count <= (APP_LIMITS[index] if application else maximum),
                        'journal row count outside frozen bounds')
                offset += count * struct.calcsize(layout)
            require(offset == info.st_size and fields[10] == offset - 128,
                    'journal short/trailing bytes or payload length differs')
            self.mapping = mmap.mmap(descriptor, 0, access=mmap.ACCESS_READ)
            offset = 128
            for (row_type, layout, _), count in zip(LAYOUTS, counts):
                views.append(Rows(self.mapping, offset, count, row_type, layout))
                offset += count * struct.calcsize(layout)
            self.cycles, self.polls, self.peers, self.reads, self.frames = views
            self.counts = dict(zip(('cycles', 'polls', 'peers', 'reads', 'frames'), counts))
            self.validate()
        except BaseException:
            self.close()
            raise
        finally:
            os.close(descriptor)

    def close(self):
        if self.mapping is not None:
            self.mapping.close()
            self.mapping = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def cycle_peers(self, cycle):
        return [self.peers[index] for index in range(cycle.peer_begin, cycle.peer_begin + cycle.peer_count)]

    def target_peer(self, cycle, connection):
        return next((peer for peer in self.cycle_peers(cycle) if peer.connection == connection), None)

    def validate(self):
        peer_cursor = poll_cursor = last = 0
        identities = {}
        self.ends = array('Q')
        for index, cycle in enumerate(self.cycles):
            require(cycle.id == index + 1 and 0 < cycle.entry <= cycle.exit and last <= cycle.entry
                    and cycle.success in (0, 1) and cycle.peer_begin == peer_cursor
                    and cycle.poll_begin == poll_cursor
                    and cycle.peer_count <= len(self.peers) - peer_cursor
                    and cycle.poll_count <= len(self.polls) - poll_cursor
                    and (not cycle.success or cycle.poll_count > 0), 'invalid cycle order/ranges')
            poll_end = cycle.entry
            for cursor in range(poll_cursor, poll_cursor + cycle.poll_count):
                row = self.polls[cursor]
                require(row.cycle == cycle.id and poll_end <= row.entry <= row.returned <= cycle.exit
                        and row.result >= -1 and (row.result < 0) == (row.error > 0),
                        'invalid poll order/result')
                poll_end = row.returned
            visited, connections = poll_end, set()
            for peer in self.cycle_peers(cycle):
                require(peer.cycle == cycle.id and peer.connection > 0 and peer.connection not in connections
                        and peer.pid > 0 and peer.start > 0 and peer.flags <= 3
                        and peer.events <= 65535 and peer.revents <= 65535 and peer.alive in (0, 1)
                        and (not peer.flags or not peer.events & 1), 'invalid peer interest/identity')
                connections.add(peer.connection)
                identity = (peer.pid, peer.start)
                require(identities.setdefault(peer.connection, identity) == identity,
                        'connection changed authenticated owner')
                if cycle.success or peer.visit:
                    require(visited <= peer.visit <= cycle.exit, 'peer visit order differs')
                    visited = peer.visit
            peer_cursor += cycle.peer_count
            poll_cursor += cycle.poll_count
            last = cycle.exit
            self.ends.append(last)
        require(peer_cursor == len(self.peers) and poll_cursor == len(self.polls), 'orphan peer/poll rows')
        self.read_offsets = array('Q', [0]) * (len(self.cycles) + 1)
        first_count = last_cycle = last_return = 0
        token_last = {}
        for index, row in enumerate(self.reads):
            require(0 < row.cycle <= len(self.cycles) and row.cycle >= last_cycle,
                    'read cycle order differs')
            cycle = self.cycles[row.cycle - 1]
            peer = self.target_peer(cycle, row.connection)
            require(peer is not None and peer.alive and peer.revents & 1 and peer.events & 1
                    and not peer.flags and peer.visit <= row.entry <= row.returned <= cycle.exit
                    and row.entry >= last_return and row.result >= -1
                    and (row.result < 0) == (row.error > 0), 'read lacks exact readable peer/clock/result')
            require(row.result <= 0 or row.token > 0, 'positive read lacks token')
            if row.token:
                require(0 < row.token <= index + 1, 'forward or missing read token')
                first = self.reads[row.token - 1]
                require(first.token == row.token and first.first_read > 0 and first.result > 0
                        and first.connection == row.connection, 'read token belongs to another frame')
                token_last[row.token] = (row.cycle, row.returned, index)
            if row.first_read:
                require(row.token == index + 1 and row.result > 0
                        and row.returned <= row.first_read <= cycle.exit, 'invalid first-positive stamp')
                first_count += 1
            self.read_offsets[row.cycle] += 1
            last_cycle, last_return = row.cycle, max(row.returned, row.first_read)
        for index in range(1, len(self.read_offsets)):
            self.read_offsets[index] += self.read_offsets[index - 1]
        completed, self.by_wire = set(), {}
        last_cycle = last_stamp = 0
        for frame in self.frames:
            require(0 < frame.cycle <= len(self.cycles) and frame.cycle >= last_cycle
                    and frame.received >= last_stamp and 0 < frame.token <= len(self.reads)
                    and frame.token not in completed and frame.wire > 0 and frame.opcode > 0,
                    'invalid or duplicate frame association')
            first, cycle = self.reads[frame.token - 1], self.cycles[frame.cycle - 1]
            peer = self.target_peer(cycle, frame.connection)
            require(peer is not None and (peer.pid, peer.start) == (frame.pid, frame.start)
                    and first.connection == frame.connection and first.first_read == frame.first_read > 0
                    and first.cycle <= frame.cycle and cycle.entry <= frame.received <= cycle.exit
                    and frame.first_read <= frame.received and frame.token in token_last
                    and token_last[frame.token][0] == frame.cycle
                    and token_last[frame.token][1] <= frame.received, 'frame clock/owner/last read differs')
            next_read = token_last[frame.token][2] + 1
            require(next_read == len(self.reads) or frame.received <= self.reads[next_read].entry,
                    'frame completion crosses the next receive observation')
            key = (frame.connection, frame.pid, frame.start, frame.wire)
            require(key not in self.by_wire, 'duplicate authenticated frame wire')
            self.by_wire[key] = frame
            completed.add(frame.token)
            last_cycle, last_stamp = frame.cycle, frame.received
        require(first_count == len(self.frames) and set(token_last) == completed,
                'dangling, missing or duplicate first-positive associations')

    def partition(self, frame, accepted_ns, *, fully_flushed=True):
        """Return a disjoint exhaustive wall partition; no CPU/readiness causality."""
        first = self.reads[frame.token - 1]
        begin, end = accepted_ns, frame.first_read
        require(type(begin) is int and begin > 0 and first.first_read == end,
                'invalid accepted/first-read boundary')
        if begin >= end:
            return []
        if not fully_flushed:
            return [(begin, end, 'unresolved')]
        candidates = []

        def add(left, right, phase):
            left, right = max(begin, left), min(end, right)
            if left < right:
                candidates.append((left, right, phase))

        index = bisect_right(self.ends, begin)
        while index < len(self.cycles) and self.cycles[index].entry < end:
            cycle = self.cycles[index]
            peer = self.target_peer(cycle, frame.connection)
            if index:
                previous = self.cycles[index - 1]
                prior_peer = self.target_peer(previous, frame.connection)
                if peer and prior_peer and not peer.flags and not prior_peer.flags:
                    add(previous.exit, cycle.entry, 'outside_poll_call')
            reads = [self.reads[item] for item in range(self.read_offsets[index], self.read_offsets[index + 1])
                     if self.reads[item].connection == frame.connection]
            eligible = peer and not peer.flags and peer.events & 1 and peer.revents & 1 and peer.alive
            eligible = eligible and all(not row.token or row.token == frame.token for row in reads)
            if eligible and cycle.poll_count:
                polls = [self.polls[item] for item in range(cycle.poll_begin, cycle.poll_begin + cycle.poll_count)]
                add(cycle.entry, polls[0].entry, 'call_setup')
                for poll in polls:
                    add(poll.entry, poll.returned, 'ppoll')
                add(polls[-1].returned, peer.visit, 'poll_return_to_target_visit')
                if cycle.id == first.cycle:
                    cursor = peer.visit
                    for read in reads:
                        if read.entry >= first.entry:
                            break
                        # Unsuccessful attempts cannot be retrospectively assigned a wire.
                        add(cursor, read.entry, 'target_visit_to_recv_entry')
                        cursor = read.returned
                    add(cursor, first.entry, 'target_visit_to_recv_entry')
                    add(first.entry, first.returned, 'recv_syscall')
                    add(first.returned, first.first_read, 'positive_return_to_first_read')
            index += 1
        result, cursor = [], begin
        for left, right, phase in sorted(candidates):
            require(cursor <= left < right, 'overlapping poll phase intervals')
            if cursor < left:
                result.append((cursor, left, 'unresolved'))
            result.append((left, right, phase))
            cursor = right
        if cursor < end:
            result.append((cursor, end, 'unresolved'))
        require(sum(right - left for left, right, _ in result) == end - begin, 'poll parent duration lost')
        return result


def summarize(journal, selected, gaps, window):
    """Keep exact active multisets and both independent sum and union checks."""
    import run_bert_delivery_boundary as delivery
    begin, end = window
    spans, parents, waits, receipts = [], [], [], []
    arithmetic = Counter({phase: 0 for phase in PHASES})
    starts = [left for left, _ in gaps]
    ends = [right for _, right in gaps]

    def overlap(left, right):
        index, total = bisect_right(ends, left), 0
        while left < right and index < len(starts) and starts[index] < right:
            total += max(0, min(right, ends[index]) - max(left, starts[index]))
            index += 1
        return total
    for row in selected:
        if not begin <= row['send_end'] < end:
            continue
        frame = row['frame']
        left, right = row['send_end'], min(end, frame.first_read)
        if left < right:
            parents.append((left, right, (row['wire'],)))
        if row['send_end'] < min(end, row['reply']):
            waits.append((row['send_end'], min(end, row['reply']), (row['wire'],)))
        phase_sums = Counter({phase: 0 for phase in PHASES})
        for a, b, phase in journal.partition(frame, row['send_end'], fully_flushed=row['fully_flushed']):
            a, b = max(begin, a), min(end, b)
            if a >= b:
                continue
            spans.append((a, b, (row['wire'], phase)))
            # Independent direct intersection, not the active-multiset sweep.
            elapsed = overlap(a, b)
            phase_sums[phase] += elapsed
        expected = overlap(left, right)
        require(sum(phase_sums.values()) == expected, 'per-launch poll parent intersection differs')
        arithmetic.update(phase_sums)
        receipts.append([row['wire'], frame.token, journal.reads[frame.token - 1].cycle,
                         row['send_end'], frame.first_read, row['reply'], *[phase_sums[p] for p in PHASES]])
    buckets, missing = delivery.intake.partition(gaps, spans)
    parent_buckets, parent_missing = delivery.intake.partition(gaps, parents)
    wait_buckets, _ = delivery.intake.partition(gaps, waits)
    collapsed, weighted = Counter(), Counter({phase: 0 for phase in PHASES})
    exclusive = Counter({phase: 0 for phase in PHASES})
    mixed = 0
    for active, elapsed in buckets.items():
        bare = Counter()
        for signature, count in active:
            bare[signature[:-1]] += count
            weighted[signature[-1]] += elapsed * count
        collapsed[tuple(sorted(bare.items()))] += elapsed
        phases = {signature[-1] for signature, _ in active}
        if len(phases) == 1:
            exclusive[next(iter(phases))] += elapsed
        else:
            mixed += elapsed
    require(collapsed == parent_buckets and missing == parent_missing and weighted == arithmetic,
            'poll partition changed exact parent multiset or weighted durations')
    total = sum(parent_buckets.values())
    require(sum(exclusive.values()) + mixed == total, 'exclusive/mixed union lost parent duration')
    chosen = [phase for phase, value in exclusive.items() if phase != 'unresolved' and total and 5 * value >= 3 * total]
    whole_gap = delivery.intake.duration(gaps)
    return dict(primary_launches=len(receipts), whole_observation_launches=len(selected),
        accepted_to_first_read_gap_union_ns=total, launch_wait_gap_union_ns=sum(wait_buckets.values()),
        service_supply_gap_ns=whole_gap, fraction_of_whole_gap=total / whole_gap if whole_gap else None,
        exclusive_phase_ns=dict(exclusive), mixed_phase_ns=mixed, weighted_phase_ns=dict(weighted),
        phase_fraction_of_parent={name: value / total if total else None for name, value in exclusive.items()},
        next_boundary=chosen[0] if len(chosen) == 1 else 'mixed_or_inconclusive',
        primary_receipt_fields=['wire', 'frame_token', 'first_read_cycle', 'accepted_ns', 'first_read_ns',
                                'reply_ns', *[phase + '_gap_ns' for phase in PHASES]],
        primary_receipts=receipts,
        buckets=[dict(active_multiset=[dict(signature=key, count=count) for key, count in active],
                      duration_ns=elapsed) for active, elapsed in sorted(buckets.items())])


def bind_analyzer(inputs, scope):
    """Compare every maintained analysis dependency to the frozen capture seal."""
    import analyze_bert_frontend_lifecycle as frontend
    import run_bert_delivery_boundary as delivery
    from run_rnn_v8_validation import digest
    plan_path = Path(scope['plan']['path'])
    plan = inputs.read(plan_path, scope['plan']['sha256'])
    require(plan['status'] == 'frozen_poll_boundary_application'
            and plan['decision_role'] == 'inference', 'frozen inference decision required')
    source = plan['analysis_source']
    require(source['path'] == str(Path(__file__).resolve())
            and source['sha256'] == digest(Path(__file__)), 'direct frozen poll analyzer identity differs')
    seal = inputs.read(plan_path.with_name('build-seal.json'))
    require(seal['plan_sha256'] == scope['plan']['sha256'], 'poll analysis plan changed')
    require(seal['identities'].get(source['path']) == source['sha256'], 'analyzer plan/seal binding differs')
    for module in (frontend, delivery, delivery.intake, delivery.lifecycle,
                   delivery.lifecycle.frontend, delivery.lifecycle.base):
        path = Path(module.__file__).resolve()
        require(str(path) in seal['identities'] and digest(path) == seal['identities'][str(path)],
                'frozen poll analysis dependency differs')
        inputs.identities[str(path)] = seal['identities'][str(path)]
    path = Path(__file__).resolve()
    require(str(path) in seal['identities'] and digest(path) == seal['identities'][str(path)],
            'frozen poll analyzer differs')
    inputs.identities[str(path)] = seal['identities'][str(path)]
    for name, checksum in seal['identities'].items():
        path = Path(name)
        if path.parent == delivery.PROJECT_ROOT / 'scripts' and path.suffix == '.py':
            require(digest(path) == checksum, 'frozen analysis support changed')
            inputs.identities[name] = checksum


def analyze(attempt, scope):
    """Join this capture's immutable owner/transport/frontend evidence only."""
    import analyze_bert_frontend_lifecycle as frontend
    import run_bert_delivery_boundary as delivery
    from run_rnn_v8_validation import digest
    inputs = delivery.BoundInputs()
    bind_analyzer(inputs, scope)
    worker = inputs.read(attempt / 'equal-worker.json')
    job = inputs.read(attempt / 'equal/helper-attempt.json')
    request = inputs.read(attempt / 'request-analysis.json')
    coarse = inputs.read(attempt / 'frontend-analysis.json')
    window = coarse['begin_ns'], coarse['end_ns']
    require(window[1] - window[0] == 5_000_000_000, 'fixed five-second window required')
    stage_receipt, poll_receipt = worker['stage_output'], worker['poll_journal']
    document = inputs.read(Path(stage_receipt['path']), stage_receipt['sha256'])
    delivery.validate_intake(document)
    journal_path = Path(poll_receipt['path'])
    inputs.identities[str(journal_path)] = digest(journal_path)
    require(inputs.identities[str(journal_path)] == poll_receipt['sha256'], 'poll ledger changed')
    states = frontend.service(document, *window)
    retained, outcomes = {}, {}
    with Journal(journal_path, job['server_identity'], application=True) as journal:
        require(journal.counts == poll_receipt['counts'], 'normal receipt counts differ')
        joined_frames = set()
        for frame_ns, owner_ns, retained_ns, index, wire, opcode, first_read in document['intake']['records']:
            connection, pid, start = document['slots'][index]
            key = (connection, pid, start, wire)
            require(key in journal.by_wire and key not in joined_frames, 'missing/duplicate retained frame join')
            frame = journal.by_wire[key]
            require((frame.first_read, frame.received, frame.opcode) == (first_read, frame_ns, opcode),
                    'journal does not match unchanged intake boundaries')
            joined_frames.add(key)
            retained[index, wire] = (frame, owner_ns, retained_ns)
        require(joined_frames == journal.by_wire.keys(), 'unretained journal frames cannot qualify complete evidence')
        for _, index, kind, wire, command, detail in document['lifecycle']['events']:
            if kind == 4:
                require((index, wire) not in outcomes, 'duplicate lifecycle reply')
                outcomes[index, wire] = (detail & 0xffffffff, detail >> 32, command)
        roles = {}
        for role, identity in job['application_identities'].items():
            owned = [(index, slot, state) for index, (slot, state) in enumerate(zip(document['slots'], states))
                     if tuple(slot[1:]) == (identity['pid'], identity['start']) and state['kind'] == 0]
            require(len(owned) == 1, 'exactly one compute connection per application role required')
            index, slot, state = owned[0]
            coverage = [row for row in request['request_lifecycle']['slots']
                        if row['role'] == role and row['compute_connection']]
            require(len(coverage) == 1 and coverage[0]['slot'] == slot and coverage[0]['coverage_gate']
                    and .95 <= coverage[0]['lifecycle_coverage'] <= 1, 'complete role lifecycle coverage required')
            receipt = worker['frontend_outputs'][role]
            require(receipt['application_identity'] == identity, 'frontend role identity differs')
            observed = inputs.read(Path(receipt['path']), receipt['sha256'])
            flushed = delivery.validate_delivery(observed)
            _, calls, _ = frontend.frontend(observed, identity, *window)
            matched, selected = set(), []
            expected = {wire for wire, call in state['calls'].items()
                        if call['receive'] < window[1] and call['reply'] >= window[0]}
            for row in calls:
                if row['session'] != slot[0] or not row['wire']:
                    continue
                wire = row['wire']
                require(wire in state['calls'] and (index, wire) in retained and (index, wire) in outcomes,
                        'frontend compute wire lacks retained/reply evidence')
                frame, owner_ns, retained_ns = retained[index, wire]
                call = state['calls'][wire]
                require(row['opcode'] == frame.opcode == call['opcode']
                        and row['send_begin'] <= frame.first_read <= frame.received <= owner_ns
                        <= retained_ns == call['receive'] <= call['reply'] <= row['reply'],
                        'exact frontend/transport/service chronology differs')
                if wire in expected:
                    matched.add(wire)
                disposition, status, command = outcomes[index, wire]
                if row['opcode'] == 11 and disposition == 4:
                    require(status == 0 and command > 0, 'accepted launch lacks successful real command')
                    selected.append(dict(wire=wire, send_end=row['send_end'], reply=row['reply'], frame=frame,
                                         fully_flushed=flushed[row['session'], wire]))
            require(expected == matched and matched and selected, 'complete fixed-window compute join required')
            gaps = delivery.intake.union(state['gaps'], *window)
            result = summarize(journal, selected, gaps, window)
            require(result['service_supply_gap_ns'] == coverage[0]['zero_wire_and_zero_unissued_ns'],
                    'role supply-gap denominator changed')
            if role != 'inference':
                result['next_boundary'] = 'descriptive_only'
            roles[role] = dict(identity=identity, compute_slot=slot, correlated_compute_wires=len(matched), **result)
        result = dict(status='poll_boundary_requires_independent_review', begin_ns=window[0], end_ns=window[1],
            journal_counts=journal.counts, joined_frames=len(joined_frames), roles=roles,
            decision_role='inference', next_boundary=roles['inference']['next_boundary'],
            lifecycle_scope=scope, raw_inputs=inputs.identities, diagnostic_only=True,
            full_device_qualified=False, performance_qualified=False, saturation_qualified=False,
            priority_qualified=False, gpu_gap_causality_qualified=False)
    inputs.verify()
    encoded = json.dumps(result, indent=2, allow_nan=False) + '\n'
    require(len(encoded.encode()) < 32 * 1024**2, 'bounded poll analysis output required')
    with (attempt / 'poll-analysis.json').open('x') as output:
        output.write(encoded)
    inputs.verify()
    print(json.dumps(dict(decision_role='inference', next_boundary=result['next_boundary'])))
    return 0
