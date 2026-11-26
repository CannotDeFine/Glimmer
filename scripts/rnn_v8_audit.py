"""Independent raw-receipt audit for the fixed six-case SDK9 LSTM input."""

from collections import Counter
import gzip
import json
import math
from pathlib import Path
import struct

GUARD = b'\xa7' * 256
INPUTS = {'x', 'dy', 'hx', 'cx', 'dhy', 'dcy', 'weight', 'lengths'}
OUTPUTS = {'y', 'dx', 'hy', 'cy', 'dhx', 'dcx', 'dweight'}
OPAQUE = {'workspace', 'reserve'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def seeds():
    p = []
    for gate in range(8):
        p.extend(((gate * 5 + i * 3) % 17 + 1) / (-128 if gate % 2 else 128)
                 for i in range(4))
        p.extend((gate * 2 + i + 1) / (256 if gate % 3 else -256) for i in range(2))
    return p, dict(x=[(i + 1) / (-32 if i % 2 else 32) for i in range(12)],
                   hx=[(i + 1) / (-16 if i % 2 else 16) for i in range(4)],
                   cx=[(i + 2) / (32 if i % 2 else -32) for i in range(4)],
                   dy=[(i + 3) / (64 if i % 3 else -64) for i in range(12)],
                   dhy=[(i + 2) / (32 if i % 2 else -32) for i in range(4)],
                   dcy=[(i + 4) / (-32 if i % 2 else 32) for i in range(4)])


def forward(p, x, hx, cx, lengths):
    y, hy, cy = [0.] * 12, [0.] * 4, [0.] * 4
    for batch, length in enumerate(lengths):
        h, c = hx[2 * batch:2 * batch + 2], cx[2 * batch:2 * batch + 2]
        for step in range(length):
            start = 4 * step + 2 * batch
            gates = []
            for gate in range(4):
                values = []
                for row in range(2):
                    z = p[6 * gate + 4 + row] + p[6 * (gate + 4) + 4 + row]
                    for col in range(2):
                        z += p[6 * gate + 2 * row + col] * x[start + col]
                        z += p[6 * (gate + 4) + 2 * row + col] * h[col]
                    values.append(math.tanh(z) if gate == 2 else 1 / (1 + math.exp(-z)))
                gates.append(values)
            c = [gates[1][r] * c[r] + gates[0][r] * gates[2][r] for r in range(2)]
            h = [gates[3][r] * math.tanh(c[r]) for r in range(2)]
            y[start:start + 2] = h
        hy[2 * batch:2 * batch + 2], cy[2 * batch:2 * batch + 2] = h, c
    return dict(y=y, hy=hy, cy=cy)


def reference(lengths):
    """Central differences, independent of the input's analytic backprop code."""
    p, inputs = seeds()
    arguments = [p, inputs['x'], inputs['hx'], inputs['cx']]

    def loss():
        out = forward(*arguments, lengths)
        return sum(a * b for name in ('y', 'hy', 'cy')
                   for a, b in zip(out[name], inputs['d' + name], strict=True))

    gradients = []
    for argument in arguments:
        values = []
        for index, value in enumerate(argument):
            argument[index] = value + 1e-6
            plus = loss()
            argument[index] = value - 1e-6
            minus = loss()
            argument[index] = value
            values.append((plus - minus) / 2e-6)
        gradients.append(values)
    return dict(**forward(*arguments, lengths), dx=gradients[1], dhx=gradients[2],
                dcx=gradients[3]), gradients[0]


def sequence(values, layout, lengths):
    indices = ([(t, b) for b in range(2) for t in range(3)] if layout == 2 else
               [(t, b) for t in range(3) for b in range(2) if layout != 1 or t < lengths[b]])
    return [values[(2 * t + b) * 2 + c] for t, b in indices for c in range(2)]


def payload(record):
    raw = bytes.fromhex(record['guarded_hex'])
    require(record['guard_bytes'] == 256 and len(raw) == record['payload_bytes'] + 512 and
            raw[:256] == GUARD and raw[-256:] == GUARD, 'complete raw bytes and every guard byte')
    return raw[256:-256]


def close(raw, expected, code):
    width = struct.calcsize(code)
    require(len(raw) == len(expected) * width, 'complete logical output extent')
    values = struct.unpack('<' + code * len(expected), raw)
    for actual, wanted in zip(values, expected, strict=True):
        tolerance = (2e-9 + 2e-8 * abs(wanted) if code == 'd' else 3e-5 + 3e-4 * abs(wanted))
        require(math.isfinite(actual) and abs(actual - wanted) <= tolerance,
                f'unchanged independent numerical oracle: {actual} versus {wanted}')


def one(records, kind):
    found = [r for r in records if r['kind'] == kind]
    require(len(found) == 1, 'exactly one ' + kind)
    return found[0]


def audit_layout(records, config, width, dtype, weight_bytes, bases):
    queries = [r for r in records if r['kind'] == 'query']
    require(len(queries) == 32 and [r['query'] for r in queries] ==
            list(range((config - 1) * 32 + 1, config * 32 + 1)), 'all32 ordered original queries')
    shapes = [r for r in records if r['kind'] == 'shape']
    require(len(shapes) == 55, 'all55 existing shape observations')
    for item in shapes:
        require(item['status'] == 0 and item['rank'] in (0, 3), 'original bounded shape')
        for name in ('dimensions', 'strides'):
            array = item[name]
            require(array['before'] == 0x123456789abcdef0 and
                    array['after'] == 0xfedcba9876543210 and len(array['values']) == 8 and
                    array['values'][item['rank']:] == [-8123] * (8 - item['rank']),
                    'all shape guards and unused output capacity')
    locations = [r for r in records if r['kind'] == 'scalar_location']
    require(len(locations) == 144, 'three independent48-scalar location maps')
    mappings = []
    for group, base in enumerate((*bases, 0)):
        offsets = []
        for linear in range(8):
            q = queries[group * 8 + linear]
            require(q['status'] == 0 and q['linear'] == linear and q['base'] == base and
                    q['bytes'] == weight_bytes and q['matrix_descriptor'] and q['bias_descriptor']
                    and q['matrix_descriptor'] != q['bias_descriptor'] and q['matrix_output'] and
                    q['bias_output'] and q['matrix_output'] != q['bias_output'], 'original mapping query')
            found = [r for r in shapes if r['query'] == q['query']]
            require(len(found) == 2, 'both native parameter shapes')
            expected_locations = []
            for part, name in enumerate(('matrix', 'bias')):
                shape = found[part]
                require(shape['descriptor'] == q[name + '_descriptor'] and shape['type'] == dtype and
                        shape['rank'] == 3 and shape['dimensions']['values'][:3] == [1, 2, 1 if part else 2]
                        and all(s > 0 for s in shape['strides']['values'][:3]), 'native logical shape')
                start = q[name + '_address'] - base
                strides = shape['strides']['values']
                for row in range(2):
                    for col in range(1 if part else 2):
                        offset = start + width * (row * strides[1] + col * strides[2])
                        require(start >= 0 and 0 <= offset <= weight_bytes - width and offset % width == 0,
                                'authoritative complete scalar offset')
                        expected_locations.append((linear, part, row, col, offset))
                        offsets.append(offset)
            actual = [(r['linear'], r['part'], r['row'], r['column'], r['offset'])
                      for r in locations if r['query'] == q['query']]
            require(actual == expected_locations, 'reported locations agree with raw native metadata')
        require(len(set(offsets)) == 48, 'all logical parameters are nonoverlapping')
        mappings.append(offsets)
    require(mappings[0] == mappings[1] == mappings[2], 'weight/gradient/null-base maps agree')
    for index, q in enumerate(queries[24:]):
        require(q['status'] == 0 and q['linear'] == (0 if index < 4 else 8), 'optional query status')
        if index < 4:
            require(q['base'] == bases[0] and q['bytes'] == weight_bytes, 'optional original range')
            if index != 1:
                require(q['matrix_address'] == bases[0] + mappings[0][4 if index == 3 else 0],
                        'matrix/aliased address output')
            if index != 0:
                require(q['bias_address'] == bases[0] + mappings[0][4], 'bias address output')
        else:
            require(q['bytes'] == 0 and q['matrix_address'] == q['bias_address'] == 0,
                    'absent projection at zero capacity')
    a, b, alias, host_alias, *absent = queries[24:]
    require(a['matrix_descriptor'] and a['matrix_output'] and not a['bias_descriptor'] and
            not a['bias_output'] and not b['matrix_descriptor'] and not b['matrix_output'] and
            b['bias_descriptor'] and b['bias_output'] and
            alias['matrix_descriptor'] == alias['bias_descriptor'] != 0 and
            host_alias['matrix_output'] == host_alias['bias_output'] != 0,
            'unchanged optional/alias controls')
    require([q['base'] for q in absent] == [bases[0], bases[0] + weight_bytes + 256, 0, 0] and
            all(absent[-1][name] == 0 for name in
                ('matrix_descriptor', 'bias_descriptor', 'matrix_output', 'bias_output')),
            'original endpoint/null/absent controls')
    extra = [r for r in shapes if r['query'] > queries[23]['query']]
    require(extra[0]['query'] == alias['query'] and extra[0]['rank'] == 3 and
            extra[0]['dimensions']['values'][:3] == [1, 2, 1] and
            [r['query'] for r in extra[1:]] == [q['query'] for q in absent[:3] for _ in range(2)] and
            all(r['rank'] == 0 for r in extra[1:]), 'alias shape and absent descriptors')
    return mappings[0]


def audit_configuration(records, config):
    """Require all original bytes/math for one genuinely completed configuration."""
    checked_scalars = 0
    selected = [r for r in records if r.get('configuration') == config]
    case, capacities = one(selected, 'configuration'), one(selected, 'capacities')
    dtype, layout = (0 if config % 2 else 1), (config - 1) // 2
    require((case['type'], case['layout']) == (dtype, layout), 'unchanged type/layout order')
    code, width = ('f', 4) if dtype == 0 else ('d', 8)
    lengths = [3, 2] if layout == 1 else [3, 3]
    buffers = [r for r in selected if r['kind'] == 'buffer']
    require(len(buffers) == 86 and all(r['name'] in INPUTS | OUTPUTS | OPAQUE for r in buffers),
            'exact original reset/read observations')
    initial = {r['name']: r for r in buffers if r['iteration'] == 0 and r['action'] == 'reset'}
    require(set(initial) == INPUTS | OUTPUTS | OPAQUE, 'all17 original allocations')
    for r in buffers:
        payload(r)
        require(r['base'] == initial[r['name']]['base'] and
                r['payload_bytes'] == initial[r['name']]['payload_bytes'], 'stable owned allocation')
    size = capacities['weight_bytes']
    require(0 < size <= 65536 and initial['weight']['payload_bytes'] == size and
            initial['dweight']['payload_bytes'] == size, 'bounded native weight capacities')
    for name in OPAQUE:
        require(0 <= capacities[name + '_bytes'] <= 64 * 1024 ** 2 and
                initial[name]['payload_bytes'] == capacities[name + '_bytes'], 'opaque capacities')
    offsets = audit_layout(selected, config, width, dtype, size,
                           [initial[n]['base'] + 256 for n in ('weight', 'dweight')])
    p, inputs = seeds()
    for name, values in inputs.items():
        values = sequence(values, layout, lengths) if name in ('x', 'dy') else values
        require(payload(initial[name]) == struct.pack('<' + code * len(values), *values),
                'frozen input bytes: ' + name)
    require(payload(initial['lengths']) == struct.pack('<ii', *lengths), 'frozen device lengths')
    weight, gradient = bytearray(size), bytearray(size)
    for offset, value in zip(offsets, p, strict=True):
        struct.pack_into('<' + code, weight, offset, value)
        struct.pack_into('<' + code, gradient, offset, value * .25 + .125)
    require(payload(initial['weight']) == weight and payload(initial['dweight']) == gradient,
            'entire nonzero initial weight/ADD gradient bytes including padding')
    values, derivatives = reference(lengths)
    receipts = [r for r in selected if r['kind'] == 'iteration']
    require([r['iteration'] for r in receipts] == [1, 2] and
            all(r['complete_oracle_passed'] is True and r['mapped_parameters'] == 48 and
                math.isfinite(r['cpu_finite_difference_error']) and
                r['cpu_finite_difference_error'] < 2e-9 for r in receipts), 'two complete oracle cycles')
    for iteration in (1, 2):
        reads = [r for r in buffers if r['iteration'] == iteration and r['action'] == 'read']
        require(Counter(r['name'] for r in reads) ==
                Counter({n: 2 if n in OUTPUTS else 1 for n in INPUTS | OUTPUTS | OPAQUE}),
                'all17 guarded allocations plus original seven duplicate snapshots')
        current = {}
        for r in reads:
            raw = payload(r)
            require(r['name'] not in current or current[r['name']] == raw, 'same-cycle duplicate read')
            current[r['name']] = raw
        for name in INPUTS:
            require(current[name] == payload(initial[name]), 'complete immutable input: ' + name)
        for name, wanted in values.items():
            wanted = sequence(wanted, layout, lengths) if name in ('y', 'dx') else wanted
            close(current[name], wanted, code)
            checked_scalars += len(wanted)
        for offset, derivative in zip(offsets, derivatives, strict=True):
            start = struct.unpack_from('<' + code, gradient, offset)[0]
            close(current['dweight'][offset:offset + width], [start + iteration * derivative], code)
            checked_scalars += 1
    require(sum(r['kind'] == 'device_release' and r['status'] == 0 for r in selected) == 17,
            'all17 allocation releases for this completed configuration')
    return checked_scalars


def audit_records(records, cached):
    sdk = one(records, 'sdk')
    require(sdk['runtime_version'] == sdk['header_version'] == 91002 and sdk['cached'] is cached,
            'pinned SDK9 version and selected lookup mode')
    result = one(records, 'rnn_v8_result')
    expected = dict(parameter_queries=192, numerical_cases=12, forward_calls=12,
                    backward_data_calls=12, backward_weight_calls=12, resources_destroyed=81,
                    device_releases=102, live_libraries=0, logical_parameter_gradient_elements=576,
                    logical_output_and_gradient_elements=1040)
    require(result['status'] == 'passed' and result['normal_cleanup'] is True and
            all(result[k] == v for k, v in expected.items()), 'complete normal numerical/lifecycle receipt')
    flags = ('opaque_padding_gradient_qualified', 'temporary_mathematics_qualified',
             'all_rnn_models_qualified', 'full_device_qualified', 'saturation_qualified',
             'persistent_qualified', 'priority_qualified', 'performance_qualified', 'full_goal_qualified')
    require(all(result[k] is False for k in flags), 'unqualified scopes remain false')
    require(all(r['status'] == 0 for r in records if r['kind'] in
                ('status', 'resource_destroy', 'device_release')), 'every original call and cleanup succeeds')
    require(sum(r['kind'] == 'resource_destroy' for r in records) == 81 and
            sum(r['kind'] == 'device_release' for r in records) == 102, 'individual normal cleanup records')
    require([r['configuration'] for r in records if r['kind'] == 'configuration'] == list(range(1, 7)),
            'exact six configurations in original order')
    require(all(r.get('aux_flags') == (0 if r['layout'] == 1 else 1)
                for r in records if r['kind'] == 'configuration'),
            'packed disables and both unpacked layouts explicitly enable padded I/O')
    checked_scalars = sum(audit_configuration(records, config) for config in range(1, 7))
    require(checked_scalars == 1040, 'all1040 output/gradient scalars independently recomputed')
    return dict(logical_scalars=checked_scalars, parameter_queries=192, guarded_records=516,
                normal_cleanup=True, full_device_qualified=False)


def audit(path, cached):
    with open_receipt(path) as stream:
        records = [json.loads(line) for line in stream]
    return audit_records(records, cached)


def open_receipt(path):
    path = Path(path)
    return gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()


def audit_partial(path, cached):
    """Audit completed configurations without inventing a final route receipt."""
    records, partial_tail_bytes = [], 0
    with open_receipt(path) as stream:
        for line in stream:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                require(stream.read() == '', 'only the final interrupted JSON record may be incomplete')
                partial_tail_bytes = len(line.encode())
                break
    require(not any(r['kind'] == 'rnn_v8_result' for r in records), 'partial audit is not full-route acceptance')
    sdk = one(records, 'sdk')
    require(sdk['runtime_version'] == sdk['header_version'] == 91002 and sdk['cached'] is cached,
            'partial pinned SDK identity')
    completed = [config for config in range(1, 7) if
                 [r['iteration'] for r in records if r['kind'] == 'iteration' and
                  r.get('configuration') == config] == [1, 2] and
                 sum(r['kind'] == 'device_release' and r.get('configuration') == config
                     for r in records) == 17]
    require(completed, 'at least one genuinely completed configuration')
    scalars = sum(audit_configuration(records, config) for config in completed)
    return dict(status='partial_configurations_independently_audited', configurations=completed,
                logical_scalars=scalars, interrupted_tail_bytes=partial_tail_bytes,
                full_route_qualified=False, normal_route_cleanup=False, full_device_qualified=False)
