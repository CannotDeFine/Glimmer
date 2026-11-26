#!/usr/bin/env python3
"""Audit the unchanged fixed SDK8 LSTM receipts; opaque mathematics stays unqualified."""

import gzip
import json
import math
from pathlib import Path
import struct

GUARD = bytes([0xa7]) * 256
INPUT_NAMES = {'x', 'dy', 'hx', 'cx', 'dhy', 'dcy', 'weight'}
OUTPUT_NAMES = {'y', 'dx', 'hy', 'cy', 'dhx', 'dcx', 'dweight'}
OPAQUE_NAMES = {'dropout_state', 'workspace', 'reserve'}
LEGACY_CALLS = {
    'cudnnOpsInferVersionCheck', 'cudnnOpsTrainVersionCheck', 'cudnnCnnInferVersionCheck',
    'cudnnCnnTrainVersionCheck', 'cudnnAdvInferVersionCheck', 'cudnnAdvTrainVersionCheck',
    'cudnnCreateAlgorithmDescriptor', 'cudnnSetAlgorithmDescriptor', 'cudnnDestroyAlgorithmDescriptor',
    'cudnnSetRNNDescriptor_v6', 'cudnnSetRNNMatrixMathType', 'cudnnGetRNNMatrixMathType',
    'cudnnSetRNNBiasMode', 'cudnnGetRNNBiasMode', 'cudnnRNNSetClip', 'cudnnGetRNNWorkspaceSize',
    'cudnnGetRNNTrainingReserveSize', 'cudnnGetRNNParamsSize', 'cudnnGetRNNLinLayerMatrixParams',
    'cudnnGetRNNLinLayerBiasParams', 'cudnnSetRNNAlgorithmDescriptor',
    'cudnnGetRNNForwardInferenceAlgorithmMaxCount', 'cudnnRNNForwardTraining',
    'cudnnRNNBackwardData', 'cudnnRNNBackwardWeights',
}


def require(value, message):
    if not value:
        raise ValueError(message)


def seeds():
    parameters = []
    for gate in range(8):
        parameters.extend(((gate * 5 + i * 3) % 17 + 1) / (-128 if gate % 2 else 128)
                          for i in range(4))
        parameters.extend((gate * 2 + i + 1) / (256 if gate % 3 else -256) for i in range(2))
    x = [(i + 1) / (-32 if i % 2 else 32) for i in range(10)]
    hx = [(i + 1) / (-16 if i % 2 else 16) for i in range(4)]
    cx = [(i + 2) / (32 if i % 2 else -32) for i in range(4)]
    dy = [(i + 3) / (64 if i % 3 else -64) for i in range(10)]
    dhy = [(i + 2) / (32 if i % 2 else -32) for i in range(4)]
    dcy = [(i + 4) / (-32 if i % 2 else 32) for i in range(4)]
    return parameters, x, hx, cx, dy, dhy, dcy


def forward(parameters, x, hx, cx):
    y, hy, cy = [0.] * 10, [0.] * 4, [0.] * 4
    for batch, length in enumerate((3, 2)):
        h, c = hx[2 * batch:2 * batch + 2], cx[2 * batch:2 * batch + 2]
        for step in range(length):
            start = 4 * step + 2 * batch
            gates = []
            for gate in range(4):
                values = []
                for row in range(2):
                    value = parameters[6 * gate + 4 + row] + parameters[6 * (gate + 4) + 4 + row]
                    for col in range(2):
                        value += parameters[6 * gate + 2 * row + col] * x[start + col]
                        value += parameters[6 * (gate + 4) + 2 * row + col] * h[col]
                    values.append(math.tanh(value) if gate == 2 else 1 / (1 + math.exp(-value)))
                gates.append(values)
            c = [gates[1][row] * c[row] + gates[0][row] * gates[2][row] for row in range(2)]
            h = [gates[3][row] * math.tanh(c[row]) for row in range(2)]
            y[start:start + 2] = h
        hy[2 * batch:2 * batch + 2], cy[2 * batch:2 * batch + 2] = h, c
    return y, hy, cy


def reference():
    p, x, hx, cx, dy, dhy, dcy = seeds()
    arguments = [p, x, hx, cx]

    def loss():
        y, hy, cy = forward(*arguments)
        return sum(a * b for left, right in ((y, dy), (hy, dhy), (cy, dcy))
                   for a, b in zip(left, right, strict=True))

    gradients = []
    for argument in arguments:
        gradient = []
        for index, value in enumerate(argument):
            argument[index] = value + 1e-6
            plus = loss()
            argument[index] = value - 1e-6
            minus = loss()
            argument[index] = value
            gradient.append((plus - minus) / 2e-6)
        gradients.append(gradient)
    y, hy, cy = forward(*arguments)
    return dict(y=y, hy=hy, cy=cy, dx=gradients[1], dhx=gradients[2], dcx=gradients[3]), gradients[0]


def floats(payload):
    require(len(payload) % 4 == 0, 'complete aligned float payload')
    return list(struct.unpack('<' + 'f' * (len(payload) // 4), payload))


def close(observed, expected):
    require(len(observed) == len(expected), 'complete scalar extent')
    for actual, wanted in zip(observed, expected, strict=True):
        require(math.isfinite(actual) and abs(actual - wanted) <= 3e-5 + 3e-4 * abs(wanted),
                f'independent full finite-difference/forward numerical mismatch: {actual} vs {wanted}')


def audit_job(path, cached, remote, sdk, seals):
    opener = gzip.open(path, 'rt') if Path(path).suffix == '.gz' else Path(path).open()
    with opener as stream:
        records = [json.loads(line) for line in stream]

    def one(kind):
        values = [item for item in records if item['kind'] == kind]
        require(len(values) == 1, 'exactly one ' + kind + ' record')
        return values[0]

    header, model, result = one('sdk'), one('model'), one('legacy_sdk_result')
    require(header['header_version'] == 8907 and header['cached'] is cached, 'exact SDK8/lookup identity')
    expected_model = dict(layers=1, input=2, hidden=2, steps=3, batch=2, lengths=[3, 2],
                          packed_floats=10, seed=17, dropout=0, algorithm='standard', type='float',
                          bias='double', direction='uni', input_mode='linear', math='default',
                          clip='none', tensor_descriptors=18)
    require(all(model.get(key) == value for key, value in expected_model.items()), 'frozen LSTM model')
    require(math.isfinite(model['central_difference_max_error']) and
            model['central_difference_max_error'] < 2e-9, 'unchanged analytic oracle self-check')
    require(0 < model['weight_bytes'] <= 65536 and model['algorithm_count'] > 0,
            'bounded original weight/inference queries')
    for name in ('workspace', 'reserve', 'dropout'):
        require(0 <= model[name + '_bytes'] <= 64 * 1024 ** 2, 'bounded original temporary query')
    mapped = set()
    for name in one('providers')['paths']:
        provider = Path(name).resolve(strict=True)
        require(provider.parent == sdk / 'lib' and str(provider) in seals, 'actual sealed provider')
        mapped.add(str(provider))
    require(1 <= len(mapped) <= 7 and (remote or len(mapped) == 7), 'actual selected SDK8 mappings')
    require(result['status'] == 'passed' and result['iterations'] == 2 and
            result['mapped_parameters'] == 48 and result['normal_cleanup'] is True and
            result['live_resources'] == 0, 'complete normal SDK lifecycle')
    require(all(result.get(name) is False for name in (
        'opaque_padding_mathematics_qualified', 'temporary_mathematics_qualified',
        'full_device_qualified', 'saturation_qualified', 'persistent_qualified',
        'priority_qualified', 'performance_qualified', 'full_goal_qualified')), 'qualification boundaries')
    calls = result['legacy_calls']
    require(set(calls) == LEGACY_CALLS and all(type(count) is int and count > 0 for count in calls.values()),
            'all25 original SDK callback routes succeeded')
    require(all(calls[name] == 2 for name in ('cudnnRNNForwardTraining', 'cudnnRNNBackwardData',
                                            'cudnnRNNBackwardWeights')) and
            calls['cudnnGetRNNLinLayerMatrixParams'] == calls['cudnnGetRNNLinLayerBiasParams'] == 16,
            'exact six computations and32 original subregion queries')
    layouts = [item for item in records if item['kind'] == 'layout']
    require(len(layouts) == 32, 'complete independent weight and gradient layout discovery')
    offsets = {}
    for name in ('weight', 'dweight'):
        selected = [item for item in layouts if item['name'] == name]
        found, byte_ranges = {}, set()
        for item in selected:
            key = (item['linear'], item['part'])
            count = 4 if item['part'] == 'matrix' else 2
            require(0 <= key[0] < 8 and key[1] in ('matrix', 'bias') and key not in found,
                    'all eight distinct linear matrix/bias records')
            require(len(item['dimensions']) == 3 and math.prod(item['dimensions']) == count and
                    all(d > 0 for d in item['dimensions']) and item['extent'] == 4 * count,
                    'native complete contiguous filter extent')
            expected = list(range(item['offset'], item['offset'] + item['extent'], 4))
            require(item['logical_offsets'] == expected and all(0 <= o <= model['weight_bytes'] - 4
                    and o % 4 == 0 for o in expected), 'bounded row-major legacy filter mapping')
            region = set(range(item['offset'], item['offset'] + item['extent']))
            require(not byte_ranges & region, 'disjoint full parameter regions')
            byte_ranges.update(region)
            found[key] = expected
        require(len(found) == 16 and len(byte_ranges) == 192, 'all48 scalar locations')
        offsets[name] = [offset for linear in range(8) for part in ('matrix', 'bias')
                         for offset in found[(linear, part)]]
    require(offsets['weight'] == offsets['dweight'], 'independent parameter layouts agree')
    buffers = {}
    for item in (record for record in records if record['kind'] == 'buffer'):
        key, name = (item['iteration'], item['name']), item['name']
        require(key not in buffers and item['guard_bytes'] == 256, 'unique complete guarded snapshot')
        size = item['payload_bytes']
        if item['opaque']:
            capacity = 'dropout_bytes' if name == 'dropout_state' else name + '_bytes'
            require(name in OPAQUE_NAMES and size == model[capacity] and
                    bytes.fromhex(item['prefix_hex']) == GUARD and
                    bytes.fromhex(item['suffix_hex']) == GUARD and
                    len(item['payload_fnv1a64']) == 16, 'all opaque allocation guards/capacities')
            buffers[key] = None
        else:
            require(name in INPUT_NAMES | OUTPUT_NAMES, 'known full tensor/gradient snapshot')
            raw = bytes.fromhex(item['guarded_hex'])
            require(len(raw) == size + 512 and raw[:256] == GUARD and raw[-256:] == GUARD,
                    'every prefix/suffix byte and full payload preserved')
            buffers[key] = raw[256:-256]
    require({name for iteration, name in buffers if iteration == 0} ==
            INPUT_NAMES | {'dweight'} | OPAQUE_NAMES and len(buffers) == 45,
            'exact initial11 and twice all17 allocation snapshots')
    p, x, hx, cx, dy, dhy, dcy = seeds()
    for name, expected in dict(x=x, hx=hx, cx=cx, dy=dy, dhy=dhy, dcy=dcy).items():
        require(buffers[(0, name)] == struct.pack('<' + 'f' * len(expected), *expected), 'frozen input bytes')
    weight = bytearray(model['weight_bytes'])
    initial = bytearray(model['weight_bytes'])
    for offset, value in zip(offsets['weight'], p, strict=True):
        struct.pack_into('<f', weight, offset, value)
        struct.pack_into('<f', initial, offset, value * .25 + .125)
    require(buffers[(0, 'weight')] == weight and buffers[(0, 'dweight')] == initial,
            'entire initial weight/gradient including opaque padding')
    values, parameter_gradient = reference()
    for iteration in (1, 2):
        require({name for number, name in buffers if number == iteration} ==
                INPUT_NAMES | OUTPUT_NAMES | OPAQUE_NAMES, 'all17 live allocation snapshots')
        for name in INPUT_NAMES:
            require(buffers[(iteration, name)] == buffers[(0, name)], 'entire immutable input payload')
        for name, expected in values.items():
            close(floats(buffers[(iteration, name)]), expected)
        gradients = [struct.unpack_from('<f', buffers[(iteration, 'dweight')], offset)[0]
                     for offset in offsets['dweight']]
        close(gradients, [value * .25 + .125 + iteration * gradient
                          for value, gradient in zip(p, parameter_gradient, strict=True)])
    iterations = [item for item in records if item['kind'] == 'iteration']
    require(len(iterations) == 2 and {item['iteration'] for item in iterations} == {1, 2} and
            all(item['tensor_scalars_checked'] == 36 and item['parameter_gradients_checked'] == 48
                and item['guarded_allocations_checked'] == 17 and item['unchanged_inputs_checked'] == 7
                and item['complete_oracle_passed'] is True for item in iterations), 'complete oracle receipts')
    return dict(complete_tensor_and_gradient_scalars=168, guarded_snapshots=45,
                finite_difference_gradient_scalars=66, parameter_layouts=32,
                original_legacy_routes=25, actual_providers=sorted(mapped),
                opaque_padding_math_qualified=False, full_device_qualified=False)
