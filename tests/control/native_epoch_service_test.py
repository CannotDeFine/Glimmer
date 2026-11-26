#!/usr/bin/env python3
"""Test explicit process-bound epoch service policy without CUDA issue."""

import pathlib
import socket
import subprocess
import sys
import tempfile
import time


POLICY = ['--native-epoch-max-calls', '8', '--native-epoch-prefix-calls', '4',
          '--native-epoch-pending-prefixes', '2', '--native-epoch-acceptance-us', '1000000']
REQUEST_SEQUENCE = 0


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def request(connection, line):
    global REQUEST_SEQUENCE
    is_epoch = line.startswith('GLIMMER_EPOCH_V2 ')
    if is_epoch:
        REQUEST_SEQUENCE += 1
        line += f' {REQUEST_SEQUENCE}'
    connection.sendall(line.encode() + b'\n')
    response = bytearray()
    while not response.endswith(b'\n'):
        byte = connection.recv(1)
        require(byte, 'complete service response')
        response.extend(byte)
        require(len(response) <= 256, 'bounded response')
    tokens = response.decode().split()
    if is_epoch:
        require(tokens[-1] == str(REQUEST_SEQUENCE), 'exact actual request correlation')
        tokens.pop()
    return tokens


def connect(path):
    connection = socket.socket(socket.AF_UNIX)
    connection.settimeout(2)
    connection.connect(path)
    return connection


def main():
    if sys.argv[1] == '--foreign':
        with connect(sys.argv[2]) as connection:
            require(request(connection, f'GLIMMER_EPOCH_V2 CLAIM {sys.argv[3]}') ==
                    ['GLIMMER_EPOCH_V2', 'ERROR', 'INVALID_REQUEST'],
                    'actual foreign process cannot inherit epoch')
        return 0
    service = pathlib.Path(sys.argv[1]).resolve()
    require(service.is_file(), 'real service executable')
    with tempfile.TemporaryDirectory(prefix='glimmer-epoch-policy-') as directory:
        path = str(pathlib.Path(directory) / 'control.sock')
        base = [str(service), '--socket', path, '--quota-bytes', '100']
        remote = ['--execution-mode', 'remote', '--bind-leases-to-process']
        bad = [POLICY, remote + POLICY[:-2], ['--execution-mode', 'remote'] + POLICY,
               remote + POLICY + ['--native-epoch-max-calls', '8']]
        for offset, value in ((1, '0'), (1, '65537'), (3, '9'), (5, '17'), (7, '1000001'), (7, '-1')):
            changed = POLICY.copy()
            changed[offset] = value
            bad.append(remote + changed)
        for arguments in bad:
            result = subprocess.run(base + arguments, capture_output=True, text=True, timeout=5)
            require(result.returncode == 1 and 'Usage:' in result.stderr and not pathlib.Path(path).exists(),
                    'invalid/simulated/unbound/incomplete/repeated policy rejects before socket creation')
        try:
            probe = socket.socket(socket.AF_UNIX)
            probe.close()
        except PermissionError:
            print('SKIP: Unix sockets unavailable')
            return 77
        process = subprocess.Popen(base + remote + POLICY + ['--max-requests', '6'],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            deadline = time.monotonic() + 5
            while not pathlib.Path(path).exists():
                require(process.poll() is None and time.monotonic() < deadline, 'configured service starts')
                time.sleep(0.005)
            with connect(path) as connection:
                grant = request(connection, 'GLIMMER_EPOCH_V2 ACQUIRE client 1 1 8 0 8 4 2 1000000')
                require(len(grant) == 9 and grant[:2] == ['GLIMMER_EPOCH_V2', 'GRANTED'] and
                        grant[4:8] == ['8', '4', '2', '1000000'] and int(grant[2]) > 0 and
                        int(grant[3]) > 0 and int(grant[8]) > 0, 'real configured epoch grant')
                subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve()), '--foreign', path, grant[2]],
                               check=True, timeout=5)
                identity = ' '.join(grant[2:4])
                require(request(connection, f'GLIMMER_EPOCH_V2 HEARTBEAT {identity}') ==
                        ['GLIMMER_EPOCH_V2', 'ACK', 'HEARTBEAT', *grant[2:4], '0'], 'same process heartbeat')
                # Synthetic backend stop declaration, not a CUDA completion proof.
                terminal = f'GLIMMER_EPOCH_V2 COMPLETE {identity} 2'
                require(request(connection, terminal) ==
                        ['GLIMMER_EPOCH_V2', 'ACK', 'COMPLETE', *grant[2:4], '2'], 'exact terminal acknowledgement')
                require(request(connection, terminal) == ['GLIMMER_EPOCH_V2', 'ERROR', 'INVALID_REQUEST'],
                        'actual service rejects terminal replay')
                stats = request(connection, 'GLIMMER_TASK_V1 STATS')
                require(stats[:2] == ['GLIMMER_TASK_V1', 'STATS'] and stats[2:6] == ['1', '0', '0', '1'] and
                        stats[9:11] == ['0', '0'], 'ordinary stats confirm fully drained service')
            stdout, stderr = process.communicate(timeout=5)
            require(process.returncode == 0 and not stderr, f'configured process exits cleanly: {stdout} {stderr}')
        finally:
            if process.poll() is None:
                process.terminate()
                process.communicate(timeout=5)
    print('Native epoch service policy tests passed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
