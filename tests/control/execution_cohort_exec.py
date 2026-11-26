"""Bounded owner-side cold exec gate; never an application or CUDA interface."""

import json
import os
import signal
import sys

MAX_BOOTSTRAP_BYTES = 4096


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate bootstrap field")
        result[key] = value
    return result


def decode_bootstrap(data):
    if not data or len(data) > MAX_BOOTSTRAP_BYTES or not data.endswith(b"\n") or data.count(b"\n") != 1:
        raise ValueError("one bounded complete bootstrap frame required")
    value = json.loads(data.decode("utf-8"), object_pairs_hook=unique_object,
                       parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    if type(value) is not dict or set(value) != {"server_pid", "server_start", "socket", "preload"}:
        raise ValueError("exact bootstrap fields required")
    if type(value["server_pid"]) is not int or not 0 < value["server_pid"] < 1 << 32 or \
            type(value["server_start"]) is not int or not 0 < value["server_start"] < 1 << 64:
        raise ValueError("positive exact service generation required")
    socket, preload = value["socket"], value["preload"]
    if type(socket) is not str or not socket.startswith("/") or not 1 < len(os.fsencode(socket)) < 108 or \
            any(character in socket for character in ("\0", "\r", "\n")):
        raise ValueError("bounded absolute private socket required")
    if type(preload) is not str or len(preload) > 4096 or (preload and not os.path.isabs(preload)) or \
            any(character.isspace() or character in "\0:" for character in preload):
        raise ValueError("one exact optional preload path required")
    return value


def main():
    if len(sys.argv) < 3 or not sys.argv[1].isdecimal() or int(sys.argv[1]) < 3 or not os.path.isabs(sys.argv[2]):
        raise ValueError("dedicated inherited control descriptor and absolute executable required")
    control = int(sys.argv[1])
    data = bytearray()
    try:
        while True:
            chunk = os.read(control, MAX_BOOTSTRAP_BYTES + 1 - len(data))
            if not chunk:
                break
            data.extend(chunk)
            if len(data) > MAX_BOOTSTRAP_BYTES:
                raise ValueError("oversized bootstrap frame")
    finally:
        os.close(control)
    value = decode_bootstrap(bytes(data))
    environment = dict(os.environ)
    names = ("GLIMMER_EXECUTION_SOCKET", "GLIMMER_EXECUTION_SERVER_PID", "GLIMMER_EXECUTION_SERVER_START")
    if value["preload"]:
        environment.update(LD_PRELOAD=value["preload"])
        environment.update(zip(names, (value["socket"], str(value["server_pid"]), str(value["server_start"]))))
    else:
        for name in ("LD_PRELOAD", *names):
            environment.pop(name, None)
    # Match direct subprocess exec rather than retaining Python's ignored pipe
    # and file-size signals in the unchanged native application.
    for name in ("SIGPIPE", "SIGXFZ", "SIGXFSZ"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), signal.SIG_DFL)
    os.execve(sys.argv[2], sys.argv[2:], environment)
    return 126


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"Cohort cold exec rejected: {error}", file=sys.stderr)
        raise SystemExit(126) from None
