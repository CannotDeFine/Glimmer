#!/usr/bin/env python3
"""Run isolated, GPU-free Driver-proxy lookup configuration cases."""

import argparse
import json
import os
import pathlib
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "interceptor", "first", "second", "caller"):
        parser.add_argument(f"--{name}", type=pathlib.Path, required=True)
    parser.add_argument("--preload", help="explicit sanitizer-first preload chain when applicable")
    args = parser.parse_args()
    for name in ("binary", "interceptor", "first", "second", "caller"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    env = {key: value for key, value in os.environ.items()
           if key in {"PATH", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "ASAN_OPTIONS", "UBSAN_OPTIONS", "LSAN_OPTIONS"}}
    env.update(LD_PRELOAD=args.preload or str(args.interceptor), GLIMMER_COOPERATIVE_RUNTIME_PREVIEW="1")
    with tempfile.TemporaryDirectory(prefix="glimmer driver proxy ") as directory:
        folder = pathlib.Path(directory)
        symbolic = folder / "symbolic alias.so"
        symbolic.symlink_to(args.first)
        cases = [("unset", None, True, (0, 0)), ("empty", "", True, (0, 0)),
                 ("configured", str(args.first), True, (1, 0)),
                 ("other_provider", str(args.second), True, (0, 1)),
                 ("dot_segment", f"{args.first.parent}/./{args.first.name}", True, (1, 0)),
                 ("repeated_separator", f"{args.first.parent}//{args.first.name}", True, (1, 0)),
                 ("symlink_declined", str(symbolic), True, (0, 0)),
                 ("parent_traversal", f"{args.first.parent}/../{args.first.parent.name}/{args.first.name}", True, (0, 0)),
                 ("oversized", "/" + "p" * 4096, True, (0, 0)),
                 ("relative", args.first.name, True, (0, 0)),
                 ("missing", str(folder / "missing.so"), True, (0, 0)),
                 ("directory", str(folder), True, (0, 0)),
                 ("runtime_off", str(args.first), False, (0, 0))]
        for name, proxy, runtime, expected in cases:
            child = dict(env)
            child["GLIMMER_COOPERATIVE_RUNTIME_PREVIEW"] = "1" if runtime else "0"
            if proxy is not None:
                child["GLIMMER_PROFILE_DRIVER_PROXY_PATH"] = proxy
            command = [str(args.binary), str(args.first), str(args.second), str(args.caller),
                       *(str(value) for value in expected)]
            result = subprocess.run(command, env=child, text=True, capture_output=True, timeout=20)
            print(json.dumps({"case": name, "returncode": result.returncode,
                              "stdout": result.stdout, "stderr": result.stderr}), flush=True)
            if result.returncode:
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
