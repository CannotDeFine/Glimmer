"""Check official startup selection without submitting any CUDA work."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def run_service(binary, directory, arguments, expected):
    socket = directory / "service.sock"
    log = directory / "service.log"
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    environment.pop("LD_PRELOAD", None)
    environment.pop("LD_AUDIT", None)
    with log.open("w+") as output:
        process = subprocess.Popen([str(binary), "--socket", str(socket), *arguments],
                                   stdout=output, stderr=subprocess.STDOUT, env=environment)
        try:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                output.flush()
                if "Execution service ready:" in log.read_text(errors="replace"):
                    break
                if process.poll() is not None:
                    raise ValueError("service exited before readiness: " +
                                     log.read_text(errors="replace"))
                time.sleep(0.05)
            else:
                raise TimeoutError("service did not become ready")
            if not socket.is_socket():
                raise ValueError("ready service did not create its socket")
            process.terminate()
            if process.wait(timeout=5) != 0:
                raise ValueError("service did not shut down cleanly")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
        output.flush()
        content = log.read_text(errors="replace")
    if socket.exists() or content.count(f"sharded_gemm_signal={expected}") != 1:
        raise ValueError(f"selection or cleanup differs: {content}")


def main():
    binary = Path(sys.argv[1]).resolve(strict=True)
    provider = sys.argv[2].upper() in {"ON", "TRUE", "YES", "1"}
    with tempfile.TemporaryDirectory(prefix="glimmer-sharded-cli-") as temporary:
        directory = Path(temporary)
        replacement = ["--replace-gemm", "--gemm-tile-side", "32", "--persistent-workers", "320"]
        run_service(binary, directory, replacement if provider else [], "false")
        if provider:
            run_service(binary, directory,
                        [*replacement, "--experimental-sharded-gemm-signal"], "true")
        else:
            result = subprocess.run([str(binary), "--socket", str(directory / "service.sock"),
                                     *replacement, "--experimental-sharded-gemm-signal"],
                                    capture_output=True, text=True, timeout=5, check=False)
            if result.returncode != 1 or "compiled sharded provider" not in result.stdout + result.stderr:
                raise ValueError("unavailable provider did not reject the selection")
    print("Official service selection and cleanup passed without CUDA work")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
