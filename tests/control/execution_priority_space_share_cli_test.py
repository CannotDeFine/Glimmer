"""Cold official CLI rejection controls; no valid GPU startup is attempted."""

import pathlib
import subprocess
import sys
import tempfile


def main():
    binary = pathlib.Path(sys.argv[1]).resolve(strict=True)
    help_result = subprocess.run([str(binary), "--help"], text=True, capture_output=True,
                                 timeout=5, check=False)
    if help_result.returncode or "priority-space-share" not in help_result.stdout:
        raise ValueError("official help omits the actual space-sharing strategy")
    with tempfile.TemporaryDirectory(prefix="glimmer-space-share-cli-") as directory:
        socket = pathlib.Path(directory) / "service.sock"
        base = [str(binary), "--socket", str(socket), "--scheduler", "priority-space-share"]
        pins = ["--client-priority", "1000001:1:0:original",
                "--client-priority", "1000002:1:100:original"]
        cases = [[], pins[:2], [*pins, "--persistent-workers", "320"],
                 [*pins, "--kernel-mode", "preemptive"], [*pins, "--replace-gemm"],
                 [*pins, "--profile-input", "/nonexistent/single"],
                 [*pins, "--profile-output", "/nonexistent/new-single"],
                 [*pins, "--pair-workers", "160"], [*pins, "--priority-idle-us", "0"],
                 [*pins, "--sharing-threads-per-sm", "32"],
                 [*pins, "--ptx-cache-input", "/nonexistent/ptx"],
                 [*pins, "--ptx-cache-output", "/nonexistent/new-ptx"],
                 [*pins, "--observe-block-completion"], [*pins, "--max-clients", "1"],
                 [*pins, "--client-priority", "1000003:1:50:configured"]]
        for index, arguments in enumerate(cases):
            result = subprocess.run([*base, *arguments], text=True, capture_output=True,
                                    timeout=5, check=False)
            output = result.stdout + result.stderr
            if (result.returncode != 1 or socket.exists() or "Unknown execution scheduler" in output
                    or "Execution service ready:" in output
                    or not any(marker in output for marker in
                               ("Priority space sharing requires", "Configured client kernels require"))):
                raise ValueError(f"cold CLI case{index} did not reject the precise conflict: {output}")
    print("Official priority-space-share help and15 cold conflict cases passed; no valid GPU startup")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
