"""CPU-only cold exec framing, EOF and unchanged application-input controls."""

import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import unittest

SHIM = pathlib.Path(__file__).resolve().parents[1] / "control/execution_cohort_exec.py"
spec = importlib.util.spec_from_file_location("cohort_exec", SHIM)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def bootstrap():
    return dict(server_pid=os.getpid(), server_start=19, socket="/tmp/private-cohort.sock", preload="")


class ColdExec(unittest.TestCase):
    def test_exact_frame_and_native_selection(self):
        self.assertEqual(gate.decode_bootstrap((json.dumps(bootstrap()) + "\n").encode()), bootstrap())

    def test_invalid_types_bounds_and_loader_paths(self):
        for key, value in (("server_pid", False), ("server_pid", 0), ("server_pid", 1 << 32),
                           ("server_start", 1.0), ("server_start", 0), ("server_start", 1 << 64),
                           ("socket", "relative"), ("socket", "/" + "x" * 107), ("socket", "/a\0b"),
                           ("preload", "relative"), ("preload", "/a:/b"), ("preload", "/a b")):
            frame = bootstrap()
            frame[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                gate.decode_bootstrap((json.dumps(frame) + "\n").encode())

    def test_missing_duplicate_trailing_and_oversized_frames(self):
        frame = (json.dumps(bootstrap()) + "\n").encode()
        for changed in (b"", frame.rstrip(), frame + b"\n", frame + b"extra",
                        b"x" * 4097, frame.replace(b'"server_pid":', b'"server_pid": 1, "server_pid":'),
                        frame.replace(b'"preload": ""', b'"other": ""')):
            with self.subTest(frame=changed[:30]), self.assertRaises((ValueError, UnicodeError)):
                gate.decode_bootstrap(changed)

    def launch(self, payload, input_bytes=b"run\n"):
        read_end, write_end = os.pipe2(os.O_CLOEXEC)
        code = "import os,sys; print('PID',os.getpid()); print('INPUT',repr(sys.stdin.buffer.read())); print('PRELOAD',repr(os.getenv('LD_PRELOAD')))"
        process = subprocess.Popen([sys.executable, "-B", str(SHIM), str(read_end),
                                    sys.executable, "-c", code], pass_fds=(read_end,), stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        os.close(read_end)
        try:
            if payload:
                self.assertEqual(os.write(write_end, payload), len(payload))
        finally:
            os.close(write_end)
        try:
            output, errors = process.communicate(input_bytes, timeout=5)
        except subprocess.TimeoutExpired:
            # This unreaped Popen child is ours; never leave a failed CPU fixture
            # or its input receiver stranded after a bounded wait.
            process.kill()
            process.communicate()
            raise
        return process, output.decode(), errors.decode()

    def test_actual_exec_keeps_pid_and_all_ordinary_input(self):
        process, output, errors = self.launch((json.dumps(bootstrap()) + "\n").encode(), b"run\nnext\n")
        self.assertEqual(process.returncode, 0, errors)
        self.assertEqual(output.splitlines(), [f"PID {process.pid}", "INPUT b'run\\nnext\\n'", "PRELOAD None"])

    def test_parent_eof_cancels_before_application_start(self):
        process, output, errors = self.launch(b"")
        self.assertEqual(process.returncode, 126)
        self.assertEqual(output, "")
        self.assertIn("Cohort cold exec rejected", errors)

    def test_malformed_control_cannot_execute_application(self):
        process, output, errors = self.launch(b"{}\n")
        self.assertEqual(process.returncode, 126)
        self.assertEqual(output, "")
        self.assertIn("exact bootstrap fields", errors)


if __name__ == "__main__":
    unittest.main()
