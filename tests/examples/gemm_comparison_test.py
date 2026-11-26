"""CPU-only regression tests for standalone GEMM evidence validation."""

import importlib.util
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import sys
import unittest
from unittest import mock

SOURCE = Path(__file__).resolve().parents[2] / "examples/gemm_comparison/run_comparison.py"
SPEC = importlib.util.spec_from_file_location("gemm_comparison_runner", SOURCE)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class EvidenceTest(unittest.TestCase):
    def records(self, check_only=False):
        rows = []
        for name, (m, n, k) in RUNNER.CASES.items():
            rows.append({
                "case": name, "m": m, "n": n, "k": k,
                "workers": 320, "threads": 128, "sm_count": 80,
                "logical_tiles": 144, "resident_blocks_per_sm": 4,
                "ordinary_persistent_bitwise_equal": True, "retreat_resume": True,
                "ordinary_compute_sms": 80, "persistent_compute_sms": 80,
                "max_native_abs_error": 1e-6, "samples_per_mode": 0 if check_only else 21,
                "gpu_samples_us": [([] if check_only else [100.0] * 21) for _ in range(3)],
                "native_median_us": 0 if check_only else 100.0,
                "ordinary_median_us": 0 if check_only else 100.0,
                "persistent_median_us": 0 if check_only else 100.0,
            })
        rows.append({"compiled_source_sha256": "a" * 64,
                     "status": "ok", "cases": 9, "diagnostic_only": True,
                     "application_acceptance": False, "priority_acceptance": False})
        return rows

    def validate(self, records, check_only=False):
        return RUNNER.validate_output("\n".join(json.dumps(row) for row in records), check_only)

    def test_complete_timing_and_correctness(self):
        for check_only in (False, True):
            self.assertEqual(len(self.validate(self.records(check_only), check_only)), 9)

    def replay_output(self, groups):
        return "\n".join("==PROF== Application replay pass\n" +
                         "\n".join(json.dumps(row) for row in records) for records in groups)

    def test_all_application_replay_receipts_are_validated(self):
        for count in (1, 8):
            output = self.replay_output([self.records(True) for _ in range(count)])
            result = RUNNER.validate_application_output(output, True, application_replay=True)
            self.assertEqual(result["application_runs"], count)
            self.assertEqual(result["rows"], self.records(True)[:-1])
            self.assertEqual(result["compiled_source_sha256"], "a" * 64)

    def test_replay_does_not_relax_ordinary_single_run_or_timing(self):
        output = self.replay_output([self.records(True)] * 2)
        with self.assertRaises(ValueError):
            RUNNER.validate_application_output(output, True)
        with self.assertRaises(ValueError):
            RUNNER.validate_application_output(output, False, application_replay=True)

    def test_exact_profiler_progress_interleaving_reconstructs_all_receipts(self):
        records = self.records(True)
        output = "\n".join(json.dumps(row) for row in records)
        for position in (output.index("native_abs_error") + 6,
                         output.index("forward_qkv") + 3,
                         output.index("1e-06") + 2):
            message = '==PROF== Profiling "volta_sgemm_128x32_nt" - 177: Application replay pass 1\n'
            interleaved = output[:position] + message + output[position:]
            with self.subTest(position=position):
                result = RUNNER.validate_application_output(
                    interleaved + "\n" + output, True, application_replay=True)
                self.assertEqual(result["application_runs"], 2)
                self.assertEqual(result["rows"], records[:-1])
                with self.assertRaises(ValueError):
                    RUNNER.validate_application_output(interleaved, True)

    def test_unknown_partial_or_warning_interleaving_is_not_repaired(self):
        output = "\n".join(json.dumps(row) for row in self.records(True))
        position = output.index("native_abs_error") + 6
        for message in (
                '==PROF== Warning: invalid results\n',
                '==PROF== Profiling "Kernel" - 177: Application replay pass 1',
                '==PROF== Profiling "Kernel" - unknown: Application replay pass 1\n',
                '==PROF== Profiling "Kernel" - 177: Unexpected text\n'):
            with self.subTest(message=message), self.assertRaises(ValueError):
                RUNNER.validate_application_output(
                    output[:position] + message + output[position:], True, application_replay=True)

    def test_progress_repair_does_not_hide_changed_or_incomplete_replays(self):
        output = "\n".join(json.dumps(row) for row in self.records(True))
        position = output.index("native_abs_error") + 6
        message = '==PROF== Profiling "Kernel" - 177: Application replay pass 1\n'
        first = output[:position] + message + output[position:]
        for second in (output.replace('"persistent_compute_sms": 80',
                                      '"persistent_compute_sms": 79', 1),
                       output.replace('"max_native_abs_error": 1e-06',
                                      '"max_native_abs_error": 2e-06', 1),
                       "\n".join(output.splitlines()[:-1])):
            with self.assertRaises(ValueError):
                RUNNER.validate_application_output(first + "\n" + second, True,
                                                   application_replay=True)

    def test_invalid_earlier_replay_cannot_be_hidden_by_final_success(self):
        for key, value in (("ordinary_compute_sms", 79), ("persistent_compute_sms", 79),
                           ("retreat_resume", False), ("workers", 3)):
            groups = [self.records(True) for _ in range(8)]
            groups[0][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                RUNNER.validate_application_output(self.replay_output(groups), True, application_replay=True)

    def test_replay_rejects_incomplete_duplicate_reordered_and_changed_results(self):
        bad_groups = []
        for position in (0, 1):
            groups = [self.records(True), self.records(True)]
            groups[position].pop()
            bad_groups.append(groups)
        for key, value in (("compiled_source_sha256", "b" * 64), ("application_acceptance", True)):
            groups = [self.records(True), self.records(True)]
            groups[0][-1][key] = value
            bad_groups.append(groups)
        for change in ("duplicate", "reordered", "numerical"):
            groups = [self.records(True), self.records(True)]
            if change == "duplicate":
                groups[0][1] = groups[0][0]
            elif change == "reordered":
                groups[0][0], groups[0][1] = groups[0][1], groups[0][0]
            else:
                groups[0][0]["max_native_abs_error"] = 2e-6
            bad_groups.append(groups)
        for groups in bad_groups:
            with self.assertRaises(ValueError):
                RUNNER.validate_application_output(self.replay_output(groups), True, application_replay=True)
        with self.assertRaises(ValueError):
            RUNNER.validate_application_output("==PROF== No application output", True, application_replay=True)

    def test_partial_and_duplicate_fail(self):
        for duplicate in (False, True):
            rows = self.records()
            if duplicate:
                rows[1] = rows[0]
            else:
                rows.pop(0)
            with self.assertRaises(ValueError):
                self.validate(rows)

    def test_reordered_case_cannot_change_native_launch_mapping(self):
        rows = self.records(True)
        rows[0], rows[1] = rows[1], rows[0]
        with self.assertRaises(ValueError):
            self.validate(rows, True)

    def test_contract_mismatches_fail(self):
        for key, value in (("workers", 3), ("threads", 256), ("sm_count", 79),
                           ("m", 1024), ("resident_blocks_per_sm", 3),
                           ("retreat_resume", False),
                           ("ordinary_persistent_bitwise_equal", False),
                           ("samples_per_mode", 20), ("logical_tiles", 36)):
            with self.subTest(key=key):
                rows = self.records()
                rows[0][key] = value
                with self.assertRaises(ValueError):
                    self.validate(rows)

    def test_normal_specialization_requires_explicit_correctness(self):
        for value in (None, False, 1):
            rows = self.records(True)
            rows[0]["persistent_audit_specialized"] = True
            if value is not None:
                rows[0]["normal_kernel_checks"] = value
            with self.assertRaises(ValueError):
                self.validate(rows, True)
        rows[0]["normal_kernel_checks"] = True
        self.assertEqual(len(self.validate(rows, True)), 9)

    def test_invalid_timing_fails(self):
        for value in (0, -1, float("nan"), float("inf")):
            rows = self.records()
            rows[0]["native_median_us"] = value
            with self.assertRaises(ValueError):
                self.validate(rows)

    def test_raw_sample_inventory_fails(self):
        for samples in ([], [[1.0] * 21] * 2, [[0.0] * 21] * 3):
            rows = self.records()
            rows[0]["gpu_samples_us"] = samples
            with self.assertRaises(ValueError):
                self.validate(rows)

    def test_compute_coverage_is_not_fabricated(self):
        rows = self.records()
        rows[0]["persistent_compute_sms"] = 74
        with self.assertRaises(ValueError):
            self.validate(rows)
        result = RUNNER.validate_output("\n".join(json.dumps(row) for row in rows), False, True)
        self.assertEqual(result[0]["persistent_compute_sms"], 74)
        for value in (0, 81, True, 80.0):
            rows[0]["persistent_compute_sms"] = value
            with self.assertRaises(ValueError):
                self.validate(rows)

    def test_ordinary_partial_coverage_also_fails(self):
        rows = self.records()
        rows[0]["ordinary_compute_sms"] = 79
        with self.assertRaises(ValueError):
            self.validate(rows)

    def metric_csv(self, count=18, minimum="100", sms="80", kernel="volta_sgemm_128x64_nn"):
        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_ALL)
        writer.writerow(["ID", "Kernel Name", *RUNNER.NATIVE_METRICS])
        writer.writerow(["", "", "", "cycle", "%", "%", "%"])
        for index in range(count):
            writer.writerow([index, kernel, sms, minimum, "85.0", "50.0", "30.0"])
        return output.getvalue()

    def test_native_requires_every_launch_and_positive_minimum_on_all_sms(self):
        rows = RUNNER.native_metric_rows(self.metric_csv(minimum="1,234.5"))
        self.assertEqual(len(rows), 18)
        self.assertEqual(rows[0]["case"], "forward_qkv")
        self.assertEqual(rows[-1]["case"], "weight_contract")
        self.assertEqual(rows[-1]["beta"], 1)
        self.assertEqual(rows[0]["minimum_active_cycles"], 1234.5)
        for kwargs in ({"count": 1}, {"count": 17}, {"count": 19}, {"minimum": "0"},
                       {"minimum": "-1"}, {"minimum": "nan"}, {"minimum": "inf"},
                       {"sms": "79"}, {"kernel": "glimmer_persistent_sgemm"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RUNNER.native_metric_rows(self.metric_csv(**kwargs))
        duplicate = self.metric_csv().replace('"17",', '"0",')
        with self.assertRaises(ValueError):
            RUNNER.native_metric_rows(duplicate)
        extra = '"18","vendor_split_k_reduce","80","100","85","50","30"\n'
        with self.assertRaises(ValueError):
            RUNNER.native_metric_rows(self.metric_csv() + extra)
        support = extra.replace('vendor_split_k_reduce', 'void (anonymous namespace)::persistent_gemm<fixture>()')
        self.assertEqual(len(RUNNER.native_metric_rows(self.metric_csv() + support)), 18)

    def test_ncu_unnamed_helpers_are_explicitly_source_owned(self):
        for name in ("<unnamed>::read_sm_id_extent(unsigned int *)",
                     "void <unnamed>::persistent_gemm<fixture>()",
                     "void <unnamed>::audited_ordinary_gemm<fixture>()"):
            support = '"18","' + name + '","80","100","85","50","30"\n'
            self.assertEqual(len(RUNNER.native_metric_rows(self.metric_csv() + support)), 18)
        for name in ("<unnamed>::vendor_split_k_reduce()", "<unnamed>::persistent_gemm_extra()",
                     "vendor::persistent_gemm<fixture>()"):
            support = '"18","' + name + '","80","100","85","50","30"\n'
            with self.assertRaises(ValueError):
                RUNNER.native_metric_rows(self.metric_csv() + support)

    def compute_csv(self, mutations=None):
        names = ("volta_sgemm_128x32_nn", "void cutlass::Kernel<fixture>()",
                 "void <unnamed>::persistent_gemm<fixture, true>()",
                 "void <unnamed>::audited_ordinary_gemm<fixture>()")
        modes = (0, 1, 2, 3, 2, 2, 2, 4, 4, 2, 4)
        inventory = [{"ID": "0", "Kernel Name": "<unnamed>::read_sm_id_extent(unsigned int *)"}]
        for operation in range(18):
            for position, mode in enumerate(modes):
                name = names[mode] if mode < 4 else "void <unnamed>::persistent_gemm<fixture, false>()"
                grid = 144 if mode in (0, 1, 3) else 1 if position in (5, 9) else 320
                compute = position in (0, 1, 2, 7)
                inventory.append(dict(zip(("ID", "Kernel Name", *RUNNER.NATIVE_METRICS,
                                           RUNNER.COMPUTE_METRIC, "launch__grid_size", "launch__block_size",
                                           "launch__registers_per_thread"),
                                          (str(len(inventory)), name, "80", "100" if compute else "0",
                                           "90", "20", "75", "1234" if compute else "0",
                                           str(grid), "128", "108"))))
        if mutations:
            mutations(inventory)
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=list(inventory[1]), quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(inventory)
        return output.getvalue()

    def test_normal_compute_capture_excludes_retreat_controls(self):
        output = self.compute_csv()
        result = RUNNER.persistent_metric_rows(output, self.records(True)[:-1])
        self.assertEqual(len(result), 72)
        self.assertEqual(len(RUNNER.native_metric_rows(output)), 18)
        self.assertEqual({row["mode"] for row in result}, {"native", "ordinary", "audit", "normal"})
        normal = [row for row in result if row["mode"] == "normal"]
        self.assertEqual(len(normal), 18)
        self.assertEqual(normal[0]["launch_id"], "8")
        self.assertEqual(normal[-1]["case"], "weight_contract")
        self.assertEqual(normal[-1]["beta"], 1)
        self.assertTrue(all(row["minimum_ffma_instructions"] > 0 for row in normal))

    def test_ncu_template_bool_spellings_preserve_exact_specialization(self):
        original = self.compute_csv()
        for audit, normal in (("true", "false"), ("(bool)1", "(bool)0"), ("1", "0")):
            output = original.replace("fixture, true>", "nested<0, 1>, " + audit + ">")
            output = output.replace("fixture, false>", "nested<1, 0>, " + normal + ">")
            with self.subTest(audit=audit, normal=normal):
                result = RUNNER.persistent_metric_rows(output, self.records(True)[:-1])
                self.assertEqual(len(result), 72)
                self.assertEqual(len([row for row in result if row["mode"] == "normal"]), 18)
        for final in ("2", "-1", "11", "01", "false", "(bool)0", "0", "1U"):
            output = original.replace("fixture, true>", "nested<0, 1>, " + final + ">")
            with self.subTest(final=final), self.assertRaises(ValueError):
                RUNNER.persistent_metric_rows(output, self.records(True)[:-1])
        output = original.replace("persistent_gemm<", "persistent_gemm_extra<")
        with self.assertRaises(ValueError):
            RUNNER.persistent_metric_rows(output, self.records(True)[:-1])

    def test_compute_capture_rejects_idle_missing_or_changed_launches(self):
        for index in (1, 2, 3, 8):
            for key, value in ((RUNNER.COMPUTE_METRIC, "0"), (RUNNER.COMPUTE_METRIC, "nan"),
                               ("sm__cycles_active.min", "0"), ("launch__sm_count", "79")):
                def mutate(rows, index=index, key=key, value=value):
                    rows[index][key] = value
                with self.subTest(index=index, key=key), self.assertRaises(ValueError):
                    RUNNER.persistent_metric_rows(self.compute_csv(mutate), self.records(True)[:-1])
        mutations = [lambda rows: rows.pop(),
                     lambda rows: rows[8].update({"Kernel Name": rows[3]["Kernel Name"]}),
                     lambda rows: rows[8].update({"launch__grid_size": "1"}),
                     lambda rows: rows[8].update({"launch__block_size": "64"}),
                     lambda rows: rows[8].update({"ID": "0"}),
                     lambda rows: rows[8].pop(RUNNER.COMPUTE_METRIC),
                     lambda rows: rows[8].update({"Kernel Name": "vendor_helper"})]
        for mutation in mutations:
            with self.assertRaises((ValueError, KeyError)):
                RUNNER.persistent_metric_rows(self.compute_csv(mutation), self.records(True)[:-1])

    def test_normal_full_launch_cannot_be_replaced_by_pre_retreat(self):
        def swap(rows):
            rows[8], rows[9] = rows[9], rows[8]
            rows[8]["ID"], rows[9]["ID"] = "8", "9"
        with self.assertRaises(ValueError):
            RUNNER.persistent_metric_rows(self.compute_csv(swap), self.records(True)[:-1])

    def test_retained_native_evidence_revalidates_all_replays_and_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "native.ncu-rep").write_bytes(b"fixture capture")
            (root / "native-metrics.csv").write_text(self.metric_csv())
            inventory = "0, Tesla V100, GPU-fixture, 560.35.05, 0, 0"
            current = dict(binary_sha256="b" * 64, source_sha256="a" * 64,
                           cutlass_revision=RUNNER.PINNED_CUTLASS,
                           vendor_identities={}, gpu_before=inventory)
            report = dict(current, status="native_full_sm_coverage_verified",
                          collect_native_sm=True, check_only=True,
                          native_full_sm_verified=True, cutlass_compute_full_sm_verified=True,
                          coverage_policy="require_all_sms", full_device_coverage_verified=True,
                          host_sanitizers=False, device_tool=None, performance_acceptance=False,
                          application_acceptance=False, priority_acceptance=False,
                          returncode=0, timed_out=False, gpu_after=inventory,
                          finished_at_unix_s=RUNNER.time.time(),
                          native_rows=RUNNER.native_metric_rows(self.metric_csv()))
            for count in (1, 8):
                groups = [self.records(True) for _ in range(count)]
                (root / "stdout.log").write_text(self.replay_output(groups))
                report["application_replay_passes"] = count
                for name, key in (("native.ncu-rep", "capture_sha256"),
                                  ("native-metrics.csv", "metrics_sha256"), ("stdout.log", "stdout_sha256")):
                    report[key] = RUNNER.digest(root / name)
                path = root / "report.json"
                path.write_text(json.dumps(report))
                self.assertEqual(RUNNER.verify_native_evidence(path, current)["native_launches"], 18)
                report["application_replay_passes"] = count + 1
                path.write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError, "replay count"):
                    RUNNER.verify_native_evidence(path, current)
            report["application_replay_passes"] = 8
            groups[0][0]["persistent_compute_sms"] = 79
            (root / "stdout.log").write_text(self.replay_output(groups))
            report["stdout_sha256"] = RUNNER.digest(root / "stdout.log")
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "partial compute SM"):
                RUNNER.verify_native_evidence(path, current)

            groups = [self.records(True) for _ in range(8)]
            for group in groups:
                for row in group[:-1]:
                    row.update(persistent_audit_specialized=True, normal_kernel_checks=True)
            (root / "stdout.log").write_text(self.replay_output(groups))
            (root / "native-metrics.csv").write_text(self.compute_csv())
            report.update(stdout_sha256=RUNNER.digest(root / "stdout.log"),
                          metrics_sha256=RUNNER.digest(root / "native-metrics.csv"),
                          native_rows=RUNNER.native_metric_rows(self.compute_csv()),
                          compute_rows=RUNNER.persistent_metric_rows(self.compute_csv(), groups[0][:-1]),
                          normal_compute_full_sm_verified=True)
            path.write_text(json.dumps(report))
            self.assertEqual(RUNNER.verify_native_evidence(path, current)["native_launches"], 18)
            for key in ("normal_compute_full_sm_verified", "compute_rows"):
                value = report.pop(key)
                path.write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError, "useful-compute qualification"):
                    RUNNER.verify_native_evidence(path, current)
                report[key] = value

    def test_native_capture_is_untimed_complete_and_does_not_change_gpu_policy(self):
        for privileged in (False, True):
            command = RUNNER.native_profile_command(Path("/binary"), Path("/evidence"),
                                                    {"PATH": "/usr/bin", "LANG": "C"}, privileged)
            self.assertEqual(command[-2:], ["/binary", "--check-only"])
            self.assertIn("none", command)
            self.assertNotIn("--force-overwrite", command)
            self.assertNotIn("--launch-count", command)
            self.assertNotIn("--kill", command)
            self.assertNotIn("--kernel-name", command)
            self.assertIn(RUNNER.COMPUTE_METRIC, command[command.index("--metrics") + 1])
            if privileged:
                self.assertEqual(command[:3], ["/usr/bin/sudo", "-n", "/usr/bin/timeout"])
                self.assertIn("-i", command)

    def test_privileged_launcher_preserves_session_but_isolates_process_group(self):
        probe = "import os,json; print(json.dumps(dict(pid=os.getpid(),sid=os.getsid(0),pgid=os.getpgrp())))"
        for preserve in (False, True):
            with self.subTest(preserve=preserve):
                code, output, errors, timed_out = RUNNER.run_bounded(
                    [sys.executable, "-I", "-c", probe], {"PATH": "/usr/bin:/bin"}, 5,
                    preserve_session=preserve)
                state = json.loads(output)
                self.assertEqual(code, 0)
                self.assertEqual(errors, "")
                self.assertFalse(timed_out)
                self.assertEqual(state["pgid"], state["pid"])
                self.assertNotEqual(state["pgid"], os.getpgrp())
                if preserve:
                    self.assertEqual(state["sid"], os.getsid(0))
                else:
                    self.assertEqual(state["sid"], state["pid"])
                    self.assertNotEqual(state["sid"], os.getsid(0))

    def test_timeout_signals_only_the_owned_group_in_both_launch_modes(self):
        for preserve in (False, True):
            child = mock.Mock(pid=123456, returncode=-9)
            child.communicate.side_effect = [RUNNER.subprocess.TimeoutExpired(["fixture"], 1),
                                             RUNNER.subprocess.TimeoutExpired(["fixture"], 5),
                                             ("retained stdout", "retained stderr")]
            with mock.patch.object(RUNNER.subprocess, "Popen", return_value=child) as popen, \
                    mock.patch.object(RUNNER.os, "killpg") as kill:
                result = RUNNER.run_bounded(["/fixture"], {}, 1, preserve_session=preserve)
                self.assertTrue(result[-1])
                self.assertEqual(result[1:3], ("retained stdout", "retained stderr"))
                self.assertEqual(kill.call_args_list, [mock.call(123456, RUNNER.signal.SIGTERM),
                                                      mock.call(123456, RUNNER.signal.SIGKILL)])
                self.assertEqual(popen.call_args.kwargs["start_new_session"], not preserve)

    def test_same_session_launch_retains_a_real_controlling_terminal(self):
        driver = '''
import fcntl, importlib.util, json, os, sys, termios
fcntl.ioctl(0, termios.TIOCSCTTY, 0)
spec = importlib.util.spec_from_file_location("fixture_runner", sys.argv[1])
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
probe = "import os,json; s=open('/proc/self/stat').read().rsplit(')',1)[1].split(); print(json.dumps(dict(sid=os.getsid(0),tty=int(s[4]))))"
parent = open('/proc/self/stat').read().rsplit(')',1)[1].split()
results = dict(sid=os.getsid(0), tty=int(parent[4]))
for preserve in (False, True):
    code, output, errors, expired = runner.run_bounded([sys.executable, '-I', '-c', probe], {}, 5, preserve_session=preserve)
    if code or errors or expired:
        raise RuntimeError('terminal process fixture failed')
    results[str(preserve)] = json.loads(output)
print(json.dumps(results))
'''
        master, slave = os.openpty()
        try:
            result = RUNNER.subprocess.run([sys.executable, "-I", "-c", driver, str(SOURCE)],
                stdin=slave, stdout=RUNNER.subprocess.PIPE, stderr=RUNNER.subprocess.PIPE,
                start_new_session=True, text=True, timeout=15, check=True)
        finally:
            os.close(slave)
            os.close(master)
        state = json.loads(result.stdout)
        self.assertNotEqual(state["tty"], 0)
        self.assertEqual(state["True"], {"sid": state["sid"], "tty": state["tty"]})
        self.assertEqual(state["False"]["tty"], 0)
        self.assertNotEqual(state["False"]["sid"], state["sid"])

    def test_root_group_permission_failure_cannot_be_called_successful_containment(self):
        child = mock.Mock(pid=123456)
        child.communicate.side_effect = RUNNER.subprocess.TimeoutExpired(["fixture"], 1)
        with mock.patch.object(RUNNER.subprocess, "Popen", return_value=child), \
                mock.patch.object(RUNNER.os, "killpg", side_effect=PermissionError):
            with self.assertRaisesRegex(RuntimeError, "containment is unverified"):
                RUNNER.run_bounded(["/fixture"], {}, 1, preserve_session=True)

    def test_fresh_native_identity_and_raw_artifacts_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "native.ncu-rep").write_bytes(b"fixture capture")
            (root / "native-metrics.csv").write_text(self.metric_csv())
            (root / "stdout.log").write_text("\n".join(json.dumps(row) for row in self.records(True)))
            current = dict(binary_sha256="b" * 64, source_sha256="a" * 64,
                           cutlass_revision=RUNNER.PINNED_CUTLASS, vendor_identities={"blas": "hash"},
                           gpu_before="0, V100, GPU-a, driver-a, 0, 0")
            report = dict(current, status="native_full_sm_coverage_verified", collect_native_sm=True,
                          check_only=True, native_full_sm_verified=True, cutlass_compute_full_sm_verified=True,
                          coverage_policy="require_all_sms", full_device_coverage_verified=True,
                          host_sanitizers=False, device_tool=None, gpu_after=current["gpu_before"],
                          performance_acceptance=False, application_acceptance=False, priority_acceptance=False,
                          returncode=0, timed_out=False, finished_at_unix_s=999,
                          native_rows=RUNNER.native_metric_rows(self.metric_csv()))
            for name, key in (("native.ncu-rep", "capture_sha256"),
                              ("native-metrics.csv", "metrics_sha256"), ("stdout.log", "stdout_sha256")):
                report[key] = RUNNER.digest(root / name)
            path = root / "report.json"
            path.write_text(json.dumps(report))
            with mock.patch.object(RUNNER.time, "time", return_value=1000):
                self.assertEqual(RUNNER.verify_native_evidence(path, current)["native_launches"], 18)
                for key, value in (("source_sha256", "c" * 64), ("binary_sha256", "d" * 64),
                                   ("vendor_identities", {}), ("status", "failed"),
                                   ("gpu_before", "0, V100, GPU-other, driver-a, 0, 0"),
                                   ("finished_at_unix_s", -100000), ("finished_at_unix_s", 1001),
                                   ("timed_out", True), ("native_rows", [])):
                    with self.subTest(key=key), self.assertRaises(ValueError):
                        path.write_text(json.dumps(dict(report, **{key: value})))
                        RUNNER.verify_native_evidence(path, current)
                path.write_text(json.dumps(report))
                (root / "native.ncu-rep").write_bytes(b"changed capture")
                with self.assertRaises(ValueError):
                    RUNNER.verify_native_evidence(path, current)

    def test_default_timing_rejects_missing_native_evidence_before_gpu_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "binary"
            binary.write_bytes(b"fixture executable")
            (root / "build-inputs.json").write_text(json.dumps(dict(
                source_sha256=RUNNER.digest(SOURCE.with_name("gemm_comparison.cu")),
                cutlass_revision=RUNNER.PINNED_CUTLASS, host_sanitizers=False)))
            command = [str(SOURCE), "--binary", str(binary), "--cutlass-root", str(root),
                       "--output-root", str(root / "evidence")]
            results = [mock.Mock(stdout=RUNNER.PINNED_CUTLASS), mock.Mock(stdout="")]
            with mock.patch.object(sys, "argv", command), \
                    mock.patch.object(RUNNER.subprocess, "run", side_effect=results), \
                    mock.patch.object(RUNNER, "gpu_inventory") as inventory, \
                    mock.patch.object(RUNNER, "run_bounded") as execute:
                self.assertEqual(RUNNER.main(), 1)
                inventory.assert_not_called()
                execute.assert_not_called()
            report = json.loads(next((root / "evidence").glob("*/report.json")).read_text())
            self.assertFalse(report["full_device_coverage_verified"])
            self.assertIn("--native-sm-evidence", report["error"])

    def test_default_output_root_is_the_artifact_helper_not_a_build_tree(self):
        with mock.patch.object(sys, "argv", [str(SOURCE), "--binary", "/missing-binary",
                                            "--cutlass-root", "/missing-cutlass"]), \
                mock.patch.object(RUNNER, "resolve_artifact_root", wraps=RUNNER.resolve_artifact_root) as resolve, \
                mock.patch.object(RUNNER, "create_attempt") as create:
            with self.assertRaises(FileNotFoundError):
                RUNNER.main()
            resolve.assert_called_once_with("gemm-comparison", output_root=None)
            create.assert_not_called()

    def test_help_is_cold_outside_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, "-B", str(SOURCE), "--help"],
                                    cwd=directory, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("artifacts/runs/gemm-comparison", result.stdout)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_application_acceptance_is_forbidden(self):
        rows = self.records()
        rows[-1]["application_acceptance"] = True
        with self.assertRaises(ValueError):
            self.validate(rows)

    def test_missing_build_identity_fails(self):
        rows = self.records()
        del rows[-1]["compiled_source_sha256"]
        with self.assertRaises(ValueError):
            self.validate(rows)


if __name__ == "__main__":
    unittest.main()
