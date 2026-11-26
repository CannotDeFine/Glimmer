"""CPU-only contracts for the optional real BERT workload; no framework imports."""

import argparse
import contextlib
import copy
import io
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] /
                       "examples/framework_workloads/bert"))
import bert_contract as contract
import bert_workload as workload
import prepare_bert
import run_bert_baseline as baseline
import onnx_bert_workload as onnx_workload
import profile_bert as profiler
import run_bert_qualification as qualification


class BertContractTest(unittest.TestCase):
    def profile(self, owned=False):
        return {"trace_quality": "no_reported_warnings", "kernel_duration": {"count": 4, "total_ns": 100},
                "kernel_duration_by_name": {"unknown_total_ns": 0, "total_ns": 100, "groups": [
                    {"name": "glimmer_persistent_sgemm" if owned else "vendor_gemm", "count": 2, "total_ns": 80},
                    {"name": "layernorm", "count": 2, "total_ns": 20}]}}

    def test_gpu_denominator_accepts_native_zero_coverage_without_claiming_preemption(self):
        native = profiler.coverage(self.profile(), "native")
        self.assertEqual(native["original_time_fraction"], 1)
        self.assertEqual(native["owned_persistent_launches"], 0)
        owned = profiler.coverage(self.profile(True), "cooperative")
        self.assertEqual(owned["original_time_fraction"], 0.2)
        self.assertEqual(owned["original_launch_fraction"], 0.5)
        with self.assertRaises(ValueError):
            profiler.coverage(self.profile(True), "native")

    def test_gpu_denominator_rejects_warning_unknown_names_and_incomplete_inventory(self):
        cases = []
        value = self.profile()
        value["trace_quality"] = "warnings_require_review"
        cases.append(value)
        for name, replacement in (("unknown_total_ns", 1), ("total_ns", 99)):
            value = self.profile()
            value["kernel_duration_by_name"][name] = replacement
            cases.append(value)
        for name, replacement in (("name", "__glimmer_unclassified"), ("name", None),
                                  ("total_ns", -1), ("count", True), ("count", 0)):
            value = self.profile()
            value["kernel_duration_by_name"]["groups"][0][name] = replacement
            cases.append(value)
        for value in cases:
            with self.subTest(value=value), self.assertRaises(ValueError):
                profiler.coverage(value, "cooperative")

    def test_profile_preloads_only_target_and_retains_failed_output(self):
        args = argparse.Namespace(mode="cooperative", build=pathlib.Path("/build"), engine="pytorch",
                                  python=pathlib.Path("/python"), assets=pathlib.Path("/assets"),
                                  output=pathlib.Path("/output"), role="training")
        command = profiler.command(args, "/nsys")
        profiler_index = command.index("/nsys")
        child = command.index("/usr/bin/env", profiler_index)
        self.assertEqual(command[:2], ["/usr/bin/env", "-i"])
        self.assertNotIn("-i", command[profiler_index:])
        self.assertFalse(any("LD_PRELOAD=" in value for value in command[:child]))
        self.assertTrue(any("LD_PRELOAD=" in value for value in command[child:]))
        self.assertIn("--profile-cuda", command)
        self.assertFalse(any(value.startswith("--capture-range") for value in command))
        with tempfile.TemporaryDirectory() as temporary:
            args.output = pathlib.Path(temporary) / "failed"
            args.nsys = "missing"
            with mock.patch.object(profiler.shutil, "which", return_value=None), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(profiler.run(args), 1)
            report = json.loads((args.output / "profile.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertFalse(report["preemption_acceptance"])
            self.assertNotIn("coverage", report)

    def test_framework_caches_are_local_and_benchmark_network_is_disabled(self):
        with mock.patch.dict(contract.os.environ, {}, clear=True):
            contract.configure_cache()
            expected = pathlib.Path(contract.__file__).resolve().parents[3] / ".cache" / "bert"
            for name in ("HF_HOME", "TORCH_HOME", "XDG_CACHE_HOME", "CUDA_CACHE_PATH"):
                self.assertTrue(pathlib.Path(contract.os.environ[name]).is_relative_to(expected))
            self.assertEqual(contract.os.environ["HF_HUB_OFFLINE"], "1")

    def test_barrier_rejects_stale_signals_and_preserves_timeout_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            ready, start = root / "ready.json", root / "start"
            start.touch()
            with self.assertRaises(ValueError):
                contract.wait_for_start(ready, start, "gpu")
            start.unlink()
            with self.assertRaises(TimeoutError):
                contract.wait_for_start(ready, start, "gpu", timeout=0)
            self.assertEqual(json.loads(ready.read_text())["device_uuid"], "gpu")
            with self.assertRaises(FileExistsError):
                contract.wait_for_start(ready, start, "gpu", timeout=0)

    def test_baseline_removes_inherited_instrumentation_and_credentials(self):
        with mock.patch.dict(baseline.os.environ, {"GLIMMER_PERSISTENT_AUDIT": "1", "LD_PRELOAD": "bad",
                                                  "CUBLAS_LOGINFO_DBG": "1", "AWS_SECRET_ACCESS_KEY": "secret",
                                                  "CUDA_VISIBLE_DEVICES": "0"}, clear=True):
            native = baseline.environment("native", pathlib.Path("/build"))
            self.assertNotIn("LD_PRELOAD", native)
            self.assertNotIn("GLIMMER_PERSISTENT_AUDIT", native)
            self.assertNotIn("CUBLAS_LOGINFO_DBG", native)
            self.assertNotIn("AWS_SECRET_ACCESS_KEY", native)
            self.assertEqual(native["CUDA_VISIBLE_DEVICES"], "0")
            cooperative = baseline.environment("cooperative", pathlib.Path("/build"))
            self.assertEqual(cooperative["GLIMMER_PERSISTENT_BLAS"], "1")
            self.assertEqual(cooperative["GLIMMER_PERSISTENT_WORKER_BLOCKS"], "320")
            self.assertNotIn("GLIMMER_PERSISTENT_WORKER_BLOCKS", native)
            self.assertNotIn("GLIMMER_PERSISTENT_AUDIT", cooperative)

    def test_benchmark_geometry_overrides_inherited_correctness_default(self):
        with mock.patch.dict(baseline.os.environ, {"GLIMMER_PERSISTENT_WORKER_BLOCKS": "3"}, clear=True):
            args = argparse.Namespace(mode="cooperative", build=pathlib.Path("/build"), engine="pytorch",
                                      python=pathlib.Path("/python"), assets=pathlib.Path("/assets"),
                                      output=pathlib.Path("/output"), role="training")
            command = profiler.command(args, "/nsys")
            self.assertIn("GLIMMER_PERSISTENT_WORKER_BLOCKS=320", command)
            self.assertNotIn("GLIMMER_PERSISTENT_WORKER_BLOCKS=3", command)

    def test_profile_can_keep_complete_training_oracle_and_matching_warmup(self):
        args = argparse.Namespace(mode="cooperative", build=pathlib.Path("/build"), engine="pytorch",
                                  python=pathlib.Path("/python"), assets=pathlib.Path("/assets"),
                                  output=pathlib.Path("/output"), role="training", warmup=1,
                                  reference=pathlib.Path("/native-reference"))
        command = profiler.command(args, "/nsys")
        self.assertEqual(command[command.index("--warmup") + 1], "1")
        self.assertEqual(command[command.index("--reference") + 1], "/native-reference")
        self.assertEqual(command[command.index("--iterations") + 1], "2")

    def qualification_result(self, role="training"):
        result = dict(status="ok", role=role, profile=contract.PROFILE[role], seed=contract.SEED,
                      dtype="float32", warmup=1, iterations=2, samples_ns=[1, 2],
                      timing=contract.summarize([1, 2], 4, contract.PROFILE[role]["batch_size"]),
                      measured_begin_ns=10, measured_end_ns=14,
                      preemption_acceptance=False, performance_acceptance=False)
        if role == "training":
            result["numerical_comparison"] = dict(rtol=2e-4, atol=2e-5,
                checks=["loss", "start_logits", "end_logits", *[f"{kind}/tensor-{index}"
                    for kind in ("gradient", "parameter") for index in range(199)]])
        else:
            result.update(providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                          validation=dict(reference="native PyTorch logits", rtol=2e-4, atol=2e-5,
                                          checked_requests=2))
        return result

    def qualification_log(self):
        return "".join(f"[glimmer] {name} accepted=1\n" * count
                       for name, count in qualification.SELECTIONS.items())

    def test_qualification_requires_exact_tensor_and_actual_selection_inventories(self):
        result, log = self.qualification_result(), self.qualification_log()
        self.assertEqual(qualification.validate_control(result, "cooperative", "training", log),
                         qualification.SELECTIONS)
        for field, value in (("checks", result["numerical_comparison"]["checks"][:-1]),
                             ("checks", ["loss"] * 401), ("rtol", 1e-3), ("atol", 1e-4)):
            changed = copy.deepcopy(result)
            changed["numerical_comparison"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                qualification.validate_control(changed, "cooperative", "training", log)
        for changed in ("", log.replace("persistent_lt_bias", "unexpected"),
                        log + "[glimmer] persistent_sgemm accepted=1\n"):
            with self.assertRaises(ValueError):
                qualification.validate_control(result, "cooperative", "training", changed)
        with self.assertRaises(ValueError):
            qualification.validate_control(result, "native", "training", log)
        qualification.validate_control(result, "native", "training", "")

    def test_qualification_requires_cuda_inference_oracle_and_unchanged_training_pairing(self):
        result = self.qualification_result("inference")
        qualification.validate_control(result, "cooperative", "inference", "")
        for changed in (dict(result, providers=["CPUExecutionProvider"]),
                        dict(result, validation=dict(result["validation"], checked_requests=1))):
            with self.assertRaises(ValueError):
                qualification.validate_control(changed, "native", "inference", "")
        with self.assertRaises(ValueError):
            qualification.validate_control(result, "cooperative", "inference", self.qualification_log())
        training = self.qualification_result()
        training["numerical_comparison"]["checks"][-1] = "parameter/unknown"
        with self.assertRaises(ValueError):
            qualification.validate_control(training, "cooperative", "training", self.qualification_log())

    def test_qualification_failure_preserves_report_and_never_claims_device_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = argparse.Namespace(output=pathlib.Path(temporary) / "qualification",
                                      build=pathlib.Path("/missing"))
            with mock.patch.object(qualification, "collect") as collect, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(qualification.run(args), 1)
                collect.assert_not_called()
            report = json.loads((args.output / "qualification.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertFalse(report["full_card_acceptance"])
            self.assertFalse(report["performance_acceptance"])
            self.assertFalse(report["preemption_acceptance"])
            self.assertEqual(report["controls"], [])

    def test_qualification_foreign_process_rejection_precedes_application(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary) / "control"
            args = argparse.Namespace(build=pathlib.Path("/build"), python=pathlib.Path("/python"),
                                      assets=pathlib.Path("/assets"), reference=pathlib.Path("/reference"))
            with mock.patch.object(qualification, "check_gpu", side_effect=RuntimeError("foreign GPU")), \
                    mock.patch.object(qualification.subprocess, "Popen") as spawn:
                with self.assertRaisesRegex(RuntimeError, "foreign GPU"):
                    qualification.collect(args, "cooperative", "training", root)
                spawn.assert_not_called()
            report = json.loads((root / "control.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertIn("foreign GPU", report["error"])

    def test_qualification_timeout_and_child_failure_reap_only_owned_group(self):
        for timeout in (False, True):
            with self.subTest(timeout=timeout), tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary) / "control"
                args = argparse.Namespace(build=pathlib.Path("/build"), python=pathlib.Path("/python"),
                                          assets=pathlib.Path("/assets"), reference=pathlib.Path("/reference"),
                                          timeout=1)
                process = mock.Mock(pid=4321, returncode=1)
                process.poll.return_value = None if timeout else 1
                with mock.patch.object(qualification, "check_gpu", return_value={}), \
                        mock.patch.object(qualification, "mapped_vendor_libraries", return_value=set()), \
                        mock.patch.object(qualification.time, "monotonic", side_effect=[0, 2]), \
                        mock.patch.object(qualification.subprocess, "Popen", return_value=process) as spawn, \
                        mock.patch.object(qualification, "stop_group") as cleanup:
                    with self.assertRaises(TimeoutError if timeout else RuntimeError):
                        qualification.collect(args, "cooperative", "training", root)
                    cleanup.assert_called_once_with(process)
                    self.assertTrue(spawn.call_args.kwargs["start_new_session"])
                report = json.loads((root / "control.json").read_text())
                self.assertEqual(report["status"], "failed")
                self.assertEqual(report["pid"], 4321)
                self.assertFalse(report["full_card_acceptance"])
                self.assertNotIn("result_sha256", report)

    def test_baseline_rejects_missing_samples_and_false_preemption_claim(self):
        result = {"status": "ok", "role": "inference", "profile": contract.PROFILE["inference"],
                  "iterations": 2, "samples_ns": [1, 2], "timing": contract.summarize([1, 2], 4, 1),
                  "measured_begin_ns": 10, "measured_end_ns": 14,
                  "preemption_acceptance": False, "performance_acceptance": False}
        baseline.validate_result(result, "inference", 2)
        for field, value in (("preemption_acceptance", True), ("samples_ns", [1]), ("status", "failed"),
                             ("timing", dict(result["timing"], p99_ms=10)), ("measured_end_ns", 12)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                baseline.validate_result(dict(result, **{field: value}), "inference", 2)

    def document(self):
        return {"version": "1.1", "data": [{"paragraphs": [{"context": "This is a real answer.",
                 "qas": [{"id": "sample-1", "question": "What is this?", "answers": [
                     {"answer_start": 10, "text": "real answer"}]}]}]}]}

    def test_real_answer_identity_and_offsets(self):
        selected = contract.squad_examples(self.document(), 1)
        self.assertEqual(selected[0]["id"], "sample-1")
        self.assertEqual(selected[0]["answer_start"], 10)
        for count in (0, -1, 2, True):
            with self.subTest(count=count), self.assertRaises(ValueError):
                contract.squad_examples(self.document(), count)
        wrong = self.document()
        wrong["data"][0]["paragraphs"][0]["qas"][0]["answers"][0]["answer_start"] = 11
        with self.assertRaises(ValueError):
            contract.squad_examples(wrong, 1)

    def test_duplicate_ids_and_wrong_dataset_rejected(self):
        document = self.document()
        questions = document["data"][0]["paragraphs"][0]["qas"]
        questions.append(copy.deepcopy(questions[0]))
        with self.assertRaises(ValueError):
            contract.squad_examples(document, 2)
        document["version"] = "2.0"
        with self.assertRaises(ValueError):
            contract.squad_examples(document, 1)

    def test_answer_tokens_exclude_question_padding_and_partial_spans(self):
        offsets = [(0, 0), (0, 4), (0, 0), (0, 4), (5, 7), (8, 9), (10, 14), (15, 21), (0, 0)]
        segments = [None, 0, None, 1, 1, 1, 1, 1, None]
        self.assertEqual(contract.answer_tokens(offsets, segments, 10, 21), (6, 7))
        self.assertIsNone(contract.answer_tokens(offsets, segments, 10, 25))
        self.assertIsNone(contract.answer_tokens(offsets, [None] * len(offsets), 10, 21))
        with self.assertRaises(ValueError):
            contract.answer_tokens(offsets, segments, 4, 5)

    def test_timing_retains_samples_and_labels_insufficient_p99(self):
        result = contract.summarize([1_000_000, 3_000_000], 5_000_000, 2)
        self.assertEqual(result["p99_ms"], 3)
        self.assertEqual(result["mean_ms"], 2)
        self.assertEqual(result["samples_per_second"], 800)
        self.assertFalse(result["p99_sufficient_samples"])
        self.assertTrue(contract.summarize([1] * 1000, 1000, 1)["p99_sufficient_samples"])
        for values, elapsed in (([], 10), ([0], 10), ([float("nan")], 10), ([11], 10), ([True], 10)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                contract.summarize(values, elapsed, 1)

    def test_failed_assets_never_start_framework_or_report_acceptance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            args = argparse.Namespace(output=root / "run", assets=root / "missing", role="training",
                                      warmup=3, iterations=2, profile_cuda=False,
                                      snapshot=False, reference=None)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(workload.run(args), 1)
            report = json.loads((args.output / "result.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertFalse(report["preemption_acceptance"])
            self.assertFalse(report["performance_acceptance"])
            self.assertNotIn("timing", report)
            with self.assertRaises(FileExistsError):
                workload.run(args)

    def test_download_failure_is_preserved_and_not_retried_as_synthetic(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary) / "assets"
            with mock.patch.object(prepare_bert.urllib.request, "urlopen", side_effect=OSError("offline")) as fetch:
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(OSError):
                    prepare_bert.prepare(output)
                self.assertEqual(fetch.call_count, 1)
            report = json.loads((output / "manifest.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["files"], {})
            with self.assertRaises(ValueError):
                contract.validate_assets(output)

    def test_profile_is_complete_real_model_and_official_pinned_assets(self):
        self.assertEqual(contract.PROFILE["training"], {"batch_size": 2, "sequence_length": 384})
        self.assertEqual(contract.PROFILE["inference"], {"batch_size": 1, "sequence_length": 128})
        self.assertEqual(len(contract.WEIGHTS_SHA256), 64)
        for name, url in contract.ASSETS.items():
            self.assertTrue(url.startswith("https://"))
            self.assertNotIn("/main/", url)
            self.assertFalse(name.endswith((".pkl", ".bin")))

    def test_asset_inventory_hash_and_model_shape_are_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            config = {"model_type": "bert", "hidden_size": 768, "num_hidden_layers": 12,
                      "num_attention_heads": 12, "intermediate_size": 3072, "vocab_size": 30522}
            manifest = {"status": "ok", "model_id": contract.MODEL_ID,
                        "model_revision": contract.MODEL_REVISION,
                        "data_revision": contract.DATA_REVISION, "files": {}}
            for name, url in contract.ASSETS.items():
                path = root / name
                path.write_text(json.dumps(config) if name == "config.json" else "fixture")
                manifest["files"][name] = {"url": url, "bytes": path.stat().st_size,
                                         "sha256": contract.sha256(path)}
            contract.write_json(root / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "official pinned"):
                contract.validate_assets(root)
            with mock.patch.object(contract, "WEIGHTS_SHA256", contract.sha256(root / "model.safetensors")), \
                    mock.patch.object(contract, "WEIGHTS_BYTES", 7):
                self.assertEqual(contract.validate_assets(root), manifest)
                (root / "tokenizer.json").write_text("untracked")
                with self.assertRaisesRegex(ValueError, "untracked asset"):
                    contract.validate_assets(root)
                (root / "tokenizer.json").unlink()
                (root / "vocab.txt").write_text("changed")
                with self.assertRaisesRegex(ValueError, "asset identity"):
                    contract.validate_assets(root)
                (root / "vocab.txt").write_text("fixture")
                config["num_hidden_layers"] = 2
                contract.write_json(root / "config.json", config)
                manifest["files"]["config.json"].update(
                    bytes=(root / "config.json").stat().st_size,
                    sha256=contract.sha256(root / "config.json"))
                contract.write_json(root / "manifest.json", manifest)
                with self.assertRaisesRegex(ValueError, "complete BERT-base"):
                    contract.validate_assets(root)

    def test_download_rejects_truncation_and_checks_resume_range(self):
        class Response(io.BytesIO):
            url = "https://official.example/model"
            status = 206
            headers = {"Content-Length": "3", "Content-Range": "bytes 3-5/6"}

        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "weights"
            path.write_bytes(b"abc")
            with mock.patch.object(prepare_bert.urllib.request, "urlopen", return_value=Response(b"def")) as fetch:
                self.assertEqual(prepare_bert.download(Response.url, path, 6, 3, 6), 6)
                self.assertEqual(fetch.call_args.args[0].headers["Range"], "bytes=3-")
            self.assertEqual(path.read_bytes(), b"abcdef")
            path.write_bytes(b"abc")
            with mock.patch.object(prepare_bert.urllib.request, "urlopen", return_value=Response(b"de")):
                with self.assertRaisesRegex(ValueError, "incomplete download"):
                    prepare_bert.download(Response.url, path, 6, 3, 6)
            wrong = Response(b"def")
            wrong.status = 200
            path.write_bytes(b"abc")
            with mock.patch.object(prepare_bert.urllib.request, "urlopen", return_value=wrong):
                with self.assertRaisesRegex(ValueError, "exact requested resume range"):
                    prepare_bert.download(Response.url, path, 6, 3, 6)
            self.assertEqual(path.read_bytes(), b"abc")

    def test_resume_copies_verified_failed_evidence_without_modifying_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            previous, target = root / "old", root / "new"
            previous.mkdir()
            target.mkdir()
            manifest = {"status": "failed", "model_id": contract.MODEL_ID,
                        "model_revision": contract.MODEL_REVISION,
                        "data_revision": contract.DATA_REVISION, "files": {}}
            for name, url in contract.ASSETS.items():
                path = previous / name
                path.write_bytes(b"abc")
                manifest["files"][name] = {"url": url, "bytes": 3, "sha256": contract.sha256(path)}
            contract.write_json(previous / "manifest.json", manifest)
            self.assertEqual(prepare_bert.copy_previous(previous, target), manifest["files"])
            self.assertEqual((target / "model.safetensors").read_bytes(), b"abc")
            self.assertEqual(json.loads((previous / "manifest.json").read_text()), manifest)
            (previous / "model.safetensors").write_bytes(b"abcunrecorded-tail")
            prepare_bert.copy_previous(previous, target)
            self.assertEqual((target / "model.safetensors").read_bytes(), b"abcunrecorded-tail")
            (previous / "model.safetensors").write_bytes(b"abd")
            with self.assertRaisesRegex(ValueError, "resume source identity"):
                prepare_bert.copy_previous(previous, target)

    def test_bounded_resume_requests_exact_end_and_total(self):
        class Response(io.BytesIO):
            url = "https://official.example/model"
            status = 206
            headers = {"Content-Length": "3", "Content-Range": "bytes 3-5/9"}

        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "weights"
            path.write_bytes(b"abc")
            with mock.patch.object(prepare_bert.urllib.request, "urlopen", return_value=Response(b"def")) as fetch:
                self.assertEqual(prepare_bert.download(Response.url, path, 9, 3, 9, 5), 6)
                self.assertEqual(fetch.call_args.args[0].headers["Range"], "bytes=3-5")
            self.assertEqual(path.read_bytes(), b"abcdef")

    def test_onnx_rejects_failed_or_changed_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            contract.write_json(root / "export.json", {"status": "failed"})
            with self.assertRaises(ValueError):
                onnx_workload.validate_export(root, {})
            (root / "bert.onnx").write_bytes(b"fixture")
            report = {"status": "ok", "assets": {}, "seed": contract.SEED,
                      "dtype": "float32", "opset": 17,
                      "model_sha256": contract.sha256(root / "bert.onnx"),
                      "input_names": ["input_ids", "attention_mask", "token_type_ids"],
                      "output_names": ["start_logits", "end_logits"],
                      "source_sha256": {name: contract.sha256(pathlib.Path(onnx_workload.__file__).with_name(name))
                                         for name in ("export_bert.py", "bert_workload.py", "bert_contract.py")}}
            contract.write_json(root / "export.json", report)
            self.assertEqual(onnx_workload.validate_export(root, {}), report)
            (root / "bert.onnx").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                onnx_workload.validate_export(root, {})


if __name__ == "__main__":
    unittest.main()
