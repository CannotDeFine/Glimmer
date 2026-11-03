"""Test check-script behavior without building or changing the real index."""

import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

CHECK = pathlib.Path(__file__).resolve().parents[2] / "scripts/check.sh"
PROJECT = CHECK.parents[1]


class CheckScriptTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="glimmer check ")
        self.addCleanup(temporary.cleanup)
        self.root = pathlib.Path(temporary.name)
        (self.root / "scripts").mkdir()
        (self.root / "examples").mkdir()
        self.script = self.root / "scripts/check.sh"
        shutil.copyfile(CHECK, self.script)
        shutil.copyfile(PROJECT / "scripts/build_safety.sh", self.root / "scripts/build_safety.sh")
        self.bin = self.root / "tools"
        self.bin.mkdir()
        self.log = self.root / "calls.jsonl"
        self.environment = {key: value for key, value in os.environ.items()
                            if not key.startswith("GIT_") and key not in {"LD_PRELOAD", "LD_AUDIT"}}
        self.environment.update(PATH=f"{self.bin}:/usr/bin:/bin", CHECK_SCRIPT_LOG=str(self.log))
        self.tool("shellcheck", """
import json, os, pathlib, sys
if sys.argv[1:2] != ['-x']:
    raise SystemExit(22)
with open(os.environ['CHECK_SCRIPT_LOG'], 'a') as target:
    target.write(json.dumps(sys.argv[2:]) + '\\n')
raise SystemExit(23 if any('BAD_SCRIPT' in pathlib.Path(p).read_text() for p in sys.argv[2:]) else 0)
""")
        self.tool("cmake", "raise SystemExit(47)\n")
        self.git("init", "-q")
        (self.root / ".gitignore").write_text(".venv/\noutput/\n")
        self.git("add", ".gitignore", "scripts/check.sh", "scripts/build_safety.sh")

    def tool(self, name, source):
        path = self.bin / name
        path.write_text(f"#!{sys.executable}\n{source}")
        path.chmod(0o755)

    def git(self, *arguments):
        subprocess.run(["/usr/bin/git", "-C", str(self.root), *arguments], env=self.environment,
                       check=True, capture_output=True, text=True, timeout=10)

    def source(self, name, tracked=False, bad=False):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\n# BAD_SCRIPT\n" if bad else "#!/bin/sh\nexit 0\n")
        if tracked:
            self.git("add", "-f", "--", name)
        return path

    def run_check(self, *arguments):
        result = subprocess.run(["bash", str(self.script), *arguments], env=self.environment,
                                capture_output=True, text=True, timeout=10)
        calls = [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []
        return result, [name for call in calls for name in call]

    def test_help_and_invalid_arguments_do_not_start_checks_or_change_cache(self):
        cache = self.root / "build/debug/CMakeCache.txt"
        cache.parent.mkdir(parents=True)
        content = "GLIMMER_ENABLE_CUDA_GPU_TESTS:BOOL=ON\n"
        cache.write_text(content)
        for arguments, code in ((["--help"], 0), (["-h"], 0),
                                (["--unknown"], 2), (["--help", "extra"], 2)):
            with self.subTest(arguments=arguments):
                result, paths = self.run_check(*arguments)
                self.assertEqual(result.returncode, code, result.stdout + result.stderr)
                self.assertIn("Usage:", result.stdout + result.stderr)
                self.assertEqual(paths, [])
                self.assertEqual(cache.read_text(), content)

    def test_build_budget_rejects_full_or_unmeasurable_storage(self):
        self.tool("du", """
import os, sys
value = os.environ['BUDGET_FIXTURE_BYTES']
if value == 'error':
    raise SystemExit(7)
print(value + '\\t' + sys.argv[-1])
""")
        helper = self.root / "scripts/build_safety.sh"
        command = ["bash", "-c", 'source "$1"; check_build_budget "$2"', "_", str(helper), str(self.root)]
        self.environment["BUDGET_FIXTURE_BYTES"] = "error"
        result = subprocess.run(command, env=self.environment, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)  # A missing build has no allocated bytes.
        (self.root / "build").mkdir()
        for value, expected in (("0", 0), ("9999999999", 0), ("10000000000", 1),
                                ("999999999999999999999999", 1), ("invalid", 1), ("error", 1)):
            self.environment["BUDGET_FIXTURE_BYTES"] = value
            result = subprocess.run(command, env=self.environment, capture_output=True, text=True, timeout=10)
            with self.subTest(value=value):
                self.assertEqual(result.returncode, expected, result.stderr)

    def test_build_scripts_stop_after_budget_is_exceeded(self):
        build_script = self.root / "scripts/build.sh"
        shutil.copyfile(PROJECT / "scripts/build.sh", build_script)
        spdlog = self.root / "3rdparty/spdlog/CMakeLists.txt"
        spdlog.parent.mkdir(parents=True)
        spdlog.write_text("fixture\n")
        (self.root / "build").mkdir()
        self.tool("du", """
import os, pathlib, sys
counter = pathlib.Path(os.environ['BUDGET_FIXTURE_COUNT'])
count = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(count))
value = '10000000000' if count >= int(os.environ['BUDGET_FIXTURE_LIMIT']) else '9999999999'
print(value + '\\t' + sys.argv[-1])
""")
        self.tool("cmake", """
import json, os, sys
with open(os.environ['BUDGET_FIXTURE_CMAKE'], 'a') as stream:
    stream.write(json.dumps(sys.argv[1:]) + '\\n')
""")
        self.tool("clang-format", "raise SystemExit(0)\n")
        self.tool("ctest", "raise SystemExit(0)\n")
        counter = self.root / "budget-count"
        cmake_log = self.root / "budget-cmake.jsonl"
        self.environment.update(BUDGET_FIXTURE_COUNT=str(counter), BUDGET_FIXTURE_CMAKE=str(cmake_log))
        for script, trigger, expected_cmake in ((build_script, 1, 0), (build_script, 2, 1),
                                                (build_script, 3, 2), (self.script, 1, 0),
                                                (self.script, 3, 1), (self.script, 4, 2),
                                                (self.script, 5, 3), (self.script, 6, 3)):
            counter.unlink(missing_ok=True)
            cmake_log.unlink(missing_ok=True)
            self.environment["BUDGET_FIXTURE_LIMIT"] = str(trigger)
            result = subprocess.run(["bash", str(script)], cwd=self.root, env=self.environment,
                                    capture_output=True, text=True, timeout=10)
            calls = cmake_log.read_text().splitlines() if cmake_log.exists() else []
            with self.subTest(script=script.name, trigger=trigger):
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn("must remain below", result.stderr)
                self.assertEqual(len(calls), expected_cmake)

    def test_sources_include_untracked_but_exclude_ignored_packages(self):
        self.source("scripts/tracked.sh", tracked=True)
        self.source("examples/new.sh")
        self.source("examples/workload/.venv/package/completion.sh", bad=True)
        self.source("examples/workload/output/generated.sh", bad=True)
        result, paths = self.run_check()
        self.assertEqual(result.returncode, 47, result.stderr)
        self.assertEqual(set(paths), {"scripts/check.sh", "scripts/build_safety.sh",
                                     "scripts/tracked.sh", "examples/new.sh"})

    def test_tracked_source_is_not_hidden_by_ignore_rules(self):
        self.source("examples/.venv/intentional.sh", tracked=True)
        result, paths = self.run_check()
        self.assertEqual(result.returncode, 47, result.stderr)
        self.assertIn("examples/.venv/intentional.sh", paths)

    def test_literal_paths_and_deleted_or_symbolic_files(self):
        name = "examples/a space;literal\nline.sh"
        self.source(name)
        self.source("scripts/deleted.sh", tracked=True).unlink()
        target = self.source("examples/target.data", bad=True)
        (self.root / "examples/link.sh").symlink_to(target)
        result, paths = self.run_check()
        self.assertEqual(result.returncode, 47, result.stderr)
        self.assertIn(name, paths)
        self.assertNotIn("scripts/deleted.sh", paths)
        self.assertNotIn("examples/link.sh", paths)

    def test_source_diagnostic_still_fails_before_build(self):
        self.source("examples/bad.sh", bad=True)
        result, paths = self.run_check()
        self.assertEqual(result.returncode, 23, result.stderr)
        self.assertIn("examples/bad.sh", paths)

    def test_ci_shell_check_matches_source_scope_and_reports_missing_tool(self):
        workflow = (PROJECT / ".github/workflows/ci.yml").read_text()
        section = workflow.split("      - name: Check shell scripts when ShellCheck is available\n", 1)[1]
        section = section.split("      - name:", 1)[0]
        script = textwrap.dedent(section.split("        run: |\n", 1)[1])
        self.source("scripts/new.sh")
        self.source("examples/.venv/package/ignored.sh", bad=True)
        command = ["/bin/bash", "-e", "-o", "pipefail", "-c", script]
        result = subprocess.run(command, cwd=self.root, env=self.environment,
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = [name for line in self.log.read_text().splitlines() for name in json.loads(line)]
        self.assertEqual(set(calls), {"scripts/check.sh", "scripts/build_safety.sh", "scripts/new.sh"})

        self.source("examples/bad.sh", bad=True)
        result = subprocess.run(command, cwd=self.root, env=self.environment,
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 23, result.stdout + result.stderr)

        (self.bin / "shellcheck").unlink()
        missing_tool_environment = {**self.environment, "PATH": str(self.bin)}
        result = subprocess.run(command, cwd=self.root, env=missing_tool_environment,
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("::warning::ShellCheck is unavailable", result.stdout)

    def test_build_script_does_not_fetch_or_overwrite_user_files(self):
        build_script = self.root / "scripts/build.sh"
        shutil.copyfile(PROJECT / "scripts/build.sh", build_script)
        for name in ("git", "cmake"):
            self.tool(name, """
import json, os, sys
with open(os.environ['CHECK_SCRIPT_LOG'], 'a') as target:
    target.write(json.dumps(sys.argv) + '\\n')
""")

        def run_build(*arguments):
            return subprocess.run(["/bin/bash", str(build_script), *arguments], cwd=self.root,
                                  env=self.environment, capture_output=True, text=True, timeout=10)

        for arguments, code in ((("--help",), 0), (("-h",), 0), (("--unknown",), 2)):
            with self.subTest(arguments=arguments):
                result = run_build(*arguments)
                self.assertEqual(result.returncode, code, result.stdout + result.stderr)
                self.assertIn("Usage:", result.stdout + result.stderr)
                self.assertFalse(self.log.exists())

        missing = run_build()
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn("3rdparty/spdlog", missing.stderr)
        self.assertFalse(self.log.exists(), "missing dependencies must not run git or CMake")

        spdlog = self.root / "3rdparty/spdlog/CMakeLists.txt"
        spdlog.parent.mkdir(parents=True)
        spdlog.touch()
        database = self.root / "compile_commands.json"
        database.write_text("user data\n")
        refused = run_build()
        self.assertNotEqual(refused.returncode, 0)
        self.assertEqual(database.read_text(), "user data\n")
        self.assertFalse(self.log.exists())
        database.unlink()

        slot = self.root / "build/debug"
        slot.mkdir(parents=True)
        retained = slot / ".retained"
        retained.write_text("keep\n")
        refused = run_build()
        self.assertNotEqual(refused.returncode, 0)
        self.assertEqual(retained.read_text(), "keep\n")
        self.assertFalse(self.log.exists())
        retained.unlink()

        marker = slot / ".glimmer-frozen"
        marker.symlink_to(self.root / "missing-marker")
        refused = run_build()
        self.assertNotEqual(refused.returncode, 0)
        self.assertTrue(marker.is_symlink())
        self.assertFalse(self.log.exists())
        marker.unlink()

        cache = slot / "CMakeCache.txt"
        cache.symlink_to(self.root / "missing-cache")
        refused = run_build()
        self.assertNotEqual(refused.returncode, 0)
        self.assertTrue(cache.is_symlink())
        self.assertFalse(self.log.exists())
        cache.unlink()

        slot.rmdir()
        retained_slot = self.root / "retained-slot"
        retained_slot.mkdir()
        slot.symlink_to(retained_slot, target_is_directory=True)
        refused = run_build()
        self.assertNotEqual(refused.returncode, 0)
        self.assertTrue(slot.is_symlink())
        self.assertFalse(self.log.exists())
        slot.unlink()
        slot.mkdir()
        cache.write_text("GLIMMER_ENABLE_CUDA_GPU_TESTS:BOOL=OFF\n")
        (slot / "compile_commands.json").write_text("generated\n")
        result = run_build()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = [json.loads(line)[1:] for line in self.log.read_text().splitlines()]
        self.assertEqual(calls, [["--preset", "debug"], ["--build", "--preset", "debug"]])
        self.assertFalse(database.exists())

    def test_git_inventory_failure_does_not_pass_an_empty_scope(self):
        self.tool("git", "import sys\nraise SystemExit(64 if 'ls-files' in sys.argv else 0)\n")
        result, paths = self.run_check()
        self.assertEqual(result.returncode, 64, result.stderr)
        self.assertEqual(paths, [])

    def test_staged_whitespace_failure_stops_before_build(self):
        path = self.source("scripts/staged.sh")
        path.write_text("#!/bin/sh\nexit 0  \n")
        self.git("add", "scripts/staged.sh")
        result, paths = self.run_check()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("trailing whitespace", result.stdout + result.stderr)
        self.assertEqual(paths, [])

    def test_gpu_enabled_or_frozen_preset_is_rejected_before_configure(self):
        cache = self.root / "build/debug/CMakeCache.txt"
        cache.parent.mkdir(parents=True)
        for value in ("ON", "TRUE", "1", "2", "yes"):
            with self.subTest(value=value):
                content = f"GLIMMER_ENABLE_CUDA_GPU_TESTS:BOOL={value}\n"
                cache.write_text(content)
                result, paths = self.run_check()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("GPU tests are enabled", result.stderr)
                self.assertEqual(cache.read_text(), content)
                self.assertEqual(paths, [])
        cache.write_text("GLIMMER_ENABLE_CUDA_GPU_TESTS:BOOL=OFF\n")
        marker = cache.parent / ".glimmer-frozen"
        marker.touch()
        result, paths = self.run_check()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("frozen", result.stderr)
        self.assertEqual(paths, [])
        marker.unlink()
        marker.symlink_to(self.root / "missing-marker-target")
        result, paths = self.run_check()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("frozen", result.stderr)
        self.assertTrue(marker.is_symlink())
        self.assertEqual(cache.read_text(), "GLIMMER_ENABLE_CUDA_GPU_TESTS:BOOL=OFF\n")
        self.assertEqual(paths, [])

    def test_all_selected_caches_are_checked_before_building(self):
        cache = self.root / "build/asan-ubsan/CMakeCache.txt"
        cache.parent.mkdir(parents=True)
        cache.write_text("GLIMMER_ENABLE_CUDA_GPU_TESTS:BOOL=ON\n")
        result, paths = self.run_check()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("build/asan-ubsan/CMakeCache.txt", result.stderr)
        self.assertEqual(paths, [])
        cache.write_text("GLIMMER_ENABLE_CUDA_GPU_TESTS:BOOL=OFF\n")
        perf_cache = self.root / "build/cuda-perf/CMakeCache.txt"
        perf_cache.parent.mkdir(parents=True)
        content = "GLIMMER_ENABLE_CUDA_GPU_TESTS:BOOL=ON\n"
        perf_cache.write_text(content)
        self.tool("nvcc", "pass\n")
        result, paths = self.run_check()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("build/cuda-perf/CMakeCache.txt", result.stderr)
        self.assertEqual(perf_cache.read_text(), content)
        self.assertEqual(paths, [])

    def test_check_rejects_non_symlink_compile_database_before_checks(self):
        database = self.root / "compile_commands.json"
        database.write_text("user data\n")
        result, paths = self.run_check()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("non-symlink", result.stderr)
        self.assertEqual(database.read_text(), "user data\n")
        self.assertEqual(paths, [])

    def test_symlinked_build_slots_are_rejected_before_configure(self):
        content = "GLIMMER_ENABLE_CUDA_GPU_TESTS:BOOL=OFF\n"
        parent = self.root / "build"
        external = self.root / "retained"
        external.mkdir()
        cache = external / "CMakeCache.txt"
        cache.write_text(content)
        parent.mkdir()
        slot = parent / "debug"
        slot.symlink_to(external, target_is_directory=True)
        result, paths = self.run_check()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("symlink", result.stderr)
        self.assertEqual(cache.read_text(), content)
        self.assertEqual(paths, [])
        slot.unlink()
        parent.rmdir()

        other = self.root / "retained-parent"
        (other / "debug").mkdir(parents=True)
        other_cache = other / "debug/CMakeCache.txt"
        other_cache.write_text(content)
        parent.symlink_to(other, target_is_directory=True)
        result, paths = self.run_check()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("symlink", result.stderr)
        self.assertEqual(other_cache.read_text(), content)
        self.assertEqual(paths, [])

    def test_unconfigured_or_linked_cache_slots_are_not_overwritten(self):
        slot = self.root / "build/debug"
        slot.mkdir(parents=True)
        retained = slot / ".retained"
        retained.write_text("keep this\n")
        result, paths = self.run_check()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("nonempty", result.stderr)
        self.assertEqual(retained.read_text(), "keep this\n")
        self.assertEqual(paths, [])
        retained.unlink()

        cache = slot / "CMakeCache.txt"
        target = self.root / "retained-cache"
        for exists in (True, False):
            with self.subTest(target_exists=exists):
                if exists:
                    target.write_text("GLIMMER_ENABLE_CUDA_GPU_TESTS:BOOL=OFF\n")
                cache.symlink_to(target)
                result, paths = self.run_check()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("symlink", result.stderr)
                self.assertTrue(cache.is_symlink())
                if exists:
                    self.assertEqual(target.read_text(), "GLIMMER_ENABLE_CUDA_GPU_TESTS:BOOL=OFF\n")
                    target.unlink()
                self.assertEqual(paths, [])
                cache.unlink()

        slot.rmdir()
        slot.write_text("not a build directory\n")
        result, paths = self.run_check()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not a directory", result.stderr)
        self.assertEqual(slot.read_text(), "not a build directory\n")
        self.assertEqual(paths, [])
        slot.unlink()
        slot.mkdir()

        # An empty slot remains available for an initial configuration.
        result, paths = self.run_check()
        self.assertEqual(result.returncode, 47, result.stderr)
        self.assertIn("scripts/check.sh", paths)

    def test_cuda_tool_warnings_describe_path_lookup(self):
        self.environment["PATH"] = str(self.bin)
        (self.bin / "dirname").symlink_to("/usr/bin/dirname")
        (self.bin / "bash").symlink_to("/bin/bash")
        for name in ("git", "cmake", "ctest"):
            self.tool(name, "pass\n")
        result, _ = self.run_check()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("nvcc or clang-tidy not found on PATH", result.stderr)
        self.assertIn("nvcc not found on PATH", result.stderr)

    def test_host_presets_disable_gpu_tests_and_ctest_hides_devices(self):
        preset_log = self.root / "preset-calls.jsonl"
        self.environment["CHECK_PRESET_LOG"] = str(preset_log)
        self.environment.pop("ASAN_OPTIONS", None)
        self.tool("cmake", """
import json, os, sys
with open(os.environ['CHECK_PRESET_LOG'], 'a') as target:
    target.write(json.dumps(['cmake', sys.argv[1:]]) + '\\n')
""")
        self.tool("ctest", """
import json, os, sys
with open(os.environ['CHECK_PRESET_LOG'], 'a') as target:
    target.write(json.dumps(['ctest', sys.argv[1:], os.environ.get('CUDA_VISIBLE_DEVICES'),
                             os.environ.get('ASAN_OPTIONS')]) + '\\n')
""")
        for name in ("clang-format", "clang-tidy", "nvcc"):
            self.tool(name, "pass\n")
        result, _ = self.run_check()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("review skipped checks and warnings", result.stdout)
        calls = [json.loads(line) for line in preset_log.read_text().splitlines()]
        configurations = {args[1]: args for tool, args, *rest in calls
                          if tool == "cmake" and args[:1] == ["--preset"]}
        self.assertEqual(set(configurations), {"debug", "asan-ubsan", "lint", "cuda-lint", "cuda-perf"})
        for preset in ("debug", "asan-ubsan", "lint", "cuda-lint", "cuda-perf"):
            self.assertIn("-DGLIMMER_ENABLE_CUDA_GPU_TESTS=OFF", configurations[preset])
        format_checks = [args for tool, args, *rest in calls
                         if tool == "cmake" and "format-check" in args]
        self.assertEqual(format_checks, [["--build", "--preset", "debug", "--target", "format-check"]])
        tests = [call for call in calls if call[0] == "ctest"]
        self.assertEqual(len(tests), 5)
        self.assertTrue(all(call[2] == "" for call in tests))
        self.assertTrue(all("--no-tests=error" in call[1] for call in tests))
        for preset in ("asan-ubsan", "lint", "cuda-lint", "cuda-perf"):
            arguments = next(call[1] for call in tests if preset in call[1])
            label = arguments.index("-LE")
            self.assertEqual(arguments[label + 1], "^no-instrumented-code$")
        perf = next(call[1] for call in tests if "cuda-perf" in call[1])
        self.assertEqual(perf[-2:], ["-E", "_gpu_test$"])
        asan = next(call for call in tests if "asan-ubsan" in call[1])
        debug = next(call for call in tests if "debug" in call[1])
        self.assertNotIn("-LE", debug[1])
        self.assertEqual(asan[3], "detect_leaks=0")
        self.assertIn("leak detection is disabled", result.stderr)
        self.environment["ASAN_OPTIONS"] = "detect_leaks=1:halt_on_error=1"
        overridden, _ = self.run_check()
        self.assertEqual(overridden.returncode, 0, overridden.stderr)
        self.assertNotIn("leak detection is disabled", overridden.stderr)
        calls = [json.loads(line) for line in preset_log.read_text().splitlines()]
        asan = [call for call in calls if call[0] == "ctest" and "asan-ubsan" in call[1]][-1]
        self.assertEqual(asan[3], "detect_leaks=1:halt_on_error=1")


    @unittest.skipUnless(shutil.which("cmake") and shutil.which("c++"),
                         "cmake and a C++ compiler are required")
    def test_compile_database_target_preserves_non_symlink_files(self):
        source = self.root / "build/link source"
        source.mkdir(parents=True)
        for name in ("CMakeLists.txt", "CMakePresets.json"):
            shutil.copyfile(PROJECT / name, source / name)
        for name in ("cmake", "src", "include", "3rdparty"):
            (source / name).symlink_to(PROJECT / name, target_is_directory=True)
        build = self.root / "link build"
        cmake = shutil.which("cmake")
        configured = subprocess.run(
            [cmake, "-S", str(source), "-B", str(build), "-DBUILD_TESTING=OFF"],
            env=self.environment, capture_output=True, text=True, timeout=30)
        self.assertEqual(configured.returncode, 0, configured.stdout + configured.stderr)
        command = [cmake, "--build", str(build), "--target", "glimmer_compile_commands"]
        database = source / "compile_commands.json"
        database.write_text("user data\n")
        refused = subprocess.run(command, env=self.environment, capture_output=True, text=True, timeout=30)
        self.assertNotEqual(refused.returncode, 0, refused.stdout + refused.stderr)
        self.assertFalse(database.is_symlink())
        self.assertEqual(database.read_text(), "user data\n")
        database.unlink()
        database.mkdir()
        refused = subprocess.run(command, env=self.environment, capture_output=True, text=True, timeout=30)
        self.assertNotEqual(refused.returncode, 0, refused.stdout + refused.stderr)
        self.assertTrue(database.is_dir())
        database.rmdir()
        old_database = source / "old database.json"
        old_database.write_text("old build data\n")
        for previous in (None, str(old_database), "missing-build/compile_commands.json"):
            if previous is not None:
                database.symlink_to(previous)
            updated = subprocess.run(command, env=self.environment, capture_output=True, text=True, timeout=30)
            self.assertEqual(updated.returncode, 0, updated.stdout + updated.stderr)
            self.assertTrue(database.is_symlink())
            self.assertEqual(database.resolve(), build / "compile_commands.json")
            database.unlink()
        self.assertEqual(old_database.read_text(), "old build data\n")

    @unittest.skipUnless(shutil.which("cmake") and shutil.which("c++"),
                         "cmake and a C++ compiler are required")
    def test_compile_database_respects_an_explicit_off_setting(self):
        source = self.root / "compile export"
        source.mkdir()
        (source / "probe.cc").write_text("int main() { return 0; }\n")
        (source / "CMakeLists.txt").write_text(f'''
cmake_minimum_required(VERSION 3.20)
project(CompileExport LANGUAGES CXX)
include("{(PROJECT / "cmake/ProjectOptions.cmake").as_posix()}")
add_executable(scope_probe probe.cc)
''')
        cmake = shutil.which("cmake")
        for setting, expected in (([], True), (["-DCMAKE_EXPORT_COMPILE_COMMANDS=OFF"], False)):
            build = self.root / ("compile-on" if expected else "compile-off")
            configured = subprocess.run([cmake, "-S", str(source), "-B", str(build), *setting],
                                       env=self.environment, capture_output=True, text=True, timeout=30)
            self.assertEqual(configured.returncode, 0, configured.stdout + configured.stderr)
            self.assertEqual((build / "compile_commands.json").exists(), expected)


    @unittest.skipUnless(shutil.which("clang-tidy") and shutil.which("cmake") and shutil.which("c++"),
                         "clang-tidy, cmake and a C++ compiler are required")
    def test_clang_tidy_checks_project_headers_in_a_different_checkout_path(self):
        source = self.root / "project (checkout).v1"
        source.mkdir()
        (source / "src").mkdir()
        (source / "src/code").mkdir()
        vendor = source / "3rdparty/vendor/src"
        vendor.mkdir(parents=True)
        shutil.copyfile(PROJECT / ".clang-tidy", source / ".clang-tidy")
        (source / "src/probe.h").write_text("inline int BadName() { return 0; }\n")
        (vendor / "vendor.h").write_text("inline int VendorBadName() { return 0; }\n")
        (source / "src/code/probe.cc").write_text(
            '#include "../probe.h"\n#include <vendor.h>\n'
            'int main() { return BadName() + VendorBadName(); }\n')
        (source / "CMakeLists.txt").write_text(f'''
cmake_minimum_required(VERSION 3.20)
project(TidyScope LANGUAGES CXX)
include("{(PROJECT / "cmake/ProjectOptions.cmake").as_posix()}")
add_executable(scope_probe src/code/probe.cc)
target_include_directories(scope_probe PRIVATE "{vendor.as_posix()}")
glimmer_enable_clang_tidy(scope_probe)
''')
        cmake, tidy = shutil.which("cmake"), shutil.which("clang-tidy")
        build = self.root / "tidy-build"
        configured = subprocess.run([cmake, "-S", str(source), "-B", str(build),
                                    "-DGLIMMER_ENABLE_CLANG_TIDY=ON",
                                    "-DGLIMMER_CLANG_TIDY=" + tidy],
                                   env=self.environment, capture_output=True, text=True, timeout=30)
        self.assertEqual(configured.returncode, 0, configured.stdout + configured.stderr)
        command = [cmake, "--build", str(build), "-j1"]
        failing = subprocess.run(command, env=self.environment, capture_output=True, text=True, timeout=30)
        self.assertNotEqual(failing.returncode, 0, failing.stdout + failing.stderr)
        self.assertIn("BadName", failing.stdout + failing.stderr)
        self.assertNotIn("VendorBadName", failing.stdout + failing.stderr)
        (source / "src/probe.h").write_text("inline int bad_name() { return 0; }\n")
        (source / "src/code/probe.cc").write_text(
            '#include "../probe.h"\n#include <vendor.h>\n'
            'int main() { return bad_name() + VendorBadName(); }\n')
        passing = subprocess.run(command, env=self.environment, capture_output=True, text=True, timeout=30)
        self.assertEqual(passing.returncode, 0, passing.stdout + passing.stderr)


if __name__ == "__main__":
    unittest.main()
