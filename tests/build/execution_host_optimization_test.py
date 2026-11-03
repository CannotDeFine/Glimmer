"""Configure-level controls for production-only Debug host optimization."""

import json
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[2]
TARGETS = (
    "glimmer_core", "glimmer_control", "glimmer_backend", "glimmer_transform",
    "glimmer_cuda_backend", "glimmer_execution_service_runtime",
    "glimmer_execution_service", "glimmer_execution_interceptor",
)


class ExecutionHostOptimizationTest(unittest.TestCase):
    def fixture(self, *, execution=True, compiler=None):
        temporary = tempfile.TemporaryDirectory(prefix="glimmer-host-options-")
        self.addCleanup(temporary.cleanup)
        source = Path(temporary.name)
        (source / "host.cc").write_text("int glimmer_fixture() { return 0; }\n")
        (source / "entry.S").write_text("\n")
        targets = TARGETS if execution else TARGETS[:6]
        declarations = "\n".join(
            f"add_library({target} STATIC host.cc entry.S)" for target in targets)
        (source / "CMakeLists.txt").write_text(f'''
cmake_minimum_required(VERSION 3.20)
project(HostOptionsFixture LANGUAGES CXX ASM)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
{declarations}
add_library(vendor STATIC host.cc)
add_library(fixture STATIC host.cc)
target_link_libraries(fixture PRIVATE glimmer_core)
{"set(CMAKE_CXX_COMPILER_ID " + compiler + ")" if compiler else ""}
include("{PROJECT.as_posix()}/cmake/ExecutionHostOptimization.cmake")
glimmer_configure_execution_host_optimization()
''')
        return source

    def configure(self, source, *options, success=True):
        result = subprocess.run(
            ["cmake", "-S", str(source), "-B", str(source / "build"),
             "-G", "Unix Makefiles", "-DCMAKE_BUILD_TYPE=Debug", *options],
            capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode == 0, success, result.stdout + result.stderr)
        return result

    def commands(self, source):
        return {row["command"].split("CMakeFiles/", 1)[1].split(".dir/", 1)[0]
                + ":" + Path(row["file"]).suffix: shlex.split(row["command"])
                for row in json.loads((source / "build/compile_commands.json").read_text())}

    def test_default_off_target_language_scope_and_rollback(self):
        source = self.fixture()
        self.configure(source)
        before = self.commands(source)
        self.assertEqual(len(before), len(TARGETS) * 2 + 2)
        self.assertTrue(all("-O2" not in argv and "-DNDEBUG" not in argv
                            for argv in before.values()))
        self.configure(source, "-DGLIMMER_OPTIMIZE_EXECUTION_HOST=ON")
        enabled = self.commands(source)
        self.assertEqual(before.keys(), enabled.keys())
        for key, argv in enabled.items():
            intended = key in {target + ":.cc" for target in TARGETS}
            self.assertEqual(argv.count("-O2"), int(intended), key)
            self.assertEqual([arg for arg in argv if arg != "-O2"], before[key], key)
        self.configure(source, "-DGLIMMER_OPTIMIZE_EXECUTION_HOST=OFF")
        self.assertEqual(before, self.commands(source))

    def test_unsupported_configurations_and_compiler_reject(self):
        for options, compiler, message in (
                (["-DCMAKE_BUILD_TYPE=Release"], None, "single-configuration Debug"),
                (["-DCMAKE_BUILD_TYPE=RelWithDebInfo"], None, "single-configuration Debug"),
                (["-DCMAKE_CONFIGURATION_TYPES=Debug;Release"], None, "single-configuration Debug"),
                ([], "Unsupported", "GCC or Clang")):
            with self.subTest(options=options, compiler=compiler):
                source = self.fixture(compiler=compiler)
                result = self.configure(source, "-DGLIMMER_OPTIMIZE_EXECUTION_HOST=ON",
                                        *options, success=False)
                self.assertIn(message, result.stderr)

    def test_execution_target_required_only_when_enabled(self):
        source = self.fixture(execution=False)
        self.configure(source)
        result = self.configure(source, "-DGLIMMER_OPTIMIZE_EXECUTION_HOST=ON", success=False)
        self.assertIn("execution service or frontend", result.stderr)


if __name__ == "__main__":
    unittest.main()
