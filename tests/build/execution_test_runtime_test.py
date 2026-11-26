"""Small CMake testing-configuration and execution-fixture controls."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[2]
APPLICATION = "glimmer_execution_dnn_rnn_weight_params_test"


class TestingConfigurationTest(unittest.TestCase):
    def test_python_is_required_only_when_testing_is_enabled(self):
        with tempfile.TemporaryDirectory(prefix="glimmer-testing-config-") as temporary:
            for enabled in (False, True):
                with self.subTest(testing=enabled):
                    build = Path(temporary) / str(enabled)
                    result = subprocess.run(
                        ["cmake", "-S", str(PROJECT), "-B", str(build),
                         f"-DBUILD_TESTING={'ON' if enabled else 'OFF'}",
                         "-DCMAKE_EXPORT_COMPILE_COMMANDS=OFF",
                         "-DCMAKE_DISABLE_FIND_PACKAGE_Python3=TRUE"],
                        capture_output=True, text=True, timeout=60)
                    if enabled:
                        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                        self.assertIn("Python3", result.stderr)
                    else:
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class ExecutionTestRuntimeTest(unittest.TestCase):
    def command(self, *command, cwd=None, success=True):
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=60)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def fixture(self, *, enabled=True, profile="9", wrap=False):
        temporary = tempfile.TemporaryDirectory(prefix="glimmer-test-runtime-")
        self.addCleanup(temporary.cleanup)
        source = Path(temporary.name)
        (source / "runtime.cc").write_text(
            'extern "C" int dependency();\nextern "C" int runtime() { return dependency(); }\n')
        (source / "dependency.cc").write_text('extern "C" int dependency() { return 1; }\n')
        (source / "main.cc").write_text(
            'extern "C" int runtime();\nint main() { return runtime() != 1; }\n')
        (source / "wrapper.cc").write_text(
            'extern "C" int runtime();\nextern "C" int __wrap_dependency() { return 7; }\n'
            'int main() { return runtime() != 7; }\n')
        (source / "CMakeLists.txt").write_text(f'''
cmake_minimum_required(VERSION 3.20)
project(ExecutionFixture LANGUAGES CXX)
set(BUILD_TESTING ON)
set(GLIMMER_SHARE_EXECUTION_TEST_RUNTIME {"ON" if enabled else "OFF"})
set(GLIMMER_CUDNN_ABI_PROFILE "{profile}")
set(PROJECT_SOURCE_DIR "{PROJECT.as_posix()}")
function(glimmer_enable_clang_tidy target)
endfunction()
include("{PROJECT.as_posix()}/cmake/ExecutionTestRuntime.cmake")
add_library(options INTERFACE)
add_library(Glimmer::project_options ALIAS options)
add_library(dependency STATIC dependency.cc)
set_target_properties(dependency PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_library(glimmer_execution_service_runtime STATIC runtime.cc)
target_link_libraries(glimmer_execution_service_runtime PRIVATE dependency)
glimmer_create_execution_test_runtime()
add_executable({APPLICATION} main.cc)
target_link_libraries({APPLICATION} PRIVATE glimmer_execution_service_runtime dependency)
{"target_link_options(" + APPLICATION + " PRIVATE -Wl,--wrap=dependency)" if wrap else ""}
glimmer_share_execution_fixture({APPLICATION})
glimmer_share_execution_fixture({APPLICATION})
add_executable(wrapper wrapper.cc)
target_link_libraries(wrapper PRIVATE glimmer_execution_service_runtime dependency)
target_link_options(wrapper PRIVATE -Wl,--wrap=dependency)
glimmer_share_execution_fixture(wrapper)
add_executable(production main.cc)
target_link_libraries(production PRIVATE glimmer_execution_service_runtime dependency)
glimmer_share_execution_fixture(production)
''')
        return source, source / "build"

    def test_shared_runtime_and_static_wrapper_keep_behavior_and_reuse(self):
        source, build = self.fixture()
        self.command("cmake", "-S", str(source), "-B", str(build))
        self.command("cmake", "--build", str(build), "-j1")
        shared = build / "lib/test-support/libglimmer_execution_test_runtime.so"
        self.assertTrue(shared.is_file())
        needed = self.command("readelf", "-d", str(build / APPLICATION)).stdout
        self.assertIn("libglimmer_execution_test_runtime.so", needed)
        for target in ("wrapper", "production"):
            self.assertNotIn("libglimmer_execution_test_runtime.so",
                             self.command("readelf", "-d", str(build / target)).stdout)
        for target in (APPLICATION, "wrapper", "production"):
            self.command(str(build / target))
        before = {path: path.stat().st_mtime_ns for path in build.rglob("*.o")}
        self.command("cmake", "--build", str(build), "-j1")
        self.assertEqual(before, {path: path.stat().st_mtime_ns for path in before})

    def test_opt_out_and_sdk8_remain_static(self):
        for options in ({"enabled": False}, {"profile": "8"}):
            with self.subTest(options=options):
                source, build = self.fixture(**options)
                self.command("cmake", "-S", str(source), "-B", str(build))
                self.command("cmake", "--build", str(build), "-j1")
                self.assertFalse((build / "lib/test-support/libglimmer_execution_test_runtime.so").exists())
                self.command(str(build / APPLICATION))
                self.command(str(build / "wrapper"))

    def test_allowlisted_symbol_rewriting_is_a_configuration_error(self):
        source, build = self.fixture(wrap=True)
        result = self.command("cmake", "-S", str(source), "-B", str(build), success=False)
        self.assertIn("symbol-rewriting", result.stderr)


if __name__ == "__main__":
    if not all(shutil.which(tool) for tool in ("cmake", "c++", "readelf")):
        raise SystemExit("CMake/linker controls require cmake, c++ and readelf")
    unittest.main()
