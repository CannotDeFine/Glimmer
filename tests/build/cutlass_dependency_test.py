"""CPU-only CMake dependency gates; no CUDA, network, or real index changes."""

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

PARSER = argparse.ArgumentParser(add_help=False)
PARSER.add_argument("--cmake", default="cmake")
OPTIONS, UNITTEST_ARGUMENTS = PARSER.parse_known_args()
HELPER = Path(__file__).resolve().parents[2] / "cmake/CutlassDependency.cmake"
PINNED_REVISION = "f7b19de32c5d1f3cedfc735c2849f12b537522ee"


class CutlassDependencyTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="glimmer-cutlass-gate-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.root = self.directory / "checkout with spaces"
        self.make_checkout(self.root)
        self.git = self.directory / "fixture-git"
        self.git.write_text(f"#!{sys.executable}\n" + """
from pathlib import Path
import sys
arguments = sys.argv[1:]
if arguments == ['--version']:
    print('git version 2.43.0')
    sys.exit(0)
if len(arguments) < 4 or arguments[0] != '-C':
    sys.exit(2)
root = Path(arguments[1])
command = arguments[2:]
if (root / '.fail-git').exists():
    sys.exit(3)
if command == ['rev-parse', '--verify', 'HEAD']:
    print((root / '.revision').read_text().strip())
elif command == ['status', '--porcelain', '--untracked-files=all',
                 '--', 'include', 'LICENSE.txt']:
    if (root / '.dirty').exists():
        print((root / '.dirty').read_text().strip())
else:
    sys.exit(4)
""")
        self.git.chmod(0o700)

    def make_checkout(self, root):
        header = root / "include/cutlass/gemm/device/gemm.h"
        header.parent.mkdir(parents=True)
        header.write_text("// Test fixture, not CUTLASS source.\n")
        (root / "LICENSE.txt").write_text("Test fixture license\n")
        (root / ".revision").write_text(PINNED_REVISION)

    def run_cmake(self, *arguments):
        return subprocess.run(
            [OPTIONS.cmake, *map(str, arguments)], capture_output=True,
            text=True, timeout=30, check=False,
        )

    def script_check(self):
        return self.run_cmake(
            f"-DGLIMMER_CUTLASS_ROOT={self.root}",
            f"-DGIT_EXECUTABLE={self.git}", "-P", HELPER,
        )

    def require_failure(self, result, message):
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(message, result.stdout + result.stderr)

    def test_pinned_clean_input_and_literal_spaces(self):
        result = self.script_check()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_missing_header_and_license(self):
        for path in (self.root / "include/cutlass/gemm/device/gemm.h",
                     self.root / "LICENSE.txt"):
            original = path.read_text()
            path.unlink()
            self.require_failure(self.script_check(), "CUTLASS is missing")
            path.write_text(original)

    def test_wrong_revision(self):
        (self.root / ".revision").write_text("0" * 40)
        self.require_failure(self.script_check(), "reviewed v3.5.1 revision")

    def test_changed_and_untracked_headers_and_license(self):
        for change in (" M include/cutlass/gemm/device/gemm.h",
                       "?? include/local.h", " M LICENSE.txt"):
            (self.root / ".dirty").write_text(change)
            self.require_failure(self.script_check(), "must be unchanged")

    def test_git_failure(self):
        (self.root / ".fail-git").touch()
        self.require_failure(self.script_check(), "reviewed v3.5.1 revision")

    def configure_target(self, extra=""):
        source = self.directory / "source"
        source.mkdir()
        (source / "main.cc").write_text("int main() { return 0; }\n")
        (source / "CMakeLists.txt").write_text(f"""
cmake_minimum_required(VERSION 3.20)
project(CutlassGateFixture LANGUAGES CXX)
include("{HELPER}")
glimmer_add_cutlass_headers("{self.root}")
glimmer_add_cutlass_headers("{self.root}")
if(NOT GLIMMER_CUTLASS_REVISION STREQUAL "{PINNED_REVISION}")
    message(FATAL_ERROR "Missing revision output")
endif()
get_target_property(system_headers glimmer_cutlass_headers
    INTERFACE_SYSTEM_INCLUDE_DIRECTORIES)
if(NOT system_headers STREQUAL "{self.root}/include")
    message(FATAL_ERROR "Missing SYSTEM include boundary")
endif()
add_executable(consumer main.cc)
target_link_libraries(consumer PRIVATE Glimmer::cutlass_headers)
{extra}
""")
        build = self.directory / "build"
        result = self.run_cmake("-S", source, "-B", build,
                                f"-DGIT_EXECUTABLE={self.git}")
        return result, build

    def test_header_target_build_and_post_configure_revalidation(self):
        result, build = self.configure_target()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        result = self.run_cmake("--build", build, "--target", "consumer")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue((build / "consumer").is_file())
        (self.root / ".dirty").write_text(" M include/cutlass/gemm/device/gemm.h")
        self.require_failure(
            self.run_cmake("--build", build, "--target", "consumer"),
            "must be unchanged",
        )
        (self.root / ".dirty").unlink()
        (self.root / ".revision").write_text("0" * 40)
        self.require_failure(
            self.run_cmake("--build", build, "--target", "consumer"),
            "reviewed v3.5.1 revision",
        )

    def test_conflicting_roots_fail(self):
        other = self.directory / "other checkout"
        self.make_checkout(other)
        result, _ = self.configure_target(f'glimmer_add_cutlass_headers("{other}")')
        self.require_failure(result, "Conflicting CUTLASS roots")

    def test_disabled_dependency_needs_no_checkout_or_git(self):
        source = self.directory / "default-source"
        thirdparty = source / "3rdparty"
        spdlog = thirdparty / "spdlog"
        spdlog.mkdir(parents=True)
        (spdlog / "CMakeLists.txt").write_text("add_library(spdlog INTERFACE)\n")
        original = HELPER.parent.parent / "3rdparty/CMakeLists.txt"
        (thirdparty / "CMakeLists.txt").write_text(original.read_text())
        (source / "CMakeLists.txt").write_text("""
cmake_minimum_required(VERSION 3.20)
project(DisabledCutlassFixture LANGUAGES NONE)
set(GLIMMER_BUILD_GEMM_COMPARISON OFF)
add_subdirectory(3rdparty)
if(TARGET Glimmer::cutlass_headers OR TARGET glimmer_verify_cutlass)
    message(FATAL_ERROR "Disabled CUTLASS was initialized")
endif()
""")
        result = self.run_cmake(
            "-S", source, "-B", self.directory / "default-build",
            "-DCMAKE_DISABLE_FIND_PACKAGE_Git=TRUE",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], *UNITTEST_ARGUMENTS])
