"""CPU-only installed NCCL discovery checks; no SDK code or network execution."""

import argparse
from pathlib import Path
import subprocess
import tempfile
import unittest


PARSER = argparse.ArgumentParser(add_help=False)
PARSER.add_argument("--cmake", default="cmake")
OPTIONS, UNITTEST_ARGUMENTS = PARSER.parse_known_args()
HELPER = Path(__file__).resolve().parents[2] / "cmake/NcclDependency.cmake"
HEADER = """#define NCCL_MAJOR 2
#define NCCL_MINOR 29
#define NCCL_PATCH 3
#define NCCL_VERSION_CODE 22903
"""


class NcclDependencyTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="glimmer-nccl-sdk-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.root = self.directory / "installed SDK with spaces"
        self.make_sdk(self.root)
        self.source = self.directory / "source"
        self.source.mkdir()
        self.build = self.directory / "build"

    @staticmethod
    def make_sdk(root):
        (root / "include").mkdir(parents=True)
        (root / "lib").mkdir()
        (root / "include/nccl.h").write_text(HEADER)
        # A discovery-only fixture. Never compiled, linked or executed.
        (root / "lib/libnccl.so.2").write_bytes(b"NCCL discovery fixture\n")

    def configure(self, body=None, extra=()):
        if body is None:
            body = f'glimmer_add_nccl_sdk("{self.root}")\n'
        (self.source / "CMakeLists.txt").write_text(f"""
cmake_minimum_required(VERSION 3.20)
project(NcclDiscoveryFixture LANGUAGES NONE)
include("{HELPER}")
{body}
""")
        return subprocess.run(
            [OPTIONS.cmake, "-S", str(self.source), "-B", str(self.build), *extra],
            text=True, capture_output=True, timeout=30, check=False,
        )

    def require_failure(self, result, message):
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(message, result.stdout + result.stderr)

    def test_positive_boundaries_repeated_root_and_literal_spaces(self):
        result = self.configure(f"""
glimmer_add_nccl_sdk("{self.root}")
glimmer_add_nccl_sdk("{self.root}")
get_target_property(headers glimmer_nccl_headers INTERFACE_SYSTEM_INCLUDE_DIRECTORIES)
get_target_property(definitions glimmer_nccl_headers INTERFACE_COMPILE_DEFINITIONS)
get_target_property(library Glimmer::nccl IMPORTED_LOCATION)
get_target_property(links glimmer_nccl_headers INTERFACE_LINK_LIBRARIES)
if(NOT headers STREQUAL "{self.root}/include" OR
        NOT definitions STREQUAL "GLIMMER_HAS_NCCL=1" OR
        NOT library STREQUAL "{self.root}/lib/libnccl.so.2" OR links)
    message(FATAL_ERROR "Incorrect private header/vendor link boundary")
endif()
""")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_no_sdk_when_discovery_not_called(self):
        result = self.configure("""
if(TARGET Glimmer::nccl OR TARGET Glimmer::nccl_headers)
    message(FATAL_ERROR "Unexpected unconditional NCCL dependency")
endif()
""")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_execution_activation_requires_all_explicit_inputs(self):
        inputs = (
            f"-DGLIMMER_NCCL_ROOT={self.root}",
            "-DGLIMMER_BUILD_CUDA_EXECUTION_SERVICE=ON",
            "-DGLIMMER_BUILD_CUDA_EXECUTION_INTERCEPTOR=ON",
        )
        for missing in range(len(inputs)):
            with self.subTest(missing=inputs[missing]):
                result = subprocess.run(
                    [OPTIONS.cmake, "-S", str(HELPER.parent.parent),
                     "-B", str(self.directory / f"activation-{missing}"),
                     "-DBUILD_TESTING=OFF", "-DGLIMMER_ENABLE_NCCL_EXECUTION=ON",
                     "-DGLIMMER_BUILD_CUDA_BACKEND=ON",
                     *(value for index, value in enumerate(inputs) if index != missing)],
                    text=True, capture_output=True, timeout=30, check=False,
                )
                self.require_failure(result, "NCCL execution requires GLIMMER_NCCL_ROOT")

    def test_execution_activation_does_not_bypass_backend_requirement(self):
        result = subprocess.run(
            [OPTIONS.cmake, "-S", str(HELPER.parent.parent),
             "-B", str(self.directory / "activation-no-backend"),
             "-DBUILD_TESTING=OFF", "-DGLIMMER_ENABLE_NCCL_EXECUTION=ON",
             f"-DGLIMMER_NCCL_ROOT={self.root}",
             "-DGLIMMER_BUILD_CUDA_EXECUTION_SERVICE=ON",
             "-DGLIMMER_BUILD_CUDA_EXECUTION_INTERCEPTOR=ON"],
            text=True, capture_output=True, timeout=30, check=False,
        )
        self.require_failure(result, "native NCCL packets require")

    def test_relative_root(self):
        self.require_failure(self.configure('glimmer_add_nccl_sdk("relative")'),
                             "absolute installed NCCL SDK root")

    def test_missing_header(self):
        (self.root / "include/nccl.h").unlink()
        self.require_failure(self.configure(), "absolute installed NCCL SDK root")

    def test_missing_library_ignores_preexisting_cache(self):
        (self.root / "lib/libnccl.so.2").unlink()
        self.require_failure(self.configure(extra=(
            "-D_glimmer_nccl_library=/tmp/not-a-real-approved-nccl.so",)),
            "No installed NCCL library")

    def test_unsupported_version(self):
        (self.root / "include/nccl.h").write_text(HEADER.replace("MINOR 29", "MINOR 28"))
        self.require_failure(self.configure(), "audited SDK 2.29.3")

    def test_missing_or_ambiguous_version(self):
        header = self.root / "include/nccl.h"
        for content in (HEADER.replace("#define NCCL_PATCH 3\n", ""),
                        HEADER + "#define NCCL_PATCH 4\n"):
            header.write_text(content)
            self.require_failure(self.configure(), "ambiguous version header")

    def test_inconsistent_version_code(self):
        (self.root / "include/nccl.h").write_text(HEADER.replace("22903", "22902"))
        self.require_failure(self.configure(), "inconsistent version code")

    def test_conflicting_roots(self):
        second = self.directory / "second SDK"
        self.make_sdk(second)
        self.require_failure(self.configure(f"""
glimmer_add_nccl_sdk("{self.root}")
glimmer_add_nccl_sdk("{second}")
"""), "Only one NCCL SDK root")

    def test_equivalent_symlink_root(self):
        alias = self.directory / "SDK alias"
        alias.symlink_to(self.root, target_is_directory=True)
        result = self.configure(f"""
glimmer_add_nccl_sdk("{self.root}")
glimmer_add_nccl_sdk("{alias}")
""")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main(argv=[__file__, *UNITTEST_ARGUMENTS])
