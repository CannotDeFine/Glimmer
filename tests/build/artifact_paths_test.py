"""CPU-only path policy and fresh-attempt regressions."""

from concurrent.futures import ThreadPoolExecutor
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from scripts.artifact_paths import ArtifactRoot, create_attempt, resolve_artifact_root


class ArtifactPathsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="glimmer-artifact-paths-")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.project = self.base / "project"
        self.project.mkdir()

    def resolve(self, namespace="fixture", **options):
        return resolve_artifact_root(namespace, project_root=self.project, **options)

    def test_default_resolution_does_not_create_directories(self):
        root = self.resolve()
        self.assertEqual(root.path, self.project / "artifacts/runs/fixture")
        self.assertEqual(list(self.project.iterdir()), [])

    def test_import_never_creates_output(self):
        import scripts.artifact_paths as paths
        with mock.patch.object(Path, "mkdir", side_effect=AssertionError("cold import wrote output")), \
                mock.patch.object(paths.tempfile, "mkdtemp", side_effect=AssertionError("cold import wrote output")):
            importlib.reload(paths)

    def test_bad_names_and_categories(self):
        for name in ("", ".", "..", "a/b", "a\\b", "a b", "a" * 65, None):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.resolve(name)
        with self.assertRaises(ValueError):
            self.resolve(category="build")

    def test_source_and_ancestor_roots_are_rejected(self):
        for root in (self.project, self.base, Path("/"), self.project / "src/results",
                     self.project / "assets/results", self.project / ".deps/results",
                     self.project / ".git/results", self.project / "build/results"):
            with self.subTest(root=root), self.assertRaises(ValueError):
                self.resolve(output_root=root)
        self.assertEqual(list(self.project.iterdir()), [])

    def test_atomic_unique_attempts_and_metadata(self):
        root = self.resolve(category="profiles")
        with ThreadPoolExecutor(max_workers=4) as pool:
            attempts = list(pool.map(lambda _: create_attempt(root), range(16)))
        self.assertEqual(len(set(attempts)), 16)
        for attempt in attempts:
            record = json.loads((attempt / "run.json").read_text())
            self.assertEqual(record["layout_version"], 1)
            self.assertEqual(record["category"], "profiles")
            self.assertFalse(record["legacy_build_root"])
            self.assertEqual(record["output_root"], str(root.path))

    def test_legacy_write_exception_is_retired(self):
        previous = Path.cwd()
        os.chdir(self.project)
        self.addCleanup(os.chdir, previous)
        for output in (Path("build/legacy"), self.project / "build/legacy"):
            with self.subTest(output=output):
                with self.assertRaisesRegex(ValueError, "legacy write exception is retired"):
                    self.resolve(output_root=output, allow_legacy_build_root=True)
        with self.assertRaisesRegex(ValueError, "read-only"):
            create_attempt(ArtifactRoot(self.project / "build/legacy", "fixture", "runs", True))
        self.assertFalse((self.project / "build").exists())

    def test_creation_revalidates_actual_path_not_legacy_flag(self):
        for path in (self.project / "build/forged", self.project / "src/forged"):
            root = ArtifactRoot(path, "fixture", "runs", project_root=self.project)
            with self.subTest(path=path), self.assertRaises(ValueError):
                create_attempt(root)
            self.assertFalse(path.exists())
        root = self.resolve()
        root.path.parent.mkdir(parents=True)
        source = self.project / "src"
        source.mkdir()
        root.path.symlink_to(source, target_is_directory=True)
        with self.assertRaises(ValueError):
            create_attempt(root)
        self.assertEqual(list(source.iterdir()), [])

    def test_symlinks_cannot_disguise_build_or_source_output(self):
        artifacts = self.project / "artifacts"
        artifacts.mkdir()
        (self.project / "build").mkdir()
        (self.project / "src").mkdir()
        for target in (self.project / "build", self.project / "src", self.base):
            alias = artifacts / "alias"
            alias.symlink_to(target, target_is_directory=True)
            with self.subTest(target=target), self.assertRaises(ValueError):
                self.resolve(output_root=alias / "run", allow_legacy_build_root=True)
            alias.unlink()
        external_alias = self.base / "alias"
        external_alias.symlink_to(self.project / "build", target_is_directory=True)
        with self.assertRaises(ValueError):
            self.resolve(output_root=external_alias / "run")

    def test_source_alias_cannot_redirect_project_local_output_outside_project(self):
        source = self.project / "src"
        source.mkdir()
        external = self.base / "external"
        external.mkdir()
        alias = source / "alias"
        alias.symlink_to(external, target_is_directory=True)
        for spelling in (alias / "run", self.project / "artifacts/../src/alias/run"):
            with self.subTest(spelling=spelling), self.assertRaises(ValueError):
                self.resolve(output_root=spelling)
        self.assertEqual(list(external.iterdir()), [])

    def test_default_boundary_cannot_be_redirected(self):
        (self.project / "artifacts").symlink_to(self.base, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.resolve()

    def test_build_root_symlink_does_not_enable_external_evidence(self):
        external = self.base / "external-build"
        external.mkdir()
        (self.project / "build").symlink_to(external, target_is_directory=True)
        for path in (self.project / "build/run", external / "run"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.resolve(output_root=path, allow_legacy_build_root=True)

    def test_existing_file_is_not_an_output_directory(self):
        target = self.base / "file"
        target.touch()
        with self.assertRaises(ValueError):
            self.resolve(output_root=target)

    def test_cli_help_and_import_are_cold_and_work_outside_repository(self):
        script = PROJECT / "scripts/artifact_paths.py"
        for command in ([sys.executable, "-B", str(script), "--help"],
                        [sys.executable, "-B", "-c",
                         f"import sys; sys.path.insert(0, {str(PROJECT)!r}); import scripts.artifact_paths"]):
            result = subprocess.run(command, cwd=self.base, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(list(self.project.iterdir()), [])
        output = self.base / "external"
        result = subprocess.run([sys.executable, "-B", str(script), "--name", "cli",
                                 "--output-root", str(output)], cwd=self.base,
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((Path(result.stdout.strip()) / "run.json").is_file())


if __name__ == "__main__":
    unittest.main()
