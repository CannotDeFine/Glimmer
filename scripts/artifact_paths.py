"""Create fresh experiment attempts outside reusable CMake build trees."""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ("runs", "profiles", "maintenance", "archives")


@dataclass(frozen=True)
class ArtifactRoot:
    path: Path
    namespace: str
    category: str
    legacy_build_root: bool = False
    project_root: Path = PROJECT_ROOT


def resolve_artifact_root(namespace, *, category="runs", output_root=None,
                          project_root=PROJECT_ROOT, allow_legacy_build_root=False):
    """Validate without creating files; explicit relative roots use the caller CWD.

    Canonicalization prevents existing symlinks from disguising a build/source
    output as an artifact root. This is a developer tool, not a security boundary
    against concurrent hostile directory replacement.
    The deprecated allow_legacy_build_root argument no longer enables writes.
    """
    if (not isinstance(namespace, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", namespace) is None):
        raise ValueError("artifact namespace must be a simple name of at most 64 characters")
    if category not in CATEGORIES:
        raise ValueError("unknown artifact category")
    project = Path(project_root).resolve()
    # The default boundary itself must not be redirected into another tree.
    artifact_directory = project / "artifacts"
    if artifact_directory.resolve() != artifact_directory:
        raise ValueError("project artifacts directory must not be a symlink")
    selected = (artifact_directory / category / namespace
                if output_root is None else Path(os.path.abspath(output_root)))
    canonical = selected.resolve()
    if canonical == Path(canonical.anchor) or canonical == project or canonical in project.parents:
        raise ValueError("artifact root must not be the repository or an ancestor")
    legacy = (selected.is_relative_to(project / "build")
              or canonical.is_relative_to((project / "build").resolve()))
    if legacy:
        raise ValueError("new evidence belongs in artifacts/, not build/; the legacy write exception is retired")
    if selected.is_relative_to(project) and not selected.is_relative_to(artifact_directory):
        raise ValueError("project-local evidence must be inside artifacts/")
    if canonical.is_relative_to(project) and not canonical.is_relative_to(artifact_directory):
        raise ValueError("project-local evidence must be inside artifacts/")
    # Do not silently redirect a project-local spelling to an external tree.
    if selected.is_relative_to(artifact_directory) and not canonical.is_relative_to(artifact_directory):
        raise ValueError("artifact symlink escapes the artifacts directory")
    if canonical.exists() and not canonical.is_dir():
        raise ValueError("artifact root is not a directory")
    return ArtifactRoot(canonical, namespace, category, legacy, project)


def create_attempt(root):
    """Atomically allocate a unique directory; never reuse an existing attempt."""
    if root.legacy_build_root:
        raise ValueError("historical build evidence is read-only; create a fresh artifacts/ attempt")
    root = resolve_artifact_root(root.namespace, category=root.category,
                                 output_root=root.path, project_root=root.project_root)
    root.path.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=root.path))
    metadata = {
        "layout_version": 1, "namespace": root.namespace, "category": root.category,
        "output_root": str(root.path), "legacy_build_root": root.legacy_build_root,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    with (attempt / "run.json").open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")
    return attempt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="short experiment namespace")
    parser.add_argument("--category", choices=CATEGORIES, default="runs")
    parser.add_argument("--output-root", type=Path, help="explicit dedicated output directory")
    parser.add_argument("--allow-legacy-build-root", action="store_true",
                        help="deprecated; no longer permits new output under build/")
    args = parser.parse_args()
    try:
        root = resolve_artifact_root(
            args.name, category=args.category, output_root=args.output_root,
            allow_legacy_build_root=args.allow_legacy_build_root)
    except ValueError as error:
        parser.error(str(error))
    print(create_attempt(root))


if __name__ == "__main__":
    main()
