#!/usr/bin/env python3
"""Explicitly download pinned public BERT/SQuAD assets into a new local directory."""

import argparse
import hashlib
import json
import pathlib
import shutil
import urllib.request

from bert_contract import (ASSETS, DATA_REVISION, MODEL_ID, MODEL_REVISION, WEIGHTS_BYTES,
                           sha256, validate_assets, write_json)


def copy_previous(previous, output):
    """Copy only verified artifacts from a failed transfer; never mutate its evidence."""
    manifest = json.loads((previous / "manifest.json").read_text())
    if (manifest.get("status") != "failed" or manifest.get("model_id") != MODEL_ID
            or manifest.get("model_revision") != MODEL_REVISION
            or manifest.get("data_revision") != DATA_REVISION
            or set(manifest.get("files", {})) != set(ASSETS)):
        raise ValueError("resume requires a complete recorded failed asset inventory")
    for name, url in ASSETS.items():
        record, path = manifest["files"][name], previous / name
        if record.get("url") != url or path.is_symlink() or not path.is_file():
            raise ValueError("resume source identity changed")
        size = path.stat().st_size
        if name == "model.safetensors" and type(record.get("bytes")) is int and 0 < record["bytes"] < size < WEIGHTS_BYTES:
            # An interrupted older preparer can have an unrecorded downloaded
            # tail. Prove its recorded prefix; the entire candidate must still
            # pass the original official hash before it can become an asset.
            digest, remaining = hashlib.sha256(), record["bytes"]
            with path.open("rb") as source:
                while remaining:
                    block = source.read(min(remaining, 1024 * 1024))
                    if not block:
                        raise ValueError("resume prefix was truncated")
                    digest.update(block)
                    remaining -= len(block)
            matched = digest.hexdigest() == record.get("sha256")
        else:
            matched = record.get("bytes") == size and record.get("sha256") == sha256(path)
        if not matched:
            raise ValueError("resume source identity changed")
    if not 0 < manifest["files"]["model.safetensors"]["bytes"] < WEIGHTS_BYTES:
        raise ValueError("resume requires a strictly partial weight file")
    for name in ASSETS:
        shutil.copyfile(previous / name, output / name)
    return manifest["files"]


def download(url, path, limit, offset=0, expected_total=None, range_end=None):
    end = expected_total - 1 if range_end is None and expected_total is not None else range_end
    headers = {"Range": f"bytes={offset}-{end if range_end is not None else ''}"} if offset else {}
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        if not response.url.startswith("https://"):
            raise ValueError("asset redirect must retain HTTPS")
        if offset and (response.status != 206 or response.headers.get("Content-Range")
                       != f"bytes {offset}-{end}/{expected_total}"):
            raise ValueError("server did not confirm the exact requested resume range")
        declared = response.headers.get("Content-Length")
        declared = int(declared) if declared is not None else None
        total = offset
        with path.open("ab" if offset else "xb") as target:
            while block := response.read(1024 * 1024):
                total += len(block)
                if total > limit:
                    raise ValueError(f"asset exceeds size bound: {path.name}")
                target.write(block)
        if (total == offset or (declared is not None and total - offset != declared)
                or (end is not None and total != end + 1)):
            raise ValueError(f"incomplete download: {path.name}, received {total} bytes")
    return total


def prepare(output, previous=None):
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "failed", "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                "data_revision": DATA_REVISION, "files": {}}
    try:
        if previous is not None:
            manifest["files"] = copy_previous(previous, output)
        for name, url in ASSETS.items():
            if previous is not None and name != "model.safetensors":
                continue
            limit = 450_000_000 if name == "model.safetensors" else 40_000_000
            path = output / name
            offset = path.stat().st_size if previous is not None else 0
            print(f"Downloading {name}", flush=True)
            stops = (range(offset, WEIGHTS_BYTES, 8 * 1024 * 1024) if offset else (0,))
            for position in stops:
                end = min(position + 8 * 1024 * 1024, WEIGHTS_BYTES) - 1 if offset else None
                try:
                    download(url, path, limit, position,
                             WEIGHTS_BYTES if name == "model.safetensors" else None, end)
                finally:
                    if path.is_file():
                        manifest["files"][name] = {"url": url, "bytes": path.stat().st_size, "sha256": sha256(path)}
                        write_json(output / "manifest.json", manifest)
        manifest["status"] = "ok"
        write_json(output / "manifest.json", manifest)
        validate_assets(output)
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        write_json(output / "manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--resume-from", type=pathlib.Path,
                        help="copy a recorded failed download and resume only its partial weights")
    arguments = parser.parse_args()
    prepare(arguments.output, arguments.resume_from)


if __name__ == "__main__":
    main()
