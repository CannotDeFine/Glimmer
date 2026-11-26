#!/usr/bin/env python3
"""Qualify a host-range candidate with byte-identical checkpoint sharing."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile

import analyze_bert_host_range_optimization as optimization
import run_bert_frontend_wait as wait
from artifact_paths import PROJECT_ROOT

require = wait.require
FLOOR = 8 * 1024 ** 3
SNAPSHOT_BYTES = 871285428
MARGIN = 128 * 1024 ** 2
CAPTURE_BUDGET = 912680551
PLAN = PROJECT_ROOT / 'artifacts/maintenance/bert-host-range-optimization/attempt-uox8yg0a/plan.json'
CANONICAL = {
    'native': (PROJECT_ROOT / 'artifacts/runs/bert-current-candidate/attempt-zdzg6i9e/native/training/snapshot.pt',
               1342341, 0o444, '6ebc766be1716e340b9759549febab78b9567d46fce9bbb8984a57cf2b82677a'),
    'equal': (PROJECT_ROOT / 'artifacts/runs/bert-current-candidate/attempt-7ktvaxbo/equal/training/snapshot.pt',
              1342646, 0o664, '25343f4507ccfa1d8f6a67e25145537863e4592404de2d71a0abff5209b6c125'),
}


def snapshot(path):
    value = path.lstat()
    require(stat.S_ISREG(value.st_mode) and path.resolve(strict=True) == path,
            'canonical regular snapshot path required')
    return dict(device=value.st_dev, inode=value.st_ino, uid=value.st_uid,
                mode=stat.S_IMODE(value.st_mode), links=value.st_nlink, bytes=value.st_size,
                allocated=value.st_blocks * 512, mtime_ns=str(value.st_mtime_ns),
                ctime_ns=str(value.st_ctime_ns))


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish(path, value):
    """Publish complete receipts without replacing an earlier observation."""
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
        sync_directory(path.parent)
    finally:
        os.unlink(temporary)


def share_checkpoint(attempt, mode, unit, worker):
    """Share only a newly produced duplicate after the complete numerical oracle."""
    require(unit['unit_removed'] is True and unit['returncode'] == 0
            and worker['status'] == 'passed_requires_review'
            and worker['reparse'] == dict(tensors=401, elements=217787909,
                full_tensor_recomparison=True, source_bound_complete_inference=True,
                first_command_handoff=False), 'complete fresh numerical qualification required')
    wait.lifecycle.candidate.normal_cleanup(worker['job'])
    for child in worker['job']['authenticated_child_cleanup'].values():
        try:
            current = wait.lifecycle.candidate.identity(child['pid'])
        except (FileNotFoundError, ProcessLookupError):
            continue
        require(current != child['start'], 'a qualified child has not retired')
    target = attempt / mode / 'training/snapshot.pt'
    source, inode, permissions, checksum = CANONICAL[mode]
    original, fresh = snapshot(source), snapshot(target)
    require(original['inode'] == inode and original['mode'] == permissions
            and original['device'] == fresh['device'] and original['uid'] == fresh['uid'] == os.getuid()
            and original['bytes'] == fresh['bytes'] == SNAPSHOT_BYTES
            and fresh['links'] == 1 and fresh['mode'] == 0o664
            and worker['job']['artifact_identities']['training/snapshot.pt'] == checksum,
            'pinned canonical and fresh independent snapshot required')
    sha = hashlib.sha256()
    with source.open('rb') as first, target.open('rb') as second:
        while True:
            block = first.read(1024 ** 2)
            require(block == second.read(1024 ** 2), 'preserve nonidentical fresh snapshot')
            if not block:
                break
            sha.update(block)
    require(sha.hexdigest() == checksum and snapshot(source) == original and snapshot(target) == fresh,
            'snapshot changed during complete byte comparison')
    free_before = shutil.disk_usage(attempt).free
    receipt = dict(source=str(source), target=str(target), sha256=checksum,
                   canonical_before=original, fresh_before=fresh, free_before=free_before,
                   scope='After fresh full numerical validation; matching is only a storage gate.',
                   metadata_change='Canonical link count and ctime change; bytes/path/mode do not.',
                   writable_alias=bool(permissions & 0o222),
                   retention='Policy-immutable aliases; never modify either in place. Other independent historical copies are retained.')
    publish(attempt / (mode + '-checkpoint-share-intent.json'), receipt)
    with tempfile.TemporaryDirectory(prefix='.checkpoint-share-', dir=target.parent) as temporary:
        link = Path(temporary) / 'snapshot.pt'
        require(snapshot(source) == original and snapshot(target) == fresh, 'snapshot changed before sharing')
        os.link(source, link, follow_symlinks=False)
        linked = snapshot(link)
        require(linked['inode'] == inode and linked['device'] == original['device']
                and linked['mode'] == permissions and linked['links'] == original['links'] + 1
                and snapshot(target) == fresh, 'temporary link identity changed')
        os.replace(link, target)
        sync_directory(target.parent)
    after, retained = snapshot(source), snapshot(target)
    require(after == retained and after['inode'] == inode and after['mode'] == permissions
            and after['links'] == original['links'] + 1 and after['mtime_ns'] == original['mtime_ns']
            and wait.lifecycle.digest(target) == checksum, 'post-share identity or bytes changed')
    free_after = shutil.disk_usage(attempt).free
    publish(attempt / (mode + '-checkpoint-share-result.json'), dict(receipt,
        canonical_after=after, target_after=retained, free_after=free_after,
        free_delta=free_after - free_before, reclaimed_allocation=fresh['allocated']))
    require(free_after >= FLOOR + MARGIN
            and free_after - free_before >= fresh['allocated'] - 16 * 1024 ** 2,
            'free space not recovered; stop before another GPU unit')


def launch_with_retention(original_launch, attempt, mode, *, worker_script=None):
    numerical = worker_script is None
    require(mode in CANONICAL, 'explicit native/equal retention mode required')
    if numerical:
        require((attempt / 'manifest.json').is_file() and not (attempt / 'plan.json').exists(),
                'numerical qualification artifact shape required')
    else:
        require(worker_script == Path(wait.lifecycle.collector.__file__)
                and (attempt / 'plan.json').is_file() and not (attempt / 'manifest.json').exists(),
                'explicit capture worker and artifact shape required')
    budget = SNAPSHOT_BYTES + MARGIN if numerical else CAPTURE_BUDGET + MARGIN
    require(shutil.disk_usage(attempt).free >= FLOOR + budget, 'insufficient reserved output space')
    unit, worker = original_launch(attempt, mode, worker_script=worker_script)
    if numerical:
        share_checkpoint(attempt, mode, unit, worker)
    return unit, worker


def configure():
    scope = ORIGINAL_CONFIGURE()
    candidate = wait.lifecycle.candidate
    original_launch = candidate.launch_unit
    common, _ = wait.lifecycle.stage.BASE_HELPERS()
    original_inputs = common.input_paths

    def inputs(h):
        return (original_inputs(h) | {entry[0] for entry in CANONICAL.values()}
                | set(optimization.BASELINE_INPUTS) | {Path(optimization.host.__file__).resolve()})

    def launch(attempt, mode, *, worker_script=None):
        return launch_with_retention(original_launch, attempt, mode, worker_script=worker_script)

    common.input_paths, candidate.launch_unit = inputs, launch
    return scope


ORIGINAL_CONFIGURE = wait.configure


def main():
    wait.ENTRY = Path(__file__).resolve()
    wait.BUILD = PROJECT_ROOT / 'build/bert-host-range-optimization-1'
    wait.PLAN = PLAN
    wait.OUTPUT_SCHEMA, wait.HOST_ANALYZER = 3, optimization
    wait.configure = configure
    return wait.main()


if __name__ == '__main__':
    raise SystemExit(main())
