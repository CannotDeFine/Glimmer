"""Standard-library provenance, input and measurement contracts for BERT intake."""

import hashlib
import atexit
import ctypes
import json
import math
import os
import pathlib
import select
import stat
import struct
import threading
import time


MODEL_ID = "google-bert/bert-base-uncased"
MODEL_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"
WEIGHTS_SHA256 = "68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3"
WEIGHTS_BYTES = 440449768
DATA_REVISION = "240e165ab706d95bd4323653bb92421f446208cf"
SEED = 817
PROFILE = {"training": {"batch_size": 2, "sequence_length": 384},
           "inference": {"batch_size": 1, "sequence_length": 128}}
ASSETS = {
    **{name: f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/{name}"
       for name in ("config.json", "vocab.txt", "tokenizer_config.json", "model.safetensors")},
    **{name: f"https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/{DATA_REVISION}/dataset/{name}"
       for name in ("train-v1.1.json", "dev-v1.1.json")},
}


def sha256(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    pathlib.Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def configure_cache():
    """Keep optional framework caches inside the ignored project workspace."""
    root = pathlib.Path(__file__).resolve().parents[3] / ".cache" / "bert"
    for variable, name in (("HF_HOME", "huggingface"), ("TORCH_HOME", "torch"),
                           ("XDG_CACHE_HOME", "xdg"), ("CUDA_CACHE_PATH", "cuda")):
        os.environ[variable] = str(root / name)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


_START_EVENT = struct.Struct("iIII")
_START_PUBLISHED = 0x00000008 | 0x00000080  # IN_CLOSE_WRITE | IN_MOVED_TO
_START_INVALIDATED = 0x00000400 | 0x00000800 | 0x00002000 | 0x00004000 | 0x00008000
_START_MAX_NOTIFICATIONS = 8
# Process-owned bounded leases, including setup/active waits. Successful leases
# survive measured requests; only orderly exit or explicit quiescent retirement
# closes their descriptors. Never perform native close while holding this lock.
_start_notifications = []
_start_notifications_lock = threading.Lock()
_start_retirement_active = False


def _retire_start_notifications():
    global _start_retirement_active
    with _start_notifications_lock:
        if _start_retirement_active or any(owner.is_waiting for owner in _start_notifications):
            raise RuntimeError("cannot retire an active start barrier")
        owners = list(_start_notifications)
        _start_retirement_active = True
    first_error = None
    try:
        for owner in owners:
            try:
                owner.close()
            except OSError as error:
                if first_error is None:
                    first_error = error
    finally:
        with _start_notifications_lock:
            _start_retirement_active = False
    if first_error is not None:
        raise first_error


def _retire_start_notifications_at_exit():
    try:
        _retire_start_notifications()
    except BaseException:
        # Python normally ignores atexit exceptions and can still exit0. Make
        # failed cleanup observable; process containment closes remaining FDs.
        try:
            os.write(2, b"BERT start notification retirement failed\n")
        finally:
            os._exit(1)


atexit.register(_retire_start_notifications_at_exit)


def _start_event(data, watch, name):
    """Parse complete directory events; creation alone never publishes input."""
    offset, published = 0, False
    while offset < len(data):
        if len(data) - offset < _START_EVENT.size:
            raise ValueError("truncated start notification")
        owner, mask, _, length = _START_EVENT.unpack_from(data, offset)
        offset += _START_EVENT.size
        if (owner != watch or mask & _START_INVALIDATED or not length or length % 4
                or length > len(data) - offset):
            raise ValueError("start notification lost identity or overflowed")
        padded = data[offset:offset + length]
        offset += length
        leaf, separator, tail = padded.partition(b"\0")
        if not separator or not leaf or b"/" in leaf or any(tail):
            raise ValueError("noncanonical start notification name")
        if leaf == name:
            if not mask & _START_PUBLISHED or mask & ~_START_PUBLISHED:
                raise ValueError("start signal is not a published regular input")
            published = True
    return published


class _StartNotification:
    """Owned Linux watch installed before readiness; no polling fallback."""

    def __init__(self, path):
        self.directory = self.descriptor = -1
        self.is_waiting = True
        self.name = os.fsencode(path.name)
        with _start_notifications_lock:
            if _start_retirement_active:
                raise RuntimeError("start notification retirement is active")
            if len(_start_notifications) >= _START_MAX_NOTIFICATIONS:
                raise RuntimeError("start notification ownership capacity exhausted")
            _start_notifications.append(self)
        try:
            self.directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
            libc = ctypes.CDLL(None, use_errno=True)
            libc.inotify_init1.argtypes = [ctypes.c_int]
            libc.inotify_init1.restype = ctypes.c_int
            libc.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
            libc.inotify_add_watch.restype = ctypes.c_int
            self.descriptor = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
            if self.descriptor < 0:
                code = ctypes.get_errno()
                raise OSError(code, os.strerror(code))
            self.watch = libc.inotify_add_watch(
                self.descriptor, os.fsencode(f"/proc/self/fd/{self.directory}"),
                _START_PUBLISHED | 0x400 | 0x800)
            if self.watch < 0:
                code = ctypes.get_errno()
                raise OSError(code, os.strerror(code))
            self.require_absent()
        except BaseException:
            self.close()
            raise

    def require_absent(self):
        try:
            os.stat(self.name, dir_fd=self.directory, follow_symlinks=False)
        except FileNotFoundError:
            return
        raise ValueError("require a fresh start signal")

    def wait(self, timeout):
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("workload start barrier timed out")
            try:
                readable, _, _ = select.select([self.descriptor], [], [], remaining)
                if not readable:
                    raise TimeoutError("workload start barrier timed out")
                data = os.read(self.descriptor, 65536)
            except (InterruptedError, BlockingIOError):
                continue
            if not data:
                raise OSError("start notification ended")
            if _start_event(data, self.watch, self.name):
                signal = os.stat(self.name, dir_fd=self.directory, follow_symlinks=False)
                if not stat.S_ISREG(signal.st_mode) or signal.st_uid != os.geteuid():
                    raise ValueError("start signal must be an owned regular file")
                return

    def close(self):
        first_error = None
        for attribute in ("descriptor", "directory"):
            descriptor = getattr(self, attribute)
            setattr(self, attribute, -1)
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError as error:
                    if first_error is None:
                        first_error = error
        # Keep the lease charged while native close can still block. Failure
        # paths cannot admit replacement descriptors before both closes finish.
        with _start_notifications_lock:
            if self in _start_notifications:
                _start_notifications.remove(self)
        if first_error is not None:
            raise first_error

    def __enter__(self):
        return self

    def __exit__(self, exception_type, *_):
        if exception_type is not None:
            self.close()
        else:
            # inotify close can block for milliseconds. Keep its already
            # reserved bounded process lease, not a request-path cleanup thread.
            with _start_notifications_lock:
                self.is_waiting = False


def wait_for_start(ready, start, device_uuid, timeout=120):
    """Common event-driven measurement barrier, without GPU/scheduling work."""
    if not ready and not start:
        return
    if not ready or not start:
        raise ValueError("require paired barrier paths and a fresh start signal")
    if ready.resolve() == start.resolve():
        raise ValueError("readiness cannot also be the start signal")
    if type(timeout) not in (int, float) or not math.isfinite(timeout):
        raise ValueError("require a finite barrier timeout")
    with _StartNotification(start) as notification:
        # Exclusive creation prevents stale readiness. The watch already exists
        # when a collector sees this file, including an immediate close/release.
        with ready.open("x", encoding="utf-8") as target:
            json.dump({"pid": os.getpid(), "ready_ns": time.monotonic_ns(), "device_uuid": device_uuid}, target)
        notification.wait(timeout)


def validate_assets(directory):
    directory = pathlib.Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if {path.name for path in directory.iterdir()} != set(ASSETS) | {"manifest.json"}:
        raise ValueError("untracked asset files could change model or tokenizer loading")
    if (manifest.get("status") != "ok" or manifest.get("model_id") != MODEL_ID
            or manifest.get("model_revision") != MODEL_REVISION
            or manifest.get("data_revision") != DATA_REVISION
            or set(manifest.get("files", {})) != set(ASSETS)):
        raise ValueError("missing or mismatched asset provenance")
    for name, url in ASSETS.items():
        record = manifest["files"][name]
        path = directory / name
        if (record.get("url") != url or path.is_symlink() or not path.is_file()
                or record.get("bytes") != path.stat().st_size
                or record.get("sha256") != sha256(path)):
            raise ValueError(f"asset identity mismatch: {name}")
    if (manifest["files"]["model.safetensors"]["sha256"] != WEIGHTS_SHA256
            or manifest["files"]["model.safetensors"]["bytes"] != WEIGHTS_BYTES):
        raise ValueError("weights do not match the official pinned safetensors hash")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    required = {"model_type": "bert", "hidden_size": 768, "num_hidden_layers": 12,
                "num_attention_heads": 12, "intermediate_size": 3072, "vocab_size": 30522}
    if any(config.get(key) != value for key, value in required.items()):
        raise ValueError("the acceptance model must be complete BERT-base")
    return manifest


def squad_examples(document, count):
    """Select valid answerable examples in original order, without random fallback."""
    if type(count) is not int or count <= 0 or document.get("version") != "1.1":
        raise ValueError("expected SQuAD v1.1 and a positive example count")
    examples, identifiers = [], set()
    for article in document["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for question in paragraph["qas"]:
                identifier = question["id"]
                if not identifier or identifier in identifiers:
                    raise ValueError("empty or duplicate SQuAD example identity")
                identifiers.add(identifier)
                answer = question["answers"][0]
                begin, text = answer["answer_start"], answer["text"]
                if (type(begin) is not int or begin < 0 or not text
                        or context[begin:begin + len(text)] != text):
                    raise ValueError("SQuAD answer span does not match its context")
                examples.append({"id": identifier, "question": question["question"],
                                 "context": context, "answer_start": begin, "answer": text})
                if len(examples) == count:
                    return examples
    raise ValueError("insufficient real SQuAD examples")


def answer_tokens(offsets, sequence_ids, begin, end):
    """Return an answer wholly inside the context window; never train on a partial span."""
    context = [index for index, segment in enumerate(sequence_ids) if segment == 1]
    if not context or offsets[context[0]][0] > begin or offsets[context[-1]][1] < end:
        return None
    starts = [index for index in context if offsets[index][0] <= begin < offsets[index][1]]
    ends = [index for index in context if offsets[index][0] < end <= offsets[index][1]]
    if len(starts) != 1 or len(ends) != 1 or starts[0] > ends[0]:
        raise ValueError("answer cannot be mapped to complete context tokens")
    return starts[0], ends[0]


def summarize(samples_ns, elapsed_ns, batch_size):
    if (not samples_ns or any(type(value) is not int or value <= 0 for value in samples_ns)
            or type(elapsed_ns) is not int or elapsed_ns < sum(samples_ns)
            or type(batch_size) is not int or batch_size <= 0):
        raise ValueError("invalid measured interval or sample inventory")
    ordered = sorted(samples_ns)
    return {"samples": len(samples_ns), "mean_ms": sum(samples_ns) / len(samples_ns) / 1e6,
            "p99_ms": ordered[math.ceil(0.99 * len(ordered)) - 1] / 1e6,
            "p99_method": "nearest rank", "p99_sufficient_samples": len(samples_ns) >= 1000,
            "elapsed_seconds": elapsed_ns / 1e9,
            "steps_per_second": len(samples_ns) * 1e9 / elapsed_ns,
            "samples_per_second": len(samples_ns) * batch_size * 1e9 / elapsed_ns}
