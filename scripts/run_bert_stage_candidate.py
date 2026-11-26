#!/usr/bin/env python3
"""Qualify a service-only diagnostic candidate against complete BERT outputs.

The immutable baseline libraries and frontend are reused. The new service is
explicitly selected; no prior numerical checkpoint qualifies this binary.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess

import run_bert_candidate_validation as candidate
from artifact_paths import PROJECT_ROOT
from run_rnn_v8_validation import digest

BUILD = PROJECT_ROOT / 'build/bert-host-stage-diagnostic-1'
SERVER = BUILD / 'bin/glimmer_execution_service'
DEADLINE = datetime.fromisoformat('2026-10-31T18:02:53+00:00')
BASE_HELPERS = candidate.load_helpers
require = candidate.require


def authorization_budget(seconds):
    require((DEADLINE - datetime.now(timezone.utc)).total_seconds() >= seconds,
            'insufficient remaining diagnostic authorization')


def build_identity():
    """Check the actual inherited compile/link inputs, not build-type labels."""
    baseline = candidate.BUILD
    original = json.loads((baseline / 'compile_commands.json').read_text())
    current = json.loads((BUILD / 'compile_commands.json').read_text())

    def commands(rows):
        result = {}
        for row in rows:
            if '/src/execution_service/' not in row['file']:
                continue
            args = shlex.split(row['command'])
            output = args.index('-o')
            result[row['file']] = args[:output] + args[output + 2:]
        return result

    old, new = commands(original), commands(current)
    require(len(old) == 7 and all(new.get(path) == args for path, args in old.items()),
            'preexisting service translation-unit compile flags differ')

    def dynamic(path):
        text = subprocess.check_output(['readelf', '-d', str(path)], text=True, timeout=10)
        return [line.split(')', 1)[1].strip() for line in text.splitlines()
                if any('(' + key + ')' in line for key in ('NEEDED', 'RUNPATH', 'RPATH'))]

    old_server = baseline / 'bin/glimmer_execution_service'
    require(dynamic(old_server) == dynamic(SERVER), 'candidate dynamic dependency contract differs')
    old_link = baseline / 'src/execution_service/CMakeFiles/glimmer_execution_service.dir/link.txt'
    new_link = BUILD / 'service/CMakeFiles/glimmer_execution_service.dir/link.txt'

    def link(path):
        args = shlex.split(path.read_text())
        resolved = [(path.parents[2] / value).resolve() if not value.startswith('-') else value
                    for value in args]
        return [str(value) for value in resolved]

    before, after = link(old_link), link(new_link)
    substitutions = {
        str((baseline / 'src/execution_service/CMakeFiles/glimmer_execution_service.dir/main.cc.o').resolve()):
            str((BUILD / 'service/CMakeFiles/glimmer_execution_service.dir/main.cc.o').resolve()),
        str(old_server): str(SERVER),
        str(baseline / 'src/execution_service/libglimmer_execution_service_runtime.a'):
            str(BUILD / 'service/libglimmer_execution_service_runtime.a'),
    }
    require([substitutions.get(arg, arg) for arg in before] == after,
            'candidate ordered link contract differs beyond service replacement')
    paths = {Path(arg) for arg in before + after if Path(arg).is_absolute() and Path(arg).is_file()}
    paths.update({old_link, new_link, BUILD / 'CMakeCache.txt', BUILD / 'compile_commands.json',
                  baseline / 'CMakeCache.txt', baseline / 'compile_commands.json',
                  PROJECT_ROOT / 'cmake/ExecutionServiceDiagnosticOverlay/CMakeLists.txt'})
    return dict(compile_equivalent_files=sorted(old), dynamic_dependencies=dynamic(SERVER),
                baseline_link=before, candidate_link=after,
                identities={str(path): digest(path) for path in sorted(paths)})


def helpers():
    common, h = BASE_HELPERS()
    h.training.SERVER = SERVER
    return common, h


def bind_workers(expected_sha):
    """Select the candidate in the actual systemd worker, not just its parent."""
    original_launch = candidate.launch_unit
    original_cleanup = candidate.normal_cleanup

    def launch(attempt, mode, *, worker_script=None):
        require(worker_script is None, 'qualifier owns its explicit worker entrypoint')
        return original_launch(attempt, mode, worker_script=Path(__file__).resolve())

    def cleanup(job):
        original_cleanup(job)
        if job['mode'] != 'native':
            require(job['server_command'][0] == str(SERVER)
                    and digest(SERVER) == expected_sha
                    and '--host-stage-output' not in job['server_command'],
                    'actual numerical worker must execute the selected uninstrumented candidate')

    candidate.launch_unit, candidate.normal_cleanup = launch, cleanup


def main():
    authorization_budget(2600)
    binding = build_identity()
    common, _ = BASE_HELPERS()
    original_inputs = common.input_paths

    def inputs(h):
        return original_inputs(h) | {Path(__file__).resolve(),
            PROJECT_ROOT / 'scripts/run_bert_candidate_validation.py',
            PROJECT_ROOT / 'src/execution_service/host_stage_metrics.h',
            PROJECT_ROOT / 'src/execution_service/host_stage_metrics.cc',
            *(Path(path) for path in binding['identities'])}

    common.input_paths = inputs
    candidate.load_helpers = helpers
    original_save = candidate.save

    def save(path, value):
        if path.name == 'manifest.json':
            original_save(path.parent / 'service-build-identity.json', binding)
        return original_save(path, value)

    candidate.save = save
    bind_workers(binding['identities'][str(SERVER)])
    return candidate.main()


if __name__ == '__main__':
    raise SystemExit(main())
