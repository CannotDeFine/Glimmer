#!/usr/bin/env python3
"""Prepare or explicitly collect current-Debug sustained BERT observations.

Default preparation/help stays cold. Collection needs actual-context privilege,
idle hardware and task-only memory limits. A plan is not hardware qualification.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from aggregate_sm_audit import FIELDS, PERIOD_US, reject_constant, require, unique_object
from artifact_paths import PROJECT_ROOT, create_attempt, resolve_artifact_root
import bert_device_observer as observer
from run_bert_candidate_validation import BUILD, MODES, SUCCESS, normal_cleanup
from run_rnn_v8_validation import digest, save

REFERENCE = PROJECT_ROOT / 'artifacts/runs/bert-current-candidate/attempt-0yofu28n'
LEGACY = PROJECT_ROOT / 'build/tally-full-implementation-1'
POINT = observer.COMPUTE / 'resource-parity-diagnostics/attempt-7t7ho455/point.json'
CATALOGUE = observer.COMPUTE / 'dcgm-catalogues/catalogue-attempt-nfk8r3jt/report.json'
FIXED_POINT = dict(period_ms=7, requests=4286, slo_ms=12, burst_size=1, repetitions=1, warmup=1)
PRIVILEGED_COMMAND = ['/usr/bin/sudo', '-n', '/usr/bin/timeout', '--signal=TERM', '--kill-after=5s', '710s',
    '/usr/bin/env', '-i', 'PATH=/usr/bin:/bin', 'LD_LIBRARY_PATH=' + str(observer.LIBRARIES),
    'CUDA_VISIBLE_DEVICES=' + observer.GPU_UUID, 'PYTHONDONTWRITEBYTECODE=1',
    '/usr/bin/python3', '-B', str(Path(__file__).resolve()), '--sampler-folder', '{observer_folder}']
RECIPE = dict(modes=list(observer.MODES), workers=320, gemm_tile_side=32, priority_idle_us=2000,
              memory_max=16 * 1024 ** 3, swap_max=0, unit_runtime_s=800, unit_stop_s=5,
              outer_timeout_s=820, sampler_work_s=700, sampler_timeout_s=710, sampler_kill_s=5,
              sampler_arm_s=10, sampler_ready_s=20, journal_max=observer.JOURNAL_MAX,
              aggregate_output_max=6 * 1024 ** 3, minimum_free_bytes=8 * 1024 ** 3,
              file_max=1024 ** 3, tasks_max=256, sampled_idle_s=60, idle_wait_s=600,
              fields=list(FIELDS), period_us=PERIOD_US, retries=0, privileged_command=PRIVILEGED_COMMAND,
              worker_preflight_max_age_s=30, full_input_hash_scope='outer-before-and-after-each-unit')


def read(path):
    require(path.stat().st_size <= 8 * 1024 ** 2, 'bounded preparation JSON required')
    return json.loads(path.read_text(), object_pairs_hook=unique_object, parse_constant=reject_constant)


def production_input(path):
    """Old test/doc edits do not invalidate unchanged qualified runtime bytes."""
    return (any(path.is_relative_to(PROJECT_ROOT / prefix) for prefix in
                ('src', 'include', 'examples/framework_workloads', '3rdparty'))
            or path.is_relative_to(BUILD) or not path.is_relative_to(PROJECT_ROOT)
            or path.suffix in ('.pt', '.onnx')
            or path.is_relative_to(PROJECT_ROOT / 'build/bert-intake-assets-3')
            or path.is_relative_to(PROJECT_ROOT / 'build/bert-export-1'))


def qualify_checkpoint(reference):
    """Bind existing qualified receipts; this is not a new tensor comparison."""
    report = read(reference / 'report.json')
    require(report.get('status') == 'current_candidate_observed_controls_require_review'
            and report.get('input_identity_unchanged') is True
            and [item['mode'] for item in report['jobs']] == list(MODES),
            'complete current-candidate numerical/physical checkpoint required')
    old = read(reference / 'inputs-before.json')
    require(old == read(reference / 'inputs-after.json'), 'checkpoint input seals differ')
    paths = {reference / name for name in ('report.json', 'inputs-before.json', 'inputs-after.json', 'manifest.json')}
    for item in report['jobs']:
        mode, unit, worker = item['mode'], item['unit'], item['worker']
        require(unit['unit_removed'] is True and type(unit['returncode']) is int and unit['returncode'] == 0
                and worker['status'] == 'passed_requires_review' and worker['scoped_publication_valid'] is True,
                'normal bounded checkpoint control required')
        reparse = worker['reparse']
        require(reparse == dict(tensors=401, elements=217787909, full_tensor_recomparison=True,
                               source_bound_complete_inference=True, first_command_handoff=mode == 'priority'),
                'unchanged full-tensor/complete-inference/first-handoff checkpoint required')
        require(read(reference / (mode + '-worker.json')) == worker
                and read(reference / (mode + '-unit.json')) == unit
                and read(reference / mode / 'attempt.json') == worker['job'],
                'raw checkpoint receipts differ from reviewed envelope')
        normal_cleanup(worker['job'])
        require(worker['job']['status'] == SUCCESS, 'failed raw checkpoint')
        paths.update((reference / (mode + '-worker.json'), reference / (mode + '-unit.json'),
                      reference / mode / 'attempt.json'))
    required = {BUILD / 'bin/glimmer_execution_service', BUILD / 'lib/libglimmer_execution_interceptor.so',
                BUILD / 'src/backend/cuda/cuda_sgemm.ptx', BUILD / 'src/backend/cuda/cuda_sgemm_tile32.ptx'}
    selected = {Path(name): value for name, value in old.items() if production_input(Path(name))}
    require(required <= selected.keys(), 'checkpoint omits current binaries/PTX')
    for path, value in selected.items():
        require(str(path.resolve(strict=True)) == value['target'] and path.stat().st_size == value['bytes']
                and digest(path) == value['sha256'], 'qualified runtime/input changed: ' + str(path))
    return report, selected, paths


def helper_variant(reference, specification):
    """Bind the separately reviewed native/equal candidate; never widen the old qualifier."""
    spec = read(specification)
    batch_path, application_path = (Path(spec[key]['path']) for key in ('batch_plan', 'application_plan'))
    batch, application = read(batch_path), read(application_path)
    scope = read(reference / 'numerical-scope.json')
    depth = spec['status'] == 'frozen_issue_depth_hardware_application'
    events = spec.get('issue_events', False)
    require(type(events) is bool and application.get('issue_events', False) is events
            and batch.get('issue_events', False) is events and (not events or depth),
            'exact owner-event hardware selection required')
    generations = spec.get('observation_generations', False)
    require(type(generations) is bool and application.get('observation_generations', False) is generations
            and batch.get('observation_generations', False) is generations and (not generations or events),
            'exact observation-generation hardware selection required')
    selections = spec.get('selected_commands', False)
    require(type(selections) is bool and application.get('selected_commands', False) is selections
            and batch.get('selected_commands', False) is selections and (not selections or generations)
            and all(item.get('preparation_boundary', False) is selections
                    for item in (spec, application, batch)), 'exact selected-command hardware selection required')
    if selections:
        require(all(item.get('application_purpose') == 'selected_command_observations'
                    for item in (spec, application, batch)), 'explicit selected-command purpose required')
    require(spec['status'] == ('frozen_issue_depth_hardware_application' if depth else 'frozen_helper_hardware_application')
            and spec['reference'] == str(reference) and type(spec['helper_plans']) is bool
            and batch['status'] == ('frozen_issue_depth_hardware_batch' if depth else 'frozen_helper_hardware_batch')
            and batch['order'] == (['native', 'equal'] if depth else ['off/native', 'off/equal', 'on/native', 'on/equal'])
            and application['status'] == 'frozen_helper_planning_application'
            and application['helper_plans'] is spec['helper_plans']
            and application['modes'] == ['native', 'equal']
            and application['build'] == batch['build'] == str(PROJECT_ROOT / 'build/candidate')
            and application['runtime_environment'] == {'GLIMMER_EXECUTION_HOST_RANGE_SEEK': '1'},
            'exact reviewed helper hardware selection required')
    require(digest(batch_path) == spec['batch_plan']['sha256']
            and digest(application_path) == spec['application_plan']['sha256']
            and scope['plan'] == spec['application_plan']
            and all(application[key] == batch[key] == scope[key] for key in ('server_sha256', 'frontend_sha256'))
            and scope['server'] == str(Path(batch['build']) / 'bin/glimmer_execution_service')
            and scope['frontend'] == str(Path(batch['build']) / 'lib/libglimmer_execution_interceptor.so'),
            'numerical scope differs from hardware candidate')
    require(all(batch[key] is False for key in ('full_device_qualified', 'performance_qualified',
            'saturation_qualified', 'gpu_gap_causality_qualified', 'priority_qualified')),
            'hardware plan is diagnostic only')
    extra = {}
    if depth:
        require(spec['helper_plans'] is True and application.get('issue_depth') is True,
                'depth hardware requires observer-on numerical qualification')
        closure_path = Path(spec['numerical_closure']['path'])
        closure = read(closure_path)
        require(digest(closure_path) == spec['numerical_closure']['sha256']
                and closure['status'] == ('fresh_numerical_and_selected_commands_independently_reviewed' if selections
                                         else 'fresh_numerical_and_generations_independently_reviewed' if generations
                                         else 'fresh_numerical_and_owner_events_independently_reviewed'
                                         if events else 'fresh_numerical_and_depth_independently_reviewed')
                and closure['reference'] == str(reference)
                and closure['service_sha256'] == batch['server_sha256']
                and closure['frontend_sha256'] == batch['frontend_sha256'], 'reviewed depth checkpoint required')
        proofs = {str(closure_path): spec['numerical_closure']['sha256']}
        proof_names = [('audit.py', 'audit_source_sha256'), ('audit.json', 'audit_sha256')]
        if not events:
            proof_names.append(('audit-initial-error.json', 'initial_offline_error_sha256'))
        for name, key in proof_names:
            path = closure_path.parent / name
            require(digest(path) == closure[key], 'numerical audit proof changed')
            proofs[str(path)] = closure[key]
        extra = dict(issue_depth=True, numerical_closure=spec['numerical_closure'], numerical_proofs=proofs)
        if events:
            extra['issue_events'] = True
        if generations:
            extra['observation_generations'] = True
        if selections:
            analyzer = batch_path.with_name('analyze.py')
            require(batch.get('analyzer') == str(analyzer)
                    and digest(analyzer) == batch['analyzer_sha256'], 'selected-command analyzer changed')
            extra.update(selected_commands=True, preparation_boundary=True,
                         application_purpose='selected_command_observations',
                         analysis_source=dict(path=str(analyzer), sha256=batch['analyzer_sha256']))
    else:
        require(not application.get('issue_depth', False), 'old helper batch cannot enable depth')
    return dict(specification=dict(path=str(specification), sha256=digest(specification)),
                batch_plan=spec['batch_plan'], application_plan=spec['application_plan'],
                build=batch['build'], helper_plans=spec['helper_plans'],
                runtime_environment=application['runtime_environment'], deadline_utc=batch['deadline_utc'],
                server_sha256=batch['server_sha256'], frontend_sha256=batch['frontend_sha256'], **extra)


def helper_runtime_paths(variant):
    return {Path(variant['build']) / name for name in ('bin/glimmer_execution_service',
        'lib/libglimmer_execution_interceptor.so', 'CMakeCache.txt', 'compile_commands.json')}


def validate_helper_job(job, variant, *, numerical=False):
    """Reject an enabled observer or a missing/duplicate runtime switch in either phase."""
    require(job['mode'] in ('native', 'equal'), 'helper comparison excludes other modes')
    environments = [job['application_environment'], *job.get('application_environments', {}).values()]
    for environment in environments:
        require('LD_AUDIT' not in environment, 'unexpected audit loader')
        keys = {key for key in environment if key.startswith('GLIMMER_EXECUTION_')}
        if job['mode'] == 'native':
            require(not keys and 'LD_PRELOAD' not in environment and 'server_command' not in job,
                    'native helper control is contaminated')
        else:
            allowed = {'GLIMMER_EXECUTION_SOCKET', 'GLIMMER_EXECUTION_SERVER_PID',
                       'GLIMMER_EXECUTION_SERVER_START', 'GLIMMER_EXECUTION_PREPARATION_TIMEOUT_MS',
                       'GLIMMER_EXECUTION_HOST_RANGE_SEEK'}
            require(keys <= allowed and environment.get('GLIMMER_EXECUTION_HOST_RANGE_SEEK') == '1'
                    and environment.get('GLIMMER_EXECUTION_PREPARATION_TIMEOUT_MS') == '240000'
                    and environment.get('LD_PRELOAD') == str(Path(variant['build']) / 'lib/libglimmer_execution_interceptor.so'),
                    'equal runtime or observer environment differs')
    if job['mode'] == 'equal':
        argv = job['server_command']
        require(argv[0] == str(Path(variant['build']) / 'bin/glimmer_execution_service')
                and argv.count('--skip-idle-helper-plans') == int(variant['helper_plans'])
                and '--host-stage-output' not in argv
                and argv.count('--issue-depth-output') == int(variant.get('issue_depth', False))
                and argv.count('--issue-depth-events') == int(variant.get('issue_events', False))
                and argv.count('--issue-depth-selections') == int(variant.get('selected_commands', False)),
                'actual helper service/observer selection differs')
        for flag, value in (('--persistent-workers', '320'), ('--gemm-tile-side', '32')):
            require(argv.count(flag) == 1 and argv[argv.index(flag) + 1] == value, 'helper geometry differs')
        if not numerical:
            require(not any(flag.startswith('--audit-') or flag == '--observe-block-completion' for flag in argv),
                    'hardware run contains per-operation observation')


def reviewed_depth_inputs(reference, variant):
    """Require the exact raw bytes independently reviewed, not only current consistency."""
    audit_path = Path(variant['numerical_closure']['path']).parent / 'audit.json'
    require(digest(audit_path) == variant['numerical_proofs'][str(audit_path)], 'reviewed audit changed')
    audit = read(audit_path)
    expected = ('fresh_numerical_and_selected_command_evidence_requires_independent_review'
                if variant.get('selected_commands', False)
                else 'fresh_numerical_and_generation_evidence_requires_independent_review'
                if variant.get('observation_generations', False)
                else 'fresh_numerical_and_owner_event_evidence_requires_independent_review'
                if variant.get('issue_events', False) else 'fresh_numerical_and_depth_evidence_requires_independent_review')
    require(audit['status'] == expected
            and audit['reference'] == str(reference), 'reviewed numerical scope differs')
    reviewed = audit['inputs']
    required = {reference / name for name in ('report.json', 'inputs-before.json', 'inputs-after.json',
        'numerical-scope.json', 'equal/issue-depth-receipt.json')}
    required.update(reference / name for mode in ('native', 'equal')
                    for name in (mode+'-worker.json', mode+'-unit.json', mode+'/attempt.json'))
    application = Path(variant['application_plan']['path'])
    required.update((application, application.with_name('build-seal.json')))
    require({str(path) for path in required} <= reviewed.keys(), 'reviewed raw proof inventory incomplete')
    for path in required:
        require(digest(path) == reviewed[str(path)], 'raw numerical proof differs from independent review: '+str(path))
    return reviewed


def qualify_helper_checkpoint(reference, variant):
    reviewed = reviewed_depth_inputs(reference, variant) if variant.get('issue_depth', False) else None
    report, old = read(reference / 'report.json'), read(reference / 'inputs-before.json')
    scope = read(reference / 'numerical-scope.json')
    require(report['status'] == 'current_candidate_observed_controls_require_review'
            and report['input_identity_unchanged'] is True and report['qualification_scope'] == scope
            and old == read(reference / 'inputs-after.json')
            and [item['mode'] for item in report['jobs']] == ['native', 'equal'],
            'complete matching native/equal numerical checkpoint required')
    paths = {reference / name for name in ('report.json', 'inputs-before.json', 'inputs-after.json',
                                         'manifest.json', 'numerical-scope.json')}
    paths.update(Path(variant[key]['path']) for key in ('specification', 'batch_plan', 'application_plan'))
    if variant.get('selected_commands', False):
        paths.add(Path(variant['analysis_source']['path']))
    for item in report['jobs']:
        mode, unit, worker = item['mode'], item['unit'], item['worker']
        require(unit['unit_removed'] is True and type(unit['returncode']) is int and unit['returncode'] == 0
                and worker['status'] == 'passed_requires_review' and worker['scoped_publication_valid'] is True
                and worker['reparse'] == dict(tensors=401, elements=217787909, full_tensor_recomparison=True,
                    source_bound_complete_inference=True, first_command_handoff=False),
                'complete normally retired helper numerical control required')
        for name, value in ((mode + '-worker.json', worker), (mode + '-unit.json', unit),
                            (mode + '/attempt.json', worker['job'])):
            require(read(reference / name) == value, 'raw helper numerical receipt differs')
            paths.add(reference / name)
        normal_cleanup(worker['job'])
        validate_helper_job(worker['job'], variant, numerical=True)
        if mode == 'equal' and variant.get('issue_depth', False):
            import run_bert_delivery_boundary as delivery
            path = reference / 'equal/issue-depth.json'
            receipt_path = path.with_name('issue-depth-receipt.json')
            receipt = read(receipt_path)
            actual = delivery.issue_depth_receipt(path, worker['job'],
                                                  expected_events=variant.get('issue_events', False),
                                                  expected_generations=variant.get('observation_generations', False),
                                                  expected_preparation=variant.get('preparation_boundary', False),
                                                  expected_selections=variant.get('selected_commands', False))
            require(receipt == json.loads(json.dumps(actual, allow_nan=False))
                    and reviewed.get(str(path)) == receipt['sha256'], 'qualified depth receipt changed')
            argv = worker['job']['server_command']
            require(argv[argv.index('--issue-depth-output') + 1] == str(path), 'numerical depth path differs')
            paths.update((path, receipt_path, *(Path(name) for name in variant['numerical_proofs'])))
        for name, checksum in worker['job']['artifact_identities'].items():
            path = reference / mode / name
            require(reviewed is None or reviewed.get(str(path)) == checksum,
                    'numerical output identity differs from independent review')
            require(path.resolve(strict=True).is_relative_to(reference) and digest(path) == checksum,
                    'helper numerical output changed')
            paths.add(path)
    # These are collector-only edits for this adapter, not changed execution oracles.
    changed_tools = {PROJECT_ROOT / name for name in ('scripts/run_bert_device_observation.py',
        'scripts/bert_device_capture.py', 'tests/examples/bert_device_capture_test.py',
        'scripts/bert_device_observer.py', 'scripts/aggregate_sm_audit.py',
        'tests/examples/aggregate_sm_audit_test.py', 'tests/examples/bert_device_observer_test.py')}
    selected = {Path(name): value for name, value in old.items() if Path(name) not in changed_tools}
    require(helper_runtime_paths(variant) <= selected.keys(), 'helper checkpoint omits current runtime inputs')
    for path, value in selected.items():
        require(str(path.resolve(strict=True)) == value['target'] and path.stat().st_size == value['bytes']
                and digest(path) == value['sha256'], 'helper qualified input changed: ' + str(path))
    require(digest(Path(variant['build']) / 'bin/glimmer_execution_service') == variant['server_sha256']
            and digest(Path(variant['build']) / 'lib/libglimmer_execution_interceptor.so') == variant['frontend_sha256'],
            'helper candidate bytes changed')
    return report, selected, paths


def observer_paths():
    # The runtime closure uses top-level bindings and DSOs, not the SDK's static
    # stub archive or its unrelated command-line-client common/ package.
    # Derive required files from the retained catalogue, not the live directory:
    # removing both a library and its plan entry must still fail validation.
    installed = {Path(name) for name in read(CATALOGUE)['identities']
                 if (Path(name).parent == observer.LIBRARIES and '.so' in Path(name).name)
                 or (Path(name).parent == observer.BINDINGS and Path(name).suffix == '.py')
                 or Path(name) == observer.TOOLS / 'layout.json'}
    required_tools = {observer.LIBRARIES / name for name in
                      ('libdcgm.so.4.0.0', 'libdcgmmoduleprofiling.so.4.0.0', 'libnvperf_dcgm_host.so')}
    required_tools.update(observer.BINDINGS / name for name in
                          ('dcgm_agent.py', 'dcgm_structs.py', 'dcgm_fields.py', 'dcgmvalue.py'))
    require(required_tools <= installed and all(path.is_file() for path in installed),
            'mandatory pinned DCGM runtime file is missing')
    # Preserve the Python import closure, not the historical launcher behavior.
    helpers = {path for root in (observer.COMPUTE, LEGACY) for path in root.glob('*.py')}
    helpers.update(observer.COMPUTE / name for name in
                    ('dcgm_sampling.py', 'dcgm_catalogue.py', 'collect_application_counters.py'))
    helpers.update(LEGACY / name for name in ('collect_sustained_application.py',
                    'collect_ordinary_application.py', 'collect_application_block_colocation.py'))
    helpers.update((Path(__file__), PROJECT_ROOT / 'scripts/bert_device_observer.py',
                    PROJECT_ROOT / 'scripts/bert_device_capture.py',
                    PROJECT_ROOT / 'scripts/aggregate_sm_audit.py', PROJECT_ROOT / 'scripts/artifact_paths.py',
                    PROJECT_ROOT / 'scripts/run_bert_candidate_validation.py',
                    PROJECT_ROOT / 'scripts/run_rnn_v8_validation.py', PROJECT_ROOT / 'scripts/rnn_v8_audit.py',
                    PROJECT_ROOT / 'scripts/cudnn8_legacy_audit.py'))
    # DCGM can load the system NVML provider lazily; bind the actual object too.
    nvml = Path('/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1')
    require(nvml.is_file(), 'installed NVML provider required for a future observation')
    helpers.update((nvml, nvml.resolve(), Path(sys.executable).resolve()))
    for name in ('/usr/bin/python3', '/usr/bin/timeout', '/usr/bin/env', '/usr/bin/sudo'):
        helpers.update((Path(name), Path(name).resolve(strict=True)))
    return installed, helpers


def observer_seals():
    old = read(CATALOGUE)['identities']
    installed, helpers = observer_paths()
    require(installed, 'installed DCGM runtime is missing')
    for path in installed:
        require(str(path) in old and digest(path) == old[str(path)],
                'installed DCGM differs from the retained pinned catalogue: ' + str(path))
    seals = {str(path): digest(path) for path in sorted(installed | helpers)}
    observer.verify_seals(seals)
    return seals


def required_inputs(reference, variant=None):
    require(reference.is_absolute() and reference.resolve(strict=True) == reference
            and reference.parent == PROJECT_ROOT / 'artifacts/runs/bert-current-candidate'
            and reference.name.startswith('attempt-'), 'canonical current checkpoint location required')
    selected = {name for name in read(reference / 'inputs-before.json') if production_input(Path(name))}
    required = {BUILD / name for name in ('bin/glimmer_execution_service',
                'lib/libglimmer_execution_interceptor.so', 'src/backend/cuda/cuda_sgemm.ptx',
                'src/backend/cuda/cuda_sgemm_tile32.ptx', 'CMakeCache.txt', 'compile_commands.json')}
    required.update((POINT, CATALOGUE,
        PROJECT_ROOT / 'examples/framework_workloads/bert/bert_colocation_worker.py',
        PROJECT_ROOT / 'examples/framework_workloads/bert/run_bert_colocation.py'))
    required.update(reference / name for name in ('report.json', 'inputs-before.json', 'inputs-after.json', 'manifest.json'))
    for mode in ('native', 'equal') if variant is not None else MODES:
        required.update((reference / (mode + '-worker.json'), reference / (mode + '-unit.json'),
                         reference / mode / 'attempt.json'))
    if variant is not None:
        required.update(helper_runtime_paths(variant))
        required.add(reference / 'numerical-scope.json')
        required.update(Path(variant[key]['path']) for key in ('specification', 'batch_plan', 'application_plan'))
        if variant.get('selected_commands', False):
            required.add(Path(variant['analysis_source']['path']))
        if variant.get('issue_depth', False):
            required.update((reference / 'equal/issue-depth.json', reference / 'equal/issue-depth-receipt.json',
                             *(Path(name) for name in variant['numerical_proofs'])))
    return selected | {str(path) for path in required}


def expected_recipe(plan):
    result = dict(RECIPE)
    if 'helper_variant' in plan:
        require('worker_probe' not in plan and 'native_equal' not in plan, 'one helper selection only')
        return dict(result, modes=['native', 'equal'])
    require(not ('worker_probe' in plan and 'native_equal' in plan), 'one explicit diagnostic selection required')
    if 'worker_probe' in plan:
        require(plan['worker_probe'] is True, 'explicit diagnostic worker probe required')
        workers = plan.get('recipe', {}).get('workers')
        require(type(workers) is int and workers in (320, 640, 1280), 'bounded worker probe count required')
        result.update(workers=workers, modes=['equal'])
    if 'native_equal' in plan:
        require(plan['native_equal'] is True, 'explicit native/equal pair required')
        result['modes'] = ['native', 'equal']
    return result


def authorization_budget(plan, seconds):
    if 'helper_variant' in plan:
        deadline = datetime.fromisoformat(plan['helper_variant']['deadline_utc'])
        require((deadline - datetime.now(timezone.utc)).total_seconds() >= seconds,
                'insufficient remaining hardware diagnostic authorization')


def compare_helper_hardware(off, on):
    """Compare one ordered pair using conservative fixed-window counter bounds."""
    summaries = []
    for group in (off, on):
        require([worker['mode'] for worker in group] == ['native', 'equal'], 'exact hardware pair required')
        controls = {}
        for worker in group:
            require(worker['status'] == 'passed_requires_review', 'complete hardware worker required')
            metrics, fields = worker['job']['metrics'], worker['activity']['fields']
            duration = metrics['window_end_ns'] - metrics['window_begin_ns']
            require(duration == 30_002_000_000 and set(fields) == {str(field) for field in FIELDS},
                    'fixed offered-window hardware fields required')
            sm = fields['1002']
            margin = 20_000_000 / duration
            completed = metrics['at_offer_end']['completed_requests']
            require(type(completed) is int and completed > 0 and sm['interior_duration_weighted_ratio'] > 0,
                    'positive offered-window/control denominators required')
            controls[worker['mode']] = dict(sm_interior=sm['interior_duration_weighted_ratio'],
                sm_lower=max(0, sm['whole_window_ratio_lower_bound'] - margin),
                sm_upper=min(1, sm['whole_window_ratio_upper_bound'] + margin),
                clock_margin_ratio=margin, completed_requests=completed,
                training_completions=metrics['training_completions_in_window'],
                coverage_passed=all(row['interior_coverage_fraction'] >= .95 for row in fields.values()),
                fields=fields)
        summaries.append(controls)
    before, after = summaries
    native_off, native_on = before['native'], after['native']
    equal_off, equal_on = before['equal'], after['equal']
    native_sm_drift = abs(native_on['sm_interior'] - native_off['sm_interior']) / native_off['sm_interior']
    gates = dict(counter_coverage=all(row['coverage_passed'] for group in summaries for row in group.values()),
        sm_activity_improvement=equal_on['sm_lower'] >= 1.15 * equal_off['sm_upper'],
        equal_completions=equal_on['completed_requests'] * 100 >= equal_off['completed_requests'] * 95,
        native_sm_drift=native_sm_drift <= .20,
        native_uncertainty_overlap=max(native_off['sm_lower'], native_on['sm_lower'])
                                  <= min(native_off['sm_upper'], native_on['sm_upper']),
        native_completion_drift=abs(native_on['completed_requests'] - native_off['completed_requests']) * 100
                                <= native_off['completed_requests'] * 20)
    return dict(status='helper_hardware_response_requires_review', gates=gates, all_gates_passed=all(gates.values()),
                off=before, on=after, native_sm_relative_drift=native_sm_drift,
                diagnostic_only=True, full_device_qualified=False, saturation_qualified=False,
                performance_qualified=False, gpu_gap_causality_qualified=False)


def validate_structure(plan):
    variant = plan.get('helper_variant')
    if variant is not None:
        require(variant == helper_variant(Path(plan['reference']), Path(variant['specification']['path'])),
                'helper hardware selection changed')
    require(type(plan.get('version')) is int and plan['version'] == 1
            and plan.get('status') == 'prepared_not_capture_ready'
            and plan.get('live_capture_enabled') is False and plan.get('counter_collection') is False
            and plan.get('ordinary_timing') is False,
            'only a prepare-only current plan is accepted')
    require(plan.get('recipe') == expected_recipe(plan) and plan.get('point') == FIXED_POINT
            and plan.get('gpu_uuid') == observer.GPU_UUID
            and plan.get('build') == (variant['build'] if variant is not None else str(BUILD)),
            'frozen candidate/whole-device/arrival recipe differs')
    require(all(type(plan['point'][name]) is int for name in FIXED_POINT)
            and all(type(plan['recipe'][name]) is type(value) for name, value in RECIPE.items()),
            'fixed recipe scalar types differ')
    for key in ('full_device_qualified', 'saturation_qualified', 'performance_qualified', 'full_goal_complete'):
        require(plan.get(key) is False, 'preparation cannot publish hardware qualification')
    require(type(plan.get('input_identities')) is dict and plan['input_identities'], 'input seals required')
    require(type(plan.get('observer_identities')) is dict, 'observer seals required')
    installed, helpers = observer_paths()
    require({str(path) for path in installed | helpers} <= plan['observer_identities'].keys(),
            'mandatory observer/helper/tool identity omitted')
    require(type(plan.get('reference')) is str
            and required_inputs(Path(plan['reference']), variant) <= plan['input_identities'].keys(),
            'mandatory candidate/PTX/workload/checkpoint identity omitted')


def validate_plan(plan):
    validate_structure(plan)
    observer.verify_seals(plan['observer_identities'])
    for name, value in plan['input_identities'].items():
        path = Path(name)
        require(str(path.resolve(strict=True)) == value['target'] and path.stat().st_size == value['bytes']
                and digest(path) == value['sha256'], 'prepared input identity changed: ' + name)


def prepare(reference, worker_probe=None, *, native_equal=False, helper_plan=None):
    require(type(native_equal) is bool and not (native_equal and worker_probe is not None),
            'native/equal pair cannot change the worker policy')
    require(helper_plan is None or not native_equal and worker_probe is None, 'one helper preparation only')
    reference = reference.resolve(strict=True)
    # Fail small tool/helper preflights before reading the multi-GiB model set.
    seals = observer_seals()
    variant = helper_variant(reference, helper_plan.resolve(strict=True)) if helper_plan is not None else None
    checkpoint, qualified, proofs = (qualify_helper_checkpoint(reference, variant) if variant is not None
                                    else qualify_checkpoint(reference))
    point = read(POINT)
    require(all(type(point.get(name)) is int and point[name] == value for name, value in FIXED_POINT.items()),
            'previously frozen sustained point changed')
    paths = proofs | {POINT, CATALOGUE}
    paths.update((PROJECT_ROOT / 'examples/framework_workloads/bert/bert_colocation_worker.py',
                  PROJECT_ROOT / 'examples/framework_workloads/bert/run_bert_colocation.py'))
    # Qualified entries have just been checked byte-for-byte. Reuse those
    # records, then revalidate the complete union once before publication.
    inputs = {str(path): dict(value) for path, value in qualified.items()}
    inputs.update({str(path): dict(target=str(path.resolve(strict=True)), bytes=path.stat().st_size, sha256=digest(path))
                   for path in sorted(paths)})
    plan = dict(version=1, status='prepared_not_capture_ready', live_capture_enabled=False,
                counter_collection=False, ordinary_timing=False, full_device_qualified=False,
                saturation_qualified=False, performance_qualified=False, full_goal_complete=False,
                build=str(BUILD), gpu_uuid=observer.GPU_UUID, reference=str(reference),
                checkpoint_gpu=checkpoint['gpu_identity_after'], point=dict(FIXED_POINT), recipe=dict(RECIPE),
                observer_identities=seals, input_identities=inputs,
                integration_remaining=['capture-readiness direction review',
                    'actual-context noninteractive authentication and idle hardware'],
                hardware_gates_remaining=['whole-path80-SM participation', 'actual concurrent activity',
                                           'compute and memory throughput', 'separate ordinary timing and CPU cost'])
    if worker_probe is not None:
        plan.update(worker_probe=True)
        plan['recipe'] = {**RECIPE, 'workers': worker_probe, 'modes': ['equal']}
        plan['integration_remaining'].append('worker-count diagnostic only; no new full-tensor qualification')
    if native_equal:
        plan.update(native_equal=True, recipe={**RECIPE, 'modes': ['native', 'equal']})
    if variant is not None:
        plan.update(helper_variant=variant, build=variant['build'], recipe={**RECIPE, 'modes': ['native', 'equal']})
    validate_plan(plan)
    attempt = create_attempt(resolve_artifact_root('bert-current-device', category='profiles'))
    save(attempt / 'plan.json', plan)
    return attempt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=REFERENCE, help='existing current numerical/physical checkpoint')
    parser.add_argument('--check-plan', type=Path, help='revalidate an existing preparation without launching anything')
    parser.add_argument('--collect', type=Path, help='explicit bounded counter batch; never ordinary timing')
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--worker-probe', type=int, choices=(320, 640, 1280),
                           help='prepare one equal-priority worker-count diagnostic; unchanged workload')
    selection.add_argument('--native-equal', action='store_true',
                           help='prepare only native/equal observations with unchanged workload and worker policy')
    selection.add_argument('--helper-plan', type=Path,
                           help='prepare the explicitly reviewed current helper-candidate hardware variant')
    parser.add_argument('--worker-attempt', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--worker-mode', choices=observer.MODES, help=argparse.SUPPRESS)
    parser.add_argument('--sampler-folder', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        require(sum(value is not None for value in
                    (args.check_plan, args.collect, args.worker_attempt, args.sampler_folder)) <= 1,
                'one explicit operation only')
        require((args.worker_probe is None and not args.native_equal and args.helper_plan is None) or
                not any((args.check_plan, args.collect, args.worker_attempt, args.sampler_folder)),
                'diagnostic selection applies to preparation only; collect its frozen plan separately')
        if args.collect or args.worker_attempt or args.sampler_folder:
            import bert_device_capture as capture
            if args.collect:
                return capture.collect(args.collect.resolve(strict=True))
            if args.sampler_folder:
                return capture.sampler_entry(args.sampler_folder)
            require(args.worker_mode is not None, 'explicit internal worker mode required')
            return capture.worker(args.worker_attempt.resolve(strict=True), args.worker_mode)
        if args.check_plan:
            validate_plan(read(args.check_plan))
            print('prepared_not_capture_ready: seals unchanged; no experiment launched')
        else:
            print('Preparation only: ' + str(prepare(args.reference, args.worker_probe,
                                                     native_equal=args.native_equal, helper_plan=args.helper_plan) / 'plan.json'), flush=True)
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f'Preparation failed: {type(error).__name__}: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
