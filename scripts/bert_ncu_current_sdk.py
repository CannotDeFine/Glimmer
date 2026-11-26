"""Bind a current SDK counter experiment to completed BERT numerical evidence."""

from copy import deepcopy
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import subprocess

import bert_ncu_replay_exec as bridge
from bert_sgemm_coverage_candidates import authenticated_training_sessions
from execution_semantic_audit import parse_semantic_audit
import run_bert_candidate_validation as candidate
import run_bert_host_pair_numerical as pair
from run_rnn_v8_validation import digest

require = pair.require
ORACLE = pair.ORACLE
VARIANTS = ('scalar', 'sharded')
FLAGS = dict(diagnostic_only=True, full_device_qualified=False, performance_qualified=False,
             saturation_qualified=False, priority_qualified=False, hardware_counters=False)
LAUNCH_ALLOWANCE_SECONDS = 840
FRESH_CONTRACT_FIELDS = ('schema', 'variant', 'target', 'max_seconds', 'deadline_utc', 'runtime',
                         'service_environment', 'numerical', 'reference')
RUN_TIMING_FIELDS = frozenset(('measured_begin_ns', 'measured_end_ns', 'samples_ns', 'timing'))


def read(path):
    return json.loads(Path(path).read_text())


def native_result_projection(root, native):
    """Authenticate result records and omit only independently measured timing fields."""
    result = {}
    for name in ('training/result.json', 'inference/result.json'):
        path = root / 'native' / name
        require(path == path.resolve(strict=True) and path.is_file() and
                digest(path) == native['artifact_identities'][name], 'native result record changed')
        record = read(path)
        require(type(record) is dict and RUN_TIMING_FIELDS <= record.keys(), 'native result timing fields missing')
        result[name] = {key: value for key, value in record.items() if key not in RUN_TIMING_FIELDS}
    return result


def validate_fresh_applications(root, manifest, native, equal):
    """Check the fixed same-run BERT commands and uninstrumented environments."""
    project = bridge.PROJECT
    python = str(project / 'examples/framework_workloads/pytorch_smoke/.venv/bin/python')
    assets = str(project / 'build/bert-intake-assets-3')
    workload = project / 'examples/framework_workloads/bert'
    startup = Path(manifest['startup_reference_bundle'])
    for mode, job in (('native', native), ('equal', equal)):
        require(set(job['application_commands']) == set(job['application_environments']) == {'low', 'high'},
                'two fixed BERT roles required')
        directory = root / mode
        reference = startup / 'training' if mode == 'native' else root / 'native/training'
        low = [python, '-B', str(workload / 'bert_workload.py'), '--assets', assets,
               '--output', str(directory / 'training'), '--role', 'training', '--warmup', '1',
               '--iterations', '2', '--snapshot', '--reference', str(reference),
               '--ready-file', str(directory / 'low.ready.json'), '--start-file', str(directory / 'low.start')]
        high = [python, '-B', str(workload / 'onnx_bert_workload.py'), '--assets', assets,
                '--export', str(startup / 'export'), '--output', str(directory / 'inference'),
                '--warmup', '1', '--iterations', '2', '--ready-file', str(directory / 'high.ready.json'),
                '--start-file', str(directory / 'high.start')]
        require(job['application_commands'] == dict(low=low, high=high),
                'fixed BERT command or same-run native reference differs')
        environments = (job['application_environment'], *job['application_environments'].values())
        allowed = {'GLIMMER_EXECUTION_SOCKET', 'GLIMMER_EXECUTION_SERVER_PID',
                   'GLIMMER_EXECUTION_SERVER_START', 'GLIMMER_EXECUTION_PREPARATION_TIMEOUT_MS',
                   'GLIMMER_EXECUTION_HOST_RANGE_SEEK'}
        extras = set() if mode == 'native' else allowed | {'LD_PRELOAD'}
        require(all(set(env) == pair.BASE_ENVIRONMENT_KEYS | extras for env in environments),
                'unexpected numerical application environment')
        require(all(all(env[key] == '2' for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'))
                    and env['CUDA_VISIBLE_DEVICES'] == manifest['gpu'].split(',')[0].strip()
                    for env in environments), 'numerical application CPU or GPU differs')
        if mode == 'equal':
            expected = dict(LD_PRELOAD=equal['application_environment']['LD_PRELOAD'],
                            GLIMMER_EXECUTION_SOCKET=equal['socket_path'],
                            GLIMMER_EXECUTION_SERVER_PID=str(equal['server_identity']['pid']),
                            GLIMMER_EXECUTION_SERVER_START=str(equal['server_identity']['start']),
                            GLIMMER_EXECUTION_PREPARATION_TIMEOUT_MS='240000', GLIMMER_EXECUTION_HOST_RANGE_SEEK='0')
            require(set(equal['server_environment']) == pair.BASE_ENVIRONMENT_KEYS and
                    all(env == dict(equal['server_environment'], **expected) for env in environments),
                    'numerical service or application environment differs')
        else:
            require(all(env == environments[0] for env in environments), 'native role environments differ')
    require(all(native['application_environment'][key] == equal['server_environment'][key]
                for key in pair.BASE_ENVIRONMENT_KEYS), 'native and replacement base environments differ')
    artifacts = ('training/result.json', 'training/snapshot.pt', 'inference/result.json')
    references = equal['reference_before']
    require(references == equal['reference_after'] and
            set(references) == {str(root / 'native' / name) for name in artifacts} and
            all(references[str(root / 'native' / name)]['target'] == str(root / 'native' / name) and
                references[str(root / 'native' / name)]['sha256'] == native['artifact_identities'][name]
                for name in artifacts), 'same-run native training or inference reference differs')


def _require_caps0(uid, gid):
    require(os.getresuid() == (uid, uid, uid) and os.getresgid() == (gid, gid, gid) and uid > 0,
            'fresh checker must remain unprivileged')
    status = Path('/proc/self/status').read_text()
    capabilities = {name: value.strip() for name, separator, value in
                    (line.partition(':') for line in status.splitlines()) if separator and
                    name in ('CapInh', 'CapPrm', 'CapEff', 'CapAmb')}
    require(len(capabilities) == 4 and all(int(value, 16) == 0 for value in capabilities.values()),
            'fresh checker cannot retain capabilities')


def fresh_numerical_evidence(root, expected_seals, variant, runtime, provenance, uid, gid):
    """Validate one independently reviewed fresh native/equal BERT batch.

    The caller must authenticate the supplied seals through its reviewed plan.
    This checks numerical and provider prerequisites, not descriptor identity,
    SM participation, priority or performance.
    """
    _require_caps0(uid, gid)
    root = Path(root)
    require(root.is_absolute() and root == root.resolve(strict=True) and
            root.is_relative_to(bridge.PROJECT / 'artifacts/maintenance') and
            variant in VARIANTS and root.is_dir(), 'canonical fresh numerical variant and directory required')
    names = ('report.json', 'manifest.json', 'batch-publication.json', 'native-worker.json',
             'equal-worker.json', 'review.md', 'native/attempt.json', 'equal/attempt.json', 'equal/server.log',
             'native-unit.log', 'equal-unit.log')
    require(type(expected_seals) is dict and set(expected_seals) == set(names),
            'exact reviewed fresh numerical evidence required')
    for name in names:
        path = root / name
        require(path == path.resolve(strict=True) and path.is_file() and
                digest(path) == expected_seals[name], 'fresh numerical evidence changed: ' + name)
    report, manifest, publication = (read(root / name) for name in names[:3])
    require(report['status'] == 'completed_requires_review' and report['diagnostic_only'] is True and
            all(report[key] is False for key in ('hardware_counters', 'full_device_qualified',
                'performance_qualified', 'priority_qualified', 'saturation_qualified', 'application_qualified')) and
            report['inputs'] == manifest['inputs'] and report['gpu_identity'] == manifest['gpu'] and
            report['resources_before'] == report['resources_after'] and report['final_gpu_processes'] == [] and
            report['manifest_sha256'] == publication['manifest_sha256'] == expected_seals['manifest.json'] and
            publication['report_sha256'] == expected_seals['report.json'] and
            report['batch_publication_path'] == str(root / 'batch-publication.json'),
            'fresh numerical publication or diagnostic scope differs')
    claim = Path(report['claim'])
    require(claim.parent == root.parent and claim == claim.resolve(strict=True) and claim.is_file() and
            digest(claim) == publication['claim_sha256'], 'fresh numerical claim differs')
    bridge.exact_fields(provenance, ('plan', 'launch', 'runner', 'auditor'))
    for entry in provenance.values():
        bridge.exact_fields(entry, ('path', 'sha256'))
        path = Path(entry['path'])
        require(path.parent == root.parent.parent and path == path.resolve(strict=True) and path.is_file() and
                digest(path) == entry['sha256'] == manifest['inputs'][str(path)]['sha256'],
                'fresh numerical provenance changed')
    launch, claim_record = read(provenance['launch']['path']), read(claim)
    require(manifest['plan_sha256'] == provenance['plan']['sha256'] == launch['plan_sha256'] and
            manifest['launch_sha256'] == provenance['launch']['sha256'] and
            provenance['runner']['sha256'] == launch['runner_sha256'] and
            claim_record == dict(attempt=str(root), plan_sha256=manifest['plan_sha256'],
                                 launch_sha256=manifest['launch_sha256'], order=['native', 'equal']),
            'fresh numerical plan, runner or claim differs')
    workers, active_samples = {}, {}
    bridge.exact_fields(publication, ('report_sha256', 'manifest_sha256', 'claim_sha256', 'workers'))
    require(type(report['jobs']) is list and len(report['jobs']) == 2 and
            [item['mode'] for item in report['jobs']] == ['native', 'equal'] and
            set(publication['workers']) == {'native', 'equal'}, 'fresh numerical job inventory differs')
    for row, mode in zip(report['jobs'], ('native', 'equal')):
        worker_path = root / (mode + '-worker.json')
        worker = read(worker_path)
        unit_log = (root / (mode + '-unit.log')).read_text()
        active, terminal = row['active_properties'], row['terminal_properties']
        observed_active = active['ActiveState'] == 'active' and active['MainPID'].isdecimal() and int(active['MainPID']) > 0
        observed_before_start = active['ActiveState'] == 'inactive' and active['MainPID'] == '0'
        require(active['LoadState'] == 'loaded' and (observed_active or observed_before_start) and
                active['Description'] == 'Glimmer fresh numerical ' + root.name + ' ' + mode and
                row['unit'] == 'glimmer-bert-' + root.name + '-' + mode + '.service' and
                active['RuntimeMaxUSec'] in ('13min 20s', '800s') and active['TimeoutStopUSec'] == '5s' and
                active['MemoryMax'] == str(16 * 1024 ** 3) and active['MemorySwapMax'] == '0' and
                active['TasksMax'] == '256' and active['LimitFSIZE'] == str(1024 ** 3) and
                active['NoNewPrivileges'] == 'yes' and active['KillMode'] == 'control-group' and
                '--property=OOMPolicy=kill' in row['command'] and terminal['ActiveState'] == 'inactive' and
                terminal['MainPID'] == '0', 'numerical unit resource or terminal state differs')
        active_samples[mode] = observed_active
        require(row['status'] == worker['status'] == 'passed_requires_review' and
                row['result'] == str(worker_path) and row['terminal_properties']['LoadState'] == 'not-found' and
                row['command'].count('--worker') == 1 and
                row['command'][row['command'].index('--worker') - 1] == provenance['runner']['path'] and
                unit_log.count('Finished with result: success\n') == 1 and
                unit_log.count('Main processes terminated with: code=exited/status=0\n') == 1 and
                publication['workers'][mode] == expected_seals[mode + '-worker.json'] and
                worker['mode'] == mode and worker['scoped_publication_valid'] is False and
                worker['reparse'] == ORACLE and row['reparse'] == ORACLE and
                all(type(worker['reparse'][key]) is type(value) for key, value in ORACLE.items()) and
                read(root / mode / 'attempt.json') == worker['job'], 'fresh numerical worker differs')
        resources = worker['resources']
        require(type(resources['resource_checks']) is int and resources['resource_checks'] > 0 and
                resources['observer']['cgroup'].endswith('/' + row['unit']),
                'numerical resource observer or task unit differs')
        participants = resources['participants']
        job = worker['job']
        identities = list(job['application_identities'].values())
        if mode == 'equal':
            identities.append(job['server_identity'])
        require(type(participants) is list and len(participants) == len(identities) and
                {(item['pid'], item['start']) for item in participants} ==
                {(item['pid'], item['start']) for item in identities} and
                all(item['cgroup'] == resources['observer']['cgroup'] for item in participants),
                'numerical participants did not share the observed task cgroup')
        for phase in ('initial_memory', 'last_memory'):
            memory = resources[phase]
            events = dict(line.split() for line in memory['memory.events'].splitlines())
            require(memory['memory.max'] == str(16 * 1024 ** 3) and memory['memory.swap.max'] == '0' and
                    memory['memory.oom.group'] == '1' and events and
                    all(int(value) == 0 for value in events.values()), 'numerical memory limit or event differs')
        task = resources['last_task_and_swap']
        require(task['pids.max'] == '256' and task['memory.swap.current'] == '0' and
                dict(line.split() for line in task['pids.events'].splitlines()) == {'max': '0'},
                'numerical task or swap limit reached')
        candidate.normal_cleanup(worker['job'])
        workers[mode] = worker
    native, equal = workers['native']['job'], workers['equal']['job']
    validate_fresh_applications(root, manifest, native, equal)
    require(all('LD_PRELOAD' not in env and not any(key.startswith('GLIMMER_EXECUTION_') for key in env)
                for env in (native['application_environment'], *native['application_environments'].values())),
            'native numerical environment was intercepted')
    command = equal['server_command']
    binding = dict(schema='fresh_runtime_v2', variant=variant)
    expected = bridge.service_command(dict(clients=equal['application_identities'], socket=equal['socket_path']),
                                      service=Path(runtime['service']['path']), signal_binding=binding)
    for option in ('--audit-sgemm-target-client', '--audit-sgemm-range-library'):
        index = expected.index(option)
        del expected[index:index + 2]
    require(command == expected, 'numerical service recipe differs')
    require(equal['service_configuration'] == dict(scheduler='priority', kernel_mode='preemptive',
                configured_workers=320, nccl_enabled=False, sharded_gemm_signal=variant == 'sharded',
                full_device_qualified=False), 'numerical service configuration receipt differs')
    require(set(equal['application_environments']) == {'low', 'high'} and
            set(equal['application_identities']) == {'low', 'high'}, 'two numerical applications required')
    environments = (equal['application_environment'], *equal['application_environments'].values())
    require(all(env.get('LD_PRELOAD') == runtime['frontend']['path'] and
                env.get('GLIMMER_EXECUTION_HOST_RANGE_SEEK') == '0' for env in environments),
            'numerical frontend environment differs')
    maps = workers['equal']['loaded_mappings']
    identities = equal['application_identities']
    for item in maps.values():
        bridge.exact_fields(item, ('pid', 'start', 'path', 'device', 'inode', 'maps', 'seek'))
    require(set(maps) == {str(identities[role]['pid']) for role in ('low', 'high')} and
            all(maps[str(item['pid'])]['pid'] == item['pid'] and maps[str(item['pid'])]['start'] == item['start'] and
                maps[str(item['pid'])]['path'] == runtime['frontend']['path'] and
                maps[str(item['pid'])]['device'] == runtime['frontend']['device'] and
                maps[str(item['pid'])]['inode'] == runtime['frontend']['inode'] and
                maps[str(item['pid'])]['maps'] == 5 and maps[str(item['pid'])]['seek'] == '0'
                for item in identities.values()), 'numerical frontend mappings differ')
    provider = workers['equal']['provider']
    require(provider['configured_sessions'] == [2] and provider['completed_sgemm'] == 438 and
            provider['unique_keys'] == 438 and provider['workers'] == 320 and
            provider['admission_status'] == {'true': 438}, 'complete numerical SGEMM provider evidence required')
    require((provider.get('single_physical_launch_per_sgemm') is True if variant == 'scalar'
             else 'single_physical_launch_per_sgemm' not in provider),
            'numerical physical-launch receipt differs')
    log = root / 'equal/server.log'
    require(expected_seals['equal/server.log'] == equal['artifact_identities']['server.log'],
            'numerical service audit differs')
    content = log.read_text()
    config = ('Execution service configuration: scheduler=priority, kernel_mode=preemptive, '
              'configured_workers=320, nccl_enabled=false, sharded_gemm_signal=' +
              ('true' if variant == 'sharded' else 'false') + '; full_device_qualified=false')
    require(content.count('Execution service configuration:') == 1 and content.count(config) == 1,
            'numerical signal configuration differs')
    semantic = parse_semantic_audit(content, require_complete=True)
    sessions = authenticated_training_sessions(semantic, equal)
    require(sessions == {2}, 'target training session differs')
    audit_path = root.parent.parent / 'official_bert_provider_audit.py'
    require(audit_path == audit_path.resolve(strict=True) and
            str(audit_path) == provenance['auditor']['path'] and
            digest(audit_path) == provenance['auditor']['sha256'] == manifest['inputs'][str(audit_path)]['sha256'],
            'numerical provider audit source changed')
    spec = importlib.util.spec_from_file_location('fresh_bert_provider_audit', audit_path)
    auditor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auditor)
    audited = auditor.audit(content, sessions)
    require(all(provider[key] == value for key, value in audited.items()), 'numerical provider audit differs')
    if variant == 'scalar':
        keys = {(int(row[1]), int(row[2])) for row in re.finditer(
            r'Native admission audit: session=(\d+), sequence=(\d+), accepted=true, operation=30(?:\s|$)',
            content) if int(row[1]) in sessions}
        launches = [row for row in re.finditer(
            r'Native persistent audit: session=(\d+), sequence=(\d+), total_blocks=(\d+), '
            r'physical_launches=(\d+), workers=(\d+)(?:\s|$)', content)
            if (int(row[1]), int(row[2])) in keys]
        require(len(keys) == len(launches) == 438 and
                all(int(row[4]) == 1 and int(row[5]) == 320 for row in launches),
                'scalar numerical SGEMM launch count differs')
    patterns = (
        r'Native admission audit: session=2, sequence=3127, accepted=true, operation=30(?:\s|$)',
        r'Native persistent audit: session=2, sequence=3127, total_blocks=2304, '
        r'physical_launches=1, workers=320(?:\s|$)',
        r'Native issue audit: session=2, sequence=3127, status=0(?:\s|$)',
        r'Native execution audit: session=2, issued=3127, retained=0, retired=1, status=0(?:\s|$)',
        r'Priority execution audit: event=low_logical_done, session=2, priority=0, sequence=3127, '
        r'completed_blocks=2304, total_blocks=2304, physical_launches=1, status=0(?:\s|$)',
    )
    matches = [list(re.finditer(pattern, content)) for pattern in patterns]
    require(all(len(found) == 1 for found in matches) and
            [found[0].start() for found in matches] == sorted(found[0].start() for found in matches),
            'fresh numerical target did not complete normally')
    bridge.validate_numerical_runtime(runtime, manifest['inputs'], uid, gid)
    return dict(variant=variant, native_job=native, manifest=manifest, runtime=runtime,
                service_environment=dict(equal['server_environment']),
                native_result_projection=native_result_projection(root, native),
                provenance=provenance, numerical_only=True, active_samples=active_samples)


def match_fresh_numerical_pair(scalar, sharded):
    """Compare the two checked numerical batches without claiming counter coverage."""
    require(scalar['variant'] == 'scalar' and sharded['variant'] == 'sharded' and
            scalar['numerical_only'] is True and sharded['numerical_only'] is True and
            scalar['runtime'] == sharded['runtime'] and
            scalar['service_environment'] == sharded['service_environment'],
            'fresh numerical runtime or service environment differs')
    first, second = scalar['manifest'], sharded['manifest']
    require(all(first[key] == second[key] for key in
                ('gpu', 'startup_reference_bundle', 'startup_reference_report_sha256', 'dynamic_libraries')),
            'fresh numerical GPU, reference or dynamic closure differs')
    require(scalar['native_job']['artifact_identities']['training/snapshot.pt'] ==
            sharded['native_job']['artifact_identities']['training/snapshot.pt'],
            'fresh numerical native training snapshots differ')
    require(scalar['native_result_projection'] == sharded['native_result_projection'],
            'fresh numerical native result semantics differ')
    exempt = {record['path'] for evidence in (scalar, sharded)
              for name, record in evidence['provenance'].items() if name in ('plan', 'launch', 'runner')}
    first_inputs = {name: value for name, value in first['inputs'].items() if name not in exempt}
    second_inputs = {name: value for name, value in second['inputs'].items() if name not in exempt}
    require(first_inputs == second_inputs, 'fresh numerical non-run-specific inputs differ')
    return dict(numerical_only=True, same_runtime_and_inputs=True,
                service_environment=dict(scalar['service_environment']), runtime=deepcopy(scalar['runtime']),
                full_device_qualified=False,
                performance_qualified=False, priority_qualified=False, saturation_qualified=False)


def fresh_application_inventory(common, h, candidates, manifests, environment):
    """Select recorded application inputs and check each binary's loader closure.

    This is an inventory of inputs recorded by the numerical runs, not a claim
    that those runs recorded every Python module subsequently imported.
    """
    project = bridge.PROJECT
    base = project / 'build/tally-full-implementation-1'
    workload = project / 'examples/framework_workloads/bert'
    packages = h.training.PYTHON.parent.parent
    directories = (packages, h.training.ASSET_ROOT, h.training.ORIGINAL, common.STARTUP_INPUTS,
                   Path('/usr/local/cuda-12.6'), Path('/usr/lib/x86_64-linux-gnu'),
                   Path('/lib/x86_64-linux-gnu'))
    exact = {workload / name for name in ('bert_contract.py', 'bert_workload.py', 'onnx_bert_workload.py',
                                           'export_bert.py')}
    exact.update(base / name for name in ('collect_application_block_colocation.py',
        'collect_owned_bert_training.py', 'collect_frontend_gpu.py', 'collect_service_gpu.py',
        'collect_dispatcher_gpu.py', 'native_command_audit.py', 'collect_priority_gpu.py',
        'arrival_notification.py', 'owned_child_cleanup.py', 'owned_onnx_result.py'))
    exact.update(project / 'scripts' / name for name in ('application_priority_audit.py',
        'execution_priority_audit.py', 'execution_semantic_audit.py', 'execution_application_receipt.py'))
    exact.update((common.OLD_ONNX, h.training.SERVER, h.training.PRELOAD, Path('/usr/bin/ldd')))
    roots = (h.training.SERVER, h.training.PRELOAD, h.training.PYTHON,
             h.training.PACKAGES / 'nvidia/cudnn/lib/libcudnn.so.9',
             h.training.PACKAGES / 'nvidia/nccl/lib/libnccl.so.2')
    require(len(set(roots)) == 5 and not environment.get('LD_PRELOAD') and not environment.get('LD_AUDIT'),
            'clean five-binary application loader inventory required')
    dynamic = {}
    for root in roots:
        output = subprocess.check_output(['/usr/bin/ldd', str(root)], env=environment, text=True, timeout=20)
        require(output.strip() and 'not found' not in output, 'unresolved application dependency: ' + str(root))
        for line in output.splitlines():
            line = line.strip()
            if re.fullmatch(r'linux-vdso\.so\.\d+ \(0x[0-9a-fA-F]+\)', line):
                continue
            match = re.fullmatch(r'(?:\S+ => )?(/\S+) \(0x[0-9a-fA-F]+\)', line)
            require(match is not None, 'unrecognized application loader row: ' + line)
            exact.add(Path(match[1]))
        exact.add(root)
        dynamic[str(root)] = re.sub(r'0x[0-9a-fA-F]+', '0xADDRESS', output)
    require(len(manifests) == 2 and all(item['dynamic_libraries'] == dynamic for item in manifests),
            'application dynamic loader closure differs from numerical runs')

    def selected(name):
        path = Path(name)
        return path in exact or any(path.is_relative_to(directory) for directory in directories)

    historical = [{name: value for name, value in item['inputs'].items() if selected(name)}
                  for item in manifests]
    require(historical[0] == historical[1], 'numerical application inputs differ')
    current = {str(path) for path in candidates if selected(path)} | {str(path) for path in exact}
    require(current == set(historical[0]), 'recorded application input inventory differs')
    return {Path(name) for name in current}, historical[0], fresh_application_contract(historical[0], dynamic)


def fresh_application_contract(expected, dynamic):
    """Summarize exactly which recorded identities and loader outputs were checked."""
    def rows_hash(rows):
        encoded = json.dumps(rows, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()
        return hashlib.sha256(encoded).hexdigest()

    rows = [[name, value['resolved'], value['identity'], value['sha256']]
            for name, value in sorted(expected.items())]
    hashed = [row for row in rows if row[3] is not None]
    stat_only = [row for row in rows if row[3] is None]
    return dict(schema='recorded_application_inputs_v1', recorded_files=len(rows),
                historical_sha256=len(hashed), historical_stat_only=len(stat_only),
                recorded_digest=rows_hash(rows), historical_sha256_digest=rows_hash(hashed),
                historical_stat_only_digest=rows_hash(stat_only),
                loader_sha256={name: hashlib.sha256(output.encode()).hexdigest()
                               for name, output in sorted(dynamic.items())},
                complete_python_import_closure=False)


def validate_fresh_application_inputs(expected, observed):
    """Check recorded metadata and only those historical hashes that exist."""
    hashed = stat_only = 0
    for name, old in expected.items():
        require(name in observed, 'recorded application input missing: ' + name)
        path, item = Path(name), observed[name]
        info = path.stat()
        identity = [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]
        require(identity == old['identity'] and item['target'] == old['resolved'] and
                item['bytes'] == info.st_size, 'application input identity changed: ' + name)
        if old['sha256'] is None:
            stat_only += 1
        else:
            require(item['sha256'] == old['sha256'], 'application input bytes changed: ' + name)
            hashed += 1
    return dict(recorded_files=len(expected), historical_sha256=hashed, historical_stat_only=stat_only,
                complete_python_import_closure=False)


def validate_fresh_contract(plan, review, variant, *, now=None, minimum_remaining=LAUNCH_ALLOWANCE_SECONDS):
    """Validate the independently reviewed fixed capture contract, without launching."""
    ordinary = plan.get('schema') == 'fresh_runtime_sgemm_v3'
    fields = (*FRESH_CONTRACT_FIELDS, 'specialization') if ordinary else FRESH_CONTRACT_FIELDS
    bridge.exact_fields(plan, (*fields, 'review'))
    bridge.exact_fields(plan['review'], ('path', 'sha256'))
    bridge.exact_fields(review, ('status', 'contract', 'source_sha256'))
    contract = {key: plan[key] for key in fields}
    require(review['status'] == ('fresh_runtime_v3_ready' if ordinary else 'fresh_runtime_v2_ready') and
            review['contract'] == contract and
            contract['schema'] == ('fresh_runtime_sgemm_v3' if ordinary else 'fresh_runtime_sgemm_v2') and
            (not ordinary or contract['specialization'] == 'ordinary') and contract['variant'] == variant and
            variant in VARIANTS and contract['target'] == '2:3127' and
            type(contract['max_seconds']) is int and contract['max_seconds'] == 800,
            'reviewed fixed fresh capture contract required')
    deadline = datetime.fromisoformat(contract['deadline_utc'])
    current = datetime.now(timezone.utc) if now is None else now
    require(type(minimum_remaining) is int and minimum_remaining >= 0 and
            deadline.tzinfo is not None and current.tzinfo is not None and
            (deadline - current).total_seconds() > 0 and
            (deadline - current).total_seconds() >= minimum_remaining,
            'insufficient time for bounded fresh capture')
    bridge.exact_fields(contract['runtime'], ('service', 'frontend'))
    bridge.exact_fields(contract['service_environment'], bridge.ENV_KEYS)
    bridge.exact_fields(contract['numerical'], VARIANTS)
    for item in contract['numerical'].values():
        bridge.exact_fields(item, ('root', 'seals', 'provenance'))
        bridge.exact_fields(item['provenance'], ('plan', 'launch', 'runner', 'auditor'))
    bridge.exact_fields(contract['reference'], ('path', 'targets'))
    bridge.exact_fields(contract['reference']['targets'],
                        ('training/result.json', 'training/snapshot.pt', 'inference/result.json'))
    for item in contract['reference']['targets'].values():
        bridge.exact_fields(item, ('target', 'sha256'))
    require(type(review['source_sha256']) is dict and review['source_sha256'],
            'reviewed capture sources required')
    return contract


def _check_fresh_binding_metadata(binding, *, minimum_remaining=LAUNCH_ALLOWANCE_SECONDS):
    """Check the reviewed contract and sources, without authenticating its evidence."""
    bridge.exact_fields(binding, ('schema', 'variant', 'contract', 'plan', 'plan_sha256',
                                  'review', 'review_sha256'))
    require(binding['schema'] in ('fresh_runtime_v2', 'fresh_runtime_v3') and binding['variant'] in VARIANTS,
            'fresh runtime binding required')
    specialization = bridge.capture_specialization(binding)
    plan_path, review_path = Path(binding['plan']), Path(binding['review'])
    require(plan_path == plan_path.resolve(strict=True) and review_path == review_path.resolve(strict=True) and
            plan_path.is_relative_to(bridge.PROJECT / 'artifacts/maintenance') and
            review_path.parent == plan_path.parent and digest(plan_path) == binding['plan_sha256'] and
            digest(review_path) == binding['review_sha256'], 'fresh binding plan or review changed')
    plan, review = read(plan_path), read(review_path)
    contract = validate_fresh_contract(plan, review, binding['variant'], minimum_remaining=minimum_remaining)
    require((specialization == 'ordinary') == (contract['schema'] == 'fresh_runtime_sgemm_v3') and
            contract == binding['contract'] and plan['review'] ==
            dict(path=str(review_path), sha256=binding['review_sha256']), 'fresh binding contract changed')
    require(all(Path(path).is_absolute() and digest(path) == expected
                for path, expected in review['source_sha256'].items()), 'fresh binding source changed')
    return contract, review


def _file_identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid,
            info.st_size, info.st_mtime_ns, info.st_ctime_ns)


class ReferenceGuardError(ValueError):
    """Report a reference verification or cleanup failure."""


class _PinnedReference:
    """Hold one hashed reference FD; identity checks assume a trusted same-UID owner."""

    def __init__(self, link, target, expected):
        self.link, self.target, self.expected = link, target, expected
        self.pid = os.getpid()
        self.fd = None
        self.link_fd = None
        self.entered = False

    def __enter__(self):
        require(not self.entered and os.getpid() == self.pid, 'reference guard cannot be reused or forked')
        self.entered = True
        link, target = self.link, self.target
        require(link.is_symlink() and os.readlink(link) == str(target) and target.is_absolute() and
                target == target.resolve(strict=True) and link.resolve(strict=True) == target,
                'canonical snapshot reference link required')
        self.link_fd = os.open(link, os.O_PATH | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            link_info = os.fstat(self.link_fd)
            require(stat.S_ISLNK(link_info.st_mode) and not os.get_inheritable(self.link_fd),
                    'non-inheritable snapshot link required')
            self.link_identity = _file_identity(link_info)
            self.fd = os.open(target, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
            info = os.fstat(self.fd)
            require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid() and 0 < info.st_size <= 1024 ** 3 and
                    not os.get_inheritable(self.fd), 'owned non-inheritable snapshot file required')
            self.identity = _file_identity(info)
            self.verify()
            hasher = hashlib.sha256()
            remaining = info.st_size
            while remaining:
                block = os.read(self.fd, min(remaining, 1024 * 1024))
                require(block, 'snapshot ended during hashing')
                hasher.update(block)
                remaining -= len(block)
            require(not os.read(self.fd, 1), 'snapshot grew during hashing')
            require(hasher.hexdigest() == self.expected, 'snapshot content differs')
            self.verify()
            return self
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup_error:
                raise ReferenceGuardError(f'reference entry failed: {error}; cleanup failed: {cleanup_error}') from error
            raise

    def close(self):
        failures = []
        for name in ('fd', 'link_fd'):
            descriptor = getattr(self, name)
            if descriptor is not None:
                setattr(self, name, None)
                try:
                    os.close(descriptor)
                except OSError as error:
                    failures.append(f'{name}: {error}')
        if failures:
            raise ReferenceGuardError('reference close failed: ' + '; '.join(failures))

    def verify(self):
        require(self.fd is not None and self.link_fd is not None and os.getpid() == self.pid,
                'reference guard is closed or forked')
        require(_file_identity(os.fstat(self.fd)) == self.identity and
                _file_identity(self.target.lstat()) == self.identity and
                _file_identity(os.fstat(self.link_fd)) == self.link_identity and
                _file_identity(self.link.lstat()) == self.link_identity and
                self.link.is_symlink() and os.readlink(self.link) == str(self.target) and
                self.target == self.target.resolve(strict=True) and
                self.link.resolve(strict=True) == self.target,
                'snapshot reference identity changed')

    def __exit__(self, exception_type, exception, traceback):
        failures = []
        try:
            self.verify()
        except BaseException as verification_error:
            failures.append(f'postflight {type(verification_error).__name__}: {verification_error}')
        try:
            self.close()
        except BaseException as cleanup_error:
            failures.append(f'cleanup {type(cleanup_error).__name__}: {cleanup_error}')
        if failures:
            if exception is not None:
                failures.append(f'body {type(exception).__name__}: {exception}')
            raise ReferenceGuardError('; '.join(failures)) from exception


def _check_reference_target(reference_path, name, item, native):
    link, target = reference_path / name, Path(item['target'])
    require(link.is_symlink() and os.readlink(link) == str(target) and target.is_absolute() and
            target == target.resolve(strict=True) and link.resolve(strict=True) == target and
            target.is_file() and item['sha256'] == native['artifact_identities'][name] and
            (name == 'training/snapshot.pt' or digest(target) == item['sha256']),
            'fresh native reference bytes or destination differ')
    return link, target


@contextmanager
def prepare_fresh(plan_path, variant):
    """Hold a prepared V2 capture reference; no GPU or privileged launch.

    Only the reference snapshot is pinned. Service authentication and frontend
    mappings require their own checks; a same-UID adversary is outside this guard.
    """
    plan_path = Path(plan_path)
    require(plan_path.is_absolute() and plan_path == plan_path.resolve(strict=True) and
            plan_path.is_relative_to(bridge.PROJECT / 'artifacts/maintenance') and plan_path.is_file(),
            'canonical fresh capture plan required')
    plan = read(plan_path)
    review_path = Path(plan['review']['path'])
    require(review_path.parent == plan_path.parent and review_path == review_path.resolve(strict=True) and
            review_path.is_file() and digest(review_path) == plan['review']['sha256'],
            'independent fresh capture review changed')
    review = read(review_path)
    contract = validate_fresh_contract(plan, review, variant)
    required_sources = {str(Path(__file__).resolve()), str(bridge.SCRIPT), str(Path(candidate.__file__).resolve()),
                        *(str(Path(__file__).with_name(name).resolve(strict=True)) for name in
                          ('run_bert_sgemm_ncu_capture.py', 'bert_ncu_sibling.py', 'bert_ncu_relay.py',
                           'bert_ncu_service_proxy.py', 'run_bert_sgemm_descriptor_probe.py',
                           'bert_sgemm_coverage_candidates.py', 'bert_sgemm_ncu_process.py'))}
    sources = review['source_sha256']
    require(required_sources <= set(sources) and
            all(Path(path).is_absolute() and digest(path) == expected for path, expected in sources.items()),
            'reviewed fresh capture sources changed')
    evidence = {}
    for name in VARIANTS:
        item = contract['numerical'][name]
        evidence[name] = fresh_numerical_evidence(item['root'], item['seals'], name, contract['runtime'],
                                                  item['provenance'], os.getuid(), os.getgid())
    pair_receipt = match_fresh_numerical_pair(evidence['scalar'], evidence['sharded'])
    require(pair_receipt['runtime'] == contract['runtime'] and
            pair_receipt['service_environment'] == contract['service_environment'],
            'capture runtime or service environment differs from both numerical runs')
    reference = contract['reference']
    reference_path = Path(reference['path'])
    require(reference_path == reference_path.resolve(strict=True) and reference_path.is_dir() and
            reference_path.is_relative_to(bridge.PROJECT / 'artifacts/maintenance') and
            reference_path.stat().st_uid == os.getuid(), 'owned fresh numerical reference view required')
    paths = {plan_path, review_path, Path(contract['runtime']['service']['path']),
             Path(contract['runtime']['frontend']['path'])}
    paths.update(Path(path) for path in sources)
    snapshot = reference['targets']['training/snapshot.pt']
    native = evidence[variant]['native_job']
    require(snapshot['sha256'] == native['artifact_identities']['training/snapshot.pt'],
            'fresh native snapshot identity differs')
    with _PinnedReference(reference_path / 'training/snapshot.pt', Path(snapshot['target']), snapshot['sha256']) as pin:
        result_identities = {}
        for name, item in reference['targets'].items():
            link, target = _check_reference_target(reference_path, name, item, native)
            paths.update((link, target))
            if name != 'training/snapshot.pt':
                result_identities[name] = (_file_identity(link.lstat()), _file_identity(target.lstat()))
        for item in contract['numerical'].values():
            root = Path(item['root'])
            paths.update(root / name for name in item['seals'])
            paths.update(Path(value['path']) for value in item['provenance'].values())
            paths.add(Path(read(root / 'report.json')['claim']))
        binding = dict(schema='fresh_runtime_v3' if contract['schema'] == 'fresh_runtime_sgemm_v3'
                       else 'fresh_runtime_v2', variant=variant, contract=contract, plan=str(plan_path),
                       plan_sha256=digest(plan_path), review=str(review_path), review_sha256=plan['review']['sha256'])
        _check_fresh_binding_metadata(binding)
        application = dict(manifests=[evidence[name]['manifest'] for name in VARIANTS], expected=None, summary=None)
        yield binding, paths, native, pin, application
        for name, item in reference['targets'].items():
            if name != 'training/snapshot.pt':
                link, target = _check_reference_target(reference_path, name, item, native)
                require((_file_identity(link.lstat()), _file_identity(target.lstat())) == result_identities[name],
                        'fresh native result identity changed')


def _read_claim_input(path, limit, *, retain=False):
    """Bound and seal one owned input through the same no-follow file descriptor."""
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        before = os.fstat(descriptor)
        require(path == path.resolve(strict=True) and stat.S_ISREG(before.st_mode) and
                (before.st_uid, before.st_gid, before.st_nlink) == (os.getuid(), os.getgid(), 1) and
                0 < before.st_size <= limit and _file_identity(before) == _file_identity(path.lstat()),
                'fresh counter input file differs')
        remaining, hasher, blocks = before.st_size, hashlib.sha256(), []
        while remaining:
            data = os.read(descriptor, min(remaining, 1024 * 1024))
            require(data, 'fresh counter input ended during reading')
            hasher.update(data)
            if retain:
                blocks.append(data)
            remaining -= len(data)
        require(not os.read(descriptor, 1) and _file_identity(os.fstat(descriptor)) == _file_identity(before) and
                _file_identity(path.lstat()) == _file_identity(before), 'fresh counter input changed')
        return hasher.hexdigest(), b''.join(blocks) if retain else None
    finally:
        os.close(descriptor)


def claim_fresh(binding, attempt, application_contract):
    """Consume one reviewed plan before spawn; failed writes never remove the claim."""
    contract, _ = _check_fresh_binding_metadata(binding)
    plan = Path(binding['plan'])
    attempt = Path(attempt)
    require(attempt == attempt.resolve(strict=True) and attempt.is_dir() and
            attempt.parent == bridge.PROJECT / 'artifacts/profiles/ncu' and
            re.fullmatch(r'attempt-[A-Za-z0-9_]+', attempt.name) is not None and
            (attempt.stat().st_uid, attempt.stat().st_gid) == (os.getuid(), os.getgid()) and
            stat.S_IMODE(attempt.stat().st_mode) == 0o700,
            'fresh owned counter attempt required')
    manifest, inputs = attempt / 'manifest.json', attempt / 'inputs-before.json'
    manifest_sha, manifest_data = _read_claim_input(manifest, bridge.MESSAGE_MAX, retain=True)
    inputs_sha, _ = _read_claim_input(inputs, 64 * 1024 ** 2)
    template = bridge.decode(manifest_data)
    service = contract['runtime']['service']
    require(type(application_contract) is dict and
            application_contract.get('schema') == 'recorded_application_inputs_v1' and
            template.get('application_input_contract') == application_contract and
            template.get('fresh_runtime') == binding and template.get('service_cap_sys_admin') is True and
            template.get('sgemm_target') == contract['target'] == '2:3127' and
            template.get('service_environment') == contract['service_environment'] and
            template.get('bridge_seals', {}).get(service['path']) == service['sha256'] and
            (template.get('uid'), template.get('gid')) == (os.getuid(), os.getgid()),
            'fresh counter manifest differs from reviewed binding')
    encoded_binding = json.dumps(binding, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    require(len(encoded_binding) <= bridge.MESSAGE_MAX, 'fresh binding exceeds claim limit')
    specialization = bridge.capture_specialization(binding)
    record = dict(schema='fresh_runtime_claim_v2' if specialization == 'ordinary' else 'fresh_runtime_claim_v1',
                  attempt=str(attempt), variant=binding['variant'],
                  target=contract['target'], plan=str(plan), plan_sha256=binding['plan_sha256'],
                  review=binding['review'], review_sha256=binding['review_sha256'],
                  binding_sha256=hashlib.sha256(encoded_binding).hexdigest(),
                  manifest_sha256=manifest_sha, inputs_sha256=inputs_sha)
    if specialization == 'ordinary':
        record['specialization'] = specialization
    payload = (json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    require(len(payload) <= 4096, 'fresh claim exceeds size limit')
    directory = plan.parent
    directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    descriptor, created, errors = None, False, []
    path = directory / (plan.name + '.launched')
    try:
        observed, current = os.fstat(directory_fd), directory.lstat()
        require(directory == directory.resolve(strict=True) and stat.S_ISDIR(observed.st_mode) and
                observed.st_uid == os.getuid() and observed.st_mode & 0o022 == 0 and
                _file_identity(observed) == _file_identity(current), 'fresh claim directory changed')
        descriptor = os.open(path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                             0o600, dir_fd=directory_fd)
        created = True
        os.fchmod(descriptor, 0o600)
        info = os.fstat(descriptor)
        require(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o600 and
                (info.st_uid, info.st_gid, info.st_nlink) == (os.getuid(), os.getgid(), 1),
                'fresh claim ownership or mode differs')
        remaining = memoryview(payload)
        while remaining:
            count = os.write(descriptor, remaining)
            require(count > 0, 'short fresh claim write')
            remaining = remaining[count:]
        os.fsync(descriptor)
    except BaseException as error:
        errors.append(error)
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError as error:
                errors.append(error)
        if created:
            try:
                os.fsync(directory_fd)
            except OSError as error:
                errors.append(error)
        try:
            os.close(directory_fd)
        except OSError as error:
            errors.append(error)
    if errors:
        raise ReferenceGuardError('fresh claim failed: ' + '; '.join(str(error) for error in errors)) from errors[0]
    actual_sha, _ = _read_claim_input(path, 4096)
    require(stat.S_IMODE(path.lstat().st_mode) == 0o600 and actual_sha == hashlib.sha256(payload).hexdigest(),
            'fresh claim changed after publication')
    return dict(path=str(path), sha256=actual_sha)


def numerical_evidence(root, expected_seals, variant):
    """Check the published numerical prerequisite without repeating tensor comparisons.

    This establishes numerical evidence only. It does not establish SM coverage,
    performance, or a descriptor match for a future profiler launch.
    """
    root = Path(root).resolve(strict=True)
    require(variant in VARIANTS, 'unknown current SDK variant')
    report_path, plan_path = root / 'report.json', root / 'plan.json'
    expected_paths = {root / name for name in ('report.json', 'plan.json', 'batch-publication.json',
                                               'inputs-before.json', 'inputs-after.json')}
    for name in ('native', *VARIANTS):
        directory = root / f'{name}-{root.name}'
        mode = 'native' if name == 'native' else 'equal'
        expected_paths.update((directory / f'{mode}-worker.json', directory / mode / 'attempt.json'))
    require(set(expected_seals) == {str(path) for path in expected_paths} and
            all(digest(path) == expected_seals[str(path)] for path in expected_paths),
            'independently bound numerical evidence changed')
    publication = read(root / 'batch-publication.json')
    require(publication['complete'] is True and
            publication['qualification'] == 'numerical_evidence_only' and
            publication['report'] == str(report_path) and
            publication['report_sha256'] == expected_seals[str(report_path)] and
            publication['plan_sha256'] == digest(plan_path), 'numerical publication changed')
    report, plan = read(report_path), read(plan_path)
    require(report['status'] == 'fresh_native_scalar_sharded_numerical_requires_review' and
            report['numerical_three_path_passed'] is True and report['input_identity_unchanged'] is True and
            read(root / 'inputs-before.json') == read(root / 'inputs-after.json') and
            plan['variants'] == ['native', 'scalar', 'sharded'], 'complete unchanged numerical batch required')
    require(set(publication['workers']) == {'native', *VARIANTS} and
            len(report['jobs']) == 3, 'exact three-path numerical inventory required')
    rows = {row['variant']: row for row in report['jobs']}
    require(set(rows) == {'native', *VARIANTS}, 'duplicate or missing numerical variant')
    for name, row in rows.items():
        worker = publication['workers'][name]
        worker_path = Path(worker['path'])
        mode = 'native' if name == 'native' else 'equal'
        directory = root / f'{name}-{root.name}'
        require(worker_path == directory / f'{mode}-worker.json' and
                worker_path == worker_path.resolve(strict=True) and
                digest(worker_path) == worker['sha256'] and read(worker_path) == row['worker'] and
                row['unit']['returncode'] == 0 and row['unit']['unit_removed'] is True and
                row['worker']['status'] == 'passed_requires_review' and
                row['worker']['reparse'] == ORACLE and row['worker']['variant'] == name and
                row['worker']['mode'] == mode and row['worker']['job']['mode'] == mode and
                read(directory / mode / 'attempt.json') == row['worker']['job'] and
                row['worker']['scoped_publication_valid'] is False and
                row['worker']['batch_valid'] is False, 'numerical worker evidence changed')
        require(all(row['worker'].get(key) is value for key, value in FLAGS.items()),
                'worker numerical-only scope changed')
        candidate.normal_cleanup(row['worker']['job'])
    require(all(record.get(key) is value for record in (report, publication, plan)
                for key, value in FLAGS.items()), 'batch numerical-only scope changed')
    worker = rows[variant]['worker']
    job = worker['job']
    log = root / f'{variant}-{root.name}' / 'equal/server.log'
    require(digest(log) == job['artifact_identities']['server.log'], 'numerical service log changed')
    content = log.read_text()
    require(content.count('GEMM signal diagnostic: sharded=true') == (variant == 'sharded'),
            'numerical signal selector differs')
    semantic = parse_semantic_audit(content, require_complete=True)
    require(2 in authenticated_training_sessions(semantic, job),
            'target session is not the authenticated configured training client')
    patterns = (
        r'Native admission audit: session=2, sequence=3127, accepted=true, operation=30(?:\s|$)',
        r'Native persistent audit: session=2, sequence=3127, total_blocks=2304, '
        r'physical_launches=1, workers=320(?:\s|$)',
        r'Native issue audit: session=2, sequence=3127, status=0(?:\s|$)',
        r'Native execution audit: session=2, issued=3127, retained=0, retired=1, status=0(?:\s|$)',
        r'Priority execution audit: event=low_logical_done, session=2, priority=0, sequence=3127, '
        r'completed_blocks=2304, total_blocks=2304, physical_launches=1, status=0(?:\s|$)',
    )
    matches = [list(re.finditer(pattern, content)) for pattern in patterns]
    require(all(len(found) == 1 for found in matches) and
            [found[0].start() for found in matches] == sorted(found[0].start() for found in matches),
            'exact target was not observed completing in numerical batch')
    return plan, rows['native']['worker']['job']


def check_binding(binding):
    """Recheck the explicit reviewed runtime; never fall back to a legacy service."""
    for key in ('plan', 'runtime_manifest', 'launch_review'):
        require(digest(binding[key]) == binding[key + '_sha256'], key + ' changed')
    plan, review = read(binding['plan']), read(binding['launch_review'])
    require(plan['schema'] == 'current_sdk_sgemm_v1' and binding['variant'] in VARIANTS and
            plan['variant'] == binding['variant'] and
            review['status'] == 'current_sdk_ncu_ready' and review['variant'] == binding['variant'] and
            review['numerical_acceptance'] == 'fresh_three_path_numerical_accepted' and
            review['numerical_seals'] == plan['numerical_seals'] and
            review['reference_targets'] == plan['reference_targets'] and
            review['native_reference'] == plan['native_reference'] and
            review['target'] == plan['target'] == '2:3127' and review['max_seconds'] == plan['max_seconds'] == 800 and
            review['runtime_manifest_sha256'] == binding['runtime_manifest_sha256'] and
            all(plan[key] == binding[key] for key in ('runtime_manifest', 'runtime_manifest_sha256',
                'launch_review', 'launch_review_sha256', 'native_reference')),
            'independent current SDK launch review required')
    service = bridge.selected_service(dict(uid=os.getuid(), current_sdk=binding))
    frontend = Path(binding['frontend'])
    product = read(binding['runtime_manifest'])['frontend']
    metadata = frontend.stat()
    require(str(service) == binding['server'] and frontend == frontend.resolve(strict=True) and
            str(frontend) == product['path'] and stat.S_ISREG(metadata.st_mode) and
            metadata.st_uid == os.getuid() and
            (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_ctime_ns) ==
            (product['device'], product['inode'], product['size'], product['ctime_ns']) and
            digest(frontend) == product['sha256'] == binding['frontend_sha256'],
            'selected SDK runtime changed')
    return plan, review


def prepare(plan_path, variant):
    """Resolve a reviewed candidate and its numerical reference before GPU allocation."""
    plan_path = Path(plan_path).resolve(strict=True)
    plan = read(plan_path)
    deadline = datetime.fromisoformat(plan['deadline_utc'])
    require(variant in VARIANTS and plan['variant'] == variant and plan['target'] == '2:3127' and
            deadline.tzinfo is not None and
            (deadline - datetime.now(timezone.utc)).total_seconds() >= LAUNCH_ALLOWANCE_SECONDS and
            plan['max_seconds'] == 800 and
            plan['runtime_environment'] == {'GLIMMER_EXECUTION_HOST_RANGE_SEEK': '1'},
            'fixed current SDK target and runtime required')
    numerical_plan, native_job = numerical_evidence(plan['numerical_root'], plan['numerical_seals'], variant)
    runtime_path = Path(plan['runtime_manifest']).resolve(strict=True)
    runtime = read(runtime_path)
    original = runtime['variants'][variant]['original_server']
    server = runtime['variants'][variant]['glimmer_execution_service']['path']
    frontend = runtime['frontend']['path']
    original_frontend = runtime['original_frontend']
    require(numerical_plan['artifact_sha256'][original] ==
            runtime['variants'][variant]['glimmer_execution_service']['sha256'] and
            numerical_plan['artifact_sha256'][original_frontend] == runtime['frontend']['sha256'],
            'counter binaries do not match the numerical batch')
    binding = dict(plan=str(plan_path), plan_sha256=digest(plan_path), variant=variant,
                   runtime_manifest=str(runtime_path), runtime_manifest_sha256=plan['runtime_manifest_sha256'],
                   launch_review=plan['launch_review'], launch_review_sha256=plan['launch_review_sha256'],
                   server=server, frontend=frontend, frontend_sha256=runtime['frontend']['sha256'],
                   native_reference=plan['native_reference'])
    _, review = check_binding(binding)
    required_sources = {str(Path(__file__).resolve()), str(bridge.SCRIPT),
                        str(Path(__file__).with_name('run_bert_sgemm_ncu_capture.py').resolve()),
                        str(Path(__file__).with_name('bert_ncu_sibling.py').resolve()),
                        str(Path(__file__).with_name('bert_ncu_relay.py').resolve())}
    require(required_sources <= set(review['source_sha256']) and
            all(digest(path) == expected for path, expected in review['source_sha256'].items()),
            'reviewed counter source changed')
    reference = Path(binding['native_reference'])
    require(reference == reference.resolve(strict=True) and reference.is_dir(),
            'canonical native reference view required')
    expected_artifacts = native_job['artifact_identities']
    paths = {plan_path, runtime_path, Path(binding['launch_review']), Path(server), Path(frontend)}
    paths.update(Path(path) for path in plan['numerical_seals'])
    paths.update(Path(path) for path in review['source_sha256'])
    paths.update(Path(path) for path in numerical_plan['runtime_identity'])
    source_hashes = {name: sha for name, sha in numerical_plan['artifact_sha256'].items()
                     if name.startswith(str(bridge.PROJECT / 'src') + '/')}
    require(source_hashes, 'numerical source identities required')
    paths.update(Path(path) for path in source_hashes)
    binding['numerical_runtime_identity'] = numerical_plan['runtime_identity']
    binding['numerical_source_sha256'] = source_hashes
    for name in ('training/result.json', 'training/snapshot.pt', 'inference/result.json'):
        path = reference / name
        target = path.resolve(strict=True)
        expected = plan['reference_targets'][name]
        require(path.is_symlink() and os.readlink(path) == expected['link'] and
                str(target) == expected['target'] and digest(target) == expected['sha256'] ==
                expected_artifacts[name], 'deduplicated native reference changed')
        paths.update((path, target))
    require(set(plan['reference_targets']) == {'training/result.json', 'training/snapshot.pt',
                                               'inference/result.json'}, 'exact native reference view required')
    return binding, paths, native_job


def configure(binding):
    check_binding(binding)
    return pair.numerical.configure(dict(server=binding['server'], frontend=binding['frontend'],
                                         runtime_environment={'GLIMMER_EXECUTION_HOST_RANGE_SEEK': '1'}))


def configure_fresh(binding):
    """Configure a replay after the caller authenticates its full caps0 status.

    This checks frozen metadata, not reference lifetime or launch authorization.
    """
    uid, gid = os.getuid(), os.getgid()
    _require_caps0(uid, gid)
    contract, _ = _check_fresh_binding_metadata(binding, minimum_remaining=0)
    runtime = contract['runtime']
    service = bridge.validate_runtime_product(runtime['service'], uid, gid)
    frontend = bridge.validate_runtime_product(runtime['frontend'], uid, gid)
    require(service != frontend and
            (runtime['service']['device'], runtime['service']['inode']) !=
            (runtime['frontend']['device'], runtime['frontend']['inode']),
            'fresh service and frontend must differ')
    common, h = pair.numerical.configure(dict(server=str(service), frontend=str(frontend),
                                              runtime_environment={'GLIMMER_EXECUTION_HOST_RANGE_SEEK': '0'}))
    require(h.training.SERVER == service and h.training.PRELOAD == frontend,
            'fresh replay helper runtime differs')
    sharded = binding['variant'] == 'sharded'

    def configuration(content):
        lines = [line for line in content.splitlines() if 'Execution service configuration:' in line]
        expected = (common.PREFIX + r'Execution service configuration: scheduler=priority, '
                    r'kernel_mode=preemptive, configured_workers=320, nccl_enabled=false, '
                    r'sharded_gemm_signal=' + str(sharded).lower() + r'; full_device_qualified=false')
        require(len(lines) == 1 and re.fullmatch(expected, lines[0]) is not None,
                'fresh replay service configuration differs')
        return dict(scheduler='priority', kernel_mode='preemptive', configured_workers=320,
                    nccl_enabled=False, sharded_gemm_signal=sharded, full_device_qualified=False)

    common.validate_service_configuration = configuration
    return common, h


def validate_numerical_inputs(binding, inputs):
    require(binding['numerical_runtime_identity'] and binding['numerical_source_sha256'] and
            all(inputs[path] == identity for path, identity in
                binding['numerical_runtime_identity'].items()) and
            all(inputs[path]['sha256'] == expected for path, expected in
                binding['numerical_source_sha256'].items()),
            'numerically validated runtime or source changed')


def validate_job(binding, job):
    check_binding(binding)
    require(job['mode'] == 'equal', 'replacement counter path required')
    command = list(job['server_command'])
    low = job['application_identities']['low']
    for option, expected in (
            ('--audit-sgemm-target-client', f"{low['pid']}:{low['start']}:3127"),
            ('--audit-sgemm-range-library', str(bridge.RANGE_LIBRARY.resolve(strict=True)))):
        require(command.count(option) == 1, 'one exact counter selector required')
        index = command.index(option)
        require(index + 1 < len(command) and command[index + 1] == expected, 'counter selector differs')
        del command[index:index + 2]
    pair.validate_job(dict(server=binding['server'], frontend=binding['frontend'],
                           runtime_environment={'GLIMMER_EXECUTION_HOST_RANGE_SEEK': '1'}),
                      dict(job, server_command=command))


def claim_launch(binding):
    plan, _ = check_binding(binding)
    deadline = datetime.fromisoformat(plan['deadline_utc'])
    require(deadline.tzinfo is not None and plan['max_seconds'] == 800 and
            (deadline - datetime.now(timezone.utc)).total_seconds() >= LAUNCH_ALLOWANCE_SECONDS,
            'insufficient time for the bounded profiler and cleanup')
    plan_path = Path(binding['plan']).resolve(strict=True)
    variant = binding['variant']
    path = plan_path.with_name('ncu-' + variant + '-launch.json')
    with path.open('x') as stream:
        json.dump(dict(plan=str(plan_path), plan_sha256=binding['plan_sha256'], variant=variant), stream)
        stream.write('\n')
