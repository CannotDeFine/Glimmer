"""Complete unchanged BERT replay in a caps0 sibling of the NCU process tree."""

import hashlib
import json
import os
from pathlib import Path
import tempfile

import bert_ncu_replay_exec as bridge
from bert_ncu_service_proxy import (FrontendMappings, RelayConnection, service_launches, service_signal,
                                    validate_launch_order, validate_ordinary_release)
from bert_sgemm_coverage_candidates import targeted_training_binding, targeted_issue_range
from bert_sgemm_ncu_process import proc_mapped_paths
from execution_semantic_audit import parse_semantic_audit
import run_bert_candidate_validation as candidate
from run_bert_sgemm_descriptor_probe import FLAGS, parse_target, selected_service_target


def run(outer, endpoint, supervisor, serial):
    # Import only at execution time to keep all collectors' cold help CPU-only.
    from run_bert_sgemm_ncu_capture import publish_json, task_ancestors

    outer = outer.resolve(strict=True)
    report, guard, relay, directory, launch_order = dict(status='started', **FLAGS), None, None, None, None
    try:
        manifest = bridge.decode((outer / 'manifest.json').read_bytes())
        bridge.require(manifest['service_cap_sys_admin'] is True, 'explicit service-capability mode required')
        bridge.validate_status(Path('/proc/self/status').read_text(), manifest['uid'], manifest['gid'], False)
        bridge.require(not proc_mapped_paths(os.getpid()), 'sibling controller must be outside NCU injection')
        environment = manifest['service_environment']
        expected = dict(environment, TMPDIR=str(outer / 'ncu-tmp'), CUDA_CACHE_DISABLE='1')
        bridge.require(dict(os.environ) == expected, 'sibling controller inherited unexpected environment')
        go = json.loads((outer / 'ncu-go.json').read_text())
        deadline_ns = go['unit_started_us'] * 1000 + 780_000_000_000
        relay = RelayConnection(outer, endpoint, supervisor, serial, deadline_ns)
        guard = candidate.ResourceGuard(outer, 'equal')
        report.update(controller_identity=bridge.process_identity(os.getpid()),
                      ancestors=task_ancestors(guard.group), topology='clean-ncu-sibling', serial=serial)
        directory = Path(tempfile.mkdtemp(prefix=f'pass-{serial:04d}-', dir=outer / 'passes'))
        report['pass_directory'] = str(directory)
        binding = bridge.selected_signal_binding(manifest)
        if binding is not None:
            from bert_ncu_current_sdk import configure_fresh
            bridge.require(environment == binding['contract']['service_environment'] and
                           manifest['sgemm_target'] == binding['contract']['target'],
                           'fresh replay environment or target differs')
            common, h = configure_fresh(binding)
        elif 'current_sdk' in manifest:
            from bert_ncu_current_sdk import configure
            common, h = configure(manifest['current_sdk'])
        elif 'host_pair' in manifest:
            from bert_ncu_host_pair import configure
            common, h = configure(manifest['host_pair'])
        else:
            common, h = candidate.load_helpers()
        h.startup_reference, h.export = common.application_references(h, manifest)
        gpu = manifest['gpu'].split(',')[0].strip()
        bridge.require(h.query('gpu', h.training.GPU_FIELDS) == manifest['gpu'] and
                       h.training.clean_environment(gpu) == environment,
                       'GPU or fixed application environment changed')
        provider = Path(manifest['range_library'])
        bridge.require(provider == bridge.RANGE_LIBRARY.resolve(strict=True) and
                       bridge.seal(provider) == manifest['range_library_sha256'], 'range provider changed')
        _, sequence = parse_target(manifest['sgemm_target'])
        bridge.require(manifest['sgemm_selector'] == dict(kind='client', role='low', sequence=sequence),
                       'fixed training selector differs')
        native = (Path(binding['contract']['reference']['path']) if binding is not None
                  else Path(manifest['current_sdk']['native_reference']) if 'current_sdk' in manifest
                  else Path(manifest['host_pair']['native_reference']) if 'host_pair' in manifest
                  else candidate.EXISTING_NATIVE / 'native')
        service = h.training.SERVER if binding is not None else bridge.selected_service(manifest)
        if binding is not None:
            bridge.require(service == Path(binding['contract']['runtime']['service']['path']),
                           'fresh controller service differs from its authenticated runtime')
        bridge.require(h.training.SERVER == service, 'controller and sealed service selection differ')
        frontend_observer = (FrontendMappings(binding['contract']['runtime']['frontend'],
                                               binding['contract']['service_environment']) if binding is not None
                             else None)
        with selected_service_target(common, manifest['sgemm_target'], provider, client_bound=True), \
                service_signal(common, binding), \
                service_launches(common, h, relay, directory, environment, service=service,
                                 signal_binding=binding, frontend_observer=frontend_observer) as launch_order:
            job = candidate.guarded_control(common, h, directory, 'equal', environment, guard, native)
        report['job'] = job
        if frontend_observer is not None:
            report['frontend_mappings'] = frontend_observer.for_job(job)
            validate_launch_order(job, launch_order)
            report['launch_order'] = launch_order
        report['reparse'] = candidate.reparse(common, h, directory, 'equal', job, gpu, native)
        expected_command = bridge.service_command(dict(clients=job['application_identities'],
                                                        socket=job['socket_path']), service=service,
                                                  signal_binding=binding)
        bridge.require(job['server_command'] == expected_command, 'final fixed service command differs')
        log = directory / 'equal/server.log'
        bridge.require(log.stat().st_size <= 64 * 1024 * 1024, 'service audit exceeds parser capacity')
        data = log.read_bytes()
        bridge.require(len(data) <= 64 * 1024 * 1024 and hashlib.sha256(data).hexdigest() ==
                       job['artifact_identities']['server.log'], 'service audit changed')
        content = data.decode('utf-8')
        semantic = parse_semantic_audit(content, require_complete=True)
        report['target_binding'] = targeted_training_binding(content, semantic, job, sequence, manifest['uid'])
        if binding is not None and bridge.capture_specialization(binding) == 'ordinary':
            report['logical_release'] = validate_ordinary_release(content, job, report['target_binding'])
        session = report['target_binding']['session']
        report['range'] = targeted_issue_range(content, session, sequence)
        guard.check(force=True)
        report['status'] = 'pass_requires_profiler_review'
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        if launch_order is not None:
            report['launch_order_state'] = launch_order
    finally:
        if guard is not None:
            report['resources'] = guard.receipt
        try:
            if directory is not None:
                publish_json(directory / 'pass.json', report)
            if relay is not None:
                relay.request('complete', directory=str(directory) if directory else None,
                              accepted=report['status'] == 'pass_requires_profiler_review',
                              report_sha256=bridge.seal(directory / 'pass.json') if directory else None)
        except BaseException as error:
            report.update(status='failed', completion_error=f'{type(error).__name__}: {error}')
            print(json.dumps(report), flush=True)
        finally:
            if relay is not None:
                relay.close()
    print(json.dumps(dict(status=report['status'], error=report.get('error'),
                          directory=str(directory) if directory else None)), flush=True)
    return int(report['status'] != 'pass_requires_profiler_review')
