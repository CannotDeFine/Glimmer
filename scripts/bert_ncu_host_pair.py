"""Explicit immutable host-pair selection for separate BERT counter captures."""

import json
from pathlib import Path

import run_bert_host_pair_numerical as pair
from run_rnn_v8_validation import digest

require, read = pair.require, pair.read


def launch_review(plan, variant):
    """O2 requires a separate review bound to the independently accepted O0 capture."""
    review = read(plan['launch_review'])
    require(digest(plan['launch_review']) == plan['launch_review_sha256'] and
            review['status'] == 'host_pair_ncu_ready' and review['variant'] == variant,
            'variant-specific counter launch review required')
    if variant == 'o2':
        previous = review['predecessor']
        root = Path(previous['attempt']).resolve(strict=True)
        require(digest(root / 'report.json') == previous['report_sha256'] and
                digest(root / 'review.json') == previous['review_sha256'], 'O0 predecessor changed')
        report, accepted = read(root / 'report.json'), read(root / 'review.json')
        require(accepted['status'] == 'host_pair_replacement_target_80sm_accepted' and
                accepted['variant'] == 'o0' and accepted['report_sha256'] == previous['report_sha256'] and
                report['status'] == 'counter_observation_requires_independent_review' and
                report['input_identity_unchanged'] is True and report['unit']['returncode'] == 0 and
                report['unit']['unit_removed'] is True and
                report['observation']['metrics']['minimums_positive'] is True and
                report['observation']['metrics']['metrics']['launch__sm_count'] == 80 and
                report['host_pair']['variant'] == 'o0' and
                report['host_pair']['runtime_manifest_sha256'] == plan['runtime_manifest_sha256'] and
                report['host_pair']['numerical_report_sha256'] == plan['numerical_report_sha256'],
                'O0 must pass counters, lifecycle and independent acceptance before O2')
    return review


def check_binding(binding):
    """Authenticate selected runtime and reference without reloading helpers."""
    for key in ('plan', 'runtime_manifest', 'numerical_report', 'numerical_review'):
        require(digest(binding[key]) == binding[key + '_sha256'], key + ' changed')
    runtime = read(binding['runtime_manifest'])
    require(binding['variant'] in pair.VARIANTS, 'known immutable variant required')
    products = runtime['variants'][binding['variant']]
    for key, product in (('server', 'glimmer_execution_service'),
                         ('frontend', 'glimmer_execution_interceptor')):
        require(binding[key] == products[product]['path'] and
                digest(binding[key]) == products[product]['sha256'], 'runtime selection changed')
    report = read(binding['numerical_report'])
    expected_native = Path(report['variant_attempts'][binding['variant']]) / 'native'
    require(Path(binding['native_reference']) == expected_native, 'own-variant native reference required')
    review = read(binding['numerical_review'])
    require(review['status'] == 'fresh_host_pair_numerical_accepted' and
            review['report_sha256'] == binding['numerical_report_sha256'], 'numerical acceptance changed')
    return report


def prepare(plan_path, variant):
    """Resolve the reviewed prerequisite before allocating a GPU experiment."""
    plan_path = Path(plan_path).resolve(strict=True)
    plan = read(plan_path)
    pair.numerical.authorization(plan, 900)
    require(plan['variant_order'] == list(pair.VARIANTS) and variant in pair.VARIANTS and
            plan['runtime_environment'] == {'GLIMMER_EXECUTION_HOST_RANGE_SEEK': '1'},
            'fixed host pair counter contract required')
    review = launch_review(plan, variant)
    for name, expected in review['source_sha256'].items():
        require(digest(name) == expected, 'reviewed counter source changed: ' + name)
    reference = Path(plan['numerical_reference']).resolve(strict=True)
    numerical_plan = read(reference / 'plan.json')
    pair.require_launch_review(numerical_plan)
    paths = pair.validate_plan(numerical_plan)
    require(plan['runtime_manifest'] == numerical_plan['runtime_manifest'] and
            plan['runtime_manifest_sha256'] == numerical_plan['runtime_manifest_sha256'],
            'counter and numerical runtime pair differ')
    selected = pair.selected_plan(numerical_plan, variant)
    report = read(reference / 'report.json')
    binding = dict(plan=str(plan_path), plan_sha256=digest(plan_path), variant=variant,
                   runtime_manifest=plan['runtime_manifest'],
                   runtime_manifest_sha256=plan['runtime_manifest_sha256'],
                   numerical_report=str(reference / 'report.json'),
                   numerical_report_sha256=plan['numerical_report_sha256'],
                   numerical_review=str(reference / 'review.json'),
                   numerical_review_sha256=plan['numerical_review_sha256'],
                   server=selected['server'], frontend=selected['frontend'],
                   native_reference=str(Path(report['variant_attempts'][variant]) / 'native'))
    check_binding(binding)
    require(report['status'] == 'fresh_o0_o2_native_equal_numerical_requires_review' and
            report['input_identity_unchanged'] is True and
            read(reference / 'inputs-before.json') == read(reference / 'inputs-after.json'),
            'fresh complete numerical prerequisite required')
    rows = [row for row in report['jobs'] if row['variant'] == variant and row['mode'] == 'native']
    require(len(rows) == 1, 'one own-variant native prerequisite required')
    row = rows[0]
    require(row['worker']['status'] == 'passed_requires_review' and row['worker']['reparse'] == pair.ORACLE
            and row['unit']['returncode'] == 0 and row['unit']['unit_removed'] is True,
            'native numerical prerequisite incomplete')
    job = row['worker']['job']
    native = Path(binding['native_reference'])
    require(read(native / 'attempt.json') == job, 'raw native receipt differs')
    pair.validate_job(selected, job)
    paths.update(reference / name for name in ('plan.json', 'report.json', 'review.json',
                                               'inputs-before.json', 'inputs-after.json'))
    paths.update((plan_path, Path(plan['launch_review']), native / 'attempt.json', Path(__file__).resolve(),
                  Path(pair.__file__).resolve(), Path(pair.numerical.__file__).resolve()))
    paths.update(Path(name) for name in review['source_sha256'])
    if variant == 'o2':
        paths.update(Path(review['predecessor']['attempt']) / name for name in ('report.json', 'review.json'))
    for name, expected in job['artifact_identities'].items():
        path = native / name
        require(path.resolve(strict=True).is_relative_to(native) and digest(path) == expected,
                'native reference artifact changed')
        paths.add(path)
    return binding, paths, job


def configure(binding):
    """One helper instance with explicit immutable binaries and existing seek policy."""
    check_binding(binding)
    return pair.numerical.configure(dict(server=binding['server'], frontend=binding['frontend'],
                                        runtime_environment={'GLIMMER_EXECUTION_HOST_RANGE_SEEK': '1'}))


def validate_job(binding, job):
    """Retain the full numerical job contract with only the exact target range added."""
    check_binding(binding)
    require(job['mode'] == 'equal', 'replacement counter path required')
    command = list(job['server_command'])
    low = job['application_identities']['low']
    for option, expected in (
            ('--audit-sgemm-target-client', f"{low['pid']}:{low['start']}:3127"),
            ('--audit-sgemm-range-library', str(Path('/usr/local/cuda/lib64/libnvToolsExt.so.1').resolve(strict=True)))):
        require(command.count(option) == 1, 'one exact counter selector required')
        index = command.index(option)
        require(index + 1 < len(command) and command[index + 1] == expected, 'counter selector differs')
        del command[index:index + 2]
    pair.validate_job(dict(server=binding['server'], frontend=binding['frontend']),
                      dict(job, server_command=command))


def claim_launch(plan_path, variant):
    """A failed attempt remains claimed; this entry point never silently retries it."""
    plan_path = Path(plan_path).resolve(strict=True)
    launch_review(read(plan_path), variant)
    path = plan_path.with_name('ncu-' + variant + '-launch.json')
    with path.open('x') as stream:
        json.dump(dict(plan=str(Path(plan_path).resolve()), plan_sha256=digest(plan_path),
                       variant=variant), stream)
        stream.write('\n')
