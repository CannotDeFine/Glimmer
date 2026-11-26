"""Reject unrelated failures before the explicitly partial offline audit."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import audit_bert_overhead_fragment as audit


class Tests(unittest.TestCase):
    def fixture(self):
        owners = dict(low=dict(pid=1, start=11), high=dict(pid=2, start=12))
        service = dict(pid=3, start=13)
        job = dict(status='failed', cleanup_errors=['timeline clock: lost wait records or probe error'],
                   reference_before={'reference': 1}, reference_after={'reference': 1}, socket_retained=False,
                   application_identities=owners, server_identity=service,
                   final_returncodes=dict(low=0, high=0, service=0), artifact_identities={'profile': 'hash'},
                   authenticated_child_cleanup={name: dict(owner, exit_confirmed=True, signals=[])
                                                for name, owner in dict(owners, service=service).items()})
        documents = {'report.json': dict(status='failed', input_identity_unchanged=True),
                     'inputs-before.json': {'input': 'sealed'}, 'inputs-after.json': {'input': 'sealed'},
                     'equal-unit.json': dict(unit_removed=True), 'equal-worker.json': dict(job=job),
                     'attempt.json': job}
        return documents, job

    def test_valid_failure_remains_failed(self):
        documents, job = self.fixture()
        with patch.object(audit, 'read', side_effect=lambda p: documents[p.name]), \
                patch.object(audit, 'digest', return_value='hash'):
            result, _ = audit.validate_failed_capture(Path('/source'))
        self.assertIs(result, job)
        self.assertEqual(job['status'], 'failed')
        self.assertEqual(documents['report.json']['status'], 'failed')

    def test_unrelated_error_cleanup_signal_or_changed_bytes_rejected(self):
        for fault in ('error', 'cleanup', 'signal', 'exit', 'seals', 'bytes', 'unit'):
            documents, job = self.fixture()
            if fault == 'error':
                job['error'] = 'numerical failure'
            elif fault == 'cleanup':
                job['cleanup_errors'].append('another failure')
            elif fault == 'signal':
                job['authenticated_child_cleanup']['service']['signals'] = [15]
            elif fault == 'exit':
                job['final_returncodes']['low'] = 1
            elif fault == 'seals':
                documents['inputs-after.json'] = {'different': 1}
            elif fault == 'unit':
                documents['equal-unit.json']['unit_removed'] = False
            with self.subTest(fault=fault), \
                    patch.object(audit, 'read', side_effect=lambda p: documents[p.name]), \
                    patch.object(audit, 'digest', return_value='changed' if fault == 'bytes' else 'hash'), \
                    self.assertRaises(ValueError):
                audit.validate_failed_capture(Path('/source'))


if __name__ == '__main__':
    unittest.main()
