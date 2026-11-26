"""Literal source/RPC/owner clock and authentication controls; no GPU imports."""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))
from execution_copy_audit import first_post_release_copy, parse_source_copies, parse_owner_copies
from semantic_ordering_fixture import SemanticTrace


def fixture(two_threads=False):
    source = ("Glimmer execution copy audit: v=1, pid=811, serial=3, api=4, bytes=1024, entry_ns=100, "
              "locked_ns=110, finish_ns=300, cpu_entry_ns=10, cpu_finish_ns=30, has_result=1, status=0, calls=2, dropped=0\n"
              "Glimmer execution copy rpc: v=1, pid=811, serial=3, call=1, operation=22, session=1, request=6, "
              "begin_ns=120, submitted_ns=130, done_ns=180, has_reply=1, status=0\n"
              "Glimmer execution copy rpc: v=1, pid=811, serial=3, call=2, operation=25, session=1, request=7, "
              "begin_ns=200, submitted_ns=210, done_ns=290, has_reply=1, status=0\n")
    trace = SemanticTrace(actor=False)
    for request, operation, frame, owner, reply in ((6, 22, 140, 150, 170), (7, 25, 220, 240, 280)):
        if operation == 25:
            trace.append("Native admission audit: session=1, sequence=14, accepted=true, operation=25\n"
                         "Priority execution audit: event=low_issued, session=1, priority=0, sequence=14, "
                         "completed_blocks=0, total_blocks=0, physical_launches=0, status=0\n")
        trace.append(f"Copy request audit: v=1, connection=1, session=1, request={request}, "
                     f"command={14 if operation == 25 else 0}, pid=811, "
                     f"operation={operation}, frame_ns={frame}, owner_ns={owner}, reply_queued_ns={reply}, "
                     "status=0; gpu_completion_qualified=false\n")
    if two_threads:
        # Earlier API entrance does not imply earlier submission under contention.
        source += ("Glimmer execution copy audit: v=1, pid=811, serial=4, api=4, bytes=1024, entry_ns=90, "
                   "locked_ns=190, finish_ns=400, cpu_entry_ns=20, cpu_finish_ns=40, has_result=1, status=0, calls=1, dropped=0\n"
                   "Glimmer execution copy rpc: v=1, pid=811, serial=4, call=1, operation=25, session=1, request=9, "
                   "begin_ns=300, submitted_ns=310, done_ns=390, has_reply=1, status=0\n")
        trace.append("Native admission audit: session=1, sequence=15, accepted=true, operation=25\n"
                     "Priority execution audit: event=low_issued, session=1, priority=0, sequence=15, "
                     "completed_blocks=0, total_blocks=0, physical_launches=0, status=0\n"
                     "Copy request audit: v=1, connection=1, session=1, request=9, command=15, pid=811, "
                     "operation=25, frame_ns=320, owner_ns=340, reply_queued_ns=380, "
                     "status=0; gpu_completion_qualified=false\n")
    return source, trace.finish()


class CopyAudit(unittest.TestCase):
    def test_generic_admission_is_not_a_fabricated_copy_clock(self):
        source, owner = fixture()
        fake_source = source.replace("operation=25", "operation=141")
        fake_owner = owner.replace("operation=25", "operation=141")
        mixed_source = source.replace("calls=2", "calls=3") + (
            "Glimmer execution copy rpc: v=1, pid=811, serial=3, call=3, operation=141, "
            "session=1, request=8, begin_ns=291, submitted_ns=292, done_ns=295, "
            "has_reply=1, status=0\n")
        for changed in (fake_source, mixed_source):
            with self.subTest(source=changed), self.assertRaisesRegex(ValueError, "timing scope"):
                parse_source_copies(changed, 811)
        with self.assertRaises(ValueError):
            parse_owner_copies(fake_owner)
        for client, server in ((fake_source, owner), (source, fake_owner), (fake_source, fake_owner)):
            with self.subTest(client=client), self.assertRaises(ValueError):
                first_post_release_copy(client, server, 811, 80, 90)
        # A real ordinary141 admission/issue elsewhere in the service log is
        # recognized as memory, but cannot add an uninstrumented RPC or clock.
        generic = ("Native admission audit: session=1, sequence=15, accepted=true, operation=141\n"
                   "Priority execution audit: event=low_issued, session=1, priority=0, sequence=15, "
                   "completed_blocks=0, total_blocks=0, physical_launches=0, status=0\n")
        end = owner.rfind("Execution semantic audit:")
        self.assertGreater(end, 0)
        self.assertEqual(first_post_release_copy(source, owner[:end] + generic + owner[end:],
                                                811, 80, 90),
                         first_post_release_copy(source, owner, 811, 80, 90))

    def test_exact_join_preserves_release_bracket_and_cpu_scope(self):
        source, owner = fixture()
        result = first_post_release_copy(source, owner, 811, 80, 90)
        self.assertEqual(result["release_to_api_entry_lower_ns"], 10)
        self.assertEqual(result["source_api_wall_ns"], 200)
        self.assertEqual(result["source_api_thread_cpu_ns"], 20)
        self.assertEqual(result["joined_copy_rpcs"][0]["owner_to_reply_queue_ns"], 20)
        self.assertFalse(result["gpu_completion_qualified"])
        self.assertEqual(first_post_release_copy(source, owner, 811, 99, 105)["release_to_api_entry_lower_ns"], 0)

    def test_every_incomplete_foreign_or_inconsistent_copy_rejects(self):
        source, owner = fixture()
        for old, new in (("pid=811", "pid=812"), ("calls=2", "calls=3"), ("dropped=0", "dropped=1"),
                         ("locked_ns=110", "locked_ns=99"), ("finish_ns=300", "finish_ns=170"),
                         ("call=2", "call=1"), ("submitted_ns=210", "submitted_ns=250"),
                         ("has_result=1", "has_result=0"), ("has_reply=1", "has_reply=0"),
                         ("serial=3", "serial=0"), ("status=0", "status=701")):
            with self.subTest(old=old), self.assertRaises(ValueError):
                first_post_release_copy(source.replace(old, new), owner, 811, 80, 90)
        for changed in (source.rstrip(), source + source, source.replace("bytes=1024", "bytes=18446744073709551616")):
            with self.assertRaises(ValueError):
                parse_source_copies(changed, 811)

    def test_owner_frame_binding_and_queued_reply_are_not_completion(self):
        source, owner = fixture()
        for old, new in (("pid=811", "pid=812"), ("session=1", "session=2"),
                         ("frame_ns=220", "frame_ns=209"), ("owner_ns=240", "owner_ns=219"),
                         ("reply_queued_ns=280", "reply_queued_ns=291"), ("request=7", "request=8"),
                         ("command=14", "command=0"), ("command=14", "command=15"),
                         ("operation=25", "operation=23"), ("qualified=false", "qualified=true")):
            with self.subTest(old=old), self.assertRaises(ValueError):
                first_post_release_copy(source, owner.replace(old, new), 811, 80, 90)
        with self.assertRaises(ValueError):
            first_post_release_copy(source, owner, 811, 301, 310)

    def test_actual_gpu_issue_selects_copy_not_api_entry_or_completion(self):
        source, owner = fixture(two_threads=True)
        self.assertEqual(parse_source_copies(source, 811)[0]["serial"], 4)
        self.assertEqual(first_post_release_copy(source, owner, 811, 80, 90)["api"]["serial"], 3)
        issue = ("Priority execution audit: event=low_issued, session=1, priority=0, sequence=14, "
                 "completed_blocks=0, total_blocks=0, physical_launches=0, status=0\n")
        for changed in (owner.replace(issue, ""), owner.replace(issue, issue + issue)):
            with self.assertRaises(ValueError):
                first_post_release_copy(source, changed, 811, 80, 90)


if __name__ == "__main__":
    unittest.main()
