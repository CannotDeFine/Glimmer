"""Independent numeric semantic-log construction for offline verifier tests.

These CPU fixtures describe protocol lifecycles, not native execution or GPU
qualification. Keep the literal schema independent of the production parser.
"""

import json


ACTOR_FIELDS = """
actor_serial event subject action binding id peer priority caller0 caller1 group
sequence ordinary_sequence cleanup_parent origin_source reservation generation
cutoff input_reservation input_generation input_cutoff native_sequence
native_operation native_kind ordinary_opcode connection wire_sequence attempt
source stream capture command_sequence related index count allocation ticket depth
stage previous_stage phase outcome disposition host_origin status carrier
original_status async_status native_status completion_status library_status
accepted sequence_consumed has_original has_async has_library has_host
has_output_written output_written reply_ready reply_released enqueue_complete
physical_complete has_physical_work open_group unknown retired bound consumed active_call
ready_continuation caller_primary_domain has_context before_kind before_identity
before_context before_device after_kind after_identity after_context after_device
communicator_domain communicator_context communicator_object output_contract
""".split()


class SemanticTrace:
    """Append typed records and legacy text in a single deterministic owner order."""

    def __init__(self, *, actor=True, source=1, peer=11, priority=0):
        self.parts = []
        self.serial = 0
        self.actor_serial = 0
        self.public_sequence = 0
        self.native_sequence = 0
        self.caller_sequences = {}
        self.lane_sequences = {}
        self.cutoff = 0
        self.depth = 0
        self.has_actor = actor
        self.peer_id = peer
        self.priority = priority
        self.pending = {}
        self.batches = {}
        self.source_attachments = {}
        self.progress_grants = {}
        self.peer_sources = {}
        self.communicator_inputs = {}
        self.runtime(1, pid=700, incarnation0=3, incarnation1=4)
        self.peer(source, peer=peer, priority=priority)
        if actor:
            self.actor(1)

    @property
    def position(self):
        return sum(map(len, self.parts))

    def append(self, text):
        self.parts.append(text)

    def emit(self, family, **fields):
        self.serial += 1
        row = dict(v=1, service=37, serial=self.serial, family=family)
        row.update(fields)
        self.append("Execution semantic audit: " + json.dumps(row, separators=(",", ":")) + "\n")
        return self.serial

    def runtime(self, event, **fields):
        return self.emit(2, event=event, **fields)

    def peer(self, connection, *, peer=11, priority=0, **fields):
        row = dict(event=1, connection=connection, peer=peer, pid=800 + peer,
                   start=900 + peer, uid=1000, priority=priority, role=0,
                   policy=11, kernel_mode=2, effective=1, retained=1)
        row.update(fields)
        if row["role"] == 0:
            self.peer_sources.setdefault(peer, connection)
        return self.emit(4, **row)

    def actor(self, event, **fields):
        row = dict.fromkeys(ACTOR_FIELDS, 0)
        self.actor_serial += 1
        row.update(subject=1, event=event, actor_serial=self.actor_serial)
        row.update(fields)
        return self.emit(1, **row)

    def ordering(self, event, command=None, **fields):
        row = dict(connection=1, priority=0, sequence=10, opcode=48, capture=101,
                   event_source=1, event_object=51, record_sequence=10,
                   event_capture=101, status=0)
        if command is not None:
            row.update(command)
        row.update(fields)
        self.emit(3, event=event, **row)
        return row

    def admit_native(self, kind=9, *, reservation=9, group=0, communicator=None):
        self.public_sequence += 1
        self.cutoff += 1
        sequence = self.caller_sequences.get(self.peer_id, 0) + 1
        self.caller_sequences[self.peer_id] = sequence
        key = dict(subject=3, id=self.public_sequence, peer=self.peer_id,
                   priority=self.priority, caller0=7, caller1=8, group=group,
                   sequence=sequence, reservation=reservation,
                   generation=1, cutoff=self.cutoff, native_kind=kind, stage=1,
                   output_contract=2)
        default_object = (100000 + self.public_sequence if kind in (2, 3)
                          else 1000 + self.peer_id)
        self.communicator_inputs[key["id"]] = ((1, 1, default_object) if communicator is None
                                               else communicator)
        self.actor(2, **key, action=1, accepted=1, sequence_consumed=1)
        self.batches[reservation] = self.peer_id, self.priority
        return key

    def grant(self, key, *, mode=1, source=0, leaf=None, capture=0, path=()):
        if mode == 2:
            self.attach_source(key, source)
        grant = self.runtime(
            10, policy=11, peer=key["peer"], priority=key["priority"],
            reservation=key["reservation"], generation=key["generation"],
            cutoff=key["cutoff"], target_kind=1, target=key["id"], action=5,
            mode=mode, root_source=source, leaf_source=source if leaf is None else leaf,
            capture=capture, edges=len(path), path_kind=0)
        for index, edge in enumerate(path):
            self.runtime(11, grant=grant, index=index, **edge)
        # A selected foreign leaf uses the exact backend frontier directly. Only
        # zero-edge Request progress actually enters the actor Observe call.
        if mode == 2 and not path:
            inputs = dict(input_reservation=key["reservation"], input_generation=key["generation"],
                          input_cutoff=key["cutoff"])
            attachment = self.source_attachments[key["id"]]
            observed = dict(key, source=attachment["source"], capture=attachment["capture"])
            self.actor(3, **observed, **inputs, action=8)
            self.progress_grants[grant] = observed, inputs
        return grant

    def attach_source(self, key, source):
        """Admit the real source turn under the initial native execution grant.

        Its blocked return changes the request cutoff before a separate source
        progress grant. This is not an unbound Prepared request permission.
        """
        if key["id"] in self.source_attachments:
            if self.source_attachments[key["id"]]["source"] != source:
                raise ValueError("fixture source cannot change after initial turn admission")
            return
        grant = self.grant(key)
        inputs = dict(input_reservation=key["reservation"], input_generation=key["generation"],
                      input_cutoff=key["cutoff"])
        self.actor(3, **key, **inputs, action=2)
        attachment = dict(source=source, capture=10000 + key["id"], stream=0,
                          index=0, count=1)
        self.actor(7, **key, **attachment, action=1, binding=3)
        self.source_attachments[key["id"]] = dict(attachment, reservation=key["reservation"],
                                                 finished=False)
        previous = key["stage"]
        self.cutoff += 1
        key.update(stage=2, cutoff=self.cutoff)
        self.actor(5, **key, previous_stage=previous)
        self.actor(4, **key, **inputs, action=2, status=600)
        self.result(grant, status=600)
        self.batch(key["reservation"])

    def result(self, grant, **fields):
        row = dict(status=0, issue_state=0, command_sequence=0, native_status=0,
                   completion_status=0, has_library=0, library_status=0, source=0)
        row.update(fields)
        if grant in self.progress_grants:
            key, inputs = self.progress_grants.pop(grant)
            self.actor(4, **key, **inputs, action=8, status=row["status"],
                       outcome=row["issue_state"] + 1, command_sequence=row["command_sequence"],
                       native_status=row["native_status"], completion_status=row["completion_status"],
                       has_library=row["has_library"], library_status=row["library_status"])
        return self.runtime(12, grant=grant, **row)

    def source_issue(self, key, source, sequence, *, root=None, capture=0, path=()):
        grant = self.grant(key, mode=2, source=source if root is None else root,
                           leaf=source, capture=capture, path=path)
        self.result(grant, issue_state=1, command_sequence=sequence, source=source)
        return grant

    def run_native(self, key, *, physical=False):
        if key["native_kind"] in (9, 10, 11):
            self.attach_source(key, self.peer_sources[key["peer"]])
        grant = self.grant(key)
        inputs = dict(input_reservation=key["reservation"], input_generation=key["generation"],
                      input_cutoff=key["cutoff"])
        self.actor(3, **key, **inputs, action=2)
        source = self.source_attachments.get(key["id"])
        attachment = ({name: source[name] for name in
                       ("source", "capture", "stream", "index", "count")}
                      if source is not None else None)
        if attachment is not None:
            self.actor(7, **key, **attachment, action=2, binding=3)
        self.native_sequence += 1
        lane_sequence = self.lane_sequences.get(key["peer"], 0) + 1
        self.lane_sequences[key["peer"]] = lane_sequence
        key.update(native_sequence=lane_sequence, native_operation=self.native_sequence)
        if key["native_kind"] in (2, 3, 4, 5, 6, 9, 10, 11):
            domain, context, communicator = self.communicator_inputs[key["id"]]
            key.update(communicator_domain=domain, communicator_context=context,
                       communicator_object=communicator)
            if key["native_kind"] in (2, 3):
                key["allocation"] = communicator
        self.actor(8, **key)
        self.cutoff += 1
        previous = key["stage"]
        key.update(stage=3, cutoff=self.cutoff)
        self.actor(5, **key, previous_stage=previous)
        if attachment is not None:
            self.actor(7, **key, **attachment, action=3, binding=3)
            source["finished"] = True
        self.actor(4, **key, **inputs, action=2)
        self.result(grant)
        key.update(has_original=1, original_status=0, phase=5 if physical else 3,
                   physical_complete=int(physical), enqueue_complete=int(physical),
                   has_context=1, before_kind=1, after_kind=1)
        self.actor(9, **dict(key, stage=0, cutoff=0))
        if key["native_kind"] in (7, 8):
            old_depth = self.depth
            self.depth += 1 if key["native_kind"] == 7 else -1
            self.actor(10, **key, index=old_depth, depth=self.depth)
        self.cutoff += 1
        key.update(stage=9 if physical else 5, cutoff=self.cutoff)
        self.actor(5, **key, previous_stage=3)
        key["reply_ready"] = 1
        self.actor(11, **key, action=9)
        key["reply_released"] = 1
        self.actor(12, **key, action=10)
        if physical:
            self.actor(13, **key, action=11)
            self.actor(13, **key, action=12)
        else:
            self.pending[key["id"]] = key
        self.batch(key["reservation"])
        return key

    def native(self, kind=9, *, physical=False, reservation=9, group=0, communicator=None):
        return self.run_native(self.admit_native(kind, reservation=reservation, group=group,
                                                communicator=communicator),
                               physical=physical)

    def batch(self, reservation=9, *, remove=False, **fields):
        members = [key for key in self.pending.values() if key["reservation"] == reservation]
        source_active = any(source["reservation"] == reservation and not source["finished"]
                            for source in self.source_attachments.values())
        peer, priority = self.batches[reservation]
        row = dict(subject=2, id=reservation, peer=peer, priority=priority,
                   reservation=reservation, generation=1, open_group=int(self.depth > 0),
                   has_physical_work=int(self.depth > 0 or bool(members) or source_active),
                   count=len(members))
        if remove:
            row.update(action=12, retired=1)
        row.update(fields)
        self.actor(16, **row)
        if remove:
            del self.batches[reservation]

    def finish_native(self, key):
        key.update(physical_complete=1, enqueue_complete=1, phase=5)
        self.actor(9, **dict(key, stage=0, cutoff=0, reply_released=0))
        self.cutoff += 1
        previous = key["stage"]
        key.update(stage=9, cutoff=self.cutoff)
        self.actor(5, **key, previous_stage=previous)
        self.actor(13, **key, action=11)
        self.actor(13, **key, action=12)
        del self.pending[key["id"]]
        self.batch(key["reservation"])

    def finish(self):
        if self.has_actor:
            for reservation in sorted(self.batches):
                self.batch(reservation, remove=True)
            self.actor(18)
            self.actor(19, retired=1)
        self.runtime(2)
        return "".join(self.parts)
