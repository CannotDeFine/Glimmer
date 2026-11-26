# Architecture

[Project home](../README.md) / [Documentation](../docs/README.md) / Architecture

Glimmer separates scheduling policy from CUDA execution and application-facing
interception. The [build guide](BUILDING.md) identifies the available components;
[Usage](USAGE.md) describes their service and application configuration.

## Runtime flow

With launch admission, the application retains its GPU resources and execution:

```text
Application -> launch interceptor -> control service grants a lease
                    |
                    +-> native CUDA launch -> completion event -> release lease
```

With centralized execution, supported operations cross a process boundary:

```text
Application -> execution frontend -> authenticated local transport
                                           |
                                           v
                                  execution service scheduler
                                           |
                                           v
                                  CUDA resources and execution
```

The frontend represents application-visible objects while the execution service
owns their native counterparts. A request being accepted is not proof of GPU
completion; dependent work and resource retirement must respect actual progress.
If execution control submits independent asynchronous device work, its exact
completion must also be established before the associated resources are reused.

## Components

| Component | Responsibility |
| --- | --- |
| `src/core/` | Host-side scheduling policy, tenant/task state and resource accounting. |
| `src/control/` | Admission and execution contracts, local transport and client/service coordination. |
| `src/backend/` | Execution contracts and simulated execution for hardware-independent tests. |
| `src/backend/cuda/` | Optional CUDA-specific resource, execution and completion handling. |
| `src/interceptor/` | Optional process-local CUDA interception and launch admission. |
| `src/interceptor/execution/` | Optional frontend that sends supported operations to the centralized execution service. |
| `src/execution_service/` | Authenticated client handling and service-owned CUDA execution. |
| `src/transform/` | Preparation and transformation of eligible kernel inputs. |
| `examples/`, `tests/` | Workloads, correctness fixtures and integration checks, not scheduling authority. |

## Execution boundaries

The host control service and client are available without CUDA. They exercise
admission, scheduling and task lifecycles; the bootstrap `glimmer` executable is
not a complete GPU application launcher.

The launch-admission interceptor keeps CUDA execution in the application
process. It adds supported accounting and scheduling boundaries. Admission
order alone cannot preempt an already-running kernel.

The centralized execution path uses a different preload frontend and service.
The service owns its native GPU resources, while the frontend preserves
supported application-facing operations through the control protocol. Client
authentication, object identity and physical completion are separate from host
request acceptance. The two preload libraries are not interchangeable.

The transport can inspect a complete authenticated request without transferring
ownership; a borrowed view expires at the next non-const server call, and an
exact transfer rechecks frame identity. Inspection and diagnostics do not make
scheduling decisions or grant native execution or physical-completion authority.
Execution transport is externally serialized and owned by its constructing
process. Every data chunk is authenticated, and each transferred descriptor
reference has one owner. A fork child must not use the inherited endpoint and must
promptly exec or `_exit`. A lost reply leaves delivery uncertain and does not
authorize replay.

The native epoch codec only encodes and validates bounded, versioned messages;
it does not authenticate peers, grant execution, retry requests or establish
GPU completion.

The generation-bound task codec has a separate C++ interface and wire envelope
from the original task codec. Decoding a message is not peer authentication,
execution authorization or evidence of GPU completion.

The execution admission service owns its scheduler and lease state. Its
cooperative and native epoch transitions require authenticated identities;
an expired running execution is not proof of physical completion. The original
admission interface retains its existing ownership and ABI.
The service owner must stop and join backend work and retire leases before
destruction; destroying the service does not stop a GPU or fence a context.
If poisoned private accounting cannot be retired individually, the owner may
destroy the entire non-reusable service only after all callers and backend work
are joined, physical execution is stopped and external resources are retired.
Cooperative admission is marked before publication; ordinary task operations
cannot claim it, and a generation grant is issued at most once per dispatch.

The execution endpoint owns its admission service and routes each complete line
by its exact protocol envelope. A response that cannot be built before publication
retires newly issued cooperative or native work and rolls back its resource binding
only when non-issue is known. An uncertain transport delivery is not retried or
treated as proof of non-issue. Completion observers are not resource ownership.

The shared task client uses an authenticated local channel and pins the first
service incarnation for its logical session. It does not replay a request after
an uncertain reply or fall back to another transport.
The separate versioned transport client can select socket, shared memory or
automatic negotiation. Automatic fallback is restricted to an explicit
unsupported reply on the same connection before publishing a domain request.
Shared sessions pin the service incarnation; socket sessions retire after a
lost exchange because the legacy wire does not identify an incarnation.
The detailed result distinguishes requests not published from uncertain
delivery; callers must validate replies and retain uncertain ownership.
This transport is not yet integrated into the ordinary launch-admission path.
The cooperative launch gate uses the versioned execution service for remote
admission. It serializes requests and permanently quarantines the gate if a
reply is lost, inconsistent, or cannot establish a safe native outcome. A
quarantined gate sends no further requests and does not release remote leases;
the service owner remains responsible for proving physical stop and retiring
the associated resources. This gate is not yet used by the ordinary launch path.
Its asynchronous adapter can use a bounded CPU mailbox with caller-driven
progress. It exposes quarantine before publishing an uncertain result; neither
mailbox progress nor a future establishes physical GPU completion.
The execution task server accepts domain messages only after shared-channel
negotiation and joins its connection workers before endpoint teardown. A lost
transport reply does not authorize cancellation or prove physical completion.
Endpoint shutdown prevents new grants while allowing owners to retire existing
work. Internal lease counts do not prove that external resources are released
or that GPU execution has stopped.

The host-only task execution service uses its own socket and scheduler; it
does not own CUDA work or external resources. It binds leases to the connecting
process and accepts clients with the service's effective UID. On shutdown it
refuses new grants while continuing to receive completion or cancellation
reports, and exits normally only when all task and lease counts are empty.
Clients must report completion only after physical execution has stopped.
This service does not recover tasks after a process or service restart.

The separate recovery session binds operation identities to a peer, nonce and
service incarnation. It retains bounded results until acknowledged and permits
an identical request to recover a lost reply. The session client delivers each
result at most once within the process; its caller remains responsible for
issuing a grant at most once and recording acceptance before acknowledgement.
Neither acknowledgements nor disconnects prove physical completion. Recovery
state is not durable across process or service restart, and this internal path
is not enabled by the ordinary launch-admission quick start.

Eligible transformed kernels and selected library replacements can use
cooperative execution. Unsupported or unqualified paths must not be described
as universally transformable, preemptible or transparent. Software quotas and
local client authentication do not provide hardware tenant isolation.

## Dependency and ownership rules

- Keep the scheduling core independent of CUDA, NVML, dynamic symbol lookup and
  transport implementation details.
- Keep vendor-specific calls in their owning backend or frontend. Use narrow
  control/backend contracts instead of exposing scheduler internals.
- Do not expose CUDA or third-party types through a public header without an
  explicit architecture decision.
- Track admission, issue, physical completion and resource retirement
  separately. A reply or timeout is not proof that GPU work has stopped.
- Retain uncertain resource ownership until a valid completion or containment
  boundary; do not infer successful cleanup or replay an ambiguous operation.
- Keep profiling and observation separate from runtime execution authority.

Before changing a module's responsibility, dependency direction or cross-module
contract, update this overview with the new contract and its rationale. Add tests
for observable changes under the [development lifecycle](DEVELOPMENT_LIFECYCLE.md).
