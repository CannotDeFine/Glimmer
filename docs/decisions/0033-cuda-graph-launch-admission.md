# ADR 0033: Admit CUDA graph launches at the graph boundary

- Status: Accepted
- Date: 2026-09-03
- Deciders: Glimmer maintainers

## Context

CUDA frameworks increasingly capture work into an executable CUDA Graph and
submit it with `cuGraphLaunch` or `cudaGraphLaunch`. A scheduler that only
intercepts individual kernel-launch entry points misses this submission path:
the graph can bypass transparent priority and concurrency admission even though
the application still uses the same CUDA stream.

Graph execution is opaque at the launch ABI. The call exposes an executable
graph and a stream, but not the number, order, or resource usage of its
internal nodes. Graph memory allocation nodes also have lifetimes that span
capture, launch, update, auto-free, and process teardown.

## Decision

Glimmer intercepts the Driver and Runtime graph-launch families:

- `cuGraphLaunch` and `cuGraphLaunch_ptsz`;
- `cudaGraphLaunch` and `cudaGraphLaunch_ptsz`.

Each successful graph launch is one scheduling task. In enforce mode it uses
the same local or authenticated remote `LaunchGate` as a kernel launch, records
one internal non-timing Driver event on the supplied stream, and releases the
lease only after that event completes. Batching applies at the call boundary in
the same way as other covered launches. Observe mode emits an allocation-free
graph-specific diagnostic; timing diagnostics use the existing launch timing
schema.

The interceptor does not inspect, rewrite, split, or preempt graph nodes. Graph
memory allocation nodes remain fail-closed while quota mode is enabled; quota-
disabled mode preserves native behavior. Cooperative launches and device-side
graph launches remain outside this decision.

## Consequences

Positive:

- Frameworks that submit captured graphs receive the same transparent admission
  and priority boundary as direct kernel launches.
- Completion tracking remains stream ordered and does not move CUDA handles
  between processes.
- The graph path has an explicit diagnostic instead of being mislabeled as a
  single kernel launch.

Trade-offs and limits:

- One graph consumes one scheduler slot even if it contains many kernels; this
  is intentionally conservative because node-level preemption is unavailable.
- A graph can still allocate through unsupported graph memory nodes only when
  quota mode is disabled. Supporting those nodes requires a separate lifetime
  and accounting design.
- Graph launch support does not claim cooperative-launch or device-side launch
  coverage.

## Verification

- Fake Driver and Runtime preload tests resolve direct, `dlsym`, and
  `cuGetProcAddress` graph-launch symbols and exercise both PTDS aliases.
- Observe and enforce tests verify graph diagnostics, admission, and completion
  behavior without changing existing kernel-launch coverage.
- CUDA GPU tests remain the compatibility check for a real graph workload when
  a graph-capable toolkit and driver are available.
