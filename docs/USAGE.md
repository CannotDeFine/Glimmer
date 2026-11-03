# Using Glimmer

[Project home](../README.md) / [Documentation](README.md) / Usage

The launch-admission library schedules covered CUDA launches while the
application retains its CUDA context and performs its own execution. Follow the
[quick start](../README.md#get-started) to build it, start a control service and
launch an application. Check [CUDA compatibility](CUDA_API_COVERAGE.md) for
the scope and limitations of interception.

## Scheduling

The service controls cross-process admission. Its `--scheduler-policy` accepts
`fifo`, `weighted_rr`, `drr` and `priority`. Under `priority`, larger values of
`GLIMMER_SCHEDULER_PRIORITY` take precedence among queued work. This does not
interrupt an already-running kernel or guarantee a latency target.

`--max-concurrent-tasks` limits running leases across clients;
`GLIMMER_MAX_CONCURRENT_KERNELS` controls the application's launch gate. These
limits are not GPU partitions or utilization guarantees. Tenant names identify
scheduling work, not security identities. With process-bound leases, the
submitting process must claim and complete its own work.

## Allocation limits

The service's `--quota-bytes` limits its task reservations; it does not by
itself limit CUDA allocations in applications. To set a process-local limit
for covered allocations, add `GLIMMER_MEMORY_LIMIT_BYTES` to the application's
environment. A software limit is not hardware isolation and cannot account for
allocation paths that are not intercepted.

## Shutdown

Allow applications to finish before stopping the service. The control client
can inspect service state with `stats`. Stop the service with Ctrl+C, then use
`rmdir` to remove the empty private socket directory. Investigate a directory
that is not empty rather than removing it blindly.
