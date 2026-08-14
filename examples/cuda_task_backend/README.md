# Explicit CUDA task backend

This example demonstrates the opt-in task-boundary path. It loads a small PTX
kernel with the CUDA Driver API, submits two tasks through
`CudaTaskController`, and waits for the backend-owned CUDA completion events.
The scheduler reserves one MiB per task from an eight MiB quota by default and
dispatches one task at a time in the default configuration.

Build it with the CUDA workload preset:

```sh
cmake --preset cuda-gpu
cmake --build --preset cuda-gpu --target glimmer_cuda_task_backend_demo -j2
```

Run it without the preload interceptor:

```sh
./build/cuda-gpu/examples/cuda_task_backend/glimmer_cuda_task_backend_demo
```

Expected output:

```text
explicit_cuda_task_demo status=ok tasks=2 completed=2 quota_bytes=8388608
```

This is an explicit integration example, not a transparent application mode.
The caller must keep kernel argument storage, the CUDA module, the function,
and any referenced stream or allocation alive until the controller reports a
terminal task state. The preload interceptor does not call this controller or
queue arbitrary application launches.

The directory also builds `glimmer_cuda_multi_tenant_demo`. It submits three
tasks for `tenant-a` with weight 2 and two tasks for `tenant-b` with weight 1
through `control::TaskAdmissionService`. The real CUDA backend must complete
them in this weighted round-robin order:

```text
tenant-a1,tenant-a2,tenant-b1,tenant-a3,tenant-b2
```

Run it with:

```sh
./build/cuda-gpu/examples/cuda_task_backend/glimmer_cuda_multi_tenant_demo
```

The directory also contains `glimmer_cuda_lease_worker`, a real worker for the
remote control-service mode. Start the service in one shell, submit a task in a
second shell, and run the worker in a third shell:

```sh
./build/cuda-gpu/bin/glimmer_control_service \
    --socket /tmp/glimmer-control.sock \
    --quota-bytes 8388608 \
    --execution-mode remote \
    --lease-timeout-ms 5000 \
    --max-concurrent-tasks 2 \
    --max-queued-tasks 64 \
    --scheduler-policy weighted_rr \
    --bind-leases-to-process
./build/cuda-gpu/bin/glimmer_control_client \
    --socket /tmp/glimmer-control.sock submit tenant-a 1048576 1 1
./build/cuda-gpu/examples/cuda_task_backend/glimmer_cuda_lease_worker \
    --socket /tmp/glimmer-control.sock \
    --heartbeat-interval-ms 1000
```

The worker creates its own CUDA context, allocates the leased number of bytes,
launches a small no-op Driver-API PTX kernel, synchronizes, frees the local
allocation, and reports `COMPLETE`. While the task is running it sends
`HEARTBEAT` requests at the configured interval. If any CUDA or heartbeat step
fails it reports `FAIL`. Set the interval to `0` only when the service lease
timeout is disabled; set the service's `--lease-timeout-ms` to `0` to disable
lease expiry. The `--execution-delay-ms` option is a deterministic
demonstration/testing aid for exercising heartbeats with a long-running task.
The wire protocol carries only logical task metadata; CUDA handles and device
pointers never cross the socket.

`--max-concurrent-tasks` controls how many workers may hold a running lease at
once. It defaults to `1`; increase it only when the GPU workload and quota are
intended to overlap. Each worker must use a distinct process and claims are
serialized by the service.

The service supports `weighted_rr` (the default) and `fifo` scheduling
policies. Both apply at explicit task boundaries; they do not preempt a CUDA
kernel after submission.

With `--bind-leases-to-process`, the worker that claims a lease must also send
its heartbeats and terminal report. The service authenticates this ownership
with the Unix peer PID, UID, and process start-time identity.

`--max-queued-tasks` provides optional admission backpressure for waiting work.
When the bound is reached, the service returns `QUEUE_FULL` without reserving
GPU quota; its default value is unlimited.

The standalone client can inspect the service without changing task state:

```sh
./build/cuda-gpu/bin/glimmer_control_client \
    --socket /tmp/glimmer-control.sock stats
```

The response contains task-state counts, quota bytes, and configured scheduler
limits. It is a bounded diagnostic snapshot rather than a persistent metrics
export.
