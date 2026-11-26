# Using Glimmer

[Project home](../README.md) / [Documentation](README.md) / Usage

## Choose an execution path

For shared launch scheduling without changing application source, use
`libglimmer_cuda_interceptor.so` with `glimmer_control_service`. The application
keeps its CUDA contexts, memory and kernel execution. The service grants launch
leases and orders waiting work across processes.

Centralized execution instead uses `libglimmer_execution_interceptor.so` with
`glimmer_execution_service`. GPU resources belong to that service. It is an
experimental integration with a different configuration and compatibility scope;
see [centralized execution](#centralized-execution).

Do not preload both libraries. Neither path provides hardware isolation between
untrusted users. Check [compatibility](CUDA_API_COVERAGE.md) before choosing one.

## Start shared launch scheduling

Build the CUDA launch-admission targets using the
[build guide](BUILDING.md#cuda-launch-admission-and-examples). All commands below
run from the repository root. Use terminals without inherited `GLIMMER_*`,
`LD_PRELOAD` or `LD_AUDIT` settings, and a build directory owned by your user.

Start the service in a dedicated terminal. This creates a private directory for
the socket and prints its path:

```sh
glimmer_run_dir="$(mktemp -d /tmp/glimmer.XXXXXXXX)" &&
printf 'Socket path: %s/control.sock\n' "$glimmer_run_dir" &&
env -u LD_PRELOAD -u LD_AUDIT ./build/cuda-debug/bin/glimmer_control_service \
  --socket "$glimmer_run_dir/control.sock" \
  --execution-mode remote \
  --quota-bytes 1073741824 \
  --scheduler-policy priority \
  --max-concurrent-tasks 2 \
  --max-queued-tasks 128 \
  --lease-timeout-ms 5000 \
  --bind-leases-to-process
```

Wait for the socket to exist, copy the printed path into another terminal, and
confirm that the service responds:

```sh
glimmer_socket="/replace/with/the/printed/socket/path"
./build/cuda-debug/bin/glimmer_control_client \
  --socket "$glimmer_socket" stats
```

For a separate `glimmer_task_execution_service` socket, add `--execution`
before `--socket` to query `stats` or `metrics`. Its responses use the
versioned execution protocol and are not interchangeable with the ordinary
control-service responses.

The service checks the peer UID; the private directory also limits access to the
socket path by other unprivileged users. Use a distinct socket path for each service. Do not
remove a socket belonging to a running service. `--quota-bytes` is the
task-reservation budget; it does not by itself limit CUDA allocations made by
applications.

## Run your applications

In another terminal, launch an application with a priority and tenant name:

```sh
glimmer_socket="/replace/with/the/printed/socket/path"
env -u LD_AUDIT \
  LD_PRELOAD="$PWD/build/cuda-debug/lib/libglimmer_cuda_interceptor.so" \
  GLIMMER_SCHEDULER_MODE=enforce \
  GLIMMER_SCHEDULER_CONTROL_SOCKET="$glimmer_socket" \
  GLIMMER_SCHEDULER_TENANT_ID=inference \
  GLIMMER_SCHEDULER_PRIORITY=100 \
  GLIMMER_MAX_CONCURRENT_KERNELS=2 \
  GLIMMER_SCHEDULER_BATCH_SIZE=1 \
  ./your_cuda_application --your-application-options
```

Replace the executable and arguments with your own. For a Python application,
use its interpreter and script on the last line, for example
`python3 inference.py`. Do not preload the service, compiler, package manager or
your whole shell. The `env` assignments affect only this application and its
children; framework multiprocessing behavior still needs application-specific
verification.

Start a second application in another terminal with the same socket, tenant
name `training`, and priority `10`. Both must target the same GPU to demonstrate
sharing it. A tenant label identifies scheduling work; it is not a security
identity. Process-bound leases require the submitting process to renew and
complete its own work.

The CUDA API and numerical behavior your application uses must fall within the
selected path's coverage. An application starting successfully does not prove
every operation is scheduled. Initialization/configuration diagnostics must be
resolved before relying on enforcement.

## Scheduling settings

The service owns the cross-process policy. Its `--scheduler-policy` accepts
`fifo`, `weighted_rr`, `drr` and `priority`; the default is `weighted_rr`.

- `--max-concurrent-tasks` bounds active leases across clients. With one slot,
  admitted launches are serialized. More slots allow overlap but are not a
  partition of GPU SMs or a utilization guarantee.
- `GLIMMER_MAX_CONCURRENT_KERNELS` controls the client launch gate. Set it
  explicitly; the default is one. The examples use two at both ends.
- `GLIMMER_SCHEDULER_PRIORITY` is a nonnegative integer; higher values take
  precedence under the priority policy. It defaults to zero.
- `GLIMMER_SCHEDULER_WEIGHT` is a positive integer for weighted policies.
- `GLIMMER_SCHEDULER_BATCH_SIZE` defaults to one launch per lease. Larger batches
  delay scheduling decisions; keep one for latency-sensitive work.
- `GLIMMER_SCHEDULER_MODE=observe` does not enforce scheduling. `enforce` selects
  enforcement; with no control socket it uses a process-local gate instead of
  sharing a service. `GLIMMER_SCHEDULER_POLICY` selects that local policy.

Priority changes admission order, not the execution of an already-running
ordinary kernel. A graph replay is not made node-preemptible by admission.
Unsupported calls and unobserved allocations are not made safe by a priority or
quota setting.

## Optional allocation limits

Add `GLIMMER_MEMORY_LIMIT_BYTES=4294967296` to the application's `env` command
to select a 4 GiB process-local accounting limit for covered allocations.
Choose a limit that fits the model and device; the example value is not a
capacity recommendation. This is separate from the service's task-reservation
budget. Setting the same value on two applications does not create a shared
4 GiB allocation pool.

Memory accounting covers supported allocation and release paths, not all
driver/framework overhead or arbitrary external mappings. Quota enforcement can
reject paths it cannot account for. See the
[CUDA allocation examples](../examples/cuda_workloads/README.md) and
[compatibility boundaries](CUDA_API_COVERAGE.md).

## Shutdown and native execution

Let all applications finish first, then query `stats` again. Investigate queued,
running or failed work instead of assuming process exit proves a clean drain.
Stop the foreground service with Ctrl+C after the clients have finished.
Remove the empty private directory with `rmdir "$glimmer_run_dir"` in the
service terminal. If it is not empty, inspect it rather than deleting it blindly.
An active client cannot transparently switch to native CUDA when its service
disappears; covered remote transport or lease failures are errors.

For ordinary execution without Glimmer, start the application from a clean
terminal without either preload library:

```sh
env -u LD_PRELOAD -u LD_AUDIT ./your_cuda_application
```

This command also removes other loader instrumentation. Use the application's
usual environment if it legitimately depends on another preload library.

## Centralized execution

Build both opt-in targets as described in
[Building Glimmer](BUILDING.md#centralized-cuda-execution-experimental).
This is not a drop-in substitution in the launch-admission commands above.
The frontend requires all of the following from a trusted launcher:

- `GLIMMER_EXECUTION_SOCKET`: the execution service's Unix socket path.
- `GLIMMER_EXECUTION_SERVER_PID`: the exact running service process ID.
- `GLIMMER_EXECUTION_SERVER_START`: that process's Linux `/proc/PID/stat`
  start-time value, in clock ticks, not a wall-clock timestamp.
- `LD_PRELOAD`: the matching `libglimmer_execution_interceptor.so`.

The frontend authenticates the service's PID, start identity and effective UID.
The launch-admission `GLIMMER_SCHEDULER_*` variables do not configure this
protocol. Centralized per-client priorities are selected on the service using
`--client-priority PID:START:PRIORITY`, not supplied as trusted priority by an
application. Memory limits use `--memory-per-client` and
`--host-memory-per-client`.

In priority scheduling, the service can briefly hold lower-priority work after
the highest-priority work drains, including after successful native allocation
release or BLAS handle destruction. `--priority-idle-us` sets this interval
in microseconds; it defaults to 100 and zero disables it. Actual resumption may
take longer because the interval does not bypass other scheduling or cleanup.

There is not yet a general-purpose application launcher that assembles this
configuration. Integrations must manage identities, service readiness, client
lifetime and cleanup, and qualify their required operations. Do not infer
arbitrary framework support from building the frontend. Cooperative kernel modes
and library replacements need additional, workload-specific qualification;
the default launch-admission quick start does not enable them.

For an experimental private GEMM signal layout, add
`--experimental-sharded-gemm-signal` to the execution service along with
`--replace-gemm`, `--gemm-tile-side 32`, and `--persistent-workers N`, where
`N` is positive. Preemptive kernel mode is required. Startup rejects an
unavailable compiled provider or incompatible settings. The provider checks
for SM70 hardware during preparation, before launching a provider kernel;
service readiness alone does not establish device compatibility. This selection
is fixed for the service's private GEMM sessions and cannot be changed per
client. It adds 32 mapped-host signal words, spaced 64 bytes apart, per command
slot, within the existing private-host memory budget. Other execution paths
retain their existing signal layout.
The option alone does not establish full-device participation, saturation,
or a performance improvement.

For optional execution-service diagnostics, run
`./build/cuda-debug/bin/glimmer_execution_service --help` to list the available
switches. Host-side timing reports describe service activity, not GPU execution time or SM
coverage; incomplete or invalid output cannot support a measurement claim.

The frontend can also record host RPC diagnostics by setting
`GLIMMER_EXECUTION_RPC_OUTPUT_DIR` to an existing, user-owned canonical
absolute directory. Recording is disabled by default. When enabled, it reserves
at least 512 MiB of fixed buffers per process and attempts to write two fresh
JSON files after normal shutdown, each limited to 1023 MiB. Account for memory and
disk overhead before enabling it; missing or invalid output is not usable.
