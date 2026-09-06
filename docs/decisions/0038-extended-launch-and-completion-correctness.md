# 0038: Close extended-launch and completion-monitor gaps

Status: Accepted

## Scope and acceptance

Correct the functional findings from ADR 0037 before policy tuning: cover the
extended Driver/Runtime launch ABIs and PTDS aliases and serialize event removal
with batch terminal decisions. Evaluate, but do not accept without measurement,
a replacement for repeated not-ready polling. Preserve all launch
attributes, argument storage, streams, native error codes, and reentrancy.
Missing real entry points fail closed. No new dependency or scheduler policy.

Keep completion ownership in the interceptor. An event remains pending while
its query and destruction run outside the state mutex; removing it and deciding
the batch outcome must be one locked transition. Never reinterpret a real CUDA
failure as success. Pending work retains its lease until completion is proven.

Retain the existing immediate retry policy for pending events after rejecting
two timed-backoff candidates during verification. The one-millisecond idle
condition-variable wait is not a delay between event queries. CPU polling cost
remains a performance follow-up, not a solved problem. No device synchronization
is inserted into ordinary launch calls and no polling policy moves into the
scheduler core.

Extended launch configurations and their attribute arrays are forwarded intact;
the scheduler does not rewrite cooperative/programmatic dependency attributes.
This adds boundary interception, not a guarantee of scheduling benefit for all
attribute combinations. Completion uses an ordinary stream-ordered event, not
a launch-completion attribute event. PTDS null streams are mapped only for
completion tracking; the original native launch arguments remain unchanged.
Proc-address PTDS selection uses the same central wrapper registry as direct
lookup. Nested Driver lookups retain the stream flags and use cached async
allocation/free entry points without re-entering `dlsym`.

Review also corrects reserved-capacity admission to count lower-priority
running tasks against the unreserved allowance. High-priority tasks occupying
reserved slots must not prevent filling a free unreserved slot. Paired framework
experiments now reject failed/cancelled tasks and undrained scheduler resources,
independently of model result validation.

## Risks and verification

Exact ABI/alias forwarding and nested Runtime-to-Driver calls require fake and
real GPU tests. A blocked fake event-destroy operation must reproduce the
close-batch race deterministically and prove exactly one terminal result after
the fix, including failed destruction. Test missing symbols, invalid arguments,
failed launches/events, multiple pending events, and stop/cleanup.

Run sanitizer, lint, formatting, optimized and GPU tests. Repeat the unprofiled
framework launch-count check and remote experiments with strict drained-task
checks. Keep profiler warnings visible; warning-bearing traces cannot prove a
performance gain. Compare untraced optimized baselines after correctness gates.
Review the source/test/documentation diffs before the authorized commit.

## Review dispositions and evidence

- Major: extended launch entry points and PTDS lookup flags were not fully
  covered. Central registry selection and cached Driver forwarding now retain
  the flags. Fake ABI tests cover all four exports, original attributes and
  argument pointers, native errors, and missing Runtime symbols. Real Driver
  PTX and Runtime tests validate ordinary and PTDS extended launches.
- Major: removing a pending event before cleanup could let `close_batch()`
  publish a terminal result before the monitor. A deterministic blocked-destroy
  regression now keeps the task running until cleanup finishes, for both
  successful and failed destruction. Creation, recording, query, multiple-event,
  and shutdown paths are also covered. Shutdown no longer allocates a set while
  collecting batches to fail.
- Major: reserved-capacity admission counted high-priority running tasks
  against the lower-priority allowance. A regression now fills an available
  unreserved slot while high-priority work remains running.
- Major: model validation alone could report a successful experiment with a
  failed scheduler task. Paired measurements now require all admitted tasks to
  complete, with no failed/cancelled tasks or retained resources.
- Minor: the manually maintained symbol-table size left two empty entries.
  Deduce its size from the entries so additions cannot silently leave padding.
- Major: CTest's required-output matching ignores a nonzero process exit code.
  Tests checking interceptor diagnostics now also reject an explicit assertion
  failure marker, including Runtime result-validation and cleanup failures.
  Matching an earlier hook log must not mask a later failure.
  A negative CTest fixture reproduced the old false pass with matching output
  followed by exit status 1; the hardened configuration correctly rejected it.

Verification on Linux/WSL2, GTX 1660 SUPER, CUDA 13.2, and PyTorch 2.12.1+cu132:

- `./scripts/check.sh` passed formatting, ShellCheck, Debug, ASan/UBSan, lint,
  CUDA lint, and optimized no-GPU checks. The five socket/process tests skipped
  in the sandbox were rerun outside it for each affected preset and passed.
- `ctest --preset cuda-perf --output-on-failure` passed all 69 tests on the
  actual GPU. A separate CUDA-enabled ASan/UBSan build passed the dispatch and
  completion-monitor tests.
- Unprofiled training with one and two measured iterations, no warmup, batch
  size 32, hidden size 1024, and two work units produced 47 and 88 observations:
  41 additional launches per step, including six `cuLaunchKernelEx` calls.
  This closes the observed 35-versus-41 gap for this workload, not every possible
  framework launch path.
- Three-second static-reservation and adaptive co-location runs passed the
  common-window, same-GPU, model, and strict drain checks. Static mode completed
  all 10,874 admitted tasks; adaptive mode completed all 12,385. Both had zero
  failed/cancelled tasks and no retained reservation or allocation.
- A repeated remote training capture completed all 3,368 admitted tasks; the
  original terminal-report failure did not recur. Nsight still warned that CUDA
  events might be missing, so the trace is not evidence of complete coverage or
  a performance gain.

These short workload runs establish functional regression evidence. They do
not establish hard real-time guarantees, arbitrary framework compatibility, or
high-performance scheduling.

## Rejected polling candidates and follow-up

The correctness-fixed immediate-polling baseline measured a median of about
177 local training steps/second. A candidate with eight immediate NOT_READY
probes followed by exponential 1-to-32-microsecond waits measured about 153.
Replacing the probe count with a 100-microsecond spin window still measured
about 152 in a separate clean run. Each matrix used optimized binaries, five
seconds per role/mode, three repetitions, fixed model parameters, and no launch
tracing; all remote task-drain checks passed. The first candidate's collection
overlapped a small test build, so it is diagnostic only. The later clean
spin-window run did not establish a benefit either.

Do not accept either candidate as a performance fix. Restore immediate retries
while retaining the completion ownership fix. A future polling change must
measure CPU cost and inference tails as well as training throughput on repeated,
untraced runs. OS wakeup latency is not bounded by a requested microsecond wait.
Reduced profiler event-query counts alone do not satisfy this acceptance gate.

After restoring immediate retries, another complete 30-run untraced matrix
passed all quota and remote-drain checks. Local training recovered to a median
170.8 steps/second, compared with 151.8 for the timed spin-window candidate.
Native and remote training measured 301.4 and 77.4 steps/second respectively;
substantial admission overhead remains. The comparison supports rejecting the
candidate, not claiming that scheduling is faster than native execution.

Final static and adaptive co-location repetitions completed all 13,742 and
14,351 admitted tasks, respectively, without failed/cancelled tasks or retained
resources. These checks were performed after restoring immediate polling.

## Review decision

Approved after the functional, ABI, ownership, failure-path, documentation,
script, sanitizer, and real-GPU checks above. No unresolved Blocker or Major
finding remains in this reviewed change. The final required-output failure
checks were rebuilt and verified in all 69 optimized tests; the measured
interceptor and control binary hashes were unchanged by that test-only fix.
CPU polling cost and profiler completeness remain explicitly deferred
limitations. Do not describe this integration as a performance improvement or
a hard latency guarantee.

## Rollback

These changes remain within existing interceptor dispatch, wrappers, registry,
and completion-monitor responsibilities. They do not change the wire protocol,
quota ownership, or public scheduler API. Revert the corresponding complete
logical change if its regression or workload gates fail; do not weaken checks.
