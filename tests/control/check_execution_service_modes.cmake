if(NOT DEFINED SERVICE)
    message(FATAL_ERROR "SERVICE executable is required")
endif()

execute_process(COMMAND "${SERVICE}" --help RESULT_VARIABLE status OUTPUT_VARIABLE usage
    ERROR_VARIABLE errors TIMEOUT 5)
if(NOT status EQUAL 0 OR NOT usage MATCHES "--scheduler priority\\|naive" OR
        NOT usage MATCHES "--kernel-mode original\\|preemptive" OR
        NOT usage MATCHES "--ptx-cache-bytes N" OR NOT usage MATCHES "--ptx-cache-entries N" OR
        NOT usage MATCHES "--ptx-cache-input ABSOLUTE_PATH" OR
        NOT usage MATCHES "--ptx-cache-output NEW_ABSOLUTE_PATH" OR
        NOT usage MATCHES "workload-agnostic" OR
        NOT usage MATCHES "--sharing-threads-per-sm N" OR
        NOT usage MATCHES "--sharing-min-speed-percent N" OR
        NOT usage MATCHES "PID:START:PRIORITY" OR
        NOT usage MATCHES "auto\\|original\\|configured" OR
        NOT usage MATCHES "--observe-block-completion" OR
        NOT usage MATCHES "--gemm-tile-side 32\\|64" OR
        NOT usage MATCHES "--experimental-sharded-gemm-signal" OR
        NOT usage MATCHES "--audit-copy-requests" OR
        NOT usage MATCHES "--audit-sgemm-target SESSION:SEQUENCE" OR
        NOT usage MATCHES "--audit-sgemm-target-client PID:START:SEQUENCE" OR
        NOT usage MATCHES "--audit-sgemm-range-library ABSOLUTE_PATH" OR
        NOT usage MATCHES "workload-aware" OR
        NOT usage MATCHES "--pair-workers N,N" OR
        NOT usage MATCHES "--pair-profile-input ABSOLUTE_PATH" OR
        NOT usage MATCHES "--pair-profile-output NEW_ABSOLUTE_PATH" OR
        NOT usage MATCHES "--pair-cache-pairs N" OR
        NOT usage MATCHES "--pair-cache-bytes N" OR
        NOT usage MATCHES "--pair-cache-candidates N" OR
        NOT usage MATCHES "--pair-warmup N" OR NOT usage MATCHES "--pair-samples N")
    message(FATAL_ERROR "Execution mode help is incomplete: ${usage}${errors}")
endif()

# All cases terminate in argument validation, before SDK resources or sockets.
function(expect_rejection diagnostic)
    execute_process(COMMAND "${SERVICE}" ${ARGN} RESULT_VARIABLE result
        OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 5)
    if(NOT result EQUAL 1 OR NOT "${output}${error}" MATCHES "${diagnostic}")
        message(FATAL_ERROR "Expected ${diagnostic}, received ${result}: ${output}${error}")
    endif()
endfunction()

expect_rejection("Missing execution-service option value" --audit-sgemm-target)
foreach(target 0:1 1:0 1 1:2:3 abc:1 1:-1)
    expect_rejection("Invalid or duplicate SGEMM audit target" --audit-sgemm-target "${target}")
endforeach()
expect_rejection("Invalid or duplicate SGEMM audit target"
    --audit-sgemm-target 1:2 --audit-sgemm-target 1:3)
foreach(target 0:1:1 1:0:1 1:1:0 1:2 1:2:3:4 4294967296:1:1 1:-1:2)
    expect_rejection("Invalid or duplicate SGEMM audit target" --audit-sgemm-target-client "${target}")
endforeach()
expect_rejection("Invalid or duplicate SGEMM audit target"
    --audit-sgemm-target 1:2 --audit-sgemm-target-client 1:2:3)
expect_rejection("Invalid or duplicate SGEMM audit target"
    --audit-sgemm-target-client 1:2:3 --audit-sgemm-target 1:2)
expect_rejection("SGEMM audit target requires native auditing"
    --socket unused-sgemm-audit.sock --audit-sgemm-target 1:2)
expect_rejection("Missing execution-service option value" --audit-sgemm-range-library)
expect_rejection("Invalid or duplicate SGEMM range library path" --audit-sgemm-range-library relative.so)
expect_rejection("Invalid or duplicate SGEMM range library path"
    --audit-sgemm-range-library /tmp/a --audit-sgemm-range-library /tmp/b)
expect_rejection("SGEMM range library requires an exact SGEMM audit target"
    --socket unused-sgemm-range.sock --audit-sgemm-range-library /tmp/a)
set(sgemm_range_base --socket unused-sgemm-range.sock --audit-native --audit-sgemm-target 1:2)
expect_rejection("cannot load SGEMM range provider"
    ${sgemm_range_base} --audit-sgemm-range-library /nonexistent/glimmer-nvtx.so)
expect_rejection("SGEMM range symbols differ from selected provider"
    ${sgemm_range_base} --audit-sgemm-range-library /lib/x86_64-linux-gnu/libm.so.6)

expect_rejection("Missing execution-service option value" --gemm-tile-side)
foreach(side 0 16 33 128)
    expect_rejection("Unknown or out-of-range execution-service option" --gemm-tile-side "${side}")
endforeach()
expect_rejection("Unknown or out-of-range execution-service option"
    --gemm-tile-side 32 --gemm-tile-side 64)
foreach(side 32 64)
    expect_rejection("Explicit GEMM tile geometry requires private replacement"
        --socket unused-gemm-geometry.sock --persistent-workers 320 --gemm-tile-side "${side}")
    expect_rejection("Invalid execution-service numeric value"
        --gemm-tile-side "${side}" --replace-gemm --persistent-workers 320 --device invalid)
endforeach()

expect_rejection("Duplicate sharded GEMM signal selection"
    --experimental-sharded-gemm-signal --experimental-sharded-gemm-signal)
set(sharded_base --socket unused-sharded-gemm.sock --experimental-sharded-gemm-signal)
expect_rejection("Sharded GEMM signal requires" ${sharded_base})
expect_rejection("Sharded GEMM signal requires" ${sharded_base}
    --persistent-workers 320 --gemm-tile-side 32)
expect_rejection("Sharded GEMM signal requires" ${sharded_base} --replace-gemm
    --gemm-tile-side 32)
expect_rejection("Sharded GEMM signal requires" ${sharded_base} --replace-gemm
    --persistent-workers 320)
expect_rejection("Sharded GEMM signal requires" ${sharded_base} --replace-gemm
    --persistent-workers 320 --gemm-tile-side 64)
foreach(mode original static block-static dynamic sliced)
    expect_rejection("Sharded GEMM signal requires" ${sharded_base} --replace-gemm
        --persistent-workers 320 --gemm-tile-side 32 --kernel-mode "${mode}")
endforeach()
expect_rejection("Invalid execution-service numeric value" ${sharded_base} --replace-gemm
    --persistent-workers 320 --gemm-tile-side 32 --device invalid)

if(NOT usage MATCHES "--priority-idle-us")
    message(FATAL_ERROR "Missing priority idle interval in help")
endif()
expect_rejection("Missing execution-service option value" --priority-idle-us)
expect_rejection("Invalid execution-service numeric value" --priority-idle-us -1)
expect_rejection("Duplicate or out-of-range priority idle interval" --priority-idle-us 1000001)
expect_rejection("Duplicate or out-of-range priority idle interval"
    --priority-idle-us 100 --priority-idle-us 0)
foreach(value 0 100 1000000)
    expect_rejection("Invalid execution-service numeric value"
        --priority-idle-us "${value}" --device invalid)
endforeach()
expect_rejection("Explicit priority idle interval requires"
    --socket unused-idle.sock --priority-idle-us 100)
foreach(mode naive profile workload-agnostic)
    expect_rejection("Explicit priority idle interval requires"
        --socket unused-idle.sock --priority-idle-us 100 --scheduler "${mode}"
        --client-priority 1:2:0)
endforeach()

expect_rejection("Duplicate block-completion observation selection"
    --observe-block-completion --observe-block-completion)
set(block_base --socket unused-block-observation.sock --observe-block-completion)
expect_rejection("Block-completion observation requires" ${block_base})
expect_rejection("Block-completion observation requires" ${block_base}
    --persistent-workers 320 --client-priority 1:2:0:configured)
foreach(mode original static block-static dynamic sliced)
    expect_rejection("Block-completion observation requires" ${block_base}
        --persistent-workers 320 --audit-priority --client-priority 1:2:0:configured
        --kernel-mode "${mode}")
endforeach()
foreach(mode naive profile workload-agnostic)
    expect_rejection("Block-completion observation requires" ${block_base}
        --persistent-workers 320 --audit-priority --client-priority 1:2:0:configured
        --scheduler "${mode}")
endforeach()
expect_rejection("Block-completion observation requires" ${block_base}
    --persistent-workers 320 --audit-priority --client-priority 1:2:0:configured
    --profile-input /tmp/not-read-block-profile)

if(NOT usage MATCHES "--enable-nccl")
    message(FATAL_ERROR "Missing NCCL cold selection in help")
endif()
if(NCCL_ENABLED)
    expect_rejection("Duplicate NCCL execution selection" --enable-nccl --enable-nccl)
    expect_rejection("Invalid execution-service numeric value" --enable-nccl --device invalid)
    expect_rejection("NCCL execution requires fewer than 64 source clients"
        --enable-nccl --socket unused-nccl.sock --max-clients 64)
else()
    expect_rejection("NCCL execution is unavailable in this build" --enable-nccl)
endif()

expect_rejection("Unknown execution scheduler" --scheduler invalid)
expect_rejection("Unknown kernel mode" --kernel-mode invalid)
expect_rejection("Missing execution-service option value" --scheduler)
expect_rejection("Missing execution-service option value" --kernel-mode)
foreach(pin "1:2:0" "1:2:0:auto" "1:2:0:original" "1:2:0:configured")
    # Reaching the later numeric parser proves the pin itself was accepted;
    # this command never constructs a service or opens a CUDA library.
    expect_rejection("Invalid execution-service numeric value" --client-priority "${pin}"
        --scheduler priority --persistent-workers 320 --device invalid)
endforeach()
foreach(pin "1:2:0:" "1:2:0:unknown" "1:2:0:auto:configured" "1:2:0:Configured"
        "1:2:-1:configured" "1:2:4294967296:original" "0:2:0:auto" "1:0:0:auto")
    expect_rejection("Invalid or oversized trusted execution priority table"
        --client-priority "${pin}")
endforeach()
foreach(second "1:2:0" "1:2:0:auto" "1:2:0:original" "1:2:100:configured")
    expect_rejection("Duplicate or conflicting trusted execution priority identity"
        --client-priority 1:2:0 --client-priority "${second}")
endforeach()
foreach(scheduler naive profile workload-agnostic)
    expect_rejection("Configured client kernels require priority scheduling"
        --socket unused-client-kernels.sock --scheduler "${scheduler}"
        --persistent-workers 320 --client-priority 1:2:0:configured)
endforeach()
expect_rejection("Configured client kernels require priority scheduling"
    --socket unused-client-kernels.sock --scheduler priority --persistent-workers 320
    --kernel-mode original --client-priority 1:2:0:configured)
expect_rejection("Configured client kernels require priority scheduling"
    --socket unused-client-kernels.sock --scheduler priority --client-priority 1:2:0:configured)
expect_rejection("Original kernel mode conflicts" --socket unused-mode-test.sock
    --kernel-mode original --replace-gemm --persistent-workers 320)
expect_rejection("Transformed kernel mode requires explicit persistent workers"
    --socket unused-mode-test.sock --kernel-mode preemptive)
expect_rejection("Invalid execution-service numeric value" --scheduler naive
    --kernel-mode original --device invalid)
expect_rejection("Invalid execution-service numeric value" --scheduler priority
    --kernel-mode preemptive --persistent-workers 320 --device invalid)
expect_rejection("Invalid execution-service numeric value" --ptx-cache-bytes 0
    --ptx-cache-entries 0 --device invalid)
expect_rejection("Unknown or out-of-range execution-service option" --ptx-cache-entries 4097)
expect_rejection("PTX archive requires an absolute file path" --ptx-cache-input relative.cache)
expect_rejection("Duplicate PTX archive option" --ptx-cache-output /tmp/first.cache
    --ptx-cache-output /tmp/second.cache)
expect_rejection("PTX archives require enabled transformation and cache"
    --socket unused-cache.sock --ptx-cache-input /tmp/input.cache)
expect_rejection("PTX archives require enabled transformation and cache"
    --socket unused-cache.sock --persistent-workers 320 --ptx-cache-bytes 0
    --ptx-cache-output /tmp/output.cache)
expect_rejection("PTX archives require enabled transformation and cache"
    --socket unused-cache.sock --persistent-workers 320 --kernel-mode original
    --ptx-cache-input /tmp/input.cache)
expect_rejection("Execution archive output paths must be distinct"
    --socket unused-cache.sock --persistent-workers 320
    --ptx-cache-input /tmp/input.cache --ptx-cache-output /tmp/input.cache)
expect_rejection("Execution archive output paths must be distinct"
    --socket unused-cache.sock --persistent-workers 320
    --ptx-cache-output /tmp/output.cache --profile-output /tmp/output.cache)
expect_rejection("Execution archive output paths must be distinct"
    --socket unused-cache.sock --persistent-workers 320
    --ptx-cache-output /tmp/input.cache --profile-input /tmp/input.cache)
expect_rejection("Execution archive output paths must be distinct"
    --socket unused-cache.sock --persistent-workers 320
    --ptx-cache-input /tmp/input.cache --profile-output /tmp/input.cache)
expect_rejection("Invalid execution-service numeric value" --ptx-cache-input /tmp/input.cache
    --ptx-cache-output /tmp/new.cache --persistent-workers 320 --device invalid)
foreach(kind static block-static dynamic sliced)
    if(NOT usage MATCHES "${kind}")
        message(FATAL_ERROR "Missing execution kind ${kind} in help")
    endif()
    expect_rejection("Transformed kernel mode requires explicit persistent workers"
        --socket unused-mode-test.sock --kernel-mode "${kind}")
    expect_rejection("Invalid execution-service numeric value"
        --kernel-mode "${kind}" --persistent-workers 320 --device invalid)
    expect_rejection("Private GEMM replacement requires preemptive kernel mode"
        --socket unused-mode-test.sock --kernel-mode "${kind}"
        --persistent-workers 320 --replace-gemm)
endforeach()
expect_rejection("Invalid profile worker list" --profile-workers 80,80)
expect_rejection("Invalid profile worker list" --profile-workers 80,)
expect_rejection("Invalid profile worker list" --profile-workers 0,80)
expect_rejection("Profile options require the profile scheduler" --socket unused-profile.sock
    --profile-workers 80,160)
expect_rejection("Profile requires explicit compatible workers" --socket unused-profile.sock
    --scheduler profile)
expect_rejection("Profile requires explicit compatible workers" --socket unused-profile.sock
    --scheduler profile --persistent-workers 80)
expect_rejection("Invalid execution-service numeric value" --scheduler profile
    --profile-workers 80,160,320 --profile-warmup 0 --profile-samples 1 --persistent-workers 320
    --profile-cache-functions 8 --profile-cache-bytes 65536 --device invalid)
expect_rejection("Unknown or out-of-range execution-service option" --profile-samples 0)
expect_rejection("Invalid profile family list" --profile-kinds static,static)
expect_rejection("Invalid profile family list" --profile-kinds sliced,)
expect_rejection("Invalid profile family list" --profile-kinds original)
expect_rejection("Invalid profile family list" --profile-kinds unknown)
expect_rejection("Profile options require the profile scheduler" --socket unused-profile.sock
    --profile-kinds dynamic,sliced)
expect_rejection("Profile requires explicit compatible workers" --socket unused-profile.sock
    --scheduler profile --kernel-mode original --profile-kinds dynamic)
expect_rejection("Invalid execution-service numeric value" --scheduler profile
    --profile-kinds static,block-static,dynamic,preemptive,sliced --persistent-workers 320
    --device invalid)
expect_rejection("Profile archive requires an absolute file path" --profile-input relative.cache)
expect_rejection("Duplicate profile archive option" --profile-input /tmp/first.cache
    --profile-input /tmp/second.cache)
expect_rejection("Profile output requires the profile scheduler" --socket unused-profile.sock
    --profile-output /tmp/new.cache)
expect_rejection("Profile output must not replace the input archive" --socket unused-profile.sock
    --scheduler profile --kernel-mode original
    --profile-input /tmp/input.cache --profile-output /tmp/input.cache)
expect_rejection("Invalid execution-service numeric value" --scheduler naive
    --profile-input /tmp/input.cache --profile-kinds dynamic,sliced --persistent-workers 320
    --device invalid)

expect_rejection("Workload-agnostic scheduling needs a readonly profile"
    --socket unused-sharing.sock --scheduler workload-agnostic --persistent-workers 320)
expect_rejection("Workload-agnostic scheduling needs a readonly profile"
    --socket unused-sharing.sock --scheduler workload-agnostic --profile-input /tmp/input.cache)
expect_rejection("Workload-agnostic scheduling needs a readonly profile"
    --socket unused-sharing.sock --scheduler workload-agnostic --persistent-workers 320
    --profile-input /tmp/input.cache --profile-output /tmp/output.cache)
expect_rejection("Workload-agnostic scheduling needs a readonly profile"
    --socket unused-sharing.sock --scheduler workload-agnostic --persistent-workers 320
    --profile-input /tmp/input.cache --kernel-mode original)
expect_rejection("Workload-agnostic scheduling needs a readonly profile"
    --socket unused-sharing.sock --scheduler workload-agnostic --persistent-workers 320
    --profile-input /tmp/input.cache --replace-gemm)
expect_rejection("Workload-agnostic scheduling needs a readonly profile"
    --socket unused-sharing.sock --scheduler workload-agnostic --persistent-workers 320
    --profile-input /tmp/input.cache --client-priority 1:2:0)
foreach(scheduler priority naive profile)
    expect_rejection("Workload-agnostic scheduling needs a readonly profile"
        --socket unused-sharing.sock --scheduler "${scheduler}" --sharing-threads-per-sm 512)
    expect_rejection("Workload-agnostic scheduling needs a readonly profile"
        --socket unused-sharing.sock --scheduler "${scheduler}" --sharing-min-speed-percent 90)
endforeach()
foreach(value 0 65537)
    expect_rejection("Unknown or out-of-range execution-service option"
        --sharing-threads-per-sm "${value}")
endforeach()
foreach(value 0 10001)
    expect_rejection("Unknown or out-of-range execution-service option"
        --sharing-min-speed-percent "${value}")
endforeach()
expect_rejection("Invalid execution-service numeric value" --scheduler workload-agnostic
    --profile-input /tmp/input.cache --profile-workers 80,160,320 --profile-kinds dynamic
    --persistent-workers 320 --sharing-threads-per-sm 512 --sharing-min-speed-percent 90
    --device invalid)
