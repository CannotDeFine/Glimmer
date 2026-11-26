if(NOT DEFINED SERVICE)
    message(FATAL_ERROR "SERVICE executable is required")
endif()

# ASan interposes the vendor's lazy dlopen, so its $ORIGIN RUNPATH is not
# sufficient to select split libraries. Bind this controlled cold-CLI fixture
# to its configured SDK; production provider/version rejection stays unchanged.
set(service_command "${SERVICE}")
if(DEFINED CUDNN_ROOT AND NOT CUDNN_ROOT STREQUAL "")
    set(service_command "${CMAKE_COMMAND}" -E env
        "LD_LIBRARY_PATH=${CUDNN_ROOT}/lib:${CUDNN_ROOT}/lib64" "${SERVICE}")
endif()

# Separate bounded cold subprocess group; preserve the existing mode-test limit.
function(expect_rejection diagnostic)
    execute_process(COMMAND ${service_command} ${ARGN} RESULT_VARIABLE result
        OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 5)
    if(NOT result EQUAL 1 OR NOT "${output}${error}" MATCHES "${diagnostic}")
        message(FATAL_ERROR "Expected ${diagnostic}, received ${result}: ${output}${error}")
    endif()
endfunction()

expect_rejection("Duplicate copy-request audit selection" --audit-copy-requests --audit-copy-requests)
expect_rejection("Invalid execution-service numeric value" --audit-copy-requests --device invalid)

foreach(workers "0" "160," "160,160" "80,160,240,320,400,480,560,640,720")
    expect_rejection("Invalid or duplicate pair worker list" --pair-workers "${workers}")
endforeach()
expect_rejection("Invalid or duplicate pair worker list" --pair-workers 160 --pair-workers 80)
expect_rejection("Pair profile archive requires an absolute file path" --pair-profile-input relative.cache)
expect_rejection("Pair profile archive requires an absolute file path" --pair-profile-output /)
expect_rejection("Duplicate pair profile archive option" --pair-profile-input /tmp/one --pair-profile-input /tmp/two)
foreach(option pair-cache-pairs pair-cache-bytes pair-cache-candidates pair-samples)
    expect_rejection("Unknown or out-of-range execution-service option" "--${option}" 0)
endforeach()
expect_rejection("Unknown or out-of-range execution-service option" --pair-cache-pairs 1025)
expect_rejection("Unknown or out-of-range execution-service option" --pair-cache-candidates 65)
expect_rejection("Unknown or out-of-range execution-service option" --pair-cache-bytes 134217729)
expect_rejection("Unknown or out-of-range execution-service option" --pair-warmup 1001)
expect_rejection("Unknown or out-of-range execution-service option" --pair-samples 1001)
foreach(mode priority naive profile workload-agnostic)
    expect_rejection("Pair options require workload-aware scheduling" --socket unused-pair.sock
        --scheduler "${mode}" --pair-workers 160)
endforeach()
expect_rejection("Workload-aware scheduling requires readonly singles" --socket unused-pair.sock
    --scheduler workload-aware --persistent-workers 320)
set(pair_base --socket unused-pair.sock --scheduler workload-aware --persistent-workers 320
    --profile-input /tmp/not-read-pair-single --pair-workers 160
    --client-priority 1:2:0:configured --client-priority 2:3:0:configured)
expect_rejection("Workload-aware scheduling requires readonly singles" ${pair_base} --max-clients 1)
expect_rejection("Workload-aware scheduling requires readonly singles" ${pair_base} --profile-output /tmp/not-written)
foreach(kind static block-static dynamic)
    expect_rejection("Workload-aware scheduling requires readonly singles" ${pair_base} --kernel-mode "${kind}")
endforeach()
expect_rejection("Block-completion observation requires" ${pair_base} --observe-block-completion)
expect_rejection("Pair archive outputs must be distinct" ${pair_base}
    --pair-profile-output /tmp/not-read-pair-single)
expect_rejection("Pair archive outputs must be distinct" ${pair_base}
    --pair-profile-input /tmp/pair-in --ptx-cache-output /tmp/pair-in)
expect_rejection("Pair archive outputs must be distinct" ${pair_base}
    --memory-per-client 18446744073709551615)
expect_rejection("Pair profiling requires exactly two equal configured pins" ${pair_base}
    --pair-profile-output /tmp/not-written-pair --client-priority 3:4:100:original)
expect_rejection("Pair profiling requires exactly two equal configured pins"
    --socket unused-pair.sock --scheduler workload-aware --persistent-workers 320
    --profile-input /tmp/not-read-pair-single --pair-workers 80,160 --pair-cache-candidates 1
    --client-priority 1:2:0:configured --client-priority 2:3:0:configured
    --pair-profile-output /tmp/not-written-pair)
# Valid mode/pin/catalog syntax reaches protected archive validation, still
# before native domain initialization or socket creation. Stale pin values are
# not execution authority and are never used by a live application here.
expect_rejection("invalid, incompatible or inaccessible profile archive" ${pair_base})
expect_rejection("invalid, incompatible or inaccessible profile archive" ${pair_base}
    --pair-profile-output /tmp/not-written-pair --pair-cache-candidates 1
    --pair-warmup 0 --pair-samples 1)
expect_rejection("invalid, incompatible or inaccessible profile archive" ${pair_base}
    --client-priority 3:4:100:original --pair-profile-input /tmp/not-read-pair)
