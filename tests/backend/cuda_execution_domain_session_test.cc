#include "cuda_execution_domain_fixture.h"

namespace {
using namespace glimmer::backend::cuda;
using namespace glimmer::test::cuda_domain;

enum class PairScenario { kSharedResume, kSecondNativeError, kColdOriginal };

enum class PairMetadataCase {
    kPrepared,
    kMissingDeviceQuery,
    kMissingFunctionQuery,
    kDeviceError,
    kFunctionError,
    kOccupancyError,
    kDynamicFamily,
};
struct PairMetadataObservation {
    PairMetadataCase scenario{};
    bool is_observing{};
    std::array<unsigned int, 3> queries{};
};
PairMetadataObservation pair_metadata;

CUresult CUDAAPI observe_pair_device(int* output, CUdevice_attribute attribute, CUdevice device) {
    if (pair_metadata.is_observing) {
        ++pair_metadata.queries[0];
        if (pair_metadata.scenario == PairMetadataCase::kDeviceError)
            return CUDA_ERROR_NOT_SUPPORTED;
    }
    return pair_dispatch().device_attribute(output, attribute, device);
}
CUresult CUDAAPI observe_pair_function(int* output, CUfunction_attribute attribute,
                                       CUfunction function) {
    if (pair_metadata.is_observing) {
        ++pair_metadata.queries[1];
        if (pair_metadata.scenario == PairMetadataCase::kFunctionError)
            return CUDA_ERROR_NOT_SUPPORTED;
    }
    return pair_dispatch().function_get_attribute(output, attribute, function);
}
CUresult CUDAAPI observe_pair_occupancy(int* output, CUfunction function, int threads,
                                        std::size_t shared_bytes) {
    if (pair_metadata.is_observing) {
        ++pair_metadata.queries[2];
        if (pair_metadata.scenario == PairMetadataCase::kOccupancyError)
            return CUDA_ERROR_NOT_SUPPORTED;
    }
    return occupancy(output, function, threads, shared_bytes);
}

void pair_catalog_errors_are_not_structural_exclusions() {
    using Disposition = CudaSessionPairKernelDisposition;
    const CudaSessionPairKernelResult absent;
    expect(absent.status != CUDA_SUCCESS && absent.disposition == Disposition::kFailed &&
               !absent.kernel,
           "an absent pair preparation is a failure, never an empty prepared or optional catalog");
    constexpr std::array cases{PairMetadataCase::kPrepared,
                               PairMetadataCase::kMissingDeviceQuery,
                               PairMetadataCase::kMissingFunctionQuery,
                               PairMetadataCase::kDeviceError,
                               PairMetadataCase::kFunctionError,
                               PairMetadataCase::kOccupancyError,
                               PairMetadataCase::kDynamicFamily};
    for (const auto scenario : cases) {
        driver = {};
        pair_metadata = {.scenario = scenario};
        auto dispatch = pair_dispatch();
        dispatch.device_attribute =
            scenario == PairMetadataCase::kMissingDeviceQuery ? nullptr : observe_pair_device;
        dispatch.function_get_attribute =
            scenario == PairMetadataCase::kMissingFunctionQuery ? nullptr : observe_pair_function;
        dispatch.occupancy = observe_pair_occupancy;
        CudaSessionPersistentConfig persistent{320};
        if (scenario == PairMetadataCase::kDynamicFamily)
            persistent.kind = glimmer::transform::PtxExecutionKind::kDynamic;
        CudaExecutionSession session({1, 99}, 101, dispatch, limits(1000), persistent);
        success(session.initialize(0));
        const auto module = session.load_ptx(kSource);
        success(module);
        const auto function = session.get_function(module.id, "kernel");
        success(function);
        const auto buffer = session.allocate(32);
        success(buffer);
        KernelInput input({function.id, 0}, buffer.address);
        const auto accepted = session.accept_kernel(1, input.kernel);
        const auto pending = session.pending_command();
        const auto progress = session.persistent_progress(1);
        expect(accepted.accepted() && pending.status == CUDA_SUCCESS && pending.command &&
                   pending.command.description().command.is_persistent && progress &&
                   progress->kind == persistent.kind,
               "test uses a real accepted persistent command of the requested public family");
        const auto configuration_calls = driver.configuration_calls;
        const auto parameter_calls = driver.parameter_calls;
        pair_metadata.is_observing = true;
        const auto prepared = session.prepare_pair_kernel(pending.command);
        const bool is_prepared = scenario == PairMetadataCase::kPrepared;
        const bool is_unsupported = scenario == PairMetadataCase::kDynamicFamily;
        const auto expected = is_prepared      ? Disposition::kPrepared
                              : is_unsupported ? Disposition::kUnsupported
                                               : Disposition::kFailed;
        expect(prepared.status == (is_prepared ? CUDA_SUCCESS : CUDA_ERROR_NOT_SUPPORTED) &&
                   prepared.disposition == expected && bool(prepared.kernel) == is_prepared,
               "only known structural exclusion is skippable; identical native error stays failed");
        std::array<unsigned int, 3> expected_queries{};
        if (is_prepared || scenario == PairMetadataCase::kOccupancyError)
            expected_queries = {5, 2, 1};
        else if (scenario == PairMetadataCase::kDeviceError)
            expected_queries = {1, 0, 0};
        else if (scenario == PairMetadataCase::kFunctionError)
            expected_queries = {5, 1, 0};
        expect(
            pair_metadata.queries == expected_queries,
            "metadata failure stops at its real callback; structural exclusion queries no catalog");
        if (is_prepared) {
            const auto refreshed = session.refresh_pair_kernel(prepared.kernel, pending.command);
            expect(refreshed.status == CUDA_SUCCESS &&
                       refreshed.disposition == Disposition::kPrepared && refreshed.kernel &&
                       refreshed.kernel.description().catalog_generation ==
                           prepared.kernel.description().catalog_generation &&
                       pair_metadata.queries == expected_queries,
                   "exact refresh preserves prepared identity without native metadata queries");
        }
        const auto after = session.persistent_progress(1);
        expect(session.pending_command().command.description() == pending.command.description() &&
                   after && after->kind == progress->kind && after->workers == progress->workers &&
                   after->total_blocks == progress->total_blocks && !after->physical_launches &&
                   !after->physical_stops && !after->completed_blocks &&
                   driver.configuration_calls == configuration_calls &&
                   driver.parameter_calls == parameter_calls && !driver.launches &&
                   !driver.counter_initializations && !session.snapshot().is_quarantined,
               "catalog classification cannot select geometry, initialize counters, issue or "
               "quarantine");
        pair_metadata.is_observing = false;
        expect(session.reject_queued(CUDA_ERROR_INVALID_VALUE) == 1 &&
                   session.acknowledge_result(accepted.receipt),
               "cancel and acknowledge the exact never-issued test command before normal cleanup");
        success(session.release(buffer.id));
        success(session.unload_module(module.id));
        success(session.shutdown());
        expect(
            !driver.live_resources() && driver.context_destroys == 1 && !driver.current &&
                !driver.launches && !driver.counter_initializations,
            "all preparation outcomes retain zero native work and complete owned resource cleanup");
    }
}

void configured_functions_reuse_values_not_epochs() {
    driver = {};
    driver.requires_explicit_cleanup = true;
    driver.has_profile_function_metadata = true;
    driver.next_claimed = 8;
    CudaProfileConfig settings;
    settings.workers = {320};
    settings.warmup_calls = 0;
    settings.samples_per_candidate = 1;
    auto learning = std::make_shared<CudaProfileCache>(settings, pair_clock);
    CudaExecutionSession session({1, 99}, 101, pair_dispatch(), limits(1000), {320}, {}, learning);
    success(session.initialize(0));
    const auto module = session.load_ptx(kSource);
    success(module);
    const auto function = session.get_function(module.id, "kernel");
    success(function);
    const auto buffer = session.allocate(32);
    success(buffer);
    const auto configure = [&](int value) {
        const auto calls = driver.configuration_calls;
        const auto result =
            session.function_control(CudaFunctionControl::kSetAttribute, function.id,
                                     CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, value);
        expect(result.status == CUDA_SUCCESS && result.original_status == CUDA_SUCCESS &&
                   driver.configuration_calls == calls + 2,
               "one requested setter and one distinct private setter, unchanged original status");
    };
    std::uint64_t sequence{};
    const auto execute = [&](unsigned int candidate) {
        KernelInput input({function.id, 0}, buffer.address);
        const auto accepted = session.accept_kernel(++sequence, input.kernel);
        pair_now_ns = 1000;
        const auto issued = session.issue(session.pending_command().command);
        expect(accepted.accepted() && issued.state == CudaExecutionIssueState::kIssued &&
                   issued.native_status == CUDA_SUCCESS && issued.completion_status == CUDA_SUCCESS,
               "configured function executes the original request once");
        pair_now_ns = candidate ? 1100 : 1200;
        expect(
            session.acknowledge_result(accepted.receipt) && session.poll().status == CUDA_SUCCESS,
            "real final retirement verifies actual function attributes before training");
    };
    for (const int value : {1024, 2048}) {
        configure(value);
        execute(0);
        execute(1);
    }
    expect(learning->snapshot().cases == 2 && learning->snapshot().trained_cases == 2 &&
               learning->snapshot().samples == 4,
           "two actual function configurations train separate original/private cases");
    configure(1024);
    execute(1);
    expect(learning->snapshot().cases == 2 && learning->snapshot().samples == 4 &&
               driver.physical_workers == std::vector<unsigned int>{8, 320, 8, 320, 320},
           "same canonical function attributes reuse at a later control epoch without replay");
    const auto cache = session.function_control(CudaFunctionControl::kSetCache, function.id,
                                                CU_FUNC_CACHE_PREFER_SHARED);
    expect(cache.status == CUDA_SUCCESS && cache.original_status == CUDA_SUCCESS,
           "explicit cache preference is not confused with CACHE_MODE_CA");
    execute(0);
    execute(1);
    expect(learning->snapshot().cases == 3 && learning->snapshot().samples == 6,
           "unqueryable cache preference provenance distinguishes otherwise equal attributes");
    const auto failed_samples = learning->snapshot().samples;
    driver.function_configuration_status = CUDA_ERROR_INVALID_VALUE;
    const auto failed = session.function_control(CudaFunctionControl::kSetCache, function.id,
                                                 CU_FUNC_CACHE_PREFER_L1);
    expect(failed.status == CUDA_SUCCESS && failed.original_status == CUDA_ERROR_INVALID_VALUE,
           "failed original setter retains its actual SDK error");
    execute(0);
    expect(driver.physical_workers.back() == 8 && learning->snapshot().samples == failed_samples,
           "unknown post-error history cannot inherit or train configured private data");
    std::array<std::uint32_t, 8> output{};
    success(session.download({buffer.id, 0}, std::as_writable_bytes(std::span{output})));
    expect(std::ranges::all_of(output, [](auto value) { return value == 56; }),
           "all eight non-idempotent inputs execute exactly once across configuration changes");
    expect(session.poll().status == CUDA_SUCCESS, "retire actual readback prefix");
    success(session.release(buffer.id));
    success(session.unload_module(module.id));
    success(session.shutdown());
    expect(driver.live_resources() == 0 && driver.context_destroys == 1,
           "configured function measurements retain and clean their real owners once");
    const auto archive = learning->encode("function-profile-session");
    CudaProfileCache restored(settings, pair_clock, false);
    expect(archive &&
               (*archive)[4 + std::string_view{"Glimmer profile archive"}.size()] == std::byte{3} &&
               restored.restore(*archive, "function-profile-session") &&
               restored.snapshot().loaded_cases == 3,
           "actual Session function measurements restore with explicit V3 provenance");
}

void unexpected_function_state_rejects_sample() {
    const auto child = fork();
    expect(child >= 0, "fork explicit retained-uncertainty function control");
    if (!child) {
        try {
            driver = {};
            driver.has_profile_function_metadata = true;
            driver.next_claimed = 8;
            CudaProfileConfig settings;
            settings.workers = {320};
            settings.warmup_calls = 0;
            settings.samples_per_candidate = 1;
            auto learning = std::make_shared<CudaProfileCache>(settings, pair_clock);
            CudaExecutionSession session({1, 99}, 101, pair_dispatch(), limits(1000), {320}, {},
                                         learning);
            success(session.initialize(0));
            const auto module = session.load_ptx(kSource);
            success(module);
            const auto function = session.get_function(module.id, "kernel");
            success(function);
            const auto configured = session.function_control(
                CudaFunctionControl::kSetCache, function.id, CU_FUNC_CACHE_PREFER_SHARED);
            expect(configured.original_status == CUDA_SUCCESS, "actual requested function setter");
            const auto buffer = session.allocate(32);
            success(buffer);
            KernelInput input({function.id, 0}, buffer.address);
            const auto accepted = session.accept_kernel(1, input.kernel);
            expect(accepted.accepted(), "one distinct function-bound sample");
            driver.on_launch = [] {
                for (auto& loaded : driver.modules)
                    if (loaded.is_live) ++loaded.original.attributes[8];
            };
            pair_now_ns = 1000;
            const auto issued = session.issue(session.pending_command().command);
            driver.on_launch = {};
            expect(issued.native_status == CUDA_SUCCESS && issued.completion_status == CUDA_SUCCESS,
                   "unexpected function effects cannot invent an original launch error");
            // A public mutation still cannot enter while accepted physical work is held.
            driver.ready_launch = 0;
            const auto calls = driver.configuration_calls;
            const auto deferred = session.function_control(CudaFunctionControl::kSetCache,
                                                           function.id, CU_FUNC_CACHE_PREFER_L1);
            expect(deferred.disposition == CudaSessionDisposition::kDeferred &&
                       !deferred.original_status && driver.configuration_calls == calls,
                   "function identity capture never authorizes a physical-tail mutation");
            driver.ready_launch = 100;
            pair_now_ns = 1200;
            expect(session.acknowledge_result(accepted.receipt), "original function sample ACK");
            (void)session.poll();
            const auto measured = learning->snapshot();
            expect(
                !measured.samples && !measured.trained_cases && measured.rejected_samples == 1 &&
                    session.snapshot().is_quarantined && !driver.context_destroys,
                "final native function readback rejects before commit and retains unknown owners");
            // Native uncertainty intentionally retains resources. This child
            // does not claim healthy cleanup or leak-free positive execution.
            _exit(0);
        } catch (const std::exception& error) {
            std::cerr << "Function configuration uncertainty: " << error.what() << '\n';
            _exit(1);
        }
    }
    int status{};
    expect(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "explicit function-uncertainty child obeys its retained-owner oracle");
}
void unexpected_configuration_cannot_commit_measurement() {
    const auto child = fork();
    expect(child >= 0, "fork intentional configured native-uncertainty control");
    if (!child) {
        try {
            driver = {};
            driver.next_claimed = 8;
            CudaProfileConfig settings;
            settings.workers = {320};
            settings.warmup_calls = 0;
            settings.samples_per_candidate = 1;
            auto learning = std::make_shared<CudaProfileCache>(settings, pair_clock);
            CudaExecutionDomainOwner owner(configuration_domain_dispatch(),
                                           {99, 0, CU_CTX_MAP_HOST, 1, 256});
            expect(owner.initialize() == CUDA_SUCCESS, "unproven configured sample owner");
            auto member = owner.create_member({{1, 99}, 101});
            CudaExecutionSession session({1, 99}, 101, pair_dispatch(), limits(1000), {320}, {},
                                         learning, false, {}, std::move(member.member));
            success(session.initialize(0));
            const auto configured =
                session.context_control(CudaContextControl::kSetCache, CU_FUNC_CACHE_PREFER_SHARED);
            expect(configured.status == CUDA_SUCCESS && configured.original_status == CUDA_SUCCESS,
                   "known requested context setter");
            const auto module = session.load_ptx(kSource);
            success(module);
            const auto function = session.get_function(module.id, "kernel");
            success(function);
            const auto output = session.allocate(32);
            success(output);
            KernelInput input({function.id, 0}, output.address);
            const auto accepted = session.accept_kernel(1, input.kernel);
            expect(accepted.accepted(), "one unique configured observation");
            driver.on_launch = [] { driver.context_limits[0] += 16; };
            pair_now_ns = 1000;
            const auto issued = session.issue(session.pending_command().command);
            driver.on_launch = {};
            expect(issued.state == CudaExecutionIssueState::kIssued &&
                       issued.native_status == CUDA_SUCCESS &&
                       issued.completion_status == CUDA_SUCCESS,
                   "unexpected configuration does not forge a failed original launch");
            pair_now_ns = 1200;
            expect(session.acknowledge_result(accepted.receipt), "original native success ACK");
            (void)session.poll();
            const auto measured = learning->snapshot();
            expect(
                !measured.samples && !measured.trained_cases && measured.rejected_samples == 1 &&
                    owner.snapshot().is_quarantined && !driver.context_destroys,
                "native final-retirement proof rejects changed configuration before sample commit");
            // Intentional native uncertainty retains the complete owner. This
            // child is not a healthy-cleanup/leak-free positive control.
            _exit(0);
        } catch (const std::exception& error) {
            std::cerr << "Configured observation child: " << error.what() << '\n';
            _exit(1);
        }
    }
    int status{};
    expect(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "configured sample uncertainty control exits with its explicit retained-owner oracle");
}
void configured_context_measurements_are_not_epochs() {
    driver = {};
    driver.requires_explicit_cleanup = true;
    driver.next_claimed = 8;
    CudaProfileConfig settings;
    settings.workers = {320};
    settings.warmup_calls = 0;
    settings.samples_per_candidate = 1;
    auto learning = std::make_shared<CudaProfileCache>(settings, pair_clock);
    CudaExecutionDomainOwner owner(configuration_domain_dispatch(),
                                   {99, 0, CU_CTX_MAP_HOST, 2, 256});
    expect(owner.initialize() == CUDA_SUCCESS, "configured profile native domain");
    auto member = owner.create_member({{1, 99}, 101});
    auto inspection = member.member;
    std::array<CudaProfileGeometry, 2> geometries{};
    {
        CudaExecutionSession session({1, 99}, 101, pair_dispatch(), limits(1000), {320}, {},
                                     learning, false, {}, std::move(member.member));
        success(session.initialize(0));
        const auto set_cache = [&](std::uint32_t cache) {
            const auto result = session.context_control(CudaContextControl::kSetCache, cache);
            expect(result.status == CUDA_SUCCESS && result.original_status == CUDA_SUCCESS,
                   "requested original setter and canonical native readback");
        };
        set_cache(CU_FUNC_CACHE_PREFER_SHARED);
        const auto module = session.load_ptx(kSource);
        success(module);
        const auto function = session.get_function(module.id, "kernel");
        success(function);
        const auto buffer = session.allocate(32);
        success(buffer);
        std::uint64_t sequence{};
        for (std::size_t profile = 0; profile < geometries.size(); ++profile) {
            if (profile) set_cache(CU_FUNC_CACHE_PREFER_L1);
            const auto observed = inspection.configuration();
            expect(observed.status == CUDA_SUCCESS && (observed.available & 7U) == 7U &&
                       observed.values[1] ==
                           (profile ? CU_FUNC_CACHE_PREFER_L1 : CU_FUNC_CACHE_PREFER_SHARED),
                   "CPU configuration view exposes exact normalized values, never an epoch key");
            geometries[profile] = {{8, 1, 1},
                                   {32, 1, 1},
                                   0,
                                   glimmer::transform::PtxExecutionKind::kPreemptive,
                                   {observed.available, observed.values}};
            for (unsigned int candidate = 0; candidate < 2; ++candidate) {
                KernelInput input({function.id, 0}, buffer.address);
                const auto accepted = session.accept_kernel(++sequence, input.kernel);
                expect(accepted.accepted(), "distinct configured original/private input");
                pair_now_ns = 1000;
                const auto issued = session.issue(session.pending_command().command);
                expect(issued.state == CudaExecutionIssueState::kIssued &&
                           issued.native_status == CUDA_SUCCESS &&
                           issued.completion_status == CUDA_SUCCESS,
                       "real configured Session observer begins at original native issue");
                expect(inspection.configuration().status == CUDA_ERROR_NOT_READY,
                       "unobserved implicit effects cannot qualify another configured identity");
                pair_now_ns = candidate ? 1100 : 1200;
                expect(session.acknowledge_result(accepted.receipt) &&
                           session.poll().status == CUDA_SUCCESS,
                       "configured measurement ends only at actual retirement");
                success(session.deactivate());
                success(session.activate());
                expect(inspection.configuration().status == CUDA_SUCCESS,
                       "actual quiescent native verification restores configuration freshness");
            }
        }
        expect(learning->snapshot().cases == 2 && learning->snapshot().trained_cases == 2 &&
                   learning->snapshot().samples == 4,
               "two normalized configurations train independent complete candidates");
        const auto old_epoch = inspection.view().configuration_epoch;
        set_cache(CU_FUNC_CACHE_PREFER_SHARED);
        expect(inspection.view().configuration_epoch > old_epoch &&
                   inspection.configuration().values == geometries[0].context.values,
               "returning to the same values is not returning to the same epoch number");
        KernelInput fifth({function.id, 0}, buffer.address);
        const auto accepted = session.accept_kernel(++sequence, fifth.kernel);
        const auto issued = session.issue(session.pending_command().command);
        expect(accepted.accepted() && issued.state == CudaExecutionIssueState::kIssued &&
                   issued.native_status == CUDA_SUCCESS &&
                   issued.completion_status == CUDA_SUCCESS &&
                   session.acknowledge_result(accepted.receipt) &&
                   session.poll().status == CUDA_SUCCESS && learning->snapshot().cases == 2 &&
                   learning->snapshot().samples == 4 &&
                   driver.physical_workers == std::vector<unsigned int>{8, 320, 8, 320, 320},
               "exact same verified configuration reuses its measured candidate without replay");
        std::array<std::uint32_t, 8> output{};
        success(session.download({buffer.id, 0}, std::as_writable_bytes(std::span{output})));
        expect(std::ranges::all_of(output, [](auto value) { return value == 35; }),
               "all five distinct non-idempotent configured commands execute once");
        expect(session.poll().status == CUDA_SUCCESS, "retire configured readback tail");
        success(session.shutdown());
    }
    inspection = {};
    const auto archive = learning->encode("configured-session-profile");
    CudaProfileCache restored(settings, pair_clock, false);
    expect(archive && restored.restore(*archive, "configured-session-profile") &&
               restored.snapshot().loaded_cases == 2,
           "actual configured Session measurements restore in the V2 single archive");
    const auto function = restored.bind({{std::byte{1}}, 12060, CUDA_VERSION, kCudaPtxGeneratorAbi},
                                        kSource, {}, "kernel");
    for (const auto& geometry : geometries)
        expect(restored.original_measurement(function, geometry).has_value(),
               "both exact configured original measurements remain qualified");
    auto initial = geometries[0];
    initial.context = {};
    expect(!restored.original_measurement(function, initial),
           "configured measurements cannot be relabeled as legacy initial-only data");
    expect(owner.close() == CUDA_SUCCESS && owner.shutdown() == CUDA_SUCCESS &&
               driver.live_resources() == 0 && driver.context_destroys == 1,
           "configured profiling retains and cleans its actual native owner once");
}
void isolated_pair_samples_execute_once() {
    driver = {};
    driver.requires_explicit_cleanup = true;
    auto singles = pair_singles();
    const CudaPairProfileConfig config{
        .max_candidates = 4, .warmup_calls = 0, .samples_per_candidate = 1};
    CudaPairProfileCache learning(singles, config, pair_clock);
    CudaExecutionDomainOwner owner(domain_dispatch(), {99, 0, CU_CTX_MAP_HOST, 2, 256});
    expect(owner.initialize() == CUDA_SUCCESS, "isolated sample physical owner");
    auto first = owner.create_member({{1, 99}, 101}), second = owner.create_member({{2, 99}, 202});
    {
        CudaExecutionSession left({1, 99}, 101, pair_dispatch(), limits(1000), {320}, {}, singles,
                                  false, {}, std::move(first.member));
        CudaExecutionSession right({2, 99}, 202, pair_dispatch(), limits(2000), {320}, {}, singles,
                                   false, {}, std::move(second.member));
        std::array<CudaExecutionSession*, 2> sessions{&left, &right};
        std::array<CudaSessionResult, 2> buffers{}, functions{};
        for (std::size_t leg = 0; leg < 2; ++leg) {
            success(sessions[leg]->initialize(0));
            buffers[leg] = sessions[leg]->allocate(32);
            success(buffers[leg]);
            const auto module = sessions[leg]->load_ptx(kSource);
            success(module);
            functions[leg] = sessions[leg]->get_function(module.id, "kernel");
            success(functions[leg]);
            success(sessions[leg]->deactivate());
        }
        constexpr std::array workers{80U, 160U};
        for (std::uint64_t sequence = 1; sequence <= 4; ++sequence) {
            std::array<glimmer::backend::ExecutionAcceptance, 2> accepted{};
            std::array<CudaSessionPairKernel, 2> kernels{};
            for (std::size_t leg = 0; leg < 2; ++leg) {
                success(sessions[leg]->activate());
                KernelInput input({functions[leg].id, 0}, buffers[leg].address);
                accepted[leg] = sessions[leg]->accept_kernel(sequence, input.kernel);
                expect(accepted[leg].accepted(),
                       "new unique logical input, never replayed for profiling");
                const auto prepared =
                    sessions[leg]->prepare_pair_kernel(sessions[leg]->pending_command().command);
                expect(prepared.status == CUDA_SUCCESS, "actual sample resource catalog");
                kernels[leg] = prepared.kernel;
            }
            driver.next_context_query_status = CUDA_ERROR_INVALID_CONTEXT;
            if (sequence == 1) {
                auto budget = config;
                budget.max_bytes = 1;
                CudaPairProfileCache limited(singles, budget, pair_clock);
                expect(
                    CudaExecutionSession::reserve_pair_sample(left, kernels[0], right, kernels[1],
                                                              limited, workers)
                                .status == CUDA_ERROR_OUT_OF_MEMORY &&
                        driver.next_context_query_status == CUDA_ERROR_INVALID_CONTEXT &&
                        !driver.launches && !limited.snapshot().active &&
                        left.pending_command().command.description() ==
                            kernels[0].description().pending &&
                        right.pending_command().command.description() ==
                            kernels[1].description().pending,
                    "sampling capacity failure is explicit, SDK-free and does not select geometry");
            }
            auto sampled = CudaExecutionSession::reserve_pair_sample(left, kernels[0], right,
                                                                     kernels[1], learning, workers);
            expect(sampled.status == CUDA_SUCCESS && sampled.reservation &&
                       sampled.selection.is_profile_sample() &&
                       !sampled.selection.choice().is_trained &&
                       sampled.reservation.candidate == sequence - 1 &&
                       driver.next_context_query_status == CUDA_ERROR_INVALID_CONTEXT &&
                       learning.snapshot().active == 1,
                   "SDK-free sample selection binds one actual checked fresh pair");
            expect(CudaExecutionSession::reserve_pair_sample(left, kernels[0], right, kernels[1],
                                                             learning, workers)
                               .status != CUDA_SUCCESS &&
                       driver.next_context_query_status == CUDA_ERROR_INVALID_CONTEXT,
                   "stale duplicate sampling cannot replace either attached observer");
            driver.next_context_query_status = CUDA_SUCCESS;
            pair_now_ns = 1000;
            driver.next_claimed = sequence == 1 ? 3 : 8;
            const auto issued = CudaExecutionSession::issue_pair(
                left, right, sampled.selection, pair_authorization(sampled.selection), {101, 202});
            expect(issued.status == CUDA_SUCCESS && issued.attempted[0] && issued.attempted[1],
                   "exact isolated native pair");
            pair_now_ns = 2000;
            for (std::size_t leg = 0; leg < 2; ++leg)
                expect(sessions[leg]->acknowledge_result(accepted[leg].receipt) &&
                           sessions[leg]->poll().status == CUDA_SUCCESS,
                       "SDK ACK and independent actual stop");
            if (sequence == 1) {
                expect(learning.snapshot().active == 1 && !learning.snapshot().samples,
                       "partial physical stop cannot commit a logical joint sample");
                const auto partial = left.prepare_pair_kernel(left.pending_command().command);
                expect(CudaExecutionSession::reserve_pair_sample(left, partial.kernel, right,
                                                                 kernels[1], learning, workers)
                               .status != CUDA_SUCCESS,
                       "partial work cannot begin another sample or replace its observers");
                driver.next_claimed = 8;
                pair_now_ns = 3000;
                for (auto* session : sessions)
                    expect(session->issue(session->pending_command().command).state ==
                               CudaExecutionIssueState::kIssued,
                           "same-counter exact retained continuation");
                pair_now_ns = 4000;
                for (auto* session : sessions)
                    expect(session->poll().status == CUDA_SUCCESS,
                           "actual final logical retirement");
            }
            expect(learning.snapshot().samples == sequence && !learning.snapshot().active &&
                       !learning.snapshot().rejected_samples && !learning.snapshot().cancelled,
                   "one sample per two distinct logical inputs, not per physical epoch");
        }
        const auto measured = learning.snapshot();
        const auto archive = learning.encode("isolated-pair-backend");
        CudaPairProfileCache restored(singles, config, pair_clock, false);
        expect(
            measured.trained_pairs == 1 && archive &&
                restored.restore(*archive, "isolated-pair-backend") &&
                restored.snapshot().trained_pairs == 1 && restored.snapshot().loaded_pairs == 1 &&
                driver.launches == 10 && driver.counter_initializations == 8,
            "complete real-observer archive restores; continuation never reinitializes counters");
        for (std::size_t leg = 0; leg < 2; ++leg) {
            std::array<std::uint32_t, 8> output{};
            success(sessions[leg]->download({buffers[leg].id, 0},
                                            std::as_writable_bytes(std::span{output})));
            expect(std::ranges::all_of(output, [](auto value) { return value == 28; }),
                   "all four non-idempotent outputs execute exactly once");
            expect(sessions[leg]->poll().status == CUDA_SUCCESS, "retire actual readback tail");
        }
        success(left.shutdown());
        success(right.activate());
        success(right.shutdown());
    }
    expect(owner.close() == CUDA_SUCCESS && owner.shutdown() == CUDA_SUCCESS &&
               driver.live_resources() == 0 && driver.context_destroys == 1,
           "isolated samples retain and clean independent owners normally");
}
void actual_pending_pairs_preserve_independent_ownership(PairScenario scenario) {
    driver = {};
    driver.requires_explicit_cleanup = true;
    auto singles = pair_singles();
    CudaExecutionDomainOwner owner(domain_dispatch(), {99, 0, CU_CTX_MAP_HOST, 2, 256});
    expect(owner.initialize() == CUDA_SUCCESS, "actual pair physical owner initializes once");
    auto left_member = owner.create_member({{1, 99}, 101});
    auto right_member = owner.create_member({{2, 99}, 202});
    {
        CudaExecutionSession left({1, 99}, 101, pair_dispatch(), limits(1000), {320}, {}, singles,
                                  false, {}, std::move(left_member.member));
        CudaExecutionSession right({2, 99}, 202, pair_dispatch(), limits(2000), {320}, {}, singles,
                                   false, {}, std::move(right_member.member));
        std::array<CudaExecutionSession*, 2> sessions{&left, &right};
        std::array<CudaSessionResult, 2> buffers{};
        std::array<glimmer::backend::ExecutionAcceptance, 2> accepted{};
        std::array<CudaSessionPairKernel, 2> kernels{};
        for (std::size_t leg = 0; leg < sessions.size(); ++leg) {
            auto& session = *sessions[leg];
            success(session.initialize(0));
            buffers[leg] = session.allocate(32);
            const auto module = session.load_ptx(kSource);
            success(buffers[leg]);
            success(module);
            const auto function = session.get_function(module.id, "kernel");
            success(function);
            KernelInput input({function.id, 0}, buffers[leg].address);
            accepted[leg] = session.accept_kernel(1, input.kernel);
            const auto prepared = session.prepare_pair_kernel(session.pending_command().command);
            expect(accepted[leg].accepted() && prepared.status == CUDA_SUCCESS && prepared.kernel &&
                       prepared.disposition == CudaSessionPairKernelDisposition::kPrepared,
                   "actual exact pending commands carry private SDK resource metadata");
            kernels[leg] = prepared.kernel;
            success(session.deactivate());
        }
        const bool is_cold = scenario == PairScenario::kColdOriginal;
        auto measurements = pair_measurements(
            singles, {kernels[0].description(), kernels[1].description()}, is_cold);
        constexpr std::array workers{80U, 160U};
        driver.next_context_query_status = CUDA_ERROR_INVALID_CONTEXT;
        const std::array original_pending{left.pending_command().command.description(),
                                          right.pending_command().command.description()};
        const auto foreign = left.prepare_pair_kernel(right.pending_command().command);
        const auto empty = left.prepare_pair_kernel({});
        const auto foreign_refresh =
            left.refresh_pair_kernel(kernels[1], left.pending_command().command);
        expect(foreign.status == CUDA_ERROR_INVALID_HANDLE &&
                   foreign.disposition == CudaSessionPairKernelDisposition::kFailed &&
                   !foreign.kernel && empty.status == CUDA_ERROR_INVALID_HANDLE &&
                   empty.disposition == CudaSessionPairKernelDisposition::kFailed &&
                   !empty.kernel && foreign_refresh.status == CUDA_ERROR_INVALID_HANDLE &&
                   foreign_refresh.disposition == CudaSessionPairKernelDisposition::kFailed &&
                   !foreign_refresh.kernel &&
                   driver.next_context_query_status == CUDA_ERROR_INVALID_CONTEXT &&
                   !driver.launches && !driver.counter_initializations,
               "foreign or missing pair capabilities fail before native proof and never become "
               "exclusions");
        for (unsigned int repeat = 0; repeat < 3; ++repeat) {
            const auto evaluated = CudaExecutionSession::evaluate_pair(
                left, kernels[0], right, kernels[1], *measurements, workers);
            expect(evaluated.status == CUDA_SUCCESS && evaluated.choice.is_shared == !is_cold &&
                       left.pending_command().command.description() == original_pending[0] &&
                       right.pending_command().command.description() == original_pending[1] &&
                       left.persistent_progress(1)->workers == 320 &&
                       right.persistent_progress(1)->workers == 320 &&
                       driver.next_context_query_status == CUDA_ERROR_INVALID_CONTEXT &&
                       !driver.launches && !driver.counter_initializations,
                   "readonly pair evaluation preserves both exact command geometries without SDK "
                   "effects");
        }
        expect(
            CudaExecutionSession::evaluate_pair(left, kernels[1], right, kernels[0], *measurements,
                                                workers)
                        .status == CUDA_ERROR_INVALID_HANDLE &&
                CudaExecutionSession::evaluate_pair(left, kernels[0], left, kernels[0],
                                                    *measurements, workers)
                        .status == CUDA_ERROR_INVALID_VALUE &&
                CudaExecutionSession::evaluate_pair(left, kernels[0], right, kernels[1],
                                                    *measurements, {})
                        .status == CUDA_ERROR_INVALID_VALUE &&
                driver.next_context_query_status == CUDA_ERROR_INVALID_CONTEXT,
            "foreign/same-Session/empty-catalog evaluations reject without callbacks or mutation");
        auto selected = CudaExecutionSession::select_pair(left, kernels[0], right, kernels[1],
                                                          *measurements, workers);
        expect(selected.status == CUDA_SUCCESS && selected.selection &&
                   selected.selection.choice().is_shared == !is_cold &&
                   driver.next_context_query_status == CUDA_ERROR_INVALID_CONTEXT &&
                   !driver.launches && !driver.counter_initializations,
               "CPU readonly selection performs no SDK/replay/counter reset");
        driver.next_context_query_status = CUDA_SUCCESS;
        auto authorization = pair_authorization(selected.selection);
        expect(authorization.is_valid && authorization.grant_count == (is_cold ? 1U : 2U),
               "actual commands receive only exact serial/shared Core grants");
        auto forged = authorization;
        ++forged.grants[0].command.geometry_epoch;
        driver.next_context_query_status = CUDA_ERROR_INVALID_CONTEXT;
        expect(CudaExecutionSession::issue_pair(left, right, selected.selection, forged, {101, 202})
                           .status != CUDA_SUCCESS &&
                   driver.next_context_query_status == CUDA_ERROR_INVALID_CONTEXT &&
                   !driver.launches,
               "stale grant rejects before any native context proof");
        forged = authorization;
        forged.priority.decisions[forged.priority.size++] = {
            .session = 303, .action = glimmer::core::ExecutionPriorityAction::kExecute};
        expect(CudaExecutionSession::issue_pair(left, right, selected.selection, forged, {101, 202})
                           .status != CUDA_SUCCESS &&
                   driver.next_context_query_status == CUDA_ERROR_INVALID_CONTEXT &&
                   !driver.launches,
               "an unrelated ordinary Execute cannot ride an exact pair grant");
        if (!is_cold) {
            const auto stale_command = right.pending_command().command;
            expect(right.select_pending_kernel(right.pending_command().command, {.workers = 80})
                           .status == CUDA_SUCCESS,
                   "independent second-leg selection invalidates its prior pair capability");
            const auto stale = right.prepare_pair_kernel(stale_command);
            const auto stale_refresh = right.refresh_pair_kernel(kernels[1], stale_command);
            expect(stale.status == CUDA_ERROR_INVALID_HANDLE &&
                       stale.disposition == CudaSessionPairKernelDisposition::kFailed &&
                       !stale.kernel && stale_refresh.status == CUDA_ERROR_INVALID_HANDLE &&
                       stale_refresh.disposition == CudaSessionPairKernelDisposition::kFailed &&
                       !stale_refresh.kernel &&
                       driver.next_context_query_status == CUDA_ERROR_INVALID_CONTEXT &&
                       !driver.launches && !driver.counter_initializations,
                   "stale preparation and refresh remain failures before SDK proof or counter "
                   "mutation");
            expect(CudaExecutionSession::evaluate_pair(left, kernels[0], right, kernels[1],
                                                       *measurements, workers)
                               .status == CUDA_ERROR_INVALID_HANDLE &&
                       driver.next_context_query_status == CUDA_ERROR_INVALID_CONTEXT,
                   "stale speculative metadata cannot evaluate a different geometry epoch");
            expect(CudaExecutionSession::issue_pair(left, right, selected.selection, authorization,
                                                    {101, 202})
                               .status != CUDA_SUCCESS &&
                       driver.next_context_query_status == CUDA_ERROR_INVALID_CONTEXT &&
                       !driver.launches,
                   "stale second leg rejects BOTH grants before first-leg native effects");
            const auto left_fresh =
                left.refresh_pair_kernel(kernels[0], left.pending_command().command);
            const auto right_fresh =
                right.refresh_pair_kernel(kernels[1], right.pending_command().command);
            expect(left_fresh.status == CUDA_SUCCESS && right_fresh.status == CUDA_SUCCESS &&
                       left_fresh.disposition == CudaSessionPairKernelDisposition::kPrepared &&
                       right_fresh.disposition == CudaSessionPairKernelDisposition::kPrepared &&
                       left_fresh.kernel && right_fresh.kernel &&
                       driver.next_context_query_status == CUDA_ERROR_INVALID_CONTEXT,
                   "fresh geometry refresh is explicitly prepared and stays entirely CPU-only");
            selected = CudaExecutionSession::select_pair(
                left, left_fresh.kernel, right, right_fresh.kernel, *measurements, workers);
            expect(selected.status == CUDA_SUCCESS && selected.selection.choice().is_shared,
                   "fresh exact geometry observations can select the trained pair again");
            authorization = pair_authorization(selected.selection);
        }
        driver.next_context_query_status = CUDA_SUCCESS;
        success(left.activate());
        const bool has_native_error = scenario == PairScenario::kSecondNativeError;
        if (has_native_error) {
            driver.on_launch = [&] {
                if (driver.launches == 1)
                    driver.next_launch_status = CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES;
            };
        }
        driver.query_status = CUDA_ERROR_NOT_READY;
        const auto issued = CudaExecutionSession::issue_pair(left, right, selected.selection,
                                                             authorization, {101, 202});
        driver.on_launch = {};
        expect(issued.status ==
                       (has_native_error ? CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES : CUDA_SUCCESS) &&
                   issued.attempted[0] && issued.attempted[1] == !is_cold &&
                   driver.launches == (is_cold ? 1U : 2U),
               "exact grants execute once; second failure cannot roll back the first native issue");
        if (has_native_error) {
            expect(issued.issues[0].native_status == CUDA_SUCCESS &&
                       issued.issues[1].native_status == CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES &&
                       left.snapshot().submissions.awaiting_physical_stop == 1 &&
                       right.snapshot().submissions.awaiting_physical_stop == 1,
                   "successful work and failed-call tail remain independently retained");
        }
        const auto launches = driver.launches;
        driver.next_context_query_status = CUDA_ERROR_INVALID_CONTEXT;
        expect(CudaExecutionSession::issue_pair(left, right, selected.selection, authorization,
                                                {101, 202})
                           .status != CUDA_SUCCESS &&
                   driver.next_context_query_status == CUDA_ERROR_INVALID_CONTEXT &&
                   driver.launches == launches,
               "duplicate pair cannot substitute another command or reenter the SDK");
        driver.next_context_query_status = CUDA_SUCCESS;
        expect(left.acknowledge_result(accepted[0].receipt), "first logical result ACK");
        if (!is_cold)
            expect(right.acknowledge_result(accepted[1].receipt), "second logical result ACK");
        driver.query_status = CUDA_SUCCESS;
        expect(left.poll().status == CUDA_SUCCESS && right.poll().status == CUDA_SUCCESS,
               "actual event/readback proofs, not ACK, supply independent physical stop");
        if (!is_cold) {
            expect(left.persistent_progress(1)->completed_blocks == 3,
                   "first partial preserves actual incomplete non-idempotent computation");
            if (!has_native_error) {
                const auto left_catalog = left.prepare_pair_kernel(left.pending_command().command);
                const auto right_catalog =
                    right.prepare_pair_kernel(right.pending_command().command);
                CudaPairProfileCache cold(singles, {}, pair_clock, false);
                const auto serial = CudaExecutionSession::select_pair(
                    left, left_catalog.kernel, right, right_catalog.kernel, cold, workers);
                const auto serial_plan = pair_authorization(serial.selection);
                expect(serial.status == CUDA_SUCCESS && !serial.selection.choice().is_shared &&
                           serial_plan.grant_count == 1 &&
                           serial.selection.kernels()[0].pending.command.is_resume &&
                           serial.selection.kernels()[0].variant.workers == 160 &&
                           serial.selection.kernels()[1].variant.workers == 160,
                       "cache miss preserves partial family/counter for serial continuation");
                driver.next_claimed = 8;
                const auto resumed = CudaExecutionSession::issue_pair(left, right, serial.selection,
                                                                      serial_plan, {101, 202});
                expect(resumed.status == CUDA_SUCCESS && resumed.attempted[0] &&
                           !resumed.attempted[1] && left.poll().status == CUDA_SUCCESS &&
                           right.persistent_progress(1)->completed_blocks == 3,
                       "serial fallback resumes only the exact selected partial leg");
            } else {
                driver.next_claimed = 8;
                expect(left.issue(left.pending_command().command).state ==
                               CudaExecutionIssueState::kIssued &&
                           left.poll().status == CUDA_SUCCESS,
                       "second failure preserves first-leg independent continuation");
            }
        }
        if (!has_native_error) {
            driver.next_claimed = 8;
            expect(right.issue(right.pending_command().command).state ==
                           CudaExecutionIssueState::kIssued &&
                       right.poll().status == CUDA_SUCCESS,
                   "remaining exact command drains independently without pair replay");
            if (is_cold)
                expect(right.acknowledge_result(accepted[1].receipt), "serial second host result");
        }
        for (std::size_t leg = 0; leg < sessions.size(); ++leg) {
            std::array<std::uint32_t, 8> output{};
            success(sessions[leg]->download({buffers[leg].id, 0},
                                            std::as_writable_bytes(std::span{output})));
            const auto value = has_native_error && leg == 1 ? 0U : 7U;
            expect(std::ranges::all_of(output, [value](auto actual) { return actual == value; }),
                   "all non-idempotent outputs checked, including known second-call failure");
            expect(sessions[leg]->poll().status == CUDA_SUCCESS, "retire actual copy tail");
        }
        expect(driver.previous_counters == (is_cold ? std::vector<std::uint64_t>{}
                                            : has_native_error
                                                ? std::vector<std::uint64_t>{0, 3}
                                                : std::vector<std::uint64_t>{0, 0, 3, 3}),
               "each actual private ABI retains its own counter without original replay");
        success(left.shutdown());
        success(right.activate());
        success(right.shutdown());
    }
    expect(owner.close() == CUDA_SUCCESS && owner.shutdown() == CUDA_SUCCESS &&
               driver.context_creates == 1 && driver.context_destroys == 1 &&
               driver.live_resources() == 0 && !driver.current,
           "all pair paths normally clean independent owners and one physical context");
}

void shared_nonpersistent_flags_keep_mapped_host_capability() {
    driver = {};
    driver.requires_explicit_cleanup = true;
    CudaExecutionDomainOwner owner(configuration_domain_dispatch(),
                                   {99, 0, CU_CTX_MAP_HOST, 1, 256});
    expect(owner.initialize() == CUDA_SUCCESS,
           "fresh mapped-host domain for a nonpersistent member");
    auto member = owner.create_member({{1, 99}, 101});
    {
        CudaExecutionSession session({1, 99}, 101, session_dispatch(), limits(1000), {}, {}, {},
                                     false, {}, std::move(member.member));
        success(session.initialize(0, {CU_CTX_MAP_HOST}));
        success(session.set_context_flags(CU_CTX_SCHED_YIELD));
        expect(
            session.get_context_flags().id == (CU_CTX_MAP_HOST | CU_CTX_SCHED_YIELD) &&
                driver.next_host == 0,
            "a member without counters cannot disable the domain's private mapped-host capability");
        success(session.shutdown());
    }
    expect(owner.close() == CUDA_SUCCESS && owner.shutdown() == CUDA_SUCCESS &&
               driver.live_resources() == 0 && driver.context_creates == 1 &&
               driver.context_destroys == 1,
           "zero-progress-pool member still retires its actual proxy/context once");
}

void shared_initial_flags_defer_and_restore() {
    driver = {};
    driver.requires_explicit_cleanup = true;
    CudaExecutionDomainOwner owner(configuration_domain_dispatch(),
                                   {99, 0, CU_CTX_MAP_HOST, 2, 256});
    expect(owner.initialize() == CUDA_SUCCESS, "initial logical flags domain");
    auto first = owner.create_member({{1, 99}, 101}, CU_CTX_SCHED_YIELD);
    auto second = owner.create_member({{2, 99}, 202}, CU_CTX_SCHED_SPIN);
    expect(first.status == CUDA_SUCCESS && second.status == CUDA_SUCCESS,
           "cold independent flag profiles");
    {
        CudaExecutionSession low({1, 99}, 101, session_dispatch(), limits(1000), {}, {}, {}, false,
                                 {}, std::move(first.member));
        CudaExecutionSession high({2, 99}, 202, session_dispatch(), limits(2000), {}, {}, {}, false,
                                  {}, std::move(second.member));
        success(low.initialize(0, {CU_CTX_SCHED_YIELD}));
        expect(low.get_context_flags().id == (CU_CTX_SCHED_YIELD | CU_CTX_MAP_HOST),
               "new Session materializes requested flags before resource callbacks");
        const auto buffer = low.allocate(32);
        const auto module = low.load_ptx(kSource);
        success(buffer);
        success(module);
        const auto function = low.get_function(module.id, "kernel");
        success(function);
        KernelInput input({function.id, 0}, buffer.address);
        const auto accepted = low.accept_kernel(1, input.kernel);
        auto frontier = low.capture_command_frontier(1);
        auto pending = low.pending_command();
        driver.query_status = CUDA_ERROR_NOT_READY;
        expect(accepted.accepted() && frontier.status == CUDA_SUCCESS && pending.command &&
                   low.issue(pending.command).state == CudaExecutionIssueState::kIssued &&
                   low.acknowledge_result(accepted.receipt),
               "real low tail remains pending");
        success(low.deactivate());
        const auto controls = driver.context_control_calls;
        const auto deferred = high.initialize(0, {CU_CTX_SCHED_SPIN});
        expect(deferred.disposition == CudaSessionDisposition::kDeferred &&
                   !deferred.original_status && !high.snapshot().has_context &&
                   driver.context_control_calls == controls && !driver.current,
               "new flags defer before attachment/native mutation while sibling is physical");
        success(low.activate());
        driver.query_status = CUDA_SUCCESS;
        expect(low.poll().status == CUDA_SUCCESS &&
                   low.query_frontier(frontier.frontier) == CUDA_SUCCESS,
               "only actual low completion clears its physical owner");
        std::array<std::uint32_t, 8> actual{};
        success(low.download({buffer.id, 0}, std::as_writable_bytes(std::span{actual})));
        expect(std::ranges::all_of(actual, [](auto value) { return value == 7; }),
               "every non-idempotent low output is exact before context retirement");
        expect(low.poll().status == CUDA_SUCCESS, "retire the synchronous download tail");
        frontier.frontier = {};
        success(low.deactivate());
        success(high.initialize(0, {CU_CTX_SCHED_SPIN}));
        expect(high.get_context_flags().id == (CU_CTX_SCHED_SPIN | CU_CTX_MAP_HOST) &&
                   driver.context_creates == 1,
               "retry only deferred initialization, without a second physical context");
        success(high.shutdown());
        expect(!driver.context_destroys && owner.snapshot().attached_members == 1,
               "member shutdown cannot destroy a surviving sibling");
        success(low.activate());
        expect(low.get_context_flags().id == (CU_CTX_SCHED_YIELD | CU_CTX_MAP_HOST),
               "surviving logical flags restore after sibling shutdown");
        success(low.shutdown());
    }
    expect(owner.close() == CUDA_SUCCESS && owner.shutdown() == CUDA_SUCCESS &&
               driver.context_creates == 1 && driver.context_destroys == 1 &&
               !driver.live_resources(),
           "initial flag profiles preserve exact final domain and resource cleanup");
}

void shared_configuration_and_retained_commands() {
    driver = {};
    driver.requires_explicit_cleanup = true;
    CudaExecutionDomainOwner owner(configuration_domain_dispatch(),
                                   {99, 0, CU_CTX_MAP_HOST, 2, 256});
    expect(owner.initialize() == CUDA_SUCCESS, "fresh actual configuration baseline");
    auto first = owner.create_member({{1, 99}, 101});
    auto second = owner.create_member({{2, 99}, 202});
    auto first_member = first.member;
    auto second_member = second.member;
    {
        CudaExecutionSession low({1, 99}, 101, session_dispatch(), limits(1000), {320}, {}, {},
                                 false, {}, std::move(first.member));
        CudaExecutionSession high({2, 99}, 202, session_dispatch(), limits(2000), {320}, {}, {},
                                  false, {}, std::move(second.member));
        success(low.initialize(0));
        expect(!low.can_share_configuration(high),
               "a cold unattached sibling is not an execution sharing capability");
        success(low.set_context_flags(CU_CTX_SCHED_YIELD));
        const auto set_control = [&](CudaContextControl operation, std::uint32_t argument,
                                     std::uint64_t value = 0) {
            const auto result = low.context_control(operation, argument, value);
            expect(result.status == CUDA_SUCCESS && result.original_status == CUDA_SUCCESS,
                   "shared requested Driver configuration retains exact native success");
        };
        set_control(CudaContextControl::kSetCache, CU_FUNC_CACHE_PREFER_L1);
        set_control(CudaContextControl::kSetSharedMemory,
                    CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE);
        set_control(CudaContextControl::kSetLimit, CU_LIMIT_STACK_SIZE, 1500);
        expect(first_member.view().configuration_epoch == 5 &&
                   low.context_control(CudaContextControl::kGetLimit, CU_LIMIT_STACK_SIZE).value ==
                       1536,
               "logical epoch and SDK-normalized limit are retained independently");
        const auto buffer = low.allocate(32);
        const auto module = low.load_ptx(kSource);
        success(buffer);
        success(module);
        const auto function = low.get_function(module.id, "kernel");
        success(function);
        success(low.deactivate());
        success(high.initialize(0));
        expect(!low.can_share_configuration(high),
               "one native context does not imply equal logical configurations");
        expect(high.get_context_flags().id == CU_CTX_MAP_HOST &&
                   high.context_control(CudaContextControl::kGetCache, 0).value ==
                       CU_FUNC_CACHE_PREFER_NONE &&
                   high.context_control(CudaContextControl::kGetLimit, CU_LIMIT_STACK_SIZE).value ==
                       1024 &&
                   second_member.view().configuration_epoch == 1,
               "new Session restores the fresh baseline before its own resource setup");
        success(high.deactivate());
        success(low.activate());
        expect(low.get_context_flags().id == (CU_CTX_SCHED_YIELD | CU_CTX_MAP_HOST) &&
                   driver.cache == CU_FUNC_CACHE_PREFER_L1 && driver.context_limits[0] == 1536,
               "reactivation restores only this Session's canonical native configuration");
        KernelInput input({function.id, 0}, buffer.address);
        const auto accepted = low.accept_kernel(1, input.kernel);
        auto frontier = low.capture_command_frontier(1);
        const auto pending = low.pending_command();
        expect(accepted.accepted() && frontier.status == CUDA_SUCCESS,
               "exact retained low command");
        expect(
            pending.command && pending.command.description().control_epoch == 5 &&
                pending.command.description().member_generation == first_member.view().generation &&
                pending.command.description().configuration_epoch == 5 &&
                pending.command.description().physical_configuration_epoch ==
                    first_member.view().physical_configuration_epoch,
            "pending command binds actual logical control/member/materialization epochs");
        driver.query_status = CUDA_ERROR_NOT_READY;
        expect(low.issue(pending.command).state == CudaExecutionIssueState::kIssued &&
                   low.acknowledge_result(accepted.receipt),
               "physical low work survives host ACK");
        const auto controls = driver.context_control_calls;
        const auto limit_during_work =
            low.context_control(CudaContextControl::kGetLimit, CU_LIMIT_STACK_SIZE);
        expect(limit_during_work.disposition == CudaSessionDisposition::kDeferred &&
                   !limit_during_work.original_status,
               "unobserved kernel-driven effects cannot leak through a logical limit query");
        expect(low.context_control(CudaContextControl::kSetCache, 0).disposition ==
                       CudaSessionDisposition::kDeferred &&
                   driver.context_control_calls == controls,
               "own held physical command cannot mutate configuration after host ACK");
        success(low.deactivate());
        expect(high.activate().disposition == CudaSessionDisposition::kDeferred &&
                   driver.context_control_calls == controls && !driver.current,
               "different high configuration defers and restores previous NULL TLS without SDK "
               "setters");
        success(low.activate());
        expect(low.request_retreat(1) == CUDA_SUCCESS, "request incomplete actual retreat");
        driver.query_status = CUDA_SUCCESS;
        const auto stopped = low.poll();
        expect(stopped.status == CUDA_SUCCESS && stopped.paused == 1 &&
                   low.persistent_progress(1)->completed_blocks == 3,
               "only actual trailing event/readback supplies incomplete physical stop");
        const auto old_resume = low.pending_command();
        expect(old_resume.command && old_resume.command.description().command.is_resume &&
                   old_resume.command.description().command.physical_generation == 1,
               "known-stopped retained command has one exact resumable generation");
        success(low.deactivate());
        success(high.activate());
        const auto changed_runtime =
            high.context_control(CudaContextControl::kSetCache, cudaFuncCachePreferShared, 0, true);
        expect(changed_runtime.status == CUDA_SUCCESS &&
                   changed_runtime.original_status == cudaErrorInvalidDeviceFunction &&
                   second_member.view().configuration_epoch == 2 &&
                   driver.cache == CU_FUNC_CACHE_PREFER_SHARED,
               "actual Runtime error status is not remapped; changed post-state remains captured");
        const auto unmaterialized = low.pending_command();
        driver.next_context_query_status = CUDA_ERROR_INVALID_CONTEXT;
        expect(low.prepare_pair_kernel(unmaterialized.command).status == CUDA_ERROR_NOT_READY &&
                   driver.next_context_query_status == CUDA_ERROR_INVALID_CONTEXT &&
                   !low.snapshot().is_quarantined && !high.snapshot().is_quarantined,
               "known unmaterialized resource queries defer before SDK without quarantining the "
               "cohort");
        driver.next_context_query_status = CUDA_SUCCESS;
        const auto resources = owner.snapshot().reserved_device_bytes;
        success(high.shutdown());
        expect(driver.context_destroys == 0 && owner.snapshot().reserved_device_bytes < resources,
               "sibling cleanup releases only its member and never resets shared "
               "configuration/context");
        success(low.activate());
        expect(driver.cache == CU_FUNC_CACHE_PREFER_L1 &&
                   low.context_control(CudaContextControl::kGetLimit, CU_LIMIT_STACK_SIZE).value ==
                       1536,
               "remaining owner restores its exact state after sibling's mutation and cleanup");
        driver.next_context_query_status = CUDA_ERROR_INVALID_CONTEXT;
        expect(low.issue(old_resume.command).native_status == CUDA_ERROR_INVALID_HANDLE &&
                   driver.next_context_query_status == CUDA_ERROR_INVALID_CONTEXT &&
                   driver.launches == 1,
               "configuration restoration invalidates an old grant before any SDK proof or launch");
        driver.next_context_query_status = CUDA_SUCCESS;
        const auto restored = low.pending_command();
        expect(restored.command &&
                   restored.command.description().control_epoch ==
                       old_resume.command.description().control_epoch &&
                   restored.command.description().configuration_epoch ==
                       old_resume.command.description().configuration_epoch &&
                   restored.command.description().physical_configuration_epoch !=
                       old_resume.command.description().physical_configuration_epoch,
               "same logical configuration cannot reuse a prior physical materialization grant");
        driver.next_claimed = 8;
        expect(low.issue(restored.command).state == CudaExecutionIssueState::kIssued &&
                   low.poll().status == CUDA_SUCCESS &&
                   low.query_frontier(frontier.frontier) == CUDA_SUCCESS &&
                   driver.previous_counters == std::vector<std::uint64_t>{0, 3},
               "same prepared command/counter resumes exactly once after configuration switching");
        std::array<std::uint32_t, 8> output{};
        success(low.download({buffer.id, 0}, std::as_writable_bytes(std::span{output})));
        expect(std::ranges::all_of(output, [](auto value) { return value == 7; }),
               "all non-idempotent outputs remain exact across member configuration changes");
        expect(low.poll().status == CUDA_SUCCESS,
               "retire actual private synchronous-download tail");
        frontier.frontier = {};
        success(low.shutdown());
    }
    first_member = {};
    second_member = {};
    expect(owner.close() == CUDA_SUCCESS && owner.shutdown() == CUDA_SUCCESS &&
               driver.context_creates == 1 && driver.context_destroys == 1 && !driver.current &&
               driver.live_resources() == 0,
           "configuration Sessions retire resources and native domain exactly once");
}

void shared_lifetime_quota_reentry_and_retained_continuation() {
    driver = {};
    driver.requires_explicit_cleanup = true;
    CudaExecutionDomainOwner owner(domain_dispatch(), {99, 0, CU_CTX_MAP_HOST, 2, 192});
    expect(owner.initialize() == CUDA_SUCCESS && !driver.current && driver.context_creates == 1,
           "domain creates its physical context exactly once and restores NULL TLS");
    auto first = owner.create_member({{1, 99}, 101});
    auto second = owner.create_member({{2, 99}, 202});
    expect(first.status == CUDA_SUCCESS && second.status == CUDA_SUCCESS,
           "create two exact logical member capabilities");
    auto retained_view = first.member.view();
    {
        const auto other_view = second.member.view();
        expect(retained_view.status == CUDA_SUCCESS && other_view.status == CUDA_SUCCESS &&
                   retained_view.context.native == other_view.context.native &&
                   retained_view.logical.context == 101 && other_view.logical.context == 202 &&
                   retained_view.generation != other_view.generation,
               "same physical context never merges logical identity or member generation");
    }
    {
        CudaExecutionSession first_session({1, 99}, 101, session_dispatch(), limits(1000), {320},
                                           {}, {}, false, {}, std::move(first.member));
        CudaExecutionSession second_session({2, 99}, 202, session_dispatch(), limits(2000), {320},
                                            {}, {}, false, {}, std::move(second.member));
        success(first_session.initialize(0));
        success(first_session.deactivate());
        success(second_session.initialize(0));
        expect(driver.context_creates == 1 && driver.next_host == 2 && driver.next_stream == 2 &&
                   owner.snapshot().reserved_device_bytes == 32 &&
                   first_session.can_share_configuration(second_session) &&
                   second_session.can_share_configuration(first_session) &&
                   !first_session.can_share_configuration(first_session),
               "borrowed Sessions create independent counters/hosts/proxies, not native contexts");
        const auto second_counter = driver.allocations[1].address;
        const auto second_proxy = reinterpret_cast<CUstream>(&driver.streams[1]);
        const auto second_buffer = second_session.allocate(64);
        const auto second_stream = second_session.create_stream(CU_STREAM_DEFAULT);
        const auto second_module = second_session.load_ptx(kSource);
        success(second_buffer);
        success(second_stream);
        success(second_module);
        const auto function = second_session.get_function(second_module.id, "kernel");
        success(function);
        const auto second_native_module = &driver.modules[0];
        success(second_session.deactivate());
        success(first_session.activate());
        const auto first_buffer = first_session.allocate(64);
        const auto first_stream = first_session.create_stream(CU_STREAM_DEFAULT);
        const auto first_module = first_session.load_ptx(kSource);
        const auto first_event = first_session.create_event(CU_EVENT_DISABLE_TIMING);
        for (const auto& result : {first_buffer, first_stream, first_module, first_event})
            success(result);
        expect(first_buffer.id != second_buffer.id &&
                   first_buffer.address != second_buffer.address &&
                   !first_session.release(second_buffer.id).original_status &&
                   !first_session.get_function(second_module.id, "kernel").original_status,
               "another logical member cannot release or resolve opaque resource IDs");
        expect(std::ranges::all_of(
                   std::span{driver.streams}.first(driver.next_stream),
                   [](const auto& stream) { return stream.flags == CU_STREAM_NON_BLOCKING; }),
               "logical blocking/default streams remain physically member-local nonblocking");
        bool blocks_reentry{};
        driver.on_allocate = [&] {
            const auto allocations = driver.allocation_calls;
            const auto other = second_session.allocate(4);
            const auto close = second_session.shutdown();
            blocks_reentry = other.status == CUDA_ERROR_NOT_READY &&
                             close.disposition == CudaSessionDisposition::kDeferred &&
                             driver.allocation_calls == allocations &&
                             !first_session.can_share_configuration(second_session);
        };
        const auto extra = first_session.allocate(16);
        success(extra);
        expect(blocks_reentry, "one member's native callback cannot enter or close a sibling");
        success(first_session.release(extra.id));
        const auto allocations = driver.allocation_calls;
        const auto over_quota = first_session.allocate(40);
        expect(over_quota.status == CUDA_ERROR_OUT_OF_MEMORY && !over_quota.original_status &&
                   driver.allocation_calls == allocations &&
                   owner.snapshot().reserved_device_bytes == 160,
               "aggregate device quota rejects before SDK despite independent local headroom");
        success(first_session.deactivate());
        success(second_session.activate());
        KernelInput input({function.id, second_stream.id}, second_buffer.address);
        const auto accepted = second_session.accept_kernel(1, input.kernel);
        auto frontier = second_session.capture_command_frontier(1);
        expect(accepted.accepted() && frontier.status == CUDA_SUCCESS,
               "accept one exact persistent logical command and immutable frontier");
        input.parameters.fill(std::byte{0xa5});
        driver.query_status = CUDA_ERROR_NOT_READY;
        expect(second_session.issue_next().state == CudaExecutionIssueState::kIssued &&
                   second_session.acknowledge_result(accepted.receipt) &&
                   driver.previous_counters == std::vector<std::uint64_t>{0},
               "one host ACK leaves actual first physical epoch and original ABI retained");
        success(second_session.deactivate());
        success(first_session.activate());
        const auto cleanup_calls = driver.cleanup_calls();
        first_session.close();
        const auto delayed = first_session.shutdown();
        expect(delayed.disposition == CudaSessionDisposition::kDeferred &&
                   delayed.status == CUDA_ERROR_NOT_READY &&
                   driver.cleanup_calls() == cleanup_calls && driver.context_destroys == 0 &&
                   owner.snapshot().reserved_device_bytes == 160,
               "sibling physical ownership defers cleanup before any SDK destruction or refund");
        success(first_session.deactivate());
        success(second_session.activate());
        expect(
            second_session.request_retreat(1) == CUDA_SUCCESS && second_session.poll().paused == 0,
            "retreat intent alone cannot permit sibling native cleanup");
        success(second_session.deactivate());
        success(first_session.activate());
        const auto still_delayed = first_session.shutdown();
        expect(still_delayed.disposition == CudaSessionDisposition::kDeferred &&
                   still_delayed.status == CUDA_ERROR_NOT_READY &&
                   driver.cleanup_calls() == cleanup_calls &&
                   owner.snapshot().reserved_device_bytes == 160,
               "even after retreat publication no cleanup or refund precedes actual tail stop");
        success(first_session.deactivate());
        success(second_session.activate());
        driver.query_status = CUDA_SUCCESS;
        expect(second_session.poll().paused == 1 &&
                   second_session.persistent_progress(1)->completed_blocks == 3 &&
                   second_session.query_frontier(frontier.frontier) == CUDA_ERROR_NOT_READY,
               "actual event readiness proves incomplete physical stop, not logical completion");
        success(second_session.deactivate());
        success(first_session.activate());
        success(first_session.shutdown());
        expect(driver.context_destroys == 0 && second_native_module->is_live &&
                   driver.hosts[1].is_live && driver.allocations[1].is_live &&
                   driver.allocations[1].address == second_counter &&
                   reinterpret_cast<FakeStream*>(second_proxy)->is_live &&
                   owner.snapshot().reserved_device_bytes == 80,
               "member cleanup releases only its own resources, preserving sibling continuation");
        success(second_session.activate());
        driver.next_claimed = 8;
        expect(second_session.resume(1).state == CudaExecutionIssueState::kIssued &&
                   second_session.poll().retired == 1 &&
                   driver.previous_counters == std::vector<std::uint64_t>{0, 3} &&
                   driver.physical_workers == std::vector<unsigned int>{320, 320} &&
                   driver.counter_initializations == 1 &&
                   !second_session.host_result(accepted.receipt) &&
                   second_session.query_frontier(frontier.frontier) == CUDA_SUCCESS,
               "resume uses the same once-ACKed task/counter/frontier without replay or reset");
        std::array<std::uint32_t, 16> output{};
        success(second_session.download({second_buffer.id, 0},
                                        std::as_writable_bytes(std::span{output})));
        for (std::size_t index = 0; index < output.size(); ++index)
            expect(output[index] == (index < 8 ? 7U : 0U),
                   "all non-idempotent output and unused guard words survive sibling cleanup");
        expect(second_session.poll().status == CUDA_SUCCESS, "retire actual private download tail");
        frontier = {};
        success(second_session.shutdown());
        expect(driver.context_destroys == 0 && owner.snapshot().reserved_device_bytes == 0 &&
                   driver.live_resources() == 0,
               "all member cleanup is explicit, once-only and independent of context destroy");
    }
    expect(context_set_current(nullptr) == CUDA_SUCCESS && owner.close() == CUDA_SUCCESS &&
               owner.shutdown() == CUDA_ERROR_NOT_READY && driver.context_destroys == 0,
           "a surviving exported view still pins the domain after actual member cleanup");
    retained_view = {};
    expect(owner.shutdown() == CUDA_SUCCESS && driver.context_destroys == 1 &&
               !owner.snapshot().has_context && !driver.current,
           "only the last retired physical owner destroys the shared native context");
}

void clean_initialization_rejections_preserve_other_members() {
    for (unsigned int failure = 0; failure < 4; ++failure) {
        driver = {};
        driver.requires_explicit_cleanup = true;
        CudaExecutionDomainOwner owner(domain_dispatch(),
                                       {99, 0, CU_CTX_MAP_HOST, 2, failure == 0 ? 8U : 64U});
        expect(owner.initialize() == CUDA_SUCCESS, "initialize clean failure cohort");
        auto first = owner.create_member({{1, 99}, 101});
        auto second = owner.create_member({{2, 99}, 202});
        expect(first.status == CUDA_SUCCESS && second.status == CUDA_SUCCESS,
               "create independent members before one initialization rejects");
        if (failure == 1) driver.next_allocation_status = CUDA_ERROR_OUT_OF_MEMORY;
        if (failure == 2) driver.next_host_status = CUDA_ERROR_OUT_OF_MEMORY;
        if (failure == 3) driver.next_host_map_status = CUDA_ERROR_OUT_OF_MEMORY;
        {
            CudaExecutionSession rejected_session({1, 99}, 101, session_dispatch(), limits(1000),
                                                  {320}, {}, {}, false, {},
                                                  std::move(first.member));
            const auto result = rejected_session.initialize(0);
            expect(result.status == CUDA_ERROR_OUT_OF_MEMORY && !driver.current &&
                       !owner.snapshot().is_quarantined &&
                       owner.snapshot().reserved_device_bytes == 0 &&
                       driver.live_resources() == 0 && driver.context_destroys == 0,
                   "clean budget/counter/host/map rejection rolls back only known own resources");
            if (failure == 0)
                expect(driver.allocation_calls == 0, "aggregate budget rejects before allocation");
            success(rejected_session.shutdown());
            CudaExecutionSession healthy({2, 99}, 202, session_dispatch(), limits(2000), {}, {}, {},
                                         false, {}, std::move(second.member));
            success(healthy.initialize(0, {CU_CTX_MAP_HOST}));
            success(healthy.allocate(4));
            expect(
                !healthy.snapshot().is_quarantined && owner.snapshot().reserved_device_bytes == 4,
                "another member retains independent initialization and allocation authority");
            success(healthy.shutdown());
        }
        expect(!driver.current && driver.live_resources() == 0 &&
                   owner.snapshot().reserved_device_bytes == 0,
               "clean rejected and healthy members retire with exact aggregate accounting");
        expect(owner.close() == CUDA_SUCCESS && owner.shutdown() == CUDA_SUCCESS &&
                   driver.context_creates == 1 && driver.context_destroys == 1,
               "only the cohort owner destroys its settled context");
    }
}

void synchronous_tail_requires_actual_retirement() {
    for (const bool uses_drain : {false, true}) {
        driver = {};
        driver.requires_explicit_cleanup = true;
        CudaExecutionDomainOwner owner(domain_dispatch(), {99, 0, 0, 2, 128});
        expect(owner.initialize() == CUDA_SUCCESS, "initialize synchronous tail cohort");
        auto first = owner.create_member({{1, 99}, 101});
        auto second = owner.create_member({{2, 99}, 202});
        expect(first.status == CUDA_SUCCESS && second.status == CUDA_SUCCESS,
               "create both synchronous tail members");
        {
            CudaExecutionSession first_session({1, 99}, 101, session_dispatch(), limits(1000), {},
                                               {}, {}, false, {}, std::move(first.member));
            CudaExecutionSession second_session({2, 99}, 202, session_dispatch(), limits(2000), {},
                                                {}, {}, false, {}, std::move(second.member));
            success(first_session.initialize(0));
            const auto buffer = first_session.allocate(16);
            success(buffer);
            success(first_session.deactivate());
            success(second_session.initialize(0));
            success(second_session.deactivate());
            success(first_session.activate());
            constexpr std::array<std::uint32_t, 4> input{13, 17, 23, 29};
            driver.query_status = CUDA_ERROR_NOT_READY;
            success(first_session.upload({buffer.id, 0}, std::as_bytes(std::span{input})));
            expect(first_session.snapshot().external_work_in_flight,
                   "host-complete upload still owns an actual private physical tail");
            success(first_session.deactivate());
            success(second_session.activate());
            const auto allocation_calls = driver.allocation_calls;
            const auto deferred = second_session.allocate(4);
            expect(deferred.disposition == CudaSessionDisposition::kDeferred &&
                       deferred.status == CUDA_ERROR_NOT_READY &&
                       driver.allocation_calls == allocation_calls,
                   "sibling cannot start exclusive SDK work after host return before tail stop");
            success(second_session.deactivate());
            success(first_session.activate());
            driver.query_status = CUDA_SUCCESS;
            const auto retired = uses_drain ? first_session.drain_issued() : first_session.poll();
            expect(
                retired.status == CUDA_SUCCESS && !first_session.snapshot().external_work_in_flight,
                "both actual poll and explicit drain settle the private tail");
            success(first_session.deactivate());
            success(second_session.activate());
            success(second_session.allocate(4));
            success(second_session.shutdown());
            success(first_session.activate());
            std::array<std::uint32_t, 4> output{};
            success(
                first_session.download({buffer.id, 0}, std::as_writable_bytes(std::span{output})));
            expect(output == input, "complete copied values survive sibling native cleanup");
            expect(first_session.poll().status == CUDA_SUCCESS,
                   "readback private tail retires independently of the host output");
            success(first_session.shutdown());
        }
        expect(!driver.current && driver.live_resources() == 0 &&
                   owner.snapshot().reserved_device_bytes == 0 && owner.close() == CUDA_SUCCESS &&
                   owner.shutdown() == CUDA_SUCCESS && driver.context_destroys == 1,
               "all physical tails and members retire before the only native context destroy");
    }
}

void isolated_legacy_context_destruction_is_unchanged() {
    driver = {};
    {
        CudaExecutionSession session({3, 99}, 303, session_dispatch(), limits(3000));
        success(session.initialize(0));
        success(session.allocate(64));
        success(session.load_ptx(kSource));
        success(session.create_stream(CU_STREAM_DEFAULT));
        expect(driver.context_creates == 1 && driver.streams[0].flags == CU_STREAM_DEFAULT,
               "unattached Session preserves native context and original physical stream flags");
        success(session.shutdown());
        expect(driver.context_destroys == 1 && driver.release_calls == 0 &&
                   driver.module_unloads == 0 && driver.stream_destroys == 0 &&
                   driver.live_resources() == 0 && !driver.current,
               "isolated context destruction retains its existing implicit resource cleanup");
    }
    expect(driver.context_destroys == 1, "isolated destructor cannot repeat native shutdown");
}

void shared_initialization_and_boundary_faults_are_retained() {
    for (unsigned int failure = 0; failure < 10; ++failure) {
        const auto child = fork();
        expect(child >= 0, "fork intentionally retained cohort boundary control");
        if (child == 0) {
            try {
                driver = {};
                CudaExecutionDomainOwner owner(domain_dispatch(), {99, 0, CU_CTX_MAP_HOST, 2, 192});
                expect(owner.initialize() == CUDA_SUCCESS, "initialize boundary fault cohort");
                auto first = owner.create_member({{1, 99}, 101});
                auto second = owner.create_member({{2, 99}, 202});
                expect(first.status == CUDA_SUCCESS && second.status == CUDA_SUCCESS,
                       "prepare both boundary fault member capabilities");
                CudaExecutionSession first_session({1, 99}, 101, session_dispatch(), limits(1000),
                                                   {320}, {}, {}, false, {},
                                                   std::move(first.member));
                CudaExecutionSession second_session({2, 99}, 202, session_dispatch(), limits(2000),
                                                    {320}, {}, {}, false, {},
                                                    std::move(second.member));
                if (failure < 2) {
                    if (failure == 0)
                        driver.on_allocate = [] {
                            driver.next_context_query_status = CUDA_ERROR_INVALID_CONTEXT;
                        };
                    else
                        driver.next_allocation_status = CUDA_ERROR_UNKNOWN;
                    expect(first_session.initialize(0).status == CUDA_ERROR_UNKNOWN &&
                               driver.allocation_calls == 1 && driver.next_host == 0 &&
                               owner.snapshot().reserved_device_bytes == 16,
                           "unproved/unknown counter allocation retains charge and never enters "
                           "host allocation");
                } else {
                    success(first_session.initialize(0));
                    success(first_session.deactivate());
                    success(second_session.initialize(0));
                    success(second_session.deactivate());
                    success(first_session.activate());
                    if (failure < 4) {
                        if (failure == 2)
                            driver.on_allocate = [] {
                                driver.next_context_query_status = CUDA_ERROR_INVALID_CONTEXT;
                            };
                        else
                            driver.next_allocation_status = CUDA_ERROR_UNKNOWN;
                        const auto result = first_session.allocate(64);
                        expect(result.status == CUDA_ERROR_UNKNOWN && !result.id &&
                                   result.original_status ==
                                       (failure == 2 ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN) &&
                                   owner.snapshot().reserved_device_bytes == 96,
                               "resource proof loss preserves native status without publishing or "
                               "refunding allocation");
                    } else if (failure == 4) {
                        const auto accepted =
                            first_session.accept_external_barrier(0, std::make_shared<int>(0));
                        expect(accepted.status == CUDA_SUCCESS &&
                                   first_session.begin_external_barrier(accepted.barrier) ==
                                       CUDA_SUCCESS &&
                                   first_session.fail_external_barrier(
                                       accepted.barrier, CUDA_ERROR_UNKNOWN) == CUDA_SUCCESS,
                               "explicit begun native barrier failure is reported without polling");
                    } else if (failure == 5) {
                        const auto accepted =
                            first_session.accept_enqueue_turn(0, std::make_shared<int>(0));
                        expect(
                            accepted.status == CUDA_SUCCESS &&
                                first_session.begin_enqueue_turn(accepted.turn) == CUDA_SUCCESS &&
                                first_session.fail_enqueue_turn(
                                    accepted.turn, CUDA_ERROR_UNKNOWN) == CUDA_ERROR_UNKNOWN,
                            "explicit begun native turn failure is reported without polling");
                    } else if (failure == 6 || failure == 9) {
                        driver.on_module_load = [] {
                            driver.next_context_query_status = CUDA_ERROR_INVALID_CONTEXT;
                        };
                        if (failure == 9) driver.next_module_status = CUDA_ERROR_INVALID_PTX;
                        const auto result = first_session.load_ptx(kSource);
                        expect(result.status != CUDA_SUCCESS && !result.id &&
                                   result.original_status ==
                                       (failure == 9 ? CUDA_ERROR_INVALID_PTX : CUDA_SUCCESS) &&
                                   driver.module_load_calls == 1 && driver.module_unloads == 0,
                               "unproved load retains original status and never retries original "
                               "intake or unloads");
                    } else {
                        const auto module = first_session.load_ptx(kSource);
                        success(module);
                        driver.on_function_get = [failure](bool is_private) {
                            if (is_private == (failure == 8))
                                driver.next_context_query_status = CUDA_ERROR_INVALID_CONTEXT;
                        };
                        const auto result = first_session.get_function(module.id, "kernel");
                        expect(result.status == CUDA_ERROR_UNKNOWN && !result.id &&
                                   result.original_status == CUDA_SUCCESS &&
                                   driver.parameter_calls == (failure == 8 ? 2U : 0U) &&
                                   driver.configuration_calls == (failure == 8 ? 1U : 0U) &&
                                   driver.module_unloads == 0,
                               "public/private lookup proof loss stops every later ABI query and "
                               "configuration");
                    }
                }
                const auto allocation_calls = driver.allocation_calls;
                const auto cleanup_calls = driver.cleanup_calls();
                const auto launches = driver.launches;
                expect(owner.snapshot().is_quarantined && first_session.snapshot().is_quarantined &&
                           second_session.snapshot().is_quarantined &&
                           second_session.issue_next().state == CudaExecutionIssueState::kError &&
                           second_session.allocate(4).status != CUDA_SUCCESS &&
                           first_session.shutdown().status != CUDA_SUCCESS &&
                           second_session.shutdown().status != CUDA_SUCCESS &&
                           driver.allocation_calls == allocation_calls &&
                           driver.cleanup_calls() == cleanup_calls && driver.launches == launches,
                       "quarantine reaches every sibling immediately with no continuation, cleanup "
                       "or refund");
                _exit(0);
            } catch (const std::exception& error) {
                std::cerr << "Shared boundary child: " << error.what() << '\n';
                _exit(1);
            }
        }
        int status{};
        expect(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
               "retained boundary fault control finishes without signaling or fake recovery");
    }
}

void shared_native_fault_is_retained_in_a_child(bool uses_drain = false) {
    const auto child = fork();
    expect(child >= 0, "fork intentional shared native-uncertainty control");
    if (child == 0) {
        try {
            driver = {};
            CudaExecutionDomainOwner owner(domain_dispatch(), {100, 0, CU_CTX_MAP_HOST, 2, 192});
            expect(owner.initialize() == CUDA_SUCCESS, "initialize uncertain domain control");
            auto first = owner.create_member({{4, 100}, 404});
            auto second = owner.create_member({{5, 100}, 505});
            expect(first.status == CUDA_SUCCESS && second.status == CUDA_SUCCESS,
                   "create exact fault-cohort members");
            CudaExecutionSession first_session({4, 100}, 404, session_dispatch(), limits(4000),
                                               {320}, {}, {}, false, {}, std::move(first.member));
            CudaExecutionSession second_session({5, 100}, 505, session_dispatch(), limits(5000),
                                                {320}, {}, {}, false, {}, std::move(second.member));
            success(first_session.initialize(0));
            const auto buffer = first_session.allocate(64);
            const auto module = first_session.load_ptx(kSource);
            success(buffer);
            success(module);
            const auto function = first_session.get_function(module.id, "kernel");
            success(function);
            success(first_session.deactivate());
            success(second_session.initialize(0));
            success(second_session.deactivate());
            success(first_session.activate());
            KernelInput input({function.id, 0}, buffer.address);
            const auto accepted = first_session.accept_kernel(1, input.kernel);
            expect(accepted.accepted() &&
                       first_session.issue_next().state == CudaExecutionIssueState::kIssued,
                   "issue the exact original fault-control command");
            driver.query_status = CUDA_ERROR_UNKNOWN;
            const auto result = uses_drain ? first_session.drain_issued() : first_session.poll();
            expect(result.status != CUDA_SUCCESS && first_session.snapshot().is_quarantined &&
                       second_session.snapshot().is_quarantined && owner.snapshot().is_quarantined,
                   "actual dispatcher tail failure quarantines every shared member immediately");
            const auto calls = driver.allocation_calls;
            expect(second_session.allocate(4).status != CUDA_SUCCESS &&
                       driver.allocation_calls == calls && driver.context_destroys == 0 &&
                       owner.snapshot().reserved_device_bytes == 96,
                   "shared uncertainty cannot issue, refund or use context teardown as recovery");
            // This child deliberately retains an unproved native tail and all
            // of its resources. It is not a healthy leak-free exit claim.
            _exit(0);
        } catch (const std::exception& error) {
            std::cerr << "Shared fault child: " << error.what() << '\n';
            _exit(1);
        }
    }
    int status{};
    expect(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "shared native fault fails closed without fake cleanup or an inherited owner");
}
}  // namespace

int main() {
    try {
        pair_catalog_errors_are_not_structural_exclusions();
        configured_functions_reuse_values_not_epochs();
        unexpected_function_state_rejects_sample();
        configured_context_measurements_are_not_epochs();
        unexpected_configuration_cannot_commit_measurement();
        isolated_pair_samples_execute_once();
        actual_pending_pairs_preserve_independent_ownership(PairScenario::kSharedResume);
        actual_pending_pairs_preserve_independent_ownership(PairScenario::kSecondNativeError);
        actual_pending_pairs_preserve_independent_ownership(PairScenario::kColdOriginal);
        shared_nonpersistent_flags_keep_mapped_host_capability();
        shared_initial_flags_defer_and_restore();
        shared_configuration_and_retained_commands();
        shared_lifetime_quota_reentry_and_retained_continuation();
        clean_initialization_rejections_preserve_other_members();
        synchronous_tail_requires_actual_retirement();
        isolated_legacy_context_destruction_is_unchanged();
        shared_initialization_and_boundary_faults_are_retained();
        shared_native_fault_is_retained_in_a_child();
        shared_native_fault_is_retained_in_a_child(true);
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Execution domain Session integration: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
