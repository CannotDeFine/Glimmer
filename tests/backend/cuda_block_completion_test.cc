#include "cuda_execution_domain_fixture.h"

#include <atomic>
#include <thread>

namespace {
using namespace glimmer::backend::cuda;
using namespace glimmer::test::cuda_domain;
using Kind = glimmer::transform::PtxExecutionKind;
CUdeviceptr completion{};
std::vector<CUdeviceptr> observed;

// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact native callback ABI.
CUresult CUDAAPI observed_parameter(CUfunction function, std::size_t index, std::size_t* offset,
                                    std::size_t* bytes) {
    if (reinterpret_cast<FakeFunction*>(function)->is_private && index == 7) {
        ++driver.parameter_calls;
        *offset = 40;
        *bytes = 8;
        return CUDA_SUCCESS;
    }
    return parameter_info(function, index, offset, bytes);
}
CUresult CUDAAPI observed_launch(CUfunction function, unsigned int gx, unsigned int gy,
                                 unsigned int gz, unsigned int bx, unsigned int by, unsigned int bz,
                                 unsigned int shared, CUstream stream, void** parameters,
                                 void** extra) {
    expect(reinterpret_cast<FakeFunction*>(function)->is_private && !parameters && extra &&
               *static_cast<const std::size_t*>(extra[3]) == 48,
           "observed native ABI explicitly adds one pointer, not a legacy retreat extension");
    const auto* input = static_cast<const std::byte*>(extra[1]);
    CUdeviceptr retreat{};
    std::memcpy(&retreat, input + 32, 8);
    std::memcpy(&completion, input + 40, 8);
    expect(completion == retreat + 4 && completion % alignof(std::uint32_t) == 0,
           "new pointer binds the separately initialized aligned word in the owned host slot");
    observed.push_back(completion);
    std::size_t legacy_bytes{40};
    void* legacy[]{CU_LAUNCH_PARAM_BUFFER_POINTER, extra[1], CU_LAUNCH_PARAM_BUFFER_SIZE,
                   &legacy_bytes, CU_LAUNCH_PARAM_END};
    return launch(function, gx, gy, gz, bx, by, bz, shared, stream, parameters, legacy);
}
// NOLINTEND(bugprone-easily-swappable-parameters)

void transform_and_cache() {
    using glimmer::transform::PtxTransformLimits;
    using glimmer::transform::transform_persistent_ptx;
    PtxTransformLimits settings;
    settings.observes_block_completion = true;
    const auto normal = transform_persistent_ptx(kSource, Kind::kPreemptive, true);
    const auto diagnostic = transform_persistent_ptx(kSource, Kind::kPreemptive, true, settings);
    expect(normal && diagnostic && !normal->kernels[0].observes_block_completion &&
               diagnostic->kernels[0].observes_block_completion &&
               normal->image.find("__glimmer_ptb_completed") == std::string::npos,
           "default generated PTX has no observation parameter, code or publication");
    const auto& code = diagnostic->image;
    const auto exit = code.find("__glimmer_ptb_next:");
    const auto barrier = code.find("bar.sync 0;", exit);
    const auto publication = code.find("st.release.sys.global.u32");
    expect(
        exit < barrier && barrier < publication &&
            code.find("mov.pred %__glimmer_ptb_publish_completion, 0;", publication) !=
                std::string::npos &&
            code.find("mov.u32 %__glimmer_ptb_physical_worker, %ctaid.x;") != std::string::npos &&
            code.find("not.pred %__glimmer_ptb_leader, %__glimmer_ptb_follower;") !=
                std::string::npos,
        "only physical CTA0/thread0 publishes once after the complete source-block barrier");
    for (const auto kind : {Kind::kStatic, Kind::kBlockStatic, Kind::kDynamic, Kind::kSliced})
        expect(!transform_persistent_ptx(kSource, kind, true, settings),
               "observation cannot silently alter another family's native ABI");
    for (const auto& source : {std::string(kSource).replace(kSource.find("7.0"), 3, "5.0"),
                               std::string(kSource).replace(kSource.find("sm_70"), 5, "sm_60")}) {
        expect(!transform_persistent_ptx(source, Kind::kPreemptive, true, settings) &&
                   transform_persistent_ptx(source, Kind::kPreemptive, true),
               "new system-release eligibility does not narrow ordinary legacy PTX support");
    }
    CudaPtxCache cache;
    const auto ordinary = cache.lookup(kSource, Kind::kPreemptive, true);
    const auto instrumented = cache.lookup(kSource, Kind::kPreemptive, true, settings);
    expect(ordinary.image && instrumented.image && ordinary.image != instrumented.image &&
               cache.snapshot().entries == 2,
           "one exact source has distinct immutable normal/observed preparation identities");
    const auto archive = cache.encode("completion-fixture");
    CudaPtxCache restored;
    expect(archive && restored.restore(*archive, "completion-fixture"),
           "observation metadata restores transactionally under the new generator ABI");
    expect(restored.lookup(kSource, Kind::kPreemptive, true).image->image == normal->image &&
               restored.lookup(kSource, Kind::kPreemptive, true, settings).image->image == code &&
               !restored.snapshot().transforms,
           "restore cannot alias an observed artifact into an ordinary hit or retransform it");
}

void observation_is_not_stop() {
    driver = {};
    driver.query_status = CUDA_ERROR_NOT_READY;
    completion = 0;
    observed.clear();
    auto sdk = session_dispatch();
    sdk.parameter_info = observed_parameter;
    sdk.execution.launch_kernel = observed_launch;
    CudaSessionPersistentConfig configuration{320};
    configuration.observes_block_completion = true;
    CudaProfileConfig profile;
    profile.workers = {320};
    profile.warmup_calls = 0;
    profile.samples_per_candidate = 1;
    auto external = std::make_shared<CudaProfileCache>(profile, pair_clock, false);
    CudaExecutionSession session({1, 99}, 101, sdk, limits(1000), configuration, {}, external);
    const auto measured = pair_singles()->encode("late-observation-cache");
    expect(measured && external->restore(*measured, "late-observation-cache"),
           "external readonly cache can acquire ordinary measured data after Session construction");
    success(session.initialize(0));
    const auto module = session.load_ptx(kSource);
    success(module);
    const auto function = session.get_function(module.id, "kernel");
    success(function);
    const auto buffer = session.allocate(32);
    success(buffer);
    KernelInput input({function.id, 0}, buffer.address);
    const auto accepted = session.accept_kernel(1, input.kernel);
    expect(accepted.accepted(), "accept unchanged original non-idempotent command");
    expect(!session.persistent_progress(1)->has_completed_block,
           "accepted bytes do not fabricate completed GPU work");
    const auto issued = session.issue_next();
    expect(issued.state == CudaExecutionIssueState::kIssued &&
               issued.native_status == CUDA_SUCCESS &&
               session.acknowledge_result(accepted.receipt) && completion &&
               !session.persistent_progress(1)->has_completed_block,
           "an actual successful SDK call and host ACK still do not fabricate GPU publication");
    expect(driver.physical_workers == std::vector<unsigned int>{320},
           "observed Session never retains/reuses an initially empty cache after late restore");
    const auto host = std::ranges::find_if(driver.hosts, [&](const auto& entry) {
        const auto base = reinterpret_cast<CUdeviceptr>(entry.data.data());
        return entry.is_live && completion >= base &&
               completion - base <= entry.data.size() - sizeof(std::uint32_t);
    });
    expect(host != driver.hosts.end(), "witness is inside its exact live host owner");
    const auto offset = completion - reinterpret_cast<CUdeviceptr>(host->data.data());
    auto& word = *reinterpret_cast<std::uint32_t*>(host->data.data() + offset);
    expect(std::atomic_ref(word).load() == 0, "word initialized before the first physical call");
    std::atomic_ref(word).store(1,
                                std::memory_order_release);  // Injected GPU writer, not real GPU.
    const auto cleanup = driver.cleanup_calls();
    const auto before = session.persistent_progress(1);
    expect(before && before->observes_block_completion && before->has_completed_block &&
               before->completed_blocks == 0 && before->physical_launches == 1 &&
               before->physical_stops == 0 && driver.cleanup_calls() == cleanup,
           "publication witnesses one block, never count retirement or physical completion");
    const auto poll = session.poll();
    expect(!poll.paused && !poll.retired && session.persistent_progress(1)->physical_stops == 0,
           "publication cannot turn an actual pending event into stopped work");
    std::optional<CudaSessionPersistentProgress> foreign;
    std::thread caller([&] { foreign = session.persistent_progress(1); });
    caller.join();
    expect(!foreign && !session.persistent_progress(999),
           "foreign owner and sequence cannot borrow another command's publication");
    expect(session.request_retreat(1) == CUDA_SUCCESS, "request actual retained retreat");
    driver.ready_launch = 1;
    expect(session.poll().paused == 1, "only the independently ready trailing event proves stop");
    expect(session.persistent_progress(1)->completed_blocks == 3 &&
               session.persistent_progress(1)->has_completed_block,
           "stopped count remains independent of the live completion publication");
    driver.next_claimed = 8;
    const auto resumed = session.resume(1);
    expect(resumed.state == CudaExecutionIssueState::kIssued &&
               resumed.native_status == CUDA_SUCCESS &&
               session.persistent_progress(1)->has_completed_block && observed.size() == 2 &&
               observed[0] == observed[1] && driver.counter_initializations == 1,
           "retained resume preserves both witness and counter, with no second initialization");
    driver.query_status = CUDA_SUCCESS;
    expect(session.poll().retired == 1, "final independent physical event retires original work");
    const auto previous_word = completion;
    driver.query_status = CUDA_ERROR_NOT_READY;
    driver.next_claimed = 3;
    const auto fresh = session.accept_kernel(2, input.kernel);
    expect(fresh.accepted() && !session.persistent_progress(2)->has_completed_block,
           "new accepted command cannot inherit the previous generation's publication");
    const auto fresh_issue = session.issue_next();
    expect(fresh_issue.state == CudaExecutionIssueState::kIssued &&
               session.acknowledge_result(fresh.receipt) && completion == previous_word &&
               std::atomic_ref(word).load() == 0 &&
               !session.persistent_progress(2)->has_completed_block &&
               driver.counter_initializations == 2,
           "safe slot reuse clears its witness exactly once before the new physical launch");
    std::atomic_ref(word).store(2, std::memory_order_release);
    expect(!session.persistent_progress(2)->has_completed_block,
           "noncanonical word cannot fabricate a publication");
    std::atomic_ref(word).store(1, std::memory_order_release);
    expect(session.request_retreat(2) == CUDA_SUCCESS,
           "fresh command retains separate stop authority");
    driver.ready_launch = 3;
    expect(session.poll().paused == 1, "only fresh physical event proves its partial stop");
    driver.next_claimed = 8;
    expect(session.resume(2).state == CudaExecutionIssueState::kIssued,
           "fresh command resumes only its own retained work");
    driver.query_status = CUDA_SUCCESS;
    expect(session.poll().retired == 1 && driver.counter_initializations == 2,
           "two logical commands initialize twice, not once per physical generation");
    std::array<std::uint32_t, 8> values{};
    success(session.download({buffer.id, 0}, std::as_writable_bytes(std::span{values})));
    expect(std::ranges::all_of(values, [](auto value) { return value == 14; }),
           "every non-idempotent output reflects exactly two independent commands");
    expect(session.poll().status == CUDA_SUCCESS, "retire real readback prefix");
    success(session.release(buffer.id));
    success(session.unload_module(module.id));
    success(session.shutdown());
    expect(driver.live_resources() == 0 && driver.context_destroys == 1,
           "witness storage follows the existing resource owner through normal final cleanup");
}

void unsupported_original_never_publishes() {
    driver = {};
    auto sdk = session_dispatch();
    CudaSessionPersistentConfig configuration{320};
    configuration.observes_block_completion = true;
    CudaExecutionSession session({1, 99}, 101, sdk, limits(1000), configuration);
    success(session.initialize(0));
    auto source = std::string(kSource);
    source.replace(source.find("sm_70"), 5, "sm_60");
    const auto module = session.load_ptx(source);
    success(module);
    const auto function = session.get_function(module.id, "kernel");
    success(function);
    const auto buffer = session.allocate(32);
    success(buffer);
    KernelInput input({function.id, 0}, buffer.address);
    const auto accepted = session.accept_kernel(1, input.kernel);
    expect(accepted.accepted() && !session.persistent_progress(1),
           "unsupported observation target stays original without a fabricated witness");
    const auto issued = session.issue_next();
    expect(issued.state == CudaExecutionIssueState::kIssued &&
               issued.native_status == CUDA_SUCCESS &&
               session.acknowledge_result(accepted.receipt) && !session.persistent_progress(1),
           "original SDK fallback never acquires completion-publication capability");
    expect(session.poll().status == CUDA_SUCCESS, "actual original trailing event retires");
    std::array<std::uint32_t, 8> values{};
    success(session.download({buffer.id, 0}, std::as_writable_bytes(std::span{values})));
    expect(std::ranges::all_of(values, [](auto value) { return value == 7; }),
           "original fallback keeps all useful non-idempotent outputs");
    expect(session.poll().status == CUDA_SUCCESS, "original readback retires");
    success(session.release(buffer.id));
    success(session.unload_module(module.id));
    success(session.shutdown());
}

void invalid_observation_configuration_rejects_before_native() {
    for (unsigned int scenario = 0; scenario < 4; ++scenario) {
        driver = {};
        CudaSessionPersistentConfig configuration{320};
        configuration.observes_block_completion = true;
        std::shared_ptr<CudaProfileCache> learning;
        if (scenario == 0) configuration.workers = 0;
        if (scenario == 1) configuration.kind = Kind::kSliced;
        if (scenario == 2) configuration.uses_workload_agnostic = true;
        if (scenario == 3) learning = std::make_shared<CudaProfileCache>(CudaProfileConfig{});
        bool rejected{};
        try {
            CudaExecutionSession session({1, 99}, 101, session_dispatch(), limits(1000),
                                         configuration, {}, learning);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        expect(rejected && driver.context_creates == 0 && driver.allocation_calls == 0 &&
                   driver.launches == 0 && driver.cleanup_calls() == 0,
               "invalid diagnostic/profile composition rejects before every native side effect");
    }
}
}  // namespace
int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--dump-observed") {
            glimmer::transform::PtxTransformLimits settings;
            settings.observes_block_completion = true;
            const auto image = glimmer::transform::transform_persistent_ptx(
                kSource, Kind::kPreemptive, true, settings);
            expect(image.has_value(), "actual observed image for offline compiler validation");
            std::cout << image->image;
            return EXIT_SUCCESS;
        }
        expect(argc == 1, "only the explicit offline image option is accepted");
        transform_and_cache();
        observation_is_not_stop();
        unsupported_original_never_publishes();
        invalid_observation_configuration_rejects_before_native();
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Block completion: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
