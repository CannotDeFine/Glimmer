#include "cuda_execution_domain_fixture.h"
#include "internal/cuda_sgemm_image.h"
#include "internal/cuda_gemm_geometry.h"
#include "internal/cuda_gemm_signal_layout.h"

#include <atomic>
#include <string>

namespace {
using namespace glimmer::backend::cuda;
using namespace glimmer::test::cuda_domain;
constexpr std::array<std::string_view, 25> kNames{
    "glimmer_owned_sgemm_nn",        "glimmer_owned_sgemm_nt",   "glimmer_owned_sgemm_tn",
    "glimmer_owned_sgemm_tt",        "glimmer_owned_hgemm_nn",   "glimmer_owned_hgemm_nt",
    "glimmer_owned_hgemm_tn",        "glimmer_owned_hgemm_tt",   "glimmer_owned_hsgemm_nn",
    "glimmer_owned_hsgemm_nt",       "glimmer_owned_hsgemm_tn",  "glimmer_owned_hsgemm_tt",
    "glimmer_owned_bsgemm_nn",       "glimmer_owned_bsgemm_nt",  "glimmer_owned_bsgemm_tn",
    "glimmer_owned_bsgemm_tt",       "glimmer_owned_bhgemm_nn",  "glimmer_owned_bhgemm_nt",
    "glimmer_owned_bhgemm_tn",       "glimmer_owned_bhgemm_tt",  "glimmer_owned_bhsgemm_nn",
    "glimmer_owned_bhsgemm_nt",      "glimmer_owned_bhsgemm_tn", "glimmer_owned_bhsgemm_tt",
    "glimmer_owned_sgemm_nn_split_k"};
constexpr std::array<std::string_view, 3> kLtNames{
    "glimmer_owned_lt_sgemm", "glimmer_owned_lt_hgemm", "glimmer_owned_lt_hsgemm"};

void retreat_load_contract() {
    for (const auto& [side, sharded] :
         {std::pair{32U, false}, std::pair{64U, false}, std::pair{32U, true}}) {
        const auto image = sharded ? cuda_sgemm_sharded_image(side) : cuda_sgemm_image(side);
        expect(!image.empty(), "each supported GEMM tile family has an image");
        const auto check_entries = [&](const auto& names) {
            for (const auto name : names) {
                for (const bool observes : {false, true}) {
                    const auto entry = std::string{".visible .entry "} + std::string{name} +
                                       (observes ? "_observed(" : "(");
                    const auto begin = image.find(entry);
                    expect(begin != std::string_view::npos, "GEMM provider entry exists");
                    const auto next = image.find(".visible .entry ", begin + entry.size());
                    const auto body = image.substr(begin, next - begin);
                    expect(body.find("ld.acquire.sys.global.u32") != std::string_view::npos,
                           "each GEMM entry acquires the system-visible retreat flag");
                }
            }
        };
        check_entries(kNames);
        check_entries(kLtNames);
    }
    expect(cuda_sgemm_sharded_image(64).empty() &&
               cuda_sgemm_sharded_image(32) != cuda_sgemm_image(32),
           "the sharded layout has a distinct, explicitly supported image");
}
struct State {
    std::array<FakeFunction, kNames.size()> functions{};
    bool observes{}, is_batched{}, has_bad_metadata{}, uses_sharded_host_retreat{};
    unsigned int lookups{}, launches{}, native_calls{};
    std::array<int, 4> handles{};
    std::array<cudaStream_t, 4> blas_streams{};
    std::size_t handle_count{};
    CUdeviceptr completed{}, counter{}, retreat{}, generic_retreat{};
    unsigned int tile_side{64};
    std::vector<CUdeviceptr> witnesses, counters;
} state;

template <typename T>
T parameter(const std::byte* data, std::size_t offset) {
    T value{};
    std::memcpy(&value, data + offset, sizeof(value));
    return value;
}
std::uint32_t& mapped_word(CUdeviceptr address) {
    const auto found = std::ranges::find_if(driver.hosts, [address](const auto& host) {
        const auto base = reinterpret_cast<CUdeviceptr>(host.data.data());
        return host.is_live && address >= base &&
               address - base <= host.data.size() - sizeof(std::uint32_t) &&
               (address - base) % alignof(std::uint32_t) == 0;
    });
    expect(found != driver.hosts.end(), "word belongs to an aligned live fake host mapping");
    const auto offset = address - reinterpret_cast<CUdeviceptr>(found->data.data());
    return *reinterpret_cast<std::uint32_t*>(found->data.data() + offset);
}
std::uint32_t& completion_word() {
    return mapped_word(state.completed);
}

CudaSessionDispatch dispatch() {
    auto sdk = session_dispatch();
    sdk.module_load = [](CUmodule* output, const void* image) {
        const auto selected = state.uses_sharded_host_retreat
                                  ? cuda_sgemm_sharded_image(state.tile_side)
                                  : cuda_sgemm_image(state.tile_side);
        expect(image == selected.data(),
               "native module intake uses this Session's exact selected immutable image");
        return glimmer::test::cuda_domain::module_load(output, image);
    };
    sdk.upload_async = [](CUdeviceptr destination, const void* source, std::size_t bytes,
                          CUstream stream) {
        expect(bytes == (state.observes ? 16U : 8U),
               "observed provider initializes claim/election once; default stays eight bytes");
        expect(parameter<std::uint64_t>(static_cast<const std::byte*>(source), 0) == 0 &&
                   (!state.observes ||
                    parameter<std::uint64_t>(static_cast<const std::byte*>(source), 8) == 0),
               "new logical lease initializes every private device word to zero");
        if (bytes == 16) ++driver.counter_initializations;
        return glimmer::test::cuda_domain::upload_async(destination, source, bytes, stream);
    };
    sdk.device_attribute = [](int* output, CUdevice_attribute attribute, CUdevice) {
        expect(attribute == CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR ||
                   attribute == CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
               "provider queries only its qualified target");
        *output = attribute == CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR ? 7 : 0;
        return CUDA_SUCCESS;
    };
    sdk.function_get = [](CUfunction* output, CUmodule module, const char* name) {
        expect(module && reinterpret_cast<FakeModule*>(module)->is_live,
               "provider lookup retains the real fake module owner");
        std::string_view entry{name};
        expect(entry.ends_with("_observed") == state.observes,
               "ordinary and observed Sessions select distinct actual native entries");
        if (state.observes) entry.remove_suffix(std::string_view("_observed").size());
        const auto found = std::ranges::find(kNames, entry);
        expect(found != kNames.end(), "all twenty-five required GEMM entries are prepared");
        const auto index = static_cast<std::size_t>(found - kNames.begin());
        state.functions[index].is_private = true;
        *output = reinterpret_cast<CUfunction>(&state.functions[index]);
        ++state.lookups;
        return CUDA_SUCCESS;
    };
    // NOLINTBEGIN(bugprone-easily-swappable-parameters): exact injected native ABIs.
    sdk.parameter_info = [](CUfunction function, std::size_t index, std::size_t* offset,
                            std::size_t* bytes) {
        const auto position = std::ranges::find_if(state.functions, [&](const auto& entry) {
            return &entry == reinterpret_cast<const FakeFunction*>(function);
        });
        expect(position != state.functions.end(), "metadata belongs to a prepared provider entry");
        const auto variant = static_cast<std::size_t>(position - state.functions.begin());
        const bool split = variant == 24;
        const auto arity = split ? std::size_t{17} : variant < 12 ? std::size_t{16} : 20;
        constexpr std::array<std::size_t, 20> offsets{0,  8,  16, 24, 28, 32, 36, 40,  44,  48,
                                                      52, 56, 64, 72, 80, 88, 96, 104, 112, 120};
        constexpr std::array<std::size_t, 20> widths{8, 8, 8, 4, 4, 4, 4, 4, 4, 4,
                                                     4, 8, 8, 4, 8, 8, 8, 8, 8, 4};
        if (index == arity && state.observes) {
            *offset = split ? 104 : variant < 12 ? 96 : 128;
            *bytes = state.has_bad_metadata ? 4 : 8;
            return CUDA_SUCCESS;
        }
        if (index >= arity) return CUDA_ERROR_INVALID_VALUE;
        *offset = offsets[index];
        *bytes = split && index == 16 ? 4 : widths[index];
        return CUDA_SUCCESS;
    };
    sdk.blas.create = [](cublasHandle_t* output) {
        expect(state.handle_count < state.handles.size(), "bounded fake BLAS handle arena");
        *output = reinterpret_cast<cublasHandle_t>(&state.handles[state.handle_count++]);
        return CUBLAS_STATUS_SUCCESS;
    };
    sdk.blas.destroy = [](cublasHandle_t) { return CUBLAS_STATUS_SUCCESS; };
    sdk.blas.set_stream = [](cublasHandle_t handle, cudaStream_t stream) {
        for (std::size_t index = 0; index < state.handle_count; ++index)
            if (handle == reinterpret_cast<cublasHandle_t>(&state.handles[index])) {
                state.blas_streams[index] = stream;
                return CUBLAS_STATUS_SUCCESS;
            }
        return CUBLAS_STATUS_INVALID_VALUE;
    };
    sdk.blas.get_stream = [](cublasHandle_t handle, cudaStream_t* stream) {
        for (std::size_t index = 0; index < state.handle_count; ++index)
            if (handle == reinterpret_cast<cublasHandle_t>(&state.handles[index])) {
                *stream = state.blas_streams[index];
                return CUBLAS_STATUS_SUCCESS;
            }
        return CUBLAS_STATUS_INVALID_VALUE;
    };
    sdk.blas.sgemm = [](cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
                        const float*, const float*, int, const float*, int, const float*, float*,
                        int) {
        ++state.native_calls;
        return CUBLAS_STATUS_SUCCESS;
    };
    sdk.blas.strided_sgemm = [](cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
                                const float*, const float*, int, long long, const float*, int,
                                long long, const float*, float*, int, long long, int) {
        ++state.native_calls;
        return CUBLAS_STATUS_SUCCESS;
    };
    // Assignments instantiate the SDK's exact function-pointer signatures.
    // Geometry-only Ex cases must never call either original implementation.
    sdk.blas.gemm_ex = [](auto...) {
        ++state.native_calls;
        return CUBLAS_STATUS_SUCCESS;
    };
    sdk.blas.strided_gemm_ex = [](auto...) {
        ++state.native_calls;
        return CUBLAS_STATUS_SUCCESS;
    };
    sdk.execution.launch_kernel = [](CUfunction function, unsigned int workers, unsigned int,
                                     unsigned int, unsigned int threads, unsigned int, unsigned int,
                                     unsigned int, CUstream, void** parameters, void** extra) {
        expect(
            function == reinterpret_cast<CUfunction>(&state.functions[state.is_batched ? 12 : 0]) &&
                workers == 320 && threads == 128 && !parameters && extra,
            "actual selected provider launch keeps explicit worker/block geometry");
        const auto base_bytes = state.is_batched ? std::size_t{128} : 96;
        expect(*static_cast<const std::size_t*>(extra[3]) == base_bytes + (state.observes ? 8 : 0),
               "observed packing appends only the explicit pointer; default ABI is unchanged");
        const auto* data = static_cast<const std::byte*>(extra[1]);
        state.counter = parameter<CUdeviceptr>(data, 80);
        state.retreat = parameter<CUdeviceptr>(data, 88);
        if (state.uses_sharded_host_retreat) {
            const auto owner = std::ranges::find_if(driver.hosts, [](const auto& host) {
                const auto base = reinterpret_cast<CUdeviceptr>(host.data.data());
                return host.is_live && state.retreat >= base + 128 &&
                       state.retreat + kGemmSignalSlotBytes <= base + host.data.size() &&
                       (state.retreat - base - 128) % kGemmSignalSlotBytes == 0;
            });
            expect(owner != driver.hosts.end(),
                   "GEMM receives its exact bounded mapped shard array");
            for (std::size_t index = 0; index < kGemmSignalShards; ++index) {
                auto& word = mapped_word(state.retreat + index * kGemmSignalShardBytes);
                expect(std::atomic_ref(word).load(std::memory_order_acquire) == 0,
                       "each GEMM shard is cleared before initial issue or stopped resume");
            }
        }
        if (state.observes) {
            state.completed = parameter<CUdeviceptr>(data, base_bytes);
            if (!state.uses_sharded_host_retreat)
                expect(state.completed == state.retreat + 4,
                       "completion and legacy retreat are distinct aligned mapped words");
            state.witnesses.push_back(state.completed);
        }
        state.counters.push_back(state.counter);
        const auto previous = parameter<std::uint64_t>(device_bytes(state.counter, 8), 0);
        if (state.observes) {
            auto* election = device_bytes(state.counter + 8, 8);
            const auto previous_election = parameter<std::uint64_t>(election, 0);
            expect(previous_election == (previous ? 1U : 0U),
                   "resume retains the exact device election; new logical work resets it");
            const std::uint64_t winner = 1;
            std::memcpy(election, &winner, sizeof(winner));
        }
        const auto total = state.is_batched ? std::uint64_t{8} : 1;
        const auto claimed = std::min(driver.next_claimed, total);
        expect(previous <= claimed, "retained fake GEMM counter cannot replay completed tiles");
        for (auto tile = previous; tile < claimed; ++tile) {
            const auto offset = tile * sizeof(float);
            const auto a =
                parameter<float>(device_bytes(parameter<CUdeviceptr>(data, 0) + offset, 4), 0);
            const auto b =
                parameter<float>(device_bytes(parameter<CUdeviceptr>(data, 8) + offset, 4), 0);
            auto* output = device_bytes(parameter<CUdeviceptr>(data, 16) + offset, 4);
            const auto result = parameter<float>(data, 48) * a * b +
                                parameter<float>(data, 52) * parameter<float>(output, 0);
            std::memcpy(output, &result, sizeof(result));
        }
        std::memcpy(device_bytes(state.counter, 8), &claimed, sizeof(claimed));
        ++state.launches;
        ++driver.launches;
        return CUDA_SUCCESS;  // Deliberately does NOT publish block completion.
    };
    // NOLINTEND(bugprone-easily-swappable-parameters)
    return sdk;
}

void run(bool observes, bool is_batched, unsigned int tile_side = 64,
         bool uses_sharded_host_retreat = false) {
    driver = {};
    state = {};
    state.observes = observes;
    state.is_batched = is_batched;
    state.tile_side = tile_side;
    state.uses_sharded_host_retreat = uses_sharded_host_retreat;
    CudaSessionPersistentConfig settings{320};
    settings.replace_sgemm = true;
    settings.gemm_tile_side = tile_side;
    settings.observes_block_completion = observes;
    settings.uses_sharded_host_retreat = uses_sharded_host_retreat;
    CudaExecutionSession session({1, 99}, 101, dispatch(), limits(1000), settings);
    success(session.initialize(0));
    const auto handle = session.create_blas();
    expect(handle.status == CUDA_SUCCESS && handle.original_status == CUBLAS_STATUS_SUCCESS &&
               state.lookups == 25,
           "all provider metadata is checked before accepting ordinary application calls");
    const auto a = session.allocate(32), b = session.allocate(32), c = session.allocate(32);
    success(a);
    success(b);
    success(c);
    std::array<float, 8> values{};
    for (const auto& [allocation, value] :
         {std::pair{a.id, 2.0F}, std::pair{b.id, 4.0F}, std::pair{c.id, 6.0F}}) {
        values.fill(value);
        success(session.upload({allocation, 0}, std::as_bytes(std::span{values})));
    }
    CudaSessionStridedGemmEx problem;
    problem.operation.operation = {.handle = handle.id,
                                   .transpose_a = CUBLAS_OP_N,
                                   .transpose_b = CUBLAS_OP_N,
                                   .rows = 1,
                                   .columns = 1,
                                   .depth = 1,
                                   .leading_a = 1,
                                   .leading_b = 1,
                                   .leading_c = 1,
                                   .a = {a.id, 0},
                                   .b = {b.id, 0},
                                   .c = {c.id, 0},
                                   .alpha = 2.0F,
                                   .beta = 0.5F};
    std::memcpy(problem.operation.alpha.data(), &problem.operation.operation.alpha, 4);
    std::memcpy(problem.operation.beta.data(), &problem.operation.operation.beta, 4);
    problem.strides = {1, 1, 1};
    problem.batches = 8;
    const auto accept = [&](std::uint64_t sequence) {
        return is_batched ? session.accept_strided_sgemm(sequence, problem)
                          : session.accept_sgemm(sequence, problem.operation.operation);
    };
    std::array queued{accept(1), accept(2)};
    expect(queued[0].accepted() && queued[1].accepted() && session.snapshot().persistent_slots == 2,
           "two accepted logical commands independently reserve both progress slots");
    std::array<CUdeviceptr, 3> logical_counters{};
    for (std::uint64_t sequence = 1; sequence <= 3; ++sequence) {
        driver.query_status = CUDA_ERROR_NOT_READY;
        driver.ready_launch = 0;
        driver.next_claimed = 3;
        const auto accepted = sequence <= 2 ? queued[sequence - 1] : accept(sequence);
        if (sequence <= 2) queued[sequence - 1] = {};
        expect(accepted.accepted() && !session.persistent_progress(sequence)->has_completed_block,
               "new logical library work cannot borrow a prior slot publication");
        const auto issue = session.issue_next();
        expect(issue.state == CudaExecutionIssueState::kIssued &&
                   issue.native_status == CUDA_SUCCESS &&
                   session.acknowledge_result(accepted.receipt),
               "actual Driver issue is independent of completed useful computation");
        logical_counters[sequence - 1] = state.counter;
        if (sequence == 2)
            expect(logical_counters[1] - logical_counters[0] == (observes ? 16U : 8U),
                   "neighboring live claim/election slots never alias each other");
        if (sequence == 3)
            expect(logical_counters[2] == logical_counters[0],
                   "physical retirement permits exact slot reuse with fresh device election");
        const auto current_witness = state.completed;
        auto progress = session.persistent_progress(sequence);
        expect(progress && progress->observes_block_completion == observes &&
                   !progress->has_completed_block && !progress->physical_stops,
               "SDK success and counter claims do not fabricate completion publication");
        if (observes) {
            expect(std::atomic_ref(completion_word()).load() == 0,
                   "initial launch clears the exact reused witness");
            std::atomic_ref(completion_word()).store(1, std::memory_order_release);
            expect(session.persistent_progress(sequence)->has_completed_block &&
                       !session.poll().retired &&
                       !session.persistent_progress(sequence)->physical_stops,
                   "injected publication cannot retire an actually pending event");
        }
        if (is_batched) {
            expect(session.request_retreat(sequence) == CUDA_SUCCESS,
                   "request retained library retreat");
            if (uses_sharded_host_retreat) {
                const auto base = reinterpret_cast<CUdeviceptr>(driver.hosts[0].data.data()) + 128;
                for (std::size_t slot = 0; slot < 2; ++slot)
                    for (std::size_t index = 0; index < kGemmSignalShards; ++index) {
                        auto& word = mapped_word(base + slot * kGemmSignalSlotBytes +
                                                 index * kGemmSignalShardBytes);
                        expect(std::atomic_ref(word).load(std::memory_order_acquire) ==
                                   (base + slot * kGemmSignalSlotBytes == state.retreat ? 1U : 0U),
                               "retreat publishes every shard of only the selected slot");
                    }
            }
            driver.ready_launch = driver.launches;
            expect(session.poll().paused == 1 &&
                       session.persistent_progress(sequence)->completed_blocks == 3,
                   "independent stopped event proves incomplete work, not the witness");
            driver.next_claimed = 8;
            expect(session.resume(sequence).state == CudaExecutionIssueState::kIssued,
                   "resume exactly the retained original library command");
            if (observes)
                expect(session.persistent_progress(sequence)->has_completed_block &&
                           state.witnesses.back() == current_witness,
                       "physical continuation preserves the original mapped witness");
        }
        driver.query_status = CUDA_SUCCESS;
        expect(session.poll().retired == 1 && driver.counter_initializations == sequence,
               "only logical initial issue zeroes the counter; final event retires once");
    }
    success(session.download({c.id, 0}, std::as_writable_bytes(std::span{values})));
    for (std::size_t index = 0; index < values.size(); ++index)
        expect(values[index] == (is_batched || !index ? 28.75F : 6.0F),
               "every fake output/padding applies non-idempotent beta exactly three times");
    expect(!state.native_calls && state.launches == (is_batched ? 6U : 3U),
           "replacement never replays native cuBLAS compute");
    expect(session.poll().status == CUDA_SUCCESS, "retire readback prefix");
    expect(session.destroy_blas(handle.id).status == CUDA_SUCCESS, "destroy original owned handle");
    success(session.shutdown());
    expect(driver.live_resources() == 0,
           "all native and mapped owners clean once after real tails");
}
void geometry_contract(unsigned int tile_side, bool is_batched) {
    driver = {};
    state = {};
    state.tile_side = tile_side;
    CudaSessionPersistentConfig settings{320};
    settings.replace_sgemm = true;
    settings.gemm_tile_side = tile_side;
    CudaExecutionSession session({1, 99}, 101, dispatch(), limits(1000), settings);
    success(session.initialize(0));
    const auto handle = session.create_blas();
    expect(handle.status == CUDA_SUCCESS && handle.original_status == CUBLAS_STATUS_SUCCESS,
           "selected geometry owns its validated original BLAS handle");
    const auto a = session.allocate(4), b = session.allocate(132), c = session.allocate(132);
    success(a);
    success(b);
    success(c);
    CudaSessionStridedGemmEx problem;
    problem.operation.operation = {.handle = handle.id,
                                   .transpose_a = CUBLAS_OP_N,
                                   .transpose_b = CUBLAS_OP_N,
                                   .rows = 1,
                                   .columns = 33,
                                   .depth = 1,
                                   .leading_a = 1,
                                   .leading_b = 1,
                                   .leading_c = 1,
                                   .a = {a.id, 0},
                                   .b = {b.id, 0},
                                   .c = {c.id, 0},
                                   .alpha = 1,
                                   .beta = 0.5F};
    std::memcpy(problem.operation.alpha.data(), &problem.operation.operation.alpha, 4);
    std::memcpy(problem.operation.beta.data(), &problem.operation.operation.beta, 4);
    problem.operation.a_type = problem.operation.b_type = problem.operation.c_type = CUDA_R_16F;
    problem.strides = {1, 33, 33};
    problem.batches = 2;
    const auto accepted = is_batched ? session.accept_strided_gemm_ex(1, problem)
                                     : session.accept_gemm_ex(1, problem.operation);
    const auto expected_tiles = (tile_side == 32 ? 2U : 1U) * (is_batched ? 2U : 1U);
    const auto progress = session.persistent_progress(1);
    expect(accepted.accepted() && progress && progress->total_blocks == expected_tiles &&
               !progress->physical_launches && !state.launches && !state.native_calls,
           "exact independent complete-tile count is fixed before any GPU issue");
    session.close();
    expect(session.reject_queued(CUDA_ERROR_DEINITIALIZED) == 1,
           "the unissued command receives one explicit pre-issue retirement");
    const auto rejected = session.host_result(accepted.receipt);
    expect(rejected && rejected->status == CUDA_ERROR_DEINITIALIZED &&
               session.acknowledge_result(accepted.receipt),
           "consume the exact unissued rejection before releasing its owner");
    success(session.shutdown());
    expect(driver.context_destroys == 1, "unissued geometry and provider owners clean up exactly");
}
void invalid_geometry() {
    for (const auto side : {0U, 16U, 33U, 128U}) {
        driver = {};
        CudaSessionPersistentConfig settings{320};
        settings.replace_sgemm = true;
        settings.gemm_tile_side = side;
        bool rejected{};
        try {
            CudaExecutionSession session({1, 99}, 101, dispatch(), limits(1000), settings);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        expect(rejected && driver.context_creates == 0 && cuda_sgemm_image(side).empty(),
               "unsupported tile families reject before SDK ownership");
    }
    {
        driver = {};
        CudaSessionPersistentConfig settings{320};
        settings.gemm_tile_side = 32;
        bool rejected{};
        try {
            CudaExecutionSession session({1, 99}, 101, dispatch(), limits(1000), settings);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        expect(rejected && driver.context_creates == 0,
               "nondefault geometry cannot silently apply to a native-only Session");
    }
    expect(
        cuda_gemm_matrix_tiles(768, 3072, 64) == 576 &&
            cuda_gemm_matrix_tiles(768, 3072, 32) == 2304 &&
            cuda_gemm_matrix_tiles(UINT64_MAX, UINT64_MAX, 32) == 0 &&
            cuda_gemm_matrix_tiles(1, 1, 0) == 0 &&
            cuda_gemm_matrix_tiles(0, UINT64_MAX, 64) == 0 &&
            cuda_sgemm_image(32) != cuda_sgemm_image(64),
        "family geometry has checked counts and distinct immutable code, never wrapped progress");
}
void pool_budget() {
    for (const auto& [observes, replaces] :
         {std::pair{false, true}, std::pair{true, false}, std::pair{true, true}}) {
        driver = {};
        state = {};
        const auto bytes = std::size_t{2} * (observes && replaces ? 16 : 8);
        CudaSessionPersistentConfig settings{320};
        settings.replace_sgemm = replaces;
        settings.observes_block_completion = observes;
        auto budget = limits(1000);
        budget.max_device_bytes = bytes - 1;
        bool rejected{};
        try {
            CudaExecutionSession too_small({1, 99}, 101, dispatch(), budget, settings);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        expect(rejected && !driver.context_creates && !driver.allocation_calls,
               "full progress stride budget rejects before native resource entry");
        budget.max_device_bytes = bytes;
        auto sdk = dispatch();
        sdk.upload_async = glimmer::test::cuda_domain::upload_async;
        CudaExecutionSession exact({1, 99}, 101, sdk, budget, settings);
        success(exact.initialize(0));
        expect(exact.snapshot().device_bytes == bytes,
               "observed GEMM charges both device words; generic/default retain old charge");
        success(exact.shutdown());
        expect(driver.live_resources() == 0, "exact bounded progress storage retires normally");
    }
}
void sharded_signal_budget() {
    driver = {};
    state = {};
    state.tile_side = 32;
    state.uses_sharded_host_retreat = true;
    CudaSessionPersistentConfig settings{320};
    settings.replace_sgemm = true;
    settings.gemm_tile_side = 32;
    settings.uses_sharded_host_retreat = true;
    auto budget = limits(1000);
    constexpr auto required = 2 * (64 + kGemmSignalSlotBytes);
    budget.max_private_host_bytes = required - 1;
    bool rejected{};
    try {
        CudaExecutionSession too_small({1, 99}, 101, dispatch(), budget, settings);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    expect(rejected && !driver.context_creates && !driver.allocation_calls,
           "sharded progress rejects an insufficient host budget before native work");
    budget.max_private_host_bytes = required;
    CudaExecutionSession exact({1, 99}, 101, dispatch(), budget, settings);
    success(exact.initialize(0));
    expect(exact.snapshot().private_host_bytes == required,
           "the private host budget charges every mapped shard and progress slot");
    success(exact.shutdown());
    expect(!driver.live_resources(), "the single expanded progress mapping retires normally");

    for (unsigned int invalid = 0; invalid < 3; ++invalid) {
        driver = {};
        state = {};
        auto bad = settings;
        auto bad_budget = limits(1000);
        if (invalid == 0) bad.gemm_tile_side = 64;
        if (invalid == 1) bad.replace_sgemm = false;
        if (invalid == 2) {
            bad_budget.max_private_host_bytes = std::numeric_limits<std::size_t>::max();
            bad_budget.submissions.max_commands =
                std::numeric_limits<std::size_t>::max() / (64 + kGemmSignalSlotBytes) + 1;
        }
        rejected = false;
        try {
            CudaExecutionSession invalid_session({1, 99}, 101, dispatch(), bad_budget, bad);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        expect(rejected && !driver.context_creates && !driver.allocation_calls,
               "unsupported shard layouts and overflow reject before native acquisition");
    }

    driver = {};
    state = {};
    state.tile_side = 32;
    state.uses_sharded_host_retreat = true;
    driver.requires_explicit_cleanup = true;
    CudaExecutionDomainOwner owner(domain_dispatch(), {99, 0, CU_CTX_MAP_HOST, 2});
    expect(owner.initialize() == CUDA_SUCCESS, "create the shared shard failure domain");
    auto first = owner.create_member({{1, 99}, 101});
    auto second = owner.create_member({{2, 99}, 202});
    {
        CudaExecutionSession healthy({1, 99}, 101, dispatch(), limits(1000), settings, {}, {},
                                     false, {}, std::move(first.member));
        CudaExecutionSession failing({2, 99}, 202, dispatch(), limits(2000), settings, {}, {},
                                     false, {}, std::move(second.member));
        success(healthy.initialize(0));
        success(healthy.deactivate());
        driver.next_host_map_status = CUDA_ERROR_OUT_OF_MEMORY;
        const auto result = failing.initialize(0);
        expect(result.status == CUDA_ERROR_OUT_OF_MEMORY && !owner.snapshot().is_quarantined &&
                   driver.hosts[0].is_live && !driver.hosts[1].is_live && driver.host_frees == 1,
               "a clean shard mapping failure rolls back only the failing member");
        success(failing.shutdown());
        success(healthy.activate());
        expect(healthy.snapshot().private_host_bytes == required,
               "the other member retains its exact mapped shard budget");
        success(healthy.shutdown());
    }
    expect(owner.close() == CUDA_SUCCESS && owner.shutdown() == CUDA_SUCCESS &&
               !driver.live_resources(),
           "shared shard mapping rollback preserves normal domain cleanup");
}

void sharded_shared_domain_isolation() {
    driver = {};
    state = {};
    state.tile_side = 32;
    state.is_batched = state.uses_sharded_host_retreat = true;
    driver.requires_explicit_cleanup = true;
    CudaExecutionDomainOwner owner(domain_dispatch(), {99, 0, CU_CTX_MAP_HOST, 2});
    expect(owner.initialize() == CUDA_SUCCESS, "create the shared physical domain");
    auto first = owner.create_member({{1, 99}, 101});
    auto second = owner.create_member({{2, 99}, 202});
    expect(first.status == CUDA_SUCCESS && second.status == CUDA_SUCCESS,
           "create independent logical members");
    CudaSessionPersistentConfig settings{320};
    settings.replace_sgemm = settings.uses_sharded_host_retreat = true;
    settings.gemm_tile_side = 32;
    const auto checked = [](const CudaSessionResult& result, const char* operation) {
        if (result.disposition != CudaSessionDisposition::kComplete ||
            result.status != CUDA_SUCCESS)
            throw std::runtime_error(std::string{operation} + ": " +
                                     std::to_string(static_cast<int>(result.status)));
    };
    {
        CudaExecutionSession left({1, 99}, 101, dispatch(), limits(1000), settings, {}, {}, false,
                                  {}, std::move(first.member));
        CudaExecutionSession right({2, 99}, 202, dispatch(), limits(2000), settings, {}, {}, false,
                                   {}, std::move(second.member));
        struct Input {
            std::uint64_t handle{}, output{};
            glimmer::backend::ExecutionReceipt receipt{};
            CUdeviceptr signal{};
            CudaSessionStridedGemmEx problem{};
        };
        const auto begin = [&](CudaExecutionSession& session) {
            const auto handle = session.create_blas();
            if (handle.status != CUDA_SUCCESS)
                throw std::runtime_error("prepare shared GEMM provider: " +
                                         std::to_string(static_cast<int>(handle.status)));
            const auto a = session.allocate(32), b = session.allocate(32), c = session.allocate(32);
            checked(a, "allocate shared A");
            checked(b, "allocate shared B");
            checked(c, "allocate shared C");
            std::array<float, 8> data{};
            for (const auto& [allocation, value] :
                 {std::pair{a.id, 2.0F}, std::pair{b.id, 4.0F}, std::pair{c.id, 6.0F}}) {
                data.fill(value);
                checked(session.upload({allocation, 0}, std::as_bytes(std::span{data})),
                        "upload shared operand");
                expect(session.poll().status == CUDA_SUCCESS,
                       "retire the actual shared proxy upload before cold setup continues");
            }
            CudaSessionStridedGemmEx problem;
            problem.operation.operation = {.handle = handle.id,
                                           .transpose_a = CUBLAS_OP_N,
                                           .transpose_b = CUBLAS_OP_N,
                                           .rows = 1,
                                           .columns = 1,
                                           .depth = 1,
                                           .leading_a = 1,
                                           .leading_b = 1,
                                           .leading_c = 1,
                                           .a = {a.id, 0},
                                           .b = {b.id, 0},
                                           .c = {c.id, 0},
                                           .alpha = 2.0F,
                                           .beta = 0.5F};
            std::memcpy(problem.operation.alpha.data(), &problem.operation.operation.alpha, 4);
            std::memcpy(problem.operation.beta.data(), &problem.operation.operation.beta, 4);
            problem.strides = {1, 1, 1};
            problem.batches = 8;
            return Input{handle.id, c.id, {}, 0, problem};
        };
        const auto check_shards = [](CUdeviceptr signal, std::uint32_t expected) {
            for (std::size_t index = 0; index < kGemmSignalShards; ++index) {
                auto& word = mapped_word(signal + index * kGemmSignalShardBytes);
                expect(std::atomic_ref(word).load(std::memory_order_acquire) == expected,
                       "all exact member shards have the expected generation");
            }
        };
        checked(left.initialize(0), "initialize left");
        auto left_input = begin(left);
        checked(left.deactivate(), "deactivate prepared left");
        checked(right.initialize(0), "initialize right");
        auto right_input = begin(right);
        const auto right_accepted = right.accept_strided_sgemm(1, right_input.problem);
        expect(right_accepted.accepted(), "second member accepts its sharded GEMM");
        right_input.receipt = right_accepted.receipt;
        checked(right.deactivate(), "deactivate prepared right");
        checked(left.activate(), "activate prepared left");
        const auto left_accepted = left.accept_strided_sgemm(1, left_input.problem);
        expect(left_accepted.accepted(), "first member accepts its sharded GEMM");
        left_input.receipt = left_accepted.receipt;
        driver.query_status = CUDA_ERROR_NOT_READY;
        driver.next_claimed = 3;
        expect(left.issue_next().state == CudaExecutionIssueState::kIssued &&
                   left.acknowledge_result(left_input.receipt),
               "issue the first member's sharded GEMM");
        left_input.signal = state.retreat;
        checked(left.deactivate(), "deactivate issued left");
        checked(right.activate(), "activate issued right");
        expect(right.issue_next().state == CudaExecutionIssueState::kIssued &&
                   right.acknowledge_result(right_input.receipt),
               "issue the second member's sharded GEMM");
        right_input.signal = state.retreat;
        expect(left_input.signal + kGemmSignalSlotBytes <= right_input.signal ||
                   right_input.signal + kGemmSignalSlotBytes <= left_input.signal,
               "members have non-overlapping mapped signal arrays");
        checked(right.deactivate(), "deactivate issued right");
        checked(left.activate(), "activate left to retreat");
        expect(left.request_retreat(1) == CUDA_SUCCESS, "retreat only the first member");
        check_shards(left_input.signal, 1);
        check_shards(right_input.signal, 0);
        driver.ready_launch = 1;
        expect(left.poll().paused == 1, "first member physically stops partially");
        driver.next_claimed = 8;
        expect(left.resume(1).state == CudaExecutionIssueState::kIssued,
               "first member resumes after its physical stop");
        check_shards(left_input.signal, 0);
        checked(left.deactivate(), "deactivate resumed left");
        checked(right.activate(), "activate right to retreat");
        expect(right.request_retreat(1) == CUDA_SUCCESS, "retreat only the second member");
        check_shards(left_input.signal, 0);
        check_shards(right_input.signal, 1);
        driver.ready_launch = 2;
        expect(right.poll().paused == 1, "second member physically stops partially");
        expect(right.resume(1).state == CudaExecutionIssueState::kIssued,
               "second member resumes after its physical stop");
        check_shards(right_input.signal, 0);
        driver.query_status = CUDA_SUCCESS;
        expect(right.poll().retired == 1, "second member completes its exact command");
        std::array<float, 8> output{};
        checked(right.deactivate(), "deactivate completed right");
        checked(left.activate(), "activate left to retire");
        expect(left.poll().retired == 1, "first member completes its exact command");
        checked(left.download({left_input.output, 0}, std::as_writable_bytes(std::span{output})),
                "download left");
        expect(left.poll().status == CUDA_SUCCESS, "retire left proxy readback");
        expect(std::ranges::all_of(output, [](float value) { return value == 19.0F; }),
               "first member preserves every non-idempotent result");
        expect(left.destroy_blas(left_input.handle).status == CUDA_SUCCESS,
               "retire first owned BLAS handle");
        checked(left.shutdown(), "shutdown left");
        checked(right.activate(), "activate right to retire");
        checked(right.download({right_input.output, 0}, std::as_writable_bytes(std::span{output})),
                "download right");
        expect(right.poll().status == CUDA_SUCCESS, "retire right proxy readback");
        expect(std::ranges::all_of(output, [](float value) { return value == 19.0F; }),
               "second member preserves every non-idempotent result");
        expect(right.destroy_blas(right_input.handle).status == CUDA_SUCCESS,
               "retire second owned BLAS handle");
        checked(right.shutdown(), "shutdown right");
    }
    expect(owner.close() == CUDA_SUCCESS && owner.shutdown() == CUDA_SUCCESS &&
               !driver.live_resources(),
           "shared members release their own mappings before domain teardown");
}

void sharded_and_generic_signal_isolation() {
    driver = {};
    state = {};
    state.tile_side = 32;
    state.is_batched = state.uses_sharded_host_retreat = true;
    auto sdk = dispatch();
    sdk.module_load = glimmer::test::cuda_domain::module_load;
    sdk.function_get = [](CUfunction* output, CUmodule module, const char* name) {
        return module == reinterpret_cast<CUmodule>(&driver.modules[0])
                   ? dispatch().function_get(output, module, name)
                   : glimmer::test::cuda_domain::function_get(output, module, name);
    };
    // NOLINTBEGIN(bugprone-easily-swappable-parameters): exact injected CUDA ABIs.
    sdk.parameter_info = [](CUfunction function, std::size_t index, std::size_t* offset,
                            std::size_t* bytes) {
        const bool is_gemm = std::ranges::any_of(state.functions, [&](const auto& candidate) {
            return function == reinterpret_cast<CUfunction>(const_cast<FakeFunction*>(&candidate));
        });
        return is_gemm ? dispatch().parameter_info(function, index, offset, bytes)
                       : glimmer::test::cuda_domain::parameter_info(function, index, offset, bytes);
    };
    sdk.execution.launch_kernel = [](CUfunction function, unsigned int x, unsigned int y,
                                     unsigned int z, unsigned int bx, unsigned int by,
                                     unsigned int bz, unsigned int shared, CUstream stream,
                                     void** parameters, void** extra) {
        const bool is_gemm = std::ranges::any_of(state.functions, [&](const auto& candidate) {
            return function == reinterpret_cast<CUfunction>(const_cast<FakeFunction*>(&candidate));
        });
        if (is_gemm)
            return dispatch().execution.launch_kernel(function, x, y, z, bx, by, bz, shared, stream,
                                                      parameters, extra);
        std::memcpy(&state.generic_retreat, static_cast<const std::byte*>(extra[1]) + 32,
                    sizeof(state.generic_retreat));
        return glimmer::test::cuda_domain::launch(function, x, y, z, bx, by, bz, shared, stream,
                                                  parameters, extra);
    };
    // NOLINTEND(bugprone-easily-swappable-parameters)
    CudaSessionPersistentConfig settings{320};
    settings.replace_sgemm = settings.uses_sharded_host_retreat = true;
    settings.gemm_tile_side = 32;
    CudaExecutionSession session({1, 99}, 101, sdk, limits(1000), settings);
    success(session.initialize(0));
    const auto handle = session.create_blas();
    expect(handle.status == CUDA_SUCCESS, "prepare the sharded GEMM image first");
    const auto a = session.allocate(32), b = session.allocate(32), c = session.allocate(32);
    const auto generic_output = session.allocate(32);
    const auto stream = session.create_stream(CU_STREAM_NON_BLOCKING);
    const auto module = session.load_ptx(kSource);
    for (const auto& result : {a, b, c, generic_output, stream, module}) success(result);
    const auto function = session.get_function(module.id, "kernel");
    success(function);
    std::array<float, 8> values{};
    for (const auto& [allocation, value] :
         {std::pair{a.id, 2.0F}, std::pair{b.id, 4.0F}, std::pair{c.id, 6.0F}}) {
        values.fill(value);
        success(session.upload({allocation, 0}, std::as_bytes(std::span{values})));
    }
    KernelInput generic({function.id, stream.id}, generic_output.address);
    const auto generic_accepted = session.accept_kernel(1, generic.kernel);
    CudaSessionStridedGemmEx problem;
    problem.operation.operation = {.handle = handle.id,
                                   .transpose_a = CUBLAS_OP_N,
                                   .transpose_b = CUBLAS_OP_N,
                                   .rows = 1,
                                   .columns = 1,
                                   .depth = 1,
                                   .leading_a = 1,
                                   .leading_b = 1,
                                   .leading_c = 1,
                                   .a = {a.id, 0},
                                   .b = {b.id, 0},
                                   .c = {c.id, 0},
                                   .alpha = 2.0F,
                                   .beta = 0.5F};
    std::memcpy(problem.operation.alpha.data(), &problem.operation.operation.alpha, 4);
    std::memcpy(problem.operation.beta.data(), &problem.operation.operation.beta, 4);
    problem.strides = {1, 1, 1};
    problem.batches = 8;
    const auto gemm_accepted = session.accept_strided_sgemm(2, problem);
    expect(generic_accepted.accepted() && gemm_accepted.accepted(),
           "accept generic and sharded GEMM work on independent streams");
    driver.query_status = CUDA_ERROR_NOT_READY;
    driver.next_claimed = 3;
    expect(session.issue_next().state == CudaExecutionIssueState::kIssued &&
               session.acknowledge_result(generic_accepted.receipt),
           "issue the generic transformed kernel");
    const auto generic_signal = state.generic_retreat;
    expect(session.issue_next().state == CudaExecutionIssueState::kIssued &&
               session.acknowledge_result(gemm_accepted.receipt),
           "issue the independently scheduled sharded GEMM");
    const auto gemm_signal = state.retreat;
    const auto base = reinterpret_cast<CUdeviceptr>(driver.hosts[0].data.data());
    expect(generic_signal == base && gemm_signal == base + 128 + kGemmSignalSlotBytes,
           "generic PTX keeps its scalar slot while GEMM uses its own appended shard array");
    auto& generic_word = mapped_word(generic_signal);
    const auto check_shards = [&](std::uint32_t expected) {
        for (std::size_t index = 0; index < kGemmSignalShards; ++index) {
            auto& word = mapped_word(gemm_signal + index * kGemmSignalShardBytes);
            expect(std::atomic_ref(word).load(std::memory_order_acquire) == expected,
                   "GEMM shards are isolated from the generic scalar flag");
        }
    };
    expect(session.request_retreat(1) == CUDA_SUCCESS &&
               std::atomic_ref(generic_word).load(std::memory_order_acquire) == 1,
           "generic request sets only its scalar flag");
    check_shards(0);
    expect(session.request_retreat(2) == CUDA_SUCCESS, "request the independent GEMM retreat");
    check_shards(1);
    driver.ready_launch = 1;
    expect(session.poll().paused == 1, "generic physical event proves partial stop");
    driver.next_claimed = 8;
    expect(session.resume(1).state == CudaExecutionIssueState::kIssued &&
               std::atomic_ref(generic_word).load(std::memory_order_acquire) == 0,
           "generic continuation clears only its own stopped scalar flag");
    check_shards(1);
    driver.ready_launch = 2;
    expect(
        session.poll().paused == 1 && session.resume(2).state == CudaExecutionIssueState::kIssued,
        "GEMM continuation waits for its own physical stop");
    check_shards(0);
    driver.query_status = CUDA_SUCCESS;
    expect(session.poll().retired == 2, "both independent physical tails retire");
    std::array<std::uint32_t, 8> generic_values{};
    success(session.download({generic_output.id, 0},
                             std::as_writable_bytes(std::span{generic_values})));
    expect(std::ranges::all_of(generic_values, [](auto value) { return value == 7U; }),
           "generic transformed output is neither lost nor replayed");
    success(session.download({c.id, 0}, std::as_writable_bytes(std::span{values})));
    expect(std::ranges::all_of(values, [](float value) { return value == 19.0F; }),
           "GEMM output is neither lost nor replayed");
    expect(session.destroy_blas(handle.id).status == CUDA_SUCCESS, "retire owned BLAS handle");
    success(session.shutdown());
    expect(!driver.live_resources(), "all scalar and shard resources retire exactly once");
}

void invalid_metadata() {
    const auto child = ::fork();
    expect(child >= 0, "isolated unknown-provider ownership control");
    if (!child) {
        driver = {};
        state = {};
        state.observes = state.has_bad_metadata = true;
        CudaSessionPersistentConfig settings{320};
        settings.replace_sgemm = settings.observes_block_completion = true;
        CudaExecutionSession session({1, 99}, 101, dispatch(), limits(1000), settings);
        success(session.initialize(0));
        const auto handle = session.create_blas();
        // Partially prepared uncertain owners are retained. This child does
        // not claim healthy leak-free cleanup for that quarantined state.
        std::_Exit(handle.status == CUDA_ERROR_INVALID_VALUE && !handle.original_status &&
                           !state.launches && !state.native_calls && !driver.launches
                       ? EXIT_SUCCESS
                       : EXIT_FAILURE);
    }
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "wrong appended pointer width rejects before original SDK or compute entry");
}
}  // namespace

int main() {
    try {
        if (cuda_sgemm_image().empty()) {
            std::cout
                << "GEMM completion provider unavailable; GPU provider checks remain required\n";
            return EXIT_SUCCESS;
        }
        retreat_load_contract();
        for (const bool observes : {false, true})
            for (const bool batched : {false, true}) run(observes, batched);
        for (const bool observes : {false, true})
            for (const bool batched : {false, true}) run(observes, batched, 32);
        for (const bool observes : {false, true})
            for (const bool batched : {false, true}) run(observes, batched, 32, true);
        for (const auto side : {32U, 64U})
            for (const bool batched : {false, true}) geometry_contract(side, batched);
        invalid_geometry();
        pool_budget();
        sharded_signal_budget();
        sharded_shared_domain_isolation();
        sharded_and_generic_signal_isolation();
        invalid_metadata();
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "GEMM completion: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
