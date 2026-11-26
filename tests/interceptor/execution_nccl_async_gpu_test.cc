#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nccl.h>

#include <array>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>

namespace {
constexpr std::size_t kCount = 1024;
constexpr std::size_t kGuard = 16;
constexpr std::size_t kExtent = kCount + 2 * kGuard;
constexpr std::size_t kBytes = kExtent * sizeof(float);
constexpr std::size_t kMaximumQueries = 4096;
constexpr float kMarker = -1048576.0F;
using Values = std::array<float, kExtent>;

struct Profile {
    std::string_view placement;
    bool grouped{};
    std::string_view cleanup_kind;
};

struct CallSite {
    std::string_view operation;
    unsigned int phase{}, communicator{};
};

struct Observation {
    CallSite site;
    int status{}, before{}, after{}, async_status{-1};
};

struct Report {
    std::array<Observation, kMaximumQueries + 64> observations{};
    std::size_t observation_count{};
    std::array<std::uint64_t, 12> checksums{};
    std::size_t checksum_count{}, checked_values{};
    bool has_cleanup{}, input_unchanged{}, has_numerical_pass{};
};

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void checked(CUresult status, const char* operation) {
    if (status != CUDA_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": Driver " +
                                 std::to_string(static_cast<int>(status)));
}

void checked_runtime(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": Runtime " +
                                 std::to_string(static_cast<int>(status)));
}

template <typename T>
struct Guarded {
    std::array<std::byte, 16> before{};
    T value;
    std::array<std::byte, 16> after{};

    explicit Guarded(T initial = {}) : value(initial) {
        before.fill(std::byte{0x5a});
        after.fill(std::byte{0xa5});
    }

    [[nodiscard]] bool intact() const noexcept {
        for (const auto byte : before)
            if (byte != std::byte{0x5a}) return false;
        for (const auto byte : after)
            if (byte != std::byte{0xa5}) return false;
        return true;
    }
};

// Byte-aligned suffixes catch an overrun immediately after each public output,
// including four-byte scalar outputs; no alignment padding hides a write.
static_assert(offsetof(Guarded<int>, after) == 16 + sizeof(int));
static_assert(offsetof(Guarded<ncclResult_t>, after) == 16 + sizeof(ncclResult_t));
static_assert(offsetof(Guarded<ncclUniqueId>, after) == 16 + sizeof(ncclUniqueId));
static_assert(offsetof(Guarded<ncclComm_t>, after) == 16 + sizeof(ncclComm_t));
static_assert(offsetof(Guarded<ncclConfig_t>, after) == 16 + sizeof(ncclConfig_t));
static_assert(sizeof(ncclResult_t) == sizeof(int));
static_assert(sizeof(float) == sizeof(std::uint32_t));

struct Communicator {
    Guarded<ncclUniqueId> id;
    Guarded<ncclComm_t> output;
    Guarded<ncclConfig_t> config{NCCL_CONFIG_INITIALIZER};
    std::array<std::byte, sizeof(ncclUniqueId)> saved_id{};
    std::array<std::byte, sizeof(ncclConfig_t)> saved_config{};
    CUstream stream{};
    std::array<CUdeviceptr, 2> memory{};
    bool has_id{}, is_ready{}, is_retired{};

    Communicator() {
        config.value.blocking = 0;
        std::memcpy(saved_config.data(), &config.value, saved_config.size());
    }
};

Values expected_values(unsigned int communicator, unsigned int round) {
    Values result{};
    result.fill(kMarker);
    for (std::size_t index = 0; index != kCount; ++index)
        result[kGuard + index] = static_cast<float>(index) * 0.25F +
                                 static_cast<float>(communicator) * 16.0F +
                                 static_cast<float>(round) * 0.5F;
    return result;
}

std::uint64_t fingerprint(const Values& values) noexcept {
    std::uint64_t result = 14695981039346656037ULL;
    for (const auto value : values) {
        const auto bits = std::bit_cast<std::uint32_t>(value);
        for (unsigned int byte = 0; byte != 4; ++byte) {
            result ^= (bits >> (8 * byte)) & 255U;
            result *= 1099511628211ULL;
        }
    }
    return result;
}

// All SDK owners have explicit checked teardown only after every readiness loop
// and actual source-stream completion. Destruction is CPU-only: on an unknown
// result, failed proof or timeout, retain all native resources until process exit.
// In particular, failure never speculates an Abort, GroupEnd, synchronize or reset.
class Probe {
   public:
    Probe(Profile profile, Report& report) : profile_(profile), report_(report) {}

    void run() {
        checked(cuInit(0), "initialize Driver");
        checked_runtime(cudaSetDevice(0), "establish Runtime primary context");
        checked(cuCtxGetCurrent(&primary_), "observe primary context");
        require(primary_ != nullptr, "primary context exists");
        checked(cuCtxCreate(&user_, 0, 0), "create explicit user context");
        require(user_ && user_ != primary_ && current() == user_,
                "distinct explicit user context is current");
        resource_context_ = profile_.placement == "primary-after" ? primary_ : user_;
        if (resource_context_ == user_) create_resources();

        Guarded<int> version;
        success({"GetVersion", 0, 0}, [&] { return ncclGetVersion(&version.value); });
        require(version.intact() && version.value == 22903,
                "guarded version identifies the installed 22903 SDK");
        for (unsigned int index = 0; index != communicators_.size(); ++index) initialize(index);
        require(communicators_[0].output.value != communicators_[1].output.value,
                "independent initializers publish distinct live communicator handles");
        const auto observed_post_init = current();
        if (resource_context_ == primary_) create_resources();
        calling_context_ = profile_.placement == "user-restored" ? user_ : observed_post_init;
        if (profile_.placement == "primary-after") calling_context_ = primary_;

        for (unsigned int round = 1; round != 4; ++round) run_round(round);
        require(report_.checked_values == 12672 && report_.checksum_count == 12,
                "all rounds checked four full independent guarded arrays");
        cleanup();
        validate_inputs();
        report_.input_unchanged = true;
        report_.has_numerical_pass = true;
    }

   private:
    Profile profile_;
    Report& report_;
    CUcontext primary_{}, user_{}, resource_context_{}, calling_context_{};
    std::array<Communicator, 2> communicators_{};
    std::size_t query_count_{};

    CUcontext current() const {
        CUcontext result{};
        checked(cuCtxGetCurrent(&result), "observe actual current context");
        require(result && (result == primary_ || result == user_),
                "current context has a known retained identity");
        return result;
    }

    int context_id() const {
        return current() == primary_ ? 1 : 2;
    }

    void select(CUcontext context) const {
        checked(cuCtxSetCurrent(context), "select exact source/caller context");
        require(current() == context, "selected context is independently observed");
    }

    void validate_inputs() const {
        for (const auto& item : communicators_) {
            require(item.id.intact() && item.output.intact() && item.config.intact(),
                    "all UID, communicator and configuration boundary guards remain intact");
            require(std::memcmp(&item.config.value, item.saved_config.data(),
                                item.saved_config.size()) == 0,
                    "every configuration byte remains unchanged");
            if (item.has_id)
                require(
                    std::memcmp(&item.id.value, item.saved_id.data(), item.saved_id.size()) == 0,
                    "every generated UID byte remains unchanged after initialization");
        }
    }

    template <typename Call>
    int invoke(CallSite site, Call call, const ncclResult_t* async_result = nullptr) {
        require(report_.observation_count < report_.observations.size(),
                "bounded native observation ledger");
        const auto before = context_id();
        const auto status = static_cast<int>(call());
        auto& row = report_.observations[report_.observation_count++];
        row = {site, status, before, 0, -1};
        if (async_result) std::memcpy(&row.async_status, async_result, sizeof(row.async_status));
        // This is a CPU current-context observation, not a stream operation or
        // readiness assumption. Preserve the original result even if it fails.
        row.after = context_id();
        validate_inputs();
        require(status >= 0 && status <= 7, "native result is in the actual NCCL domain");
        return status;
    }

    template <typename Call>
    void success(CallSite site, Call call) {
        require(invoke(site, call) == 0, "expected original NCCL success");
    }

    template <typename Call>
    void asynchronous(CallSite site, Call call) {
        const auto status = invoke(site, call);
        require(status == ncclSuccess || status == ncclInProgress,
                "original asynchronous NCCL call returned success or InProgress");
    }

    void ready(unsigned int index, unsigned int phase) {
        auto& item = communicators_[index];
        require(item.output.value && !item.is_retired,
                "query only the exact live published communicator");
        item.is_ready = false;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        for (;;) {
            require(query_count_ < kMaximumQueries && std::chrono::steady_clock::now() < deadline,
                    "bounded actual communicator readiness queries");
            ++query_count_;
            Guarded<ncclResult_t> observed{ncclInternalError};
            const auto status = invoke(
                {"GetAsyncError", phase, index + 1},
                [&] { return ncclCommGetAsyncError(item.output.value, &observed.value); },
                &observed.value);
            int raw{};
            std::memcpy(&raw, &observed.value, sizeof(raw));
            require(observed.intact(), "async output preserves both exact boundary guards");
            require(status == ncclSuccess && (raw == ncclSuccess || raw == ncclInProgress),
                    "query preserves original success and raw asynchronous zero or seven");
            require(std::chrono::steady_clock::now() < deadline,
                    "actual query completion remains within its readiness deadline");
            if (raw == ncclSuccess) {
                item.is_ready = true;
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    void initialize(unsigned int index) {
        auto& item = communicators_[index];
        select(user_);  // Each independent initializer starts in the user context.
        success({"GetUniqueId", 0, index + 1}, [&] { return ncclGetUniqueId(&item.id.value); });
        std::memcpy(item.saved_id.data(), &item.id.value, item.saved_id.size());
        item.has_id = true;
        if (profile_.grouped)
            success({"GroupStartInit", 0, index + 1}, [] { return ncclGroupStart(); });
        asynchronous({"InitRankConfig", 0, index + 1}, [&] {
            return ncclCommInitRankConfig(&item.output.value, 1, item.id.value, 0,
                                          &item.config.value);
        });
        if (profile_.grouped)
            asynchronous({"GroupEndInit", 0, index + 1}, [] { return ncclGroupEnd(); });
        // The selected SDK publishes this output during InitRankDev before its
        // asynchronous job. Readiness still requires this exact owner's query.
        require(item.output.value != nullptr, "initializer publishes a guarded communicator");
        ready(index, 0);
    }

    void create_resources() {
        select(resource_context_);
        for (auto& item : communicators_) {
            checked(cuStreamCreate(&item.stream, CU_STREAM_NON_BLOCKING),
                    "create one explicit source stream per communicator");
            require(item.stream != nullptr, "source stream output is nonnull");
            for (auto& address : item.memory) {
                checked(cuMemAlloc(&address, kBytes), "allocate a complete guarded source buffer");
                require(address != 0, "source allocation output is nonzero");
            }
        }
        require(communicators_[0].stream != communicators_[1].stream,
                "communicators have distinct actual source streams");
        for (std::size_t first = 0; first != 4; ++first)
            for (std::size_t second = first + 1; second != 4; ++second)
                require(communicators_[first / 2].memory[first % 2] !=
                            communicators_[second / 2].memory[second % 2],
                        "all four guarded allocations are independent");
        require(current() == resource_context_, "resource creation preserves exact source context");
    }

    void initialize_round(unsigned int round) {
        require(communicators_[0].is_ready && communicators_[1].is_ready,
                "both native participants are ready before ordinary CUDA work");
        select(resource_context_);
        Values empty{};
        empty.fill(kMarker);
        for (unsigned int index = 0; index != communicators_.size(); ++index) {
            const auto input = expected_values(index + 1, round);
            const auto& item = communicators_[index];
            checked(cuMemcpyHtoD(item.memory[0], input.data(), kBytes), "upload full guarded send");
            checked(cuMemcpyHtoD(item.memory[1], empty.data(), kBytes),
                    "upload full guarded receive");
        }
        checked(cuCtxSynchronize(), "complete every round upload before source-stream execution");
        select(calling_context_);
    }

    void seed_broadcasts() {
        const auto saved = current();
        select(resource_context_);
        // In grouped mode these ordinary source commands are inside the group.
        // Both precede either inner call, so an inner InProgress never authorizes
        // another CUDA enqueue. Distinct buffers preserve eager/deferred results.
        for (const auto& item : communicators_)
            checked(cuMemcpyDtoDAsync(item.memory[1] + kGuard * sizeof(float),
                                      item.memory[0] + kGuard * sizeof(float),
                                      kCount * sizeof(float), item.stream),
                    "seed each source stream's distinct in-place broadcast payload");
        select(saved);
    }

    void collective(unsigned int index, unsigned int round) {
        auto& item = communicators_[index];
        require(item.is_ready, "collective uses its actually ready communicator");
        item.is_ready = false;
        const auto stream = reinterpret_cast<cudaStream_t>(item.stream);
        // Actual SDK device addresses; no caller address is sent through a wire.
        // NOLINTBEGIN(performance-no-int-to-ptr)
        const auto* send = reinterpret_cast<const float*>(item.memory[0] + kGuard * sizeof(float));
        auto* receive = reinterpret_cast<float*>(item.memory[1] + kGuard * sizeof(float));
        // NOLINTEND(performance-no-int-to-ptr)
        if (round == 1)
            asynchronous({"AllReduce", round, index + 1}, [&] {
                return ncclAllReduce(send, receive, kCount, ncclFloat32, ncclSum, item.output.value,
                                     stream);
            });
        else if (round == 2)
            asynchronous({"AllGather", round, index + 1}, [&] {
                return ncclAllGather(send, receive, kCount, ncclFloat32, item.output.value, stream);
            });
        else
            asynchronous({"Bcast", round, index + 1}, [&] {
                return ncclBcast(receive, kCount, ncclFloat32, 0, item.output.value, stream);
            });
    }

    void check_round(unsigned int round) {
        require(communicators_[0].is_ready && communicators_[1].is_ready,
                "both exact participants are ready before any completion or readback");
        const auto saved = current();
        select(resource_context_);
        for (const auto& item : communicators_)
            checked(cuStreamSynchronize(item.stream), "observe actual source-stream completion");
        for (unsigned int index = 0; index != communicators_.size(); ++index) {
            const auto expected = expected_values(index + 1, round);
            for (const auto address : communicators_[index].memory) {
                Values actual{};
                checked(cuMemcpyDtoH(actual.data(), address, kBytes),
                        "read every payload and guard float from the source context");
                bool matches = true;
                for (std::size_t element = 0; element != actual.size(); ++element)
                    if (std::bit_cast<std::uint32_t>(actual[element]) !=
                        std::bit_cast<std::uint32_t>(expected[element]))
                        matches = false;
                require(report_.checksum_count < report_.checksums.size(), "bounded full hashes");
                report_.checksums[report_.checksum_count++] = fingerprint(actual);
                report_.checked_values += actual.size();
                require(matches, "complete exact send/receive payload and guard comparison");
            }
        }
        select(saved);
    }

    void run_round(unsigned int round) {
        initialize_round(round);
        if (profile_.grouped)
            success({"GroupStartCollective", round, 0}, [] { return ncclGroupStart(); });
        if (round == 3) seed_broadcasts();
        for (unsigned int index = 0; index != communicators_.size(); ++index) {
            collective(index, round);
            if (!profile_.grouped) ready(index, round);
        }
        if (profile_.grouped) {
            asynchronous({"GroupEndCollective", round, 0}, [] { return ncclGroupEnd(); });
            for (unsigned int index = 0; index != communicators_.size(); ++index)
                ready(index, round);
        }
        check_round(round);
    }

    void cleanup() {
        for (unsigned int index = 0; index != communicators_.size(); ++index) {
            auto& item = communicators_[index];
            require(item.is_ready && !item.is_retired, "cleanup follows readiness and stream stop");
            if (profile_.cleanup_kind == "destroy")
                success({"Destroy", 4, index + 1},
                        [&] { return ncclCommDestroy(item.output.value); });
            else
                success({"Abort", 4, index + 1}, [&] { return ncclCommAbort(item.output.value); });
            item.is_retired = true;
        }
        // No communicator query or access through NCCL follows destructive calls.
        select(resource_context_);
        for (auto& item : communicators_) {
            require(item.is_retired,
                    "release only a successfully retired communicator's resources");
            checked(cuStreamDestroy(item.stream), "destroy completed source stream");
            item.stream = nullptr;
            for (auto& address : item.memory) {
                checked(cuMemFree(address), "free exact completed source allocation");
                address = 0;
            }
        }
        select(user_);
        checked(cuCtxDestroy(user_), "destroy user context after all resources retire");
        user_ = nullptr;
        select(primary_);
        checked_runtime(cudaDeviceReset(), "release Runtime primary after all actual cleanup");
        primary_ = nullptr;
        report_.has_cleanup = true;
    }
};

void print(const Profile& profile, const Report& report) {
    std::cout << "{\"status\":\"nccl_async_public_observation\",\"placement\":\""
              << profile.placement << "\",\"grouped\":" << (profile.grouped ? "true" : "false")
              << ",\"cleanup_kind\":\"" << profile.cleanup_kind
              << "\",\"ranks\":1,\"communicators\":2,\"blocking\":0,\"observations\":[";
    for (std::size_t index = 0; index != report.observation_count; ++index) {
        if (index) std::cout << ',';
        const auto& row = report.observations[index];
        std::cout << "{\"operation\":\"" << row.site.operation << "\",\"phase\":" << row.site.phase
                  << ",\"communicator\":" << row.site.communicator << ",\"status\":" << row.status
                  << ",\"before\":" << row.before << ",\"after\":" << row.after
                  << ",\"async_status\":" << row.async_status << '}';
    }
    std::cout << "],\"checked_values\":" << report.checked_values << ",\"checksums\":[";
    for (std::size_t index = 0; index != report.checksum_count; ++index) {
        if (index) std::cout << ',';
        std::cout << report.checksums[index];
    }
    std::cout << "],\"cleanup\":" << (report.has_cleanup ? "true" : "false")
              << ",\"input_unchanged\":" << (report.input_unchanged ? "true" : "false")
              << ",\"outcome\":\"" << (report.has_numerical_pass ? "numerical_pass" : "failed")
              << "\",\"multi_rank_qualified\":false,\"full_device_qualified\":false,"
                 "\"physical_priority_qualified\":false,\"performance_qualified\":false}\n";
}

constexpr std::string_view kUsage =
    "usage: execution_nccl_async_gpu_test "
    "primary-after|user-observed|user-restored direct|grouped destroy|abort\n";
}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::cout << kUsage;
        return EXIT_SUCCESS;
    }
    if (argc != 4 ||
        (std::string_view(argv[1]) != "primary-after" &&
         std::string_view(argv[1]) != "user-observed" &&
         std::string_view(argv[1]) != "user-restored") ||
        (std::string_view(argv[2]) != "direct" && std::string_view(argv[2]) != "grouped") ||
        (std::string_view(argv[3]) != "destroy" && std::string_view(argv[3]) != "abort")) {
        std::cerr << kUsage;
        return EXIT_FAILURE;
    }
    const Profile profile{argv[1], std::string_view(argv[2]) == "grouped", argv[3]};
    Report report;
    try {
        Probe probe(profile, report);
        probe.run();
        print(profile, report);
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Public nonblocking NCCL observation: " << error.what() << '\n';
        print(profile, report);
        return EXIT_FAILURE;
    }
}
