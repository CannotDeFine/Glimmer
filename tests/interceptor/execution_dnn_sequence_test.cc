#include "execution_dnn_profile_fixture.h"
#include "runtime.h"
#include "internal/process_identity.h"
#include "glimmer/control/execution_dnn_protocol.h"
#include "../backend/dnn_sequence_fixture.h"

#include <cerrno>
#include <future>
#include <iostream>
#include <string_view>
#include <dlfcn.h>
#include <sys/mman.h>
#include <sys/wait.h>

extern "C" unsigned int glimmer_dnn_sequence_native_calls();
namespace {
using namespace glimmer::app;
using namespace glimmer::control;
using glimmer::core::ExecutionSchedulingMode;
namespace fixture = glimmer::tests::dnn_sequence;
using Clock = std::chrono::steady_clock;
constexpr unsigned int kPoisonCases = 13, kQueryHostileCases = 25, kHostileCases = 53;
void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void check(cudnnStatus_t result) {
    expect(result == CUDNN_STATUS_SUCCESS, "original sequence success");
}
struct Cached {
    decltype(&cudnnCreateRNNDataDescriptor) create_rnn{};
    decltype(&cudnnDestroyRNNDataDescriptor) destroy_rnn{};
    decltype(&cudnnSetRNNDataDescriptor) set_rnn{};
    decltype(&cudnnGetRNNDataDescriptor) get_rnn{};
    decltype(&cudnnCreateSeqDataDescriptor) create_seq{};
    decltype(&cudnnDestroySeqDataDescriptor) destroy_seq{};
    decltype(&cudnnSetSeqDataDescriptor) set_seq{};
    decltype(&cudnnGetSeqDataDescriptor) get_seq{};
};
Cached lookup(const char* provider) {
    auto* library = ::dlopen(provider, RTLD_NOW | RTLD_LOCAL);
    expect(library, "open independent sequence caller provider");
    Cached result;
#define LOOKUP(member, symbol)                                               \
    do {                                                                     \
        auto* selected = ::dlsym(library, #symbol);                          \
        expect(selected == reinterpret_cast<void*>(&symbol) &&               \
                   selected == ::dlsym(RTLD_DEFAULT, #symbol),               \
               "direct/global/explicit provider symbols stay owned");        \
        result.member = reinterpret_cast<decltype(result.member)>(selected); \
    } while (false)
    LOOKUP(create_rnn, cudnnCreateRNNDataDescriptor);
    LOOKUP(destroy_rnn, cudnnDestroyRNNDataDescriptor);
    LOOKUP(set_rnn, cudnnSetRNNDataDescriptor);
    LOOKUP(get_rnn, cudnnGetRNNDataDescriptor);
    LOOKUP(create_seq, cudnnCreateSeqDataDescriptor);
    LOOKUP(destroy_seq, cudnnDestroySeqDataDescriptor);
    LOOKUP(set_seq, cudnnSetSeqDataDescriptor);
    LOOKUP(get_seq, cudnnGetSeqDataDescriptor);
#undef LOOKUP
    expect(::dlclose(library) == 0, "cached controlled symbols survive provider close");
    return result;
}
struct Page {
    std::size_t bytes{};
    void* mapping{};
    Page() {
        const auto size = ::sysconf(_SC_PAGESIZE);
        expect(size > 0, "actual page size");
        bytes = static_cast<std::size_t>(size);
        mapping =
            ::mmap(nullptr, bytes * 2, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        expect(mapping != MAP_FAILED, "create readable page and inaccessible successor");
        expect(::mprotect(static_cast<std::byte*>(mapping) + bytes, bytes, PROT_NONE) == 0,
               "protect next page");
    }
    ~Page() {
        if (::munmap(mapping, bytes * 2) != 0) std::abort();
    }
    void* end(std::size_t width) {
        return static_cast<std::byte*>(mapping) + bytes - width;
    }
};
struct Output {
    cudnnDataType_t type{CUDNN_DATA_DOUBLE};
    int dimensions{-77};
    std::size_t count{SIZE_MAX - 77};
    std::array<int, 6> sizes{91, 91, 91, 91, 91, 91};
    std::array<cudnnSeqDataAxis_t, 6> axes{CUDNN_SEQDATA_VECT_DIM, CUDNN_SEQDATA_VECT_DIM,
                                           CUDNN_SEQDATA_VECT_DIM, CUDNN_SEQDATA_VECT_DIM,
                                           CUDNN_SEQDATA_VECT_DIM, CUDNN_SEQDATA_VECT_DIM};
    std::array<int, 6> lengths{-91, -91, -91, -91, -91, -91};
    std::array<std::byte, 16> padding{};
    Output() {
        padding.fill(std::byte{0xa5});
    }
    bool operator==(const Output&) const = default;
};
cudnnStatus_t query(const Cached& api, cudnnSeqDataDescriptor_t descriptor, Output& output,
                    int dimensions, std::size_t lengths, unsigned int mask) {
    return api.get_seq(
        descriptor, mask & 1U ? &output.type : nullptr, mask & 2U ? &output.dimensions : nullptr,
        dimensions, mask & 4U ? output.sizes.data() + 1 : nullptr,
        mask & 8U ? output.axes.data() + 1 : nullptr, mask & 16U ? &output.count : nullptr, lengths,
        mask & 32U ? output.lengths.data() + 1 : nullptr,
        mask & 64U ? output.padding.data() + 4 : nullptr);
}
void verify(const Output& output, const glimmer::backend::cuda::CudaDnnSeqDataState& input,
            int dimensions, std::size_t lengths, unsigned int mask) {
    Output expected;
    if (mask & 1U) expected.type = static_cast<cudnnDataType_t>(input.type);
    if (mask & 2U) expected.dimensions = 4;
    if (mask & 16U) expected.count = input.sequence_count;
    for (int i = 0; i < std::min(dimensions, 4); ++i) {
        if (mask & 4U) expected.sizes[i + 1] = input.sizes[i];
        if (mask & 8U) expected.axes[i + 1] = static_cast<cudnnSeqDataAxis_t>(input.axes[i]);
    }
    if (mask & 32U)
        std::copy_n(input.lengths.begin(), std::min(lengths, input.lengths.size()),
                    expected.lengths.begin() + 1);
    if (mask & 64U)
        std::copy_n(input.padding.begin(), fixture::padding_width(input.type, false),
                    expected.padding.begin() + 4);
    expect(output == expected, "all scalar, requested-prefix, padding and guard bytes match");
}
cudnnStatus_t set(const Cached& api, cudnnSeqDataDescriptor_t descriptor,
                  glimmer::backend::cuda::CudaDnnSeqDataState input) {
    std::array<cudnnSeqDataAxis_t, 4> axes{};
    for (std::size_t i = 0; i != 4; ++i) axes[i] = static_cast<cudnnSeqDataAxis_t>(input.axes[i]);
    return api.set_seq(descriptor, static_cast<cudnnDataType_t>(input.type), input.dimensions,
                       input.presence & 1U ? input.sizes.data() : nullptr,
                       input.presence & 2U ? axes.data() : nullptr, input.sequence_count,
                       input.presence & 4U ? input.lengths.data() : nullptr,
                       input.presence & 8U ? input.padding.data() : nullptr);
}
void wait_flag(const std::atomic<bool>& flag) {
    const auto deadline = Clock::now() + std::chrono::seconds{5};
    while (!flag.load()) {
        expect(Clock::now() < deadline, "bounded explicitly-started concurrent operation");
        std::this_thread::yield();
    }
}
void pending(const Cached& api, cudnnSeqDataDescriptor_t descriptor) {
    const auto original = fixture::seq();
    auto input = original;
    std::array<cudnnSeqDataAxis_t, 4> axes{CUDNN_SEQDATA_BATCH_DIM, CUDNN_SEQDATA_TIME_DIM,
                                           CUDNN_SEQDATA_BEAM_DIM, CUDNN_SEQDATA_VECT_DIM};
    fixture::control.entered.store(false);
    fixture::control.release.store(false);
    fixture::control.pause_set.store(true);
    auto setter = std::async(std::launch::async, [&] {
        return api.set_seq(descriptor, CUDNN_DATA_FLOAT, 4, input.sizes.data(), axes.data(), 4,
                           input.lengths.data(), input.padding.data());
    });
    wait_flag(fixture::control.entered);
    // Original callback is paused before reading its server-owned argument
    // snapshots. No frontend reads of these caller arrays remain active here.
    input.sizes.fill(99);
    input.lengths.assign(4, 77);
    input.padding.fill(std::byte{0xff});
    axes.fill(CUDNN_SEQDATA_TIME_DIM);
    std::atomic<bool> started{};
    auto reader = std::async(std::launch::async, [&] {
        started.store(true);
        Output output;
        const auto status = query(api, descriptor, output, 4, 4, 127);
        if (status == CUDNN_STATUS_SUCCESS) verify(output, original, 4, 4, 127);
        return status;
    });
    wait_flag(started);
    expect(reader.wait_for(std::chrono::milliseconds{20}) == std::future_status::timeout &&
               setter.wait_for(std::chrono::milliseconds{0}) == std::future_status::timeout,
           "paused original mutation cannot publish an overlapping getter early");
    fixture::control.release.store(true);
    check(setter.get());
    check(reader.get());
    // This public check includes synchronous service serialization; the direct
    // registry lease/reentry tests independently prove the exclusion mechanism.
}
void ordinary(const char* provider) {
    const auto api = lookup(provider);
    cudnnRNNDataDescriptor_t rnn{};
    cudnnSeqDataDescriptor_t seq{};
    check(cudnnCreateRNNDataDescriptor(&rnn));
    check(cudnnCreateSeqDataDescriptor(&seq));
    glimmer::backend::cuda::CudaDnnSeqDataState fresh{.dimensions = 4, .lengths = {}};
    Output initial;
    check(query(api, seq, initial, INT32_MAX, SIZE_MAX, 127));
    verify(initial, fresh, INT32_MAX, SIZE_MAX, 127);
    for (const auto type :
         {CUDNN_DATA_FLOAT, CUDNN_DATA_DOUBLE, CUDNN_DATA_HALF, CUDNN_DATA_BFLOAT16}) {
        const auto input = fixture::seq(type);
        Page padding, lengths, sizes, axes;
        std::memcpy(padding.end(input.padding_bytes), input.padding.data(), input.padding_bytes);
        std::memcpy(lengths.end(16), input.lengths.data(), 16);
        std::memcpy(sizes.end(16), input.sizes.data(), 16);
        std::memcpy(axes.end(16), input.axes.data(), 16);
        check(api.set_seq(seq, type, 4, static_cast<int*>(sizes.end(16)),
                          static_cast<cudnnSeqDataAxis_t*>(axes.end(16)), 4,
                          static_cast<int*>(lengths.end(16)), padding.end(input.padding_bytes)));
        for (unsigned int mask = 0; mask != 128; ++mask) {
            Output output;
            const auto status = query(api, seq, output, 4, SIZE_MAX, mask);
            if (type == CUDNN_DATA_BFLOAT16 && (mask & 64U)) {
                expect(
                    status == CUDNN_STATUS_INTERNAL_ERROR && output == Output{},
                    "BF16 padding getter fails natively without any partial scalar/array output");
            } else {
                check(status);
                verify(output, input, 4, SIZE_MAX, mask);
            }
        }
        if (type != CUDNN_DATA_BFLOAT16) {
            check(api.get_seq(seq, nullptr, nullptr, 0, nullptr, nullptr, nullptr, 0, nullptr,
                              padding.end(input.padding_bytes)));
            expect(!std::memcmp(padding.end(input.padding_bytes), input.padding.data(),
                                input.padding_bytes),
                   "getter writes exactly its 2/4/8-byte actual padding width at page boundary");
            const auto rnn_input = fixture::rnn(type);
            Page rnn_lengths;
            std::memcpy(rnn_lengths.end(12), rnn_input.lengths.data(), 12);
            for (const auto layout :
                 {CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED, CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED,
                  CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED})
                check(api.set_rnn(rnn, type, layout, 5, 3, 7,
                                  static_cast<int*>(rnn_lengths.end(12)),
                                  padding.end(input.padding_bytes)));
        }
    }
    const auto input = fixture::seq();
    check(set(api, seq, input));
    for (const int requested : {0, 1, 3, 4, INT32_MAX})
        for (const auto count : std::array<std::size_t, 6>{0, 1, 3, 4, 5, SIZE_MAX}) {
            Output output;
            check(query(api, seq, output, requested, count, 127));
            verify(output, input, requested, count, 127);
        }
    Output failed;
    fixture::control.get_status.store(CUDNN_STATUS_BAD_PARAM_SHAPE_MISMATCH);
    expect(query(api, seq, failed, 4, 4, 127) == CUDNN_STATUS_BAD_PARAM_SHAPE_MISMATCH &&
               failed == Output{},
           "original getter error touches private outputs only");
    fixture::control.get_status.store(CUDNN_STATUS_SUCCESS);
    expect(
        query(api, seq, failed, -1, SIZE_MAX, 127) == CUDNN_STATUS_BAD_PARAM && failed == Output{},
        "negative raw getter capacity preserves native error and no outputs");
    Page boundary;
    for (unsigned int selected = 0; selected != 4; ++selected) {
        Output output;
        auto* short_span = static_cast<int*>(boundary.end(4));
        *short_span = 12345;
        const auto status =
            api.get_seq(seq, &output.type, &output.dimensions, 4,
                        selected == 0 ? short_span : output.sizes.data() + 1,
                        selected == 1 ? reinterpret_cast<cudnnSeqDataAxis_t*>(short_span)
                                      : output.axes.data() + 1,
                        &output.count, 4, selected == 2 ? short_span : output.lengths.data() + 1,
                        selected == 3 ? boundary.end(2) : output.padding.data() + 4);
        expect(status == CUDNN_STATUS_BAD_PARAM && output == Output{} && *short_span == 12345,
               "every output span is validated before any scalar or array publication");
    }
    const auto callbacks = fixture::control.callbacks.load();
    const auto invalid = static_cast<int*>(boundary.end(0));
    const std::array<int, 4> overflow{5, INT32_MAX, 2, 7};
    expect(api.set_seq(seq, CUDNN_DATA_FLOAT, 4, overflow.data(), nullptr, 0, invalid, invalid) ==
                   CUDNN_STATUS_BAD_PARAM &&
               api.set_seq(seq, CUDNN_DATA_FLOAT, 4, nullptr, nullptr, 4, nullptr, nullptr) ==
                   CUDNN_STATUS_BAD_PARAM &&
               callbacks == fixture::control.callbacks.load(),
           "overflow and native-dereference missing-array policy reject before issue");
    check(query(api, seq, failed, 4, 4, 127));
    verify(failed, input, 4, 4, 127);
    Output guard;
    expect(api.get_rnn(rnn, &guard.type, nullptr, &guard.dimensions, nullptr, nullptr, 0,
                       guard.lengths.data(), guard.padding.data()) == CUDNN_STATUS_NOT_SUPPORTED &&
               guard == Output{},
           "precisely typed pinned RNNData getter remains a no-output guard");
    pending(api, seq);
    const auto child = ::fork();
    expect(child >= 0, "fork inherited sequence frontend");
    if (!child) {
        Output output;
        ::_exit(query(api, seq, output, 4, 4, 127) == CUDNN_STATUS_NOT_INITIALIZED &&
                        output == Output{} && !glimmer_dnn_sequence_native_calls()
                    ? 0
                    : 1);
    }
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && !WEXITSTATUS(status),
           "fork child cannot issue metadata or caller-native SDK");
    expect(api.destroy_rnn(reinterpret_cast<cudnnRNNDataDescriptor_t>(seq)) ==
               CUDNN_STATUS_NOT_INITIALIZED,
           "cross-kind opaque descriptor rejects before destruction");
    check(api.destroy_seq(seq));
    check(api.destroy_rnn(rnn));
    expect(api.destroy_seq(seq) == CUDNN_STATUS_NOT_INITIALIZED &&
               !glimmer_dnn_sequence_native_calls(),
           "stale descriptors never call provider SDK");
}
void poison(const char* provider, unsigned int which) {
    const auto api = lookup(provider);
    cudnnRNNDataDescriptor_t rnn{};
    cudnnSeqDataDescriptor_t seq{};
    check(api.create_rnn(&rnn));
    check(api.create_seq(&seq));
    Page page;
    auto* invalid = static_cast<int*>(page.end(0));
    const auto* invalid_axes = reinterpret_cast<cudnnSeqDataAxis_t*>(invalid);
    cudnnStatus_t result{}, expected = CUDNN_STATUS_BAD_PARAM;
    if (which == 0) {
        fixture::control.set_status.store(CUDNN_STATUS_BAD_PARAM);
        auto input = fixture::rnn();
        result = api.set_rnn(rnn, CUDNN_DATA_FLOAT, CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED, 5, 3,
                             7, input.lengths.data(), input.padding.data());
    } else if (which == 1) {
        fixture::control.set_status.store(CUDNN_STATUS_ALLOC_FAILED);
        expected = CUDNN_STATUS_ALLOC_FAILED;
        result = set(api, seq, fixture::seq());
    } else if (which == 2 || which == 3) {
        expected = which == 2 ? CUDNN_STATUS_NOT_SUPPORTED : CUDNN_STATUS_BAD_PARAM;
        result = api.set_rnn(rnn, CUDNN_DATA_BFLOAT16, CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED, 5,
                             3, 7, which == 2 ? invalid : nullptr, invalid);
    } else if (which == 4 || which == 5 || which == 10) {
        if (which == 10) expected = CUDNN_STATUS_NOT_SUPPORTED;
        result = api.set_rnn(rnn, CUDNN_DATA_FLOAT,
                             static_cast<cudnnRNNDataLayout_t>(which == 4 ? 3 : 0),
                             which == 5    ? 0
                             : which == 10 ? 65536
                                           : 5,
                             3, 7, invalid, invalid);
    } else if (which == 6 || which == 7 || which == 11) {
        expected = which == 7 ? CUDNN_STATUS_NOT_SUPPORTED : CUDNN_STATUS_BAD_PARAM;
        result = api.set_seq(seq, static_cast<cudnnDataType_t>(which == 6 ? 17 : 0),
                             which == 7    ? 3
                             : which == 11 ? 0
                                           : 4,
                             invalid, invalid_axes, SIZE_MAX, invalid, invalid);
    } else if (which == 12) {
        auto input = fixture::seq();
        input.axes = {3, 2, 1, 0};
        result = set(api, seq, input);
    } else {
        auto input = fixture::seq();
        if (which == 8) input.sizes[0] = 0;
        result = api.set_seq(seq, CUDNN_DATA_FLOAT, 4, input.sizes.data(), invalid_axes,
                             which == 9 ? SIZE_MAX : 4, invalid, invalid);
    }
    expect(result == expected,
           "early native scalar errors preserve original precedence without inaccessible reads");
    const auto calls = fixture::control.callbacks.load();
    Output output;
    expect(query(api, seq, output, 4, 4, 127) == CUDNN_STATUS_NOT_INITIALIZED &&
               api.destroy_rnn(rnn) == CUDNN_STATUS_NOT_INITIALIZED && output == Output{} &&
               calls == fixture::control.callbacks.load() && !glimmer_dnn_sequence_native_calls(),
           "all issued setter failures quarantine without replay, cleanup or caller fallback");
}
void endpoint(const std::string& path, TaskPeerIdentity identity) {
    const auto pid = std::to_string(identity.pid),
               start = std::to_string(identity.start_time_ticks);
    expect(::setenv("GLIMMER_EXECUTION_SOCKET", path.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_PID", pid.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_START", start.c_str(), 1) == 0,
           "configure authenticated DNM1 endpoint");
}
void run(const char* provider, ExecutionSchedulingMode mode, int failure = -1) {
    const auto identity = current_process_identity();
    expect(identity.has_value(), "actual service peer identity");
    ExecutionServiceConfig config;
    config.transport.socket_path = "/tmp/glimmer-sequence-" + std::to_string(::getpid()) + ".sock";
    config.scheduling_mode = mode;
    config.uses_original_kernels = true;
    config.exit_when_idle = true;
    if (mode == ExecutionSchedulingMode::kPriority) config.client_priorities = {{*identity, 7}};
    endpoint(config.transport.socket_path, *identity);
    fixture::reset();
    std::future<void> client;
    ExecutionService service(std::move(config), fixture::session_dispatch());
    expect(service.start(), "start genuine metadata-only service");
    client = std::async(std::launch::async, [=] {
        if (failure >= 0)
            poison(provider, static_cast<unsigned int>(failure));
        else
            ordinary(provider);
    });
    const auto deadline = Clock::now() + std::chrono::seconds{30};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready)
        expect(service.step(1) && Clock::now() < deadline, "bounded metadata-only progress");
    client.get();
    expect(fixture::foundation::is_resource_free() && !glimmer_dnn_sequence_native_calls(),
           "no Driver, context, compute or caller-native invocation in metadata pipeline");
    if (failure >= 0) {
        expect(!fixture::state.destroys && fixture::control.callbacks.load() == 1,
               "isolated issued error retains both native descriptors without retry");
        ::_exit(0);
    }
    expect(service.request_stop() && service.finished() && fixture::state.creates == 2 &&
               fixture::state.destroys == 2,
           "all originals retire exactly once");
}

// Hostile receipt tests are defined below; the peer still uses the real authenticated
// transport and canonical bootstrap, but never invokes a native provider.
struct ReplyCase {
    ExecutionWireReply reply;
    ExecutionDnnSeqDataQuery query{4, 4, 127};
    bool setter{}, clean{};
    cudnnStatus_t status{CUDNN_STATUS_INTERNAL_ERROR};
};
ExecutionDnnSeqDataState wire_state() {
    const auto original = fixture::seq();
    ExecutionDnnSeqDataState value;
    value.type = original.type;
    value.dimensions = original.dimensions;
    value.presence = original.presence;
    value.sequence_count = original.sequence_count;
    value.sizes = original.sizes;
    value.axes = original.axes;
    value.lengths = original.lengths;
    value.padding_bytes = original.padding_bytes;
    value.padding = original.padding;
    return value;
}
ReplyCase hostile(unsigned int index) {
    ReplyCase test;
    auto& reply = test.reply;
    reply.disposition = ExecutionWireDisposition::kComplete;
    reply.original_status = CUDNN_STATUS_SUCCESS;
    auto value = wire_state();
    test.setter = index >= 19;
    if (index == 0) reply.parameter_bytes = 1;
    if (index == 1) reply.id = 17;
    if (index == 2) reply.address = 17;
    if (index == 3) {
        reply.parameter_bytes = 1;
        reply.arguments = {{0, 1}};
    }
    if (index == 4) reply.original_status.reset();
    if (index == 5) reply.original_status = CUDNN_STATUS_BAD_PARAM;
    if (index == 6) value.dimensions = 3;
    if (index == 7) value.type = 32;
    if (index == 8) value.sequence_count = kExecutionDnnMaxSequenceLengths + 1;
    if (index == 9) value.lengths.pop_back();
    if (index == 10) value.lengths.push_back(1);
    if (index == 11) value.presence = 7;
    if (index == 12) value.axes[0] = 4;
    if (index == 13) value.padding_bytes = 8;
    if (index == 14) {
        test.query = {0, 0, 0};
        value.presence = 0;
        value.sizes = {};
        value.axes = {};
        value.lengths.clear();
        value.padding_bytes = 0;
        value.padding = {};
        reply.parameter_bytes = 1;
    }
    if (index == 15 || index == 16 || index == 22 || index == 23) {
        reply.status = CUDNN_STATUS_BAD_PARAM_SHAPE_MISMATCH;
        reply.original_status = reply.status;
        test.status = CUDNN_STATUS_BAD_PARAM_SHAPE_MISMATCH;
        test.clean = index == 15;
        if (index == 16) {
            reply.address = 17;
            test.status = CUDNN_STATUS_INTERNAL_ERROR;
        }
        if (index == 22) reply.parameter_bytes = 1;
        if (index == 23) test.status = CUDNN_STATUS_INTERNAL_ERROR;
    }
    if (index == 17 || index == 24) {
        test.status = CUDNN_STATUS_SUCCESS;
        test.clean = true;
    }
    if (index == 18) {
        reply.disposition = ExecutionWireDisposition::kRejected;
        reply.status = CUDA_ERROR_NOT_SUPPORTED;
        reply.original_status.reset();
        test.status = CUDNN_STATUS_NOT_SUPPORTED;
        test.clean = true;
    }
    if (index == 19) reply.parameter_bytes = 1;
    if (index == 20) reply.data = {std::byte{7}};
    if (index == 21) reply.id = 17;
    if (!test.setter && index != 15 && index != 16 && index != 18) {
        // Write raw fields so intentionally malformed states reach the frontend,
        // rather than asking a semantic state encoder to bless them.
        ExecutionPayloadWriter output;
        expect(output.write_u32(value.type) &&
                   output.write_u32(static_cast<std::uint32_t>(value.dimensions)) &&
                   output.write_u32(value.presence) && output.write_u64(value.sequence_count),
               "encode hostile fixed sequence prefix");
        for (const auto size : value.sizes)
            expect(output.write_u32(static_cast<std::uint32_t>(size)), "encode size");
        for (const auto axis : value.axes) expect(output.write_u32(axis), "encode axis");
        ExecutionPayloadWriter lengths;
        for (const auto length : value.lengths)
            expect(lengths.write_u32(static_cast<std::uint32_t>(length)), "encode length");
        expect(output.write_blob(lengths.take()) && output.write_blob(std::span<const std::byte>(
                                                        value.padding.data(), value.padding_bytes)),
               "encode hostile bounded variable arrays");
        reply.data = output.take();
    }
    return test;
}
void lifecycle_case(const char* provider, unsigned int index) {
    const unsigned int variant = index % 14;
    const bool rnn_kind = index >= 14, destroying = variant >= 7, clean = variant == 6;
    const bool native_error = variant == 4 || variant == 6 || variant == 11;
    ExecutionWireReply injected;
    injected.disposition = ExecutionWireDisposition::kComplete;
    injected.original_status = CUDNN_STATUS_SUCCESS;
    if (!destroying) injected.id = 99;
    if (variant == 0 || variant == 7) injected.parameter_bytes = 1;
    if (variant == 1 || variant == 8) injected.address = 17;
    if (variant == 2 || variant == 9) injected.data = {std::byte{7}};
    if (variant == 3 || variant == 10) {
        injected.parameter_bytes = 1;
        injected.arguments = {{0, 1}};
    }
    if (variant == 5) injected.id = 0;
    if (variant == 12) injected.id = 99;
    if (native_error || variant == 13) {
        injected.id = 0;
        injected.status = CUDNN_STATUS_ALLOC_FAILED;
        injected.original_status = injected.status;
        injected.parameter_bytes = variant == 6 || variant == 13 ? 0 : 1;
    }
    const auto identity = current_process_identity();
    expect(identity.has_value(), "actual hostile lifecycle peer identity");
    ExecutionSocketConfig config;
    config.socket_path = "/tmp/glimmer-sequence-life-" + std::to_string(::getpid()) + ".sock";
    endpoint(config.socket_path, *identity);
    std::future<void> client;
    ExecutionSocketServer server(std::move(config));
    expect(server.start(), "start authenticated hostile lifecycle peer");
    client = std::async(std::launch::async, [=] {
        const auto api = lookup(provider);
        cudnnSeqDataDescriptor_t seed{};
        cudnnRNNDataDescriptor_t rnn{};
        check(api.create_seq(&seed));
        if (destroying && rnn_kind) check(api.create_rnn(&rnn));
        cudnnStatus_t status{};
        if (destroying)
            status = rnn_kind ? api.destroy_rnn(rnn) : api.destroy_seq(seed);
        else if (rnn_kind) {
            auto* value = reinterpret_cast<cudnnRNNDataDescriptor_t>(0xfee);
            status = api.create_rnn(&value);
            expect(value == reinterpret_cast<cudnnRNNDataDescriptor_t>(0xfee),
                   "malformed or failed RNNData create never publishes a token");
        } else {
            auto* value = reinterpret_cast<cudnnSeqDataDescriptor_t>(0xfee);
            status = api.create_seq(&value);
            expect(value == reinterpret_cast<cudnnSeqDataDescriptor_t>(0xfee),
                   "malformed or failed SeqData create never publishes a token");
        }
        expect(status == (native_error ? CUDNN_STATUS_ALLOC_FAILED : CUDNN_STATUS_INTERNAL_ERROR),
               "lifecycle malformed receipt and exact native-error control remain distinct");
        Output output;
        const auto follow = query(api, seed, output, 4, 4, 127);
        if (clean) {
            check(follow);
            verify(output, fixture::seq(), 4, 4, 127);
        } else {
            expect(follow == CUDNN_STATUS_NOT_INITIALIZED && output == Output{},
                   "lifecycle uncertainty disables every existing metadata token locally");
            cudnnSeqDataDescriptor_t new_output{};
            expect(api.create_seq(&new_output) == CUDNN_STATUS_INTERNAL_ERROR && !new_output,
                   "quarantined create cannot open a replacement owner");
        }
        expect(!glimmer_dnn_sequence_native_calls(), "lifecycle containment never calls provider");
    });
    unsigned int hellos{}, requests{};
    const unsigned int setup = destroying && rnn_kind ? 2 : 1;
    glimmer::tests::ExecutionDnnProfileFixture profiles;
    const auto deadline = Clock::now() + std::chrono::seconds{10};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
        expect(Clock::now() < deadline && server.poll_for(1), "bounded hostile lifecycle progress");
        if (auto request = server.take_request()) {
            expect(request->peer == *identity, "authenticate lifecycle peer");
            if (profiles.respond(server, *request)) continue;
            ExecutionWireReply reply;
            reply.disposition = ExecutionWireDisposition::kComplete;
            ExecutionPayloadReader input(request->message.payload);
            if (request->message.header.operation == ExecutionOpcode::kHello) {
                std::uint32_t role{}, flags{};
                expect(input.read_u32(role) && role == kExecutionDnnMetadataRole &&
                           input.read_u32(flags) && !flags && input.empty(),
                       "canonical DNM1 Hello");
                ++hellos;
                const auto ownership = encode_execution_bootstrap_ownership(
                    ExecutionBootstrapOwnership::kMetadataRetained);
                expect(ownership.has_value(), "encode lifecycle ownership bootstrap");
                reply.data.assign(ownership->begin(), ownership->end());
            } else {
                ++requests;
                if (requests <= setup || (requests == setup + 1 && !destroying)) {
                    std::uint32_t kind{};
                    const bool expected_rnn = requests == 2 && setup == 2 ? true
                                              : requests == setup + 1     ? rnn_kind
                                                                          : false;
                    expect(
                        request->message.header.operation == ExecutionOpcode::kCreateDnnMetadata &&
                            input.read_u32(kind) && input.empty() &&
                            kind == static_cast<unsigned int>(
                                        expected_rnn ? ExecutionDnnObjectKind::kRnnData
                                                     : ExecutionDnnObjectKind::kSeqData),
                        "new lifecycle sends exact original descriptor kind");
                } else if (requests == setup + 1) {
                    std::uint64_t id{};
                    expect(
                        request->message.header.operation == ExecutionOpcode::kDestroyDnnMetadata &&
                            input.read_u64(id) && id == (rnn_kind ? 18U : 17U) && input.empty(),
                        "destroy uses existing opaque owner identity");
                } else {
                    std::uint64_t id{};
                    ExecutionDnnSeqDataQuery value;
                    expect(
                        clean && requests == setup + 2 &&
                            request->message.header.operation == ExecutionOpcode::kGetDnnSeqData &&
                            input.read_u64(id) && id == 17 &&
                            read_execution_dnn_seq_data_query(input, value) && input.empty(),
                        "only known-null native creation failure permits a follow-up RPC");
                }
                if (requests <= setup) {
                    reply.original_status = CUDNN_STATUS_SUCCESS;
                    reply.id = requests + 16;
                } else if (requests == setup + 1)
                    reply = injected;
                else
                    reply = hostile(17).reply;
            }
            const auto encoded = encode_execution_reply(reply);
            expect(encoded && decode_execution_reply(*encoded),
                   "hostile lifecycle envelope is wire-valid");
            auto header = request->message.header;
            header.kind = ExecutionMessageKind::kReply;
            header.session = request->connection;
            header.incarnation = {71, 93};
            expect(server.reply(request->connection, {header, *encoded}),
                   "deliver lifecycle receipt");
        }
    }
    client.get();
    profiles.verify();
    expect(hellos == 1 && requests == setup + (clean ? 2U : 1U),
           "bad lifecycle receipts never invoke hidden cleanup or reinitialization");
    server.stop();
}
void hostile_case(const char* provider, unsigned int index) {
    if (index >= kQueryHostileCases) {
        lifecycle_case(provider, index - kQueryHostileCases);
        return;
    }
    const auto identity = current_process_identity();
    expect(identity.has_value(), "hostile endpoint identity");
    ExecutionSocketConfig config;
    config.socket_path = "/tmp/glimmer-sequence-hostile-" + std::to_string(::getpid()) + ".sock";
    endpoint(config.socket_path, *identity);
    const auto test = hostile(index);
    std::future<void> client;
    ExecutionSocketServer server(std::move(config));
    expect(server.start(), "start authenticated hostile sequence peer");
    client = std::async(std::launch::async, [=] {
        const auto api = lookup(provider);
        cudnnSeqDataDescriptor_t descriptor{};
        check(api.create_seq(&descriptor));
        Output output;
        const auto status = test.setter
                                ? set(api, descriptor, fixture::seq())
                                : query(api, descriptor, output, test.query.dimensions_requested,
                                        test.query.lengths_requested, test.query.outputs);
        expect(status == test.status, "hostile envelope produces exact contained result");
        if (index == 17)
            verify(output, fixture::seq(), 4, 4, 127);
        else
            expect(output == Output{}, "no failed or uncertain receipt publishes any output");
        Output next;
        const auto follow = query(api, descriptor, next, 4, 4, 127);
        if (test.clean) {
            check(follow);
            verify(next, fixture::seq(), 4, 4, 127);
        } else {
            expect(follow == CUDNN_STATUS_NOT_INITIALIZED && next == Output{},
                   "quarantine blocks subsequent lookup without another RPC");
        }
        expect(!glimmer_dnn_sequence_native_calls(),
               "hostile transport never falls through caller SDK");
    });
    unsigned int hellos{}, creates{}, queries{};
    glimmer::tests::ExecutionDnnProfileFixture profiles;
    const auto deadline = Clock::now() + std::chrono::seconds{10};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
        expect(Clock::now() < deadline && server.poll_for(1), "bounded hostile sequence progress");
        if (auto request = server.take_request()) {
            expect(request->peer == *identity, "hostile fixture still verifies peer identity");
            if (profiles.respond(server, *request)) continue;
            ExecutionWireReply reply;
            reply.disposition = ExecutionWireDisposition::kComplete;
            ExecutionPayloadReader input(request->message.payload);
            if (request->message.header.operation == ExecutionOpcode::kHello) {
                std::uint32_t role{}, flags{};
                expect(input.read_u32(role) && role == kExecutionDnnMetadataRole &&
                           input.read_u32(flags) && !flags && input.empty(),
                       "sequence metadata opens only canonical DNM1 bootstrap");
                ++hellos;
                const auto ownership = encode_execution_bootstrap_ownership(
                    ExecutionBootstrapOwnership::kMetadataRetained);
                expect(ownership.has_value(), "encode retained metadata bootstrap");
                reply.data.assign(ownership->begin(), ownership->end());
            } else if (request->message.header.operation == ExecutionOpcode::kCreateDnnMetadata) {
                std::uint32_t kind{};
                expect(input.read_u32(kind) &&
                           kind == static_cast<unsigned int>(ExecutionDnnObjectKind::kSeqData) &&
                           input.empty(),
                       "create exact original SeqData kind");
                ++creates;
                reply.original_status = CUDNN_STATUS_SUCCESS;
                reply.id = 17;
            } else {
                std::uint64_t descriptor{};
                expect(input.read_u64(descriptor) && descriptor == 17,
                       "opaque descriptor id is exact");
                ++queries;
                if (test.setter && queries == 1) {
                    ExecutionDnnSeqDataState value;
                    expect(request->message.header.operation == ExecutionOpcode::kSetDnnSeqData &&
                               read_execution_dnn_seq_data(input, value) && input.empty() &&
                               value.lengths == fixture::seq().lengths,
                           "full original setter argument payload reaches peer");
                } else {
                    ExecutionDnnSeqDataQuery query;
                    expect(request->message.header.operation == ExecutionOpcode::kGetDnnSeqData &&
                               read_execution_dnn_seq_data_query(input, query) && input.empty(),
                           "exact getter request reaches peer");
                }
                if (queries == 1)
                    reply = test.reply;
                else {
                    expect(test.clean && queries == 2,
                           "only healthy receipt permits subsequent RPC");
                    reply = hostile(17).reply;
                }
            }
            const auto encoded = encode_execution_reply(reply);
            expect(encoded && decode_execution_reply(*encoded),
                   "encode intended hostile but transport-valid envelope");
            auto header = request->message.header;
            header.kind = ExecutionMessageKind::kReply;
            header.session = request->connection;
            header.incarnation = {71, 93};
            expect(server.reply(request->connection, {header, *encoded}),
                   "deliver hostile sequence reply");
        }
    }
    client.get();
    profiles.verify();
    expect(hellos == 1 && creates == 1 && queries == (test.clean ? 2U : 1U),
           "bad receipts never reopen metadata or invoke recovery queries");
    server.stop();
}
unsigned int number(const char* text) {
    const std::string_view value{text};
    expect(!value.empty() && value.size() <= 2, "bounded child index");
    unsigned int result{};
    for (const char digit : value) {
        expect(digit >= '0' && digit <= '9', "decimal child index");
        result = result * 10 + static_cast<unsigned int>(digit - '0');
    }
    return result;
}
void children(const char* executable, const char* provider, bool malformed) {
    for (unsigned int index = 0; index != (malformed ? kHostileCases : kPoisonCases); ++index) {
        const auto value = std::to_string(index);
        const auto child = ::fork();
        expect(child >= 0, "fork isolated sequence process");
        if (!child) {
            ::execl("/proc/self/exe", executable, provider,
                    malformed ? "--malformed-case" : "--poison-case", value.c_str(),
                    static_cast<char*>(nullptr));
            ::_exit(127);
        }
        int status{};
        pid_t waited{};
        do {
            waited = ::waitpid(child, &status, 0);
        } while (waited < 0 && errno == EINTR);
        if (waited != child || !WIFEXITED(status) || WEXITSTATUS(status))
            throw std::runtime_error("isolated sequence case failed: " + value);
    }
}
}  // namespace
int main(int argc, char** argv) {
    try {
        expect(argc >= 2 && argc <= 4, "provider and optional sequence fixture mode");
        const std::string_view mode = argc >= 3 ? argv[2] : "";
        if (mode == "--malformed-replies" || mode == "--poison") {
            expect(argc == 3, "isolated aggregate mode takes no case argument");
            children(argv[0], argv[1], mode == "--malformed-replies");
        } else if (mode == "--malformed-case" || mode == "--poison-case") {
            expect(argc == 4, "isolated mode requires case index");
            const auto index = number(argv[3]);
            if (mode == "--malformed-case") {
                expect(index < kHostileCases, "known hostile case");
                hostile_case(argv[1], index);
                ::_exit(0);
            }
            expect(index < kPoisonCases, "known native failure case");
            run(argv[1], ExecutionSchedulingMode::kNaive, static_cast<int>(index));
        } else {
            expect(argc <= 3 && (mode.empty() || mode == "--profile" || mode == "--priority"),
                   "known ordinary scheduling mode");
            run(argv[1], mode == "--profile"    ? ExecutionSchedulingMode::kProfile
                         : mode == "--priority" ? ExecutionSchedulingMode::kPriority
                                                : ExecutionSchedulingMode::kNaive);
        }
        std::cout << "Sequence original metadata CPU/local-IPC checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
