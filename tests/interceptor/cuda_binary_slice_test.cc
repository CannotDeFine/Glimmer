#include "internal/cuda_binary_slice.h"

#include <cstring>
#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <cerrno>
#include <cstdio>
#include <unistd.h>

namespace {
using namespace glimmer::interceptor;
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
struct Input {
    int count{2049};
    float alpha{0.5F};
    std::array<std::uint64_t, 3> pointers{0x10000, 0x20000, 0x30000};
    std::array<void*, 3> addresses{&count, &alpha, pointers.data()};
    CUfunction function{reinterpret_cast<CUfunction>(0x4000)};
    std::shared_ptr<void> resources{std::make_shared<int>(1)};
    QualifiedAddMapping mapping{QualifiedAddMapping::kSingleCta};
    std::optional<CudaQueuedKernel> prepare(std::array<unsigned int, 3> grid = {3, 1, 1},
                                            std::array<unsigned int, 3> block = {128, 1, 1}) {
        return prepare_mapped_add_slices(function, grid, block, addresses.data(), resources, false,
                                         mapping);
    }
};
std::uint64_t index(const OwnedKernelArguments& arguments) {
    std::uint64_t result{};
    std::memcpy(&result, static_cast<const std::byte*>(arguments.addresses()[2]) + 24, 8);
    return result;
}
void range_preparation_and_state() {
    for (int count : {32769, 32770, 1048576, 1048577, 4194304, 4194305}) {
        Input input;
        input.count = count;
        input.pointers = {0x10000000, 0x20000000, 0x30000000};
        const auto blocks = (static_cast<unsigned int>(count) + 1023) / 1024;
        auto prepared = prepare_mapped_add_slices(input.function, {blocks, 1, 1}, {128, 1, 1},
                                                  input.addresses.data(), input.resources, false,
                                                  QualifiedAddMapping::kRange);
        require(prepared.has_value(), "qualified range preparation");
        auto progress =
            std::dynamic_pointer_cast<CudaBinarySliceContinuation>(prepared->continuation);
        input.count = 0;
        input.pointers.fill(0);
        input.addresses.fill(nullptr);
        const std::uint64_t width = count <= 32769 ? 1 : 1024;
        std::uint64_t base{}, launches{};
        while (base < blocks) {
            const auto physical = std::min(width, blocks - base);
            bool complete = true;
            require(
                progress->prepare_launch(prepared->grid, prepared->arguments) == CUDA_SUCCESS &&
                    prepared->grid ==
                        std::array<unsigned int, 3>{static_cast<unsigned int>(physical), 1, 1} &&
                    index(prepared->arguments) == base &&
                    progress->progress().completed_blocks == base &&
                    progress->prepare_launch(prepared->grid, prepared->arguments) ==
                        CUDA_ERROR_INVALID_VALUE &&
                    progress->request_retreat() == CUDA_SUCCESS &&
                    progress->inspect_stopped(complete) == CUDA_ERROR_INVALID_VALUE,
                "prepared range owns its base but does not prove completion");
            require(progress->stage_completion(nullptr) == CUDA_SUCCESS &&
                        progress->stage_completion(nullptr) == CUDA_ERROR_INVALID_VALUE &&
                        progress->inspect_stopped(complete) == CUDA_SUCCESS &&
                        complete == (base + physical == blocks),
                    "only staged proof advances range");
            base += physical;
            ++launches;
            require(progress->progress().completed_blocks == base &&
                        progress->progress().physical_launches == launches &&
                        progress->inspect_stopped(complete) == CUDA_ERROR_INVALID_VALUE,
                    "range completion advances exactly once including final partial range");
        }
        require(progress->prepare_launch(prepared->grid, prepared->arguments) ==
                    CUDA_ERROR_INVALID_VALUE,
                "completed range cannot replay");
    }
    Input input;
    input.pointers = {0x10000000, 0x20000000, 0x30000000};
    for (auto mapping : {QualifiedAddMapping::kRange, static_cast<QualifiedAddMapping>(255)}) {
        input.count = mapping == QualifiedAddMapping::kRange ? 4194306 : 2049;
        const auto blocks = (static_cast<unsigned int>(input.count) + 1023) / 1024;
        require(!prepare_mapped_add_slices(input.function, {blocks, 1, 1}, {128, 1, 1},
                                           input.addresses.data(), input.resources, false, mapping),
                "unknown mapping or unqualified large count rejects");
    }
}
void preparation_and_state() {
    Input input;
    auto prepared = input.prepare();
    require(prepared.has_value(), "qualified launch preparation");
    auto progress = std::dynamic_pointer_cast<CudaBinarySliceContinuation>(prepared->continuation);
    require(progress != nullptr, "typed continuation");
    auto kernel = std::move(*prepared);
    prepared.reset();
    input.count = 0;
    input.alpha = 0;
    input.pointers.fill(0);
    input.addresses.fill(nullptr);
    int count{};
    std::memcpy(&count, kernel.arguments.addresses()[0], sizeof(count));
    require(count == 2049 && progress->progress().completed_blocks == 0, "owned original count");
    std::array<unsigned int, 3> grid{};
    bool complete = true;
    require(progress->inspect_stopped(complete) == CUDA_ERROR_INVALID_VALUE && !complete &&
                progress->stage_completion(nullptr) == CUDA_ERROR_INVALID_VALUE,
            "unissued completion cannot advance");
    auto empty = OwnedKernelArguments::copy({}, nullptr);
    require(progress->prepare_launch(grid, *empty) == CUDA_ERROR_INVALID_VALUE &&
                progress->progress().physical_launches == 0,
            "private slot rejection cannot prepare or advance");
    for (std::uint64_t block = 0; block != 3; ++block) {
        require(progress->prepare_launch(grid, kernel.arguments) == CUDA_SUCCESS &&
                    grid == std::array<unsigned int, 3>{1, 1, 1} &&
                    index(kernel.arguments) == block,
                "one physical block receives retained logical index");
        require(progress->prepare_launch(grid, kernel.arguments) == CUDA_ERROR_INVALID_VALUE &&
                    progress->request_retreat() == CUDA_SUCCESS &&
                    progress->progress().completed_blocks == block,
                "duplicate preparation and retreat cannot advance");
        require(progress->inspect_stopped(complete) == CUDA_ERROR_INVALID_VALUE &&
                    progress->stage_completion(nullptr) == CUDA_SUCCESS &&
                    progress->stage_completion(nullptr) == CUDA_ERROR_INVALID_VALUE &&
                    progress->inspect_stopped(complete) == CUDA_SUCCESS &&
                    complete == (block == 2) &&
                    progress->inspect_stopped(complete) == CUDA_ERROR_INVALID_VALUE,
                "only one staged physical proof advances each slice");
    }
    require(progress->progress().completed_blocks == 3 &&
                progress->progress().physical_launches == 3 &&
                progress->prepare_launch(grid, kernel.arguments) == CUDA_ERROR_INVALID_VALUE,
            "complete continuation cannot replay");
}
void diagnostic_completion_is_opt_in() {
    // This fixture has no background threads. Restore stderr on every exit.
    struct Capture {
        FILE* file{std::tmpfile()};
        int saved{::dup(STDERR_FILENO)};
        ~Capture() {
            if (saved >= 0 && (::dup2(saved, STDERR_FILENO) != STDERR_FILENO || ::close(saved)))
                std::_Exit(EXIT_FAILURE);
            if (file && std::fclose(file)) std::_Exit(EXIT_FAILURE);
        }
    } capture;
    require(capture.file && capture.saved >= 0 &&
                ::dup2(::fileno(capture.file), STDERR_FILENO) == STDERR_FILENO,
            "capture private slice diagnostic");
    for (bool should_trace : {false, true}) {
        Input input;
        input.count = 1024;
        auto prepared =
            prepare_mapped_add_slices(input.function, {1, 1, 1}, {128, 1, 1},
                                      input.addresses.data(), input.resources, should_trace);
        require(prepared.has_value(), "prepare diagnostic continuation");
        auto& continuation = *prepared->continuation;
        bool complete{};
        require(
            continuation.inspect_stopped(complete) == CUDA_ERROR_INVALID_VALUE &&
                continuation.prepare_launch(prepared->grid, prepared->arguments) == CUDA_SUCCESS &&
                continuation.stage_completion(nullptr) == CUDA_SUCCESS,
            "prepare and stage diagnostic slice");
        require(::lseek(::fileno(capture.file), 0, SEEK_CUR) == 0,
                "default-off and unproved slices emit nothing");
        errno = EDOM;
        require(continuation.inspect_stopped(complete) == CUDA_SUCCESS && complete &&
                    errno == EDOM &&
                    continuation.inspect_stopped(complete) == CUDA_ERROR_INVALID_VALUE,
                "physical inspection preserves errno and rejects duplicate completion");
    }
    require(std::fseek(capture.file, 0, SEEK_SET) == 0, "rewind slice diagnostic");
    std::array<char, 160> text{};
    const auto bytes = std::fread(text.data(), 1, text.size(), capture.file);
    require(!std::ferror(capture.file) &&
                std::string_view(text.data(), bytes) ==
                    "[glimmer] binary_slice blocks=1 completed=1 launches=1\n",
            "only enabled exactly-once physical inspection emits a diagnostic");
}
void rejection() {
    for (int count : {-1, 0, 32770}) {
        Input input;
        input.count = count;
        require(!input.prepare(), "unsupported count");
    }
    Input input;
    require(!input.prepare({2, 1, 1}) && !input.prepare({3, 2, 1}) &&
                !input.prepare({3, 1, 1}, {256, 1, 1}),
            "unsupported launch shape");
    input.pointers[1] += 4;
    require(!input.prepare(), "misaligned pointer");
    input.pointers[1] = 0;
    require(!input.prepare(), "null pointer");
    input.pointers[1] = 0x20000;
    for (const auto pointer : {std::uint64_t{0x10010}, std::uint64_t{0xfff0},
                               std::numeric_limits<std::uint64_t>::max() - 15}) {
        input.pointers[1] = pointer;
        require(!input.prepare(), "partial output/input overlap or wrapped range");
    }
    input.count = 2048;
    input.pointers[1] = 0x12000;
    require(input.prepare({2, 1, 1}).has_value(), "adjacent disjoint ranges remain valid");
    input.count = 2049;
    input.pointers[1] = 0x20000;
    input.addresses[1] = nullptr;
    require(!input.prepare(), "missing alpha storage");
    input.addresses[1] = &input.alpha;
    input.function = nullptr;
    require(!input.prepare(), "missing qualified function");
    input.function = reinterpret_cast<CUfunction>(0x4000);
    input.resources.reset();
    require(!input.prepare(), "missing lifetime owner");
}
struct Event {
    bool ready{};
};
struct Fake {
    static inline Fake* active{};
    CUcontext context{reinterpret_cast<CUcontext>(0x1000)};
    std::vector<std::unique_ptr<Event>> events;
    std::vector<std::uint64_t> issued;
    std::vector<unsigned int> widths;
    unsigned int maximum_width{1};
    CUresult launch_error{CUDA_SUCCESS};
    CUresult record_error{CUDA_SUCCESS};
    int drains{};
    Fake() {
        active = this;
    }
    ~Fake() {
        active = nullptr;
    }
    static CUresult current(CUcontext* output) {
        *output = active->context;
        return CUDA_SUCCESS;
    }
    static CUresult create(CUevent* output, unsigned int) {
        auto event = std::make_unique<Event>();
        *output = reinterpret_cast<CUevent>(event.get());
        active->events.push_back(std::move(event));
        return CUDA_SUCCESS;
    }
    static CUresult record(CUevent, CUstream) {
        return active->record_error;
    }
    static CUresult query(CUevent event) {
        return reinterpret_cast<Event*>(event)->ready ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
    }
    static CUresult destroy(CUevent) {
        return CUDA_SUCCESS;
    }
    static CUresult drain(CUstream) {
        ++active->drains;
        return CUDA_SUCCESS;
    }
    static CUresult launch(CUfunction, unsigned int gx, unsigned int gy, unsigned int gz,
                           unsigned int bx, unsigned int by, unsigned int bz, unsigned int shared,
                           CUstream, void** arguments, void**) {
        require(gx >= 1 && gx <= active->maximum_width && gy == 1 && gz == 1 && bx == 128 &&
                    by == 1 && bz == 1 && shared == 0,
                "native call must use mapped shape");
        std::uint64_t block{};
        std::memcpy(&block, static_cast<const std::byte*>(arguments[2]) + 24, 8);
        active->issued.push_back(block);
        active->widths.push_back(gx);
        return active->launch_error;
    }
    static CudaQueueDispatch dispatch() {
        CudaQueueDispatch result{};
        result.context_get_current = current;
        result.launch_kernel = launch;
        result.event_create = create;
        result.event_record = record;
        result.event_query = query;
        result.event_destroy = destroy;
        result.stream_synchronize = drain;
        return result;
    }
    void ready() {
        for (auto& event : events) event->ready = true;
    }
};
void queue_continuation(bool cancel, bool ranges = false) {
    Fake fake;
    Input input;
    if (ranges) {
        input.count = 4194305;
        input.pointers = {0x10000000, 0x20000000, 0x30000000};
        input.mapping = QualifiedAddMapping::kRange;
        fake.maximum_width = 1024;
    }
    const std::uint64_t blocks = ranges ? 4097 : 3;
    auto prepared = input.prepare({static_cast<unsigned int>(blocks), 1, 1});
    auto progress = std::dynamic_pointer_cast<CudaBinarySliceContinuation>(prepared->continuation);
    std::weak_ptr<void> lifetime = input.resources;
    CudaOperationQueue queue(fake.context, Fake::dispatch());
    auto stream = queue.bind_stream(reinterpret_cast<CUstream>(0x8000),
                                    OrderedStreamKind::kNonBlocking, std::make_shared<int>(1));
    auto work = queue.submit(stream, std::move(*prepared));
    auto dependent = queue.submit(stream, CudaQueuedMarker{});
    prepared.reset();
    input.resources.reset();
    require(work && dependent && !lifetime.expired(), "queue retains payload independently");
    require(queue.pump_admitted({}) == CUDA_SUCCESS && fake.issued.empty(), "no grant no launch");
    const std::array admitted{work};
    for (std::uint64_t base = 0, slice = 0; base < blocks; ++slice) {
        const auto physical = std::min<std::uint64_t>(fake.maximum_width, blocks - base);
        require(queue.pump_admitted(admitted) == CUDA_SUCCESS && fake.issued.back() == base &&
                    fake.widths.back() == physical,
                "admitted slice uses owned index");
        require(queue.pump_admitted({}) == CUDA_SUCCESS && !queue.cancel_stopped(work) &&
                    progress->progress().completed_blocks == base &&
                    queue.request_retreat(work) == CUDA_SUCCESS,
                "pending event is not stop proof");
        fake.ready();
        require(queue.pump_admitted({}) == CUDA_SUCCESS &&
                    progress->progress().completed_blocks == base + physical,
                "actual trailing proof advances once");
        base += physical;
        if (base != blocks) {
            require(queue.state(work) == OrderedOperationState::kSuspended &&
                        queue.query(dependent) == CUDA_ERROR_NOT_READY &&
                        queue.pump_admitted({}) == CUDA_SUCCESS && fake.issued.size() == slice + 1,
                    "suspended slices hold dependencies and never auto-issue");
            if (cancel) {
                require(queue.cancel_stopped(work) && queue.shutdown() == CUDA_SUCCESS,
                        "cancel proven stop");
                progress.reset();
                require(lifetime.expired(), "cancelled ownership releases after drain");
                return;
            }
            require(queue.resume(work) && !queue.resume(work), "explicit resume once");
        }
    }
    fake.ready();
    require(queue.pump_admitted({}) == CUDA_SUCCESS && queue.query(dependent) == CUDA_SUCCESS &&
                fake.issued == (ranges ? std::vector<std::uint64_t>{0, 1024, 2048, 3072, 4096}
                                       : std::vector<std::uint64_t>{0, 1, 2}) &&
                queue.shutdown() == CUDA_SUCCESS,
            "all slices precede dependent completion");
    progress.reset();
    require(lifetime.expired(), "completed payload releases resources");
}
void failures(bool ranges = false) {
    for (bool fail_launch : {false, true}) {
        Fake fake;
        Input input;
        if (ranges) {
            input.count = 4194305;
            input.pointers = {0x10000000, 0x20000000, 0x30000000};
            input.mapping = QualifiedAddMapping::kRange;
            fake.maximum_width = 1024;
        }
        auto prepared = input.prepare({ranges ? 4097U : 3U, 1, 1});
        auto progress =
            std::dynamic_pointer_cast<CudaBinarySliceContinuation>(prepared->continuation);
        CudaOperationQueue queue(fake.context, Fake::dispatch());
        auto stream = queue.bind_stream(reinterpret_cast<CUstream>(0x8000),
                                        OrderedStreamKind::kNonBlocking, input.resources);
        auto work = queue.submit(stream, std::move(*prepared));
        (fail_launch ? fake.launch_error : fake.record_error) = CUDA_ERROR_UNKNOWN;
        require(queue.pump() == CUDA_ERROR_UNKNOWN && queue.pump() == CUDA_ERROR_UNKNOWN &&
                    !queue.resume(work) && !queue.cancel_stopped(work) && fake.issued.size() == 1 &&
                    progress->progress().completed_blocks == 0,
                "uncertain launch or fence cannot replay or advance");
        require(queue.shutdown() == CUDA_SUCCESS && fake.drains == 1,
                "uncertain execution requires physical drain");
    }
}
}  // namespace
int main() {
    try {
        preparation_and_state();
        range_preparation_and_state();
        diagnostic_completion_is_opt_in();
        rejection();
        queue_continuation(false);
        queue_continuation(true);
        queue_continuation(false, true);
        queue_continuation(true, true);
        failures();
        failures(true);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
