#include "internal/cuda_binary_slice.h"
#include "internal/allocation_registry.h"
#include "internal/cuda_copy_range.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <unistd.h>

namespace {
using namespace glimmer::interceptor;
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
struct Input {
    alignas(8) std::array<std::byte, 176> bytes{};
    std::array<void*, 1> arguments{bytes.data()};
    std::array<unsigned int, 3> grid{2, 2, 3};
    std::array<unsigned int, 3> block{256, 1, 1};
    CUfunction function{reinterpret_cast<CUfunction>(0x4000)};
    std::shared_ptr<void> resources{std::make_shared<int>(1)};
    template <typename T>
    void put(std::size_t offset, T value) {
        require(offset + sizeof(value) <= bytes.size(), "test parameter overflow");
        std::memcpy(bytes.data() + offset, &value, sizeof(value));
    }
    Input() {
        for (const auto offset : {0U, 8U, 16U, 168U})
            put(offset, std::uint64_t{0x10000000} * (offset + 1));
        put(24, std::uint64_t{129} * 4);
        put(32, std::uint64_t{33} * 4);
        put(40, std::uint64_t{129} * 32);
        put(48, std::uint64_t{33} * 32);
        put(56, std::uint64_t{129} * 16);
        put(64, std::uint64_t{33} * 16);
        put(72, std::uint64_t{129} * 33);
        put(80, std::uint64_t{129} * 33);
        for (const auto offset : {88U, 96U, 100U, 104U}) put(offset, std::uint32_t{129});
        for (const auto offset : {92U, 108U}) put(offset, std::uint32_t{33});
        for (const auto offset : {112U, 128U}) put(offset, std::uint32_t{16});
        put(132, std::uint32_t{1});
        put(152, 1.25F);
        put(156, -0.5F);
    }
    std::optional<CudaQueuedKernel> prepare(
        bool should_trace = false, QualifiedGemmMapping mapping = QualifiedGemmMapping::kPlane) {
        return prepare_mapped_gemm_slices(function, grid, block, arguments.data(), resources,
                                          should_trace, mapping);
    }
};

void diagnostic_requires_physical_proof() {
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
            "capture GEMM trace");
    for (const bool should_trace : {false, true}) {
        Input input;
        input.grid[2] = 1;
        auto prepared = input.prepare(should_trace);
        require(prepared.has_value(), "prepare traced GEMM");
        auto& continuation = *prepared->continuation;
        bool complete{};
        require(
            continuation.inspect_stopped(complete) == CUDA_ERROR_INVALID_VALUE &&
                continuation.prepare_launch(prepared->grid, prepared->arguments) == CUDA_SUCCESS &&
                continuation.stage_completion(nullptr) == CUDA_SUCCESS &&
                ::lseek(::fileno(capture.file), 0, SEEK_CUR) == 0,
            "unproved/default-off GEMM emits no completion");
        errno = EDOM;
        require(continuation.inspect_stopped(complete) == CUDA_SUCCESS && complete &&
                    errno == EDOM &&
                    continuation.inspect_stopped(complete) == CUDA_ERROR_INVALID_VALUE,
                "GEMM inspection preserves errno and cannot repeat");
    }
    require(std::fseek(capture.file, 0, SEEK_SET) == 0, "rewind GEMM trace");
    std::array<char, 160> bytes{};
    const auto size = std::fread(bytes.data(), 1, bytes.size(), capture.file);
    require(
        !std::ferror(capture.file) && std::string_view(bytes.data(), size) ==
                                          "[glimmer] gemm_slice blocks=4 completed=4 launches=1\n",
        "exact GEMM completion trace");
}

struct MemoryProof {
    struct Range {
        CUdeviceptr base;
        std::size_t size;
    };
    inline static MemoryProof* active{};
    Input input;
    CUcontext context{reinterpret_cast<CUcontext>(0x5000)};
    CUdevice device{};
    AllocationRegistry allocations;
    std::array<Range, 4> ranges{{{0x10000000, std::size_t{129} * 16 * 3 * 4},
                                 {0x90000000, std::size_t{33} * 16 * 3 * 4},
                                 {0x110000000, std::size_t{129} * 33 * 3 * 4},
                                 {0xa90000000, std::size_t{129} * 33 * 3 * 4}}};
    unsigned int calls{}, fail_call{};
    CUmemorytype type{CU_MEMORYTYPE_DEVICE};
    CUcontext reported_context{context};
    unsigned int managed{}, mapped{1};
    bool mutate_original{};
    MemoryProof() {
        active = this;
    }
    ~MemoryProof() {
        active = nullptr;
    }
    MemoryProof(const MemoryProof&) = delete;
    MemoryProof& operator=(const MemoryProof&) = delete;
    static CUresult CUDAAPI attribute(void* value, CUpointer_attribute key, CUdeviceptr) {
        auto& proof = *active;
        if (++proof.calls == proof.fail_call) return CUDA_ERROR_UNKNOWN;
        if (proof.mutate_original) proof.input.bytes.fill(std::byte{0xff});
        switch (key) {
            case CU_POINTER_ATTRIBUTE_MEMORY_TYPE:
                *static_cast<CUmemorytype*>(value) = proof.type;
                break;
            case CU_POINTER_ATTRIBUTE_CONTEXT:
                *static_cast<CUcontext*>(value) = proof.reported_context;
                break;
            case CU_POINTER_ATTRIBUTE_IS_MANAGED:
                *static_cast<unsigned int*>(value) = proof.managed;
                break;
            case CU_POINTER_ATTRIBUTE_MAPPED:
                *static_cast<unsigned int*>(value) = proof.mapped;
                break;
            default:
                return CUDA_ERROR_INVALID_VALUE;
        }
        return CUDA_SUCCESS;
    }
    static CUresult CUDAAPI address_range(CUdeviceptr* base, std::size_t* size, CUdeviceptr ptr) {
        auto& proof = *active;
        if (++proof.calls == proof.fail_call) return CUDA_ERROR_UNKNOWN;
        for (const auto& range : proof.ranges) {
            if (ptr >= range.base && ptr - range.base < range.size) {
                *base = range.base;
                *size = range.size;
                return CUDA_SUCCESS;
            }
        }
        return CUDA_ERROR_INVALID_VALUE;
    }
    CudaCopyRangeDispatch dispatch{attribute, address_range, nullptr};
    void record(std::size_t skip = 4,
                AllocationProvenance provenance = AllocationProvenance::kIndependentDevice,
                AllocationScope scope = AllocationScope::kContextBound) {
        for (std::size_t index = 0; index != ranges.size(); ++index) {
            if (index == skip) continue;
            require(allocations.record({ranges[index].base, context, device}, ranges[index].size,
                                       scope, provenance),
                    "proof allocation record");
        }
    }
    std::optional<CudaQueuedKernel> prepare() {
        return prepare_checked_mapped_gemm_slices(input.function, input.grid, input.block,
                                                  input.arguments.data(), input.resources, dispatch,
                                                  allocations, context, device);
    }
};

void allocation_proof() {
    {
        MemoryProof proof;
        proof.record();
        const auto bytes = proof.input.bytes;
        proof.mutate_original = true;
        auto prepared = proof.prepare();
        require(
            prepared && proof.calls == 24 &&
                std::memcmp(prepared->arguments.addresses()[0], bytes.data(), bytes.size()) == 0,
            "all four full extents use owned parameters, not mutated caller storage");
    }
    for (unsigned int failure = 1; failure <= 24; ++failure) {
        MemoryProof proof;
        proof.record();
        proof.fail_call = failure;
        require(!proof.prepare() && proof.calls == failure, "every metadata failure rejects");
    }
    for (std::size_t index = 0; index != 4; ++index) {
        {
            MemoryProof proof;
            proof.record(index);
            require(!proof.prepare(), "every operand requires observed allocation provenance");
        }
        {
            MemoryProof proof;
            proof.record();
            --proof.ranges[index].size;
            require(!proof.prepare(), "complete final padded extent needs native coverage");
        }
        {
            MemoryProof proof;
            --proof.ranges[index].size;
            proof.record();
            ++proof.ranges[index].size;
            require(!proof.prepare(), "native bounds cannot enlarge ledger allocation");
        }
        {
            MemoryProof proof;
            proof.record();
            auto [status, ticket] = proof.allocations.begin_release(
                {proof.ranges[index].base, proof.context, proof.device});
            require(
                status == AllocationRegistry::ReleaseStatus::kStarted && ticket && !proof.prepare(),
                "release in progress rejects");
            proof.allocations.cancel_release(*ticket);
            require(proof.prepare().has_value(), "failed free preserves provenance");
            auto release = proof.allocations.begin_release(ticket->identity);
            require(release.second && proof.allocations.complete_release(*release.second) &&
                        !proof.prepare(),
                    "completed free removes eligibility");
            require(proof.allocations.record(ticket->identity, ticket->memory_bytes) &&
                        !proof.prepare(),
                    "recycled pointer cannot inherit old provenance");
        }
    }
    for (unsigned int mode = 0; mode != 11; ++mode) {
        MemoryProof proof;
        proof.record(
            4,
            mode == 0 ? AllocationProvenance::kUnknown : AllocationProvenance::kIndependentDevice,
            mode == 1 ? AllocationScope::kContextIndependent : AllocationScope::kContextBound);
        if (mode == 2) proof.type = CU_MEMORYTYPE_HOST;
        if (mode == 3) proof.reported_context = reinterpret_cast<CUcontext>(0x6000);
        if (mode == 4) proof.managed = 1;
        if (mode == 5) proof.mapped = 0;
        if (mode == 6) proof.dispatch.pointer_attribute = nullptr;
        if (mode == 7) proof.dispatch.address_range = nullptr;
        if (mode == 8) proof.device = 1;
        if (mode == 9) proof.allocations.mark_accounting_degraded();
        if (mode == 10) proof.context = nullptr;
        require(!proof.prepare(), "unproven storage or unavailable metadata rejects");
    }
    {
        MemoryProof proof;
        for (auto& range : proof.ranges) {
            range.base -= 128;
            range.size += 256;
        }
        proof.record();
        require(proof.prepare().has_value(), "guarded interior allocation spans accepted");
        proof.input.put(168, std::uint64_t{0x110000000});
        require(proof.prepare().has_value(), "exact in-place C allocation accepted");
    }
    {
        MemoryProof proof;
        proof.record();
        proof.input.put(132, std::uint32_t{9});
        require(!proof.prepare() && proof.calls == 0, "invalid original ABI performs no queries");
    }
}
std::uint64_t index(const OwnedKernelArguments& arguments) {
    std::uint64_t result{};
    std::memcpy(&result, static_cast<const std::byte*>(arguments.addresses()[0]) + 176, 8);
    return result;
}
void ownership_and_state() {
    Input input;
    const auto original = input.bytes;
    auto prepared = input.prepare();
    require(prepared.has_value() && input.bytes == original, "nonmutating preparation");
    auto kernel = std::move(*prepared);
    prepared.reset();
    input.bytes.fill(std::byte{0xff});
    input.arguments.fill(nullptr);
    require(std::memcmp(kernel.arguments.addresses()[0], original.data(), original.size()) == 0 &&
                index(kernel.arguments) == 0,
            "owned original prefix and initialized private slot");
    auto progress = std::dynamic_pointer_cast<CudaBinarySliceContinuation>(kernel.continuation);
    require(progress && progress->progress().logical_blocks == 12, "logical CTA count includes XY");
    bool complete = true;
    require(progress->inspect_stopped(complete) == CUDA_ERROR_INVALID_VALUE && !complete &&
                progress->stage_completion(nullptr) == CUDA_ERROR_INVALID_VALUE,
            "unissued completion rejects");
    auto empty = OwnedKernelArguments::copy({}, nullptr);
    require(progress->prepare_launch(kernel.grid, *empty) == CUDA_ERROR_INVALID_VALUE &&
                progress->progress().physical_launches == 0,
            "missing private slot does not advance");
    for (std::uint64_t batch = 0; batch != 3; ++batch) {
        require(
            progress->prepare_launch(kernel.grid, kernel.arguments) == CUDA_SUCCESS &&
                kernel.grid == std::array<unsigned int, 3>{2, 2, 1} &&
                index(kernel.arguments) == batch &&
                std::memcmp(kernel.arguments.addresses()[0], original.data(), original.size()) == 0,
            "only private index changes and XY geometry survives");
        require(
            progress->prepare_launch(kernel.grid, kernel.arguments) == CUDA_ERROR_INVALID_VALUE &&
                progress->request_retreat() == CUDA_SUCCESS &&
                progress->inspect_stopped(complete) == CUDA_ERROR_INVALID_VALUE &&
                progress->progress().completed_blocks == batch * 4,
            "duplicate preparation, retreat and missing proof cannot advance");
        require(progress->stage_completion(nullptr) == CUDA_SUCCESS &&
                    progress->stage_completion(nullptr) == CUDA_ERROR_INVALID_VALUE &&
                    progress->inspect_stopped(complete) == CUDA_SUCCESS &&
                    complete == (batch == 2) &&
                    progress->progress().completed_blocks == (batch + 1) * 4 &&
                    progress->progress().physical_launches == batch + 1 &&
                    progress->inspect_stopped(complete) == CUDA_ERROR_INVALID_VALUE,
                "exactly one proof advances a complete plane");
    }
    require(progress->prepare_launch(kernel.grid, kernel.arguments) == CUDA_ERROR_INVALID_VALUE &&
                progress->request_retreat() == CUDA_ERROR_INVALID_VALUE,
            "completed work cannot replay");
}
void range_ownership_and_state() {
    for (unsigned int batches : {1U, 7U, 8U, 9U, 16U, 17U, 36U}) {
        Input input;
        input.grid[2] = batches;
        const auto original = input.bytes;
        auto prepared = input.prepare(false, QualifiedGemmMapping::kRange);
        require(prepared && input.bytes == original, "nonmutating range preparation");
        input.bytes.fill(std::byte{0xff});
        auto progress =
            std::dynamic_pointer_cast<CudaBinarySliceContinuation>(prepared->continuation);
        bool complete = true;
        auto empty = OwnedKernelArguments::copy({}, nullptr);
        require(progress && progress->progress().logical_blocks == std::uint64_t{batches} * 4 &&
                    progress->prepare_launch(prepared->grid, *empty) == CUDA_ERROR_INVALID_VALUE &&
                    progress->progress().physical_launches == 0,
                "range preserves logical XYZ and rejects a missing private slot");
        std::uint64_t launches{};
        for (unsigned int base = 0; base < batches; base += 8) {
            const auto planes = std::min(8U, batches - base);
            require(progress->prepare_launch(prepared->grid, prepared->arguments) == CUDA_SUCCESS &&
                        prepared->grid == std::array{2U, 2U, planes} &&
                        index(prepared->arguments) == base &&
                        std::memcmp(prepared->arguments.addresses()[0], original.data(),
                                    original.size()) == 0,
                    "range changes only private base and retains bounded XY/tail geometry");
            require(progress->prepare_launch(prepared->grid, prepared->arguments) ==
                            CUDA_ERROR_INVALID_VALUE &&
                        progress->request_retreat() == CUDA_SUCCESS &&
                        progress->inspect_stopped(complete) == CUDA_ERROR_INVALID_VALUE &&
                        progress->progress().completed_blocks == std::uint64_t{base} * 4,
                    "range cannot advance without actual physical proof");
            require(
                progress->stage_completion(nullptr) == CUDA_SUCCESS &&
                    progress->stage_completion(nullptr) == CUDA_ERROR_INVALID_VALUE &&
                    progress->inspect_stopped(complete) == CUDA_SUCCESS &&
                    complete == (base + planes == batches) &&
                    progress->progress().completed_blocks == (std::uint64_t{base} + planes) * 4 &&
                    progress->progress().physical_launches == ++launches &&
                    progress->inspect_stopped(complete) == CUDA_ERROR_INVALID_VALUE,
                "one proof advances exactly the issued range, including partial tail");
        }
        require(progress->prepare_launch(prepared->grid, prepared->arguments) ==
                        CUDA_ERROR_INVALID_VALUE &&
                    progress->request_retreat() == CUDA_ERROR_INVALID_VALUE,
                "completed range cannot replay");
    }
    Input input;
    require(!input.prepare(false, static_cast<QualifiedGemmMapping>(99)),
            "unknown GEMM mapping rejects");
    input.put(132, std::uint32_t{9});
    require(!input.prepare(false, QualifiedGemmMapping::kRange), "range cannot enable split mode");
}
void inactive_semaphore_bytes() {
    for (const auto value : {std::uint64_t{0}, std::uint64_t{1}, std::uint64_t{0x1234567800000000},
                             ~std::uint64_t{0}}) {
        Input input;
        input.put(120, value);
        auto prepared = input.prepare();
        require(prepared && std::memcmp(prepared->arguments.addresses()[0], input.bytes.data(),
                                        input.bytes.size()) == 0,
                "mode-one preserves opaque inactive semaphore bytes");
        input.put(132, std::uint32_t{9});
        require(!input.prepare(), "nonzero or zero semaphore never enables split mode");
    }
}
void rejection() {
    for (std::size_t index = 0; index != 4; ++index) {
        const std::uint32_t rows = index == 1 ? 33 : 129;
        for (const auto leading : {rows - 1, rows + 6}) {
            Input input;
            input.put(88 + index * 4, leading);
            require(!input.prepare(), "each leading dimension enforces both bounds");
        }
        const std::uint64_t extent = std::uint64_t{rows} * (index < 2 ? 16 : 33);
        for (const auto stride : {extent - 1, extent + 6}) {
            Input input;
            input.put(56 + index * 8, stride);
            require(!input.prepare(), "each batch stride enforces independence and padding bounds");
        }
    }
    for (const auto offset : {140U, 148U}) {
        Input input;
        input.put(offset, std::uint32_t{1});
        require(!input.prepare(), "upper halves of unsupported pointers reject");
    }
    for (const auto [offset, value] : std::array<std::array<unsigned int, 2>, 19>{{{104, 0},
                                                                                   {104, 130},
                                                                                   {108, 0},
                                                                                   {108, 34},
                                                                                   {112, 0},
                                                                                   {112, 2049},
                                                                                   {128, 15},
                                                                                   {132, 0},
                                                                                   {132, 9},
                                                                                   {132, 4096},
                                                                                   {116, 1},
                                                                                   {136, 1},
                                                                                   {144, 1},
                                                                                   {160, 1},
                                                                                   {164, 1},
                                                                                   {88, 128},
                                                                                   {92, 32},
                                                                                   {96, 135},
                                                                                   {100, 135}}}) {
        Input input;
        input.put(offset, value);
        const auto before = input.bytes;
        require(!input.prepare() && input.bytes == before, "invalid ABI rejects without mutation");
    }
    for (auto offset : {24U, 32U, 40U, 48U, 56U, 64U, 72U, 80U}) {
        for (auto value : {std::uint64_t{0}, std::numeric_limits<std::uint64_t>::max()}) {
            Input input;
            input.put(offset, value);
            require(!input.prepare(), "incorrect increments, stride or overflow rejects");
        }
    }
    for (auto offset : {0U, 8U, 16U, 168U}) {
        for (auto value :
             {std::uint64_t{0}, std::uint64_t{1}, std::numeric_limits<std::uint64_t>::max() - 15}) {
            Input input;
            input.put(offset, value);
            require(!input.prepare(), "null, misaligned or wrapped allocation rejects");
        }
    }
    for (auto offset : {0U, 8U, 168U}) {
        for (auto delta : {std::int64_t{-16}, std::int64_t{0}, std::int64_t{16}}) {
            Input input;
            input.put(offset, static_cast<std::uint64_t>(0x110000000LL + delta));
            require(input.prepare().has_value() == (offset == 168 && delta == 0),
                    "only exact input C/output C alias qualifies");
        }
    }
    for (auto offset : {80U, 100U}) {
        Input input;
        input.put(168, std::uint64_t{0x110000000});
        if (offset == 80)
            input.put(offset, std::uint64_t{129 * 33 + 1});
        else {
            input.put(offset, std::uint32_t{130});
            input.put(80, std::uint64_t{130} * 33);
        }
        require(!input.prepare(), "same pointer with differing C layouts rejects");
    }
    Input input;
    input.put(96, std::uint32_t{132});
    input.put(72, std::uint64_t{132} * 33);
    input.put(0, std::uint64_t{0x110000000} + std::uint64_t{132} * 33 * 3 * 4);
    require(input.prepare().has_value(), "adjacent disjoint output/input extents qualify");
    for (auto grid : {std::array{0U, 2U, 3U}, std::array{2U, 0U, 3U}, std::array{2U, 2U, 0U},
                      std::array{1U, 2U, 3U}, std::array{2U, 1U, 3U}, std::array{2U, 2U, 37U}}) {
        input.grid = grid;
        require(!input.prepare(), "invalid geometry rejects");
    }
    input.grid = {2, 2, 36};
    // Restore disjoint addresses before increasing the batch count.
    input.put(0, std::uint64_t{0x10000000});
    require(input.prepare().has_value(), "maximum batch count qualifies");
    input.block = {128, 1, 1};
    require(!input.prepare(), "wrong thread shape rejects");
    input.block = {256, 1, 1};
    input.arguments[0] = nullptr;
    require(!input.prepare(), "missing argument rejects");
    input.arguments[0] = input.bytes.data();
    input.function = nullptr;
    require(!input.prepare(), "missing function rejects");
    input.function = reinterpret_cast<CUfunction>(0x4000);
    input.resources.reset();
    require(!input.prepare(), "missing resource owner rejects");
}

void padded_and_boundary_inputs() {
    Input input;
    for (std::size_t index = 0; index != 4; ++index) {
        const std::uint32_t leading = (index == 1 ? 33U : 129U) + 5;
        input.put(88 + index * 4, leading);
        input.put(56 + index * 8, std::uint64_t{leading} * (index < 2 ? 16 : 33) + 5);
        if (index < 2) {
            input.put(24 + index * 8, std::uint64_t{leading} * 4);
            input.put(40 + index * 8, std::uint64_t{leading} * 32);
        }
    }
    require(input.prepare().has_value(), "maximum leading and batch padding qualifies");
    input.put(112, std::uint32_t{2048});
    input.put(128, std::uint32_t{2048});
    input.put(56, std::uint64_t{134 * 2048 + 5});
    input.put(64, std::uint64_t{38 * 2048 + 5});
    input.grid[2] = 36;
    require(input.prepare().has_value(), "maximum bounded extents qualify without integer wrap");
    for (const auto offset : {88U, 92U, 96U, 100U, 104U, 108U, 112U, 128U})
        input.put(offset, std::uint32_t{1});
    for (const auto offset : {56U, 64U, 72U, 80U}) input.put(offset, std::uint64_t{1});
    for (const auto offset : {24U, 32U}) input.put(offset, std::uint64_t{4});
    for (const auto offset : {40U, 48U}) input.put(offset, std::uint64_t{32});
    input.grid = {1, 1, 1};
    require(input.prepare().has_value(), "minimum extents qualify");
    require(!prepare_mapped_gemm_slices(input.function, input.grid, input.block, nullptr,
                                        input.resources),
            "missing argument table rejects");
    for (const auto block : {std::array{256U, 2U, 1U}, std::array{256U, 1U, 2U}}) {
        input.block = block;
        require(!input.prepare(), "nontrivial block Y/Z rejects");
    }
}
struct Event {
    bool ready{};
};
struct Fake {
    static inline Fake* active{};
    CUcontext context{reinterpret_cast<CUcontext>(0x1000)};
    std::vector<std::unique_ptr<Event>> events;
    std::vector<std::uint64_t> issued;
    CUresult launch_error{CUDA_SUCCESS}, record_error{CUDA_SUCCESS}, query_error{CUDA_SUCCESS};
    int drains{};
    unsigned int width{1};
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
        if (active->query_error != CUDA_SUCCESS) return active->query_error;
        return reinterpret_cast<Event*>(event)->ready ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
    }
    static CUresult destroy(CUevent) {
        return CUDA_SUCCESS;
    }
    static CUresult drain(CUstream) {
        ++active->drains;
        return CUDA_SUCCESS;
    }
    // NOLINTBEGIN(bugprone-easily-swappable-parameters): exact CUDA launch callback ABI.
    static CUresult launch(CUfunction, unsigned int gx, unsigned int gy, unsigned int gz,
                           unsigned int bx, unsigned int by, unsigned int bz, unsigned int shared,
                           CUstream, void** arguments, void**) {
        require(gx == 2 && gy == 2 && bx == 256 && by == 1 && bz == 1 && !shared,
                "mapped XY/block shape");
        std::uint64_t batch{};
        std::memcpy(&batch, static_cast<const std::byte*>(arguments[0]) + 176, 8);
        require(
            gz == (active->width == 1 ? 1U : std::min(8U, 19U - static_cast<unsigned int>(batch))),
            "mapped plane/range tail shape");
        active->issued.push_back(batch);
        return active->launch_error;
    }
    // NOLINTEND(bugprone-easily-swappable-parameters)
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
void queue_lifecycle(bool cancel, QualifiedGemmMapping mapping = QualifiedGemmMapping::kPlane) {
    Fake fake;
    Input input;
    const unsigned int width = mapping == QualifiedGemmMapping::kRange ? 8U : 1U;
    fake.width = width;
    if (width == 8) input.grid[2] = 19;
    auto prepared = input.prepare(false, mapping);
    auto progress = std::dynamic_pointer_cast<CudaBinarySliceContinuation>(prepared->continuation);
    std::weak_ptr<void> lifetime = input.resources;
    CudaOperationQueue queue(fake.context, Fake::dispatch());
    auto stream = queue.bind_stream(reinterpret_cast<CUstream>(0x8000),
                                    OrderedStreamKind::kNonBlocking, std::make_shared<int>(1));
    auto work = queue.submit(stream, std::move(*prepared));
    auto dependent = queue.submit(stream, CudaQueuedMarker{});
    prepared.reset();
    input.resources.reset();
    input.bytes.fill(std::byte{});
    require(work && dependent && !lifetime.expired(), "owned queue payload");
    require(queue.pump_admitted({}) == CUDA_SUCCESS && fake.issued.empty(), "no grant no issue");
    for (std::uint64_t batch = 0; batch != 3; ++batch) {
        const auto completed = std::min<std::uint64_t>(input.grid[2], (batch + 1) * width) * 4;
        require(queue.pump_admitted(std::array{work}) == CUDA_SUCCESS &&
                    fake.issued.back() == batch * width,
                "explicitly admitted batch");
        require(queue.pump_admitted({}) == CUDA_SUCCESS && !queue.cancel_stopped(work) &&
                    progress->progress().completed_blocks == batch * width * 4 &&
                    queue.request_retreat(work) == CUDA_SUCCESS,
                "pending event is not proof");
        fake.ready();
        require(queue.pump_admitted({}) == CUDA_SUCCESS &&
                    progress->progress().completed_blocks == completed,
                "proven plane advances");
        if (batch == 2) continue;
        require(queue.state(work) == OrderedOperationState::kSuspended &&
                    queue.query(dependent) == CUDA_ERROR_NOT_READY &&
                    queue.pump_admitted({}) == CUDA_SUCCESS && fake.issued.size() == batch + 1,
                "suspended work retains dependencies and cannot auto-issue");
        if (cancel) {
            require(queue.cancel_stopped(work) && queue.shutdown() == CUDA_SUCCESS &&
                        fake.issued == std::vector<std::uint64_t>{0} &&
                        progress->progress().completed_blocks == std::uint64_t{width} * 4,
                    "stopped cancel");
            progress.reset();
            require(lifetime.expired(), "cancelled payload releases only after proof");
            return;
        }
        require(queue.resume(work) && !queue.resume(work), "explicit exactly-once resume");
    }
    fake.ready();
    require(queue.pump_admitted({}) == CUDA_SUCCESS && queue.query(dependent) == CUDA_SUCCESS &&
                fake.issued == std::vector<std::uint64_t>{0, width, std::uint64_t{2} * width} &&
                queue.shutdown() == CUDA_SUCCESS,
            "dependent completes after all planes");
    progress.reset();
    require(lifetime.expired(), "complete payload releases ownership");
}
void uncertain_failures(QualifiedGemmMapping mapping = QualifiedGemmMapping::kPlane) {
    for (unsigned failure = 0; failure != 3; ++failure) {
        Fake fake;
        Input input;
        if (mapping == QualifiedGemmMapping::kRange) {
            fake.width = 8;
            input.grid[2] = 19;
        }
        auto prepared = input.prepare(false, mapping);
        auto progress =
            std::dynamic_pointer_cast<CudaBinarySliceContinuation>(prepared->continuation);
        CudaOperationQueue queue(fake.context, Fake::dispatch());
        auto stream = queue.bind_stream(reinterpret_cast<CUstream>(0x8000),
                                        OrderedStreamKind::kNonBlocking, input.resources);
        auto work = queue.submit(stream, std::move(*prepared));
        if (failure == 0) fake.launch_error = CUDA_ERROR_UNKNOWN;
        if (failure == 1) fake.record_error = CUDA_ERROR_UNKNOWN;
        if (failure == 2) {
            require(queue.pump() == CUDA_SUCCESS, "issued before query error");
            fake.query_error = CUDA_ERROR_UNKNOWN;
        }
        require(queue.pump() == CUDA_ERROR_UNKNOWN && queue.pump() == CUDA_ERROR_UNKNOWN &&
                    !queue.resume(work) && !queue.cancel_stopped(work) && fake.issued.size() == 1 &&
                    progress->progress().completed_blocks == 0,
                "uncertainty cannot replay or advance");
        require(queue.shutdown() == CUDA_SUCCESS && fake.drains == 1, "uncertainty requires drain");
    }
}
}  // namespace
int main() {
    try {
        ownership_and_state();
        range_ownership_and_state();
        inactive_semaphore_bytes();
        rejection();
        padded_and_boundary_inputs();
        queue_lifecycle(false);
        queue_lifecycle(true);
        uncertain_failures();
        queue_lifecycle(false, QualifiedGemmMapping::kRange);
        queue_lifecycle(true, QualifiedGemmMapping::kRange);
        uncertain_failures(QualifiedGemmMapping::kRange);
        allocation_proof();
        diagnostic_requires_physical_proof();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
