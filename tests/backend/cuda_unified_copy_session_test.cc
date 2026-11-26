#include "cuda_execution_domain_fixture.h"

#include <limits>
#include <memory>
#include <new>
#include <thread>
#include <fcntl.h>
#include <sys/mman.h>

namespace {
// Fault only the caller's next allocation inside an explicitly bounded admission.
// Clear before throwing so exception storage and all cleanup remain ordinary.
thread_local bool should_fail_allocation{};
thread_local unsigned int injected_allocation_failures{};
struct AllocationRequest {
    std::size_t bytes{}, alignment{};
};
void* allocate_test_memory(AllocationRequest request) {
    if (std::exchange(should_fail_allocation, false)) {
        ++injected_allocation_failures;
        throw std::bad_alloc();
    }
    void* pointer{};
    if (request.alignment) {
        if (::posix_memalign(&pointer, request.alignment,
                             std::max(request.bytes, std::size_t{1})) != 0)
            throw std::bad_alloc();
    } else {
        pointer = std::malloc(std::max(request.bytes, std::size_t{1}));
    }
    if (!pointer) throw std::bad_alloc();
    return pointer;
}
}  // namespace

// Complete replacement allocation/deallocation pairs keep the fixture's fault
// injection compatible with array, sized, aligned and nothrow callers under ASan.
void* operator new(std::size_t bytes) {
    return allocate_test_memory({bytes, 0});
}
void* operator new[](std::size_t bytes) {
    return allocate_test_memory({bytes, 0});
}
void* operator new(std::size_t bytes, std::align_val_t alignment) {
    return allocate_test_memory({bytes, static_cast<std::size_t>(alignment)});
}
void* operator new[](std::size_t bytes, std::align_val_t alignment) {
    return allocate_test_memory({bytes, static_cast<std::size_t>(alignment)});
}
void* operator new(std::size_t bytes, const std::nothrow_t&) noexcept {
    try {
        return ::operator new(bytes);
    } catch (const std::bad_alloc&) {
        return nullptr;
    }
}
void* operator new[](std::size_t bytes, const std::nothrow_t&) noexcept {
    try {
        return ::operator new[](bytes);
    } catch (const std::bad_alloc&) {
        return nullptr;
    }
}
void* operator new(std::size_t bytes, std::align_val_t alignment, const std::nothrow_t&) noexcept {
    try {
        return ::operator new(bytes, alignment);
    } catch (const std::bad_alloc&) {
        return nullptr;
    }
}
void* operator new[](std::size_t bytes, std::align_val_t alignment,
                     const std::nothrow_t&) noexcept {
    try {
        return ::operator new[](bytes, alignment);
    } catch (const std::bad_alloc&) {
        return nullptr;
    }
}
void operator delete(void* pointer) noexcept {
    std::free(pointer);
}
void operator delete[](void* pointer) noexcept {
    std::free(pointer);
}
void operator delete(void* pointer, std::size_t) noexcept {
    std::free(pointer);
}
void operator delete[](void* pointer, std::size_t) noexcept {
    std::free(pointer);
}
void operator delete(void* pointer, std::align_val_t) noexcept {
    std::free(pointer);
}
void operator delete[](void* pointer, std::align_val_t) noexcept {
    std::free(pointer);
}
void operator delete(void* pointer, std::size_t, std::align_val_t) noexcept {
    std::free(pointer);
}
void operator delete[](void* pointer, std::size_t, std::align_val_t) noexcept {
    std::free(pointer);
}
void operator delete(void* pointer, const std::nothrow_t&) noexcept {
    std::free(pointer);
}
void operator delete[](void* pointer, const std::nothrow_t&) noexcept {
    std::free(pointer);
}
void operator delete(void* pointer, std::align_val_t, const std::nothrow_t&) noexcept {
    std::free(pointer);
}
void operator delete[](void* pointer, std::align_val_t, const std::nothrow_t&) noexcept {
    std::free(pointer);
}

namespace {
using namespace glimmer::backend;
using namespace glimmer::backend::cuda;
using namespace glimmer::test::cuda_domain;
constexpr std::size_t kBytes = 128, kOffset = 17, kCopied = 59;
struct Invocation {
    CUdeviceptr destination{}, source{};
    std::size_t bytes{};
    CUstream stream{};
    bool is_asynchronous{};
};
struct Trace {
    std::vector<Invocation> calls;
    std::vector<void*> registered;
    std::vector<CUstream> recorded_streams;
    std::vector<CUstream> waited_streams;
    unsigned int directional{}, registrations{}, maps{}, unregistrations{};
    CUresult status{CUDA_SUCCESS};
} trace;

std::byte* endpoint_bytes(CUdeviceptr address, std::size_t bytes) {
    for (const auto& entry : driver.allocations)
        if (entry.is_live && address >= entry.address && address - entry.address <= entry.bytes)
            return device_bytes(address, bytes);
    // The exact generic SDK ABI encodes a previously validated HOST pointer as CUdeviceptr.
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    return reinterpret_cast<std::byte*>(static_cast<std::uintptr_t>(address));
}
CUresult invoke_copy(Invocation invocation) {
    expect(driver.current, "generic copy executes on the actual selected owner");
    trace.calls.push_back(invocation);
    if (trace.status == CUDA_SUCCESS && invocation.bytes)
        std::memcpy(endpoint_bytes(invocation.destination, invocation.bytes),
                    endpoint_bytes(invocation.source, invocation.bytes), invocation.bytes);
    return trace.status;
}
// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact original CUDA callback ABIs.
CUresult CUDAAPI unified(CUdeviceptr destination, CUdeviceptr source, std::size_t bytes) {
    return invoke_copy({destination, source, bytes, nullptr, false});
}
CUresult CUDAAPI unified_async(CUdeviceptr destination, CUdeviceptr source, std::size_t bytes,
                               CUstream stream) {
    return invoke_copy({destination, source, bytes, stream, true});
}
CUresult CUDAAPI directional_copy(CUdeviceptr, CUdeviceptr, std::size_t) {
    ++trace.directional;
    return CUDA_ERROR_UNKNOWN;
}
CUresult CUDAAPI directional_copy_async(CUdeviceptr, CUdeviceptr, std::size_t, CUstream) {
    ++trace.directional;
    return CUDA_ERROR_UNKNOWN;
}
// NOLINTEND(bugprone-easily-swappable-parameters)
CudaSessionDispatch dispatch(bool has_unified = true) {
    auto sdk = session_dispatch();
    sdk.execution.copy_device = directional_copy;
    sdk.execution.copy_device_async = directional_copy_async;
    sdk.execution.upload_async = [](CUdeviceptr, const void*, std::size_t, CUstream) {
        ++trace.directional;
        return CUDA_ERROR_UNKNOWN;
    };
    sdk.execution.download_async = [](void*, CUdeviceptr, std::size_t, CUstream) {
        ++trace.directional;
        return CUDA_ERROR_UNKNOWN;
    };
    sdk.upload_async = sdk.execution.upload_async;
    sdk.download_async = sdk.execution.download_async;
    if (has_unified) {
        sdk.execution.copy_unified = unified;
        sdk.execution.copy_unified_async = unified_async;
    }
    sdk.execution.event_record = [](CUevent event, CUstream stream) {
        trace.recorded_streams.push_back(stream);
        return event_record(event, stream);
    };
    sdk.stream_wait_event = [](CUstream stream, CUevent event, unsigned int flags) {
        trace.waited_streams.push_back(stream);
        return stream_wait_event(stream, event, flags);
    };
    sdk.execution.stream_wait_event = sdk.stream_wait_event;
    sdk.host_register = [](void* pointer, std::size_t, unsigned int flags) {
        expect(flags == (CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP),
               "registered host imports retain their existing flags");
        ++trace.registrations;
        trace.registered.push_back(pointer);
        return CUDA_SUCCESS;
    };
    sdk.host_unregister = [](void* pointer) {
        expect(std::erase(trace.registered, pointer) == 1, "unregister exactly one native owner");
        ++trace.unregistrations;
        return CUDA_SUCCESS;
    };
    sdk.host_map = [](CUdeviceptr* address, void* pointer, unsigned int flags) {
        expect(flags == 0, "native host alias query preserves flags");
        ++trace.maps;
        if (std::ranges::find(trace.registered, pointer) == trace.registered.end())
            return CUDA_ERROR_INVALID_VALUE;
        // Distinct device alias proves generic host operands use pointer, not address.
        *address = reinterpret_cast<CUdeviceptr>(pointer) + CUdeviceptr{1024} * 1024;
        return CUDA_SUCCESS;
    };
    return sdk;
}
struct Descriptor {
    int value{-1};
    explicit Descriptor(std::size_t bytes) {
        value = ::memfd_create("glimmer-unified-copy-test", MFD_CLOEXEC | MFD_ALLOW_SEALING);
        expect(value >= 0, "create a private sealed host backing descriptor");
        expect(::ftruncate(value, static_cast<off_t>(bytes)) == 0 &&
                   ::fcntl(value, F_ADD_SEALS, F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL) == 0,
               "complete backing extent and required seals");
    }
    ~Descriptor() {
        if (value >= 0 && ::close(value) != 0) std::abort();
    }
    Descriptor(const Descriptor&) = delete;
    Descriptor& operator=(const Descriptor&) = delete;
};
struct OwnerOptions {
    bool uses_proxy{}, has_unified{true}, has_registration{true};
    std::size_t host_pages{4};
};
struct Owner {
    std::unique_ptr<CudaExecutionSession> session;
    std::size_t page{};
    std::uint64_t next{1};
    explicit Owner(OwnerOptions options = {}) {
        driver = {};
        trace = {};
        driver.requires_explicit_cleanup = true;
        const auto system_page = ::sysconf(_SC_PAGESIZE);
        expect(system_page > 0, "positive platform page size");
        page = static_cast<std::size_t>(system_page);
        auto sdk = dispatch(options.has_unified);
        if (!options.has_registration) {
            sdk.host_register = nullptr;
            sdk.host_unregister = nullptr;
            sdk.host_map = nullptr;
        }
        CudaSessionLimits bounds;
        bounds.submissions = {2, 4096};
        bounds.max_host_memory_bytes = page * options.host_pages;
        session = std::make_unique<CudaExecutionSession>(
            ExecutionSessionIdentity{51, 52}, 53, sdk, bounds, CudaSessionPersistentConfig{},
            nullptr, nullptr, false,
            CudaSessionStreamConfig{.uses_default_proxy = options.uses_proxy});
        success(session->initialize(0));
    }
    CudaSessionResult import(bool is_pageable = true) {
        Descriptor backing(page);
        auto result = session->import_host_memory(backing.value, kBytes, page, 0, is_pageable);
        success(result);
        expect(result.id && (is_pageable ? !result.address : result.address != 0),
               "pageable import never invents a registered device alias");
        expect(is_pageable ? !result.original_status : result.original_status == CUDA_SUCCESS,
               "unregistered CPU import is synthetic, while real registration keeps provenance");
        return result;
    }
    ExecutionAcceptance accept(const CudaSessionUvaCopy& copy) {
        auto accepted = session->accept_uva_copy(next, copy);
        if (accepted.accepted()) ++next;
        return accepted;
    }
    void issue(const ExecutionAcceptance& accepted) {
        expect(accepted.accepted() && !session->host_result(accepted.receipt),
               "queue acceptance cannot manufacture a generic SDK result");
        driver.query_status = CUDA_ERROR_NOT_READY;
        const auto issued = session->issue_next();
        const auto reply = session->host_result(accepted.receipt);
        expect(issued.state == CudaExecutionIssueState::kIssued &&
                   issued.native_status == trace.status && reply &&
                   reply->origin == ExecutionReplyOrigin::kNativeSdk &&
                   reply->status == trace.status && !issued.library_status,
               "exact original Driver result survives independent physical retirement");
        expect(session->acknowledge_result(accepted.receipt), "ACK exact native host result once");
        expect(session->poll().retired == 0, "SDK return and ACK cannot fabricate physical stop");
    }
    void retire() {
        driver.query_status = CUDA_SUCCESS;
        expect(session->poll().retired == 1, "only independently released trailing event retires");
    }
    void close() {
        driver.query_status = CUDA_SUCCESS;
        success(session->shutdown());
        expect(driver.live_resources() == 0 && driver.context_destroys == 1 && !driver.current &&
                   trace.registered.empty() && trace.directional == 0,
               "normal cleanup finishes before the exact callback and lifetime oracle");
    }
};
CudaSessionUvaEndpoint device_endpoint(const CudaSessionResult& allocation) {
    return {.kind = CudaUvaEndpointKind::kDevice, .device = {allocation.id, kOffset}};
}
CudaSessionUvaEndpoint host_endpoint(Owner& owner, const CudaSessionResult& allocation) {
    const auto view = owner.session->host_memory({allocation.id, kOffset}, kCopied);
    expect(view.has_value(), "host endpoint is a checked bounded owner borrow");
    return {.kind = CudaUvaEndpointKind::kHost, .host = *view};
}
void exact_calls_and_full_images() {
    for (const bool is_asynchronous : {false, true})
        for (unsigned int scenario = 0; scenario != 16; ++scenario) {
            const auto direction = scenario % 4;
            const auto host_classes = scenario / 4;
            Owner owner;
            const auto destination = owner.session->allocate(kBytes);
            const auto source = owner.session->allocate(kBytes);
            success(destination);
            success(source);
            const auto host_destination = owner.import((host_classes & 1U) != 0);
            const auto host_source = owner.import((host_classes & 2U) != 0);
            auto destination_view = owner.session->host_memory({host_destination.id, 0}, kBytes);
            auto source_view = owner.session->host_memory({host_source.id, 0}, kBytes);
            expect(destination_view && source_view, "both complete host owners are available");
            std::array<std::byte, kBytes> input{}, output{};
            output.fill(std::byte{0xa7});
            for (std::size_t index = 0; index != input.size(); ++index)
                input[index] = static_cast<std::byte>((index * 19 + 7) & 255);
            success(owner.session->upload({source.id, 0}, input));
            success(owner.session->upload({destination.id, 0}, output));
            std::memcpy(source_view->pointer, input.data(), input.size());
            std::memcpy(destination_view->pointer, output.data(), output.size());
            const bool to_host = (direction & 1U) != 0, from_host = (direction & 2U) != 0;
            CudaSessionUvaCopy copy{
                .destination =
                    to_host ? host_endpoint(owner, host_destination) : device_endpoint(destination),
                .source = from_host ? host_endpoint(owner, host_source) : device_endpoint(source),
                .bytes = kCopied,
                .is_asynchronous = is_asynchronous};
            const auto stream = owner.session->create_stream(CU_STREAM_NON_BLOCKING);
            success(stream);
            if (is_asynchronous) copy.stream = stream.id;
            const auto current = driver.current;
            driver.current = nullptr;
            const auto accepted = owner.accept(copy);
            driver.current = current;
            owner.issue(accepted);
            expect(
                trace.calls.size() == 1 && trace.calls[0].is_asynchronous == is_asynchronous &&
                    trace.calls[0].bytes == kCopied &&
                    trace.calls[0].destination ==
                        (to_host
                             ? reinterpret_cast<CUdeviceptr>(destination_view->pointer) + kOffset
                             : destination.address + kOffset) &&
                    trace.calls[0].source ==
                        (from_host ? reinterpret_cast<CUdeviceptr>(source_view->pointer) + kOffset
                                   : source.address + kOffset) &&
                    (is_asynchronous
                         ? trace.calls[0].stream == reinterpret_cast<CUstream>(&driver.streams[0])
                         : trace.calls[0].stream == nullptr),
                "exact generic ABI retains both endpoint classes and native stream");
            std::copy_n(input.begin() + kOffset, kCopied, output.begin() + kOffset);
            const auto* actual = to_host ? static_cast<std::byte*>(destination_view->pointer)
                                         : device_bytes(destination.address, kBytes);
            expect(std::equal(output.begin(), output.end(), actual),
                   "complete result and both untouched guards match an independent byte oracle");
            expect(std::equal(input.begin(), input.end(),
                              from_host ? static_cast<std::byte*>(source_view->pointer)
                                        : device_bytes(source.address, kBytes)),
                   "generic copy never changes source bytes or guards");
            copy = {};
            destination_view.reset();
            source_view.reset();
            owner.retire();
            success(owner.session->release_host_memory(host_destination.id));
            success(owner.session->release_host_memory(host_source.id));
            success(owner.session->release(destination.id));
            success(owner.session->release(source.id));
            success(owner.session->destroy_stream(stream.id));
            owner.close();
            const auto registrations =
                2U - ((host_classes & 1U) != 0 ? 1U : 0U) - ((host_classes & 2U) != 0 ? 1U : 0U);
            expect(trace.registrations == registrations && trace.maps == registrations * 2 &&
                       trace.unregistrations == registrations,
                   "ordinary registered host import and cleanup retain their exact SDK calls");
        }
}
void dual_pageable_leases_and_quota() {
    Owner owner({.has_registration = false, .host_pages = 2});
    const auto a = owner.import(), b = owner.import();
    expect(owner.session->snapshot().host_memory_bytes == 2 * owner.page,
           "both unregistered mapping extents are charged before publication");
    Descriptor extra(owner.page);
    const auto full = owner.session->import_host_memory(extra.value, kBytes, owner.page, 0, true);
    expect(full.status == CUDA_ERROR_OUT_OF_MEMORY && !full.id &&
               owner.session->snapshot().host_memory_bytes == 2 * owner.page,
           "pageable mapping quota rejection neither admits nor reduces ownership");
    auto destination = host_endpoint(owner, a), source = host_endpoint(owner, b);
    std::weak_ptr<const void> destination_lease = destination.host.lifetime;
    std::weak_ptr<const void> source_lease = source.host.lifetime;
    std::memset(source.host.pointer, 0x5d, kCopied);
    auto accepted = owner.accept({destination, source, kCopied, 0, true});
    destination = {};
    source = {};
    expect(destination_lease.use_count() == 2 && source_lease.use_count() == 2,
           "queued UVA copy retains both independent host mappings");
    owner.issue(accepted);
    const auto release_a = owner.session->release_host_memory(a.id);
    const auto release_b = owner.session->release_host_memory(b.id);
    expect(release_a.disposition == CudaSessionDisposition::kDeferred &&
               release_b.disposition == CudaSessionDisposition::kDeferred &&
               !owner.session->host_memory({a.id, 0}, 1) &&
               !owner.session->host_memory({b.id, 0}, 1) && destination_lease.use_count() == 2 &&
               source_lease.use_count() == 2,
           "retiring maps cannot be borrowed or freed while a physical borrower remains");
    owner.retire();
    expect(destination_lease.use_count() == 1 && source_lease.use_count() == 1,
           "physical retirement alone restores exact idle baseline ownership");
    const auto cleaned_a = owner.session->release_host_memory(a.id);
    const auto cleaned_b = owner.session->release_host_memory(b.id);
    success(cleaned_a);
    success(cleaned_b);
    expect(!cleaned_a.original_status && !cleaned_b.original_status,
           "CPU-only unmap cannot invent an unregister SDK result");
    expect(destination_lease.expired() && source_lease.expired() &&
               owner.session->snapshot().host_memory_bytes == 0 && trace.registrations == 0 &&
               trace.maps == 0 && trace.unregistrations == 0,
           "pageable imports never register, query aliases or unregister, including cleanup");
    const auto reused = owner.import();
    success(owner.session->release_host_memory(reused.id));
    owner.close();
}
void canonical_rejections_and_zero() {
    Owner owner;
    const auto allocation = owner.session->allocate(kBytes);
    success(allocation);
    const auto host = owner.import();
    auto endpoint = host_endpoint(owner, host);
    CudaSessionUvaCopy valid{device_endpoint(allocation), endpoint, kCopied, 0, false};
    auto reject = [&](const CudaSessionUvaCopy& copy) {
        const auto accepted = owner.accept(copy);
        expect(!accepted.accepted() && trace.calls.empty(),
               "malformed endpoints fail before original entry or sequence consumption");
    };
    auto bad = valid;
    bad.source.kind = CudaUvaEndpointKind::kNull;
    reject(bad);
    bad = valid;
    bad.source.device = {allocation.id, 0};
    reject(bad);
    bad = valid;
    bad.source.host.lifetime.reset();
    reject(bad);
    bad = valid;
    bad.source.host.address = 1;
    reject(bad);
    bad = valid;
    bad.source.host.bytes = kCopied - 1;
    reject(bad);
    bad = valid;
    bad.destination.device.offset = kBytes - kCopied + 1;
    reject(bad);
    bad = valid;
    bad.destination.device.allocation = UINT64_MAX;
    reject(bad);
    bad = valid;
    bad.destination.device.offset = std::numeric_limits<std::size_t>::max();
    reject(bad);
    bad = valid;
    bad.destination.kind = static_cast<CudaUvaEndpointKind>(99);
    reject(bad);
    bad = valid;
    bad.stream = 99;
    reject(bad);
    bad = valid;
    bad.destination = bad.source;
    reject(bad);
    bad = valid;
    // Deliberately invalid boundary bits must be rejected without any dereference.
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    bad.source.host.pointer = reinterpret_cast<void*>(UINTPTR_MAX - 1);
    reject(bad);
    expect(
        owner.session->accept_host_transfer(owner.next, {allocation.id, 0}, endpoint.host, 0, true)
                .status == ExecutionAcceptStatus::kInvalid,
        "unregistered backing cannot enter legacy pinned directional transfers");
    const auto zero = owner.accept({});
    owner.issue(zero);
    expect(trace.calls.size() == 1 && !trace.calls[0].bytes && !trace.calls[0].source &&
               !trace.calls[0].destination,
           "canonical zero NULL copy still invokes the actual SDK once");
    owner.retire();
    bad = {};
    valid = {};
    endpoint = {};
    success(owner.session->release_host_memory(host.id));
    success(owner.session->release(allocation.id));
    owner.close();
}
void original_error_and_legacy_bridge() {
    for (const bool is_asynchronous : {false, true}) {
        Owner owner({.uses_proxy = true});
        const auto destination = owner.session->allocate(kBytes);
        const auto source = owner.session->allocate(kBytes);
        success(destination);
        success(source);
        trace.status = CUDA_ERROR_INVALID_VALUE;
        const auto accepted = owner.accept(
            {device_endpoint(destination), device_endpoint(source), kCopied, 0, is_asynchronous});
        owner.issue(accepted);
        const auto before = driver.release_calls;
        expect(owner.session->release(destination.id).disposition ==
                       CudaSessionDisposition::kDeferred &&
                   driver.release_calls == before,
               "a real SDK error does not free its unretired device owner");
        if (is_asynchronous)
            expect(trace.calls[0].stream != nullptr && trace.recorded_streams.size() == 1,
                   "async default uses the owned proxy without a fake synchronous callback");
        else
            expect(trace.calls[0].stream == nullptr && trace.recorded_streams.size() == 2 &&
                       trace.recorded_streams[0] == nullptr &&
                       trace.recorded_streams[1] != nullptr && trace.waited_streams.size() == 1 &&
                       trace.waited_streams[0] == trace.recorded_streams[1],
                   "sync exact generic call is joined from real legacy stream into source proxy");
        owner.retire();
        success(owner.session->release(destination.id));
        success(owner.session->release(source.id));
        owner.close();
    }
    Owner owner({.has_unified = false});
    expect(!owner.accept({}).accepted() && !owner.accept({.is_asynchronous = true}).accepted() &&
               trace.calls.empty(),
           "missing exact generic callback cannot substitute a directional API");
    owner.close();
}
void bounded_queue_rejection_and_reuse() {
    Owner owner;
    const auto first = owner.accept({}), second = owner.accept({});
    expect(first.accepted() && second.accepted() &&
               owner.accept({}).status == ExecutionAcceptStatus::kFull && trace.calls.empty(),
           "bounded generic admission rejects before SDK and does not advance sequence");
    owner.issue(first);
    owner.retire();
    owner.issue(second);
    owner.retire();
    const auto reused = owner.accept({.is_asynchronous = true});
    owner.issue(reused);
    owner.retire();
    owner.close();
    expect(trace.calls.size() == 3 && !trace.calls[0].is_asynchronous &&
               !trace.calls[1].is_asynchronous && trace.calls[2].is_asynchronous,
           "capacity is reused only after exact retirement with no dropped or duplicate copies");
}
void admission_owner_and_closing_status() {
    for (const bool is_asynchronous : {false, true}) {
        Owner owner;
        const auto destination = owner.session->allocate(kBytes);
        const auto source = owner.session->allocate(kBytes);
        success(destination);
        success(source);
        const CudaSessionUvaCopy copy{device_endpoint(destination), device_endpoint(source),
                                      kCopied, 0, is_asynchronous};
        const auto sequence = owner.next;
        ExecutionAcceptance foreign;
        driver.next_context_query_status = CUDA_ERROR_OUT_OF_MEMORY;
        std::thread caller([&] { foreign = owner.session->accept_uva_copy(sequence, copy); });
        caller.join();
        const auto after_foreign = owner.session->snapshot();
        expect(foreign.status == ExecutionAcceptStatus::kQuarantined && !foreign.receipt &&
                   owner.next == sequence && !after_foreign.is_quarantined &&
                   !after_foreign.submissions.outstanding && trace.calls.empty() &&
                   driver.next_context_query_status == CUDA_ERROR_OUT_OF_MEMORY,
               "foreign admission rejects before SDK proof without poisoning the real owner");
        driver.next_context_query_status = CUDA_SUCCESS;
        const auto accepted = owner.accept(copy);
        expect(accepted.receipt.sequence() == sequence,
               "foreign admission cannot consume the owner's exact command sequence");
        owner.issue(accepted);
        owner.retire();
        owner.session->close();
        const auto closed_sequence = owner.next;
        const auto closed = owner.accept(copy);
        const auto after_closed = owner.session->snapshot();
        const bool has_exact_closing_rejection =
            closed.status == ExecutionAcceptStatus::kClosed && !closed.receipt &&
            owner.next == closed_sequence && after_closed.submissions.is_closing &&
            !after_closed.is_quarantined && !after_closed.submissions.outstanding &&
            after_closed.allocations == 2 && trace.calls.size() == 1 && driver.release_calls == 0;
        success(owner.session->release(destination.id));
        success(owner.session->release(source.id));
        owner.close();
        expect(has_exact_closing_rejection && driver.release_calls == 2 && trace.calls.size() == 1,
               "healthy closing rejects as Closed, not quarantine, before exact normal cleanup");
    }
}
void admission_allocation_failure_and_recovery() {
    for (const bool is_asynchronous : {false, true}) {
        Owner owner;
        const auto destination = owner.session->allocate(kBytes);
        const auto source = owner.session->allocate(kBytes);
        success(destination);
        success(source);
        std::array<std::byte, kBytes> input{}, expected{};
        expected.fill(std::byte{0xa6});
        for (std::size_t index = 0; index != input.size(); ++index)
            input[index] = static_cast<std::byte>((index * 23 + 9) & 255);
        success(owner.session->upload({source.id, 0}, input));
        success(owner.session->upload({destination.id, 0}, expected));
        const CudaSessionUvaCopy copy{device_endpoint(destination), device_endpoint(source),
                                      kCopied, 0, is_asynchronous};
        const auto sequence = owner.next;
        const auto injected_before = injected_allocation_failures;
        should_fail_allocation = true;
        const auto failed = owner.accept(copy);
        const bool was_consumed = !std::exchange(should_fail_allocation, false);
        const auto after = owner.session->snapshot();
        expect(was_consumed && injected_allocation_failures == injected_before + 1 &&
                   failed.status == ExecutionAcceptStatus::kOutOfMemory && !failed.receipt &&
                   owner.next == sequence && !after.submissions.outstanding &&
                   !after.submissions.unacknowledged_replies && !after.is_quarantined &&
                   after.allocations == 2 && trace.calls.empty() && driver.release_calls == 0,
               "actual allocation failure publishes no command, reply, native call or retirement");
        expect(std::equal(input.begin(), input.end(), device_bytes(source.address, kBytes)) &&
                   std::equal(expected.begin(), expected.end(),
                              device_bytes(destination.address, kBytes)),
               "rejected admission leaves both complete device byte images unchanged");
        const auto recovered = owner.accept(copy);
        expect(recovered.accepted() && recovered.receipt.sequence() == sequence,
               "one-shot allocation failure leaves the same exact sequence available");
        owner.issue(recovered);
        std::copy_n(input.begin() + kOffset, kCopied, expected.begin() + kOffset);
        expect(std::equal(input.begin(), input.end(), device_bytes(source.address, kBytes)) &&
                   std::equal(expected.begin(), expected.end(),
                              device_bytes(destination.address, kBytes)),
               "recovery executes once with every source, destination and guard byte checked");
        owner.retire();
        success(owner.session->release(destination.id));
        success(owner.session->release(source.id));
        owner.close();
        expect(trace.calls.size() == 1 && driver.release_calls == 2 &&
                   trace.calls.front().is_asynchronous == is_asynchronous,
               "allocation rejection and recovery finish normal cleanup without replay");
    }
}
}  // namespace
int main() {
    try {
        exact_calls_and_full_images();
        dual_pageable_leases_and_quota();
        canonical_rejections_and_zero();
        original_error_and_legacy_bridge();
        bounded_queue_rejection_and_reuse();
        admission_owner_and_closing_status();
        admission_allocation_failure_and_recovery();
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
