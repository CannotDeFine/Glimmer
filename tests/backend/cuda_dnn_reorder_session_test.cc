#include "dnn_reorder_fixture.h"

#include <iostream>
#include <source_location>
#include <sys/wait.h>

namespace {
using namespace glimmer::backend::cuda;
namespace fixture = glimmer::tests::dnn_reorder;
namespace foundation = fixture::driver;
namespace base = fixture::base;
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void success(const CudaSessionResult& result,
             std::source_location where = std::source_location::current()) {
    if (result.disposition != CudaSessionDisposition::kComplete || result.status != CUDA_SUCCESS)
        throw std::runtime_error("reorder Driver failure at " + std::to_string(where.line()));
}
void success(const CudaDnnResult& result,
             std::source_location where = std::source_location::current()) {
    if (result.disposition != CudaDnnDisposition::kComplete || result.status != CUDA_SUCCESS ||
        result.original_status != CUDNN_STATUS_SUCCESS)
        throw std::runtime_error("reorder SDK failure at " + std::to_string(where.line()));
}
enum class Layout { kSeparate, kDisjoint, kReadAlias };
struct Owner {
    std::unique_ptr<CudaDnnRegistry> metadata;
    std::unique_ptr<CudaExecutionSession> session;
    std::uint64_t descriptor{}, handle{};
    CudaDnnFilter filter;
    std::uint32_t reorder{};
    int flag{};
    std::vector<CudaSessionResult> allocations;
    std::vector<std::vector<std::byte>> images;
    std::array<std::size_t, 4> extents{}, slots{}, offsets{};
    std::array<CudaAllocationRange, 4> buffers{};
    Owner(std::uint32_t type, std::uint32_t mode, int bias, Layout layout = Layout::kSeparate,
          bool explicit_stream = false, unsigned int missing = 0)
        : filter(fixture::filter(type)), reorder(mode), flag(bias) {
        fixture::reset();
        auto sdk = fixture::session_dispatch();
        auto dnn = *sdk.dnn;
        if (missing == 1) dnn.version = nullptr;
        if (missing == 2) dnn.reorder_filter_bias = nullptr;
        if (missing == 3) dnn.filter_get_nd = nullptr;
        if (missing == 4) dnn.filter_size = nullptr;
        sdk.dnn = std::make_shared<const CudaDnnDispatch>(dnn);
        metadata = std::make_unique<CudaDnnRegistry>(dnn);
        const auto created = metadata->create(CudaDnnObjectKind::kFilter);
        success(created);
        descriptor = created.id;
        success(metadata->set_filter(descriptor, filter));
        session = std::make_unique<CudaExecutionSession>(
            glimmer::backend::ExecutionSessionIdentity{51, 52}, 53, sdk);
        success(session->initialize(0));
        const auto native = session->create_dnn();
        success(native);
        handle = native.id;
        if (explicit_stream) {
            const auto stream = session->create_stream(CU_STREAM_NON_BLOCKING);
            success(stream);
            success(session->set_dnn_stream(handle, stream.id));
        }
        extents = {fixture::filter_bytes(filter), fixture::filter_bytes(filter),
                   flag ? fixture::bias_bytes(filter, mode) : 0,
                   flag ? fixture::bias_bytes(filter, mode) : 0};
        std::size_t combined{};
        for (std::size_t i = 0; i != 4; ++i) {
            if (!extents[i]) continue;
            if (layout == Layout::kReadAlias && i == 2) {
                slots[i] = slots[0];
                offsets[i] = offsets[0];
                expect(extents[i] <= extents[0], "read-only alias fits original input");
                continue;
            }
            if (layout == Layout::kDisjoint) {
                slots[i] = 0;
                offsets[i] = combined + 16;
                combined += extents[i] + 32;
            } else {
                slots[i] = images.size();
                offsets[i] = 16;
                images.emplace_back(extents[i] + 32, std::byte{0xcd});
            }
        }
        if (layout == Layout::kDisjoint) images.emplace_back(combined, std::byte{0xcd});
        for (std::size_t i : {0U, 2U}) {
            if (!extents[i]) continue;
            if (layout == Layout::kReadAlias && i == 2) continue;
            for (std::size_t j = 0; j != extents[i]; ++j)
                images[slots[i]][offsets[i] + j] =
                    static_cast<std::byte>((j * 37 + i * 19 + 11) & 255);
        }
        for (const auto& image : images) {
            const auto allocation = session->allocate(image.size());
            success(allocation);
            allocations.push_back(allocation);
            success(session->upload({allocation.id, 0}, image));
        }
        for (std::size_t i = 0; i != 4; ++i)
            if (extents[i]) buffers[i] = {allocations[slots[i]].id, offsets[i]};
    }
    CudaSessionDnnReorder problem() {
        return {handle, *metadata->borrow(descriptor, CudaDnnObjectKind::kFilter), reorder, flag,
                buffers};
    }
    void retained() {
        expect(
            metadata->set_filter(descriptor, filter).disposition == CudaDnnDisposition::kDeferred &&
                metadata->destroy(descriptor).disposition == CudaDnnDisposition::kDeferred &&
                session->destroy_dnn(handle).disposition == CudaDnnDisposition::kDeferred,
            "queued/pending reorder retains immutable filter and owned handle");
        for (const auto& allocation : allocations)
            expect(session->release(allocation.id).disposition == CudaSessionDisposition::kDeferred,
                   "every active allocation remains owned through physical retirement");
    }
    void compare(bool invoked, bool failed) {
        auto expected = images;
        if (invoked) {
            for (std::size_t input : {0U, 2U}) {
                if (!extents[input] || (failed && input == 2)) continue;
                std::vector<std::byte> bytes(
                    images[slots[input]].begin() + static_cast<std::ptrdiff_t>(offsets[input]),
                    images[slots[input]].begin() +
                        static_cast<std::ptrdiff_t>(offsets[input] + extents[input]));
                if (!reorder) std::reverse(bytes.begin(), bytes.end());
                std::copy(bytes.begin(), bytes.end(),
                          expected[slots[input + 1]].begin() +
                              static_cast<std::ptrdiff_t>(offsets[input + 1]));
            }
        }
        for (std::size_t i = 0; i != allocations.size(); ++i)
            expect(foundation::driver.memory.at(allocations[i].address).bytes == expected[i],
                   "complete synthetic output, immutable inputs and every guard byte match");
    }
    void finish() {
        for (const auto& allocation : allocations) success(session->release(allocation.id));
        success(metadata->destroy(descriptor));
        success(metadata->shutdown());
        success(session->shutdown());
        expect(fixture::state.creates == 1 && fixture::state.destroys == 1 &&
                   base::state.handle_creates == 1 && base::state.handle_destroys == 1 &&
                   foundation::driver.frees == allocations.size() &&
                   foundation::driver.memory.empty() && foundation::driver.contexts.empty() &&
                   foundation::driver.streams.empty() && foundation::driver.events.empty() &&
                   !fixture::state.four_queries && !foundation::control.context_synchronizes.load(),
               "all original ownership retires once, without Filter4d or forced context drain");
    }
};
void execute(std::uint32_t type, std::uint32_t mode, int flag, Layout layout, bool explicit_stream,
             bool failed = false, bool cancel = false) {
    Owner owner(type, mode, flag, layout, explicit_stream);
    auto problem = owner.problem();
    const auto accepted = owner.session->accept_dnn_reorder(1, problem);
    expect(accepted.acceptance.accepted() && !fixture::state.calls && fixture::state.nd_queries &&
               fixture::state.size_queries && fixture::state.version_queries,
           "admission uses original Nd/size/version but invokes no original reorder");
    problem = {};
    owner.retained();
    if (cancel) {
        expect(owner.session->reject_queued(CUDA_ERROR_INVALID_VALUE) == 1,
               "cancel precisely one queued reorder");
        const auto host = owner.session->host_result(accepted.acceptance.receipt);
        expect(host &&
                   host->origin == glimmer::backend::ExecutionReplyOrigin::kRejectedBeforeIssue &&
                   host->status == CUDA_ERROR_INVALID_VALUE && !fixture::state.calls &&
                   owner.session->acknowledge_result(accepted.acceptance.receipt),
               "preissue cancellation publishes no fabricated native result");
    } else {
        foundation::control.can_stop.store(false);
        fixture::control.compute_status.store(failed ? CUDNN_STATUS_EXECUTION_FAILED
                                                     : CUDNN_STATUS_SUCCESS);
        const auto issued = owner.session->issue_next();
        const auto host = owner.session->host_result(accepted.acceptance.receipt);
        const auto status = failed ? CUDNN_STATUS_EXECUTION_FAILED : CUDNN_STATUS_SUCCESS;
        expect(
            issued.state == CudaExecutionIssueState::kIssued && issued.library_status == status &&
                host && host->origin == glimmer::backend::ExecutionReplyOrigin::kNativeSdk &&
                host->status == status && fixture::state.calls == 1 &&
                fixture::state.last.reorder_bias == flag && fixture::state.last.reorder == mode &&
                (explicit_stream ? fixture::state.last.stream != nullptr
                                 : fixture::state.last.stream == nullptr),
            "original invocation receives exact mode/signed flag/stream and returns exact status");
        expect(owner.session->acknowledge_result(accepted.acceptance.receipt) &&
                   !owner.session->poll().retired,
               "host status and ACK cannot retire successful or partially failed work");
        owner.retained();
        success(owner.metadata->get_filter(owner.descriptor, 8));
        success(owner.metadata->filter_size(owner.descriptor));
        expect(!foundation::control.can_stop.load() && !owner.session->poll().retired &&
                   fixture::state.calls == 1,
               "read-only metadata queries do not drain or repeat pending computation");
        foundation::control.can_stop.store(true);
        expect(owner.session->poll().retired == 1, "one trailing physical event retires one call");
    }
    owner.compare(!cancel, failed);
    owner.finish();
}
void rejected(unsigned int mode) {
    const bool inactive = mode == 17 || mode == 18;
    Owner owner(8, mode == 16 ? 0 : 1, inactive ? 0 : -1, Layout::kDisjoint, false,
                mode <= 4 ? mode : 0);
    auto problem = owner.problem();
    if (mode >= 5 && mode <= 7)
        fixture::control.version.store(mode == 5 ? 0 : mode == 6 ? 91001 : 92301);
    if (mode >= 8 && mode <= 11) {
        const auto i = mode - 8;
        problem.buffers[i].offset = owner.images[0].size() - owner.extents[i] + 1;
    }
    if (mode == 12) problem.buffers[1] = problem.buffers[0];
    if (mode == 13) problem.buffers[3] = problem.buffers[2];
    if (mode == 14) problem.buffers[3] = problem.buffers[0];
    if (mode == 15) problem.buffers[1] = problem.buffers[2];
    if (mode == 16) problem.buffers[3] = problem.buffers[1];
    if (mode == 17) problem.buffers[2] = problem.buffers[0];
    if (mode == 18) problem.buffers[3].offset = 1;
    if (mode == 19) problem.buffers[2] = {};
    if (mode == 20) problem.buffers[3] = {};
    if (mode == 21) problem.reorder = 2;
    if (mode == 22) problem.filter = {};
    if (mode == 23) fixture::control.reported_type.store(17);
    if (mode == 24) fixture::control.reported_size.store(std::numeric_limits<std::size_t>::max());
    const auto accepted = owner.session->accept_dnn_reorder(1, problem);
    expect(!accepted.acceptance.accepted() && !fixture::state.calls,
           "invalid capability, complete range, shape or active write overlap rejects before SDK");
    if (mode <= 7)
        expect(accepted.rejection_status == CUDA_ERROR_NOT_SUPPORTED && !accepted.original_status &&
                   !fixture::state.nd_queries && !fixture::state.size_queries,
               "missing/unqualified capability reports NOT_SUPPORTED before metadata queries");
    problem = {};
    owner.compare(false, false);
    owner.finish();
}
void original_errors(bool size_error) {
    Owner owner(8, 0, -1);
    (size_error ? fixture::control.size_status : fixture::control.query_status)
        .store(CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE);
    auto problem = owner.problem();
    const auto accepted = owner.session->accept_dnn_reorder(1, problem);
    expect(!accepted.acceptance.accepted() &&
               accepted.original_status == CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE &&
               !fixture::state.calls && !owner.session->snapshot().is_quarantined,
           "original metadata query failure retains exact status without issuing compute");
    problem = {};
    owner.finish();
}
void native_unsupported() {
    Owner owner(CUDNN_DATA_FLOAT, 0, 1);
    auto problem = owner.problem();
    const auto accepted = owner.session->accept_dnn_reorder(1, problem);
    expect(accepted.acceptance.accepted(),
           "safe unsupported DEFAULT type reaches native validation");
    problem = {};
    const auto issued = owner.session->issue_next();
    const auto host = owner.session->host_result(accepted.acceptance.receipt);
    expect(issued.library_status == CUDNN_STATUS_NOT_SUPPORTED && host &&
               host->status == CUDNN_STATUS_NOT_SUPPORTED &&
               owner.session->acknowledge_result(accepted.acceptance.receipt),
           "native unsupported combination is not replaced by admission success");
    expect(owner.session->poll().retired == 1,
           "unsupported native call still has physical frontier");
    owner.compare(false, false);
    owner.finish();
}
}  // namespace
int main() {
    try {
        for (std::uint32_t mode : {0U, 1U})
            for (int flag : {0, 1, -1, INT_MIN, INT_MAX})
                execute(8, mode, flag, Layout::kSeparate, flag != 0);
        for (std::uint32_t type = 0; type != 17; ++type)
            execute(type, 1, -1, Layout::kDisjoint, true);
        execute(8, 0, -1, Layout::kReadAlias, false);
        execute(8, 1, INT_MIN, Layout::kReadAlias, true);
        execute(8, 0, -1, Layout::kDisjoint, true, true);
        execute(8, 1, 1, Layout::kSeparate, false, true);
        execute(8, 0, 1, Layout::kSeparate, false, false, true);
        for (unsigned int mode = 1; mode != 25; ++mode) rejected(mode);
        original_errors(false);
        original_errors(true);
        native_unsupported();
        std::cout << "Reorder synthetic CPU byte and ownership checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        foundation::control.can_stop.store(true);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
