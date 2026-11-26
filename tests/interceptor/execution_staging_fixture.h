#pragma once

#include "glimmer/control/execution_socket.h"

#include <cstring>
#include <map>
#include <optional>
#include <span>
#include <set>
#include <stdexcept>
#include <sys/mman.h>

// Shared CPU backing for legacy public-ABI fixtures. This is a fake SDK, not
// a native completion oracle; the service fixture tests actual event gating.
class ExecutionStagingFixture final {
   public:
    ExecutionStagingFixture(std::span<std::byte> device, std::uint64_t allocation)
        : device_(device), allocation_(allocation) {}
    ~ExecutionStagingFixture() {
        // Fake SDK CPU mappings only; no CUDA registration or GPU ownership.
        for (const auto& [id, mapping] : mappings_) {
            (void)id;
            if (::munmap(mapping.pointer, mapping.extent) != 0) std::terminate();
        }
    }
    ExecutionStagingFixture(const ExecutionStagingFixture&) = delete;
    ExecutionStagingFixture& operator=(const ExecutionStagingFixture&) = delete;
    void add_device(std::uint64_t id, std::span<std::byte> device) {
        require(id != allocation_ && !devices_.contains(id));
        devices_.emplace(id, device);
    }
    void allow_stream(std::uint64_t id) {
        streams_.insert(id);
    }
    std::optional<glimmer::control::ExecutionWireReply> handle(
        const glimmer::control::ExecutionReceivedRequest& request) {
        using namespace glimmer::control;
        ExecutionPayloadReader input(request.message.payload);
        ExecutionWireReply reply;
        reply.disposition = ExecutionWireDisposition::kComplete;
        reply.original_status = 0;
        const auto operation = request.message.header.operation;
        std::uint64_t id{}, bytes{}, extent{};
        std::uint32_t flags{};
        if (operation == ExecutionOpcode::kHello && request.message.payload.size() == 8) {
            require(input.read_u32(flags) && flags == kExecutionStagingOwnerRole &&
                    input.read_u32(flags) && flags == 8 && input.empty());
            const auto ownership =
                encode_execution_bootstrap_ownership(ExecutionBootstrapOwnership::kRetained);
            reply.data.assign(ownership->begin(), ownership->end());
        } else if (operation == ExecutionOpcode::kImportHostMemory) {
            require(input.read_u64(bytes) && input.read_u64(extent) && input.read_u32(flags) &&
                    !flags && input.empty() && bytes && bytes <= extent && extent == 4096);
            void* mapping = ::mmap(nullptr, static_cast<std::size_t>(extent),
                                   PROT_READ | PROT_WRITE, MAP_SHARED, request.descriptor.get(), 0);
            require(mapping != MAP_FAILED);
            reply.id = ++next_;
            reply.address = reinterpret_cast<std::uintptr_t>(mapping);
            mappings_.emplace(reply.id, Mapping{mapping, static_cast<std::size_t>(extent),
                                                static_cast<std::size_t>(bytes)});
            ExecutionPayloadWriter ownership;
            require(ownership.write_u32(0));
            reply.data = ownership.take();
        } else if (operation == ExecutionOpcode::kQueryStagingIdle) {
            require(input.read_u64(id) && input.empty() && mappings_.contains(id));
            reply.id = id;
            reply.original_status.reset();
        } else if (operation == ExecutionOpcode::kReleaseHostMemory) {
            require(input.read_u64(id) && input.empty() && mappings_.contains(id));
            require(::munmap(mappings_.at(id).pointer, mappings_.at(id).extent) == 0);
            mappings_.erase(id);
        } else if (operation == ExecutionOpcode::kStagedHostTransfer) {
            std::uint64_t owner{}, host_offset{}, allocation{}, device_offset{}, stream{};
            std::uint32_t upload{};
            require(input.read_u64(owner) && owner && input.read_u64(id) &&
                    mappings_.contains(id) && input.read_u64(host_offset) && !host_offset &&
                    input.read_u64(allocation) &&
                    (allocation == allocation_ || devices_.contains(allocation)) &&
                    input.read_u64(device_offset) && input.read_u64(bytes) &&
                    input.read_u64(stream) && (!stream || streams_.contains(stream)) &&
                    input.read_u32(upload) && upload <= 1 && input.empty());
            const auto selected = allocation == allocation_ ? device_ : devices_.at(allocation);
            require(device_offset <= selected.size() && bytes <= selected.size() - device_offset &&
                    bytes <= mappings_.at(id).bytes);
            auto* host = mappings_.at(id).pointer;
            auto* device = selected.data() + device_offset;
            if (upload)
                std::memcpy(device, host, static_cast<std::size_t>(bytes));
            else
                std::memcpy(host, device, static_cast<std::size_t>(bytes));
            reply.id = 1;
        } else
            return std::nullopt;
        return reply;
    }
    bool empty() const noexcept {
        return mappings_.empty();
    }

   private:
    struct Mapping {
        void* pointer{};
        std::size_t extent{}, bytes{};
    };
    static void require(bool valid) {
        if (!valid) throw std::runtime_error("invalid owned staging fixture operation");
    }
    std::span<std::byte> device_;
    std::uint64_t allocation_, next_{};
    std::map<std::uint64_t, Mapping> mappings_;
    std::map<std::uint64_t, std::span<std::byte>> devices_;
    std::set<std::uint64_t> streams_;
};
