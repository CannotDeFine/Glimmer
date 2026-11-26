#pragma once

#include "glimmer/control/execution_socket.h"

#include <cuda.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

// Explicit CPU command/frontier model for public-ABI endpoint fixtures. Admission
// never supplies SDK status or completion; each test separately publishes both.
// The actual ExecutionService fixture independently exercises real event gating.
class ExecutionAsyncFixture final {
   public:
    static void tail(glimmer::control::ExecutionPayloadReader& input) {
        std::uint32_t magic{};
        require(input.read_u32(magic) && magic == glimmer::control::kExecutionAsyncLaunchMagic &&
                input.empty());
    }
    // Expected fields follow the public launch wire order, checked one by one.
    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    static void zero_argument_launch(glimmer::control::ExecutionPayloadReader& input,
                                     std::uint64_t expected_function,
                                     std::uint32_t expected_blocks = 320,
                                     std::uint32_t expected_threads = 32) {
        std::uint64_t function{}, stream{};
        require(input.read_u64(function) && function == expected_function &&
                input.read_u64(stream) && stream == 0);
        for (const auto expected :
             std::array{expected_blocks, 1U, 1U, expected_threads, 1U, 1U, 0U}) {
            std::uint32_t dimension{};
            require(input.read_u32(dimension) && dimension == expected);
        }
        std::span<const std::byte> parameters;
        require(input.read_blob(parameters) && parameters.empty());
        tail(input);
    }
    // NOLINTEND(bugprone-easily-swappable-parameters)
    // Scope is first and the within-scope identity second, matching the endpoint.
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    glimmer::control::ExecutionWireReply admit(std::uint64_t connection, std::uint64_t stream) {
        using namespace glimmer::control;
        auto& commands = commands_[connection];
        const auto sequence = ++sequences_[connection];
        auto command = std::make_shared<Command>();
        command->stream = stream;
        commands.emplace(sequence, command);
        const auto frontier = ++next_frontier_;
        frontiers_.emplace(frontier, Frontier{connection, {std::move(command)}, sequence});
        ExecutionWireReply reply;
        reply.disposition = ExecutionWireDisposition::kAdmissionAccepted;
        reply.id = sequence;
        reply.address = frontier;
        return reply;
    }
    void complete(std::uint64_t connection, std::uint64_t sequence, CUresult status,
                  bool is_physically_retired = true) {
        auto& command = *commands_.at(connection).at(sequence);
        command.status = status;
        command.is_physically_retired = is_physically_retired;
    }
    glimmer::control::ExecutionWireReply capture(std::uint64_t connection,
                                                 std::optional<std::uint64_t> stream = {}) {
        using namespace glimmer::control;
        Frontier frontier{connection, {}};
        for (const auto& [sequence, command] : commands_[connection]) {
            (void)sequence;
            if (!stream || command->stream == *stream) frontier.commands.push_back(command);
        }
        ExecutionWireReply reply;
        reply.disposition = ExecutionWireDisposition::kComplete;
        reply.id = ++next_frontier_;
        reply.address = sequences_[connection];
        frontiers_.emplace(reply.id, std::move(frontier));
        return reply;
    }
    glimmer::control::ExecutionWireReply observe(
        const glimmer::control::ExecutionReceivedRequest& request) {
        using namespace glimmer::control;
        if (request.message.header.operation == ExecutionOpcode::kQueryAsyncFrontiers ||
            request.message.header.operation == ExecutionOpcode::kReleaseAsyncFrontiers)
            return observe_batch(request);
        ExecutionPayloadReader input(request.message.payload);
        std::uint64_t id{};
        require(input.read_u64(id) && input.empty());
        const auto found = frontiers_.find(id);
        require(found != frontiers_.end() && found->second.connection == request.connection);
        ExecutionWireReply reply;
        reply.disposition = ExecutionWireDisposition::kComplete;
        if (request.message.header.operation == ExecutionOpcode::kReleaseFrontier) {
            frontiers_.erase(found);
        } else {
            require(request.message.header.operation == ExecutionOpcode::kQueryFrontier);
            bool all_retired = true;
            for (const auto& command : found->second.commands) {
                all_retired = all_retired && command->is_physically_retired;
                if (command->status != CUDA_SUCCESS &&
                    (reply.status == CUDA_SUCCESS || reply.status == CUDA_ERROR_NOT_READY))
                    reply.status = command->status;
            }
            reply.parameter_bytes = all_retired ? 1U : 0U;
            if (!all_retired && reply.status == CUDA_SUCCESS) reply.status = CUDA_ERROR_NOT_READY;
        }
        return reply;
    }
    void close(std::uint64_t connection) {
        require(std::ranges::none_of(frontiers_, [connection](const auto& item) {
            return item.second.connection == connection;
        }));
        for (const auto& [sequence, command] : commands_[connection]) {
            (void)sequence;
            require(command->is_physically_retired);
        }
        commands_.erase(connection);
        sequences_.erase(connection);
    }
    // Both identifiers are validated against the same retained owner before erase.
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    void abandon(std::uint64_t connection, std::uint64_t frontier) {
        const auto found = frontiers_.find(frontier);
        require(found != frontiers_.end() && found->second.connection == connection);
        frontiers_.erase(found);
    }
    [[nodiscard]] bool empty() const {
        return commands_.empty() && frontiers_.empty();
    }

   private:
    static void require(bool value) {
        if (!value)
            throw std::runtime_error("exact asynchronous command/frontier fixture contract");
    }
    struct Command {
        std::uint64_t stream{};
        CUresult status{CUDA_ERROR_NOT_READY};
        bool is_physically_retired{};
    };
    struct Frontier {
        std::uint64_t connection{};
        std::vector<std::shared_ptr<Command>> commands;
        std::uint64_t asynchronous_sequence{};
    };
    glimmer::control::ExecutionWireReply observe_batch(
        const glimmer::control::ExecutionReceivedRequest& request) {
        using namespace glimmer::control;
        const auto keys = decode_execution_async_frontier_batch(request.message.payload);
        require(keys.has_value());
        const bool is_release =
            request.message.header.operation == ExecutionOpcode::kReleaseAsyncFrontiers;
        std::vector<ExecutionAsyncFrontierObservation> observations;
        for (const auto key : *keys) {
            const auto found = frontiers_.find(key.id);
            require(found != frontiers_.end() && found->second.connection == request.connection &&
                    found->second.asynchronous_sequence == key.sequence &&
                    found->second.commands.size() == 1);
            const auto& command = *found->second.commands.front();
            require(!is_release || command.is_physically_retired);
            observations.push_back(
                {key, static_cast<std::int32_t>(command.status), command.is_physically_retired});
        }
        auto data = is_release ? encode_execution_async_frontier_batch(*keys)
                               : encode_execution_async_frontier_observations(observations);
        require(data.has_value());
        if (is_release)
            for (const auto key : *keys) frontiers_.erase(key.id);
        ExecutionWireReply reply;
        reply.disposition = ExecutionWireDisposition::kComplete;
        reply.data = std::move(*data);
        return reply;
    }
    std::uint64_t next_frontier_{1000};
    std::map<std::uint64_t, std::uint64_t> sequences_;
    std::map<std::uint64_t, std::map<std::uint64_t, std::shared_ptr<Command>>> commands_;
    std::map<std::uint64_t, Frontier> frontiers_;
};
