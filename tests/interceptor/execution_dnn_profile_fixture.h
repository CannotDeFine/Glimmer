#pragma once

#include "glimmer/control/execution_dnn_protocol.h"
#include "glimmer/control/execution_socket.h"

#include <cudnn.h>

#include <array>
#include <cstdint>
#include <map>
#include <stdexcept>

namespace glimmer::tests {

// Test-only setup for existing authenticated fake endpoints. The caller retains
// its original peer/Hello checks, semantic dispatch and malformed-reply oracle.
// Binding is accounted separately and cannot consume an old semantic request.
class ExecutionDnnProfileFixture final {
   public:
    [[nodiscard]] bool respond(control::ExecutionSocketServer& server,
                               const control::ExecutionReceivedRequest& request) {
        using namespace control;
        const auto operation = request.message.header.operation;
        if (operation == ExecutionOpcode::kHello) {
            require(channels_.try_emplace(request.connection).second,
                    "profile fixture observes exactly one Hello per connection");
            return false;
        }
        if (operation == ExecutionOpcode::kBindDnnProfile) {
            const auto found = channels_.find(request.connection);
            require(found != channels_.end(), "profile binding follows its own Hello");
            auto& channel = found->second;
            require(!channel.bindings && !channel.has_dnn && request.descriptor.get() < 0 &&
                        request.message.header.session == request.connection &&
                        request.message.header.incarnation[0] &&
                        request.message.header.incarnation[1],
                    "profile binds once on the exact retained channel without a descriptor");
            ExecutionPayloadReader input(request.message.payload);
            std::uint32_t version{};
            require(read_execution_dnn_profile(input, version) && version == CUDNN_VERSION,
                    "profile binding carries exactly the selected SDK header version");
            ExecutionPayloadWriter output;
            require(write_execution_dnn_profile(output, version), "encode canonical profile echo");
            ExecutionWireReply reply;
            reply.disposition = ExecutionWireDisposition::kComplete;
            reply.data = output.take();
            const auto encoded = encode_execution_reply(reply);
            require(encoded.has_value(), "encode SDK-free profile binding receipt");
            auto header = request.message.header;
            header.kind = ExecutionMessageKind::kReply;
            require(server.reply(request.connection, {header, *encoded}),
                    "deliver binding echo on its original authenticated channel");
            ++channel.bindings;
            channel.incarnation = header.incarnation;
            return true;
        }
        if (execution_opcode_is_dnn(operation)) {
            const auto found = channels_.find(request.connection);
            require(found != channels_.end() && found->second.bindings == 1 &&
                        found->second.incarnation == request.message.header.incarnation &&
                        request.message.header.session == request.connection,
                    "every semantic DNN request follows exactly one same-epoch binding");
            found->second.has_dnn = true;
        }
        return false;
    }

    void verify() const {
        unsigned int bound_channels{};
        for (const auto& [connection, channel] : channels_) {
            static_cast<void>(connection);
            if (!channel.bindings && !channel.has_dnn) continue;
            require(channel.bindings == 1 && channel.has_dnn,
                    "each used DNN channel has exactly one independent binding");
            ++bound_channels;
        }
        require(bound_channels != 0, "fake DNN endpoint exercised profile binding");
    }

   private:
    struct Channel {
        unsigned int bindings{};
        bool has_dnn{};
        std::array<std::uint64_t, 2> incarnation{};
    };
    static void require(bool condition, const char* message) {
        if (!condition) throw std::runtime_error(message);
    }
    std::map<std::uint64_t, Channel> channels_;
};

}  // namespace glimmer::tests
