#include "glimmer/control/execution_socket.h"

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>
#include <unistd.h>

namespace {
using namespace glimmer::control;
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
std::uint64_t number(std::string_view value) {
    std::uint64_t result{};
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    expect(parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size(),
           "valid trusted process identity");
    return result;
}
constexpr const char* kSource = R"ptx(
.version 7.0
.target sm_70
.address_size 64
.visible .entry bounded_add(.param .u64 output, .param .u64 duration,
                            .param .u32 offset, .param .u32 value) {
 .reg .pred %p;
 .reg .b32 %r<8>;
 .reg .b64 %rd<8>;
 ld.param.u64 %rd1, [output]; ld.param.u64 %rd2, [duration];
 ld.param.u32 %r1, [offset]; ld.param.u32 %r2, [value];
 mov.u64 %rd3, %globaltimer;
WAIT:
 mov.u64 %rd4, %globaltimer; sub.u64 %rd5, %rd4, %rd3;
 setp.lt.u64 %p, %rd5, %rd2; @%p bra WAIT;
 mov.u32 %r3, %ctaid.x; mov.u32 %r4, %ntid.x; mov.u32 %r5, %tid.x;
 mad.lo.u32 %r6, %r3, %r4, %r5; add.u32 %r6, %r6, %r1;
 mul.wide.u32 %rd6, %r6, 4; add.u64 %rd7, %rd1, %rd6;
 atom.global.add.u32 %r7, [%rd7], %r2;
 ret;
}
)ptx";

ExecutionWireReply successful(std::optional<ExecutionWireReply> result, bool original = true) {
    expect(
        result && result->disposition == ExecutionWireDisposition::kComplete && result->status == 0,
        "remote operation must return complete success");
    if (original) expect(result->original_status == 0, "actual SDK result, not admission success");
    return std::move(*result);
}
void run(ExecutionSocketClientConfig config) {
    ExecutionSocketClient client(std::move(config));
    successful(client.open());
    const auto call = [&](ExecutionOpcode operation, ExecutionPayloadWriter input) {
        return successful(client.exchange(operation, input.take()));
    };
    ExecutionPayloadWriter load;
    expect(load.write_string(kSource), "serialize owned PTX source");
    const auto module = call(ExecutionOpcode::kLoadPtx, std::move(load));
    ExecutionPayloadWriter lookup;
    expect(lookup.write_u64(module.id) && lookup.write_string("bounded_add"),
           "serialize function lookup");
    const auto function = call(ExecutionOpcode::kGetFunction, std::move(lookup));
    expect(function.arguments.size() == 4 && function.parameter_bytes == 24,
           "real server-discovered Driver parameter ABI");
    constexpr std::uint32_t workers = 320, elements = workers * 32;
    std::vector<std::uint32_t> output(std::size_t{2} * elements, 0);
    ExecutionPayloadWriter allocate;
    expect(allocate.write_u64(output.size() * sizeof(output[0])), "serialize memory size");
    const auto memory = call(ExecutionOpcode::kAllocate, std::move(allocate));
    ExecutionPayloadWriter upload;
    expect(upload.write_u64(memory.id) && upload.write_u64(0) &&
               upload.write_blob(std::as_bytes(std::span{output})),
           "serialize copied source data");
    call(ExecutionOpcode::kUpload, std::move(upload));
    ExecutionPayloadWriter create_stream;
    expect(create_stream.write_u32(1), "serialize nonblocking stream flag");
    const auto stream = call(ExecutionOpcode::kCreateStream, std::move(create_stream));
    std::vector<std::byte> parameters(static_cast<std::size_t>(function.parameter_bytes));
    const auto set = [&](std::size_t index, const auto& value) {
        expect(function.arguments[index].bytes == sizeof(value), "exact argument type size");
        std::memcpy(parameters.data() + function.arguments[index].offset, &value, sizeof(value));
    };
    constexpr std::uint64_t duration_ns = 500000000;
    const auto launch = [&](std::uint32_t offset, std::uint32_t value) {
        set(0, memory.address);
        set(1, duration_ns);
        set(2, offset);
        set(3, value);
        ExecutionPayloadWriter input;
        expect(input.write_u64(function.id) && input.write_u64(stream.id), "serialize handles");
        for (const auto dimension : {workers, 1U, 1U, 32U, 1U, 1U})
            expect(input.write_u32(dimension), "serialize explicit worker geometry");
        expect(input.write_u32(0) && input.write_blob(parameters), "serialize copied parameters");
        const auto result = call(ExecutionOpcode::kLaunch, std::move(input));
        std::fill(parameters.begin(), parameters.end(), std::byte{0});
        return result.id;
    };
    expect(launch(0, 1) == 1 && launch(elements, 3) == 2,
           "two original launches have distinct nonrecycled sequence IDs");
    const auto pending = successful(client.exchange(ExecutionOpcode::kSnapshot), false);
    ExecutionPayloadReader snapshot(pending.data);
    std::uint64_t outstanding{}, awaiting{};
    expect(snapshot.read_u64(outstanding) && snapshot.read_u64(awaiting) && outstanding == 2 &&
               awaiting == 2,
           "both real original launches replied before first physical completion");
    successful(client.exchange(ExecutionOpcode::kDrain), false);
    ExecutionPayloadWriter download;
    expect(download.write_u64(memory.id) && download.write_u64(0) &&
               download.write_u64(output.size() * sizeof(output[0])),
           "serialize bounded download range");
    const auto downloaded = call(ExecutionOpcode::kDownload, std::move(download));
    expect(downloaded.data.size() == output.size() * sizeof(output[0]), "complete output bytes");
    std::memcpy(output.data(), downloaded.data.data(), downloaded.data.size());
    for (std::size_t index = 0; index < output.size(); ++index)
        expect(output[index] == (index < elements ? 1U : 3U),
               "every native output executes exactly once through a separate process");
    for (const auto& [operation, id] : {std::pair{ExecutionOpcode::kRelease, memory.id},
                                        std::pair{ExecutionOpcode::kDestroyStream, stream.id},
                                        std::pair{ExecutionOpcode::kUnloadModule, module.id}}) {
        ExecutionPayloadWriter input;
        expect(input.write_u64(id), "serialize owned resource release");
        call(operation, std::move(input));
    }
    successful(client.exchange(ExecutionOpcode::kClose));
    expect(!client.exchange(ExecutionOpcode::kSnapshot), "closed session cannot reconnect");
    std::cout << "Cross-process CUDA execution verified: workers=320, elements=" << output.size()
              << "; copied_pipeline=true; actual_sdk_results=true; resource_lifecycle=true; "
                 "public_integration=false; full_device_qualified=false\n";
}
}  // namespace
int main(int argc, char** argv) {
    try {
        expect(argc == 4, "Usage: execution_service_gpu_client SOCKET SERVER_PID SERVER_START");
        const auto pid = number(argv[2]);
        expect(pid > 0 && pid <= UINT32_MAX, "bounded trusted server PID");
        run({argv[1],
             {static_cast<std::uint32_t>(pid), static_cast<std::uint32_t>(::geteuid()),
              number(argv[3])},
             30000});
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
