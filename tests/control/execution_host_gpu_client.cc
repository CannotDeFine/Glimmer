#include "glimmer/control/execution_socket.h"

#include <array>
#include <charconv>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace {
using namespace glimmer::control;
constexpr std::size_t kCount = std::size_t{320} * 32, kBytes = kCount * sizeof(std::uint32_t);
constexpr std::string_view kPtx = R"ptx(
.version 7.0
.target sm_70
.address_size 64
.visible .entry increment(.param .u64 data, .param .u32 count) {
 .reg .b32 %r<6>;
 .reg .b64 %rd<4>;
 .reg .pred %p;
 ld.param.u64 %rd0, [data];
 ld.param.u32 %r0, [count];
 mov.u32 %r1, %ctaid.x;
 mov.u32 %r2, %ntid.x;
 mov.u32 %r3, %tid.x;
 mad.lo.u32 %r4, %r1, %r2, %r3;
 setp.ge.u32 %p, %r4, %r0;
 @%p bra done;
 mul.wide.u32 %rd1, %r4, 4;
 add.u64 %rd2, %rd0, %rd1;
 ld.global.u32 %r5, [%rd2];
 add.u32 %r5, %r5, 1;
 st.global.u32 [%rd2], %r5;
done:
 ret;
}
)ptx";
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
std::uint64_t number(std::string_view value) {
    std::uint64_t result{};
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    expect(parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size(),
           "valid identity number");
    return result;
}
ExecutionWireReply complete(std::optional<ExecutionWireReply> result, bool is_native = true) {
    if (!result || result->status || result->disposition != ExecutionWireDisposition::kComplete ||
        (is_native && result->original_status != 0))
        throw std::runtime_error("host GPU operation failed: status=" +
                                 std::to_string(result ? result->status : -1));
    return std::move(*result);
}
std::vector<std::byte> id_payload(std::uint64_t id) {
    ExecutionPayloadWriter writer;
    expect(writer.write_u64(id), "encode owned resource identity");
    return writer.take();
}
void run(const std::string& path, TaskPeerIdentity identity, bool should_close_owner) {
    ExecutionSocketClient app({path, identity, 10000});
    ExecutionPayloadWriter app_flags;
    expect(app_flags.write_u32(8), "request mapped-host application context");
    complete(app.open(app_flags.take()));
    ExecutionSocketClient host({path, identity, 10000});
    ExecutionPayloadWriter role;
    expect(role.write_u32(kExecutionHostOwnerRole) && role.write_u32(8), "encode host owner role");
    complete(host.open(role.take()));
    ExecutionOwnedDescriptor descriptor(
        ::memfd_create("glimmer-host-gpu", MFD_ALLOW_SEALING | MFD_CLOEXEC));
    expect(
        descriptor.get() >= 0 && ::ftruncate(descriptor.get(), kBytes) == 0 &&
            ::fcntl(descriptor.get(), F_ADD_SEALS, F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL) == 0,
        "create bounded sealed shared host storage");
    ExecutionPayloadWriter import;
    expect(import.write_u64(kBytes) && import.write_u64(kBytes) && import.write_u32(3),
           "encode import");
    const auto resource = complete(
        host.exchange(ExecutionOpcode::kImportHostMemory, import.take(), descriptor.get()));
    expect(resource.address && resource.address <= std::numeric_limits<std::uintptr_t>::max(),
           "valid native mapped address");
    // Checked native UVA address, never a destructive MAP_FIXED replacement.
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    void* desired = reinterpret_cast<void*>(static_cast<std::uintptr_t>(resource.address));
    void* mapping = ::mmap(desired, kBytes, PROT_READ | PROT_WRITE,
                           MAP_SHARED | MAP_FIXED_NOREPLACE, descriptor.get(), 0);
    expect(mapping != MAP_FAILED && reinterpret_cast<std::uintptr_t>(mapping) == resource.address,
           "non-destructive shared UVA mapping must preserve original argument bits");
    auto* values = static_cast<std::uint32_t*>(mapping);
    for (std::size_t index = 0; index < kCount; ++index) values[index] = 37;
    ExecutionPayloadWriter source;
    expect(source.write_string(kPtx), "upload original unmodified PTX");
    const auto module = complete(app.exchange(ExecutionOpcode::kLoadPtx, source.take()));
    ExecutionPayloadWriter lookup;
    expect(lookup.write_u64(module.id) && lookup.write_string("increment"),
           "lookup original kernel");
    const auto function = complete(app.exchange(ExecutionOpcode::kGetFunction, lookup.take()));
    expect(function.parameter_bytes == 12 && function.arguments.size() == 2,
           "original parameter ABI only");
    const auto first = complete(app.exchange(ExecutionOpcode::kAllocate, id_payload(kBytes)));
    const auto second = complete(app.exchange(ExecutionOpcode::kAllocate, id_payload(kBytes)));
    ExecutionPayloadWriter stream_flags;
    expect(stream_flags.write_u32(1), "explicit independent nonblocking stream");
    const auto stream = complete(app.exchange(ExecutionOpcode::kCreateStream, stream_flags.take()));
    const auto transfer = [&](std::uint64_t allocation, bool is_upload) {
        ExecutionPayloadWriter input;
        expect(input.write_u64(host.session()) && input.write_u64(resource.id) &&
                   input.write_u64(0) && input.write_u64(allocation) && input.write_u64(0) &&
                   input.write_u64(kBytes) && input.write_u64(stream.id) &&
                   input.write_u32(is_upload ? 1 : 0),
               "encode owned async transfer");
        complete(app.exchange(ExecutionOpcode::kHostTransfer, input.take()));
    };
    const auto launch = [&](std::uint64_t pointer) {
        std::array<std::byte, 12> parameters{};
        const auto count = static_cast<std::uint32_t>(kCount);
        std::memcpy(parameters.data(), &pointer, sizeof(pointer));
        std::memcpy(parameters.data() + sizeof(pointer), &count, sizeof(count));
        ExecutionPayloadWriter input;
        expect(input.write_u64(function.id) && input.write_u64(stream.id) && input.write_u32(320) &&
                   input.write_u32(1) && input.write_u32(1) && input.write_u32(32) &&
                   input.write_u32(1) && input.write_u32(1) && input.write_u32(0) &&
                   input.write_blob(parameters),
               "original explicit full-card geometry and ABI");
        complete(app.exchange(ExecutionOpcode::kLaunch, input.take()));
    };
    transfer(first.id, true);
    launch(first.address);
    transfer(first.id, false);
    transfer(second.id, true);
    launch(second.address);
    transfer(second.id, false);
    launch(resource.address);
    if (should_close_owner)
        complete(host.exchange(ExecutionOpcode::kClose), false);
    else {
        complete(host.exchange(ExecutionOpcode::kReleaseHostMemory, id_payload(resource.id)));
        complete(host.exchange(ExecutionOpcode::kClose), false);
    }
    for (std::size_t index = 0; index < kCount; ++index)
        expect(values[index] == 40,
               "every host/device/alias/mapped-kernel result must match exactly");
    complete(app.exchange(ExecutionOpcode::kRelease, id_payload(first.id)));
    complete(app.exchange(ExecutionOpcode::kRelease, id_payload(second.id)));
    complete(app.exchange(ExecutionOpcode::kDestroyStream, id_payload(stream.id)));
    complete(app.exchange(ExecutionOpcode::kUnloadModule, id_payload(module.id)));
    complete(app.exchange(ExecutionOpcode::kClose));
    expect(::munmap(mapping, kBytes) == 0,
           "retire independent application mapping after native release");
    std::cout
        << "Shared host GPU dataflow verified: values=" << kCount
        << "; workers=320; original_mapped_argument=true; close_owner=" << should_close_owner
        << "; public_integration=false; full_device_qualified=false; performance_qualified=false\n";
}
}  // namespace
int main(int argc, char** argv) {
    try {
        expect(argc == 6, "usage: client SOCKET PID UID START CLOSE_OWNER");
        run(argv[1],
            {static_cast<std::uint32_t>(number(argv[2])),
             static_cast<std::uint32_t>(number(argv[3])), number(argv[4])},
            number(argv[5]) != 0);
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
