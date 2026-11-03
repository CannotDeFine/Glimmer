#pragma once

#include "glimmer/control/execution_protocol.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace glimmer::control {

inline constexpr std::size_t kExecutionNcclUniqueIdBytes = 128;
inline constexpr std::size_t kExecutionNcclScopeBytes = 40;
inline constexpr std::size_t kExecutionNcclObservationBytes = 172;
inline constexpr std::size_t kExecutionNcclConfigStringLimit = 65535;
inline constexpr std::uint32_t kExecutionNcclReplyMagic = 0x3152434e;  // NCR1
inline constexpr std::uint32_t kExecutionNcclReplyVersion = 1;
inline constexpr std::size_t kExecutionNcclReplyPrefixBytes = 240;
inline constexpr std::size_t kExecutionNcclPublicationBytes = 212;
inline constexpr std::size_t kExecutionNcclPublicationLimit = 256;

// The fixed enum types and values are part of the wire protocol.
// NOLINTNEXTLINE(performance-enum-size)
enum class ExecutionNcclOperation : std::uint32_t {
    kGetUniqueId = 1,
    kInitRank = 2,
    kInitRankConfig = 3,
    kGetAsyncError = 4,
    kDestroy = 5,
    kAbort = 6,
    kGroupStart = 7,
    kGroupEnd = 8,
    kAllGather = 9,
    kAllReduce = 10,
    kBcast = 11,
};
struct ExecutionNcclScope {
    std::array<std::uint64_t, 2> thread{};
    std::uint64_t sequence{}, group{}, context{};
    bool operator==(const ExecutionNcclScope&) const = default;
};
struct ExecutionNcclCommunicator {
    std::uint64_t domain{}, object{};
    bool operator==(const ExecutionNcclCommunicator&) const = default;
};
// NOLINTNEXTLINE(performance-enum-size)
enum class ExecutionNcclStreamKind : std::uint32_t {
    kDefault = 0,
    kOwned = 1,
    kLegacy = 2,
    kPerThread = 3,
};
struct ExecutionNcclStream {
    ExecutionNcclStreamKind kind{};
    std::uint64_t owner{}, object{};
    bool operator==(const ExecutionNcclStream&) const = default;
};
// NOLINTNEXTLINE(performance-enum-size)
enum class ExecutionNcclBufferKind : std::uint32_t { kNull = 0, kOwned = 1, kUnresolved = 2 };
struct ExecutionNcclBuffer {
    ExecutionNcclBufferKind kind{};
    std::uint64_t source{}, allocation{}, offset{};
    bool operator==(const ExecutionNcclBuffer&) const = default;
};
using ExecutionNcclUniqueId = std::array<std::byte, kExecutionNcclUniqueIdBytes>;

// Fixed audited LP64 88-byte native layout, not sizeof/a pointer-bearing blob.
// Fields occupy native offsets16/20/24/28/40/44/56/60/64/68/72/76/80/84.
// Scalars outside min(declared_bytes,88) have canonical zero bytes; partial
// scalar prefixes retain only copied low bytes. declared_bytes itself survives.
// Sizes33..39/49..55 cut a native pointer and reject. Missing pointer fields
// require absent strings. Present strings are bounded, non-NUL opaque bytes;
// NULL and empty differ. This is representation, not SDK ABI qualification.
struct ExecutionNcclConfig {
    std::uint64_t declared_bytes{};
    std::uint32_t magic{}, version{};
    std::array<std::int32_t, 14> fields{};
    std::optional<std::string> net_name, comm_name;
    bool operator==(const ExecutionNcclConfig&) const = default;
};
struct ExecutionNcclGetUniqueId {
    bool has_output{};
    bool operator==(const ExecutionNcclGetUniqueId&) const = default;
};
struct ExecutionNcclInitRank {
    bool has_output{};
    std::int32_t ranks{}, rank{};
    ExecutionNcclUniqueId unique_id{};
    bool operator==(const ExecutionNcclInitRank&) const = default;
};
struct ExecutionNcclInitRankConfig {
    ExecutionNcclInitRank initialization;
    std::optional<ExecutionNcclConfig> config;
    bool operator==(const ExecutionNcclInitRankConfig&) const = default;
};
struct ExecutionNcclGetAsyncError {
    ExecutionNcclCommunicator communicator;
    bool has_output{};
    bool operator==(const ExecutionNcclGetAsyncError&) const = default;
};
struct ExecutionNcclDestroy {
    ExecutionNcclCommunicator communicator;
    bool operator==(const ExecutionNcclDestroy&) const = default;
};
struct ExecutionNcclAbort {
    ExecutionNcclCommunicator communicator;
    bool operator==(const ExecutionNcclAbort&) const = default;
};
struct ExecutionNcclGroupStart {
    bool operator==(const ExecutionNcclGroupStart&) const = default;
};
struct ExecutionNcclGroupEnd {
    bool operator==(const ExecutionNcclGroupEnd&) const = default;
};
struct ExecutionNcclAllGather {
    ExecutionNcclCommunicator communicator;
    ExecutionNcclStream stream;
    ExecutionNcclBuffer send, receive;
    std::uint64_t count{};
    std::int32_t datatype{};
    bool operator==(const ExecutionNcclAllGather&) const = default;
};
struct ExecutionNcclAllReduce {
    ExecutionNcclAllGather collective;
    std::int32_t reduction{};
    bool operator==(const ExecutionNcclAllReduce&) const = default;
};
struct ExecutionNcclBcast {
    ExecutionNcclCommunicator communicator;
    ExecutionNcclStream stream;
    ExecutionNcclBuffer buffer;
    std::uint64_t count{};
    std::int32_t datatype{}, root{};
    bool operator==(const ExecutionNcclBcast&) const = default;
};
using ExecutionNcclCall =
    std::variant<ExecutionNcclGetUniqueId, ExecutionNcclInitRank, ExecutionNcclInitRankConfig,
                 ExecutionNcclGetAsyncError, ExecutionNcclDestroy, ExecutionNcclAbort,
                 ExecutionNcclGroupStart, ExecutionNcclGroupEnd, ExecutionNcclAllGather,
                 ExecutionNcclAllReduce, ExecutionNcclBcast>;
struct ExecutionNcclRequest {
    ExecutionNcclScope scope;
    ExecutionNcclCall call;
    bool operator==(const ExecutionNcclRequest&) const = default;
};

// Effect bits1/2/4 describe UID/communicator/async output writes independently
// of requested native status, queueing, readiness or GPU completion. UID bits
// are little-endian within each mask byte. Inactive value bytes are zero.
// Raw async scalar bytes avoid evaluating an invalid native enum on error.
// A communicator write may be NULL(0/0) or an authorized nonzero token pair.
// The native adapter must independently prove observations and lifetime; this
// codec does not make poison-equal writes detectable or grant native authority.
struct ExecutionNcclObservation {
    std::uint32_t effects{};
    std::array<std::byte, 16> unique_id_mask{};
    ExecutionNcclUniqueId unique_id{};
    ExecutionNcclCommunicator communicator;
    std::uint32_t async_error_mask{};
    std::int32_t async_error{};
    bool operator==(const ExecutionNcclObservation&) const = default;
};

// NOLINTNEXTLINE(performance-enum-size)
enum class ExecutionNcclContextEffectKind : std::uint32_t {
    kNone = 0,
    kSelectPrimary = 1,
    kClearCurrent = 2,
};
struct ExecutionNcclContextEffect {
    ExecutionNcclContextEffectKind kind{};
    std::int32_t device{};
    std::uint64_t domain{};
    bool operator==(const ExecutionNcclContextEffect&) const = default;
};
struct ExecutionNcclPublication {
    ExecutionNcclScope scope;
    ExecutionNcclObservation observation;
    bool operator==(const ExecutionNcclPublication&) const = default;
};
// Representation only: the enclosing actor proves native write effects, retained
// domain authority, request/publication correlation and context side effects.
// A rejected record is otherwise empty. Acceptance requires a nonzero operation,
// but does not require an original status: a carrier failure may precede native
// entry. Present native statuses preserve all signed bits, including errors.
// None/ClearCurrent have zero device/domain; SelectPrimary requires device>=0
// and domain!=0. ClearCurrent describes an actual observed native NULL context,
// not an opcode-derived instruction or a context/resource retirement proof.
// Publication order is preserved; the consumer must authenticate each original
// request/caller/group binding and reject duplicate publications. The codec does
// not infer native write order or turn changed-byte masks into a write-effect proof.
struct ExecutionNcclReply {
    bool is_accepted{};
    std::uint64_t operation{};
    ExecutionNcclCommunicator reserved_communicator;
    std::optional<std::int32_t> original_status;
    ExecutionNcclObservation observation;
    ExecutionNcclContextEffect context_effect;
    std::vector<ExecutionNcclPublication> publications;
    bool operator==(const ExecutionNcclReply&) const = default;
};

// Explicit little-endian operation/reserved0 + scope + operation-specific body.
// Readers permit enclosing composition and preserve cursor/output on failure;
// writers preflight the whole record before changing a previously written prefix.
// Structural checks preserve raw counts/ranks/enums, not access authority or
// native validation precedence. The enclosing owner authenticates every token.
[[nodiscard]] bool write_execution_nccl_request(ExecutionPayloadWriter& output,
                                                const ExecutionNcclRequest& value) noexcept;
[[nodiscard]] bool read_execution_nccl_request(ExecutionPayloadReader& input,
                                               ExecutionNcclRequest& value) noexcept;
[[nodiscard]] bool write_execution_nccl_observation(ExecutionPayloadWriter& output,
                                                    const ExecutionNcclObservation& value) noexcept;
[[nodiscard]] bool read_execution_nccl_observation(ExecutionPayloadReader& input,
                                                   ExecutionNcclObservation& value) noexcept;

// LE magic/version/reserved0/accepted + operation + reserved communicator pair +
// original-present/raw-status + observation + context-kind/device/domain + count,
// followed by count*(full scope + observation). The fixed prefix is240 bytes;
// each publication is212 bytes and count<=256. Absent original status encodes0.
// As with requests, aggregate preflight and transactional enclosing composition
// apply; no public role/opcode or native execution is enabled by this envelope.
[[nodiscard]] bool write_execution_nccl_reply(ExecutionPayloadWriter& output,
                                              const ExecutionNcclReply& value) noexcept;
[[nodiscard]] bool read_execution_nccl_reply(ExecutionPayloadReader& input,
                                             ExecutionNcclReply& value) noexcept;

}  // namespace glimmer::control
