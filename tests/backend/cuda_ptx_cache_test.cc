#include "internal/cuda_ptx_cache.h"
#include "../transform/device_call_ptx_fixture.h"
#include "../transform/computation_ptx_fixture.h"
#include "../transform/uniform_loop_ptx_fixture.h"

#include <array>
#include <cstdlib>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

namespace {
using namespace glimmer::backend::cuda;
using glimmer::transform::PtxExecutionKind;
using glimmer::transform::PtxTransformLimits;
constexpr auto kKind = PtxExecutionKind::kPreemptive;
constexpr const char* kPtx =
    ".version 7.0\n.target sm_70\n.address_size 64\n"
    ".visible .entry kernel() { .reg .u32 %r; mov.u32 %r, 1; ret; }\n";
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
std::size_t collision(std::string_view) noexcept {
    return 0;
}
std::string other(char value) {
    std::string result{kPtx};
    result[result.find(", 1;") + 2] = value;
    return result;
}
void exact_immutable_identity_and_kind() {
    CudaPtxCache cache({}, collision);
    std::string input{kPtx};
    const auto first = cache.lookup(input, kKind);
    expect(first.status == CudaPtxCacheStatus::kReady && first.image && !first.was_hit,
           "initial source creates an immutable transformed artifact");
    const auto image = first.image->image;
    input = other('2');
    const auto second = cache.lookup(input, kKind);
    expect(second.image && second.image != first.image && second.image->image != image &&
               first.image->image == image,
           "same-length forced hash collision and caller mutation cannot change or alias code "
           "identity");
    expect(cache.lookup(kPtx, kKind).image == first.image &&
               cache.lookup(input, kKind).image == second.image,
           "full byte identities survive a shared hash discriminator");
    for (const auto kind : std::array{PtxExecutionKind::kStatic, PtxExecutionKind::kBlockStatic,
                                      PtxExecutionKind::kDynamic, PtxExecutionKind::kSliced}) {
        const auto variant = cache.lookup(kPtx, kind);
        expect(variant.image && variant.image != first.image &&
                   variant.image->kernels.front().execution_kind == kind &&
                   cache.lookup(kPtx, kind).image == variant.image,
               "execution kind separates every generator contract despite forced hash collision");
    }
    const auto snapshot = cache.snapshot();
    expect(snapshot.is_valid && snapshot.entries == 6 && snapshot.transforms == 6 &&
               snapshot.hits == 6,
           "one pure transformation per exact source/kind, never application execution");
}
void negative_and_disabled_controls() {
    CudaPtxCache cache;
    constexpr auto unsupported =
        ".version 7.0\n.target sm_70\n.address_size 64\n"
        ".visible .entry barrier() { bar.sync 15; ret; }";
    const auto first = cache.lookup(unsupported, kKind), second = cache.lookup(unsupported, kKind);
    expect(first.status == CudaPtxCacheStatus::kUnsupported && !first.image && !first.was_hit &&
               second.status == CudaPtxCacheStatus::kUnsupported && second.was_hit &&
               !second.image && cache.snapshot().transforms == 1,
           "deterministic unsupported source can be reused without caching SDK failure");
    expect(cache.lookup({}, kKind).status == CudaPtxCacheStatus::kInvalid &&
               cache.lookup(kPtx, static_cast<PtxExecutionKind>(7)).status ==
                   CudaPtxCacheStatus::kInvalid &&
               cache.lookup(std::string("bad\0image", 9), kKind).status ==
                   CudaPtxCacheStatus::kInvalid,
           "invalid source and unknown generator kind never acquire cache authority");
    for (const auto limits : std::array{CudaPtxCacheLimits{0, 4096}, CudaPtxCacheLimits{8, 0},
                                        CudaPtxCacheLimits{8, 1}}) {
        CudaPtxCache disabled(limits);
        expect(disabled.lookup(kPtx, kKind).status == CudaPtxCacheStatus::kBypass &&
                   disabled.snapshot().entries == 0 && disabled.snapshot().retained_bytes == 0 &&
                   disabled.snapshot().transforms == 0,
               "zero or undersized cache never suppresses the caller's uncached path");
    }
    std::string oversized(glimmer::transform::kMaxPersistentPtxBytes + 1, ' ');
    expect(cache.lookup(oversized, kKind).status == CudaPtxCacheStatus::kBypass,
           "large original modules bypass cache retention and hashing");
    bool rejected{};
    try {
        CudaPtxCache invalid({4097, 1});
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    expect(rejected, "unbounded entry configuration rejects at construction");
    const auto stateful = std::string(kPtx) + "\n.global .u32 state;\n";
    CudaPtxCache scoped;
    expect(scoped.lookup(stateful, kKind).status == CudaPtxCacheStatus::kUnsupported,
           "separate-module contract rejects stateful source");
    const auto owned = scoped.lookup(stateful, kKind, true);
    expect(owned.image && owned.image->has_module_variables &&
               scoped.lookup(stateful, kKind, true).image == owned.image &&
               scoped.lookup(stateful, kKind).status == CudaPtxCacheStatus::kUnsupported &&
               scoped.snapshot().entries == 2 && scoped.snapshot().transforms == 2 &&
               scoped.snapshot().hits == 2,
           "same-module ownership is part of exact cache identity, including negative entries");
}
void pinned_capacity_and_lru() {
    CudaPtxCache pinned({1, std::size_t{1024} * 1024});
    auto first = pinned.lookup(kPtx, kKind);
    const auto retained = pinned.snapshot().retained_bytes;
    const auto blocked = pinned.lookup(other('2'), kKind);
    auto hit = pinned.lookup(kPtx, kKind);
    expect(first.image && blocked.image && hit.was_hit && pinned.snapshot().entries == 1 &&
               pinned.snapshot().bypasses == 1 && pinned.prune() && pinned.snapshot().entries == 1,
           "active module borrow pins the cache; pressure returns a valid uncached artifact");
    first.image.reset();
    hit.image.reset();
    const auto admitted = pinned.lookup(other('2'), kKind);
    expect(admitted.image && !admitted.was_hit && pinned.snapshot().entries == 1 &&
               pinned.snapshot().evictions == 1 && blocked.image->image == admitted.image->image,
           "eviction waits for CPU borrows but never invalidates an uncached artifact");
    CudaPtxCache byte_bounded({8, retained - 1});
    expect(byte_bounded.lookup(kPtx, kKind).image && byte_bounded.snapshot().entries == 0 &&
               byte_bounded.snapshot().retained_bytes == 0 && byte_bounded.snapshot().bypasses == 1,
           "post-transform metadata charge cannot exceed the retained-byte budget");
    CudaPtxCache lru({2, std::size_t{1024} * 1024});
    expect(lru.lookup(kPtx, kKind).image && lru.lookup(other('2'), kKind).image,
           "fill two LRU entries");
    expect(lru.lookup(kPtx, kKind).was_hit && lru.lookup(other('3'), kKind).image &&
               lru.lookup(kPtx, kKind).was_hit && !lru.lookup(other('2'), kKind).was_hit,
           "least-recently-used unborrowed entry, not the hot exact source, is evicted");
    expect(lru.prune() && lru.snapshot().entries == 0 && lru.snapshot().retained_bytes == 0,
           "idle cache retention can be released without SDK cleanup");
}

void preparation_policy_identity() {
    constexpr PtxTransformLimits defaults{};
    const std::array positive{
        defaults,
        PtxTransformLimits{defaults.source_bytes + 1, defaults.output_bytes, defaults.token_count},
        PtxTransformLimits{defaults.source_bytes, defaults.output_bytes + 1, defaults.token_count},
        PtxTransformLimits{defaults.source_bytes, defaults.output_bytes, defaults.token_count + 1}};
    // Both are valid policies for this source; one cannot fit generated output,
    // and the other cannot tokenize the source. Neither is a malformed request.
    const std::array negative{
        PtxTransformLimits{defaults.source_bytes, std::string_view{kPtx}.size(),
                           defaults.token_count},
        PtxTransformLimits{defaults.source_bytes, defaults.output_bytes, 1}};
    CudaPtxCache cache({}, collision);
    std::array<std::shared_ptr<const glimmer::transform::PersistentPtx>, 4> images;
    for (std::size_t index = 0; index < positive.size(); ++index) {
        const auto result = cache.lookup(kPtx, kKind, false, positive[index]);
        expect(result.status == CudaPtxCacheStatus::kReady && result.image && !result.was_hit,
               "each valid limit field participates in a distinct positive cache key");
        images[index] = result.image;
        for (std::size_t previous = 0; previous < index; ++previous)
            expect(images[previous] != images[index] &&
                       images[previous]->image == images[index]->image,
                   "equal generated text and colliding hashes do not collapse policy identity");
    }
    for (const auto policy : negative) {
        const auto first = cache.lookup(kPtx, kKind, false, policy);
        const auto second = cache.lookup(kPtx, kKind, false, policy);
        expect(first.status == CudaPtxCacheStatus::kUnsupported && !first.image && !first.was_hit &&
                   second.status == CudaPtxCacheStatus::kUnsupported && second.was_hit,
               "bounded token/output failures are negative only for their exact policy");
    }
    expect(cache.snapshot().entries == 6 && cache.snapshot().transforms == 6,
           "four positive and two negative policies coexist without accidental reuse");
    for (std::size_t index = 0; index < positive.size(); ++index) {
        const auto result = cache.lookup(kPtx, kKind, false, positive[index]);
        expect(result.was_hit && result.image == images[index],
               "negative policies cannot poison previously successful preparation");
    }
    const auto archive = cache.encode("policy-runtime");
    CudaPtxCache restored({}, collision);
    expect(archive && restored.restore(*archive, "policy-runtime") &&
               restored.snapshot().loaded_entries == 6,
           "archive restores distinct positive and negative preparation limits");
    for (std::size_t index = 0; index < positive.size(); ++index) {
        const auto result = restored.lookup(kPtx, kKind, false, positive[index]);
        expect(result.was_hit && result.image && result.image->image == images[index]->image,
               "restored positive policy retains bytes without retransforming");
    }
    for (const auto policy : negative) {
        const auto result = restored.lookup(kPtx, kKind, false, policy);
        expect(result.was_hit && result.status == CudaPtxCacheStatus::kUnsupported && !result.image,
               "restored negative eligibility remains policy scoped");
    }
    expect(restored.snapshot().transforms == 0, "restored decisions need no generator execution");
    const PtxTransformLimits unrecorded{defaults.source_bytes + 2, defaults.output_bytes,
                                        defaults.token_count};
    expect(!restored.lookup(kPtx, kKind, false, unrecorded).was_hit &&
               restored.snapshot().transforms == 1,
           "unrecorded valid policy does not inherit a restored decision");

    CudaPtxCache invalid;
    for (unsigned int field = 0; field < 3; ++field) {
        for (const bool excessive : {false, true}) {
            auto policy = defaults;
            if (field == 0)
                policy.source_bytes =
                    excessive ? glimmer::transform::kMaxPtxTransformSourceBytes + 1 : 0;
            if (field == 1)
                policy.output_bytes =
                    excessive ? glimmer::transform::kMaxPtxTransformOutputBytes + 1 : 0;
            if (field == 2)
                policy.token_count = excessive ? glimmer::transform::kMaxPtxTransformTokens + 1 : 0;
            expect(
                invalid.lookup(kPtx, kKind, false, policy).status == CudaPtxCacheStatus::kInvalid,
                "malformed trusted bounds do not become negative cache entries");
        }
    }
    expect(!invalid.snapshot().entries && !invalid.snapshot().transforms,
           "invalid policies have no generator or retention side effects");

    std::string large{kPtx};
    large.resize(defaults.source_bytes + 1, ' ');
    const PtxTransformLimits trusted{large.size(), std::size_t{8} * 1024 * 1024,
                                     defaults.token_count};
    CudaPtxCache bounded({8, large.size() + 1});
    const auto result = bounded.lookup(large, kKind, false, trusted);
    expect(result.status == CudaPtxCacheStatus::kReady && result.image &&
               result.image->image.starts_with(large) && !bounded.snapshot().entries &&
               bounded.snapshot().transforms == 1 && bounded.snapshot().bypasses == 1,
           "a prepared image exceeding cache retention is returned once as Ready, not discarded");
    expect(bounded.lookup(large, kKind).status == CudaPtxCacheStatus::kBypass &&
               bounded.snapshot().transforms == 1,
           "trusted oversized source does not weaken the legacy cache eligibility bound");
}

void archive_roundtrip_and_rejections() {
    constexpr std::array kinds{PtxExecutionKind::kStatic, PtxExecutionKind::kBlockStatic,
                               PtxExecutionKind::kDynamic, kKind, PtxExecutionKind::kSliced};
    CudaPtxCache original({}, collision);
    for (const auto kind : kinds)
        expect(original.lookup(kPtx, kind).image != nullptr, "prepare every archived family");
    expect(original.lookup(other('2'), kKind).image != nullptr, "archive a forced hash collision");
    const auto stateful = std::string(kPtx) + "\n.global .u32 state;\n";
    expect(original.lookup(stateful, kKind, true).image != nullptr &&
               original.lookup(stateful, kKind).status == CudaPtxCacheStatus::kUnsupported,
           "archive distinct owned-state and unsupported separate-module contracts");
    const auto archive = original.encode("runtime-A");
    expect(archive && !archive->empty(), "encode complete immutable preparation records");
    CudaPtxCache restored({}, collision);
    expect(restored.restore(*archive, "runtime-A") && restored.snapshot().entries == 8 &&
               restored.snapshot().loaded_entries == 8 && restored.snapshot().transforms == 0,
           "restore cached code and metadata without running the generator");
    for (const auto kind : kinds) {
        const auto expected = original.lookup(kPtx, kind);
        const auto actual = restored.lookup(kPtx, kind);
        expect(actual.was_hit && actual.image && actual.image->image == expected.image->image &&
                   actual.image->kernels.front().execution_kind == kind &&
                   actual.image->kernels.front().persistent_name ==
                       expected.image->kernels.front().persistent_name,
               "restored exact source/family reuses identical code and metadata");
    }
    auto restored_state = restored.lookup(stateful, kKind, true);
    expect(restored.lookup(other('2'), kKind).was_hit && restored_state.image &&
               restored_state.image->has_module_variables &&
               restored.lookup(stateful, kKind).status == CudaPtxCacheStatus::kUnsupported &&
               restored.snapshot().hits == 8 && restored.snapshot().transforms == 0 &&
               !restored.restore(*archive, "runtime-A"),
           "collision, sharing and negative entries survive without replacing a live cache");
    // Drop the stateful borrow before checking single-entry pinning below.
    restored_state.image.reset();
    auto pinned = restored.lookup(kPtx, kKind).image;
    expect(restored.prune() && restored.snapshot().entries == 1 &&
               restored.lookup(kPtx, kKind).image == pinned,
           "restored code obeys the same immutable-borrow eviction policy");
    pinned.reset();
    expect(restored.prune() && restored.snapshot().entries == 0 &&
               !restored.restore(*archive, "runtime-A"),
           "pruning does not turn a used cache into a pristine restore target");
    for (const auto limits :
         std::array{CudaPtxCacheLimits{7, std::size_t{1024} * 1024}, CudaPtxCacheLimits{128, 1}}) {
        CudaPtxCache bounded(limits);
        expect(!bounded.restore(*archive, "runtime-A") && !bounded.snapshot().entries &&
                   !bounded.snapshot().retained_bytes && !bounded.snapshot().loaded_entries,
               "restore enforces all entry/retention bounds transactionally");
    }
    CudaPtxCache empty;
    const auto empty_archive = empty.encode("runtime-A");
    expect(empty_archive && empty.restore(*empty_archive, "runtime-A") &&
               !empty.snapshot().entries && !empty.snapshot().transforms &&
               !empty.restore(*empty_archive, "runtime-A"),
           "an empty preparation cache has a valid bounded archive");

    constexpr auto parameterized =
        ".version 7.0\n.target sm_70\n.address_size 64\n"
        ".entry kernel(.param .align 8 .b8 payload[16], .param .u32 value) { ret; }\n";
    CudaPtxCache source;
    expect(source.lookup(parameterized, kKind).image != nullptr, "parameter-layout archive input");
    const auto encoded = source.encode("runtime-A");
    expect(encoded.has_value(), "encode one parameterized artifact");
    const auto reject = [&](std::span<const std::byte> bytes,
                            std::string_view identity = "runtime-A") {
        CudaPtxCache candidate({}, collision);
        expect(!candidate.restore(bytes, identity) && !candidate.snapshot().entries &&
                   !candidate.snapshot().retained_bytes && !candidate.snapshot().transforms &&
                   !candidate.snapshot().loaded_entries,
               "invalid archive rejects before publishing any preparation state");
    };
    reject(*encoded, "runtime-B");
    reject(*encoded, "");
    for (std::size_t length = 0; length < encoded->size(); ++length)
        reject(std::span{*encoded}.first(length));
    auto corrupt = *encoded;
    corrupt[corrupt.size() / 2] ^= std::byte{1};
    reject(corrupt);
    const auto resign = [](std::vector<std::byte>& bytes) {
        std::uint64_t value = 14695981039346656037ULL;
        for (const auto byte : std::span{bytes}.first(bytes.size() - 8))
            value = (value ^ std::to_integer<unsigned int>(byte)) * 1099511628211ULL;
        for (std::size_t index = 0; index < 8; ++index)
            bytes[bytes.size() - 8 + index] = static_cast<std::byte>(value >> (index * 8));
    };
    const auto mutate = [&](std::size_t offset, std::uint32_t value) {
        auto bytes = *encoded;
        expect(offset + 4 <= bytes.size() - 8, "bounded test archive field");
        for (std::size_t index = 0; index < 4; ++index)
            bytes[offset + index] = static_cast<std::byte>(value >> (index * 8));
        resign(bytes);
        reject(bytes);
    };
    std::size_t cursor{};
    const auto number = [&] {
        expect(cursor + 4 <= encoded->size() - 8, "bounded test number read");
        std::uint32_t value{};
        for (std::size_t index = 0; index < 4; ++index)
            value |= std::to_integer<std::uint32_t>((*encoded)[cursor++]) << (index * 8);
        return value;
    };
    const auto text = [&] {
        const auto length = number();
        const auto start = cursor;
        expect(length <= encoded->size() - 8 - cursor, "bounded test text read");
        cursor += length;
        return start;
    };
    (void)text();  // Magic.
    mutate(cursor, 1);
    mutate(cursor, 3);
    expect(number() == 2, "limit-bearing archives use codec version two");
    mutate(cursor, 6);
    mutate(cursor, kCudaPtxGeneratorAbi + 1);
    mutate(cursor, 7);
    mutate(cursor, 8);
    mutate(cursor, 9);
    expect(number() == 10, "block-completion preparation identity uses generator ABI ten");
    (void)text();  // Runtime identity.
    const auto entries_field = cursor;
    mutate(cursor, 2);
    (void)number();  // Entry count / truncated second entry.
    const auto entry_start = cursor;
    mutate(cursor, 99);
    (void)number();  // Family.
    mutate(cursor, 2);
    (void)number();  // Sharing boolean.
    for (const auto ceiling :
         std::array<std::uint32_t, 3>{64 * 1024 * 1024, 256 * 1024 * 1024, 16 * 1024 * 1024}) {
        mutate(cursor, 0);
        mutate(cursor, ceiling + 1);
        (void)number();  // Source, output and token policy fields, respectively.
    }
    mutate(cursor, 2);
    mutate(cursor, 1);
    (void)number();  // Exact observation policy boolean, distinct from ordinary preparation.
    const auto source_start = text();
    mutate(source_start, 0);  // Embedded NUL / source-prefix mismatch.
    mutate(cursor, 2);
    (void)number();  // Ready boolean.
    mutate(cursor, 1);
    (void)number();  // Stateful image without shared-module ownership.
    const auto image_start = text();
    mutate(image_start, 0x41414141U);  // Original source prefix must be exact.
    mutate(cursor, 0);
    (void)number();  // No variants.
    const auto original_name = text(), generated_name = text();
    mutate(original_name, 0);             // Invalid native entry name.
    mutate(generated_name, 0x41414141U);  // Deterministic generated entry name.
    mutate(cursor, 257);
    (void)number();
    mutate(cursor, 2);
    mutate(cursor, 1);
    (void)number();  // Variant observation metadata must agree with preparation policy.
    mutate(cursor, 0);
    (void)number();  // Zero parameter size.
    mutate(cursor, 3);
    (void)number();         // Non-power-of-two alignment.
    mutate(cursor, 65536);  // Sum exceeds the original 64-KiB parameter contract.
    corrupt = *encoded;
    corrupt.insert(corrupt.end() - 8, std::byte{0});
    resign(corrupt);
    reject(corrupt);
    corrupt = *encoded;
    corrupt[entries_field] = std::byte{2};
    corrupt.insert(corrupt.end() - 8, encoded->begin() + static_cast<std::ptrdiff_t>(entry_start),
                   encoded->end() - 8);
    resign(corrupt);
    reject(corrupt);
    CudaPtxCache accepted;
    expect(!accepted.restore(corrupt, "runtime-A") && !accepted.snapshot().entries &&
               !accepted.snapshot().retained_bytes && !accepted.snapshot().loaded_entries &&
               !accepted.snapshot().transforms && accepted.restore(*encoded, "runtime-A"),
           "failed transactional restoration permits a later complete valid restore");
    const auto image = accepted.lookup(parameterized, kKind).image;
    expect(image && image->kernels[0].parameter_count == 2 &&
               image->kernels[0].parameter_layout[0].size_bytes == 16 &&
               image->kernels[0].parameter_layout[0].alignment_bytes == 8,
           "restored parameter size/alignment is preserved without deriving a new ABI");
    expect(!original.encode("") && !original.encode(std::string("bad\0identity", 12)),
           "invalid runtime identities are not encoded");
    CudaPtxCache raced;
    auto first =
        std::async(std::launch::async, [&] { return raced.restore(*encoded, "runtime-A"); });
    auto second =
        std::async(std::launch::async, [&] { return raced.restore(*encoded, "runtime-A"); });
    expect(first.get() != second.get() && raced.snapshot().entries == 1 &&
               raced.snapshot().loaded_entries == 1 && !raced.snapshot().transforms,
           "concurrent restores commit exactly one complete pristine-cache transaction");
}

void collective_loop_archive_identity(std::string_view source, std::uint32_t prior_abi,
                                      std::string_view runtime_identity) {
    constexpr std::array kinds{PtxExecutionKind::kStatic, PtxExecutionKind::kBlockStatic,
                               PtxExecutionKind::kDynamic, PtxExecutionKind::kPreemptive,
                               PtxExecutionKind::kSliced};
    CudaPtxCache original;
    for (const auto kind : kinds) {
        const auto result = original.lookup(source, kind);
        expect(
            result.status == (kind == PtxExecutionKind::kStatic ? CudaPtxCacheStatus::kUnsupported
                                                                : CudaPtxCacheStatus::kReady) &&
                !result.was_hit &&
                static_cast<bool>(result.image) == (kind != PtxExecutionKind::kStatic),
            "collective loops cache four cooperative artifacts and one negative static decision");
    }
    const auto archive = original.encode(runtime_identity);
    expect(archive.has_value(), "encode complete collective-loop preparation decisions");

    // Keep a valid checksum while marking the archive as the previous generator.
    // This isolates ABI invalidation; these are synthetic test bytes, not a
    // retained historical binary or evidence of historical numerical behavior.
    const auto number = [&](std::size_t offset) {
        expect(offset <= archive->size() - 8 && 4 <= archive->size() - 8 - offset,
               "bounded collective-loop archive field");
        std::uint32_t value{};
        for (std::size_t index = 0; index < 4; ++index)
            value |= std::to_integer<std::uint32_t>((*archive)[offset + index]) << (index * 8);
        return value;
    };
    const auto generator_offset = std::size_t{4} + number(0) + 4;
    expect(kCudaPtxGeneratorAbi == 10 && number(generator_offset) == 10,
           "current loop eligibility is bound to exact generator ABI ten");
    auto prior = *archive;
    for (std::size_t index = 0; index < 4; ++index)
        prior[generator_offset + index] = static_cast<std::byte>(prior_abi >> (index * 8));
    std::uint64_t digest = 14695981039346656037ULL;
    for (const auto byte : std::span{prior}.first(prior.size() - 8))
        digest = (digest ^ std::to_integer<unsigned int>(byte)) * 1099511628211ULL;
    for (std::size_t index = 0; index < 8; ++index)
        prior[prior.size() - 8 + index] = static_cast<std::byte>(digest >> (index * 8));

    CudaPtxCache restored;
    expect(!restored.restore(prior, runtime_identity) && !restored.snapshot().entries &&
               !restored.snapshot().retained_bytes && !restored.snapshot().loaded_entries &&
               !restored.snapshot().transforms,
           "checksum-valid prior ABI cannot publish stale positive or negative loop decisions");
    expect(restored.restore(*archive, runtime_identity),
           "rejected prior ABI leaves the cache pristine for the current exact archive");
    for (const auto kind : kinds) {
        const auto before = original.lookup(source, kind);
        const auto after = restored.lookup(source, kind);
        expect(after.was_hit && after.status == before.status,
               "all five restored loop decisions are exact cache hits");
        if (kind == PtxExecutionKind::kStatic) {
            expect(!after.image, "negative static eligibility never acquires a generated image");
            continue;
        }
        expect(before.image && after.image && after.image->image == before.image->image &&
                   after.image->image.starts_with(source) && after.image->kernels.size() == 1 &&
                   after.image->kernels[0].execution_kind == kind &&
                   after.image->kernels[0].persistent_name ==
                       before.image->kernels[0].persistent_name &&
                   after.image->kernels[0].parameter_count == 2 &&
                   after.image->kernels[0].parameter_layout.size() == 2 &&
                   after.image->kernels[0].parameter_layout[0].size_bytes == 8 &&
                   after.image->kernels[0].parameter_layout[0].alignment_bytes == 8 &&
                   after.image->kernels[0].parameter_layout[1].size_bytes == 4 &&
                   after.image->kernels[0].parameter_layout[1].alignment_bytes == 4,
               "restored loop code and public parameter ABI are unchanged");
    }
    const auto state = restored.snapshot();
    expect(state.entries == 5 && state.loaded_entries == 5 && state.hits == 5 && !state.transforms,
           "restored collective-loop decisions require zero transformations");
}

void opaque_static_archive_identity() {
    constexpr std::string_view parameters =
        ".param .align 16 .b8 packet[24], .param .u32 value, .param .align 8 .b8 flags[3]";
    const std::string source =
        ".version 7.0\n.target sm_70\n.address_size 64\n.visible .entry opaque(" +
        std::string(parameters) + ") { ret; }\n";
    constexpr auto kind = PtxExecutionKind::kStatic;
    constexpr PtxTransformLimits negative_policy{std::size_t{4} * 1024 * 1024,
                                                 std::size_t{4} * 1024 * 1024, 1};
    constexpr std::string_view identity = "opaque-static-runtime";
    expect(kCudaPtxGeneratorAbi == 10,
           "opaque Static eligibility uses exact current generator ABI 10");
    const auto unchanged = [](const CudaPtxCacheSnapshot& before,
                              const CudaPtxCacheSnapshot& after) {
        expect(before.entries == after.entries && before.retained_bytes == after.retained_bytes &&
                   before.lookups == after.lookups && before.hits == after.hits &&
                   before.transforms == after.transforms && before.evictions == after.evictions &&
                   before.bypasses == after.bypasses && before.is_valid == after.is_valid &&
                   before.loaded_entries == after.loaded_entries,
               "rejected ABI restore leaves every cache counter and ownership charge unchanged");
    };
    // Exercise positive-only, negative-only and mixed archives independently.
    // The negative is a real bounded-token decision for this exact source, not
    // a fabricated historical generator result or a native compilation error.
    for (unsigned int mode = 0; mode < 3; ++mode) {
        CudaPtxCache original({}, collision);
        std::shared_ptr<const glimmer::transform::PersistentPtx> image;
        if (mode != 1) {
            const auto first = original.lookup(source, kind);
            expect(
                first.status == CudaPtxCacheStatus::kReady && first.image && !first.was_hit,
                "Static accepts a bounded opaque signature without flattening original parameters");
            image = first.image;
            const auto hit = original.lookup(source, kind);
            expect(hit.was_hit && hit.image == image && image->image.starts_with(source) &&
                       image->image.find(parameters, source.size()) != std::string::npos &&
                       !image->has_module_variables && image->kernels.size() == 1,
                   "Static cache hit preserves immutable source and exact private parameter text");
            const auto& variant = image->kernels.front();
            expect(variant.original_name == "opaque" &&
                       variant.persistent_name == "opaque__glimmer_ptb" &&
                       variant.execution_kind == kind && variant.parameter_count == 3 &&
                       variant.parameter_layout.size() == 3 &&
                       variant.parameter_layout[0].size_bytes == 24 &&
                       variant.parameter_layout[0].alignment_bytes == 16 &&
                       variant.parameter_layout[1].size_bytes == 4 &&
                       variant.parameter_layout[1].alignment_bytes == 4 &&
                       variant.parameter_layout[2].size_bytes == 3 &&
                       variant.parameter_layout[2].alignment_bytes == 8,
                   "cached opaque layout retains all three original widths and alignments");
        }
        if (mode != 0) {
            const auto first = original.lookup(source, kind, false, negative_policy);
            const auto hit = original.lookup(source, kind, false, negative_policy);
            expect(first.status == CudaPtxCacheStatus::kUnsupported && !first.image &&
                       !first.was_hit && hit.status == CudaPtxCacheStatus::kUnsupported &&
                       !hit.image && hit.was_hit,
                   "valid narrow policy has its own immutable negative cache entry");
        }
        const auto archive = original.encode(identity);
        expect(archive && archive->size() >= 16,
               "encode current opaque positive and negative cache decisions");
        const auto number = [&](std::size_t offset) {
            expect(offset <= archive->size() - 8 && 4 <= archive->size() - 8 - offset,
                   "bounded independent archive header read");
            std::uint32_t value{};
            for (std::size_t index = 0; index < 4; ++index)
                value |= std::to_integer<std::uint32_t>((*archive)[offset + index]) << (index * 8);
            return value;
        };
        const auto version_offset = std::size_t{4} + number(0);
        const auto generator_offset = version_offset + 4;
        expect(number(version_offset) == 2 && number(generator_offset) == 10,
               "archive envelope remains version 2 with generator ABI 10 observation metadata");
        // Change the real serialized ABI word and repair its checksum. These
        // counterfactual bytes isolate ABI rejection, not historical execution.
        auto prior = *archive;
        for (std::size_t index = 0; index < 4; ++index)
            prior[generator_offset + index] =
                static_cast<std::byte>(std::uint32_t{9} >> (index * 8));
        std::uint64_t digest = 14695981039346656037ULL;
        for (const auto byte : std::span{prior}.first(prior.size() - 8))
            digest = (digest ^ std::to_integer<unsigned int>(byte)) * 1099511628211ULL;
        for (std::size_t index = 0; index < 8; ++index)
            prior[prior.size() - 8 + index] = static_cast<std::byte>(digest >> (index * 8));

        CudaPtxCache restored({}, collision);
        const auto empty = restored.snapshot();
        expect(!restored.restore(prior, identity),
               "checksum-valid ABI 8 rejects both positive and negative opaque decisions");
        unchanged(empty, restored.snapshot());
        expect(restored.restore(*archive, identity) &&
                   restored.snapshot().loaded_entries == (mode == 2 ? 2U : 1U),
               "prior ABI rejection leaves a pristine target for a complete ABI 9 restore");
        if (mode != 1) {
            const auto result = restored.lookup(source, kind);
            expect(result.status == CudaPtxCacheStatus::kReady && result.was_hit && result.image &&
                       result.image->image == image->image && result.image->kernels.size() == 1 &&
                       result.image->kernels.front().parameter_count == 3 &&
                       result.image->kernels.front().parameter_layout.size() == 3,
                   "restored Static opaque image is an exact positive cache hit");
            const auto& variant = result.image->kernels.front();
            expect(variant.original_name == "opaque" &&
                       variant.persistent_name == "opaque__glimmer_ptb" &&
                       variant.execution_kind == kind,
                   "restored opaque variant retains the exact original/private identity");
            for (std::size_t index = 0; index < variant.parameter_layout.size(); ++index)
                expect(variant.parameter_layout[index].size_bytes ==
                               image->kernels.front().parameter_layout[index].size_bytes &&
                           variant.parameter_layout[index].alignment_bytes ==
                               image->kernels.front().parameter_layout[index].alignment_bytes,
                       "opaque restoration preserves every original size and alignment");
        }
        if (mode != 0) {
            const auto result = restored.lookup(source, kind, false, negative_policy);
            expect(result.status == CudaPtxCacheStatus::kUnsupported && result.was_hit &&
                       !result.image,
                   "restored current negative remains scoped to its exact policy");
        }
        expect(!restored.snapshot().transforms,
               "trusted ABI 9 positive and negative restore does not regenerate PTX");

        CudaPtxCache live;
        const auto pinned = live.lookup(kPtx, kKind);
        expect(pinned.image != nullptr, "retain a distinct already-used cache image");
        const auto before = live.snapshot();
        const auto encoded_before = live.encode(identity);
        expect(
            encoded_before && !live.restore(prior, identity) && !live.restore(*archive, identity),
            "neither stale nor current archives replace an existing cache");
        unchanged(before, live.snapshot());
        expect(
            live.encode(identity) == encoded_before &&
                live.lookup(kPtx, kKind).image == pinned.image,
            "failed restore preserves existing serialized decisions and borrowed image identity");
    }
}
CudaPtxCache* hash_observer{};
std::size_t reentrant_hash(std::string_view) noexcept {
    return hash_observer && hash_observer->snapshot().is_valid ? 0 : 1;
}
void threads_reentrancy_and_fork() {
    CudaPtxCache cache({}, reentrant_hash);
    hash_observer = &cache;
    constexpr std::size_t threads = 8;
    std::vector<std::future<CudaPtxCacheResult>> jobs;
    for (std::size_t index = 0; index < threads; ++index)
        jobs.push_back(std::async(std::launch::async, [&] { return cache.lookup(kPtx, kKind); }));
    auto first = jobs.front().get();
    for (std::size_t index = 1; index < threads; ++index)
        expect(jobs[index].get().image == first.image,
               "concurrent exact lookups share one immutable artifact");
    expect(cache.snapshot().transforms == 1 && cache.snapshot().hits == threads - 1,
           "CPU generation is deduplicated and external hash observation does not hold cache "
           "exclusion");
    const auto archive = cache.encode("runtime-A");
    expect(archive.has_value(), "archive pinned CPU image without SDK ownership");
    const auto child = ::fork();
    expect(child >= 0, "fork cache process guard control");
    if (!child)
        ::_exit(cache.lookup(kPtx, kKind).status == CudaPtxCacheStatus::kWrongProcess &&
                        !cache.snapshot().is_valid && !cache.prune() &&
                        !cache.encode("runtime-A") && !cache.restore(*archive, "runtime-A")
                    ? 0
                    : 1);
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "inherited cache rejects before any mutex or hash callback");
    CudaPtxCache imported({}, reentrant_hash);
    hash_observer = &imported;
    expect(imported.restore(*archive, "runtime-A") && imported.lookup(kPtx, kKind).was_hit,
           "restore hash observation runs outside exclusion too");
    hash_observer = &cache;
    auto release = std::async(std::launch::async,
                              [image = std::move(first.image)]() mutable { image.reset(); });
    release.get();
    expect(cache.prune() && cache.snapshot().entries == 0,
           "off-thread CPU borrow release needs no SDK owner");
    hash_observer = nullptr;
}
}  // namespace
int main() {
    try {
        exact_immutable_identity_and_kind();
        negative_and_disabled_controls();
        pinned_capacity_and_lru();
        preparation_policy_identity();
        archive_roundtrip_and_rejections();
        collective_loop_archive_identity(kUniformLoopPtx, 5, "uniform-loop-runtime");
        collective_loop_archive_identity(kReconvergentLoopPtx, 8, "reconvergent-loop-runtime");
        opaque_static_archive_identity();
        threads_reentrancy_and_fork();
        CudaPtxCache calls;
        for (const auto kind :
             {PtxExecutionKind::kStatic, PtxExecutionKind::kBlockStatic, PtxExecutionKind::kDynamic,
              PtxExecutionKind::kPreemptive, PtxExecutionKind::kSliced})
            expect(calls.lookup(kDeviceCallPtx, kind).status == CudaPtxCacheStatus::kReady,
                   "every family caches the full private helper call graph");
        const auto archive = calls.encode("call-graph-runtime");
        CudaPtxCache restored;
        expect(archive && restored.restore(*archive, "call-graph-runtime"),
               "durable preparation preserves helper definitions and private calling ABI");
        for (const auto kind :
             {PtxExecutionKind::kStatic, PtxExecutionKind::kBlockStatic, PtxExecutionKind::kDynamic,
              PtxExecutionKind::kPreemptive, PtxExecutionKind::kSliced}) {
            const auto result = restored.lookup(kDeviceCallPtx, kind);
            expect(result.was_hit && result.image &&
                       result.image->image == calls.lookup(kDeviceCallPtx, kind).image->image &&
                       result.image->kernels.size() == 1 &&
                       result.image->kernels.front().parameter_count == 2,
                   "restored helper code remains exact without exposing private parameters");
        }
        expect(restored.snapshot().transforms == 0 && restored.snapshot().loaded_entries == 5,
               "restored call-graph artifacts do not regenerate");
        CudaPtxCache computation;
        for (const auto source : {kAtomicPtx, kWarpPtx})
            for (const auto kind : {PtxExecutionKind::kStatic, PtxExecutionKind::kBlockStatic,
                                    PtxExecutionKind::kDynamic, PtxExecutionKind::kPreemptive,
                                    PtxExecutionKind::kSliced})
                expect(computation.lookup(source, kind).status ==
                           (source == kWarpPtx && kind == PtxExecutionKind::kStatic
                                ? CudaPtxCacheStatus::kUnsupported
                                : CudaPtxCacheStatus::kReady),
                       "computation cache separates independent and whole-CTA eligibility");
        const auto computation_archive = computation.encode("computation-runtime");
        CudaPtxCache computation_restored;
        expect(computation_archive &&
                   computation_restored.restore(*computation_archive, "computation-runtime"),
               "atomic/warp preparation restores without changing memory or convergence semantics");
        for (const auto source : {kAtomicPtx, kWarpPtx})
            for (const auto kind : {PtxExecutionKind::kStatic, PtxExecutionKind::kBlockStatic,
                                    PtxExecutionKind::kDynamic, PtxExecutionKind::kPreemptive,
                                    PtxExecutionKind::kSliced}) {
                const auto before = computation.lookup(source, kind);
                const auto after = computation_restored.lookup(source, kind);
                expect(after.was_hit && after.status == before.status &&
                           ((!before.image && !after.image) ||
                            (before.image && after.image &&
                             before.image->image == after.image->image)),
                       "positive and negative computation eligibility survive archive identity");
            }
        expect(computation_restored.snapshot().transforms == 0 &&
                   computation_restored.snapshot().loaded_entries == 10,
               "restored computation artifacts never regenerate");
        std::cout << "Bounded immutable PTX preparation cache checks passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
