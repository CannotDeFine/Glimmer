#include "cuda_image_fixture.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace {
using namespace glimmer::transform;
using namespace glimmer::test;
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

void append_members(std::vector<std::byte>& destination, const std::vector<std::byte>& source) {
    expect(source.size() >= 16 && destination.size() >= 16, "independent member fixture bounds");
    destination.insert(destination.end(), source.begin() + 16, source.end());
    image_store(destination, 8, static_cast<std::uint64_t>(destination.size() - 16));
}

std::vector<std::byte> repeated_tail_ptx(std::string_view prefix, std::size_t decoded_bytes) {
    expect(prefix.size() >= 15 && decoded_bytes >= prefix.size() + 24,
           "independent LZ4 literal prefix and repeated-byte match bounds");
    std::vector<std::byte> payload{std::byte{0xff}};
    const auto extension = [&](std::size_t count) {
        while (count >= 255) {
            payload.push_back(std::byte{255});
            count -= 255;
        }
        payload.push_back(static_cast<std::byte>(count));
    };
    extension(prefix.size() - 15);
    const auto literals = std::as_bytes(std::span{prefix.data(), prefix.size()});
    payload.insert(payload.end(), literals.begin(), literals.end());
    payload.push_back(std::byte{1});  // Match the last literal byte, distance one.
    payload.push_back(std::byte{0});
    extension(decoded_bytes - prefix.size() - 24);
    // A block ends with a literal-only sequence, including the final five
    // decoded bytes. Ending directly after the match is not a valid fixture.
    payload.push_back(std::byte{0x50});
    for (unsigned int index = 0; index < 5; ++index)
        payload.push_back(static_cast<std::byte>(prefix.back()));
    std::vector<std::byte> image(80);
    image_store(image, 0, std::uint32_t{0xba55ed50});
    image_store(image, 4, std::uint16_t{1});
    image_store(image, 6, std::uint16_t{16});
    image_store(image, 16, std::uint16_t{1});
    image_store(image, 18, std::uint16_t{0x101});
    image_store(image, 20, std::uint32_t{64});
    image_store(image, 24, static_cast<std::uint64_t>(payload.size()));
    image_store(image, 32, static_cast<std::uint32_t>(payload.size()));
    image_store(image, 44, std::uint32_t{70});
    image_store(image, 56, std::uint64_t{0x2011});
    image_store(image, 72, static_cast<std::uint64_t>(decoded_bytes));
    image.insert(image.end(), payload.begin(), payload.end());
    image_store(image, 8, static_cast<std::uint64_t>(image.size() - 16));
    return image;
}

void independent_preparation_budgets() {
    constexpr std::string_view prefix =
        ".version 7.0\n.target sm_70\n.address_size 64\n.entry kernel(){ret;} ";
    constexpr std::size_t expanded = 4096;
    const auto cubin = cuda_test_cubin({});
    auto image = cuda_test_fatbin(cubin);
    append_members(image, repeated_tail_ptx(prefix, expanded));
    auto limits = CudaImageLimits{};
    limits.image_bytes = image.size();
    limits.decoded_bytes = expanded;
    limits.retained_bytes = 8192;
    limits.work_bytes = std::size_t{1024} * 1024;
    limits.compact_ptx_arities = true;
    const auto result = inspect_cuda_module(image, 70, limits);
    if (!result) throw std::runtime_error("expected bounded compressed module metadata");
    const auto expected = std::string(prefix) + std::string(expanded - prefix.size(), ' ');
    expect(image.size() < expanded && result && result->ptx == expected &&
               result->kernels.size() == 1 && result->kernels[0].arguments.empty() &&
               result->validation_ptx.empty() && result->ptx_arities.size() == 1 &&
               result->ptx_arities[0].parameter_count == 0,
           "small compressed raw image has independently bounded decoded source and native ABI");
    for (unsigned int field = 0; field < 4; ++field) {
        auto insufficient = limits;
        if (field == 0) --insufficient.image_bytes;
        if (field == 1) --insufficient.decoded_bytes;
        if (field == 2) insufficient.retained_bytes = result->retained_bytes - 1;
        if (field == 3) insufficient.work_bytes = expanded - 1;
        expect(!inspect_cuda_module(image, 70, insufficient),
               "raw, single decode, retained metadata and actual work each reject independently");
        for (const bool excessive : {false, true}) {
            auto invalid = limits;
            if (field == 0) invalid.image_bytes = excessive ? kMaxCudaModuleBytes + 1 : 0;
            if (field == 1) invalid.decoded_bytes = excessive ? kMaxCudaModuleBytes + 1 : 0;
            if (field == 2) invalid.retained_bytes = excessive ? kMaxCudaModuleBytes + 1 : 0;
            if (field == 3) invalid.work_bytes = excessive ? kMaxCudaModuleWorkBytes + 1 : 0;
            expect(!inspect_cuda_module(image, 70, invalid),
                   "zero or excessive preparation limits cannot enlarge a trusted hard ceiling");
        }
    }
    limits.retained_bytes = result->retained_bytes;
    expect(inspect_cuda_module(image, 70, limits).has_value(),
           "exact reported logical retention is sufficient without a hidden source duplicate");
    limits.work_bytes = kMaxCudaModuleWorkBytes;
    expect(inspect_cuda_module(image, 70, limits).has_value(),
           "explicit one-GiB cumulative-work ceiling does not change metadata");

    auto second_source = repeated_tail_ptx(prefix, expanded);
    image_store(second_source, 44, std::uint32_t{60});
    append_members(image, second_source);
    limits.image_bytes = image.size();
    limits.retained_bytes = 8192;  // Also permits later sources' temporary name inventories.
    const auto repeated = inspect_cuda_module(image, 70, limits);
    expect(
        repeated && repeated->ptx == expected && repeated->retained_bytes == result->retained_bytes,
        "two separately bounded decoded members do not share a single-member size budget");

    // An incompatible PTX member has a valid bounded declaration but corrupt
    // compressed content. It is not a source the selected device will decode.
    auto unused = repeated_tail_ptx(prefix, expanded);
    image_store(unused, 44, std::uint32_t{80});
    unused[80] = std::byte{0};
    append_members(image, unused);
    limits.image_bytes = image.size();
    const auto ignored = inspect_cuda_module(image, 70, limits);
    expect(ignored && ignored->ptx == expected && ignored->retained_bytes == result->retained_bytes,
           "incompatible PTX consumes raw intake but no decoded source or retained metadata");
    image_store(image, image.size() - unused.size() + 16 + 28, std::uint32_t{70});
    expect(!inspect_cuda_module(image, 70, limits),
           "the same malformed member cannot be ignored when its architecture is eligible");
}

void padded_source_retention() {
    constexpr std::size_t expanded = std::size_t{2} * 1024 * 1024;
    const std::string source =
        ".version 7.0\n.target sm_70\n.address_size 64\n.entry kernel(){ret;}\n";
    auto prefix = source;
    prefix.push_back('\0');
    const auto image = repeated_tail_ptx(prefix, expanded);
    expect(image.size() < expanded / 100 && source.size() < 100,
           "independent compressed fixture has a tiny text prefix and multi-MiB zero tail");
    for (const bool compact : {false, true}) {
        auto limits = CudaImageLimits{};
        limits.compact_ptx_arities = compact;
        limits.retained_bytes = 4096;
        expect(!inspect_cuda_module(image, 70, limits),
               "tiny retained quota cannot authorize moved decoded capacity hidden after NUL");
        limits.retained_bytes = expanded * 3;
        const auto result = inspect_cuda_module(image, 70, limits);
        if (!result || !result->ptx) throw std::runtime_error("expected retained PTX metadata");
        expect(result && result->is_ptx_selected && result->ptx == source &&
                   result->validation_ptx.size() == (compact ? 0 : 1) &&
                   result->ptx_arities.size() == (compact ? 1 : 0),
               "ample explicit retention preserves compact/legacy exact text and arity behavior");
        auto storage = result->ptx->capacity() + 1;
        for (const auto& text : result->validation_ptx) {
            expect(text == source,
                   "legacy validation preserves text without exposing zero padding");
            storage += text.capacity() + 1;
        }
        expect(storage >= expanded && result->retained_bytes >= storage &&
                   result->retained_bytes <= limits.retained_bytes,
               "published logical retention covers actual persistent source capacities");
        auto malformed_prefix = prefix;
        malformed_prefix.push_back('x');
        expect(!inspect_cuda_module(repeated_tail_ptx(malformed_prefix, expanded), 70, limits),
               "nonzero bytes after the first NUL remain malformed even with ample retention");
    }
}

void compact_arity_facts() {
    const std::array<CudaImageArgument, 2> native_arguments{{{0, 8}, {16, 16}}};
    const std::string header = ".version 7.0\n.target sm_70\n.address_size 64\n";
    const auto selected = header +
                          ".entry kernel(.param .u64 p,.param .align 16 .b8 a[16]){ret;}\n"
                          ".entry zero(){ret;}\n"
                          ".entry first(.param .u32 p){ret;}\n"
                          ".entry first(.param .u32 p,.param .u32 q){ret;}\n"
                          ".entry tombstone(.param .u32 p,){ret;}\n"
                          ".entry tombstone(.param .u32 p){ret;}\n"
                          ".entry from_later(.param .u32 p,){ret;}\n"
                          ".entry conflict(.param .u32 p){ret;}\n"
                          ".entry agreed(.param .u32 p,.param .u32 q){ret;}\n"
                          ".entry exceeds(.param .u32 p,.param .u32 q,.param .u32 r){ret;}\n"
                          ".entry exceeds(.param .u32 p){ret;}\n"
                          "// .entry comment_fake(.param .u64 p){}\n"
                          "/* .entry block_fake(){} */\n.file 1 \".entry quoted_fake(){}\"\n";
    const auto lower = header +
                       ".entry from_later(.param .u32 p){ret;}\n"
                       ".entry conflict(.param .u32 p,.param .u32 q){ret;}\n"
                       ".entry agreed(.param .u32 p,.param .u32 q){ret;}\n";
    auto image = cuda_test_fatbin(cuda_test_cubin(native_arguments), selected);
    auto fallback = cuda_test_fatbin({}, lower);
    image_store(fallback, 44, std::uint32_t{60});
    append_members(image, fallback);
    fallback = cuda_test_fatbin({}, header + ".entry conflict(.param .u32 p){ret;}\n");
    image_store(fallback, 44, std::uint32_t{50});
    append_members(image, fallback);
    const auto legacy = inspect_cuda_module(image, 70);
    if (!legacy) throw std::runtime_error("expected legacy module metadata");
    expect(legacy && legacy->validation_ptx.size() == 3 && legacy->ptx_arities.empty() &&
               legacy->ptx == selected,
           "default inspection preserves all legacy validation sources without compact facts");
    auto limits = CudaImageLimits{};
    limits.compact_ptx_arities = true;
    const auto compact = inspect_cuda_module(image, 70, limits);
    if (!compact) throw std::runtime_error("expected compact module metadata");
    expect(compact && compact->validation_ptx.empty() && compact->ptx == selected &&
               compact->ptx_arities.size() == 7 &&
               compact->retained_bytes < legacy->retained_bytes && compact->kernels.size() == 1 &&
               compact->kernels[0].arguments ==
                   std::vector(native_arguments.begin(), native_arguments.end()),
           "compact source facts retain one companion and never replace native argument layout");
    const auto fact = [&](std::string_view name, std::size_t count, bool conflict) {
        const auto found = std::ranges::find(compact->ptx_arities, name, &CudaImagePtxArity::name);
        expect(found != compact->ptx_arities.end() && found->parameter_count == count &&
                   found->has_conflicting_counts == conflict,
               "independent compact arity, first-definition and sticky-conflict oracle");
    };
    fact("kernel", 2, false);
    fact("zero", 0, false);
    fact("first", 1, false);
    fact("from_later", 1, false);
    fact("agreed", 2, false);
    fact("conflict", 1, true);
    fact("exceeds", 3, false);
    for (const auto absent : {"tombstone", "missing", "comment_fake", "block_fake", "quoted_fake"})
        expect(std::ranges::find(compact->ptx_arities, absent, &CudaImagePtxArity::name) ==
                   compact->ptx_arities.end(),
               "malformed first entry, absent entry and lexical decoys never become zero arity");
    limits.parameters = 2;
    const auto bounded = inspect_cuda_module(image, 70, limits);
    expect(bounded && bounded->ptx_arities.size() == 6 &&
               std::ranges::find(bounded->ptx_arities, "exceeds", &CudaImagePtxArity::name) ==
                   bounded->ptx_arities.end(),
           "over-limit first definition is a tombstone, not repaired by a later small duplicate");
    limits.parameters = 1024;
    limits.retained_bytes = legacy->retained_bytes - 1;
    expect(inspect_cuda_module(image, 70, limits).has_value(),
           "compact sources and bounded temporary inventories fit below legacy retention");
    limits.compact_ptx_arities = false;
    expect(!inspect_cuda_module(image, 70, limits),
           "the same retention quota still rejects larger legacy source duplication");

    limits = {};
    limits.compact_ptx_arities = true;
    const auto ptx_only = inspect_cuda_module(cuda_test_fatbin({}, selected), 70, limits);
    expect(ptx_only && ptx_only->is_ptx_selected && ptx_only->ptx == selected &&
               ptx_only->validation_ptx.empty() && ptx_only->kernels.empty() &&
               !ptx_only->ptx_arities.empty(),
           "PTX-only intake carries count facts without inventing native argument offsets");
    append_members(image, cuda_test_fatbin({}, selected));
    const auto ambiguous = inspect_cuda_module(image, 70, limits);
    expect(ambiguous && !ambiguous->ptx && !ambiguous->is_ptx_selected &&
               ambiguous->kernels.size() == 1 && ambiguous->ptx_arities.size() == 7 &&
               ambiguous->validation_ptx.empty(),
           "ambiguous companions retain native ABI and compact facts but select no source");
}

void symbol_metadata() {
    const std::array<CudaImageArgument, 2> arguments{{{0, 8}, {8, 4}}};
    const std::array<CudaImageVariable, 2> variables{{{"state", 16}, {"bias", 4}}};
    const auto image = cuda_test_cubin(arguments, 0, variables);
    const auto metadata = inspect_cuda_module(image, 70);
    if (!metadata) throw std::runtime_error("expected native module metadata");
    expect(metadata && metadata->has_complete_kernel_metadata &&
               metadata->variables == std::vector(variables.begin(), variables.end()) &&
               metadata->static_storage_bytes == 20,
           "owned CUDA object names/sizes and complete entry metadata are retained");
    auto limits = CudaImageLimits{};
    limits.variables = 1;
    expect(!inspect_cuda_module(image, 70, limits), "named object count is independently bounded");
    Elf64_Ehdr header{};
    std::memcpy(&header, image.data(), sizeof(header));
    Elf64_Shdr symbols{}, info{};
    std::memcpy(&symbols, image.data() + header.e_shoff + 3 * sizeof(symbols), sizeof(symbols));
    std::memcpy(&info, image.data() + header.e_shoff + 5 * sizeof(info), sizeof(info));
    Elf64_Sym variable{};
    const auto variable_offset = symbols.sh_offset + 2 * sizeof(variable);
    std::memcpy(&variable, image.data() + variable_offset, sizeof(variable));
    for (unsigned int mode = 0; mode < 6; ++mode) {
        auto changed = image;
        auto invalid = variable;
        if (mode == 0) invalid.st_value = UINT64_MAX;
        if (mode == 1) invalid.st_size = 0;
        if (mode == 2) invalid.st_size = 21;
        if (mode == 3) invalid.st_name = UINT32_MAX;
        if (mode == 4) invalid.st_shndx = 1000;
        image_store(changed, variable_offset + (mode == 5 ? sizeof(variable) : 0), invalid);
        const auto result = inspect_cuda_module(changed, 70);
        if (mode == 1)
            expect(result && !result->has_complete_variable_metadata,
                   "zero-sized nonqueryable objects preserve original-only intake");
        else
            expect(!result,
                   "object overflow, oversized extent, bad name/section and duplicates reject");
    }
    auto missing = image;
    info.sh_name = 0;
    image_store(missing, header.e_shoff + 5 * sizeof(info), info);
    const auto incomplete = inspect_cuda_module(missing, 70);
    expect(incomplete && !incomplete->has_complete_kernel_metadata && incomplete->kernels.empty() &&
               incomplete->variables == metadata->variables,
           "missing entry ABI is explicit incomplete metadata, not zero parameters");
    auto merged = cuda_test_fatbin(image);
    const auto duplicate = cuda_test_fatbin(image);
    merged.insert(merged.end(), duplicate.begin() + 16, duplicate.end());
    image_store(merged, 8, std::uint64_t{merged.size() - 16});
    const auto compatible = inspect_cuda_module(merged, 70);
    if (!compatible) throw std::runtime_error("expected compatible module metadata");
    expect(compatible && compatible->variables == metadata->variables,
           "matching selected members deduplicate identical named objects");
    expect(compatible->static_storage_bytes == 40,
           "all selected native members reserve their physical static section extents");
    const std::array<CudaImageVariable, 2> conflicts{{{"state", 20}, {"bias", 4}}};
    const auto conflicting = cuda_test_fatbin(cuda_test_cubin(arguments, 0, conflicts));
    merged = cuda_test_fatbin(image);
    merged.insert(merged.end(), conflicting.begin() + 16, conflicting.end());
    image_store(merged, 8, std::uint64_t{merged.size() - 16});
    expect(!inspect_cuda_module(merged, 70), "selected members cannot disagree on object sizes");
}
void private_info_link() {
    const std::array<CudaImageArgument, 3> layout{{{0, 8}, {8, 4}, {16, 16}}};
    const auto image = cuda_test_cubin(layout);
    Elf64_Ehdr header{};
    std::memcpy(&header, image.data(), sizeof(header));
    const auto section = [&](std::size_t index) {
        Elf64_Shdr result{};
        std::memcpy(&result, image.data() + header.e_shoff + index * sizeof(result),
                    sizeof(result));
        return result;
    };
    const auto names = section(1), strings = section(2), symbols = section(3), text = section(4);
    const auto info = section(5);
    expect(symbols.sh_link == 2 && info.sh_link == 3 && info.sh_info == 4,
           "independent fixture keeps ordinary ELF and CUDA-private links distinct");
    const auto complete = [&](const auto& metadata) {
        expect(metadata && metadata->has_complete_kernel_metadata &&
                   metadata->kernels.size() == 1 && metadata->kernels[0].name == "kernel" &&
                   metadata->kernels[0].arguments == std::vector(layout.begin(), layout.end()) &&
                   !metadata->is_ptx_selected && !metadata->ptx,
               "opaque CUDA-private linkage preserves the complete ordered kernel ABI");
    };
    // These values do not participate in KPARAM parsing or symbol resolution.
    // In particular, a self-link must not be followed as a guessed link chain.
    for (const std::uint32_t link : {3U, 5U, 0U, 4U, UINT32_MAX}) {
        auto changed = image;
        auto private_info = info;
        private_info.sh_link = link;
        image_store(changed, header.e_shoff + 5 * sizeof(info), private_info);
        complete(inspect_cuda_module(changed, 70));
        complete(inspect_cuda_module(cuda_test_fatbin(changed), 70));
    }
    auto stale = image;
    auto private_info = info;
    private_info.sh_link = 5;
    image_store(stale, header.e_shoff + 5 * sizeof(info), private_info);
    for (const std::uint32_t owner : {0U, 3U, 5U, UINT32_MAX}) {
        auto changed = stale;
        auto invalid = private_info;
        invalid.sh_info = owner;
        image_store(changed, header.e_shoff + 5 * sizeof(info), invalid);
        expect(!inspect_cuda_module(changed, 70),
               "private info must still identify the symbol's owning executable section");
    }
    auto changed = stale;
    auto invalid_info = private_info;
    invalid_info.sh_type = SHT_PROGBITS;
    image_store(changed, header.e_shoff + 5 * sizeof(info), invalid_info);
    expect(!inspect_cuda_module(changed, 70),
           "private info must retain the exact CUDA section type");
    changed = stale;
    changed[names.sh_offset + info.sh_name + std::string_view{".nv.info.kernel"}.size() - 1] =
        std::byte{'x'};
    const auto missing = inspect_cuda_module(changed, 70);
    expect(missing && !missing->has_complete_kernel_metadata && missing->kernels.empty(),
           "wrong private-info name withholds layout rather than fabricating a zero-argument ABI");
    changed = stale;
    changed[names.sh_offset + text.sh_name + std::string_view{".text.kernel"}.size() - 1] =
        std::byte{'x'};
    expect(!inspect_cuda_module(changed, 70), "symbol and executable section names must agree");
    changed = stale;
    auto invalid_text = text;
    invalid_text.sh_flags = 0;
    image_store(changed, header.e_shoff + 4 * sizeof(text), invalid_text);
    expect(!inspect_cuda_module(changed, 70), "kernel authority still requires executable text");
    for (const std::uint32_t link : {0U, 4U, 5U, UINT32_MAX}) {
        changed = stale;
        auto invalid_symbols = symbols;
        invalid_symbols.sh_link = link;
        image_store(changed, header.e_shoff + 3 * sizeof(symbols), invalid_symbols);
        expect(!inspect_cuda_module(changed, 70),
               "generic ELF symbol links still require an in-range string table");
    }
    changed = stale;
    auto invalid_strings = strings;
    invalid_strings.sh_type = SHT_PROGBITS;
    image_store(changed, header.e_shoff + 2 * sizeof(strings), invalid_strings);
    expect(!inspect_cuda_module(changed, 70), "generic symbol strings retain their section type");
    changed = stale;
    auto duplicate_header = header;
    ++duplicate_header.e_shnum;
    image_store(changed, 0, duplicate_header);
    image_store(changed, changed.size(), private_info);
    expect(!inspect_cuda_module(changed, 70),
           "duplicate private-info names cannot choose a layout");
    changed = stale;
    Elf64_Sym symbol{};
    std::memcpy(&symbol, image.data() + symbols.sh_offset + sizeof(symbol), sizeof(symbol));
    image_store(changed, symbols.sh_offset, symbol);
    expect(!inspect_cuda_module(changed, 70),
           "duplicate entry symbols cannot choose an ABI winner");
    changed = stale;
    invalid_info = private_info;
    --invalid_info.sh_size;
    image_store(changed, header.e_shoff + 5 * sizeof(info), invalid_info);
    expect(!inspect_cuda_module(changed, 70), "truncated parameter records remain invalid");
    expect(!inspect_cuda_module(std::span{stale}.first(stale.size() - 1), 70),
           "opaque linkage does not waive a truncated ELF section table");
}
void synthetic() {
    const std::array<CudaImageArgument, 3> layout{{{0, 8}, {8, 4}, {16, 16}}};
    const auto image = cuda_test_cubin(layout);
    const auto metadata = inspect_cuda_module(image, 70);
    expect(metadata && metadata->kernels.size() == 1 && metadata->kernels[0].name == "kernel" &&
               metadata->kernels[0].arguments == std::vector(layout.begin(), layout.end()),
           "real ELF metadata ordering preserves gaps, sizes and argument ordinals");
    expect(cuda_module_image_extent(image) == image.size(), "derive complete owned ELF extent");
    expect(inspect_cuda_module(cuda_test_cubin({}), 70)->kernels[0].arguments.empty(),
           "explicit metadata proves zero arguments");
    for (std::size_t bytes = 0; bytes < image.size(); ++bytes)
        expect(!inspect_cuda_module(std::span{image}.first(bytes), 70),
               "every truncated ELF rejects");
    auto changed = image;
    Elf64_Ehdr header{};
    std::memcpy(&header, image.data(), sizeof(header));
    header.e_shoff = UINT64_MAX - 32;
    image_store(changed, 0, header);
    expect(!cuda_module_table_extent(changed), "overflowing section extent never allocates");
    changed = image;
    std::memcpy(&header, image.data(), sizeof(header));
    header.e_shnum = 0;
    image_store(changed, 0, header);
    expect(!inspect_cuda_module(changed, 70), "extended/absent section counts are not guessed");
    const auto fatbin = cuda_test_fatbin(image, ".version 7.0\n.target sm_70\n.address_size 64\n");
    const auto combined = inspect_cuda_module(fatbin, 70);
    expect(combined && combined->kernels.size() == 1 && combined->ptx && !combined->is_ptx_selected,
           "matching native and PTX members retain exact native ABI");
    for (std::size_t bytes = 0; bytes < fatbin.size(); ++bytes)
        expect(!inspect_cuda_module(std::span{fatbin}.first(bytes), 70),
               "every truncated container rejects");
    const auto ptx = inspect_cuda_module(cuda_test_fatbin({}, ".version 7.0\n"), 70);
    expect(ptx && ptx->ptx && ptx->is_ptx_selected, "PTX-only containers preserve selected source");
    const std::string stripped = "\n.target sm_70\n.address_size 64\n";
    auto stripped_image = cuda_test_fatbin({}, stripped);
    image_store(stripped_image, 16 + 24, std::uint16_t{4});
    image_store(stripped_image, 16 + 26, std::uint16_t{7});
    const auto restored = inspect_cuda_module(stripped_image, 70);
    expect(restored && restored->ptx == ".version 7.4\n" + stripped &&
               restored->validation_ptx.size() == 1 &&
               restored->validation_ptx.front() == *restored->ptx,
           "restore stripped source version from the exact member major/minor, not architecture");
    auto no_header = cuda_test_fatbin({}, stripped);
    expect(!inspect_cuda_module(no_header, 70),
           "missing version cannot be invented for stripped PTX");
    for (const auto& version : {std::pair{0U, 4U}, std::pair{100U, 4U}, std::pair{7U, 10U}}) {
        auto invalid = stripped_image;
        image_store(invalid, 16 + 26, static_cast<std::uint16_t>(version.first));
        image_store(invalid, 16 + 24, static_cast<std::uint16_t>(version.second));
        expect(!inspect_cuda_module(invalid, 70),
               "invalid header version rejects before publication");
    }
    auto explicit_image = cuda_test_fatbin({}, ".version 7.0\n.target sm_70\n");
    image_store(explicit_image, 16 + 24, std::uint16_t{4});
    image_store(explicit_image, 16 + 26, std::uint16_t{7});
    expect(inspect_cuda_module(explicit_image, 70)->ptx == ".version 7.0\n.target sm_70\n",
           "explicit version source remains byte-identical");
    auto limited = CudaImageLimits{};
    limited.parameters = 2;
    expect(!inspect_cuda_module(image, 70, limited), "arity cannot escape configured bounds");
    limited = {};
    limited.image_bytes = image.size() - 1;
    expect(!inspect_cuda_module(image, 70, limited), "image size obeys configured bounds");
    // Synthetic CPU-only code bytes exercise real section extent, not merely
    // trailing padding. No real SDK acceptance is implied by this fixture.
    const auto large = cuda_test_cubin(layout, std::size_t{17} * 1024 * 1024);
    expect(cuda_module_image_extent(large) == large.size() && inspect_cuda_module(large, 70),
           "bounded executable inspection accepts actual extents above the inline message cap");
    limited = {};
    limited.image_bytes = std::size_t{16} * 1024 * 1024;
    expect(!cuda_module_image_extent(large, limited) && !inspect_cuda_module(large, 70, limited),
           "caller-specific inspection quotas remain enforced for large images");
    auto boundary = fatbin;
    image_store(boundary, 8, std::uint64_t{kMaxCudaModuleBytes - 16});
    expect(cuda_module_image_extent(boundary) == kMaxCudaModuleBytes &&
               !inspect_cuda_module(boundary, 70),
           "exact per-image extent is bounded before reading the missing claimed bytes");
    image_store(boundary, 8, std::uint64_t{kMaxCudaModuleBytes - 15});
    expect(!cuda_module_image_extent(boundary), "one byte beyond the hard image cap rejects");
    std::memcpy(&header, image.data(), sizeof(header));
    Elf64_Shdr info{};
    std::memcpy(&info, image.data() + header.e_shoff + 5 * sizeof(info), sizeof(info));
    for (unsigned int mutation = 0; mutation < 5; ++mutation) {
        changed = image;
        if (mutation == 0) image_store(changed, info.sh_offset + 12, std::uint16_t{0});
        if (mutation == 1) image_store(changed, info.sh_offset + 12, UINT16_MAX);
        if (mutation == 2) image_store(changed, info.sh_offset + 14, std::uint16_t{8});
        if (mutation == 3) image_store(changed, info.sh_offset + 16, std::uint32_t{0x1f000});
        if (mutation == 4) image_store(changed, info.sh_offset + 2, std::uint16_t{31});
        expect(!inspect_cuda_module(changed, 70),
               "duplicate/gapped ordinals, overlap, zero size and bank underrun reject");
    }
    changed = fatbin;
    image_store(changed, 16 + 28, std::uint32_t{80});
    expect(!inspect_cuda_module(changed, 70), "member architecture cannot disagree with its ELF");
    auto conflict = cuda_test_fatbin(cuda_test_cubin(std::array<CudaImageArgument, 1>{{{0, 4}}}));
    changed = cuda_test_fatbin(image);
    changed.insert(changed.end(), conflict.begin() + 16, conflict.end());
    image_store(changed, 8, std::uint64_t{changed.size() - 16});
    expect(!inspect_cuda_module(changed, 70),
           "ambiguous selected-image ABI cannot select a winner");
    // An independently built literal-only LZ4 member exercises the decoder
    // without depending on a compressor or accepting a malformed ELF first.
    const std::string source = ".version 7.0\n";
    std::vector<std::byte> compressed(16 + 64 + 16);
    image_store(compressed, 0, std::uint32_t{0xba55ed50});
    image_store(compressed, 4, std::uint16_t{1});
    image_store(compressed, 6, std::uint16_t{16});
    image_store(compressed, 8, std::uint64_t{compressed.size() - 16});
    image_store(compressed, 16, std::uint16_t{1});
    image_store(compressed, 18, std::uint16_t{0x101});
    image_store(compressed, 20, std::uint32_t{64});
    image_store(compressed, 24, std::uint64_t{16});
    image_store(compressed, 32, static_cast<std::uint32_t>(source.size() + 1));
    image_store(compressed, 44, std::uint32_t{70});
    image_store(compressed, 56, std::uint64_t{0x2011});
    image_store(compressed, 72, static_cast<std::uint64_t>(source.size()));
    compressed[80] = static_cast<std::byte>(source.size() << 4);
    std::memcpy(compressed.data() + 81, source.data(), source.size());
    const auto decoded = inspect_cuda_module(compressed, 70);
    expect(decoded && decoded->ptx == source, "decode exact literal-only compressed PTX");
    for (unsigned int mutation = 0; mutation < 4; ++mutation) {
        changed = compressed;
        if (mutation == 0) image_store(changed, 72, std::uint64_t{UINT64_MAX});
        if (mutation == 1) image_store(changed, 32, std::uint32_t{17});
        if (mutation == 2) changed[80] = std::byte{0xf0};  // Missing literal bytes.
        if (mutation == 3) changed[80] = std::byte{};      // Invalid backward distance.
        expect(!inspect_cuda_module(changed, 70),
               "decoded-size bomb, payload overflow and malformed LZ4 reject");
    }
}
void compiler(const char* filename) {
    std::ifstream stream(filename, std::ios::binary | std::ios::ate);
    expect(static_cast<bool>(stream), "open actual compiler-produced image");
    const auto count = stream.tellg();
    expect(count > 0 && count <= static_cast<std::streamoff>(kMaxCudaModuleBytes),
           "bounded compiler image");
    std::vector<std::byte> bytes(static_cast<std::size_t>(count));
    stream.seekg(0);
    expect(static_cast<bool>(stream.read(reinterpret_cast<char*>(bytes.data()), count)),
           "read full compiler image");
    const auto result = inspect_cuda_module(bytes, 70);
    if (!result) throw std::runtime_error("expected compiler module metadata");
    expect(result && result->kernels.size() == 3, "parse actual nvcc ELF and compressed fatbinary");
    const auto found =
        std::ranges::find(result->kernels, std::string{"binary_add"}, &CudaImageKernel::name);
    expect(found != result->kernels.end() &&
               found->arguments == std::vector<CudaImageArgument>{{0, 8}, {8, 4}, {16, 16}},
           "cuobjdump-qualified compiler layout");
}
}  // namespace
int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view{argv[1]} == "--private-info-link") {
            private_info_link();
            std::cout << "CUDA-private info linkage and ABI authority checks passed\n";
            return EXIT_SUCCESS;
        }
        symbol_metadata();
        private_info_link();
        synthetic();
        independent_preparation_budgets();
        padded_source_retention();
        compact_arity_facts();
        for (int index = 1; index < argc; ++index) compiler(argv[index]);
        std::cout << "Bounded CUDA module image and ABI metadata checks passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
