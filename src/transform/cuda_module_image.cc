#include "glimmer/transform/cuda_module_image.h"

#include <algorithm>
#include <bit>
#include <cctype>
#include <cstring>
#include <limits>
#include <map>
#include <string_view>
#include <elf.h>

namespace glimmer::transform {
namespace {
constexpr std::uint32_t kFatbinMagic = 0xba55ed50;
constexpr std::uint32_t kCudaInfo = 0x70000000;
constexpr unsigned int kKernelParameter = 0x17, kParameterBankSize = 0x19;
constexpr unsigned char kCudaEntry = 0x10;
static_assert(sizeof(Elf64_Ehdr) == 64 && sizeof(Elf64_Shdr) == 64 && sizeof(Elf64_Phdr) == 56 &&
              sizeof(Elf64_Sym) == 24);

bool range(std::uint64_t offset, std::uint64_t bytes, std::size_t limit) noexcept {
    return offset <= limit && bytes <= limit - offset;
}
struct WorkBudget {
    std::size_t remaining{};
    bool is_exhausted{};

    bool take(std::size_t bytes) noexcept {
        if (bytes > remaining) {
            is_exhausted = true;
            return false;
        }
        remaining -= bytes;
        return true;
    }
};
bool retain(CudaModuleMetadata& metadata, std::size_t bytes,
            const CudaImageLimits& limits) noexcept {
    if (!range(metadata.retained_bytes, bytes, limits.retained_bytes)) return false;
    metadata.retained_bytes += bytes;
    return true;
}
template <typename T>
std::optional<T> record(std::span<const std::byte> bytes, std::size_t offset) noexcept {
    if (!range(offset, sizeof(T), bytes.size())) return std::nullopt;
    T value{};
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    return value;
}
std::uint64_t integer(std::span<const std::byte> bytes) noexcept {
    std::uint64_t value{};
    for (std::size_t index = 0; index < bytes.size(); ++index)
        value |= std::uint64_t{std::to_integer<unsigned char>(bytes[index])} << (8 * index);
    return value;
}
bool is_fatbin(std::span<const std::byte> bytes) noexcept {
    return bytes.size() >= 4 && integer(bytes.first(4)) == kFatbinMagic;
}
std::optional<Elf64_Ehdr> elf_header(std::span<const std::byte> bytes,
                                     const CudaImageLimits& limits) noexcept {
    auto header = record<Elf64_Ehdr>(bytes, 0);
    if (!header || std::endian::native != std::endian::little ||
        std::memcmp(header->e_ident, ELFMAG, SELFMAG) != 0 ||
        header->e_ident[EI_CLASS] != ELFCLASS64 || header->e_ident[EI_DATA] != ELFDATA2LSB ||
        header->e_ident[EI_VERSION] != EV_CURRENT || header->e_machine != EM_CUDA ||
        header->e_type != ET_EXEC || header->e_ehsize != sizeof(Elf64_Ehdr) || !header->e_shnum ||
        header->e_shnum > limits.sections || header->e_shentsize != sizeof(Elf64_Shdr) ||
        header->e_shstrndx == SHN_UNDEF || header->e_shstrndx >= header->e_shnum ||
        header->e_shoff < sizeof(Elf64_Ehdr) || header->e_phnum == PN_XNUM ||
        header->e_phnum > limits.sections ||
        (header->e_phnum &&
         (header->e_phentsize != sizeof(Elf64_Phdr) || header->e_phoff < sizeof(Elf64_Ehdr))))
        return std::nullopt;
    return header;
}
std::optional<std::string_view> name_at(std::span<const std::byte> strings, std::size_t offset,
                                        WorkBudget& work) noexcept {
    if (offset >= strings.size()) return std::nullopt;
    // A name longer than 4096 was already invalid. Bound the search itself,
    // including repeated references into a large or overlapping string table.
    const auto tail = strings.subspan(offset, std::min<std::size_t>(4097, strings.size() - offset));
    if (!work.take(tail.size())) return std::nullopt;
    const auto end = std::find(tail.begin(), tail.end(), std::byte{});
    if (end == tail.end() || end - tail.begin() > 4096) return std::nullopt;
    return std::string_view(reinterpret_cast<const char*>(tail.data()),
                            static_cast<std::size_t>(end - tail.begin()));
}
std::optional<std::vector<CudaImageArgument>> parameters(std::span<const std::byte> bytes,
                                                         const CudaImageLimits& limits) {
    struct Parameter {
        std::size_t ordinal{};
        CudaImageArgument layout;
    };
    std::vector<Parameter> parsed;
    std::optional<std::size_t> bank_size;
    while (!bytes.empty()) {
        if (bytes.size() < 4) return std::nullopt;
        const auto format = std::to_integer<unsigned int>(bytes[0]);
        const auto attribute = std::to_integer<unsigned int>(bytes[1]);
        const auto size = integer(bytes.subspan(2, 2));
        if (format < 1 || format > 4) return std::nullopt;
        const auto payload = format == 4 ? size : 0;
        if (payload > bytes.size() - 4) return std::nullopt;
        if (attribute == kKernelParameter) {
            if (format != 4 || size != 12 || parsed.size() == limits.parameters)
                return std::nullopt;
            const auto body = bytes.subspan(4, 12);
            const auto ordinal = integer(body.subspan(4, 2));
            const auto offset = integer(body.subspan(6, 2));
            // CUDA ELF KPARAM's upper 14 descriptor bits encode byte size.
            const auto argument_bytes = integer(body.subspan(8, 4)) >> 18;
            if (!argument_bytes || ordinal >= limits.parameters ||
                !range(offset, argument_bytes, limits.parameter_bytes))
                return std::nullopt;
            parsed.push_back(
                {static_cast<std::size_t>(ordinal),
                 {static_cast<std::size_t>(offset), static_cast<std::size_t>(argument_bytes)}});
        } else if (attribute == kParameterBankSize) {
            if (format != 3 || bank_size || size > limits.parameter_bytes) return std::nullopt;
            bank_size = size;
        }
        bytes = bytes.subspan(4 + payload);
    }
    std::ranges::sort(parsed, {}, &Parameter::ordinal);
    std::vector<CudaImageArgument> result;
    std::size_t end{};
    for (std::size_t index = 0; index < parsed.size(); ++index) {
        const auto& item = parsed[index];
        if (item.ordinal != index || item.layout.offset < end) return std::nullopt;
        end = item.layout.offset + item.layout.bytes;
        result.push_back(item.layout);
    }
    if ((!result.empty() && (!bank_size || end > *bank_size)) ||
        (result.empty() && bank_size.value_or(0) != 0))
        return std::nullopt;
    return result;
}
std::optional<CudaModuleMetadata> inspect_elf(std::span<const std::byte> image,
                                              const CudaImageLimits& limits, WorkBudget& work) {
    const auto extent = cuda_module_image_extent(image, limits);
    const auto header = elf_header(image, limits);
    if (!extent || *extent > image.size() || !header) return std::nullopt;
    std::vector<Elf64_Shdr> sections;
    for (std::size_t index = 0; index < header->e_shnum; ++index)
        sections.push_back(
            *record<Elf64_Shdr>(image, header->e_shoff + index * sizeof(Elf64_Shdr)));
    const auto& names = sections[header->e_shstrndx];
    if (names.sh_type != SHT_STRTAB) return std::nullopt;
    const auto section_strings = image.subspan(names.sh_offset, names.sh_size);
    std::vector<std::string_view> section_names;
    std::map<std::string_view, std::size_t> information;
    std::size_t static_storage_bytes{};
    for (const auto& section : sections) {
        const auto name = name_at(section_strings, section.sh_name, work);
        if (!name) return std::nullopt;
        section_names.push_back(*name);
        const bool is_static =
            *name == ".nv.global" || *name == ".nv.global.init" ||
            (name->starts_with(".nv.constant") && !name->starts_with(".nv.constant0."));
        if (is_static && (section.sh_flags & SHF_ALLOC) && section.sh_size) {
            const auto alignment = std::max<std::uint64_t>(1, section.sh_addralign);
            const auto maximum = std::numeric_limits<std::size_t>::max();
            if ((alignment & (alignment - 1)) || alignment - 1 > maximum - static_storage_bytes ||
                section.sh_size > maximum - static_storage_bytes - (alignment - 1))
                return std::nullopt;
            static_storage_bytes += section.sh_size + alignment - 1;
        }
        if (name->starts_with(".nv.info.") &&
            !information.emplace(*name, section_names.size() - 1).second)
            return std::nullopt;
    }
    CudaModuleMetadata result;
    result.static_storage_bytes = static_storage_bytes;
    std::size_t candidates{};
    for (std::size_t table_index = 0; table_index < sections.size(); ++table_index) {
        const auto& table = sections[table_index];
        if (table.sh_type != SHT_SYMTAB) continue;
        if (table.sh_link >= sections.size() || table.sh_entsize != sizeof(Elf64_Sym) ||
            table.sh_size % sizeof(Elf64_Sym))
            return std::nullopt;
        const auto& string_section = sections[table.sh_link];
        if (string_section.sh_type != SHT_STRTAB) return std::nullopt;
        if (!work.take(table.sh_size)) return std::nullopt;
        const auto strings = image.subspan(string_section.sh_offset, string_section.sh_size);
        for (std::size_t index = 0; index < table.sh_size / sizeof(Elf64_Sym); ++index) {
            const auto symbol =
                *record<Elf64_Sym>(image, table.sh_offset + index * sizeof(Elf64_Sym));
            if (ELF64_ST_TYPE(symbol.st_info) == STT_OBJECT && symbol.st_shndx >= sections.size() &&
                symbol.st_shndx < SHN_LORESERVE)
                return std::nullopt;
            if (ELF64_ST_TYPE(symbol.st_info) == STT_OBJECT && symbol.st_shndx != SHN_UNDEF &&
                symbol.st_shndx < sections.size()) {
                const auto section = section_names[symbol.st_shndx];
                if (section == ".nv.global" || section == ".nv.global.init" ||
                    section.starts_with(".nv.constant")) {
                    const auto name = name_at(strings, symbol.st_name, work);
                    if (!name ||
                        !range(symbol.st_value, symbol.st_size, sections[symbol.st_shndx].sh_size))
                        return std::nullopt;
                    if (name->empty() || !symbol.st_size) {
                        result.has_complete_variable_metadata = false;
                        continue;  // Nonqueryable objects do not reject original binary intake.
                    }
                    if (result.variables.size() == limits.variables ||
                        std::ranges::any_of(result.variables,
                                            [&](const auto& item) { return item.name == *name; }))
                        return std::nullopt;
                    if (!retain(result, name->size() + 1 + sizeof(CudaImageVariable), limits))
                        return std::nullopt;
                    result.variables.push_back(
                        {std::string(*name), static_cast<std::size_t>(symbol.st_size)});
                }
            }
            if (ELF64_ST_TYPE(symbol.st_info) != STT_FUNC || !(symbol.st_other & kCudaEntry))
                continue;
            if (++candidates > limits.kernels) return std::nullopt;
            if (symbol.st_shndx >= sections.size()) return std::nullopt;
            const auto name = name_at(strings, symbol.st_name, work);
            if (!name || name->empty()) return std::nullopt;
            const auto& text = sections[symbol.st_shndx];
            if (!(text.sh_flags & SHF_EXECINSTR) ||
                section_names[symbol.st_shndx] != std::string(".text.") + std::string(*name))
                return std::nullopt;
            const auto info_name = std::string(".nv.info.") + std::string(*name);
            const auto metadata = information.find(info_name);
            if (metadata == information.end()) {
                result.has_complete_kernel_metadata = false;  // Missing layout is not zero arity.
                continue;
            }
            const auto& info = sections[metadata->second];
            // CUDA-private sh_link is opaque: parameter parsing never uses it
            // to resolve an address or establish kernel ABI authority.
            if (info.sh_type != kCudaInfo || info.sh_info != symbol.st_shndx) return std::nullopt;
            if (!work.take(info.sh_size)) return std::nullopt;
            auto layout = parameters(image.subspan(info.sh_offset, info.sh_size), limits);
            if (!layout || result.kernels.size() == limits.kernels ||
                std::ranges::any_of(result.kernels,
                                    [&](const auto& item) { return item.name == *name; }))
                return std::nullopt;
            if (!retain(result,
                        name->size() + 1 + sizeof(CudaImageKernel) +
                            layout->size() * sizeof(CudaImageArgument),
                        limits))
                return std::nullopt;
            result.kernels.push_back({std::string(*name), std::move(*layout)});
        }
    }
    return result;
}
// Adapt the existing fixed-image decoder's bounded overlap-safe algorithm.
// The caller authorizes the single decoded buffer and cumulative work separately.
bool decompress(std::span<const std::byte> input, std::span<std::byte> output) noexcept {
    std::size_t source{}, destination{};
    auto length = [&](std::size_t value) -> std::optional<std::size_t> {
        if (value == 15) {
            unsigned int extra{};
            do {
                if (source == input.size()) return std::nullopt;
                extra = std::to_integer<unsigned int>(input[source++]);
                if (extra > output.size() || value > output.size() - extra) return std::nullopt;
                value += extra;
            } while (extra == 255);
        }
        return value;
    };
    while (source < input.size()) {
        const auto token = std::to_integer<unsigned int>(input[source++]);
        const auto literals = length(token >> 4);
        if (!literals || *literals > input.size() - source ||
            *literals > output.size() - destination)
            return false;
        if (*literals) std::memcpy(output.data() + destination, input.data() + source, *literals);
        source += *literals;
        destination += *literals;
        if (source == input.size()) return destination == output.size();
        if (input.size() - source < 2) return false;
        const auto distance = integer(input.subspan(source, 2));
        source += 2;
        if (!distance || distance > destination) return false;
        const auto count = length(token & 15);
        if (!count || output.size() - destination < 4 || *count > output.size() - destination - 4)
            return false;
        for (std::size_t index = 0; index < *count + 4; ++index) {
            output[destination] = output[destination - distance];
            ++destination;
        }
    }
    return false;
}
struct Member {
    unsigned int kind{}, architecture{};
    std::span<const std::byte> payload;
    std::size_t decoded{};
    bool is_compressed{};
    unsigned int ptx_major{}, ptx_minor{};
};
std::size_t expanded_bytes(const Member& member) noexcept {
    return member.is_compressed ? member.decoded : member.payload.size();
}
std::optional<std::vector<std::byte>> member_bytes(const Member& member, WorkBudget& work) {
    if (!work.take(expanded_bytes(member))) return std::nullopt;
    if (!member.is_compressed)
        return std::vector<std::byte>(member.payload.begin(), member.payload.end());
    std::vector<std::byte> result(member.decoded);
    if (!decompress(member.payload, result)) return std::nullopt;
    return result;
}
// This lexical convention intentionally matches the session's legacy arity
// lookup, including ignored comments/strings and malformed first definitions.
// The outer pass is prepaid in full; copied parameter cursors charge each byte
// before advancing, so pathological nested definitions cannot evade the work cap.
class ArityTokens {
   public:
    explicit ArityTokens(std::string_view source) noexcept : source_(source) {}
    void budget(WorkBudget& work) noexcept {
        work_ = &work;
    }
    [[nodiscard]] std::string_view next() noexcept {
        while (position_ < source_.size()) {
            const auto value = source_[position_];
            if (std::isspace(static_cast<unsigned char>(value))) {
                if (!advance()) return {};
                continue;
            }
            if (source_.substr(position_, 2) == "//") {
                while (position_ < source_.size()) {
                    const auto current = source_[position_];
                    if (!advance()) return {};
                    if (current == '\n') break;
                }
                continue;
            }
            if (source_.substr(position_, 2) == "/*") {
                if (!advance(2)) return {};
                while (position_ < source_.size() && source_.substr(position_, 2) != "*/")
                    if (!advance()) return {};
                if (position_ < source_.size() && !advance(2)) return {};
                continue;
            }
            if (value == '"') {
                if (!advance()) return {};
                while (position_ < source_.size()) {
                    const auto current = source_[position_];
                    if (!advance()) return {};
                    if (current == '"') break;
                    if (current == '\\' && position_ < source_.size() && !advance()) return {};
                }
                continue;
            }
            const auto begin = position_;
            if (!advance()) return {};
            constexpr std::string_view punctuation = "()[]{},;";
            if (punctuation.find(value) == std::string_view::npos)
                while (position_ < source_.size() &&
                       !std::isspace(static_cast<unsigned char>(source_[position_])) &&
                       punctuation.find(source_[position_]) == std::string_view::npos &&
                       source_[position_] != '"' && source_.substr(position_, 2) != "//" &&
                       source_.substr(position_, 2) != "/*")
                    if (!advance()) return {};
            return source_.substr(begin, position_ - begin);
        }
        return {};
    }

   private:
    bool advance(std::size_t bytes = 1) noexcept {
        if (work_ && !work_->take(bytes)) return false;
        position_ += bytes;
        return true;
    }
    std::string_view source_;
    std::size_t position_{};
    WorkBudget* work_{};
};
std::optional<std::size_t> arity(ArityTokens tokens, std::size_t maximum,
                                 WorkBudget& work) noexcept {
    tokens.budget(work);
    if (tokens.next() != "(") return std::nullopt;
    std::size_t depth = 1, count{};
    bool is_next_parameter = true;
    for (auto token = tokens.next(); !token.empty(); token = tokens.next()) {
        if (token == "(") {
            ++depth;
            continue;
        }
        if (token == ")") {
            if (--depth == 0)
                return is_next_parameter && count ? std::nullopt : std::optional{count};
            continue;
        }
        if (depth != 1) continue;
        if (token == ",") {
            if (is_next_parameter) return std::nullopt;
            is_next_parameter = true;
        }
        if (token == ".param") {
            if (!is_next_parameter || count == maximum) return std::nullopt;
            ++count;
            is_next_parameter = false;
        }
    }
    return std::nullopt;
}
bool collect_arities(std::string_view source, CudaModuleMetadata& metadata,
                     const CudaImageLimits& limits, WorkBudget& work) {
    if (!work.take(source.size())) return false;
    // Views are valid only during this source. Tombstones must not leak into
    // another candidate's first-definition decision, nor copy unbounded names.
    std::map<std::string_view, bool> seen;
    std::size_t scratch_bytes{};
    ArityTokens tokens(source);
    for (auto token = tokens.next(); !token.empty(); token = tokens.next()) {
        if (token != ".entry") continue;
        const auto name = tokens.next();
        if (name.empty() || name.size() > 4096 || seen.contains(name)) continue;
        constexpr auto node_bytes =
            sizeof(std::pair<const std::string_view, bool>) + 4 * sizeof(void*);
        if (seen.size() == limits.kernels ||
            !range(scratch_bytes, node_bytes, limits.retained_bytes) ||
            !range(metadata.retained_bytes, scratch_bytes + node_bytes, limits.retained_bytes))
            return false;
        scratch_bytes += node_bytes;
        seen.emplace(name, true);
        const auto count = arity(tokens, limits.parameters, work);
        if (work.is_exhausted) return false;
        if (!count) continue;
        const auto existing =
            std::ranges::lower_bound(metadata.ptx_arities, name, {}, &CudaImagePtxArity::name);
        if (existing != metadata.ptx_arities.end() && existing->name == name) {
            existing->has_conflicting_counts |= existing->parameter_count != *count;
            continue;
        }
        const auto bytes = name.size() + 1 + sizeof(CudaImagePtxArity);
        if (metadata.ptx_arities.size() == limits.kernels ||
            !range(metadata.retained_bytes, bytes, limits.retained_bytes) ||
            !range(metadata.retained_bytes + bytes, scratch_bytes, limits.retained_bytes) ||
            !retain(metadata, bytes, limits))
            return false;
        metadata.ptx_arities.insert(existing, {std::string(name), *count, false});
    }
    return true;
}
std::optional<std::string> member_text(const Member& member, const CudaImageLimits& limits,
                                       WorkBudget& work) {
    const auto expanded = expanded_bytes(member);
    if (!work.take(expanded)) return std::nullopt;
    std::string source(expanded, '\0');
    auto decode = [&](std::span<char> destination) {
        auto bytes = std::as_writable_bytes(destination);
        if (member.is_compressed) return decompress(member.payload, bytes);
        std::memcpy(bytes.data(), member.payload.data(), member.payload.size());
        return true;
    };
    if (!decode(source) || !work.take(expanded)) return std::nullopt;
    const auto end = source.find('\0');
    const auto length = end == std::string::npos ? source.size() : end;
    if (!length ||
        (end != std::string::npos && source.find_first_not_of('\0', end) != std::string::npos))
        return std::nullopt;
    source.resize(length);
    if (!work.take(length)) return std::nullopt;
    const auto first = source.find_first_not_of(" \t\r\n");
    if (first != std::string::npos &&
        std::string_view(source).substr(first).starts_with(".target")) {
        // The version comes only from the original record. Re-decode into the
        // normalized buffer if growth is necessary; never hold two full decoded
        // members concurrently merely to prepend the small version directive.
        if (!member.ptx_major || member.ptx_major > 99 || member.ptx_minor > 9) return std::nullopt;
        const auto prefix = ".version " + std::to_string(member.ptx_major) + "." +
                            std::to_string(member.ptx_minor) + "\n";
        if (!range(length, prefix.size(), limits.decoded_bytes)) return std::nullopt;
        if (length + prefix.size() <= source.capacity()) {
            if (!work.take(length + prefix.size())) return std::nullopt;
            source.resize(length + prefix.size());
            std::memmove(source.data() + prefix.size(), source.data(), length);
        } else {
            if (!range(expanded, prefix.size(), limits.decoded_bytes) ||
                !work.take(expanded + prefix.size()))
                return std::nullopt;
            std::string{}.swap(source);
            source.resize(expanded + prefix.size());
            if (!decode(std::span(source).subspan(prefix.size(), expanded))) return std::nullopt;
            source.resize(length + prefix.size());
        }
        std::memcpy(source.data(), prefix.data(), prefix.size());
    }
    return source;
}
std::optional<CudaModuleMetadata> inspect_fatbin(std::span<const std::byte> image,
                                                 unsigned int target, const CudaImageLimits& limits,
                                                 WorkBudget& work) {
    const auto extent = cuda_module_image_extent(image, limits);
    if (!extent || *extent != image.size()) return std::nullopt;
    std::vector<Member> members;
    std::size_t position = 16;
    auto decoded_limits = limits;
    decoded_limits.image_bytes = limits.decoded_bytes;
    unsigned int selected_cubin{}, selected_ptx{};
    while (position < image.size()) {
        if (image.size() - position < 64 || members.size() == 1024) return std::nullopt;
        const auto header = image.subspan(position);
        const auto kind = integer(header.first(2)), version = integer(header.subspan(2, 2));
        const auto header_bytes = integer(header.subspan(4, 4)),
                   payload_bytes = integer(header.subspan(8, 8));
        if ((kind != 1 && kind != 2) || version != 0x101 || header_bytes < 64 ||
            header_bytes > 4096 || !range(position, header_bytes, image.size()) || !payload_bytes ||
            !range(position + header_bytes, payload_bytes, image.size()))
            return std::nullopt;
        const auto architecture = integer(header.subspan(28, 4)),
                   flags = integer(header.subspan(40, 8));
        if (!(flags & 1) || (flags & ~std::uint64_t{0x2011}) || architecture > 999)
            return std::nullopt;
        Member member{static_cast<unsigned int>(kind), static_cast<unsigned int>(architecture),
                      image.subspan(position + header_bytes, payload_bytes), 0,
                      (flags & 0x2000) != 0};
        if (kind == 1) {
            member.ptx_minor = static_cast<unsigned int>(integer(header.subspan(24, 2)));
            member.ptx_major = static_cast<unsigned int>(integer(header.subspan(26, 2)));
        }
        if (member.is_compressed) {
            const auto compressed = integer(header.subspan(16, 4)),
                       decoded = integer(header.subspan(56, 8));
            if (!compressed || compressed > payload_bytes || !decoded ||
                decoded > limits.decoded_bytes)
                return std::nullopt;
            member.payload = member.payload.first(compressed);
            member.decoded = decoded;
        }
        if (expanded_bytes(member) > limits.decoded_bytes) return std::nullopt;
        if (kind == 2) {
            const auto bytes = member_bytes(member, work);
            if (!bytes) return std::nullopt;
            const auto elf = elf_header(*bytes, decoded_limits);
            if (!elf || (elf->e_flags & 0xff) != architecture) return std::nullopt;
        }
        if (kind == 2 && architecture <= target && architecture / 10 == target / 10)
            selected_cubin = std::max(selected_cubin, static_cast<unsigned int>(architecture));
        if (kind == 1 && architecture <= target)
            selected_ptx = std::max(selected_ptx, static_cast<unsigned int>(architecture));
        members.push_back(member);
        position += header_bytes + payload_bytes;
    }
    if (!selected_cubin && !selected_ptx) return std::nullopt;
    CudaModuleMetadata result;
    for (const auto& member : members) {
        if (member.kind != 2 || member.architecture != selected_cubin) continue;
        const auto bytes = member_bytes(member, work);
        if (!bytes) return std::nullopt;
        const auto header = elf_header(*bytes, decoded_limits);
        if (!header || (header->e_flags & 0xff) != member.architecture) return std::nullopt;
        if (!work.take(bytes->size())) return std::nullopt;
        auto metadata_limits = decoded_limits;
        metadata_limits.retained_bytes -= result.retained_bytes;
        auto metadata = inspect_elf(*bytes, metadata_limits, work);
        if (!metadata) return std::nullopt;
        if (metadata->static_storage_bytes >
            std::numeric_limits<std::size_t>::max() - result.static_storage_bytes)
            return std::nullopt;
        result.static_storage_bytes += metadata->static_storage_bytes;
        result.has_complete_kernel_metadata &= metadata->has_complete_kernel_metadata;
        result.has_complete_variable_metadata &= metadata->has_complete_variable_metadata;
        for (auto& kernel : metadata->kernels) {
            const auto existing =
                std::ranges::find(result.kernels, kernel.name, &CudaImageKernel::name);
            if (existing != result.kernels.end()) {
                if (existing->arguments != kernel.arguments) return std::nullopt;
            } else {
                if (result.kernels.size() == limits.kernels) return std::nullopt;
                if (!retain(result,
                            kernel.name.size() + 1 + sizeof(CudaImageKernel) +
                                kernel.arguments.size() * sizeof(CudaImageArgument),
                            limits))
                    return std::nullopt;
                result.kernels.push_back(std::move(kernel));
            }
        }
        for (auto& variable : metadata->variables) {
            const auto existing =
                std::ranges::find(result.variables, variable.name, &CudaImageVariable::name);
            if (existing != result.variables.end()) {
                if (existing->bytes != variable.bytes) return std::nullopt;
            } else {
                if (result.variables.size() == limits.variables) return std::nullopt;
                if (!retain(result, variable.name.size() + 1 + sizeof(CudaImageVariable), limits))
                    return std::nullopt;
                result.variables.push_back(std::move(variable));
            }
        }
    }
    const auto ptx_architecture = selected_cubin ? selected_cubin : selected_ptx;
    const auto companion_count = std::ranges::count_if(members, [&](const auto& member) {
        return member.kind == 1 && member.architecture == ptx_architecture;
    });
    std::optional<std::size_t> companion;
    for (const auto& member : members) {
        if (member.kind != 1 || member.architecture > target) continue;
        auto source = member_text(member, limits, work);
        if (!source) return std::nullopt;
        const bool is_companion = companion_count == 1 && member.architecture == ptx_architecture;
        if (limits.compact_ptx_arities) {
            if (!collect_arities(*source, result, limits, work)) return std::nullopt;
            if (is_companion) {
                // resize at the first NUL does not release decoded padding.
                // Moving the source retains its entire allocation capacity.
                if (!retain(result, source->capacity() + 1, limits)) return std::nullopt;
                result.ptx = std::move(*source);
            }
        } else {
            if (!retain(result, source->capacity() + 1 + sizeof(std::string), limits))
                return std::nullopt;
            result.validation_ptx.push_back(std::move(*source));
            if (is_companion) companion = result.validation_ptx.size() - 1;
        }
    }
    result.is_ptx_selected = !selected_cubin;
    if (companion) {
        const auto& source = result.validation_ptx[*companion];
        // Authorize the copy conservatively before allocating. A library that
        // grows the copy beyond the source capacity cannot publish that result.
        if (!retain(result, source.capacity() + 1, limits) || !work.take(source.size()))
            return std::nullopt;
        result.ptx = source;
        if (result.ptx->capacity() > source.capacity()) return std::nullopt;
    }
    if (result.is_ptx_selected && !result.ptx) return std::nullopt;
    return result;
}
}  // namespace

std::optional<std::size_t> cuda_module_table_extent(std::span<const std::byte> prefix,
                                                    CudaImageLimits limits) noexcept {
    if (!limits.image_bytes || limits.image_bytes > kMaxCudaModuleBytes) return std::nullopt;
    if (is_fatbin(prefix)) {
        if (prefix.size() < 16 || integer(prefix.subspan(4, 2)) != 1 ||
            integer(prefix.subspan(6, 2)) != 16)
            return std::nullopt;
        return 16;
    }
    const auto header = elf_header(prefix, limits);
    if (!header ||
        !range(header->e_shoff, std::size_t{header->e_shnum} * sizeof(Elf64_Shdr),
               limits.image_bytes) ||
        (header->e_phnum &&
         !range(header->e_phoff, std::size_t{header->e_phnum} * sizeof(Elf64_Phdr),
                limits.image_bytes)))
        return std::nullopt;
    return std::max<std::size_t>(
        header->e_shoff + std::size_t{header->e_shnum} * sizeof(Elf64_Shdr),
        header->e_phnum ? header->e_phoff + std::size_t{header->e_phnum} * sizeof(Elf64_Phdr)
                        : sizeof(Elf64_Ehdr));
}
std::optional<std::size_t> cuda_module_image_extent(std::span<const std::byte> tables,
                                                    CudaImageLimits limits) noexcept {
    const auto table_bytes = cuda_module_table_extent(tables, limits);
    if (!table_bytes || *table_bytes > tables.size()) return std::nullopt;
    if (is_fatbin(tables)) {
        const auto data = integer(tables.subspan(8, 8));
        return data && range(16, data, limits.image_bytes) ? std::optional<std::size_t>{16 + data}
                                                           : std::nullopt;
    }
    const auto header = elf_header(tables, limits);
    if (!header) return std::nullopt;
    std::size_t extent = *table_bytes;
    for (std::size_t index = 0; index < header->e_shnum; ++index) {
        const auto section =
            record<Elf64_Shdr>(tables, header->e_shoff + index * sizeof(Elf64_Shdr));
        if (!section) return std::nullopt;
        if (section->sh_type == SHT_NOBITS) continue;
        if (!range(section->sh_offset, section->sh_size, limits.image_bytes)) return std::nullopt;
        extent = std::max<std::size_t>(extent, section->sh_offset + section->sh_size);
    }
    for (std::size_t index = 0; index < header->e_phnum; ++index) {
        const auto program =
            record<Elf64_Phdr>(tables, header->e_phoff + index * sizeof(Elf64_Phdr));
        if (!program || !range(program->p_offset, program->p_filesz, limits.image_bytes))
            return std::nullopt;
        extent = std::max<std::size_t>(extent, program->p_offset + program->p_filesz);
    }
    return extent;
}
std::optional<CudaModuleMetadata> inspect_cuda_module(std::span<const std::byte> image,
                                                      unsigned int compute_capability,
                                                      CudaImageLimits limits) noexcept {
    if (image.empty() || !limits.image_bytes || limits.image_bytes > kMaxCudaModuleBytes ||
        image.size() > limits.image_bytes || !limits.decoded_bytes ||
        limits.decoded_bytes > kMaxCudaModuleBytes || !limits.retained_bytes ||
        limits.retained_bytes > kMaxCudaModuleBytes || !limits.work_bytes ||
        limits.work_bytes > kMaxCudaModuleWorkBytes || !limits.sections || !limits.kernels ||
        !limits.parameters || !limits.parameter_bytes)
        return std::nullopt;
    try {
        WorkBudget work{limits.work_bytes};
        if (is_fatbin(image)) return inspect_fatbin(image, compute_capability, limits, work);
        if (image.size() > limits.decoded_bytes || !work.take(image.size())) return std::nullopt;
        return inspect_elf(image, limits, work);
    } catch (...) {
        return std::nullopt;
    }
}
}  // namespace glimmer::transform
