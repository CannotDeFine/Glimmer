#pragma once

#include "glimmer/transform/cuda_module_image.h"

#include <array>
#include <cstring>
#include <span>
#include <string>
#include <vector>
#include <elf.h>

namespace glimmer::test {
template <typename T>
void image_store(std::vector<std::byte>& bytes, std::size_t offset, const T& value) {
    if (offset + sizeof(T) > bytes.size()) bytes.resize(offset + sizeof(T));
    std::memcpy(bytes.data() + offset, &value, sizeof(T));
}
inline std::vector<std::byte> cuda_test_cubin(
    std::span<const transform::CudaImageArgument> arguments, std::size_t extra_text_bytes = 0,
    std::span<const transform::CudaImageVariable> variables = {}) {
    std::vector<std::string> names{"",        ".shstrtab",    ".strtab",
                                   ".symtab", ".text.kernel", ".nv.info.kernel"};
    if (!variables.empty()) names.emplace_back(".nv.global");
    std::vector<std::byte> image(sizeof(Elf64_Ehdr));
    std::vector<Elf64_Shdr> sections(names.size());
    sections[1].sh_offset = image.size();
    for (std::size_t index = 0; index < names.size(); ++index) {
        sections[index].sh_name = image.size() - sections[1].sh_offset;
        for (char value : names[index])
            image.push_back(std::byte{static_cast<unsigned char>(value)});
        image.push_back(std::byte{});
    }
    sections[1].sh_size = image.size() - sections[1].sh_offset;
    sections[1].sh_type = SHT_STRTAB;
    sections[2].sh_offset = image.size();
    for (char value : std::string("\0kernel", 7))
        image.push_back(std::byte{static_cast<unsigned char>(value)});
    image.push_back(std::byte{});
    std::vector<std::size_t> variable_names;
    for (const auto& variable : variables) {
        variable_names.push_back(image.size() - sections[2].sh_offset);
        for (const auto value : variable.name)
            image.push_back(std::byte{static_cast<unsigned char>(value)});
        image.push_back(std::byte{});
    }
    sections[2].sh_size = image.size() - sections[2].sh_offset;
    sections[2].sh_type = SHT_STRTAB;
    sections[3].sh_offset = image.size();
    Elf64_Sym symbol{};
    image_store(image, image.size(), symbol);
    symbol.st_name = 1;
    symbol.st_info = ELF64_ST_INFO(STB_GLOBAL, STT_FUNC);
    symbol.st_other = 0x10;
    symbol.st_shndx = 4;
    image_store(image, image.size(), symbol);
    std::size_t variable_bytes{};
    for (std::size_t index = 0; index < variables.size(); ++index) {
        symbol = {};
        symbol.st_name = variable_names[index];
        symbol.st_info = ELF64_ST_INFO(STB_GLOBAL, STT_OBJECT);
        symbol.st_shndx = 6;
        symbol.st_value = variable_bytes;
        symbol.st_size = variables[index].bytes;
        variable_bytes += variables[index].bytes;
        image_store(image, image.size(), symbol);
    }
    sections[3].sh_size = (2 + variables.size()) * sizeof(Elf64_Sym);
    sections[3].sh_type = SHT_SYMTAB;
    sections[3].sh_entsize = sizeof(Elf64_Sym);
    sections[3].sh_link = 2;
    sections[4].sh_offset = image.size();
    sections[4].sh_size = 16 + extra_text_bytes;
    sections[4].sh_type = SHT_PROGBITS;
    sections[4].sh_flags = SHF_EXECINSTR;
    image.resize(image.size() + sections[4].sh_size);
    sections[5].sh_offset = image.size();
    sections[5].sh_type = 0x70000000;
    sections[5].sh_link = 3;
    sections[5].sh_info = 4;
    const std::uint16_t bank =
        arguments.empty() ? 0 : arguments.back().offset + arguments.back().bytes;
    image.push_back(std::byte{3});
    image.push_back(std::byte{0x19});
    image_store(image, image.size(), bank);
    for (std::size_t index = arguments.size(); index > 0; --index) {
        image.push_back(std::byte{4});
        image.push_back(std::byte{0x17});
        image_store(image, image.size(), std::uint16_t{12});
        image_store(image, image.size(), std::uint32_t{});
        image_store(image, image.size(), static_cast<std::uint16_t>(index - 1));
        image_store(image, image.size(), static_cast<std::uint16_t>(arguments[index - 1].offset));
        image_store(image, image.size(),
                    static_cast<std::uint32_t>((arguments[index - 1].bytes << 18) | 0x1f000));
    }
    sections[5].sh_size = image.size() - sections[5].sh_offset;
    if (!variables.empty()) {
        sections[6].sh_offset = image.size();
        sections[6].sh_size = variable_bytes;
        sections[6].sh_type = SHT_NOBITS;
        sections[6].sh_flags = SHF_ALLOC | SHF_WRITE;
    }
    Elf64_Ehdr header{};
    std::memcpy(header.e_ident, ELFMAG, SELFMAG);
    header.e_ident[EI_CLASS] = ELFCLASS64;
    header.e_ident[EI_DATA] = ELFDATA2LSB;
    header.e_ident[EI_VERSION] = EV_CURRENT;
    header.e_type = ET_EXEC;
    header.e_machine = EM_CUDA;
    header.e_ehsize = sizeof(header);
    header.e_shoff = image.size();
    header.e_shnum = sections.size();
    header.e_shentsize = sizeof(Elf64_Shdr);
    header.e_shstrndx = 1;
    header.e_flags = 70;
    for (const auto& section : sections) image_store(image, image.size(), section);
    image_store(image, 0, header);
    return image;
}
inline std::vector<std::byte> cuda_test_fatbin(std::span<const std::byte> cubin,
                                               std::string_view ptx = {}) {
    std::vector<std::byte> image(16);
    const auto member = [&](unsigned int kind, std::span<const std::byte> bytes) {
        const auto position = image.size();
        image.resize(position + 64);
        image_store(image, position, static_cast<std::uint16_t>(kind));
        image_store(image, position + 2, std::uint16_t{0x101});
        image_store(image, position + 4, std::uint32_t{64});
        const std::uint64_t padded = (bytes.size() + 7) & ~std::uint64_t{7};
        image_store(image, position + 8, padded);
        image_store(image, position + 28, std::uint32_t{70});
        image_store(image, position + 40, std::uint64_t{0x11});
        image.insert(image.end(), bytes.begin(), bytes.end());
        image.resize(position + 64 + padded);
    };
    if (!cubin.empty()) member(2, cubin);
    if (!ptx.empty()) member(1, std::as_bytes(std::span{ptx.data(), ptx.size()}));
    image_store(image, 0, std::uint32_t{0xba55ed50});
    image_store(image, 4, std::uint16_t{1});
    image_store(image, 6, std::uint16_t{16});
    image_store(image, 8, std::uint64_t{image.size() - 16});
    return image;
}
}  // namespace glimmer::test
