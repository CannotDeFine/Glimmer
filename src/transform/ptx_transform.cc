#include "glimmer/transform/ptx_transform.h"

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cctype>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <utility>

namespace glimmer::transform {
namespace {

constexpr std::string_view kNounrollPragma = "\"nounroll\"";

struct Token {
    std::string_view text;
    std::size_t begin = 0;
    std::size_t end = 0;
};

bool is_word(char character) {
    return std::isalnum(static_cast<unsigned char>(character)) != 0 || character == '_' ||
           character == '.' || character == '%' || character == '$';
}

std::optional<std::vector<Token>> tokenize(std::string_view source, std::size_t maximum_tokens) {
    std::vector<Token> tokens;
    for (std::size_t offset = 0; offset < source.size();) {
        const char character = source[offset];
        if (std::isspace(static_cast<unsigned char>(character)) != 0) {
            ++offset;
            continue;
        }
        if (source.substr(offset, 2) == "//") {
            const auto end = source.find('\n', offset + 2);
            offset = end == std::string_view::npos ? source.size() : end + 1;
            continue;
        }
        if (source.substr(offset, 2) == "/*") {
            const auto end = source.find("*/", offset + 2);
            if (end == std::string_view::npos) {
                return std::nullopt;
            }
            offset = end + 2;
            continue;
        }
        if (character == '\0' || static_cast<unsigned char>(character) > 127) {
            return std::nullopt;
        }
        const auto begin = offset;
        ++offset;
        if (character == '"') {
            bool is_closed{};
            while (offset < source.size()) {
                const auto value = source[offset++];
                if (value == '"') {
                    is_closed = true;
                    break;
                }
                if (value == '\0' || value == '\n' || value == '\r' ||
                    static_cast<unsigned char>(value) > 127)
                    return std::nullopt;
                if (value == '\\') {
                    if (offset == source.size() || source[offset] == '\0' ||
                        source[offset] == '\n' || source[offset] == '\r' ||
                        static_cast<unsigned char>(source[offset]) > 127)
                        return std::nullopt;
                    ++offset;
                }
            }
            if (!is_closed) return std::nullopt;
        } else if (is_word(character)) {
            while (offset < source.size() && is_word(source[offset])) {
                ++offset;
            }
        }
        if (tokens.size() == maximum_tokens) {
            return std::nullopt;
        }
        tokens.push_back(Token{source.substr(begin, offset - begin), begin, offset});
    }
    return tokens;
}

bool is_identifier(std::string_view name) {
    if (name.empty() || (std::isalpha(static_cast<unsigned char>(name.front())) == 0 &&
                         name.front() != '_' && name.front() != '$')) {
        return false;
    }
    return std::all_of(name.begin(), name.end(), [](char character) {
        return std::isalnum(static_cast<unsigned char>(character)) != 0 || character == '_' ||
               character == '$';
    });
}

std::optional<std::uint32_t> parse_unsigned(std::string_view value) {
    std::uint32_t number = 0;
    const auto result = std::from_chars(value.data(), value.data() + value.size(), number);
    if (result.ec != std::errc{} || result.ptr != value.data() + value.size()) {
        return std::nullopt;
    }
    return number;
}

bool skip_file(const std::vector<Token>& tokens, std::size_t& index) {
    if (index + 2 >= tokens.size()) return false;
    const auto number = parse_unsigned(tokens[++index].text);
    const auto filename = tokens[++index].text;
    ++index;
    if (!number || !*number || filename.size() < 3 || filename.front() != '"' ||
        filename.back() != '"')
        return false;
    if (index < tokens.size() && tokens[index].text == ",") {
        for (unsigned int field = 0; field < 2; ++field) {
            if (index + 1 >= tokens.size() || tokens[index++].text != ",") return false;
            const auto value = tokens[index++].text;
            std::uint64_t parsed{};
            const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
            if (result.ec != std::errc{} || result.ptr != value.data() + value.size()) return false;
        }
    }
    return true;
}

bool skip_location(const std::vector<Token>& tokens, std::size_t& index, std::size_t end) {
    const auto coordinates = [&] {
        for (unsigned int field = 0; field < 3; ++field) {
            if (index == end) return false;
            const auto number = parse_unsigned(tokens[index++].text);
            if (!number || (field == 0 && !*number)) return false;
        }
        return true;
    };
    ++index;
    if (!coordinates()) return false;
    if (index < end && tokens[index].text == ",") {
        if (index + 2 >= end || tokens[index + 1].text != "function_name" ||
            !is_identifier(tokens[index + 2].text))
            return false;
        index += 3;
        if (index < end && tokens[index].text == "+") {
            ++index;
            if (index == end || !parse_unsigned(tokens[index].text)) return false;
            ++index;
        }
        if (index + 1 >= end || tokens[index++].text != "," ||
            tokens[index++].text != "inlined_at" || !coordinates())
            return false;
    }
    return true;
}

bool skip_debug_section(const std::vector<Token>& tokens, std::size_t& index) {
    if (index + 2 >= tokens.size() || !tokens[index + 1].text.starts_with(".debug_") ||
        tokens[index + 2].text != "{")
        return false;
    index += 3;
    for (; index < tokens.size(); ++index) {
        const auto text = tokens[index].text;
        if (text == "}") {
            ++index;
            return true;
        }
        // Debug sections contain data, not executable definitions or nested scopes.
        if (text == "{" || (text.starts_with('.') && text != ".b8" && text != ".b16" &&
                            text != ".b32" && text != ".b64"))
            return false;
    }
    return false;
}

bool valid_entry_directives(const std::vector<Token>& tokens, std::size_t begin, std::size_t end) {
    constexpr std::array<std::string_view, 5> directives{".maxntid", ".reqntid", ".maxnreg",
                                                         ".minnctapersm", ".maxnctapersm"};
    std::array<bool, directives.size()> seen{};
    while (begin < end) {
        const auto found = std::ranges::find(directives, tokens[begin++].text);
        if (found == directives.end()) return false;
        const auto kind = static_cast<std::size_t>(found - directives.begin());
        if (seen[kind]) return false;
        seen[kind] = true;
        unsigned int dimensions{};
        do {
            if (begin == end) return false;
            const auto value = parse_unsigned(tokens[begin++].text);
            if (!value || !*value || ++dimensions > (kind < 2 ? 3U : 1U)) return false;
            if (begin == end || tokens[begin].text != ",") break;
            ++begin;
            if (begin == end) return false;
        } while (true);
    }
    return !seen[0] || !seen[1];
}

// Preserve declarations, never interpret initializers or allocate device state.
// The caller must keep original and generated entries in one native module.
bool skip_module_variable(const std::vector<Token>& tokens, std::size_t& index, bool is_external) {
    ++index;  // .global or .const.
    if (index < tokens.size() && tokens[index].text == ".align") {
        if (++index == tokens.size()) return false;
        const auto alignment = parse_unsigned(tokens[index++].text);
        if (!alignment || !*alignment || *alignment > 128 || (*alignment & (*alignment - 1)))
            return false;
    }
    if (index < tokens.size() && (tokens[index].text == ".v2" || tokens[index].text == ".v4"))
        ++index;
    constexpr auto types =
        std::to_array<std::string_view>({".b8", ".b16", ".b32", ".b64", ".u8", ".u16", ".u32",
                                         ".u64", ".s8", ".s16", ".s32", ".s64", ".f32", ".f64"});
    if (index + 1 >= tokens.size() || std::ranges::find(types, tokens[index].text) == types.end() ||
        !is_identifier(tokens[index + 1].text))
        return false;
    index += 2;
    std::uint64_t elements = 1;
    bool has_unsized_array{};
    while (index < tokens.size() && tokens[index].text == "[") {
        if (++index == tokens.size()) return false;
        if (tokens[index].text == "]") {
            has_unsized_array = true;
            ++index;
            continue;
        }
        const auto count = parse_unsigned(tokens[index++].text);
        if (!count || !*count || elements > UINT64_MAX / *count || index == tokens.size() ||
            tokens[index++].text != "]")
            return false;
        elements *= *count;
    }
    if (index == tokens.size()) return false;
    if (tokens[index].text == ";") {
        ++index;
        return is_external || !has_unsized_array;
    }
    if (is_external || tokens[index++].text != "=") return false;
    std::array<std::string_view, 16> delimiters{};
    std::size_t depth{}, values{};
    while (index < tokens.size()) {
        const auto text = tokens[index++].text;
        if (text == ";") return depth == 0 && values != 0;
        if (text == "{" || text == "(" || text == "[") {
            if (depth == delimiters.size()) return false;
            delimiters[depth++] = text;
        } else if (text == "}" || text == ")" || text == "]") {
            const auto expected = text == "}" ? "{" : text == ")" ? "(" : "[";
            if (!depth || delimiters[--depth] != expected) return false;
        } else {
            if (text.starts_with('%') || text.starts_with('"') ||
                (text.starts_with('.') &&
                 (text.size() < 2 || !std::isdigit(static_cast<unsigned char>(text[1])))))
                return false;
            ++values;
        }
    }
    return false;
}

struct RegisterRange {
    std::string_view name;
    std::uint32_t count = 0;
};

bool parse_registers(const std::vector<Token>& tokens, std::size_t begin, std::size_t end,
                     std::vector<RegisterRange>& registers) {
    constexpr auto types =
        std::to_array<std::string_view>({".pred", ".b16", ".b32", ".b64", ".u16", ".u32", ".u64",
                                         ".s16", ".s32", ".s64", ".f32", ".f64"});
    if (begin + 2 >= end ||
        std::find(types.begin(), types.end(), tokens[begin + 1].text) == types.end()) {
        return false;
    }
    std::size_t index = begin + 2;
    while (index < end) {
        const auto name = tokens[index++].text;
        if (!is_identifier(name.starts_with('%') ? name.substr(1) : name)) {
            return false;
        }
        std::uint32_t count = 0;
        if (index < end && tokens[index].text == "<") {
            if (index + 2 >= end || tokens[index + 2].text != ">") {
                return false;
            }
            const auto parsed = parse_unsigned(tokens[index + 1].text);
            if (!parsed.has_value() || *parsed == 0 || *parsed > 65536) {
                return false;
            }
            count = *parsed;
            index += 3;
        }
        registers.push_back({name, count});
        if (index < end && tokens[index++].text != ",") {
            return false;
        }
    }
    return true;
}

bool has_register(std::string_view name, const std::vector<RegisterRange>& registers,
                  bool is_cooperative) {
    // A read-only device timer is independent of logical CTA/grid remapping.
    // Keep the original instruction/clock value in every physical execution.
    if (name == "%ctaid.x" || name == "%nctaid.x" || name == "%tid.x" || name == "%ntid.x" ||
        name == "%globaltimer") {
        return true;
    }
    if (is_cooperative &&
        (name == "%ctaid.y" || name == "%ctaid.z" || name == "%nctaid.y" || name == "%nctaid.z" ||
         name == "%tid.y" || name == "%tid.z" || name == "%ntid.y" || name == "%ntid.z" ||
         name == "%laneid" || name == "%lanemask_eq" || name == "%lanemask_lt" ||
         name == "%lanemask_le" || name == "%lanemask_gt" || name == "%lanemask_ge")) {
        return true;
    }
    for (const auto& range : registers) {
        if (range.count == 0 && range.name == name) {
            return true;
        }
        if (range.count > 0 && name.starts_with(range.name)) {
            const auto number = parse_unsigned(name.substr(range.name.size()));
            if (number.has_value() && *number < range.count) {
                return true;
            }
        }
    }
    return false;
}

bool is_warp_collective(std::string_view opcode) {
    constexpr auto instructions = std::to_array<std::string_view>(
        {"shfl.sync.up.b32", "shfl.sync.down.b32", "shfl.sync.bfly.b32", "shfl.sync.idx.b32",
         "vote.sync.all.pred", "vote.sync.any.pred", "vote.sync.uni.pred", "vote.sync.ballot.b32",
         "match.any.sync.b32", "match.any.sync.b64", "match.all.sync.b32", "match.all.sync.b64",
         "bar.warp.sync"});
    return std::ranges::find(instructions, opcode) != instructions.end();
}

// Match the Sm70 scalar atomic grammar rather than admitting unknown operations
// by prefix. Preserve all operands and memory-order qualifiers for SDK validation.
bool is_atomic_instruction(std::string_view opcode, bool is_cooperative) {
    const bool is_reduction = opcode.starts_with("red.");
    if (!is_reduction && !opcode.starts_with("atom.")) return false;
    auto tail = opcode.substr(is_reduction ? 4 : 5);
    constexpr auto operations = std::to_array<std::string_view>(
        {"and", "or", "xor", "cas", "exch", "add", "inc", "dec", "min", "max"});
    constexpr auto types = std::to_array<std::string_view>(
        {"b16", "b32", "b64", "u32", "u64", "s32", "s64", "f16", "f16x2", "f32", "f64"});
    std::string_view operation, type, space, semantics, scope;
    bool has_noftz{};
    const auto assign = [](std::string_view& slot, std::string_view value) {
        if (!slot.empty()) return false;
        slot = value;
        return true;
    };
    // PTX permits these qualifiers in multiple orders. Preserve the spelling
    // and defaults instead of manufacturing stronger or weaker memory semantics.
    while (!tail.empty()) {
        const auto dot = tail.find('.');
        const auto component = tail.substr(0, dot);
        if (component == "global" || component == "shared") {
            if (!assign(space, component)) return false;
        } else if (component == "cta" || component == "gpu" || component == "sys") {
            if (!assign(scope, component)) return false;
        } else if (component == "relaxed" || component == "release" ||
                   (!is_reduction && (component == "acquire" || component == "acq_rel"))) {
            if (!assign(semantics, component)) return false;
        } else if (std::ranges::find(operations, component) != operations.end()) {
            if (!assign(operation, component)) return false;
        } else if (std::ranges::find(types, component) != types.end()) {
            if (!assign(type, component)) return false;
        } else if (component == "noftz" && !has_noftz) {
            has_noftz = true;
        } else {
            return false;
        }
        if (dot == std::string_view::npos) break;
        tail.remove_prefix(dot + 1);
        if (tail.empty()) return false;
    }
    // An unspecified state space can address shared storage too.
    if ((!is_cooperative && space != "global") || operation.empty() || type.empty() ||
        (is_reduction && (operation == "cas" || operation == "exch")))
        return false;
    if (has_noftz || type == "f16" || type == "f16x2")
        return has_noftz && operation == "add" && (type == "f16" || type == "f16x2");
    return type != "b16" || (!is_reduction && operation == "cas");
}

bool is_ordered_memory_instruction(std::string_view opcode, bool is_cooperative) {
    const bool is_load = opcode.starts_with("ld.");
    if (!is_load && !opcode.starts_with("st.")) return false;
    auto tail = opcode.substr(3);
    const auto take = [&tail](std::string_view component) {
        if (!tail.starts_with(component) || tail.size() <= component.size() ||
            tail[component.size()] != '.')
            return false;
        tail.remove_prefix(component.size() + 1);
        return true;
    };
    if (!take("volatile")) {
        if (!(take("relaxed") || take(is_load ? "acquire" : "release")) ||
            !(take("cta") || take("gpu") || take("sys")))
            return false;
    }
    if (!take("global") && !(is_cooperative && take("shared"))) return false;
    static_cast<void>(take("v2") || take("v4"));
    constexpr auto types =
        std::to_array<std::string_view>({"b8", "b16", "b32", "b64", "u8", "u16", "u32", "u64", "s8",
                                         "s16", "s32", "s64", "f32", "f64"});
    return std::ranges::find(types, tail) != types.end();
}

bool is_supported_instruction(std::string_view opcode, bool is_cooperative) {
    if (is_atomic_instruction(opcode, is_cooperative) ||
        is_ordered_memory_instruction(opcode, is_cooperative))
        return true;
    if (is_cooperative &&
        (opcode == "bar.sync" || opcode.starts_with("ld.shared.") ||
         opcode.starts_with("st.shared.") || opcode.starts_with("cvta.to.shared.") ||
         opcode.starts_with("cvta.shared.") || opcode == "membar.cta" || opcode == "membar.gl" ||
         opcode == "membar.sys" || opcode == "fence.acq_rel.cta" || opcode == "fence.acq_rel.gpu" ||
         opcode == "fence.acq_rel.sys" || opcode == "fence.sc.cta" || opcode == "fence.sc.gpu" ||
         opcode == "fence.sc.sys" || opcode == "activemask.b32" || is_warp_collective(opcode))) {
        return true;
    }
    if (opcode == "bra" || opcode == "bra.uni") {
        return true;
    }
    if (opcode == "ret" || opcode == "ret.uni") {
        return true;
    }
    if (opcode.starts_with("ld.param.") || opcode.starts_with("st.param.") ||
        opcode.starts_with("ld.global.") || opcode.starts_with("ld.const.") ||
        opcode.starts_with("st.global.")) {
        return opcode.find(".volatile") == std::string_view::npos &&
               opcode.find(".acquire") == std::string_view::npos &&
               opcode.find(".release") == std::string_view::npos &&
               opcode.find(".relaxed") == std::string_view::npos;
    }
    if (opcode.starts_with("cvta.to.global.") || opcode.starts_with("cvta.const.") ||
        opcode.starts_with("cvta.to.const.") || opcode.starts_with("cvta.local.") ||
        opcode.starts_with("cvta.to.local.") || opcode.starts_with("ld.local.") ||
        opcode.starts_with("st.local.")) {
        return true;
    }
    constexpr auto arithmetic = std::to_array<std::string_view>(
        {"mov",  "cvt",   "add",  "addc", "sub",  "subc", "mul",   "mad", "madc",     "fma",
         "div",  "rem",   "set",  "setp", "selp", "slct", "testp", "shl", "shr",      "shf",
         "and",  "or",    "xor",  "not",  "cnot", "neg",  "abs",   "min", "max",      "bfe",
         "bfi",  "bfind", "brev", "clz",  "popc", "prmt", "lop3",  "sad", "copysign", "rcp",
         "sqrt", "rsqrt", "sin",  "cos",  "lg2",  "ex2"});
    const auto dot = opcode.find('.');
    return dot != std::string_view::npos && std::find(arithmetic.begin(), arithmetic.end(),
                                                      opcode.substr(0, dot)) != arithmetic.end();
}

struct Replacement {
    std::size_t begin = 0;
    std::size_t end = 0;
    std::string text;
};

struct BodyStatement {
    std::size_t opcode = 0;
    bool is_predicated = false;
    bool has_barrier = false;
    std::size_t end{};
    std::string_view predicate;
};

struct PreparedFunction {
    std::string name, prototype, definition;
    std::size_t parameters{}, returns{}, call_depth{1};
    bool has_barrier{};
    std::vector<std::string_view> callees;
};
using FunctionResolver = std::function<const PreparedFunction*(std::string_view)>;
struct BodyAnalysis {
    std::vector<Replacement> replacements;
    std::vector<std::string_view> callees;
    bool has_barrier{};
    std::size_t call_depth{1};
};
constexpr std::array<std::string_view, 6> kLogicalRegisters{
    "%__glimmer_ptb_index", "%__glimmer_ptb_y",      "%__glimmer_ptb_z",
    "%__glimmer_ptb_grid",  "%__glimmer_ptb_grid_y", "%__glimmer_ptb_grid_z"};

bool call_operands(const std::vector<Token>& tokens, std::size_t& index, std::size_t end,
                   std::size_t& count) {
    if (index == end || tokens[index++].text != "(") return false;
    bool has_operand{};
    for (; index < end; ++index) {
        const auto text = tokens[index].text;
        if (text == ")") {
            ++index;
            if (count && !has_operand) return false;
            if (has_operand) ++count;
            return count <= 256;
        }
        if (text == ",") {
            if (!has_operand || ++count >= 256) return false;
            has_operand = false;
        } else {
            if (text == "(" || text == "{" || text == "}" || text == ";" || text.starts_with('"') ||
                text.starts_with('.'))
                return false;
            has_operand = true;
        }
    }
    return false;
}

bool analyze_call(const std::vector<Token>& tokens, std::size_t opcode, std::size_t end,
                  const FunctionResolver& resolve, BodyAnalysis& analysis, bool is_kernel,
                  bool is_cooperative, bool is_predicated, bool& has_barrier) {
    auto index = opcode + 1;
    std::size_t returns{}, parameters{};
    if (index < end && tokens[index].text == "(") {
        if (!call_operands(tokens, index, end, returns) || index == end ||
            tokens[index++].text != ",")
            return false;
    }
    if (index == end || !is_identifier(tokens[index].text) || !resolve) return false;
    const auto target = index++;
    const auto* function = resolve(tokens[target].text);
    if (!function || function->returns != returns) return false;
    const bool has_operands = index < end;
    if (has_operands &&
        (tokens[index++].text != "," || !call_operands(tokens, index, end, parameters)))
        return false;
    if (index != end || function->parameters != parameters ||
        (function->has_barrier && (!is_cooperative || is_predicated)))
        return false;
    has_barrier = function->has_barrier;
    analysis.call_depth = std::max(analysis.call_depth, function->call_depth + 1);
    analysis.callees.push_back(tokens[target].text);
    analysis.replacements.push_back({tokens[target].begin, tokens[target].end, function->name});
    std::string arguments;
    for (std::size_t axis = 0; axis < kLogicalRegisters.size(); ++axis) {
        if (axis) arguments += ", ";
        // Independent-thread entries have a one-dimensional logical grid.
        if (is_kernel && !is_cooperative && axis != 0 && axis != 3)
            arguments += axis < 3 ? "0" : "1";
        else
            arguments += kLogicalRegisters[axis];
    }
    if (has_operands) {
        if (parameters) arguments.insert(0, ", ");
        const auto position = tokens[index - 1].begin;
        analysis.replacements.push_back({position, position, std::move(arguments)});
    } else {
        const auto position = tokens[target].end;
        analysis.replacements.push_back({position, position, ", (" + arguments + ")"});
    }
    return true;
}

struct UniformStatement {
    std::vector<std::size_t> outputs, inputs;
    std::optional<std::size_t> predicate;
    bool is_uniform{};
};

bool is_uniform_special_register(std::string_view name) {
    constexpr std::array<std::string_view, 9> names{"%ctaid.x",  "%ctaid.y",  "%ctaid.z",
                                                    "%nctaid.x", "%nctaid.y", "%nctaid.z",
                                                    "%ntid.x",   "%ntid.y",   "%ntid.z"};
    return std::ranges::find(names, name) != names.end();
}

bool is_uniform_scalar_operation(std::string_view opcode) {
    // Carry-consuming operations have an implicit input that this analysis does
    // not track. Memory and warp results are not assumed equal across a CTA.
    constexpr auto operations = std::to_array<std::string_view>(
        {"mov",  "cvt",  "add",      "sub", "mul",  "mad",   "fma",   "div",  "rem", "set",  "setp",
         "selp", "slct", "testp",    "shl", "shr",  "shf",   "and",   "or",   "xor", "not",  "cnot",
         "neg",  "abs",  "min",      "max", "bfe",  "bfi",   "bfind", "brev", "clz", "popc", "prmt",
         "lop3", "sad",  "copysign", "rcp", "sqrt", "rsqrt", "sin",   "cos",  "lg2", "ex2"});
    return std::ranges::find(operations, opcode.substr(0, opcode.find('.'))) != operations.end();
}

bool is_scalar_literal(std::string_view text) {
    // Native validation still owns literal spelling, width and opcode legality.
    return !text.empty() && std::isdigit(static_cast<unsigned char>(text.front())) != 0;
}

bool uniform_parameter_load(const std::vector<Token>& tokens, std::size_t begin, std::size_t end,
                            const std::vector<std::string_view>& parameters) {
    if (begin + 3 > end || tokens[begin].text != "[" || tokens[end - 1].text != "]" ||
        std::ranges::find(parameters, tokens[begin + 1].text) == parameters.end())
        return false;
    return begin + 3 == end || (begin + 5 == end && tokens[begin + 2].text == "+" &&
                                is_scalar_literal(tokens[begin + 3].text));
}

struct ReconvergentRegions {
    std::vector<bool> branches, interiors;
};

// This structural proof never consults tentative uniformity facts. Every
// accepted region has a unique immediate postdominator and is independently
// checked for single entry, finite acyclic paths and no collective/exit. Its
// interior writes must be tainted even when an arm writes a constant.
std::optional<ReconvergentRegions> reconvergent_regions(
    const std::vector<Token>& tokens, const std::vector<BodyStatement>& statements,
    const std::vector<std::vector<std::size_t>>& successors, const std::vector<bool>& reachable) {
    constexpr auto maximum_graph_work = std::size_t{16} * 1024 * 1024;
    std::size_t work{};
    const auto charge = [&work](std::size_t amount) {
        if (amount > maximum_graph_work - work) return false;
        work += amount;
        return true;
    };
    const auto count = statements.size();
    const auto words = (count + 1 + 63) / 64;
    using Bits = std::vector<std::uint64_t>;
    const auto contains = [](const Bits& bits, std::size_t index) {
        return (bits[index / 64] & (std::uint64_t{1} << (index % 64))) != 0;
    };
    const auto insert = [](Bits& bits, std::size_t index) {
        bits[index / 64] |= std::uint64_t{1} << (index % 64);
    };
    std::vector<std::vector<std::size_t>> predecessors(count + 1);
    Bits universe(words);
    insert(universe, count);  // Synthetic exit, never an eligible region join.
    for (std::size_t index = 0; index < count; ++index) {
        if (!charge(successors[index].size() + 1)) return std::nullopt;
        if (reachable[index]) insert(universe, index);
        for (const auto successor : successors[index]) predecessors[successor].push_back(index);
        if (successors[index].empty()) predecessors[count].push_back(index);
    }
    std::vector<bool> reaches_exit(count + 1);
    std::vector<std::size_t> pending{count};
    while (!pending.empty()) {
        const auto index = pending.back();
        pending.pop_back();
        if (reaches_exit[index]) continue;
        reaches_exit[index] = true;
        if (!charge(predecessors[index].size() + 1)) return std::nullopt;
        for (const auto predecessor : predecessors[index]) pending.push_back(predecessor);
    }
    for (std::size_t index = 0; index < count; ++index)
        if (reachable[index] && !reaches_exit[index]) return std::nullopt;
    if (!charge((count + 1) * words)) return std::nullopt;
    std::vector<Bits> postdominators(count + 1, universe);
    postdominators[count].assign(words, 0);
    insert(postdominators[count], count);
    bool changed = true;
    while (changed) {
        changed = false;
        for (std::size_t index = count; index-- > 0;) {
            if (!reachable[index]) continue;
            if (!charge((successors[index].size() + 3) * words)) return std::nullopt;
            Bits next = successors[index].empty() ? postdominators[count] : universe;
            for (const auto successor : successors[index])
                for (std::size_t word = 0; word < words; ++word)
                    next[word] &= postdominators[successor][word];
            insert(next, index);
            if (next != postdominators[index]) {
                postdominators[index] = std::move(next);
                changed = true;
            }
        }
    }
    if (!charge((count + 1) * words)) return std::nullopt;
    std::vector<std::size_t> depths(count + 1);
    for (std::size_t index = 0; index <= count; ++index)
        for (const auto word : postdominators[index])
            depths[index] += static_cast<std::size_t>(std::popcount(word));
    ReconvergentRegions result{std::vector<bool>(count), std::vector<bool>(count)};
    for (std::size_t branch = 0; branch < count; ++branch) {
        const auto opcode = tokens[statements[branch].opcode].text;
        if (!reachable[branch] || !statements[branch].is_predicated ||
            (opcode != "bra" && opcode != "bra.uni"))
            continue;
        if (!charge(count + words)) return std::nullopt;
        auto join = count;
        for (std::size_t index = 0; index < count; ++index)
            if (index != branch && contains(postdominators[branch], index) &&
                depths[index] > depths[join])
                join = index;
        if (join == count) continue;
        auto strict = postdominators[branch];
        strict[branch / 64] &= ~(std::uint64_t{1} << (branch % 64));
        if (strict != postdominators[join]) continue;

        if (!charge(count * 2)) return std::nullopt;
        std::vector<bool> interior(count);
        std::vector<std::size_t> members;
        pending = successors[branch];
        bool valid = true;
        while (!pending.empty() && valid) {
            const auto index = pending.back();
            pending.pop_back();
            if (index == join || interior[index]) continue;
            const auto body_opcode = tokens[statements[index].opcode].text;
            if (index == branch || index == 0 || statements[index].has_barrier ||
                body_opcode == "call" || body_opcode == "call.uni" || body_opcode == "ret" ||
                body_opcode == "ret.uni" || successors[index].empty()) {
                valid = false;
                break;
            }
            if (!charge(successors[index].size() + 1)) return std::nullopt;
            interior[index] = true;
            members.push_back(index);
            for (const auto successor : successors[index]) pending.push_back(successor);
        }
        if (!valid) continue;
        std::vector<std::size_t> indegrees(count);
        pending.clear();
        for (const auto index : members) {
            if (!charge(predecessors[index].size() + 1)) return std::nullopt;
            for (const auto predecessor : predecessors[index]) {
                if (predecessor != branch && !interior[predecessor]) valid = false;
                if (interior[predecessor]) ++indegrees[index];
            }
            if (indegrees[index] == 0) pending.push_back(index);
        }
        if (!valid) continue;
        std::size_t removed{};
        while (!pending.empty()) {
            const auto index = pending.back();
            pending.pop_back();
            ++removed;
            if (!charge(successors[index].size() + 1)) return std::nullopt;
            for (const auto successor : successors[index])
                if (successor != join && --indegrees[successor] == 0) pending.push_back(successor);
        }
        if (removed != members.size()) continue;
        result.branches[branch] = true;
        for (const auto index : members) result.interiors[index] = true;
    }
    return result;
}

// This fallback proves a stronger property than ordinary barrier dominance:
// every control decision is equal across the whole CTA. Under that invariant,
// intersected dataflow facts remain valid at joins and across loop iterations.
// A second pass can admit structurally reconvergent ordinary regions, after
// tainting their writes independently of predicate facts. Nothing here trusts
// a .uni hint, a helper argument, or a device-memory load.
bool validate_uniform_control_flow(const std::vector<Token>& tokens,
                                   const std::vector<BodyStatement>& statements,
                                   const std::vector<std::vector<std::size_t>>& successors,
                                   const std::vector<RegisterRange>& registers,
                                   const std::vector<std::string_view>& parameters) {
    constexpr std::size_t max_statements = 8192, max_registers = 2048;
    constexpr std::size_t max_evaluations = 131072;
    constexpr std::size_t max_work = std::size_t{16} * 1024 * 1024;
    if (statements.size() > max_statements) return false;
    std::unordered_map<std::string_view, std::size_t> register_ids;
    for (const auto& statement : statements) {
        const auto opcode = tokens[statement.opcode].text;
        if (opcode == "call" || opcode == "call.uni" || opcode.starts_with("st.param."))
            return false;
        if (opcode.starts_with('.')) continue;
        if (!statement.predicate.empty() && !register_ids.contains(statement.predicate))
            register_ids.emplace(statement.predicate, register_ids.size());
        for (auto index = statement.opcode + 1; index < statement.end; ++index) {
            const auto text = tokens[index].text;
            if (has_register(text, registers, false) && !is_uniform_special_register(text) &&
                text != "%tid.x" && text != "%globaltimer" && !register_ids.contains(text))
                register_ids.emplace(text, register_ids.size());
        }
        if (register_ids.size() > max_registers) return false;
    }
    if (register_ids.empty()) return false;
    std::vector<UniformStatement> facts(statements.size());
    for (std::size_t index = 0; index < statements.size(); ++index) {
        const auto& statement = statements[index];
        auto& fact = facts[index];
        const auto opcode = tokens[statement.opcode].text;
        if (!statement.predicate.empty()) {
            const auto found = register_ids.find(statement.predicate);
            if (found == register_ids.end()) return false;
            fact.predicate = found->second;
        }
        if (opcode.starts_with('.') || opcode == "bra" || opcode == "bra.uni" || opcode == "ret" ||
            opcode == "ret.uni" || opcode.starts_with("st.") || opcode.starts_with("red.") ||
            opcode.starts_with("bar.") || opcode.starts_with("membar.") ||
            opcode.starts_with("fence."))
            continue;
        // A comma inside a vector destination is not the source separator.
        // Until vector writes have their own transfer rule, reject this proof
        // rather than accidentally retaining an overwritten component's fact.
        if (statement.opcode + 1 == statement.end || tokens[statement.opcode + 1].text == "{")
            return false;
        auto comma = statement.opcode + 1;
        while (comma < statement.end && tokens[comma].text != ",") ++comma;
        if (comma == statement.end) return false;
        for (auto operand = statement.opcode + 1; operand < comma; ++operand) {
            const auto text = tokens[operand].text;
            if (const auto found = register_ids.find(text); found != register_ids.end())
                fact.outputs.push_back(found->second);
            else if (text != "{" && text != "}" && text != "|" && text != "_")
                return false;
        }
        fact.is_uniform = is_uniform_scalar_operation(opcode);
        if (opcode.starts_with("ld.param.")) {
            fact.is_uniform = uniform_parameter_load(tokens, comma + 1, statement.end, parameters);
            continue;
        }
        for (auto operand = comma + 1; operand < statement.end; ++operand) {
            const auto text = tokens[operand].text;
            if (const auto found = register_ids.find(text); found != register_ids.end())
                fact.inputs.push_back(found->second);
            else if (!is_uniform_special_register(text) && !is_scalar_literal(text) &&
                     text != "," && text != "{" && text != "}" && text != "-" && text != "+" &&
                     text != "!" && text != "|")
                fact.is_uniform = false;
        }
    }
    std::vector<std::vector<std::size_t>> predecessors(statements.size());
    std::vector<bool> reachable(statements.size());
    std::vector<std::size_t> pending{0};
    while (!pending.empty()) {
        const auto index = pending.back();
        pending.pop_back();
        if (reachable[index]) continue;
        reachable[index] = true;
        for (const auto successor : successors[index]) {
            predecessors[successor].push_back(index);
            pending.push_back(successor);
        }
    }
    const auto words = (register_ids.size() + 63) / 64;
    using State = std::vector<std::uint64_t>;
    const auto test = [](const State& state, std::size_t reg) {
        return (state[reg / 64] & (std::uint64_t{1} << (reg % 64))) != 0;
    };
    const auto assign = [](State& state, std::size_t reg, bool uniform) {
        const auto mask = std::uint64_t{1} << (reg % 64);
        if (uniform)
            state[reg / 64] |= mask;
        else
            state[reg / 64] &= ~mask;
    };
    std::size_t evaluations{};
    std::size_t work{};
    std::optional<ReconvergentRegions> regions;
    // Preserve the original proof first. Both passes share its work/evaluation
    // caps. The second pass starts afresh: optimistic facts from the first
    // pass cannot select regions or survive their control-dependence kills.
    for (unsigned pass = 0; pass < 2; ++pass) {
        std::vector<State> outputs(statements.size(), State(words, ~std::uint64_t{}));
        std::vector<bool> queued = reachable;
        for (std::size_t index = statements.size(); index-- > 0;)
            if (reachable[index]) pending.push_back(index);
        while (!pending.empty()) {
            if (++evaluations > max_evaluations) return false;
            const auto index = pending.back();
            pending.pop_back();
            queued[index] = false;
            const auto cost = (predecessors[index].size() + 3) * words +
                              facts[index].inputs.size() + facts[index].outputs.size() * 2;
            if (cost > max_work - work) return false;
            work += cost;
            State state(words, index == 0 ? 0 : ~std::uint64_t{});
            for (const auto predecessor : predecessors[index])
                for (std::size_t word = 0; word < words; ++word)
                    state[word] &= outputs[predecessor][word];
            const auto& fact = facts[index];
            bool uniform = fact.is_uniform && (!regions || !regions->interiors[index]);
            for (const auto input : fact.inputs) uniform = uniform && test(state, input);
            if (fact.predicate) uniform = uniform && test(state, *fact.predicate);
            // For a uniform false predicate, the previous value survives. It
            // must already be uniform; a predicated write cannot seed a fact.
            for (const auto output : fact.outputs)
                if (fact.predicate) uniform = uniform && test(state, output);
            for (const auto output : fact.outputs) assign(state, output, uniform);
            if (state == outputs[index]) continue;
            outputs[index] = std::move(state);
            for (const auto successor : successors[index])
                if (!queued[successor]) {
                    queued[successor] = true;
                    pending.push_back(successor);
                }
        }
        bool valid = true;
        for (std::size_t index = 0; index < statements.size(); ++index) {
            const auto opcode = tokens[statements[index].opcode].text;
            if (reachable[index] &&
                (opcode == "bra" || opcode == "bra.uni" || opcode == "ret" ||
                 opcode == "ret.uni") &&
                facts[index].predicate && !test(outputs[index], *facts[index].predicate) &&
                (!regions || !regions->branches[index])) {
                valid = false;
                break;
            }
        }
        if (valid) return true;
        if (pass == 0) {
            regions = reconvergent_regions(tokens, statements, successors, reachable);
            if (!regions) return false;
        }
    }
    return false;
}

// Removing any barrier must make every return unreachable. This conservative
// dominance check excludes early exits that would deadlock a wrapped block.
// Cyclic/bypassed barriers additionally require whole-CTA control uniformity.
bool validate_control_flow(const std::vector<Token>& tokens,
                           const std::vector<BodyStatement>& statements,
                           const std::unordered_map<std::string_view, std::size_t>& labels,
                           const std::vector<RegisterRange>& registers,
                           const std::vector<std::string_view>& parameters,
                           bool can_prove_uniformity) {
    if (statements.size() > 65536) {
        return false;
    }
    std::vector<std::vector<std::size_t>> successors(statements.size());
    std::vector<std::size_t> barriers;
    for (std::size_t index = 0; index < statements.size(); ++index) {
        const auto& statement = statements[index];
        const auto opcode = tokens[statement.opcode].text;
        const bool is_branch = opcode == "bra" || opcode == "bra.uni";
        const bool is_return = opcode == "ret" || opcode == "ret.uni";
        if (is_branch) {
            const auto found = labels.find(tokens[statement.opcode + 1].text);
            if (found == labels.end() || found->second >= statements.size()) {
                return false;
            }
            successors[index].push_back(found->second);
        }
        if ((!is_branch && !is_return) || statement.is_predicated) {
            if (index + 1 == statements.size()) {
                // No implicit fall-through out of a wrapped logical block.
                return false;
            }
            successors[index].push_back(index + 1);
        }
        if (statement.has_barrier) {
            barriers.push_back(index);
        }
    }
    if (barriers.size() > 64) {
        return false;
    }
    const auto uniform_fallback = [&] {
        return can_prove_uniformity &&
               validate_uniform_control_flow(tokens, statements, successors, registers, parameters);
    };
    for (const auto barrier : barriers) {
        std::vector<bool> visited(statements.size(), false);
        std::vector<std::size_t> pending = successors[barrier];
        while (!pending.empty()) {
            const auto index = pending.back();
            pending.pop_back();
            if (index == barrier) {
                return uniform_fallback();
            }
            if (visited[index]) {
                continue;
            }
            visited[index] = true;
            for (const auto successor : successors[index]) {
                pending.push_back(successor);
            }
        }
        std::fill(visited.begin(), visited.end(), false);
        pending.push_back(0);
        while (!pending.empty()) {
            const auto index = pending.back();
            pending.pop_back();
            if (index == barrier || visited[index]) {
                continue;
            }
            visited[index] = true;
            const auto opcode = tokens[statements[index].opcode].text;
            if (opcode == "ret" || opcode == "ret.uni") {
                return uniform_fallback();
            }
            for (const auto successor : successors[index]) {
                pending.push_back(successor);
            }
        }
    }
    return true;
}

bool parse_storage(const std::vector<Token>& tokens, std::size_t begin, std::size_t end) {
    auto index = begin + 1;
    if (index < end && tokens[index].text == ".align") {
        if (index + 1 >= end) {
            return false;
        }
        const auto alignment = parse_unsigned(tokens[index + 1].text);
        if (!alignment.has_value() || *alignment == 0 || *alignment > 128 ||
            (*alignment & (*alignment - 1)) != 0) {
            return false;
        }
        index += 2;
    }
    constexpr auto types =
        std::to_array<std::string_view>({".b8", ".b16", ".b32", ".b64", ".u8", ".u16", ".u32",
                                         ".u64", ".s8", ".s16", ".s32", ".s64", ".f32", ".f64"});
    if (index < end && (tokens[index].text == ".v2" || tokens[index].text == ".v4")) ++index;
    if (index + 1 >= end ||
        std::find(types.begin(), types.end(), tokens[index].text) == types.end() ||
        !is_identifier(tokens[index + 1].text)) {
        return false;
    }
    index += 2;
    if (index == end) {
        return true;
    }
    if (index + 3 != end || tokens[index].text != "[" || tokens[index + 2].text != "]") {
        return false;
    }
    const auto count = parse_unsigned(tokens[index + 1].text);
    return count.has_value() && *count > 0 && *count <= 65536;
}

bool analyze_body(const std::vector<Token>& tokens, std::size_t begin, std::size_t end,
                  BodyAnalysis& analysis, bool is_cooperative, bool is_kernel = true,
                  const FunctionResolver& resolve = {}, std::vector<RegisterRange> registers = {},
                  const std::vector<std::string_view>& parameters = {}) {
    std::vector<BodyStatement> statements;
    std::unordered_map<std::string_view, std::size_t> labels;
    bool has_return = false;
    bool has_nested_scope = false;
    std::size_t depth{};
    std::size_t statement = begin;
    while (statement < end) {
        if (tokens[statement].text == "{") {
            if (++depth > 64) return false;
            has_nested_scope = true;
            ++statement;
            continue;
        }
        if (tokens[statement].text == "}") {
            if (!depth) return false;
            --depth;
            ++statement;
            continue;
        }
        if (tokens[statement].text == ".loc") {
            if (!skip_location(tokens, statement, end)) return false;
            continue;
        }
        if (statement + 1 < end && is_identifier(tokens[statement].text) &&
            tokens[statement + 1].text == ":") {
            if (!labels.emplace(tokens[statement].text, statements.size()).second) {
                return false;
            }
            statement += 2;
            continue;
        }
        auto finish = statement;
        while (finish < end && tokens[finish].text != ";") {
            ++finish;
        }
        if (finish == end || finish == statement) {
            return false;
        }
        auto opcode = statement;
        bool is_predicated = false;
        bool has_barrier = false;
        std::string_view predicate;
        if (tokens[opcode].text == ".reg") {
            if (!parse_registers(tokens, statement, finish, registers)) {
                return false;
            }
        } else if (tokens[opcode].text == ".local" || tokens[opcode].text == ".param" ||
                   (is_cooperative && tokens[opcode].text == ".shared")) {
            if (!parse_storage(tokens, statement, finish)) {
                return false;
            }
        } else if (tokens[opcode].text == ".pragma") {
            // nvcc emits this non-executable loop annotation. Accept exactly
            // one known literal; other pragmas/strings remain unsupported.
            if (opcode + 2 != finish || tokens[opcode + 1].text != kNounrollPragma) {
                return false;
            }
        } else {
            if (tokens[opcode].text == "@") {
                is_predicated = true;
                ++opcode;
                if (opcode < finish && tokens[opcode].text == "!") {
                    ++opcode;
                }
                if (opcode >= finish ||
                    !(tokens[opcode].text.starts_with('%') || is_identifier(tokens[opcode].text))) {
                    return false;
                }
                predicate = tokens[opcode].text;
                ++opcode;
            }
            if (opcode >= finish) return false;
            const bool is_call = tokens[opcode].text == "call" || tokens[opcode].text == "call.uni";
            if (is_call) {
                if (!analyze_call(tokens, opcode, finish, resolve, analysis, is_kernel,
                                  is_cooperative, is_predicated, has_barrier))
                    return false;
            } else if (!is_supported_instruction(tokens[opcode].text, is_cooperative)) {
                return false;
            }
            // Independent threads may be on different logical CTAs. Preserve
            // control flow, but do not retain the original warp-uniform promise.
            if (!is_cooperative &&
                (tokens[opcode].text == "call.uni" || tokens[opcode].text == "bra.uni"))
                analysis.replacements.push_back(
                    {tokens[opcode].begin, tokens[opcode].end,
                     tokens[opcode].text == "call.uni" ? "call" : "bra"});
            if (tokens[opcode].text == "bar.sync" &&
                (is_predicated || opcode + 2 != finish || tokens[opcode + 1].text != "0")) {
                return false;
            }
            has_barrier = has_barrier || tokens[opcode].text == "bar.sync";
            if (is_warp_collective(tokens[opcode].text)) {
                // A logical return branches to our footer rather than exiting
                // the physical lane. Do not leave that lane in a later mask.
                if (is_predicated) return false;
                has_barrier = true;
            }
            if ((tokens[opcode].text == "bra" || tokens[opcode].text == "bra.uni") &&
                (opcode + 2 != finish || !is_identifier(tokens[opcode + 1].text))) {
                return false;
            }
            if (tokens[opcode].text == "ret" || tokens[opcode].text == "ret.uni") {
                if (opcode + 1 != finish) {
                    return false;
                }
                if (is_kernel)
                    analysis.replacements.push_back(
                        {tokens[opcode].begin, tokens[opcode].end, "bra __glimmer_ptb_next"});
                else if (!is_cooperative && tokens[opcode].text == "ret.uni")
                    analysis.replacements.push_back(
                        {tokens[opcode].begin, tokens[opcode].end, "ret"});
                has_return = true;
            }
        }
        analysis.has_barrier = analysis.has_barrier || has_barrier;
        statements.push_back({opcode, is_predicated, has_barrier, finish, predicate});
        if (statements.size() > 65536) return false;
        statement = finish + 1;
    }
    for (auto index = begin; index < end; ++index) {
        const auto name = tokens[index].text;
        if (name.starts_with('%')) {
            // Declaration prefixes are not operands of the instruction stream.
            if (index + 1 < end && tokens[index + 1].text == "<") {
                continue;
            }
            if (!has_register(name, registers, is_cooperative || !is_kernel)) {
                return false;
            }
            if (name == "%ctaid.x") {
                analysis.replacements.push_back(
                    {tokens[index].begin, tokens[index].end, "%__glimmer_ptb_index"});
            } else if (name == "%nctaid.x") {
                analysis.replacements.push_back(
                    {tokens[index].begin, tokens[index].end, "%__glimmer_ptb_grid"});
            } else if ((is_cooperative || !is_kernel) && name == "%ctaid.y") {
                analysis.replacements.push_back(
                    {tokens[index].begin, tokens[index].end, "%__glimmer_ptb_y"});
            } else if ((is_cooperative || !is_kernel) && name == "%ctaid.z") {
                analysis.replacements.push_back(
                    {tokens[index].begin, tokens[index].end, "%__glimmer_ptb_z"});
            } else if ((is_cooperative || !is_kernel) && name == "%nctaid.y") {
                analysis.replacements.push_back(
                    {tokens[index].begin, tokens[index].end, "%__glimmer_ptb_grid_y"});
            } else if ((is_cooperative || !is_kernel) && name == "%nctaid.z") {
                analysis.replacements.push_back(
                    {tokens[index].begin, tokens[index].end, "%__glimmer_ptb_grid_z"});
            }
        }
    }
    std::size_t replacement_bytes{};
    for (const auto& replacement : analysis.replacements) {
        if (replacement.text.size() > kMaxPersistentPtxBytes - replacement_bytes) return false;
        replacement_bytes += replacement.text.size();
    }
    return !depth && has_return &&
           ((!is_cooperative && is_kernel) ||
            validate_control_flow(tokens, statements, labels, registers, parameters,
                                  is_kernel && !has_nested_scope));
}

std::optional<std::vector<PtxParameterLayout>> parameter_layout(const std::vector<Token>& tokens,
                                                                std::size_t begin, std::size_t end,
                                                                bool allows_registers = false) {
    constexpr auto types =
        std::to_array<std::string_view>({".b8", ".b16", ".b32", ".b64", ".u8", ".u16", ".u32",
                                         ".u64", ".s8", ".s16", ".s32", ".s64", ".f32", ".f64"});
    std::vector<PtxParameterLayout> layout;
    std::size_t total_bytes = 0;
    for (auto index = begin; index < end;) {
        const bool is_register = allows_registers && tokens[index].text == ".reg";
        if (tokens[index++].text != ".param" && !is_register) {
            return std::nullopt;
        }
        std::size_t alignment = 0;
        if (!is_register && index < end && tokens[index].text == ".align") {
            if (++index >= end) {
                return std::nullopt;
            }
            const auto parsed_alignment = parse_unsigned(tokens[index++].text);
            if (!parsed_alignment.has_value() || *parsed_alignment == 0 ||
                *parsed_alignment > 128 || (*parsed_alignment & (*parsed_alignment - 1)) != 0) {
                return std::nullopt;
            }
            alignment = *parsed_alignment;
        }
        if (index + 1 >= end ||
            std::find(types.begin(), types.end(), tokens[index].text) == types.end() ||
            !(is_register ? is_identifier(tokens[index + 1].text.starts_with('%')
                                              ? tokens[index + 1].text.substr(1)
                                              : tokens[index + 1].text)
                          : is_identifier(tokens[index + 1].text))) {
            return std::nullopt;
        }
        const auto bits = parse_unsigned(tokens[index].text.substr(2));
        if (!bits.has_value()) {
            return std::nullopt;
        }
        std::size_t bytes = *bits / 8;
        if (is_register && bytes < 4) return std::nullopt;
        if (alignment == 0) {
            alignment = bytes;
        }
        index += 2;
        if (!is_register && index < end && tokens[index].text == "[") {
            if (index + 2 >= end || tokens[index + 2].text != "]") {
                return std::nullopt;
            }
            const auto count = parse_unsigned(tokens[index + 1].text);
            if (!count.has_value() || *count == 0 || *count > 65536 / bytes) {
                return std::nullopt;
            }
            bytes *= *count;
            index += 3;
        }
        if (bytes > 65536 - total_bytes || layout.size() == 256) {
            return std::nullopt;
        }
        if (index < end) {
            if (tokens[index].text != "," || index + 1 == end) {
                return std::nullopt;
            }
            ++index;
        }
        total_bytes += bytes;
        layout.push_back({bytes, alignment});
    }
    return layout;
}

std::string rewrite_body(std::string_view source, std::size_t begin, std::size_t end,
                         std::vector<Replacement> replacements) {
    std::sort(replacements.begin(), replacements.end(),
              [](const auto& left, const auto& right) { return left.begin < right.begin; });
    std::string body;
    for (const auto& replacement : replacements) {
        body.append(source.substr(begin, replacement.begin - begin));
        body.append(replacement.text);
        begin = replacement.end;
    }
    body.append(source.substr(begin, end - begin));
    return body;
}

std::string make_variant(std::string_view parameters, std::string_view directives,
                         std::string_view body, const PtxKernelVariant& kernel) {
    std::string image = "\n.visible .entry " + kernel.persistent_name + "(";
    image.append(parameters);
    if (kernel.parameter_count > 0) {
        // A newline terminates a possible trailing line comment before the comma.
        image += "\n,";
    }
    image += R"ptx(
    .param .u32 __glimmer_ptb_original_grid
)
)ptx";
    image.append(directives);
    image += R"ptx(
{
    .reg .u32 %__glimmer_ptb_index, %__glimmer_ptb_stride;
    .reg .u32 %__glimmer_ptb_grid, %__glimmer_ptb_remaining;
    .reg .pred %__glimmer_ptb_done;
    ld.param.u32 %__glimmer_ptb_grid, [__glimmer_ptb_original_grid];
    mov.u32 %__glimmer_ptb_index, %ctaid.x;
    mov.u32 %__glimmer_ptb_stride, %nctaid.x;
__glimmer_ptb_loop:
)ptx";
    image.append(body);
    // Test the remaining distance before addition to avoid index overflow.
    image += R"ptx(
__glimmer_ptb_next:
    sub.u32 %__glimmer_ptb_remaining, %__glimmer_ptb_grid, %__glimmer_ptb_index;
    setp.le.u32 %__glimmer_ptb_done, %__glimmer_ptb_remaining, %__glimmer_ptb_stride;
    @%__glimmer_ptb_done ret;
    add.u32 %__glimmer_ptb_index, %__glimmer_ptb_index, %__glimmer_ptb_stride;
    bra __glimmer_ptb_loop;
}
)ptx";
    return image;
}

std::string make_cooperative_variant(std::string_view parameters, std::string_view directives,
                                     std::string_view body, const PtxKernelVariant& kernel,
                                     PtxExecutionKind kind) {
    std::string image = "\n.visible .entry " + kernel.persistent_name + "(";
    image.append(parameters);
    if (kernel.parameter_count > 0) {
        image += "\n,";
    }
    image += R"ptx(
    .param .u32 __glimmer_ptb_original_grid,
    .param .u32 __glimmer_ptb_original_grid_y,
    .param .u32 __glimmer_ptb_original_grid_z
)ptx";
    if (kind == PtxExecutionKind::kSliced) {
        image += R"ptx(
    ,
    .param .u32 __glimmer_ptb_slice_start,
    .param .u32 __glimmer_ptb_slice_count
)ptx";
    } else if (kind != PtxExecutionKind::kBlockStatic) {
        image += "    ,\n    .param .u64 __glimmer_ptb_counter";
        if (kind == PtxExecutionKind::kPreemptive) {
            image += ",\n    .param .u64 __glimmer_ptb_retreat";
            if (kernel.observes_block_completion)
                image += ",\n    .param .u64 __glimmer_ptb_completed";
        }
        image += '\n';
    }
    image += R"ptx(
)
)ptx";
    image.append(directives);
    image += R"ptx(
{
    .reg .u32 %__glimmer_ptb_index, %__glimmer_ptb_y, %__glimmer_ptb_z;
    .reg .u32 %__glimmer_ptb_grid, %__glimmer_ptb_grid_y, %__glimmer_ptb_grid_z;
    .reg .u32 %__glimmer_ptb_flat, %__glimmer_ptb_quotient, %__glimmer_ptb_value;
    .reg .u64 %__glimmer_ptb_total, %__glimmer_ptb_claim;
    .reg .pred %__glimmer_ptb_done;
    ld.param.u32 %__glimmer_ptb_grid, [__glimmer_ptb_original_grid];
    ld.param.u32 %__glimmer_ptb_grid_y, [__glimmer_ptb_original_grid_y];
    ld.param.u32 %__glimmer_ptb_grid_z, [__glimmer_ptb_original_grid_z];
    mul.wide.u32 %__glimmer_ptb_total, %__glimmer_ptb_grid, %__glimmer_ptb_grid_y;
    cvt.u64.u32 %__glimmer_ptb_claim, %__glimmer_ptb_grid_z;
    mul.lo.u64 %__glimmer_ptb_total, %__glimmer_ptb_total, %__glimmer_ptb_claim;
)ptx";
    if (kind == PtxExecutionKind::kBlockStatic) {
        image += R"ptx(
    .reg .u32 %__glimmer_ptb_worker;
    .reg .u64 %__glimmer_ptb_stride;
    mov.u32 %__glimmer_ptb_worker, %nctaid.x;
    cvt.u64.u32 %__glimmer_ptb_stride, %__glimmer_ptb_worker;
    mov.u32 %__glimmer_ptb_worker, %ctaid.x;
    cvt.u64.u32 %__glimmer_ptb_claim, %__glimmer_ptb_worker;
__glimmer_ptb_claim_next:
    setp.ge.u64 %__glimmer_ptb_done, %__glimmer_ptb_claim, %__glimmer_ptb_total;
    @%__glimmer_ptb_done ret;
    cvt.u32.u64 %__glimmer_ptb_flat, %__glimmer_ptb_claim;
)ptx";
    } else if (kind == PtxExecutionKind::kSliced) {
        image += R"ptx(
    .reg .u32 %__glimmer_ptb_start, %__glimmer_ptb_count;
    .reg .u64 %__glimmer_ptb_extent;
    ld.param.u32 %__glimmer_ptb_start, [__glimmer_ptb_slice_start];
    ld.param.u32 %__glimmer_ptb_count, [__glimmer_ptb_slice_count];
    mov.u32 %__glimmer_ptb_flat, %ctaid.x;
    setp.ge.u32 %__glimmer_ptb_done, %__glimmer_ptb_flat, %__glimmer_ptb_count;
    @%__glimmer_ptb_done ret;
    cvt.u64.u32 %__glimmer_ptb_claim, %__glimmer_ptb_start;
    cvt.u64.u32 %__glimmer_ptb_extent, %__glimmer_ptb_flat;
    add.u64 %__glimmer_ptb_claim, %__glimmer_ptb_claim, %__glimmer_ptb_extent;
    setp.ge.u64 %__glimmer_ptb_done, %__glimmer_ptb_claim, %__glimmer_ptb_total;
    @%__glimmer_ptb_done ret;
    cvt.u32.u64 %__glimmer_ptb_flat, %__glimmer_ptb_claim;
)ptx";
    } else {
        image += R"ptx(
    .reg .u32 %__glimmer_ptb_thread, %__glimmer_ptb_thread_y;
    .reg .u64 %__glimmer_ptb_counter_address;
    .reg .pred %__glimmer_ptb_follower;
    .shared .align 4 .u32 __glimmer_ptb_broadcast[2];
    ld.param.u64 %__glimmer_ptb_counter_address, [__glimmer_ptb_counter];
    cvta.to.global.u64 %__glimmer_ptb_counter_address, %__glimmer_ptb_counter_address;
    mov.u32 %__glimmer_ptb_thread, %tid.x;
    mov.u32 %__glimmer_ptb_thread_y, %tid.y;
    or.b32 %__glimmer_ptb_thread, %__glimmer_ptb_thread, %__glimmer_ptb_thread_y;
    mov.u32 %__glimmer_ptb_thread_y, %tid.z;
    or.b32 %__glimmer_ptb_thread, %__glimmer_ptb_thread, %__glimmer_ptb_thread_y;
    setp.ne.u32 %__glimmer_ptb_follower, %__glimmer_ptb_thread, 0;
)ptx";
        if (kind == PtxExecutionKind::kPreemptive) {
            image += R"ptx(
    .reg .u64 %__glimmer_ptb_retreat_address;
    ld.param.u64 %__glimmer_ptb_retreat_address, [__glimmer_ptb_retreat];
    cvta.to.global.u64 %__glimmer_ptb_retreat_address, %__glimmer_ptb_retreat_address;
)ptx";
            if (kernel.observes_block_completion) {
                image += R"ptx(
    .reg .u64 %__glimmer_ptb_completed_address;
    .reg .u32 %__glimmer_ptb_physical_worker;
    .reg .pred %__glimmer_ptb_publish_completion, %__glimmer_ptb_leader;
    ld.param.u64 %__glimmer_ptb_completed_address, [__glimmer_ptb_completed];
    cvta.to.global.u64 %__glimmer_ptb_completed_address, %__glimmer_ptb_completed_address;
    mov.u32 %__glimmer_ptb_physical_worker, %ctaid.x;
    setp.eq.u32 %__glimmer_ptb_publish_completion, %__glimmer_ptb_physical_worker, 0;
    not.pred %__glimmer_ptb_leader, %__glimmer_ptb_follower;
    and.pred %__glimmer_ptb_publish_completion, %__glimmer_ptb_publish_completion, %__glimmer_ptb_leader;
)ptx";
            }
        }
        image += R"ptx(
__glimmer_ptb_claim_next:
    @%__glimmer_ptb_follower bra __glimmer_ptb_broadcast_ready;
)ptx";
        if (kind == PtxExecutionKind::kPreemptive) {
            image += R"ptx(
    ld.volatile.global.u32 %__glimmer_ptb_value, [%__glimmer_ptb_retreat_address];
    setp.ne.u32 %__glimmer_ptb_done, %__glimmer_ptb_value, 0;
    mov.u64 %__glimmer_ptb_claim, %__glimmer_ptb_total;
    @%__glimmer_ptb_done bra __glimmer_ptb_publish;
)ptx";
        }
        image += R"ptx(
    atom.global.add.u64 %__glimmer_ptb_claim, [%__glimmer_ptb_counter_address], 1;
__glimmer_ptb_publish:
    setp.ge.u64 %__glimmer_ptb_done, %__glimmer_ptb_claim, %__glimmer_ptb_total;
    cvt.u32.u64 %__glimmer_ptb_flat, %__glimmer_ptb_claim;
    selp.u32 %__glimmer_ptb_value, 1, 0, %__glimmer_ptb_done;
    st.shared.u32 [__glimmer_ptb_broadcast], %__glimmer_ptb_flat;
    st.shared.u32 [__glimmer_ptb_broadcast+4], %__glimmer_ptb_value;
__glimmer_ptb_broadcast_ready:
    bar.sync 0;
    ld.shared.u32 %__glimmer_ptb_flat, [__glimmer_ptb_broadcast];
    ld.shared.u32 %__glimmer_ptb_value, [__glimmer_ptb_broadcast+4];
    setp.ne.u32 %__glimmer_ptb_done, %__glimmer_ptb_value, 0;
    @%__glimmer_ptb_done ret;
)ptx";
    }
    image += R"ptx(
    rem.u32 %__glimmer_ptb_index, %__glimmer_ptb_flat, %__glimmer_ptb_grid;
    div.u32 %__glimmer_ptb_quotient, %__glimmer_ptb_flat, %__glimmer_ptb_grid;
    rem.u32 %__glimmer_ptb_y, %__glimmer_ptb_quotient, %__glimmer_ptb_grid_y;
    div.u32 %__glimmer_ptb_z, %__glimmer_ptb_quotient, %__glimmer_ptb_grid_y;
)ptx";
    image.append(body);
    // Source barriers either dominate exits without cycles or have proven
    // whole-CTA control uniformity. Their final phase has finished here, so
    // reuse barrier zero sequentially. A
    // high-numbered private barrier can impede independent-kernel concurrency.
    image += "\n__glimmer_ptb_next:\n    bar.sync 0;\n";
    if (kernel.observes_block_completion) {
        // Only physical CTA0/thread0 publishes after every source thread has
        // reached the block exit. This is NOT a claim count or stop proof.
        image += R"ptx(
    @!%__glimmer_ptb_publish_completion bra __glimmer_ptb_completion_published;
    st.release.sys.global.u32 [%__glimmer_ptb_completed_address], 1;
    mov.pred %__glimmer_ptb_publish_completion, 0;
__glimmer_ptb_completion_published:
)ptx";
    }
    if (kind == PtxExecutionKind::kBlockStatic) {
        // The final valid claim is at most UINT32_MAX - 1, and the worker
        // stride is at most UINT32_MAX. Their 64-bit sum cannot wrap.
        image += "    add.u64 %__glimmer_ptb_claim, %__glimmer_ptb_claim, %__glimmer_ptb_stride;\n";
    }
    image += kind == PtxExecutionKind::kSliced ? "    ret;\n}\n"
                                               : "    bra.uni __glimmer_ptb_claim_next;\n}\n";
    return image;
}

struct Function {
    std::size_t start{}, name{}, returns_begin{}, returns_end{}, parameters_begin{},
        parameters_end{}, header_begin{}, open_brace{}, close_brace{};
    bool is_entry{}, has_parameters{}, is_definition{};
};

bool signature_tuple(const std::vector<Token>& tokens, std::size_t& index, std::size_t& begin,
                     std::size_t& end) {
    begin = ++index;
    while (index < tokens.size() && tokens[index].text != ")") {
        if (tokens[index].text == "(" || tokens[index].text == "{" || tokens[index].text == ";")
            return false;
        ++index;
    }
    if (index == tokens.size()) return false;
    end = index++;
    return true;
}

std::optional<Function> read_function(const std::vector<Token>& tokens, std::size_t& index) {
    Function function;
    function.start = index;
    function.is_entry = tokens[index].text == ".entry";
    if (!function.is_entry && tokens[index].text != ".func") return std::nullopt;
    ++index;
    function.returns_begin = function.returns_end = index;
    if (!function.is_entry && index < tokens.size() && tokens[index].text == "(" &&
        !signature_tuple(tokens, index, function.returns_begin, function.returns_end))
        return std::nullopt;
    if (index == tokens.size() || !is_identifier(tokens[index].text)) return std::nullopt;
    function.name = index++;
    function.parameters_begin = function.parameters_end = index;
    function.has_parameters = index < tokens.size() && tokens[index].text == "(";
    if (function.has_parameters &&
        !signature_tuple(tokens, index, function.parameters_begin, function.parameters_end))
        return std::nullopt;
    if (function.is_entry && !function.has_parameters) return std::nullopt;
    function.header_begin = index;
    while (index < tokens.size() && tokens[index].text != "{" && tokens[index].text != ";") ++index;
    if (index == tokens.size()) return std::nullopt;
    if (tokens[index].text == ";") {
        ++index;
        return function;
    }
    function.is_definition = true;
    function.open_brace = index++;
    std::size_t depth = 1;
    while (index < tokens.size() && depth) {
        if (tokens[index].text == "{" && ++depth > 65) return std::nullopt;
        if (tokens[index].text == "}") --depth;
        ++index;
    }
    if (depth) return std::nullopt;
    function.close_brace = index - 1;
    return function;
}

bool signature_registers(const std::vector<Token>& tokens, std::size_t begin, std::size_t end,
                         std::vector<RegisterRange>& registers) {
    while (begin < end) {
        auto next = begin;
        while (next < end && tokens[next].text != ",") ++next;
        if (tokens[begin].text == ".reg" && !parse_registers(tokens, begin, next, registers))
            return false;
        begin = next + 1;
    }
    return true;
}

std::string_view family_suffix(PtxExecutionKind kind) {
    constexpr std::array<std::string_view, 5> suffixes{"", "_block_static", "_dynamic",
                                                       "_preemptive", "_sliced"};
    const auto index = static_cast<unsigned int>(kind);
    return index < suffixes.size() ? suffixes[index] : std::string_view{};
}

std::optional<PreparedFunction> prepare_function(std::string_view ptx,
                                                 const std::vector<Token>& tokens,
                                                 const Function& function, PtxExecutionKind kind,
                                                 const FunctionResolver& resolve) {
    const auto inputs =
        parameter_layout(tokens, function.parameters_begin, function.parameters_end, true);
    const auto outputs =
        parameter_layout(tokens, function.returns_begin, function.returns_end, true);
    std::vector<RegisterRange> registers;
    if (!inputs || !outputs || outputs->size() > 1 ||
        !valid_entry_directives(tokens, function.header_begin, function.open_brace) ||
        !signature_registers(tokens, function.parameters_begin, function.parameters_end,
                             registers) ||
        !signature_registers(tokens, function.returns_begin, function.returns_end, registers))
        return std::nullopt;
    BodyAnalysis analysis;
    if (!analyze_body(tokens, function.open_brace + 1, function.close_brace, analysis,
                      kind != PtxExecutionKind::kStatic, false, resolve, std::move(registers)) ||
        analysis.call_depth > 64)
        return std::nullopt;
    PreparedFunction prepared;
    prepared.name = std::string(tokens[function.name].text) + "__glimmer_ptb_func" +
                    std::string(family_suffix(kind));
    prepared.parameters = inputs->size();
    prepared.returns = outputs->size();
    prepared.has_barrier = analysis.has_barrier;
    prepared.call_depth = analysis.call_depth;
    prepared.callees = std::move(analysis.callees);
    prepared.prototype =
        std::string(ptx.substr(tokens[function.start].begin,
                               tokens[function.name].begin - tokens[function.start].begin)) +
        prepared.name + "(";
    if (function.has_parameters) {
        const auto begin = tokens[function.parameters_begin - 1].end;
        prepared.prototype.append(ptx.substr(begin, tokens[function.parameters_end].begin - begin));
    }
    if (!inputs->empty()) prepared.prototype += "\n,";
    for (std::size_t axis = 0; axis < kLogicalRegisters.size(); ++axis) {
        if (axis) prepared.prototype += ",\n";
        prepared.prototype += ".param .u32 __glimmer_ptb_arg_" + std::to_string(axis);
    }
    prepared.prototype += ")\n";
    const auto header_start =
        function.has_parameters ? tokens[function.parameters_end].end : tokens[function.name].end;
    prepared.prototype.append(
        ptx.substr(header_start, tokens[function.open_brace].begin - header_start));
    prepared.definition = prepared.prototype + "\n{\n";
    for (std::size_t axis = 0; axis < kLogicalRegisters.size(); ++axis) {
        const auto reg = std::string(kLogicalRegisters[axis]);
        prepared.definition.append(".reg .u32 ")
            .append(reg)
            .append(";\nld.param.u32 ")
            .append(reg)
            .append(", [__glimmer_ptb_arg_")
            .append(std::to_string(axis))
            .append("];\n");
    }
    prepared.definition +=
        rewrite_body(ptx, tokens[function.open_brace].end, tokens[function.close_brace].begin,
                     std::move(analysis.replacements));
    prepared.definition += "\n}\n";
    prepared.prototype += ";\n";
    if (prepared.definition.size() > kMaxPersistentPtxBytes ||
        prepared.prototype.size() > kMaxPersistentPtxBytes - prepared.definition.size())
        return std::nullopt;
    return prepared;
}

}  // namespace

std::optional<PersistentPtx> transform_persistent_ptx(std::string_view ptx, PtxExecutionKind kind,
                                                      bool shares_original_module) {
    return transform_persistent_ptx(ptx, kind, shares_original_module, {});
}

std::optional<PersistentPtx> transform_persistent_ptx(std::string_view ptx, PtxExecutionKind kind,
                                                      bool shares_original_module,
                                                      PtxTransformLimits limits) {
    if (!valid_ptx_transform_limits(limits) || ptx.empty() || ptx.size() > limits.source_bytes ||
        ptx.size() > limits.output_bytes || ptx.find("__glimmer_ptb") != std::string_view::npos) {
        return std::nullopt;
    }
    const auto parsed = tokenize(ptx, limits.token_count);
    if (!parsed.has_value()) {
        return std::nullopt;
    }
    const auto& tokens = *parsed;
    if (tokens.size() < 6 || tokens[0].text != ".version" || tokens[2].text != ".target" ||
        !tokens[3].text.starts_with("sm_")) {
        return std::nullopt;
    }
    if (limits.observes_block_completion) {
        const auto version = tokens[1].text;
        const auto major = parse_unsigned(version.substr(0, version.find('.')));
        const auto target = parse_unsigned(tokens[3].text.substr(3));
        if (kind != PtxExecutionKind::kPreemptive || !major || *major < 6 || !target ||
            *target < 70)
            return std::nullopt;
    }
    std::size_t first = 4;
    while (first < tokens.size() && tokens[first].text == ",") {
        if (++first == tokens.size()) return std::nullopt;
        const auto option = tokens[first++].text;
        if (option != "debug" && option != "texmode_unified" && option != "texmode_independent" &&
            option != "map_f64_to_f32")
            return std::nullopt;
    }
    if (first + 1 >= tokens.size() || tokens[first].text != ".address_size" ||
        tokens[first + 1].text != "64")
        return std::nullopt;
    PersistentPtx transformed{std::string(ptx), {}};
    std::vector<Function> entries, functions;
    std::unordered_map<std::string_view, std::size_t> definitions;
    for (std::size_t index = first + 2; index < tokens.size();) {
        if (tokens[index].text == ".file") {
            if (!skip_file(tokens, index)) return std::nullopt;
            continue;
        }
        if (tokens[index].text == ".section") {
            if (!skip_debug_section(tokens, index)) return std::nullopt;
            continue;
        }
        const bool is_weak = tokens[index].text == ".weak";
        if (tokens[index].text == ".visible" || is_weak) {
            ++index;
        }
        const bool is_external = index < tokens.size() && tokens[index].text == ".extern";
        if (is_external) ++index;
        if (index < tokens.size() &&
            (tokens[index].text == ".global" || tokens[index].text == ".const")) {
            if (!shares_original_module || !skip_module_variable(tokens, index, is_external))
                return std::nullopt;
            transformed.has_module_variables = true;
            continue;
        }
        if (index == tokens.size()) return std::nullopt;
        const auto function = read_function(tokens, index);
        if (!function || (is_external && function->is_definition)) return std::nullopt;
        if (!function->is_definition) continue;
        for (auto token = function->open_brace + 1; token < function->close_brace; ++token) {
            if (tokens[token].text == ".global" || tokens[token].text == ".const") {
                if (!shares_original_module) return std::nullopt;
                transformed.has_module_variables = true;
            }
        }
        if (function->is_entry) {
            entries.push_back(*function);
        } else {
            if (functions.size() >= 4096 ||
                !definitions.emplace(tokens[function->name].text, functions.size()).second)
                return std::nullopt;
            functions.push_back(*function);
        }
    }
    // The source inventory is now immutable. Resolve complete call graphs
    // before appending any generated entry; recursion and unknown callees reject.
    enum class Visit : std::uint8_t { kNew, kVisiting, kRejected, kReady };
    std::vector<Visit> visits(functions.size());
    std::vector<std::optional<PreparedFunction>> helpers(functions.size());
    bool depth_overflow{};
    std::function<const PreparedFunction*(std::string_view, std::size_t)> resolve;
    resolve = [&](std::string_view name, std::size_t depth) -> const PreparedFunction* {
        const auto found = definitions.find(name);
        if (depth > 64) {
            depth_overflow = true;
            return nullptr;
        }
        if (found == definitions.end()) return nullptr;
        const auto index = found->second;
        if (visits[index] == Visit::kVisiting || visits[index] == Visit::kRejected) return nullptr;
        if (visits[index] == Visit::kReady) {
            if (helpers[index]->call_depth > 65 - depth) {
                depth_overflow = true;
                return nullptr;
            }
            return &*helpers[index];
        }
        visits[index] = Visit::kVisiting;
        helpers[index] =
            prepare_function(ptx, tokens, functions[index], kind,
                             [&](std::string_view callee) { return resolve(callee, depth + 1); });
        // A too-deep caller cannot poison the same helper used by a shorter chain.
        visits[index] = helpers[index]   ? Visit::kReady
                        : depth_overflow ? Visit::kNew
                                         : Visit::kRejected;
        return helpers[index] ? &*helpers[index] : nullptr;
    };
    std::vector<bool> used(functions.size());
    std::string variants;
    for (const auto& entry : entries) {
        const auto name = tokens[entry.name].text;
        const auto parameters_begin = entry.parameters_begin, parameters_end = entry.parameters_end;
        const auto open_brace = entry.open_brace, close_brace = entry.close_brace;
        const auto layout = parameter_layout(tokens, parameters_begin, parameters_end);
        std::vector<std::string_view> parameter_names;
        for (auto index = parameters_begin; index < parameters_end; ++index)
            if (is_identifier(tokens[index].text)) parameter_names.push_back(tokens[index].text);
        BodyAnalysis analysis;
        if (!valid_entry_directives(tokens, entry.header_begin, open_brace) ||
            !layout.has_value() ||
            !analyze_body(
                tokens, open_brace + 1, close_brace, analysis, kind != PtxExecutionKind::kStatic,
                true,
                [&](std::string_view callee) {
                    depth_overflow = false;
                    return resolve(callee, 1);
                },
                {}, parameter_names)) {
            continue;
        }
        PtxKernelVariant kernel{
            std::string(name),
            std::string(name) + "__glimmer_ptb" + std::string(family_suffix(kind)), layout->size(),
            *layout, kind};
        kernel.observes_block_completion = limits.observes_block_completion;
        auto pending = std::move(analysis.callees);
        while (!pending.empty()) {
            const auto callee = definitions.at(pending.back());
            pending.pop_back();
            if (used[callee]) continue;
            used[callee] = true;
            pending.insert(pending.end(), helpers[callee]->callees.begin(),
                           helpers[callee]->callees.end());
        }
        const auto body = rewrite_body(ptx, tokens[open_brace].end, tokens[close_brace].begin,
                                       std::move(analysis.replacements));
        const auto parameter_offset = tokens[parameters_begin - 1].end;
        const auto parameters =
            ptx.substr(parameter_offset, tokens[parameters_end].begin - parameter_offset);
        const auto directives = ptx.substr(tokens[parameters_end].end,
                                           tokens[open_brace].begin - tokens[parameters_end].end);
        auto variant = kind == PtxExecutionKind::kStatic
                           ? make_variant(parameters, directives, body, kernel)
                           : make_cooperative_variant(parameters, directives, body, kernel, kind);
        if (variant.size() > kMaxPersistentPtxBytes ||
            variant.size() > limits.output_bytes - ptx.size() - variants.size())
            return std::nullopt;
        variants += variant;
        transformed.kernels.push_back(std::move(kernel));
    }
    if (transformed.kernels.empty()) {
        return std::nullopt;
    }
    const auto append = [&](std::string_view text) {
        if (text.size() > limits.output_bytes - transformed.image.size()) return false;
        transformed.image += text;
        return true;
    };
    for (std::size_t index = 0; index < helpers.size(); ++index)
        if (used[index] && !append("\n" + helpers[index]->prototype)) return std::nullopt;
    for (std::size_t index = 0; index < helpers.size(); ++index)
        if (used[index] && !append(helpers[index]->definition)) return std::nullopt;
    if (!append(variants)) return std::nullopt;
    return transformed;
}

std::optional<PersistentPtx> transform_transparent_ptx(std::string_view ptx) {
    auto independent = transform_persistent_ptx(ptx);
    auto block_static = transform_persistent_ptx(ptx, PtxExecutionKind::kBlockStatic);
    if (!independent.has_value()) {
        return block_static;
    }
    if (!block_static.has_value()) {
        return independent;
    }
    bool has_additional_entry = false;
    for (auto& kernel : block_static->kernels) {
        const auto existing = std::find_if(
            independent->kernels.begin(), independent->kernels.end(),
            [&](const auto& candidate) { return candidate.original_name == kernel.original_name; });
        if (existing == independent->kernels.end()) {
            independent->kernels.push_back(std::move(kernel));
            has_additional_entry = true;
        }
    }
    if (has_additional_entry) {
        // Original definitions appear once; appended variant names are distinct.
        if (block_static->image.size() - ptx.size() >
            kMaxPersistentPtxBytes - independent->image.size())
            return std::nullopt;
        independent->image.append(block_static->image, ptx.size(), std::string::npos);
    }
    return independent;
}

}  // namespace glimmer::transform
