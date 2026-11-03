#include "glimmer/transform/cuda_module_image.h"
#include "glimmer/transform/ptx_transform.h"

#include <array>
#include <charconv>
#include <cctype>
#include <limits>
#include <utility>

namespace glimmer::transform {
namespace {
constexpr auto kMaximum = std::numeric_limits<std::size_t>::max();
bool add(std::size_t& value, std::size_t extra) noexcept {
    if (extra > kMaximum - value) return false;
    value += extra;
    return true;
}
bool multiply(std::size_t& value, std::size_t factor) noexcept {
    if (factor && value > kMaximum / factor) return false;
    value *= factor;
    return true;
}
class Tokens {
   public:
    explicit Tokens(std::string_view input) noexcept : source_(input) {
        advance();
    }
    [[nodiscard]] std::string_view peek() const noexcept {
        return token_;
    }
    [[nodiscard]] bool valid() const noexcept {
        return is_valid_;
    }
    bool take(std::string_view value) noexcept {
        if (token_ != value) return false;
        advance();
        return true;
    }
    void advance() noexcept {
        token_ = {};
        while (position_ < source_.size()) {
            if (std::isspace(static_cast<unsigned char>(source_[position_]))) {
                ++position_;
                continue;
            }
            if (source_.substr(position_, 2) == "//") {
                const auto end = source_.find('\n', position_ + 2);
                position_ = end == std::string_view::npos ? source_.size() : end + 1;
                continue;
            }
            if (source_.substr(position_, 2) == "/*") {
                const auto end = source_.find("*/", position_ + 2);
                if (end == std::string_view::npos) {
                    is_valid_ = false;
                    position_ = source_.size();
                    return;
                }
                position_ = end + 2;
                continue;
            }
            const auto begin = position_++;
            if (source_[begin] == '"') {
                bool closed{};
                while (position_ < source_.size()) {
                    const auto character = source_[position_++];
                    if (character == '"') {
                        closed = true;
                        break;
                    }
                    if (character == '\\' && position_ < source_.size()) ++position_;
                }
                if (!closed) is_valid_ = false;
            } else if (word(source_[begin])) {
                while (position_ < source_.size() && word(source_[position_])) ++position_;
            } else if ((source_[begin] == '<' || source_[begin] == '>') &&
                       position_ < source_.size() && source_[position_] == source_[begin]) {
                ++position_;
            }
            token_ = source_.substr(begin, position_ - begin);
            return;
        }
    }

   private:
    static bool word(char value) noexcept {
        return std::isalnum(static_cast<unsigned char>(value)) || value == '_' || value == '$' ||
               value == '.';
    }
    std::string_view source_, token_;
    std::size_t position_{};
    bool is_valid_{true};
};
std::optional<std::size_t> integer(std::string_view text) noexcept {
    if (text.ends_with('U') || text.ends_with('u')) text.remove_suffix(1);
    int base = 10;
    if (text.starts_with("0x") || text.starts_with("0X")) {
        text.remove_prefix(2);
        base = 16;
    } else if (text.size() > 1 && text.front() == '0') {
        base = 8;
    }
    if (text.empty()) return std::nullopt;
    std::size_t value{};
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value, base);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) return std::nullopt;
    return value;
}
unsigned int precedence(std::string_view token) noexcept {
    if (token == "|") return 1;
    if (token == "^") return 2;
    if (token == "&") return 3;
    if (token == "<<" || token == ">>") return 4;
    if (token == "+" || token == "-") return 5;
    if (token == "*" || token == "/" || token == "%") return 6;
    return 0;
}
struct ExpressionBounds {
    unsigned int minimum{1};
    unsigned int depth{};
};
std::optional<std::size_t> expression(Tokens& tokens, ExpressionBounds bounds = {}) noexcept {
    if (bounds.depth == 32) return std::nullopt;
    std::optional<std::size_t> value;
    if (tokens.take("(")) {
        value = expression(tokens, {.minimum = 1, .depth = bounds.depth + 1});
        if (!tokens.take(")")) return std::nullopt;
    } else if (tokens.take("+")) {
        value = expression(tokens, {.minimum = 7, .depth = bounds.depth + 1});
    } else {
        value = integer(tokens.peek());
        tokens.advance();
    }
    if (!value) return std::nullopt;
    while (precedence(tokens.peek()) >= bounds.minimum) {
        const auto operation = tokens.peek();
        const auto rank = precedence(operation);
        tokens.advance();
        const auto right = expression(tokens, {.minimum = rank + 1, .depth = bounds.depth + 1});
        if (!right) return std::nullopt;
        if (operation == "+" && !add(*value, *right)) return std::nullopt;
        if (operation == "-") {
            if (*right > *value) return std::nullopt;
            *value -= *right;
        }
        if (operation == "*" && !multiply(*value, *right)) return std::nullopt;
        if (operation == "/" || operation == "%") {
            if (!*right) return std::nullopt;
            *value = operation == "/" ? *value / *right : *value % *right;
        }
        if (operation == "<<" || operation == ">>") {
            if (*right >= std::numeric_limits<std::size_t>::digits) return std::nullopt;
            if (operation == "<<" && *value > (kMaximum >> *right)) return std::nullopt;
            *value = operation == "<<" ? *value << *right : *value >> *right;
        }
        if (operation == "&") *value &= *right;
        if (operation == "|") *value |= *right;
        if (operation == "^") *value ^= *right;
    }
    return value;
}
bool identifier(std::string_view text) noexcept {
    if (text.empty() || !(std::isalpha(static_cast<unsigned char>(text.front())) ||
                          text.front() == '_' || text.front() == '$'))
        return false;
    for (const auto value : text)
        if (!(std::isalnum(static_cast<unsigned char>(value)) || value == '_' || value == '$'))
            return false;
    return true;
}
std::optional<std::size_t> type_bytes(std::string_view type) noexcept {
    constexpr std::array<std::pair<std::string_view, std::size_t>, 19> sizes{{{".b8", 1},
                                                                              {".u8", 1},
                                                                              {".s8", 1},
                                                                              {".b16", 2},
                                                                              {".u16", 2},
                                                                              {".s16", 2},
                                                                              {".f16", 2},
                                                                              {".bf16", 2},
                                                                              {".b32", 4},
                                                                              {".u32", 4},
                                                                              {".s32", 4},
                                                                              {".f32", 4},
                                                                              {".f16x2", 4},
                                                                              {".bf16x2", 4},
                                                                              {".b64", 8},
                                                                              {".u64", 8},
                                                                              {".s64", 8},
                                                                              {".f64", 8},
                                                                              {".b128", 16}}};
    for (const auto& [name, bytes] : sizes)
        if (type == name) return bytes;
    return std::nullopt;
}
// Count outer initializer elements; nested rows consume one first-dimension
// element, including partially initialized rows. Values remain SDK-validated.
std::optional<std::size_t> initializer(Tokens& tokens) noexcept {
    std::array<std::string_view, 32> stack{};
    std::size_t depth{}, count{};
    const bool is_list = tokens.peek() == "{";
    bool has_element{}, has_value{};
    while (!tokens.peek().empty()) {
        const auto token = tokens.peek();
        if (!depth && (token == ";" || token == ","))
            return is_list     ? std::optional{count}
                   : has_value ? std::optional<std::size_t>{1}
                               : std::nullopt;
        tokens.advance();
        has_value = true;
        if (token == "(" || token == "[" || token == "{") {
            if (depth == stack.size()) return std::nullopt;
            if (depth == 1 && is_list) has_element = true;
            stack[depth++] = token;
        } else if (token == ")" || token == "]" || token == "}") {
            const auto expected = token == ")" ? "(" : token == "]" ? "[" : "{";
            if (!depth || stack[depth - 1] != expected) return std::nullopt;
            if (depth == 1 && is_list && has_element && !add(count, 1)) return std::nullopt;
            --depth;
        } else if (token == "," && depth == 1 && is_list) {
            if (!has_element || !add(count, 1)) return std::nullopt;
            has_element = false;
        } else {
            if (token == ";" || token == "=" || token == ".global" || token == ".const")
                return std::nullopt;
            if (depth == 1 && is_list) has_element = true;
        }
    }
    return std::nullopt;
}
bool declaration(Tokens& tokens, bool is_external, std::size_t& total) noexcept {
    std::size_t alignment{}, vector_width{1};
    for (;;) {
        if (tokens.take(".align")) {
            if (alignment) return false;
            const auto parsed = expression(tokens);
            if (!parsed || !*parsed || (*parsed & (*parsed - 1))) return false;
            alignment = *parsed;
        } else if (tokens.peek() == ".v2" || tokens.peek() == ".v4") {
            if (vector_width != 1) return false;
            vector_width = tokens.peek() == ".v2" ? 2 : 4;
            tokens.advance();
        } else if (tokens.take(".attribute")) {
            // Attribute names do not change a declared variable's extent.
            if (!tokens.take("(") || !tokens.take(".managed") || !tokens.take(")")) return false;
        } else {
            break;
        }
    }
    auto width = type_bytes(tokens.peek());
    if (!width || !multiply(*width, vector_width) || *width > 16) return false;
    tokens.advance();
    if (!alignment) alignment = *width;
    for (;;) {
        if (!identifier(tokens.peek())) return false;
        tokens.advance();
        std::size_t elements{1}, dimensions{};
        bool is_inferred{};
        if (tokens.take("<")) {
            const auto count = expression(tokens);
            if (!count || !*count || !tokens.take(">")) return false;
            elements = *count;
        }
        while (tokens.take("[")) {
            if (++dimensions > 32) return false;
            if (tokens.take("]")) {
                if (dimensions != 1) return false;
                is_inferred = true;
            } else {
                const auto count = expression(tokens);
                if (!count || !*count || !tokens.take("]") || !multiply(elements, *count))
                    return false;
            }
        }
        if (tokens.take("=")) {
            if (is_external || (is_inferred && tokens.peek() != "{")) return false;
            const auto count = initializer(tokens);
            if (!count || (is_inferred && (!*count || !multiply(elements, *count)))) return false;
        } else if (is_inferred && !is_external) {
            return false;
        }
        if (!is_external &&
            (!multiply(elements, *width) || !add(elements, alignment - 1) || !add(total, elements)))
            return false;
        if (!tokens.take(",")) return tokens.take(";");
    }
}
}  // namespace

std::optional<std::size_t> inspect_ptx_static_storage(std::string_view source) noexcept {
    return inspect_ptx_static_storage(source, kMaxCudaModuleBytes);
}

std::optional<std::size_t> inspect_ptx_static_storage(std::string_view source,
                                                      std::size_t source_bytes) noexcept {
    if (!source_bytes || source_bytes > kMaxPtxTransformOutputBytes ||
        source.size() > source_bytes || source.find('\0') != std::string_view::npos)
        return std::nullopt;
    Tokens tokens(source);
    std::size_t total{}, braces{}, parentheses{};
    bool is_external{}, is_pointer{};
    while (!tokens.peek().empty()) {
        const auto token = tokens.peek();
        tokens.advance();
        if (token == "{") {
            if (++braces > 256) return std::nullopt;
        } else if (token == "}") {
            if (!braces) return std::nullopt;
            --braces;
        } else {
            if (token == "(") {
                if (++parentheses > 32) return std::nullopt;
            } else if (token == ")") {
                if (!parentheses) return std::nullopt;
                --parentheses;
            } else if (!parentheses) {
                if (token == ".ptr") is_pointer = true;
                if (token == ".extern") is_external = true;
                if (token == ".entry" || token == ".func" || token == ";") {
                    is_external = false;
                    is_pointer = false;
                }
                if (!is_pointer && (token == ".global" || token == ".const")) {
                    if (!declaration(tokens, is_external, total)) return std::nullopt;
                    is_external = false;
                }
            }
        }
    }
    return tokens.valid() && !braces && !parentheses ? std::optional{total} : std::nullopt;
}
}  // namespace glimmer::transform
