#include "glimmer/transform/cuda_module_image.h"
#include "glimmer/transform/ptx_transform.h"

#include <array>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace {
using glimmer::transform::inspect_ptx_static_storage;
void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void declarations() {
    const std::array<std::pair<std::string_view, std::size_t>, 16> cases{{
        {".global .u32 a;", 7},
        {".global .align 1 .b8 a[16];", 16},
        {".const .v4 .f32 vector = {1,2,3,4};", 31},
        {".global .align 16 .b8 a[4];", 19},
        {".global .b16 a[3][5];", 31},
        {".const .u32 a[] = {1,2,3};", 15},
        {".global .s32 a[][2] = {{-1,0},{0,-1},{1,0},{0,1}};", 35},
        {".global .u64 pointers<3>;", 31},
        {".global .u8 a, b[4];", 5},
        {".global .u32 a[(2+3)*4-(8/2)][1<<2];", 259},
        {".global .b8 a[(0x10U|2)^(3&1)];", 19},
        {".global .u8 a[17%8];", 1},
        {".extern .global .b8 external[]; .visible .global .u32 a;", 7},
        {".common .global .attribute(.managed) .u64 a;", 15},
        {".weak .global .align 8 .u64 a = generic(other+8);", 15},
        {".const .u16 a[010];", 17},
    }};
    for (const auto& [source, bytes] : cases)
        expect(inspect_ptx_static_storage(source) == bytes,
               "scalar/vector/array sizing includes each declared alignment bound");
    expect(inspect_ptx_static_storage(".file 1 \"fake .global .u64 evil[999999]; { \\\" }\"\n"
                                      "// .global .u8 fake[999999];\n"
                                      "/* .const .u32 fake[999999]; */\n"
                                      ".entry kernel(.param .u64 .ptr .global address) {\n"
                                      ".param .u64 .ptr .global argument; .reg .u32 %r<128>;\n"
                                      ".local .b8 scratch[1024]; .shared .b8 tile[1024];\n"
                                      "ld.global.u32 %r0, [address]; ret; }") == 0,
           "strings/comments, parameter pointers, code and per-thread/SM data are not module data");
    expect(inspect_ptx_static_storage(
               ".func helper() { .global .b8 scoped[9]; { .const .u32 table[2] = {1,2}; } ret; }"
               ".entry kernel() { ret; }") == 20,
           "function-scoped global/constant data cannot disappear from the module inventory");
    expect(inspect_ptx_static_storage(".extern .func function(.param .u64 x); .global .u8 a;") == 1,
           "external function linkage cannot suppress later owned variable charges");
}
void invalid() {
    for (const auto source : {".global .align 3 .u32 a;",
                              ".global .align 0 .u32 a;",
                              ".global .u32 a[0];",
                              ".global .u64 a[18446744073709551615];",
                              ".global .b8 a[18446744073709551615+1];",
                              ".global .b8 a[1<<64];",
                              ".global .b8 a[1/0];",
                              ".global .b8 a[1-2];",
                              ".global .b8 a[unknown];",
                              ".global .b8 a[];",
                              ".global .b8 a[][0];",
                              ".global .b8 a[2][];",
                              ".global .u32 a[] = 1;",
                              ".global .u32 a[] = {};",
                              ".global .u32 a = ;",
                              ".global .u32 a = {1,2;",
                              ".global .u32 a = {1,2);",
                              ".extern .global .u32 a = 1;",
                              ".global .v4 .f64 a;",
                              ".global .pred a;",
                              ".global .texture a;",
                              ".global .u32 a",
                              ".entry x() {",
                              ".entry x( {}",
                              "/* unterminated",
                              ".file 1 \"unterminated"}) {
        expect(!inspect_ptx_static_storage(source),
               "unsupported or overflowing owned extents reject before SDK loading");
    }
    expect(!inspect_ptx_static_storage(std::string{"x\0y", 3}),
           "embedded terminator cannot hide later declarations");
    expect(
        !inspect_ptx_static_storage(std::string(glimmer::transform::kMaxCudaModuleBytes + 1, ' ')),
        "streaming input still has an explicit byte bound");
    std::string expression(40, '(');
    expression += "1";
    expression += std::string(40, ')');
    expect(!inspect_ptx_static_storage(".global .b8 a[" + expression + "];"),
           "constant expression recursion is bounded");
}
void explicit_generated_image_bounds() {
    namespace transform = glimmer::transform;
    constexpr std::string_view source = ".global .u32 value;";
    expect(inspect_ptx_static_storage(source, source.size()) == 7,
           "an explicit byte bound includes its exact boundary");
    expect(!inspect_ptx_static_storage(source, source.size() - 1),
           "explicit generated-image scanning rejects one byte over budget");
    expect(!inspect_ptx_static_storage(source, 0) &&
               !inspect_ptx_static_storage(source, transform::kMaxPtxTransformOutputBytes + 1),
           "zero and excessive trusted bounds reject even a small source");
    expect(!inspect_ptx_static_storage(std::string{"x\0y", 3}, 3) &&
               !inspect_ptx_static_storage(".global .u32 value[unknown];", 1024),
           "a larger bound cannot hide invalid text or unknown static storage");
    std::string generated(transform::kMaxCudaModuleBytes + 1, ' ');
    generated += source;
    expect(!inspect_ptx_static_storage(generated) &&
               inspect_ptx_static_storage(generated, generated.size()) == 7,
           "only explicit generated-image policy scans beyond the legacy64MiB bound");
}
}  // namespace
int main() {
    try {
        declarations();
        invalid();
        explicit_generated_image_bounds();
        std::cout << "PTX static-storage inventory checks passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
