#include "internal/ptx_transform.h"

#include "cooperative_ptx_fixture.h"
#include "persistent_ptx_fixture.h"
#include "../transform/compiler_ptx_fixture.h"
#include "../transform/computation_ptx_fixture.h"
#include "../transform/device_call_ptx_fixture.h"
#include "../transform/aggregate_ptx_fixture.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>

namespace {

using glimmer::interceptor::transform_persistent_ptx;

bool expect(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << message << '\n';
    }
    return condition;
}

std::string module(std::string_view body, std::string_view parameters = "") {
    return ".version 7.0\n.target sm_50\n.address_size 64\n.visible .entry test(" +
           std::string(parameters) + ") {\n" + std::string(body) + "\n}";
}

bool test_aggregate_parameters() {
    using Kind = glimmer::interceptor::PtxExecutionKind;
    constexpr std::array kinds{Kind::kStatic, Kind::kBlockStatic, Kind::kDynamic, Kind::kPreemptive,
                               Kind::kSliced};
    constexpr std::array<std::size_t, 4> sizes{4, 8, 32, 4}, alignments{4, 8, 16, 4};
    constexpr std::array<std::size_t, 4> offsets{0, 8, 16, 48};
    bool passed = expect(kAggregatePtx.find(kAggregateParameterText) != std::string_view::npos,
                         "independent public parameter text is exact");
    const auto oversized =
        std::string(kAggregatePtx) + std::string(glimmer::transform::kMaxPersistentPtxBytes, ' ');
    std::string maximum_count;
    for (unsigned int index = 0; index < 256; ++index) {
        if (index) maximum_count += ", ";
        maximum_count += ".param .u8 value" + std::to_string(index);
    }
    for (const auto kind : kinds) {
        const auto transformed = transform_persistent_ptx(kAggregatePtx, kind);
        if (!expect(transformed && transformed->kernels.size() == 1,
                    "all five families accept the same bounded opaque entry layout"))
            return false;
        const auto& kernel = transformed->kernels.front();
        passed &=
            expect(transformed->image.starts_with(kAggregatePtx) &&
                       kernel.original_name == "aggregate_once" && kernel.execution_kind == kind &&
                       kernel.parameter_count == 4 && kernel.parameter_layout.size() == 4,
                   "opaque aggregate remains one public parameter in original order");
        if (kernel.parameter_layout.size() != 4) return false;
        std::size_t end{};
        for (std::size_t index = 0; index < sizes.size(); ++index) {
            const auto layout = kernel.parameter_layout[index];
            passed &= expect(
                layout.size_bytes == sizes[index] && layout.alignment_bytes == alignments[index],
                "exact scalar/array sizes and alignments are family-independent");
            end = (end + alignments[index] - 1) / alignments[index] * alignments[index];
            passed &= expect(end == offsets[index], "public padded offsets are 0/8/16/48");
            end += sizes[index];
        }
        passed &= expect(end == 52, "public opaque argument bytes end at 52");
        const auto generated = transformed->image.substr(kAggregatePtx.size());
        passed &=
            expect(generated.find("(" + std::string(kAggregateParameterText) + "\n,") !=
                           std::string::npos &&
                       generated.find(".param .align 16 .b8 values[32]") != std::string::npos &&
                       generated.find("ld.param.u32 %r9, [values+28]") != std::string::npos,
                   "private entry retains exact original declarations and reads the final word");
        for (const auto parameters :
             {".param .align 1 .b8 bytes[1]", ".param .align 128 .b8 bytes[32]",
              ".param .b8 bytes[65536]", ".param .b64 bytes[8192]"}) {
            const auto boundary = transform_persistent_ptx(module("ret;", parameters), kind);
            passed &= expect(boundary && boundary->kernels.size() == 1 &&
                                 boundary->kernels.front().parameter_count == 1,
                             "inclusive alignment and payload bounds are shared by all families");
        }
        const auto count = transform_persistent_ptx(module("ret;", maximum_count), kind);
        passed &= expect(count && count->kernels.front().parameter_count == 256 &&
                             !transform_persistent_ptx(
                                 module("ret;", maximum_count + ", .param .u8 overflow"), kind),
                         "256 original declarations are distinct from appended private arguments");
        for (const auto parameters :
             {".param .align 0 .b8 a[1]", ".param .align 3 .b8 a[1]", ".param .align 256 .b8 a[1]",
              ".param .align -1 .b8 a[1]", ".param .align 18446744073709551616 .b8 a[1]",
              ".param .b8 a[0]", ".param .b8 a[-1]", ".param .b8 a[]", ".param .b8 a[2][2]",
              ".param .b8 a[65537]", ".param .b64 a[8193]", ".param .b64 a[18446744073709551615]",
              ".param .b8 a[18446744073709551616]", ".param .b8 a[65536], .param .u8 b",
              ".param .align .b8 a[1]", ".param .b8 a[1", ".param .b8 a[1],", ".reg .u32 a"})
            passed &= expect(!transform_persistent_ptx(module("ret;", parameters), kind),
                             "malformed, overflow and over-budget entry layouts stay native");
        for (const auto argument : {".reg .align 16 .u32 value", ".reg .u32 value[2]"}) {
            const auto helper = ".version 7.4\n.target sm_70\n.address_size 64\n.func helper(" +
                                std::string(argument) +
                                ") { ret; }\n.visible .entry entry() { "
                                ".reg .u32 %r; mov.u32 %r, 1; call helper, (%r); ret; }";
            passed &= expect(!transform_persistent_ptx(helper, kind),
                             "register-form helper ABI does not inherit opaque entry qualifiers");
        }
        passed &= expect(!transform_persistent_ptx(oversized, kind) &&
                             !transform_persistent_ptx(kAggregatePtx, kind, false,
                                                       {kAggregatePtx.size() - 1, 65536, 4096}),
                         "entry eligibility cannot bypass default or explicit source budgets");
    }
    for (const auto body :
         {"bar.sync 0; ret;", ".shared .u32 data[4]; ret;",
          ".reg .u32 %r; mov.u32 %r, %ctaid.y; ret;", ".reg .u32 %r; mov.u32 %r, %nctaid.z; ret;"})
        passed &= expect(!transform_persistent_ptx(module(body, kAggregateParameterText)),
                         "opaque parameters do not relax Static synchronization or grid rules");
    const auto mixed = std::string(kAggregatePtx) + "\n.visible .entry aggregate_barrier(" +
                       std::string(kAggregateParameterText) + ") { bar.sync 0; ret; }\n";
    const auto selected = glimmer::interceptor::transform_transparent_ptx(mixed);
    passed &= expect(
        selected && selected->image.starts_with(mixed) && selected->kernels.size() == 2 &&
            selected->kernels[0].original_name == "aggregate_once" &&
            selected->kernels[0].execution_kind == Kind::kStatic &&
            selected->kernels[1].original_name == "aggregate_barrier" &&
            selected->kernels[1].execution_kind == Kind::kBlockStatic &&
            selected->kernels[0].parameter_count == 4 && selected->kernels[1].parameter_count == 4,
        "mixed transparent selection prefers independent Static and retains barrier BlockStatic");
    return passed;
}

bool test_transform_limits() {
    using namespace glimmer::transform;
    constexpr std::size_t mib = std::size_t{1024} * 1024;
    constexpr PtxTransformLimits defaults{};
    bool passed =
        expect(defaults.source_bytes == 4 * mib && defaults.output_bytes == 4 * mib &&
                   defaults.token_count == 2 * mib && kMaxPtxTransformSourceBytes == 64 * mib &&
                   kMaxPtxTransformOutputBytes == 256 * mib && kMaxPtxTransformTokens == 16 * mib,
               "legacy eligibility and trusted hard ceilings remain independent");
    constexpr PtxTransformLimits maximum{64 * mib, 256 * mib, 16 * mib};
    passed &=
        expect(valid_ptx_transform_limits(maximum) && valid_ptx_transform_limits({1024, 1, 1}),
               "valid policy fields need not have source/output ordering");
    // These are fourteen lexical tokens, independent of any generator expansion.
    constexpr std::string_view source =
        ".version 7.0 .target sm_70 .address_size 64 .entry kernel ( ) { ret ; }";
    constexpr auto kind = PtxExecutionKind::kPreemptive;
    const auto baseline = transform_persistent_ptx(source, kind);
    if (!baseline) return expect(false, "bounded-policy fixture must transform");
    if (!expect(baseline && baseline->kernels.size() == 1,
                "bounded-policy fixture has one independently specified zero-arity entry"))
        return false;
    PtxTransformLimits exact{source.size(), baseline->image.size(), 14};
    const auto boundary = transform_persistent_ptx(source, kind, false, exact);
    passed &= expect(boundary && boundary->image == baseline->image &&
                         boundary->kernels.front().parameter_count == 0,
                     "inclusive source/output/token limits preserve exact generated bytes");
    for (unsigned int field = 0; field < 3; ++field) {
        auto shorter = exact;
        if (field == 0) --shorter.source_bytes;
        if (field == 1) --shorter.output_bytes;
        if (field == 2) --shorter.token_count;
        passed &= expect(!transform_persistent_ptx(source, kind, false, shorter),
                         "one-byte or one-token deficit rejects independently");
        for (const bool excessive : {false, true}) {
            auto invalid = defaults;
            if (field == 0) invalid.source_bytes = excessive ? 64 * mib + 1 : 0;
            if (field == 1) invalid.output_bytes = excessive ? 256 * mib + 1 : 0;
            if (field == 2) invalid.token_count = excessive ? 16 * mib + 1 : 0;
            passed &= expect(!valid_ptx_transform_limits(invalid) &&
                                 !transform_persistent_ptx(source, kind, false, invalid),
                             "zero/excessive trusted bounds never authorize preparation");
        }
    }
    const auto comments = std::string(source) + "\n// ignored .entry counterfeit() { ret; }\n";
    auto commented_limits = exact;
    commented_limits.source_bytes = comments.size();
    commented_limits.output_bytes += comments.size() - source.size();
    passed &= expect(transform_persistent_ptx(comments, kind, false, commented_limits).has_value(),
                     "comment text does not consume lexical-token authority");

    std::string large{source};
    large.resize(4 * mib + 1, ' ');
    passed &= expect(!transform_persistent_ptx(large, kind),
                     "legacy caller still rejects source one byte above four MiB");
    const PtxTransformLimits trusted{large.size(), 8 * mib, 14};
    for (const auto family : {PtxExecutionKind::kStatic, PtxExecutionKind::kBlockStatic,
                              PtxExecutionKind::kDynamic, kind, PtxExecutionKind::kSliced}) {
        const auto prepared = transform_persistent_ptx(large, family, false, trusted);
        passed &= expect(prepared && prepared->image.starts_with(large) &&
                             prepared->kernels.size() == 1 &&
                             prepared->kernels[0].execution_kind == family,
                         "explicit bounded whole-module policy works for all five families");
    }
    const auto stateful = std::string(source) + "\n.global .u32 state;\n";
    passed &= expect(!transform_persistent_ptx(stateful, kind, false, maximum) &&
                         transform_persistent_ptx(stateful, kind, true, maximum).has_value() &&
                         !transform_persistent_ptx(module("bra missing;"), kind, true, maximum) &&
                         !transform_persistent_ptx(module("ret;", ".param .b64 huge[65536]"), kind,
                                                   true, maximum),
                     "larger budgets do not relax ownership, grammar or per-entry ABI bounds");
    const auto legacy_again = transform_persistent_ptx(source, kind);
    passed &= expect(legacy_again && legacy_again->image == baseline->image,
                     "explicit policies never mutate the legacy transform contract");
    return passed;
}

bool test_transform() {
    const auto transformed = transform_persistent_ptx(kPersistentTestPtx);
    if (!transformed) return expect(false, "ordinary PTX must transform");
    if (!expect(transformed.has_value(), "ordinary one-dimensional PTX must be transformed")) {
        return false;
    }
    bool passed = expect(transformed->image.starts_with(kPersistentTestPtx),
                         "original entries must be retained byte-for-byte");
    passed &= expect(transformed->kernels.size() == 1 &&
                         transformed->kernels[0].original_name == "write_indices" &&
                         transformed->kernels[0].persistent_name == "write_indices__glimmer_ptb" &&
                         transformed->kernels[0].parameter_count == 2,
                     "barrier kernel must remain native while the independent entry transforms");
    const auto variant = transformed->image.substr(sizeof(kPersistentTestPtx) - 1);
    passed &= expect(variant.find("@%p1 bra __glimmer_ptb_next;") != std::string::npos,
                     "predicated return must advance to the next logical block");
    passed &= expect(variant.find("mov.u32 %r2, %__glimmer_ptb_index;") != std::string::npos &&
                         variant.find("mov.u32 %r5, %__glimmer_ptb_grid;") != std::string::npos,
                     "application block index and grid size must retain logical values");
    passed &= expect(variant.find("%__glimmer_ptb_remaining") != std::string::npos,
                     "loop termination must avoid unsigned index addition overflow");
    const auto empty = transform_persistent_ptx(module("ret;"));
    passed &= expect(empty.has_value() && empty->kernels[0].parameter_count == 0,
                     "zero-parameter entry must receive a private grid parameter");
    const auto comments =
        transform_persistent_ptx(module("// ret; bar.sync 0; %smid\n/* .shared .b32 x; */\nret;",
                                        ".param .u32 value // parameter comment\n"));
    passed &= expect(comments.has_value(), "comments must not be treated as instructions");
    const auto branches =
        transform_persistent_ptx(module(".reg .pred %p; .reg .u32 %r; mov.u32 %r, %ctaid.x; "
                                        "setp.eq.u32 %p, %r, 0; @%p bra $exit; bra.uni $loop; "
                                        "$loop: add.u32 %r, %r, 1; $exit: ret;"));
    passed &=
        expect(branches.has_value() && branches->image.find("@%p bra $exit;") != std::string::npos,
               "direct compiler branches and labels must survive transformation");
    return passed;
}

bool test_read_only_device_timer() {
    using glimmer::interceptor::PtxExecutionKind;
    const auto source = module(".reg .u64 %timer; mov.u64 %timer, %globaltimer; ret;");
    bool passed = true;
    for (const auto kind :
         {PtxExecutionKind::kStatic, PtxExecutionKind::kBlockStatic, PtxExecutionKind::kDynamic,
          PtxExecutionKind::kPreemptive, PtxExecutionKind::kSliced}) {
        const auto transformed = transform_persistent_ptx(source, kind);
        passed &= expect(
            transformed && transformed->kernels.size() == 1 &&
                transformed->image.starts_with(source) &&
                transformed->image.substr(source.size()).find("mov.u64 %timer, %globaltimer;") !=
                    std::string::npos,
            "read-only wall timer retains its original instruction in every variant");
        passed &=
            expect(!transform_persistent_ptx(
                       module(".reg .u64 %timer; mov.u64 %timer, %globaltimer_bad; ret;"), kind),
                   "unknown special registers cannot enter by timer prefix matching");
    }
    return passed;
}
bool test_transparent_selection() {
    using glimmer::interceptor::PtxExecutionKind;
    using glimmer::interceptor::transform_transparent_ptx;
    const auto mixed = transform_transparent_ptx(kPersistentTestPtx);
    if (!mixed) return expect(false, "transparent selection must transform");
    if (!expect(mixed.has_value() && mixed->kernels.size() == 2,
                "transparent selection must include independent and safe barrier entries")) {
        return false;
    }
    bool passed = expect(mixed->kernels[0].original_name == "write_indices" &&
                             mixed->kernels[0].execution_kind == PtxExecutionKind::kStatic &&
                             mixed->kernels[1].original_name == "native_barrier" &&
                             mixed->kernels[1].execution_kind == PtxExecutionKind::kBlockStatic &&
                             mixed->image.starts_with(kPersistentTestPtx) &&
                             mixed->image.find(".entry write_indices(",
                                               sizeof(kPersistentTestPtx)) == std::string::npos,
                         "selection must retain static ABI and append original definitions once");
    const auto cooperative = transform_transparent_ptx(kCooperativePtx);
    passed &= expect(cooperative.has_value() && cooperative->kernels.size() == 2 &&
                         cooperative->kernels[0].original_name == "cooperative" &&
                         cooperative->kernels[0].execution_kind == PtxExecutionKind::kBlockStatic &&
                         cooperative->kernels[0].parameter_count == 3,
                     "shared 3D kernels must select block-static without progress pointers");
    passed &=
        expect(!transform_transparent_ptx(module("bar.sync 15; ret;")).has_value() &&
                   !transform_transparent_ptx(module("again: bar.sync 0; bra again;")).has_value(),
               "unsafe synchronization must remain native in transparent selection");
    return passed;
}

bool test_owned_module_variables() {
    using glimmer::interceptor::PtxExecutionKind;
    constexpr std::string_view prefix = ".version 7.0\n.target sm_70\n.address_size 64\n";
    constexpr std::string_view declarations =
        ".visible .global .align 4 .u32 state = 7;\n"
        ".const .align 4 .u32 bias[2] = {3, 5};\n"
        ".extern .global .b8 external[];\n";
    const std::string body =
        ".visible .entry kernel() { .reg .u32 %r; ld.global.u32 %r, [state]; "
        "ld.const.u32 %r, [bias]; st.global.u32 [state], %r; ret; }\n";
    const auto source = std::string(prefix) + std::string(declarations) + body;
    bool passed = true;
    for (const auto kind :
         {PtxExecutionKind::kStatic, PtxExecutionKind::kBlockStatic, PtxExecutionKind::kDynamic,
          PtxExecutionKind::kPreemptive, PtxExecutionKind::kSliced}) {
        const auto transformed = transform_persistent_ptx(source, kind, true);
        passed &=
            expect(transformed && transformed->has_module_variables &&
                       transformed->kernels.size() == 1 && transformed->image.starts_with(source) &&
                       transformed->image.substr(source.size()).find(".global .align") ==
                           std::string::npos &&
                       transformed->image.substr(source.size()).find("ld.const.u32 %r, [bias]") !=
                           std::string::npos,
                   "owned variants reference the sole original global/constant definitions");
        passed &= expect(!transform_persistent_ptx(source, kind),
                         "legacy separate-module adapters must not duplicate module state");
    }
    for (const auto declaration :
         {".weak .global .align 8 .u64 pointer = generic(state);", ".global .s32 negative = -2;",
          ".const .f32 value = 0f3f800000;", ".global .u32 inferred[] = {1, 2};",
          ".global .align 16 .v4 .u32 vector = {1, 2, 3, 4};",
          ".global .u32 matrix[2][3] = {{1, 2, 3}, {4, 5, 6}};"}) {
        const auto transformed = transform_persistent_ptx(
            std::string(prefix) + std::string(declarations) + declaration + body,
            PtxExecutionKind::kPreemptive, true);
        passed &= expect(transformed && transformed->has_module_variables,
                         "preserve bounded native initializer syntax without evaluating it");
    }
    for (const auto declaration :
         {".global .align 3 .u32 state;", ".global .u32 state[0];",
          ".global .u32 state[4294967296];", ".global .u32 state[];", ".global .u32 state = {1, 2;",
          ".global .u32 state = {1, 2);", ".global .u32 state = ;", ".global .u32 state = {};",
          ".extern .global .u32 state = 1;", ".global .bad state;", ".shared .u32 state;",
          ".global .u32 state = { .entry injected() { ret; } };", ".global .u32 state = %r;"}) {
        passed &= expect(!transform_persistent_ptx(std::string(prefix) + declaration + body,
                                                   PtxExecutionKind::kPreemptive, true),
                         "malformed or unsupported module declarations retain original routing");
    }
    return passed;
}

bool test_compiler_structure() {
    using Kind = glimmer::interceptor::PtxExecutionKind;
    bool passed = true;
    for (const auto kind :
         {Kind::kStatic, Kind::kBlockStatic, Kind::kDynamic, Kind::kPreemptive, Kind::kSliced}) {
        const auto transformed = transform_persistent_ptx(kCompilerPtx, kind);
        passed &= expect(transformed && transformed->kernels.size() == 1 &&
                             transformed->image.starts_with(kCompilerPtx) &&
                             transformed->kernels.front().original_name == "add_once" &&
                             transformed->kernels.front().parameter_layout.size() == 2 &&
                             !transformed->has_module_variables,
                         "compiler metadata, weak entry and local storage work in every family");
        if (transformed) {
            const auto generated = transformed->image.substr(kCompilerPtx.size());
            for (const auto retained :
                 {".maxntid 32, 1, 1", ".minnctapersm 2", ".maxnreg 64", ".loc 1 26 0",
                  ".local .align 16 .b8 scratch[16]", "cvta.local.u64", "cvta.to.local.u64",
                  "st.local.u32", "ld.local.u32"})
                passed &=
                    expect(generated.find(retained) != std::string::npos,
                           "generated entry retains original constraints and local operations");
            passed &=
                expect(generated.find(".file") == std::string::npos &&
                           generated.find(".section") == std::string::npos &&
                           generated.find("mov.u32 %r2, %__glimmer_ptb_index") != std::string::npos,
                       "module debug data occurs once while executable indices are remapped");
        }
        for (const auto body :
             {"{ ret; }", ".local .u32 value; ret;", ".local .align 8 .v2 .s32 value[2]; ret;",
              ".reg .pred %p; .loc 1 0 0 @%p ret; .loc 1 2 3 ret;",
              ".loc 1 3 0, function_name label+4, inlined_at 1 2 0 ret;"})
            passed &= expect(transform_persistent_ptx(module(body), kind).has_value(),
                             "lexical scopes and debug locations preserve ordinary control flow");
        for (const auto body :
             {".loc 0 1 0 ret;", ".loc 1 ret;", ".loc 1 2 -1 ret;", ".loc 1 2 3, unknown ret;",
              ".loc 1 2 3, function_name name ret;", ".local .align 3 .u32 value; ret;",
              ".local .u32 value[]; ret;", ".local .u32 value[0]; ret;",
              ".local .u32 value[4294967296]; ret;", ".local .u32 value = 1; ret;",
              ".local .bad value; ret;", "{ ret;", "ret; }", "\"ret;\";", "\"unterminated",
              "\"bad\\\nstring\"; ret;", ".loc 1 2 3 .section .debug_str {} ret;"})
            passed &= expect(!transform_persistent_ptx(module(body), kind),
                             "malformed compiler structure cannot create a private variant");
        for (const auto header :
             {".maxntid 32", ".reqntid 8, 2, 2", ".maxnreg 32", ".maxntid 32 .maxnctapersm 2"})
            passed &= expect(
                transform_persistent_ptx(".version 7.4\n.target sm_70, debug\n.address_size 64\n"
                                         ".visible .entry test() " +
                                             std::string(header) + " { ret; }",
                                         kind)
                    .has_value(),
                "per-block constraints and target options are preserved");
        for (const auto header :
             {".maxntid 0", ".maxntid 1,", ".reqntid 1,1,1,1", ".maxnreg -1", ".maxnreg 32,64",
              ".maxntid 32 .reqntid 32", ".maxntid 32 .maxntid 64", ".minnctapersm",
              ".explicitcluster", ".maxclusterrank 2"})
            passed &=
                expect(!transform_persistent_ptx(".version 7.4\n.target sm_70\n.address_size 64\n"
                                                 ".visible .entry test() " +
                                                     std::string(header) + " { ret; }",
                                                 kind),
                       "invalid or grid-changing launch directives stay native");
    }
    for (const auto metadata :
         {R"ptx(.file 1 "file \"name\" \\ path")ptx", ".file 7 \"file name.cu\", 4294967297, 123",
          ".section .debug_info { label: .b32 1,2 .b8 0 }"})
        passed &= expect(transform_persistent_ptx(module("ret;") + "\n" + metadata).has_value(),
                         "bounded module debug data is retained without interpreting code");
    for (const auto metadata :
         {".file 0 \"name\"", ".file 1 \"\"", ".file 1 name", ".file 1 \"name\", 1",
          ".file 1 \"name\", -1, 0", ".file 1 \"name\", 18446744073709551616, 1",
          ".file 1 \"unterminated", ".section .text { .b8 0 }",
          ".section .debug_info { .entry hidden() {} }", ".section .debug_info { { .b8 0 } }",
          ".section .debug_info { .global .u32 hidden; }", ".section .debug_info { .b8 0"})
        passed &= expect(!transform_persistent_ptx(module("ret;") + "\n" + metadata),
                         "malformed debug directives cannot hide executable or module state");
    const auto too_deep = module(std::string(65, '{') + "ret;" + std::string(65, '}'));
    passed &= expect(!transform_persistent_ptx(too_deep), "lexical nesting has a fixed bound");
    for (const auto body : {".reg .pred %p; .loc 1 1 0 @%p ret; { .loc 1 2 0 bar.sync 0; } ret;",
                            "again: .loc 1 1 0 { bar.sync 0; } bra again;"})
        passed &= expect(!transform_persistent_ptx(module(body), Kind::kPreemptive),
                         "debug/scopes cannot bypass barrier dominance and cycle checks");
    return passed;
}

bool test_device_calls() {
    using Kind = glimmer::interceptor::PtxExecutionKind;
    constexpr std::string_view prefix = ".version 7.4\n.target sm_70\n.address_size 64\n";
    const auto with_functions = [&](std::string_view functions, std::string_view body) {
        return std::string(prefix) + std::string(functions) + "\n.visible .entry test() { " +
               std::string(body) + " }\n";
    };
    bool passed = true;
    for (const auto kind :
         {Kind::kStatic, Kind::kBlockStatic, Kind::kDynamic, Kind::kPreemptive, Kind::kSliced}) {
        const auto result = transform_persistent_ptx(kDeviceCallPtx, kind);
        passed &= expect(result && result->kernels.size() == 1 &&
                             result->image.starts_with(kDeviceCallPtx) &&
                             result->kernels.front().parameter_count == 2,
                         "all families clone the complete mixed-ABI direct call chain");
        if (result) {
            const auto generated = result->image.substr(kDeviceCallPtx.size());
            for (const auto name : {"leaf", "middle", "empty_helper", "register_helper"})
                passed &= expect(
                    generated.find(std::string(name) + "__glimmer_ptb_func") != std::string::npos,
                    "every reachable helper has a private family-specific target");
            passed &= expect(
                generated.find("call.uni (returned), leaf,") == std::string::npos &&
                    generated.find("mov.u32 %r1, %__glimmer_ptb_index") != std::string::npos &&
                    generated.find("mov.u32 %r7, %__glimmer_ptb_grid_y") != std::string::npos &&
                    generated.find("mov.u32 %r8, %__glimmer_ptb_grid_z") != std::string::npos,
                "nested callees observe logical coordinates, never physical grids");
            const auto entry = generated.find(".visible .entry");
            passed &= expect(
                entry != std::string::npos &&
                    generated.substr(0, entry).find("bra __glimmer_ptb_next") ==
                        std::string::npos &&
                    generated.substr(0, entry).find(kind == Kind::kStatic ? "ret;" : "ret.uni;") !=
                        std::string::npos,
                "helper returns preserve call semantics; only entries advance CTAs");
            if (kind == Kind::kStatic)
                passed &=
                    expect(generated.find("call.uni") == std::string::npos &&
                               generated.find("ret.uni") == std::string::npos,
                           "independent logical CTAs cannot retain warp-uniform call promises");
        }
        for (const auto functions :
             {".func helper() { ret; }", ".func helper(); .func helper() { ret; }",
              ".extern .func unused(); .func helper() { ret; }",
              ".func unused() { bar.sync 15; ret; } .func helper() { ret; }"})
            passed &= expect(
                transform_persistent_ptx(with_functions(functions, "call helper; ret;"), kind)
                    .has_value(),
                "prototypes and unused unsupported functions do not poison entries");
        for (const auto body :
             {"call missing; ret;", "call helper, (1); ret;",
              ".reg .u32 %r; call (%r), helper; ret;", "call helper, (1,); ret;",
              "call helper, (), prototype; ret;", ".reg .u64 %fp; call %fp; ret;"})
            passed &= expect(
                !transform_persistent_ptx(with_functions(".func helper() { ret; }", body), kind),
                "unknown/indirect/malformed/arity-mismatched calls remain original");
        for (const auto helpers :
             {".func helper() { call helper; ret; }",
              ".func second(); .func helper() { call second; ret; }"
              ".func second() { call helper; ret; }",
              ".extern .func external(); .func helper() { call external; ret; }",
              ".func helper() { bar.sync 15; ret; }"})
            passed &= expect(
                !transform_persistent_ptx(with_functions(helpers, "call helper; ret;"), kind),
                "recursive, unresolved and unsupported transitive callees reject");
        const auto unaffected = transform_persistent_ptx(
            with_functions(".func helper() { call helper; ret; }", "call helper; ret;") +
                ".visible .entry independent() { ret; }",
            kind);
        passed &= expect(unaffected && unaffected->kernels.size() == 1 &&
                             unaffected->kernels.front().original_name == "independent",
                         "a rejected call graph does not disable independent kernels");
        passed &=
            expect(transform_persistent_ptx(
                       with_functions(".func (.reg .u32 result) helper(.reg .u32 input) {"
                                      " mov.u32 result, input; ret; }",
                                      ".reg .u32 temporary; call (temporary), helper, (1); ret;"),
                       kind)
                       .has_value(),
                   "compiler-generated registers need not use a percent prefix");
    }
    const auto synchronized =
        ".func sync() { bar.sync 0; ret; } .func helper() { call sync; ret; }";
    for (const auto kind : {Kind::kBlockStatic, Kind::kDynamic, Kind::kPreemptive, Kind::kSliced}) {
        passed &=
            expect(transform_persistent_ptx(with_functions(synchronized, "call helper; ret;"), kind)
                       .has_value(),
                   "unconditional transitively uniform full-CTA helper barriers are retained");
        for (const auto body :
             {".reg .pred %p; @%p call helper; ret;", ".reg .pred %p; @%p ret; call helper; ret;",
              "loop: call helper; bra loop; ret;",
              ".reg .pred %p; @%p bra done; call helper; done: ret;"})
            passed &=
                expect(!transform_persistent_ptx(with_functions(synchronized, body), kind),
                       "helper barriers participate in caller predicate/dominance/cycle checks");
        passed &= expect(
            !transform_persistent_ptx(with_functions(".func helper() { .reg .pred %p; @%p ret;"
                                                     " bar.sync 0; ret; }",
                                                     "call helper; ret;"),
                                      kind),
            "a callee's own divergent barrier control flow remains rejected");
    }
    passed &= expect(
        !transform_persistent_ptx(with_functions(synchronized, "call helper; ret;"), Kind::kStatic),
        "independent-thread variants cannot call a CTA-synchronized helper");
    const auto uniform_source = with_functions(
        ".func helper() { .reg .pred %p; .reg .u32 %r;"
        " mov.u32 %r, %ctaid.x; setp.eq.u32 %p, %r, 1;"
        " @%p bra.uni done; done: ret.uni; }",
        ".reg .pred predicate; .reg .u32 block; mov.u32 block, %ctaid.x;"
        " setp.eq.u32 predicate, block, 1; @predicate call.uni helper; ret;");
    const auto uniform = transform_persistent_ptx(uniform_source, Kind::kStatic);
    passed &= expect(
        uniform && uniform->image.substr(uniform_source.size()).find(".uni") == std::string::npos &&
            uniform->image.substr(uniform_source.size()).find("@predicate call ") !=
                std::string::npos,
        "static calls keep predicates while dropping uniform branch/return hints");
    std::string chain;
    for (unsigned int index = 0; index < 65; ++index)
        chain += ".func helper" + std::to_string(index) + "();\n";
    for (unsigned int index = 0; index < 65; ++index) {
        chain += ".func helper" + std::to_string(index) + "() { ";
        if (index + 1 < 65) chain += "call helper" + std::to_string(index + 1) + "; ";
        chain += "ret; }\n";
    }
    const auto deep =
        transform_persistent_ptx(with_functions(chain, "call helper0; ret;") +
                                     ".visible .entry shallow() { call helper64; ret; }",
                                 Kind::kPreemptive);
    passed &= expect(
        deep && deep->kernels.size() == 1 && deep->kernels.front().original_name == "shallow",
        "excessive depth cannot poison a helper later used by a short chain");
    const auto helper_state = with_functions(".func unused() { .global .u32 state; ret; }", "ret;");
    const auto owned = transform_persistent_ptx(helper_state, Kind::kPreemptive, true);
    passed &= expect(owned && owned->has_module_variables &&
                         !transform_persistent_ptx(helper_state, Kind::kPreemptive),
                     "function-scoped module data retains explicit same-module ownership");
    return passed;
}

bool test_computation() {
    using Kind = glimmer::interceptor::PtxExecutionKind;
    bool passed = true;
    constexpr auto registers = ".reg .b32 %r; .reg .b64 %a; .reg .pred %p; ";
    for (const auto kind :
         {Kind::kStatic, Kind::kBlockStatic, Kind::kDynamic, Kind::kPreemptive, Kind::kSliced}) {
        const auto atomic = transform_persistent_ptx(kAtomicPtx, kind);
        passed &= expect(
            atomic && atomic->kernels.size() == 1 && atomic->kernels.front().parameter_count == 2 &&
                atomic->image.starts_with(kAtomicPtx) &&
                atomic->image.substr(kAtomicPtx.size())
                        .find("atom.relaxed.gpu.global.add.u32 %r7, [%rd3], %r5;") !=
                    std::string::npos &&
                atomic->image.substr(kAtomicPtx.size())
                        .find("red.release.gpu.global.add.u32 [%rd3], %r8;") != std::string::npos,
            "all families preserve atomics, reductions, scalar math and public ABI");
        for (const auto instruction :
             {"atom.global.cas.b32 %r, [%a], 0, 1;",
              "atom.acquire.sys.global.exch.b64 %a, [%a], 0;",
              "atom.acq_rel.cta.global.max.s32 %r, [%a], 1;",
              "atom.global.acquire.sys.add.u32 %r, [%a], 1;",
              "atom.acquire.global.add.u32 %r, [%a], 1;",
              "atom.global.add.noftz.f16x2 %r, [%a], %r;", "red.global.xor.b32 [%a], %r;",
              "ld.acquire.gpu.global.u32 %r, [%a];", "ld.relaxed.sys.global.u32 %r, [%a];",
              "st.release.gpu.global.u32 [%a], %r;", "st.volatile.global.u32 [%a], %r;"}) {
            const auto source = module(std::string(registers) + instruction + " ret;");
            const auto result = transform_persistent_ptx(source, kind);
            passed &= expect(result && result->image.substr(source.size()).find(instruction) !=
                                           std::string::npos,
                             "atomic and ordered-memory qualifiers are retained exactly");
        }
        for (const auto instruction :
             {"atom.cluster.global.add.u32 %r, [%a], 1;",
              "atom.relaxed.acquire.gpu.global.add.u32 %r, [%a], 1;",
              "atom.global.gpu.sys.add.u32 %r, [%a], 1;", "atom.global.unknown.u32 %r, [%a], 1;",
              "atom.global.cas.b128 %r, [%a], 0, 1;", "red.global.cas.b32 [%a], 0, 1;",
              "red.acquire.gpu.global.add.u32 [%a], 1;", "atom.global.add.noftz.bf16 %r, [%a], 1;",
              "atom.shared.local.add.u32 %r, [%a], 1;", "fence.sc.cluster;",
              "ld.acquire.cluster.global.u32 %r, [%a];", "st.acquire.gpu.global.u32 [%a], %r;",
              "ld.release.gpu.global.u32 %r, [%a];"}) {
            passed &= expect(!transform_persistent_ptx(
                                 module(std::string(registers) + instruction + " ret;"), kind),
                             "unknown, malformed and non-Sm70 memory families remain native");
        }
        const auto warp = transform_persistent_ptx(kWarpPtx, kind);
        passed &=
            expect(kind == Kind::kStatic
                       ? !warp
                       : warp && warp->kernels.size() == 1 && warp->image.starts_with(kWarpPtx) &&
                             warp->image.substr(kWarpPtx.size())
                                     .find("shfl.sync.idx.b32 %r6|%p1, %r1, 0, 31, -1;") !=
                                 std::string::npos,
                   "warp/shared computation and nested collectives require whole-CTA routing");
        for (const auto instruction :
             {"atom.shared.add.u32 %r, [%a], 1;", "red.relaxed.cta.shared.add.u32 [%a], 1;",
              "atom.add.u32 %r, [%a], 1;", "ld.acquire.cta.shared.u32 %r, [%a];", "fence.sc.sys;",
              "activemask.b32 %r;", "mov.u32 %r, %lanemask_eq;",
              "shfl.sync.down.b32 %r, %r, 1, 31, -1;", "vote.sync.all.pred %p, %p, -1;",
              "match.all.sync.b64 %r|%p, %a, -1;", "bar.warp.sync -1;"}) {
            const auto result = transform_persistent_ptx(
                module(std::string(registers) + instruction + " ret;"), kind);
            passed &=
                expect(result.has_value() == (kind != Kind::kStatic),
                       "shared/generic/warp eligibility cannot leak into independent threads");
        }
        if (kind == Kind::kStatic) continue;
        for (const auto body : {"@%p ret; shfl.sync.idx.b32 %r, %r, 0, 31, -1; ret;",
                                "@%p bra done; vote.sync.ballot.b32 %r, %p, -1; done: ret;",
                                "@%p bar.warp.sync -1; ret;",
                                "loop: match.any.sync.b32 %r, %r, -1; @%p bra loop; ret;"}) {
            passed &= expect(!transform_persistent_ptx(module(std::string(registers) + body), kind),
                             "retired, predicated or repeated collective participants stay native");
        }
        const std::string helper =
            ".version 7.4\n.target sm_70\n.address_size 64\n"
            ".func helper() { bar.warp.sync -1; ret; }\n"
            ".visible .entry test() { .reg .pred %p; @%p ret; call helper; ret; }\n";
        passed &= expect(!transform_persistent_ptx(helper, kind),
                         "nested warp collective constrains kernel early returns transitively");
        passed &= expect(
            transform_persistent_ptx(
                module(std::string(registers) + "@%p ret; atom.global.add.u32 %r, [%a], 1; ret;"),
                kind)
                .has_value(),
            "atomic side effects alone do not require convergent control flow");
    }
    const auto selected = glimmer::interceptor::transform_transparent_ptx(kWarpPtx);
    passed &= expect(selected && selected->kernels.size() == 1 &&
                         selected->kernels.front().execution_kind == Kind::kBlockStatic,
                     "automatic selection routes warp work through a cooperative family");
    return passed;
}

bool test_fallback() {
    bool passed = true;
    using glimmer::interceptor::PtxExecutionKind;
    for (const auto kind :
         {PtxExecutionKind::kStatic, PtxExecutionKind::kBlockStatic, PtxExecutionKind::kDynamic,
          PtxExecutionKind::kPreemptive, PtxExecutionKind::kSliced}) {
        const auto annotated = transform_persistent_ptx(module(".pragma \"nounroll\"; ret;"), kind);
        passed &=
            expect(annotated && annotated->image.find(".pragma \"nounroll\";") != std::string::npos,
                   "nvcc nounroll metadata must survive without changing logical control flow");
        for (const auto pragma : {".pragma \"unknown\"; ret;", ".pragma \"nounroll\", 1; ret;",
                                  ".pragma nounroll; ret;", "\"nounroll\"; ret;",
                                  ".reg .pred %p; @%p .pragma \"nounroll\"; ret;"}) {
            passed &= expect(!transform_persistent_ptx(module(pragma), kind),
                             "only the exact unpredicated nounroll annotation is supported");
        }
    }
    for (const auto body :
         {"bar.sync 0; ret;", "atom.global.add.u32 %r1, [%rd1], 1; ret;",
          ".shared .b32 buffer[32]; ret;", "call function; ret;", "brx.idx %r1, targets; ret;",
          "bra; ret;", "exit;", ".reg .u32 %r1; mov.u32 %r1, %smid; ret;",
          ".reg .u32 %r1; mov.u32 %r1, %clock; ret;", ".reg .u32 %r1; mov.u32 %r1, %ctaid.y; ret;",
          ".reg .u32 %r1; mov.u32 %r2, 1; ret;", ".reg .u32 %r<2>; mov.u32 %r2, 1; ret;",
          "ld.volatile.global.u32 %r1, [%rd1]; ret;", "/* missing terminator", "ret", "@ ret;"}) {
        passed &= expect(!transform_persistent_ptx(module(body)).has_value(),
                         std::string("unsupported body must stay native: ") + body);
    }
    for (const auto parameters : {".param .u32 value,", ".param .u64",
                                  ".param .u32 value, .param .u32", ".param .ptr value"}) {
        passed &= expect(!transform_persistent_ptx(module("ret;", parameters)).has_value(),
                         "unsupported parameter layout must stay native");
    }
    passed &= expect(
        !transform_persistent_ptx("").has_value() &&
            !transform_persistent_ptx("not PTX").has_value() &&
            !transform_persistent_ptx(module("ret;") + "\n.global .u32 state;").has_value() &&
            !transform_persistent_ptx(module("ret; // __glimmer_ptb")).has_value() &&
            !transform_persistent_ptx(
                 std::string(glimmer::interceptor::kMaxPersistentPtxBytes + 1, ' '))
                 .has_value(),
        "invalid, stateful, colliding, and oversized modules must stay native");
    return passed;
}

bool test_cooperative_variants() {
    using glimmer::interceptor::PtxExecutionKind;
    bool passed = true;
    for (const auto kind : {PtxExecutionKind::kBlockStatic, PtxExecutionKind::kDynamic,
                            PtxExecutionKind::kPreemptive, PtxExecutionKind::kSliced}) {
        const auto fixture = transform_persistent_ptx(kCooperativePtx, kind);
        passed &= expect(fixture.has_value() && fixture->kernels.size() == 2 &&
                             fixture->kernels.front().original_name == "cooperative" &&
                             fixture->kernels.front().parameter_count == 3 &&
                             fixture->kernels.front().parameter_layout.size() == 3 &&
                             fixture->kernels.front().parameter_layout[2].size_bytes == 4,
                         "bounded volatile-work fixture must select its actual entry and ABI");
    }
    const auto native_subset =
        transform_persistent_ptx(module("bar.sync 0; ret;", ".param .u64 unsupported") +
                                 "\n.visible .entry supported(.param .u64 value) { ret; }");
    passed &= expect(native_subset.has_value() && native_subset->kernels.size() == 1 &&
                         native_subset->kernels.front().original_name == "supported" &&
                         native_subset->kernels.front().parameter_count == 1,
                     "an unsupported entry must not borrow another entry's variant or ABI");
    const auto shared = module(
        ".reg .u32 %r; .shared .align 4 .b8 storage[64]; "
        "mov.u32 %r, %ctaid.y; st.shared.u32 [storage], %r; "
        "bar.sync 0; ld.shared.u32 %r, [storage]; ret;",
        ".param .u64 output");
    for (const auto kind : {PtxExecutionKind::kBlockStatic, PtxExecutionKind::kDynamic,
                            PtxExecutionKind::kPreemptive, PtxExecutionKind::kSliced}) {
        const auto result = transform_persistent_ptx(shared, kind);
        passed &=
            expect(result.has_value() && result->image.starts_with(shared) &&
                       result->kernels.size() == 1 &&
                       result->image.find("bar.sync 0;") != std::string::npos &&
                       result->image.find("bar.sync 15;") == std::string::npos &&
                       result->image.find("mov.u32 %r, %__glimmer_ptb_y;") != std::string::npos,
                   "cooperative variants must preserve shared barriers and logical 3D indices");
        if (result.has_value()) {
            const auto variant = result->image.substr(shared.size());
            passed &= expect(variant.find("__glimmer_ptb_original_grid_z") != std::string::npos,
                             "original grid dimensions must be explicit variant parameters");
            if (kind == PtxExecutionKind::kBlockStatic) {
                passed &=
                    expect(variant.find("__glimmer_ptb_counter") == std::string::npos &&
                               variant.find("atom.global") == std::string::npos &&
                               variant.find("add.u64 %__glimmer_ptb_claim") != std::string::npos,
                           "block-static PTB should stride uniformly without a claim counter");
            } else if (kind == PtxExecutionKind::kSliced) {
                passed &= expect(variant.find("__glimmer_ptb_slice_start") != std::string::npos &&
                                     variant.find("atom.global") == std::string::npos,
                                 "slicing must use checked ranges without dynamic claims");
            } else {
                passed &= expect(variant.find("atom.global.add.u64") != std::string::npos,
                                 "dynamic claims must not wrap a 32-bit counter");
                passed &= expect((variant.find("ld.volatile.global.u32") != std::string::npos) ==
                                     (kind == PtxExecutionKind::kPreemptive),
                                 "only the preemptive variant observes the retreat flag");
            }
        }
    }
    for (const auto body :
         {".reg .pred %p; @%p ret; bar.sync 0; ret;",
          ".reg .pred %p; @%p bra done; bar.sync 0; done: ret;",
          ".reg .pred %p; @%p bar.sync 0; ret;", "bar.sync 15; ret;", "bar.sync 0, 32; ret;",
          "bar.arrive 0; ret;", "bra missing; ret;", "same: ret; same: ret;", "@%p ret;",
          ".reg .pred %p; repeated: bar.sync 0; @%p bra repeated; ret;",
          ".reg .pred %p; first: bar.sync 0; bar.sync 0; @%p bra first; ret;",
          ".shared .align 3 .u32 x; ret;", ".shared .u32 x[0]; ret;",
          ".reg .u32 %r; shfl.sync.bad.b32 %r, %r, 1, 31, -1; ret;"}) {
        passed &= expect(
            !transform_persistent_ptx(module(body), PtxExecutionKind::kPreemptive).has_value(),
            std::string("unsafe cooperative control flow must stay native: ") + body);
    }
    const auto early_return = transform_persistent_ptx(module(".reg .pred %p; @%p ret; ret;"),
                                                       PtxExecutionKind::kPreemptive);
    passed &= expect(early_return.has_value(),
                     "divergent returns without original barriers may join the private footer");
    const auto aggregate = transform_persistent_ptx(
        module("ret;", ".param .u64 ptr, .param .align 16 .b8 aggregate[32], .param .u32 count"),
        PtxExecutionKind::kSliced);
    passed &= expect(aggregate.has_value() && aggregate->kernels[0].parameter_count == 3 &&
                         aggregate->kernels[0].parameter_layout[0].size_bytes == 8 &&
                         aggregate->kernels[0].parameter_layout[1].size_bytes == 32 &&
                         aggregate->kernels[0].parameter_layout[1].alignment_bytes == 16 &&
                         aggregate->kernels[0].parameter_layout[2].size_bytes == 4,
                     "owned launch metadata must preserve scalar and aligned aggregate byte sizes");
    passed &= expect(!transform_persistent_ptx(module("ret;", ".param .b64 huge[65536]"),
                                               PtxExecutionKind::kDynamic)
                          .has_value(),
                     "aggregate byte bounds must be checked before retaining layout");
    std::string many_entries = ".version 7.0\n.target sm_70\n.address_size 64\n";
    for (unsigned int index = 0; index < 2000; ++index)
        many_entries += ".visible .entry entry_" + std::to_string(index) + "() { ret; }\n";
    passed &=
        expect(many_entries.size() < glimmer::transform::kMaxPersistentPtxBytes &&
                   !transform_persistent_ptx(many_entries, PtxExecutionKind::kPreemptive),
               "small input with many entries must not expand beyond the generated-image budget");
    return passed;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--aggregate-parameters")
        return test_aggregate_parameters() ? EXIT_SUCCESS : EXIT_FAILURE;
    if ((argc == 3 && (std::string_view(argv[1]) == "--emit-compiler" ||
                       std::string_view(argv[1]) == "--emit-calls" ||
                       std::string_view(argv[1]) == "--emit-atomic" ||
                       std::string_view(argv[1]) == "--emit-warp" ||
                       std::string_view(argv[1]) == "--emit-aggregate")) ||
        (argc == 4 && (std::string_view(argv[1]) == "--transform-compiler" ||
                       std::string_view(argv[1]) == "--transform-warp"))) {
        using Kind = glimmer::interceptor::PtxExecutionKind;
        const std::string_view mode{argv[1]};
        const bool is_warp = mode == "--emit-warp" || mode == "--transform-warp";
        std::string source{mode == "--emit-aggregate" ? kAggregatePtx
                           : mode == "--emit-calls"   ? kDeviceCallPtx
                           : mode == "--emit-atomic"  ? kAtomicPtx
                           : is_warp                  ? kWarpPtx
                                                      : kCompilerPtx};
        if (argc == 4) {
            std::ifstream input(argv[2], std::ios::binary | std::ios::ate);
            const auto bytes = input.tellg();
            if (!input || bytes <= 0 ||
                static_cast<std::uint64_t>(bytes) > glimmer::transform::kMaxPersistentPtxBytes)
                return EXIT_FAILURE;
            source.resize(static_cast<std::size_t>(bytes));
            input.seekg(0);
            input.read(source.data(), static_cast<std::streamsize>(source.size()));
            if (!input) return EXIT_FAILURE;
        }
        std::string combined{source};
        for (const auto kind : {Kind::kStatic, Kind::kBlockStatic, Kind::kDynamic,
                                Kind::kPreemptive, Kind::kSliced}) {
            const auto transformed = transform_persistent_ptx(source, kind);
            if (is_warp && kind == Kind::kStatic) {
                if (transformed) return EXIT_FAILURE;
                continue;
            }
            if (!transformed || transformed->kernels.size() != 1) return EXIT_FAILURE;
            combined.append(transformed->image, source.size(), std::string::npos);
        }
        std::ofstream output(argv[argc - 1], std::ios::binary | std::ios::trunc);
        output.write(combined.data(), static_cast<std::streamsize>(combined.size()));
        output.close();
        return output ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    if (argc != 1) return EXIT_FAILURE;
    return test_aggregate_parameters() && test_transform_limits() && test_transform() &&
                   test_read_only_device_timer() && test_transparent_selection() &&
                   test_fallback() && test_cooperative_variants() &&
                   test_owned_module_variables() && test_compiler_structure() &&
                   test_device_calls() && test_computation()
               ? EXIT_SUCCESS
               : EXIT_FAILURE;
}
