#include "glimmer/transform/ptx_transform.h"

#include "uniform_loop_ptx_fixture.h"

#include <array>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <string_view>

namespace {

using glimmer::transform::PtxExecutionKind;
using glimmer::transform::transform_persistent_ptx;
constexpr std::array kCooperativeKinds{PtxExecutionKind::kBlockStatic, PtxExecutionKind::kDynamic,
                                       PtxExecutionKind::kPreemptive, PtxExecutionKind::kSliced};

bool expect(bool condition, std::string_view message) {
    if (!condition) std::cerr << message << '\n';
    return condition;
}

std::string module(std::string_view body,
                   std::string_view parameters = ".param .u32 count, .param .u64 address") {
    return ".version 7.4\n.target sm_70\n.address_size 64\n.visible .entry test(" +
           std::string(parameters) +
           ") {\n.reg .pred %p<8>; .reg .b32 %r<16>; "
           ".reg .b64 %rd<4>;\n" +
           std::string(body) + "\n}\n";
}

std::string loop(std::string_view setup, std::string_view body = "",
                 std::string_view continuation = "setp.lt.u32 %p1, %r1, %r0; @%p1 bra again;") {
    return module(std::string(setup) +
                  " mov.u32 %r1, 0; setp.eq.u32 %p0, %r0, 0; "
                  "@%p0 bra done; again: bar.sync 0; " +
                  std::string(body) + " add.u32 %r1, %r1, 1; " + std::string(continuation) +
                  " done: ret;");
}

bool accepted(std::string_view source) {
    bool passed = true;
    for (const auto kind : kCooperativeKinds) {
        const auto result = transform_persistent_ptx(source, kind);
        passed &= expect(result && result->kernels.size() == 1 && result->image.starts_with(source),
                         "proven uniform loop must retain original and one cooperative variant");
    }
    return passed;
}

bool rejected(std::string_view source) {
    bool passed = true;
    for (const auto kind : kCooperativeKinds)
        passed &= expect(!transform_persistent_ptx(source, kind),
                         "unproved repeated collective participation must remain native");
    return passed;
}

bool test_uniform_loops() {
    bool passed = accepted(kUniformLoopPtx);
    passed &= expect(!transform_persistent_ptx(kUniformLoopPtx, PtxExecutionKind::kStatic),
                     "uniform control cannot permit independent-thread warp/barrier execution");
    const auto automatic = glimmer::transform::transform_transparent_ptx(kUniformLoopPtx);
    passed &= expect(automatic && automatic->kernels.size() == 1 &&
                         automatic->kernels[0].execution_kind == PtxExecutionKind::kBlockStatic,
                     "automatic routing must select a cooperative loop family");
    for (const auto setup :
         {"ld.param.u32 %r0, [count];", "mov.u32 %r0, 7;", "mov.u32 %r0, %ntid.x;",
          "mov.u32 %r0, %ctaid.x;", "mov.u32 %r0, %nctaid.x;",
          "mov.u32 %r0, %ctaid.y; add.u32 %r0, %r0, %nctaid.z;"})
        passed &= accepted(loop(setup));
    passed &= accepted(loop("ld.param.u32 %r0, [count];",
                            "mov.u32 %r2, %tid.x; setp.eq.u32 %p3, %r2, 0; "
                            "@%p3 st.global.u32 [%rd0], %r2;"));
    passed &=
        accepted(module("ld.param.u32 %r0, [count]; setp.eq.u32 %p0, %r0, 0; "
                        "@%p0 ret; mov.u32 %r1, 0; again: bar.sync 0; "
                        "add.u32 %r1, %r1, 1; setp.lt.u32 %p1, %r1, %r0; "
                        "@%p1 bra again; ret;"));
    passed &=
        accepted(module("ld.param.u32 %r0, [count]; mov.u32 %r1, 0; "
                        "setp.eq.u32 %p0, %r0, 0; @%p0 bra done; outer: "
                        "mov.u32 %r2, 0; inner: bar.sync 0; "
                        "add.u32 %r2, %r2, 1; setp.lt.u32 %p2, %r2, 3; @%p2 bra inner; "
                        "add.u32 %r1, %r1, 1; setp.lt.u32 %p1, %r1, %r0; "
                        "@%p1 bra outer; done: ret;"));
    passed &=
        accepted(loop("ld.param.u32 %r0, [count]; mov.u32 %r4, 3; "
                      "setp.gt.u32 %p4, %r0, 7; @%p4 mov.u32 %r4, 5;",
                      "add.u32 %r4, %r4, 1;"));
    passed &=
        accepted(loop("ld.param.u32 %r0, [count]; setp.gt.u32 %p4, %r0, 7; "
                      "@%p4 bra alternate; mov.u32 %r0, 3; bra joined; "
                      "alternate: mov.u32 %r0, 5; joined:"));
    passed &=
        accepted(module(".reg .b32 counter; .reg .pred more; mov.u32 counter, 0; "
                        "again: bar.sync 0; add.u32 counter, counter, 1; "
                        "setp.lt.u32 more, counter, 3; @more bra again; ret;"));
    for (const auto kind : kCooperativeKinds) {
        const auto result = transform_persistent_ptx(kUniformLoopPtx, kind);
        if (!result) {
            passed = false;
            continue;
        }
        const auto suffix = result->image.substr(sizeof(kUniformLoopPtx) - 1);
        passed &= expect(
            suffix.find("shfl.sync.bfly.b32 %r8, %r3, 1, 31, -1;") != std::string::npos &&
                suffix.find("@%p1 bra again;") != std::string::npos &&
                suffix.find("atom.global.add.u32 %r11, [%rd2], %r10;") != std::string::npos &&
                result->kernels[0].parameter_count == 2 &&
                result->kernels[0].parameter_layout[0].size_bytes == 8 &&
                result->kernels[0].parameter_layout[1].size_bytes == 4,
            "source loop, memory effects, masks and public parameter ABI remain");
    }
    return passed;
}

bool test_rejections() {
    bool passed = true;
    const auto* conditional_update =
        "ld.param.u32 %r0, [count]; mov.u32 %r3, %tid.x; "
        "setp.eq.u32 %p3, %r3, 0; @%p3 add.u32 %r0, %r0, 1;";
    for (const auto setup :
         {"mov.u32 %r0, %tid.x;", "mov.u32 %r0, %tid.y;", "mov.u32 %r0, %laneid;",
          "mov.u64 %rd0, %globaltimer; cvt.u32.u64 %r0, %rd0;",
          "ld.param.u64 %rd0, [address]; ld.global.u32 %r0, [%rd0];",
          "ld.param.u32 %r0, [unknown];", conditional_update})
        passed &= rejected(loop(setup));
    passed &= rejected(loop("ld.param.u32 %r0, [count];", "mov.u32 %r0, %tid.x;"));
    passed &= rejected(loop("ld.param.u32 %r0, [count];", "mov.u32 %r1, %tid.x;"));
    passed &= rejected(loop("ld.param.u32 %r0, [count];",
                            "mov.u32 %r2, %tid.x; cvt.u64.u32 %rd0, %r2; "
                            "shl.b64 %rd0, %rd0, 32; mov.b64 {%r2, %r0}, %rd0;"));
    passed &=
        rejected(module("mov.u32 %r1, 3; ld.param.u64 %rd0, [address]; "
                        "mov.u32 %r2, %tid.x; mul.wide.u32 %rd1, %r2, 8; "
                        "add.u64 %rd0, %rd0, %rd1; again: bar.sync 0; "
                        "ld.global.v2.u32 {%r0, %r1}, [%rd0]; "
                        "setp.ne.u32 %p0, %r1, 0; @%p0 bra again; ret;"));
    passed &=
        rejected(module("mov.u32 %r0, 3; mov.u32 %r1, 0; "
                        "setp.lt.u32 %p1, %r1, %r0; again: bar.sync 0; "
                        "@!%p1 ret; mov.u32 %r2, %tid.x; "
                        "setp.eq.u32 %p1, %r2, 0; bra again;"));
    passed &=
        rejected(loop("ld.param.u32 %r0, [count];", "mov.u32 %r2, %tid.x; setp.eq.u32 %p1, %r2, 0;",
                      "@%p1 bra.uni again;"));
    passed &= rejected(loop("ld.param.u32 %r0, [count];",
                            "mov.u32 %r2, %tid.x; setp.eq.u32 %p3, %r2, 0; "
                            "selp.u32 %r0, 2, 3, %p3;"));
    passed &=
        rejected(loop("ld.param.u32 %r0, [count];", "mov.u32 %r2, %tid.x; setp.eq.u32 %p3, %r2, 0;",
                      "setp.lt.and.u32 %p1, %r1, %r0, %p3; @%p1 bra again;"));
    passed &= rejected(loop("ld.param.u32 %r0, [count];", "", "@%p7 bra again;"));
    passed &= rejected(loop("ld.param.u32 %r0, [count];", "addc.u32 %r0, %r0, 0;"));
    passed &= rejected(loop("ld.param.u32 %r0, [count];", "vote.sync.ballot.b32 %r0, %p3, -1;"));
    passed &=
        rejected(loop("ld.param.u32 %r0, [count];", "shfl.sync.idx.b32 %r0, %r0, 0, 31, -1;"));
    passed &= rejected(loop("ld.param.u32 %r0, [count];",
                            "mov.u32 %r3, %tid.x; setp.eq.u32 %p3, %r3, 0; @%p3 ret;"));
    passed &= rejected(loop("ld.param.u32 %r0, [count];",
                            "mov.u32 %r3, %tid.x; setp.eq.u32 %p3, %r3, 0; "
                            "@%p3 bra split; mov.u32 %r0, 2; bra join; "
                            "split: mov.u32 %r0, 3; join:"));
    passed &= rejected(loop("ld.param.u32 %r0, [count];", "{ .reg .b32 %r0; mov.u32 %r0, 3; }"));
    passed &=
        rejected(module("ld.param.u32 %r0, [count]; mov.u32 %r2, %tid.x; "
                        "setp.eq.u32 %p2, %r2, 0; @%p2 mov.u32 %r1, 0; "
                        "again: bar.sync 0; add.u32 %r1, %r1, 1; "
                        "setp.lt.u32 %p1, %r1, %r0; @%p1 bra again; ret;"));
    // A callee's .param value can differ by lane even though its spelling looks
    // like a kernel argument. Do not use kernel-parameter uniformity in helpers.
    const std::string helper =
        ".version 7.4\n.target sm_70\n.address_size 64\n"
        ".func helper(.param .u32 count) { .reg .b32 %r<2>; .reg .pred %p; "
        "ld.param.u32 %r0, [count]; mov.u32 %r1, 0; again: bar.sync 0; "
        "add.u32 %r1, %r1, 1; setp.lt.u32 %p, %r1, %r0; @%p bra again; ret; }\n"
        ".visible .entry test() { .reg .b32 %r; mov.u32 %r, %tid.x; "
        "call helper, (%r); ret; }\n";
    passed &= rejected(helper);
    passed &= rejected(loop("ld.param.u32 %r0, [count];", "st.param.u32 [count], %r1;"));
    return passed;
}

bool test_reconvergent_regions() {
    constexpr std::string_view setup =
        "ld.param.u32 %r0, [count]; mov.u32 %r2, %tid.x; mov.u32 %r3, 0; "
        "ld.param.u64 %rd0, [address]; setp.lt.u32 %p2, %r2, 16;";
    constexpr std::string_view diamond =
        "@!%p2 bra alternate; add.u32 %r3, %r3, 3; bra joined; "
        "alternate: add.u32 %r3, %r3, 7; joined:";
    bool passed = accepted(kReconvergentLoopPtx);
    passed &= expect(!transform_persistent_ptx(kReconvergentLoopPtx, PtxExecutionKind::kStatic),
                     "reconvergence cannot give Static collective support");
    const auto automatic = glimmer::transform::transform_transparent_ptx(kReconvergentLoopPtx);
    passed &=
        expect(automatic && automatic->kernels.size() == 1 &&
                   automatic->kernels.front().execution_kind == PtxExecutionKind::kBlockStatic,
               "transparent selection must use the cooperative reconvergent variant");
    passed &= accepted(loop(setup, diamond));
    passed &= accepted(loop(setup, "@%p2 bra joined; st.global.u32 [%rd0], %r2; joined:"));
    passed &= accepted(loop(setup,
                            "@!%p2 bra outer_else; setp.eq.u32 %p3, %r2, 0; "
                            "@%p3 bra inner_else; add.u32 %r3, %r3, 1; bra inner_join; "
                            "inner_else: add.u32 %r3, %r3, 2; inner_join: bra outer_join; "
                            "outer_else: add.u32 %r3, %r3, 3; outer_join:"));
    passed &= accepted(loop(
        setup, std::string(diamond) + " @%p2 bra second_join; add.u32 %r3, %r3, 11; second_join:"));
    // Region writes cannot supply uniform control, but an unconditional fresh
    // original-parameter load after the join can recover a uniform fact.
    passed &= accepted(loop(setup,
                            "@%p2 bra alternate; mov.u32 %r0, 2; bra joined; "
                            "alternate: mov.u32 %r0, 3; joined: ld.param.u32 %r0, [count];"));
    passed &= accepted(
        module(std::string(setup) +
               " mov.u32 %r1, 0; setp.eq.u32 %p0, %r0, 0; @%p0 bra done; outer: "
               "mov.u32 %r4, 0; inner: " +
               std::string(diamond) +
               " bar.sync 0; add.u32 %r4, %r4, 1; setp.lt.u32 %p4, %r4, 3; @%p4 bra inner; "
               "add.u32 %r1, %r1, 1; setp.lt.u32 %p1, %r1, %r0; @%p1 bra outer; done: ret;"));
    for (const auto kind : kCooperativeKinds) {
        const auto result = transform_persistent_ptx(kReconvergentLoopPtx, kind);
        if (!result) {
            passed = false;
            continue;
        }
        const auto suffix = result->image.substr(sizeof(kReconvergentLoopPtx) - 1);
        passed &= expect(
            suffix.find("@!%p2 bra odd;") != std::string::npos &&
                suffix.find("st.shared.u32 [%rd4], %r10;") != std::string::npos &&
                suffix.find("ld.shared.u32 %r11, [%rd5];") != std::string::npos &&
                suffix.find("@%p1 bra again;") != std::string::npos &&
                suffix.find("atom.global.add.u32 %r14, [%rd2], %r13;") != std::string::npos &&
                result->kernels[0].parameter_count == 2 &&
                result->kernels[0].parameter_layout[0].size_bytes == 8 &&
                result->kernels[0].parameter_layout[1].size_bytes == 4,
            "diamond computation, memory, trip count and public parameter ABI must be retained");
    }

    for (const auto body : {"@%p2 bra alternate; mov.u32 %r0, 2; bra joined; "
                            "alternate: mov.u32 %r0, 3; joined:",
                            "@%p2 bra joined; mov.u32 %r0, 3; joined:",
                            "@%p2 bra alternate; @%p2 mov.u32 %r0, 2; bra joined; "
                            "alternate: @!%p2 mov.u32 %r0, 3; joined:",
                            "@%p2 bra alternate; mov.u32 %r1, 0; bra joined; "
                            "alternate: mov.u32 %r1, 1; joined:",
                            "@%p2 bra alternate; setp.eq.u32 %p3, 0, 0; bra joined; "
                            "alternate: setp.eq.u32 %p3, 0, 1; joined: selp.u32 %r0, 2, 3, %p3;"})
        passed &= rejected(loop(setup, body));
    // Both predicate outputs, vector writes and implicit carry consumers must
    // remain conservative; none may hide an overwritten loop-control value.
    passed &= rejected(
        loop(setup, "@%p2 bra joined; setp.eq.u32 %p3|%p4, 0, 0; joined:", "@%p4 bra again;"));
    passed &= rejected(loop(setup, "@%p2 bra joined; mov.b64 {%r3, %r0}, %rd0; joined:"));
    passed &= rejected(loop(setup,
                            "@%p2 bra joined; add.cc.u32 %r3, %r2, 1; joined: "
                            "addc.u32 %r0, %r0, 0;"));
    for (const auto body :
         {"@%p2 ret;", "@%p2 bra joined; bar.sync 0; joined:",
          "@%p2 bra joined; shfl.sync.idx.b32 %r3, %r2, 0, 31, -1; joined:",
          "@%p2 bra joined; cycle: add.u32 %r3, %r3, 1; @%p2 bra cycle; joined:",
          "@%p2 bra alternate; bar.sync 0; bra joined; alternate: bar.sync 0; joined:"})
        passed &= rejected(loop(setup, body));
    // Even an unreachable incoming edge prevents this conservative structural
    // proof from treating the arm as a single-entry region.
    passed &=
        rejected(module(std::string(setup) +
                        " mov.u32 %r1, 0; again: bar.sync 0; @%p2 bra alternate; arm: "
                        "add.u32 %r3, %r3, 3; bra joined; alternate: add.u32 %r3, %r3, 7; joined: "
                        "add.u32 %r1, %r1, 1; setp.lt.u32 %p1, %r1, %r0; @%p1 bra again; ret; "
                        "foreign: bra arm;"));
    // An independent noncollective helper remains outside the new proof: its
    // writes and control dependence are not guessed from the caller's inputs.
    passed &= rejected(
        ".version 7.4\n.target sm_70\n.address_size 64\n.func helper() { ret; }\n" +
        loop(setup, "@%p2 bra joined; call helper; joined:")
            .substr(std::string(".version 7.4\n.target sm_70\n.address_size 64\n").size()));

    auto independent = loop(setup, diamond);
    independent += ".visible .entry unaffected() { ret; }\n";
    for (const auto kind : kCooperativeKinds) {
        const auto result = transform_persistent_ptx(independent, kind);
        passed &= expect(result && result->kernels.size() == 2,
                         "new proof must not lose unrelated entries");
    }
    std::string many_diamonds;
    for (std::size_t index = 0; index < 1200; ++index) {
        const auto label = std::to_string(index);
        many_diamonds.append("@%p2 bra alt")
            .append(label)
            .append("; mov.u32 %r3, 1; bra join")
            .append(label)
            .append("; alt")
            .append(label)
            .append(": mov.u32 %r3, 2; join")
            .append(label)
            .append(":");
    }
    passed &= rejected(loop(setup, many_diamonds));
    return passed;
}

bool test_bounds_and_independent_entries() {
    bool passed = true;
    auto source = loop("ld.param.u32 %r0, [count];");
    source += ".visible .entry untouched() { ret; }\n";
    for (const auto kind : kCooperativeKinds) {
        const auto result = transform_persistent_ptx(source, kind);
        passed &= expect(result && result->kernels.size() == 2,
                         "uniform loop preparation cannot lose independent entries");
    }
    auto oversized =
        module("mov.u32 %r0, 2; mov.u32 %r1, 0; again: bar.sync 0; " + std::string(1, ' '));
    const auto position = oversized.rfind('}');
    std::string statements;
    for (std::size_t index = 0; index < 8200; ++index) statements += "add.u32 %r1, %r1, 1;";
    statements += "setp.lt.u32 %p0, %r1, %r0; @%p0 bra again; ret;";
    oversized.insert(position, statements);
    passed &= rejected(oversized);
    return passed;
}

int emit(std::string_view source, const char* path) {
    std::string image(source);
    for (const auto kind : kCooperativeKinds) {
        const auto result = transform_persistent_ptx(source, kind);
        if (!result || result->kernels.size() != 1 || !result->image.starts_with(source))
            return EXIT_FAILURE;
        image += result->image.substr(source.size());
    }
    if (transform_persistent_ptx(source, PtxExecutionKind::kStatic)) return EXIT_FAILURE;
    std::ofstream output(path, std::ios::binary);
    output.write(image.data(), static_cast<std::streamsize>(image.size()));
    return output ? EXIT_SUCCESS : EXIT_FAILURE;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 3 && std::string_view(argv[1]) == "--emit") return emit(kUniformLoopPtx, argv[2]);
    if (argc == 3 && std::string_view(argv[1]) == "--emit-reconvergent")
        return emit(kReconvergentLoopPtx, argv[2]);
    if (argc == 4 && std::string_view(argv[1]) == "--transform") {
        std::ifstream input(argv[2], std::ios::binary);
        if (!input) return EXIT_FAILURE;
        const std::string source((std::istreambuf_iterator<char>(input)), {});
        return emit(source, argv[3]);
    }
    if (argc != 1) return EXIT_FAILURE;
    return test_uniform_loops() && test_rejections() && test_reconvergent_regions() &&
                   test_bounds_and_independent_entries()
               ? EXIT_SUCCESS
               : EXIT_FAILURE;
}
