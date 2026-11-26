#include "internal/embedded_ptx.h"
#include "internal/ptx_transform.h"
#include "persistent_ptx_fixture.h"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>

namespace {

bool expect(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << message << '\n';
    }
    return condition;
}

bool run(const std::string& tool) {
    using glimmer::interceptor::extract_embedded_ptx;
    using glimmer::interceptor::parse_embedded_ptx;
    const auto output = std::string("Fatbin ptx code:\ncompressed\n") + kPersistentTestPtx +
                        "\nFatbin elf code:\narch = sm_70\n";
    const auto parsed = parse_embedded_ptx(output);
    bool passed =
        expect(parsed.has_value() && parsed->size() == 1 &&
                   glimmer::interceptor::transform_persistent_ptx(parsed->front()).has_value(),
               "cuobjdump framing must leave valid standalone PTX");
    passed &= expect(!parse_embedded_ptx("Fatbin elf code:\n").has_value() &&
                         !parse_embedded_ptx(std::string(".version 7.0\0", 13)).has_value(),
                     "absent and binary output must stay native");
    passed &=
        expect(!extract_embedded_ptx("relative-tool", "/good").has_value() &&
                   !extract_embedded_ptx(tool, "relative-binary").has_value() &&
                   !extract_embedded_ptx("/nonexistent/glimmer-cuobjdump", "/good").has_value(),
               "invalid and missing tools must fail without execution");
    const auto extracted = extract_embedded_ptx(tool, "/good; not a shell command");
    passed &= expect(extracted.has_value() && extracted->size() == 1,
                     "argv extraction must preserve literal arguments and sanitize the child");
    const auto duplicate = extract_embedded_ptx(tool, "/duplicate");
    passed &= expect(duplicate.has_value() && duplicate->size() == 2,
                     "multiple architectures must remain separate images");
    for (const auto mode : {"/failure", "/no-ptx", "/overflow"}) {
        passed &= expect(!extract_embedded_ptx(tool, mode).has_value(),
                         "failed, absent, or excessive tool output must stay native");
    }
    const auto started = std::chrono::steady_clock::now();
    passed &=
        expect(!extract_embedded_ptx(tool, "/timeout", std::chrono::milliseconds{50}).has_value() &&
                   std::chrono::steady_clock::now() - started < std::chrono::seconds{2},
               "hung owned child must be terminated and reaped within the timeout");
    return passed;
}

}  // namespace

int main(int argc, char** argv) {
    return argc == 2 && run(argv[1]) ? EXIT_SUCCESS : EXIT_FAILURE;
}
