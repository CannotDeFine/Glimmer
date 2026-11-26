#include "sgemm_issue_range.h"

#include <cstdlib>
#include <stdexcept>
#include <string_view>

namespace {
using glimmer::app::ExecutionSgemmRangeHooks;
using glimmer::app::SgemmIssueRangeTracker;
int pushes{}, pops{}, push_result{}, pop_result{};
bool throws_push{}, throws_pop{};

void expect(bool value) {
    if (!value) throw std::runtime_error("SGEMM issue-range assertion failed");
}
int push(const char* label) {
    expect(std::string_view(label) == "glimmer.sgemm.target.issue.v1");
    ++pushes;
    if (throws_push) throw std::runtime_error("push failure");
    return push_result;
}
int pop() {
    ++pops;
    if (throws_pop) throw std::runtime_error("pop failure");
    return pop_result;
}
void reset() {
    pushes = pops = 0;
    push_result = pop_result = 0;
    throws_push = throws_pop = false;
}
void record(SgemmIssueRangeTracker& tracker, bool target, bool resume, bool issued = true,
            int native = 0, int completion = 0, bool has_library = false, int library = 0) {
    const auto first = tracker.begin(target, resume);
    tracker.finish(first, issued, {.native = native, .completion = completion}, has_library,
                   library);
}
void cases() {
    const ExecutionSgemmRangeHooks hooks{push, pop};
    reset();
    SgemmIssueRangeTracker success(hooks);
    record(success, false, false);
    expect(!pushes && !pops && !success.summary().attempts);
    record(success, true, false);
    auto info = success.summary();
    expect(pushes == 1 && pops == 1 && info.attempts == 1 && info.resumes == 0 && info.push == 0 &&
           info.pop == 0 && info.first_issued && info.native_status == 0 &&
           info.completion_status == 0 && !info.has_library_status && info.library_status == -1);

    reset();
    SgemmIssueRangeTracker retried(hooks);
    record(retried, true, false, false, 600);
    record(retried, true, false);
    info = retried.summary();
    expect(pushes == 1 && pops == 1 && info.attempts == 2 && !info.first_issued &&
           info.native_status == 600);

    reset();
    SgemmIssueRangeTracker resumed(hooks);
    record(resumed, true, false);
    record(resumed, true, true);
    info = resumed.summary();
    expect(pushes == 1 && pops == 1 && info.attempts == 2 && info.resumes == 1);

    reset();
    SgemmIssueRangeTracker first_resume(hooks);
    record(first_resume, true, true);
    record(first_resume, true, false);
    info = first_resume.summary();
    expect(!pushes && !pops && info.attempts == 2 && info.resumes == 1 && !info.first_issued);

    reset();
    push_result = -1;
    SgemmIssueRangeTracker missing(hooks);
    record(missing, true, false);
    info = missing.summary();
    expect(pushes == 1 && !pops && info.push == -1 && info.pop == -1 && info.first_issued);

    reset();
    throws_push = true;
    SgemmIssueRangeTracker push_exception(hooks);
    record(push_exception, true, false);
    expect(pushes == 1 && !pops && push_exception.summary().push == -1);

    reset();
    throws_pop = true;
    SgemmIssueRangeTracker pop_exception(hooks);
    record(pop_exception, true, false, true, 0, 0, true, 13);
    info = pop_exception.summary();
    expect(pushes == 1 && pops == 1 && info.pop == -1 && info.has_library_status &&
           info.library_status == 13);
}
}  // namespace

int main() {
    try {
        cases();
        return EXIT_SUCCESS;
    } catch (...) {
        return EXIT_FAILURE;
    }
}
