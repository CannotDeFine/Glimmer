#include "internal/persistent_sgemm.h"

#include <array>
#include <cstdlib>
#include <iostream>
#include <limits>

namespace {
using glimmer::interceptor::persistent_sgemm_layout;
using glimmer::interceptor::PersistentSgemmProblem;

bool require(bool condition, const char* message) {
    if (!condition) std::cerr << message << '\n';
    return condition;
}
}  // namespace

int main() {
    const std::array<PersistentSgemmProblem, 14> problems{{
        {4096, 32, 1024, 4096, 1024, 4096, 0},
        {4096, 1024, 32, 4096, 1024, 4096, 1},
        {1024, 4096, 32, 1024, 4096, 1024, 1},
        {4096, 32, 1024, 1024, 1024, 4096, 0, 1.0F, 0.0F, 1},
        {1024, 32, 4096, 4096, 4096, 1024, 0, 1.0F, 0.0F, 1},
        {768, 768, 768, 768, 768, 768, 0},
        {3072, 768, 768, 3072, 768, 3072, 0},
        {768, 768, 3072, 768, 3072, 768, 0},
        {768, 768, 768, 768, 768, 768, 1},
        {3072, 768, 768, 3072, 768, 3072, 1},
        {768, 3072, 768, 768, 3072, 768, 1},
        {768, 768, 768, 768, 768, 768, 0, 1.0F, 0.0F, 1},
        {3072, 768, 768, 768, 768, 3072, 0, 1.0F, 0.0F, 1},
        {768, 768, 3072, 3072, 3072, 768, 0, 1.0F, 0.0F, 1},
    }};
    for (const auto& problem : problems) {
        const auto layout = persistent_sgemm_layout(problem);
        if (!require(layout.has_value(), "qualified GEMM shape rejected") ||
            !require(layout->grid ==
                         std::array<unsigned int, 3>{problem.rows / 16, problem.columns / 16, 1},
                     "incorrect logical tile geometry") ||
            !require(layout->block == std::array<unsigned int, 3>{256, 1, 1}, "incorrect block") ||
            !require(layout->a_bytes == std::size_t{problem.rows} * problem.depth * sizeof(float),
                     "incorrect A extent") ||
            !require(
                layout->b_bytes == std::size_t{problem.columns} * problem.depth * sizeof(float),
                "incorrect B extent") ||
            !require(layout->c_bytes == std::size_t{problem.rows} * problem.columns * sizeof(float),
                     "incorrect C extent"))
            return EXIT_FAILURE;
        auto padded = problem;
        auto expected_reduction = problem.depth;
        if (problem.rows == 768 && problem.columns == 768) {
            if (problem.transpose_a && problem.depth == 768) expected_reduction = 128;
            if (problem.transpose_a && problem.depth == 3072) expected_reduction = 1024;
            if (!problem.transpose_a && !problem.transpose_b && problem.depth == 3072)
                expected_reduction = 768;
        }
        if (!require(layout->reduction_depth == expected_reduction &&
                         expected_reduction % 16 == 0 && problem.depth % expected_reduction == 0,
                     "incorrect contiguous reduction partition"))
            return EXIT_FAILURE;
        padded.leading_a += 17;
        padded.leading_b += 19;
        padded.leading_c += 23;
        padded.alpha = -0.75F;
        padded.beta = problem.transpose_a ? 0.0F : 0.5F;
        const auto result = persistent_sgemm_layout(padded);
        if (!require(
                result &&
                    result->a_bytes ==
                        (std::size_t{problem.transpose_a ? problem.rows - 1 : problem.depth - 1} *
                             padded.leading_a +
                         (problem.transpose_a ? problem.depth : problem.rows)) *
                            sizeof(float) &&
                    result->b_bytes == (std::size_t{problem.transpose_b ? problem.depth - 1
                                                                        : problem.columns - 1} *
                                            padded.leading_b +
                                        (problem.transpose_b ? problem.columns : problem.depth)) *
                                           sizeof(float) &&
                    result->c_bytes ==
                        (std::size_t{problem.columns - 1} * padded.leading_c + problem.rows) *
                            sizeof(float),
                "padded operand extent mismatch"))
            return EXIT_FAILURE;
        for (auto field :
             {&PersistentSgemmProblem::rows, &PersistentSgemmProblem::columns,
              &PersistentSgemmProblem::depth, &PersistentSgemmProblem::leading_a,
              &PersistentSgemmProblem::leading_b, &PersistentSgemmProblem::leading_c}) {
            auto invalid = problem;
            invalid.*field = 0;
            if (!require(!persistent_sgemm_layout(invalid), "zero extent accepted"))
                return EXIT_FAILURE;
            invalid.*field = 1;
            if (!require(!persistent_sgemm_layout(invalid), "undersized extent accepted"))
                return EXIT_FAILURE;
            invalid.*field = std::numeric_limits<std::uint32_t>::max();
            if (!require(!persistent_sgemm_layout(invalid),
                         "overflow/out-of-scope extent accepted"))
                return EXIT_FAILURE;
        }
        auto invalid = problem;
        invalid.transpose_a = 2;
        if (!require(!persistent_sgemm_layout(invalid), "unknown A transpose accepted"))
            return EXIT_FAILURE;
        if (!require(layout->bias_bytes == (problem.transpose_a ? problem.rows * sizeof(float) : 0),
                     "incorrect bias extent"))
            return EXIT_FAILURE;
        if (problem.transpose_a) {
            invalid = problem;
            invalid.beta = 0.5F;
            if (!require(!persistent_sgemm_layout(invalid), "forward nonzero beta accepted"))
                return EXIT_FAILURE;
        }
        invalid = problem;
        invalid.transpose_b = 2;
        if (!require(!persistent_sgemm_layout(invalid), "unknown transpose accepted"))
            return EXIT_FAILURE;
        invalid = problem;
        --invalid.rows;
        if (!require(!persistent_sgemm_layout(invalid), "unaligned shape accepted"))
            return EXIT_FAILURE;
        for (const auto value : {0.0F, std::numeric_limits<float>::infinity(),
                                 std::numeric_limits<float>::quiet_NaN()}) {
            invalid = problem;
            invalid.alpha = value;
            if (!require(!persistent_sgemm_layout(invalid), "unsupported alpha accepted"))
                return EXIT_FAILURE;
        }
        for (const auto value :
             {std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
              std::numeric_limits<float>::quiet_NaN()}) {
            invalid = problem;
            invalid.beta = value;
            if (!require(!persistent_sgemm_layout(invalid), "nonfinite beta accepted"))
                return EXIT_FAILURE;
        }
    }
    for (const auto& problem : std::array<PersistentSgemmProblem, 7>{{
             {768, 3072, 768, 768, 768, 768, 0},
             {768, 768, 3072, 768, 768, 768, 1},
             {3072, 3072, 768, 3072, 3072, 3072, 1},
             {768, 128, 768, 768, 768, 768, 0, 1.0F, 0.0F, 1},
             {768, 768, 768, 768, 768, 768, 1, 1.0F, 0.0F, 1},
             {2, 768, 768, 768, 768, 2, 0, 1.0F, 0.0F, 1},
             {384, 384, 64, 384, 64, 384, 0},
         }}) {
        if (!require(!persistent_sgemm_layout(problem), "unqualified BERT shape accepted"))
            return EXIT_FAILURE;
    }
    std::cout << "Persistent SGEMM layout tests passed\n";
    return EXIT_SUCCESS;
}
