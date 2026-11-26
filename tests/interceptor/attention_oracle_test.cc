#include "attention_oracle.h"

#include <iostream>
#include <limits>

namespace oracle = glimmer::tests::attention_oracle;
namespace {
void near(double actual, double expected, double tolerance = 1e-12) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance)
        throw std::runtime_error("independent attention numerical check failed");
}
void derivatives() {
    const auto parameters = oracle::seeded_parameters();
    const auto upstream = oracle::seeded_upstream();
    for (const bool residual : {false, true})
        for (const double scale : {.0, .75, -1.25})
            for (unsigned int window = 0; window != 3; ++window) {
                auto input = oracle::seeded_inputs();
                input.scale = scale;
                input.has_residual = residual;
                if (window == 1) {
                    input.lower = {0, 1};
                    input.upper = {2, 3};
                }
                if (window == 2) {
                    input.lower = {1, 2};
                    input.upper = {2, 3};
                }
                near(oracle::max_finite_difference_error(parameters, input, upstream), 0, 1e-8);
                const auto gradient = oracle::backward(parameters, input, upstream);
                for (std::size_t i = 0; i != 4; ++i)
                    near(gradient.residual[i], residual ? upstream.out[i] : 0);
                for (const auto derivative : gradient.parameters.biases[1]) near(derivative, 0);
                if (!window && scale != 0)
                    for (const auto& matrix : gradient.parameters.matrices)
                        for (const auto value : matrix)
                            if (!std::isfinite(value) || std::abs(value) < 1e-10)
                                throw std::runtime_error(
                                    "projected matrix-gradient case is trivial");
            }
}
void closed_form() {
    // Q=K=0 implies uniform attention independently of all query/key inputs.
    // Identity V/O then reduces to arithmetic window means plus known biases.
    oracle::Parameters parameters;
    parameters.matrices[2] = parameters.matrices[3] = {1, 0, 0, 1};
    parameters.biases[2] = {.125, -.25};
    parameters.biases[3] = {-.375, .5};
    auto input = oracle::seeded_inputs();
    input.lower = {0, 1};
    input.upper = {2, 3};
    input.has_residual = true;
    const auto upstream = oracle::seeded_upstream();
    const auto output = oracle::forward(parameters, input);
    const auto gradient = oracle::backward(parameters, input, upstream);
    for (std::size_t channel = 0; channel != 2; ++channel) {
        for (std::size_t query = 0; query != 2; ++query) {
            const auto first = static_cast<std::size_t>(input.lower[query]);
            const auto mean =
                .5 * (input.values[2 * first + channel] + input.values[2 * (first + 1) + channel]);
            near(output.out[2 * query + channel], mean + parameters.biases[2][channel] +
                                                      parameters.biases[3][channel] +
                                                      input.residual[2 * query + channel]);
        }
        near(gradient.values[channel], .5 * upstream.out[channel]);
        near(gradient.values[2 + channel],
             .5 * (upstream.out[channel] + upstream.out[2 + channel]));
        near(gradient.values[4 + channel], .5 * upstream.out[2 + channel]);
    }
    for (const auto value : gradient.queries) near(value, 0);
    for (const auto value : gradient.keys) near(value, 0);
    // A constant key-projection bias changes every score equally and cannot
    // affect the softmax, even with distinct windows and nonzero projections.
    auto original = oracle::seeded_parameters();
    const auto baseline = oracle::forward(original, input);
    original.biases[1] = {5.0, -7.0};
    const auto shifted = oracle::forward(original, input);
    for (std::size_t i = 0; i != 4; ++i) near(shifted.out[i], baseline.out[i]);
}
void invalid() {
    const auto parameters = oracle::seeded_parameters();
    for (const auto value :
         {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
        auto upstream = oracle::seeded_upstream();
        upstream.out[0] = value;
        bool rejected{};
        try {
            (void)oracle::max_finite_difference_error(parameters, oracle::seeded_inputs(),
                                                      upstream);
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        if (!rejected) throw std::runtime_error("non-finite gradients must not validate");
    }
    for (unsigned int kind = 0; kind != 3; ++kind) {
        auto input = oracle::seeded_inputs();
        if (!kind) input.lower[0] = -1;
        if (kind == 1) input.upper[0] = 4;
        if (kind == 2) input.upper[0] = input.lower[0];
        bool rejected{};
        try {
            (void)oracle::forward(parameters, input);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        if (!rejected) throw std::runtime_error("unsupported numerical window must be explicit");
    }
}
}  // namespace
int main() {
    try {
        derivatives();
        closed_form();
        invalid();
        std::cout << "Independent 24-parameter projected-attention oracle checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
