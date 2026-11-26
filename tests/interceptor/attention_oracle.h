#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>

// Independent one-head mathematical attention. No native weight offsets,
// SeqData packing, dropout, opaque reserve contents or GPU behavior are modeled.
namespace glimmer::tests::attention_oracle {
inline constexpr std::size_t kQuerySteps = 2, kKeySteps = 3, kWidth = 2;
struct Parameters {
    // Q, K, V, O; rows are projected/output coordinates, columns are inputs.
    std::array<std::array<double, 4>, 4> matrices{};
    std::array<std::array<double, 2>, 4> biases{};
};
struct Inputs {
    std::array<double, 4> queries{};
    std::array<double, 6> keys{}, values{};
    std::array<double, 4> residual{};
    std::array<int, 2> lower{0, 0}, upper{3, 3};
    double scale{.75};
    bool has_residual{};
};
struct Upstream {
    std::array<double, 4> out{};
};
struct Output {
    std::array<double, 4> out{};
};
struct Gradient {
    Parameters parameters;
    std::array<double, 4> queries{};
    std::array<double, 6> keys{}, values{};
    // This is mathematically dout when enabled. Native BackwardData does not
    // publish this residual gradient or add it to dqueries automatically.
    std::array<double, 4> residual{};
};
namespace detail {
using Vector = std::array<double, 2>;
struct Evaluation {
    Output output;
    std::array<Vector, kQuerySteps> queries{}, context{};
    std::array<Vector, kKeySteps> keys{}, values{};
    std::array<std::array<double, kKeySteps>, kQuerySteps> probabilities{};
};
inline void validate(const Inputs& input) {
    if (!std::isfinite(input.scale)) throw std::invalid_argument("non-finite attention scale");
    for (std::size_t t = 0; t != kQuerySteps; ++t)
        if (input.lower[t] < 0 || input.upper[t] > static_cast<int>(kKeySteps) ||
            input.lower[t] >= input.upper[t])
            throw std::invalid_argument("attention oracle requires a bounded nonempty window");
}
template <std::size_t N>
inline Vector project(const Parameters& parameters, std::size_t kind,
                      const std::array<double, N>& input, std::size_t time) {
    Vector result = parameters.biases[kind];
    for (std::size_t row = 0; row != kWidth; ++row)
        for (std::size_t col = 0; col != kWidth; ++col)
            result[row] +=
                parameters.matrices[kind][kWidth * row + col] * input[kWidth * time + col];
    return result;
}
inline Evaluation evaluate(const Parameters& parameters, const Inputs& input) {
    validate(input);
    Evaluation result;
    for (std::size_t time = 0; time != kKeySteps; ++time) {
        result.keys[time] = project(parameters, 1, input.keys, time);
        result.values[time] = project(parameters, 2, input.values, time);
    }
    for (std::size_t time = 0; time != kQuerySteps; ++time) {
        result.queries[time] = project(parameters, 0, input.queries, time);
        const auto lower = static_cast<std::size_t>(input.lower[time]);
        const auto upper = static_cast<std::size_t>(input.upper[time]);
        std::array<double, kKeySteps> scores{};
        for (std::size_t key = lower; key != upper; ++key)
            for (std::size_t channel = 0; channel != kWidth; ++channel)
                scores[key] +=
                    input.scale * result.queries[time][channel] * result.keys[key][channel];
        const auto maximum = *std::max_element(scores.begin() + lower, scores.begin() + upper);
        double denominator{};
        for (std::size_t key = lower; key != upper; ++key) {
            result.probabilities[time][key] = std::exp(scores[key] - maximum);
            denominator += result.probabilities[time][key];
        }
        for (std::size_t key = lower; key != upper; ++key) {
            auto& probability = result.probabilities[time][key];
            probability /= denominator;
            for (std::size_t channel = 0; channel != kWidth; ++channel)
                result.context[time][channel] += probability * result.values[key][channel];
        }
        const auto output = project(parameters, 3, result.context[time], 0);
        for (std::size_t channel = 0; channel != kWidth; ++channel)
            result.output.out[kWidth * time + channel] =
                output[channel] +
                (input.has_residual ? input.residual[kWidth * time + channel] : 0.0);
    }
    return result;
}
template <std::size_t N>
inline void reverse_projection(const Parameters& parameters, std::size_t kind,
                               const std::array<double, N>& input, std::size_t time,
                               const Vector& derivative, Parameters& parameter_gradient,
                               std::array<double, N>& input_gradient) {
    for (std::size_t row = 0; row != kWidth; ++row) {
        parameter_gradient.biases[kind][row] += derivative[row];
        for (std::size_t col = 0; col != kWidth; ++col) {
            parameter_gradient.matrices[kind][kWidth * row + col] +=
                derivative[row] * input[kWidth * time + col];
            input_gradient[kWidth * time + col] +=
                parameters.matrices[kind][kWidth * row + col] * derivative[row];
        }
    }
}
}  // namespace detail
inline Output forward(const Parameters& parameters, const Inputs& input) {
    return detail::evaluate(parameters, input).output;
}
inline double loss(const Output& output, const Upstream& upstream) {
    double result{};
    for (std::size_t i = 0; i != output.out.size(); ++i) result += output.out[i] * upstream.out[i];
    return result;
}
inline Gradient backward(const Parameters& parameters, const Inputs& input,
                         const Upstream& upstream) {
    const auto cache = detail::evaluate(parameters, input);
    Gradient result;
    std::array<detail::Vector, kKeySteps> dkeys{}, dvalues{};
    for (std::size_t time = 0; time != kQuerySteps; ++time) {
        const detail::Vector dout{upstream.out[kWidth * time], upstream.out[kWidth * time + 1]};
        detail::Vector dcontext{}, dquery{};
        detail::reverse_projection(parameters, 3, cache.context[time], 0, dout, result.parameters,
                                   dcontext);
        if (input.has_residual)
            for (std::size_t channel = 0; channel != kWidth; ++channel)
                result.residual[kWidth * time + channel] = dout[channel];
        const auto lower = static_cast<std::size_t>(input.lower[time]);
        const auto upper = static_cast<std::size_t>(input.upper[time]);
        std::array<double, kKeySteps> dprobabilities{};
        double expectation{};
        for (std::size_t key = lower; key != upper; ++key) {
            for (std::size_t channel = 0; channel != kWidth; ++channel) {
                dvalues[key][channel] += cache.probabilities[time][key] * dcontext[channel];
                dprobabilities[key] += cache.values[key][channel] * dcontext[channel];
            }
            expectation += cache.probabilities[time][key] * dprobabilities[key];
        }
        for (std::size_t key = lower; key != upper; ++key) {
            const auto dscore =
                cache.probabilities[time][key] * (dprobabilities[key] - expectation);
            for (std::size_t channel = 0; channel != kWidth; ++channel) {
                dquery[channel] += input.scale * dscore * cache.keys[key][channel];
                dkeys[key][channel] += input.scale * dscore * cache.queries[time][channel];
            }
        }
        detail::reverse_projection(parameters, 0, input.queries, time, dquery, result.parameters,
                                   result.queries);
    }
    for (std::size_t key = 0; key != kKeySteps; ++key) {
        detail::reverse_projection(parameters, 1, input.keys, key, dkeys[key], result.parameters,
                                   result.keys);
        detail::reverse_projection(parameters, 2, input.values, key, dvalues[key],
                                   result.parameters, result.values);
    }
    return result;
}
inline Parameters seeded_parameters() {
    Parameters result;
    result.matrices = {{{.375, -.25, .125, .5},
                        {-.25, .375, .5, .125},
                        {.5, .25, -.375, .625},
                        {.75, -.125, .25, .5}}};
    result.biases = {{{.125, -.25}, {-.125, .25}, {.0625, -.125}, {.125, .0625}}};
    return result;
}
inline Inputs seeded_inputs() {
    Inputs result;
    result.queries = {.5, -.25, -.375, .625};
    result.keys = {.25, -.5, .625, .125, -.375, .75};
    result.values = {-.5, .25, .375, -.75, .625, .5};
    result.residual = {.25, -.125, -.375, .5};
    return result;
}
inline Upstream seeded_upstream() {
    return {{.375, -.25, .5, .625}};
}
inline double max_finite_difference_error(Parameters parameters, Inputs input,
                                          const Upstream& upstream) {
    const auto analytic = backward(parameters, input, upstream);
    constexpr double epsilon = 1e-6;
    double error{};
    const auto check = [&](double& value, double expected) {
        const auto saved = value;
        value = saved + epsilon;
        const auto plus = loss(forward(parameters, input), upstream);
        value = saved - epsilon;
        const auto minus = loss(forward(parameters, input), upstream);
        value = saved;
        const auto observed = (plus - minus) / (2 * epsilon);
        if (!std::isfinite(plus) || !std::isfinite(minus) || !std::isfinite(observed) ||
            !std::isfinite(expected))
            throw std::runtime_error("non-finite attention central-difference validation");
        error = std::max(error, std::abs(observed - expected));
    };
    for (std::size_t kind = 0; kind != 4; ++kind) {
        for (std::size_t i = 0; i != 4; ++i)
            check(parameters.matrices[kind][i], analytic.parameters.matrices[kind][i]);
        for (std::size_t i = 0; i != 2; ++i)
            check(parameters.biases[kind][i], analytic.parameters.biases[kind][i]);
    }
    for (std::size_t i = 0; i != input.queries.size(); ++i) {
        check(input.queries[i], analytic.queries[i]);
        check(input.residual[i], analytic.residual[i]);
    }
    for (std::size_t i = 0; i != input.keys.size(); ++i) {
        check(input.keys[i], analytic.keys[i]);
        check(input.values[i], analytic.values[i]);
    }
    return error;
}
}  // namespace glimmer::tests::attention_oracle
