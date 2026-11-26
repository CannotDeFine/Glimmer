#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>

// Independent logical LSTM model. Native parameter offsets, storage padding,
// tensor strides and GPU execution are deliberately not represented here.
namespace glimmer::tests::rnn_lstm_oracle {
inline constexpr std::size_t kInput = 2, kHidden = 2, kSteps = 3, kBatch = 2;
struct Parameters {
    // Native linear IDs: 0/4 input, 1/5 forget, 2/6 candidate, 3/7 output.
    // Rows are hidden/output coordinates; columns are input/previous hidden.
    std::array<std::array<double, 4>, 8> matrices{};
    std::array<std::array<double, 2>, 8> biases{};
};
struct Inputs {
    std::array<double, 12> x{};
    std::array<double, 4> hx{}, cx{};
    std::array<int, 2> lengths{3, 2};
};
struct Upstream {
    std::array<double, 12> dy{};
    std::array<double, 4> dhy{}, dcy{};
};
struct Output {
    std::array<double, 12> y{};
    std::array<double, 4> hy{}, cy{};
};
struct Gradient {
    Parameters parameters;
    std::array<double, 12> dx{};
    std::array<double, 4> dhx{}, dcx{};
};
namespace detail {
struct Cell {
    std::array<double, 2> previous_hidden{}, previous_cell{}, cell{};
    std::array<std::array<double, 2>, 4> gates{};
};
struct Forward {
    Output output;
    std::array<Cell, kSteps * kBatch> cells{};
};
inline void validate(const Inputs& input) {
    for (const auto length : input.lengths)
        if (length < 0 || length > static_cast<int>(kSteps))
            throw std::invalid_argument("LSTM oracle sequence length is out of bounds");
}
inline Forward evaluate(const Parameters& parameters, const Inputs& input) {
    validate(input);
    Forward result;
    for (std::size_t batch = 0; batch != kBatch; ++batch) {
        std::array<double, 2> hidden{input.hx[2 * batch], input.hx[2 * batch + 1]};
        std::array<double, 2> cell{input.cx[2 * batch], input.cx[2 * batch + 1]};
        for (int step = 0; step != input.lengths[batch]; ++step) {
            const auto index = static_cast<std::size_t>(step) * kBatch + batch;
            auto& cache = result.cells[index];
            cache.previous_hidden = hidden;
            cache.previous_cell = cell;
            for (std::size_t gate = 0; gate != 4; ++gate)
                for (std::size_t row = 0; row != kHidden; ++row) {
                    double z = parameters.biases[gate][row] + parameters.biases[gate + 4][row];
                    for (std::size_t col = 0; col != 2; ++col)
                        z += parameters.matrices[gate][2 * row + col] * input.x[2 * index + col] +
                             parameters.matrices[gate + 4][2 * row + col] * hidden[col];
                    cache.gates[gate][row] = gate == 2 ? std::tanh(z) : 1.0 / (1.0 + std::exp(-z));
                }
            for (std::size_t row = 0; row != kHidden; ++row) {
                cell[row] =
                    cache.gates[1][row] * cell[row] + cache.gates[0][row] * cache.gates[2][row];
                hidden[row] = cache.gates[3][row] * std::tanh(cell[row]);
                cache.cell[row] = cell[row];
                result.output.y[2 * index + row] = hidden[row];
            }
        }
        for (std::size_t row = 0; row != 2; ++row) {
            result.output.hy[2 * batch + row] = hidden[row];
            result.output.cy[2 * batch + row] = cell[row];
        }
    }
    return result;
}
}  // namespace detail
inline Output forward(const Parameters& parameters, const Inputs& input) {
    return detail::evaluate(parameters, input).output;
}
inline double loss(const Output& value, const Upstream& upstream) {
    double result{};
    for (std::size_t i = 0; i != value.y.size(); ++i) result += value.y[i] * upstream.dy[i];
    for (std::size_t i = 0; i != value.hy.size(); ++i)
        result += value.hy[i] * upstream.dhy[i] + value.cy[i] * upstream.dcy[i];
    return result;
}
inline Gradient backward(const Parameters& parameters, const Inputs& input,
                         const Upstream& upstream) {
    const auto cached = detail::evaluate(parameters, input);
    Gradient result;
    for (std::size_t batch = 0; batch != kBatch; ++batch) {
        std::array<double, 2> dh{upstream.dhy[2 * batch], upstream.dhy[2 * batch + 1]};
        std::array<double, 2> dc{upstream.dcy[2 * batch], upstream.dcy[2 * batch + 1]};
        for (int step = input.lengths[batch] - 1; step >= 0; --step) {
            const auto index = static_cast<std::size_t>(step) * kBatch + batch;
            const auto& cache = cached.cells[index];
            std::array<std::array<double, 2>, 4> dz{};
            for (std::size_t row = 0; row != kHidden; ++row) {
                dh[row] += upstream.dy[2 * index + row];
                const auto tanh_cell = std::tanh(cache.cell[row]);
                dc[row] += dh[row] * cache.gates[3][row] * (1 - tanh_cell * tanh_cell);
                const auto input_gate = cache.gates[0][row], forget_gate = cache.gates[1][row],
                           candidate = cache.gates[2][row], output_gate = cache.gates[3][row];
                dz[0][row] = dc[row] * candidate * input_gate * (1 - input_gate);
                dz[1][row] = dc[row] * cache.previous_cell[row] * forget_gate * (1 - forget_gate);
                dz[2][row] = dc[row] * input_gate * (1 - candidate * candidate);
                dz[3][row] = dh[row] * tanh_cell * output_gate * (1 - output_gate);
                dc[row] *= forget_gate;
            }
            dh = {};
            for (std::size_t gate = 0; gate != 4; ++gate)
                for (std::size_t row = 0; row != kHidden; ++row) {
                    const auto value = dz[gate][row];
                    result.parameters.biases[gate][row] += value;
                    result.parameters.biases[gate + 4][row] += value;
                    for (std::size_t col = 0; col != 2; ++col) {
                        result.parameters.matrices[gate][2 * row + col] +=
                            value * input.x[2 * index + col];
                        result.parameters.matrices[gate + 4][2 * row + col] +=
                            value * cache.previous_hidden[col];
                        result.dx[2 * index + col] +=
                            value * parameters.matrices[gate][2 * row + col];
                        dh[col] += value * parameters.matrices[gate + 4][2 * row + col];
                    }
                }
        }
        for (std::size_t row = 0; row != 2; ++row) {
            result.dhx[2 * batch + row] = dh[row];
            result.dcx[2 * batch + row] = dc[row];
        }
    }
    return result;
}
inline Parameters seeded_parameters() {
    Parameters result;
    for (std::size_t gate = 0; gate != 8; ++gate) {
        for (std::size_t i = 0; i != 4; ++i)
            result.matrices[gate][i] =
                static_cast<double>((gate * 5 + i * 3) % 17 + 1) / (gate % 2 ? -128.0 : 128.0);
        for (std::size_t i = 0; i != 2; ++i)
            result.biases[gate][i] =
                static_cast<double>(gate * 2 + i + 1) / (gate % 3 ? 256.0 : -256.0);
    }
    return result;
}
inline Inputs seeded_inputs() {
    Inputs result;
    for (std::size_t i = 0; i != result.x.size(); ++i)
        result.x[i] = static_cast<double>(i + 1) / (i % 2 ? -32.0 : 32.0);
    for (std::size_t i = 0; i != result.hx.size(); ++i) {
        result.hx[i] = static_cast<double>(i + 1) / (i % 2 ? -16.0 : 16.0);
        result.cx[i] = static_cast<double>(i + 2) / (i % 2 ? 32.0 : -32.0);
    }
    return result;
}
inline Upstream seeded_upstream() {
    Upstream result;
    for (std::size_t i = 0; i != result.dy.size(); ++i)
        result.dy[i] = static_cast<double>(i + 3) / (i % 3 ? 64.0 : -64.0);
    for (std::size_t i = 0; i != result.dhy.size(); ++i) {
        result.dhy[i] = static_cast<double>(i + 2) / (i % 2 ? 32.0 : -32.0);
        result.dcy[i] = static_cast<double>(i + 4) / (i % 2 ? -32.0 : 32.0);
    }
    return result;
}
inline double max_finite_difference_error(Parameters parameters, Inputs input,
                                          const Upstream& upstream) {
    const auto analytic = backward(parameters, input, upstream);
    constexpr double epsilon = 1e-6;
    double error{};
    const auto check = [&](double& value, double expected) {
        const auto old = value;
        value = old + epsilon;
        const auto plus = loss(forward(parameters, input), upstream);
        value = old - epsilon;
        const auto minus = loss(forward(parameters, input), upstream);
        value = old;
        const auto observed = (plus - minus) / (2 * epsilon);
        if (!std::isfinite(plus) || !std::isfinite(minus) || !std::isfinite(observed) ||
            !std::isfinite(expected))
            throw std::runtime_error("non-finite central-difference validation");
        error = std::max(error, std::abs(observed - expected));
    };
    for (std::size_t gate = 0; gate != 8; ++gate) {
        for (std::size_t i = 0; i != 4; ++i)
            check(parameters.matrices[gate][i], analytic.parameters.matrices[gate][i]);
        for (std::size_t i = 0; i != 2; ++i)
            check(parameters.biases[gate][i], analytic.parameters.biases[gate][i]);
    }
    for (std::size_t i = 0; i != input.x.size(); ++i) check(input.x[i], analytic.dx[i]);
    for (std::size_t i = 0; i != input.hx.size(); ++i) {
        check(input.hx[i], analytic.dhx[i]);
        check(input.cx[i], analytic.dcx[i]);
    }
    return error;
}
}  // namespace glimmer::tests::rnn_lstm_oracle
