#include "rnn_lstm_oracle.h"

#include <iostream>
#include <limits>

namespace oracle = glimmer::tests::rnn_lstm_oracle;
int main() {
    try {
        const auto parameters = oracle::seeded_parameters();
        const auto upstream = oracle::seeded_upstream();
        for (const auto lengths :
             {std::array{3, 3}, std::array{3, 2}, std::array{0, 3}, std::array{0, 0}}) {
            auto input = oracle::seeded_inputs();
            input.lengths = lengths;
            const auto error = oracle::max_finite_difference_error(parameters, input, upstream);
            if (!std::isfinite(error) || error > 1e-8)
                throw std::runtime_error("complete central-difference gradient cross-check failed");
            const auto output = oracle::forward(parameters, input);
            const auto gradient = oracle::backward(parameters, input, upstream);
            for (std::size_t batch = 0; batch != 2; ++batch)
                for (int step = lengths[batch]; step != 3; ++step)
                    for (std::size_t row = 0; row != 2; ++row) {
                        const auto index = (static_cast<std::size_t>(step) * 2 + batch) * 2 + row;
                        if (output.y[index] != 0.0 || gradient.dx[index] != 0.0)
                            throw std::runtime_error("padded logical values must be zero");
                    }
            if (lengths == std::array{3, 3})
                for (std::size_t gate = 0; gate != 8; ++gate) {
                    for (const auto value : gradient.parameters.matrices[gate])
                        if (!std::isfinite(value) || std::abs(value) < 1e-10)
                            throw std::runtime_error("matrix-gradient oracle is not nontrivial");
                    for (const auto value : gradient.parameters.biases[gate])
                        if (!std::isfinite(value) || std::abs(value) < 1e-10)
                            throw std::runtime_error("bias-gradient oracle is not nontrivial");
                }
        }
        // An independent closed-form single-cell control catches gate-ID swaps
        // even if forward and reverse implementations shared the same mistake.
        oracle::Parameters zero;
        auto input = oracle::seeded_inputs();
        input.lengths = {1, 1};
        zero.biases[2].fill(.25);  // Candidate, not output, is linear ID 2.
        const auto output = oracle::forward(zero, input);
        for (std::size_t i = 0; i != 4; ++i) {
            const auto cell = .5 * input.cx[i] + .5 * std::tanh(.25);
            if (std::abs(output.cy[i] - cell) > 1e-14 ||
                std::abs(output.hy[i] - .5 * std::tanh(cell)) > 1e-14)
                throw std::runtime_error("independent native gate-ID control failed");
        }
        for (const auto invalid :
             {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
            auto bad = upstream;
            bad.dy[0] = invalid;
            bool rejected{};
            try {
                (void)oracle::max_finite_difference_error(parameters, oracle::seeded_inputs(), bad);
            } catch (const std::runtime_error&) {
                rejected = true;
            }
            if (!rejected) throw std::runtime_error("non-finite validation must fail closed");
        }
        std::cout << "Independent 48-parameter LSTM oracle checks passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
