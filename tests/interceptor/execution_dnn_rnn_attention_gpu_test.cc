#include <cudnn.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

// Ordinary installed-SDK metadata input, not an injected fixture or a GPU CTest.
// This observes original status/lifecycle only: the two guarded getters are not
// invoked, so neither complete internal state nor computation is qualified.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
namespace {
bool has_failure{};
unsigned int calls{};
void expect(bool condition, const char* message) {
    if (!condition) {
        has_failure = true;
        throw std::runtime_error(message);
    }
}
void checked(cudnnStatus_t status, std::string_view operation) {
    ++calls;
    std::cout << "rnn_attention_case=" << calls << "; operation=" << operation
              << "; status=" << static_cast<int>(status) << '\n';
    if (status != CUDNN_STATUS_SUCCESS) {
        has_failure = true;
        throw std::runtime_error(std::string(operation) + ": " + std::to_string(status));
    }
}
template <typename Handle, auto Create, auto Destroy>
class Descriptor {
   public:
    Descriptor() = default;
    Descriptor(const Descriptor&) = delete;
    Descriptor& operator=(const Descriptor&) = delete;
    ~Descriptor() {
        // Unexpected mutation failure has no general native rollback proof.
        // Exit retains that uncertain process-owned metadata; never retry it.
        if (value_ && !has_failure && Destroy(std::exchange(value_, {})) != CUDNN_STATUS_SUCCESS)
            has_failure = true;
    }
    void create(std::string_view operation) {
        checked(Create(&value_), operation);
        expect(value_ != nullptr, "successful creation publishes a nonnull original identity");
    }
    void close(std::string_view operation) {
        expect(value_ != nullptr, "metadata destruction occurs exactly once");
        checked(Destroy(std::exchange(value_, {})), operation);
    }
    Handle get() const {
        return value_;
    }

   private:
    Handle value_{};
};
using Dropout = Descriptor<cudnnDropoutDescriptor_t, cudnnCreateDropoutDescriptor,
                           cudnnDestroyDropoutDescriptor>;
using Rnn = Descriptor<cudnnRNNDescriptor_t, cudnnCreateRNNDescriptor, cudnnDestroyRNNDescriptor>;
using Attention =
    Descriptor<cudnnAttnDescriptor_t, cudnnCreateAttnDescriptor, cudnnDestroyAttnDescriptor>;

void set_rnn(cudnnRNNDescriptor_t descriptor, cudnnDropoutDescriptor_t dropout,
             std::string_view operation) {
    checked(cudnnSetRNNDescriptor_v8(descriptor, CUDNN_RNN_ALGO_STANDARD, CUDNN_LSTM,
                                     CUDNN_RNN_DOUBLE_BIAS, CUDNN_UNIDIRECTIONAL,
                                     CUDNN_LINEAR_INPUT, CUDNN_DATA_FLOAT, CUDNN_DATA_FLOAT,
                                     CUDNN_DEFAULT_MATH, 4, 4, 4, 1, dropout, 0),
            operation);
}
void set_attention(cudnnAttnDescriptor_t descriptor, cudnnDropoutDescriptor_t first,
                   cudnnDropoutDescriptor_t second, std::string_view operation) {
    checked(
        cudnnSetAttnDescriptor(descriptor, 0, 1, 1.0, CUDNN_DATA_FLOAT, CUDNN_DATA_FLOAT,
                               CUDNN_DEFAULT_MATH, first, second, 4, 4, 4, 0, 0, 0, 0, 2, 2, 1, 1),
        operation);
}
void exercise(bool parents_first) {
    Rnn rnn;
    Attention attention;
    Dropout first, second;
    if (parents_first) {
        rnn.create("create-rnn-before-dropouts");
        attention.create("create-attention-before-dropouts");
    }
    first.create("create-first-dropout");
    second.create("create-second-dropout");
    expect(first.get() != second.get(), "simultaneously live dropouts remain distinct");
    if (!parents_first) {
        rnn.create("create-rnn-after-dropouts");
        attention.create("create-attention-after-dropouts");
    }
    // Fresh native dropout descriptors require no context or RNG initialization
    // for these shallow-pointer metadata setters.
    set_rnn(rnn.get(), nullptr, "rnn-null");
    set_attention(attention.get(), nullptr, nullptr, "attention-null-null");
    set_rnn(rnn.get(), first.get(), "rnn-first");
    set_attention(attention.get(), first.get(), first.get(), "attention-first-alias");
    set_rnn(rnn.get(), second.get(), "rnn-rebind-second");
    set_attention(attention.get(), first.get(), second.get(), "attention-distinct");
    set_rnn(rnn.get(), nullptr, "rnn-detach");
    set_attention(attention.get(), nullptr, second.get(), "attention-detach-first");
    first.close("destroy-detached-first-dropout");
    set_rnn(rnn.get(), second.get(), "rnn-rebind-live-second");
    set_attention(attention.get(), second.get(), second.get(), "attention-second-alias");
    // Do not compare destruction of a still-referenced child: that is an
    // explicit owned-policy rejection, not asserted native error equivalence.
    attention.close("destroy-attention-before-second-dropout");
    rnn.close("destroy-rnn-before-second-dropout");
    second.close("destroy-second-dropout-after-parents");
}
}  // namespace
int main(int argc, char** argv) {
    try {
        expect(argc == 1 || (argc == 2 && std::string_view(argv[1]) == "--metadata-only"),
               "usage: RNN/attention input [--metadata-only]");
        exercise(false);
        exercise(true);
        expect(calls == 36 && !has_failure,
               "complete status/lifecycle matrix with checked cleanup");
        std::cout << "cuDNN RNN/attention metadata input passed: api_calls=36; cases=36; "
                     "creation_orders=2; guarded_getter_calls=0; explicit_driver_calls=0; "
                     "explicit_device_calls=0; compute_calls=0; full_state_qualified=false; "
                     "full_device_qualified=false; priority_qualified=false\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
#pragma GCC diagnostic pop
