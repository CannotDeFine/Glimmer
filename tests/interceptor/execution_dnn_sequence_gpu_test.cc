#include <cudnn.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// Ordinary installed-SDK metadata input. No fixture, Driver/context/device call,
// computation or automatic GPU CTest. Actual native/owned execution is separate.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
namespace {
bool cleanup_failed{};
std::size_t calls{};
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void checked(cudnnStatus_t status, const char* operation) {
    ++calls;
    if (status != CUDNN_STATUS_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": " + std::to_string(status));
}
template <typename Handle, auto Destroy>
class Descriptor {
   public:
    Descriptor() = default;
    Descriptor(const Descriptor&) = delete;
    Descriptor& operator=(const Descriptor&) = delete;
    ~Descriptor() {
        if (value_ && Destroy(std::exchange(value_, {})) != CUDNN_STATUS_SUCCESS)
            cleanup_failed = true;
    }
    Handle* output() {
        return &value_;
    }
    Handle get() const {
        return value_;
    }
    void close() {
        checked(Destroy(std::exchange(value_, {})), "destroy metadata");
    }

   private:
    Handle value_{};
};
using Rnn = Descriptor<cudnnRNNDataDescriptor_t, cudnnDestroyRNNDataDescriptor>;
using Seq = Descriptor<cudnnSeqDataDescriptor_t, cudnnDestroySeqDataDescriptor>;
struct Expected {
    cudnnDataType_t type{CUDNN_DATA_FLOAT};
    std::array<int, 4> sizes{};
    std::array<cudnnSeqDataAxis_t, 4> axes{};
    std::vector<int> lengths;
    std::size_t padding_bytes{4};
    alignas(double) std::array<std::byte, 8> padding{};
};
struct Outputs {
    std::array<cudnnDataType_t, 3> type{CUDNN_DATA_INT32, CUDNN_DATA_INT32, CUDNN_DATA_INT32};
    std::array<int, 3> rank{-71, -72, -73};
    std::array<std::size_t, 3> count{71, 72, 73};
    std::array<int, 8> sizes{}, lengths{};
    std::array<cudnnSeqDataAxis_t, 8> axes{};
    alignas(double) std::array<std::byte, 24> padding{};
    Outputs() {
        sizes.fill(-101);
        lengths.fill(-202);
        axes.fill(CUDNN_SEQDATA_VECT_DIM);
        padding.fill(std::byte{0xad});
    }
    bool operator==(const Outputs&) const = default;
};
void observe(cudnnSeqDataDescriptor_t descriptor, const Expected& expected, unsigned int mask,
             int requested, std::size_t capacity) {
    Outputs actual;
    auto wanted = actual;
    const bool native_padding_error = expected.type == CUDNN_DATA_BFLOAT16 && (mask & 64U);
    // This particular native error can write earlier non-padding outputs. The
    // differential error case requests padding only, not an invented rollback.
    expect(!native_padding_error || mask == 64, "native error observation excludes prior outputs");
    if (!native_padding_error) {
        if (mask & 1) wanted.type[1] = expected.type;
        if (mask & 2) wanted.rank[1] = 4;
        if (mask & 16) wanted.count[1] = expected.lengths.size();
        const auto dimensions = static_cast<std::size_t>(std::min(requested, 4));
        if (mask & 4) std::copy_n(expected.sizes.begin(), dimensions, wanted.sizes.begin() + 1);
        if (mask & 8) std::copy_n(expected.axes.begin(), dimensions, wanted.axes.begin() + 1);
        if (mask & 32)
            std::copy_n(expected.lengths.begin(), std::min(capacity, expected.lengths.size()),
                        wanted.lengths.begin() + 1);
        if (mask & 64)
            std::copy_n(expected.padding.begin(), expected.padding_bytes,
                        wanted.padding.begin() + 8);
    }
    const auto status = cudnnGetSeqDataDescriptor(
        descriptor, (mask & 1) ? &actual.type[1] : nullptr, (mask & 2) ? &actual.rank[1] : nullptr,
        requested, (mask & 4) ? actual.sizes.data() + 1 : nullptr,
        (mask & 8) ? actual.axes.data() + 1 : nullptr, (mask & 16) ? &actual.count[1] : nullptr,
        capacity, (mask & 32) ? actual.lengths.data() + 1 : nullptr,
        (mask & 64) ? actual.padding.data() + 8 : nullptr);
    ++calls;
    expect(status == (native_padding_error ? CUDNN_STATUS_INTERNAL_ERROR : CUDNN_STATUS_SUCCESS),
           "original sequence getter status");
    expect(actual == wanted, "complete scalar, prefix, padding and unrequested guards");
    std::cout << "sequence_case=" << calls << "; type=" << expected.type << "; mask=" << mask
              << "; requested=" << requested << "; capacity=" << capacity << "; status=" << status
              << "; rank=" << actual.rank[1] << "; count=" << actual.count[1];
    const auto print = [](std::string_view name, const auto& values) {
        std::cout << "; " << name << '=';
        for (std::size_t i = 0; i != values.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << static_cast<int>(values[i]);
        }
    };
    print("types", actual.type);
    print("sizes", actual.sizes);
    print("axes", actual.axes);
    print("lengths", actual.lengths);
    print("padding", actual.padding);
    std::cout << '\n';
}
Expected configuration(cudnnDataType_t type) {
    Expected expected;
    expected.type = type;
    expected.sizes = {3, 2, 1, 4};
    expected.axes = {CUDNN_SEQDATA_TIME_DIM, CUDNN_SEQDATA_BATCH_DIM, CUDNN_SEQDATA_BEAM_DIM,
                     CUDNN_SEQDATA_VECT_DIM};
    expected.lengths = {3, 2};
    // Exactly representable 1.5 in all four admitted padding formats.
    std::uint64_t bits{};
    if (type == CUDNN_DATA_DOUBLE) {
        expected.padding_bytes = 8;
        bits = std::bit_cast<std::uint64_t>(1.5);
    } else if (type == CUDNN_DATA_FLOAT) {
        expected.padding_bytes = 4;
        bits = std::bit_cast<std::uint32_t>(1.5F);
    } else {
        expected.padding_bytes = 2;
        bits = type == CUDNN_DATA_HALF ? 0x3e00 : 0x3fc0;
    }
    for (std::size_t i = 0; i != expected.padding_bytes; ++i)
        expected.padding[i] = static_cast<std::byte>(bits >> (i * 8));
    return expected;
}
void run() {
    {
        Seq fresh;
        checked(cudnnCreateSeqDataDescriptor(fresh.output()), "create fresh sequence");
        observe(fresh.get(), {}, 127, 4, std::numeric_limits<std::size_t>::max());
        fresh.close();
    }
    for (const auto type : {CUDNN_DATA_FLOAT, CUDNN_DATA_DOUBLE, CUDNN_DATA_HALF}) {
        auto value = configuration(type);
        for (const auto layout :
             {CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED, CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED,
              CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED}) {
            Rnn descriptor;
            checked(cudnnCreateRNNDataDescriptor(descriptor.output()), "create RNN data");
            checked(cudnnSetRNNDataDescriptor(descriptor.get(), type, layout, 3, 2, 4,
                                              value.lengths.data(), value.padding.data()),
                    "set RNN data");
            descriptor.close();
        }
    }
    for (const auto type :
         {CUDNN_DATA_FLOAT, CUDNN_DATA_DOUBLE, CUDNN_DATA_HALF, CUDNN_DATA_BFLOAT16}) {
        for (const bool alternate_axes : {false, true}) {
            auto value = configuration(type);
            // Original 9.10.2 requires the vector axis last; permute only the
            // first three axes when selecting another successful layout.
            if (alternate_axes) std::swap(value.axes[0], value.axes[1]);
            Seq descriptor;
            checked(cudnnCreateSeqDataDescriptor(descriptor.output()), "create sequence");
            checked(cudnnSetSeqDataDescriptor(descriptor.get(), type, 4, value.sizes.data(),
                                              value.axes.data(), value.lengths.size(),
                                              value.lengths.data(), value.padding.data()),
                    "set sequence");
            const unsigned int mask = type == CUDNN_DATA_BFLOAT16 ? 63U : 127U;
            observe(descriptor.get(), value, mask, 0, 0);
            observe(descriptor.get(), value, mask, 2, 1);
            observe(descriptor.get(), value, mask, std::numeric_limits<int>::max(),
                    std::numeric_limits<std::size_t>::max());
            if (type == CUDNN_DATA_FLOAT && !alternate_axes)
                for (unsigned int outputs = 0; outputs != 128; ++outputs)
                    observe(descriptor.get(), value, outputs, 4, 2);
            if (type == CUDNN_DATA_BFLOAT16) observe(descriptor.get(), value, 64, 4, 2);
            descriptor.close();
        }
    }
    expect(calls == 208 && !cleanup_failed, "complete metadata matrix and checked destruction");
}
}  // namespace
int main(int argc, char** argv) {
    try {
        expect(argc == 1 || (argc == 2 && std::string_view(argv[1]) == "--metadata-only"),
               "usage: sequence input [--metadata-only]");
        run();
        std::cout << "cuDNN sequence metadata input passed: api_calls=" << calls
                  << "; sequence_cases=155; output_masks=128; explicit_driver_calls=0; "
                     "explicit_device_calls=0; "
                     "compute_calls=0; full_device_qualified=false; priority_qualified=false\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
#pragma GCC diagnostic pop
