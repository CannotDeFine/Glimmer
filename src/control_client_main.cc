#include "glimmer/control/task_protocol.h"
#include "glimmer/control/unix_socket_client.h"

#include <charconv>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>

namespace {

void print_usage(std::ostream& output, std::string_view program) {
    output << "Usage:\n"
           << "  " << program
           << " --socket PATH submit TENANT MEMORY_BYTES WEIGHT WORK_UNITS [PRIORITY]\n"
           << "  " << program << " --socket PATH query TASK_ID\n"
           << "  " << program << " --socket PATH cancel TASK_ID\n"
           << "  " << program << " --socket PATH claim\n"
           << "  " << program << " --socket PATH heartbeat TASK_ID\n"
           << "  " << program << " --socket PATH complete TASK_ID\n"
           << "  " << program << " --socket PATH fail TASK_ID\n"
           << "  " << program << " --socket PATH stats\n"
           << "  " << program << " --socket PATH metrics\n";
}

template <typename Integer>
bool parse_positive(std::string_view text, Integer* value) {
    if (value == nullptr || text.empty()) {
        return false;
    }
    Integer parsed = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), parsed);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size() || parsed == 0) {
        return false;
    }
    *value = parsed;
    return true;
}

template <typename Integer>
bool parse_unsigned(std::string_view text, Integer* value) {
    if (value == nullptr || text.empty()) {
        return false;
    }
    Integer parsed = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), parsed);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
        return false;
    }
    *value = parsed;
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        print_usage(std::cout, argv[0]);
        return EXIT_SUCCESS;
    }
    if (argc < 4 || std::string_view(argv[1]) != "--socket") {
        print_usage(std::cerr, argv[0]);
        return EXIT_FAILURE;
    }
    const std::string_view socket_path = argv[2];
    const std::string_view operation = argv[3];
    glimmer::control::TaskProtocolRequest request;
    if (operation == "submit") {
        if (argc != 8 && argc != 9) {
            print_usage(std::cerr, argv[0]);
            return EXIT_FAILURE;
        }
        request.operation = glimmer::control::TaskProtocolOperation::kSubmit;
        request.admission.tenant_id = argv[4];
        if (!parse_positive(argv[5], &request.admission.memory_bytes) ||
            !parse_positive(argv[6], &request.admission.weight) ||
            !parse_positive(argv[7], &request.admission.work_units)) {
            std::cerr << "invalid submit values\n";
            return EXIT_FAILURE;
        }
        if (argc == 9 && !parse_unsigned(argv[8], &request.admission.priority)) {
            std::cerr << "invalid submit priority\n";
            return EXIT_FAILURE;
        }
    } else if (operation == "claim") {
        if (argc != 4) {
            print_usage(std::cerr, argv[0]);
            return EXIT_FAILURE;
        }
        request.operation = glimmer::control::TaskProtocolOperation::kClaim;
    } else if (operation == "stats" || operation == "metrics") {
        if (argc != 4) {
            print_usage(std::cerr, argv[0]);
            return EXIT_FAILURE;
        }
        request.operation = operation == "stats"
                                ? glimmer::control::TaskProtocolOperation::kStats
                                : glimmer::control::TaskProtocolOperation::kMetrics;
    } else {
        if (argc != 5 || !parse_positive(argv[4], &request.task_id)) {
            print_usage(std::cerr, argv[0]);
            return EXIT_FAILURE;
        }
        if (operation == "query") {
            request.operation = glimmer::control::TaskProtocolOperation::kQuery;
        } else if (operation == "cancel") {
            request.operation = glimmer::control::TaskProtocolOperation::kCancel;
        } else if (operation == "heartbeat") {
            request.operation = glimmer::control::TaskProtocolOperation::kHeartbeat;
        } else if (operation == "complete") {
            request.operation = glimmer::control::TaskProtocolOperation::kComplete;
        } else if (operation == "fail") {
            request.operation = glimmer::control::TaskProtocolOperation::kFail;
        } else {
            print_usage(std::cerr, argv[0]);
            return EXIT_FAILURE;
        }
    }

    const auto encoded_request = glimmer::control::format_task_protocol_request(request);
    if (!encoded_request.has_value()) {
        std::cerr << "request is invalid or exceeds the protocol limit\n";
        return EXIT_FAILURE;
    }
    const glimmer::control::UnixSocketControlClient client{std::string(socket_path)};
    const auto response = client.request(encoded_request.value());
    if (!response.has_value()) {
        std::cerr << "failed to communicate with control service\n";
        return EXIT_FAILURE;
    }
    std::cout << response.value();
    const auto parsed = glimmer::control::parse_task_protocol_response(response.value());
    return parsed.parsed() && parsed.response.has_value() &&
                   parsed.response->kind != glimmer::control::TaskProtocolResponseKind::kError
               ? EXIT_SUCCESS
               : EXIT_FAILURE;
}
