#include "internal/owned_control_exchange.h"

#include <array>
#include <barrier>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <iostream>
#include <poll.h>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

namespace {

using namespace glimmer::control;
using namespace glimmer::control::internal;
using Clock = OwnedControlExchange::Clock;
using namespace std::chrono_literals;

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

class Descriptor final {
   public:
    explicit Descriptor(int descriptor) : descriptor_(descriptor) {
        require(descriptor >= 0, "owned descriptor must create");
    }
    ~Descriptor() {
        close();
    }
    Descriptor(const Descriptor&) = delete;
    Descriptor& operator=(const Descriptor&) = delete;
    int get() const {
        return descriptor_;
    }
    void close() {
        if (descriptor_ < 0) return;
        require(::close(descriptor_) == 0, "owned descriptor must close");
        descriptor_ = -1;
    }

   private:
    int descriptor_;
};

class Directory final {
   public:
    Directory() {
        std::array<char, 48> pattern{};
        constexpr std::string_view prefix = "/tmp/glimmer-owned-exchange-XXXXXX";
        std::memcpy(pattern.data(), prefix.data(), prefix.size());
        require(::mkdtemp(pattern.data()) != nullptr, "private directory must create");
        directory_ = pattern.data();
        path_ = directory_ + "/control.sock";
    }
    ~Directory() {
        remove();
    }
    Directory(const Directory&) = delete;
    Directory& operator=(const Directory&) = delete;
    const std::string& path() const {
        return path_;
    }
    void remove() {
        if (!has_directory_) return;
        require(::rmdir(directory_.c_str()) == 0, "only the owned empty directory must release");
        has_directory_ = false;
    }

   private:
    std::string directory_;
    std::string path_;
    bool has_directory_ = true;
};

class Listener final {
   public:
    Listener() : descriptor_(::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0)) {
        sockaddr_un address{};
        address.sun_family = AF_UNIX;
        require(directory_.path().size() < sizeof(address.sun_path), "socket path must fit");
        std::memcpy(address.sun_path, directory_.path().c_str(), directory_.path().size() + 1);
        const int result =
            ::bind(descriptor_.get(), reinterpret_cast<const sockaddr*>(&address), sizeof(address));
        if (result != 0 && (errno == EPERM || errno == EACCES || errno == ENOSYS)) {
            const int bind_error = errno;
            directory_.remove();
            std::cerr << "Unix socket bind unavailable: errno=" << bind_error << '\n';
            std::exit(77);
        }
        require(result == 0 && ::listen(descriptor_.get(), 4) == 0, "owned listener must bind");
    }
    ~Listener() {
        require(::unlink(directory_.path().c_str()) == 0,
                "only the owned socket inode must remove");
    }
    Listener(const Listener&) = delete;
    Listener& operator=(const Listener&) = delete;
    const std::string& path() const {
        return directory_.path();
    }
    int accept() const {
        return ::accept4(descriptor_.get(), nullptr, nullptr, SOCK_CLOEXEC);
    }

   private:
    Directory directory_;
    Descriptor descriptor_;
};

void send_bytes(int descriptor, std::string_view bytes) {
    require(::send(descriptor, bytes.data(), bytes.size(), MSG_DONTWAIT | MSG_NOSIGNAL) ==
                static_cast<ssize_t>(bytes.size()),
            "scripted peer must send exact bounded bytes");
}

void read_open(int descriptor) {
    std::string line;
    for (std::size_t index = 0; index < kOwnerControlOpen.size(); ++index) {
        char byte = 0;
        require(::recv(descriptor, &byte, 1, 0) == 1, "cold peer must receive OPEN");
        line.push_back(byte);
    }
    require(line == kOwnerControlOpen, "only explicit V3 negotiation is permitted");
}

class Session final {
   public:
    explicit Session(OwnedControlExchange& exchange) {
        auto status = OwnedExchangeColdStatus::kRejected;
        std::thread initialize([&] { status = exchange.initialize({listener.path(), 1000ms}); });
        peer_ = std::make_unique<Descriptor>(listener.accept());
        read_open(peer_->get());
        send_bytes(peer_->get(), kOwnerControlReady);
        require(server.create_server({17, 29}, SharedControlMode::kReplyHintV3),
                "sealed scripted offer must create");
        require(server.send_offer(peer_->get(), 1000ms), "scripted offer must send");
        initialize.join();
        require(status == OwnedExchangeColdStatus::kReady, "cold owned endpoint must initialize");
    }
    void receive() {
        std::string line;
        require(
            server.receive_request(peer_->get(), 1000, line) == SharedControlReadStatus::kLine &&
                line == "request\n",
            "actual request must arrive once");
    }
    void reply() {
        require(server.publish_response_with_hint(peer_->get(), "response\n"),
                "actual response must publish before hint");
    }
    int peer() const {
        return peer_->get();
    }
    void close_peer() {
        peer_->close();
    }

    Listener listener;
    SharedControlChannel server;

   private:
    std::unique_ptr<Descriptor> peer_;
};

OwnedExchangePublication begin(OwnedControlExchange& exchange, Clock::time_point now) {
    auto operation = exchange.begin("request\n", now);
    require(operation.status == SharedControlPublication::kPublished &&
                static_cast<bool>(operation.ticket),
            "owned begin must publish once and return its exact ticket");
    return operation;
}

void test_binding() {
    ServiceIncarnationBinding binding;
    require(!binding.bind({0, 29}) && !binding.bind({17, 0}) && !binding.is_bound(),
            "invalid identity must not partially bind");
    require(binding.bind({17, 29}) && binding.bind({17, 29}) && binding.is_bound(),
            "same logical service may bind repeatedly");
    require(!binding.bind({18, 29}) && !binding.bind({17, 30}) && binding.matches({17, 29}),
            "both words are immutable");
    for (std::size_t index = 0; index < 40; ++index) {
        ServiceIncarnationBinding concurrent;
        std::barrier ready{3};
        bool first = false;
        bool second = false;
        std::thread a([&] {
            ready.arrive_and_wait();
            first = concurrent.bind({17, 29});
        });
        std::thread b([&] {
            ready.arrive_and_wait();
            second = concurrent.bind({18, 30});
        });
        ready.arrive_and_wait();
        a.join();
        b.join();
        require(!(first && second) && concurrent.is_bound(),
                "conflicting concurrent first offers cannot both authorize");
        require(!first || concurrent.matches({17, 29}), "first winner must match both words");
        require(!second || concurrent.matches({18, 30}), "second winner must match both words");
    }
}

void test_tickets_and_idle_reconnect() {
    ServiceIncarnationBinding binding;
    OwnedControlExchange exchange{binding};
    OwnedControlExchange foreign{binding};
    const auto now = Clock::now();
    OwnedExchangeTicket previous;
    for (std::size_t connection = 0; connection < 2; ++connection) {
        Session session{exchange};
        for (const std::string& invalid :
             {std::string{}, std::string{"missing"}, std::string{"two\nlines\n"},
              std::string(257, 'x'), std::string("embedded\0\n", 10)})
            require(exchange.begin(invalid, now).status == SharedControlPublication::kRejected,
                    "invalid framing must reject without consuming ready endpoint");
        require(exchange.begin("request\n", Clock::time_point::max()).status ==
                    SharedControlPublication::kRejected,
                "deadline overflow must reject before publication");
        for (std::size_t index = 0; index < 100; ++index) {
            const auto operation = begin(exchange, now);
            const auto wait = exchange.wait_state(operation.ticket);
            require(wait && wait->deadline == now + 1000ms, "absolute deadline must be fixed");
            require((::fcntl(wait->descriptor, F_GETFL) & O_NONBLOCK) != 0 &&
                        (::fcntl(wait->descriptor, F_GETFD) & FD_CLOEXEC) != 0,
                    "owned socket must be nonblocking and close-on-exec");
            session.receive();
            std::string line;
            require(
                exchange.poll({}, line, now) == OwnedExchangeProgress::kRejected &&
                    exchange.poll(previous, line, now) == OwnedExchangeProgress::kRejected &&
                    foreign.poll(operation.ticket, line, now) == OwnedExchangeProgress::kRejected,
                "default/stale/foreign tickets cannot consume pending request");
            require(
                !exchange.abandon(previous) && !exchange.close_idle() &&
                    exchange.initialize({session.listener.path(), 1000ms}) ==
                        OwnedExchangeColdStatus::kRejected &&
                    exchange.begin("request\n", now).status == SharedControlPublication::kRejected,
                "accepted pending work prevents idle reset, reconnect and duplicate begin");
            const int flags = ::fcntl(wait->descriptor, F_GETFL);
            require(flags >= 0 && ::fcntl(wait->descriptor, F_SETFL, flags & ~O_NONBLOCK) == 0,
                    "fixture must clear mutable descriptor flag");
            for (std::size_t attempt = 0; attempt < 8; ++attempt) {
                require(
                    exchange.poll(operation.ticket, line, now) == OwnedExchangeProgress::kPending,
                    "unready operation must poll without sleeping despite changed flags");
                require(exchange.wait_state(operation.ticket)->deadline == wait->deadline,
                        "polling cannot renew deadline");
            }
            require(::fcntl(wait->descriptor, F_SETFL, flags) == 0, "fixture must restore flags");
            session.reply();
            require(
                exchange.poll(operation.ticket, line, now) == OwnedExchangeProgress::kResponse &&
                    line == "response\n" && !exchange.wait_state(operation.ticket) &&
                    exchange.poll(operation.ticket, line, now) == OwnedExchangeProgress::kRejected,
                "only actual response completes, exactly once");
            previous = operation.ticket;
        }
        require(exchange.close_idle() && binding.matches({17, 29}),
                "idle close retires transport but preserves logical session and generations");
    }
    require(!exchange.has_cleanup_error(), "owned cleanup must succeed");
}

void test_uncertainty() {
    enum class Failure : std::uint8_t {
        kDeadline,
        kBackwardsClock,
        kMalformedHint,
        kClose,
        kAbandon
    };
    for (const auto failure : {Failure::kDeadline, Failure::kBackwardsClock,
                               Failure::kMalformedHint, Failure::kClose, Failure::kAbandon}) {
        ServiceIncarnationBinding binding;
        OwnedControlExchange exchange{binding};
        Session session{exchange};
        const auto now = Clock::now();
        const auto operation = begin(exchange, now);
        session.receive();
        std::string line;
        auto observed = now;
        if (failure == Failure::kDeadline) {
            for (std::size_t count = 0; count < 32; ++count)
                require(exchange.poll(operation.ticket, line, now + 999ms) ==
                            OwnedExchangeProgress::kPending,
                        "repeated idle polls cannot extend expiry");
            observed = now + 1000ms;
        } else if (failure == Failure::kBackwardsClock) {
            observed = now - 1ms;
        } else if (failure == Failure::kMalformedHint) {
            send_bytes(session.peer(), "X");
        } else if (failure == Failure::kClose) {
            session.close_peer();
        } else {
            require(exchange.abandon(operation.ticket), "explicit abandonment must quarantine");
        }
        require(
            exchange.poll(operation.ticket, line, observed) == OwnedExchangeProgress::kUncertain,
            "published failure must become sticky uncertainty, not rejection or success");
        require(!exchange.wait_state(operation.ticket) && !exchange.close_idle() &&
                    exchange.initialize({session.listener.path(), 1000ms}) ==
                        OwnedExchangeColdStatus::kRejected &&
                    exchange.begin("request\n", now).status == SharedControlPublication::kRejected,
                "quarantine cannot reset, reconnect or replay");
        if (failure != Failure::kClose)
            require(!session.server.publish_response_with_hint(session.peer(), "late\n"),
                    "late response cannot repair closed uncertain transport");
        require(exchange.poll(operation.ticket, line, now + 2000ms) ==
                        OwnedExchangeProgress::kUncertain &&
                    binding.matches({17, 29}),
                "late publication or timer cannot clear uncertainty or logical pin");
    }
    ServiceIncarnationBinding binding;
    OwnedControlExchange exchange{binding};
    Session session{exchange};
    const auto now = Clock::now();
    const auto operation = begin(exchange, now);
    session.receive();
    session.reply();
    session.close_peer();
    std::string line;
    require(
        exchange.poll(operation.ticket, line, now + 2000ms) == OwnedExchangeProgress::kResponse &&
            line == "response\n",
        "already published actual response precedes timeout/closure classification");
    require(exchange.begin("request\n", now + 2000ms).status == SharedControlPublication::kRejected,
            "closed connection cannot publish a successor even after valid response");
}

void test_fork() {
    ServiceIncarnationBinding binding;
    OwnedControlExchange exchange{binding};
    Session session{exchange};
    const auto now = Clock::now();
    const auto operation = begin(exchange, now);
    session.receive();
    const pid_t child = ::fork();
    require(child >= 0, "owned fork must create");
    if (child == 0) {
        std::string line;
        const bool is_rejected =
            exchange.poll(operation.ticket, line, now) == OwnedExchangeProgress::kRejected &&
            !exchange.wait_state(operation.ticket) && !exchange.abandon(operation.ticket) &&
            exchange.begin("request\n", now).status == SharedControlPublication::kRejected &&
            exchange.initialize({session.listener.path(), 1000ms}) ==
                OwnedExchangeColdStatus::kRejected &&
            binding.matches({17, 29});
        // Explicit CPU destructor checks inherited close without shutdown.
        exchange.~OwnedControlExchange();
        ::_exit(is_rejected ? 0 : 1);
    }
    int status = 0;
    require(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "fork rejection child must terminate and be reaped");
    session.reply();
    std::string line;
    require(exchange.poll(operation.ticket, line, now) == OwnedExchangeProgress::kResponse &&
                line == "response\n",
            "child cleanup cannot shutdown parent connection or consume parent operation");
}

void test_cold_rejection() {
    ServiceIncarnationBinding binding;
    OwnedControlExchange exchange{binding};
    for (const auto timeout : {0ms, -1ms, 60001ms, std::chrono::milliseconds::max()})
        require(exchange.initialize({"/unopened", timeout}) == OwnedExchangeColdStatus::kRejected,
                "invalid timeout must reject before startup");
    for (const std::string& path : {std::string{}, std::string(108, 'x'), std::string("x\0y", 3)})
        require(exchange.initialize({path, 1000ms}) == OwnedExchangeColdStatus::kRejected,
                "invalid socket path must reject before connect");
    require(exchange.initialize({"/tmp/glimmer-absent-exchange.sock", 1000ms}) ==
                    OwnedExchangeColdStatus::kRejected &&
                !binding.is_bound(),
            "failed connect must not bind a logical session");
    Session session{exchange};
    require(exchange.close_idle(), "rejected cold setup must leave endpoint reusable");
    for (const bool is_pinned : {false, true}) {
        ServiceIncarnationBinding logical;
        if (is_pinned) require(logical.bind({17, 29}), "pinned unsupported fixture must bind");
        OwnedControlExchange endpoint{logical};
        Listener listener;
        auto result = OwnedExchangeColdStatus::kReady;
        std::thread initialize([&] { result = endpoint.initialize({listener.path(), 1000ms}); });
        Descriptor peer{listener.accept()};
        read_open(peer.get());
        send_bytes(peer.get(), kOwnerControlUnsupported);
        initialize.join();
        require(result == (is_pinned ? OwnedExchangeColdStatus::kRejected
                                     : OwnedExchangeColdStatus::kUnsupported) &&
                    logical.is_bound() == is_pinned,
                "explicit unsupported cannot erase a pinned session or publish domain work");
    }
    for (const bool is_partial : {false, true}) {
        ServiceIncarnationBinding logical;
        OwnedControlExchange endpoint{logical};
        Listener listener;
        auto result = OwnedExchangeColdStatus::kReady;
        std::thread initialize([&] { result = endpoint.initialize({listener.path(), 25ms}); });
        Descriptor peer{listener.accept()};
        read_open(peer.get());
        send_bytes(peer.get(), is_partial ? "GLIMMER_SHM_V3 RE" : "WRONG\n");
        initialize.join();
        require(
            result == OwnedExchangeColdStatus::kRejected && !logical.is_bound() &&
                endpoint.begin("request\n").status == SharedControlPublication::kRejected,
            "malformed/withheld negotiation must reject within cold budget without publication");
    }
}

void test_changed_incarnation_rejects_before_publication() {
    ServiceIncarnationBinding binding;
    OwnedControlExchange exchange{binding};
    {
        Session original{exchange};
        require(binding.matches({17, 29}) && exchange.close_idle(),
                "original logical session binds and closes while idle");
    }
    for (const ServiceIncarnation changed :
         {ServiceIncarnation{18, 29}, ServiceIncarnation{17, 30}}) {
        Listener listener;
        auto status = OwnedExchangeColdStatus::kReady;
        std::thread initialize([&] { status = exchange.initialize({listener.path(), 1000ms}); });
        Descriptor peer{listener.accept()};
        read_open(peer.get());
        send_bytes(peer.get(), kOwnerControlReady);
        SharedControlChannel offered;
        require(offered.create_server(changed, SharedControlMode::kReplyHintV3) &&
                    offered.send_offer(peer.get(), 1000ms),
                "different incarnation is offered by the fake server");
        initialize.join();
        std::string line;
        require(status == OwnedExchangeColdStatus::kRejected && binding.matches({17, 29}) &&
                    exchange.begin("request\n").status == SharedControlPublication::kRejected &&
                    offered.receive_request(peer.get(), 0, line) != SharedControlReadStatus::kLine,
                "changed service identity rejects without publishing a request");
    }
}

std::size_t descriptor_count() {
    DIR* directory = ::opendir("/proc/self/fd");
    require(directory != nullptr, "descriptor inventory must open");
    std::size_t count = 0;
    errno = 0;
    while (const auto* entry = ::readdir(directory)) {
        if (std::strcmp(entry->d_name, ".") != 0 && std::strcmp(entry->d_name, "..") != 0) ++count;
    }
    require(errno == 0 && ::closedir(directory) == 0, "descriptor inventory must close");
    return count;
}

}  // namespace

int main() {
    const auto descriptors = descriptor_count();
    test_binding();
    test_tickets_and_idle_reconnect();
    test_uncertainty();
    test_fork();
    test_cold_rejection();
    test_changed_incarnation_rejects_before_publication();
    require(descriptor_count() == descriptors, "all descriptors must release");
}
