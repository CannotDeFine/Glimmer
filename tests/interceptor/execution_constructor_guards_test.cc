#include <cuda.h>
#include <cuda_runtime_api.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <string_view>

#ifdef cuGetProcAddress
#undef cuGetProcAddress
#endif
extern "C" CUresult CUDAAPI cuGetProcAddress(const char*, void**, int, cuuint64_t);

namespace {
constexpr std::array kExportedNames{"cuLinkCreate_v2",  "cuLinkAddData_v2",
                                    "cuLinkAddFile_v2", "cuLinkComplete",
                                    "cuLinkDestroy",    "cuLaunchCooperativeKernel"};
constexpr std::array kQueriedNames{"cuLinkCreate",   "cuLinkAddData", "cuLinkAddFile",
                                   "cuLinkComplete", "cuLinkDestroy", "cuLaunchCooperativeKernel"};
}  // namespace

#ifdef GLIMMER_CONSTRUCTOR_GUARD_PROVIDER
namespace {
std::atomic<unsigned int> native_consumers{};
std::atomic<unsigned int> initializations{};
}  // namespace
extern "C" unsigned int glimmer_constructor_guard_native_calls() {
    return native_consumers.load();
}
extern "C" unsigned int glimmer_constructor_guard_initializations() {
    return initializations.load();
}
extern "C" CUresult CUDAAPI cuInit(unsigned int) {
    ++initializations;
    return CUDA_SUCCESS;
}
extern "C" cudaError_t CUDARTAPI cudaRuntimeGetVersion(int* version) {
    if (!version) return cudaErrorInvalidValue;
    *version = CUDART_VERSION;
    return cudaSuccess;
}
extern "C" CUresult CUDAAPI cuLinkCreate_v2(unsigned int, CUjit_option*, void**, CUlinkState*) {
    ++native_consumers;
    return CUDA_ERROR_UNKNOWN;
}
extern "C" CUresult CUDAAPI cuLinkAddData_v2(CUlinkState, CUjitInputType, void*, std::size_t,
                                             const char*, unsigned int, CUjit_option*, void**) {
    ++native_consumers;
    return CUDA_ERROR_UNKNOWN;
}
extern "C" CUresult CUDAAPI cuLinkAddFile_v2(CUlinkState, CUjitInputType, const char*, unsigned int,
                                             CUjit_option*, void**) {
    ++native_consumers;
    return CUDA_ERROR_UNKNOWN;
}
extern "C" CUresult CUDAAPI cuLinkComplete(CUlinkState, void**, std::size_t*) {
    ++native_consumers;
    return CUDA_ERROR_UNKNOWN;
}
extern "C" CUresult CUDAAPI cuLinkDestroy(CUlinkState) {
    ++native_consumers;
    return CUDA_ERROR_UNKNOWN;
}
extern "C" CUresult CUDAAPI cuLaunchCooperativeKernel(CUfunction, unsigned int, unsigned int,
                                                      unsigned int, unsigned int, unsigned int,
                                                      unsigned int, unsigned int, CUstream,
                                                      void**) {
    ++native_consumers;
    return CUDA_ERROR_UNKNOWN;
}
// Installed Driver ABI fixes the order and numeric types of version and flags.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
extern "C" CUresult CUDAAPI cuGetProcAddress_v2(const char* name, void** output, int version,
                                                cuuint64_t flags,
                                                CUdriverProcAddressQueryResult* status) {
    if (!name || !output || flags) return CUDA_ERROR_INVALID_VALUE;
    *output = nullptr;
    if (version == 1) {
        if (status) *status = CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT;
        return CUDA_SUCCESS;
    }
    const std::array addresses{reinterpret_cast<void*>(&cuLinkCreate_v2),
                               reinterpret_cast<void*>(&cuLinkAddData_v2),
                               reinterpret_cast<void*>(&cuLinkAddFile_v2),
                               reinterpret_cast<void*>(&cuLinkComplete),
                               reinterpret_cast<void*>(&cuLinkDestroy),
                               reinterpret_cast<void*>(&cuLaunchCooperativeKernel)};
    for (std::size_t index = 0; index < kQueriedNames.size(); ++index) {
        if (name == std::string_view(kQueriedNames[index]) ||
            name == std::string_view(kExportedNames[index])) {
            // A genuine native address that does not match the selected ABI
            // must not be turned into a supported controlled entry.
            *output = version == 12059 ? reinterpret_cast<void*>(&cuInit) : addresses[index];
        }
    }
    if (status)
        *status = *output ? CU_GET_PROC_ADDRESS_SUCCESS : CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
    return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuGetProcAddress(const char* name, void** output, int version,
                                             cuuint64_t flags) {
    return cuGetProcAddress_v2(name, output, version, flags, nullptr);
}
#else
#include "internal/process_identity.h"

#include <cerrno>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <dlfcn.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

extern "C" unsigned int glimmer_constructor_guard_native_calls();
extern "C" unsigned int glimmer_constructor_guard_initializations();
namespace {
void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

class Descriptor final {
   public:
    explicit Descriptor(int value) : value_(value) {
        expect(value >= 0, "create CPU-only owned descriptor");
    }
    ~Descriptor() {
        if (value_ >= 0) (void)::close(value_);
    }
    Descriptor(const Descriptor&) = delete;
    Descriptor& operator=(const Descriptor&) = delete;
    [[nodiscard]] int get() const {
        return value_;
    }
    void close() {
        const auto value = value_;
        value_ = -1;
        expect(::close(value) == 0, "close CPU-only owned descriptor");
    }

   private:
    int value_;
};

class Diagnostics final {
   public:
    Diagnostics()
        : saved_(::dup(STDERR_FILENO)), storage_(::memfd_create("guard-log", MFD_CLOEXEC)) {
        expect(::dup2(storage_.get(), STDERR_FILENO) == STDERR_FILENO, "capture rejection records");
    }
    ~Diagnostics() {
        if (active_) (void)::dup2(saved_.get(), STDERR_FILENO);
    }
    Diagnostics(const Diagnostics&) = delete;
    Diagnostics& operator=(const Diagnostics&) = delete;
    [[nodiscard]] std::string content() const {
        struct stat info {};
        expect(::fstat(storage_.get(), &info) == 0 && info.st_size >= 0 && info.st_size < 65536,
               "diagnostic capture is bounded");
        std::string result(static_cast<std::size_t>(info.st_size), '\0');
        std::size_t offset{};
        while (offset < result.size()) {
            const auto count = ::pread(storage_.get(), result.data() + offset,
                                       result.size() - offset, static_cast<off_t>(offset));
            expect(count > 0, "read complete diagnostic capture");
            offset += static_cast<std::size_t>(count);
        }
        return result;
    }
    void restore() {
        expect(::dup2(saved_.get(), STDERR_FILENO) == STDERR_FILENO, "restore original stderr");
        active_ = false;
        saved_.close();
        storage_.close();
    }

   private:
    Descriptor saved_, storage_;
    bool active_{true};
};

class ProtectedPage final {
   public:
    ProtectedPage() {
        const auto page_size = ::sysconf(_SC_PAGESIZE);
        expect(page_size > 0, "read page size");
        size_ = static_cast<std::size_t>(page_size);
        address_ = ::mmap(nullptr, size_, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        expect(address_ != MAP_FAILED, "create inaccessible caller storage");
    }
    ~ProtectedPage() {
        if (address_ != MAP_FAILED) (void)::munmap(address_, size_);
    }
    ProtectedPage(const ProtectedPage&) = delete;
    ProtectedPage& operator=(const ProtectedPage&) = delete;
    [[nodiscard]] void* get() const {
        return address_;
    }
    void close() {
        expect(::munmap(address_, size_) == 0, "release inaccessible caller storage");
        address_ = MAP_FAILED;
    }

   private:
    void* address_{MAP_FAILED};
    std::size_t size_{};
};

struct Functions {
    decltype(&cuLinkCreate_v2) create{};
    decltype(&cuLinkAddData_v2) data{};
    decltype(&cuLinkAddFile_v2) file{};
    decltype(&cuLinkComplete) complete{};
    decltype(&cuLinkDestroy) destroy{};
    decltype(&cuLaunchCooperativeKernel) launch{};
};

bool owned(void* address) {
    Dl_info selected{}, owner{};
    return address && ::dladdr(address, &selected) &&
           ::dladdr(reinterpret_cast<void*>(&cuGetProcAddress), &owner) &&
           selected.dli_fbase == owner.dli_fbase && owner.dli_fname &&
           std::string_view(owner.dli_fname).find("glimmer") != std::string_view::npos;
}

Functions lookup(unsigned int route, void* provider) {
    std::array<void*, 6> addresses{};
    for (std::size_t index = 0; index < addresses.size(); ++index) {
        if (route < 2) {
            addresses[index] = ::dlsym(route ? provider : RTLD_DEFAULT, kExportedNames[index]);
        } else if (route == 2) {
            expect(
                cuGetProcAddress(kQueriedNames[index], &addresses[index], 12060, 0) == CUDA_SUCCESS,
                "Driver v1 metadata query succeeds");
        } else if (route == 3) {
            CUdriverProcAddressQueryResult status = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
            expect(cuGetProcAddress_v2(kQueriedNames[index], &addresses[index], 12060, 0,
                                       &status) == CUDA_SUCCESS &&
                       status == CU_GET_PROC_ADDRESS_SUCCESS,
                   "Driver v2 metadata query preserves success");
        } else {
            cudaDriverEntryPointQueryResult status = cudaDriverEntryPointSymbolNotFound;
            const auto result =
                route == 4
                    ? cudaGetDriverEntryPoint(kQueriedNames[index], &addresses[index], 0, &status)
                    : cudaGetDriverEntryPointByVersion(kQueriedNames[index], &addresses[index],
                                                       12060, 0, &status);
            expect(result == cudaSuccess && status == cudaDriverEntryPointSuccess,
                   "Runtime cached metadata query preserves success");
        }
        if (!owned(addresses[index]))
            throw std::runtime_error(std::string("missing owned constructor guard: ") +
                                     kExportedNames[index]);
    }
    return {reinterpret_cast<decltype(&cuLinkCreate_v2)>(addresses[0]),
            reinterpret_cast<decltype(&cuLinkAddData_v2)>(addresses[1]),
            reinterpret_cast<decltype(&cuLinkAddFile_v2)>(addresses[2]),
            reinterpret_cast<decltype(&cuLinkComplete)>(addresses[3]),
            reinterpret_cast<decltype(&cuLinkDestroy)>(addresses[4]),
            reinterpret_cast<decltype(&cuLaunchCooperativeKernel)>(addresses[5])};
}

void record(CUresult status, std::size_t operation, std::string& expected) {
    expect(status == CUDA_ERROR_NOT_SUPPORTED, "guard returns only explicit owned NOT_SUPPORTED");
    expected += std::string("Glimmer: unsupported owned CUDA operation: ") +
                kExportedNames.at(operation) + '\n';
}

void exercise(const Functions& api, void* inaccessible, std::string& expected) {
    const auto initializations = glimmer_constructor_guard_initializations();
    std::array<unsigned char, 24> data{};
    data.fill(0xa5);
    const auto original_data = data;
    std::array options{CU_JIT_OPTIMIZATION_LEVEL, CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES};
    const auto original_options = options;
    std::array<void*, 2> values{data.data(), &options};
    const auto original_values = values;
    auto state = reinterpret_cast<CUlinkState>(data.data());
    const auto original_state = state;
    void* image = data.data();
    std::size_t bytes = 0x123456;
    unsigned int count = 17;
    void* pointer = data.data();
    std::array<void*, 2> parameters{&count, &pointer};
    const auto original_parameters = parameters;
    const auto function = reinterpret_cast<CUfunction>(data.data());
    const auto stream = reinterpret_cast<CUstream>(data.data());
    record(api.create(2, options.data(), values.data(), &state), 0, expected);
    record(api.data(state, CU_JIT_INPUT_CUBIN, data.data(), data.size(), "fixture-input", 2,
                    options.data(), values.data()),
           1, expected);
    record(api.file(state, CU_JIT_INPUT_OBJECT, "caller-file-must-not-be-opened", 2, options.data(),
                    values.data()),
           2, expected);
    record(api.complete(state, &image, &bytes), 3, expected);
    record(api.destroy(state), 4, expected);
    record(api.launch(function, 1, 1, 1, 32, 1, 1, 0, stream, parameters.data()), 5, expected);
    record(api.complete(state, &image, nullptr), 3, expected);
    record(api.complete(state, nullptr, &bytes), 3, expected);
    expect(state == original_state && image == data.data() && bytes == 0x123456 &&
               options == original_options && values == original_values && data == original_data &&
               count == 17 && pointer == data.data() && parameters == original_parameters,
           "all visible outputs, option slots, input data and kernel arguments stay unchanged");

    record(api.create(0, nullptr, nullptr, nullptr), 0, expected);
    record(api.data(nullptr, CU_JIT_INPUT_PTX, nullptr, 0, nullptr, 0, nullptr, nullptr), 1,
           expected);
    record(api.file(nullptr, CU_JIT_INPUT_LIBRARY, nullptr, 0, nullptr, nullptr), 2, expected);
    record(api.complete(nullptr, nullptr, nullptr), 3, expected);
    record(api.destroy(nullptr), 4, expected);
    record(api.launch(nullptr, 0, 0, 0, 0, 0, 0, 0, nullptr, nullptr), 5, expected);

    // Deliberately invalid caller addresses must never be probed by these leaves.
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    void* poison = reinterpret_cast<void*>(std::numeric_limits<std::uintptr_t>::max());
    for (void* address : {inaccessible, poison}) {
        const auto link = reinterpret_cast<CUlinkState>(address);
        auto* option_pointer = static_cast<CUjit_option*>(address);
        auto** value_pointer = static_cast<void**>(address);
        const auto maximum = std::numeric_limits<unsigned int>::max();
        record(
            api.create(maximum, option_pointer, value_pointer, static_cast<CUlinkState*>(address)),
            0, expected);
        record(api.data(link, CU_JIT_INPUT_NVVM, address, std::numeric_limits<std::size_t>::max(),
                        static_cast<const char*>(address), maximum, option_pointer, value_pointer),
               1, expected);
        record(api.file(link, CU_JIT_INPUT_FATBINARY, static_cast<const char*>(address), maximum,
                        option_pointer, value_pointer),
               2, expected);
        record(api.complete(link, value_pointer, static_cast<std::size_t*>(address)), 3, expected);
        record(api.destroy(link), 4, expected);
        record(api.launch(reinterpret_cast<CUfunction>(address), maximum, maximum, maximum, maximum,
                          maximum, maximum, maximum, reinterpret_cast<CUstream>(address),
                          value_pointer),
               5, expected);
    }
    expect(glimmer_constructor_guard_native_calls() == 0 &&
               glimmer_constructor_guard_initializations() == initializations,
           "guard calls never initialize CUDA or invoke a native resource consumer");
}

void negative_queries(void* provider) {
    for (const auto* name : kQueriedNames) {
        void* address = provider;
        CUdriverProcAddressQueryResult status = CU_GET_PROC_ADDRESS_SUCCESS;
        expect(cuGetProcAddress_v2(name, &address, 1, 0, &status) == CUDA_SUCCESS && !address &&
                   status == CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT,
               "native unsupported version remains authoritative");
        address = provider;
        expect(cuGetProcAddress(name, &address, 1, 0) == CUDA_SUCCESS && !address,
               "Driver v1 missing version does not manufacture availability");
        address = provider;
        status = CU_GET_PROC_ADDRESS_SUCCESS;
        expect(cuGetProcAddress_v2(name, &address, 12059, 0, &status) == CUDA_ERROR_NOT_SUPPORTED &&
                   !address && status == CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND,
               "a mismatched native ABI address cannot escape or acquire a guard");
    }
    void* address = provider;
    CUdriverProcAddressQueryResult status = CU_GET_PROC_ADDRESS_SUCCESS;
    expect(cuGetProcAddress_v2("cuConstructorGuardMissing", &address, 12060, 0, &status) ==
                   CUDA_SUCCESS &&
               !address && status == CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND,
           "native missing-symbol result remains authoritative");
    for (const auto* obsolete : {"cuLinkCreate", "cuLinkAddData", "cuLinkAddFile"})
        expect(!::dlsym(RTLD_DEFAULT, obsolete) && !::dlsym(provider, obsolete),
               "obsolete unversioned ELF ABI is not exposed as modern v2");
}

void failed_diagnostic(const Functions& api) {
    const auto initializations = glimmer_constructor_guard_initializations();
    Descriptor saved(::dup(STDERR_FILENO));
    expect(::close(STDERR_FILENO) == 0, "close diagnostic sink for failure control");
    const auto status = api.destroy(nullptr);
    const auto restored = ::dup2(saved.get(), STDERR_FILENO);
    saved.close();
    expect(restored == STDERR_FILENO && status == CUDA_ERROR_NOT_SUPPORTED &&
               glimmer_constructor_guard_initializations() == initializations,
           "failed diagnostic write cannot change rejection or retry native work");
}

void run(const char* provider_path, bool lookup_only) {
    Diagnostics diagnostics;
    void* provider = ::dlopen(provider_path, RTLD_NOW | RTLD_LOCAL);
    expect(provider, "load independent Driver provider");
    std::array<Functions, 6> entries{};
    std::string expected;
    ProtectedPage protected_page;
    for (unsigned int route = 0; route < entries.size(); ++route) {
        const auto before_lookup = diagnostics.content();
        entries[route] = lookup(route, provider);
        expect(diagnostics.content() == before_lookup,
               "lookup alone emits no rejection diagnostic");
        if (route < 4)
            expect(
                glimmer_constructor_guard_initializations() == 0,
                "global, explicit and Driver guards are callable before any CUDA initialization");
        if (!lookup_only) exercise(entries[route], protected_page.get(), expected);
    }
    negative_queries(provider);
    expect(::dlclose(provider) == 0, "close explicit provider handle");
    if (!lookup_only) {
        // Every resolved function remains a genuine typed owned entry after
        // dlclose; the Runtime's metadata initialization is not a guard call.
        exercise(entries.back(), protected_page.get(), expected);
        failed_diagnostic(entries.back());
    }
    expect(diagnostics.content() == expected, "exact fixed operation diagnostics and no extras");
    expect(glimmer_constructor_guard_native_calls() == 0, "zero native link/cooperative consumers");
    protected_page.close();
    diagnostics.restore();
}

void configured_run(const char* provider_path) {
    char pattern[] = "/tmp/glimmer-constructor-guards-XXXXXX";
    const auto* directory = ::mkdtemp(pattern);
    expect(directory, "create private unused endpoint directory");
    const auto path = std::string(directory) + "/service.sock";
    try {
        Descriptor listener(::socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0));
        sockaddr_un address{};
        address.sun_family = AF_UNIX;
        expect(path.size() < sizeof(address.sun_path), "bounded private socket path");
        std::memcpy(address.sun_path, path.c_str(), path.size() + 1);
        expect(::bind(listener.get(), reinterpret_cast<const sockaddr*>(&address),
                      sizeof(address)) == 0 &&
                   ::chmod(path.c_str(), 0600) == 0 && ::listen(listener.get(), 4) == 0,
               "prepare authenticated but unused execution endpoint");
        const auto identity = glimmer::control::current_process_identity();
        expect(identity && ::setenv("GLIMMER_EXECUTION_SOCKET", path.c_str(), 1) == 0 &&
                   ::setenv("GLIMMER_EXECUTION_SERVER_PID", std::to_string(identity->pid).c_str(),
                            1) == 0 &&
                   ::setenv("GLIMMER_EXECUTION_SERVER_START",
                            std::to_string(identity->start_time_ticks).c_str(), 1) == 0,
               "configure actual peer identity without starting a service");
        run(provider_path, false);
        const auto connection = ::accept4(listener.get(), nullptr, nullptr, SOCK_CLOEXEC);
        if (connection >= 0) {
            Descriptor unexpected(connection);
            throw std::runtime_error("constructor guards unexpectedly opened IPC");
        }
        expect(errno == EAGAIN || errno == EWOULDBLOCK, "no IPC connection was attempted");
        listener.close();
        expect(::unlink(path.c_str()) == 0 && ::rmdir(directory) == 0, "remove unused endpoint");
    } catch (...) {
        (void)::unlink(path.c_str());
        (void)::rmdir(directory);
        throw;
    }
}
}  // namespace

int main(int argc, char** argv) {
    try {
        expect(argc == 2 || argc == 3 || argc == 4, "expected provider and optional isolated mode");
        if (argc == 2) {
            for (const auto* mode : {"no-config", "configured"}) {
                const auto child = ::fork();
                expect(child >= 0, "create isolated preinitialization case");
                if (!child) {
                    ::execl(argv[0], argv[0], argv[1], "--case", mode, nullptr);
                    _exit(127);
                }
                int status{};
                expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) &&
                           WEXITSTATUS(status) == 0,
                       "isolated constructor guard case failed");
            }
            return EXIT_SUCCESS;
        }
        expect(::unsetenv("GLIMMER_EXECUTION_SOCKET") == 0 &&
                   ::unsetenv("GLIMMER_EXECUTION_SERVER_PID") == 0 &&
                   ::unsetenv("GLIMMER_EXECUTION_SERVER_START") == 0,
               "clear unrelated execution configuration");
        if (argc == 3) {
            expect(std::string_view(argv[2]) == "--lookup-only", "explicit lookup-only RED mode");
            run(argv[1], true);
        } else {
            expect(std::string_view(argv[2]) == "--case", "explicit isolated case selector");
            if (std::string_view(argv[3]) == "configured")
                configured_run(argv[1]);
            else {
                expect(std::string_view(argv[3]) == "no-config", "known preinitialization mode");
                run(argv[1], false);
            }
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
#endif
