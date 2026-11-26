#include "internal/driver_cooperative_router.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace {
using namespace glimmer::interceptor;
int context_token;
struct State {
    int active{1};
    unsigned int queries{}, retains{}, balances{}, originals{}, retirements{};
    CUresult query_status{CUDA_SUCCESS};
    CUresult balance_status{CUDA_SUCCESS};
    bool guards{true};
} state;
CUcontext context() {
    return reinterpret_cast<CUcontext>(&context_token);
}
void require(bool value, const char* message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}
CUresult get_state(CUdevice, unsigned int* flags, int* active) {
    state.guards &= is_inside_driver_call();
    ++state.queries;
    if (state.query_status == CUDA_SUCCESS) {
        *flags = 0;
        *active = state.active;
    }
    return state.query_status;
}
CUresult retain(CUcontext* output, CUdevice) {
    state.guards &= is_inside_driver_call();
    ++state.retains;
    *output = context();
    return CUDA_SUCCESS;
}
CUresult balance(CUdevice) {
    state.guards &= is_inside_driver_call();
    ++state.balances;
    return state.balance_status;
}
CUresult resolve(const char*, void** output, int, cuuint64_t) {
    // Constructor-only placeholder addresses. No frontend is ever imported;
    // invoking any queue, context or pool callback would violate this fixture.
    state.guards &= is_inside_driver_call();
    *output = reinterpret_cast<void*>(&balance);
    return CUDA_SUCCESS;
}
void qualify(int mode) {
    state = {};
    DriverFunctionTable table;
    table.primary_context_get_state = get_state;
    table.primary_context_retain = retain;
    table.primary_context_release = balance;
    table.get_proc_address = resolve;
    DriverDispatch driver(table);
    DriverCooperativeRouter router(driver, {}, false);
    ProcessCallGate gate;
    bool uncertain = true;
    const auto original = [&]() -> CUresult {
        ++state.originals;
        require(gate.is_entered_by_current_thread(), "release owns enclosing process call");
        if (mode == 3 || mode == 10) {
            return CUDA_ERROR_INVALID_CONTEXT;
        }
        if (mode == 13) {
            return CUDA_ERROR_DEINITIALIZED;
        }
        if (mode == 16) {
            return CUDA_ERROR_INVALID_CONTEXT;
        }
        if (mode == 7) {
            return CUDA_ERROR_INVALID_DEVICE;
        }
        if (mode == 4 || mode == 17) {
            throw std::runtime_error("original effect unknown");
        }
        if (mode != 1) {
            state.active = 0;
        }
        if (mode == 5) {
            state.query_status = CUDA_ERROR_UNKNOWN;
        }
        if (mode == 12) {
            state.active = 2;
        }
        return CUDA_SUCCESS;
    };
    const auto retire = [&](CUcontext target) -> CUresult {
        ++state.retirements;
        require(target == context(), "retirement uses only identified destroyed owner");
        if (mode == 6) {
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        if (mode == 11) {
            throw std::bad_alloc();
        }
        return CUDA_SUCCESS;
    };
    require(router.release_primary_context(gate, 0, original, retire, uncertain) ==
                    CUDA_ERROR_INVALID_VALUE &&
                !uncertain && state.queries == 0,
            "missing process call rejects without any native identity/effect");
    const ProcessCallGate::Call call(gate);
    if (mode == 0 || mode == 10) {
        state.active = 0;
    }
    if (mode == 7) {
        state.query_status = CUDA_ERROR_INVALID_DEVICE;
    }
    if (mode == 8) {
        state.balance_status = CUDA_ERROR_UNKNOWN;
    }
    if (mode == 13 || mode == 14 || mode == 16 || mode == 17) {
        state.query_status = CUDA_ERROR_DEINITIALIZED;
    }
    if (mode == 15) {
        state.balance_status = CUDA_ERROR_DEINITIALIZED;
    }
    if (mode == 9) {
        ProcessCallGate::Transition transition(gate);
        transition.set_native_only(true);
    }
    const auto result = router.release_primary_context(gate, 0, original, retire, uncertain);
    const bool failure = (mode >= 3 && mode <= 6) || mode == 8 || mode == 11 || mode == 12 ||
                         (mode >= 14 && mode <= 17);
    const auto expected = mode == 3 || mode == 10 || mode == 16 ? CUDA_ERROR_INVALID_CONTEXT
                          : mode == 13 || mode == 15            ? CUDA_ERROR_DEINITIALIZED
                          : mode == 6                           ? CUDA_ERROR_OUT_OF_MEMORY
                          : mode == 7                           ? CUDA_ERROR_INVALID_DEVICE
                          : mode == 9                           ? CUDA_ERROR_NOT_SUPPORTED
                          : failure                             ? CUDA_ERROR_UNKNOWN
                                                                : CUDA_SUCCESS;
    if (result != expected || uncertain != failure || !state.guards || is_inside_driver_call()) {
        std::cerr << "release mode=" << mode << " actual=" << result << " expected=" << expected
                  << " uncertain=" << uncertain << " guards=" << state.guards << '\n';
    }
    require(result == expected && uncertain == failure && state.guards && !is_inside_driver_call(),
            "release preserves exact failure, uncertainty and lexical original guards");
    require(state.originals == (mode == 8 || mode == 9 || mode == 15 ? 0U : 1U) &&
                state.retirements == (mode == 2 || mode == 6 || mode == 11 ? 1U : 0U),
            "cold/live/last/failed release has exact original and CPU retirement count");
    require(state.retains == (mode == 0 || mode == 7 || mode == 9 || mode == 10 || mode == 13 ||
                                      mode == 14 || mode == 16 || mode == 17
                                  ? 0U
                                  : 1U) &&
                state.balances == state.retains,
            "identity never activates a cold primary and balances every temporary reference");
    if (failure) {
        const auto queries = state.queries;
        require(router.normal_owner_status() != CUDA_SUCCESS &&
                    router.release_primary_context(gate, 0, original, retire, uncertain) !=
                        CUDA_SUCCESS &&
                    !uncertain && state.queries == queries &&
                    state.originals == (mode == 8 || mode == 15 ? 0U : 1U),
                "actor-free unknown lifetime is sticky and cannot issue SDK again");
        std::shared_ptr<CudaCooperativeFrontend> owner;
        require(router.existing_frontend(context(), owner) != CUDA_SUCCESS && !owner &&
                    router.frontend(context(), owner) != CUDA_SUCCESS && !owner,
                "quarantine forbids lazy actor import and normal lookup");
    } else {
        require(router.normal_owner_status() == CUDA_SUCCESS,
                "cold/invalid/source rejection does not poison healthy normal ownership");
    }
}
void qualify_primary_destroy_admission() {
    state = {};
    DriverFunctionTable table;
    table.get_proc_address = resolve;
    DriverDispatch driver(table);
    DriverCooperativeRouter router(driver, {}, false);
    ProcessCallGate gate;
    CudaPrimaryDestroyDisposition disposition;
    const auto& handled = disposition.was_handled;
    const auto& uncertain = disposition.has_uncertain_lifetime;
    unsigned int originals{};
    const auto original = [&] {
        ++originals;
        return CUDA_ERROR_INVALID_CONTEXT;
    };
    require(router.try_rejected_primary_destroy(gate, context(), original, disposition) ==
                    CUDA_ERROR_INVALID_VALUE &&
                handled && !uncertain && originals == 0,
            "primary-destroy correction requires an enclosing process call before any effect");
    const ProcessCallGate::Call call(gate);
    require(router.try_rejected_primary_destroy(gate, context(), {}, disposition) ==
                    CUDA_ERROR_INVALID_VALUE &&
                handled && !uncertain && originals == 0,
            "missing primary-destroy callback rejects without ownership changes");
    require(router.try_rejected_primary_destroy(gate, context(), original, disposition) ==
                    CUDA_SUCCESS &&
                !handled && !uncertain && originals == 0 && state.queries == 0 &&
                state.retains == 0 && state.balances == 0,
            "unowned context performs no speculative binding, identity or original issue");
    {
        ProcessCallGate::Transition transition(gate);
        transition.set_native_only(true);
    }
    require(router.try_rejected_primary_destroy(gate, context(), original, disposition) ==
                    CUDA_ERROR_NOT_SUPPORTED &&
                handled && !uncertain && originals == 0,
            "retained native-only source cannot issue primary destruction");
}
void qualify_user_destroy_admission(int mode) {
    state = {};
    DriverFunctionTable table;
    table.get_proc_address = resolve;
    DriverDispatch driver(table);
    // Opt-in tracing must not invent or dereference an absent frontend.
    DriverCooperativeRouter router(driver, {}, true);
    ProcessCallGate gate;
    CudaContextDestroyDisposition disposition;
    const auto original = [&]() -> CUresult {
        ++state.originals;
        require(gate.is_entered_by_current_thread() && gate.is_native_only_on_current_thread(),
                "user original owns exclusive enclosing process admission");
        if (mode == 1) {
            return CUDA_ERROR_INVALID_CONTEXT;
        }
        if (mode == 2) {
            throw std::runtime_error("unknown user destruction effect");
        }
        return CUDA_SUCCESS;
    };
    const auto retire = [&](CUcontext target) -> CUresult {
        ++state.retirements;
        require(target == context() && gate.is_native_only_on_current_thread(),
                "successful actor-free retirement uses exact target under exclusion");
        if (mode == 3) {
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        if (mode == 4) {
            throw std::bad_alloc();
        }
        return CUDA_SUCCESS;
    };
    require(router.destroy_context(gate, context(), original, retire, true, disposition) ==
                    CUDA_ERROR_INVALID_VALUE &&
                !disposition.has_uncertain_lifetime && state.originals == 0 &&
                state.retirements == 0,
            "user destruction rejects missing process admission before any native effect");
    const ProcessCallGate::Call call(gate);
    require(router.destroy_context(gate, context(), {}, retire, true, disposition) ==
                    CUDA_ERROR_INVALID_VALUE &&
                router.destroy_context(gate, context(), original, {}, true, disposition) ==
                    CUDA_ERROR_INVALID_VALUE &&
                state.originals == 0 && state.retirements == 0,
            "missing user original or retirement callback cannot mutate ownership");
    if (mode == 5) {
        ProcessCallGate::Transition transition(gate);
        transition.set_native_only(true);
    }
    const auto target = mode == 6 ? nullptr : context();
    const auto result = router.destroy_context(gate, target, original, retire, true, disposition);
    const bool has_failed = mode == 2 || mode == 3 || mode == 4 || mode == 6;
    const auto expected = mode == 1    ? CUDA_ERROR_INVALID_CONTEXT
                          : mode == 3  ? CUDA_ERROR_OUT_OF_MEMORY
                          : mode == 5  ? CUDA_ERROR_NOT_SUPPORTED
                          : has_failed ? CUDA_ERROR_UNKNOWN
                                       : CUDA_SUCCESS;
    require(
        result == expected && disposition.was_handled &&
            disposition.has_uncertain_lifetime == has_failed &&
            state.originals == (mode == 5 ? 0U : 1U) &&
            state.retirements == (mode == 0 || mode == 3 || mode == 4 ? 1U : 0U) &&
            state.queries == 0 && state.retains == 0 && state.balances == 0 &&
            !is_inside_driver_call(),
        "actor-free lifecycle preserves exact original result/count and avoids identity/import");
    std::shared_ptr<CudaCooperativeFrontend> owner;
    if (has_failed) {
        require(router.normal_owner_status() != CUDA_SUCCESS &&
                    router.destroy_context(gate, target, original, retire, true, disposition) !=
                        CUDA_SUCCESS &&
                    !disposition.has_uncertain_lifetime && state.originals == 1 &&
                    router.existing_frontend(context(), owner) != CUDA_SUCCESS && !owner,
                "unknown user destruction or retirement is sticky and cannot retry/import");
    } else {
        require(router.normal_owner_status() == CUDA_SUCCESS &&
                    router.existing_frontend(context(), owner) == CUDA_SUCCESS && !owner,
                "actor-free success/validation/source rejection never invents a frontend");
    }
}
}  // namespace
int main() {
    try {
        qualify_primary_destroy_admission();
        for (int mode = 0; mode < 7; ++mode) {
            qualify_user_destroy_admission(mode);
        }
        for (int mode = 0; mode < 18; ++mode) {
            qualify(mode);
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "primary release routing: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
