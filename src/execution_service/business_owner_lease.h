#pragma once

#include <memory>
#include <stdexcept>

namespace glimmer::app {

// A dedicated control block distinguishes explicit business leases from local
// observation copies. The anchor owns the resource but grants no native issue
// authority. Serialize bind/acquire/has_leases with owner lifecycle; acquired
// copies may be released by other threads.
template <class Resource>
class BusinessOwnerLease final {
   public:
    BusinessOwnerLease() = default;
    BusinessOwnerLease(const BusinessOwnerLease&) = delete;
    BusinessOwnerLease& operator=(const BusinessOwnerLease&) = delete;
    BusinessOwnerLease(BusinessOwnerLease&&) noexcept = default;
    BusinessOwnerLease& operator=(BusinessOwnerLease&&) noexcept = default;
    // Bind exactly once before publishing the owning slot.
    void bind(const std::shared_ptr<Resource>& resource) {
        if (!resource || anchor_) throw std::logic_error("invalid business owner binding");
        anchor_ = std::shared_ptr<Resource>(
            resource.get(), [lifetime = resource](Resource*) mutable { lifetime.reset(); });
    }
    [[nodiscard]] std::shared_ptr<Resource> acquire() const noexcept {
        return anchor_;
    }
    [[nodiscard]] bool has_leases() const noexcept {
        return anchor_ && anchor_.use_count() > 1;
    }

   private:
    std::shared_ptr<Resource> anchor_;
};

}  // namespace glimmer::app
