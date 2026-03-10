// UISurfaceRegistry.cpp — runtime registry of live UI surfaces.
//======================================================//

#include "UISurfaceRegistry.hpp"

namespace GRIM::MMO {

// ─── Singleton ────────────────────────────────────────────

UISurfaceRegistry& UISurfaceRegistry::instance() {
    static UISurfaceRegistry inst;
    return inst;
}

// ─── Create ───────────────────────────────────────────────

std::string UISurfaceRegistry::create(const UISurfaceSpec& spec) {
    std::string err = validateSurfaceSpec(spec);
    if (!err.empty()) return err;

    std::lock_guard<std::mutex> lock(mutex_);
    if (surfaces_.count(spec.surface_id)) {
        return "Surface '" + spec.surface_id + "' already exists";
    }

    surfaces_[spec.surface_id] = spec;
    notify(SurfaceEvent::Created, spec.surface_id);
    return "";
}

// ─── Update ───────────────────────────────────────────────

std::string UISurfaceRegistry::update(const UISurfaceSpec& spec) {
    std::string err = validateSurfaceSpec(spec);
    if (!err.empty()) return err;

    std::lock_guard<std::mutex> lock(mutex_);
    auto it = surfaces_.find(spec.surface_id);
    if (it == surfaces_.end()) {
        return "Surface '" + spec.surface_id + "' not found";
    }

    it->second = spec;
    notify(SurfaceEvent::Updated, spec.surface_id);
    return "";
}

// ─── Show / Hide ──────────────────────────────────────────

bool UISurfaceRegistry::show(const std::string& surface_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = surfaces_.find(surface_id);
    if (it == surfaces_.end()) return false;

    it->second.visibility = VisibilityState::Visible;
    notify(SurfaceEvent::Shown, surface_id);
    return true;
}

bool UISurfaceRegistry::hide(const std::string& surface_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = surfaces_.find(surface_id);
    if (it == surfaces_.end()) return false;

    it->second.visibility = VisibilityState::Hidden;
    notify(SurfaceEvent::Hidden, surface_id);
    return true;
}

// ─── Destroy ──────────────────────────────────────────────

bool UISurfaceRegistry::destroy(const std::string& surface_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = surfaces_.find(surface_id);
    if (it == surfaces_.end()) return false;

    surfaces_.erase(it);
    notify(SurfaceEvent::Destroyed, surface_id);
    return true;
}

// ─── Lookup ───────────────────────────────────────────────

std::optional<UISurfaceSpec> UISurfaceRegistry::get(
    const std::string& surface_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = surfaces_.find(surface_id);
    if (it == surfaces_.end()) return std::nullopt;
    return it->second;
}

bool UISurfaceRegistry::exists(const std::string& surface_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return surfaces_.count(surface_id) > 0;
}

// ─── Queries ──────────────────────────────────────────────

std::vector<UISurfaceSpec> UISurfaceRegistry::visibleSurfaces() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<UISurfaceSpec> result;
    for (const auto& [id, spec] : surfaces_) {
        if (spec.visibility == VisibilityState::Visible) {
            result.push_back(spec);
        }
    }
    return result;
}

std::vector<UISurfaceSpec> UISurfaceRegistry::surfacesByKind(
    SurfaceKind kind) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<UISurfaceSpec> result;
    for (const auto& [id, spec] : surfaces_) {
        if (spec.kind == kind) {
            result.push_back(spec);
        }
    }
    return result;
}

size_t UISurfaceRegistry::count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return surfaces_.size();
}

// ─── Callbacks ────────────────────────────────────────────

void UISurfaceRegistry::onSurfaceChange(SurfaceChangeCallback callback) {
    std::lock_guard<std::mutex> lock(mutex_);
    callbacks_.push_back(std::move(callback));
}

void UISurfaceRegistry::destroyAll() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> ids;
    ids.reserve(surfaces_.size());
    for (const auto& [id, _] : surfaces_) {
        ids.push_back(id);
    }
    surfaces_.clear();
    for (const auto& id : ids) {
        notify(SurfaceEvent::Destroyed, id);
    }
}

// ─── Notification ─────────────────────────────────────────

void UISurfaceRegistry::notify(SurfaceEvent event,
                               const std::string& surface_id) {
    // Called under lock — callbacks should be fast and non-blocking
    for (const auto& cb : callbacks_) {
        cb(event, surface_id);
    }
}

} // namespace GRIM::MMO
