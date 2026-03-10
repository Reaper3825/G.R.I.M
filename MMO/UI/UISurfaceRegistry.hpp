// UISurfaceRegistry — runtime registry of live UI surfaces.
//
// Canonical owner of all active surfaces. Tools create surfaces
// through this registry after validation. The registry enforces
// uniqueness, lifetime policies, and provides lookup.
//
// Thread-safe: all operations serialized under mutex.
//======================================================//
#pragma once

#include "UISurfaceSpec.hpp"

#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace GRIM::MMO {

// =========================================================
// SurfaceChangeCallback — notified when surfaces change
//
// Used by the UI renderer to react to surface creation,
// visibility changes, and destruction.
// =========================================================
enum class SurfaceEvent : uint8_t {
    Created   = 0,
    Shown     = 1,
    Hidden    = 2,
    Updated   = 3,
    Destroyed = 4
};

using SurfaceChangeCallback =
    std::function<void(SurfaceEvent event, const std::string& surface_id)>;

// =========================================================
// UISurfaceRegistry
//
// Usage:
//   auto& reg = UISurfaceRegistry::instance();
//   auto err = reg.create(spec);  // validates + registers
//   reg.show("my_panel");
//   reg.hide("my_panel");
//   reg.destroy("my_panel");
//
// Surfaces are keyed by surface_id (unique).
// =========================================================
class UISurfaceRegistry {
public:
    static UISurfaceRegistry& instance();

    // Create and register a new surface. Validates the spec first.
    // Returns empty string on success, or error description.
    std::string create(const UISurfaceSpec& spec);

    // Update an existing surface's spec (re-validates).
    std::string update(const UISurfaceSpec& spec);

    // Show a surface (sets visibility to Visible).
    bool show(const std::string& surface_id);

    // Hide a surface (sets visibility to Hidden).
    bool hide(const std::string& surface_id);

    // Destroy and unregister a surface.
    bool destroy(const std::string& surface_id);

    // Look up a surface by id. Returns nullopt if not found.
    std::optional<UISurfaceSpec> get(const std::string& surface_id) const;

    // Check if a surface exists.
    bool exists(const std::string& surface_id) const;

    // Get all visible surfaces.
    std::vector<UISurfaceSpec> visibleSurfaces() const;

    // Get all surfaces of a specific kind.
    std::vector<UISurfaceSpec> surfacesByKind(SurfaceKind kind) const;

    // Get total registered surface count.
    size_t count() const;

    // Register a callback for surface change events.
    void onSurfaceChange(SurfaceChangeCallback callback);

    // Destroy all surfaces (e.g. on shutdown).
    void destroyAll();

private:
    UISurfaceRegistry() = default;

    void notify(SurfaceEvent event, const std::string& surface_id);

    mutable std::mutex mutex_;
    std::unordered_map<std::string, UISurfaceSpec> surfaces_;
    std::vector<SurfaceChangeCallback> callbacks_;
};

} // namespace GRIM::MMO
