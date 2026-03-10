// UISurfaceSpec.cpp — validation of UI surface specs.
//======================================================//

#include "UISurfaceSpec.hpp"

namespace GRIM::MMO {

std::string validateSurfaceSpec(const UISurfaceSpec& spec) {
    if (spec.surface_id.empty()) {
        return "surface_id is required";
    }
    if (spec.title.empty()) {
        return "title is required";
    }

    // Modal surfaces must use Exclusive input policy
    if (spec.kind == SurfaceKind::Modal &&
        spec.input_policy != InputPolicy::Exclusive) {
        return "Modal surfaces require InputPolicy::Exclusive";
    }

    // Toast surfaces must use AutoDismiss lifetime
    if (spec.kind == SurfaceKind::Toast &&
        spec.lifetime != LifetimePolicy::AutoDismiss) {
        return "Toast surfaces require LifetimePolicy::AutoDismiss";
    }

    // AutoDismiss requires a positive timeout
    if (spec.lifetime == LifetimePolicy::AutoDismiss &&
        spec.auto_dismiss_ms <= 0) {
        return "AutoDismiss lifetime requires positive auto_dismiss_ms";
    }

    // Validate widget specs
    for (const auto& w : spec.widgets) {
        if (w.widget_id.empty()) {
            return "All widgets require a widget_id";
        }
        if (w.widget_type.empty()) {
            return "Widget '" + w.widget_id + "' requires a widget_type";
        }
    }

    // Validate layout
    if (spec.layout.direction != "vertical" &&
        spec.layout.direction != "horizontal" &&
        spec.layout.direction != "grid") {
        return "layout.direction must be 'vertical', 'horizontal', or 'grid'";
    }
    if (spec.layout.direction == "grid" && spec.layout.columns < 1) {
        return "Grid layout requires columns >= 1";
    }

    return "";  // valid
}

} // namespace GRIM::MMO
