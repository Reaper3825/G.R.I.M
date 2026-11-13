#pragma once

#include "core/input/Cursor.hpp"

namespace GRIM { namespace Perception { namespace Physical {

// Gesture-controller presentation policy. Unarmed uses the user's configured
// arrow; armed replaces it with a centered ring whose hotspot matches the
// gesture-controlled click position.
class PhysicalGestureCursor final : public GRIM::Input::Cursor {
public:
    bool SyncArmedState(bool armed, float pinch_openness = 1.0f);
    bool Shutdown();

protected:
    bool BuildBitmap(GRIM::Input::CursorBitmap& bitmap,
                     std::string& error) const override;

private:
    bool armed_requested_ = false;
    int requested_aperture_level_ = 12;
    int applied_aperture_level_ = -1;
};

}}} // namespace GRIM::Perception::Physical
