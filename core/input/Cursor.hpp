#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace GRIM { namespace Input {

// Platform-neutral pixel contract for a custom system cursor. Pixels use
// premultiplied ARGB (0xAARRGGBB); the platform implementation owns conversion
// to the native cursor handle.
struct CursorBitmap {
    int width = 0;
    int height = 0;
    int hotspot_x = 0;
    int hotspot_y = 0;
    std::vector<uint32_t> argb;
};

// Reusable ownership primitive for the operating system's normal pointer.
// Derived classes only describe their bitmap. Cursor centralizes native-handle
// lifetime, exclusive ownership, application, and restoration of the user's
// configured cursor scheme.
class Cursor {
public:
    Cursor() = default;
    virtual ~Cursor();

    Cursor(const Cursor&) = delete;
    Cursor& operator=(const Cursor&) = delete;
    Cursor(Cursor&&) = delete;
    Cursor& operator=(Cursor&&) = delete;

    bool Apply();
    bool Restore();

    bool IsApplied() const noexcept { return applied_; }
    const std::string& LastError() const noexcept { return last_error_; }

protected:
    virtual bool BuildBitmap(CursorBitmap& bitmap,
                             std::string& error) const = 0;

private:
    bool applied_ = false;
    std::string last_error_;
};

}} // namespace GRIM::Input
