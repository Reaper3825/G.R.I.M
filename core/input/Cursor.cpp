#include "Cursor.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <utility>

#ifdef _WIN32
#include "core/grim_platform.h"
#endif

namespace GRIM { namespace Input {

namespace {

struct CursorOwnership {
    std::mutex mutex;
    Cursor* owner = nullptr;
};

// Deliberately process-lifetime storage. Cursor instances may be function
// statics, so leaking this tiny coordinator avoids static-destruction ordering
// hazards while their destructors restore the system cursor.
CursorOwnership& Ownership() {
    static auto* ownership = new CursorOwnership();
    return *ownership;
}

bool ValidateBitmap(const CursorBitmap& bitmap, std::string& error) {
    if (bitmap.width <= 0 || bitmap.height <= 0 ||
        bitmap.width > 256 || bitmap.height > 256) {
        error = "cursor dimensions must be between 1 and 256 pixels";
        return false;
    }
    const size_t width = static_cast<size_t>(bitmap.width);
    const size_t height = static_cast<size_t>(bitmap.height);
    if (width > std::numeric_limits<size_t>::max() / height ||
        bitmap.argb.size() != width * height) {
        error = "cursor pixel count does not match its dimensions";
        return false;
    }
    if (bitmap.hotspot_x < 0 || bitmap.hotspot_x >= bitmap.width ||
        bitmap.hotspot_y < 0 || bitmap.hotspot_y >= bitmap.height) {
        error = "cursor hotspot lies outside the bitmap";
        return false;
    }
    return true;
}

#ifdef _WIN32

std::string Win32Failure(const char* operation) {
    const DWORD code = ::GetLastError();
    char* detail = nullptr;
    DWORD count = ::FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
            FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr, code, 0, reinterpret_cast<char*>(&detail), 0, nullptr);
    std::string message = operation;
    message += " failed (Win32 ";
    message += std::to_string(code);
    message += ")";
    if (count != 0 && detail) {
        while (count != 0 &&
               (detail[count - 1] == '\r' || detail[count - 1] == '\n')) {
            detail[--count] = '\0';
        }
        message += ": ";
        message += detail;
    }
    if (detail) ::LocalFree(detail);
    return message;
}

HCURSOR CreateNativeCursor(const CursorBitmap& bitmap, std::string& error) {
    BITMAPV5HEADER header{};
    header.bV5Size = sizeof(header);
    header.bV5Width = bitmap.width;
    header.bV5Height = -bitmap.height; // top-down pixels
    header.bV5Planes = 1;
    header.bV5BitCount = 32;
    header.bV5Compression = BI_BITFIELDS;
    header.bV5RedMask = 0x00FF0000;
    header.bV5GreenMask = 0x0000FF00;
    header.bV5BlueMask = 0x000000FF;
    header.bV5AlphaMask = 0xFF000000;

    void* pixels = nullptr;
    HDC screen_dc = ::GetDC(nullptr);
    HBITMAP color = ::CreateDIBSection(
        screen_dc, reinterpret_cast<const BITMAPINFO*>(&header),
        DIB_RGB_COLORS, &pixels, nullptr, 0);
    if (screen_dc) ::ReleaseDC(nullptr, screen_dc);
    if (!color || !pixels) {
        if (color) ::DeleteObject(color);
        error = Win32Failure("CreateDIBSection");
        return nullptr;
    }
    std::memcpy(pixels, bitmap.argb.data(),
                bitmap.argb.size() * sizeof(uint32_t));

    const size_t mask_stride =
        static_cast<size_t>(((bitmap.width + 15) / 16) * 2);
    std::vector<uint8_t> mask_bits(
        mask_stride * static_cast<size_t>(bitmap.height), 0);
    HBITMAP mask = ::CreateBitmap(bitmap.width, bitmap.height, 1, 1,
                                  mask_bits.data());
    if (!mask) {
        ::DeleteObject(color);
        error = Win32Failure("CreateBitmap");
        return nullptr;
    }

    ICONINFO info{};
    info.fIcon = FALSE;
    info.xHotspot = static_cast<DWORD>(bitmap.hotspot_x);
    info.yHotspot = static_cast<DWORD>(bitmap.hotspot_y);
    info.hbmMask = mask;
    info.hbmColor = color;
    HCURSOR cursor = reinterpret_cast<HCURSOR>(::CreateIconIndirect(&info));
    ::DeleteObject(mask);
    ::DeleteObject(color);
    if (!cursor) error = Win32Failure("CreateIconIndirect");
    return cursor;
}

bool RestoreNativeCursor(std::string& error) {
    if (::SystemParametersInfoW(SPI_SETCURSORS, 0, nullptr, 0) == FALSE) {
        error = Win32Failure("SystemParametersInfo(SPI_SETCURSORS)");
        return false;
    }
    return true;
}

#endif

} // namespace

Cursor::~Cursor() {
    try {
        Restore();
    } catch (...) {
        // Destructors must not throw. Normal controller shutdown calls Restore
        // explicitly so failures remain observable there.
    }
}

bool Cursor::Apply() {
    CursorBitmap bitmap;
    std::string error;
    if (!BuildBitmap(bitmap, error) || !ValidateBitmap(bitmap, error)) {
        last_error_ = error.empty() ? "cursor bitmap creation failed" : error;
        return false;
    }

#ifdef _WIN32
    HCURSOR native_cursor = CreateNativeCursor(bitmap, error);
    if (!native_cursor) {
        last_error_ = std::move(error);
        return false;
    }

    auto& ownership = Ownership();
    std::lock_guard<std::mutex> lock(ownership.mutex);
    if (::SetSystemCursor(native_cursor, 32512 /* OCR_NORMAL */) == FALSE) {
        // SetSystemCursor consumes the handle even when replacing the cursor
        // fails; do not destroy it again.
        last_error_ = Win32Failure("SetSystemCursor(OCR_NORMAL)");
        return false;
    }
    if (ownership.owner && ownership.owner != this)
        ownership.owner->applied_ = false;
    ownership.owner = this;
    applied_ = true;
    last_error_.clear();
    return true;
#else
    (void)bitmap;
    last_error_ = "custom system cursors are not implemented on this platform";
    return false;
#endif
}

bool Cursor::Restore() {
    auto& ownership = Ownership();
    std::lock_guard<std::mutex> lock(ownership.mutex);
    if (ownership.owner != this) {
        applied_ = false;
        return true;
    }

#ifdef _WIN32
    std::string error;
    if (!RestoreNativeCursor(error)) {
        last_error_ = std::move(error);
        return false;
    }
#endif
    ownership.owner = nullptr;
    applied_ = false;
    last_error_.clear();
    return true;
}

}} // namespace GRIM::Input
