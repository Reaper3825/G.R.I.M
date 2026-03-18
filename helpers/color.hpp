#pragma once
#include <cstdint>
#include <string>
#include <iostream>

// ====================================================
// Color struct — platform-independent RGBA
// ====================================================
struct Color {
    uint8_t r, g, b, a;

    constexpr Color(uint8_t r_ = 255, uint8_t g_ = 255,
                    uint8_t b_ = 255, uint8_t a_ = 255)
        : r(r_), g(g_), b(b_), a(a_) {}

    constexpr uint32_t toUInt() const noexcept {
        return (uint32_t(r) << 24) |
               (uint32_t(g) << 16) |
               (uint32_t(b) << 8)  |
               uint32_t(a);
    }

    static constexpr Color fromUInt(uint32_t v) noexcept {
        return Color(
            (v >> 24) & 0xFF,
            (v >> 16) & 0xFF,
            (v >> 8)  & 0xFF,
            v & 0xFF
        );
    }
};

// ====================================================
// Predefined palette
// ====================================================
namespace Colors {
    inline constexpr Color Default = {255, 255, 255};
    inline constexpr Color Red     = {255, 80, 80};
    inline constexpr Color Green   = {100, 255, 100};
    inline constexpr Color Blue    = {100, 100, 255};
    inline constexpr Color Yellow  = {255, 255, 100};
    inline constexpr Color Cyan    = {100, 255, 255};
    inline constexpr Color Magenta = {255, 100, 255};
    inline constexpr Color Gray    = {180, 180, 180};
}

// ====================================================
// Console color helpers
// ====================================================
#ifdef _WIN32
    #include "core/grim_platform.h"

    inline void setConsoleColor(const Color& c) {
        HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
        WORD attrib = 0;

        if (c.r > 200 && c.g < 100 && c.b < 100)
            attrib = FOREGROUND_RED | FOREGROUND_INTENSITY;
        else if (c.g > 200 && c.r < 100)
            attrib = FOREGROUND_GREEN | FOREGROUND_INTENSITY;
        else if (c.b > 200 && c.r < 100)
            attrib = FOREGROUND_BLUE | FOREGROUND_INTENSITY;
        else if (c.r > 200 && c.g > 200)
            attrib = FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_INTENSITY;
        else
            attrib = FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE;

        SetConsoleTextAttribute(hConsole, attrib);
    }

    inline void resetConsoleColor() {
        HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
        SetConsoleTextAttribute(hConsole,
            FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
    }

#else // non-Windows (ANSI)
    inline void setConsoleColor(const Color& c) {
        if (c.r > 200 && c.g < 100 && c.b < 100)
            std::cout << "\033[31m";
        else if (c.g > 200 && c.r < 100)
            std::cout << "\033[32m";
        else if (c.b > 200 && c.r < 100)
            std::cout << "\033[34m";
        else if (c.r > 200 && c.g > 200)
            std::cout << "\033[33m";
        else
            std::cout << "\033[0m";
    }

    inline void resetConsoleColor() {
        std::cout << "\033[0m";
    }
#endif
