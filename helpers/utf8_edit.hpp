#pragma once
#include <algorithm>
#include <string>
#include <string_view>

// Byte offsets at UTF-8 codepoint boundaries. Not a grapheme/IME engine.
namespace UTF8Edit {
inline bool continuation(unsigned char c) { return (c & 0xc0) == 0x80; }
inline int boundary(std::string_view s, int offset) {
    offset = std::clamp(offset, 0, static_cast<int>(s.size()));
    while (offset > 0 && offset < static_cast<int>(s.size()) && continuation(s[offset])) --offset;
    return offset;
}
inline int previous(std::string_view s, int offset) { return boundary(s, std::max(0, offset - 1)); }
inline int next(std::string_view s, int offset) {
    offset = std::min(static_cast<int>(s.size()), offset + 1);
    while (offset < static_cast<int>(s.size()) && continuation(s[offset])) ++offset;
    return offset;
}
inline std::string printable(std::string_view s, bool multiline = false) {
    std::string result;
    for (size_t i = 0; i < s.size();) {
        const auto lead = static_cast<unsigned char>(s[i]);
        const size_t n = lead < 0x80 ? 1 : lead >= 0xc2 && lead <= 0xdf ? 2 :
            lead >= 0xe0 && lead <= 0xef ? 3 : lead >= 0xf0 && lead <= 0xf4 ? 4 : 0;
        if (!n || i + n > s.size()) { ++i; continue; }
        unsigned value = lead & (n == 1 ? 0x7f : n == 2 ? 0x1f : n == 3 ? 0x0f : 0x07);
        bool valid = true;
        for (size_t j = 1; j < n; ++j) {
            valid &= continuation(static_cast<unsigned char>(s[i+j]));
            value = (value << 6) | (static_cast<unsigned char>(s[i+j]) & 0x3f);
        }
        valid &= value <= 0x10ffff && !(value >= 0xd800 && value <= 0xdfff);
        valid &= n == 1 || value >= (n == 2 ? 0x80u : n == 3 ? 0x800u : 0x10000u);
        if (!valid) { ++i; continue; }
        if ((value >= 32 && !(value >= 127 && value <= 159)) ||
            (multiline && (value == '\n' || value == '\t'))) result.append(s.substr(i,n));
        i += n;
    }
    return result;
}
}
