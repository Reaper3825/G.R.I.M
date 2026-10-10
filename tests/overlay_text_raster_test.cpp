// Host-only software text rendering test. Does not start GRIM or create a window.
#include "ui/overlay_renderer.hpp"
#include "logger.hpp"
#include <stb/stb_truetype.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>

// Minimal surface/logging support for linking the actual text renderer in isolation.
void logDebug(const std::string&, const std::string&) {}
void logError(const std::string&, const std::string&) {}
void OverlayRenderer::init(int width, int height, void* pixels) {
    m_width = width; m_height = height; m_pixels = pixels;
}
void OverlayRenderer::expandDirtyRect(int, int, int, int) {}
ClipRect OverlayRenderer::activeClip() const {
    return m_clipStack.empty() ? ClipRect{0, 0, m_width, m_height} : m_clipStack.back();
}
void OverlayRenderer::pushClipRect(const Vec2& pos, const Vec2& size) {
    m_clipStack.push_back({static_cast<int>(pos.x), static_cast<int>(pos.y),
        static_cast<int>(pos.x + size.x), static_cast<int>(pos.y + size.y)});
}
void OverlayRenderer::popClipRect() { m_clipStack.pop_back(); }

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

int main(int argc, char** argv) {
    require(argc >= 2, "Pass one or more TTF font paths");
    constexpr int width = 160, height = 100;
    std::vector<uint32_t> pixels(width * height);
    OverlayRenderer renderer;
    renderer.init(width, height, pixels.data());
    for (int fontIndex = 1; fontIndex < argc; ++fontIndex) {
        std::ifstream input(argv[fontIndex], std::ios::binary);
        std::vector<unsigned char> fontBytes((std::istreambuf_iterator<char>(input)), {});
        require(!fontBytes.empty(), "Font missing");
        stbtt_fontinfo font{};
        require(stbtt_InitFont(&font, fontBytes.data(), stbtt_GetFontOffsetForIndex(fontBytes.data(), 0)),
                "Invalid test font");
        for (int baseSize : {16, 22}) {
            renderer.setFont(argv[fontIndex], baseSize);
            for (float size : {17.0f, 20.0f, 24.0f, 10.0f}) {
                const float pixelSize = baseSize * (size / UITheme::Typography::ReferenceSize);
                const float scale = stbtt_ScaleForPixelHeight(&font, pixelSize);
                int w, h, xoff, yoff;
                unsigned char* bitmap = stbtt_GetCodepointBitmap(&font, scale, scale, 'M', &w, &h, &xoff, &yoff);
                require(bitmap != nullptr, "Reference glyph missing");
                std::vector<uint32_t> expected(width * height);
                for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
                    const int dx = 10 + xoff + x;
                    const int dy = static_cast<int>(std::floor(10 + pixelSize + yoff)) + y;
                    const uint32_t coverage = bitmap[y * w + x];
                    expected[dy * width + dx] = coverage * 0x01010101u;
                }
                stbtt_FreeBitmap(bitmap, nullptr);
                for (int repeat = 0; repeat < 2; ++repeat) {
                    std::fill(pixels.begin(), pixels.end(), 0);
                    renderer.drawText({10, 10}, "M", 0xFFFFFFFF, size);
                    require(pixels == expected, "Sized text differs from native glyph rasterization");
                }
                const float baseWidth = renderer.measureTextWidth("M", UITheme::Typography::ReferenceSize);
                require(std::abs(renderer.measureTextWidth("M", size) - baseWidth * size /
                                 UITheme::Typography::ReferenceSize) < 0.001f, "Measurement scale changed");
                std::fill(pixels.begin(), pixels.end(), 0);
                renderer.pushClipRect({0, 0}, {18, 20});
                renderer.drawText({10, 10}, "M", 0xFFFFFFFF, size);
                renderer.popClipRect();
                for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x)
                    require(pixels[y * width + x] == (x < 18 && y < 20 ? expected[y * width + x] : 0),
                            "Sized glyph escaped clipping");
            }
        }
    }
    std::cout << "Native glyph rasterization, cache reuse/invalidation, measurement and clipping passed\n";
}
