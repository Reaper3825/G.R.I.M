#include "popup_renderer.hpp"
#include "logger.hpp"

#include <SFML/Graphics/Image.hpp>
#include <SFML/Graphics/Sprite.hpp>
#include <SFML/Graphics/RenderTexture.hpp>
#include <windows.h>
#include <cstdint>
#include <cstdlib>
#include <chrono>

// ===========================================================
// Globals (cached to compute scaling once)
// ===========================================================
static sf::Vector2u g_windowSize{0, 0};
static sf::Vector2u g_texSize{0, 0};

// ===========================================================
// Load resources (diffuse, opacity, shader)
// ===========================================================
bool loadResources(sf::Texture& diffuse,
                   sf::Texture& opacity,
                   sf::Shader& shader) {
    if (!diffuse.loadFromFile("D:/G.R.I.M/resources/GRIM_Listener_ss_diffuse.png")) {
        LOG_ERROR("PopupRenderer", "Failed to load diffuse texture");
        return false;
    }
    g_texSize = diffuse.getSize();
    LOG_DEBUG("PopupRenderer", "Loaded diffuse texture: " +
              std::to_string(g_texSize.x) + "x" +
              std::to_string(g_texSize.y));

    if (!opacity.loadFromFile("D:/G.R.I.M/resources/GRIM_Listener_ss_opacity.png")) {
        LOG_ERROR("PopupRenderer", "Failed to load opacity texture");
        return false;
    }
    LOG_DEBUG("PopupRenderer", "Loaded opacity texture: " +
              std::to_string(opacity.getSize().x) + "x" +
              std::to_string(opacity.getSize().y));

    if (!shader.loadFromFile("D:/G.R.I.M/resources/popup.vert", "D:/G.R.I.M/resources/popup.frag")) {
        LOG_ERROR("PopupRenderer", "Failed to load shader pair");
        return false;
    }

    shader.setUniform("diffuseMap", diffuse);
    shader.setUniform("opacityMap", opacity);

    LOG_PHASE("PopupRenderer resources loaded", true);
    return true;
}


// ===========================================================
// Update layered window with SFML texture (RGBA -> BGRA swizzle)
// ===========================================================
// New overload: accept an sf::Image directly.
void updateOverlay(HWND hwnd, const sf::Image& img) {
    if (!hwnd) return;

    auto size = img.getSize();

    if (size.x == 0 || size.y == 0) {
        LOG_ERROR("PopupRenderer", "Texture image has invalid size");
        return;
    }

    // Quick alpha/red statistics to help debug fully-transparent outputs.
    const std::uint8_t* all = img.getPixelsPtr();
    std::uint8_t maxA = 0, maxR = 0;
    uint64_t nonZeroA = 0;
    uint64_t totalPixels = static_cast<uint64_t>(size.x) * size.y;
    for (uint64_t i = 0; i < totalPixels; ++i) {
        std::uint8_t r = all[i * 4 + 0];
        std::uint8_t a = all[i * 4 + 3];
        if (a > maxA) maxA = a;
        if (r > maxR) maxR = r;
        if (a != 0) ++nonZeroA;
    }
    float pct = totalPixels ? (static_cast<float>(nonZeroA) * 100.0f / static_cast<float>(totalPixels)) : 0.0f;
    LOG_DEBUG("PopupRenderer", "Image alpha stats: maxA=" + std::to_string(maxA) +
              " maxR=" + std::to_string(maxR) + " nonZeroA=" + std::to_string(nonZeroA) +
              " (" + std::to_string(pct) + "%)");

    g_windowSize = size;

    HDC hdcScreen = GetDC(nullptr);
    HDC hdcMem    = CreateCompatibleDC(hdcScreen);

    BITMAPINFO bmi{};
    bmi.bmiHeader.biSize        = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth       = static_cast<LONG>(size.x);
    bmi.bmiHeader.biHeight      = -static_cast<LONG>(size.y); // top-down
    bmi.bmiHeader.biPlanes      = 1;
    bmi.bmiHeader.biBitCount    = 32;
    bmi.bmiHeader.biCompression = BI_RGB;

    void* bits = nullptr;
    HBITMAP hBitmap = CreateDIBSection(hdcMem, &bmi, DIB_RGB_COLORS, &bits, nullptr, 0);
    if (!hBitmap || !bits) {
        LOG_ERROR("PopupRenderer", "Failed to create DIB section for overlay");
        DeleteDC(hdcMem);
        ReleaseDC(nullptr, hdcScreen);
        return;
    }

    // 🔄 RGBA → BGRA with premultiplied alpha
    // UpdateLayeredWindow (ULW_ALPHA) expects premultiplied alpha in the
    // source bitmap. Convert and premultiply here: c' = (c * a) / 255.
    const std::uint8_t* src = img.getPixelsPtr();
    std::uint8_t* dst       = static_cast<std::uint8_t*>(bits);
    for (unsigned i = 0; i < size.x * size.y; ++i) {
        std::uint8_t a = src[3];
        if (a == 255) {
            // fully opaque: copy colors directly (B,G,R order)
            dst[0] = src[2]; // B
            dst[1] = src[1]; // G
            dst[2] = src[0]; // R
            dst[3] = a;
        } else if (a == 0) {
            // fully transparent
            dst[0] = dst[1] = dst[2] = 0;
            dst[3] = 0;
        } else {
            // premultiply with rounding
            dst[0] = static_cast<std::uint8_t>((src[2] * a + 127) / 255);
            dst[1] = static_cast<std::uint8_t>((src[1] * a + 127) / 255);
            dst[2] = static_cast<std::uint8_t>((src[0] * a + 127) / 255);
            dst[3] = a;
        }
        src += 4;
        dst += 4;
    }

    // No debug magenta fill by default. The DIB now contains the converted
    // premultiplied image data prepared above.


    // Debug: sample center pixel
    unsigned cx = size.x / 2;
    unsigned cy = size.y / 2;
    sf::Color sample = img.getPixel({cx, cy});
    // If the opacity map uses the red channel for opacity, log that value
    // as a float (0..1) to help diagnose transparent output.
    float sampledOpacity = static_cast<float>(sample.r) / 255.0f;
    LOG_DEBUG("PopupRenderer", "Sampled opacity.r=" + std::to_string(sampledOpacity));
    LOG_DEBUG("PopupRenderer", "Center pixel RGBA=" +
              std::to_string(sample.r) + "," +
              std::to_string(sample.g) + "," +
              std::to_string(sample.b) + "," +
              std::to_string(sample.a));

    HGDIOBJ oldBmp = SelectObject(hdcMem, hBitmap);

    SIZE wndSize{static_cast<LONG>(size.x), static_cast<LONG>(size.y)};
    POINT ptSrc{0, 0};
    // Use the current window position so UpdateLayeredWindow doesn't move
    // the window unexpectedly. Query the HWND for its screen coords.
    RECT wr{};
    POINT ptDst{};
    if (GetWindowRect(hwnd, &wr)) {
        ptDst.x = wr.left;
        ptDst.y = wr.top;
    } else {
        ptDst.x = 0;
        ptDst.y = 0;
    }

    BLENDFUNCTION blend{};
    blend.BlendOp             = AC_SRC_OVER;
    blend.SourceConstantAlpha = 255;
    blend.AlphaFormat         = AC_SRC_ALPHA;

    // Ensure window is visible before updating layered content.
    ShowWindow(hwnd, SW_SHOW);
    BOOL result = UpdateLayeredWindow(hwnd, hdcScreen, &ptDst, &wndSize,
                                      hdcMem, &ptSrc, 0, &blend, ULW_ALPHA);
    if (!result) {
        LOG_ERROR("PopupRenderer", "UpdateLayeredWindow failed (err=" +
                   std::to_string(GetLastError()) + ")");
    } else {
        LOG_DEBUG("PopupRenderer", "Overlay updated successfully (" +
                  std::to_string(size.x) + "x" + std::to_string(size.y) + ")");
    }

    // Force the window to remain topmost and ensure it's shown. This helps
    // avoid transient cases where the layered content is updated but the
    // window z-order causes it to be hidden by other windows.
    SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0,
                 SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE);

    // Keep normal z-order handling (no forced foregrounding).

    SelectObject(hdcMem, oldBmp);
    DeleteObject(hBitmap);
    DeleteDC(hdcMem);
    ReleaseDC(nullptr, hdcScreen);
}

// Existing API kept for compatibility: delegate to the image overload.
void updateOverlay(HWND hwnd, const sf::Texture& texture) {
    if (!hwnd) return;
    sf::Image img = texture.copyToImage();
    updateOverlay(hwnd, img);
}


// ===========================================================
// Helper: create sprite scaled to window
// ===========================================================
sf::Sprite createScaledSprite(const sf::Texture& tex, sf::Vector2u windowSize) {
    sf::Sprite sprite(tex);

    auto texSize = tex.getSize();
    // Preserve aspect ratio: scale uniformly so the whole image fits inside
    // the window (letterbox/pillarbox as needed).
    float scaleX = static_cast<float>(windowSize.x) / static_cast<float>(texSize.x);
    float scaleY = static_cast<float>(windowSize.y) / static_cast<float>(texSize.y);
    float uniform = std::min(scaleX, scaleY);

    // Set origin to texture center so scaling keeps it centered, then
    // position sprite at the window center.
    sf::FloatRect bounds = sprite.getLocalBounds();
    sprite.setOrigin(sf::Vector2f(bounds.size.x * 0.5f, bounds.size.y * 0.5f));
    sprite.setScale(sf::Vector2f(uniform, uniform));
    sprite.setPosition(sf::Vector2f(static_cast<float>(windowSize.x) * 0.5f,
                                    static_cast<float>(windowSize.y) * 0.5f));

    LOG_DEBUG("PopupRenderer", "Sprite scaled (uniform): " +
              std::to_string(texSize.x) + "x" + std::to_string(texSize.y) +
              " -> " + std::to_string(windowSize.x) + "x" + std::to_string(windowSize.y) +
              " (scale=" + std::to_string(uniform) + ")");

    return sprite;
}
