#include "popup_ui.hpp"
#include "logger.hpp"

#include <string>
#include <vector>
#include <cstdint>

// --- SFML minimal includes ---
#include <SFML/Graphics/RenderTexture.hpp>
#include <SFML/Graphics/Sprite.hpp>
#include <SFML/Graphics/Texture.hpp>
#include <SFML/Graphics/Image.hpp>
#include <SFML/Graphics/Shader.hpp>
#include <SFML/Window/VideoMode.hpp>

// --- Windows includes ---
#include <windows.h>
#include <dwmapi.h>
#pragma comment(lib, "Dwmapi.lib")

// Globals for HWND + HINSTANCE
HINSTANCE g_hInstance = GetModuleHandle(nullptr);
HWND g_hWnd = nullptr;

// Simple Win32 window procedure
LRESULT CALLBACK PopupWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    default:
        return DefWindowProc(hwnd, msg, wParam, lParam);
    }
}

// Helper: create a layered transparent window
HWND createPopupWindow(int width, int height) {
    WNDCLASSW wc{};
    wc.lpfnWndProc   = PopupWndProc;
    wc.hInstance     = g_hInstance;
    wc.lpszClassName = L"GRIM_Popup";
    RegisterClassW(&wc);

    return CreateWindowExW(
        WS_EX_LAYERED | WS_EX_TOPMOST | WS_EX_TOOLWINDOW,
        wc.lpszClassName,
        L"Popup",
        WS_POPUP,
        100, 100, width, height,
        nullptr, nullptr, g_hInstance, nullptr
    );
}

// Apply alpha-blended SFML sprite to Win32 layered window
static void applyAlphaToWindow(HWND hwnd, const sf::Texture& texture) {
    sf::Image img = texture.copyToImage();
    const std::uint8_t* pixels = img.getPixelsPtr();
    if (!pixels) return;

    int w = static_cast<int>(img.getSize().x);
    int h = static_cast<int>(img.getSize().y);

    HDC hdcScreen = GetDC(nullptr);
    HDC hdcMem    = CreateCompatibleDC(hdcScreen);

    BITMAPINFO bmi{};
    bmi.bmiHeader.biSize        = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth       = w;
    bmi.bmiHeader.biHeight      = -h; // top-down
    bmi.bmiHeader.biPlanes      = 1;
    bmi.bmiHeader.biBitCount    = 32;
    bmi.bmiHeader.biCompression = BI_RGB;

    void* bits = nullptr;
    HBITMAP hBitmap = CreateDIBSection(hdcMem, &bmi, DIB_RGB_COLORS, &bits, nullptr, 0);
    memcpy(bits, pixels, w * h * 4);

    HGDIOBJ oldBmp = SelectObject(hdcMem, hBitmap);

    SIZE sizeWnd = { w, h };
    POINT ptSrc  = { 0, 0 };
    POINT ptDst  = { 100, 100 };

    BLENDFUNCTION blend{};
    blend.BlendOp             = AC_SRC_OVER;
    blend.SourceConstantAlpha = 255;
    blend.AlphaFormat         = AC_SRC_ALPHA;

    UpdateLayeredWindow(hwnd, hdcScreen, &ptDst, &sizeWnd,
                        hdcMem, &ptSrc, 0, &blend, ULW_ALPHA);

    SelectObject(hdcMem, oldBmp);
    DeleteObject(hBitmap);
    DeleteDC(hdcMem);
    ReleaseDC(nullptr, hdcScreen);
}

void runPopupUI(int width, int height) {
    // Monitor info
    sf::VideoMode desktop = sf::VideoMode::getDesktopMode();
    LOG_DEBUG("PopupUI", "Monitor size = " +
                         std::to_string(desktop.size.x) + "x" +
                         std::to_string(desktop.size.y));

    // Create Win32 window
    g_hWnd = createPopupWindow(width, height);
    if (!g_hWnd) {
        LOG_ERROR("PopupUI", "Failed to create HWND");
        LOG_PHASE("Popup UI launched", false);
        return;
    }
    LOG_PHASE("Popup UI launched", true);

    // SFML render texture
    sf::RenderTexture rtex({ (unsigned)width, (unsigned)height });

    // Load textures
    sf::Texture texDiffuse, texOpacity;
    if (!texDiffuse.loadFromFile("D:/G.R.I.M/resources/GRIM_Listener_ss_diffuse.png")) {
        LOG_ERROR("PopupUI", "Failed to load diffuse map");
        LOG_PHASE("Diffuse map load", false);
    } else {
        LOG_DEBUG("PopupUI", "Loaded diffuse map");
        LOG_PHASE("Diffuse map load", true);
    }

    if (!texOpacity.loadFromFile("D:/G.R.I.M/resources/GRIM_Listener_ss_opacity.png")) {
        LOG_ERROR("PopupUI", "Failed to load opacity map");
        LOG_PHASE("Opacity map load", false);
    } else {
        LOG_DEBUG("PopupUI", "Loaded opacity map");
        LOG_PHASE("Opacity map load", true);
    }

    // Shader (diffuse + opacity only)
    sf::Shader shader;
    if (!shader.loadFromFile(
        "D:/G.R.I.M/resources/popup.vert",
        "D:/G.R.I.M/resources/popup.frag"))
    {
        LOG_ERROR("PopupUI", "Failed to load shader");
        LOG_PHASE("Shader load", false);
        return;
    }
    LOG_DEBUG("PopupUI", "Shaders loaded");
    LOG_PHASE("Shader load", true);

    shader.setUniform("diffuseMap", texDiffuse);
    shader.setUniform("opacityMap", texOpacity);
    shader.setUniform("debugMode", 0); // 0=final, 1=diffuse, 2=mask

    // Sprite to draw
    sf::Sprite sprite(texDiffuse);
    sprite.setPosition({ 0.f, 0.f });
    sprite.setScale(sf::Vector2f(
        static_cast<float>(width) / texDiffuse.getSize().x,
        static_cast<float>(height) / texDiffuse.getSize().y
    ));

    // Main loop
    MSG msg{};
    while (true) {
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT)
                return;
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

        // Render pass
        rtex.clear(sf::Color::Transparent);
        rtex.draw(sprite, &shader);
        rtex.display();

        // Apply to overlay
        applyAlphaToWindow(g_hWnd, rtex.getTexture());
    }
}
