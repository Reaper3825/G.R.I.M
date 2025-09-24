#include "popup_ui.hpp"
#include <iostream>
#include <SFML/Graphics.hpp>
#include <cstdint>
#include <vector>
#include <chrono>
#include <thread>

#ifdef _WIN32
    #include <windows.h>
    #include <dwmapi.h>
    #pragma comment(lib, "Dwmapi.lib")
#endif

// ============================================================
// Helper: get bottom-right position for window
// ============================================================
static sf::Vector2i getBottomRightPosition(const sf::VideoMode& mode, int frameW, int frameH) {
    int screenW = static_cast<int>(mode.size.x);
    int screenH = static_cast<int>(mode.size.y);
    int x = screenW - frameW;
    int y = screenH - frameH;
    return {x, y};
}

#ifdef _WIN32
static const wchar_t* POPUP_CLASS_NAME = L"GRIM_POPUP_CLASS";

// ============================================================
// Helper: apply per-pixel alpha (UpdateLayeredWindow)
// ============================================================
static void applyAlphaToWindow(HWND hwnd, const sf::Sprite& sprite, sf::Shader& shader, float timeSec, int alpha = 255) {
    sf::IntRect rect = sprite.getTextureRect();
    unsigned w = rect.size.x;
    unsigned h = rect.size.y;

    HDC screenDC = GetDC(nullptr);
    HDC memDC    = CreateCompatibleDC(screenDC);

    sf::RenderTexture rtex({w, h});
    rtex.clear(sf::Color::Transparent);

    sf::Sprite copy(sprite);
    copy.setOrigin(sf::Vector2f(0.f, 0.f));
    copy.setPosition(sf::Vector2f(0.f, 0.f));

    shader.setUniform("time", timeSec);
    rtex.draw(copy, &shader);
    rtex.display();

    sf::Image img = rtex.getTexture().copyToImage();
    const std::uint8_t* pixels = img.getPixelsPtr();

    std::vector<std::uint8_t> bgra(w * h * 4);
    for (size_t i = 0; i < w * h; i++) {
        std::uint8_t r = pixels[i * 4 + 0];
        std::uint8_t g = pixels[i * 4 + 1];
        std::uint8_t b = pixels[i * 4 + 2];
        std::uint8_t a = pixels[i * 4 + 3];

        bgra[i * 4 + 0] = (b * a) / 255;
        bgra[i * 4 + 1] = (g * a) / 255;
        bgra[i * 4 + 2] = (r * a) / 255;
        bgra[i * 4 + 3] = a;
    }

    BITMAPINFO bi = {};
    bi.bmiHeader.biSize        = sizeof(BITMAPINFOHEADER);
    bi.bmiHeader.biWidth       = w;
    bi.bmiHeader.biHeight      = -static_cast<int>(h);
    bi.bmiHeader.biPlanes      = 1;
    bi.bmiHeader.biBitCount    = 32;
    bi.bmiHeader.biCompression = BI_RGB;

    void* bits = nullptr;
    HBITMAP dib = CreateDIBSection(memDC, &bi, DIB_RGB_COLORS, &bits, nullptr, 0);
    if (dib && bits) {
        memcpy(bits, bgra.data(), bgra.size());
        HGDIOBJ oldBitmap = SelectObject(memDC, dib);

        RECT wndRect{};
        GetWindowRect(hwnd, &wndRect);
        POINT ptDest = {wndRect.left, wndRect.top};
        POINT ptSrc  = {0, 0};
        SIZE wndSize = {static_cast<LONG>(w), static_cast<LONG>(h)};
        BLENDFUNCTION bf = {AC_SRC_OVER, 0, (BYTE)alpha, AC_SRC_ALPHA};

        UpdateLayeredWindow(hwnd, screenDC, &ptDest, &wndSize, memDC, &ptSrc, 0, &bf, ULW_ALPHA);

        SelectObject(memDC, oldBitmap);
    }

    if (dib) DeleteObject(dib);
    DeleteDC(memDC);
    ReleaseDC(nullptr, screenDC);
}
#endif

// ============================================================
// Main popup routine with animated shader
// ============================================================
void runPopupUI(int frameW, int frameH) {
    std::cout << "[PopupUI] Starting popup with PBR maps\n";

    sf::ContextSettings settings;
    settings.majorVersion = 3;
    settings.minorVersion = 0;

    sf::Vector2u size(32, 32);
    std::cout << "[PopupUI] Creating dummy context...\n";
    sf::Context ctx(settings, size);
    std::cout << "[PopupUI] Dummy context created\n";

    auto contextSettings = ctx.getSettings();
    std::cout << "[PopupUI] OpenGL context: "
              << contextSettings.majorVersion << "."
              << contextSettings.minorVersion << "\n";

    sf::VideoMode desktop = sf::VideoMode::getDesktopMode();
    std::cout << "[PopupUI] Desktop mode: "
              << desktop.size.x << "x" << desktop.size.y << "\n";

#ifdef _WIN32
    HINSTANCE hInstance = GetModuleHandle(nullptr);
    WNDCLASSW wc = {};
    wc.lpfnWndProc   = DefWindowProcW;
    wc.hInstance     = hInstance;
    wc.lpszClassName = POPUP_CLASS_NAME;
    RegisterClassW(&wc);

    auto pos = getBottomRightPosition(desktop, frameW, frameH);
    std::cout << "[PopupUI] Window position: (" << pos.x << "," << pos.y << ")\n";

    HWND hwnd = CreateWindowExW(
        WS_EX_LAYERED | WS_EX_TOPMOST,
        POPUP_CLASS_NAME, L"GRIM Popup",
        WS_POPUP,
        pos.x, pos.y, frameW, frameH,
        nullptr, nullptr, hInstance, nullptr
    );

    if (!hwnd) {
        std::cerr << "[PopupUI] ERROR: Failed to create Win32 window\n";
        return;
    }

    ShowWindow(hwnd, SW_SHOW);
    UpdateWindow(hwnd);
    std::cout << "[PopupUI] Window created and shown\n";

    // --- Load maps ---
    std::cout << "[PopupUI] Loading textures...\n";
    sf::Texture texDiffuse, texNormal, texHeight, texEmissive, texOpacity, texAO;
    if (!texDiffuse.loadFromFile("D:/G.R.I.M/resources/GRIM_Listener_ss_diffuse.png") ||
        !texNormal.loadFromFile("D:/G.R.I.M/resources/GRIM_Listener_ss_normal.png") ||
        !texHeight.loadFromFile("D:/G.R.I.M/resources/GRIM_Listener_ss_height.png") ||
        !texEmissive.loadFromFile("D:/G.R.I.M/resources/GRIM_Listener_ss_emissive.png") ||
        !texOpacity.loadFromFile("D:/G.R.I.M/resources/GRIM_Listener_ss_opacity.png") ||
        !texAO.loadFromFile("D:/G.R.I.M/resources/GRIM_Listener_ss_AO.png"))
    {
        std::cerr << "[PopupUI] ERROR: Failed to load one or more textures\n";
        return;
    }
    std::cout << "[PopupUI] Textures loaded\n";

    sf::Sprite sprite(texDiffuse);

    // --- Load shader ---
    std::cout << "[PopupUI] Loading shader...\n";
    sf::Shader shader;
    if (!shader.loadFromFile("D:/G.R.I.M/resources/popup.vert",
                             "D:/G.R.I.M/resources/popup.frag"))
    {
        std::cerr << "[PopupUI] ERROR: Failed to load shader pair\n";
        return;
    }
    std::cout << "[PopupUI] Shader loaded\n";

    shader.setUniform("diffuseMap",  texDiffuse);
    shader.setUniform("normalMap",   texNormal);
    shader.setUniform("heightMap",   texHeight);
    shader.setUniform("emissiveMap", texEmissive);
    shader.setUniform("opacityMap",  texOpacity);
    shader.setUniform("aoMap",       texAO);

    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "[PopupUI] Entering animation loop\n";

    // Animation loop
    MSG msg = {};
    int frameCounter = 0;
    while (true) {
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) break;
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        if (GetAsyncKeyState(VK_ESCAPE) & 0x8000) break;

        auto now = std::chrono::high_resolution_clock::now();
        float timeSec = std::chrono::duration<float>(now - start).count();

        std::cout << "[PopupUI] Frame " << frameCounter++ 
                  << " time=" << timeSec << "s\n";

        applyAlphaToWindow(hwnd, sprite, shader, timeSec, 255);

        std::this_thread::sleep_for(std::chrono::milliseconds(33));
    }

    std::cout << "[PopupUI] Exiting loop, cleaning up\n";
    DestroyWindow(hwnd);
    UnregisterClassW(POPUP_CLASS_NAME, hInstance);
#endif
}
