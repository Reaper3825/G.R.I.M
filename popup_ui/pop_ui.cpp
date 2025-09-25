#include <SFML/Graphics/RenderTexture.hpp>
#include <SFML/Graphics/Sprite.hpp>
#include <SFML/Graphics/Texture.hpp>
#include <SFML/Graphics/Image.hpp>
#include <SFML/Window/VideoMode.hpp>

#include <windows.h>
#include <iostream>
#include <string>
#include <cstdint>

// ---------------- Win32 Overlay ----------------
LRESULT CALLBACK OverlayProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
        default:
            return DefWindowProc(hwnd, msg, wParam, lParam);
    }
}

HWND createOverlayWindow(int width, int height) {
    WNDCLASSW wc{};
    wc.lpfnWndProc   = OverlayProc;
    wc.hInstance     = GetModuleHandle(nullptr);
    wc.lpszClassName = L"SFML3_Overlay";
    RegisterClassW(&wc);

    HWND hwnd = CreateWindowExW(
        WS_EX_LAYERED | WS_EX_TOPMOST | WS_EX_TOOLWINDOW,
        wc.lpszClassName,
        L"Overlay",
        WS_POPUP,
        100, 100, width, height,
        nullptr, nullptr, wc.hInstance, nullptr
    );

    ShowWindow(hwnd, SW_SHOW);
    return hwnd;
}

// ---------------- Main Overlay Logic ----------------
void runPopupUI(int width, int height) {
    HWND hwnd = createOverlayWindow(width, height);

    // Offscreen buffer (SFML 3.x style)
    sf::RenderTexture rtex({static_cast<unsigned>(width),
                            static_cast<unsigned>(height)});

    // Get monitor size
    sf::VideoMode desktop = sf::VideoMode::getDesktopMode();
    unsigned monWidth  = desktop.size.x;
    unsigned monHeight = desktop.size.y;

    std::cout << "[PopupUI] Monitor size = "
              << monWidth << "x" << monHeight << std::endl;

    // ---- Load maps ----
    const std::string opacityPath = "D:/G.R.I.M/resources/GRIM_Listener_ss_opacity.png";
    const std::string diffusePath = "D:/G.R.I.M/resources/GRIM_Listener_ss_diffuse.png";

    sf::Image opacityImg;
    if (!opacityImg.loadFromFile(opacityPath)) {
        std::cerr << "[PopupUI] ERROR: could not load opacity map\n";
        return;
    }

    sf::Image diffuseImg;
    if (!diffuseImg.loadFromFile(diffusePath)) {
        std::cerr << "[PopupUI] ERROR: could not load diffuse map\n";
        return;
    }

    // ---- Apply opacity alpha to diffuse (untouched) ----
    sf::Vector2u size = opacityImg.getSize();
    if (diffuseImg.getSize() != size) {
        std::cerr << "[PopupUI] ERROR: diffuse/opacity size mismatch\n";
        return;
    }

    for (unsigned y = 0; y < size.y; ++y) {
        for (unsigned x = 0; x < size.x; ++x) {
            sf::Color d = diffuseImg.getPixel({x, y});
            sf::Color o = opacityImg.getPixel({x, y});
            diffuseImg.setPixel({x, y}, sf::Color(d.r, d.g, d.b, o.r)); // red→alpha
        }
    }

    sf::Texture texture(diffuseImg);
    texture.setSmooth(true);

    // ---- Sprite scaling + centering ----
    sf::Vector2u texSize = texture.getSize();
    sf::Sprite sprite(texture);

    float scaleX = static_cast<float>(monWidth)  / static_cast<float>(texSize.x);
    float scaleY = static_cast<float>(monHeight) / static_cast<float>(texSize.y);

    sprite.setScale(sf::Vector2f{scaleX, scaleY});
    sprite.setPosition(sf::Vector2f{
        (monWidth  - (texSize.x * scaleX)) / 2.f,
        (monHeight - (texSize.y * scaleY)) / 2.f
    });

    // ---- Message loop ----
    MSG msg{};
    while (true) {
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) return;
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

        // Draw into render texture
        rtex.clear(sf::Color(0,0,0,0));
        rtex.draw(sprite);
        rtex.display();

        // Grab pixels
        sf::Image frame = rtex.getTexture().copyToImage();
        const unsigned char* src = frame.getPixelsPtr();  // <-- FIXED TYPE

        HDC screenDC = GetDC(nullptr);
        HDC memDC    = CreateCompatibleDC(screenDC);

        BITMAPINFO bi{};
        bi.bmiHeader.biSize        = sizeof(BITMAPINFOHEADER);
        bi.bmiHeader.biWidth       = width;
        bi.bmiHeader.biHeight      = -height; // top-down DIB
        bi.bmiHeader.biPlanes      = 1;
        bi.bmiHeader.biBitCount    = 32;
        bi.bmiHeader.biCompression = BI_RGB;

        void* bits = nullptr;
        HBITMAP hBitmap = CreateDIBSection(memDC, &bi, DIB_RGB_COLORS, &bits, nullptr, 0);

        // --- RGBA -> BGRA swizzle (fix yellow tint) ---
        std::uint8_t* dst = static_cast<std::uint8_t*>(bits);
        for (int i = 0; i < width * height; ++i) {
            dst[0] = src[2]; // B
            dst[1] = src[1]; // G
            dst[2] = src[0]; // R
            dst[3] = src[3]; // A
            dst += 4;
            src += 4;
        }

        HBITMAP oldBmp = (HBITMAP)SelectObject(memDC, hBitmap);

        SIZE winSize{width, height};
        POINT srcPos{0,0};
        POINT winPos{100,100};

        BLENDFUNCTION blend{};
        blend.BlendOp             = AC_SRC_OVER;
        blend.SourceConstantAlpha = 255;
        blend.AlphaFormat         = AC_SRC_ALPHA;

        UpdateLayeredWindow(hwnd, screenDC, &winPos, &winSize, memDC, &srcPos, 0, &blend, ULW_ALPHA);

        SelectObject(memDC, oldBmp);
        DeleteObject(hBitmap);
        DeleteDC(memDC);
        ReleaseDC(nullptr, screenDC);

        Sleep(16);
    }
}
