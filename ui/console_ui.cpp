#include "console_ui.hpp"
#include "logger.hpp"

#include <bgfx/bgfx.h>
#include <bx/math.h>
#include <windows.h>
#include <thread>
#include <chrono>
#include <atomic>
#include <filesystem>
#include <algorithm>
#include "core/window_manager.hpp"

using namespace GRIMConsole;

// ===========================================================
// Globals
// ===========================================================
ConsoleState GRIMConsole::g_state;
ConsoleHistory GRIMConsole::g_history;
std::vector<Timer> GRIMConsole::g_uiTimers;
nlohmann::json GRIMConsole::g_longTermMemory;

static HWND g_hwnd = nullptr;
static uint32_t g_width = 1280;
static uint32_t g_height = 720;

// ===========================================================
// Simple inline flat-color shaders
// ===========================================================
static const char* vs_color_glsl = R"(
    #version 330 core
    layout(location = 0) in vec3 a_position;
    layout(location = 1) in vec4 a_color0;
    out vec4 v_color;
    void main() {
        gl_Position = vec4(a_position.xy, 0.0, 1.0);
        v_color = a_color0;
    }
)";

static const char* fs_color_glsl = R"(
    #version 330 core
    in vec4 v_color;
    out vec4 fragColor;
    void main() {
        fragColor = v_color;
    }
)";

// ===========================================================
// Compile GLSL into bgfx::ProgramHandle
// ===========================================================
static bgfx::ProgramHandle createColorProgram()
{
    const bgfx::Memory* vsmem = bgfx::copy(vs_color_glsl, (uint32_t)strlen(vs_color_glsl) + 1);
    const bgfx::Memory* fsmem = bgfx::copy(fs_color_glsl, (uint32_t)strlen(fs_color_glsl) + 1);

    bgfx::ShaderHandle vsh = bgfx::createShader(vsmem);
    bgfx::ShaderHandle fsh = bgfx::createShader(fsmem);

    return bgfx::createProgram(vsh, fsh, true);
}

// ===========================================================
// Helper: Draw solid color quad
// ===========================================================
static void drawQuad(float x, float y, float w, float h, uint32_t color, uint16_t viewId)
{
    struct PosColorVertex {
        float x, y, z;
        uint32_t abgr;
    };

    static const uint16_t indices[6] = { 0, 1, 2, 0, 2, 3 };
    PosColorVertex verts[4] = {
        { x,     y,     0.0f, color },
        { x + w, y,     0.0f, color },
        { x + w, y + h, 0.0f, color },
        { x,     y + h, 0.0f, color },
    };

    bgfx::VertexLayout layout;
    layout.begin()
        .add(bgfx::Attrib::Position, 3, bgfx::AttribType::Float)
        .add(bgfx::Attrib::Color0, 4, bgfx::AttribType::Uint8, true)
        .end();

    bgfx::TransientVertexBuffer tvb;
    bgfx::TransientIndexBuffer tib;

    bgfx::allocTransientVertexBuffer(&tvb, 4, layout);
    bgfx::allocTransientIndexBuffer(&tib, 6);

    if (tvb.data == nullptr || tib.data == nullptr)
        return;

    memcpy(tvb.data, verts, sizeof(verts));
    memcpy(tib.data, indices, sizeof(indices));

    static bgfx::ProgramHandle colorProgram = BGFX_INVALID_HANDLE;
    if (!bgfx::isValid(colorProgram))
        colorProgram = createColorProgram();

    bgfx::setVertexBuffer(0, &tvb);
    bgfx::setIndexBuffer(&tib);
    bgfx::setState(BGFX_STATE_WRITE_RGB | BGFX_STATE_WRITE_A);
    bgfx::submit(viewId, colorProgram);
}

// ===========================================================
// Caret blink timer (every 500ms)
// ===========================================================
static void updateCaretBlink(ConsoleState& s)
{
    uint64_t now = GetTickCount64();
    if (now - s.lastCaretToggle > 500)
    {
        s.caretVisible = !s.caretVisible;
        s.lastCaretToggle = now;
    }
}

// ===========================================================
// Window procedure (input + resize)
// ===========================================================
static LRESULT CALLBACK ConsoleWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg)
    {
    case WM_SIZE:
        g_width = LOWORD(lParam);
        g_height = HIWORD(lParam);
        return 0;

    case WM_CHAR:
        if (wParam == VK_RETURN)
        {
            std::string input = g_state.inputBuffer;
            g_state.inputBuffer.clear();
            if (!input.empty())
            {
                handleCommand(input);
                g_history.push(input, sf::Color::Green);
            }
            return 0;
        }
        else if (wParam == VK_BACK)
        {
            if (!g_state.inputBuffer.empty())
                g_state.inputBuffer.pop_back();
            return 0;
        }
        else if (wParam == VK_ESCAPE)
        {
            g_state.running = false;
            PostQuitMessage(0);
            return 0;
        }
        else if (wParam >= 32 && wParam <= 126)
        {
            g_state.inputBuffer.push_back(static_cast<char>(wParam));
            return 0;
        }
        break;

    case WM_CLOSE:
        g_state.running = false;
        PostQuitMessage(0);
        return 0;
    }

    return DefWindowProc(hwnd, msg, wParam, lParam);
}

// ===========================================================
// Create Win32 window
// ===========================================================
static HWND createConsoleWindow(int width, int height)
{
    HINSTANCE hInst = GetModuleHandle(nullptr);
    const wchar_t* className = L"GRIMConsoleClass";

    WNDCLASSW wc{};
    wc.lpfnWndProc = ConsoleWndProc;
    wc.hInstance = hInst;
    wc.lpszClassName = className;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);

    RegisterClassW(&wc);

    HWND hwnd = CreateWindowExW(
        0, className, L"G.R.I.M Console",
        WS_OVERLAPPEDWINDOW | WS_VISIBLE,
        CW_USEDEFAULT, CW_USEDEFAULT,
        width, height,
        nullptr, nullptr, hInst, nullptr);

    if (!hwnd)
    {
        LOG_ERROR("ConsoleUI", "Failed to create console window");
        return nullptr;
    }

    ShowWindow(hwnd, SW_SHOW);
    UpdateWindow(hwnd);
    return hwnd;
}

// ===========================================================
// Console render loop (shared BGFX instance)
// ===========================================================
void GRIMConsole::runConsoleUI(int width, int height)
{
    g_width = width;
    g_height = height;
    g_hwnd = createConsoleWindow(width, height);
    if (!g_hwnd) return;

    LOG_PHASE("Console UI initialized (shared BGFX)", true);

    MSG msg{};
    g_state.running = true;

    while (g_state.running)
    {
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE))
        {
            if (msg.message == WM_QUIT)
                g_state.running = false;
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

        updateCaretBlink(g_state);

        const uint16_t viewId = 0; // console view
        bgfx::setViewClear(viewId, BGFX_CLEAR_COLOR | BGFX_CLEAR_DEPTH, 0xFF121212, 1.0f, 0);
        bgfx::setViewRect(viewId, 0, 0, g_width, g_height);
        bgfx::touch(viewId);

        // Panels
        float titleH = kTitleBarH;
        float inputH = kInputBarH;
        float bodyH = g_height - titleH - inputH;

        drawQuad(0, 0, (float)g_width, titleH, 0xFF1E1A1A, viewId);
        drawQuad(0, g_height - inputH, (float)g_width, inputH, 0xFF231E1E, viewId);
        drawQuad(0, titleH, (float)g_width, bodyH, 0xFF141212, viewId);

        bgfx::dbgTextClear();
        bgfx::dbgTextPrintf(1, 1, 0x0F, "G R I M");

        std::string input = g_state.inputBuffer;
        if (g_state.caretVisible) input.push_back('|');
        bgfx::dbgTextPrintf(1, (int)((g_height / 16) - 2), 0x0F, "> %s", input.c_str());

        auto& lines = g_history.wrapped();
        int maxLines = (int)((g_height / 16) - 6);
        int start = std::max(0, (int)lines.size() - maxLines);
        int y = 3;
        for (int i = start; i < (int)lines.size(); ++i)
            bgfx::dbgTextPrintf(1, y++, 0x07, "%s", lines[i].text.c_str());

        WindowManager::endFrame();
        std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }

    LOG_PHASE("Console UI shutdown complete", true);
}
