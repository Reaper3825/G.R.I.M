#include "console_ui.hpp"
#include "logger.hpp"
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
// Caret blink timer
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
// Console logic loop (no BGFX calls)
// ===========================================================
void GRIMConsole::runConsoleUI(int width, int height)
{
    g_width = width;
    g_height = height;
    g_hwnd = createConsoleWindow(width, height);
    if (!g_hwnd) return;

    LOG_PHASE("Console UI initialized", true);

    // Register console window in WindowManager
    WindowManager::createOverlay("console", width, height, false);

    MSG msg{};
    g_state.running = true;

    while (g_state.running)
    {
        // Process input events
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE))
        {
            if (msg.message == WM_QUIT)
                g_state.running = false;
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

        updateCaretBlink(g_state);

        // Queue console draw state for renderer thread
        GRIMWindow* consoleWin = WindowManager::get("console");
        if (consoleWin)
        {
            consoleWin->visible = true;
            consoleWin->width = g_width;
            consoleWin->height = g_height;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }

    LOG_PHASE("Console UI shutdown complete", true);
}
