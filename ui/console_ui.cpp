#include "console_ui.hpp"
#include "logger.hpp"
#include <windows.h>
#include <thread>
#include <chrono>
#include <atomic>
#include <filesystem>
#include <algorithm>
#include <memory>
#include "core/window_manager.hpp"
#include "core/ui_sync.hpp"

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
    
    case WM_ERASEBKGND:
        return 1;
    
    case WM_PAINT:
        {
            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(hwnd, &ps);
            
            // Fill background
            HBRUSH brush = CreateSolidBrush(RGB(0x18, 0x18, 0x18));
            FillRect(hdc, &ps.rcPaint, brush);
            DeleteObject(brush);
            
            // Draw title bar
            RECT titleRect = {0, 0, (LONG)g_width, 40};
            HBRUSH titleBrush = CreateSolidBrush(RGB(0x20, 0x20, 0x20));
            FillRect(hdc, &titleRect, titleBrush);
            DeleteObject(titleBrush);
            
            // Draw title text
            SetBkMode(hdc, TRANSPARENT);
            SetTextColor(hdc, RGB(255, 255, 255));
            HFONT hFont = CreateFontW(24, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE,
                                     DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
                                     DEFAULT_QUALITY, DEFAULT_PITCH | FF_DONTCARE, L"Consolas");
            HFONT oldFont = (HFONT)SelectObject(hdc, hFont);
            TextOutW(hdc, 10, 8, L"G.R.I.M Console", 15);
            SelectObject(hdc, oldFont);
            DeleteObject(hFont);
            
            // Draw console history
            hFont = CreateFontW(16, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
                              DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
                              DEFAULT_QUALITY, DEFAULT_PITCH | FF_DONTCARE, L"Consolas");
            oldFont = (HFONT)SelectObject(hdc, hFont);
            
            int y = 50;
            auto& lines = g_history.wrapped();
            int maxLines = ((int)g_height - 100) / 20;
            int start = std::max(0, (int)lines.size() - maxLines);
            for (int i = start; i < (int)lines.size(); ++i) {
                std::wstring wtext(lines[i].text.begin(), lines[i].text.end());
                TextOutW(hdc, 10, y, wtext.c_str(), (int)wtext.length());
                y += 20;
            }
            
            // Draw input bar background
            RECT inputRect = {0, (LONG)g_height - 50, (LONG)g_width, (LONG)g_height};
            HBRUSH inputBrush = CreateSolidBrush(RGB(0x1E, 0x1E, 0x1E));
            FillRect(hdc, &inputRect, inputBrush);
            DeleteObject(inputBrush);
            
            // Draw input text with caret
            std::string displayInput = "> " + g_state.inputBuffer;
            if (g_state.caretVisible) displayInput += "|";
            std::wstring wInput(displayInput.begin(), displayInput.end());
            SetTextColor(hdc, RGB(0, 255, 0));
            TextOutW(hdc, 10, (int)g_height - 40, wInput.c_str(), (int)wInput.length());
            
            SelectObject(hdc, oldFont);
            DeleteObject(hFont);
            EndPaint(hwnd, &ps);
            return 0;
        }

    case WM_CHAR:
        if (wParam == VK_RETURN)
        {
            std::string input = g_state.inputBuffer;
            g_state.inputBuffer.clear();
            if (!input.empty())
            {
                handleCommand(input);
                g_history.push(input, 0xFF00FF00);
            }
            InvalidateRect(hwnd, nullptr, FALSE);
            return 0;
        }
        else if (wParam == VK_BACK)
        {
            if (!g_state.inputBuffer.empty())
                g_state.inputBuffer.pop_back();
            InvalidateRect(hwnd, nullptr, FALSE);
            return 0;
        }
        else if (wParam == VK_ESCAPE)
        {
            // Hide console instead of exiting entire program
            GRIMConsole::hideConsole();
            LOG_DEBUG("ConsoleUI", "ESC pressed - hiding console");
            return 0;
        }
        else if (wParam >= 32 && wParam <= 126)
        {
            g_state.inputBuffer.push_back(static_cast<char>(wParam));
            InvalidateRect(hwnd, nullptr, FALSE);
            return 0;
        }
        break;

    case WM_CLOSE:
        // Hide instead of destroying when user clicks X
        GRIMConsole::hideConsole();
        LOG_DEBUG("ConsoleUI", "WM_CLOSE - hiding console instead of closing");
        return 0;
    }

    return DefWindowProc(hwnd, msg, wParam, lParam);
}

// ===========================================================
// Create Win32 window
// ===========================================================
static HWND createConsoleWindow(int width, int height)
{
    LOG_TRACE("CU", "createConsoleWindow");
    HINSTANCE hInst = GetModuleHandle(nullptr);
    const wchar_t* className = L"GRIMConsoleClass";

    WNDCLASSW wc{};
    wc.lpfnWndProc = ConsoleWndProc;
    wc.hInstance = hInst;
    wc.lpszClassName = className;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = nullptr;

    RegisterClassW(&wc);
    LOG_TRACE("CU", "RegisterClassW");
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
// Console logic loop
// ===========================================================
void GRIMConsole::runConsoleUI(int width, int height)
{
    LOG_TRACE("CU", "runConsoleUI");
    g_width  = width;
    g_height = height;
    g_hwnd   = createConsoleWindow(width, height);
    if (!g_hwnd)
    {
        LOG_ERROR("ConsoleUI", "Failed to create GRIM console window");
        return;
    }

    // Register window with WindowManager
    auto consoleWin = std::make_unique<GRIMWindow>();
    consoleWin->hwnd = g_hwnd;
    consoleWin->name = "console";
    consoleWin->visible = true;
    consoleWin->isOverlay = false;
    consoleWin->width = width;
    consoleWin->height = height;
    WindowManager::registerWindow(std::move(consoleWin));
    LOG_PHASE("Console UI registered with WindowManager", true);

    MSG msg{};
    g_state.running = true;
    auto lastRedraw = std::chrono::steady_clock::now();

    // Main console event loop
    while (g_state.running)
    {
        // Handle input + window messages
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE))
        {
            if (msg.message == WM_QUIT)
            {
                g_state.running = false;
                WindowManager::requestMainLoopStop();
            }

            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

        updateCaretBlink(g_state);

        // Update window state in manager
        static uint32_t lastW = 0;
        static uint32_t lastH = 0;

        WindowManager::setVisibility("console", true);
        if (g_width != lastW || g_height != lastH)
        {
            WindowManager::updateWindowDimensions("console", g_width, g_height);
            lastW = g_width;
            lastH = g_height;
        }

        // Trigger redraw periodically for caret blink
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - lastRedraw).count() > 100) {
            InvalidateRect(g_hwnd, nullptr, FALSE);
            lastRedraw = now;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }

    LOG_PHASE("Console UI shutdown complete", true);
}

// ===========================================================
// Console visibility controls
// ===========================================================
void GRIMConsole::showConsole()
{
    if (g_hwnd)
    {
        ShowWindow(g_hwnd, SW_RESTORE);
        SetForegroundWindow(g_hwnd);
        WindowManager::setVisibility("console", true);
        LOG_DEBUG("ConsoleUI", "Console shown");
    }
}

void GRIMConsole::hideConsole()
{
    if (g_hwnd)
    {
        ShowWindow(g_hwnd, SW_HIDE);
        WindowManager::setVisibility("console", false);
        LOG_DEBUG("ConsoleUI", "Console hidden");
    }
}

void GRIMConsole::toggleConsole()
{
    if (g_hwnd)
    {
        if (IsWindowVisible(g_hwnd))
        {
            hideConsole();
        }
        else
        {
            showConsole();
        }
    }
}

void GRIMConsole::notifyConsoleActivity()
{
    // Placeholder for future activity notifications
    LOG_DEBUG("ConsoleUI", "Console activity notified");
}
