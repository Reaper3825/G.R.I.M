#include "window.hpp"
#include "logger.hpp"

#ifdef _WIN32
#include <chrono>
#include <windowsx.h>
#include <algorithm>

Window::Window() = default;
Window::~Window() { destroy(); }

// ======================================================
// Create
// ======================================================
bool Window::create(const WindowInfo& infoIn, HWND parentIn)
{
    info = infoIn;
    parent = parentIn;
    info.pid = GetCurrentProcessId();

    WNDCLASSEXW wc{};
    wc.cbSize = sizeof(WNDCLASSEXW);
    wc.lpfnWndProc = StaticWndProc;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.lpszClassName = std::wstring(info.className.begin(), info.className.end()).c_str();
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);

    if (!RegisterClassExW(&wc))
    {
        LOG_ERROR("Window", "Failed to register class: {}" + info.className);
        return false;
    }

    DWORD style = WS_OVERLAPPEDWINDOW;
    DWORD exStyle = WS_EX_LAYERED;
    if (info.topmost) exStyle |= WS_EX_TOPMOST;
    if (!info.acceptsInput) exStyle |= WS_EX_TRANSPARENT | WS_EX_NOACTIVATE;

    hwnd = CreateWindowExW(
        exStyle,
        std::wstring(info.className.begin(), info.className.end()).c_str(),
        std::wstring(info.name.begin(), info.name.end()).c_str(),
        style,
        static_cast<int>(info.position.x),
        static_cast<int>(info.position.y),
        static_cast<int>(info.size.x),
        static_cast<int>(info.size.y),
        parent,
        nullptr,
        GetModuleHandleW(nullptr),
        this
    );

    if (!hwnd)
    {
        LOG_ERROR("Window", "Failed to create window: {}" + info.name);
        return false;
    }

    SetLayeredWindowAttributes(hwnd, 0, static_cast<BYTE>(info.alpha * 255.0f), LWA_ALPHA);
    ShowWindow(hwnd, SW_HIDE);
    visible = false;
    LOG_DEBUG("Window", "Created window '{}'" + info.name);
    return true;
}

// ======================================================
// Destroy
// ======================================================
void Window::destroy()
{
    if (hwnd)
    {
        DestroyWindow(hwnd);
        hwnd = nullptr;
        visible = false;
        LOG_DEBUG("Window", "Destroyed window '{}'" + info.name);
    }
}

// ======================================================
// Show / Hide
// ======================================================
void Window::show(bool vis)
{
    if (!hwnd) return;
    ShowWindow(hwnd, vis ? SW_SHOW : SW_HIDE);
    visible = vis;
}

// ======================================================
// Setters
// ======================================================
void Window::setTitle(const std::string& title)
{
    if (hwnd) SetWindowTextW(hwnd, std::wstring(title.begin(), title.end()).c_str());
}

void Window::resize(float width, float height)
{
    if (hwnd)
    {
        info.size = { width, height };
        SetWindowPos(hwnd, nullptr, 0, 0, static_cast<int>(width), static_cast<int>(height),
            SWP_NOMOVE | SWP_NOZORDER);
    }
}

void Window::move(float x, float y)
{
    if (hwnd)
    {
        info.position = { x, y };
        SetWindowPos(hwnd, nullptr, static_cast<int>(x), static_cast<int>(y), 0, 0,
            SWP_NOSIZE | SWP_NOZORDER);
    }
}

void Window::setAlpha(float alpha)
{
    info.alpha = std::clamp(alpha, 0.0f, 1.0f);
    if (hwnd)
        SetLayeredWindowAttributes(hwnd, 0, static_cast<BYTE>(info.alpha * 255.0f), LWA_ALPHA);
}

void Window::setLocation(const Vec2& loc)
{
    info.location = loc;
}

void Window::setTopmost(bool enable)
{
    info.topmost = enable;
    if (hwnd)
    {
        SetWindowPos(hwnd, enable ? HWND_TOPMOST : HWND_NOTOPMOST, 0, 0, 0, 0,
            SWP_NOMOVE | SWP_NOSIZE);
    }
}

// ======================================================
// Lifecycle / Hooks
// ======================================================
void Window::onUpdate() {}
void Window::onRender() {}
void Window::onMessage(UINT, WPARAM, LPARAM) {}

// ======================================================
// Message Routing
// ======================================================
LRESULT CALLBACK Window::StaticWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    Window* window = nullptr;

    if (msg == WM_NCCREATE)
    {
        CREATESTRUCTW* cs = reinterpret_cast<CREATESTRUCTW*>(lParam);
        window = reinterpret_cast<Window*>(cs->lpCreateParams);
        SetWindowLongPtrW(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(window));
        window->hwnd = hwnd;
    }
    else
    {
        window = reinterpret_cast<Window*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
    }

    if (window)
        return window->InstanceWndProc(hwnd, msg, wParam, lParam);

    return DefWindowProcW(hwnd, msg, wParam, lParam);
}

LRESULT Window::InstanceWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg)
    {
    case WM_SETFOCUS:
        focused = true;
        break;

    case WM_KILLFOCUS:
        focused = false;
        if (autoHide) show(false);
        break;

    case WM_MOUSEMOVE:
        hover = true;
        mousePos.x = static_cast<float>(GET_X_LPARAM(lParam));
        mousePos.y = static_cast<float>(GET_Y_LPARAM(lParam));
        if (onMouseMove) onMouseMove(mousePos.x, mousePos.y);
        break;

    case WM_LBUTTONDOWN:
        clicked = true;
        if (onClick) onClick();
        break;

    case WM_LBUTTONUP:
        clicked = false;
        break;

    case WM_CLOSE:
        destroy();
        return 0;

    default:
        if (onEvent) onEvent(msg, wParam, lParam);
        onMessage(msg, wParam, lParam);
        break;
    }

    return DefWindowProcW(hwnd, msg, wParam, lParam);
}

#endif // _WIN32
