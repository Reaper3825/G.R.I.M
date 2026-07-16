#include "DigitalAutomationProvider.hpp"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <UIAutomation.h>
#include <wrl/client.h>
#endif

namespace GRIM { namespace Perception { namespace Digital {

namespace {

double ElapsedMs(const std::chrono::steady_clock::time_point& start) {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
}

#ifdef _WIN32

using Microsoft::WRL::ComPtr;

std::string HResultText(HRESULT hr) {
    std::ostringstream out;
    out << "HRESULT 0x" << std::hex << std::uppercase
        << static_cast<unsigned long>(hr);
    return out.str();
}

std::string WideToUtf8(const wchar_t* value, int length = -1) {
    if (!value || (length == 0)) return {};
    if (length < 0) length = static_cast<int>(wcslen(value));
    if (length <= 0) return {};
    const int bytes = WideCharToMultiByte(CP_UTF8, 0, value, length,
                                          nullptr, 0, nullptr, nullptr);
    if (bytes <= 0) return {};
    std::string result(static_cast<std::size_t>(bytes), '\0');
    WideCharToMultiByte(CP_UTF8, 0, value, length,
                        result.data(), bytes, nullptr, nullptr);
    return result;
}

std::string BstrToUtf8(BSTR value) {
    return value ? WideToUtf8(value, static_cast<int>(SysStringLen(value))) : std::string{};
}

std::string WindowTitle(HWND hwnd) {
    const int length = GetWindowTextLengthW(hwnd);
    if (length <= 0) return {};
    std::wstring title(static_cast<std::size_t>(length + 1), L'\0');
    const int copied = GetWindowTextW(hwnd, title.data(), length + 1);
    return copied > 0 ? WideToUtf8(title.data(), copied) : std::string{};
}

const char* ControlTypeName(CONTROLTYPEID type) {
    switch (type) {
        case UIA_ButtonControlTypeId: return "button";
        case UIA_CalendarControlTypeId: return "calendar";
        case UIA_CheckBoxControlTypeId: return "checkbox";
        case UIA_ComboBoxControlTypeId: return "combobox";
        case UIA_EditControlTypeId: return "edit";
        case UIA_HyperlinkControlTypeId: return "link";
        case UIA_ImageControlTypeId: return "image";
        case UIA_ListItemControlTypeId: return "list-item";
        case UIA_ListControlTypeId: return "list";
        case UIA_MenuControlTypeId: return "menu";
        case UIA_MenuBarControlTypeId: return "menu-bar";
        case UIA_MenuItemControlTypeId: return "menu-item";
        case UIA_ProgressBarControlTypeId: return "progress-bar";
        case UIA_RadioButtonControlTypeId: return "radio-button";
        case UIA_ScrollBarControlTypeId: return "scroll-bar";
        case UIA_SliderControlTypeId: return "slider";
        case UIA_SpinnerControlTypeId: return "spinner";
        case UIA_StatusBarControlTypeId: return "status-bar";
        case UIA_TabControlTypeId: return "tab";
        case UIA_TabItemControlTypeId: return "tab-item";
        case UIA_TextControlTypeId: return "text";
        case UIA_ToolBarControlTypeId: return "tool-bar";
        case UIA_ToolTipControlTypeId: return "tool-tip";
        case UIA_TreeControlTypeId: return "tree";
        case UIA_TreeItemControlTypeId: return "tree-item";
        case UIA_CustomControlTypeId: return "custom";
        case UIA_GroupControlTypeId: return "group";
        case UIA_ThumbControlTypeId: return "thumb";
        case UIA_DataGridControlTypeId: return "data-grid";
        case UIA_DataItemControlTypeId: return "data-item";
        case UIA_DocumentControlTypeId: return "document";
        case UIA_SplitButtonControlTypeId: return "split-button";
        case UIA_WindowControlTypeId: return "window";
        case UIA_PaneControlTypeId: return "pane";
        case UIA_HeaderControlTypeId: return "header";
        case UIA_HeaderItemControlTypeId: return "header-item";
        case UIA_TableControlTypeId: return "table";
        case UIA_TitleBarControlTypeId: return "title-bar";
        case UIA_SeparatorControlTypeId: return "separator";
        default: return "unknown";
    }
}

class WindowsUiAutomationProvider final : public DigitalAutomationProvider {
public:
    WindowsUiAutomationProvider() {
        const HRESULT init = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        owns_com_ = SUCCEEDED(init);
        if (FAILED(init) && init != RPC_E_CHANGED_MODE) {
            initialization_error_ = "COM initialization failed: " + HResultText(init);
            return;
        }
        const HRESULT create = CoCreateInstance(CLSID_CUIAutomation, nullptr,
                                                CLSCTX_INPROC_SERVER,
                                                IID_PPV_ARGS(automation_.GetAddressOf()));
        if (FAILED(create)) {
            initialization_error_ = "Windows UI Automation unavailable: " + HResultText(create);
        }
    }

    ~WindowsUiAutomationProvider() override {
        automation_.Reset();
        if (owns_com_) CoUninitialize();
    }

    const char* ProviderName() const noexcept override { return "windows-uia"; }

    DigitalAutomationResult InspectForeground(
        const DigitalCaptureMetadata& source_metadata,
        std::size_t max_elements) override {
        const auto start = std::chrono::steady_clock::now();
        DigitalAutomationResult result;
        result.provider = ProviderName();
        if (!automation_) {
            result.status = DigitalPrimitiveStatus::Unavailable;
            result.error = initialization_error_.empty()
                ? "Windows UI Automation was not initialized" : initialization_error_;
            result.duration_ms = ElapsedMs(start);
            return result;
        }

        const HWND foreground = GetForegroundWindow();
        if (!foreground) {
            result.status = DigitalPrimitiveStatus::Unavailable;
            result.error = "No foreground window is available";
            result.duration_ms = ElapsedMs(start);
            return result;
        }
        result.target_window = WindowTitle(foreground);
        result.target_matches_capture =
            !source_metadata.active_window_title.empty() &&
            !result.target_window.empty() &&
            result.target_window == source_metadata.active_window_title;
        result.target_changed_since_capture = !result.target_matches_capture;

        ComPtr<IUIAutomationElement> root;
        HRESULT hr = automation_->ElementFromHandle(foreground, root.GetAddressOf());
        if (FAILED(hr) || !root) {
            result.status = DigitalPrimitiveStatus::Failed;
            result.error = "ElementFromHandle failed: " + HResultText(hr);
            result.duration_ms = ElapsedMs(start);
            return result;
        }

        ComPtr<IUIAutomationTreeWalker> walker;
        hr = automation_->get_ControlViewWalker(walker.GetAddressOf());
        if (FAILED(hr) || !walker) {
            result.status = DigitalPrimitiveStatus::Failed;
            result.error = "ControlViewWalker unavailable: " + HResultText(hr);
            result.duration_ms = ElapsedMs(start);
            return result;
        }

        const std::size_t limit = std::clamp<std::size_t>(max_elements, 1, 2048);
        const std::size_t scan_limit = limit * 4;
        result.elements.reserve(limit);
        std::vector<ComPtr<IUIAutomationElement>> queue;
        queue.reserve(scan_limit);
        queue.push_back(root);

        for (std::size_t cursor = 0;
             cursor < queue.size() && result.elements.size() < limit;
             ++cursor) {
            const auto& element = queue[cursor];

            ComPtr<IUIAutomationElement> child;
            if (SUCCEEDED(walker->GetFirstChildElement(
                    element.Get(), child.GetAddressOf()))) {
                while (child) {
                    if (queue.size() >= scan_limit) {
                        result.truncated = true;
                        break;
                    }
                    queue.push_back(child);
                    ComPtr<IUIAutomationElement> sibling;
                    if (FAILED(walker->GetNextSiblingElement(
                            child.Get(), sibling.GetAddressOf()))) break;
                    child = std::move(sibling);
                }
            }

            BOOL offscreen = FALSE;
            BOOL enabled = FALSE;
            BOOL password = FALSE;
            CONTROLTYPEID control_type = 0;
            RECT rect{};
            element->get_CurrentIsOffscreen(&offscreen);
            element->get_CurrentIsEnabled(&enabled);
            element->get_CurrentIsPassword(&password);
            element->get_CurrentControlType(&control_type);
            if (FAILED(element->get_CurrentBoundingRectangle(&rect))) continue;
            if (offscreen || rect.right <= rect.left || rect.bottom <= rect.top) continue;

            BSTR raw_name = nullptr;
            BSTR raw_id = nullptr;
            element->get_CurrentName(&raw_name);
            element->get_CurrentAutomationId(&raw_id);

            DigitalUiElement item;
            item.desktop_rect = {rect.left, rect.top,
                                 rect.right - rect.left, rect.bottom - rect.top};
            item.name = password ? "[protected]" : BstrToUtf8(raw_name);
            item.automation_id = BstrToUtf8(raw_id);
            if (item.name.size() > 512) item.name.resize(512);
            if (item.automation_id.size() > 256) item.automation_id.resize(256);
            item.role = ControlTypeName(control_type);
            item.enabled = enabled != FALSE;
            item.offscreen = offscreen != FALSE;
            item.password = password != FALSE;
            if (raw_name) SysFreeString(raw_name);
            if (raw_id) SysFreeString(raw_id);

            if (item.name.empty() && item.automation_id.empty() &&
                item.role == "unknown") continue;
            result.elements.push_back(std::move(item));
        }
        if (result.elements.size() >= limit || queue.size() >= scan_limit) {
            result.truncated = true;
        }

        result.status = DigitalPrimitiveStatus::Ok;
        result.duration_ms = ElapsedMs(start);
        return result;
    }

private:
    bool owns_com_ = false;
    ComPtr<IUIAutomation> automation_;
    std::string initialization_error_;
};

#else

class UnsupportedAutomationProvider final : public DigitalAutomationProvider {
public:
    const char* ProviderName() const noexcept override { return "platform-automation-unavailable"; }
    DigitalAutomationResult InspectForeground(const DigitalCaptureMetadata&,
                                               std::size_t) override {
        DigitalAutomationResult result;
        result.status = DigitalPrimitiveStatus::Unsupported;
        result.provider = ProviderName();
        result.error = "No native automation provider is implemented for this platform";
        return result;
    }
};

#endif

} // namespace

std::unique_ptr<DigitalAutomationProvider> CreatePlatformDigitalAutomationProvider() {
#ifdef _WIN32
    return std::make_unique<WindowsUiAutomationProvider>();
#else
    return std::make_unique<UnsupportedAutomationProvider>();
#endif
}

}}} // namespace GRIM::Perception::Digital
