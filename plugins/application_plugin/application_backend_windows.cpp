#ifdef _WIN32
#include "application_catalog.hpp"
#include <windows.h>
#include <shlobj.h>
#include <shobjidl.h>
#include <shellapi.h>
#include <wrl/client.h>
#include <memory>

namespace GRIM::Applications {
namespace {
using Microsoft::WRL::ComPtr;
struct Apartment {
    HRESULT status = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    Apartment() { if (FAILED(status) && status != RPC_E_CHANGED_MODE) throw std::runtime_error("COM initialization failed"); }
    ~Apartment() { if (SUCCEEDED(status)) CoUninitialize(); }
};
struct FreeTask { void operator()(void* p) const { CoTaskMemFree(p); } };
std::string utf8(const wchar_t* value) {
    const auto n = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value, -1, nullptr, 0, nullptr, nullptr);
    if (!n) throw std::runtime_error("Invalid application Unicode metadata");
    std::string result(n, '\0'); WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value, -1, result.data(), n, nullptr, nullptr);
    result.pop_back(); return result;
}
std::wstring wide(const std::string& value) {
    const auto n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.c_str(), -1, nullptr, 0);
    if (!n) throw std::runtime_error("Invalid UTF-8 application target");
    std::wstring result(n, L'\0'); MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.c_str(), -1, result.data(), n);
    result.pop_back(); return result;
}
std::string display(IShellItem* item, SIGDN kind) {
    PWSTR value = nullptr;
    if (FAILED(item->GetDisplayName(kind, &value))) throw std::runtime_error("Application shell metadata unavailable");
    std::unique_ptr<wchar_t, FreeTask> owned(value); return utf8(value);
}
}
Backend nativeBackend() {
    return {"windows-applications-folder", [] {
        Apartment apartment;
        ComPtr<IShellItem> folder;
        if (FAILED(SHGetKnownFolderItem(FOLDERID_AppsFolder, KF_FLAG_DEFAULT, nullptr, IID_PPV_ARGS(&folder))))
            throw std::runtime_error("Windows Applications folder unavailable");
        ComPtr<IEnumShellItems> enumerator;
        if (FAILED(folder->BindToHandler(nullptr, BHID_EnumItems, IID_PPV_ARGS(&enumerator))))
            throw std::runtime_error("Cannot enumerate Windows applications");
        std::vector<Application> apps;
        for (;;) {
            ComPtr<IShellItem> item;
            const auto status = enumerator->Next(1, &item, nullptr);
            if (status == S_FALSE) break;
            if (FAILED(status)) throw std::runtime_error("Windows application enumeration failed");
            const auto target = display(item.Get(), SIGDN_DESKTOPABSOLUTEPARSING);
            apps.push_back({"windows.shell", target, display(item.Get(), SIGDN_NORMALDISPLAY), "shell-item", target, ""});
            if (apps.size() > 10000) throw std::runtime_error("Windows application limit exceeded");
        }
        return apps;
    }};
}
Backend portableBackend() {
    return {"windows-portable-and-shortcuts", [] {
        auto apps = scanPortableDirectories({L"D:\\Program Files",L"D:\\Program Files (x86)",L"D:\\Apps",L"D:\\Games",L"C:\\Program Files",L"C:\\Program Files (x86)"});
        for(const auto folder : {FOLDERID_CommonPrograms,FOLDERID_Programs}) {
            PWSTR raw=nullptr;
            if(FAILED(SHGetKnownFolderPath(folder,0,nullptr,&raw)))throw std::runtime_error("Start Menu folder unavailable");
            std::unique_ptr<wchar_t,FreeTask> owned(raw);
            const std::filesystem::path root(raw);
            if(!std::filesystem::exists(root))continue;
            for(const auto& file:std::filesystem::recursive_directory_iterator(root,std::filesystem::directory_options::skip_permission_denied)) {
                if(!file.is_regular_file())continue;
                auto ext=file.path().extension().wstring();
                std::transform(ext.begin(),ext.end(),ext.begin(),::towlower);
                if(ext!=L".lnk")continue;
                const auto target=utf8(file.path().c_str());
                apps.push_back({"windows.file",target,utf8(file.path().stem().c_str()),"file",target,""});
                if(apps.size()>10000)throw std::runtime_error("Portable/shortcut application limit exceeded");
            }
        }
        return apps;
    }};
}
void launchNative(const Application& app) {
    Apartment apartment;
    const auto target = wide(app.launchTarget);
    SHELLEXECUTEINFOW info{};
    info.cbSize = sizeof(info); info.lpVerb = L"open"; info.nShow = SW_SHOWNORMAL;
    info.fMask = SEE_MASK_FLAG_NO_UI;
    std::unique_ptr<void, FreeTask> pidl;
    if (app.launchKind == "shell-item") {
        PIDLIST_ABSOLUTE raw = nullptr;
        if (FAILED(SHParseDisplayName(target.c_str(), nullptr, &raw, 0, nullptr)))
            throw std::runtime_error("Application registration no longer exists; refresh discovery");
        pidl.reset(raw); info.fMask |= SEE_MASK_IDLIST; info.lpIDList = raw;
    } else if (app.launchKind == "file") info.lpFile = target.c_str();
    else throw std::invalid_argument("Unsupported Windows application launch kind");
    if (!ShellExecuteExW(&info)) throw std::runtime_error("Windows rejected application launch: " + std::to_string(GetLastError()));
}
}
#endif
