//======================================================//
//  UIStoragePanel — Shared cloud storage file explorer
//
//  Two tabs: My Files | Devices
//
//  My Files: breadcrumb navigation, scrollable file list,
//            upload button, disk usage footer.
//
//  Devices: registered device list with online/offline
//           status, pairing management, "Add Device".
//
//  UI owns layout + events only. All data comes from
//  DeviceCommServer snapshots + StorageIndex queries.
//======================================================//

#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "ui_panel.hpp"
#include "ui_button.hpp"
#include "ui_inputbox.hpp"
#include "ui_scrollbox.hpp"

// Forward declarations — avoid pulling device headers into UI
namespace GRIM {
    class DeviceCommServer;
    struct DeviceSnapshot;
    struct DirectoryEntry;
    struct ActiveTransfer;
}

class OverlayRenderer;
struct InputState;

// ─────────────────────────────────────────────────────────
//  View enum
// ─────────────────────────────────────────────────────────

enum class StorageView : uint8_t {
    MyFiles = 0,
    Devices = 1
};

// ─────────────────────────────────────────────────────────
//  Panel class
// ─────────────────────────────────────────────────────────

class UIStoragePanel : public UIPanel {
public:
    UIStoragePanel();
    ~UIStoragePanel() override;

    // Bind server reference (call once from main.cpp after construction)
    void setServer(GRIM::DeviceCommServer* server);

    void update(const InputState& input, float dt) override;
    bool drawOverlay(OverlayRenderer& renderer) override;

    void        setView(StorageView view);
    StorageView currentView() const { return active_view_; }

private:
    // ═══════════════════════════════════════════════════
    //  View management
    // ═══════════════════════════════════════════════════

    StorageView active_view_ = StorageView::MyFiles;
    GRIM::DeviceCommServer* server_ = nullptr;

    // ═══════════════════════════════════════════════════
    //  Tab buttons (always visible)
    // ═══════════════════════════════════════════════════

    std::shared_ptr<UIButton> tab_files_btn_;
    std::shared_ptr<UIButton> tab_devices_btn_;

    // ═══════════════════════════════════════════════════
    //  My Files tab
    // ═══════════════════════════════════════════════════

    std::vector<std::string> breadcrumb_; // current path segments
    std::shared_ptr<UIScrollBox> file_list_scroll_;
    std::shared_ptr<UIButton>    btn_upload_;
    std::shared_ptr<UIInputBox>  search_box_;

    // Cached directory listing (refreshed on navigation / timer)
    struct FileRow {
        std::string name;
        bool        is_directory = false;
        uint64_t    size_bytes   = 0;
        std::string modified_at;
        std::string file_id; // empty for directories
    };
    std::vector<FileRow> cached_files_;
    float refresh_timer_ = 0.0f;

    void navigateTo(const std::string& subdir);
    void navigateUp();
    void refreshListing();
    std::string currentPath() const;

    // ═══════════════════════════════════════════════════
    //  Devices tab
    // ═══════════════════════════════════════════════════

    std::shared_ptr<UIScrollBox> device_list_scroll_;

    // Local device code display + regenerate
    std::shared_ptr<UIButton>    btn_regenerate_code_;

    // Enter device code UI
    std::shared_ptr<UIInputBox>  device_code_input_;
    std::string                  device_code_buffer_;
    std::shared_ptr<UIButton>    btn_link_device_;
    std::string                  link_status_msg_;    // feedback after submit
    float                        link_status_timer_ = 0.0f;

    struct DeviceRow {
        std::string device_id;
        std::string device_name;
        std::string platform;
        bool        is_online     = false;
        bool        is_paired     = false;
        std::string pairing_code; // shown only while pending
    };
    std::vector<DeviceRow> cached_devices_;

    void refreshDevices();

    // ═══════════════════════════════════════════════════
    //  Transfer progress
    // ═══════════════════════════════════════════════════

    struct TransferProgress {
        std::string transfer_id;
        std::string filename;
        float       progress = 0.0f;
    };
    std::vector<TransferProgress> active_transfers_;
    void refreshTransfers();

    // ═══════════════════════════════════════════════════
    //  Draw helpers
    // ═══════════════════════════════════════════════════

    void drawTabBar(OverlayRenderer& renderer, const PanelRect& content);
    void drawFilesView(OverlayRenderer& renderer, const PanelRect& content);
    void drawDevicesView(OverlayRenderer& renderer, const PanelRect& content);
    void drawBreadcrumbs(OverlayRenderer& renderer, const PanelRect& area);
    void drawTransferOverlay(OverlayRenderer& renderer, const PanelRect& content);
};
