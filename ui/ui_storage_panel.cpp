#include "ui_storage_panel.hpp"
#include "ui_theme.hpp"
#include "control/devices/server/device_comm_server.hpp"
#include "control/devices/storage/storage_index.hpp"
#include "control/devices/transfer/file_transfer_manager.hpp"
#include "control/devices/registry/device_snapshot.hpp"
#include "overlay_renderer.hpp"

namespace {
    constexpr float kTabBarY     = 35.0f;
    constexpr float kContentTopY = 70.0f;
    constexpr float kTabWidth    = 100.0f;
    constexpr float kRowHeight   = 28.0f;
    constexpr float kRefreshInterval = 2.0f; // seconds
}

// ─── Construction ────────────────────────────────────────

UIStoragePanel::UIStoragePanel()
    : UIPanel("Shared Storage", true)
{
    position = {250, 500};
    size     = {900, 550};
    setVisible(false);
    setBackground(UITheme::Colors::PanelBg);

    // Tab buttons
    tab_files_btn_ = std::make_shared<UIButton>("My Files", [this]() {
        setView(StorageView::MyFiles);
    });
    tab_files_btn_->setSize(kTabWidth, 28.0f);

    tab_devices_btn_ = std::make_shared<UIButton>("Devices", [this]() {
        setView(StorageView::Devices);
    });
    tab_devices_btn_->setSize(kTabWidth, 28.0f);

    // Files tab widgets
    file_list_scroll_ = std::make_shared<UIScrollBox>();
    btn_upload_ = std::make_shared<UIButton>("Upload", []() {});
    btn_upload_->setSize(80.0f, 24.0f);
    search_box_       = std::make_shared<UIInputBox>();
    search_box_->setPlaceholder("Search files...");

    // Devices tab widgets
    device_list_scroll_ = std::make_shared<UIScrollBox>();
    btn_add_device_ = std::make_shared<UIButton>("Add Device", [this]() {
        if (!server_) return;
        pending_pairing_code_ = server_->createPendingDevice();
        refreshDevices();
    });
    btn_add_device_->setSize(110.0f, 28.0f);

    // Enter device code widgets
    device_code_input_ = std::make_shared<UIInputBox>(&device_code_buffer_);
    device_code_input_->setPlaceholder("Enter device code...");
    device_code_input_->setSize(180.0f, 28.0f);

    btn_link_device_ = std::make_shared<UIButton>("Link", [this]() {
        if (!server_ || device_code_buffer_.empty()) return;
        if (server_->addPendingDeviceWithCode(device_code_buffer_)) {
            link_status_msg_ = "Code accepted — waiting for device";
            device_code_buffer_.clear();
            refreshDevices();
        } else {
            link_status_msg_ = "Invalid or already used code";
        }
        link_status_timer_ = 4.0f;
    });
    btn_link_device_->setSize(60.0f, 28.0f);

    // Initial breadcrumb = root
    breadcrumb_.clear();

    // Set initial tab visibility (hides Devices tab widgets)
    setView(StorageView::MyFiles);
}

UIStoragePanel::~UIStoragePanel() = default;

// ─── Server binding ──────────────────────────────────────

void UIStoragePanel::setServer(GRIM::DeviceCommServer* server) {
    server_ = server;
    refreshListing();
    refreshDevices();
}

// ─── View switching ──────────────────────────────────────

void UIStoragePanel::setView(StorageView view) {
    active_view_ = view;

    // Show/hide widget groups
    bool files   = (view == StorageView::MyFiles);
    bool devices = (view == StorageView::Devices);

    file_list_scroll_->setVisible(files);
    btn_upload_->setVisible(files);
    search_box_->setVisible(files);

    device_list_scroll_->setVisible(devices);
    btn_add_device_->setVisible(devices);
    device_code_input_->setVisible(devices);
    btn_link_device_->setVisible(devices);
}

// ─── Update ──────────────────────────────────────────────

void UIStoragePanel::update(const InputState& input, float dt) {
    if (!isVisible()) return;
    UIPanel::update(input, dt);

    // Manually update widgets that are not panel children
    // (DataHub pattern: panel owns widgets but does not addChild them)
    if (tab_files_btn_->isVisible())   tab_files_btn_->update(input, dt);
    if (tab_devices_btn_->isVisible()) tab_devices_btn_->update(input, dt);

    switch (active_view_) {
        case StorageView::MyFiles:
            if (search_box_->isVisible())       search_box_->update(input, dt);
            if (btn_upload_->isVisible())       btn_upload_->update(input, dt);
            if (file_list_scroll_->isVisible()) file_list_scroll_->update(input, dt);
            break;
        case StorageView::Devices:
            if (btn_add_device_->isVisible())      btn_add_device_->update(input, dt);
            if (device_code_input_->isVisible())   device_code_input_->update(input, dt);
            if (btn_link_device_->isVisible())      btn_link_device_->update(input, dt);
            if (device_list_scroll_->isVisible())   device_list_scroll_->update(input, dt);
            if (link_status_timer_ > 0.0f) {
                link_status_timer_ -= dt;
                if (link_status_timer_ <= 0.0f) link_status_msg_.clear();
            }
            break;
    }

    refresh_timer_ += dt;
    if (refresh_timer_ >= kRefreshInterval) {
        refresh_timer_ = 0.0f;
        refreshListing();
        refreshDevices();
        refreshTransfers();
    }
}

// ─── Draw ────────────────────────────────────────────────

bool UIStoragePanel::drawOverlay(OverlayRenderer& renderer) {
    if (!UIPanel::drawOverlay(renderer)) return false;

    drawTabBar(renderer, getContentRect());

    PanelRect content = getContentRect();
    content.origin.y += (kContentTopY - kTabBarY);
    content.size.y   -= (kContentTopY - kTabBarY);

    switch (active_view_) {
        case StorageView::MyFiles: drawFilesView(renderer, content); break;
        case StorageView::Devices: drawDevicesView(renderer, content); break;
    }

    if (!active_transfers_.empty()) {
        drawTransferOverlay(renderer, content);
    }

    renderer.popClipRect();
    return true;
}

// ─── Tab bar ─────────────────────────────────────────────

void UIStoragePanel::drawTabBar(OverlayRenderer& renderer, const PanelRect& content) {
    float tabX = position.x + 10.0f;
    float tabY = position.y + kTabBarY;

    tab_files_btn_->setPosition(tabX, tabY);
    tab_devices_btn_->setPosition(tabX + kTabWidth + 5.0f, tabY);

    tab_files_btn_->drawOverlay(renderer, position);
    tab_devices_btn_->drawOverlay(renderer, position);

    // Active tab underline
    float indicatorX = (active_view_ == StorageView::MyFiles) ? tabX : tabX + kTabWidth + 5.0f;
    renderer.drawRect({indicatorX, tabY + 28.0f}, {kTabWidth, 2.0f}, UITheme::Colors::Primary);
}

// ─── Files view ──────────────────────────────────────────

void UIStoragePanel::drawFilesView(OverlayRenderer& renderer, const PanelRect& content) {
    // Breadcrumbs
    PanelRect breadcrumbArea = {content.origin, {content.size.x, kRowHeight}};
    drawBreadcrumbs(renderer, breadcrumbArea);

    // Search + upload bar
    float barY = content.origin.y + kRowHeight + 4.0f;
    search_box_->setPosition(content.origin.x, barY);
    search_box_->setSize(content.size.x - 90.0f, 24.0f);
    search_box_->drawOverlay(renderer, position);

    btn_upload_->setPosition(content.origin.x + content.size.x - 80.0f, barY);
    btn_upload_->drawOverlay(renderer, position);

    // File list
    float listY = barY + 30.0f;
    float listH = content.size.y - kRowHeight - 34.0f - 24.0f; // breadcrumb + bar + footer
    file_list_scroll_->setPosition(content.origin.x, listY);
    file_list_scroll_->setSize(content.size.x, listH);

    // Header row
    renderer.drawText({content.origin.x + 24.0f, listY}, "Name", UITheme::Colors::TextSecondary);
    renderer.drawText({content.origin.x + content.size.x * 0.55f, listY}, "Size", UITheme::Colors::TextSecondary);
    renderer.drawText({content.origin.x + content.size.x * 0.72f, listY}, "Modified", UITheme::Colors::TextSecondary);

    float rowY = listY + kRowHeight;
    for (size_t i = 0; i < cached_files_.size(); ++i) {
        const auto& file = cached_files_[i];

        // Alternating row background
        if (i % 2 == 0) {
            renderer.drawRect({content.origin.x, rowY}, {content.size.x, kRowHeight},
                              UITheme::Colors::RowEven);
        }

        // Icon indicator
        uint32_t iconColor = file.is_directory ? UITheme::Colors::Primary : UITheme::Colors::TextSecondary;
        std::string icon = file.is_directory ? "[D]" : "[F]";
        renderer.drawText({content.origin.x + 4.0f, rowY + 4.0f}, icon, iconColor);

        // Name
        renderer.drawText({content.origin.x + 24.0f, rowY + 4.0f}, file.name,
                  UITheme::Colors::TextPrimary);

        // Size (skip for directories)
        if (!file.is_directory) {
            std::string sizeStr;
            if (file.size_bytes < 1024) sizeStr = std::to_string(file.size_bytes) + " B";
            else if (file.size_bytes < 1024*1024) sizeStr = std::to_string(file.size_bytes/1024) + " KB";
            else sizeStr = std::to_string(file.size_bytes/(1024*1024)) + " MB";
            renderer.drawText({content.origin.x + content.size.x * 0.55f, rowY + 4.0f}, sizeStr,
                              UITheme::Colors::TextSecondary);
        }

        // Modified date
        if (!file.modified_at.empty()) {
            std::string dateStr = file.modified_at.substr(0, 10); // YYYY-MM-DD
            renderer.drawText({content.origin.x + content.size.x * 0.72f, rowY + 4.0f}, dateStr,
                              UITheme::Colors::TextSecondary);
        }

        rowY += kRowHeight;
    }

    // Footer: disk usage
    float footerY = content.origin.y + content.size.y - 20.0f;
    if (server_) {
        uint64_t usage = server_->storageManager().getDiskUsage();
        std::string usageStr;
        if (usage < 1024*1024) usageStr = std::to_string(usage/1024) + " KB used";
        else usageStr = std::to_string(usage/(1024*1024)) + " MB used";
        renderer.drawText({content.origin.x + 4.0f, footerY}, usageStr,
                  UITheme::Colors::TextSecondary);
    }
}

// ─── Breadcrumbs ─────────────────────────────────────────

void UIStoragePanel::drawBreadcrumbs(OverlayRenderer& renderer, const PanelRect& area) {
    float x = area.origin.x + 4.0f;
    float y = area.origin.y + 4.0f;

    // Root segment
    renderer.drawText({x, y}, "/", UITheme::Colors::Primary);
    x += 12.0f;

    for (size_t i = 0; i < breadcrumb_.size(); ++i) {
        renderer.drawText({x, y}, breadcrumb_[i], UITheme::Colors::TextPrimary);
        x += static_cast<float>(breadcrumb_[i].size()) * 8.0f + 4.0f;

        if (i + 1 < breadcrumb_.size()) {
            renderer.drawText({x, y}, "/", UITheme::Colors::TextSecondary);
            x += 12.0f;
        }
    }
}

// ─── Devices view ────────────────────────────────────────

void UIStoragePanel::drawDevicesView(OverlayRenderer& renderer, const PanelRect& content) {
    // ── Top bar: [Enter code...] [Link]   [Add Device] ──
    float codeInputX = content.origin.x + 4.0f;
    float codeInputY = content.origin.y + 4.0f;
    device_code_input_->setPosition(codeInputX, codeInputY);
    device_code_input_->drawOverlay(renderer, position);

    btn_link_device_->setPosition(codeInputX + 184.0f, codeInputY);
    btn_link_device_->drawOverlay(renderer, position);

    // Link status feedback
    if (!link_status_msg_.empty()) {
        renderer.drawText({codeInputX + 250.0f, codeInputY + 6.0f},
                          link_status_msg_, UITheme::Colors::Warning);
    }

    // Add Device button (right-aligned)
    btn_add_device_->setPosition(content.origin.x + content.size.x - 110.0f, content.origin.y + 4.0f);
    btn_add_device_->drawOverlay(renderer, position);

    // Pending pairing code display
    if (!pending_pairing_code_.empty()) {
        renderer.drawText({content.origin.x + 4.0f, content.origin.y + 8.0f},
                          "Pairing Code: " + pending_pairing_code_,
                          UITheme::Colors::Warning);
    }

    // Device list
    float listY = content.origin.y + 36.0f;

    // Header
    renderer.drawText({content.origin.x + 24.0f, listY}, "Device", UITheme::Colors::TextSecondary);
    renderer.drawText({content.origin.x + content.size.x * 0.4f, listY}, "Platform", UITheme::Colors::TextSecondary);
    renderer.drawText({content.origin.x + content.size.x * 0.65f, listY}, "Status", UITheme::Colors::TextSecondary);

    float rowY = listY + kRowHeight;
    for (size_t i = 0; i < cached_devices_.size(); ++i) {
        const auto& dev = cached_devices_[i];

        if (i % 2 == 0) {
            renderer.drawRect({content.origin.x, rowY}, {content.size.x, kRowHeight},
                              UITheme::Colors::RowEven);
        }

        // Status dot
        uint32_t dotColor;
        if (!dev.is_paired) {
            dotColor = UITheme::Colors::Warning; // yellow — pending
        } else if (dev.is_online) {
            dotColor = UITheme::Colors::Success; // green
        } else {
            dotColor = UITheme::Colors::TextSecondary; // gray
        }
        renderer.drawRect({content.origin.x + 6.0f, rowY + 8.0f}, {10.0f, 10.0f}, dotColor);

        // Name
        renderer.drawText({content.origin.x + 24.0f, rowY + 4.0f}, dev.device_name,
                  UITheme::Colors::TextPrimary);

        // Platform
        renderer.drawText({content.origin.x + content.size.x * 0.4f, rowY + 4.0f}, dev.platform,
                  UITheme::Colors::TextSecondary);

        // Status text
        std::string status;
        if (!dev.is_paired) status = "Pending";
        else if (dev.is_online) status = "Online";
        else status = "Offline";
        renderer.drawText({content.origin.x + content.size.x * 0.65f, rowY + 4.0f}, status,
                  dotColor);

        // Pairing code (if pending)
        if (!dev.is_paired && !dev.pairing_code.empty()) {
            renderer.drawText({content.origin.x + content.size.x * 0.8f, rowY + 4.0f},
                              dev.pairing_code,
                              UITheme::Colors::Warning);
        }

        rowY += kRowHeight;
    }
}

// ─── Transfer overlay ────────────────────────────────────

void UIStoragePanel::drawTransferOverlay(OverlayRenderer& renderer, const PanelRect& content) {
    float barH = 24.0f;
    float y = content.origin.y + content.size.y - barH * static_cast<float>(active_transfers_.size());

    for (const auto& xfer : active_transfers_) {
        // Background
        renderer.drawRect({content.origin.x, y}, {content.size.x, barH},
                          UITheme::Colors::CardSurface);

        // Progress bar fill
        float fillW = content.size.x * xfer.progress;
        renderer.drawRect({content.origin.x, y}, {fillW, barH},
                          UITheme::Colors::Primary & 0x66FFFFFF); // semi-transparent

        // Text
        int pct = static_cast<int>(xfer.progress * 100.0f);
        renderer.drawText({content.origin.x + 8.0f, y + 4.0f},
                  xfer.filename + " " + std::to_string(pct) + "%",
                  UITheme::Colors::TextPrimary);

        y += barH;
    }
}

// ─── Navigation ──────────────────────────────────────────

std::string UIStoragePanel::currentPath() const {
    std::string path;
    for (const auto& seg : breadcrumb_) {
        path += seg + "/";
    }
    return path;
}

void UIStoragePanel::navigateTo(const std::string& subdir) {
    breadcrumb_.push_back(subdir);
    refreshListing();
}

void UIStoragePanel::navigateUp() {
    if (!breadcrumb_.empty()) {
        breadcrumb_.pop_back();
        refreshListing();
    }
}

// ─── Data refresh ────────────────────────────────────────

void UIStoragePanel::refreshListing() {
    cached_files_.clear();
    if (!server_) return;

    auto listing = server_->storageManager().index().listDirectory(currentPath());
    cached_files_.reserve(listing.size());
    for (const auto& entry : listing) {
        FileRow row;
        row.name         = entry.name;
        row.is_directory = entry.is_directory;
        row.size_bytes   = entry.size_bytes;
        row.modified_at  = entry.modified_at;
        row.file_id      = entry.file_id;
        cached_files_.push_back(std::move(row));
    }
}

void UIStoragePanel::refreshDevices() {
    cached_devices_.clear();
    if (!server_) return;

    auto snapshots = server_->listDeviceSnapshots();
    cached_devices_.reserve(snapshots.size());
    for (const auto& snap : snapshots) {
        DeviceRow row;
        row.device_id    = snap.device_id;
        row.device_name  = snap.device_name;
        row.is_online    = snap.is_online;
        row.is_paired    = (snap.pairing_state == decltype(snap.pairing_state)::Paired);
        row.pairing_code = snap.pairing_code;

        // platform to string
        switch (snap.platform) {
            case decltype(snap.platform)::Windows: row.platform = "Windows"; break;
            case decltype(snap.platform)::macOS:   row.platform = "macOS";   break;
            case decltype(snap.platform)::Linux:   row.platform = "Linux";   break;
            case decltype(snap.platform)::iOS:     row.platform = "iOS";     break;
            case decltype(snap.platform)::Android: row.platform = "Android"; break;
        }

        cached_devices_.push_back(std::move(row));
    }
}

void UIStoragePanel::refreshTransfers() {
    active_transfers_.clear();
    if (!server_) return;

    auto transfers = server_->transferManager().listActive();
    active_transfers_.reserve(transfers.size());
    for (const auto& t : transfers) {
        TransferProgress tp;
        tp.transfer_id = t.transfer_id;
        // Extract filename from path
        auto pos = t.relative_path.rfind('/');
        tp.filename = (pos != std::string::npos) ? t.relative_path.substr(pos + 1) : t.relative_path;
        tp.progress = t.progress();
        active_transfers_.push_back(std::move(tp));
    }
}
