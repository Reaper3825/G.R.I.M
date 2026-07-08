#include "ui_native_3d_viewport_clear_pass.hpp"

#include "core/window_manager.hpp"

#include <algorithm>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace {
    constexpr const char* kRenderPassName = "ui_native_3d_viewport_clear_pass";

    struct ClearPassEntry {
        std::string owner;
        std::weak_ptr<UINative3DViewportAttachment> attachment;
        WindowManager::ViewIdBlock viewIds;
        uint32_t clearColorRgba = 0x153A4AFF;
    };

    std::mutex g_clearPassMutex;
    std::vector<ClearPassEntry> g_clearPassEntries;
    bool g_renderPassRegistered = false;

    void registerRenderPassIfNeeded()
    {
        if (g_renderPassRegistered)
            return;

        WindowManager::registerRenderPass(kRenderPassName, &UINative3DViewportClearPass::render, true);
        g_renderPassRegistered = true;
    }

    void unregisterRenderPassIfEmpty()
    {
        if (!g_renderPassRegistered || !g_clearPassEntries.empty())
            return;

        WindowManager::unregisterRenderPass(kRenderPassName);
        g_renderPassRegistered = false;
    }
}

void UINative3DViewportClearPass::registerClearPass(const std::string& owner,
                                                    std::weak_ptr<UINative3DViewportAttachment> attachment,
                                                    uint32_t clearColorRgba)
{
    if (owner.empty())
        throw std::runtime_error("UINative3DViewportClearPass::registerClearPass requires a non-empty owner");
    if (attachment.expired())
        throw std::runtime_error("UINative3DViewportClearPass::registerClearPass requires a live viewport attachment for '" + owner + "'");

    std::lock_guard<std::mutex> lock(g_clearPassMutex);

    for (ClearPassEntry& entry : g_clearPassEntries) {
        if (entry.owner == owner) {
            entry.attachment = std::move(attachment);
            entry.clearColorRgba = clearColorRgba;
            registerRenderPassIfNeeded();
            return;
        }
    }

    WindowManager::ViewIdBlock viewIds = WindowManager::reserveViewIds(owner,
                                                                       WindowManager::ViewIdRange::PanelViewport,
                                                                       1);
    g_clearPassEntries.push_back(ClearPassEntry{owner, std::move(attachment), viewIds, clearColorRgba});
    registerRenderPassIfNeeded();
}

void UINative3DViewportClearPass::unregisterClearPass(const std::string& owner)
{
    if (owner.empty())
        throw std::runtime_error("UINative3DViewportClearPass::unregisterClearPass requires a non-empty owner");

    std::lock_guard<std::mutex> lock(g_clearPassMutex);

    auto it = std::remove_if(g_clearPassEntries.begin(), g_clearPassEntries.end(),
        [&](const ClearPassEntry& entry) {
            if (entry.owner != owner)
                return false;

            WindowManager::releaseViewIds(entry.owner);
            return true;
        });
    g_clearPassEntries.erase(it, g_clearPassEntries.end());
    unregisterRenderPassIfEmpty();
}

void UINative3DViewportClearPass::render(uint32_t)
{
    std::lock_guard<std::mutex> lock(g_clearPassMutex);

    for (const ClearPassEntry& entry : g_clearPassEntries) {
        auto attachment = entry.attachment.lock();
        if (!attachment || !attachment->hasFrameBuffer())
            continue;

        const UI3DViewportGeometry geometry = attachment->lastGeometry();
        if (!geometry.visible || attachment->frameBufferWidth() == 0 || attachment->frameBufferHeight() == 0)
            continue;

        const bgfx::ViewId viewId = entry.viewIds.at(0);
        bgfx::setViewFrameBuffer(viewId, attachment->frameBufferHandle());
        bgfx::setViewRect(viewId, 0, 0,
                          attachment->frameBufferWidth(),
                          attachment->frameBufferHeight());
        bgfx::setViewClear(viewId,
                           BGFX_CLEAR_COLOR | BGFX_CLEAR_DEPTH,
                           entry.clearColorRgba,
                           1.0f,
                           0);
        bgfx::touch(viewId);
    }
}