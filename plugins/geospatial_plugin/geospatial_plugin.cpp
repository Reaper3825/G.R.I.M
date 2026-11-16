#include "core/native_render.hpp"
#include "core/plugin_api.hpp"
#include "core/window_manager.hpp"
#include "geospatial/geospatial_runtime.hpp"
#include "ui/primitives/ui_native_3d_viewport_attachment.hpp"
#include "ui/ui_root.hpp"
#include "ui_geospatial_panel.hpp"
#include "plugins/plugin_logging.hpp"

#include <memory>
#include <stdexcept>

namespace {

constexpr auto viewportName = "GeoSpatialViewport";

struct GeoSpatialProvider {
    UIGeoSpatialPanel panel;
    std::shared_ptr<UINative3DViewportAttachment> viewport;
    std::unique_ptr<GRIM::GeoSpatial::GeoSpatialRuntime> runtime;
    bgfx::ViewId viewId = 0;
    bool clearRegistered = false;

    GeoSpatialProvider() { panel.setVisible(false); }
    ~GeoSpatialProvider();

    void startViewport();
    void stopViewport();
    void clearViewport() const;
};

std::unique_ptr<GeoSpatialProvider> provider;

void GeoSpatialProvider::startViewport() {
    viewport = std::make_shared<UINative3DViewportAttachment>(UIRoot::get().getHWND(), viewportName);
    viewId = WindowManager::reserveViewIds(viewportName, WindowManager::ViewIdRange::PanelViewport, 1).at(0);
    runtime = std::make_unique<GRIM::GeoSpatial::GeoSpatialRuntime>(viewId);
    panel.setViewportAttachment(viewport);
    runtime->setViewportAttachment(viewport);
    runtime->setViewportReady(true);
    panel.setController(runtime.get());
    WindowManager::registerRenderPass(viewportName, [](uint32_t) { provider->clearViewport(); });
    clearRegistered = true;
}

GeoSpatialProvider::~GeoSpatialProvider() {
    stopViewport();
}

void GeoSpatialProvider::stopViewport() {
    if (clearRegistered) {
        WindowManager::unregisterRenderPass(viewportName);
        clearRegistered = false;
    }
    if (runtime) {
        runtime->setViewportAttachment({});
        runtime->setViewportReady(false);
        panel.setController(nullptr);
        runtime.reset();
    }
    panel.setViewportAttachment({});
    viewport.reset();
    if (viewId != 0) {
        GRIM::Render::resetView(viewId);
        WindowManager::releaseViewIds(viewportName);
        viewId = 0;
    }
}

void GeoSpatialProvider::clearViewport() const {
    if (!viewport || !viewport->hasFrameBuffer() || !viewport->lastGeometry().visible) return;

    // The clear and Cesium submissions share one view, so the clear stays first
    // even when other panels reserve or release view IDs between reloads.
    GRIM::Render::setViewFrameBuffer(viewId, viewport->frameBufferHandle());
    GRIM::Render::setViewRect(viewId, 0, 0, viewport->frameBufferWidth(), viewport->frameBufferHeight());
    GRIM::Render::setViewClear(viewId, BGFX_CLEAR_COLOR | BGFX_CLEAR_DEPTH, 0x153A4AFF, 1.0f, 0);
    GRIM::Render::touch(viewId);
}

} // namespace

extern "C" PLUGIN_EXPORT const GrimPluginInfo* grim_plugin_get_info() {
    static const GrimPluginInfo info{GRIM_API_VERSION, "GeoSpatial Plugin", "1.0.0", "GRIM Project",
        "Own the native GeoSpatial panel, viewport, and Cesium runtime", GRIM_PERM_UI};
    return &info;
}

extern "C" PLUGIN_EXPORT GrimResult grim_plugin_init(const GrimPluginAPI* api) {
    if (!api || (api->api_version >> 16) != GRIM_API_VERSION_MAJOR ||
        api->api_version < GRIM_API_VERSION || !api->log || !api->create_timer)
        return GRIM_ERROR_API_VERSION_MISMATCH;

    grim::plugin::setLoggingAPI(api);
    try {
        provider = std::make_unique<GeoSpatialProvider>();
        return GRIM_OK;
    } catch (const std::exception& error) {
        LOG_ERROR("GeoSpatial Plugin", error.what());
        return GRIM_ERROR_UNKNOWN;
    }
}

extern "C" PLUGIN_EXPORT GrimResult grim_plugin_activate(const GrimPluginAPI* api) {
    try {
        provider->startViewport();
        if (!api->create_timer(api, 0, true, [](uint64_t, void* context) {
                try {
                    static_cast<GeoSpatialProvider*>(context)->runtime->tick(0.016f);
                } catch (const std::exception& error) {
                    LOG_ERROR("GeoSpatial Plugin", error.what());
                }
            }, provider.get()))
            throw std::runtime_error("Could not schedule the GeoSpatial frame callback");
        return GRIM_OK;
    } catch (const std::exception& error) {
        LOG_ERROR("GeoSpatial Plugin", error.what());
        return GRIM_ERROR_UNKNOWN;
    }
}

extern "C" PLUGIN_EXPORT void grim_plugin_shutdown() {
    // The host has closed the timer gate and detached the panel on this thread.
    provider.reset();
    grim::plugin::setLoggingAPI(nullptr);
}
