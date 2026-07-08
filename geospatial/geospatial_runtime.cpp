#include "geospatial_runtime.hpp"

#ifdef OPAQUE
#undef OPAQUE
#endif

#include "cesium_bgfx_render_adapter.hpp"
#include "core/platform_input.hpp"
#include "ui/primitives/ui_native_3d_viewport_attachment.hpp"

#include <Cesium3DTilesSelection/EllipsoidTilesetLoader.h>
#include <Cesium3DTilesSelection/Tileset.h>
#include <Cesium3DTilesSelection/TilesetExternals.h>
#include <Cesium3DTilesSelection/TilesetLoadFailureDetails.h>
#include <Cesium3DTilesSelection/TilesetOptions.h>
#include <Cesium3DTilesSelection/ViewUpdateResult.h>
#include <Cesium3DTilesSelection/ViewState.h>
#include <CesiumAsync/AsyncSystem.h>
#include <CesiumAsync/CachingAssetAccessor.h>
#include <CesiumAsync/IAssetAccessor.h>
#include <CesiumAsync/IAssetRequest.h>
#include <CesiumAsync/IAssetResponse.h>
#include <CesiumAsync/ITaskProcessor.h>
#include <CesiumAsync/SqliteCache.h>
#include <CesiumCurl/CurlAssetAccessor.h>
#include <CesiumGeospatial/Cartographic.h>
#include <CesiumGeospatial/Ellipsoid.h>
#include <CesiumRasterOverlays/DebugColorizeTilesRasterOverlay.h>
#include <CesiumRasterOverlays/RasterOverlayLoadFailureDetails.h>
#include <CesiumUtility/CreditSystem.h>

#include <spdlog/spdlog.h>

#include <algorithm>
#include <condition_variable>
#include <cmath>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <memory>
#include <optional>
#include <queue>
#include <stdexcept>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifndef GRIM_ROOT_DIR
#error GRIM_ROOT_DIR must be defined by the build system
#endif

namespace GRIM::GeoSpatial {

namespace {
    constexpr int32_t kCesiumWorkerThreads = 4;
    constexpr uint64_t kCesiumAssetCacheItems = 8192;
    constexpr int32_t kCesiumRequestsPerCachePrune = 512;
    constexpr int64_t kCesiumTileCacheBytes = 512LL * 1024LL * 1024LL;
    constexpr int64_t kCesiumRasterSubTileCacheBytes = 64LL * 1024LL * 1024LL;
    constexpr double kDragZoomWheelStepsPerPixel = 0.08333333333333333;
    constexpr double kRadiansToDegrees = 57.29577951308232;

    std::string formatEcef(const glm::dvec3& ecef)
    {
        std::ostringstream stream;
        stream << std::fixed << std::setprecision(1)
               << "ECEF(" << ecef.x << ", " << ecef.y << ", " << ecef.z << ")";
        return stream.str();
    }

    std::string formatCartographic(const CesiumGeospatial::Cartographic& cartographic)
    {
        std::ostringstream stream;
        stream << std::fixed << std::setprecision(5)
               << "lon=" << (cartographic.longitude * kRadiansToDegrees)
               << " lat=" << (cartographic.latitude * kRadiansToDegrees)
               << " h=" << std::setprecision(1) << cartographic.height << "m";
        return stream.str();
    }

    std::filesystem::path cesiumCacheDatabasePath()
    {
        std::filesystem::path cacheDirectory = std::filesystem::path(GRIM_ROOT_DIR) / "cache" / "geospatial";
        std::filesystem::create_directories(cacheDirectory);
        return cacheDirectory / "cesium_asset_cache.sqlite";
    }

    std::string responseStatusSuffix(const CesiumAsync::IAssetRequest* request)
    {
        if (!request)
            return "";

        const CesiumAsync::IAssetResponse* response = request->response();
        if (!response)
            return " url=" + request->url();

        return " url=" + request->url() + " status=" + std::to_string(response->statusCode());
    }

    class QueuedTaskProcessor final : public CesiumAsync::ITaskProcessor {
    public:
        explicit QueuedTaskProcessor(int32_t workerCount)
        {
            if (workerCount <= 0)
                throw std::runtime_error("QueuedTaskProcessor requires at least one worker");

            workers_.reserve(static_cast<size_t>(workerCount));
            for (int32_t workerIndex = 0; workerIndex < workerCount; ++workerIndex)
                workers_.emplace_back([this]() { workerLoop(); });
        }

        ~QueuedTaskProcessor() override
        {
            {
                std::lock_guard lock(mutex_);
                stopping_ = true;
            }
            condition_.notify_all();

            for (std::thread& worker : workers_) {
                if (worker.joinable())
                    worker.join();
            }
        }

        void startTask(std::function<void()> task) override
        {
            if (!task)
                throw std::runtime_error("QueuedTaskProcessor received an empty task");

            {
                std::lock_guard lock(mutex_);
                if (stopping_)
                    throw std::runtime_error("QueuedTaskProcessor cannot enqueue after shutdown has started");
                tasks_.push(std::move(task));
            }
            condition_.notify_one();
        }

    private:
        void workerLoop()
        {
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock lock(mutex_);
                    condition_.wait(lock, [this]() { return stopping_ || !tasks_.empty(); });
                    if (stopping_ && tasks_.empty())
                        return;

                    task = std::move(tasks_.front());
                    tasks_.pop();
                }

                task();
            }
        }

        std::mutex mutex_;
        std::condition_variable condition_;
        std::queue<std::function<void()>> tasks_;
        std::vector<std::thread> workers_;
        bool stopping_ = false;
    };

    Cesium3DTilesSelection::TilesetOptions makeTilesetOptions()
    {
        Cesium3DTilesSelection::TilesetOptions options;
        options.maximumCachedBytes = kCesiumTileCacheBytes;
        options.maximumSimultaneousTileLoads = 24;
        options.mainThreadLoadingTimeLimit = 4.0;
        options.tileCacheUnloadTimeLimit = 1.0;
        options.enableOcclusionCulling = false;
        options.credit = std::string("Cesium Native ellipsoid terrain");
        options.loadErrorCallback = [](const Cesium3DTilesSelection::TilesetLoadFailureDetails& details) {
            throw std::runtime_error("Cesium tileset load failed: " + details.message +
                                     " status=" + std::to_string(details.statusCode));
        };
        return options;
    }

    CesiumRasterOverlays::RasterOverlayOptions makeRasterOverlayOptions()
    {
        CesiumRasterOverlays::RasterOverlayOptions options;
        options.maximumSimultaneousTileLoads = 12;
        options.subTileCacheBytes = kCesiumRasterSubTileCacheBytes;
        options.maximumTextureSize = 1024;
        options.loadErrorCallback = [](const CesiumRasterOverlays::RasterOverlayLoadFailureDetails& details) {
            throw std::runtime_error("Cesium raster overlay load failed: " + details.message +
                                     responseStatusSuffix(details.pRequest.get()));
        };
        return options;
    }

    CesiumCurl::CurlAssetAccessorOptions makeCurlOptions()
    {
        CesiumCurl::CurlAssetAccessorOptions options;
        options.userAgent = "GRIM Cesium Native Runtime/1.0";
        return options;
    }

    Cesium3DTilesSelection::TilesetExternals makeTilesetExternals(
        const CesiumAsync::AsyncSystem& asyncSystem,
        const std::shared_ptr<CesiumAsync::IAssetAccessor>& assetAccessor,
        const std::shared_ptr<CesiumUtility::CreditSystem>& creditSystem,
        const std::shared_ptr<Cesium3DTilesSelection::IPrepareRendererResources>& rendererResources)
    {
        if (!assetAccessor)
            throw std::runtime_error("GeoSpatialRuntime requires a Cesium asset accessor");
        if (!creditSystem)
            throw std::runtime_error("GeoSpatialRuntime requires a Cesium credit system");
        if (!rendererResources)
            throw std::runtime_error("GeoSpatialRuntime requires a Cesium-to-bgfx renderer adapter");

        Cesium3DTilesSelection::TilesetExternals externals{
            assetAccessor,
            rendererResources,
            asyncSystem,
            creditSystem
        };
        externals.pLogger = spdlog::default_logger();
        return externals;
    }

    Cesium3DTilesSelection::ViewState makeCameraViewState(const UICameraFrame& cameraFrame)
    {
        return Cesium3DTilesSelection::ViewState(
            cameraFrame.positionEcef,
            cameraFrame.directionEcef,
            cameraFrame.upEcef,
            glm::dvec2(cameraFrame.frustum.viewportWidth, cameraFrame.frustum.viewportHeight),
            cameraFrame.frustum.horizontalFovRadians,
            cameraFrame.frustum.verticalFovRadians,
            CesiumGeospatial::Ellipsoid::WGS84);
    }
}

struct GeoSpatialRuntime::CesiumRuntimeContext final {
    CesiumRuntimeContext()
        : taskProcessor(std::make_shared<QueuedTaskProcessor>(kCesiumWorkerThreads)),
          asyncSystem(taskProcessor),
          creditSystem(std::make_shared<CesiumUtility::CreditSystem>()),
          networkAssetAccessor(std::make_shared<CesiumCurl::CurlAssetAccessor>(makeCurlOptions())),
          cacheDatabase(std::make_shared<CesiumAsync::SqliteCache>(
              spdlog::default_logger(),
              cesiumCacheDatabasePath().string(),
              kCesiumAssetCacheItems)),
          assetAccessor(std::make_shared<CesiumAsync::CachingAssetAccessor>(
              spdlog::default_logger(),
              networkAssetAccessor,
              cacheDatabase,
              kCesiumRequestsPerCachePrune)),
          renderAdapter(std::make_shared<CesiumBgfxRenderAdapter>("GeoSpatialCesiumRender")),
          externals(makeTilesetExternals(asyncSystem, assetAccessor, creditSystem, renderAdapter)),
          tilesetOptions(makeTilesetOptions()),
          rasterOverlayOptions(makeRasterOverlayOptions())
    {
    }

    ~CesiumRuntimeContext()
    {
        tileset.reset();
        imageryOverlay.reset();
    }

    void createOrReloadTileset()
    {
        imageryOverlay.reset();
        tileset = Cesium3DTilesSelection::EllipsoidTilesetLoader::createTileset(externals, tilesetOptions);
        if (!tileset)
            throw std::runtime_error("Cesium ellipsoid tileset creation returned NULL");

        if (imageryProviderEnabled)
            attachImageryProvider();
    }

    void setImageryProviderEnabled(bool enabled)
    {
        imageryProviderEnabled = enabled;

        if (!tileset)
            return;

        if (enabled) {
            attachImageryProvider();
            return;
        }

        if (imageryOverlay) {
            tileset->getOverlays().remove(imageryOverlay);
            imageryOverlay.reset();
        }
    }

    void updateFrame(const UICameraFrame& cameraFrame, float dtSeconds)
    {
        auto mainThreadScope = asyncSystem.enterMainThread();
        asyncSystem.dispatchMainThreadTasks();
        assetAccessor->tick();

        if (!tileset) {
            renderAdapter->setFrameSelection({}, glm::dmat4(1.0), glm::dmat4(1.0));
            return;
        }

        viewStates.clear();
        viewStates.push_back(makeCameraViewState(cameraFrame));
        const Cesium3DTilesSelection::ViewUpdateResult& updateResult =
            tileset->updateViewGroup(tileset->getDefaultViewGroup(), viewStates, dtSeconds);
        renderAdapter->setFrameSelection(updateResult.tilesToRenderThisFrame,
                                         viewStates.front().getViewMatrix(),
                                         viewStates.front().getProjectionMatrix());
        tileset->loadTiles();

        loadedTileCount = tileset->getNumberOfTilesLoaded();
        loadProgress = tileset->computeLoadProgress();
        loadedBytes = tileset->getTotalDataBytes();
    }

    std::string tilesetStatus() const
    {
        if (!tileset)
            return "No tileset loaded";

        std::ostringstream stream;
        stream << "Ellipsoid terrain tileset | loaded=" << loadedTileCount
               << " progress=" << std::fixed << std::setprecision(0) << loadProgress << "%"
               << " cache=" << (loadedBytes / (1024 * 1024)) << " MiB";
        return stream.str();
    }

    std::shared_ptr<QueuedTaskProcessor> taskProcessor;
    CesiumAsync::AsyncSystem asyncSystem;
    std::shared_ptr<CesiumUtility::CreditSystem> creditSystem;
    std::shared_ptr<CesiumAsync::IAssetAccessor> networkAssetAccessor;
    std::shared_ptr<CesiumAsync::ICacheDatabase> cacheDatabase;
    std::shared_ptr<CesiumAsync::IAssetAccessor> assetAccessor;
    std::shared_ptr<CesiumBgfxRenderAdapter> renderAdapter;
    Cesium3DTilesSelection::TilesetExternals externals;
    Cesium3DTilesSelection::TilesetOptions tilesetOptions;
    CesiumRasterOverlays::RasterOverlayOptions rasterOverlayOptions;
    std::unique_ptr<Cesium3DTilesSelection::Tileset> tileset;
    CesiumUtility::IntrusivePointer<CesiumRasterOverlays::RasterOverlay> imageryOverlay;
    std::vector<Cesium3DTilesSelection::ViewState> viewStates;
    int32_t loadedTileCount = 0;
    float loadProgress = 0.0f;
    int64_t loadedBytes = 0;
    bool imageryProviderEnabled = false;

private:
    void attachImageryProvider()
    {
        if (!tileset)
            throw std::runtime_error("GeoSpatialRuntime cannot attach imagery provider without a tileset");

        if (imageryOverlay)
            return;

        imageryOverlay = new CesiumRasterOverlays::DebugColorizeTilesRasterOverlay(
            "GRIM debug imagery provider",
            rasterOverlayOptions);
        tileset->getOverlays().add(imageryOverlay);
    }
};

GeoSpatialRuntime::GeoSpatialRuntime()
{
    snapshot_.cesium_status = "Cesium Native linked; waiting for initialization";
    snapshot_.active_tileset = "No tileset loaded";
    snapshot_.camera_mode = "Home camera not initialized";
}

GeoSpatialRuntime::~GeoSpatialRuntime() = default;

GeoSpatialPanelSnapshot GeoSpatialRuntime::snapshot() const
{
    std::lock_guard lock(mutex_);
    return snapshot_;
}

void GeoSpatialRuntime::requestInitializeCesium()
{
    std::lock_guard lock(mutex_);

    const CesiumGeospatial::Cartographic home = CesiumGeospatial::Cartographic::fromDegrees(
        -75.59777,
        40.03883,
        24000000.0);
    camera_.setHomeCartographic(home);
    cesium_ = std::make_unique<CesiumRuntimeContext>();
    cesium_->renderAdapter->setViewportAttachment(viewportAttachment_);
    cesium_->createOrReloadTileset();

    initialized_ = true;
    snapshot_.cesium_ready = true;
    snapshot_.terrain_enabled = true;
    snapshot_.imagery_enabled = false;
    snapshot_.cesium_status = "Cesium Native runtime owner initialized";
    updateProviderStatusLocked();
    refreshCameraStatusLocked();
}

void GeoSpatialRuntime::requestReloadTileset()
{
    std::lock_guard lock(mutex_);
    if (!initialized_) {
        setNotInitializedStatusLocked("reload tileset");
        return;
    }

    if (!cesium_)
        throw std::runtime_error("GeoSpatialRuntime initialized without CesiumRuntimeContext");

    cesium_->createOrReloadTileset();
    updateProviderStatusLocked();
    snapshot_.cesium_status = "Cesium tileset lifecycle reloaded";
}

void GeoSpatialRuntime::requestRecenterHome()
{
    std::lock_guard lock(mutex_);
    if (!initialized_) {
        setNotInitializedStatusLocked("recenter home");
        return;
    }

    camera_.recenterHome();
    refreshCameraStatusLocked();
    snapshot_.cesium_status = "Camera recentered to WGS84 home";
}

void GeoSpatialRuntime::requestResetCamera()
{
    std::lock_guard lock(mutex_);
    if (!initialized_) {
        setNotInitializedStatusLocked("reset camera");
        return;
    }

    resetCameraLocked();
    snapshot_.cesium_status = "Camera reset to default globe orbit";
}

void GeoSpatialRuntime::requestToggleTerrain()
{
    std::lock_guard lock(mutex_);
    if (!initialized_) {
        setNotInitializedStatusLocked("toggle terrain");
        return;
    }

    if (!cesium_)
        throw std::runtime_error("GeoSpatialRuntime initialized without CesiumRuntimeContext");

    snapshot_.terrain_enabled = !snapshot_.terrain_enabled;
    if (snapshot_.terrain_enabled) {
        cesium_->createOrReloadTileset();
        snapshot_.cesium_status = "Ellipsoid terrain provider enabled";
    } else {
        cesium_->tileset.reset();
        snapshot_.cesium_status = "Terrain provider disabled; tileset released";
    }
    updateProviderStatusLocked();
}

void GeoSpatialRuntime::requestToggleImagery()
{
    std::lock_guard lock(mutex_);
    if (!initialized_) {
        setNotInitializedStatusLocked("toggle imagery");
        return;
    }

    if (!cesium_)
        throw std::runtime_error("GeoSpatialRuntime initialized without CesiumRuntimeContext");

    snapshot_.imagery_enabled = !snapshot_.imagery_enabled;
    cesium_->setImageryProviderEnabled(snapshot_.imagery_enabled);
    snapshot_.cesium_status = snapshot_.imagery_enabled ? "Debug imagery provider enabled" : "Imagery provider disabled";
    updateProviderStatusLocked();
}

void GeoSpatialRuntime::setViewportReady(bool ready)
{
    std::lock_guard lock(mutex_);
    snapshot_.viewport_ready = ready;
    if (!ready) {
        snapshot_.cesium_status = "Native viewport attachment unavailable";
    }
}

void GeoSpatialRuntime::setViewportAttachment(std::weak_ptr<UINative3DViewportAttachment> attachment)
{
    std::lock_guard lock(mutex_);
    if (auto previousAttachment = viewportAttachment_.lock())
        previousAttachment->setInputCallback({});

    viewportAttachment_ = std::move(attachment);
    if (auto liveAttachment = viewportAttachment_.lock()) {
        liveAttachment->setInputCallback([this](const PlatformWindow::ViewportInputEvent& event) {
            handleViewportInput(event);
        });
    }

    if (cesium_)
        cesium_->renderAdapter->setViewportAttachment(viewportAttachment_);
}

void GeoSpatialRuntime::tick(float dtSeconds)
{
    std::lock_guard lock(mutex_);
    if (dtSeconds < 0.0f)
        throw std::runtime_error("GeoSpatialRuntime::tick requires non-negative dtSeconds");

    elapsedSeconds_ += static_cast<double>(dtSeconds);
    if (initialized_) {
        if (!cesium_)
            throw std::runtime_error("GeoSpatialRuntime initialized without CesiumRuntimeContext");

        applyViewportKeyboardLocked(dtSeconds);

        int viewportWidth = 1280;
        int viewportHeight = 720;
        if (auto attachment = viewportAttachment_.lock()) {
            const UI3DViewportGeometry geometry = attachment->lastGeometry();
            if (geometry.pixelWidth > 0 && geometry.pixelHeight > 0) {
                viewportWidth = geometry.pixelWidth;
                viewportHeight = geometry.pixelHeight;
            }
        }

        camera_.setViewportSize(viewportWidth, viewportHeight);
        cesium_->updateFrame(camera_.frame(), dtSeconds);
        updateProviderStatusLocked();
    }
}

void GeoSpatialRuntime::refreshCameraStatusLocked()
{
    const UICameraFrame cameraFrame = camera_.frame();
    std::ostringstream stream;
    stream << "Orbit target " << formatEcef(cameraFrame.targetEcef)
           << " yaw=" << std::fixed << std::setprecision(1) << (cameraFrame.yawRadians * kRadiansToDegrees)
           << " pitch=" << (cameraFrame.pitchRadians * kRadiansToDegrees)
           << " distance=" << std::setprecision(0) << cameraFrame.orbitDistanceMeters << "m"
           << " altitude=" << cameraFrame.altitudeMeters << "m";
    snapshot_.camera_mode = stream.str();
}

void GeoSpatialRuntime::setNotInitializedStatusLocked(const char* action)
{
    snapshot_.cesium_status = std::string("Initialize Cesium before ") + action;
}

void GeoSpatialRuntime::updateProviderStatusLocked()
{
    if (!cesium_) {
        snapshot_.active_tileset = "No Cesium runtime owner";
        return;
    }

    snapshot_.active_tileset = cesium_->tilesetStatus();
}

void GeoSpatialRuntime::handleViewportInput(const PlatformWindow::ViewportInputEvent& event)
{
    std::lock_guard lock(mutex_);

    viewportInputActive_ = true;
    switch (event.type) {
        case PlatformWindow::ViewportInputEventType::MouseDown:
            viewportLastX_ = event.x;
            viewportLastY_ = event.y;
            viewportDragPixels_ = 0.0;
            if (event.button == PlatformWindow::ViewportMouseButton::Left)
                viewportDraggingLeft_ = true;
            else if (event.button == PlatformWindow::ViewportMouseButton::Right)
                viewportDraggingRight_ = true;
            else if (event.button == PlatformWindow::ViewportMouseButton::Middle)
                viewportDraggingMiddle_ = true;
            snapshot_.cesium_status = "Viewport input active";
            break;

        case PlatformWindow::ViewportInputEventType::MouseMove: {
            const double deltaX = static_cast<double>(event.x - viewportLastX_);
            const double deltaY = static_cast<double>(event.y - viewportLastY_);
            viewportLastX_ = event.x;
            viewportLastY_ = event.y;
            viewportDragPixels_ += std::abs(deltaX) + std::abs(deltaY);

            if (viewportDraggingMiddle_ || (viewportDraggingLeft_ && event.modifiers.shift)) {
                camera_.panByPixels(deltaX, deltaY, event.modifiers.ctrl);
                snapshot_.cesium_status = "Camera pan";
            } else if (viewportDraggingRight_) {
                camera_.zoomBySteps(-deltaY * kDragZoomWheelStepsPerPixel);
                snapshot_.cesium_status = "Camera zoom";
            } else if (viewportDraggingLeft_) {
                camera_.orbitByPixels(deltaX, deltaY, event.modifiers.ctrl);
                snapshot_.cesium_status = "Camera orbit";
            }
            break;
        }

        case PlatformWindow::ViewportInputEventType::MouseUp: {
            const bool wasLeftDragging = viewportDraggingLeft_;
            if (event.button == PlatformWindow::ViewportMouseButton::Left)
                viewportDraggingLeft_ = false;
            else if (event.button == PlatformWindow::ViewportMouseButton::Right)
                viewportDraggingRight_ = false;
            else if (event.button == PlatformWindow::ViewportMouseButton::Middle)
                viewportDraggingMiddle_ = false;

            if (wasLeftDragging && viewportDragPixels_ < 4.0 && !event.modifiers.shift) {
                const std::optional<CesiumGeospatial::Cartographic> picked =
                    camera_.pickWgs84Cartographic(event.x, event.y);
                snapshot_.cesium_status = picked
                    ? "Picked WGS84 " + formatCartographic(*picked)
                    : "Pick ray missed WGS84 at viewport pixel (" + std::to_string(event.x) + ", " + std::to_string(event.y) + ")";
            }
            break;
        }

        case PlatformWindow::ViewportInputEventType::MouseWheel:
            viewportLastX_ = event.x;
            viewportLastY_ = event.y;
            camera_.zoomBySteps(static_cast<double>(event.wheelDelta) / 120.0);
            snapshot_.cesium_status = "Camera wheel zoom";
            break;

        case PlatformWindow::ViewportInputEventType::KeyDown:
            if (event.keyCode == 'R') {
                resetCameraLocked();
                snapshot_.cesium_status = "Camera reset from viewport keyboard";
            }
#ifdef _WIN32
            else if (event.keyCode == VK_OEM_PLUS || event.keyCode == VK_ADD) {
                camera_.zoomBySteps(1.0);
                snapshot_.cesium_status = "Camera keyboard zoom in";
            } else if (event.keyCode == VK_OEM_MINUS || event.keyCode == VK_SUBTRACT) {
                camera_.zoomBySteps(-1.0);
                snapshot_.cesium_status = "Camera keyboard zoom out";
            }
#endif
            break;

        case PlatformWindow::ViewportInputEventType::FocusLost:
            viewportDraggingLeft_ = false;
            viewportDraggingRight_ = false;
            viewportDraggingMiddle_ = false;
            break;

        case PlatformWindow::ViewportInputEventType::KeyUp:
            break;
    }

    refreshCameraStatusLocked();
}

void GeoSpatialRuntime::applyViewportKeyboardLocked(float dtSeconds)
{
    if (!viewportInputActive_)
        return;

    const bool shift = PlatformInput::isKeyDown(static_cast<int>(PlatformInput::Key::Shift));
    const bool ctrl = PlatformInput::isKeyDown(static_cast<int>(PlatformInput::Key::Control));
    const double rotateStep = 1.5 * static_cast<double>(dtSeconds);
    const double panStep = camera_.orbitDistanceMeters() * 0.35 * static_cast<double>(dtSeconds);

    auto isDown = [](int keyCode) { return PlatformInput::isKeyDown(keyCode); };

    if (shift) {
        if (isDown(static_cast<int>(PlatformInput::Key::Left)))
            camera_.panByMeters(-panStep, 0.0);
        if (isDown(static_cast<int>(PlatformInput::Key::Right)))
            camera_.panByMeters(panStep, 0.0);
        if (isDown(static_cast<int>(PlatformInput::Key::Up)))
            camera_.panByMeters(0.0, panStep);
        if (isDown(static_cast<int>(PlatformInput::Key::Down)))
            camera_.panByMeters(0.0, -panStep);
    } else {
        if (isDown(static_cast<int>(PlatformInput::Key::Left)) || isDown('A'))
            camera_.orbitByRadians(-rotateStep, 0.0);
        if (isDown(static_cast<int>(PlatformInput::Key::Right)) || isDown('D'))
            camera_.orbitByRadians(rotateStep, 0.0);
        if (isDown(static_cast<int>(PlatformInput::Key::Up)) || isDown('W'))
            camera_.orbitByRadians(0.0, rotateStep);
        if (isDown(static_cast<int>(PlatformInput::Key::Down)) || isDown('S'))
            camera_.orbitByRadians(0.0, -rotateStep);
    }

    if (isDown('Q'))
        camera_.zoomBySteps(-static_cast<double>(dtSeconds) * 8.0);
    if (isDown('E'))
        camera_.zoomBySteps(static_cast<double>(dtSeconds) * 8.0);
    if (ctrl)
        camera_.clampOrbit();

    camera_.clampOrbit();
    refreshCameraStatusLocked();
}

void GeoSpatialRuntime::resetCameraLocked()
{
    camera_.resetOrbit();
    viewportDragPixels_ = 0.0;
    refreshCameraStatusLocked();
}

} // namespace GRIM::GeoSpatial