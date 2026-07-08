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
#include <glm/geometric.hpp>
#include <iomanip>
#include <memory>
#include <queue>
#include <stdexcept>
#include <sstream>
#include <string>
#include <thread>
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
    constexpr double kHorizontalFovRadians = 1.0471975511965976;
    constexpr double kVerticalFovRadians = 0.7853981633974483;
    constexpr double kDefaultOrbitPitchRadians = 1.25;
    constexpr double kMinOrbitPitchRadians = 0.18;
    constexpr double kMaxOrbitPitchRadians = 1.48;
    constexpr double kMinOrbitDistanceMeters = 500.0;
    constexpr double kMaxOrbitDistanceMultiplier = 5.0;
    constexpr double kOrbitRadiansPerPixel = 0.0045;
    constexpr double kFineOrbitRadiansPerPixel = 0.0015;
    constexpr double kWheelZoomStep = 0.12;
    constexpr double kDragZoomStep = 0.01;

    struct CameraBasis {
        glm::dvec3 target{0.0, 0.0, 0.0};
        glm::dvec3 east{1.0, 0.0, 0.0};
        glm::dvec3 north{0.0, 1.0, 0.0};
        glm::dvec3 up{0.0, 0.0, 1.0};
    };

    std::string formatEcef(const glm::dvec3& ecef)
    {
        std::ostringstream stream;
        stream << std::fixed << std::setprecision(1)
               << "ECEF(" << ecef.x << ", " << ecef.y << ", " << ecef.z << ")";
        return stream.str();
    }

    double defaultOrbitDistanceMeters(const glm::dvec3& homeEcef)
    {
        const double homeRadius = glm::length(homeEcef);
        if (homeRadius <= 0.0)
            throw std::runtime_error("GeoSpatialRuntime camera home ECEF radius is invalid");

        return homeRadius * 0.35;
    }

    CameraBasis makeCameraBasis(const glm::dvec3& homeEcef, double targetEastMeters, double targetNorthMeters)
    {
        CameraBasis basis;
        basis.up = glm::normalize(homeEcef);

        const glm::dvec3 worldZ{0.0, 0.0, 1.0};
        basis.east = glm::cross(worldZ, basis.up);
        if (glm::length(basis.east) < 1.0e-8)
            basis.east = glm::dvec3{1.0, 0.0, 0.0};
        else
            basis.east = glm::normalize(basis.east);

        basis.north = glm::normalize(glm::cross(basis.up, basis.east));
        basis.target = homeEcef + basis.east * targetEastMeters + basis.north * targetNorthMeters;

        basis.up = glm::normalize(basis.target);
        basis.east = glm::cross(worldZ, basis.up);
        if (glm::length(basis.east) < 1.0e-8)
            basis.east = glm::dvec3{1.0, 0.0, 0.0};
        else
            basis.east = glm::normalize(basis.east);
        basis.north = glm::normalize(glm::cross(basis.up, basis.east));
        return basis;
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

    Cesium3DTilesSelection::ViewState makeHomeViewState(const glm::dvec3& homeEcef,
                                                        double targetEastMeters,
                                                        double targetNorthMeters,
                                                        double yawRadians,
                                                        double pitchRadians,
                                                        double distanceMeters,
                                                        int viewportWidth,
                                                        int viewportHeight)
    {
        if (distanceMeters <= 0.0)
            throw std::runtime_error("GeoSpatialRuntime camera distance must be positive");

        const CameraBasis basis = makeCameraBasis(homeEcef, targetEastMeters, targetNorthMeters);
        const glm::dvec3 tangentDirection = glm::normalize(std::cos(yawRadians) * basis.north +
                                                           std::sin(yawRadians) * basis.east);
        const glm::dvec3 cameraOffsetDirection = glm::normalize(std::cos(pitchRadians) * tangentDirection +
                                                                std::sin(pitchRadians) * basis.up);
        const glm::dvec3 cameraPosition = basis.target + cameraOffsetDirection * distanceMeters;
        const glm::dvec3 direction = glm::normalize(basis.target - cameraPosition);
        glm::dvec3 right = glm::cross(direction, basis.up);
        if (glm::length(right) < 1.0e-8)
            right = basis.east;
        else
            right = glm::normalize(right);
        const glm::dvec3 up = glm::normalize(glm::cross(right, direction));

        const double width = viewportWidth > 0 ? static_cast<double>(viewportWidth) : 1280.0;
        const double height = viewportHeight > 0 ? static_cast<double>(viewportHeight) : 720.0;
        return Cesium3DTilesSelection::ViewState(
            cameraPosition,
            direction,
            up,
            glm::dvec2(width, height),
            kHorizontalFovRadians,
            kVerticalFovRadians,
            CesiumGeospatial::Ellipsoid::WGS84);
    }

    int viewportMouseButtonIndex(PlatformWindow::ViewportMouseButton button)
    {
        switch (button) {
            case PlatformWindow::ViewportMouseButton::Left: return 0;
            case PlatformWindow::ViewportMouseButton::Right: return 1;
            case PlatformWindow::ViewportMouseButton::Middle: return 2;
            default: return -1;
        }
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

    void updateFrame(const glm::dvec3& homeEcef,
                     float dtSeconds,
                     double targetEastMeters,
                     double targetNorthMeters,
                     double yawRadians,
                     double pitchRadians,
                     double distanceMeters,
                     int viewportWidth,
                     int viewportHeight)
    {
        auto mainThreadScope = asyncSystem.enterMainThread();
        asyncSystem.dispatchMainThreadTasks();
        assetAccessor->tick();

        if (!tileset) {
            renderAdapter->setFrameSelection({}, glm::dmat4(1.0), glm::dmat4(1.0));
            return;
        }

        viewStates.clear();
    viewStates.push_back(makeHomeViewState(homeEcef,
                           targetEastMeters,
                           targetNorthMeters,
                           yawRadians,
                           pitchRadians,
                           distanceMeters,
                           viewportWidth,
                           viewportHeight));
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
    homeEcef_ = CesiumGeospatial::Ellipsoid::WGS84.cartographicToCartesian(home);
    resetCameraLocked();
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

    orbitTargetEastMeters_ = 0.0;
    orbitTargetNorthMeters_ = 0.0;
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

        cesium_->updateFrame(homeEcef_,
                             dtSeconds,
                             orbitTargetEastMeters_,
                             orbitTargetNorthMeters_,
                             orbitYawRadians_,
                             orbitPitchRadians_,
                             orbitDistanceMeters_,
                             viewportWidth,
                             viewportHeight);
        updateProviderStatusLocked();
    }
}

void GeoSpatialRuntime::refreshCameraStatusLocked()
{
    std::ostringstream stream;
    stream << "Orbit target " << formatEcef(homeEcef_)
           << " yaw=" << std::fixed << std::setprecision(1) << (orbitYawRadians_ * 57.29577951308232)
           << " pitch=" << (orbitPitchRadians_ * 57.29577951308232)
           << " distance=" << std::setprecision(0) << orbitDistanceMeters_ << "m";
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
                panCameraLocked(deltaX, deltaY, event.modifiers.ctrl);
                snapshot_.cesium_status = "Camera pan";
            } else if (viewportDraggingRight_) {
                zoomCameraLocked(-deltaY * kDragZoomStep / kWheelZoomStep);
                snapshot_.cesium_status = "Camera zoom";
            } else if (viewportDraggingLeft_) {
                orbitCameraLocked(deltaX, deltaY, event.modifiers.ctrl);
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
                snapshot_.cesium_status = "Pick requested at viewport pixel (" +
                    std::to_string(event.x) + ", " + std::to_string(event.y) + ")";
            }
            break;
        }

        case PlatformWindow::ViewportInputEventType::MouseWheel:
            viewportLastX_ = event.x;
            viewportLastY_ = event.y;
            zoomCameraLocked(static_cast<double>(event.wheelDelta) / 120.0);
            snapshot_.cesium_status = "Camera wheel zoom";
            break;

        case PlatformWindow::ViewportInputEventType::KeyDown:
            if (event.keyCode == 'R') {
                resetCameraLocked();
                snapshot_.cesium_status = "Camera reset from viewport keyboard";
            }
#ifdef _WIN32
            else if (event.keyCode == VK_OEM_PLUS || event.keyCode == VK_ADD) {
                zoomCameraLocked(1.0);
                snapshot_.cesium_status = "Camera keyboard zoom in";
            } else if (event.keyCode == VK_OEM_MINUS || event.keyCode == VK_SUBTRACT) {
                zoomCameraLocked(-1.0);
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
    const double panStep = orbitDistanceMeters_ * 0.35 * static_cast<double>(dtSeconds);

    auto isDown = [](int keyCode) { return PlatformInput::isKeyDown(keyCode); };

    if (shift) {
        if (isDown(static_cast<int>(PlatformInput::Key::Left)))
            orbitTargetEastMeters_ -= panStep;
        if (isDown(static_cast<int>(PlatformInput::Key::Right)))
            orbitTargetEastMeters_ += panStep;
        if (isDown(static_cast<int>(PlatformInput::Key::Up)))
            orbitTargetNorthMeters_ += panStep;
        if (isDown(static_cast<int>(PlatformInput::Key::Down)))
            orbitTargetNorthMeters_ -= panStep;
    } else {
        if (isDown(static_cast<int>(PlatformInput::Key::Left)) || isDown('A'))
            orbitYawRadians_ -= rotateStep;
        if (isDown(static_cast<int>(PlatformInput::Key::Right)) || isDown('D'))
            orbitYawRadians_ += rotateStep;
        if (isDown(static_cast<int>(PlatformInput::Key::Up)) || isDown('W'))
            orbitPitchRadians_ += rotateStep;
        if (isDown(static_cast<int>(PlatformInput::Key::Down)) || isDown('S'))
            orbitPitchRadians_ -= rotateStep;
    }

    if (isDown('Q'))
        zoomCameraLocked(-static_cast<double>(dtSeconds) * 8.0);
    if (isDown('E'))
        zoomCameraLocked(static_cast<double>(dtSeconds) * 8.0);
    if (ctrl)
        clampCameraLocked();

    clampCameraLocked();
    refreshCameraStatusLocked();
}

void GeoSpatialRuntime::resetCameraLocked()
{
    orbitYawRadians_ = 0.0;
    orbitPitchRadians_ = kDefaultOrbitPitchRadians;
    orbitDistanceMeters_ = defaultOrbitDistanceMeters(homeEcef_);
    orbitTargetEastMeters_ = 0.0;
    orbitTargetNorthMeters_ = 0.0;
    viewportDragPixels_ = 0.0;
    clampCameraLocked();
    refreshCameraStatusLocked();
}

void GeoSpatialRuntime::clampCameraLocked()
{
    const double maxDistance = defaultOrbitDistanceMeters(homeEcef_) * kMaxOrbitDistanceMultiplier;
    orbitPitchRadians_ = std::clamp(orbitPitchRadians_, kMinOrbitPitchRadians, kMaxOrbitPitchRadians);
    orbitDistanceMeters_ = std::clamp(orbitDistanceMeters_, kMinOrbitDistanceMeters, maxDistance);
}

void GeoSpatialRuntime::orbitCameraLocked(double deltaX, double deltaY, bool fineControl)
{
    const double scale = fineControl ? kFineOrbitRadiansPerPixel : kOrbitRadiansPerPixel;
    orbitYawRadians_ -= deltaX * scale;
    orbitPitchRadians_ -= deltaY * scale;
    clampCameraLocked();
}

void GeoSpatialRuntime::panCameraLocked(double deltaX, double deltaY, bool fineControl)
{
    const double scale = orbitDistanceMeters_ * (fineControl ? 0.0002 : 0.0008);
    orbitTargetEastMeters_ -= deltaX * scale;
    orbitTargetNorthMeters_ += deltaY * scale;
}

void GeoSpatialRuntime::zoomCameraLocked(double wheelSteps)
{
    orbitDistanceMeters_ *= std::exp(-wheelSteps * kWheelZoomStep);
    clampCameraLocked();
}

} // namespace GRIM::GeoSpatial