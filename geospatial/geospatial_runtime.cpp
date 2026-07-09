#include "geospatial_runtime.hpp"

#ifdef OPAQUE
#undef OPAQUE
#endif

#include "cesium_bgfx_render_adapter.hpp"
#include "core/platform_input.hpp"
#include "logger.hpp"
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
#include <CesiumRasterOverlays/UrlTemplateRasterOverlay.h>
#include <CesiumUtility/CreditSystem.h>

#include <nlohmann/json.hpp>

#include <spdlog/spdlog.h>

#include <algorithm>
#include <condition_variable>
#include <cmath>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <limits>
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

extern nlohmann::json aiConfig;

namespace GRIM::GeoSpatial {

namespace {
    constexpr double kDragZoomWheelStepsPerPixel = 0.08333333333333333;
    constexpr double kRadiansToDegrees = 57.29577951308232;
    constexpr double kDegreesToRadians = 0.017453292519943295;

    struct GeoSpatialRuntimeConfig {
        CesiumGeospatial::Cartographic home{0.0, 0.0, 0.0};
        double initialOrbitDistanceMeters = 0.0;
        UICameraFrustum frustum;
        UICameraZoomLimits zoomLimits;
        std::filesystem::path cacheDatabasePath;
        std::string terrainProvider;
        std::string imageryProvider;
        std::string imageryUrlTemplate;
        std::string imageryCredit;
        std::string userAgent;
        uint32_t imageryMinimumLevel = 0;
        uint32_t imageryMaximumLevel = 0;
        uint32_t imageryTileWidth = 0;
        uint32_t imageryTileHeight = 0;
        std::vector<CesiumAsync::IAssetAccessor::THeader> imageryHeaders;
        int32_t workerThreads = 0;
        uint64_t assetCacheItems = 0;
        int32_t requestsPerCachePrune = 0;
        int64_t tileCacheBytes = 0;
        int64_t rasterSubTileCacheBytes = 0;
        int32_t maximumSimultaneousTileLoads = 0;
        int32_t maximumSimultaneousRasterLoads = 0;
        double maximumScreenSpaceError = 0.0;
        double culledScreenSpaceError = 0.0;
        double maximumRasterScreenSpaceError = 0.0;
        uint32_t loadingDescendantLimit = 0;
        bool forbidHoles = false;
        double mainThreadLoadingTimeLimitSeconds = 0.0;
        double tileCacheUnloadTimeLimitSeconds = 0.0;
        int32_t maximumRasterTextureSize = 0;
        bool terrainEnabledOnStart = false;
        bool imageryEnabledOnStart = false;
    };

    const nlohmann::json& requireObject(const nlohmann::json& object, const char* key)
    {
        if (!object.contains(key) || !object.at(key).is_object())
            throw std::runtime_error(std::string("GeoSpatial config requires object geospatial.") + key);
        return object.at(key);
    }

    std::string requireString(const nlohmann::json& object, const char* key, const char* path)
    {
        if (!object.contains(key) || !object.at(key).is_string())
            throw std::runtime_error(std::string("GeoSpatial config requires string ") + path + "." + key);
        std::string value = object.at(key).get<std::string>();
        if (value.empty())
            throw std::runtime_error(std::string("GeoSpatial config requires non-empty string ") + path + "." + key);
        return value;
    }

    bool requireBool(const nlohmann::json& object, const char* key, const char* path)
    {
        if (!object.contains(key) || !object.at(key).is_boolean())
            throw std::runtime_error(std::string("GeoSpatial config requires boolean ") + path + "." + key);
        return object.at(key).get<bool>();
    }

    int64_t requireInteger(const nlohmann::json& object, const char* key, const char* path)
    {
        if (!object.contains(key) || !object.at(key).is_number_integer())
            throw std::runtime_error(std::string("GeoSpatial config requires integer ") + path + "." + key);
        return object.at(key).get<int64_t>();
    }

    double requireNumber(const nlohmann::json& object, const char* key, const char* path)
    {
        if (!object.contains(key) || !object.at(key).is_number())
            throw std::runtime_error(std::string("GeoSpatial config requires number ") + path + "." + key);
        return object.at(key).get<double>();
    }

    int32_t requirePositiveInt32(const nlohmann::json& object, const char* key, const char* path)
    {
        const int64_t value = requireInteger(object, key, path);
        if (value <= 0 || value > static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
            throw std::runtime_error(std::string("GeoSpatial config integer out of range ") + path + "." + key);
        return static_cast<int32_t>(value);
    }

    int64_t requirePositiveMegabytesAsBytes(const nlohmann::json& object, const char* key, const char* path)
    {
        const int64_t value = requireInteger(object, key, path);
        if (value <= 0 || value > 32768)
            throw std::runtime_error(std::string("GeoSpatial config megabytes out of range ") + path + "." + key);
        return value * 1024LL * 1024LL;
    }

    uint32_t requireUInt32(const nlohmann::json& object, const char* key, const char* path)
    {
        const int64_t value = requireInteger(object, key, path);
        if (value < 0 || value > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()))
            throw std::runtime_error(std::string("GeoSpatial config unsigned integer out of range ") + path + "." + key);
        return static_cast<uint32_t>(value);
    }

    std::vector<CesiumAsync::IAssetAccessor::THeader> readHeaders(const nlohmann::json& object, const char* key, const char* path)
    {
        std::vector<CesiumAsync::IAssetAccessor::THeader> headers;
        if (!object.contains(key))
            return headers;
        if (!object.at(key).is_object())
            throw std::runtime_error(std::string("GeoSpatial config requires object ") + path + "." + key);

        for (const auto& [headerName, headerValue] : object.at(key).items()) {
            if (!headerValue.is_string())
                throw std::runtime_error(std::string("GeoSpatial config imagery header must be string: ") + headerName);
            const std::string value = headerValue.get<std::string>();
            if (headerName.empty() || value.empty())
                throw std::runtime_error("GeoSpatial config imagery headers require non-empty names and values");
            headers.emplace_back(headerName, value);
        }
        return headers;
    }

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

    std::filesystem::path resolveConfigPath(const std::string& value)
    {
        std::filesystem::path path(value);
        if (path.is_relative())
            path = std::filesystem::path(GRIM_ROOT_DIR) / path;

        if (!path.has_filename())
            throw std::runtime_error("GeoSpatial config cache.database_path must include a file name");

        std::filesystem::create_directories(path.parent_path());
        return path;
    }

    GeoSpatialRuntimeConfig loadGeospatialRuntimeConfig()
    {
        if (!aiConfig.contains("geospatial") || !aiConfig.at("geospatial").is_object())
            throw std::runtime_error("GeoSpatial config requires top-level geospatial object in ai_config.json");

        const nlohmann::json& root = aiConfig.at("geospatial");
        const nlohmann::json& home = requireObject(root, "home");
        const nlohmann::json& terrain = requireObject(root, "terrain");
        const nlohmann::json& imagery = requireObject(root, "imagery");
        const nlohmann::json& cache = requireObject(root, "cache");
        const nlohmann::json& loading = requireObject(root, "loading");
        const nlohmann::json& camera = requireObject(root, "camera");

        const double longitude = requireNumber(home, "longitude_degrees", "geospatial.home");
        const double latitude = requireNumber(home, "latitude_degrees", "geospatial.home");
        const double height = requireNumber(home, "height_meters", "geospatial.home");
        if (longitude < -180.0 || longitude > 180.0)
            throw std::runtime_error("GeoSpatial config longitude_degrees must be within [-180, 180]");
        if (latitude < -90.0 || latitude > 90.0)
            throw std::runtime_error("GeoSpatial config latitude_degrees must be within [-90, 90]");
        if (height <= 0.0)
            throw std::runtime_error("GeoSpatial config height_meters must be positive because it is the initial orbit distance");

        GeoSpatialRuntimeConfig config;
        config.home = CesiumGeospatial::Cartographic::fromDegrees(longitude, latitude, 0.0);
        config.initialOrbitDistanceMeters = height;
        config.terrainEnabledOnStart = requireBool(terrain, "enabled_on_start", "geospatial.terrain");
        config.terrainProvider = requireString(terrain, "provider", "geospatial.terrain");
        if (config.terrainProvider != "ellipsoid")
            throw std::runtime_error("GeoSpatial config terrain.provider must be 'ellipsoid' until a terrain adapter is implemented");

        config.imageryEnabledOnStart = requireBool(imagery, "enabled_on_start", "geospatial.imagery");
        config.imageryProvider = requireString(imagery, "provider", "geospatial.imagery");
        if (config.imageryProvider != "debug_colorize" && config.imageryProvider != "url_template" && config.imageryProvider != "none")
            throw std::runtime_error("GeoSpatial config imagery.provider must be 'debug_colorize', 'url_template', or 'none'");
        if (config.imageryEnabledOnStart && config.imageryProvider == "none")
            throw std::runtime_error("GeoSpatial config cannot enable imagery_on_start with imagery.provider='none'");
        if (config.imageryEnabledOnStart && !config.terrainEnabledOnStart)
            throw std::runtime_error("GeoSpatial config cannot enable imagery without terrain tileset on start");

        if (config.imageryProvider == "url_template") {
            const nlohmann::json& urlTemplate = requireObject(imagery, "url_template");
            config.imageryUrlTemplate = requireString(urlTemplate, "url", "geospatial.imagery.url_template");
            config.imageryCredit = requireString(urlTemplate, "credit", "geospatial.imagery.url_template");
            config.imageryMinimumLevel = requireUInt32(urlTemplate, "minimum_level", "geospatial.imagery.url_template");
            config.imageryMaximumLevel = requireUInt32(urlTemplate, "maximum_level", "geospatial.imagery.url_template");
            config.imageryTileWidth = requireUInt32(urlTemplate, "tile_width", "geospatial.imagery.url_template");
            config.imageryTileHeight = requireUInt32(urlTemplate, "tile_height", "geospatial.imagery.url_template");
            config.imageryHeaders = readHeaders(urlTemplate, "headers", "geospatial.imagery.url_template");
            if (config.imageryMaximumLevel < config.imageryMinimumLevel)
                throw std::runtime_error("GeoSpatial config imagery url_template maximum_level must be >= minimum_level");
            if (config.imageryTileWidth == 0 || config.imageryTileHeight == 0)
                throw std::runtime_error("GeoSpatial config imagery url_template tile dimensions must be positive");
        }

        config.cacheDatabasePath = resolveConfigPath(requireString(cache, "database_path", "geospatial.cache"));
        config.assetCacheItems = static_cast<uint64_t>(requirePositiveInt32(cache, "asset_cache_items", "geospatial.cache"));
        config.requestsPerCachePrune = requirePositiveInt32(cache, "requests_per_prune", "geospatial.cache");
        config.tileCacheBytes = requirePositiveMegabytesAsBytes(cache, "tile_cache_mb", "geospatial.cache");
        config.rasterSubTileCacheBytes = requirePositiveMegabytesAsBytes(cache, "raster_subtile_cache_mb", "geospatial.cache");

        config.workerThreads = requirePositiveInt32(loading, "worker_threads", "geospatial.loading");
        config.maximumSimultaneousTileLoads = requirePositiveInt32(loading, "maximum_simultaneous_tile_loads", "geospatial.loading");
        config.maximumSimultaneousRasterLoads = requirePositiveInt32(loading, "maximum_simultaneous_raster_loads", "geospatial.loading");
        config.maximumScreenSpaceError = requireNumber(loading, "maximum_screen_space_error", "geospatial.loading");
        config.culledScreenSpaceError = requireNumber(loading, "culled_screen_space_error", "geospatial.loading");
        config.maximumRasterScreenSpaceError = requireNumber(loading, "maximum_raster_screen_space_error", "geospatial.loading");
        config.loadingDescendantLimit = requireUInt32(loading, "loading_descendant_limit", "geospatial.loading");
        config.forbidHoles = requireBool(loading, "forbid_holes", "geospatial.loading");
        config.mainThreadLoadingTimeLimitSeconds = requireNumber(loading, "main_thread_loading_time_limit_seconds", "geospatial.loading");
        config.tileCacheUnloadTimeLimitSeconds = requireNumber(loading, "tile_cache_unload_time_limit_seconds", "geospatial.loading");
        config.maximumRasterTextureSize = requirePositiveInt32(loading, "maximum_raster_texture_size", "geospatial.loading");
        config.userAgent = requireString(loading, "user_agent", "geospatial.loading");
        if (config.maximumScreenSpaceError <= 0.0)
            throw std::runtime_error("GeoSpatial config maximum_screen_space_error must be positive");
        if (config.culledScreenSpaceError <= 0.0)
            throw std::runtime_error("GeoSpatial config culled_screen_space_error must be positive");
        if (config.maximumRasterScreenSpaceError <= 0.0)
            throw std::runtime_error("GeoSpatial config maximum_raster_screen_space_error must be positive");
        if (config.mainThreadLoadingTimeLimitSeconds <= 0.0)
            throw std::runtime_error("GeoSpatial config main_thread_loading_time_limit_seconds must be positive");
        if (config.tileCacheUnloadTimeLimitSeconds <= 0.0)
            throw std::runtime_error("GeoSpatial config tile_cache_unload_time_limit_seconds must be positive");

        config.frustum.horizontalFovRadians = requireNumber(camera, "horizontal_fov_degrees", "geospatial.camera") * kDegreesToRadians;
        config.frustum.verticalFovRadians = requireNumber(camera, "vertical_fov_degrees", "geospatial.camera") * kDegreesToRadians;
        config.frustum.nearPlaneMeters = requireNumber(camera, "near_plane_meters", "geospatial.camera");
        config.frustum.farPlaneMeters = requireNumber(camera, "far_plane_meters", "geospatial.camera");
        config.zoomLimits.minDistanceMeters = requireNumber(camera, "min_distance_meters", "geospatial.camera");
        config.zoomLimits.maxDistanceMeters = requireNumber(camera, "max_distance_meters", "geospatial.camera");
        if (config.frustum.horizontalFovRadians <= 0.0 || config.frustum.verticalFovRadians <= 0.0)
            throw std::runtime_error("GeoSpatial config camera FOV values must be positive");
        if (config.initialOrbitDistanceMeters < config.zoomLimits.minDistanceMeters ||
            config.initialOrbitDistanceMeters > config.zoomLimits.maxDistanceMeters) {
            throw std::runtime_error("GeoSpatial config home.height_meters initial orbit distance must be inside camera min/max distance");
        }

        return config;
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

    Cesium3DTilesSelection::TilesetOptions makeTilesetOptions(const GeoSpatialRuntimeConfig& config)
    {
        Cesium3DTilesSelection::TilesetOptions options;
        options.maximumScreenSpaceError = config.maximumScreenSpaceError;
        options.culledScreenSpaceError = config.culledScreenSpaceError;
        options.loadingDescendantLimit = config.loadingDescendantLimit;
        options.forbidHoles = config.forbidHoles;
        options.maximumCachedBytes = config.tileCacheBytes;
        options.maximumSimultaneousTileLoads = config.maximumSimultaneousTileLoads;
        options.mainThreadLoadingTimeLimit = config.mainThreadLoadingTimeLimitSeconds;
        options.tileCacheUnloadTimeLimit = config.tileCacheUnloadTimeLimitSeconds;
        options.enableOcclusionCulling = false;
        options.credit = std::string("Cesium Native ") + config.terrainProvider + " terrain";
        options.loadErrorCallback = [](const Cesium3DTilesSelection::TilesetLoadFailureDetails& details) {
            throw std::runtime_error("Cesium tileset load failed: " + details.message +
                                     " status=" + std::to_string(details.statusCode));
        };
        return options;
    }

    CesiumRasterOverlays::RasterOverlayOptions makeRasterOverlayOptions(const GeoSpatialRuntimeConfig& config)
    {
        CesiumRasterOverlays::RasterOverlayOptions options;
        options.maximumSimultaneousTileLoads = config.maximumSimultaneousRasterLoads;
        options.subTileCacheBytes = config.rasterSubTileCacheBytes;
        options.maximumTextureSize = config.maximumRasterTextureSize;
        options.maximumScreenSpaceError = config.maximumRasterScreenSpaceError;
        options.loadErrorCallback = [](const CesiumRasterOverlays::RasterOverlayLoadFailureDetails& details) {
            throw std::runtime_error("Cesium raster overlay load failed: " + details.message +
                                     responseStatusSuffix(details.pRequest.get()));
        };
        return options;
    }

    CesiumRasterOverlays::UrlTemplateRasterOverlayOptions makeUrlTemplateImageryOptions(const GeoSpatialRuntimeConfig& config)
    {
        CesiumRasterOverlays::UrlTemplateRasterOverlayOptions options;
        options.credit = config.imageryCredit;
        options.minimumLevel = config.imageryMinimumLevel;
        options.maximumLevel = config.imageryMaximumLevel;
        options.tileWidth = config.imageryTileWidth;
        options.tileHeight = config.imageryTileHeight;
        return options;
    }

    CesiumCurl::CurlAssetAccessorOptions makeCurlOptions(const GeoSpatialRuntimeConfig& config)
    {
        CesiumCurl::CurlAssetAccessorOptions options;
        options.userAgent = config.userAgent;
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
    explicit CesiumRuntimeContext(GeoSpatialRuntimeConfig runtimeConfig)
        : config(std::move(runtimeConfig)),
          taskProcessor(std::make_shared<QueuedTaskProcessor>(config.workerThreads)),
          asyncSystem(taskProcessor),
          creditSystem(std::make_shared<CesiumUtility::CreditSystem>()),
          networkAssetAccessor(std::make_shared<CesiumCurl::CurlAssetAccessor>(makeCurlOptions(config))),
          cacheDatabase(std::make_shared<CesiumAsync::SqliteCache>(
              spdlog::default_logger(),
              config.cacheDatabasePath.string(),
              config.assetCacheItems)),
          assetAccessor(std::make_shared<CesiumAsync::CachingAssetAccessor>(
              spdlog::default_logger(),
              networkAssetAccessor,
              cacheDatabase,
              config.requestsPerCachePrune)),
          renderAdapter(std::make_shared<CesiumBgfxRenderAdapter>("GeoSpatialCesiumRender")),
          externals(makeTilesetExternals(asyncSystem, assetAccessor, creditSystem, renderAdapter)),
          tilesetOptions(makeTilesetOptions(config)),
          rasterOverlayOptions(makeRasterOverlayOptions(config))
    {
    }

    ~CesiumRuntimeContext()
    {
        tileset.reset();
        imageryOverlay.reset();
    }

    void createOrReloadTileset()
    {
        if (config.terrainProvider != "ellipsoid")
            throw std::runtime_error("GeoSpatialRuntime cannot create unsupported terrain provider '" + config.terrainProvider + "'");

        imageryOverlay.reset();
        tileset = Cesium3DTilesSelection::EllipsoidTilesetLoader::createTileset(externals, tilesetOptions);
        if (!tileset)
            throw std::runtime_error("Cesium ellipsoid tileset creation returned NULL");

        if (imageryProviderEnabled)
            attachImageryProvider();
    }

    void setImageryProviderEnabled(bool enabled)
    {
        if (enabled && config.imageryProvider == "none")
            throw std::runtime_error("GeoSpatialRuntime cannot enable imagery when configured provider is 'none'");
        if (enabled && config.imageryProvider != "debug_colorize" && config.imageryProvider != "url_template")
            throw std::runtime_error("GeoSpatialRuntime cannot enable unsupported imagery provider '" + config.imageryProvider + "'");

        imageryProviderEnabled = enabled;

        if (!tileset)
            return;

        if (enabled) {
            LOG_DEBUG("GeoSpatialRuntime", "Imagery: attaching provider=" + config.imageryProvider);
            attachImageryProvider();
            LOG_DEBUG("GeoSpatialRuntime", "Imagery: provider attached");
            return;
        }

        if (imageryOverlay) {
            LOG_DEBUG("GeoSpatialRuntime", "Imagery: removing provider");
            tileset->getOverlays().remove(imageryOverlay);
            imageryOverlay.reset();
        }
    }

    void updateFrame(const UICameraFrame& cameraFrame, float dtSeconds)
    {
        const bool logFirstFrame = !firstUpdateFrameLogged;
        if (logFirstFrame)
            LOG_DEBUG("GeoSpatialRuntime", "Cesium updateFrame: enter main thread scope");
        auto mainThreadScope = asyncSystem.enterMainThread();
        if (logFirstFrame)
            LOG_DEBUG("GeoSpatialRuntime", "Cesium updateFrame: dispatch main thread tasks");
        asyncSystem.dispatchMainThreadTasks();
        if (logFirstFrame)
            LOG_DEBUG("GeoSpatialRuntime", "Cesium updateFrame: tick asset accessor");
        assetAccessor->tick();

        if (!tileset) {
            if (logFirstFrame)
                LOG_DEBUG("GeoSpatialRuntime", "Cesium updateFrame: no tileset loaded");
            renderAdapter->setFrameSelection({}, glm::dmat4(1.0), glm::dmat4(1.0));
            firstUpdateFrameLogged = true;
            return;
        }

        viewStates.clear();
        viewStates.push_back(makeCameraViewState(cameraFrame));
        if (logFirstFrame)
            LOG_DEBUG("GeoSpatialRuntime", "Cesium updateFrame: update view group");
        const Cesium3DTilesSelection::ViewUpdateResult& updateResult =
            tileset->updateViewGroup(tileset->getDefaultViewGroup(), viewStates, dtSeconds);
        if (logFirstFrame)
            LOG_DEBUG("GeoSpatialRuntime", "Cesium updateFrame: set frame selection");
        renderAdapter->setFrameSelection(updateResult.tilesToRenderThisFrame,
                                         cameraFrame.view,
                                         cameraFrame.projection);
        if (logFirstFrame)
            LOG_DEBUG("GeoSpatialRuntime", "Cesium updateFrame: load tiles");
        tileset->loadTiles();
        if (logFirstFrame)
            LOG_DEBUG("GeoSpatialRuntime", "Cesium updateFrame: load tiles complete");

        loadedTileCount = tileset->getNumberOfTilesLoaded();
        loadProgress = tileset->computeLoadProgress();
        loadedBytes = tileset->getTotalDataBytes();
        firstUpdateFrameLogged = true;
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

    std::string imageryProviderStatus() const
    {
        if (!imageryProviderEnabled)
            return "Imagery provider disabled";
        if (config.imageryProvider == "url_template")
            return "URL template imagery provider enabled: " + config.imageryCredit;
        if (config.imageryProvider == "debug_colorize")
            return "Debug colorized imagery provider enabled";
        if (config.imageryProvider == "none")
            return "No imagery provider configured";
        throw std::runtime_error("GeoSpatialRuntime has unsupported imagery provider '" + config.imageryProvider + "'");
    }

    void applyImageryConfig(const GeoSpatialRuntimeConfig& runtimeConfig)
    {
        if (imageryOverlay && tileset) {
            tileset->getOverlays().remove(imageryOverlay);
            imageryOverlay.reset();
        }

        config.imageryProvider = runtimeConfig.imageryProvider;
        config.imageryUrlTemplate = runtimeConfig.imageryUrlTemplate;
        config.imageryCredit = runtimeConfig.imageryCredit;
        config.imageryMinimumLevel = runtimeConfig.imageryMinimumLevel;
        config.imageryMaximumLevel = runtimeConfig.imageryMaximumLevel;
        config.imageryTileWidth = runtimeConfig.imageryTileWidth;
        config.imageryTileHeight = runtimeConfig.imageryTileHeight;
        config.imageryHeaders = runtimeConfig.imageryHeaders;
        rasterOverlayOptions = makeRasterOverlayOptions(runtimeConfig);

        if (imageryProviderEnabled && tileset)
            attachImageryProvider();
    }

    GeoSpatialRuntimeConfig config;
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
    bool firstUpdateFrameLogged = false;

private:
    void attachImageryProvider()
    {
        if (!tileset)
            throw std::runtime_error("GeoSpatialRuntime cannot attach imagery provider without a tileset");

        if (imageryOverlay)
            return;

        if (config.imageryProvider == "debug_colorize") {
            LOG_DEBUG("GeoSpatialRuntime", "Imagery: create DebugColorizeTilesRasterOverlay");
            imageryOverlay = new CesiumRasterOverlays::DebugColorizeTilesRasterOverlay(
                "GRIM debug imagery provider",
                rasterOverlayOptions);
        } else if (config.imageryProvider == "url_template") {
            LOG_DEBUG("GeoSpatialRuntime", "Imagery: create UrlTemplateRasterOverlay url=" + config.imageryUrlTemplate);
            imageryOverlay = new CesiumRasterOverlays::UrlTemplateRasterOverlay(
                "GRIM URL template satellite imagery",
                config.imageryUrlTemplate,
                config.imageryHeaders,
                makeUrlTemplateImageryOptions(config),
                rasterOverlayOptions);
        } else {
            throw std::runtime_error("GeoSpatialRuntime cannot attach unsupported imagery provider '" + config.imageryProvider + "'");
        }
        LOG_DEBUG("GeoSpatialRuntime", "Imagery: adding overlay to tileset");
        tileset->getOverlays().add(imageryOverlay);
        LOG_DEBUG("GeoSpatialRuntime", "Imagery: overlay added to tileset");
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

    LOG_DEBUG("GeoSpatialRuntime", "Init Cesium: loading geospatial runtime config");
    const GeoSpatialRuntimeConfig config = loadGeospatialRuntimeConfig();
    LOG_DEBUG("GeoSpatialRuntime", "Init Cesium: config loaded provider=" + config.imageryProvider +
                                       " terrain=" + config.terrainProvider);
    camera_.setHomeCartographic(config.home);
    camera_.setFrustum(config.frustum);
    camera_.setZoomLimits(config.zoomLimits);
    camera_.setOrbitDistanceMeters(config.initialOrbitDistanceMeters);

    LOG_DEBUG("GeoSpatialRuntime", "Init Cesium: constructing runtime context");
    cesium_ = std::make_unique<CesiumRuntimeContext>(config);
    LOG_DEBUG("GeoSpatialRuntime", "Init Cesium: attaching viewport; imagery starts disabled for first-frame stability");
    cesium_->renderAdapter->setViewportAttachment(viewportAttachment_);
    cesium_->setImageryProviderEnabled(false);
    if (config.terrainEnabledOnStart) {
        LOG_DEBUG("GeoSpatialRuntime", "Init Cesium: creating ellipsoid tileset");
        cesium_->createOrReloadTileset();
    }

    initialized_ = true;
    firstFrameLogged_ = false;
    snapshot_.cesium_ready = true;
    snapshot_.terrain_enabled = config.terrainEnabledOnStart;
    snapshot_.imagery_enabled = false;
    snapshot_.cesium_status = "Cesium Native runtime owner initialized";
    updateProviderStatusLocked();
    refreshCameraStatusLocked();
    LOG_DEBUG("GeoSpatialRuntime", "Init Cesium: complete");
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

    const GeoSpatialRuntimeConfig config = loadGeospatialRuntimeConfig();
    camera_.setHomeCartographic(config.home);
    camera_.setFrustum(config.frustum);
    camera_.setZoomLimits(config.zoomLimits);
    camera_.setOrbitDistanceMeters(config.initialOrbitDistanceMeters);

    cesium_ = std::make_unique<CesiumRuntimeContext>(config);
    cesium_->renderAdapter->setViewportAttachment(viewportAttachment_);
    cesium_->setImageryProviderEnabled(false);
    if (config.terrainEnabledOnStart)
        cesium_->createOrReloadTileset();

    firstFrameLogged_ = false;
    snapshot_.terrain_enabled = config.terrainEnabledOnStart;
    snapshot_.imagery_enabled = false;
    updateProviderStatusLocked();
    snapshot_.cesium_status = "Cesium runtime config reloaded from ai_config.json";
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

    const bool nextEnabled = !snapshot_.imagery_enabled;
    if (nextEnabled)
        cesium_->applyImageryConfig(loadGeospatialRuntimeConfig());

    snapshot_.imagery_enabled = nextEnabled;
    cesium_->setImageryProviderEnabled(nextEnabled);
    snapshot_.cesium_status = cesium_->imageryProviderStatus();
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
        if (!firstFrameLogged_) {
            LOG_DEBUG("GeoSpatialRuntime", "First Cesium tick: begin updateFrame terrain=" +
                                           std::to_string(snapshot_.terrain_enabled) +
                                           " imagery=" + std::to_string(snapshot_.imagery_enabled) +
                                           " viewport=" + std::to_string(viewportWidth) + "x" + std::to_string(viewportHeight));
        }
        cesium_->updateFrame(camera_.frame(), dtSeconds);
        if (!firstFrameLogged_) {
            LOG_DEBUG("GeoSpatialRuntime", "First Cesium tick: updateFrame complete");
            firstFrameLogged_ = true;
        }
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
    const UICameraZoomLimits zoomLimits = camera_.zoomLimits();
    camera_.resetOrbit();
    camera_.setZoomLimits(zoomLimits);
    if (cesium_)
        camera_.setOrbitDistanceMeters(cesium_->config.initialOrbitDistanceMeters);
    viewportDragPixels_ = 0.0;
    refreshCameraStatusLocked();
}

} // namespace GRIM::GeoSpatial