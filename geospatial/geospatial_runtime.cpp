#include "geospatial_runtime.hpp"

#ifdef OPAQUE
#undef OPAQUE
#endif

#include "cesium_bgfx_render_adapter.hpp"
#include "geospatial_catalog.hpp"
#include "geospatial_render_aggregation.hpp"
#include "core/platform_input.hpp"
#include "logger.hpp"
#include "ui/primitives/ui_native_3d_viewport_attachment.hpp"

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
#include <CesiumGeometry/QuadtreeTilingScheme.h>
#include <CesiumGeospatial/Cartographic.h>
#include <CesiumGeospatial/Ellipsoid.h>
#include <CesiumGeospatial/GeographicProjection.h>
#include <CesiumRasterOverlays/IonRasterOverlay.h>
#include <CesiumRasterOverlays/RasterOverlayLoadFailureDetails.h>
#include <CesiumUtility/CreditSystem.h>

#include <nlohmann/json.hpp>

#include <spdlog/spdlog.h>

#include <algorithm>
#include <condition_variable>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <functional>
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

    struct GeoSpatialRasterLayerConfig {
        std::string id;
        std::string name;
        int64_t ionAssetId = 0;
        std::string ionAccessToken;
        std::string credit;
        bool visibleOnStart = false;
        float opacity = 1.0f;
    };

    struct GeoSpatialRuntimeConfig {
        CesiumGeospatial::Cartographic home{0.0, 0.0, 0.0};
        double initialOrbitDistanceMeters = 0.0;
        UICameraFrustum frustum;
        UICameraZoomLimits zoomLimits;
        std::filesystem::path cacheDatabasePath;
        std::string terrainProvider;
        int64_t terrainIonAssetId = 0;
        std::string terrainIonAccessToken;
        std::vector<GeoSpatialRasterLayerConfig> rasterLayers;
        std::string globeTexturePath;
        std::string userAgent;
        int32_t workerThreads = 0;
        uint64_t assetCacheItems = 0;
        int32_t requestsPerCachePrune = 0;
        int64_t tileCacheBytes = 0;
        int64_t rasterSubTileCacheBytes = 0;
        int32_t maximumSimultaneousTileLoads = 0;
        int32_t maximumSimultaneousRasterLoads = 0;
        double maximumScreenSpaceError = 0.0;
        double culledScreenSpaceError = 0.0;
        double lodSensitivityMultiplier = 0.0;
        double lodTransitionLengthSeconds = 0.0;
        bool favorCenterTiles = false;
        double maximumRasterScreenSpaceError = 0.0;
        uint32_t loadingDescendantLimit = 0;
        bool forbidHoles = false;
        double mainThreadLoadingTimeLimitMilliseconds = 0.0;
        double tileCacheUnloadTimeLimitMilliseconds = 0.0;
        int32_t maximumRasterTextureSize = 0;
        bool terrainEnabledOnStart = false;
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

    void loadIonRasterLayers(GeoSpatialRuntimeConfig& config, const nlohmann::json& imagery)
    {
        const std::string accessToken = requireString(imagery, "access_token", "geospatial.imagery");
        if (!imagery.contains("layers") || !imagery.at("layers").is_array() || imagery.at("layers").empty())
            throw std::runtime_error("GeoSpatial config requires non-empty array geospatial.imagery.layers");

        const nlohmann::json& layers = imagery.at("layers");
        config.rasterLayers.reserve(layers.size());
        for (size_t index = 0; index < layers.size(); ++index) {
            if (!layers.at(index).is_object())
                throw std::runtime_error("GeoSpatial config geospatial.imagery.layers entries must be objects");

            const nlohmann::json& layer = layers.at(index);
            const std::string path = "geospatial.imagery.layers[" + std::to_string(index) + "]";
            GeoSpatialRasterLayerConfig parsed;
            parsed.id = requireString(layer, "id", path.c_str());
            parsed.name = requireString(layer, "name", path.c_str());
            parsed.ionAssetId = requireInteger(layer, "ion_asset_id", path.c_str());
            parsed.ionAccessToken = accessToken;
            parsed.credit = requireString(layer, "credit", path.c_str());
            parsed.visibleOnStart = requireBool(layer, "visible_on_start", path.c_str());
            parsed.opacity = static_cast<float>(requireNumber(layer, "opacity", path.c_str()));
            if (parsed.ionAssetId <= 0)
                throw std::runtime_error("GeoSpatial config " + path + ".ion_asset_id must be positive");
            if (parsed.opacity < 0.0f || parsed.opacity > 1.0f)
                throw std::runtime_error("GeoSpatial config " + path + ".opacity must be within [0, 1]");

            const auto duplicate = std::find_if(config.rasterLayers.begin(), config.rasterLayers.end(),
                [&](const GeoSpatialRasterLayerConfig& existing) { return existing.id == parsed.id; });
            if (duplicate != config.rasterLayers.end())
                throw std::runtime_error("GeoSpatial config contains duplicate imagery layer id '" + parsed.id + "'");
            config.rasterLayers.push_back(std::move(parsed));
        }
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

    std::filesystem::path resolveExistingConfigFilePath(const std::string& value, const char* pathName)
    {
        std::filesystem::path path(value);
        if (path.is_relative())
            path = std::filesystem::path(GRIM_ROOT_DIR) / path;

        if (!path.has_filename())
            throw std::runtime_error(std::string("GeoSpatial config ") + pathName + " must include a file name");
        if (!std::filesystem::exists(path))
            throw std::runtime_error(std::string("GeoSpatial config ") + pathName + " file does not exist: " + path.string());
        if (!std::filesystem::is_regular_file(path))
            throw std::runtime_error(std::string("GeoSpatial config ") + pathName + " must reference a regular file: " + path.string());
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
        if (config.terrainProvider != "cesium_ion")
            throw std::runtime_error("GeoSpatial config terrain.provider must be 'cesium_ion'");
        config.terrainIonAssetId = requireInteger(terrain, "ion_asset_id", "geospatial.terrain");
        if (config.terrainIonAssetId <= 0)
            throw std::runtime_error("GeoSpatial config terrain.ion_asset_id must be a positive Cesium ion asset id");
        config.terrainIonAccessToken = requireString(terrain, "access_token", "geospatial.terrain");

        loadIonRasterLayers(config, imagery);
        const bool rasterVisibleOnStart = std::any_of(config.rasterLayers.begin(), config.rasterLayers.end(),
            [](const GeoSpatialRasterLayerConfig& layer) { return layer.visibleOnStart; });
        if (rasterVisibleOnStart && !config.terrainEnabledOnStart)
            throw std::runtime_error("GeoSpatial config cannot show imagery layers without terrain tileset on start");

        if (imagery.contains("single_texture_path")) {
            if (!imagery.at("single_texture_path").is_string())
                throw std::runtime_error("GeoSpatial config requires string geospatial.imagery.single_texture_path");
            const std::string texturePath = imagery.at("single_texture_path").get<std::string>();
            if (!texturePath.empty())
                config.globeTexturePath = resolveExistingConfigFilePath(texturePath, "geospatial.imagery.single_texture_path").string();
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
        config.lodSensitivityMultiplier = requireNumber(loading, "lod_sensitivity_multiplier", "geospatial.loading");
        config.lodTransitionLengthSeconds = requireNumber(loading, "lod_transition_length_seconds", "geospatial.loading");
        config.favorCenterTiles = requireBool(loading, "favor_center_tiles", "geospatial.loading");
        config.maximumRasterScreenSpaceError = requireNumber(loading, "maximum_raster_screen_space_error", "geospatial.loading");
        config.loadingDescendantLimit = requireUInt32(loading, "loading_descendant_limit", "geospatial.loading");
        config.forbidHoles = requireBool(loading, "forbid_holes", "geospatial.loading");
        config.mainThreadLoadingTimeLimitMilliseconds = requireNumber(loading, "main_thread_loading_time_limit_milliseconds", "geospatial.loading");
        config.tileCacheUnloadTimeLimitMilliseconds = requireNumber(loading, "tile_cache_unload_time_limit_milliseconds", "geospatial.loading");
        config.maximumRasterTextureSize = requirePositiveInt32(loading, "maximum_raster_texture_size", "geospatial.loading");
        config.userAgent = requireString(loading, "user_agent", "geospatial.loading");
        if (config.maximumScreenSpaceError <= 0.0)
            throw std::runtime_error("GeoSpatial config maximum_screen_space_error must be positive");
        if (config.culledScreenSpaceError <= 0.0)
            throw std::runtime_error("GeoSpatial config culled_screen_space_error must be positive");
        if (config.lodSensitivityMultiplier < 0.1 || config.lodSensitivityMultiplier > 8.0)
            throw std::runtime_error("GeoSpatial config lod_sensitivity_multiplier must be within [0.1, 8.0]");
        if (config.lodTransitionLengthSeconds <= 0.0 || config.lodTransitionLengthSeconds > 5.0)
            throw std::runtime_error("GeoSpatial config lod_transition_length_seconds must be within (0, 5.0]");
        if (config.maximumRasterScreenSpaceError <= 0.0)
            throw std::runtime_error("GeoSpatial config maximum_raster_screen_space_error must be positive");
        if (config.mainThreadLoadingTimeLimitMilliseconds < 0.0)
            throw std::runtime_error("GeoSpatial config main_thread_loading_time_limit_milliseconds must be non-negative");
        if (config.tileCacheUnloadTimeLimitMilliseconds < 0.0)
            throw std::runtime_error("GeoSpatial config tile_cache_unload_time_limit_milliseconds must be non-negative");

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
        options.maximumScreenSpaceError = config.maximumScreenSpaceError / config.lodSensitivityMultiplier;
        options.culledScreenSpaceError = config.culledScreenSpaceError;
        options.enforceCulledScreenSpaceError = !config.favorCenterTiles;
        options.preloadAncestors = true;
        options.preloadSiblings = !config.favorCenterTiles;
        options.renderTilesUnderCamera = !config.favorCenterTiles;
        options.loadingDescendantLimit = config.loadingDescendantLimit;
        options.forbidHoles = config.forbidHoles;
        options.maximumCachedBytes = config.tileCacheBytes;
        options.maximumSimultaneousTileLoads = config.maximumSimultaneousTileLoads;
        options.mainThreadLoadingTimeLimit = config.mainThreadLoadingTimeLimitMilliseconds;
        options.tileCacheUnloadTimeLimit = config.tileCacheUnloadTimeLimitMilliseconds;
        options.enableOcclusionCulling = false;
        options.enableLodTransitionPeriod = true;
        options.lodTransitionLength = static_cast<float>(config.lodTransitionLengthSeconds);
        options.credit = std::string("Cesium Native ") + config.terrainProvider + " terrain";
        options.loadErrorCallback = [](const Cesium3DTilesSelection::TilesetLoadFailureDetails& details) {
            // Never throw from inside Cesium callbacks: individual tile/network
            // failures are routine (e.g. transient 404s) and must not unwind
            // through Cesium's update loop.
            LOG_ERROR("GeoSpatialRuntime", "Cesium tileset load failed: " + details.message +
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
            // Missing imagery tiles (GIBS returns 404 for no-data regions such
            // as polar night) are expected; log and let coarser imagery or the
            // fallback base color cover the area.
            LOG_ERROR("GeoSpatialRuntime", "Cesium raster overlay load failed: " + details.message +
                                           responseStatusSuffix(details.pRequest.get()));
        };
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
    struct RasterLayerState {
        GeoSpatialRasterLayerConfig config;
        bool visible = false;
        int32_t overlayTextureCoordinateID = -1;
        CesiumUtility::IntrusivePointer<CesiumRasterOverlays::RasterOverlay> overlay;
    };

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
        renderAdapter->setGlobeTexturePath(config.globeTexturePath);
        rasterLayers.reserve(config.rasterLayers.size());
        for (const GeoSpatialRasterLayerConfig& layer : config.rasterLayers)
            rasterLayers.push_back(RasterLayerState{layer, layer.visibleOnStart, -1, nullptr});
    }

    ~CesiumRuntimeContext()
    {
        tileset.reset();
        for (RasterLayerState& layer : rasterLayers)
            layer.overlay.reset();
    }

    void createOrReloadTileset()
    {
        if (config.terrainProvider != "cesium_ion")
            throw std::runtime_error("GeoSpatialRuntime cannot create unsupported terrain provider '" + config.terrainProvider + "'");

        for (RasterLayerState& layer : rasterLayers)
            layer.overlay.reset();
        std::vector<float> activeOpacities;
        for (RasterLayerState& layer : rasterLayers) {
            layer.overlayTextureCoordinateID = -1;
            if (!layer.visible)
                continue;
            layer.overlayTextureCoordinateID = static_cast<int32_t>(activeOpacities.size());
            activeOpacities.push_back(layer.config.opacity);
        }
        renderAdapter->setRasterOverlayOpacities(std::move(activeOpacities));
        tileset = std::make_unique<Cesium3DTilesSelection::Tileset>(
            externals,
            config.terrainIonAssetId,
            config.terrainIonAccessToken,
            tilesetOptions);
        if (!tileset)
            throw std::runtime_error("Cesium ion terrain tileset creation returned NULL");

        for (RasterLayerState& layer : rasterLayers) {
            if (layer.visible)
                attachRasterLayer(layer);
        }
    }

    void setLayerVisibility(const std::string& id, bool visible)
    {
        RasterLayerState& layer = requireRasterLayer(id);
        if (layer.visible == visible)
            return;
        layer.visible = visible;

        if (!tileset)
            return;

        if (visible) {
            // Tiles only receive _CESIUMOVERLAY_N texture coordinates when
            // they load with the overlay already in the collection. Attaching
            // to a live tileset leaves every already-loaded tile base-only
            // forever (imagery stuck on a single wedge of the globe), so
            // rebuild the tileset and let all tiles reload with overlay UVs.
            // Imagery bytes come back from the sqlite asset cache, so the
            // reload is cheap.
            LOG_DEBUG("GeoSpatialRuntime", "Imagery layer '" + layer.config.id +
                                           "': rebuilding tileset for overlay texture coordinates");
            createOrReloadTileset();
            return;
        }

        if (layer.overlay) {
            LOG_DEBUG("GeoSpatialRuntime", "Imagery layer '" + layer.config.id + "': removing overlay");
            tileset->getOverlays().remove(layer.overlay);
            layer.overlay.reset();
        }
    }

    void setLayerOpacity(const std::string& id, float opacity)
    {
        if (opacity < 0.0f || opacity > 1.0f)
            throw std::runtime_error("GeoSpatialRuntime layer opacity must be within [0, 1]");
        RasterLayerState& layer = requireRasterLayer(id);
        layer.config.opacity = opacity;
        updateAdapterOpacities();
    }

    std::vector<GeoSpatialLayerSnapshot> layerSnapshots(bool terrainVisible) const
    {
        std::vector<GeoSpatialLayerSnapshot> result;
        result.reserve(rasterLayers.size() + 1);
        result.push_back(GeoSpatialLayerSnapshot{"terrain", "Terrain", terrainVisible, 1.0f});
        for (const RasterLayerState& layer : rasterLayers)
            result.push_back(GeoSpatialLayerSnapshot{layer.config.id, layer.config.name, layer.visible, layer.config.opacity});
        return result;
    }

    bool anyRasterLayerVisible() const
    {
        return std::any_of(rasterLayers.begin(), rasterLayers.end(),
            [](const RasterLayerState& layer) { return layer.visible; });
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
            renderAdapter->setFrameSelection({}, {}, glm::dmat4(1.0), glm::dmat4(1.0));
            firstUpdateFrameLogged = true;
            return;
        }

        viewStates.clear();
        viewStates.push_back(makeCameraViewState(cameraFrame));
        if (logFirstFrame)
            LOG_DEBUG("GeoSpatialRuntime", "Cesium updateFrame: update view group");
        const Cesium3DTilesSelection::ViewUpdateResult& updateResult =
            tileset->updateViewGroup(tileset->getDefaultViewGroup(), viewStates, dtSeconds);
        {
            static uint32_t s_cameraDiagFrame = 0;
            if ((s_cameraDiagFrame++ % 90u) == 0) {
                LOG_DEBUG("GeoSpatialRuntime",
                          "Cesium camera diag selected=" + std::to_string(updateResult.tilesToRenderThisFrame.size()) +
                          " fadingOut=" + std::to_string(updateResult.tilesFadingOut.size()) +
                          " pos=" + formatEcef(cameraFrame.positionEcef) +
                          " dir=" + formatEcef(cameraFrame.directionEcef) +
                          " up=" + formatEcef(cameraFrame.upEcef) +
                          " target=" + formatCartographic(cameraFrame.targetCartographic) +
                          " camera=" + formatCartographic(cameraFrame.cameraCartographic) +
                          " yaw=" + std::to_string(cameraFrame.yawRadians) +
                          " pitch=" + std::to_string(cameraFrame.pitchRadians) +
                          " dist=" + std::to_string(cameraFrame.orbitDistanceMeters));
            }
        }
        if (logFirstFrame)
            LOG_DEBUG("GeoSpatialRuntime", "Cesium updateFrame: set frame selection");
        renderAdapter->setFrameSelection(updateResult.tilesToRenderThisFrame,
                                         std::vector<Cesium3DTilesSelection::Tile::ConstPointer>(
                                             updateResult.tilesFadingOut.begin(),
                                             updateResult.tilesFadingOut.end()),
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
        stream << "Cesium ion terrain tileset | loaded=" << loadedTileCount
               << " progress=" << std::fixed << std::setprecision(0) << loadProgress << "%"
               << " cache=" << (loadedBytes / (1024 * 1024)) << " MiB";
        return stream.str();
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
    std::vector<RasterLayerState> rasterLayers;
    std::vector<Cesium3DTilesSelection::ViewState> viewStates;
    int32_t loadedTileCount = 0;
    float loadProgress = 0.0f;
    int64_t loadedBytes = 0;
    bool firstUpdateFrameLogged = false;

private:
    void updateAdapterOpacities()
    {
        int32_t maximumID = -1;
        for (const RasterLayerState& layer : rasterLayers)
            maximumID = std::max(maximumID, layer.overlayTextureCoordinateID);

        std::vector<float> opacities(static_cast<size_t>(maximumID + 1), 1.0f);
        for (const RasterLayerState& layer : rasterLayers) {
            if (layer.overlayTextureCoordinateID >= 0)
                opacities[static_cast<size_t>(layer.overlayTextureCoordinateID)] = layer.config.opacity;
        }
        renderAdapter->setRasterOverlayOpacities(std::move(opacities));
    }

    RasterLayerState& requireRasterLayer(const std::string& id)
    {
        auto layer = std::find_if(rasterLayers.begin(), rasterLayers.end(),
            [&](const RasterLayerState& candidate) { return candidate.config.id == id; });
        if (layer == rasterLayers.end())
            throw std::runtime_error("GeoSpatialRuntime does not contain imagery layer '" + id + "'");
        return *layer;
    }

    void attachRasterLayer(RasterLayerState& layer)
    {
        if (!tileset)
            throw std::runtime_error("GeoSpatialRuntime cannot attach imagery layer without a tileset");
        if (layer.overlay)
            return;

        LOG_DEBUG("GeoSpatialRuntime", "[IMAGERY_MEASURE] provider=cesium_ion asset_id=" +
                                       std::to_string(layer.config.ionAssetId) + " layer=" + layer.config.id +
                                       " level=server tile=server");
        layer.overlay = new CesiumRasterOverlays::IonRasterOverlay(
            layer.config.name,
            layer.config.ionAssetId,
            layer.config.ionAccessToken,
            rasterOverlayOptions);
        tileset->getOverlays().add(layer.overlay);
        LOG_DEBUG("GeoSpatialRuntime", "Imagery layer '" + layer.config.id + "' attached: " + layer.config.credit);
    }
};

GeoSpatialRuntime::GeoSpatialRuntime()
{
    snapshot_.cesium_status = "Cesium Native linked; waiting for initialization";
    snapshot_.active_tileset = "No tileset loaded";
    snapshot_.camera_mode = "Home camera not initialized";
    loadPointCatalogLocked();
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
    try {

        LOG_DEBUG("GeoSpatialRuntime", "Init Cesium: loading geospatial runtime config");
        const GeoSpatialRuntimeConfig config = loadGeospatialRuntimeConfig();
        LOG_DEBUG("GeoSpatialRuntime", "Init Cesium: config loaded raster_layers=" +
                           std::to_string(config.rasterLayers.size()) +
                           " terrain=" + config.terrainProvider);
        camera_.setHomeCartographic(config.home);
        camera_.setFrustum(config.frustum);
        camera_.setZoomLimits(config.zoomLimits);
        camera_.setOrbitDistanceMeters(config.initialOrbitDistanceMeters);

        LOG_DEBUG("GeoSpatialRuntime", "Init Cesium: constructing runtime context");
        cesium_ = std::make_unique<CesiumRuntimeContext>(config);
        LOG_DEBUG("GeoSpatialRuntime", "Init Cesium: attaching viewport");
        cesium_->renderAdapter->setViewportAttachment(viewportAttachment_);
        if (config.terrainEnabledOnStart) {
            LOG_DEBUG("GeoSpatialRuntime", "Init Cesium: creating Cesium ion terrain tileset");
            cesium_->createOrReloadTileset();
        }

        initialized_ = true;
        firstFrameLogged_ = false;
        snapshot_.cesium_ready = true;
        snapshot_.terrain_enabled = config.terrainEnabledOnStart;
        snapshot_.cesium_status = "Cesium Native runtime owner initialized";
        updateProviderStatusLocked();
        refreshCameraStatusLocked();
        LOG_DEBUG("GeoSpatialRuntime", "Init Cesium: complete");
    } catch (const std::exception& error) {
        cesium_.reset();
        initialized_ = false;
        snapshot_.cesium_ready = false;
        snapshot_.terrain_enabled = false;
        snapshot_.imagery_enabled = false;
        snapshot_.layers.clear();
        snapshot_.active_tileset = "No tileset loaded";
        setActionFailedStatusLocked("Init Cesium", error);
    }
}

void GeoSpatialRuntime::requestReloadTileset()
{
    std::lock_guard lock(mutex_);
    try {
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
    if (config.terrainEnabledOnStart)
        cesium_->createOrReloadTileset();

    firstFrameLogged_ = false;
    snapshot_.terrain_enabled = config.terrainEnabledOnStart;
    updateProviderStatusLocked();
    snapshot_.cesium_status = "Cesium runtime config reloaded from ai_config.json";
    } catch (const std::exception& error) {
        setActionFailedStatusLocked("Reload tileset", error);
    }
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

void GeoSpatialRuntime::requestSetLayerVisibility(const std::string& id, bool visible)
{
    std::lock_guard lock(mutex_);
    try {
    if (!initialized_) {
        setNotInitializedStatusLocked("change layer visibility");
        return;
    }

    if (!cesium_)
        throw std::runtime_error("GeoSpatialRuntime initialized without CesiumRuntimeContext");

    if (id.empty())
        throw std::runtime_error("GeoSpatialRuntime layer visibility request requires a non-empty id");

    if (id == "terrain") {
        if (snapshot_.terrain_enabled == visible)
            return;
        snapshot_.terrain_enabled = visible;
        if (visible) {
            cesium_->createOrReloadTileset();
            snapshot_.cesium_status = "Terrain layer enabled";
        } else {
            cesium_->tileset.reset();
            snapshot_.cesium_status = "Terrain layer disabled; tileset released";
        }
        updateProviderStatusLocked();
        return;
    }

    cesium_->setLayerVisibility(id, visible);
    if (visible)
        snapshot_.cesium_status = "Map layer '" + id + "' enabled";
    else
        snapshot_.cesium_status = "Map layer '" + id + "' disabled";
    updateProviderStatusLocked();
    } catch (const std::exception& error) {
        updateProviderStatusLocked();
        setActionFailedStatusLocked("Set layer visibility", error);
    }
}

void GeoSpatialRuntime::requestSetLayerOpacity(const std::string& id, float opacity)
{
    std::lock_guard lock(mutex_);
    try {
        if (!initialized_) {
            setNotInitializedStatusLocked("change layer opacity");
            return;
        }
        if (!cesium_)
            throw std::runtime_error("GeoSpatialRuntime initialized without CesiumRuntimeContext");
        if (id == "terrain")
            throw std::runtime_error("Terrain layer opacity is fixed at 100 percent");

        cesium_->setLayerOpacity(id, opacity);
        snapshot_.cesium_status = "Map layer '" + id + "' opacity updated";
        updateProviderStatusLocked();
    } catch (const std::exception& error) {
        updateProviderStatusLocked();
        setActionFailedStatusLocked("Set layer opacity", error);
    }
}

void GeoSpatialRuntime::requestSavePointCatalog()
{
    std::lock_guard lock(mutex_);
    try {
        savePointCatalogLocked();
        snapshot_.point_catalog_dirty = false;
        snapshot_.point_catalog_status = "Saved " + std::to_string(snapshot_.groups.size()) + " group(s) to " + pointCatalogPath().string();
    } catch (const std::exception& error) {
        setPointCatalogFailedStatusLocked("Save point catalog", error);
    }
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
        const UICameraFrame cameraFrame = camera_.frame();
        syncPointMarkersLocked();
        cesium_->updateFrame(cameraFrame, dtSeconds);
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

void GeoSpatialRuntime::setActionFailedStatusLocked(const char* action, const std::exception& error)
{
    snapshot_.cesium_status = std::string(action) + " failed: " + error.what();
    LOG_ERROR("GeoSpatialRuntime", snapshot_.cesium_status);
}

void GeoSpatialRuntime::updateProviderStatusLocked()
{
    if (!cesium_) {
        snapshot_.active_tileset = "No Cesium runtime owner";
        return;
    }

    snapshot_.active_tileset = cesium_->tilesetStatus();
    snapshot_.imagery_enabled = cesium_->anyRasterLayerVisible();
    snapshot_.layers = cesium_->layerSnapshots(snapshot_.terrain_enabled);
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
                if (picked) {
                    snapshot_.picked_location_valid = true;
                    snapshot_.picked_longitude_degrees = picked->longitude * kRadiansToDegrees;
                    snapshot_.picked_latitude_degrees = picked->latitude * kRadiansToDegrees;
                    snapshot_.picked_height_meters = picked->height;
                }
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

    camera_.clampOrbitDistance();
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

void GeoSpatialRuntime::loadPointCatalogLocked()
{
    const std::filesystem::path path = pointCatalogPath();
    if (!std::filesystem::exists(path)) {
        snapshot_.point_catalog_status = "No saved point catalog; create the first group";
        return;
    }
    snapshot_.groups = loadPointCatalog(path);
    snapshot_.point_catalog_dirty = false;
    snapshot_.point_catalog_status = "Loaded " + std::to_string(snapshot_.groups.size()) + " saved group(s)";
}

void GeoSpatialRuntime::savePointCatalogLocked()
{
    savePointCatalog(pointCatalogPath(), snapshot_.groups);
}

void GeoSpatialRuntime::syncPointMarkersLocked()
{
    if (!cesium_)
        return;

    GeoSpatialRenderPayloads payloads = aggregateRenderPayloads(snapshot_.groups);
    cesium_->renderAdapter->setPointMarkers(std::move(payloads.point_markers));
    cesium_->renderAdapter->setAreaShapes(std::move(payloads.area_shapes));
}

void GeoSpatialRuntime::markPointCatalogChangedLocked(const std::string& status)
{
    snapshot_.point_catalog_dirty = true;
    snapshot_.point_catalog_status = status + "; save data to persist changes";
}

void GeoSpatialRuntime::setPointCatalogFailedStatusLocked(const char* action, const std::exception& error)
{
    snapshot_.point_catalog_status = std::string(action) + " failed: " + error.what();
    LOG_ERROR("GeoSpatialRuntime", snapshot_.point_catalog_status);
}

std::string GeoSpatialRuntime::generatePointCatalogIdLocked(const char* prefix) const
{
    if (!prefix)
        throw std::runtime_error("GeoSpatial point catalog ID generator prefix is NULL");
    return generatePointCatalogId(snapshot_.groups, prefix);
}

} // namespace GRIM::GeoSpatial