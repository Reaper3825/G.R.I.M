#include "ui_camera.hpp"

#include <CesiumGeospatial/Ellipsoid.h>

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/geometric.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {
    constexpr double kDefaultMinOrbitDistanceMeters = 500.0;
    constexpr double kMaxOrbitDistanceMultiplier = 5.0;
    constexpr double kOrbitRadiansPerPixel = 0.0045;
    constexpr double kFineOrbitRadiansPerPixel = 0.0015;
    constexpr double kWheelZoomStep = 0.12;
    constexpr double kRadiansEpsilon = 1.0e-8;
    constexpr double kPoleViewThreshold = 0.98;
    constexpr double kDynamicNearPlaneAltitudeFraction = 0.02;
    constexpr double kDynamicNearPlaneMaxAltitudeFraction = 0.5;
    constexpr double kDynamicFarPlaneHorizonMargin = 1.25;

    const CesiumGeospatial::Ellipsoid& wgs84()
    {
        return CesiumGeospatial::Ellipsoid::WGS84;
    }

    glm::dvec3 normalizeOrThrow(const glm::dvec3& value, const char* description)
    {
        const double length = glm::length(value);
        if (length <= 0.0)
            throw std::runtime_error(description);
        return value / length;
    }

    UICameraFrustum frustumForCameraRadius(const UICameraFrustum& source, double cameraRadius)
    {
        const double ellipsoidRadius = wgs84().getMaximumRadius();
        if (cameraRadius <= ellipsoidRadius)
            throw std::runtime_error("UICamera camera radius must be outside the WGS84 ellipsoid");

        UICameraFrustum result = source;
        const double altitudeMeters = cameraRadius - ellipsoidRadius;
        const double dynamicNear = std::clamp(
            altitudeMeters * kDynamicNearPlaneAltitudeFraction,
            source.nearPlaneMeters,
            altitudeMeters * kDynamicNearPlaneMaxAltitudeFraction);
        const double horizonDistance = std::sqrt(std::max(cameraRadius * cameraRadius - ellipsoidRadius * ellipsoidRadius, 0.0));
        const double dynamicFar = std::min(source.farPlaneMeters, horizonDistance * kDynamicFarPlaneHorizonMargin);

        result.nearPlaneMeters = dynamicNear;
        result.farPlaneMeters = std::max(dynamicFar, result.nearPlaneMeters + 1.0);
        return result;
    }
}

UICamera::UICamera()
{
    setHomeCartographic(CesiumGeospatial::Cartographic::fromDegrees(0.0, 0.0, 0.0));
}

void UICamera::setHomeCartographic(const CesiumGeospatial::Cartographic& cartographic)
{
    homeEcef_ = cartographicToEcef(cartographic);
    resetOrbit();
}

void UICamera::setHomeEcef(const glm::dvec3& ecef)
{
    homeEcef_ = normalizeOrThrow(ecef, "UICamera home ECEF must not be the ellipsoid center") * glm::length(ecef);
    ecefToCartographicOrThrow(homeEcef_);
    resetOrbit();
}

CesiumGeospatial::Cartographic UICamera::homeCartographic() const
{
    return ecefToCartographicOrThrow(homeEcef_);
}

void UICamera::setFrustum(const UICameraFrustum& frustum)
{
    if (frustum.horizontalFovRadians <= 0.0 || frustum.verticalFovRadians <= 0.0)
        throw std::runtime_error("UICamera frustum FOV must be positive");
    if (frustum.nearPlaneMeters <= 0.0)
        throw std::runtime_error("UICamera frustum near plane must be positive");
    if (frustum.farPlaneMeters <= frustum.nearPlaneMeters)
        throw std::runtime_error("UICamera frustum far plane must be greater than near plane");
    if (frustum.viewportWidth <= 0 || frustum.viewportHeight <= 0)
        throw std::runtime_error("UICamera frustum viewport size must be positive");

    frustum_ = frustum;
}

void UICamera::setViewportSize(int width, int height)
{
    if (width <= 0 || height <= 0)
        throw std::runtime_error("UICamera viewport size must be positive");

    frustum_.viewportWidth = width;
    frustum_.viewportHeight = height;
}

void UICamera::setZoomLimits(const UICameraZoomLimits& limits)
{
    if (limits.minDistanceMeters <= 0.0)
        throw std::runtime_error("UICamera minimum zoom distance must be positive");
    if (limits.maxDistanceMeters <= limits.minDistanceMeters)
        throw std::runtime_error("UICamera maximum zoom distance must be greater than minimum distance");

    zoomLimits_ = limits;
    clampOrbitDistance();
}

void UICamera::resetOrbit()
{
    const CesiumGeospatial::Cartographic home = homeCartographic();
    yawRadians_ = home.longitude;
    pitchRadians_ = home.latitude;
    orbitDistanceMeters_ = defaultOrbitDistanceMeters();
    orbitTargetEastMeters_ = 0.0;
    orbitTargetNorthMeters_ = 0.0;

    zoomLimits_.minDistanceMeters = kDefaultMinOrbitDistanceMeters;
    zoomLimits_.maxDistanceMeters = orbitDistanceMeters_ * kMaxOrbitDistanceMultiplier;
    clampOrbitDistance();
}

void UICamera::recenterHome()
{
    orbitTargetEastMeters_ = 0.0;
    orbitTargetNorthMeters_ = 0.0;
    clampOrbitDistance();
}

void UICamera::setOrbitDistanceMeters(double distanceMeters)
{
    if (distanceMeters <= 0.0)
        throw std::runtime_error("UICamera orbit distance must be positive");

    orbitDistanceMeters_ = distanceMeters;
    clampOrbitDistance();
}

void UICamera::orbitByRadians(double yawDeltaRadians, double pitchDeltaRadians)
{
    yawRadians_ += yawDeltaRadians;
    pitchRadians_ += pitchDeltaRadians;
}

void UICamera::panByMeters(double eastMeters, double northMeters)
{
    orbitByRadians(eastMeters / std::max(orbitDistanceMeters_, 1.0), northMeters / std::max(orbitDistanceMeters_, 1.0));
}

void UICamera::orbitByPixels(double deltaX, double deltaY, bool fineControl)
{
    const double scale = fineControl ? kFineOrbitRadiansPerPixel : kOrbitRadiansPerPixel;
    orbitByRadians(-deltaX * scale, -deltaY * scale);
}

void UICamera::panByPixels(double deltaX, double deltaY, bool fineControl)
{
    const double scale = orbitDistanceMeters_ * (fineControl ? 0.0002 : 0.0008);
    panByMeters(-deltaX * scale, deltaY * scale);
}

void UICamera::zoomBySteps(double wheelSteps)
{
    orbitDistanceMeters_ *= std::exp(-wheelSteps * kWheelZoomStep);
    clampOrbitDistance();
}

void UICamera::clampOrbitDistance()
{
    orbitDistanceMeters_ = std::clamp(
        orbitDistanceMeters_,
        zoomLimits_.minDistanceMeters,
        zoomLimits_.maxDistanceMeters);
}

UICameraFrame UICamera::frame() const
{
    if (orbitDistanceMeters_ <= 0.0)
        throw std::runtime_error("UICamera orbit distance must be positive before frame evaluation");

    const double cameraRadius = wgs84().getMaximumRadius() + orbitDistanceMeters_;
    const double cosPitch = std::cos(pitchRadians_);
    const glm::dvec3 radialDirection = normalizeOrThrow(
        glm::dvec3(
            cosPitch * std::cos(yawRadians_),
            cosPitch * std::sin(yawRadians_),
            std::sin(pitchRadians_)),
        "UICamera turntable radial direction is degenerate");
    const glm::dvec3 globeCenter{0.0, 0.0, 0.0};
    const glm::dvec3 cameraPosition = radialDirection * cameraRadius;
    const glm::dvec3 direction = normalizeOrThrow(globeCenter - cameraPosition, "UICamera direction is degenerate");

    glm::dvec3 referenceUp{0.0, 0.0, 1.0};
    if (std::abs(glm::dot(direction, referenceUp)) > kPoleViewThreshold) {
        referenceUp = normalizeOrThrow(
            glm::dvec3(-std::sin(yawRadians_), std::cos(yawRadians_), 0.0),
            "UICamera pole reference up is degenerate");
    }

    glm::dvec3 right = glm::cross(direction, referenceUp);
    if (glm::length(right) < kRadiansEpsilon)
        right = glm::dvec3{1.0, 0.0, 0.0};
    else
        right = glm::normalize(right);
    const glm::dvec3 up = glm::normalize(glm::cross(right, direction));

    const std::optional<glm::dvec3> targetSurface = wgs84().scaleToGeodeticSurface(cameraPosition);
    if (!targetSurface)
        throw std::runtime_error("UICamera turntable camera position cannot be projected to WGS84");

    UICameraFrustum frameFrustum = frustumForCameraRadius(frustum_, cameraRadius);
    const double aspect = static_cast<double>(frameFrustum.viewportWidth) / static_cast<double>(frameFrustum.viewportHeight);
    // The rendered projection is vertical FOV x viewport aspect, so derive the
    // horizontal FOV from those instead of trusting the configured value.
    // Cesium's screen-space-error math and our pick rays both use this
    // frustum; a mismatched pair skews LOD selection and picking.
    frameFrustum.horizontalFovRadians = 2.0 * std::atan(std::tan(frameFrustum.verticalFovRadians * 0.5) * aspect);
    UICameraFrame result;
    result.targetEcef = *targetSurface;
    result.positionEcef = cameraPosition;
    result.directionEcef = direction;
    result.upEcef = up;
    result.rightEcef = right;
    result.targetCartographic = ecefToCartographicOrThrow(*targetSurface);
    result.cameraCartographic = ecefToCartographicOrThrow(cameraPosition);
    result.frustum = frameFrustum;
    result.view = glm::lookAt(cameraPosition, globeCenter, up);
    result.projection = glm::perspective(frameFrustum.verticalFovRadians, aspect, frameFrustum.nearPlaneMeters, frameFrustum.farPlaneMeters);
    result.yawRadians = yawRadians_;
    result.pitchRadians = pitchRadians_;
    result.orbitDistanceMeters = orbitDistanceMeters_;
    result.altitudeMeters = result.cameraCartographic.height;
    return result;
}

UICameraRay UICamera::rayFromViewportPixel(double pixelX, double pixelY) const
{
    const UICameraFrame currentFrame = frame();
    const double viewportWidth = static_cast<double>(currentFrame.frustum.viewportWidth);
    const double viewportHeight = static_cast<double>(currentFrame.frustum.viewportHeight);
    if (pixelX < 0.0 || pixelY < 0.0 || pixelX > viewportWidth || pixelY > viewportHeight)
        throw std::runtime_error("UICamera pick pixel is outside the viewport");

    const double ndcX = (2.0 * (pixelX + 0.5) / viewportWidth) - 1.0;
    const double ndcY = 1.0 - (2.0 * (pixelY + 0.5) / viewportHeight);
    const double horizontalScale = std::tan(currentFrame.frustum.horizontalFovRadians * 0.5);
    const double verticalScale = std::tan(currentFrame.frustum.verticalFovRadians * 0.5);

    UICameraRay ray;
    ray.originEcef = currentFrame.positionEcef;
    ray.directionEcef = normalizeOrThrow(
        currentFrame.directionEcef + currentFrame.rightEcef * ndcX * horizontalScale + currentFrame.upEcef * ndcY * verticalScale,
        "UICamera pick ray direction is degenerate");
    return ray;
}

std::optional<glm::dvec3> UICamera::pickWgs84EllipsoidEcef(double pixelX, double pixelY) const
{
    const UICameraRay ray = rayFromViewportPixel(pixelX, pixelY);
    const glm::dvec3 radii = wgs84().getRadii();
    const glm::dvec3 scaledOrigin = ray.originEcef / radii;
    const glm::dvec3 scaledDirection = ray.directionEcef / radii;

    const double a = glm::dot(scaledDirection, scaledDirection);
    const double b = 2.0 * glm::dot(scaledOrigin, scaledDirection);
    const double c = glm::dot(scaledOrigin, scaledOrigin) - 1.0;
    const double discriminant = b * b - 4.0 * a * c;
    if (discriminant < 0.0)
        return std::nullopt;

    const double sqrtDiscriminant = std::sqrt(discriminant);
    const double nearT = (-b - sqrtDiscriminant) / (2.0 * a);
    const double farT = (-b + sqrtDiscriminant) / (2.0 * a);
    double t = nearT;
    if (t < 0.0)
        t = farT;
    if (t < 0.0)
        return std::nullopt;

    return ray.originEcef + ray.directionEcef * t;
}

std::optional<CesiumGeospatial::Cartographic> UICamera::pickWgs84Cartographic(double pixelX, double pixelY) const
{
    const std::optional<glm::dvec3> hit = pickWgs84EllipsoidEcef(pixelX, pixelY);
    if (!hit)
        return std::nullopt;

    return wgs84().cartesianToCartographic(*hit);
}

glm::dvec3 UICamera::cartographicToEcef(const CesiumGeospatial::Cartographic& cartographic)
{
    return wgs84().cartographicToCartesian(cartographic);
}

CesiumGeospatial::Cartographic UICamera::ecefToCartographicOrThrow(const glm::dvec3& ecef)
{
    const std::optional<CesiumGeospatial::Cartographic> cartographic = wgs84().cartesianToCartographic(ecef);
    if (!cartographic)
        throw std::runtime_error("UICamera ECEF position cannot be converted to WGS84 cartographic coordinates");
    return *cartographic;
}

UICamera::Basis UICamera::targetBasis() const
{
    Basis basis;
    basis.up = normalizeOrThrow(wgs84().geodeticSurfaceNormal(homeEcef_), "UICamera home geodetic normal is degenerate");

    const glm::dvec3 worldZ{0.0, 0.0, 1.0};
    basis.east = glm::cross(worldZ, basis.up);
    if (glm::length(basis.east) < kRadiansEpsilon)
        basis.east = glm::dvec3{1.0, 0.0, 0.0};
    else
        basis.east = glm::normalize(basis.east);

    basis.north = glm::normalize(glm::cross(basis.up, basis.east));
    basis.target = homeEcef_ + basis.east * orbitTargetEastMeters_ + basis.north * orbitTargetNorthMeters_;

    basis.up = normalizeOrThrow(wgs84().geodeticSurfaceNormal(basis.target), "UICamera target geodetic normal is degenerate");
    basis.east = glm::cross(worldZ, basis.up);
    if (glm::length(basis.east) < kRadiansEpsilon)
        basis.east = glm::dvec3{1.0, 0.0, 0.0};
    else
        basis.east = glm::normalize(basis.east);
    basis.north = glm::normalize(glm::cross(basis.up, basis.east));
    return basis;
}

double UICamera::defaultOrbitDistanceMeters() const
{
    const double homeRadius = glm::length(homeEcef_);
    if (homeRadius <= 0.0)
        throw std::runtime_error("UICamera home ECEF radius is invalid");

    return homeRadius * 0.35;
}
