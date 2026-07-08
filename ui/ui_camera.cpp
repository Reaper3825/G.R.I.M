#include "ui_camera.hpp"

#include <CesiumGeospatial/Ellipsoid.h>

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/geometric.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {
    constexpr double kDefaultOrbitPitchRadians = 1.25;
    constexpr double kMinOrbitPitchRadians = 0.18;
    constexpr double kMaxOrbitPitchRadians = 1.48;
    constexpr double kDefaultMinOrbitDistanceMeters = 500.0;
    constexpr double kMaxOrbitDistanceMultiplier = 5.0;
    constexpr double kOrbitRadiansPerPixel = 0.0045;
    constexpr double kFineOrbitRadiansPerPixel = 0.0015;
    constexpr double kWheelZoomStep = 0.12;
    constexpr double kRadiansEpsilon = 1.0e-8;

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
    clampOrbit();
}

void UICamera::resetOrbit()
{
    yawRadians_ = 0.0;
    pitchRadians_ = kDefaultOrbitPitchRadians;
    orbitDistanceMeters_ = defaultOrbitDistanceMeters();
    orbitTargetEastMeters_ = 0.0;
    orbitTargetNorthMeters_ = 0.0;

    zoomLimits_.minDistanceMeters = kDefaultMinOrbitDistanceMeters;
    zoomLimits_.maxDistanceMeters = orbitDistanceMeters_ * kMaxOrbitDistanceMultiplier;
    clampOrbit();
}

void UICamera::recenterHome()
{
    orbitTargetEastMeters_ = 0.0;
    orbitTargetNorthMeters_ = 0.0;
    clampOrbit();
}

void UICamera::orbitByRadians(double yawDeltaRadians, double pitchDeltaRadians)
{
    yawRadians_ += yawDeltaRadians;
    pitchRadians_ += pitchDeltaRadians;
    clampOrbit();
}

void UICamera::panByMeters(double eastMeters, double northMeters)
{
    orbitTargetEastMeters_ += eastMeters;
    orbitTargetNorthMeters_ += northMeters;
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
    clampOrbit();
}

void UICamera::clampOrbit()
{
    pitchRadians_ = std::clamp(pitchRadians_, kMinOrbitPitchRadians, kMaxOrbitPitchRadians);
    orbitDistanceMeters_ = std::clamp(
        orbitDistanceMeters_,
        zoomLimits_.minDistanceMeters,
        zoomLimits_.maxDistanceMeters);
}

UICameraFrame UICamera::frame() const
{
    if (orbitDistanceMeters_ <= 0.0)
        throw std::runtime_error("UICamera orbit distance must be positive before frame evaluation");

    const Basis basis = targetBasis();
    const glm::dvec3 tangentDirection = normalizeOrThrow(
        std::cos(yawRadians_) * basis.north + std::sin(yawRadians_) * basis.east,
        "UICamera tangent direction is degenerate");
    const glm::dvec3 cameraOffsetDirection = normalizeOrThrow(
        std::cos(pitchRadians_) * tangentDirection + std::sin(pitchRadians_) * basis.up,
        "UICamera offset direction is degenerate");
    const glm::dvec3 cameraPosition = basis.target + cameraOffsetDirection * orbitDistanceMeters_;
    const glm::dvec3 direction = normalizeOrThrow(basis.target - cameraPosition, "UICamera direction is degenerate");

    glm::dvec3 right = glm::cross(direction, basis.up);
    if (glm::length(right) < kRadiansEpsilon)
        right = basis.east;
    else
        right = glm::normalize(right);
    const glm::dvec3 up = glm::normalize(glm::cross(right, direction));

    const double aspect = static_cast<double>(frustum_.viewportWidth) / static_cast<double>(frustum_.viewportHeight);
    UICameraFrame result;
    result.targetEcef = basis.target;
    result.positionEcef = cameraPosition;
    result.directionEcef = direction;
    result.upEcef = up;
    result.rightEcef = right;
    result.targetCartographic = ecefToCartographicOrThrow(basis.target);
    result.cameraCartographic = ecefToCartographicOrThrow(cameraPosition);
    result.frustum = frustum_;
    result.view = glm::lookAt(cameraPosition, basis.target, up);
    result.projection = glm::perspective(frustum_.verticalFovRadians, aspect, frustum_.nearPlaneMeters, frustum_.farPlaneMeters);
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
