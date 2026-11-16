#pragma once

#include <CesiumGeospatial/Cartographic.h>

#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include <optional>

struct UICameraFrustum {
    double horizontalFovRadians = 1.0471975511965976;
    double verticalFovRadians = 0.7853981633974483;
    double nearPlaneMeters = 1.0;
    double farPlaneMeters = 1000000000.0;
    int viewportWidth = 1280;
    int viewportHeight = 720;
};

struct UICameraZoomLimits {
    double minDistanceMeters = 500.0;
    double maxDistanceMeters = 25000000.0;
};

struct UICameraRay {
    glm::dvec3 originEcef{0.0, 0.0, 0.0};
    glm::dvec3 directionEcef{0.0, 0.0, -1.0};
};

struct UICameraFrame {
    glm::dvec3 targetEcef{0.0, 0.0, 0.0};
    glm::dvec3 positionEcef{0.0, 0.0, 0.0};
    glm::dvec3 directionEcef{0.0, 0.0, -1.0};
    glm::dvec3 upEcef{0.0, 0.0, 1.0};
    glm::dvec3 rightEcef{1.0, 0.0, 0.0};
    CesiumGeospatial::Cartographic targetCartographic{0.0, 0.0, 0.0};
    CesiumGeospatial::Cartographic cameraCartographic{0.0, 0.0, 0.0};
    UICameraFrustum frustum{};
    glm::dmat4 view{1.0};
    glm::dmat4 projection{1.0};
    double yawRadians = 0.0;
    double pitchRadians = 0.0;
    double orbitDistanceMeters = 0.0;
    double altitudeMeters = 0.0;
};

class UICamera {
public:
    UICamera();

    void setHomeCartographic(const CesiumGeospatial::Cartographic& cartographic);
    void setHomeEcef(const glm::dvec3& ecef);

    const glm::dvec3& homeEcef() const { return homeEcef_; }
    CesiumGeospatial::Cartographic homeCartographic() const;

    void setFrustum(const UICameraFrustum& frustum);
    const UICameraFrustum& frustum() const { return frustum_; }
    void setViewportSize(int width, int height);

    void setZoomLimits(const UICameraZoomLimits& limits);
    const UICameraZoomLimits& zoomLimits() const { return zoomLimits_; }

    void resetOrbit();
    void recenterHome();
    void setOrbitDistanceMeters(double distanceMeters);
    void orbitByRadians(double yawDeltaRadians, double pitchDeltaRadians);
    void panByMeters(double eastMeters, double northMeters);
    void orbitByPixels(double deltaX, double deltaY, bool fineControl);
    void panByPixels(double deltaX, double deltaY, bool fineControl);
    void zoomBySteps(double wheelSteps);
    void clampOrbitDistance();

    UICameraFrame frame() const;
    UICameraRay rayFromViewportPixel(double pixelX, double pixelY) const;
    std::optional<glm::dvec3> pickWgs84EllipsoidEcef(double pixelX, double pixelY) const;
    std::optional<CesiumGeospatial::Cartographic> pickWgs84Cartographic(double pixelX, double pixelY) const;

    double yawRadians() const { return yawRadians_; }
    double pitchRadians() const { return pitchRadians_; }
    double orbitDistanceMeters() const { return orbitDistanceMeters_; }
    double orbitTargetEastMeters() const { return orbitTargetEastMeters_; }
    double orbitTargetNorthMeters() const { return orbitTargetNorthMeters_; }

    static glm::dvec3 cartographicToEcef(const CesiumGeospatial::Cartographic& cartographic);
    static CesiumGeospatial::Cartographic ecefToCartographicOrThrow(const glm::dvec3& ecef);

private:
    struct Basis {
        glm::dvec3 target{0.0, 0.0, 0.0};
        glm::dvec3 east{1.0, 0.0, 0.0};
        glm::dvec3 north{0.0, 1.0, 0.0};
        glm::dvec3 up{0.0, 0.0, 1.0};
    };

    Basis targetBasis() const;
    double defaultOrbitDistanceMeters() const;

    glm::dvec3 homeEcef_{0.0, 0.0, 0.0};
    double yawRadians_ = 0.0;
    double pitchRadians_ = 1.25;
    double orbitDistanceMeters_ = 0.0;
    double orbitTargetEastMeters_ = 0.0;
    double orbitTargetNorthMeters_ = 0.0;
    UICameraFrustum frustum_{};
    UICameraZoomLimits zoomLimits_{};
};
