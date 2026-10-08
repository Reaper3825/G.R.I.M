#pragma once

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <optional>

// Cartesian, right-handed, Y-up camera for data scenes. No geographic units.
struct UIOrbitCameraFrame {
    glm::dvec3 position;
    glm::dvec3 target;
    glm::dvec3 right;
    glm::dvec3 up;
    glm::dmat4 view;
    glm::dmat4 projection;
};

struct UIOrbitCameraRay {
    glm::dvec3 origin;
    glm::dvec3 direction;
};

class UIOrbitCamera {
public:
    void setViewportSize(int width, int height);
    // Pass bgfx::getCaps()->homogeneousDepth at the renderer boundary.
    void setProjection(double verticalFovRadians, double nearPlane, double farPlane,
                       bool homogeneousDepth);
    void setTarget(const glm::dvec3& target);
    void setDistanceLimits(double minimum, double maximum);
    void setDistance(double distance);
    void setOrbit(double yawRadians, double pitchRadians);
    void orbitByPixels(double dx, double dy);
    void panByPixels(double dx, double dy);
    void zoomBySteps(double steps);
    // Fits all corners; also updates distance limits and clipping planes.
    void fitBounds(const glm::dvec3& minimum, const glm::dvec3& maximum);

    UIOrbitCameraFrame frame() const;
    UIOrbitCameraRay rayFromViewportPixel(double x, double y) const;
    // Top-left viewport coordinates, depth in [0,1]; null outside the frustum.
    std::optional<glm::dvec3> projectPoint(const glm::dvec3& point) const;
    double distance() const { return distance_; }

private:
    glm::dvec3 target_{0.0};
    double yaw_ = 0.45, pitch_ = 0.25, distance_ = 5.0;
    double minDistance_ = 0.001, maxDistance_ = 1.0e6;
    double fov_ = 0.7853981633974483, near_ = 0.001, far_ = 1.0e7;
    int width_ = 1, height_ = 1;
    bool homogeneousDepth_ = false;
};
