#include "ui_orbit_camera.hpp"

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/geometric.hpp>
#include <glm/matrix.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {
constexpr double pi = 3.14159265358979323846;
void finite(double value) {
    if (!std::isfinite(value)) throw std::invalid_argument("UIOrbitCamera requires finite values");
}
void finite(const glm::dvec3& value) { finite(value.x); finite(value.y); finite(value.z); }
}

void UIOrbitCamera::setViewportSize(int width, int height) {
    if (width <= 0 || height <= 0) throw std::invalid_argument("UIOrbitCamera viewport must be positive");
    width_ = width; height_ = height;
}

void UIOrbitCamera::setProjection(double fov, double nearPlane, double farPlane, bool homogeneousDepth) {
    finite(fov); finite(nearPlane); finite(farPlane);
    if (fov <= 0 || fov >= pi || nearPlane <= 0 || farPlane <= nearPlane)
        throw std::invalid_argument("UIOrbitCamera invalid projection");
    fov_ = fov; near_ = nearPlane; far_ = farPlane; homogeneousDepth_ = homogeneousDepth;
}

void UIOrbitCamera::setTarget(const glm::dvec3& target) { finite(target); target_ = target; }

void UIOrbitCamera::setDistanceLimits(double minimum, double maximum) {
    finite(minimum); finite(maximum);
    if (minimum <= 0 || maximum < minimum) throw std::invalid_argument("UIOrbitCamera invalid distance limits");
    minDistance_ = minimum; maxDistance_ = maximum;
    distance_ = std::clamp(distance_, minDistance_, maxDistance_);
}

void UIOrbitCamera::setDistance(double distance) {
    finite(distance);
    if (distance <= 0) throw std::invalid_argument("UIOrbitCamera distance must be positive");
    distance_ = std::clamp(distance, minDistance_, maxDistance_);
}

void UIOrbitCamera::setOrbit(double yaw, double pitch) {
    finite(yaw); finite(pitch);
    yaw_ = std::remainder(yaw, 2*pi);
    pitch_ = std::clamp(pitch, -pi/2 + 0.001, pi/2 - 0.001);
}

void UIOrbitCamera::orbitByPixels(double dx, double dy) {
    finite(dx); finite(dy);
    setOrbit(yaw_ - dx * 0.005, pitch_ + dy * 0.005);
}

UIOrbitCameraFrame UIOrbitCamera::frame() const {
    UIOrbitCameraFrame result;
    result.target = target_;
    result.position = target_ + distance_ * glm::dvec3(
        std::cos(pitch_)*std::sin(yaw_), std::sin(pitch_), std::cos(pitch_)*std::cos(yaw_));
    const auto forward = glm::normalize(target_ - result.position);
    result.right = glm::normalize(glm::cross(forward, glm::dvec3(0,1,0)));
    result.up = glm::cross(result.right, forward);
    result.view = glm::lookAtRH(result.position, target_, result.up);
    const double aspect = double(width_) / height_;
    result.projection = homogeneousDepth_ ? glm::perspectiveRH_NO(fov_, aspect, near_, far_)
                                         : glm::perspectiveRH_ZO(fov_, aspect, near_, far_);
    return result;
}

void UIOrbitCamera::panByPixels(double dx, double dy) {
    finite(dx); finite(dy);
    const auto current = frame();
    const double scale = 2 * distance_ * std::tan(fov_/2) / height_;
    setTarget(target_ + scale * (-dx*current.right + dy*current.up));
}

void UIOrbitCamera::zoomBySteps(double steps) {
    finite(steps);
    const double logarithm = std::clamp(std::log(distance_) - steps*0.12,
                                       std::log(minDistance_), std::log(maxDistance_));
    distance_ = std::exp(logarithm);
}

void UIOrbitCamera::fitBounds(const glm::dvec3& minimum, const glm::dvec3& maximum) {
    finite(minimum); finite(maximum);
    if (minimum.x > maximum.x || minimum.y > maximum.y || minimum.z > maximum.z)
        throw std::invalid_argument("UIOrbitCamera bounds must be ordered");
    const glm::dvec3 extent = maximum - minimum;
    finite(extent);
    const double radius = std::max(0.001, glm::length(extent)*0.5);
    const double halfAngle = std::min(fov_/2, std::atan(std::tan(fov_/2)*width_/height_));
    const double distance = radius / std::sin(halfAngle) * 1.1;
    finite(distance * 100.0);
    setTarget(minimum + extent*0.5);
    setDistanceLimits(radius*0.01, distance*100);
    setDistance(distance);
    near_ = radius*0.001;
    far_ = distance*101 + radius;
}

UIOrbitCameraRay UIOrbitCamera::rayFromViewportPixel(double x, double y) const {
    finite(x); finite(y);
    if (x < 0 || y < 0 || x > width_ || y > height_)
        throw std::invalid_argument("UIOrbitCamera pixel outside viewport");
    const auto current = frame();
    const double nx = 2*x/width_ - 1, ny = 1 - 2*y/height_;
    const double tanHalf = std::tan(fov_/2);
    const auto forward = glm::normalize(current.target-current.position);
    return {current.position, glm::normalize(forward + nx*tanHalf*width_/height_*current.right + ny*tanHalf*current.up)};
}

std::optional<glm::dvec3> UIOrbitCamera::projectPoint(const glm::dvec3& point) const {
    finite(point);
    const auto current = frame();
    const auto clip = current.projection * current.view * glm::dvec4(point,1);
    if (clip.w <= 0) return std::nullopt;
    const glm::dvec3 ndc = glm::dvec3(clip) / clip.w;
    const double depth = homogeneousDepth_ ? (ndc.z+1)*0.5 : ndc.z;
    if (std::abs(ndc.x)>1 || std::abs(ndc.y)>1 || depth<0 || depth>1) return std::nullopt;
    return glm::dvec3((ndc.x+1)*width_*0.5, (1-ndc.y)*height_*0.5, depth);
}
