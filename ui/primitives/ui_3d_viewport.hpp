#pragma once

#include "widget.hpp"

#include <functional>
#include <memory>

struct UI3DViewportGeometry {
    Vec2 logicalOrigin{0.0f, 0.0f};
    Vec2 logicalSize{0.0f, 0.0f};
    int pixelX = 0;
    int pixelY = 0;
    int pixelWidth = 0;
    int pixelHeight = 0;
    float scale = 1.0f;
    bool visible = false;
};

class UI3DViewportAttachment {
public:
    virtual ~UI3DViewportAttachment() = default;

    virtual void onViewportGeometryChanged(const UI3DViewportGeometry& geometry) = 0;
    virtual void onViewportDetached() {}
};

class UI3DViewport : public Widget {
public:
    using GeometryChangedCallback = std::function<void(const UI3DViewportGeometry&)>;

    UI3DViewport() = default;
    UI3DViewport(const Vec2& localPos, const Vec2& logicalSize, float scale = 1.0f);

    void setScale(float scale);
    float getScale() const { return scale_; }

    UI3DViewportGeometry getGeometry() const { return geometry_; }
    void setGeometryChangedCallback(GeometryChangedCallback callback);
    void attachViewport(std::shared_ptr<UI3DViewportAttachment> attachment);
    void detachViewport();
    std::shared_ptr<UI3DViewportAttachment> getAttachedViewport() const { return attachment_; }
    void syncViewportGeometry(const Vec2& panelPos, bool parentVisible);
    bool containsScreenPoint(float x, float y) const;

    void drawOverlay(OverlayRenderer& renderer, const Vec2& panelPos) override;

private:
    UI3DViewportGeometry computeGeometry(const Vec2& panelPos) const;
    void publishGeometryIfChanged(const UI3DViewportGeometry& geometry);
    static bool sameGeometry(const UI3DViewportGeometry& a, const UI3DViewportGeometry& b);

    float scale_ = 1.0f;
    UI3DViewportGeometry geometry_{};
    bool haveGeometry_ = false;
    GeometryChangedCallback onGeometryChanged_;
    std::shared_ptr<UI3DViewportAttachment> attachment_;
};