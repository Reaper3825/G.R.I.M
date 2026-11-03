#pragma once
#include "widget.hpp"
#include <vector>
#include <memory>

// Equal-sized tiles in row-major order. The caller owns the viewport size;
// scrolling is composed with UIScrollBox rather than implemented here.
class UIGrid : public Widget {
public:
    explicit UIGrid(size_t columns = 1);
    void addWidget(std::shared_ptr<Widget> child);
    void removeWidget(Widget* child);
    void clearWidgets();
    void setColumns(size_t columns); // Fixed count, at least one.
    void setAutoColumns(float minimumTileWidth); // Responsive column count.
    void setSpacing(float horizontal, float vertical);
    void setPadding(float padding);
    void setRowHeight(float height); // Fixed row height; may exceed viewport.
    void setFitRows(bool fit); // Distribute rows across available height.
    float getContentHeight() const { return contentHeight; }
    size_t getColumnCount() const { return actualColumns; }
    void layout(); // Also called before every update and draw.
    void update(const InputState&, float) override;
    void draw(UIRenderer&) override;
    void drawOverlay(OverlayRenderer&, const Vec2&) override;
    void cancelInput() override;
private:
    std::vector<std::shared_ptr<Widget>> widgets;
    size_t columns=1, actualColumns=1;
    float minimumWidth=0, gapX=10, gapY=10, padding=0, rowHeight=40, contentHeight=0;
    bool fitRows=false;
};
