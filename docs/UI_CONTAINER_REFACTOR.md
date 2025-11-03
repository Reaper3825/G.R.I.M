# UI Container System Refactor Plan

## Current Architecture

### Hierarchy
```
UIRoot (singleton)
├─> UIPanel (has children vector)
│   ├─> UIScrollBox (has children vector)
│   │   ├─> UIButton
│   │   ├─> UISlider  
│   │   ├─> UIToggle
│   │   └─> UIDropdown
│   └─> Direct children (non-scrolled widgets)
└─> UIPanel (another panel)
```

### Existing Container Capabilities

1. **UIPanel** (`ui/ui_panel.hpp`)
   - ✅ Has `addChild(widget)` and `removeChild(widget)`
   - ✅ Has `children` vector
   - ✅ Updates all children in `update()`
   - ✅ Can draw children

2. **UIScrollBox** (`ui/ui_scrollbox.hpp`)
   - ✅ Has `addChild(widget)` and `removeChild(widget)`  
   - ✅ Has `children` vector
   - ✅ Has `autoLayoutChildren()` for automatic positioning
   - ✅ Handles scrolling with offset

3. **UIRoot** (`ui/ui_root.hpp`)
   - ✅ Manages panels as children
   - ✅ Uses OverlayRenderer for layered rendering
   - ✅ Handles z-order and visibility

## Problem with UISettingsMenu

**Current Implementation:**
```cpp
// Manual index-based button management
for (size_t i = 0; i < 4 && i < buttonsCopy.size(); ++i) {
    buttonsCopy[i]->setPosition(...);  // Manual positioning
    buttonsCopy[i]->update(...);       // Manual update
    yOffset += widgetHeight + 5;
}

// Special case for speaker button when coqui selected
if (i == 1 && !dropdownsCopy.empty()) {
    // Conditionally show dropdown
}
```

**Issues:**
- ❌ Hardcoded indices (button 0, 1, 2, 3, 4, 5...)
- ❌ Manual position calculations  
- ❌ Fragile when adding/removing widgets
- ❌ Not using container's `addChild()` properly
- ❌ Dropdown visibility handled manually

## Proposed Solution

### Use ScrollBox as True Container

**ScrollBox should own and manage all its children:**

```cpp
// In createWidgets():
scrollBox->clearChildren();

// Add buttons as children
auto backendButton = std::make_shared<UIButton>(...);
scrollBox->addChild(backendButton);

auto voiceButton = std::make_shared<UIButton>(...);
scrollBox->addChild(voiceButton);

// Conditionally add speaker button
if (voiceEngine == "coqui") {
    auto speakerButton = std::make_shared<UIButton>(...);
    scrollBox->addChild(speakerButton);
}

// Add sliders, toggles, etc.
scrollBox->addChild(temperatureSlider);
scrollBox->addChild(beamSlider);
scrollBox->addChild(suppressToggle);

// Auto-layout handles positioning
scrollBox->autoLayoutChildren();
```

**Benefits:**
- ✅ No manual indexing
- ✅ No manual positioning
- ✅ Widgets automatically update/render via parent
- ✅ Adding/removing widgets is trivial
- ✅ Clean, maintainable code

### Implementation Steps

1. **Remove manual widget arrays**
   ```cpp
   // DELETE these:
   std::vector<std::shared_ptr<UIButton>> buttons;
   std::vector<std::shared_ptr<UISlider>> sliders;
   std::vector<std::shared_ptr<UIToggle>> toggles;
   std::vector<std::shared_ptr<UIDropdown>> dropdowns;
   ```

2. **Store only what needs callbacks**
   ```cpp
   // Keep references only for widgets needing updates
   std::weak_ptr<UIButton> backendButton;
   std::weak_ptr<UIButton> speakerButton;
   // Or use tags/IDs to find them in scrollBox->getChildren()
   ```

3. **Refactor createWidgets()**
   ```cpp
   void UISettingsMenu::createWidgets() {
       scrollBox->clearChildren();
       
       // Create and add widgets directly to scrollBox
       scrollBox->addChild(makeBackendButton());
       scrollBox->addChild(makeVoiceButton());
       
       if (isCoqu enabled) {
           scrollBox->addChild(makeSpeakerButton());
       }
       
       scrollBox->addChild(makeModelButton());
       // ... etc
       
       scrollBox->autoLayoutChildren();  // Positions everything
   }
   ```

4. **Simplify update()**
   ```cpp
   void UISettingsMenu::update(InputState& input, float dt) {
       UIPanel::update(input, dt);  // Updates all children recursively
       
       // ScrollBox automatically updates its children
       // No manual iteration needed!
   }
   ```

5. **Simplify draw()**
   ```cpp
   void UISettingsMenu::drawOverlay(OverlayRenderer& renderer) {
       UIPanel::drawOverlay(renderer);  // Draws all children
       
       // ScrollBox automatically draws children with scroll offset
       // No manual rendering needed!
   }
   ```

## Widget Enhancement Needed

To make ALL widgets containable, we should add child management to the base `Widget` class:

### Option A: Add to Base Widget Class

```cpp
// helpers/widget.hpp
class Widget {
public:
    // NEW: Child management (optional, only used by containers)
    virtual void addChild(std::shared_ptr<Widget> child) {
        children.push_back(child);
        child->parent = this;
    }
    
    virtual void removeChild(Widget* child) {
        children.erase(
            std::remove_if(children.begin(), children.end(),
                [child](auto& ptr) { return ptr.get() == child; }),
            children.end()
        );
    }
    
    const std::vector<std::shared_ptr<Widget>>& getChildren() const {
        return children;
    }
    
    Widget* getParent() const { return parent; }
    
protected:
    std::vector<std::shared_ptr<Widget>> children;
    Widget* parent = nullptr;
};
```

**Pros:**
- ✅ Any widget can contain children
- ✅ Uniform API across all widgets
- ✅ Easy to create composite widgets

**Cons:**
- ❌ Memory overhead for leaf widgets (empty vectors)
- ❌ Might confuse usage (buttons with children?)

### Option B: Create UIContainer Base Class

```cpp
// ui/ui_container.hpp
class UIContainer : public Widget {
public:
    virtual void addChild(std::shared_ptr<Widget> child);
    virtual void removeChild(Widget* child);
    const std::vector<std::shared_ptr<Widget>>& getChildren() const;
    
protected:
    std::vector<std::shared_ptr<Widget>> children;
    virtual void updateChildren(InputState& input, float dt);
    virtual void drawChildren(OverlayRenderer& renderer);
};
```

Then:
```cpp
class UIPanel : public UIContainer { ... };
class UIScrollBox : public UIContainer { ... };
class UIButton : public Widget { ... };  // No children
```

**Pros:**
- ✅ Clear separation: containers vs widgets
- ✅ No memory waste on leaf widgets  
- ✅ Type-safe (can't add children to buttons)

**Cons:**
- ❌ Extra class in hierarchy
- ❌ Less flexible for future needs

## Recommendation

**Use Option B (UIContainer)** for cleaner architecture, then refactor UISettingsMenu to use proper parent/child relationships instead of manual arrays and indices.

### Migration Path

1. Create `ui/ui_container.hpp` base class
2. Make `UIPanel` and `UIScrollBox` inherit from `UIContainer`
3. Refactor `UISettingsMenu` to use `scrollBox->addChild()`
4. Remove manual widget vectors and position calculations
5. Test and verify all widgets render/update correctly

---

**Status:** Ready for implementation
**Priority:** High (fixes current button indexing bug)
**Effort:** ~2-3 hours
