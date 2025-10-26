#include "ui_manager.hpp"
#include "input_parser.hpp"

void UIManager::add(std::shared_ptr<Widget> w) {
    widgets.push_back(std::move(w));
}

void UIManager::remove(Widget* w) {
    widgets.erase(std::remove_if(widgets.begin(), widgets.end(),
        [&](const std::shared_ptr<Widget>& ptr) { return ptr.get() == w; }),
        widgets.end());
}

void UIManager::update(float dt) {
    for (auto& w : widgets)
        if (w->isVisible())
            w->update(input, dt);
}

void UIManager::draw() {
    if (!renderer) return;
    for (auto& w : widgets)
        if (w->isVisible())
            w->draw(*renderer);
}
