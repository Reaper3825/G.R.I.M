#pragma once
#include "live_world_session.hpp"
#include "ui_world_canvas.hpp"
#include "ui/primitives/ui_button.hpp"
#include "ui/primitives/ui_dropdown.hpp"
#include "ui/primitives/ui_inputbox.hpp"
#include "ui/primitives/ui_textarea.hpp"
#include "ui/overlay_renderer.hpp"
#include "ui/primitives/ui_panel.hpp"
#include "core/input_parser.hpp"
#include <memory>

// Separate live view keeps the authored document independent of live topology.
class UILiveWorldView {
public:
    UILiveWorldView() {
        canvas_.selectionOnly = true;
        canvas_.radialPaths = true;
        transcript_.setReadOnly(true);
        command_.setPlaceholder("look, tools, back, go /builtin, inspect <tool_id>");
        command_.OnTextSubmitted.Bind([this](const std::string&) { submit(); });
        commands_ = std::make_unique<UIDropdown>("", std::vector<std::string>{"Available actions"}, 0,
            [this](int i, const std::string&) {
                if (i > 0 && i <= static_cast<int>(actions_.size())) {
                    const auto action = actions_[i - 1];
                    run([&] { session_.command(action); });
                }
                commands_->setSelectedIndex(0);
            });
        commands_->setFullWidth(true); commands_->setMaxVisibleItems(8);
        send_ = std::make_unique<UIButton>("Send", [this] { submit(); });
        exit_ = std::make_unique<UIButton>("World editor", [this] { if (onExit) onExit(); });
        fit_ = std::make_unique<UIButton>("Fit", [this] { canvas_.fitWorld(); });
        refresh();
    }
    std::function<void()> onExit;
    const GRIM::WorldLab::LiveWorldSession& session() const { return session_; }
    void update(const InputState& input, float dt, const PanelRect& area, uint64_t panel) {
        run([&] { session_.synchronize(); });
        layout(area);
        if (fitPending_) { canvas_.fitWorld(); fitPending_ = false; }
        const auto clipped = input.clipped(area.origin, area.size);
        for (auto* w : widgets()) w->setPanelID(panel);
        // Dropdowns own popup input before controls beneath them.
        auto* expanded = commands_->isExpanded() ? commands_.get() : nullptr;
        if (expanded) expanded->update(clipped, dt);
        if (commands_.get() != expanded) commands_->update(clipped, dt);
        for (auto* w : widgets()) if (w != commands_.get()) w->update(clipped, dt);
    }
    void draw(OverlayRenderer& renderer, const PanelRect& area) {
        layout(area);
        renderer.pushClipRect(area.origin, area.size);
        renderer.drawText({area.origin.x + 10, area.origin.y + 12}, "Live System World", UITheme::Colors::TextPrimary, 16);
        canvas_.drawOverlay(renderer, {});
        transcript_.drawOverlay(renderer, {});
        command_.drawOverlay(renderer, {});
        for (auto* b : {send_.get(), exit_.get(), fit_.get()}) b->drawOverlay(renderer, {});
        commands_->drawOverlay(renderer, {});
        renderer.drawText({area.origin.x + 10, area.origin.y + 50},
            std::to_string(session_.world().registry().providers.size()) + " provider rooms - populated automatically", UITheme::Colors::TextMuted, 13);
        renderer.drawText({area.origin.x + 10, area.origin.y + area.size.y - 22}, status_, UITheme::Colors::TextMuted, 12);
        // Dropdown headers and popup lists have separate render passes. Draw
        // expanded options last so the canvas/transcript cannot cover them.
        commands_->drawExpandedList(renderer, {});
        renderer.popClipRect();
    }
    void hide() {
        for (auto* w : widgets()) w->cancelInput();
        commands_->collapse();
    }
private:
    void submit() { run([&] { session_.command(command_.getText()); command_.clear(); }); }
    template<class F> void run(F operation) {
        try { operation(); refresh(); }
        catch (const std::exception& error) { status_ = error.what(); }
    }
    void refresh() {
        const auto& world = session_.world();
        if (!shownDocument_.is_null() && historySize_ == session_.history().size() &&
            shownRegistry_ == world.registry().revision) return;
        shownRegistry_ = world.registry().revision;
        const auto document = world.document().json();
        if (document != shownDocument_) {
            editor_.reset(world.document()); shownDocument_ = document;
            if (!world.document().find(canvas_.selected)) canvas_.selected = world.location();
            fitPending_ = true;
        }
        const auto observation = session_.observation();
        canvas_.setPlayState(observation);
        const auto actions = observation.at("admissible_commands").get<std::vector<std::string>>();
        if (actions != actions_) {
            actions_ = actions; std::vector<std::string> labels{"Available actions"}; labels.insert(labels.end(), actions.begin(), actions.end());
            commands_->collapse(); commands_->setItems(labels); commands_->setSelectedIndex(0);
        }
        if (historySize_ != session_.history().size()) {
            historySize_ = session_.history().size();
            // Show newest entries first while retaining the full chronological session history.
            std::string text;
            for (auto it = session_.history().rbegin(); it != session_.history().rend(); ++it)
                text += "> " + it->action + "\n" + it->feedback + "\n\n";
            transcript_.setText(text); transcript_.scrollToTop();
            status_ = "Session active. Provider rooms populate automatically without restarting. Tool inspection is read-only.";
        }
    }
    std::vector<Widget*> widgets() {
        return {commands_.get(), send_.get(), exit_.get(), fit_.get(), &command_, &transcript_, &canvas_};
    }
    static void place(Widget& w, float x, float y, float width, float height = 28) {
        w.setPosition(x, y); w.setSize(std::max(1.f, width), std::max(1.f, height));
    }
    void layout(const PanelRect& a) {
        const float x = a.origin.x + 10, y = a.origin.y, w = a.size.x - 20;
        place(*exit_, x + w - 120, y + 6, 120);
        place(*commands_, x, y + 80, w - 65); place(*fit_, x + w - 55, y + 80, 55);
        const float left = w * .45f, height = a.size.y - 180;
        place(canvas_, x, y + 118, left, height); place(transcript_, x + left + 10, y + 118, w - left - 10, height);
        place(command_, x, y + a.size.y - 54, w - 75); place(*send_, x + w - 65, y + a.size.y - 54, 65);
    }
    GRIM::WorldLab::LiveWorldSession session_;
    GRIM::WorldLab::Editor editor_;
    UIWorldCanvas canvas_{editor_};
    UITextArea transcript_{"", "", {}};
    UIInputBox command_;
    std::unique_ptr<UIDropdown> commands_;
    std::unique_ptr<UIButton> send_, exit_, fit_;
    std::vector<std::string> actions_;
    nlohmann::json shownDocument_;
    size_t historySize_ = 0;
    uint64_t shownRegistry_ = 0;
    bool fitPending_ = true;
    std::string status_;
};
