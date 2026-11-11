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
#include "ui/primitives/ui_grid.hpp"
#include "ui/primitives/ui_scrollbox.hpp"
#include "ui/primitives/ui_split_pane.hpp"

// Separate live view keeps the authored document independent of live topology.
class UILiveWorldView {
public:
    explicit UILiveWorldView(std::filesystem::path navigationPath = {})
        : session_(GRIM::MMO::ProviderRegistry::instance(), std::move(navigationPath)) {
        canvas_.selectionOnly = true;
        canvas_.radialPaths = true;
        canvas_.showRoomLabels = false;
        canvas_.nodeStyle.titleSize = 18.f;
        grid_->setAutoColumns(165);
        grid_->setSpacing(6, 6);
        grid_->setRowHeight(32);
        scroll_.addChild(grid_);
        canvas_.onSelect = [this](const std::string& id) {
            if (!editor_.document().find(id)) return false;
            showRoom(id);
            return true;
        };
        transcript_.setReadOnly(true);
        command_.setPlaceholder("look, tools, resources, inspect <object>");
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
        grid_->setPanelID(panel);
        // Dropdowns own popup input before controls beneath them.
        auto* expanded = commands_->isExpanded() ? commands_.get() : nullptr;
        if (expanded) expanded->update(clipped, dt);
        if (commands_.get() != expanded) commands_->update(clipped, dt);
        resourceSplit_.update(clipped, dt);
        layout(area);
        previewSplit_.update(clipped, dt);
        layout(area);
        for (auto* w : widgets())
            if (w != commands_.get() && w != &previewSplit_ && w != &resourceSplit_) w->update(clipped, dt);
    }
    void draw(OverlayRenderer& renderer, const PanelRect& area) {
        layout(area);
        renderer.pushClipRect(area.origin, area.size);
        renderer.drawText({area.origin.x + 10, area.origin.y + 12}, "Live System World", UITheme::Colors::TextPrimary, 16);
        canvas_.drawOverlay(renderer, {});
        transcript_.drawOverlay(renderer, {});
        previewSplit_.drawOverlay(renderer, {});
        resourceSplit_.drawOverlay(renderer, {});
        renderer.drawText({scroll_.getPosition().x, scroll_.getPosition().y - 18},
            gridTitle_, UITheme::Colors::TextPrimary, 14);
        scroll_.drawOverlay(renderer, {});
        command_.drawOverlay(renderer, {});
        for (auto* b : {send_.get(), exit_.get(), fit_.get()}) b->drawOverlay(renderer, {});
        commands_->drawOverlay(renderer, {});
        renderer.drawText({area.origin.x + 10, area.origin.y + 50},
            std::to_string(session_.world().document().paths.size() - 1) + " providers - " +
            std::to_string(session_.world().registry().resourceCount()) + " live resources", UITheme::Colors::TextMuted, 13);
        renderer.drawText({area.origin.x + 10, area.origin.y + area.size.y - 22}, status_, UITheme::Colors::TextMuted, 12);
        if (!session_.navigationPersistenceError().empty())
            renderer.drawText({area.origin.x + 10, area.origin.y + 66},
                "Navigation snapshot: " + session_.navigationPersistenceError(), UITheme::Colors::Warning, 11);
        // Dropdown headers and popup lists have separate render passes. Draw
        // expanded options last so the canvas/transcript cannot cover them.
        commands_->drawExpandedList(renderer, {});
        renderer.popClipRect();
    }
    void hide() {
        for (auto* w : widgets()) w->cancelInput();
        grid_->cancelInput();
        commands_->collapse();
    }
private:
    class ResourceTile : public UIButton {
    public:
        using UIButton::UIButton;
        void drawOverlay(OverlayRenderer& renderer, const Vec2& panelPos) override {
            renderer.pushClipRect(getPosition(), getSize());
            UIButton::drawOverlay(renderer, panelPos);
            renderer.popClipRect();
        }
    };
    void showRoom(const std::string& id) {
        const auto& world = session_.world();
        const bool moved = id != gridRoom_;
        gridRoom_ = id;
        std::vector<const GRIM::MMO::ResourceDescriptor*> resources;
        if (const auto* group = world.providerInRoom(id))
            for (const auto& resource : group->resources) resources.push_back(&resource);
        std::sort(resources.begin(), resources.end(), [](const auto* a, const auto* b) {
            return std::tie(a->name, a->resource_id) < std::tie(b->name, b->resource_id);
        });
        std::vector<std::pair<std::string, std::string>> tiles;
        for (const auto* item : resources) tiles.emplace_back(item->resource_id, item->name);
        if (!moved && tiles == gridTiles_) return;
        gridTiles_ = std::move(tiles);
        gridTitle_ = world.document().find(id)->name + "  (" + std::to_string(resources.size()) + " resources)";
        grid_->clearWidgets();
        for (const auto* resource : resources) {
            grid_->addWidget(std::make_shared<ResourceTile>(resource->name, [this, key = resource->resource_id] {
                const auto* group = session_.world().providerInRoom(gridRoom_);
                if (!group) return;
                for (const auto& item : group->resources) {
                    if (item.resource_id != key) continue;
                    transcript_.setText(GRIM::WorldLab::resourceInspectionFeedback({item, group->provider}));
                    transcript_.scrollToTop();
                    break;
                }
            }));
        }
        if (moved) scroll_.scrollTo(0);
    }
    void submit() { run([&] { session_.command(command_.getText()); command_.clear(); }); }
    template<class F> void run(F operation) {
        try { operation(); refresh(); }
        catch (const std::exception& error) { status_ = error.what(); }
    }
    void refresh() {
        const auto& world = session_.world();
        if (!shownDocument_.is_null() && historySize_ == session_.history().size() &&
            shownRegistry_ == world.registry().revision && shownResources_ == world.registry().resource_revision) return;
        const bool resourcesChanged = shownResources_ != world.registry().resource_revision;
        shownRegistry_ = world.registry().revision;
        shownResources_ = world.registry().resource_revision;
        // The editor-backed canvas gets a room-only copy. Full resources stay in the live world.
        auto map = world.document();
        std::erase_if(map.entities, [](const auto& entity) { return entity.kind != "room"; });
        const auto document = map.json();
        if (document != shownDocument_) {
            editor_.reset(map);
            shownDocument_ = document;
            fitPending_ = true;
        }
        if (lastLocation_ != world.location() || !map.find(gridRoom_)) {
            lastLocation_ = world.location();
            canvas_.selected = lastLocation_;
            showRoom(lastLocation_);
        } else if (resourcesChanged) showRoom(gridRoom_);
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
        return {commands_.get(), send_.get(), exit_.get(), fit_.get(), &command_, &transcript_, &canvas_, &scroll_, &previewSplit_, &resourceSplit_};
    }
    static void place(Widget& w, float x, float y, float width, float height = 28) {
        w.setPosition(x, y); w.setSize(std::max(1.f, width), std::max(1.f, height));
    }
    void layout(const PanelRect& a) {
        const float x = a.origin.x + 10, y = a.origin.y, w = a.size.x - 20;
        place(*exit_, x + w - 120, y + 6, 120);
        place(*commands_, x, y + 80, w - 65); place(*fit_, x + w - 55, y + 80, 55);
        const float height = std::max(0.f, a.size.y - 180);
        resourceSplit_.setPosition(x, y + 118);
        resourceSplit_.setSize(std::max(0.f, w), height);
        resourceSplit_.setMinimumSizes(140, 100);
        resourceSplit_.setDefaultSecondSize(height * .40f);
        const auto preview = resourceSplit_.firstRect(), resources = resourceSplit_.secondRect();
        previewSplit_.setPosition(preview.origin);
        previewSplit_.setSize(preview.size);
        previewSplit_.setMinimumSizes(160, 200);
        previewSplit_.setDefaultSecondSize(std::max(0.f, w - 12) * .55f);
        const auto map = previewSplit_.firstRect(), viewer = previewSplit_.secondRect();
        canvas_.setPosition(map.origin); canvas_.setSize(map.size);
        transcript_.setPosition(viewer.origin); transcript_.setSize(viewer.size);
        scroll_.setPosition(resources.origin.x, resources.origin.y + std::min(22.f, resources.size.y));
        scroll_.setSize(resources.size.x, std::max(0.f, resources.size.y - 22));
        grid_->setSize(std::max(1.f, w - 16), scroll_.getSize().y);
        grid_->layout();
        const float content = grid_->getContentHeight();
        grid_->setSize(std::max(1.f, w - 16), std::max(content, scroll_.getSize().y));
        scroll_.setContentHeight(content);
        place(command_, x, y + a.size.y - 54, w - 75); place(*send_, x + w - 65, y + a.size.y - 54, 65);
    }
    GRIM::WorldLab::LiveWorldSession session_;
    GRIM::WorldLab::Editor editor_;
    UIWorldCanvas canvas_{editor_};
    UISplitPane previewSplit_{UISplitPane::Axis::Horizontal};
    UISplitPane resourceSplit_{UISplitPane::Axis::Vertical};
    UITextArea transcript_{"", "", {}};
    UIInputBox command_;
    UIScrollBox scroll_;
    std::shared_ptr<UIGrid> grid_ = std::make_shared<UIGrid>();
    std::string gridRoom_, gridTitle_, lastLocation_;
    std::vector<std::pair<std::string, std::string>> gridTiles_;
    std::unique_ptr<UIDropdown> commands_;
    std::unique_ptr<UIButton> send_, exit_, fit_;
    std::vector<std::string> actions_;
    nlohmann::json shownDocument_;
    size_t historySize_ = 0;
    uint64_t shownRegistry_ = 0;
    uint64_t shownResources_ = 0;
    bool fitPending_ = true;
    std::string status_;
};
