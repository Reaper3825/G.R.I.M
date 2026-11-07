#include "ui/world_lab/ui_live_world_view.hpp"
#include "ui/world_lab/textworld_client.hpp"
#include "ui/input/ui_input_router.hpp"
#include "core/input_parser.hpp"
#include <thread>
#include <iostream>
#include <cmath>

extern std::string renderedTestText;
extern Vec2 liveFixtureRoomLabel, liveFixtureTileLabel;

namespace {
void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
}
void runLiveWorldUITests() {
    using namespace GRIM::MMO;
    auto& registry = ToolRegistry::instance();
    const ProviderDescriptor provider{ToolProviderType::Plugin, "Live UI fixture", "first", "fixture.dll"};
    const auto initialRooms = registry.snapshot().providers.size() + 1;
    UILiveWorldView view;
    auto& router = GRIM::UI::InputRouter::get();
    OverlayRenderer renderer;
    const PanelRect area{{0, 0}, {1000, 700}};
    auto frame = [&](const InputState& input) {
        router.beginFrame(); router.currentPanel = 91;
        view.update(input, 0, area, 91);
        renderedTestText.clear(); view.draw(renderer, area); router.endFrame();
    };
    auto click = [&](float x, float y) {
        InputState input; input.mousePos = {x, y}; input.mousePressed[0] = input.mouseDown[0] = true; frame(input);
        input.mousePressed[0] = input.mouseDown[0] = false; input.mouseReleased[0] = true; frame(input);
    };
    frame({});
    check(view.session().world().document().entities.size() == initialRooms, "Live UI startup projection mismatch");
    check(registry.publishProvider(provider, {}), "UI fixture registration failed");
    frame({});
    check(view.session().world().document().entities.size() == initialRooms + 1, "Published provider room did not appear automatically");
    check(renderedTestText.find("Live UI fixture") != std::string::npos, "Automatic provider room not drawn");
    const auto originalLabel = liveFixtureRoomLabel;
    const float dividerX = 10 + (area.size.x - 20 - 12) * .45f + 6;
    InputState resize;
    resize.mousePos = {dividerX, 220}; resize.mousePressed[0] = resize.mouseDown[0] = true; frame(resize);
    resize.mousePressed[0] = false; resize.mousePos.x += 80; frame(resize);
    resize.mouseDown[0] = false; resize.mouseReleased[0] = true; frame(resize);
    click(950, 94); // Refit to the enlarged map viewport.
    check(std::abs(liveFixtureRoomLabel.x - originalLabel.x) > 1,
          "Live preview divider did not resize the map beside the viewer");
    InputState resetSplit;
    resetSplit.mousePos = {dividerX + 80, 220}; resetSplit.mousePressed[1] = resetSplit.mouseDown[1] = true; frame(resetSplit);
    frame({}); click(950, 94);
    check(std::abs(liveFixtureRoomLabel.x - originalLabel.x) < 1,
          "Live preview divider right-click did not restore its default split");
    ResourceDescriptor item;
    item.provider_name = provider.name;
    item.provider_instance = provider.instance;
    item.kind = "application";
    for (int i = 0; i < 60; ++i) {
        item.resource_id = "app:" + std::to_string(i);
        item.name = "Fixture app " + std::to_string(i);
        item.runtime_id = i + 1;
        check(registry.publishResource(item), "Fixture resource publication failed");
    }
    frame({});
    check(renderedTestText.find("Fixture app 0") == std::string::npos, "Unselected room leaked resources onto the map");
    click(liveFixtureRoomLabel.x + 2, liveFixtureRoomLabel.y + 2);
    check(view.session().world().location() == "system_hub" &&
          renderedTestText.find("Live UI fixture  (60 resources)") != std::string::npos &&
          renderedTestText.find("Fixture app 0") != std::string::npos, "Room click failed to preview its resource grid");
    const auto originalTile = liveFixtureTileLabel;
    const float resourceDividerY = 118 + (area.size.y - 180) * .60f - 6;
    InputState resizeResources;
    resizeResources.mousePos = {700, resourceDividerY};
    resizeResources.mousePressed[0] = resizeResources.mouseDown[0] = true; frame(resizeResources);
    resizeResources.mousePressed[0] = false; resizeResources.mousePos.y -= 60; frame(resizeResources);
    resizeResources.mouseDown[0] = false; resizeResources.mouseReleased[0] = true; frame(resizeResources);
    check(std::abs(liveFixtureTileLabel.y - (originalTile.y - 60)) < 1,
          "Resource divider did not enlarge the bottom viewer");
    InputState resetResources;
    resetResources.mousePos = {700, resourceDividerY - 60};
    resetResources.mousePressed[1] = resetResources.mouseDown[1] = true; frame(resetResources);
    frame({});
    check(std::abs(liveFixtureTileLabel.y - originalTile.y) < 1,
          "Resource divider right-click did not restore its default height");
    click(liveFixtureTileLabel.x + 2, liveFixtureTileLabel.y + 2);
    check(renderedTestText.find("\"kind\": \"application\"") != std::string::npos, "Grid tile failed to inspect previewed resource");
    const auto path = view.session().world().providerPath(GRIM::WorldLab::providerKey(provider));
    const auto actions = view.session().observation().at("admissible_commands").get<std::vector<std::string>>();
    const auto actionIndex = std::find(actions.begin(), actions.end(), "go " + path) - actions.begin();
    check(actionIndex < 7, "Navigation option outside visible test dropdown");
    click(100, 93);
    check(renderedTestText.find("go /Live-UI-fixture") != std::string::npos,
        "Expanded action menu did not draw navigation options");
    click(100, 108 + static_cast<float>(actionIndex + 1) * 25 + 12);
    check(view.session().world().location() != "system_hub", "Available action did not enter provider room");
    check(renderedTestText.find("Live UI fixture  (60 resources)") != std::string::npos, "Entered room grid missing");
    registry.removeResource(item);
    frame({});
    check(renderedTestText.find("Live UI fixture  (59 resources)") != std::string::npos, "Grid missed resource removal");
    const auto identity = view.session().observation().at("session_id");
    const auto history = view.session().history().size();
    view.hide();
    registry.unregisterInstance(provider.name, provider.instance);
    frame({});
    check(view.session().world().location() == "system_hub" && view.session().world().document().entities.size() == initialRooms,
        "Provider unload while hidden did not reconcile on return");
    check(view.session().history().size() > history && view.session().observation().at("session_id") == identity,
        "Live view lost session/history after hiding");
    view.hide();

    // Stop must complete even when its response is not polled immediately. No
    // request is started here, so this test never launches Docker or TextWorld.
    GRIM::WorldLab::TextWorldClient client;
    client.stop(); client.stop();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    std::optional<nlohmann::json> response;
    for (int i = 0; i < 100 && !response; ++i) {
        response = client.poll();
        if (!response) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    check(response && response->at("result").at("stopped") == true && !client.busy(), "Stop did not produce one completed response");
    std::cout << "Live world UI automatic population, navigation, hidden unload and stop tests passed\n";
}
