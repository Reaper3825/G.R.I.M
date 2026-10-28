#include "ui/training_panel/ui_training_tools_view.hpp"
#include "ui/input/ui_input_router.hpp"
#include "ui/overlay_renderer.hpp"
#include "core/input_parser.hpp"
#include <stdexcept>
#include <string>

extern std::string renderedTestText;
void runTrainingToolsTests() {
    using namespace GRIM::MMO;
    auto check = [](bool value, const char* message) { if (!value) throw std::runtime_error(message); };
    auto& registry = ToolRegistry::instance();
    ToolDescriptor tool;
    tool.tool_id = "viewer_test";
    tool.description = "Inspect metadata";
    tool.category = "information";
    tool.provider_name = "test_provider";
    tool.aliases = {"unique_alias"};
    registry.registerTool(tool);
    for (int i = 0; i < 20; ++i) registry.registerSimple("z_viewer_" + std::to_string(i), "Scrolling entry", "test");
    UITrainingToolsView view;
    OverlayRenderer renderer;
    const PanelRect content{{0,0},{1000,400}};
    auto frame = [&](InputState input = {}, float dt = 0.0f) {
        auto& router = GRIM::UI::InputRouter::get();
        router.beginFrame(false); router.currentPanel = 1;
        view.update(input, dt, content, 1);
        renderedTestText.clear();
        view.draw(renderer, content);
        router.endFrame();
    };
    auto click = [&](float x, float y) {
        InputState input; input.mousePos = {x,y}; input.mousePressed[0] = input.mouseDown[0] = true;
        frame(input); frame();
    };
    frame();
    check(renderedTestText.find("21 shown / 21 registered") != std::string::npos, "Tools count missing");
    click(30,105);
    check(renderedTestText.find("Inspect metadata") != std::string::npos, "Tool row did not select metadata");
    check(renderedTestText.find("Not documented") != std::string::npos, "Missing metadata was not identified");
    InputState wheel; wheel.mousePos = {50,200}; wheel.mouseWheelDelta = -1200;
    frame(wheel);
    click(30,105);
    check(renderedTestText.find("Scrolling entry") != std::string::npos, "Scrolled selection did not use scroll offset");
    click(30,45);
    InputState edit; edit.textInput = "UNIQUE_ALIAS"; frame(edit);
    check(renderedTestText.find("1 shown / 21 registered") != std::string::npos, "Alias search did not filter case insensitively");
    click(30,105);
    check(renderedTestText.find("Inspect metadata") != std::string::npos, "Filtered selection used incorrect row");
    registry.unregisterTool("viewer_test");
    frame({}, 2.0f);
    check(renderedTestText.find("No matching tools.") != std::string::npos, "Registry removal did not refresh search");
    check(renderedTestText.find("Inspect metadata") == std::string::npos, "Removed tool retained stale details");
    view.hide();
    check(GRIM::UI::InputRouter::get().focus.widget == 0, "Tools view retained focus after hiding");
    for (int i = 0; i < 20; ++i) registry.unregisterTool("z_viewer_" + std::to_string(i));
}
