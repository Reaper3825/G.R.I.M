#include "ui/training_panel/ui_training_tools_view.hpp"
#include "commands/command_result.hpp"
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
    tool.usage_type = ToolUsage::Debug;
    tool.provider_name = "test_provider";
    tool.aliases = {"unique_alias"};
    registry.registerTool(tool, [](const std::string&) { return CommandResult{}; });
    for (int i = 0; i < 20; ++i) registry.registerSimple("z_viewer_" + std::to_string(i), "Scrolling entry", "test", [](const std::string&) { return CommandResult{}; });
    UITrainingToolsView view;
    UITrainingToolsView* activeView = &view;
    OverlayRenderer renderer;
    const PanelRect content{{0,0},{1000,400}};
    auto frame = [&](InputState input = {}, float dt = 0.0f) {
        auto& router = GRIM::UI::InputRouter::get();
        router.beginFrame(false); router.currentPanel = 1;
        activeView->update(input, dt, content, 1);
        renderedTestText.clear();
        activeView->draw(renderer, content);
        router.endFrame();
    };
    auto click = [&](float x, float y) {
        InputState input; input.mousePos = {x,y}; input.mousePressed[0] = input.mouseDown[0] = true;
        frame(input); frame();
    };
    frame();
    check(renderedTestText.find("21 shown / 21 registered") != std::string::npos, "Tools count missing");
    click(30,159);
    check(renderedTestText.find("Inspect metadata") != std::string::npos, "Tool row did not select metadata");
    check(renderedTestText.find("Usage type: Debug") != std::string::npos, "Tool usage classification missing");
    check(renderedTestText.find("Not documented") != std::string::npos, "Missing metadata was not identified");
    InputState wheel; wheel.mousePos = {50,200}; wheel.mouseWheelDelta = -1200;
    frame(wheel);
    click(30,159);
    check(renderedTestText.find("Scrolling entry") != std::string::npos, "Scrolled selection did not use scroll offset");
    click(30,45);
    InputState edit; edit.textInput = "UNIQUE_ALIAS"; frame(edit);
    check(renderedTestText.find("1 shown / 21 registered") != std::string::npos, "Alias search did not filter case insensitively");
    click(30,159);
    check(renderedTestText.find("Inspect metadata") != std::string::npos, "Filtered selection used incorrect row");
    registry.unregisterTool("viewer_test");
    frame({}, 2.0f);
    check(renderedTestText.find("No matching tools.") != std::string::npos, "Registry removal did not refresh search");
    check(renderedTestText.find("Inspect metadata") == std::string::npos, "Removed tool retained stale details");
    view.hide();
    check(GRIM::UI::InputRouter::get().focus.widget == 0, "Tools view retained focus after hiding");
    for (int i = 0; i < 20; ++i) registry.unregisterTool("z_viewer_" + std::to_string(i));

    // Exercise the real dropdowns, combined search, and changing registry providers.
    auto add = [&](const char* id, const char* provider, ToolUsage usage, const char* description,
                   ToolProviderType type = ToolProviderType::Plugin) {
        ToolDescriptor entry;
        entry.tool_id = id; entry.provider_name = provider; entry.provider_type = type;
        entry.usage_type = usage; entry.description = description; entry.category = "fixture";
        entry.aliases = {std::string(id) + "_alias"};
        check(registry.registerTool(entry, [](const std::string&) { return CommandResult{}; }), "Filter fixture registration failed");
    };
    add("alpha_debug", "Fixture Alpha Plugin", ToolUsage::Debug, "Alpha debug details");
    add("alpha_model", "Fixture Alpha Plugin", ToolUsage::Model, "Alpha model details");
    add("beta_system", "Fixture Beta Plugin", ToolUsage::System, "Beta system details");
    add("beta_ignore", "Fixture Beta Plugin", ToolUsage::Ignore, "Beta ignored details");
    add("host_debug", "builtin", ToolUsage::Debug, "Built-in debug details", ToolProviderType::Builtin);
    UITrainingToolsView filters;
    activeView = &filters;
    auto count = [&](int shown, int total) {
        check(renderedTestText.find(std::to_string(shown) + " shown / " + std::to_string(total) + " registered") != std::string::npos,
              "Combined filter count incorrect");
    };
    auto usage = [&](int index) { click(30,100); click(30,132 + 25.0f * index); };
    auto plugin = [&](int index) { click(520,100); click(520,132 + 25.0f * index); };
    auto search = [&](const std::string& text) {
        click(30,45);
        InputState select; select.ctrl = true; select.keyPressed['A'] = true; frame(select);
        InputState replace; replace.textInput = text;
        if (text.empty()) replace.keyPressed[VK_BACK] = true;
        frame(replace);
    };
    frame(); count(5,5);
    click(520,100);
    check(renderedTestText.find("Fixture Alpha Plugin") != std::string::npos &&
          renderedTestText.find("Fixture Beta Plugin") != std::string::npos &&
          renderedTestText.find("Built-in (host)") != std::string::npos, "Detected providers missing from dropdown");
    const auto popup = GRIM::UI::InputRouter::get().popup.widget;
    frame({},2.0f);
    check(popup && GRIM::UI::InputRouter::get().popup.widget == popup, "Unchanged refresh closed plugin dropdown");
    click(520,132); // All plugins; close the menu.
    usage(1); count(2,5); // Debug.
    check(renderedTestText.find("Alpha debug details") == std::string::npos, "Dropdown click selected the underlying tool row");
    plugin(2); count(1,5); // Alpha; built-in is the first provider.
    click(30,159);
    check(renderedTestText.find("Alpha debug details") != std::string::npos, "Combined filter row selection failed");
    search("ALPHA_DEBUG_ALIAS"); count(1,5);
    usage(2); count(0,5); // Model plus the alias search must exclude Debug.
    check(renderedTestText.find("Alpha debug details") == std::string::npos, "Filtered-out selection left stale details");
    search(""); count(1,5);
    add("aardvark_model", "Aardvark Plugin", ToolUsage::Model, "New provider details");
    frame({},2.0f); count(1,6);
    check(renderedTestText.find("alpha_model") != std::string::npos &&
          renderedTestText.find("aardvark_model") == std::string::npos, "Provider refresh preserved an index instead of identity");
    registry.unregisterTool("aardvark_model");
    frame({},2.0f); count(1,5);
    plugin(1); count(0,5); // Built-in + Model.
    usage(1); count(1,5); // Built-in + Debug.
    click(30,159);
    check(renderedTestText.find("Built-in debug details") != std::string::npos, "Built-in provider filter failed");
    plugin(3); count(0,5); // Beta + Debug.
    usage(3); count(1,5); // Beta + System.
    usage(4); count(1,5); // Beta + Ignore.
    click(30,159);
    check(renderedTestText.find("Beta ignored details") != std::string::npos, "Ignore filter selected wrong row");
    registry.unregisterTool("beta_system"); registry.unregisterTool("beta_ignore");
    frame({},2.0f); count(0,3);
    check(renderedTestText.find("All plugins") != std::string::npos &&
          renderedTestText.find("Beta ignored details") == std::string::npos, "Removed provider left stale filter or details");
    usage(0); count(3,3);
    click(520,100);
    check(renderedTestText.find("Fixture Beta Plugin") == std::string::npos, "Unloaded plugin retained a dropdown option");
    filters.hide();
    check(GRIM::UI::InputRouter::get().popup.widget == 0, "Tools view retained dropdown ownership after hiding");
    registry.unregisterTool("alpha_debug"); registry.unregisterTool("alpha_model"); registry.unregisterTool("host_debug");
}
