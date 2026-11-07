#include "core/applications/application_catalog.hpp"
#include "commands/command_result.hpp"
#include <iostream>
#include <set>

using namespace GRIM::Applications;
using namespace GRIM::MMO;
using Json = nlohmann::json;
void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
template<class F> void rejects(F fn) { bool threw = false; try { fn(); } catch (const std::exception&) { threw = true; } check(threw, "Expected rejection"); }
int main() try {
    auto& registry = ToolRegistry::instance();
    ProviderDescriptor owner{ToolProviderType::Plugin, "Applications fixture", "one", ""};
    check(registry.publishProvider(owner, {}), "publish");
    Application first{"fixture", "a", "Same display name", "fixture", "first", ""};
    Application second{"fixture", "b", "Same display name", "fixture", "second", ""};
    std::vector<Application> primary{first, second}, secondary{first};
    bool fail = false; int launches = 0;
    Catalog catalog(owner, {{"primary", [&] { if (fail) throw std::runtime_error("offline"); return primary; }},
        {"secondary", [&] { return secondary; }}}, [&](const Application& app) { check(app.nativeId == "a", "Wrong launch target"); ++launches; });
    auto state = catalog.request("refresh", Json::object());
    check(state.at("complete") && state.at("applications").size() == 2, "dedup by identity, not name");
    Json selected;
    for (const auto& a : state.at("applications")) if (a.at("state").at("native_id") == "a") selected = a;
    check(selected.at("state").at("sources").size() == 2, "missing provenance");
    const auto id = selected.at("resource_id");
    const auto runtime = selected.at("runtime_id");
    Json launch{{"resource_id", id}, {"runtime_id", runtime}, {"provider_instance", owner.instance}};
    const auto revision = registry.resourceVersion();
    catalog.request("refresh", Json::object());
    check(registry.resourceVersion() == revision, "unchanged refresh churned resources");
    catalog.request("launch", launch); check(launches == 1, "launch not dispatched");
    rejects([&] { catalog.request("launch", {{"resource_id", id}}); });
    auto wrongOwner = launch; wrongOwner["provider_instance"] = "old";
    rejects([&] { catalog.request("launch", wrongOwner); });
    primary[0].launchTarget = "changed";
    catalog.request("refresh", Json::object());
    rejects([&] { catalog.request("launch", launch); });
    check(launches == 1, "stale launch reached OS");
    fail = true; secondary.clear();
    state = catalog.request("refresh", Json::object());
    check(!state.at("complete").get<bool>() && state.at("applications").size() == 2, "failure erased last good results");
    fail = false; primary.clear();
    check(catalog.request("refresh", Json::object()).at("applications").empty(), "successful removal stale");
    primary = {first};
    state = catalog.request("refresh", Json::object());
    check(state.at("applications").at(0).at("resource_id") == id && state.at("applications").at(0).at("runtime_id") != runtime,
          "recreated identity/generation incorrect");
    auto malformed = registry.snapshot().resources;
    malformed.push_back(malformed.front());
    const auto before = registry.resourceVersion();
    rejects([&] { registry.replaceResources(owner, "application", malformed); });
    check(registry.resourceVersion() == before && registry.snapshot().resources.size() == 1, "invalid batch partially published");
    auto replacement = owner; replacement.instance = "two";
    const auto latest = state.at("applications").at(0);
    auto retiredLaunch = Json{{"resource_id", latest.at("resource_id")}, {"runtime_id", latest.at("runtime_id")}, {"provider_instance", owner.instance}};
    check(registry.publishProvider(replacement, {}, owner.instance), "replacement");
    check(registry.snapshot().resources.empty(), "replacement retained old resources");
    rejects([&] { catalog.request("refresh", Json::object()); });
    rejects([&] { catalog.request("launch", retiredLaunch); });
    check(registry.snapshot().resources.empty(), "retired catalog republished");
    registry.unregisterInstance(replacement.name, replacement.instance);
    owner.instance = "three";
    check(registry.publishProvider(owner, {}), "pagination provider");
    primary.clear();
    for (int i = 0; i < 180; ++i) primary.push_back({"fixture", std::to_string(i), "Application " + std::to_string(i), "fixture", "target", ""});
    Catalog paged(owner, {{"many", [&] { return primary; }}}, [](const auto&) {});
    auto page = paged.request("refresh", Json::object());
    check(page.at("applications").size() == 25 && page.at("total") == 180, "default pagination missing");
    std::set<std::string> observed;
    for (;;) {
        check(page.dump().size() < 65536, "catalog page exceeds plugin result bound");
        for (const auto& a : page.at("applications")) check(observed.insert(a.at("resource_id")).second, "duplicate catalog page entry");
        if (page.at("next_offset").is_null()) break;
        page = paged.request("list", {{"offset", page.at("next_offset")}});
    }
    check(observed.size() == 180 && registry.snapshot().resources.size() == 180, "pagination hid registered resources");
    registry.unregisterInstance(owner.name, owner.instance);
    owner.instance = "four";
    check(registry.publishProvider(owner, {}), "failed backend provider");
    Catalog unavailable(owner, {{"unavailable", []() -> std::vector<Application> { throw std::runtime_error("offline"); }}}, [](const auto&) {});
    for (int i = 0; i < 2; ++i) {
        const auto failed = unavailable.request("refresh", Json::object());
        check(!failed.at("complete").get<bool>() && !failed.at("errors").at(0).at("retained_previous").get<bool>(),
              "Never-successful backend reported a previous snapshot");
    }
    registry.unregisterInstance(owner.name, owner.instance);
    std::cout << "Application catalog identity, refresh, failure retention, generation guards and atomic publication passed\n";
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
