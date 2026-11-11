#include "core/applications/application_catalog.hpp"
#include "commands/command_result.hpp"
#include <iostream>
#include <set>
int main() try {
    const auto backend = GRIM::Applications::nativeBackend();
    const auto apps = backend.discover();
    std::set<std::string> identities;
    for (const auto& app : apps) {
        if (app.nativeId.empty() || app.name.empty() || app.launchTarget.empty()) throw std::runtime_error("Incomplete application record");
        identities.insert(app.identityNamespace + ":" + app.nativeId);
    }
    auto& registry = GRIM::MMO::ToolRegistry::instance();
    const GRIM::MMO::ProviderDescriptor provider{GRIM::MMO::ToolProviderType::Plugin, "Discovery smoke", "fixture", ""};
    if (!registry.publishProvider(provider, {})) throw std::runtime_error("Cannot publish smoke provider");
    GRIM::Applications::Catalog catalog(provider, {{backend.name, [&] { return apps; }}},
        [](const auto&) { throw std::runtime_error("Smoke test must never launch applications"); });
    auto page = catalog.request("refresh", nlohmann::json::object());
    if (!page.at("complete").get<bool>() || page.at("total") != identities.size() || registry.snapshot().resourceCount() != identities.size())
        throw std::runtime_error("Native discovery failed resource publication");
    for (;;) {
        if (page.dump().size() > 60000) throw std::runtime_error("Native catalog page too large");
        if (page.at("next_offset").is_null()) break;
        page = catalog.request("list", {{"offset", page.at("next_offset")}});
    }
    registry.unregisterInstance(provider.name, provider.instance);
    std::cout << backend.name << ": " << apps.size() << " application records, " << identities.size() << " distinct identities; no applications launched\n";
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
