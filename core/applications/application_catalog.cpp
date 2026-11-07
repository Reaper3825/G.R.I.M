#include "application_catalog.hpp"
#include <iomanip>
#include <sstream>
#include <set>

namespace GRIM::Applications {
namespace {
bool valid(const std::string& s, size_t limit) {
    return !s.empty() && s.size() <= limit && s.find_first_of("\r\n\0", 0, 3) == std::string::npos;
}
std::string identity(const Application& a) {
    return a.identityNamespace + ":" + a.nativeId;
}
std::string resourceId(const Application& a) {
    // Fixed, deterministic encoding across processes; collisions are checked before publication.
    uint64_t hash = 14695981039346656037ULL;
    for (unsigned char c : identity(a)) { hash ^= c; hash *= 1099511628211ULL; }
    std::ostringstream out; out << "application:" << std::hex << std::setw(16) << std::setfill('0') << hash;
    return out.str();
}
}
Catalog::Catalog(MMO::ProviderDescriptor owner, std::vector<Backend> backends, Launcher launcher,
                 MMO::ToolRegistry& registry)
    : owner_(std::move(owner)), backends_(std::move(backends)), launcher_(std::move(launcher)), registry_(registry) {
    if (backends_.size() > 8 || !launcher_) throw std::invalid_argument("Invalid application catalog configuration");
    std::set<std::string> names;
    for (const auto& b : backends_)
        if (!valid(b.name, 256) || !b.discover || !names.insert(b.name).second)
            throw std::invalid_argument("Invalid or duplicate application backend");
}
MMO::ResourceDescriptor Catalog::resource(const std::string& id, const Entry& e) const {
    MMO::ResourceDescriptor r;
    r.provider_type = owner_.type; r.provider_name = owner_.name; r.provider_instance = owner_.instance;
    r.resource_id = id; r.kind = "application"; r.name = e.app.name; r.runtime_id = e.runtime;
    r.operations = {"applications.inspect", "applications.launch"};
    r.state = {{"identity_namespace", e.app.identityNamespace}, {"native_id", e.app.nativeId},
        {"launch_kind", e.app.launchKind}, {"launch_target", e.app.launchTarget},
        {"icon", e.app.icon}, {"sources", e.sources}, {"running_state", "unknown"}};
    return r;
}
void Catalog::refresh() {
    auto snapshots = backendSnapshots_;
    auto errors = nlohmann::json::array();
    for (const auto& b : backends_) {
        try {
            auto apps = b.discover();
            if (apps.size() > 10000) throw std::runtime_error("Application backend limit exceeded");
            for (const auto& a : apps) {
                if (!valid(a.identityNamespace, 128) || a.identityNamespace.find(':') != std::string::npos ||
                    !valid(a.nativeId, 8192) || !valid(a.name, 256) || !valid(a.launchKind, 128) ||
                    !valid(a.launchTarget, 8192) || a.icon.size() > 8192)
                    throw std::runtime_error("Invalid application record");
                // Validate UTF-8 and encoded size before anything reaches the live registry.
                if (nlohmann::json::array({a.identityNamespace, a.nativeId, a.name,
                        a.launchKind, a.launchTarget, a.icon}).dump().size() > 32000)
                    throw std::runtime_error("Oversized application metadata");
            }
            snapshots[b.name] = std::move(apps);
        } catch (const std::exception& e) {
            errors.push_back({{"backend", b.name}, {"message", std::string(e.what()).substr(0, 1024)},
                {"retained_previous", snapshots.count(b.name) != 0}});
        }
    }
    std::map<std::string, Entry> next;
    auto generation = nextRuntime_;
    // Backend order is precedence. Merge only identical native identities, never display names.
    for (const auto& b : backends_) {
        const auto observed = snapshots.find(b.name);
        if (observed == snapshots.end()) continue;
        for (const auto& a : observed->second) {
            const auto id = resourceId(a);
            auto [it, inserted] = next.try_emplace(id, Entry{a, {}, 0});
            if (!inserted && identity(it->second.app) != identity(a)) throw std::runtime_error("Application identity collision");
            auto& sources = it->second.sources;
            if (std::find(sources.begin(), sources.end(), b.name) == sources.end()) sources.push_back(b.name);
        }
    }
    if (next.size() > 9000) throw std::runtime_error("Application catalog limit exceeded");
    std::vector<MMO::ResourceDescriptor> resources;
    for (auto& [id, e] : next) {
        const auto old = entries_.find(id);
        e.runtime = old != entries_.end() && old->second.app == e.app && old->second.sources == e.sources
            ? old->second.runtime : generation++;
        resources.push_back(resource(id, e));
    }
    if (!registry_.replaceResources(owner_, "application", resources))
        throw std::runtime_error("Application provider is no longer current");
    backendSnapshots_ = std::move(snapshots); entries_ = std::move(next);
    nextRuntime_ = generation; errors_ = std::move(errors);
}
nlohmann::json Catalog::snapshot(size_t offset, size_t limit) const {
    auto apps = nlohmann::json::array();
    size_t index = 0, bytes = 0;
    for (const auto& [id, e] : entries_) {
        if (index++ < offset) continue;
        const nlohmann::json record = resource(id, e);
        const auto size = record.dump().size();
        if (apps.size() >= limit || (!apps.empty() && bytes + size > 48000)) break;
        apps.push_back(record); bytes += size;
    }
    const auto end = offset + apps.size();
    return {{"applications", apps}, {"total", entries_.size()},
        {"next_offset", end < entries_.size() ? nlohmann::json(end) : nlohmann::json(nullptr)},
        {"errors", errors_}, {"complete", errors_.empty()}};
}
nlohmann::json Catalog::request(const std::string& operation, const nlohmann::json& args) {
    std::lock_guard lock(mutex_);
    if (!args.is_object()) throw std::invalid_argument("Expected JSON object");
    if (operation == "refresh") {
        if (!args.empty()) throw std::invalid_argument("Refresh takes no arguments");
        refresh();
        return snapshot();
    }
    if (operation == "list") {
        for (auto it = args.begin(); it != args.end(); ++it)
            if ((it.key() != "offset" && it.key() != "limit") || !it.value().is_number_unsigned())
                throw std::invalid_argument("List accepts unsigned offset and limit only");
        const auto offset = args.value("offset", size_t{0}), limit = args.value("limit", size_t{25});
        if (offset > entries_.size() || !limit || limit > 50) throw std::invalid_argument("Invalid catalog page");
        return snapshot(offset, limit);
    }
    if (operation != "inspect" && operation != "launch") throw std::invalid_argument("Unknown application operation");
    if (args.size() != (operation == "launch" ? 3 : 1) || !args.contains("resource_id") || !args.at("resource_id").is_string())
        throw std::invalid_argument("Expected resource_id; launch also requires provider_instance and runtime_id");
    const auto id = args.at("resource_id").get<std::string>();
    const auto found = entries_.find(id);
    if (found == entries_.end()) throw std::invalid_argument("Unknown application resource; refresh discovery");
    if (operation == "launch") {
        const auto snapshot = registry_.snapshot();
        if (std::none_of(snapshot.providers.begin(), snapshot.providers.end(), [&](const auto& p) {
                return p.type == owner_.type && p.name == owner_.name && p.instance == owner_.instance;
            })) throw std::runtime_error("Application provider is no longer current");
        if (!args.contains("runtime_id") || !args.at("runtime_id").is_number_unsigned() ||
            args.at("runtime_id") != found->second.runtime || !args.contains("provider_instance") ||
            args.at("provider_instance") != owner_.instance)
            throw std::invalid_argument("Stale application identity; inspect the current resource");
        launcher_(found->second.app);
    }
    return {{"application", resource(id, found->second)}, {"launch_requested", operation == "launch"}};
}
}
