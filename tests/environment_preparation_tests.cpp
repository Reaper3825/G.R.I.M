#include "resources/models/GRIM-text/Shared/Environment/EnvironmentSession.hpp"
#include "ui/world_lab/environment_registry.hpp"

#include <iostream>
#include <chrono>
#include <stdexcept>

using nlohmann::json;
namespace fs = std::filesystem;
namespace {
int clients = 0, submissions = 0;
bool reject_begin = false;
json next_response, submitted;
void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
template<class F> void rejects(F action) {
    bool rejected = false;
    try { action(); } catch (const std::exception&) { rejected = true; }
    check(rejected, "Expected preparation failure");
    check(clients == 0, "Failed candidate leaked its client");
}
json response(const std::string& status = "active") {
    return {{"ok", true}, {"result", {{"environment_id", "content-fingerprint"},
        {"state", {{"feedback", "You are in the starting room."}, {"done", status != "active"},
            {"episode", {{"status", status}, {"identity", {{"source_id", "goal-a"},
                {"environment", "content-fingerprint"}}}}}}}}}};
}
}

// Replace only mailbox/process transport. Production preparation and registry
// resolution are compiled unchanged; no Docker or model execution occurs.
namespace GRIM::WorldLab {
struct TextWorldClient::Impl { bool pending = false; };
TextWorldClient::TextWorldClient() : impl_(std::make_unique<Impl>()) { ++clients; }
TextWorldClient::~TextWorldClient() { --clients; }
bool TextWorldClient::busy() const { return impl_->pending; }
bool TextWorldClient::begin(json request) {
    ++submissions; submitted = std::move(request);
    if (reject_begin) return false;
    impl_->pending = true; return true;
}
std::optional<json> TextWorldClient::poll() {
    if (!impl_->pending) return {};
    impl_->pending = false; return next_response;
}
void TextWorldClient::stop() { impl_->pending = false; }
}

int main() {
    const auto directory = fs::temp_directory_path() / ("grim-environment-preparation-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directory(directory);
    auto document = GRIM::WorldLab::Document::initial();
    document.environmentId = "environment-a";
    const auto path = directory / "world.grimworld.json";
    GRIM::WorldLab::saveJsonFile(path, document.json());
    GRIM::WorldLab::EnvironmentRegistry registry(directory);
    registry.registerFile(path);
    GRIM::ConceptBlockGoal goal;
    goal.target_state = "Reach the destination";
    goal.success_criteria.push_back({"Arrive", GRIM::EvidenceCondition{"player_at", document.spawn, ""}});
    std::vector<GRIM::ScheduledGoal> schedule{
        {"course-a", "environment-a", "goal-a", goal},
        {"course-b", "environment-b", "goal-b", goal}};
    using GRIM::Environment::prepareEnvironmentSession;
    for (const auto& status : {"active", "succeeded", "failed", "stopped"}) {
        next_response = response(status);
        const int before = submissions;
        auto owner = prepareEnvironmentSession(schedule, directory);
        check(owner->position() == 0 && owner->schedule().size() == 2, "Schedule/cursor lost");
        check(owner->currentAssignment().concept_block_id == "goal-a", "Wrong first Goal");
        check(owner->environment().environmentId == "environment-a", "Wrong environment");
        check(owner->environmentPath() == path, "Resolved path lost");
        check(owner->latestResponse() == next_response && !owner->client().busy(), "Initial response lost");
        check(submissions == before + 1 && submitted.at("op") == "compile", "Episode started twice");
        check(submitted.at("source_id") == "goal-a" && submitted.at("goal") == GRIM::goalToJson(goal),
              "Goal snapshot was not supplied independently");
        check(submitted.size() == 4 && !submitted.at("document").contains("solution"),
              "Unexpected authored state in request");
        auto moved = std::move(owner);
        check(clients == 1 && moved->latestResponse() == next_response, "Owner lifetime did not survive move");
    }
    check(clients == 0, "Owner did not release client");
    {
        next_response = response();
        auto owner = prepareEnvironmentSession(schedule, directory);
        const auto initial = owner->initialResponse();
        next_response["result"]["state"]["feedback"] = "Nothing changes.";
        const auto first = owner->submitCommand(10, "\n look \n");
        check(submitted.at("op") == "step" && submitted.at("command") == "look", "Command was not isolated/trimmed");
        const auto count = submissions;
        check(owner->submitCommand(10, "look") == first && submissions == count, "Command was dispatched twice");
        check(owner->initialResponse() == initial && owner->latestResponse() == first, "Initial/actual response not retained");
        check(owner->actions().size() == 1 && owner->actions()[0].status == "completed", "Missing action receipt");
        const auto invalid = [&](const std::string& command, std::size_t position) {
            bool rejected = false;
            try { owner->submitCommand(position, command); } catch (const std::exception&) { rejected = true; }
            check(rejected && submissions == count, "Invalid command reached the worker");
        };
        invalid("go north", 10); invalid("look", 9); invalid("look\ninventory", 11);
        invalid(std::string("look\0north", 10), 11); invalid(" ", 11); invalid(std::string(513, 'x'), 11);
        next_response = response("succeeded");
        owner->submitCommand(20, "go north");
        bool terminal_rejected = false;
        try { owner->submitCommand(30, "look"); } catch (const std::invalid_argument&) { terminal_rejected = true; }
        check(terminal_rejected && owner->actions().size() == 2, "Post-terminal action was accepted");
    }
    for (const auto& failure : std::vector<json>{
            {{"ok", false}, {"transport_error", true}, {"error", "timeout"}},
            {{"ok", true}, {"result", json::object()}}}) {
        next_response = response();
        auto owner = prepareEnvironmentSession(schedule, directory);
        next_response = failure;
        bool rejected = false;
        try { owner->submitCommand(10, "look"); } catch (const std::exception&) { rejected = true; }
        check(rejected && owner->actions()[0].status == "unknown", "Unknown outcome not retained");
        const auto count = submissions;
        for (const auto position : {10, 20}) {
            try { owner->submitCommand(position, "look"); } catch (const std::exception&) {}
        }
        check(submissions == count, "Uncertain action was retried or followed by another action");
    }
    {
        next_response = response();
        auto owner = prepareEnvironmentSession(schedule, directory);
        next_response["result"]["state"]["host_action"] = {{"status", "unknown"}};
        bool rejected = false;
        try { owner->submitCommand(10, "ui.show panel:settings"); } catch (const std::exception&) { rejected = true; }
        check(rejected && owner->actions()[0].status == "unknown", "Unknown host action was treated as completed");
    }
    rejects([&] { prepareEnvironmentSession({}, directory); });
    auto missing = schedule; missing[0].environment_id = "absent";
    const auto before_lookup = submissions;
    rejects([&] { prepareEnvironmentSession(missing, directory); });
    check(submissions == before_lookup, "Failed lookup contacted worker");
    reject_begin = true; rejects([&] { prepareEnvironmentSession(schedule, directory); }); reject_begin = false;
    for (const auto& invalid : std::vector<json>{
            {{"ok", false}, {"error", "binding rejected"}},
            {{"ok", false}, {"transport_error", true}, {"error", "worker unavailable"}},
            {{"ok", true}, {"result", json::object()}}, response("bogus")}) {
        next_response = invalid;
        rejects([&] { prepareEnvironmentSession(schedule, directory); });
    }
    next_response = response(); next_response["result"]["state"]["episode"]["identity"]["source_id"] = "wrong";
    rejects([&] { prepareEnvironmentSession(schedule, directory); });
    next_response = response(); next_response["result"]["state"]["done"] = true;
    rejects([&] { prepareEnvironmentSession(schedule, directory); });
    next_response = response(); next_response["result"]["environment_id"] = "wrong";
    rejects([&] { prepareEnvironmentSession(schedule, directory); });
    // Registry failures preserve authored files; tests remove only their own files.
    for (const auto& entry : fs::directory_iterator(directory)) fs::remove(entry.path());
    fs::remove(directory);
    std::cout << "Environment preparation, frozen handoff, terminal startup and failure cleanup passed\n";
}
