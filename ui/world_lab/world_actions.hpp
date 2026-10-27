#pragma once
#include "world_document.hpp"

namespace GRIM::WorldLab {
// Offline snapshot exported from the installed TextWorld knowledge base.
const nlohmann::json& actionCatalog();
nlohmann::json actionCatalog(const Document&);
bool actionTypeMatches(const std::string& type, const std::string& kind);
std::vector<std::string> actionCandidates(const Document&, const std::string& type);
std::string actionEntityName(const Document&, const std::string& id);
bool actionRelevant(const Document&, const nlohmann::json& action, const std::string& selection);
std::string composeAction(const Document&, const nlohmann::json& action,
                          const std::vector<std::string>& bindings);
}
