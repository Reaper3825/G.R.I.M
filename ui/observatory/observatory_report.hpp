#pragma once
#include "resources/models/GRIM-text/Shared/ModelConfig/CompiledModelConfig.hpp"
#include <nlohmann/json_fwd.hpp>
#include <filesystem>
#include <vector>
#include <string>

namespace GRIM::Observatory {
struct Candidate { int tokenId; std::string text; double probability; };
struct LayerReadout { unsigned layer; double entropyNats; std::vector<Candidate> candidates; };
struct PositionReadout { unsigned position; std::string inputText; std::vector<LayerReadout> layers; };
struct Report {
    std::string checkpoint, configSha256, mode, visibility;
    unsigned layerCount=0, dModel=0, vocabSize=0;
    bool synthetic=false;
    std::vector<PositionReadout> positions;
};
std::string configDigest(const Config::CompiledModelConfigSnapshot& config);
Report parseReport(const nlohmann::json& json, const Config::CompiledModelConfigSnapshot& config);
Report loadReport(const std::filesystem::path& path, const Config::CompiledModelConfigSnapshot& config);
Report makePreview(const Config::CompiledModelConfigSnapshot& config);
}
