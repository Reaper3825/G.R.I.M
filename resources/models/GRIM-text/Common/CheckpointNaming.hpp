#pragma once

#include "../Shared/HyperParameters/HyperparameterEnums.hpp"

#include <optional>
#include <stdexcept>
#include <string>

namespace GRIM::Checkpoint {

inline std::string stageQualifiedCheckpointFilename(
    HyperParameters::TrainingStage stage,
    const std::string& suffix)
{
    if (stage == HyperParameters::TrainingStage::UNSPECIFIED) {
        throw std::runtime_error(
            "stageQualifiedCheckpointFilename: training stage is UNSPECIFIED");
    }
    if (suffix.empty() || suffix.front() != '_') {
        throw std::runtime_error(
            "stageQualifiedCheckpointFilename: suffix must begin with '_'");
    }

    return "checkpoint_" +
           std::string(HyperParameters::trainingStageToString(stage)) +
           suffix + ".grimckpt";
}

inline std::optional<HyperParameters::TrainingStage>
checkpointStageFromFilename(const std::string& filename)
{
    constexpr char prefix[] = "checkpoint_";
    constexpr char extension[] = ".grimckpt";
    constexpr std::size_t prefix_length = sizeof(prefix) - 1;
    constexpr std::size_t extension_length = sizeof(extension) - 1;

    if (filename.size() <= prefix_length + extension_length ||
        filename.compare(0, prefix_length, prefix) != 0 ||
        filename.compare(filename.size() - extension_length,
                         extension_length,
                         extension) != 0) {
        return std::nullopt;
    }

    const std::size_t stage_end = filename.find('_', prefix_length);
    if (stage_end == std::string::npos) {
        return std::nullopt;
    }

    const std::string stage_token =
        filename.substr(prefix_length, stage_end - prefix_length);
    if (stage_token == "PT") {
        return HyperParameters::TrainingStage::PT;
    }
    if (stage_token == "SFT") {
        return HyperParameters::TrainingStage::SFT;
    }
    if (stage_token == "DPO") {
        return HyperParameters::TrainingStage::DPO;
    }
    if (stage_token == "RLHF") {
        return HyperParameters::TrainingStage::RLHF;
    }
    return std::nullopt;
}

} // namespace GRIM::Checkpoint