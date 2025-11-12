//======================================================//
//  UI Training Controller Implementation
//  Implementation of UI training controller utilities
//  
//  Author: GRIM Development Team
//  Date: November 12, 2025
//======================================================//

#include "ui_training_controller.hpp"
#include <sstream>
#include <iomanip>

namespace GRIM {
namespace UI {

//======================================================//
//  Training Time Formatting Utilities
//======================================================//

std::string formatTrainingTime(std::chrono::seconds duration) {
    auto hours = std::chrono::duration_cast<std::chrono::hours>(duration);
    duration -= hours;
    auto minutes = std::chrono::duration_cast<std::chrono::minutes>(duration);
    duration -= minutes;
    auto seconds = duration;
    
    std::ostringstream oss;
    
    if (hours.count() > 0) {
        oss << hours.count() << "h " << minutes.count() << "m " << seconds.count() << "s";
    } else if (minutes.count() > 0) {
        oss << minutes.count() << "m " << seconds.count() << "s";
    } else {
        oss << seconds.count() << "s";
    }
    
    return oss.str();
}

std::string formatTrainingProgress(float progress) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1) << (progress * 100.0f) << "%";
    return oss.str();
}

std::string formatMetric(float value, const std::string& unit) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4) << value;
    if (!unit.empty()) {
        oss << " " << unit;
    }
    return oss.str();
}

//======================================================//
//  Training Configuration Validation
//======================================================//

struct ValidationResult {
    bool isValid = true;
    std::string errorMessage;
};

ValidationResult validateTrainingConfig(const GRIMText::TrainingConfig& config) {
    ValidationResult result;
    
    // Validate epochs
    if (config.epochs < 1 || config.epochs > 1000) {
        result.isValid = false;
        result.errorMessage = "Epochs must be between 1 and 1000";
        return result;
    }
    
    // Validate batch size
    if (config.batchSize < 1 || config.batchSize > 128) {
        result.isValid = false;
        result.errorMessage = "Batch size must be between 1 and 128";
        return result;
    }
    
    // Validate learning rate
    if (config.learningRate <= 0.0f || config.learningRate > 1.0f) {
        result.isValid = false;
        result.errorMessage = "Learning rate must be between 0 and 1";
        return result;
    }
    
    // Validate max sequence length
    if (config.maxSeqLen < 128 || config.maxSeqLen > 32768) {
        result.isValid = false;
        result.errorMessage = "Max sequence length must be between 128 and 32768";
        return result;
    }
    
    // Validate warmup steps
    if (config.warmupSteps < 0 || config.warmupSteps > 10000) {
        result.isValid = false;
        result.errorMessage = "Warmup steps must be between 0 and 10000";
        return result;
    }
    
    // Validate paths (must be set)
    if (config.dataPath.empty()) {
        result.isValid = false;
        result.errorMessage = "Training data path is required";
        return result;
    }
    
    return result;
}

//======================================================//
//  Training Statistics Summary
//======================================================//

struct TrainingStatsSummary {
    std::string epochInfo;
    std::string batchInfo;
    std::string lossInfo;
    std::string performanceInfo;
    std::string progressInfo;
    std::string timeInfo;
};

TrainingStatsSummary summarizeTrainingStats(const GRIMText::TrainingStats& stats) {
    TrainingStatsSummary summary;
    
    // Epoch info
    std::ostringstream epochStream;
    epochStream << "Epoch " << stats.currentEpoch << "/" << stats.totalEpochs;
    summary.epochInfo = epochStream.str();
    
    // Batch info
    if (stats.totalBatches > 0) {
        std::ostringstream batchStream;
        batchStream << "Batch " << stats.currentBatch << "/" << stats.totalBatches;
        summary.batchInfo = batchStream.str();
    }
    
    // Loss info
    std::ostringstream lossStream;
    lossStream << "Loss: " << std::fixed << std::setprecision(4) << stats.currentLoss;
    if (stats.avgLoss > 0.0f) {
        lossStream << " (Avg: " << std::fixed << std::setprecision(4) << stats.avgLoss << ")";
    }
    summary.lossInfo = lossStream.str();
    
    // Performance info
    std::ostringstream perfStream;
    perfStream << std::fixed << std::setprecision(0) << stats.tokensPerSec << " tokens/sec";
    if (stats.perplexity > 0.0f) {
        perfStream << " | Perplexity: " << std::fixed << std::setprecision(2) << stats.perplexity;
    }
    summary.performanceInfo = perfStream.str();
    
    // Progress info
    summary.progressInfo = formatTrainingProgress(stats.trainingProgress);
    
    // Time info
    if (stats.elapsedTime > 0) {
        summary.timeInfo = formatTrainingTime(std::chrono::seconds(stats.elapsedTime));
    }
    
    return summary;
}

} // namespace UI
} // namespace GRIM
