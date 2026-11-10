#pragma once

#include "DataCollection/dataset_creation_service.hpp"
#include "DataCollection/data_structurer.hpp"
#include "DataCollection/huggingface_webhook.hpp"
#include "DataCollection/pipeline/pipeline_orchestrator.hpp"

namespace GRIM::DataCollection {
struct DatasetServices {
    std::shared_ptr<GRIM::Pipeline::PipelineOrchestrator> pipeline;
    std::shared_ptr<HuggingFaceWebhook> huggingFace;
    std::shared_ptr<DataStructurer> structurer;
    std::shared_ptr<DatasetCreationService> creation;
};
std::shared_ptr<DatasetServices> runtimeDatasetServices();
}

void bootstrapDatasetServices();
void shutdownDatasetServices();
