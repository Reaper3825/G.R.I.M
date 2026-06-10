#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "Serialization_views.hpp"

namespace GRIM {

struct SerializationConfig {
	bool use_gpu = true;
	bool atomic_write = true;
	std::string temp_suffix = ".tmp";
};

struct CheckpointCapabilityRequirements {
	bool requires_execution_block = false;
	bool requires_final_rms_gamma = false;
};

struct SerializationLoadReport {
	bool execution_block_loaded = false;
};

struct SerializationSaveSources {
	SerializationModelConfigView config;
	SerializationGpuEmbeddingReadView gpu_embedding;
	std::vector<SerializationEncoderLayerReadView> encoder_layers;
	SerializationLMHeadReadView lm_head;
	SerializationExecutionBlockReadView execution_block;
	DeviceReadView final_rms_gamma;
};

struct SerializationSaveRequest {
	std::string path;
	std::uint32_t model_version = 0;
	SerializationSaveSources sources;
};

struct SerializationLoadRequest {
	std::string path;
	SerializationModelConfigView config;
	CheckpointCapabilityRequirements capabilities;
	SerializationLoadReport report;
	SerializationGpuEmbeddingWriteView gpu_embedding;
	std::vector<SerializationEncoderLayerWriteView> encoder_layers;
	SerializationLMHeadWriteView lm_head;
	SerializationExecutionBlockWriteView execution_block;
	DeviceWriteView final_rms_gamma;
};

} // namespace GRIM
