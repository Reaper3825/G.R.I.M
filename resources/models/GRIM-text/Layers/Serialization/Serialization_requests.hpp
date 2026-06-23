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
	bool requires_number_encoder = false;
	bool requires_execution_block = false;
	bool requires_final_rms_gamma = false;
	bool requires_arg_selector = false;
};

struct SerializationLoadReport {
	bool number_encoder_loaded = false;
	bool execution_block_loaded = false;
	bool arg_selector_loaded = false;
};

struct SerializationSaveSources {
	SerializationModelConfigView config;
	SerializationGpuEmbeddingReadView gpu_embedding;
	std::vector<SerializationEncoderLayerReadView> encoder_layers;
	SerializationLMHeadReadView lm_head;
	SerializationNumberEncoderReadView number_encoder;
	SerializationExecutionBlockReadView execution_block;
	SerializationArgSelectorReadView arg_selector;
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
	SerializationNumberEncoderWriteView number_encoder;
	SerializationExecutionBlockWriteView execution_block;
	SerializationArgSelectorWriteView arg_selector;
	DeviceWriteView final_rms_gamma;
};

} // namespace GRIM
