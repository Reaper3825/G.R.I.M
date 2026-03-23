#pragma once
#include "Serialization_views.hpp"
#include "Serialization_requests.hpp"
#include "Serialization_validate.hpp"
#include "../grim_layer_gpu.hpp"

namespace GRIM {

class SerializationLayer final : public Layer<SerializationLayer, float> {
public:
	static constexpr LayerType layer_type = LayerType::kSerialization;

	SerializationLayer() = default;
	explicit SerializationLayer(SerializationConfig config);

	void setConfig(const SerializationConfig& config);
	const SerializationConfig& config() const noexcept { return config_; }

	bool save(const SerializationSaveRequest& request);
	bool load(SerializationLoadRequest& request);

private:
	SerializationConfig config_{};
};

} // namespace GRIM
