//======================================================//
//  Quantization_GPU.hpp
//  Lightweight CUDA quantization helpers
//======================================================//

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace GRIM {
namespace Quantization {

enum class QuantizationPrecision : std::uint8_t {
	Int8 = 0
};

struct QuantizationConfig {
	QuantizationPrecision precision = QuantizationPrecision::Int8;
	float scale = 1.0f;              // Positive scaling factor (float -> quantized)
	int zero_point = 0;              // Zero-point offset for asymmetric quantization
	float clip_min = -127.0f;        // Clamp range prior to casting
	float clip_max = 127.0f;
	bool symmetric = true;           // When true zero_point is ignored during scaling
	cudaStream_t stream = nullptr;   // Optional default stream
};

class QuantizationLayer {
public:
	QuantizationLayer() = default;
	explicit QuantizationLayer(const QuantizationConfig& config);

	void setConfig(const QuantizationConfig& config);
	const QuantizationConfig& config() const noexcept { return config_; }

	// Quantizes `elements` floats into signed int8 values.
	void quantize(const float* input,
			   std::int8_t* output,
			   std::size_t elements,
			   cudaStream_t stream = nullptr) const;

	// Dequantizes int8 values back into float using the configured parameters.
	void dequantize(const std::int8_t* input,
			     float* output,
			     std::size_t elements,
			     cudaStream_t stream = nullptr) const;

	// Convenience helper that quantizes to int8 and immediately dequantizes
	// back to float (useful for calibration/error analysis).
	void quantizeAndDequantize(const float* input,
						    float* output,
						    std::size_t elements,
						    cudaStream_t stream = nullptr) const;

	// Utility to update scale/zero-point from observed min/max range.
	void updateScaleFromRange(float min_value, float max_value);

private:
	QuantizationConfig config_{};
};

} // namespace Quantization
} // namespace GRIM

