//======================================================//
//  Quantization_GPU.cu
//  CUDA kernels for simple per-tensor INT8 quantization
//======================================================//

#include "Quantization_GPU.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

constexpr int kQuantThreads = 256;

inline void checkCuda(const char* expr, cudaError_t status, const char* file, int line) {
	if (status == cudaSuccess) {
		return;
	}
	throw std::runtime_error(std::string("Quantization CUDA error at ") + file + ":" +
				       std::to_string(line) + " (" + expr + "): " +
				       cudaGetErrorString(status));
}

#define QUANT_CUDA_CHECK(expr) checkCuda(#expr, (expr), __FILE__, __LINE__)

__global__ void quantizeKernel(const float* __restrict__ input,
					   std::int8_t* __restrict__ output,
					   std::size_t elements,
					   float scale,
					   int zero_point,
					   float clip_min,
					   float clip_max) {
	const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	if (idx >= elements) {
		return;
	}
	const float transformed = input[idx] / scale + static_cast<float>(zero_point);
	const float clipped = fminf(clip_max, fmaxf(clip_min, nearbyintf(transformed)));
	output[idx] = static_cast<std::int8_t>(clipped);
}

__global__ void dequantizeKernel(const std::int8_t* __restrict__ input,
					 float* __restrict__ output,
					 std::size_t elements,
					 float scale,
					 int zero_point) {
	const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	if (idx >= elements) {
		return;
	}
	const float shifted = static_cast<float>(static_cast<int>(input[idx]) - zero_point);
	output[idx] = shifted * scale;
}

__global__ void quantizeDequantizeKernel(const float* __restrict__ input,
							 float* __restrict__ output,
							 std::size_t elements,
							 float scale,
							 int zero_point,
							 float clip_min,
							 float clip_max) {
	const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	if (idx >= elements) {
		return;
	}
	float transformed = input[idx] / scale + static_cast<float>(zero_point);
	transformed = fminf(clip_max, fmaxf(clip_min, nearbyintf(transformed)));
	const float dequantized = (transformed - static_cast<float>(zero_point)) * scale;
	output[idx] = dequantized;
}

inline dim3 makeGrid(std::size_t elements) {
	const std::size_t blocks = (elements + kQuantThreads - 1) / kQuantThreads;
	const std::size_t capped_blocks = std::max<std::size_t>(1, blocks);
	return dim3(static_cast<unsigned int>(capped_blocks));
}

} // namespace

namespace GRIM {
namespace Quantization {

namespace {

inline void validateConfig(const QuantizationConfig& config) {
	if (config.precision != QuantizationPrecision::Int8) {
		throw std::invalid_argument("QuantizationLayer currently supports only INT8 precision");
	}
	if (!(config.clip_min < config.clip_max)) {
		throw std::invalid_argument("QuantizationLayer clip range must satisfy min < max");
	}
	if (!(config.scale > 0.0f) || !std::isfinite(config.scale)) {
		throw std::invalid_argument("QuantizationLayer scale must be a positive finite value");
	}
	if (!std::isfinite(config.clip_min) || !std::isfinite(config.clip_max)) {
		throw std::invalid_argument("QuantizationLayer clip bounds must be finite values");
	}
	if (config.symmetric && config.clip_max <= 0.0f) {
		throw std::invalid_argument("QuantizationLayer symmetric mode requires positive clip_max");
	}
}

} // namespace

QuantizationLayer::QuantizationLayer(const QuantizationConfig& config) {
	setConfig(config);
}

void QuantizationLayer::setConfig(const QuantizationConfig& config) {
	validateConfig(config);
	config_ = config;
	if (config_.symmetric) {
		config_.zero_point = 0;
		config_.clip_max = std::fabs(config_.clip_max);
		config_.clip_min = -config_.clip_max;
	}
}

void QuantizationLayer::quantize(const float* input,
				 std::int8_t* output,
				 std::size_t elements,
				 cudaStream_t stream) const {
	if (!input || !output) {
		throw std::invalid_argument("QuantizationLayer::quantize requires valid buffers");
	}
	if (elements == 0) {
		return;
	}
	const cudaStream_t launch_stream = stream ? stream : config_.stream;
	const dim3 grid = makeGrid(elements);
	quantizeKernel<<<grid, kQuantThreads, 0, launch_stream>>>(
		input,
		output,
		elements,
		config_.scale,
		config_.zero_point,
		config_.clip_min,
		config_.clip_max);
	QUANT_CUDA_CHECK(cudaGetLastError());
}

void QuantizationLayer::dequantize(const std::int8_t* input,
				   float* output,
				   std::size_t elements,
				   cudaStream_t stream) const {
	if (!input || !output) {
		throw std::invalid_argument("QuantizationLayer::dequantize requires valid buffers");
	}
	if (elements == 0) {
		return;
	}
	const cudaStream_t launch_stream = stream ? stream : config_.stream;
	const dim3 grid = makeGrid(elements);
	dequantizeKernel<<<grid, kQuantThreads, 0, launch_stream>>>(
		input,
		output,
		elements,
		config_.scale,
		config_.zero_point);
	QUANT_CUDA_CHECK(cudaGetLastError());
}

void QuantizationLayer::quantizeAndDequantize(const float* input,
							 float* output,
							 std::size_t elements,
							 cudaStream_t stream) const {
	if (!input || !output) {
		throw std::invalid_argument("QuantizationLayer::quantizeAndDequantize requires valid buffers");
	}
	if (elements == 0) {
		return;
	}
	const cudaStream_t launch_stream = stream ? stream : config_.stream;
	const dim3 grid = makeGrid(elements);
	quantizeDequantizeKernel<<<grid, kQuantThreads, 0, launch_stream>>>(
		input,
		output,
		elements,
		config_.scale,
		config_.zero_point,
		config_.clip_min,
		config_.clip_max);
	QUANT_CUDA_CHECK(cudaGetLastError());
}

void QuantizationLayer::updateScaleFromRange(float min_value, float max_value) {
	if (!(min_value < max_value)) {
		throw std::invalid_argument("QuantizationLayer::updateScaleFromRange expects min < max");
	}
	if (config_.symmetric) {
		const float bound = std::max(std::fabs(min_value), std::fabs(max_value));
		const float safe_bound = std::max(bound, std::numeric_limits<float>::min());
		config_.scale = safe_bound / std::max(1.0f, config_.clip_max);
		config_.zero_point = 0;
	} else {
		const float qmin = config_.clip_min;
		const float qmax = config_.clip_max;
		const float range = max_value - min_value;
		const float denom = qmax - qmin;
		config_.scale = std::max(range / denom, std::numeric_limits<float>::min());
		const float zero_point_real = qmin - min_value / config_.scale;
		config_.zero_point = static_cast<int>(std::round(zero_point_real));
	}
}

} // namespace Quantization
} // namespace GRIM

#undef QUANT_CUDA_CHECK

