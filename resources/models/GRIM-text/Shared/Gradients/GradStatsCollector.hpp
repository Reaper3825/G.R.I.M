#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "../LogRecorder/LogRecorder.hpp"

namespace GRIM::GradStats {

struct FlushResult {
	bool has_explosion = false;
	bool has_nan = false;
	bool has_inf = false;
	std::size_t count = 0;
};

void beginBatch();

void enqueue(const char* name,
	int layer,
	const float* data,
	std::size_t size,
	float explosion_threshold,
	cudaStream_t stream);

FlushResult flushAndLog(cudaStream_t stream,
	std::uint64_t step,
	GRIM::Logging::ModuleId module = GRIM::Logging::ModuleId::BackwardPass);

} // namespace GRIM::GradStats
