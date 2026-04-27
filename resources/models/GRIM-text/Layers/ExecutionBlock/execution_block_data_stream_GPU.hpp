#pragma once

#ifdef USE_CUDA

#include "execution_block_internal.hpp"
#include "../../Shared/Batching/BatchPayload.hpp"
#include "../../Shared/Batching/BatchDeviceBindings.hpp"

namespace GRIM::ExecutionBlockInternal {
void executeStepCoordinatorImpl(
	ExecutionBlockLayer& layer,
	Tensor& H,
	ExecutionMemory& memory,
	const int* atom_positions,
	int num_atoms,
	const Batching::BatchPayload& payload,
	const Batching::BatchDeviceBindings& bindings,
	int batch_row,
	int step,
	float temperature,
	cudaStream_t stream,
	ExecutionBlockStepOutput* diag_out,
	Tensor& trace_state,
	const std::vector<ExecutionRecord>& prior_records,
	const float* expected_target,
	const TeacherSelectionTargets* selection_targets);
}

#endif  // USE_CUDA
