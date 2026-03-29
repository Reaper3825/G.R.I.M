#pragma once

#ifdef USE_CUDA

#include "execution_block_internal.hpp"

namespace GRIM::ExecutionBlockInternal {
void executeStepCoordinatorImpl(
	ExecutionBlockLayer& layer,
	Tensor& H,
	ExecutionMemory& memory,
	const float* atom_embeddings,
	const int* atom_positions,
	const int32_t* token_to_slot_map,
	int num_atoms,
	int total_tokens,
	int step,
	float temperature,
	cudaStream_t stream,
	ExecutionBlockStepOutput* diag_out,
	int token_offset,
	int row_tokens,
	Tensor& trace_state,
	const std::vector<ExecutionRecord>& prior_records,
	const float* expected_target);
}

#endif  // USE_CUDA
