#pragma once

#ifdef USE_CUDA

#include "execution_block_internal.hpp"

namespace GRIM::ExecutionBlockInternal {
void prepareMemoryStepOrThrow(
	ExecutionBlockLayer& layer,
	const ExecutionMemory& memory,
	const int* atom_positions,
	const int32_t* token_to_slot_map,
	int num_atoms,
	int row_tokens,
	ExecutionBlockStepOutput* diag_out,
	cudaStream_t stream);

void buildValueSlotCandidates(
	ExecutionBlockLayer& layer,
	const ExecutionMemory& memory,
	cudaStream_t stream,
	StepWorkingSet& work);

void materializeSelectedOperands(
	ExecutionBlockLayer& layer,
	const ExecutionMemory& memory,
	cudaStream_t stream,
	StepWorkingSet& work);

void applyHardWriteback(
	ExecutionBlockLayer& layer,
	ExecutionMemory& memory,
	cudaStream_t stream,
	const StepWorkingSet& work);

void captureStateAfterWriteAndCheckMutations(
	ExecutionBlockLayer& layer,
	const ExecutionMemory& memory,
	ExecutionBlockStepOutput* diag_out,
	cudaStream_t stream);

void finalizeStepOrThrow(
	ExecutionBlockLayer& layer,
	ExecutionBlockStepOutput* diag_out,
	int step,
	cudaStream_t stream);

void crossAttentionReadImpl(
	ExecutionBlockLayer& layer,
	Tensor& hidden_states,
	ExecutionMemory& memory,
	cudaStream_t stream,
	int token_offset,
	int row_tokens,
	float* d_gate_accum = nullptr);
}

#endif  // USE_CUDA
