#pragma once

#ifdef USE_CUDA

#include "execution_block_internal.hpp"

namespace GRIM::ExecutionBlockInternal {
void prepareMemoryStepOrThrow(
	const HyperParameters::ExecutionBlockConstructionHP& hp,
	ExecutionBlockDiagnosticsBuffers& diag,
	const ExecutionMemory& memory,
	const int* atom_positions,
	const int32_t* token_to_slot_map,
	int num_atoms,
	int row_tokens,
	Forward::ExecutionBlockStepOutput* diag_out,
	cudaStream_t stream);

void buildValueSlotCandidates(
	const HyperParameters::ExecutionBlockConstructionHP& hp,
	const ExecutionMemory& memory,
	cudaStream_t stream,
	StepWorkingSet& work);

void materializeSelectedOperands(
	const HyperParameters::ExecutionBlockConstructionHP& hp,
	ExecutionBlockDiagnosticsBuffers& diag,
	const ExecutionMemory& memory,
	cudaStream_t stream,
	StepWorkingSet& work);

void applyHardWriteback(
	const HyperParameters::ExecutionBlockConstructionHP& hp,
	ExecutionBlockDiagnosticsBuffers& diag,
	ExecutionBlockParameterTensors& parameters,
	ExecutionMemory& memory,
	cudaStream_t stream,
	const StepWorkingSet& work);

void captureStateAfterWriteAndCheckMutations(
	const HyperParameters::ExecutionBlockConstructionHP& hp,
	ExecutionBlockDiagnosticsBuffers& diag,
	const ExecutionMemory& memory,
	Forward::ExecutionBlockStepOutput* diag_out,
	cudaStream_t stream);

void finalizeStepOrThrow(
	const HyperParameters::ExecutionBlockConstructionHP& hp,
	ExecutionBlockDiagnosticsBuffers& diag,
	Forward::ExecutionBlockStepOutput* diag_out,
	int step,
	cudaStream_t stream);

Tensor crossAttentionReadImpl(
	const HyperParameters::ExecutionBlockConstructionHP& hp,
	ExecutionBlockParameterTensors& parameters,
	const Tensor& hidden_states,
	ExecutionMemory& memory,
	cudaStream_t stream,
	int token_offset,
	int row_tokens,
	float* d_gate_accum = nullptr);
}

#endif  // USE_CUDA
