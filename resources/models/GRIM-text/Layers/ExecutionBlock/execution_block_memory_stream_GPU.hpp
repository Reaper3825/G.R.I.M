#pragma once

#ifdef USE_CUDA

#include "execution_block_internal.hpp"
#include "../../Shared/Batching/BatchPayload.hpp"

namespace GRIM::ExecutionBlockInternal {
void buildValueSlotCandidates(
	const HyperParameters::ExecutionBlockConstructionHP& hp,
	const ExecutionMemory& memory,
	const ExecutionBlockParameterTensors& parameters,
	const Batching::BatchPayload& payload,
	int batch_row,
	const std::vector<Forward::ExecutionRecord>& prior_records,
	const Tensor* selector_candidate_keys,
	const Tensor* slot_seeds,
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
	Forward::ExecutionBlockStepOutput& forward_output,
	cudaStream_t stream);

void finalizeStepOrThrow(
	const HyperParameters::ExecutionBlockConstructionHP& hp,
	ExecutionBlockDiagnosticsBuffers& diag,
	Forward::ExecutionBlockStepOutput& forward_output,
	const std::vector<Execution::CompiledSlotBinding>& slot_bindings,
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
