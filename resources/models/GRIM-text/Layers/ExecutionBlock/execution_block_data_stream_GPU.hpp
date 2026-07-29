#pragma once

#ifdef USE_CUDA

#include "execution_block_internal.hpp"
#include "../../Shared/Batching/BatchPayload.hpp"
#include "../../Shared/Batching/BatchDeviceBindings.hpp"

namespace GRIM::ExecutionBlockInternal {
void predictExecutionGateImpl(
	const HyperParameters::ExecutionBlockConstructionHP& hp,
	ExecutionBlockParameterTensors& parameters,
	Tensor& H,
	const Batching::BatchPayload& payload,
	int batch_row,
	cudaStream_t stream,
	Forward::ExecutionGateOutput* output);

void executeStepCoordinatorImpl(
	const HyperParameters::ExecutionBlockConstructionHP& hp,
	ExecutionBlockDiagnosticsBuffers& diag,
	ExecutionBlockParameterTensors& parameters,
	Tensor& H,
	ExecutionMemory& memory,
	const Batching::BatchPayload& payload,
	const Batching::BatchDeviceBindings& bindings,
	int batch_row,
	int step,
	float temperature,
	cudaStream_t stream,
	Forward::ExecutionBlockStepOutput& forward_output,
	Forward::RecordEncodeBackwardStaging& record_encode_backward_staging,
	Tensor& trace_state,
	const std::vector<Forward::ExecutionRecord>& prior_records,
	const Tensor* selector_candidate_keys,
	const Tensor* slot_seeds);

// Materialize the model's non-differentiable hard transition decision.
void materializeHardReadAndOpDecision(
	const HyperParameters::ExecutionBlockConstructionHP& hp,
	ExecutionBlockDiagnosticsBuffers& diag,
	cudaStream_t stream,
	const StepWorkingSet& work);

void materializeHardWriteDecision(
	const HyperParameters::ExecutionBlockConstructionHP& hp,
	ExecutionBlockDiagnosticsBuffers& diag,
	cudaStream_t stream,
	const StepWorkingSet& work);
}

#endif  // USE_CUDA
