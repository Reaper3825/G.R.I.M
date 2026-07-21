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
	bool teacher_force_transition,
	float temperature,
	cudaStream_t stream,
	Forward::ExecutionBlockStepOutput& forward_output,
	Forward::RecordEncodeBackwardStaging& record_encode_backward_staging,
	Tensor& trace_state,
	const std::vector<Forward::ExecutionRecord>& prior_records,
	const Tensor* selector_candidate_keys);

// Materialize only the non-differentiable hard transition decision. The caller
// supplies trajectory authority explicitly; structured CE and REINFORCE do not
// implicitly choose the executed policy. Live model tensors remain unchanged.
void materializeHardReadAndOpDecision(
	const HyperParameters::ExecutionBlockConstructionHP& hp,
	ExecutionBlockDiagnosticsBuffers& diag,
	const Batching::BatchPayload& payload,
	int batch_row,
	int step,
	bool teacher_force_transition,
	cudaStream_t stream,
	const StepWorkingSet& work);

void materializeHardWriteDecision(
	const HyperParameters::ExecutionBlockConstructionHP& hp,
	ExecutionBlockDiagnosticsBuffers& diag,
	const Batching::BatchPayload& payload,
	int batch_row,
	int step,
	bool teacher_force_transition,
	cudaStream_t stream,
	const StepWorkingSet& work);
}

#endif  // USE_CUDA
