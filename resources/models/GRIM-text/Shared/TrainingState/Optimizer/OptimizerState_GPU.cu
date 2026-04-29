//======================================================//
//  OptimizerState_GPU.cu
//  TrainingState-owned optimizer moment tensors
//======================================================//

#include "../TrainingState_GPU.hpp"

#ifdef USE_CUDA

namespace GRIM {

//======================================================//
//  Optimizer State Management (Tensor-based)
//======================================================//

void TrainingState::allocateOptimizerStates(const std::vector<size_t>& sizes, cudaStream_t stream) {
	// Free any existing states first
	freeOptimizerStates();

	// Rule 22: Get stream from centralized controller if not provided.
	// getPrimaryStream() throws if not initialized (Rule 20).
	cudaStream_t primary_stream = stream ? stream : stream_ctrl.getPrimaryStream();

	// Allocate Tensor objects for each parameter group
	optimizer_m_states.reserve(sizes.size());
	optimizer_v_states.reserve(sizes.size());

	for (size_t i = 0; i < sizes.size(); ++i) {
		if (sizes[i] > 0) {
			// Create Tensor with flat shape and zero-initialize
			optimizer_m_states.push_back(Tensor::zeros({static_cast<int>(sizes[i])}, primary_stream, "optimizer_m"));
			optimizer_v_states.push_back(Tensor::zeros({static_cast<int>(sizes[i])}, primary_stream, "optimizer_v"));
		} else {
			// Empty placeholder Tensor for zero-size groups
			optimizer_m_states.emplace_back();
			optimizer_v_states.emplace_back();
		}
	}
	optimizer_states_allocated = true;
}

void TrainingState::freeOptimizerStates() {
	// Tensors auto-cleanup via destructor - just clear the vectors
	optimizer_m_states.clear();
	optimizer_v_states.clear();
	optimizer_states_allocated = false;
}

} // namespace GRIM

#endif  // USE_CUDA
