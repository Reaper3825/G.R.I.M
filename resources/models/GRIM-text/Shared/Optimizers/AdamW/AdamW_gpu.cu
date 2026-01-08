//======================================================//
//  AdamW_gpu.cu
//  High level AdamW optimizer launcher
//======================================================//

#include "AdamW_gpu.hpp"
#include "AdamW_Kernal_GPU.hpp"

namespace GRIM {

void launchAdamWUpdate(const AdamWUpdateArgs& args) {
	if (!args.params || !args.grads || !args.moments1 || !args.moments2) {
		return;
	}
	if (args.size == 0 || args.learning_rate <= 0.0f) {
		return;
	}

	// Ensure step is at least 1 for bias correction
	const int step = (args.step <= 0) ? 1 : args.step;

	launchAdamWKernel(args.params,
					  args.grads,
					  args.moments1,
					  args.moments2,
					  args.size,
					  args.learning_rate,
					  args.weight_decay,
					  step,
					  args.stream);
}

} // namespace GRIM

