# TrainingState — Centralized GPU Resource Controller

All GPU resources MUST go through `TrainingState`. Every other struct holds **pointers only**.

| Resource | Access |
|----------|--------|
| CUDA streams | `training_state.stream_ctrl.getPrimaryStream()` |
| cuBLAS handle | `training_state.cublas_handle` |
| Gradient buffers | `ctx.model->zeroGradients()` / `ctx.model->backward()` |
| Optimizer states | `training_state.optimizer_m_states` / `optimizer_v_states` |

**Violations are bugs:**
- ❌ Raw `cudaStream_t` locals
- ❌ Separate cuBLAS handles
- ❌ Allocations owned by anything other than `TrainingState`

See [Autograd.md](Autograd.md) for the ownership taxonomy that governs which buffers belong here vs. inside the autograd tape.

## Key file
`resources/models/GRIM-text/Training/TrainingState_GPU.hpp`
