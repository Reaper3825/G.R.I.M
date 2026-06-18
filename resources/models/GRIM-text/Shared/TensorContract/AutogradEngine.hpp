#pragma once
//======================================================//
//  AutogradEngine.hpp
//  Iterative worklist (topological-sort-with-fan-in) backward engine.
//
//  Replaces the legacy DFS-recursive, first-wins ("applied" guard) traversal.
//  The engine is a pure scheduler: it owns NO gradient memory and allocates
//  nothing. It orchestrates over the two existing backbones:
//    - Leaf parameter gradients accumulate into StartupParameterRegistry-owned
//      grad_ buffers, terminally, inside each node's backward math.
//    - Non-leaf (interior / staged-activation) output gradients already live in
//      the consuming GradFn's own input-grad buffer (deferred-freed by that
//      node). The engine simply borrows the first contribution's buffer and, on
//      true fan-in, sums later contributions into it in place — no copies, no
//      per-node engine accumulator.
//
//  Each node fires exactly once, after every consumer has contributed its
//  share of that node's output gradient (true fan-in). This fixes the silent
//  gradient loss when a node has more than one consumer (MTP collapse: the LM
//  head and all K MTP heads share the encoder-output trunk).
//
//  Topology comes solely from GradFn::collect_input_edges(); the engine
//  asserts contribute-count == edge-count per node so an under-reported edge
//  fails loud (unknown contribution) and an over-reported edge fails loud
//  (undelivered edge / deadlock) instead of silently dropping gradient.
//======================================================//

#include "TensorContract_GPU.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <deque>
#include <unordered_map>
#include <vector>

namespace GRIM {

// Batching::BatchPayload / BatchDeviceBindings come in fully defined via
// TensorContract_GPU.hpp (included above), which pulls in the Batching headers.

namespace autograd {

class AutogradEngine {
public:
    AutogradEngine(cudaStream_t stream,
                   const Batching::BatchPayload* backward_payload,
                   const Batching::BatchDeviceBindings* backward_bindings);
    ~AutogradEngine();

    AutogradEngine(const AutogradEngine&) = delete;
    AutogradEngine& operator=(const AutogradEngine&) = delete;

    /**
     * Drive the full backward pass from the root grad_fn, seeded with the loss
     * gradient (device pointer + shape of the root/loss tensor). Discovers the
     * graph, seeds the root accumulator, then processes the ready-queue until
     * empty. Throws on any topology mismatch.
     */
    void run(GradFn* root,
             const float* seed_grad,
             const TensorContract::TensorShape& seed_shape);

    /**
     * Engine-side of GradFn::apply(): a downstream node hands `producer` its
     * share of the producer's output gradient. Accumulates into the producer's
     * engine accumulator and decrements its pending counter; enqueues the
     * producer once all expected contributions have arrived.
     */
    void contribute(GradFn* producer, const Tensor& grad_view);

    /**
     * The engine currently driving backward on this thread, or nullptr when a
     * legacy recursive pass (or no backward) is active. GradFn::apply() routes
     * through contribute() when this is non-null.
     */
    static AutogradEngine* active();

private:
    struct NodeState {
        int in_degree = 0;     ///< expected contributions (from discovery)
        int remaining = 0;     ///< countdown of pending contributions
        float* accum = nullptr;///< BORROWED grad buffer (owned by the producing GradFn / registry), never freed here
        std::size_t count = 0; ///< element count of accum
        TensorContract::TensorShape shape;  ///< shape of this node's output gradient
        bool queued = false;   ///< already pushed to the ready-queue
        bool executed = false; ///< run_backward has fired
    };

    NodeState& stateFor(GradFn* node);
    void discover(GradFn* root);

    cudaStream_t stream_;
    const Batching::BatchPayload* payload_;
    const Batching::BatchDeviceBindings* bindings_;

    std::unordered_map<GradFn*, NodeState> nodes_;
    std::deque<GradFn*> ready_;
};

}  // namespace autograd
}  // namespace GRIM
