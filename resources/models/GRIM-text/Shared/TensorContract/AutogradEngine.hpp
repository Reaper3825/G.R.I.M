#pragma once
//======================================================//
//  AutogradEngine.hpp
//  Iterative worklist (topological-sort-with-fan-in) backward engine.
//
//  Replaces the legacy DFS-recursive, first-wins ("applied" guard) traversal.
//  The engine is a pure scheduler: it owns no gradient tensors, accesses no
//  gradient storage, and allocates nothing. It orchestrates over the two
//  existing backbones:
//    - Leaf parameter gradients accumulate into StartupParameterRegistry-owned
//      grad_ buffers, terminally, inside each node's backward math.
//    - Every GradFn receives and owns its pending output-gradient Tensor. The
//      engine only counts arrivals and schedules the GradFn after true fan-in
//      is complete.
//
//  Each node fires exactly once, after every consumer has contributed its
//  share of that node's output gradient (true fan-in). This fixes silent
//  gradient loss when a node has more than one consumer.
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
     * Drive the full backward pass from a root GradFn that has already received
     * its loss-gradient Tensor. Discovers the graph and processes the
     * ready-queue until empty. Throws on any topology mismatch.
     */
    void run(GradFn* root);

    /**
     * Scheduler notification from GradFn::apply(): the producer has already
     * accepted one downstream gradient contribution. Decrements its pending
     * counter and enqueues it once all expected contributions have arrived.
     */
    void contribute(GradFn* producer);

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
        bool queued = false;   ///< already pushed to the ready-queue
        bool executed = false; ///< run_backward has fired
    };

    void discover(GradFn* root);

    cudaStream_t stream_;
    const Batching::BatchPayload* payload_;
    const Batching::BatchDeviceBindings* bindings_;

    std::unordered_map<GradFn*, NodeState> nodes_;
    std::deque<GradFn*> ready_;
};

}  // namespace autograd
}  // namespace GRIM
