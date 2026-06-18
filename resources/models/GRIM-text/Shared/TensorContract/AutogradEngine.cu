//======================================================//
//  AutogradEngine.cu
//  Iterative worklist backward engine implementation.
//  See AutogradEngine.hpp for the design rationale.
//======================================================//

#include "AutogradEngine.hpp"
#include "GradientAccumulation.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <unordered_set>

// Defined at global scope in TensorContract_GPU.cu. Emits per-node gradient-flow
// diagnostics when the global tape is at Debug/Trace level; no-op otherwise.
void logTensorContractApplyGradOutputStats(const GRIM::GradFn& grad_fn,
                                           const GRIM::Tensor& grad_output,
                                           cudaStream_t stream);

namespace GRIM {

namespace autograd {

namespace {
thread_local AutogradEngine* t_active_engine = nullptr;

const char* nodeName(const GradFn* node) {
    if (node != nullptr && node->op_name != nullptr) {
        return node->op_name;
    }
    return "<unnamed>";
}
}  // namespace

AutogradEngine* AutogradEngine::active() {
    return t_active_engine;
}

AutogradEngine::AutogradEngine(cudaStream_t stream,
                               const Batching::BatchPayload* backward_payload,
                               const Batching::BatchDeviceBindings* backward_bindings)
    : stream_(stream),
      payload_(backward_payload),
      bindings_(backward_bindings) {
    if (stream_ == nullptr) {
        throw std::runtime_error("[AutogradEngine] stream is NULL - backward requires a valid CUDA stream");
    }
}

AutogradEngine::~AutogradEngine() {
    // Defensive: never leave a dangling active pointer if run() threw.
    if (t_active_engine == this) {
        t_active_engine = nullptr;
    }
}

AutogradEngine::NodeState& AutogradEngine::stateFor(GradFn* node) {
    return nodes_[node];
}

void AutogradEngine::discover(GradFn* root) {
    nodes_[root];  // ensure root exists with in_degree 0

    std::vector<GradFn*> stack;
    std::unordered_set<GradFn*> seen;
    stack.push_back(root);
    seen.insert(root);

    std::vector<GradFn*> edges;
    while (!stack.empty()) {
        GradFn* node = stack.back();
        stack.pop_back();

        edges.clear();
        node->collect_input_edges(edges);
        for (GradFn* producer : edges) {
            if (producer == nullptr) {
                continue;
            }
            nodes_[producer].in_degree += 1;
            if (seen.insert(producer).second) {
                stack.push_back(producer);
            }
        }
    }

    for (auto& entry : nodes_) {
        entry.second.remaining = entry.second.in_degree;
    }
}

void AutogradEngine::contribute(GradFn* producer, const Tensor& grad_view) {
    auto it = nodes_.find(producer);
    if (it == nodes_.end()) {
        throw std::runtime_error(std::string("[AutogradEngine] contribution to undiscovered node '") +
                                 nodeName(producer) +
                                 "' - collect_input_edges() under-reported an edge (contribute-count > edge-count)");
    }
    if (grad_view.data == nullptr) {
        throw std::runtime_error(std::string("[AutogradEngine] NULL gradient contributed to node '") +
                                 nodeName(producer) + "'");
    }

    NodeState& st = it->second;
    const std::size_t count = grad_view.numel();
    if (st.accum == nullptr) {
        // First contribution: borrow the consuming node's gradient buffer
        // directly. That buffer is owned (and deferred-freed) by the producing
        // GradFn, so the scheduler allocates nothing. Single-consumer nodes —
        // the common case — therefore never trigger an accumulation kernel.
        st.accum = grad_view.data;
        st.count = count;
        st.shape = grad_view.shape;
    } else {
        if (st.count != count) {
            throw std::runtime_error("[AutogradEngine] gradient contribution size mismatch: have " +
                                     std::to_string(st.count) + " got " + std::to_string(count));
        }
        // True fan-in: sum this consumer's share into the borrowed buffer in
        // place. Serialized on stream_, so ordering with the first write holds.
        accumulate_grad(st.accum, grad_view.data, count, 1.0f, stream_, "AutogradEngine::contribute");
    }

    if (st.remaining <= 0) {
        throw std::runtime_error(std::string("[AutogradEngine] node '") + nodeName(producer) +
                                 "' received more contributions than discovered edges");
    }
    st.remaining -= 1;
    if (st.remaining == 0 && !st.queued) {
        st.queued = true;
        ready_.push_back(producer);
    }
}

void AutogradEngine::run(GradFn* root,
                         const float* seed_grad,
                         const TensorContract::TensorShape& seed_shape) {
    if (root == nullptr) {
        throw std::runtime_error("[AutogradEngine] run() called with NULL root grad_fn");
    }
    if (seed_grad == nullptr) {
        throw std::runtime_error("[AutogradEngine] run() called with NULL seed gradient");
    }

    t_active_engine = this;
    try {
        discover(root);

        // Seed the root with the loss gradient by borrowing the caller's loss
        // grad buffer directly (the loss tensor's registry/lazy grad_). The root
        // has no consumers (in_degree 0), so it is never written to and is
        // immediately ready — the scheduler allocates nothing here either.
        NodeState& root_state = nodes_[root];
        if (root_state.in_degree != 0) {
            throw std::runtime_error(std::string("[AutogradEngine] root node '") + nodeName(root) +
                                     "' has incoming edges - it must be the graph sink (loss)");
        }
        root_state.accum = const_cast<float*>(seed_grad);
        root_state.count = seed_shape.total_elements();
        root_state.shape = seed_shape;
        root_state.queued = true;
        ready_.push_back(root);

        while (!ready_.empty()) {
            GradFn* node = ready_.front();
            ready_.pop_front();

            NodeState& st = nodes_[node];
            if (st.executed) {
                continue;
            }
            st.executed = true;

            Tensor grad_view;
            grad_view.data = st.accum;
            grad_view.shape = st.shape;
            grad_view.owns_data = false;
            grad_view.stream = stream_;

            logTensorContractApplyGradOutputStats(*node, grad_view, stream_);
            // run_backward() may call contribute() (via GradFn::apply re-route)
            // for each upstream edge; the map is fully populated by discover()
            // so no structural rehash occurs and `st` stays valid.
            node->run_backward(grad_view, stream_, payload_, bindings_);
        }

        // Topology invariant: every discovered node must have fired exactly once
        // with all its edges delivered. A survivor means an over-reported edge
        // (edge-count > contribute-count) which would have silently dropped grad.
        for (const auto& entry : nodes_) {
            if (!entry.second.executed) {
                throw std::runtime_error(std::string("[AutogradEngine] node '") + nodeName(entry.first) +
                                         "' never fired (remaining=" + std::to_string(entry.second.remaining) +
                                         ") - collect_input_edges() over-reported an edge / deadlock");
            }
        }
    } catch (...) {
        t_active_engine = nullptr;
        throw;
    }
    t_active_engine = nullptr;

    // No engine-owned buffers to release: every accumulator is a borrowed
    // pointer into a consuming GradFn's buffer (or the caller's seed buffer),
    // each freed by its own owner's deferred-cleanup path.
}

}  // namespace autograd
}  // namespace GRIM
