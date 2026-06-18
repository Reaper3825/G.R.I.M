//======================================================//
//  AutogradEngine.cu
//  Iterative worklist backward engine implementation.
//  See AutogradEngine.hpp for the design rationale.
//======================================================//

#include "AutogradEngine.hpp"
#include "GradientAccumulation.hpp"
#include "../CudaAllocUtils.hpp"

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

using CudaAlloc::cudaMallocOrThrow;

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

float* AutogradEngine::ensureAccumulator(NodeState& st,
                                         const TensorContract::TensorShape& shape,
                                         std::size_t count) {
    if (st.accum == nullptr) {
        if (count == 0) {
            throw std::runtime_error("[AutogradEngine] zero-length gradient accumulator requested");
        }
        float* buffer = nullptr;
        cudaMallocOrThrow(reinterpret_cast<void**>(&buffer), count * sizeof(float), "AutogradEngine_accum");
        cudaMemsetAsync(buffer, 0, count * sizeof(float), stream_);
        st.accum = buffer;
        st.count = count;
        st.shape = shape;
        owned_buffers_.push_back(buffer);
    } else if (st.count != count) {
        throw std::runtime_error("[AutogradEngine] gradient accumulator size mismatch: have " +
                                 std::to_string(st.count) + " got " + std::to_string(count));
    }
    return st.accum;
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
    float* dst = ensureAccumulator(st, grad_view.shape, count);
    accumulate_grad(dst, grad_view.data, count, 1.0f, stream_, "AutogradEngine::contribute");

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

        // Seed the root accumulator with the loss gradient. The root has no
        // consumers (in_degree 0), so it is immediately ready.
        NodeState& root_state = nodes_[root];
        if (root_state.in_degree != 0) {
            throw std::runtime_error(std::string("[AutogradEngine] root node '") + nodeName(root) +
                                     "' has incoming edges - it must be the graph sink (loss)");
        }
        const std::size_t seed_count = seed_shape.total_elements();
        float* seed_dst = ensureAccumulator(root_state, seed_shape, seed_count);
        accumulate_grad(seed_dst, seed_grad, seed_count, 1.0f, stream_, "AutogradEngine::seed");
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

    // Accumulators are no longer needed; free them through the deferred-cleanup
    // queue so the cudaFree happens after Tensor::backward's stream sync.
    for (float* buffer : owned_buffers_) {
        queueForDeferredCleanup(buffer);
    }
    owned_buffers_.clear();
}

}  // namespace autograd
}  // namespace GRIM
