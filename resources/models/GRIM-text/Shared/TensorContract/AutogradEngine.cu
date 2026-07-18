//======================================================//
//  AutogradEngine.cu
//  Iterative worklist backward engine implementation.
//  See AutogradEngine.hpp for the design rationale.
//======================================================//

#include "AutogradEngine.hpp"

#include <cuda_runtime.h>

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <unordered_set>

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

bool syncEachNodeForDiagnostics() {
    static const bool enabled = [] {
        const char* value = std::getenv("GRIM_AUTOGRAD_SYNC_EACH_NODE");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

void synchronizeAfterNodeOrThrow(cudaStream_t stream, const GradFn* node) {
    const cudaError_t status = cudaStreamSynchronize(stream);
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string("[AutogradEngine] CUDA failure after node '") + nodeName(node) +
            "': " + cudaGetErrorString(status));
    }
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

void AutogradEngine::contribute(GradFn* producer) {
    auto it = nodes_.find(producer);
    if (it == nodes_.end()) {
        throw std::runtime_error(std::string("[AutogradEngine] contribution to undiscovered node '") +
                                 nodeName(producer) +
                                 "' - collect_input_edges() under-reported an edge (contribute-count > edge-count)");
    }
    NodeState& st = it->second;
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

void AutogradEngine::run(GradFn* root) {
    if (root == nullptr) {
        throw std::runtime_error("[AutogradEngine] run() called with NULL root grad_fn");
    }
    t_active_engine = this;
    try {
        discover(root);

        // The caller delivered the loss gradient before entering the scheduler.
        // The root has no consumers (in_degree 0), so it is immediately ready.
        NodeState& root_state = nodes_[root];
        if (root_state.in_degree != 0) {
            throw std::runtime_error(std::string("[AutogradEngine] root node '") + nodeName(root) +
                                     "' has incoming edges - it must be the graph sink (loss)");
        }
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

            // run_backward() may call contribute() (via GradFn::apply re-route)
            // for each upstream edge; the map is fully populated by discover()
            // so no structural rehash occurs and `st` stays valid.
            node->run_backward(stream_, payload_, bindings_);
            if (syncEachNodeForDiagnostics()) {
                synchronizeAfterNodeOrThrow(stream_, node);
            }
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

    // Gradient tensors remain owned by their GradFns. The engine has only
    // scheduling state to destroy.
}

}  // namespace autograd
}  // namespace GRIM
