//======================================================//
//  grim_layer_gpu.hpp
//  Lightweight declarations shared across GPU layers
//======================================================//

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace GRIM {

struct Dimensions {
    int input = 0;
    int output = 0;
};

enum class LayerType : std::uint8_t {
    kUnknown = 0,
    kEmbedding,
    kLayerNorm,
    kRMSNorm,
    kAttention,
    kFeedForward,
    kResidual,
    kEncoding,
    kSerialization, 
    kCount
};

template<typename T>
struct LayerIO {
    const T* input = nullptr;
    T* output = nullptr;
    std::size_t tokens = 0;
};

template<typename T>
struct LayerWorkspace {
    T* data = nullptr;
    std::size_t bytes = 0;
};

template<typename T>
struct LayerGradients {
    T* weights = nullptr;
    T* bias = nullptr;
};

template<typename T>
struct LayerNormWeights {
    T* gamma = nullptr;
    T* beta = nullptr;
    int dim = 0;
};

template<typename T>
struct AttentionWeights {
    T* W_qkv = nullptr;
    T* b_qkv = nullptr;
    T* W_o = nullptr;
    T* b_o = nullptr;
    int d_model = 0;
    int num_heads = 0;
};

template<typename T>
struct FeedForwardWeights {
    T* W1 = nullptr;
    T* b1 = nullptr;
    T* W2 = nullptr;
    T* b2 = nullptr;
    int d_model = 0;
    int d_ff = 0;
};

template <typename T = float>
struct LayerSnapshot {
    LayerIO<T> io;
    LayerWorkspace<T> workspace;
    LayerGradients<T> gradients;
};

template <typename Derived, typename T = float>
class Layer {
public:
    bool trainable = true;
    bool training_mode = true;

    using value_type = T;
    using derived_type = Derived;
    using io_type = LayerIO<T>;
    using workspace_type = LayerWorkspace<T>;

    // NOTE: ForwardContext/BackwardContext and delegate hooks removed per Rule 20.
    // Nobody subscribed to the OnForwardBegin/End, OnBackwardBegin/End delegates.
    // Re-add if actual debugging hooks are needed in the future.

    Layer() = default;
    explicit Layer(const Dimensions& dims) : dims_(dims) {}

    const Dimensions& dims() const noexcept { return dims_; }
    int inputDim() const noexcept { return dims_.input; }
    int outputDim() const noexcept { return dims_.output; }

    void setDimensions(int input, int output) noexcept { dims_ = {input, output}; }
    void setDimensions(const Dimensions& dims) noexcept { dims_ = dims; }

    void configure(const Dimensions& dims) {
        dims_ = dims;
        if constexpr (has_on_configure<derived_type>::value) {
            derived().onConfigure(dims);
        }
    }

    void forward(const io_type& io, workspace_type* workspace = nullptr) {
        derived().forwardImpl(io, workspace);
    }

    void backward(const io_type& io, workspace_type* workspace = nullptr) {
        derived().backwardImpl(io, workspace);
    }

    static constexpr LayerType type() noexcept {
        return deduceLayerType<derived_type>(0);
    }

protected:
    derived_type& derived() noexcept { return static_cast<derived_type&>(*this); }
    const derived_type& derived() const noexcept { return static_cast<const derived_type&>(*this); }

private:
    template <typename V, typename = void>
    struct has_on_configure : std::false_type {};

    template <typename V>
    struct has_on_configure<V, std::void_t<decltype(std::declval<V&>().onConfigure(std::declval<const Dimensions&>()))>>
        : std::true_type {};

    Dimensions dims_{};

    template <typename U>
    static constexpr auto deduceLayerType(int) -> decltype(U::layer_type) {
        return U::layer_type;
    }

    template <typename>
    static constexpr LayerType deduceLayerType(...) {
        return LayerType::kUnknown;
    }
};

} // namespace GRIM
