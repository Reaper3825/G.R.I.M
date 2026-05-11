//======================================================//
//  Reasoning Head Layer - GPU
//  Parallel head (alongside LMHead)
//
//  Gathers encoder hidden states at atom positions,
//  concatenates with ScratchBlock atom embeddings,
//  mean-pools, then projects to:
//    op_logits   [1, num_ops]
//    arg1_logits [1, num_atoms]
//    arg2_logits [1, num_atoms]
//
//  Forward:
//    H_atoms  = gather(encoder_output, atom_positions)  [num_atoms, d_model]
//    Z        = concat(H_atoms, atom_embeddings)        [num_atoms, d_total]
//    z_pool   = mean(Z, dim=0)                          [1, d_total]
//    op_logits  = z_pool @ W_op^T + b_op                [1, num_ops]
//    arg1_logits = Z @ w_arg1^T                         [num_atoms, 1] -> [1, num_atoms]
//    arg2_logits = Z @ w_arg2^T                         [num_atoms, 1] -> [1, num_atoms]
//======================================================//

#pragma once

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <memory>

#include "../../Shared/TensorContract/TensorContract_GPU.hpp"
#include "../../Shared/HyperParameters/HyperparameterGroupings.hpp"

namespace GRIM {

//======================================================//
//  ReasoningHeadOutput — returned from forward
//======================================================//
struct ReasoningHeadOutput {
    Tensor op_logits;     // [1, num_ops]
    Tensor arg1_logits;   // [1, num_atoms]
    Tensor arg2_logits;   // [1, num_atoms]
};

//======================================================//
//  ReasoningHeadGradFn — backward node for
//  gather + concat + pool (pre-linear assembly)
//
//  The three matmuls (W_op, w_arg1, w_arg2) attach their
//  own MatmulGradFn via autograd::matmul. This GradFn
//  sits below them and handles the custom kernels.
//======================================================//
struct ReasoningHeadGradFn : public GradFn {
    // Owned device copy of atom_positions [num_atoms]
    int*   saved_atom_positions = nullptr;
    int    num_atoms = 0;
    int    total_tokens = 0;
    int    d_model = 0;
    int    atom_dim = 0;
    int    d_total = 0;

    // Gradient targets — encoder branch
    float* encoder_grad = nullptr;
    bool   owns_encoder_grad = false;
    std::shared_ptr<GradFn> encoder_grad_fn;
    TensorContract::TensorShape encoder_shape;

    // Gradient targets — atom_embeddings branch
    float* atom_emb_grad = nullptr;
    bool   owns_atom_emb_grad = false;
    std::shared_ptr<GradFn> atom_emb_grad_fn;
    TensorContract::TensorShape atom_emb_shape;

    ReasoningHeadGradFn() { op_name = "reasoning_head"; }
    ~ReasoningHeadGradFn() override;

    void capture_encoder(Tensor& encoder_output, cudaStream_t stream);
    void capture_atom_emb(Tensor& atom_embeddings, cudaStream_t stream);
    void capture_positions(const int* d_positions, int n_atoms, cudaStream_t stream);

    void apply(const Tensor& grad_output, cudaStream_t stream) override;
    void release_saved() override;
};

//======================================================//
//  ReasoningHeadLayer
//======================================================//
class ReasoningHeadLayer {
public:
    ReasoningHeadLayer() = delete;

    explicit ReasoningHeadLayer(const HyperParameters::ReasoningHeadConstructionHP& hp,
                                uint64_t seed,
                                cudaStream_t init_stream);

    ~ReasoningHeadLayer() = default;

    ReasoningHeadLayer(ReasoningHeadLayer&& other) noexcept;
    ReasoningHeadLayer& operator=(ReasoningHeadLayer&& other) noexcept;

    ReasoningHeadLayer(const ReasoningHeadLayer&) = delete;
    ReasoningHeadLayer& operator=(const ReasoningHeadLayer&) = delete;

    /// Forward pass.
    /// encoder_output:  [total_tokens, d_model]  — Tensor with grad_fn from encoder
    /// atom_embeddings: [num_atoms, atom_embedding_dim] — canonical copy-first Tensor
    /// atom_positions:  device int* [num_atoms] — token indices
    /// num_atoms:       HOST int (from D2H copy of numAtomsBuffer)
    /// total_tokens:    HOST int
    ReasoningHeadOutput forward(
        Tensor& encoder_output,
        Tensor& atom_embeddings,
        int* atom_positions,
        int num_atoms,
        int total_tokens,
        cudaStream_t stream,
        cublasHandle_t cublas_handle);

    // Parameter access
    Tensor& W_op()    { return w_op_; }
    Tensor& b_op()    { return b_op_; }
    Tensor& w_arg1()  { return w_arg1_; }
    Tensor& w_arg2()  { return w_arg2_; }
    const Tensor& W_op()   const { return w_op_; }
    const Tensor& b_op()   const { return b_op_; }
    const Tensor& w_arg1() const { return w_arg1_; }
    const Tensor& w_arg2() const { return w_arg2_; }

    int d_model() const { return hp_.d_model; }
    int atom_embedding_dim() const { return hp_.atom_embedding_dim; }
    int num_ops() const { return hp_.num_ops; }
    int d_total() const { return hp_.d_model + hp_.atom_embedding_dim; }

private:
    HyperParameters::ReasoningHeadConstructionHP hp_;
    Tensor w_op_;    // [num_ops, d_total]
    Tensor b_op_;    // [num_ops]
    Tensor w_arg1_;  // [1, d_total]
    Tensor w_arg2_;  // [1, d_total]
};

}  // namespace GRIM

#endif  // USE_CUDA
