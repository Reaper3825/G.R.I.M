#pragma once

#include "PhysicalImageOperatorState.hpp"
#include "PhysicalPerceptionPrimitiveResult.hpp"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace GRIM { namespace Perception { namespace Physical {

// Forward decl — onnxruntime_cxx_api.h is included only in the .cpp.
struct PhysicalImageClassifierOrtImpl;


// ─────────────────────────────────────────────────────────────────────────────
//  PhysicalImageClassifier — zero-shot CLIP whole-frame classification.
//
//  This is NOT an ImageNet classifier. It runs the MobileCLIP-S0 image
//  encoder on the frame, L2-normalises the resulting D-dim embedding, and
//  computes cosine similarity against a precomputed bank of N text-prompt
//  embeddings. Top-K survives → softmax → out.top_k.
//
//  The label set is whatever the user wrote into the prompts file at build
//  time — there is no fixed taxonomy. See scripts/setup_mobileclip.py for
//  how the .bin embeddings are produced from prompts.txt.
//
//  Expected ONNX input  : NCHW float32, 3 x H x W, normalised with CLIP
//                         per-channel mean/std.
//  Expected ONNX output : [1, embedding_dim] float32 (image embedding).
// ─────────────────────────────────────────────────────────────────────────────
struct PhysicalImageClassifierConfig {
    // Image encoder (MobileCLIP-S0 image_encoder.onnx).
    std::string  onnx_model_path;

    // Newline-separated prompt list. Row i in this file pairs with row i
    // in text_embeddings_path. Used as the displayed class label.
    std::string  class_names_path;

    // Precomputed text embeddings produced by scripts/setup_mobileclip.py.
    // Binary format:
    //   uint32 N           (number of prompts)
    //   uint32 D           (embedding dim, must match encoder output)
    //   float32[N*D]       (row-major, L2-normalised per row)
    std::string  text_embeddings_path;

    int          input_width    = 256;       // MobileCLIP-S0 is 256x256
    int          input_height   = 256;
    float        input_scale    = 1.0f / 255.0f;

    // CLIP normalisation (NOT ImageNet — these are the OpenAI CLIP
    // statistics that MobileCLIP inherits).
    cv::Scalar   input_mean     = cv::Scalar(0.48145466, 0.4578275, 0.40821073);
    cv::Scalar   input_std      = cv::Scalar(0.26862954, 0.26130258, 0.27577711);

    bool         swap_rb        = true;      // BGR input → RGB tensor
    int          top_k          = 5;

    // CLIP logit scale = exp(t) where the model's learned t ≈ 4.6
    // → temperature ≈ 100. Higher = sharper softmax.
    float        temperature    = 100.0f;

    PhysicalOperatorCadenceConfig cadence{};
};

class PhysicalImageClassifier {
public:
    PhysicalImageClassifier();
    ~PhysicalImageClassifier();

    void LoadOnnxModelIntoPhysicalImageClassifier(const PhysicalImageClassifierConfig& cfg);

    void RouteFrameToPhysicalImageClassifier(const cv::Mat& model_image,
                                             uint64_t source_frame_counter,
                                             PhysicalImageClassifierOutput& out);

    void                       ResetPhysicalImageClassifier();
    PhysicalImageOperatorState GetPhysicalImageClassifierState() const;
    std::string                GetPhysicalImageClassifierLastError() const;

private:
    mutable std::mutex                 mutex_;
    PhysicalImageClassifierConfig      cfg_{};
    std::unique_ptr<PhysicalImageClassifierOrtImpl> ort_;
    std::vector<std::string>           class_labels_;
    // Text embeddings, [N, D] row-major, L2-normalised float32. Cosine
    // similarity = image_embedding (1xD) * text_embeddings_.t() (DxN).
    cv::Mat                            text_embeddings_;
    int                                embedding_dim_      = 0;
    PhysicalImageOperatorState         state_              = PhysicalImageOperatorState::NoModelConfigured;
    std::string                        last_error_reason_;
    uint64_t                           inference_count_    = 0;
};

}}} // namespace
