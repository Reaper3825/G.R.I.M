#pragma once

#include "PhysicalImageOperatorState.hpp"
#include "PhysicalPerceptionPrimitiveResult.hpp"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace GRIM { namespace Perception { namespace Physical {

// ─────────────────────────────────────────────────────────────────────────────
//  PhysicalInstanceSegmenter — Stage-2 SAM 2 prompted instance segmentation.
//
//  Purpose: turn the PhysicalObjectDetector's axis-aligned boxes into
//  pixel-precise per-instance binary masks at MODEL resolution. This lets
//  downstream consumers reason about partial occlusion ("the cup is partly
//  occluded by the notebook") via mask intersection — something bounding
//  boxes alone cannot express.
//
//  Backend: two cv::dnn::Net instances loading the standard SAM 2 ONNX
//  export pair:
//
//    encoder_onnx_path:
//      input  "image"            [1, 3, H_in, W_in]   (default 1024x1024)
//      output "image_embed"      [1, 256, 64, 64]
//      output "high_res_feats_0" [1, 32, 256, 256]
//      output "high_res_feats_1" [1, 64, 128, 128]
//
//    decoder_onnx_path:
//      input  "image_embed"      (from encoder)
//      input  "high_res_feats_0" (from encoder)
//      input  "high_res_feats_1" (from encoder)
//      input  "point_coords"     [1, N, 2]   in encoder-input (1024) pixels
//      input  "point_labels"     [1, N]      (2 = box-TL, 3 = box-BR per SAM)
//      input  "mask_input"       [1, 1, 256, 256]  (zeros for first call)
//      input  "has_mask_input"   [1]               (= 0 for first call)
//      output "masks"            [1, M, 256, 256]  (logits — M=1 with multimask=False)
//      output "iou_predictions"  [1, M]
//
//  This is the upstream Meta SAM 2 ONNX export. Different exports rename
//  blobs; for those, override the input/output names in the config below.
//  Mismatched shapes throw with an explicit reason — silent fallbacks are
//  forbidden by Rule 20.
//
//  Self-owned, internally managed. The only entry points called from the
//  Stage-2 loop are:
//    LoadOnnxModelsIntoPhysicalInstanceSegmenter(cfg)
//    RouteFrameAndDetectionsToPhysicalInstanceSegmenter(model_image, dets,
//                                                       frame_counter, out)
//    ResetPhysicalInstanceSegmenter()
//
//  The encoder runs ONCE per frame (frame_counter de-duplication). The
//  decoder runs ONCE PER PROMPT, so wall-time grows linearly in
//  prompt_count. cfg.max_prompts_per_frame caps that.
// ─────────────────────────────────────────────────────────────────────────────
struct PhysicalInstanceSegmenterConfig {
    // Empty paths leave the operator in NoModelConfigured (intentional silence).
    std::string  encoder_onnx_path;
    std::string  decoder_onnx_path;

    // Encoder input geometry. SAM 2 was trained on 1024x1024.
    int          encoder_input_width   = 1024;
    int          encoder_input_height  = 1024;
    float        encoder_input_scale   = 1.0f / 255.0f;
    // SAM-style ImageNet normalisation, applied AFTER scaling. Stored in
    // BGR-pre-swap convention to match cv::dnn::blobFromImage(swap_rb=true).
    cv::Scalar   encoder_input_mean    = cv::Scalar(123.675f, 116.28f, 103.53f);
    cv::Scalar   encoder_input_std_dev = cv::Scalar(58.395f,  57.12f,  57.375f);
    bool         swap_rb               = true;

    // Decoder I/O blob names. Override these if your ONNX export differs.
    std::string  encoder_input_name      = "image";
    std::string  encoder_output_image_embed_name      = "image_embed";
    std::string  encoder_output_high_res_feats_0_name = "high_res_feats_0";
    std::string  encoder_output_high_res_feats_1_name = "high_res_feats_1";
    std::string  decoder_input_image_embed_name       = "image_embed";
    std::string  decoder_input_high_res_feats_0_name  = "high_res_feats_0";
    std::string  decoder_input_high_res_feats_1_name  = "high_res_feats_1";
    std::string  decoder_input_point_coords_name      = "point_coords";
    std::string  decoder_input_point_labels_name      = "point_labels";
    std::string  decoder_input_mask_input_name        = "mask_input";
    std::string  decoder_input_has_mask_input_name    = "has_mask_input";
    std::string  decoder_output_masks_name            = "masks";
    std::string  decoder_output_iou_predictions_name  = "iou_predictions";

    // Mask decoder produces logits at this resolution; standard SAM 2 = 256.
    int          decoder_mask_resolution = 256;

    // Prompt selection from the object detector's output.
    int          max_prompts_per_frame  = 16;     // cap decoder calls per Tick
    float        min_prompt_confidence  = 0.30f;  // skip detections below this

    // Mask post-processing.
    float        mask_logit_threshold   = 0.0f;   // SAM 2 default: 0 (sigmoid > 0.5)

    // ONNX Runtime backend tuning.
    //   intra_op_num_threads: 0 = let ORT decide based on hw_concurrency.
    // CoreML EP is intentionally NOT exposed here — the current vcpkg
    // onnxruntime port does not ship the CoreML provider headers/symbols,
    // so we cannot honor an EP-selection toggle without violating Rule 20.
    int          intra_op_num_threads  = 0;

    // Latency knobs. Defaults preserve every-frame inference. Note: the
    // SAM 2 encoder is the dominant cost; turning on reuse_on_stable_scene
    // here is the single biggest win for static-camera scenarios.
    PhysicalOperatorCadenceConfig cadence{};
};

// Forward declaration — implementation lives in the .cpp to avoid leaking
// onnxruntime_cxx_api.h into every translation unit that includes this header.
struct PhysicalInstanceSegmenterOrtImpl;

class PhysicalInstanceSegmenter {
public:
    PhysicalInstanceSegmenter();
    ~PhysicalInstanceSegmenter();

    // Load encoder + decoder ONNX. Empty paths reset to NoModelConfigured;
    // any other failure throws with an explicit reason.
    void LoadOnnxModelsIntoPhysicalInstanceSegmenter(const PhysicalInstanceSegmenterConfig& cfg);

    // Run inference on one frame. Never throws — failures land in `out`.
    // `model_image` is the conditioned MODEL-resolution BGR frame from
    // PhysicalFrameBus. `detections` are the raw axis-aligned boxes from
    // PhysicalObjectDetector at the SAME source_frame_counter; we do NOT
    // re-derive them, only consume.
    void RouteFrameAndDetectionsToPhysicalInstanceSegmenter(
        const cv::Mat& model_image,
        const std::vector<PhysicalObjectDetection>& detections,
        uint64_t source_frame_counter,
        PhysicalInstanceSegmenterOutput& out);

    void                       ResetPhysicalInstanceSegmenter();
    PhysicalImageOperatorState GetPhysicalInstanceSegmenterState() const;
    std::string                GetPhysicalInstanceSegmenterLastError() const;
    bool                       IsPhysicalInstanceSegmenterReady() const;

private:
    mutable std::mutex                                mutex_;
    PhysicalInstanceSegmenterConfig                   cfg_{};
    std::unique_ptr<PhysicalInstanceSegmenterOrtImpl> ort_;
    PhysicalImageOperatorState                        state_ = PhysicalImageOperatorState::NoModelConfigured;
    std::string                                       last_error_reason_;
    uint64_t                                          inference_count_ = 0;
};

}}} // namespace GRIM::Perception::Physical
