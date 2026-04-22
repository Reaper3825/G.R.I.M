#include "PhysicalInstanceSegmenter.hpp"

#include "PhysicalPerceptionPrimitivesLogTag.hpp"
#include "logger.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>

#include <opencv2/dnn.hpp>      // for cv::dnn::blobFromImage helper only
#include <opencv2/imgproc.hpp>

#include <onnxruntime_cxx_api.h>

namespace GRIM { namespace Perception { namespace Physical {

// ORT PIMPL — owns env + sessions + cached I/O name strings. Hidden behind
// a forward declaration in the header so onnxruntime_cxx_api.h does not leak.
struct PhysicalInstanceSegmenterOrtImpl {
    Ort::Env                       env{ORT_LOGGING_LEVEL_WARNING, "GRIM.PhysicalInstanceSegmenter"};
    Ort::MemoryInfo                cpu_memory_info{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
    std::unique_ptr<Ort::Session>  encoder_session;
    std::unique_ptr<Ort::Session>  decoder_session;
    // Cached input/output names — ORT requires `const char* const*` arrays.
    // Keep storage stable for the lifetime of the session.
    std::vector<std::string>       enc_input_names_storage;
    std::vector<std::string>       enc_output_names_storage;
    std::vector<std::string>       dec_input_names_storage;
    std::vector<std::string>       dec_output_names_storage;
    std::vector<const char*>       enc_input_names;
    std::vector<const char*>       enc_output_names;
    std::vector<const char*>       dec_input_names;
    std::vector<const char*>       dec_output_names;
};

namespace {

// ─── Helpers ────────────────────────────────────────────────────────────────

std::string DescribeMatShape(const cv::Mat& m) {
    std::string s = "[";
    for (int i = 0; i < m.dims; ++i) {
        if (i) s += ",";
        s += std::to_string(m.size[i]);
    }
    s += "]";
    return s;
}

// SAM-style preprocess: BGR→RGB, scale by 1/255, subtract mean, divide by std.
// We call cv::dnn::blobFromImage to get scaled+mean-subtracted [1,3,H,W],
// then divide by std-dev channel-wise. Doing the std-divide manually is
// REQUIRED because cv::dnn::blobFromImage has no std parameter — silently
// using only mean would be a numerical-precision violation.
cv::Mat BuildSamEncoderBlob(const cv::Mat& letterboxed_bgr,
                            const PhysicalInstanceSegmenterConfig& cfg)
{
    if (letterboxed_bgr.empty()) {
        throw std::runtime_error("BuildSamEncoderBlob: input image is empty");
    }
    if (letterboxed_bgr.type() != CV_8UC3) {
        throw std::runtime_error("BuildSamEncoderBlob: input must be CV_8UC3");
    }
    if (letterboxed_bgr.cols != cfg.encoder_input_width
        || letterboxed_bgr.rows != cfg.encoder_input_height) {
        throw std::runtime_error(
            "BuildSamEncoderBlob: caller MUST pre-resize to "
            + std::to_string(cfg.encoder_input_width) + "x"
            + std::to_string(cfg.encoder_input_height) + " (got "
            + std::to_string(letterboxed_bgr.cols) + "x"
            + std::to_string(letterboxed_bgr.rows) + ")");
    }
    cv::Mat blob = cv::dnn::blobFromImage(
        letterboxed_bgr, cfg.encoder_input_scale,
        cv::Size(cfg.encoder_input_width, cfg.encoder_input_height),
        cfg.encoder_input_mean, cfg.swap_rb, /*crop=*/false);
    if (blob.dims != 4 || blob.size[0] != 1 || blob.size[1] != 3) {
        throw std::runtime_error("BuildSamEncoderBlob: blobFromImage produced unexpected shape "
                                 + DescribeMatShape(blob));
    }
    // Channel-wise std divide. blobFromImage internally swaps RB before
    // mean subtract, so channel order in the blob is (R,G,B) when swap_rb
    // is true. encoder_input_std_dev is stored in (R,G,B) order (the SAM
    // convention) — see header docs.
    const int C = blob.size[1];
    const int H = blob.size[2];
    const int W = blob.size[3];
    const double inv_std[3] = {
        cfg.encoder_input_std_dev[0] > 0.0 ? 1.0 / cfg.encoder_input_std_dev[0] : 0.0,
        cfg.encoder_input_std_dev[1] > 0.0 ? 1.0 / cfg.encoder_input_std_dev[1] : 0.0,
        cfg.encoder_input_std_dev[2] > 0.0 ? 1.0 / cfg.encoder_input_std_dev[2] : 0.0,
    };
    if (inv_std[0] == 0.0 || inv_std[1] == 0.0 || inv_std[2] == 0.0) {
        throw std::runtime_error("BuildSamEncoderBlob: encoder_input_std_dev has a zero channel — would divide by zero");
    }
    for (int c = 0; c < C && c < 3; ++c) {
        float* plane = blob.ptr<float>(0, c);
        const float k = static_cast<float>(inv_std[c]);
        const size_t n = static_cast<size_t>(H) * static_cast<size_t>(W);
        // Also fold in the *255 scale that the std expects (mean was
        // subtracted in 0..255 units by blobFromImage(scale=1/255, mean=...);
        // SAM's std is in 0..255 units too, so we just divide by std/255 ==
        // multiply by 255*inv_std). Concretely: blob currently holds
        // (pixel/255 - mean*scale) which is wrong vs SAM. Re-derive:
        //   blob[c] = (pixel - mean) / std,
        //   but blobFromImage gave us (pixel - mean) * scale = (pixel - mean)/255
        //   so multiply by 255 / std == 255 * inv_std.
        const float k_final = k * 255.0f;
        for (size_t i = 0; i < n; ++i) plane[i] *= k_final;
    }
    return blob;
}

// Letterbox-resize `bgr_in` into a `target` square, preserving aspect ratio,
// padding with zeros. Returns the (scale, offset_x, offset_y) used so the
// caller can map prompt coords from MODEL space into ENCODER space.
struct LetterboxXform {
    float scale    = 1.0f;
    float offset_x = 0.0f;
    float offset_y = 0.0f;
};

LetterboxXform LetterboxResize(const cv::Mat& bgr_in,
                               int target_w, int target_h,
                               cv::Mat& bgr_out)
{
    if (target_w <= 0 || target_h <= 0) {
        throw std::runtime_error("LetterboxResize: non-positive target size");
    }
    if (bgr_in.empty()) {
        throw std::runtime_error("LetterboxResize: input image is empty");
    }
    const float in_w = static_cast<float>(bgr_in.cols);
    const float in_h = static_cast<float>(bgr_in.rows);
    const float s    = std::min(static_cast<float>(target_w) / in_w,
                                static_cast<float>(target_h) / in_h);
    const int   new_w = std::max(1, static_cast<int>(std::round(in_w * s)));
    const int   new_h = std::max(1, static_cast<int>(std::round(in_h * s)));
    cv::Mat resized;
    cv::resize(bgr_in, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    bgr_out = cv::Mat::zeros(target_h, target_w, CV_8UC3);
    const int off_x = (target_w - new_w) / 2;
    const int off_y = (target_h - new_h) / 2;
    resized.copyTo(bgr_out(cv::Rect(off_x, off_y, new_w, new_h)));
    LetterboxXform x;
    x.scale    = s;
    x.offset_x = static_cast<float>(off_x);
    x.offset_y = static_cast<float>(off_y);
    return x;
}

// Map a MODEL-space box to (top-left, bottom-right) point pair in
// ENCODER (1024x1024) pixel space. SAM 2 represents box prompts as TWO
// points with labels {2, 3}.
void BoxToSamPointPair(const cv::Rect2f& model_box,
                       int model_w, int model_h,
                       const LetterboxXform& model_to_enc_via_letterbox,
                       float (&out_pts)[2][2])
{
    // model_box is in MODEL pixel coords. The encoder consumed a
    // letterboxed-to-encoder version of model_image, so we go MODEL → ENCODER
    // via that exact letterbox transform. We do NOT need the model dims
    // unless we want to clip; clip first to the model rect to keep prompts
    // valid even if a detector reports a slightly OOB box.
    float x0 = std::max(0.0f, model_box.x);
    float y0 = std::max(0.0f, model_box.y);
    float x1 = std::min(static_cast<float>(model_w), model_box.x + model_box.width);
    float y1 = std::min(static_cast<float>(model_h), model_box.y + model_box.height);
    if (x1 <= x0 || y1 <= y0) {
        throw std::runtime_error("BoxToSamPointPair: clipped prompt box is degenerate");
    }
    const float s   = model_to_enc_via_letterbox.scale;
    const float ox  = model_to_enc_via_letterbox.offset_x;
    const float oy  = model_to_enc_via_letterbox.offset_y;
    out_pts[0][0] = x0 * s + ox;
    out_pts[0][1] = y0 * s + oy;
    out_pts[1][0] = x1 * s + ox;
    out_pts[1][1] = y1 * s + oy;
}

// Decode one mask (CV_32F, decoder_mask_resolution × decoder_mask_resolution
// logits in ENCODER-letterboxed space) back to a tight binary mask in MODEL
// pixel space. Returns the integer model-space bbox enclosing the mask and
// the binary CV_8UC1 mask cropped to that bbox. If the mask is empty
// (all-zero), returns {bbox.area==0, mask.empty()}.
void DecodeOneMaskToModelSpace(const cv::Mat& mask_logits_lo_res,
                               int decoder_mask_resolution,
                               const LetterboxXform& model_to_enc_via_letterbox,
                               int model_w, int model_h,
                               int encoder_w, int encoder_h,
                               float logit_threshold,
                               cv::Rect2i& out_model_bbox,
                               cv::Mat& out_mask_within_bbox,
                               int& out_pixel_count)
{
    if (mask_logits_lo_res.empty()) {
        throw std::runtime_error("DecodeOneMaskToModelSpace: mask_logits_lo_res is empty");
    }
    if (mask_logits_lo_res.rows != decoder_mask_resolution
        || mask_logits_lo_res.cols != decoder_mask_resolution
        || mask_logits_lo_res.type() != CV_32FC1) {
        throw std::runtime_error(
            "DecodeOneMaskToModelSpace: expected CV_32FC1 ["
            + std::to_string(decoder_mask_resolution) + "x"
            + std::to_string(decoder_mask_resolution) + "] got "
            + std::to_string(mask_logits_lo_res.rows) + "x"
            + std::to_string(mask_logits_lo_res.cols)
            + " type=" + std::to_string(mask_logits_lo_res.type()));
    }
    // Step 1: bilinear-upsample logits to ENCODER resolution (1024x1024).
    cv::Mat mask_logits_enc;
    cv::resize(mask_logits_lo_res, mask_logits_enc,
               cv::Size(encoder_w, encoder_h), 0, 0, cv::INTER_LINEAR);

    // Step 2: crop out the letterbox padding region — only the inner
    // (model.cols * scale) x (model.rows * scale) rectangle is meaningful.
    const float s    = model_to_enc_via_letterbox.scale;
    const int   ox   = static_cast<int>(std::round(model_to_enc_via_letterbox.offset_x));
    const int   oy   = static_cast<int>(std::round(model_to_enc_via_letterbox.offset_y));
    const int   inner_w = std::max(1, static_cast<int>(std::round(model_w * s)));
    const int   inner_h = std::max(1, static_cast<int>(std::round(model_h * s)));
    if (ox < 0 || oy < 0 || ox + inner_w > encoder_w || oy + inner_h > encoder_h) {
        throw std::runtime_error("DecodeOneMaskToModelSpace: letterbox inner rect exceeds encoder rect");
    }
    cv::Mat mask_logits_inner = mask_logits_enc(cv::Rect(ox, oy, inner_w, inner_h));

    // Step 3: resize the inner region to MODEL resolution. Use bilinear so
    // soft logit edges remain stable; we threshold AFTER resize.
    cv::Mat mask_logits_model;
    cv::resize(mask_logits_inner, mask_logits_model,
               cv::Size(model_w, model_h), 0, 0, cv::INTER_LINEAR);

    // Step 4: threshold to binary 0/255.
    cv::Mat mask_full(model_h, model_w, CV_8UC1, cv::Scalar(0));
    out_pixel_count = 0;
    int min_x = model_w, min_y = model_h, max_x = -1, max_y = -1;
    for (int y = 0; y < model_h; ++y) {
        const float*  src = mask_logits_model.ptr<float>(y);
        uint8_t*      dst = mask_full.ptr<uint8_t>(y);
        for (int x = 0; x < model_w; ++x) {
            if (src[x] > logit_threshold) {
                dst[x] = 255;
                ++out_pixel_count;
                if (x < min_x) min_x = x;
                if (y < min_y) min_y = y;
                if (x > max_x) max_x = x;
                if (y > max_y) max_y = y;
            }
        }
    }
    if (out_pixel_count == 0) {
        out_model_bbox = cv::Rect2i(0, 0, 0, 0);
        out_mask_within_bbox = cv::Mat();
        return;
    }
    out_model_bbox = cv::Rect2i(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1);
    // Clone so the returned Mat owns its memory (mask_full goes out of scope
    // when the caller stores out_mask_within_bbox into the result envelope).
    out_mask_within_bbox = mask_full(out_model_bbox).clone();
}

} // anonymous namespace

PhysicalInstanceSegmenter::PhysicalInstanceSegmenter()  = default;
PhysicalInstanceSegmenter::~PhysicalInstanceSegmenter() = default;

namespace {

// Verify that every name listed in `expected` is present in the session's I/O
// list, then store them as both std::string (owner) and `const char*` (ORT).
// Mismatch is a hard error (Rule 20: no silent fallback to "first input").
template <typename GetNameFn>
void CacheOneIoSide(size_t n,
                    GetNameFn get_name_fn,
                    const std::vector<std::string>& expected,
                    std::vector<std::string>& storage,
                    std::vector<const char*>& ptrs,
                    const std::string& session_label,
                    const char* kind)
{
    Ort::AllocatorWithDefaultOptions alloc;
    std::vector<std::string> actual;
    actual.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        Ort::AllocatedStringPtr name = get_name_fn(i, alloc);
        actual.emplace_back(name.get());
    }
    for (const auto& want : expected) {
        if (std::find(actual.begin(), actual.end(), want) == actual.end()) {
            std::string have;
            for (const auto& a : actual) { have += (have.empty() ? "" : ","); have += a; }
            throw std::runtime_error(
                "PhysicalInstanceSegmenter(" + session_label + "): expected " + std::string(kind)
                + " name '" + want + "' not present in session. Have: [" + have + "]");
        }
    }
    storage = expected;
    ptrs.clear();
    ptrs.reserve(storage.size());
    for (const auto& s : storage) ptrs.push_back(s.c_str());
}

void CacheSessionIoNames(Ort::Session& sess,
                         const std::vector<std::string>& expected_inputs,
                         const std::vector<std::string>& expected_outputs,
                         std::vector<std::string>& in_storage,
                         std::vector<std::string>& out_storage,
                         std::vector<const char*>& in_ptrs,
                         std::vector<const char*>& out_ptrs,
                         const std::string& session_label)
{
    CacheOneIoSide(
        sess.GetInputCount(),
        [&](size_t i, Ort::AllocatorWithDefaultOptions& a){ return sess.GetInputNameAllocated(i, a); },
        expected_inputs, in_storage, in_ptrs, session_label, "input");
    CacheOneIoSide(
        sess.GetOutputCount(),
        [&](size_t i, Ort::AllocatorWithDefaultOptions& a){ return sess.GetOutputNameAllocated(i, a); },
        expected_outputs, out_storage, out_ptrs, session_label, "output");
}

} // anonymous namespace

void PhysicalInstanceSegmenter::LoadOnnxModelsIntoPhysicalInstanceSegmenter(
    const PhysicalInstanceSegmenterConfig& cfg)
{
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_              = cfg;
    ort_.reset();
    last_error_reason_.clear();
    inference_count_  = 0;

    if (cfg.encoder_onnx_path.empty() && cfg.decoder_onnx_path.empty()) {
        state_ = PhysicalImageOperatorState::NoModelConfigured;
        LOG_DEBUG(PHYSICAL_PERC_PRIM_LOG_TAG,
                  "PhysicalInstanceSegmenter: NoModelConfigured (both ONNX paths empty)");
        return;
    }
    // Partial config is a hard error — silent fallback would be a Rule-20 violation.
    if (cfg.encoder_onnx_path.empty() || cfg.decoder_onnx_path.empty()) {
        last_error_reason_ =
            "LoadOnnxModelsIntoPhysicalInstanceSegmenter: BOTH encoder_onnx_path and "
            "decoder_onnx_path must be set, or BOTH empty. Got encoder='"
            + cfg.encoder_onnx_path + "' decoder='" + cfg.decoder_onnx_path + "'";
        state_ = PhysicalImageOperatorState::ModelLoadFailed;
        LOG_ERROR(PHYSICAL_PERC_PRIM_LOG_TAG, last_error_reason_);
        throw std::runtime_error(last_error_reason_);
    }
    if (cfg.encoder_input_width <= 0 || cfg.encoder_input_height <= 0) {
        last_error_reason_ = "LoadOnnxModelsIntoPhysicalInstanceSegmenter: encoder_input_width/height must be positive";
        state_ = PhysicalImageOperatorState::ModelLoadFailed;
        LOG_ERROR(PHYSICAL_PERC_PRIM_LOG_TAG, last_error_reason_);
        throw std::runtime_error(last_error_reason_);
    }
    if (cfg.decoder_mask_resolution <= 0) {
        last_error_reason_ = "LoadOnnxModelsIntoPhysicalInstanceSegmenter: decoder_mask_resolution must be positive";
        state_ = PhysicalImageOperatorState::ModelLoadFailed;
        LOG_ERROR(PHYSICAL_PERC_PRIM_LOG_TAG, last_error_reason_);
        throw std::runtime_error(last_error_reason_);
    }

    try {
        auto impl = std::make_unique<PhysicalInstanceSegmenterOrtImpl>();

        Ort::SessionOptions opts;
        const int n_threads = cfg.intra_op_num_threads > 0
            ? cfg.intra_op_num_threads
            : std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
        opts.SetIntraOpNumThreads(n_threads);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        // Note: this vcpkg onnxruntime port (1.23.2, arm64-osx) ships only the
        // built-in CPU execution provider — no CoreML, no CUDA. To enable
        // CoreML/ANE acceleration in the future, the vcpkg port must be
        // rebuilt from source with `--use_coreml` and a `coreml` feature
        // gate added; until then we run on CPU. Rule 20: do not silently
        // claim CoreML when the symbol is unavailable.
        const char* ep_label = "CPU";

        impl->encoder_session = std::make_unique<Ort::Session>(
            impl->env, cfg.encoder_onnx_path.c_str(), opts);
        impl->decoder_session = std::make_unique<Ort::Session>(
            impl->env, cfg.decoder_onnx_path.c_str(), opts);

        // Cache I/O names in the order the per-frame Run() calls expect.
        CacheSessionIoNames(
            *impl->encoder_session,
            /*inputs*/  { cfg.encoder_input_name },
            /*outputs*/ { cfg.encoder_output_image_embed_name,
                          cfg.encoder_output_high_res_feats_0_name,
                          cfg.encoder_output_high_res_feats_1_name },
            impl->enc_input_names_storage, impl->enc_output_names_storage,
            impl->enc_input_names,         impl->enc_output_names,
            "encoder");

        CacheSessionIoNames(
            *impl->decoder_session,
            /*inputs*/  { cfg.decoder_input_image_embed_name,
                          cfg.decoder_input_high_res_feats_0_name,
                          cfg.decoder_input_high_res_feats_1_name,
                          cfg.decoder_input_point_coords_name,
                          cfg.decoder_input_point_labels_name,
                          cfg.decoder_input_mask_input_name,
                          cfg.decoder_input_has_mask_input_name },
            /*outputs*/ { cfg.decoder_output_masks_name,
                          cfg.decoder_output_iou_predictions_name },
            impl->dec_input_names_storage, impl->dec_output_names_storage,
            impl->dec_input_names,         impl->dec_output_names,
            "decoder");

        ort_   = std::move(impl);
        state_ = PhysicalImageOperatorState::ModelLoaded;
        LOG_DEBUG(PHYSICAL_PERC_PRIM_LOG_TAG,
                  std::string("PhysicalInstanceSegmenter: ModelLoaded ep=") + ep_label
                  + " encoder='" + cfg.encoder_onnx_path
                  + "' decoder='" + cfg.decoder_onnx_path + "'");
    } catch (const std::exception& e) {
        ort_.reset();
        last_error_reason_ = std::string("LoadOnnxModelsIntoPhysicalInstanceSegmenter failed: ") + e.what();
        state_             = PhysicalImageOperatorState::ModelLoadFailed;
        LOG_ERROR(PHYSICAL_PERC_PRIM_LOG_TAG, last_error_reason_);
        throw;
    }
}

void PhysicalInstanceSegmenter::RouteFrameAndDetectionsToPhysicalInstanceSegmenter(
    const cv::Mat& model_image,
    const std::vector<PhysicalObjectDetection>& detections,
    uint64_t source_frame_counter,
    PhysicalInstanceSegmenterOutput& out)
{
    out = PhysicalInstanceSegmenterOutput{};
    out.last_frame_counter = source_frame_counter;

    std::lock_guard<std::mutex> lk(mutex_);
    out.state             = state_;
    out.last_error_reason = last_error_reason_;
    out.inference_count   = inference_count_;
    if (state_ != PhysicalImageOperatorState::ModelLoaded) return;
    if (!ort_ || !ort_->encoder_session || !ort_->decoder_session) {
        out.state = PhysicalImageOperatorState::InferenceFailed;
        out.last_error_reason = "PhysicalInstanceSegmenter: ORT sessions null while state==ModelLoaded";
        LOG_ERROR(PHYSICAL_PERC_PRIM_LOG_TAG, out.last_error_reason);
        return;
    }
    if (model_image.empty() || model_image.type() != CV_8UC3) {
        out.state = PhysicalImageOperatorState::InferenceFailed;
        out.last_error_reason = "PhysicalInstanceSegmenter: model_image is empty or not CV_8UC3";
        LOG_ERROR(PHYSICAL_PERC_PRIM_LOG_TAG, out.last_error_reason);
        return;
    }

    // Filter prompts BEFORE running the encoder — if there are zero prompts
    // this frame, skip the whole pass. The encoder is the expensive piece
    // (~tens to hundreds of ms); running it for nothing wastes wall time.
    std::vector<size_t> prompt_indices;
    prompt_indices.reserve(detections.size());
    for (size_t i = 0; i < detections.size(); ++i) {
        if (detections[i].confidence >= cfg_.min_prompt_confidence) {
            prompt_indices.push_back(i);
        }
    }
    if (cfg_.max_prompts_per_frame > 0
        && static_cast<int>(prompt_indices.size()) > cfg_.max_prompts_per_frame) {
        // Keep the highest-confidence prompts.
        std::sort(prompt_indices.begin(), prompt_indices.end(),
                  [&](size_t a, size_t b){
                      return detections[a].confidence > detections[b].confidence;
                  });
        prompt_indices.resize(static_cast<size_t>(cfg_.max_prompts_per_frame));
    }
    out.prompt_count = static_cast<int32_t>(prompt_indices.size());
    if (prompt_indices.empty()) {
        // Successful no-op: state stays ModelLoaded, no decode happens, and
        // the consumer sees an empty instances vector with prompt_count=0.
        // This is NOT an error.
        out.state = PhysicalImageOperatorState::ModelLoaded;
        out.last_error_reason.clear();
        last_error_reason_.clear();
        return;
    }

    const auto t0 = std::chrono::steady_clock::now();
    try {
        // ── 1. Letterbox model_image into encoder input geometry ────────────
        cv::Mat enc_in_bgr;
        const LetterboxXform model_to_enc =
            LetterboxResize(model_image, cfg_.encoder_input_width, cfg_.encoder_input_height,
                            enc_in_bgr);

        // ── 2. Encoder forward (ONNX Runtime) ───────────────────────────────
        const auto t_enc_start = std::chrono::steady_clock::now();
        cv::Mat blob = BuildSamEncoderBlob(enc_in_bgr, cfg_);
        // blob is contiguous CV_32F NCHW [1,3,H,W]; wrap as Ort::Value.
        const int64_t enc_in_shape[4] = {
            1, 3,
            static_cast<int64_t>(cfg_.encoder_input_height),
            static_cast<int64_t>(cfg_.encoder_input_width),
        };
        const size_t enc_in_count = static_cast<size_t>(enc_in_shape[0])
                                  * static_cast<size_t>(enc_in_shape[1])
                                  * static_cast<size_t>(enc_in_shape[2])
                                  * static_cast<size_t>(enc_in_shape[3]);
        Ort::Value enc_in_tensor = Ort::Value::CreateTensor<float>(
            ort_->cpu_memory_info, blob.ptr<float>(), enc_in_count, enc_in_shape, 4);

        std::vector<Ort::Value> enc_outs = ort_->encoder_session->Run(
            Ort::RunOptions{nullptr},
            ort_->enc_input_names.data(),  &enc_in_tensor, 1,
            ort_->enc_output_names.data(), ort_->enc_output_names.size());
        if (enc_outs.size() != 3) {
            throw std::runtime_error(
                "Encoder.Run returned " + std::to_string(enc_outs.size())
                + " tensors (expected 3: image_embed, high_res_feats_0, high_res_feats_1)");
        }
        for (size_t i = 0; i < 3; ++i) {
            if (!enc_outs[i].IsTensor()) {
                throw std::runtime_error("Encoder output " + std::to_string(i) + " is not a tensor");
            }
        }
        out.last_encoder_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t_enc_start).count();

        // ── 3. Per-prompt decoder pass ──────────────────────────────────────
        // Pre-build the constant mask_input (zeros) and has_mask_input (=0)
        // tensors once; SAM 2 expects them on every call, but we never feed
        // a previous mask back in this single-frame implementation.
        const int M = cfg_.decoder_mask_resolution;
        cv::Mat mask_input(std::vector<int>{1, 1, M, M}, CV_32F, cv::Scalar(0));
        cv::Mat has_mask_input(std::vector<int>{1}, CV_32F, cv::Scalar(0));
        const int64_t mask_input_shape[4]    = {1, 1, M, M};
        const int64_t has_mask_shape[1]      = {1};
        const int64_t point_coords_shape[3]  = {1, 2, 2};
        const int64_t point_labels_shape[2]  = {1, 2};

        out.segmentation.instances.reserve(prompt_indices.size());
        const auto t_dec_total_start = std::chrono::steady_clock::now();

        for (size_t k = 0; k < prompt_indices.size(); ++k) {
            const size_t det_index = prompt_indices[k];
            const auto&  det       = detections[det_index];

            // Build [1, 2, 2] coords + [1, 2] labels in ENCODER pixel space.
            float pts[2][2];
            BoxToSamPointPair(det.model_box, model_image.cols, model_image.rows,
                              model_to_enc, pts);
            cv::Mat point_coords(std::vector<int>{1, 2, 2}, CV_32F);
            point_coords.ptr<float>(0, 0)[0] = pts[0][0];
            point_coords.ptr<float>(0, 0)[1] = pts[0][1];
            point_coords.ptr<float>(0, 1)[0] = pts[1][0];
            point_coords.ptr<float>(0, 1)[1] = pts[1][1];
            cv::Mat point_labels(std::vector<int>{1, 2}, CV_32F);
            // SAM convention: label 2 = box top-left, 3 = box bottom-right.
            point_labels.ptr<float>(0)[0] = 2.0f;
            point_labels.ptr<float>(0)[1] = 3.0f;

            // Decoder input order MUST match the order cached in
            // dec_input_names_storage in LoadOnnxModelsIntoPhysicalInstanceSegmenter:
            //   image_embed, high_res_feats_0, high_res_feats_1,
            //   point_coords, point_labels, mask_input, has_mask_input
            std::array<Ort::Value, 7> dec_in_tensors = {
                Ort::Value(nullptr), Ort::Value(nullptr), Ort::Value(nullptr),
                Ort::Value(nullptr), Ort::Value(nullptr), Ort::Value(nullptr),
                Ort::Value(nullptr),
            };
            // Reuse encoder output buffers directly (no copy). We construct
            // fresh non-owning Ort::Values pointing at the same float data so
            // dec_in_tensors can be passed by const* to Run() without taking
            // ownership of enc_outs[i].
            for (int i = 0; i < 3; ++i) {
                auto info  = enc_outs[i].GetTensorTypeAndShapeInfo();
                auto shape = info.GetShape();
                size_t ecnt = info.GetElementCount();
                dec_in_tensors[i] = Ort::Value::CreateTensor<float>(
                    ort_->cpu_memory_info,
                    enc_outs[i].GetTensorMutableData<float>(),
                    ecnt, shape.data(), shape.size());
            }
            dec_in_tensors[3] = Ort::Value::CreateTensor<float>(
                ort_->cpu_memory_info, point_coords.ptr<float>(),
                /*count*/ 4, point_coords_shape, 3);
            dec_in_tensors[4] = Ort::Value::CreateTensor<float>(
                ort_->cpu_memory_info, point_labels.ptr<float>(),
                /*count*/ 2, point_labels_shape, 2);
            dec_in_tensors[5] = Ort::Value::CreateTensor<float>(
                ort_->cpu_memory_info, mask_input.ptr<float>(),
                /*count*/ static_cast<size_t>(M) * static_cast<size_t>(M),
                mask_input_shape, 4);
            dec_in_tensors[6] = Ort::Value::CreateTensor<float>(
                ort_->cpu_memory_info, has_mask_input.ptr<float>(),
                /*count*/ 1, has_mask_shape, 1);

            std::vector<Ort::Value> dec_outs = ort_->decoder_session->Run(
                Ort::RunOptions{nullptr},
                ort_->dec_input_names.data(),  dec_in_tensors.data(), dec_in_tensors.size(),
                ort_->dec_output_names.data(), ort_->dec_output_names.size());
            if (dec_outs.size() != 2) {
                throw std::runtime_error(
                    "Decoder.Run returned " + std::to_string(dec_outs.size())
                    + " tensors (expected 2: masks, iou_predictions) for prompt "
                    + std::to_string(k));
            }

            auto masks_info  = dec_outs[0].GetTensorTypeAndShapeInfo();
            auto masks_shape = masks_info.GetShape();
            if (masks_shape.size() != 4 || masks_shape[0] != 1) {
                std::string s = "[";
                for (size_t i = 0; i < masks_shape.size(); ++i) { if (i) s += ","; s += std::to_string(masks_shape[i]); }
                s += "]";
                throw std::runtime_error(
                    "Decoder masks output has shape " + s
                    + " — expected [1, M, " + std::to_string(M) + ", " + std::to_string(M) + "]");
            }
            const int Mh = static_cast<int>(masks_shape[2]);
            const int Mw = static_cast<int>(masks_shape[3]);
            if (Mh != M || Mw != M) {
                throw std::runtime_error(
                    "Decoder masks spatial dims [" + std::to_string(Mh) + "x"
                    + std::to_string(Mw) + "] do not match decoder_mask_resolution="
                    + std::to_string(M));
            }
            // Pick channel 0 (highest-IoU mask when multimask_output=False;
            // for multimask exports we still use 0 — caller can switch via
            // a future cfg field).
            cv::Mat mask_logits_lo(Mh, Mw, CV_32F);
            const float* mptr = dec_outs[0].GetTensorMutableData<float>();
            std::memcpy(mask_logits_lo.data, mptr,
                        sizeof(float) * static_cast<size_t>(Mh) * static_cast<size_t>(Mw));

            float mask_iou = 0.0f;
            auto iou_info  = dec_outs[1].GetTensorTypeAndShapeInfo();
            auto iou_shape = iou_info.GetShape();
            if (iou_shape.size() == 2 && iou_shape[0] == 1 && iou_shape[1] >= 1) {
                mask_iou = dec_outs[1].GetTensorMutableData<float>()[0];
            }

            cv::Rect2i model_bbox;
            cv::Mat    mask_within_bbox;
            int        pixel_count = 0;
            DecodeOneMaskToModelSpace(mask_logits_lo, M, model_to_enc,
                                      model_image.cols, model_image.rows,
                                      cfg_.encoder_input_width, cfg_.encoder_input_height,
                                      cfg_.mask_logit_threshold,
                                      model_bbox, mask_within_bbox, pixel_count);

            PhysicalInstanceMask inst;
            inst.source_detection_index = static_cast<int32_t>(det_index);
            inst.class_id               = det.class_id;
            inst.class_label            = det.class_label;
            inst.detection_confidence   = det.confidence;
            inst.mask_confidence        = mask_iou;
            inst.prompt_model_box       = det.model_box;
            inst.mask_model_bbox        = model_bbox;
            inst.mask_within_bbox       = mask_within_bbox;
            inst.mask_pixel_count       = pixel_count;
            const double total_px =
                static_cast<double>(model_image.cols) * static_cast<double>(model_image.rows);
            inst.mask_area_fraction = total_px > 0.0
                ? static_cast<float>(pixel_count / total_px)
                : 0.0f;
            out.segmentation.instances.push_back(std::move(inst));
        }

        out.last_decoder_total_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t_dec_total_start).count();

        ++inference_count_;
        out.inference_count   = inference_count_;
        out.last_inference_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
        out.state             = PhysicalImageOperatorState::ModelLoaded;
        out.last_error_reason.clear();
        last_error_reason_.clear();
    } catch (const std::exception& e) {
        last_error_reason_ = std::string("RouteFrameAndDetectionsToPhysicalInstanceSegmenter failed: ") + e.what();
        state_             = PhysicalImageOperatorState::InferenceFailed;
        out.state          = PhysicalImageOperatorState::InferenceFailed;
        out.last_error_reason = last_error_reason_;
        LOG_ERROR(PHYSICAL_PERC_PRIM_LOG_TAG, last_error_reason_);
    }
}

void PhysicalInstanceSegmenter::ResetPhysicalInstanceSegmenter() {
    std::lock_guard<std::mutex> lk(mutex_);
    ort_.reset();
    cfg_              = PhysicalInstanceSegmenterConfig{};
    last_error_reason_.clear();
    inference_count_  = 0;
    state_            = PhysicalImageOperatorState::NoModelConfigured;
}

PhysicalImageOperatorState PhysicalInstanceSegmenter::GetPhysicalInstanceSegmenterState() const {
    std::lock_guard<std::mutex> lk(mutex_); return state_;
}
std::string PhysicalInstanceSegmenter::GetPhysicalInstanceSegmenterLastError() const {
    std::lock_guard<std::mutex> lk(mutex_); return last_error_reason_;
}
bool PhysicalInstanceSegmenter::IsPhysicalInstanceSegmenterReady() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return state_ == PhysicalImageOperatorState::ModelLoaded
        && ort_ != nullptr
        && ort_->encoder_session != nullptr
        && ort_->decoder_session != nullptr;
}

}}} // namespace GRIM::Perception::Physical
