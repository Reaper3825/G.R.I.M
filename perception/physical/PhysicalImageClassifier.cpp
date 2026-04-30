#include "PhysicalImageClassifier.hpp"

#include "PhysicalPerceptionPrimitivesLogTag.hpp"
#include "logger.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include <opencv2/dnn.hpp>     // blobFromImage helper only (CPU preprocessing)
#include <opencv2/imgproc.hpp>

#include <onnxruntime_cxx_api.h>

namespace GRIM { namespace Perception { namespace Physical {

// PIMPL: holds the onnxruntime session + cached I/O names. Definition
// is hidden in the .cpp so the public header doesn't pull onnxruntime in.
struct PhysicalImageClassifierOrtImpl {
    Ort::Env                       env{ORT_LOGGING_LEVEL_WARNING, "GRIM.PhysicalImageClassifier"};
    Ort::MemoryInfo                cpu_memory_info{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
    std::unique_ptr<Ort::Session>  session;
    std::string                    input_name;   // resolved from session
    std::string                    output_name;  // resolved from session
};

namespace {

std::vector<std::string> LoadLabelsFile(const std::string& path) {
    std::vector<std::string> out;
    if (path.empty()) {
        throw std::runtime_error(
            "PhysicalImageClassifier: class_names_path is empty (CLIP zero-shot "
            "requires a prompts file)");
    }
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error(
            "PhysicalImageClassifier: failed to open class_names_path='" + path + "'");
    }
    std::string line;
    while (std::getline(f, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t'))
            line.pop_back();
        if (!line.empty() && line[0] != '#') out.push_back(line);
    }
    return out;
}

// Loads precomputed CLIP text embeddings produced by
// scripts/setup_mobileclip.py. Throws on any inconsistency — Rule 20:
// CLIP without a valid embedding bank is meaningless, do not silently
// fall through.
//
//   uint32 N (little-endian) | uint32 D (little-endian) | float32[N*D]
cv::Mat LoadTextEmbeddingsFile(const std::string& path,
                               int expected_rows,
                               int& out_dim) {
    if (path.empty()) {
        throw std::runtime_error(
            "PhysicalImageClassifier: text_embeddings_path is empty");
    }
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error(
            "PhysicalImageClassifier: failed to open text_embeddings_path='" + path + "'");
    }
    std::uint32_t n = 0, d = 0;
    f.read(reinterpret_cast<char*>(&n), sizeof(n));
    f.read(reinterpret_cast<char*>(&d), sizeof(d));
    if (!f) {
        throw std::runtime_error(
            "PhysicalImageClassifier: text_embeddings header read failed for '" + path + "'");
    }
    if (n == 0 || d == 0 || n > 100000 || d > 8192) {
        throw std::runtime_error(
            "PhysicalImageClassifier: text_embeddings header has implausible "
            "dimensions N=" + std::to_string(n) + " D=" + std::to_string(d));
    }
    if (expected_rows > 0 && static_cast<int>(n) != expected_rows) {
        throw std::runtime_error(
            "PhysicalImageClassifier: text_embeddings row count " +
            std::to_string(n) + " does not match prompts file label count " +
            std::to_string(expected_rows));
    }
    cv::Mat emb(static_cast<int>(n), static_cast<int>(d), CV_32F);
    f.read(reinterpret_cast<char*>(emb.ptr<float>()),
           static_cast<std::streamsize>(sizeof(float)) *
               static_cast<std::streamsize>(n) *
               static_cast<std::streamsize>(d));
    if (!f) {
        throw std::runtime_error(
            "PhysicalImageClassifier: text_embeddings payload read failed for '" + path + "'");
    }
    out_dim = static_cast<int>(d);
    return emb;
}

// Numerically-stable softmax over a flat float row.
void SoftmaxInPlace(float* p, int n) {
    if (n <= 0) return;
    float m = p[0];
    for (int i = 1; i < n; ++i) if (p[i] > m) m = p[i];
    double s = 0.0;
    for (int i = 0; i < n; ++i) { p[i] = std::exp(p[i] - m); s += p[i]; }
    if (s <= 0.0) { for (int i = 0; i < n; ++i) p[i] = 1.0f / static_cast<float>(n); return; }
    const float inv = static_cast<float>(1.0 / s);
    for (int i = 0; i < n; ++i) p[i] *= inv;
}

// Divide each plane of a [1, C, H, W] float blob by its per-channel std.
// blobFromImage handles (image*scale - mean); std div has to be done here.
void ApplyPerChannelStdDiv(cv::Mat& blob, const cv::Scalar& std_bgr_or_rgb) {
    if (blob.dims != 4 || blob.size[0] != 1) {
        throw std::runtime_error(
            "PhysicalImageClassifier: blob must be 1xCxHxW, got dims=" +
            std::to_string(blob.dims));
    }
    const int C = blob.size[1];
    const int H = blob.size[2];
    const int W = blob.size[3];
    const int plane = H * W;
    if (C != 3) {
        throw std::runtime_error(
            "PhysicalImageClassifier: expected 3-channel blob, got C=" +
            std::to_string(C));
    }
    float* base = blob.ptr<float>();
    for (int c = 0; c < C; ++c) {
        const double sd = std_bgr_or_rgb[c];
        if (sd <= 0.0) {
            throw std::runtime_error(
                "PhysicalImageClassifier: input_std channel " +
                std::to_string(c) + " is non-positive");
        }
        const float inv = static_cast<float>(1.0 / sd);
        float* p = base + c * plane;
        for (int i = 0; i < plane; ++i) p[i] *= inv;
    }
}

void L2NormaliseRow(float* p, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += static_cast<double>(p[i]) * p[i];
    if (s <= 0.0) {
        throw std::runtime_error(
            "PhysicalImageClassifier: image embedding has zero norm");
    }
    const float inv = static_cast<float>(1.0 / std::sqrt(s));
    for (int i = 0; i < n; ++i) p[i] *= inv;
}

} // anonymous

PhysicalImageClassifier::PhysicalImageClassifier()  = default;
PhysicalImageClassifier::~PhysicalImageClassifier() = default;

void PhysicalImageClassifier::LoadOnnxModelIntoPhysicalImageClassifier(
    const PhysicalImageClassifierConfig& cfg)
{
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_              = cfg;
    ort_.reset();
    class_labels_.clear();
    text_embeddings_  = cv::Mat();
    embedding_dim_    = 0;
    last_error_reason_.clear();
    inference_count_  = 0;
    if (cfg.onnx_model_path.empty()) {
        state_ = PhysicalImageOperatorState::NoModelConfigured;
        return;
    }
    if (cfg.top_k <= 0) {
        last_error_reason_ = "PhysicalImageClassifier: top_k must be > 0";
        state_             = PhysicalImageOperatorState::ModelLoadFailed;
        LOG_ERROR(PHYSICAL_PERC_PRIM_LOG_TAG, last_error_reason_);
        throw std::runtime_error(last_error_reason_);
    }
    try {
        // 1) Image encoder via onnxruntime. cv::dnn cannot parse the
        //    MobileCLIP-S0 graph (mixed CNN + transformer ops produced by
        //    torch.onnx dynamo exporter) and segfaults during load.
        auto impl = std::make_unique<PhysicalImageClassifierOrtImpl>();
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    #ifdef _WIN32
            std::wstring onnx_model_path_w(cfg.onnx_model_path.begin(), cfg.onnx_model_path.end());
            impl->session = std::make_unique<Ort::Session>(
                impl->env, onnx_model_path_w.c_str(), opts);
    #else
        impl->session = std::make_unique<Ort::Session>(
            impl->env, cfg.onnx_model_path.c_str(), opts);
    #endif

        // Resolve and cache I/O names. MobileCLIP exporter emits a single
        // input ('image') and single output ('embedding').
        if (impl->session->GetInputCount() != 1 ||
            impl->session->GetOutputCount() != 1) {
            throw std::runtime_error(
                "PhysicalImageClassifier: expected 1 input and 1 output, got " +
                std::to_string(impl->session->GetInputCount()) + " / " +
                std::to_string(impl->session->GetOutputCount()));
        }
        Ort::AllocatorWithDefaultOptions alloc;
        {
            Ort::AllocatedStringPtr in  = impl->session->GetInputNameAllocated(0, alloc);
            Ort::AllocatedStringPtr out = impl->session->GetOutputNameAllocated(0, alloc);
            impl->input_name  = in.get();
            impl->output_name = out.get();
        }

        // 2) Prompts (one label per row).
        std::vector<std::string> labels = LoadLabelsFile(cfg.class_names_path);
        if (labels.empty()) {
            throw std::runtime_error(
                "PhysicalImageClassifier: prompts file '" + cfg.class_names_path +
                "' is empty");
        }

        // 3) Precomputed text embeddings — must align with labels.
        int dim = 0;
        cv::Mat embeddings = LoadTextEmbeddingsFile(
            cfg.text_embeddings_path, static_cast<int>(labels.size()), dim);

        ort_              = std::move(impl);
        class_labels_     = std::move(labels);
        text_embeddings_  = std::move(embeddings);
        embedding_dim_    = dim;
        state_            = PhysicalImageOperatorState::ModelLoaded;
        LOG_DEBUG(PHYSICAL_PERC_PRIM_LOG_TAG,
                  std::string("PhysicalImageClassifier: ModelLoaded (CLIP zero-shot, ORT) onnx='") +
                  cfg.onnx_model_path + "' prompts=" + std::to_string(class_labels_.size()) +
                  " dim=" + std::to_string(embedding_dim_) +
                  " input='" + ort_->input_name + "' output='" + ort_->output_name + "'");
    } catch (const std::exception& e) {
        ort_.reset();
        text_embeddings_  = cv::Mat();
        embedding_dim_    = 0;
        last_error_reason_ = std::string("LoadOnnxModelIntoPhysicalImageClassifier failed: ") + e.what();
        state_             = PhysicalImageOperatorState::ModelLoadFailed;
        LOG_ERROR(PHYSICAL_PERC_PRIM_LOG_TAG, last_error_reason_);
        throw;
    }
}

void PhysicalImageClassifier::RouteFrameToPhysicalImageClassifier(
    const cv::Mat& model_image,
    uint64_t source_frame_counter,
    PhysicalImageClassifierOutput& out)
{
    out = PhysicalImageClassifierOutput{};
    out.last_frame_counter = source_frame_counter;

    std::lock_guard<std::mutex> lk(mutex_);
    out.state             = state_;
    out.last_error_reason = last_error_reason_;
    if (state_ != PhysicalImageOperatorState::ModelLoaded) return;
    if (!ort_ || !ort_->session) {
        out.state = PhysicalImageOperatorState::InferenceFailed;
        out.last_error_reason = "PhysicalImageClassifier: ort session is null while state==ModelLoaded";
        LOG_ERROR(PHYSICAL_PERC_PRIM_LOG_TAG, out.last_error_reason);
        return;
    }
    if (text_embeddings_.empty() || embedding_dim_ <= 0) {
        out.state = PhysicalImageOperatorState::InferenceFailed;
        out.last_error_reason = "PhysicalImageClassifier: text embeddings missing";
        LOG_ERROR(PHYSICAL_PERC_PRIM_LOG_TAG, out.last_error_reason);
        return;
    }
    if (model_image.empty() || model_image.type() != CV_8UC3) {
        out.state = PhysicalImageOperatorState::InferenceFailed;
        out.last_error_reason = "PhysicalImageClassifier: model_image is empty or not CV_8UC3";
        LOG_ERROR(PHYSICAL_PERC_PRIM_LOG_TAG, out.last_error_reason);
        return;
    }

    const auto t0 = std::chrono::steady_clock::now();
    try {
        // Letterbox-free centred preprocessing: resize directly to the
        // encoder's expected resolution. CLIP image encoders are robust to
        // non-square inputs being squished.
        cv::Mat blob = cv::dnn::blobFromImage(
            model_image, cfg_.input_scale,
            cv::Size(cfg_.input_width, cfg_.input_height),
            cfg_.input_mean, cfg_.swap_rb, /*crop=*/false);
        ApplyPerChannelStdDiv(blob, cfg_.input_std);

        // Wrap blob as Ort::Value (non-owning view of the cv::Mat buffer).
        const std::array<int64_t, 4> in_shape = {
            1, 3,
            static_cast<int64_t>(cfg_.input_height),
            static_cast<int64_t>(cfg_.input_width)};
        const size_t in_count = static_cast<size_t>(
            in_shape[0] * in_shape[1] * in_shape[2] * in_shape[3]);
        Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
            ort_->cpu_memory_info, blob.ptr<float>(), in_count,
            in_shape.data(), in_shape.size());

        const char* in_names[]  = { ort_->input_name.c_str()  };
        const char* out_names[] = { ort_->output_name.c_str() };
        std::vector<Ort::Value> outs = ort_->session->Run(
            Ort::RunOptions{nullptr}, in_names, &in_tensor, 1, out_names, 1);
        if (outs.size() != 1) {
            throw std::runtime_error(
                "PhysicalImageClassifier: ORT Run returned " +
                std::to_string(outs.size()) + " outputs (expected 1)");
        }
        // Output shape must be [1, embedding_dim_].
        auto out_info  = outs[0].GetTensorTypeAndShapeInfo();
        auto out_shape = out_info.GetShape();
        const int64_t out_total = out_info.GetElementCount();
        if (out_shape.size() != 2 || out_shape[0] != 1 ||
            out_shape[1] != static_cast<int64_t>(embedding_dim_)) {
            std::string sh;
            for (auto d : out_shape) sh += (sh.empty() ? "" : "x") + std::to_string(d);
            throw std::runtime_error(
                "PhysicalImageClassifier: encoder output shape [" + sh +
                "] does not match expected [1x" + std::to_string(embedding_dim_) + "]");
        }
        const float* out_ptr = outs[0].GetTensorData<float>();
        cv::Mat img_emb_row(1, embedding_dim_, CV_32F);
        std::memcpy(img_emb_row.ptr<float>(0), out_ptr,
                    sizeof(float) * static_cast<size_t>(out_total));
        L2NormaliseRow(img_emb_row.ptr<float>(0), embedding_dim_);

        // Cosine similarity against all prompts: [1, D] * [D, N] = [1, N].
        // text_embeddings_ is [N, D] L2-normalised.
        cv::Mat sims = img_emb_row * text_embeddings_.t();   // [1, N]
        sims *= cfg_.temperature;                            // CLIP logit scale

        const int N = sims.cols;
        std::vector<float> probs(static_cast<size_t>(N));
        std::memcpy(probs.data(), sims.ptr<float>(0), sizeof(float) * static_cast<size_t>(N));
        SoftmaxInPlace(probs.data(), N);

        std::vector<int> idx(static_cast<size_t>(N));
        for (int i = 0; i < N; ++i) idx[static_cast<size_t>(i)] = i;
        const int K = std::min(cfg_.top_k, N);
        std::partial_sort(idx.begin(), idx.begin() + K, idx.end(),
                          [&](int a, int b) {
                              return probs[static_cast<size_t>(a)] >
                                     probs[static_cast<size_t>(b)];
                          });

        out.top_k.reserve(static_cast<size_t>(K));
        for (int i = 0; i < K; ++i) {
            const int cid = idx[static_cast<size_t>(i)];
            PhysicalImageClassification c;
            c.class_id = cid;
            c.score    = probs[static_cast<size_t>(cid)];
            if (cid >= 0 && cid < static_cast<int>(class_labels_.size())) {
                c.class_label = class_labels_[static_cast<size_t>(cid)];
            } else {
                c.class_label = "class_" + std::to_string(cid);
            }
            out.top_k.push_back(std::move(c));
        }

        ++inference_count_;
        out.inference_count   = inference_count_;
        out.last_inference_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
        out.state             = PhysicalImageOperatorState::ModelLoaded;
        out.last_error_reason.clear();
        last_error_reason_.clear();
    } catch (const std::exception& e) {
        last_error_reason_ = std::string("RouteFrameToPhysicalImageClassifier failed: ") + e.what();
        state_             = PhysicalImageOperatorState::InferenceFailed;
        out.state          = PhysicalImageOperatorState::InferenceFailed;
        out.last_error_reason = last_error_reason_;
        LOG_ERROR(PHYSICAL_PERC_PRIM_LOG_TAG, last_error_reason_);
    }
}

void PhysicalImageClassifier::ResetPhysicalImageClassifier() {
    std::lock_guard<std::mutex> lk(mutex_);
    ort_.reset();
    class_labels_.clear();
    text_embeddings_  = cv::Mat();
    embedding_dim_    = 0;
    cfg_              = PhysicalImageClassifierConfig{};
    last_error_reason_.clear();
    inference_count_  = 0;
    state_            = PhysicalImageOperatorState::NoModelConfigured;
}

PhysicalImageOperatorState PhysicalImageClassifier::GetPhysicalImageClassifierState() const {
    std::lock_guard<std::mutex> lk(mutex_); return state_;
}
std::string PhysicalImageClassifier::GetPhysicalImageClassifierLastError() const {
    std::lock_guard<std::mutex> lk(mutex_); return last_error_reason_;
}

}}} // namespace
