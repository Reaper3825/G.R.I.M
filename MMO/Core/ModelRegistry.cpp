// Multi-Model Orchestration (MMO) - Model Registry
// See ModelRegistry.hpp for interface documentation.
//======================================================//
#include "ModelRegistry.hpp"

#include <nlohmann/json.hpp>  // full definition (fwd-only in header)

#include <sstream>
#include <stdexcept>

namespace GRIM::MMO {

// =========================================================
// Singleton
// =========================================================
ModelRegistry& ModelRegistry::instance() {
    static ModelRegistry s_instance;
    return s_instance;
}

// =========================================================
// Load from ai_config.json
//
// Expected JSON shape:
//   {
//     "mmo": {
//       "enabled": true,
//       "mode": "shadow",
//       "router": { ... ModelInfo fields ... },
//       "sub_models": [ { ... }, { ... } ]
//     }
//   }
// =========================================================
void ModelRegistry::loadFromConfig(const nlohmann::json& config) {
    std::lock_guard<std::mutex> lock(mutex_);

    models_.clear();
    router_id_.clear();
    config_ = MMOConfig{};

    // ---- mmo section must exist ----
    if (!config.contains("mmo") || !config["mmo"].is_object()) {
        throw std::runtime_error(
            "ModelRegistry: ai_config.json is missing required 'mmo' object");
    }

    const auto& mmo = config["mmo"];

    // ---- top-level MMO fields ----
    config_.enabled = mmo.value("enabled", false);
    config_.mode    = mmo.value("mode", "shadow");

    if (config_.mode != "shadow" && config_.mode != "enforced") {
        throw std::runtime_error(
            "ModelRegistry: mmo.mode must be 'shadow' or 'enforced', got '"
            + config_.mode + "'");
    }

    // ---- router (required) ----
    if (!mmo.contains("router") || !mmo["router"].is_object()) {
        throw std::runtime_error(
            "ModelRegistry: mmo.router object is required");
    }

    ModelInfo router = parseModelInfo(mmo["router"]);
    validateModel(router, /*is_router=*/true);
    router_id_     = router.id;
    config_.router_id = router.id;
    models_.emplace(router.id, std::move(router));

    // ---- sub_models (optional array) ----
    if (mmo.contains("sub_models")) {
        if (!mmo["sub_models"].is_array()) {
            throw std::runtime_error(
                "ModelRegistry: mmo.sub_models must be an array");
        }

        for (size_t i = 0; i < mmo["sub_models"].size(); ++i) {
            const auto& entry = mmo["sub_models"][i];
            if (!entry.is_object()) {
                throw std::runtime_error(
                    "ModelRegistry: mmo.sub_models[" + std::to_string(i)
                    + "] is not an object");
            }

            ModelInfo sub = parseModelInfo(entry);
            validateModel(sub, /*is_router=*/false);

            if (models_.count(sub.id)) {
                throw std::runtime_error(
                    "ModelRegistry: duplicate model id '" + sub.id
                    + "' (sub_models[" + std::to_string(i) + "])");
            }

            models_.emplace(sub.id, std::move(sub));
        }
    }
}

// =========================================================
// Queries
// =========================================================

const ModelInfo* ModelRegistry::getRouter() const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = models_.find(router_id_);
    if (it == models_.end()) return nullptr;
    return &it->second;
}

const ModelInfo* ModelRegistry::getModelById(const std::string& id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = models_.find(id);
    if (it == models_.end()) return nullptr;
    return &it->second;
}

std::vector<const ModelInfo*> ModelRegistry::getSubModels() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<const ModelInfo*> result;
    for (const auto& [id, model] : models_) {
        if (id != router_id_) {
            result.push_back(&model);
        }
    }
    return result;
}

std::vector<const ModelInfo*> ModelRegistry::getTextSubModels() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<const ModelInfo*> result;
    for (const auto& [id, model] : models_) {
        if (id == router_id_)               continue;
        if (model.kind != ModelKind::Text)  continue;
        result.push_back(&model);
    }
    return result;
}

std::vector<const ModelInfo*> ModelRegistry::getVisionSubModels() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<const ModelInfo*> result;
    for (const auto& [id, model] : models_) {
        if (id == router_id_)                 continue;
        if (model.kind != ModelKind::Vision)  continue;
        result.push_back(&model);
    }
    return result;
}

std::vector<const ModelInfo*> ModelRegistry::getModelsBySubjectTag(
    const std::string& tag) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<const ModelInfo*> result;
    for (const auto& [id, model] : models_) {
        if (id == router_id_) continue;
        for (const auto& t : model.subject_tags) {
            if (t == tag) {
                result.push_back(&model);
                break;
            }
        }
    }
    return result;
}

std::vector<const ModelInfo*> ModelRegistry::getAllModels() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<const ModelInfo*> result;
    result.reserve(models_.size());
    for (const auto& [id, model] : models_) {
        result.push_back(&model);
    }
    return result;
}

size_t ModelRegistry::modelCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return models_.size();
}

bool ModelRegistry::isEnabled() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_.enabled;
}

const std::string& ModelRegistry::mode() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_.mode;
}

// =========================================================
// Lifecycle
// =========================================================

void ModelRegistry::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    models_.clear();
    router_id_.clear();
    config_ = MMOConfig{};
}

// =========================================================
// Runtime mutation
// =========================================================

const ModelInfo* ModelRegistry::registerModel(ModelInfo model) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Cannot register a router at runtime
    if (!model.lora_path.empty() || !model.hard_copy_path.empty()) {
        throw std::runtime_error(
            "ModelRegistry::registerModel: model '" + model.id
            + "' has lora_path or hard_copy_path set — "
              "only the router may have these. Sub-models are frozen information bricks.");
    }

    validateModel(model, /*is_router=*/false);

    if (models_.count(model.id)) {
        throw std::runtime_error(
            "ModelRegistry::registerModel: duplicate model id '" + model.id + "'");
    }

    auto [it, inserted] = models_.emplace(model.id, std::move(model));
    if (!inserted) {
        throw std::runtime_error(
            "ModelRegistry::registerModel: emplace failed for '" + it->first + "'");
    }
    return &it->second;
}

void ModelRegistry::removeModel(const std::string& id) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (id == router_id_) {
        throw std::runtime_error(
            "ModelRegistry::removeModel: cannot remove the router ('"
            + id + "') at runtime");
    }

    auto it = models_.find(id);
    if (it == models_.end()) {
        throw std::runtime_error(
            "ModelRegistry::removeModel: model '" + id + "' not found");
    }

    models_.erase(it);
}

// =========================================================
// Serialization
// =========================================================

nlohmann::json ModelRegistry::serializeModelToJson(const ModelInfo& model) {
    nlohmann::json j;
    j["id"]          = model.id;
    j["name"]        = model.name;
    j["version"]     = model.version;
    j["subject"]     = model.subject;
    j["description"] = model.description;
    j["model_path"]  = model.model_path;
    j["backend_type"] = backendTypeToString(model.backend_type);
    j["url"]          = model.url;
    j["subject_tags"] = model.subject_tags;
    j["usage_weight"] = model.usage_weight;
    j["estimated_ram_mb"]  = model.estimated_ram_mb;
    j["estimated_vram_mb"] = model.estimated_vram_mb;

    // Capability classification
    j["kind"] = modelKindToString(model.kind);
    if (model.kind == ModelKind::Vision) {
        nlohmann::json v;
        v["operator"]                 = visionOperatorKindToString(model.vision.operator_kind);
        v["class_names_path"]         = model.vision.class_names_path;
        v["text_embeddings_path"]     = model.vision.text_embeddings_path;
        v["input_width"]              = model.vision.input_width;
        v["input_height"]             = model.vision.input_height;
        v["confidence_threshold"]     = model.vision.confidence_threshold;
        v["iou_threshold"]            = model.vision.iou_threshold;
        v["top_k"]                    = model.vision.top_k;
        v["min_keypoint_confidence"]  = model.vision.min_keypoint_confidence;
        v["recogniser_onnx_path"]     = model.vision.recogniser_onnx_path;
        v["recogniser_charset_path"]  = model.vision.recogniser_charset_path;
        v["recogniser_input_grayscale"] = model.vision.recogniser_input_grayscale;
        v["pose_output_format"]       = model.vision.pose_output_format;
        v["num_keypoints"]            = model.vision.num_keypoints;
        v["expression_classifier_onnx_path"]        = model.vision.expression_classifier_onnx_path;
        v["expression_classifier_class_names_path"] = model.vision.expression_classifier_class_names_path;
        v["expression_classifier_input_width"]      = model.vision.expression_classifier_input_width;
        v["expression_classifier_input_height"]     = model.vision.expression_classifier_input_height;
        v["expression_classifier_input_grayscale"]  = model.vision.expression_classifier_input_grayscale;
        v["instance_seg_decoder_onnx_path"]     = model.vision.instance_seg_decoder_onnx_path;
        v["instance_seg_max_prompts_per_frame"] = model.vision.instance_seg_max_prompts_per_frame;
        v["instance_seg_min_prompt_confidence"] = model.vision.instance_seg_min_prompt_confidence;
        j["vision"] = std::move(v);
    }

    // Router-only fields — only emit if non-empty
    if (!model.lora_path.empty())      j["lora_path"]      = model.lora_path;
    if (!model.hard_copy_path.empty()) j["hard_copy_path"] = model.hard_copy_path;

    return j;
}

nlohmann::json ModelRegistry::serializeMMOSection() const {
    std::lock_guard<std::mutex> lock(mutex_);

    nlohmann::json mmo;
    mmo["enabled"] = config_.enabled;
    mmo["mode"]    = config_.mode;

    // Router
    auto rit = models_.find(router_id_);
    if (rit != models_.end()) {
        mmo["router"] = serializeModelToJson(rit->second);
    }

    // Sub-models
    nlohmann::json subs = nlohmann::json::array();
    for (const auto& [id, model] : models_) {
        if (id == router_id_) continue;
        subs.push_back(serializeModelToJson(model));
    }
    mmo["sub_models"] = subs;

    return mmo;
}

// =========================================================
// Parsing helpers
// =========================================================

ModelInfo ModelRegistry::parseModelInfo(const nlohmann::json& entry) {
    ModelInfo m;

    m.id          = entry.value("id", "");
    m.name        = entry.value("name", "");
    m.version     = entry.value("version", "");
    m.subject     = entry.value("subject", "");
    m.description = entry.value("description", "");
    m.model_path  = entry.value("model_path", "");
    m.url         = entry.value("url", "");
    m.usage_weight = entry.value("usage_weight", 0.0f);

    if (entry.contains("backend_type") && entry["backend_type"].is_string()) {
        m.backend_type = parseBackendType(entry["backend_type"].get<std::string>());
    }

    if (entry.contains("subject_tags") && entry["subject_tags"].is_array()) {
        for (const auto& tag : entry["subject_tags"]) {
            if (tag.is_string()) {
                m.subject_tags.push_back(tag.get<std::string>());
            }
        }
    }

    // Router-only fields
    m.lora_path      = entry.value("lora_path", "");
    m.hard_copy_path = entry.value("hard_copy_path", "");

    // Resource estimates
    m.estimated_ram_mb  = entry.value("estimated_ram_mb", 0L);
    m.estimated_vram_mb = entry.value("estimated_vram_mb", 0L);

    // Capability classification (kind defaults to Text)
    if (entry.contains("kind") && entry["kind"].is_string()) {
        m.kind = parseModelKind(entry["kind"].get<std::string>());
    }
    if (entry.contains("vision") && entry["vision"].is_object()) {
        const auto& v = entry["vision"];
        if (v.contains("operator") && v["operator"].is_string()) {
            m.vision.operator_kind = parseVisionOperatorKind(
                v["operator"].get<std::string>());
        }
        m.vision.class_names_path        = v.value("class_names_path", "");
        m.vision.text_embeddings_path    = v.value("text_embeddings_path", "");
        m.vision.input_width             = v.value("input_width", 0);
        m.vision.input_height            = v.value("input_height", 0);
        m.vision.confidence_threshold    = v.value("confidence_threshold", 0.0f);
        m.vision.iou_threshold           = v.value("iou_threshold", 0.0f);
        m.vision.top_k                   = v.value("top_k", 0);
        m.vision.min_keypoint_confidence = v.value("min_keypoint_confidence", 0.0f);
        m.vision.recogniser_onnx_path    = v.value("recogniser_onnx_path", "");
        m.vision.recogniser_charset_path = v.value("recogniser_charset_path", "");
        m.vision.recogniser_input_grayscale = v.value("recogniser_input_grayscale", true);
        m.vision.pose_output_format      = v.value("pose_output_format", "");
        m.vision.num_keypoints           = v.value("num_keypoints", 0);
        m.vision.expression_classifier_onnx_path        = v.value("expression_classifier_onnx_path", "");
        m.vision.expression_classifier_class_names_path = v.value("expression_classifier_class_names_path", "");
        m.vision.expression_classifier_input_width      = v.value("expression_classifier_input_width", 64);
        m.vision.expression_classifier_input_height     = v.value("expression_classifier_input_height", 64);
        m.vision.expression_classifier_input_grayscale  = v.value("expression_classifier_input_grayscale", true);

        // Monocular depth (Stage-3) preprocessing & calibration overrides.
        m.vision.depth_swap_rb              = v.value("depth_swap_rb", true);
        m.vision.depth_input_mean_r         = v.value("depth_input_mean_r", 123.675);
        m.vision.depth_input_mean_g         = v.value("depth_input_mean_g", 116.28);
        m.vision.depth_input_mean_b         = v.value("depth_input_mean_b", 103.53);
        m.vision.depth_input_std_r          = v.value("depth_input_std_r", 58.395);
        m.vision.depth_input_std_g          = v.value("depth_input_std_g", 57.12);
        m.vision.depth_input_std_b          = v.value("depth_input_std_b", 57.375);
        m.vision.depth_input_scale          = v.value("depth_input_scale", 1.0 / 255.0);
        m.vision.depth_output_is_disparity  = v.value("depth_output_is_disparity", true);
        m.vision.depth_metric_scale_meters  = v.value("depth_metric_scale_meters", 0.0);
        m.vision.depth_metric_epsilon       = v.value("depth_metric_epsilon", 1.0e-3);

        // Instance-segmenter (Stage-2 SAM 2)
        m.vision.instance_seg_decoder_onnx_path     = v.value("instance_seg_decoder_onnx_path", "");
        m.vision.instance_seg_max_prompts_per_frame = v.value("instance_seg_max_prompts_per_frame", 0);
        m.vision.instance_seg_min_prompt_confidence = v.value("instance_seg_min_prompt_confidence", 0.0f);
    }

    return m;
}

BackendType ModelRegistry::parseBackendType(const std::string& str) {
    if (str == "grim_text_server")  return BackendType::GrimTextServer;
    if (str == "llama_cpp")         return BackendType::LlamaCpp;
    if (str == "ollama")            return BackendType::Ollama;
    if (str == "external")          return BackendType::External;
    if (str == "in_process_vision") return BackendType::InProcessVision;

    throw std::runtime_error(
        "ModelRegistry: unknown backend_type '" + str
        + "' (valid: grim_text_server, llama_cpp, ollama, external, in_process_vision)");
}

std::string ModelRegistry::backendTypeToString(BackendType bt) {
    switch (bt) {
        case BackendType::GrimTextServer:  return "grim_text_server";
        case BackendType::LlamaCpp:        return "llama_cpp";
        case BackendType::Ollama:          return "ollama";
        case BackendType::External:        return "external";
        case BackendType::InProcessVision: return "in_process_vision";
    }
    throw std::runtime_error(
        "ModelRegistry::backendTypeToString: unknown BackendType value "
        + std::to_string(static_cast<int>(bt)));
}

ModelKind ModelRegistry::parseModelKind(const std::string& str) {
    if (str == "text")   return ModelKind::Text;
    if (str == "vision") return ModelKind::Vision;
    throw std::runtime_error(
        "ModelRegistry: unknown model kind '" + str
        + "' (valid: text, vision)");
}

std::string ModelRegistry::modelKindToString(ModelKind k) {
    switch (k) {
        case ModelKind::Text:   return "text";
        case ModelKind::Vision: return "vision";
    }
    throw std::runtime_error(
        "ModelRegistry::modelKindToString: unknown ModelKind "
        + std::to_string(static_cast<int>(k)));
}

VisionOperatorKind ModelRegistry::parseVisionOperatorKind(const std::string& str) {
    if (str == "object_detector")    return VisionOperatorKind::ObjectDetector;
    if (str == "semantic_segmenter") return VisionOperatorKind::SemanticSegmenter;
    if (str == "image_classifier")   return VisionOperatorKind::ImageClassifier;
    if (str == "pose_estimator")     return VisionOperatorKind::PoseEstimator;
    if (str == "scene_text_reader")  return VisionOperatorKind::SceneTextReader;
    if (str == "facial_expression_detector") return VisionOperatorKind::FacialExpressionDetector;
    if (str == "monocular_depth_estimator")  return VisionOperatorKind::MonocularDepthEstimator;
    if (str == "instance_segmenter")         return VisionOperatorKind::InstanceSegmenter;
    throw std::runtime_error(
        "ModelRegistry: unknown vision operator '" + str
        + "' (valid: object_detector, semantic_segmenter, image_classifier, "
          "pose_estimator, scene_text_reader, facial_expression_detector, "
          "monocular_depth_estimator, instance_segmenter)");
}

std::string ModelRegistry::visionOperatorKindToString(VisionOperatorKind k) {
    switch (k) {
        case VisionOperatorKind::Unknown:           return "unknown";
        case VisionOperatorKind::ObjectDetector:    return "object_detector";
        case VisionOperatorKind::SemanticSegmenter: return "semantic_segmenter";
        case VisionOperatorKind::ImageClassifier:   return "image_classifier";
        case VisionOperatorKind::PoseEstimator:     return "pose_estimator";
        case VisionOperatorKind::SceneTextReader:   return "scene_text_reader";
        case VisionOperatorKind::FacialExpressionDetector: return "facial_expression_detector";
        case VisionOperatorKind::MonocularDepthEstimator:  return "monocular_depth_estimator";
        case VisionOperatorKind::InstanceSegmenter:        return "instance_segmenter";
    }
    throw std::runtime_error(
        "ModelRegistry::visionOperatorKindToString: unknown VisionOperatorKind "
        + std::to_string(static_cast<int>(k)));
}

void ModelRegistry::validateModel(const ModelInfo& model, bool is_router) {
    // Required fields
    if (model.id.empty()) {
        throw std::runtime_error(
            "ModelRegistry: model is missing required 'id' field");
    }
    if (model.name.empty()) {
        throw std::runtime_error(
            "ModelRegistry: model '" + model.id + "' is missing required 'name' field");
    }

    // Kind / backend coherence
    const bool is_vision = (model.kind == ModelKind::Vision);
    if (is_router && is_vision) {
        throw std::runtime_error(
            "ModelRegistry: router '" + model.id
            + "' must have kind=text (the router is the reasoning engine that "
              "orchestrates sub-models, not a vision operator).");
    }

    if (is_vision) {
        // In-process vision sub-models do NOT have an HTTP url; instead
        // they require an ONNX path (in model_path) and a vision descriptor.
        if (model.backend_type != BackendType::InProcessVision) {
            throw std::runtime_error(
                "ModelRegistry: vision sub-model '" + model.id
                + "' must declare backend_type='in_process_vision'.");
        }
        if (model.vision.operator_kind == VisionOperatorKind::Unknown) {
            throw std::runtime_error(
                "ModelRegistry: vision sub-model '" + model.id
                + "' is missing required vision.operator (one of: object_detector, "
                  "semantic_segmenter, image_classifier, pose_estimator, scene_text_reader, "
                  "facial_expression_detector, monocular_depth_estimator, instance_segmenter).");
        }
        if (model.model_path.empty()) {
            throw std::runtime_error(
                "ModelRegistry: vision sub-model '" + model.id
                + "' is missing required model_path (ONNX weights).");
        }
    } else {
        // Text models (router and text sub-models) require a URL.
        if (model.url.empty()) {
            throw std::runtime_error(
                "ModelRegistry: model '" + model.id + "' is missing required 'url' field");
        }
    }

    // Sub-model invariant: no LoRA, no hard copy
    if (!is_router) {
        if (!model.lora_path.empty()) {
            throw std::runtime_error(
                "ModelRegistry: sub-model '" + model.id
                + "' has lora_path set — only the router may have LoRA. "
                  "Sub-models are frozen information bricks.");
        }
        if (!model.hard_copy_path.empty()) {
            throw std::runtime_error(
                "ModelRegistry: sub-model '" + model.id
                + "' has hard_copy_path set — only the router may have a hard copy. "
                  "Sub-models are frozen information bricks.");
        }
    }
}

} // namespace GRIM::MMO
