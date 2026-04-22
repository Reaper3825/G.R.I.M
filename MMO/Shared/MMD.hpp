// Multi-Model Orchestration (MMO) - Model Metadata Contracts
// Shared types for model descriptors, transport envelopes,
// and sub-model output contracts.
//======================================================//
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace GRIM::MMO {

// =========================================================
// Backend type — how the body connects to this model
// =========================================================
enum class BackendType : uint8_t {
    GrimTextServer   = 0,   // grim_text_server.exe (HTTP, one process per model)
    LlamaCpp         = 1,   // llama.cpp server
    Ollama           = 2,   // Ollama API
    External         = 3,   // arbitrary HTTP endpoint
    InProcessVision  = 4    // OpenCV cv::dnn vision operator hosted by
                            // PhysicalPerceptionPrimitivesLoop (no HTTP)
};

// =========================================================
// Model kind — what KIND of capability this model provides.
// Used by Orchestrator/Router to filter the candidate pool:
// text-generation routing must not consider Vision sub-models,
// and vision dispatch must not consider Text sub-models.
// =========================================================
enum class ModelKind : uint8_t {
    Text   = 0,   // produces tokens (router and text sub-models)
    Vision = 1    // produces a PhysicalPerceptionPrimitiveResult slice
};

// =========================================================
// Vision sub-model descriptor — populated only when
// ModelInfo::kind == Vision. Tells the bootstrap layer which
// PhysicalPerceptionPrimitivesLoop operator to configure with
// this model's weights.
// =========================================================
enum class VisionOperatorKind : uint8_t {
    Unknown                  = 0,
    ObjectDetector           = 1,
    SemanticSegmenter        = 2,
    ImageClassifier          = 3,
    PoseEstimator            = 4,
    SceneTextReader          = 5,
    FacialExpressionDetector = 6
};

struct VisionModelDescriptor {
    VisionOperatorKind operator_kind = VisionOperatorKind::Unknown;

    // Common preprocessing
    std::string class_names_path;
    int   input_width   = 0;
    int   input_height  = 0;

    // Object detector + classifier + pose-specific
    float confidence_threshold = 0.0f;   // detector / pose person-score
    float iou_threshold        = 0.0f;   // detector / pose NMS
    int   top_k                = 0;      // classifier
    float min_keypoint_confidence = 0.0f; // pose

    // Pose-specific. "" or "heatmap" → top-down heatmap model (HRNet/MoveNet style).
    // "yolo_anchor" → YOLOv8-pose style 3D output [1, 4+1+J*3, A].
    std::string pose_output_format;
    int         num_keypoints = 0;        // required when pose_output_format=="yolo_anchor"

    // Scene-text reader uses TWO ONNX files. ModelInfo::model_path
    // holds the detector path; recogniser is optional.
    std::string recogniser_onnx_path;
    std::string recogniser_charset_path;
    // Most CRNN-style recognisers (incl. OpenCV Zoo CRNN_EN/CN) are 1-channel
    // grayscale. CRNN_VGG_BiLSTM_CTC variants are 3-channel BGR. Default true
    // matches the OpenCV Zoo distribution.
    bool        recogniser_input_grayscale = true;

    // Facial-expression detector uses TWO ONNX files. ModelInfo::model_path
    // holds the YuNet face detector path; the per-face emotion classifier
    // (FER+ style) is optional and configured separately. classifier_class_names_path
    // is the labels file (one emotion per line, in softmax-output order).
    std::string expression_classifier_onnx_path;
    std::string expression_classifier_class_names_path;
    int         expression_classifier_input_width  = 64;
    int         expression_classifier_input_height = 64;
    bool        expression_classifier_input_grayscale = true;
};

// =========================================================
// ModelInfo — describes one model (router or sub-model)
//
// Router-only fields: lora_path, hard_copy_path
// Sub-models MUST leave those empty; ModelRegistry validates.
// =========================================================
struct ModelInfo {
    std::string id;
    std::string name;
    std::string version;
    std::string subject;
    std::string description;
    std::string model_path;         // weights file or directory
    BackendType backend_type   = BackendType::GrimTextServer;
    std::string url;                // host:port or full URL
    std::vector<std::string> subject_tags;
    float       usage_weight   = 0.0f;

    // Router-only — personalization bridge
    std::string lora_path;
    std::string hard_copy_path;

    // Estimated resource footprint (used by ResourceCoordinator)
    long        estimated_ram_mb  = 0;
    long        estimated_vram_mb = 0;

    // Capability classification. Defaults to Text so existing configs
    // remain valid; vision sub-models must declare kind=Vision and
    // populate `vision`.
    ModelKind             kind = ModelKind::Text;
    VisionModelDescriptor vision{};
};

// =========================================================
// Transport envelope — body → model (route or synthesize step)
// Every request carries these fields.
// =========================================================
struct RequestEnvelope {
    uint32_t    schema_version = 1;
    std::string request_id;
    std::string session_id;
    std::string turn_id;
    std::string target_model_id;
    std::string task;
    std::string scope;
    std::string constraints;
    std::string output_schema;
    int         max_length     = 0;
    std::string payload;
};

// =========================================================
// Sub-model output envelope — model → body
// =========================================================
enum class ResponseStatus : uint8_t {
    Ok     = 0,
    Refuse = 1,
    Error  = 2
};

struct ResponseEnvelope {
    uint32_t       schema_version = 1;
    std::string    request_id;
    std::string    target_model_id;
    ResponseStatus status         = ResponseStatus::Error;
    std::string    result;         // when status == Ok
    std::string    refusal;        // when status == Refuse
    std::string    error;          // when status == Error
};

// =========================================================
// getSubjectTags — extract subject tags from raw user input
//
// Returns tags suitable for matching against ModelInfo::subject_tags,
// enabling subject-based routing to pick the right sub-model.
// =========================================================
std::vector<std::string> getSubjectTags(const std::string& raw_input);

} // namespace GRIM::MMO