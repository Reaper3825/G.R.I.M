#include "perception_runtime.hpp"
#include "physical/PhysicalPerceptionPrimitivesLoop.hpp"
#include "physical/PhysicalSpatialGroundingLoop.hpp"
#include "physical/PhysicalMonocularDepthEstimator.hpp"
#include <set>
#include "physical/PhysicalEnvironmentLoop.hpp"
#include "physical/PhysicalLocalizationLoop.hpp"
#include "physical/PhysicalWorldStateLoop.hpp"
#include "physical/PhysicalInteractionLoop.hpp"
namespace perceptionPlugin {
namespace P = GRIM::Perception::Physical;
void configureOperator(const std::string& name, const nlohmann::json& request) {
    if (!request.is_object()) throw std::invalid_argument("Operator config must be an object");
    if (name == "object_detector") {
        P::PhysicalObjectDetectorConfig config;
        const std::set<std::string> fields{"onnx_model_path","class_names_path","input_width","input_height","input_scale","swap_rb","confidence_threshold","iou_threshold","max_detections","dnn_backend_id","dnn_target_id"};
        for (const auto& [key, value] : request.items()) if (!fields.count(key)) throw std::invalid_argument("Unsupported object_detector config field: " + key);
        if (request.contains("onnx_model_path")) request.at("onnx_model_path").get_to(config.onnx_model_path);
        if (request.contains("class_names_path")) request.at("class_names_path").get_to(config.class_names_path);
        if (request.contains("input_width")) request.at("input_width").get_to(config.input_width);
        if (request.contains("input_height")) request.at("input_height").get_to(config.input_height);
        if (request.contains("input_scale")) request.at("input_scale").get_to(config.input_scale);
        if (request.contains("swap_rb")) request.at("swap_rb").get_to(config.swap_rb);
        if (request.contains("confidence_threshold")) request.at("confidence_threshold").get_to(config.confidence_threshold);
        if (request.contains("iou_threshold")) request.at("iou_threshold").get_to(config.iou_threshold);
        if (request.contains("max_detections")) request.at("max_detections").get_to(config.max_detections);
        if (request.contains("dnn_backend_id")) request.at("dnn_backend_id").get_to(config.dnn_backend_id);
        if (request.contains("dnn_target_id")) request.at("dnn_target_id").get_to(config.dnn_target_id);
        P::RequestConfigurePhysicalObjectDetector(config);
    }
    else if (name == "semantic_segmenter") {
        P::PhysicalSemanticSegmenterConfig config;
        const std::set<std::string> fields{"onnx_model_path","class_names_path","input_width","input_height","input_scale","swap_rb","dnn_backend_id","dnn_target_id"};
        for (const auto& [key, value] : request.items()) if (!fields.count(key)) throw std::invalid_argument("Unsupported semantic_segmenter config field: " + key);
        if (request.contains("onnx_model_path")) request.at("onnx_model_path").get_to(config.onnx_model_path);
        if (request.contains("class_names_path")) request.at("class_names_path").get_to(config.class_names_path);
        if (request.contains("input_width")) request.at("input_width").get_to(config.input_width);
        if (request.contains("input_height")) request.at("input_height").get_to(config.input_height);
        if (request.contains("input_scale")) request.at("input_scale").get_to(config.input_scale);
        if (request.contains("swap_rb")) request.at("swap_rb").get_to(config.swap_rb);
        if (request.contains("dnn_backend_id")) request.at("dnn_backend_id").get_to(config.dnn_backend_id);
        if (request.contains("dnn_target_id")) request.at("dnn_target_id").get_to(config.dnn_target_id);
        P::RequestConfigurePhysicalSemanticSegmenter(config);
    }
    else if (name == "image_classifier") {
        P::PhysicalImageClassifierConfig config;
        const std::set<std::string> fields{"onnx_model_path","class_names_path","text_embeddings_path","input_width","input_height","input_scale","swap_rb","top_k","temperature"};
        for (const auto& [key, value] : request.items()) if (!fields.count(key)) throw std::invalid_argument("Unsupported image_classifier config field: " + key);
        if (request.contains("onnx_model_path")) request.at("onnx_model_path").get_to(config.onnx_model_path);
        if (request.contains("class_names_path")) request.at("class_names_path").get_to(config.class_names_path);
        if (request.contains("text_embeddings_path")) request.at("text_embeddings_path").get_to(config.text_embeddings_path);
        if (request.contains("input_width")) request.at("input_width").get_to(config.input_width);
        if (request.contains("input_height")) request.at("input_height").get_to(config.input_height);
        if (request.contains("input_scale")) request.at("input_scale").get_to(config.input_scale);
        if (request.contains("swap_rb")) request.at("swap_rb").get_to(config.swap_rb);
        if (request.contains("top_k")) request.at("top_k").get_to(config.top_k);
        if (request.contains("temperature")) request.at("temperature").get_to(config.temperature);
        P::RequestConfigurePhysicalImageClassifier(config);
    }
    else if (name == "pose_estimator") {
        P::PhysicalPoseKeypointEstimatorConfig config;
        const std::set<std::string> fields{"onnx_model_path","joint_names_path","input_width","input_height","input_scale","swap_rb","min_keypoint_confidence","num_keypoints","person_confidence_threshold","nms_iou_threshold","dnn_backend_id","dnn_target_id"};
        for (const auto& [key, value] : request.items()) if (!fields.count(key)) throw std::invalid_argument("Unsupported pose_estimator config field: " + key);
        if (request.contains("onnx_model_path")) request.at("onnx_model_path").get_to(config.onnx_model_path);
        if (request.contains("joint_names_path")) request.at("joint_names_path").get_to(config.joint_names_path);
        if (request.contains("input_width")) request.at("input_width").get_to(config.input_width);
        if (request.contains("input_height")) request.at("input_height").get_to(config.input_height);
        if (request.contains("input_scale")) request.at("input_scale").get_to(config.input_scale);
        if (request.contains("swap_rb")) request.at("swap_rb").get_to(config.swap_rb);
        if (request.contains("min_keypoint_confidence")) request.at("min_keypoint_confidence").get_to(config.min_keypoint_confidence);
        if (request.contains("num_keypoints")) request.at("num_keypoints").get_to(config.num_keypoints);
        if (request.contains("person_confidence_threshold")) request.at("person_confidence_threshold").get_to(config.person_confidence_threshold);
        if (request.contains("nms_iou_threshold")) request.at("nms_iou_threshold").get_to(config.nms_iou_threshold);
        if (request.contains("dnn_backend_id")) request.at("dnn_backend_id").get_to(config.dnn_backend_id);
        if (request.contains("dnn_target_id")) request.at("dnn_target_id").get_to(config.dnn_target_id);
        P::RequestConfigurePhysicalPoseKeypointEstimator(config);
    }
    else if (name == "scene_text_reader") {
        P::PhysicalSceneTextReaderConfig config;
        const std::set<std::string> fields{"detector_onnx_path","detector_input_width","detector_input_height","detector_input_scale","detector_binary_thresh","detector_polygon_thresh","detector_max_candidates","detector_unclip_ratio","recogniser_onnx_path","recogniser_charset_path","recogniser_input_width","recogniser_input_height","recogniser_input_scale","recogniser_input_grayscale","dnn_backend_id","dnn_target_id"};
        for (const auto& [key, value] : request.items()) if (!fields.count(key)) throw std::invalid_argument("Unsupported scene_text_reader config field: " + key);
        if (request.contains("detector_onnx_path")) request.at("detector_onnx_path").get_to(config.detector_onnx_path);
        if (request.contains("detector_input_width")) request.at("detector_input_width").get_to(config.detector_input_width);
        if (request.contains("detector_input_height")) request.at("detector_input_height").get_to(config.detector_input_height);
        if (request.contains("detector_input_scale")) request.at("detector_input_scale").get_to(config.detector_input_scale);
        if (request.contains("detector_binary_thresh")) request.at("detector_binary_thresh").get_to(config.detector_binary_thresh);
        if (request.contains("detector_polygon_thresh")) request.at("detector_polygon_thresh").get_to(config.detector_polygon_thresh);
        if (request.contains("detector_max_candidates")) request.at("detector_max_candidates").get_to(config.detector_max_candidates);
        if (request.contains("detector_unclip_ratio")) request.at("detector_unclip_ratio").get_to(config.detector_unclip_ratio);
        if (request.contains("recogniser_onnx_path")) request.at("recogniser_onnx_path").get_to(config.recogniser_onnx_path);
        if (request.contains("recogniser_charset_path")) request.at("recogniser_charset_path").get_to(config.recogniser_charset_path);
        if (request.contains("recogniser_input_width")) request.at("recogniser_input_width").get_to(config.recogniser_input_width);
        if (request.contains("recogniser_input_height")) request.at("recogniser_input_height").get_to(config.recogniser_input_height);
        if (request.contains("recogniser_input_scale")) request.at("recogniser_input_scale").get_to(config.recogniser_input_scale);
        if (request.contains("recogniser_input_grayscale")) request.at("recogniser_input_grayscale").get_to(config.recogniser_input_grayscale);
        if (request.contains("dnn_backend_id")) request.at("dnn_backend_id").get_to(config.dnn_backend_id);
        if (request.contains("dnn_target_id")) request.at("dnn_target_id").get_to(config.dnn_target_id);
        P::RequestConfigurePhysicalSceneTextReader(config);
    }
    else if (name == "facial_expression") {
        P::PhysicalFacialExpressionDetectorConfig config;
        const std::set<std::string> fields{"detector_onnx_path","detector_input_width","detector_input_height","detector_score_threshold","detector_nms_threshold","detector_top_k","classifier_onnx_path","classifier_class_names_path","classifier_input_width","classifier_input_height","classifier_input_scale","classifier_input_grayscale","classifier_swap_rb","face_crop_padding_ratio","dnn_backend_id","dnn_target_id"};
        for (const auto& [key, value] : request.items()) if (!fields.count(key)) throw std::invalid_argument("Unsupported facial_expression config field: " + key);
        if (request.contains("detector_onnx_path")) request.at("detector_onnx_path").get_to(config.detector_onnx_path);
        if (request.contains("detector_input_width")) request.at("detector_input_width").get_to(config.detector_input_width);
        if (request.contains("detector_input_height")) request.at("detector_input_height").get_to(config.detector_input_height);
        if (request.contains("detector_score_threshold")) request.at("detector_score_threshold").get_to(config.detector_score_threshold);
        if (request.contains("detector_nms_threshold")) request.at("detector_nms_threshold").get_to(config.detector_nms_threshold);
        if (request.contains("detector_top_k")) request.at("detector_top_k").get_to(config.detector_top_k);
        if (request.contains("classifier_onnx_path")) request.at("classifier_onnx_path").get_to(config.classifier_onnx_path);
        if (request.contains("classifier_class_names_path")) request.at("classifier_class_names_path").get_to(config.classifier_class_names_path);
        if (request.contains("classifier_input_width")) request.at("classifier_input_width").get_to(config.classifier_input_width);
        if (request.contains("classifier_input_height")) request.at("classifier_input_height").get_to(config.classifier_input_height);
        if (request.contains("classifier_input_scale")) request.at("classifier_input_scale").get_to(config.classifier_input_scale);
        if (request.contains("classifier_input_grayscale")) request.at("classifier_input_grayscale").get_to(config.classifier_input_grayscale);
        if (request.contains("classifier_swap_rb")) request.at("classifier_swap_rb").get_to(config.classifier_swap_rb);
        if (request.contains("face_crop_padding_ratio")) request.at("face_crop_padding_ratio").get_to(config.face_crop_padding_ratio);
        if (request.contains("dnn_backend_id")) request.at("dnn_backend_id").get_to(config.dnn_backend_id);
        if (request.contains("dnn_target_id")) request.at("dnn_target_id").get_to(config.dnn_target_id);
        P::RequestConfigurePhysicalFacialExpressionDetector(config);
    }
    else if (name == "face_recognizer") {
        P::PhysicalFaceRecognizerConfig config;
        const std::set<std::string> fields{"onnx_model_path","model_id","dnn_backend_id","dnn_target_id","minimum_face_pixels","minimum_detection_confidence"};
        for (const auto& [key, value] : request.items()) if (!fields.count(key)) throw std::invalid_argument("Unsupported face_recognizer config field: " + key);
        if (request.contains("onnx_model_path")) request.at("onnx_model_path").get_to(config.onnx_model_path);
        if (request.contains("model_id")) request.at("model_id").get_to(config.model_id);
        if (request.contains("dnn_backend_id")) request.at("dnn_backend_id").get_to(config.dnn_backend_id);
        if (request.contains("dnn_target_id")) request.at("dnn_target_id").get_to(config.dnn_target_id);
        if (request.contains("minimum_face_pixels")) request.at("minimum_face_pixels").get_to(config.minimum_face_pixels);
        if (request.contains("minimum_detection_confidence")) request.at("minimum_detection_confidence").get_to(config.minimum_detection_confidence);
        P::RequestConfigurePhysicalFaceRecognizer(config);
    }
    else if (name == "entity_tracker") {
        P::PhysicalEntityTrackerConfig config;
        const std::set<std::string> fields{"min_iou_for_match","smoothing_alpha","min_hits_to_confirm","max_tentative_misses","max_age_misses","max_predict_seconds","min_velocity_dt_seconds","detection_confidence_floor","cross_track_nms_iou"};
        for (const auto& [key, value] : request.items()) if (!fields.count(key)) throw std::invalid_argument("Unsupported entity_tracker config field: " + key);
        if (request.contains("min_iou_for_match")) request.at("min_iou_for_match").get_to(config.min_iou_for_match);
        if (request.contains("smoothing_alpha")) request.at("smoothing_alpha").get_to(config.smoothing_alpha);
        if (request.contains("min_hits_to_confirm")) request.at("min_hits_to_confirm").get_to(config.min_hits_to_confirm);
        if (request.contains("max_tentative_misses")) request.at("max_tentative_misses").get_to(config.max_tentative_misses);
        if (request.contains("max_age_misses")) request.at("max_age_misses").get_to(config.max_age_misses);
        if (request.contains("max_predict_seconds")) request.at("max_predict_seconds").get_to(config.max_predict_seconds);
        if (request.contains("min_velocity_dt_seconds")) request.at("min_velocity_dt_seconds").get_to(config.min_velocity_dt_seconds);
        if (request.contains("detection_confidence_floor")) request.at("detection_confidence_floor").get_to(config.detection_confidence_floor);
        if (request.contains("cross_track_nms_iou")) request.at("cross_track_nms_iou").get_to(config.cross_track_nms_iou);
        P::RequestConfigurePhysicalEntityTracker(config);
    }
    else if (name == "instance_segmenter") {
        P::PhysicalInstanceSegmenterConfig config;
        const std::set<std::string> fields{"encoder_onnx_path","decoder_onnx_path","encoder_input_width","encoder_input_height","encoder_input_scale","swap_rb","encoder_input_name","encoder_output_image_embed_name","encoder_output_high_res_feats_0_name","encoder_output_high_res_feats_1_name","decoder_input_image_embed_name","decoder_input_high_res_feats_0_name","decoder_input_high_res_feats_1_name","decoder_input_point_coords_name","decoder_input_point_labels_name","decoder_input_mask_input_name","decoder_input_has_mask_input_name","decoder_output_masks_name","decoder_output_iou_predictions_name","decoder_mask_resolution","max_prompts_per_frame","min_prompt_confidence","mask_logit_threshold","intra_op_num_threads"};
        for (const auto& [key, value] : request.items()) if (!fields.count(key)) throw std::invalid_argument("Unsupported instance_segmenter config field: " + key);
        if (request.contains("encoder_onnx_path")) request.at("encoder_onnx_path").get_to(config.encoder_onnx_path);
        if (request.contains("decoder_onnx_path")) request.at("decoder_onnx_path").get_to(config.decoder_onnx_path);
        if (request.contains("encoder_input_width")) request.at("encoder_input_width").get_to(config.encoder_input_width);
        if (request.contains("encoder_input_height")) request.at("encoder_input_height").get_to(config.encoder_input_height);
        if (request.contains("encoder_input_scale")) request.at("encoder_input_scale").get_to(config.encoder_input_scale);
        if (request.contains("swap_rb")) request.at("swap_rb").get_to(config.swap_rb);
        if (request.contains("encoder_input_name")) request.at("encoder_input_name").get_to(config.encoder_input_name);
        if (request.contains("encoder_output_image_embed_name")) request.at("encoder_output_image_embed_name").get_to(config.encoder_output_image_embed_name);
        if (request.contains("encoder_output_high_res_feats_0_name")) request.at("encoder_output_high_res_feats_0_name").get_to(config.encoder_output_high_res_feats_0_name);
        if (request.contains("encoder_output_high_res_feats_1_name")) request.at("encoder_output_high_res_feats_1_name").get_to(config.encoder_output_high_res_feats_1_name);
        if (request.contains("decoder_input_image_embed_name")) request.at("decoder_input_image_embed_name").get_to(config.decoder_input_image_embed_name);
        if (request.contains("decoder_input_high_res_feats_0_name")) request.at("decoder_input_high_res_feats_0_name").get_to(config.decoder_input_high_res_feats_0_name);
        if (request.contains("decoder_input_high_res_feats_1_name")) request.at("decoder_input_high_res_feats_1_name").get_to(config.decoder_input_high_res_feats_1_name);
        if (request.contains("decoder_input_point_coords_name")) request.at("decoder_input_point_coords_name").get_to(config.decoder_input_point_coords_name);
        if (request.contains("decoder_input_point_labels_name")) request.at("decoder_input_point_labels_name").get_to(config.decoder_input_point_labels_name);
        if (request.contains("decoder_input_mask_input_name")) request.at("decoder_input_mask_input_name").get_to(config.decoder_input_mask_input_name);
        if (request.contains("decoder_input_has_mask_input_name")) request.at("decoder_input_has_mask_input_name").get_to(config.decoder_input_has_mask_input_name);
        if (request.contains("decoder_output_masks_name")) request.at("decoder_output_masks_name").get_to(config.decoder_output_masks_name);
        if (request.contains("decoder_output_iou_predictions_name")) request.at("decoder_output_iou_predictions_name").get_to(config.decoder_output_iou_predictions_name);
        if (request.contains("decoder_mask_resolution")) request.at("decoder_mask_resolution").get_to(config.decoder_mask_resolution);
        if (request.contains("max_prompts_per_frame")) request.at("max_prompts_per_frame").get_to(config.max_prompts_per_frame);
        if (request.contains("min_prompt_confidence")) request.at("min_prompt_confidence").get_to(config.min_prompt_confidence);
        if (request.contains("mask_logit_threshold")) request.at("mask_logit_threshold").get_to(config.mask_logit_threshold);
        if (request.contains("intra_op_num_threads")) request.at("intra_op_num_threads").get_to(config.intra_op_num_threads);
        P::RequestConfigurePhysicalInstanceSegmenter(config);
    }
    else if (name == "depth") {
        P::PhysicalMonocularDepthEstimatorConfig config;
        const std::set<std::string> fields{"onnx_model_path","input_width","input_height","input_scale","swap_rb","output_is_disparity","metric_scale_meters","metric_epsilon","dnn_backend_id","dnn_target_id"};
        for (const auto& [key, value] : request.items()) if (!fields.count(key)) throw std::invalid_argument("Unsupported depth config field: " + key);
        if (request.contains("onnx_model_path")) request.at("onnx_model_path").get_to(config.onnx_model_path);
        if (request.contains("input_width")) request.at("input_width").get_to(config.input_width);
        if (request.contains("input_height")) request.at("input_height").get_to(config.input_height);
        if (request.contains("input_scale")) request.at("input_scale").get_to(config.input_scale);
        if (request.contains("swap_rb")) request.at("swap_rb").get_to(config.swap_rb);
        if (request.contains("output_is_disparity")) request.at("output_is_disparity").get_to(config.output_is_disparity);
        if (request.contains("metric_scale_meters")) request.at("metric_scale_meters").get_to(config.metric_scale_meters);
        if (request.contains("metric_epsilon")) request.at("metric_epsilon").get_to(config.metric_epsilon);
        if (request.contains("dnn_backend_id")) request.at("dnn_backend_id").get_to(config.dnn_backend_id);
        if (request.contains("dnn_target_id")) request.at("dnn_target_id").get_to(config.dnn_target_id);
        P::RequestConfigurePhysicalMonocularDepthEstimator(config);
    }
    else if (name == "hand_gestures") {
        P::PhysicalHandGestureConfig config;
        config.model_path = request.at("model_path").get<std::string>();
        config.enabled = request.value("enabled",true);
        config.max_hands = request.value("max_hands",2);
        config.target_fps = request.value("target_fps",30.0);
        P::RequestConfigurePhysicalHandGestures(config);
    }
    else if (name == "conditioning") {
        P::PhysicalSignalConditioningConfig config;
        const std::set<std::string> fields{"enable_resize","output_width","output_height","letterbox_pad_value","enable_denoise","denoise_strength","enable_deblur","deblur_amount","enable_stabilization","flow_max_corners","flow_quality_level","flow_min_distance"};
        for (const auto& [key, value] : request.items()) if (!fields.count(key)) throw std::invalid_argument("Unsupported conditioning config field: " + key);
        if (request.contains("enable_resize")) request.at("enable_resize").get_to(config.enable_resize);
        if (request.contains("output_width")) request.at("output_width").get_to(config.output_width);
        if (request.contains("output_height")) request.at("output_height").get_to(config.output_height);
        if (request.contains("letterbox_pad_value")) request.at("letterbox_pad_value").get_to(config.letterbox_pad_value);
        if (request.contains("enable_denoise")) request.at("enable_denoise").get_to(config.enable_denoise);
        if (request.contains("denoise_strength")) request.at("denoise_strength").get_to(config.denoise_strength);
        if (request.contains("enable_deblur")) request.at("enable_deblur").get_to(config.enable_deblur);
        if (request.contains("deblur_amount")) request.at("deblur_amount").get_to(config.deblur_amount);
        if (request.contains("enable_stabilization")) request.at("enable_stabilization").get_to(config.enable_stabilization);
        if (request.contains("flow_max_corners")) request.at("flow_max_corners").get_to(config.flow_max_corners);
        if (request.contains("flow_quality_level")) request.at("flow_quality_level").get_to(config.flow_quality_level);
        if (request.contains("flow_min_distance")) request.at("flow_min_distance").get_to(config.flow_min_distance);
        P::RequestConfigurePhysicalSignalConditioning(config);
    }
    else if (name == "spatial_grounder") {
        P::PhysicalSpatialGrounderConfig config;
        const std::set<std::string> fields{"min_range_confidence","support_strip_height_px","floor_below_delta","table_step","wall_flatness","wall_min_aspect","nav_cone_half_width_frac","nav_cone_min_y_frac","path_block_threshold","max_path_meters","moving_threshold_px_per_sec","moving_threshold_inv_per_sec","moved_quantum_px","moved_quantum_inv"};
        for (const auto& [key, value] : request.items()) if (!fields.count(key)) throw std::invalid_argument("Unsupported spatial_grounder config field: " + key);
        if (request.contains("min_range_confidence")) request.at("min_range_confidence").get_to(config.min_range_confidence);
        if (request.contains("support_strip_height_px")) request.at("support_strip_height_px").get_to(config.support_strip_height_px);
        if (request.contains("floor_below_delta")) request.at("floor_below_delta").get_to(config.floor_below_delta);
        if (request.contains("table_step")) request.at("table_step").get_to(config.table_step);
        if (request.contains("wall_flatness")) request.at("wall_flatness").get_to(config.wall_flatness);
        if (request.contains("wall_min_aspect")) request.at("wall_min_aspect").get_to(config.wall_min_aspect);
        if (request.contains("nav_cone_half_width_frac")) request.at("nav_cone_half_width_frac").get_to(config.nav_cone_half_width_frac);
        if (request.contains("nav_cone_min_y_frac")) request.at("nav_cone_min_y_frac").get_to(config.nav_cone_min_y_frac);
        if (request.contains("path_block_threshold")) request.at("path_block_threshold").get_to(config.path_block_threshold);
        if (request.contains("max_path_meters")) request.at("max_path_meters").get_to(config.max_path_meters);
        if (request.contains("moving_threshold_px_per_sec")) request.at("moving_threshold_px_per_sec").get_to(config.moving_threshold_px_per_sec);
        if (request.contains("moving_threshold_inv_per_sec")) request.at("moving_threshold_inv_per_sec").get_to(config.moving_threshold_inv_per_sec);
        if (request.contains("moved_quantum_px")) request.at("moved_quantum_px").get_to(config.moved_quantum_px);
        if (request.contains("moved_quantum_inv")) request.at("moved_quantum_inv").get_to(config.moved_quantum_inv);
        P::RequestConfigurePhysicalSpatialGrounder(config);
    }
    else if (name == "odometry") {
        P::PhysicalVisualOdometerConfig config;
        const std::set<std::string> fields{"orb_max_features","orb_scale_factor","orb_n_levels","ransac_inlier_threshold_pixels","ransac_confidence","ransac_max_iterations","min_matches_for_pose","min_median_parallax_pixels","min_inliers_for_pose","max_trajectory_samples","use_hamming_matcher"};
        for (const auto& [key, value] : request.items()) if (!fields.count(key)) throw std::invalid_argument("Unsupported odometry config field: " + key);
        if (request.contains("orb_max_features")) request.at("orb_max_features").get_to(config.orb_max_features);
        if (request.contains("orb_scale_factor")) request.at("orb_scale_factor").get_to(config.orb_scale_factor);
        if (request.contains("orb_n_levels")) request.at("orb_n_levels").get_to(config.orb_n_levels);
        if (request.contains("ransac_inlier_threshold_pixels")) request.at("ransac_inlier_threshold_pixels").get_to(config.ransac_inlier_threshold_pixels);
        if (request.contains("ransac_confidence")) request.at("ransac_confidence").get_to(config.ransac_confidence);
        if (request.contains("ransac_max_iterations")) request.at("ransac_max_iterations").get_to(config.ransac_max_iterations);
        if (request.contains("min_matches_for_pose")) request.at("min_matches_for_pose").get_to(config.min_matches_for_pose);
        if (request.contains("min_median_parallax_pixels")) request.at("min_median_parallax_pixels").get_to(config.min_median_parallax_pixels);
        if (request.contains("min_inliers_for_pose")) request.at("min_inliers_for_pose").get_to(config.min_inliers_for_pose);
        if (request.contains("max_trajectory_samples")) request.at("max_trajectory_samples").get_to(config.max_trajectory_samples);
        if (request.contains("use_hamming_matcher")) request.at("use_hamming_matcher").get_to(config.use_hamming_matcher);
        P::RequestConfigurePhysicalLocalizationOdometer(config);
    }
    else if (name == "occupancy_grid") {
        P::PhysicalOccupancyGridMapperConfig config;
        const std::set<std::string> fields{"resolution_meters","initial_cols","initial_rows","free_log_odds_increment","max_abs_log_odds"};
        for (const auto& [key, value] : request.items()) if (!fields.count(key)) throw std::invalid_argument("Unsupported occupancy_grid config field: " + key);
        if (request.contains("resolution_meters")) request.at("resolution_meters").get_to(config.resolution_meters);
        if (request.contains("initial_cols")) request.at("initial_cols").get_to(config.initial_cols);
        if (request.contains("initial_rows")) request.at("initial_rows").get_to(config.initial_rows);
        if (request.contains("free_log_odds_increment")) request.at("free_log_odds_increment").get_to(config.free_log_odds_increment);
        if (request.contains("max_abs_log_odds")) request.at("max_abs_log_odds").get_to(config.max_abs_log_odds);
        P::RequestConfigurePhysicalLocalizationGrid(config);
    }
    else if (name == "world_state") {
        P::PhysicalWorldStateBuilderConfig config;
        const std::set<std::string> fields{"max_relations_per_entity","min_overlap_for_occlusion","depth_difference_for_nearer","position_axis_dominance"};
        for (const auto& [key, value] : request.items()) if (!fields.count(key)) throw std::invalid_argument("Unsupported world_state config field: " + key);
        if (request.contains("max_relations_per_entity")) request.at("max_relations_per_entity").get_to(config.max_relations_per_entity);
        if (request.contains("min_overlap_for_occlusion")) request.at("min_overlap_for_occlusion").get_to(config.min_overlap_for_occlusion);
        if (request.contains("depth_difference_for_nearer")) request.at("depth_difference_for_nearer").get_to(config.depth_difference_for_nearer);
        if (request.contains("position_axis_dominance")) request.at("position_axis_dominance").get_to(config.position_axis_dominance);
        P::RequestConfigurePhysicalWorldStateBuilder(config);
    }
    else throw std::invalid_argument("Unknown perception operator: " + name);
}
}
