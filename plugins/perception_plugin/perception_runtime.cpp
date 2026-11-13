#include "perception_runtime.hpp"
#include "digital/DigitalEnvironmentLoop.hpp"
#include "digital/DigitalPerceptionPrimitivesLoop.hpp"
#include "digital/DigitalPerceptionPrimitiveBus.hpp"
#include "physical/PhysicalEnvironmentLoop.hpp"
#include "physical/PhysicalCameraCalibrator.hpp"
#include "physical/PhysicalFrameBus.hpp"
#include "physical/PhysicalPerceptionPrimitivesLoop.hpp"
#include "physical/PhysicalInteractionLoop.hpp"
#include "physical/PhysicalHandGestureBus.hpp"
#include "physical/PhysicalSpatialGroundingLoop.hpp"
#include "physical/PhysicalLocalizationLoop.hpp"
#include "physical/PhysicalWorldStateLoop.hpp"
#include "physical/PhysicalWorldStateBus.hpp"
#include "physical/PhysicalPerceptionPrimitiveBus.hpp"
#include "plugins/plugin_logging.hpp"
#include <stdexcept>

namespace perceptionPlugin {
namespace D = GRIM::Perception::Digital;
namespace P = GRIM::Perception::Physical;
using Json = nlohmann::json;
namespace {
Json rect(const D::DigitalRect& r) { return {{"x",r.x},{"y",r.y},{"width",r.width},{"height",r.height}}; }
Json digitalSnapshot() {
    D::DigitalPerceptionPrimitiveBus::SnapshotView view;
    uint64_t cursor = 0;
    if (!D::DigitalPerceptionPrimitiveBus::Instance().PullLatest(view, cursor))
        return {{"available",false}};
    const auto& s = *view.snapshot;
    Json regions = Json::array(), elements = Json::array();
    for (const auto& r : s.ocr.regions) regions.push_back({{"text",r.text},{"confidence",r.confidence},{"frame_rect",rect(r.frame_rect)}});
    for (const auto& e : s.automation.elements) elements.push_back({{"name",e.name},{"automation_id",e.automation_id},{"role",e.role},
        {"desktop_rect",rect(e.desktop_rect)},{"enabled",e.enabled},{"offscreen",e.offscreen},{"password",e.password}});
    return {{"available",true},{"source_frame_counter",s.source_frame_counter},{"capture_wall_ns",s.source_capture_wall_ns},
        {"analyzed_steady_ns",s.analyzed_steady_ns},{"capture_status",D::ToString(s.source_metadata.status)},
        {"source_device_id",s.source_metadata.source_device_id},{"source_platform",s.source_metadata.source_platform},
        {"source_transport",s.source_metadata.source_transport},{"source_rect",rect(s.source_metadata.source_rect)},
        {"active_window",s.source_metadata.active_window_title},{"process",s.source_metadata.active_process_name},
        {"ocr",{{"status",D::ToString(s.ocr.status)},{"error",s.ocr.error},{"text",s.ocr.full_text},
            {"confidence",s.ocr.mean_confidence},{"regions",regions}}},
        {"automation",{{"status",D::ToString(s.automation.status)},{"error",s.automation.error},
            {"target_matches_capture",s.automation.target_matches_capture},{"target_changed_since_capture",s.automation.target_changed_since_capture},
            {"truncated",s.automation.truncated},{"elements",elements}}}};
}
Json primitiveSnapshot() {
    P::PhysicalPerceptionPrimitiveBus::ResultsView view;
    uint64_t cursor = 0;
    if (!P::PhysicalPerceptionPrimitiveBus::Instance().PullLatestPhysicalPerceptionResultsView(view,cursor))
        return {{"available",false}};
    const auto& r = view.results;
    Json states = Json::object();
    const auto status = [](const auto& value) { return Json{{"state",static_cast<int>(value.state)},{"error",value.last_error_reason}}; };
    states["object_detector"] = status(r.object_detector);
    states["semantic_segmenter"] = status(r.semantic_segmenter);
    states["instance_segmenter"] = status(r.instance_segmenter);
    states["image_classifier"] = status(r.image_classifier);
    states["pose_estimator"] = status(r.pose_estimator);
    states["scene_text_reader"] = status(r.scene_text_reader);
    states["facial_expression_detector"] = status(r.facial_expression_detector);
    states["face_recognizer"] = status(r.face_recognizer);
    states["entity_tracker"] = status(r.entity_tracker);
    states["class_policy"] = status(r.class_policy);
    Json detections = Json::array(), text = Json::array();
    for (const auto& d : r.object_detector.detections) detections.push_back({{"class",d.class_label},{"confidence",d.confidence},
        {"raw_box",{{"x",d.raw_box.x},{"y",d.raw_box.y},{"width",d.raw_box.width},{"height",d.raw_box.height}}}});
    for (const auto& line : r.scene_text_reader.lines) text.push_back({{"text",line.text},{"confidence",line.confidence}});
    return {{"available",true},{"source_frame_counter",r.source_frame_counter},{"operators",states},{"detections",detections},{"text",text}};
}
Json gestureSnapshot() {
    P::PhysicalHandGestureBus::SnapshotView view; uint64_t cursor = 0;
    if (!P::PhysicalHandGestureBus::Instance().PullLatestPhysicalHandGestureSnapshot(view,cursor)) return {{"available",false}};
    Json hands = Json::array();
    for (const auto& hand : view.snapshot.hands) {
        Json landmarks = Json::array();
        for (size_t i=0;i<hand.landmark_count && i<hand.landmarks.size();++i) {
            const auto& l=hand.landmarks[i];
            landmarks.push_back({{"x",l.normalized_x},{"y",l.normalized_y},{"z",l.normalized_z}});
        }
        hands.push_back({{"gesture",hand.gesture_label},{"confidence",hand.gesture_confidence},{"handedness",static_cast<int>(hand.handedness)},{"landmarks",landmarks}});
    }
    return {{"available",true},{"source_frame_counter",view.snapshot.source_frame_counter},
        {"backend_state",static_cast<int>(view.snapshot.backend_state)},{"error",view.snapshot.last_error},{"hands",hands}};
}
Json physicalSnapshot() {
    P::PhysicalWorldStateBus::SnapshotView view;
    uint64_t cursor = 0;
    if (!P::PhysicalWorldStateBus::Instance().PullLatestPhysicalWorldStateSnapshotView(view,cursor))
        return {{"available",false}};
    Json entities = Json::array();
    for (const auto& e : view.snapshot.entities) entities.push_back({{"object_id",e.object_id},{"class",e.class_label},
        {"text",e.text_on_object},{"last_seen_frame",e.last_seen_frame_counter}});
    return {{"available",true},{"source_frame_counter",view.snapshot.source_frame_counter},
        {"built_at_steady_ns",view.snapshot.built_at_steady_ns},{"entities",entities}};
}
}
Runtime::Runtime(const GrimPluginAPI* host) : api(host) {
    // No device acquisition, model loading, or worker starts during staged init.
    publish();
    timer = api->create_timer(api, 33, true, tick, this);
    if (!timer) throw std::runtime_error("Unable to register perception tick");
}
Runtime::~Runtime() {
    if (timer) api->cancel_timer(api,timer);
    std::lock_guard lock(mutex);
    D::ShutdownDigitalPerceptionPrimitives();
    D::ShutdownDigitalEnvironment();
    stopPhysical();
}
void Runtime::stopPhysical() {
    physical = false;
    P::ShutdownPhysicalWorldState();
    P::ShutdownPhysicalLocalization();
    P::ShutdownPhysicalSpatialGrounding();
    P::ShutdownPhysicalPerceptionPrimitives();
    P::ShutdownPhysicalInteraction();
    P::PhysicalHandGestureBus::Instance().ResetPhysicalHandGestureBus();
    P::ShutdownPhysicalEnvironment();
    P::ResetPhysicalCalibrationState();
}
void Runtime::tick(uint64_t, void* context) {
    auto& self = *static_cast<Runtime*>(context);
    std::lock_guard lock(self.mutex);
    try { self.pump(); }
    catch (const std::exception& e) { self.lastError = e.what(); LOG_ERROR("PerceptionPlugin", self.lastError); }
    catch (...) { self.lastError = "Unknown perception tick failure"; }
}
void Runtime::pump() {
    if (++ticks % 30 == 0 && (physical || D::IsDigitalEnvironmentRunning())) publish();
    if (physical) {
        P::TickPhysicalEnvironment();
        if (!P::IsPhysicalCameraCalibrationCaptureActive() && P::PhysicalFrameBus::Instance().HasCalibratedModelFrame()) {
            P::TickPhysicalInteraction();
            P::TickPhysicalPerceptionPrimitives();
            P::TickPhysicalSpatialGrounding();
            P::TickPhysicalLocalization();
            P::TickPhysicalWorldState();
        }
    }
}
Json Runtime::snapshot() {
    const auto capture = D::GetDigitalEnvironmentStatusSnapshot();
    const auto primitives = D::GetDigitalPerceptionPrimitivesStatusSnapshot();
    return {{"digital_running",capture.running},{"physical_running",physical},{"last_error",lastError},
        {"capture_error",capture.last_error},{"primitive_error",primitives.last_error},
        {"capture_attempts",capture.capture_attempts},{"processed_frames",primitives.processed_frames},
        {"physical_error",P::GetLastEnvironmentError()},
        {"digital",digitalSnapshot()},{"primitives",primitiveSnapshot()},{"gestures",gestureSnapshot()},{"physical",physicalSnapshot()}};
}
void Runtime::publish() {
    const auto payload = Json::array({{{"resource_id","observations"},{"kind","perception.observations"},
        {"name","Perception observations"},{"runtime_id",1},{"state",snapshot()}}}).dump();
    if (api->replace_resources(api,"perception.observations",payload.data(),payload.size()) != GRIM_OK)
        throw std::runtime_error("Perception resource publication failed");
}
Json Runtime::request(const std::string& operation, const Json& args) {
    std::lock_guard lock(mutex);
    if (!args.is_object()) throw std::invalid_argument("Expected an object");
    if (operation == "perception.digital.start") {
        D::DigitalEnvironmentConfig config;
        const auto interval = args.value("interval_ms",1000);
        if (interval < 33 || interval > 60000) throw std::invalid_argument("interval_ms must be 33..60000");
        config.capture_interval = std::chrono::milliseconds(interval);
        const auto mode = args.value("mode",std::string("active-monitor"));
        if (mode == "active-monitor") config.request.mode = D::DigitalCaptureMode::ActiveMonitor;
        else if (mode == "active-window") config.request.mode = D::DigitalCaptureMode::ActiveWindow;
        else if (mode == "virtual-desktop") config.request.mode = D::DigitalCaptureMode::VirtualDesktop;
        else if (mode == "monitor") { config.request.mode = D::DigitalCaptureMode::Monitor; config.request.monitor_index = args.at("monitor_index").get<int>(); }
        else throw std::invalid_argument("Unknown capture mode");
        D::DigitalPerceptionPrimitivesConfig primitives;
        primitives.ocr_enabled = args.value("ocr",true);
        primitives.automation_enabled = args.value("automation",true);
        try { D::StartDigitalPerceptionPrimitives(primitives); D::StartDigitalEnvironment(config); }
        catch (...) { D::ShutdownDigitalPerceptionPrimitives(); D::ShutdownDigitalEnvironment(); throw; }
    } else if (operation == "perception.digital.stop") {
        D::ShutdownDigitalPerceptionPrimitives(); D::ShutdownDigitalEnvironment();
    } else if (operation == "perception.physical.open") {
        if (physical) stopPhysical();
        if (args.contains("left_url") || args.contains("right_url")) {
            P::PhysicalStereoCaptureConfig config;
            config.left_url = args.at("left_url").get<std::string>();
            config.right_url = args.at("right_url").get<std::string>();
            config.left_label = args.value("left_label",std::string("Left camera"));
            config.right_label = args.value("right_label",std::string("Right camera"));
            config.maximum_pair_skew_ms = args.value("maximum_pair_skew_ms",15.0);
            P::RequestOpenPhysicalStereoCameraPair(config);
        } else P::RequestOpenPhysicalCameraSource(args.at("url").get<std::string>(),args.value("label",std::string("Camera")));
        physical = true;
    } else if (operation == "perception.physical.close") stopPhysical();
    else if (operation == "perception.configure") configureOperator(args.at("operator").get<std::string>(),args.at("config"));
    else if (operation == "perception.calibrate") {
        if (!physical) throw std::runtime_error("Open a camera before calibration");
        const auto action = args.value("action",std::string("automatic"));
        if (action == "automatic") P::RequestStartAutomaticPhysicalCameraCalibration();
        else if (action == "start") P::RequestStartPhysicalCameraCalibrationCapture();
        else if (action == "stop") P::RequestStopPhysicalCameraCalibrationCapture();
        else if (action == "sample") P::RequestCapturePhysicalCalibrationSampleNow();
        else if (action == "solve") P::RequestRunIntrinsicCalibrationFromSamples();
        else if (action == "save") P::RequestSavePhysicalCalibrationToDisk();
        else if (action == "load") { if (!P::RequestLoadPhysicalCalibrationFromDisk()) throw std::runtime_error("No matching calibration profile"); }
        else if (action == "clear") P::RequestClearPhysicalCalibrationSamples();
        else if (action == "pattern") P::RequestReconfigurePhysicalCalibrationPattern(args.at("cols").get<int>(),args.at("rows").get<int>(),args.at("square_meters").get<float>());
        else throw std::invalid_argument("Unknown calibration action");
    } else if (operation != "perception.observe") throw std::invalid_argument("Unknown perception operation");
    publish();
    return snapshot();
}
}
