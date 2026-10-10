#include "ui/observatory/observatory_report.hpp"
#include "resources/models/GRIM-text/Shared/Lenses/LensInspectionReport.hpp"
#include <nlohmann/json.hpp>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <chrono>
#include <fstream>
using namespace GRIM::Observatory;
using nlohmann::json;
void check(bool condition) { if(!condition)throw std::runtime_error("Observatory report assertion failed"); }
template<class F> void rejects(F action) { bool rejected=false;try{action();}catch(const std::exception&){rejected=true;}check(rejected); }
template<class F> void rejectsWith(F action,const std::string& message) {
    try {action();} catch(const std::runtime_error& e) {check(std::string(e.what()).find(message)!=std::string::npos);return;}
    throw std::runtime_error("Expected contextual inspection error");
}
int main() {
    rejectsWith([]{parseInspectionResponse(404,"");},"Inspection endpoint is unavailable");
    rejectsWith([]{parseInspectionResponse(404,"<html>Not found</html>");},"HTTP 404");
    rejectsWith([]{parseInspectionResponse(503,R"({"error":"router_unavailable","message":"The MMO router model is not loaded"})");},"The MMO router model is not loaded");
    rejectsWith([]{parseInspectionResponse(400,R"({"error":"Selected model differs"})");},"Selected model differs");
    rejectsWith([]{parseInspectionResponse(500,"Server error");},"HTTP 500");
    rejectsWith([]{parseInspectionResponse(200,"");},"empty or invalid JSON");
    rejectsWith([]{parseInspectionResponse(200,"[]");},"capture object");
    GRIM::Config::CompiledModelConfigSnapshot config;
    config.architecture.max_seq_len=8192;config.architecture.num_layers=7;config.architecture.d_model=128;
    config.integrity.semantic_sha256[0]=0xab;
    json candidate={{"token_id",2},{"text","word"},{"probability",.4}};
    json layer={{"layer",6},{"entropy_nats",2.0},{"candidates",json::array({candidate})}};
    json source={{"schema_version",1},{"config_sha256",configDigest(config)},{"layer_count",7},{"d_model",128},
        {"vocab_size",100},{"mode","inference"},{"attention_visibility","causal"},{"checkpoint","checkpoint-42"},
        {"positions",json::array({{{"position",3},{"input_text","a"},{"layers",json::array({layer})}}})}};
    auto report=parseReport(source,config);
    check(parseInspectionResponse(200,source.dump())==source);
    check(report.layerCount==7 && report.positions[0].layers[0].layer==6);
    check(report.positions[0].layers[0].candidates[0].probability==.4); // no top-k renormalization
    auto changed=source;changed["mode"]="training_example";changed["attention_visibility"]="full_sequence";
    check(parseReport(changed,config).mode=="training_example");
    auto bad=[&](const char* field,json value){auto j=source;j[field]=std::move(value);rejects([&]{parseReport(j,config);});};
    bad("config_sha256","wrong");bad("layer_count",12);bad("d_model",256);bad("schema_version",2);bad("mode","train_live");
    bad("vocab_size",0);bad("layer_count",-1);bad("positions",json::array());
    auto mutateLayer=[&](const char* field,json value) {auto j=source;j["positions"][0]["layers"][0][field]=std::move(value);rejects([&]{parseReport(j,config);});};
    mutateLayer("layer",7);mutateLayer("entropy_nats",-1);mutateLayer("entropy_nats",10);
    mutateLayer("entropy_nats",std::numeric_limits<double>::quiet_NaN());
    mutateLayer("candidates",json::array({candidate,candidate}));
    auto invalid=candidate;invalid["probability"]=1.1;mutateLayer("candidates",json::array({invalid}));
    invalid=candidate;invalid["token_id"]=100;mutateLayer("candidates",json::array({invalid}));
    invalid=candidate;invalid["probability"]=-.1;mutateLayer("candidates",json::array({invalid}));
    invalid=candidate;invalid["token_id"]=3;invalid["probability"]=.8;mutateLayer("candidates",json::array({candidate,invalid}));
    changed=source;changed["positions"][0]["layers"].push_back(layer);rejects([&]{parseReport(changed,config);});
    changed=source;changed["positions"].push_back(changed["positions"][0]);rejects([&]{parseReport(changed,config);});
    changed=source;changed["positions"][0]["position"]=8191;
    check(parseReport(changed,config).positions[0].position==8191);
    changed["positions"][0]["position"]=8192;rejects([&]{parseReport(changed,config);});
    changed=source;auto actual=layer;actual["kind"]="actual_final";
    auto direct=layer;direct["kind"]="direct";
    changed["positions"][0]["layers"][0]={{"layer",6},{"readouts",json::array({actual,direct})},{"identity_max_abs_logit_error",0.0}};
    report=parseReport(changed,config);
    check(report.positions[0].layers[0].readouts.size()==2 && report.positions[0].layers[0].identityMaxAbsLogitError==0.0);
    selectReadout(report,"actual_final");check(report.positions[0].layers[0].candidates.size()==1);
    selectReadout(report,"identity_control");check(report.positions[0].layers[0].candidates.empty());
    changed["positions"][0]["layers"][0]["readouts"].push_back(direct);rejects([&]{parseReport(changed,config);});
    changed=source;changed["positions"]=json::array();
    for(unsigned i=0;i<5000;++i) {auto p=source["positions"][0];p["position"]=i;changed["positions"].push_back(std::move(p));}
    check(parseReport(changed,config).positions.size()==5000);
    for(unsigned count:{1u,7u,19u}) {
        config.architecture.num_layers=count;auto preview=makePreview(config);
        check(preview.synthetic && preview.layerCount==count && preview.positions[0].layers.size()==count);
        for(const auto& l:preview.positions[0].layers) {
            double sum=0;for(const auto& c:l.candidates)sum+=c.probability;
            check(std::abs(sum-1)<1e-9 && l.entropyNats>=0 && l.entropyNats<=std::log(6.));
        }
    }
    // Exercise the actual runtime transport adapter through the desktop parser.
    config.architecture.num_layers=2;
    GRIM::Lenses::LensCaptureResult capture;capture.chunk_rows=1;capture.planned_temporary_bytes=1024;
    for(int l=0;l<2;++l)for(int p=0;p<3;++p) {
        GRIM::Lenses::LensSnapshot snapshot;
        auto& m=snapshot.metadata;m.layer_count=2;m.d_model=128;m.vocab_size=100;
        m.layer_index=l;m.token_position=p;m.supervision.input_token_id=p;
        snapshot.readouts.push_back({GRIM::Lenses::ReadoutKind::Direct,{{2,1.25f,.4}},2.0});
        if(l==1) {
            snapshot.readouts.push_back({GRIM::Lenses::ReadoutKind::ActualFinal,{{2,1.25f,.4}},2.0});
            snapshot.identity_max_abs_logit_error=0.0;
        }
        capture.snapshots.push_back(std::move(snapshot));
    }
    const auto wire=GRIM::Lenses::inspectionReport(capture,config,"weights-42",[](int id){return "token"+std::to_string(id);});
    report=parseReport(json::parse(wire.dump()),config);
    check(report.positions.size()==3 && report.positions[2].layers.size()==2);
    check(report.positions[0].layers[1].identityMaxAbsLogitError==0.0);
    check(report.positions[0].layers[1].candidates[0].logit==1.25f);
    check(report.positions[0].layers[1].candidates[0].probability==.4);
    // Default-file lifecycle: missing placeholder, existing capture preservation,
    // replacement, and all readout variants surviving a disk round trip.
    const auto directory=std::filesystem::temp_directory_path()/
        ("grim-observatory-test-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(directory);
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() {std::error_code ignored;std::filesystem::remove_all(path,ignored);}
    } cleanup{directory};
    const auto capturePath=defaultCapturePath(directory/"model.grimcfg");
    check(capturePath==directory/"observatory_capture.json");
    ensureCaptureFile(capturePath);
    check(std::filesystem::exists(capturePath));
    rejects([&]{loadReport(capturePath,config);});
    saveCaptureFile(capturePath,wire);
    ensureCaptureFile(capturePath);
    auto saved=loadReport(capturePath,config);
    check(saved.checkpoint=="weights-42" && saved.positions.size()==3);
    check(saved.positions[0].layers[1].readouts.size()==2);
    check(saved.positions[0].layers[1].identityMaxAbsLogitError==0.0);
    auto replacement=wire;replacement["checkpoint"]="weights-43";
    saveCaptureFile(capturePath,replacement);
    check(loadReport(capturePath,config).checkpoint=="weights-43");
    check(!std::filesystem::exists(capturePath.string()+".tmp"));
    // An existing malformed file is preserved instead of silently replaced.
    {std::ofstream file(capturePath);file<<"invalid json";}
    ensureCaptureFile(capturePath);
    rejects([&]{loadReport(capturePath,config);});
    rejects([&]{saveCaptureFile(directory/"missing"/"capture.json",wire);});
    // Full Jacobian assembly and exact transport. Every seeded row arrives for
    // all source positions/layers; reject partial matrices rather than presenting
    // a single gradient row as a complete Jacobian.
    GRIM::Lenses::LensCaptureResult full;
    auto slice=capture;slice.jacobian_target_position=2;slice.jacobian_rows=1;
    for(int out=0;out<128;++out) {
        for(auto& snapshot:slice.snapshots) {
            snapshot.hidden_jacobian.resize(128);
            for(int in=0;in<128;++in)
                snapshot.hidden_jacobian[in]=snapshot.metadata.layer_index==1
                    ?float(snapshot.metadata.token_position==2 && out==in)
                    :float((out-in)*.001+snapshot.metadata.token_position);
        }
        GRIM::Lenses::appendHiddenJacobianRow(full,slice,out);
        if(out==0)rejects([&]{GRIM::Lenses::inspectionReport(full,config,"weights",[](int){return "token";});});
    }
    check(full.jacobian_rows==128);
    rejects([&]{GRIM::Lenses::appendHiddenJacobianRow(full,slice,127);});
    const auto matrixWire=GRIM::Lenses::inspectionReport(full,config,"weights",[](int){return "token";});
    const auto matrices=parseReport(matrixWire,config);
    check(matrices.jacobianTargetPosition==2);
    const auto& identity=*matrices.positions[2].layers[1].hiddenJacobian;
    check(identity[0]==1 && identity[1]==0 && identity[128*127+127]==1);
    check(*matrices.positions[0].layers[0].hiddenJacobian==full.snapshots[0].hidden_jacobian);
    saveCaptureFile(capturePath,matrixWire);
    check(*loadReport(capturePath,config).positions[2].layers[1].hiddenJacobian==identity);
    // Explicit selected rows use only d_model entries and retain the output index,
    // including dimension zero (an engaged optional, not absence of metadata).
    for(int dimension:{0,127}) {
        auto selected=capture;selected.jacobian_target_position=2;
        selected.jacobian_output_dimension=dimension;selected.jacobian_rows=1;
        for(auto& snapshot:selected.snapshots) {
            snapshot.hidden_jacobian.assign(128,0.f);
            snapshot.hidden_jacobian[dimension]=snapshot.metadata.token_position==2?1.f:-.25f;
        }
        const auto rowWire=GRIM::Lenses::inspectionReport(selected,config,"weights",[](int){return "token";});
        check(rowWire["jacobian"]["rows"]==1 && rowWire["jacobian"]["output_dimension"]==dimension);
        saveCaptureFile(capturePath,rowWire);
        const auto rowReport=loadReport(capturePath,config);
        check(rowReport.jacobianOutputDimension==unsigned(dimension));
        check(*rowReport.positions[0].layers[0].hiddenJacobian==selected.snapshots[0].hidden_jacobian);
        auto invalidRow=rowWire;invalidRow["jacobian"].erase("output_dimension");
        rejects([&]{parseReport(invalidRow,config);});
        invalidRow=rowWire;invalidRow["jacobian"]["output_dimension"]=128;
        rejects([&]{parseReport(invalidRow,config);});
        invalidRow=rowWire;invalidRow["jacobian"]["output_dimension"]=-1;
        rejects([&]{parseReport(invalidRow,config);});
        invalidRow=rowWire;invalidRow["jacobian"]["rows"]=128;
        rejects([&]{parseReport(invalidRow,config);});
        selected.jacobian_output_dimension=128;
        rejects([&]{GRIM::Lenses::inspectionReport(selected,config,"weights",[](int){return "token";});});
    }
    auto invalidMatrix=matrixWire;
    invalidMatrix["jacobian"]["rows"]=127;rejects([&]{parseReport(invalidMatrix,config);});
    invalidMatrix=matrixWire;invalidMatrix["positions"][0]["layers"][0]["hidden_jacobian_f32_hex"]="00000000";
    rejects([&]{parseReport(invalidMatrix,config);});
    invalidMatrix=matrixWire;invalidMatrix["positions"][0]["layers"].erase(0);
    rejects([&]{parseReport(invalidMatrix,config);});
    rejects([]{GRIM::Lenses::decodeJacobian("7fc00000",1);}); // NaN
    rejects([]{GRIM::Lenses::decodeJacobian("7f800000",1);}); // infinity
    rejects([]{GRIM::Lenses::decodeJacobian("zz000000",1);});
    check(GRIM::Lenses::hiddenJacobianBytes(768,5,12)==135ULL*1024*1024);
    rejects([]{GRIM::Lenses::hiddenJacobianBytes(0,1,1);});
    rejects([]{GRIM::Lenses::hiddenJacobianBytes(2147483647,2147483647,2147483647);});
    selectReadout(report,"actual_final");
    check(report.positions[0].layers[0].candidates.empty() && report.positions[0].layers[1].candidates.size()==1);
    rejects([&]{GRIM::Lenses::inspectionReport(GRIM::Lenses::LensCaptureResult{},config,"weights-42",[](int){return "";});});
    std::cout<<"Observatory report tests passed\n";
}
