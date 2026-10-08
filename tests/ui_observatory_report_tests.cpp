#include "ui/observatory/observatory_report.hpp"
#include "resources/models/GRIM-text/Shared/Lenses/LensInspectionReport.hpp"
#include <nlohmann/json.hpp>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace GRIM::Observatory;
using nlohmann::json;
void check(bool condition) { if(!condition)throw std::runtime_error("Observatory report assertion failed"); }
template<class F> void rejects(F action) { bool rejected=false;try{action();}catch(const std::exception&){rejected=true;}check(rejected); }
int main() {
    GRIM::Config::CompiledModelConfigSnapshot config;
    config.architecture.max_seq_len=8192;config.architecture.num_layers=7;config.architecture.d_model=128;
    config.integrity.semantic_sha256[0]=0xab;
    json candidate={{"token_id",2},{"text","word"},{"probability",.4}};
    json layer={{"layer",6},{"entropy_nats",2.0},{"candidates",json::array({candidate})}};
    json source={{"schema_version",1},{"config_sha256",configDigest(config)},{"layer_count",7},{"d_model",128},
        {"vocab_size",100},{"mode","inference"},{"attention_visibility","causal"},{"checkpoint","checkpoint-42"},
        {"positions",json::array({{{"position",3},{"input_text","a"},{"layers",json::array({layer})}}})}};
    auto report=parseReport(source,config);
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
    selectReadout(report,"actual_final");
    check(report.positions[0].layers[0].candidates.empty() && report.positions[0].layers[1].candidates.size()==1);
    rejects([&]{GRIM::Lenses::inspectionReport(GRIM::Lenses::LensCaptureResult{},config,"weights-42",[](int){return "";});});
    std::cout<<"Observatory report tests passed\n";
}
