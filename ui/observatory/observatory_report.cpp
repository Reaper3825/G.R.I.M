#include "observatory_report.hpp"
#include "resources/models/GRIM-text/Shared/Lenses/JacobianCapture.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <set>
#include <sstream>
#include <stdexcept>
#ifdef _WIN32
#include <Windows.h>
#endif

namespace GRIM::Observatory {
namespace {
void require(bool ok,const char* message) { if(!ok) throw std::runtime_error(message); }
unsigned number(const nlohmann::json& value) {
    require(value.is_number_integer(),"Capture integer field has wrong type");
    const auto n=value.get<int64_t>();
    require(n>=0 && n<=1000000,"Capture integer field out of bounds");
    return static_cast<unsigned>(n);
}
}
std::string configDigest(const Config::CompiledModelConfigSnapshot& config) {
    std::ostringstream out;
    for(auto b:config.integrity.semantic_sha256) out<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(b);
    return out.str();
}
nlohmann::json parseInspectionResponse(int status,const std::string& body) {
    require(body.size()<=512ULL*1024*1024,"Inspection exceeds the 512 MiB host response budget");
    // Missing routes commonly return an empty or HTML body. Preserve the HTTP
    // failure instead of reporting it as malformed capture JSON.
    const auto json=nlohmann::json::parse(body,nullptr,false);
    if(status!=200) {
        const auto prefix="Inspection HTTP "+std::to_string(status)+": ";
        if(status==404)
            throw std::runtime_error(prefix+"Inspection endpoint is unavailable. Rebuild and restart the GRIM-text bridge and inference worker with inspection support.");
        if(json.is_object()) {
            for(const char* key:{"message","error"}) {
                const auto found=json.find(key);
                if(found!=json.end() && found->is_string() && !found->get_ref<const std::string&>().empty())
                    throw std::runtime_error(prefix+found->get<std::string>());
            }
        }
        throw std::runtime_error(prefix+"The server returned an error without a JSON message. Check the GRIM-text bridge and loaded inference worker.");
    }
    require(!json.is_discarded(),"Inspection endpoint returned empty or invalid JSON (HTTP 200). Check the GRIM-text server response.");
    require(json.is_object(),"Inspection endpoint returned a JSON value instead of a capture object (HTTP 200).");
    return json;
}
Report parseReport(const nlohmann::json& j,const Config::CompiledModelConfigSnapshot& config) {
    require(number(j.at("schema_version"))==1,"Unsupported Observatory capture schema");
    Report r;
    r.configSha256=j.at("config_sha256").get<std::string>();
    require(r.configSha256==configDigest(config),"Capture does not match the selected .grimcfg fingerprint");
    r.layerCount=number(j.at("layer_count"));r.dModel=number(j.at("d_model"));r.vocabSize=number(j.at("vocab_size"));
    require(r.layerCount==config.architecture.num_layers && r.dModel==config.architecture.d_model,"Capture/model geometry mismatch");
    require(r.layerCount>0 && r.vocabSize>0,"Capture geometry is empty");
    r.mode=j.at("mode").get<std::string>();
    require(r.mode=="inference"||r.mode=="training_example","Unknown capture mode");
    r.visibility=j.at("attention_visibility").get<std::string>();
    require(r.visibility=="causal"||r.visibility=="full_sequence","Unknown attention visibility");
    r.checkpoint=j.at("checkpoint").get<std::string>();
    require(!r.checkpoint.empty() && r.checkpoint.size()<4096,"Missing or excessive checkpoint identity");
    r.synthetic=j.value("synthetic",false);
    if(j.contains("jacobian")) {
        const auto& jacobian=j.at("jacobian");
        require(!r.synthetic && jacobian.at("kind")=="final_hidden" && jacobian.at("layout")=="output_by_input" &&
            jacobian.at("encoding")=="ieee754_f32_hex" && jacobian.at("boundary")=="post_encoder_block","Unknown Jacobian contract");
        if(jacobian.contains("output_dimension")) {
            r.jacobianOutputDimension=number(jacobian.at("output_dimension"));
            require(*r.jacobianOutputDimension<r.dModel,"Jacobian output dimension is out of range");
        }
        require(number(jacobian.at("rows"))==(r.jacobianOutputDimension?1u:r.dModel) && number(jacobian.at("columns"))==r.dModel,"Jacobian/model dimension mismatch");
        r.jacobianTargetPosition=number(jacobian.at("target_position"));
        require(*r.jacobianTargetPosition<config.architecture.max_seq_len,"Jacobian target position is out of range");
    }
    const auto& positions=j.at("positions");
    require(positions.is_array()&&!positions.empty()&&positions.size()<=config.architecture.max_seq_len,"Invalid capture position count");
    if(r.jacobianTargetPosition)
        require(GRIM::Lenses::hiddenJacobianBytes(int(r.dModel),int(positions.size()),int(r.layerCount))/(r.jacobianOutputDimension?r.dModel:1u)<=192ULL*1024*1024,"Jacobian matrix exceeds the host capture limit");
    std::set<unsigned> seenPositions;
    for(const auto& item:positions) {
        PositionReadout p;p.position=number(item.at("position"));p.inputText=item.at("input_text").get<std::string>();
        require(p.position<config.architecture.max_seq_len && p.inputText.size()<=1024 && seenPositions.insert(p.position).second,"Invalid or repeated capture position");
        const auto& layers=item.at("layers");
        require(layers.is_array()&&!layers.empty()&&layers.size()<=r.layerCount,"Invalid layer readouts");
        std::set<unsigned> seenLayers;
        for(const auto& source:layers) {
            LayerReadout l;l.layer=number(source.at("layer"));
            if(r.jacobianTargetPosition) {
                l.hiddenJacobian=std::make_shared<const std::vector<float>>(GRIM::Lenses::decodeJacobian(source.at("hidden_jacobian_f32_hex").get_ref<const std::string&>(),size_t(r.dModel)*(r.jacobianOutputDimension?1u:r.dModel)));
            } else require(!source.contains("hidden_jacobian_f32_hex"),"Jacobian matrix has no target metadata");
            require(l.layer<r.layerCount && seenLayers.insert(l.layer).second,"Invalid or repeated layer index");
            const auto readouts=source.contains("readouts") ? source.at("readouts") : nlohmann::json::array({source});
            require(readouts.is_array()&&!readouts.empty()&&readouts.size()<=3,"Invalid readout count");
            std::set<std::string> kinds;
            for(const auto& row:readouts) {
                ReadoutVariant variant;variant.kind=row.value("kind",std::string("direct"));
                require((variant.kind=="direct" || variant.kind=="actual_final" || variant.kind=="identity_control") && kinds.insert(variant.kind).second,"Invalid or repeated readout kind");
                require(variant.kind!="actual_final" || l.layer+1==r.layerCount,"Actual logits belong to the final layer");
                variant.entropyNats=row.at("entropy_nats").get<double>();
                require(std::isfinite(variant.entropyNats)&&variant.entropyNats>=0&&variant.entropyNats<=std::log(double(r.vocabSize))+1e-5,"Invalid full-vocabulary entropy");
                const auto& candidates=row.at("candidates");
                require(candidates.is_array()&&!candidates.empty()&&candidates.size()<=r.vocabSize,"Invalid candidate count");
                std::set<int> ids;double total=0;
                for(const auto& candidate:candidates) {
                    Candidate c;c.tokenId=static_cast<int>(number(candidate.at("token_id")));c.text=candidate.at("text").get<std::string>();c.probability=candidate.at("probability").get<double>();
                    require(unsigned(c.tokenId)<r.vocabSize && ids.insert(c.tokenId).second && c.text.size()<=1024,"Invalid candidate identity");
                    require(std::isfinite(c.probability)&&c.probability>=0&&c.probability<=1,"Invalid candidate probability");
                    if(candidate.contains("logit")) {
                        require(candidate.at("logit").is_number() && std::isfinite(candidate.at("logit").get<float>()),"Invalid candidate logit");
                        c.logit=candidate.at("logit").get<float>();
                    }
                    total+=c.probability;variant.candidates.push_back(std::move(c));
                }
                require(total<=1.00001,"Candidate probabilities exceed one");
                std::stable_sort(variant.candidates.begin(),variant.candidates.end(),[](const auto& a,const auto& b){return a.probability>b.probability;});
                l.readouts.push_back(std::move(variant));
            }
            if(source.contains("identity_max_abs_logit_error")) {
                const double error=source.at("identity_max_abs_logit_error").get<double>();
                require(l.layer+1==r.layerCount && std::isfinite(error)&&error>=0,"Invalid final-layer comparison error");
                l.identityMaxAbsLogitError=error;
            }
            const auto preferred=std::find_if(l.readouts.begin(),l.readouts.end(),[](const auto& row){return row.kind=="direct";});
            const auto& initial=preferred==l.readouts.end()?l.readouts.front():*preferred;
            l.entropyNats=initial.entropyNats;l.candidates=initial.candidates;
            p.layers.push_back(std::move(l));
        }
        std::sort(p.layers.begin(),p.layers.end(),[](const auto& a,const auto& b){return a.layer<b.layer;});
        r.positions.push_back(std::move(p));
    }
    if(r.jacobianTargetPosition) {
        require(seenPositions.count(*r.jacobianTargetPosition)!=0,"Jacobian target position is not captured");
        for(const auto& p:r.positions)require(p.position<r.positions.size() && p.layers.size()==r.layerCount,"Jacobian is missing source positions or encoder layers");
    }
    return r;
}
Report loadReport(const std::filesystem::path& path,const Config::CompiledModelConfigSnapshot& config) {
    require(std::filesystem::file_size(path)<=512ULL*1024*1024,"Capture exceeds the 512 MiB host file budget");
    std::ifstream stream(path);if(!stream)throw std::runtime_error("Cannot open capture file");
    nlohmann::json json;stream>>json;
    require(!json.value("capture_pending",false),"No saved inspection yet. Enter input and choose Run inspection.");
    return parseReport(json,config);
}
std::filesystem::path defaultCapturePath(const std::filesystem::path& configPath) {
    return configPath.parent_path()/"observatory_capture.json";
}
void saveCaptureFile(const std::filesystem::path& path,const nlohmann::json& json) {
    // Replace only after a complete write, preserving the previous capture on failure.
    auto temporary=path;temporary+=".tmp";
    try {
        std::ofstream stream(temporary,std::ios::binary|std::ios::trunc);
        if(!stream)throw std::runtime_error("Cannot create Observatory capture file");
        stream<<json.dump()<<'\n';stream.close();
        if(!stream)throw std::runtime_error("Cannot write Observatory capture file");
        require(std::filesystem::file_size(temporary)<=512ULL*1024*1024,"Capture exceeds the 512 MiB host file budget");
#ifdef _WIN32
        if(!MoveFileExW(temporary.c_str(),path.c_str(),MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH))
            throw std::runtime_error("Cannot replace Observatory capture file");
#else
        std::filesystem::rename(temporary,path);
#endif
    } catch(...) {
        std::error_code ignored;std::filesystem::remove(temporary,ignored);throw;
    }
}
void ensureCaptureFile(const std::filesystem::path& path) {
    if(!std::filesystem::exists(path))
        saveCaptureFile(path,{{"schema_version",1},{"capture_pending",true}});
}
void selectReadout(Report& report,const std::string& kind) {
    for(auto& position:report.positions)for(auto& layer:position.layers) {
        if(layer.readouts.empty())continue; // Synthetic and legacy presentation.
        const auto it=std::find_if(layer.readouts.begin(),layer.readouts.end(),[&](const auto& row){return row.kind==kind;});
        layer.candidates.clear();layer.entropyNats=0;
        if(it!=layer.readouts.end()) {layer.candidates=it->candidates;layer.entropyNats=it->entropyNats;}
    }
}
Report makePreview(const Config::CompiledModelConfigSnapshot& config) {
    require(config.architecture.num_layers>0&&config.architecture.num_layers<=4096,"Preview supports 1 to 4096 configured layers");
    Report r;r.configSha256=configDigest(config);r.layerCount=config.architecture.num_layers;r.dModel=config.architecture.d_model;
    r.vocabSize=6;r.synthetic=true;r.mode="inference";r.visibility="causal";r.checkpoint="Synthetic visual preview (no checkpoint loaded)";
    const std::vector<std::string> inputs={"The","capital","of","France","is"};
    const std::vector<std::string> expected={"capital","of","France","is","Paris"};
    for(unsigned t=0;t<inputs.size();++t) {
        PositionReadout p{t,inputs[t],{}};
        for(unsigned l=0;l<r.layerCount;++l) {
            const double depth=double(l+1)/r.layerCount;
            const double target=std::clamp(.06+.7*depth+.16*std::sin(depth*22+t),.01,.91);
            LayerReadout row{l,0,{}};
            for(int k=0;k<6;++k) {
                const double probability=k==0?target:(1-target)/5;
                row.entropyNats-=probability*std::log(probability);
                row.candidates.push_back({k,k==0?expected[t]:"alternative "+std::to_string(k),probability});
            }
            std::stable_sort(row.candidates.begin(),row.candidates.end(),[](const auto& a,const auto& b){return a.probability>b.probability;});
            p.layers.push_back(std::move(row));
        }
        r.positions.push_back(std::move(p));
    }
    return r;
}
}
