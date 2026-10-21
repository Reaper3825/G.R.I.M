//======================================================//
//  Phase2_InspectionLoop.cu
//  Prefill lens capture and Jacobian inspection.
//======================================================//

#include "Phase2_InspectionLoop.hpp"

#include "../../Shared/Batching/BatchDeviceUpload.hpp"
#include "../../Shared/Forward/ModelForward_GPU.hpp"
#include "../../Shared/HyperParameters/HyperparameterGroupings.hpp"
#include "../../Shared/InferenceState/InferencePrefill.hpp"
#include "../../Shared/Lenses/JacobianCapture.hpp"

#include <chrono>
#include <memory>
#include <stdexcept>

namespace GRIMText::Training {
namespace {
struct InspectionForwardScope {
    GRIM::Forward::ModelForwardOutputs& outputs;
    ~InspectionForwardScope() { outputs.clear(); }
};
} // namespace

Phase2TextInferenceResult executePhase2Inspection(
    TrainingContext& ctx, GRIM::Tokenizer::UniByte& tokenizer,
    const std::string& prompt, const nlohmann::json* input_state,
    const GRIM::Lenses::LensCaptureRequest& capture) {
    if (prompt.empty()) throw std::invalid_argument("Inspection input is empty");
    // Use the same tokenizer/span adapter, device upload, cached prefill and
    // forward lifecycle as inference. Return before sampling or decode forwards.
    auto prefill = GRIM::Inference::buildInferencePrefill(ctx.config, tokenizer, prompt, input_state);
    if(capture.capture_jacobian) {
        if(prefill.EnableAtomIdentification || prefill.batch_size!=1 || prefill.total_tokens!=prefill.seq_lengths.at(0) ||
           capture.batch_row!=0 || !capture.all_layers || !capture.all_positions)
            throw std::invalid_argument("Jacobian row inspection requires one compact token prefill and all layers/positions");
        const auto model_hp = GRIM::HyperParameters::modelHP(ctx.config);
        const int width = model_hp.encoder_d_model;
        const int layers = model_hp.encoder_num_layers;
        const auto matrixBytes=GRIM::Lenses::hiddenJacobianBytes(width,prefill.total_tokens,layers)/width;
        const auto rowBytes=3ULL*prefill.total_tokens*width*sizeof(float);
        if(matrixBytes>192ULL*1024*1024 || matrixBytes>capture.temporary_memory_budget_bytes ||
           rowBytes>capture.temporary_memory_budget_bytes-matrixBytes)
            throw std::invalid_argument("Jacobian row exceeds capture scratch or response budget; shorten input or increase scratch MiB (response limit: 192 MiB FP32)");
        auto& training=ctx.requireTrainingState("Jacobian inspection");
        const auto stream=training.stream_ctrl.getPrimaryStream();
        GRIM::Batching::attachBatchDeviceStorage(
            prefill, GRIM::Batching::createBatchDeviceStorage(ctx.config, stream),
            "Jacobian inspection");
        const auto bindings=GRIM::Batching::uploadBatchToDevice(ctx.config,prefill,stream);
        const auto started=std::chrono::steady_clock::now();
        auto rowCapture=capture;
        GRIM::Forward::ModelForwardRequest request{};
        request.config=&ctx.config;request.gpu_encoder=ctx.gpu_model.gpu_encoder.get();
        request.parameter_registry=&ctx.parameter_registry;request.pbm=&ctx.pbm_owner.state();
        request.cublas_handle=training.cublas_handle.get();request.stream=stream;
        request.payload=&prefill;request.bindings=&bindings;request.lens_capture=&rowCapture;
        // One uncached activation-only forward/backward for the selected output dimension.
        GRIM::Forward::ModelForwardRuntimePayload runtime;
        auto outputs=GRIM::Forward::executeModelForward(request,runtime);
        InspectionForwardScope cleanup{outputs};
        if(!outputs.lens_capture_result)throw std::runtime_error("Jacobian inspection returned no capture");
        const auto matrices=outputs.lens_capture_result;
        Phase2TextInferenceResult result;
        result.prompt_token_count=prefill.total_tokens;result.sequence_token_count=prefill.total_tokens;
        result.token_ids=prefill.input_ids;
        result.generation_ms=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-started).count();
        result.prefill_lens_capture_result=matrices;
        result.prefill_lens_snapshot=std::shared_ptr<const GRIM::Lenses::LensSnapshot>(matrices,&matrices->snapshots.back());
        return result;
    }
    auto hp = GRIM::HyperParameters::generationHP(ctx.config);
    hp.num_return_sequences = 1;
    hp.max_new_tokens = 1;
    hp.min_new_tokens = 0;
    hp.strategy = GRIM::HyperParameters::SamplingStrategy::GREEDY;
    hp.do_sample = false;
    return executePhase2PayloadInference(ctx, tokenizer, prefill, hp, &capture, true);
}

} // namespace GRIMText::Training
