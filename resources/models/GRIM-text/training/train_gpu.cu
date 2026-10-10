//======================================================//
//  GRIM-text GPU Training Orchestrator
//  Three-phase training architecture
//  
//  Phases:
//  - Orchestrator handoff: load validated training startup config root
//  - Phase 1: Startup (tokenizer, model init, data loading)
//  - Phase 2: Training loop (epochs, batches, optimization)
//  - Phase 3: Cleanup (final save, status, resources)
//  
//  This file orchestrates the three phases for easier
//  debugging and maintainability. Each phase is isolated
//  in its own compilation unit for:
//  - Faster incremental builds
//  - Easier debugging and breakpoints
//  - Clear data flow contracts between phases
//  
//  Author: Austin Wadkins
//  Date: December 2025
//  Version: 3.0.0 - Three-Phase Architecture
//======================================================//

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#endif

#include "../Shared/HyperParameters/HyperParameters_GPU.hpp"
#include "Phases/Phase1_Startup.hpp"
#include "Phases/Phase2_TrainingLoop.hpp"
#include "Phases/Phase2_InferenceLoop.hpp"
#include "Phases/Phase3_Cleanup.hpp"
#include "../Shared/LogRecorder/LogRecorder.hpp"

#include "../Shared/Lenses/LensInspectionReport.hpp"
#include "Diagnostics/TokenizerDiagnostics.hpp"
#include <mutex>
#include <filesystem>
#include <chrono>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <httplib.h>
#include <nlohmann/json.hpp>

using GRIM::Logging::ModuleId;
using GRIM::Logging::EmitModuleInfo;
using GRIM::Logging::EmitModuleError;

namespace {

using json = nlohmann::json;

void printBanner() {
    EmitModuleInfo(ModuleId::TrainingOrchestrator, 
        "╔════════════════════════════════════════════════════════╗", 0);
    EmitModuleInfo(ModuleId::TrainingOrchestrator, 
        "║          GRIM-text Training v3.0.0                     ║", 0);
    EmitModuleInfo(ModuleId::TrainingOrchestrator, 
        "╚════════════════════════════════════════════════════════╝", 0);
}

void printPhaseHeader(int phase, const char* description) {
    // Emit the phase header using EmitModuleInfo directly.
    EmitModuleInfo(ModuleId::TrainingOrchestrator,
        "┌─────────────────────────────────────────────────────────┐", 0);

    // Build the centered phase line without intermediate ostringstream
    std::string line = "│  PHASE ";
    line += std::to_string(phase);
    line += ": ";
    line += description;
    // Pad to fixed width (47 chars for description area)
    const int target_width = 47;
    int desc_len = static_cast<int>(std::strlen(description));
    int pad = target_width - desc_len;
    if (pad < 0) pad = 0;
    line.append(pad, ' ');
    line += "│";
    EmitModuleInfo(ModuleId::TrainingOrchestrator, line, 0);
    EmitModuleInfo(ModuleId::TrainingOrchestrator,
        "└─────────────────────────────────────────────────────────┘", 0);
}

GRIM::HyperParameters::ModelExecutionMode requestedExecutionMode(int argc, char** argv) {
    bool requested_training = false;
    bool requested_inference = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--training") {
            requested_training = true;
        } else if (arg == "--inference") {
            requested_inference = true;
        }
    }
    if (requested_training && requested_inference) {
        throw std::runtime_error("train_gpu: --training and --inference are mutually exclusive");
    }
    return requested_inference
        ? GRIM::HyperParameters::ModelExecutionMode::INFERENCE
        : GRIM::HyperParameters::ModelExecutionMode::TRAINING;
}

int requestedInferenceWorkerPort(int argc, char** argv) {
    int port = 11436;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--inference-worker-port") {
            if (i + 1 >= argc) {
                throw std::runtime_error("train_gpu: --inference-worker-port requires a value");
            }
            port = std::stoi(argv[++i]);
        }
    }
    if (port <= 0 || port > 65535) {
        throw std::runtime_error("train_gpu: --inference-worker-port must be in 1..65535");
    }
    return port;
}

const std::string& requireJsonString(const json& request, const char* field_name) {
    if (!request.contains(field_name)) {
        throw std::runtime_error(std::string("inference worker request missing required field: ") + field_name);
    }
    const auto& value = request.at(field_name);
    if (!value.is_string()) {
        throw std::runtime_error(std::string("inference worker request field is not a string: ") + field_name);
    }
    return value.get_ref<const std::string&>();
}

GRIM::HyperParameters::GenerationHP generationHPFromRequest(
    const GRIM::Config::AiConfigSnapshot& config,
    const json& request)
{
    GRIM::HyperParameters::GenerationHP gen_config =
        GRIM::HyperParameters::generationHP(config);

    if (request.contains("max_tokens")) gen_config.max_new_tokens = request.at("max_tokens").get<int>();
    if (request.contains("temperature")) gen_config.temperature = request.at("temperature").get<float>();
    if (request.contains("top_p")) gen_config.top_p = request.at("top_p").get<float>();
    if (request.contains("top_k")) gen_config.top_k = request.at("top_k").get<int>();
    if (request.contains("min_p")) gen_config.min_p = request.at("min_p").get<float>();
    if (request.contains("typical_p")) gen_config.typical_p = request.at("typical_p").get<float>();
    if (request.contains("repetition_penalty")) gen_config.repetition_penalty = request.at("repetition_penalty").get<float>();
    if (request.contains("frequency_penalty")) gen_config.frequency_penalty = request.at("frequency_penalty").get<float>();
    if (request.contains("presence_penalty")) gen_config.presence_penalty = request.at("presence_penalty").get<float>();
    if (request.contains("no_repeat_ngram_size")) gen_config.no_repeat_ngram_size = request.at("no_repeat_ngram_size").get<int>();
    if (request.contains("seed")) gen_config.seed = request.at("seed").get<unsigned int>();
    if (request.contains("strategy")) {
        const std::string strategy_name = request.at("strategy").get<std::string>();
        gen_config.strategy = GRIM::HyperParameters::parseGenerationSamplingStrategy(strategy_name);
        if (gen_config.strategy == GRIM::HyperParameters::SamplingStrategy::GREEDY) {
            gen_config.do_sample = false;
        }
    }

    return gen_config;
}

json inferenceStatsJson(const GRIMText::Training::Phase2TextInferenceResult& result) {
    return json{
        {"prompt_token_count", result.prompt_token_count},
        {"sequence_token_count", result.sequence_token_count},
        {"encode_ms", result.encode_ms},
        {"generation_ms", result.generation_ms},
        {"decode_ms", result.decode_ms}
    };
}

std::string chatPromptFromRequest(const json& request, bool structured_state = false) {
    if (!request.contains("messages")) {
        throw std::runtime_error("inference worker chat request missing required field: messages");
    }
    const auto& messages = request.at("messages");
    if (!messages.is_array()) {
        throw std::runtime_error("inference worker chat request field messages is not an array");
    }

    if (structured_state && messages.size() == 1 &&
        requireJsonString(messages.front(), "role") == "user")
        return requireJsonString(messages.front(), "content");

    std::string prompt;
    for (const auto& msg : messages) {
        const std::string role = requireJsonString(msg, "role");
        const std::string content = requireJsonString(msg, "content");
        if (role == "system") {
            prompt += "System: " + content + "\n";
        } else if (role == "user") {
            prompt += "User: " + content + "\n";
        } else if (role == "assistant") {
            prompt += "Assistant: " + content + "\n";
        } else {
            throw std::runtime_error("inference worker chat request contains unsupported role: " + role);
        }
    }
    if (!structured_state) prompt += "Assistant: ";
    return prompt;
}

GRIMText::Training::Phase2TextInferenceResult executeWorkerInference(
    GRIMText::Training::TrainingContext& ctx, GRIM::Tokenizer::UniByte& tokenizer,
    const std::string& prompt, const json& request,
    const GRIM::HyperParameters::GenerationHP& generation_hp) {
    if (!request.contains("reasoning_state"))
        return GRIMText::Training::executePhase2TextInference(ctx, tokenizer, prompt, generation_hp);
    const auto start = std::chrono::steady_clock::now();
    auto prefill = GRIMText::Training::buildPhase2InferencePrefill(
        ctx, tokenizer, prompt, request.at("reasoning_state"));
    const auto encode_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count();
    auto result = GRIMText::Training::executePhase2PayloadInference(ctx, tokenizer, prefill, generation_hp);
    result.encode_ms = encode_ms;
    return result;
}

int runInferenceWorker(
    GRIMText::Training::TrainingContext& ctx,
    GRIM::Tokenizer::UniByte& tokenizer,
    int port) {
    httplib::Server svr;
    // All routes share model workspaces, KV state and the tokenizer.
    std::mutex execution_mutex;
    const auto paths_hp = GRIM::HyperParameters::pathsHP(ctx.config);
    std::uint64_t vocab_hash = 0;
    const std::string tokenizer_identity = GRIM::Tokenizer::AtomTable::computeFileHash(paths_hp.vocab_path, vocab_hash)
        ? "vocab-file-hash64:" + std::to_string(vocab_hash) : std::string{};

    svr.Get("/internal/status", [&](const httplib::Request&, httplib::Response& res) {
        json response = {
            {"status", "ready"},
            {"model", "grim-text"},
            {"execution_mode", "inference"},
            {"config_source", "ai_config.json"},
            {"configured_model_path", paths_hp.output_model_path},
            {"loaded_checkpoint_path", ctx.loaded_checkpoint_path},
            {"vocab_path", paths_hp.vocab_path}
        };
        res.set_content(response.dump(), "application/json");
    });

    const auto tokenizer_request = [&](const httplib::Request& req, httplib::Response& res, bool encode) {
        try {
            const auto request = json::parse(req.body);
            if (!request.is_object()) throw std::invalid_argument("Expected a JSON object");
            if (request.contains("vocab_path") || request.contains("data_path") || request.contains("config_path"))
                throw std::invalid_argument("Tokenizer diagnostics use the loaded model's artifacts");
            // Do not wait behind a long generation request: the UI can retry.
            std::unique_lock<std::mutex> lock(execution_mutex, std::try_to_lock);
            if (!lock.owns_lock()) {
                res.status = 409;
                res.set_content(json({{"error", "Model worker is busy; retry when the current operation finishes"}}).dump(), "application/json");
                return;
            }
            auto response = encode
                ? GRIMText::Diagnostics::encodeTokenizerText(tokenizer, requireJsonString(request, "text"))
                : GRIMText::Diagnostics::validateLoadedTokenizer(tokenizer);
            response["vocab_path"] = paths_hp.vocab_path;
            response["loaded_checkpoint_path"] = ctx.loaded_checkpoint_path;
            res.status = response.value("status", std::string{}) == "success" ? 200 : 422;
            res.set_content(response.dump(-1, ' ', false, json::error_handler_t::replace), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json({{"status", "error"}, {"error", e.what()}}).dump(), "application/json");
        }
    };
    svr.Post("/internal/tokenizer/run", [&](const httplib::Request& req, httplib::Response& res) {
        tokenizer_request(req, res, false);
    });
    svr.Post("/internal/tokenizer/encode", [&](const httplib::Request& req, httplib::Response& res) {
        tokenizer_request(req, res, true);
    });

    svr.Post("/internal/inspect", [&](const httplib::Request& req, httplib::Response& res) {
        try {
            const auto request = json::parse(req.body);
            if (request.value("mode", std::string("inference")) != "inference")
                throw std::invalid_argument("Training replay requires a prepared training-window payload; it is not an inference prompt");
            const auto& compiled = *ctx.config.model_config;
            if (requireJsonString(request, "config_sha256") != GRIM::Lenses::inspectionConfigDigest(compiled) ||
                !std::filesystem::equivalent(requireJsonString(request, "config_path"), compiled.source_path))
                throw std::invalid_argument("Selected model differs from the loaded inference model");
            GRIM::Lenses::LensCaptureRequest capture;
            capture.all_layers = true;
            capture.all_positions = true;
            const auto& top_k = request.at("top_k");
            if (!top_k.is_number_integer() || top_k.get<int64_t>() <= 0 || top_k.get<int64_t>() > tokenizer.vocabSize())
                throw std::invalid_argument("Inspection top-k must be in 1..vocabulary size");
            capture.top_k = top_k.get<int>();
            const auto budget = request.at("temporary_memory_budget_bytes");
            if (!budget.is_number_integer() || budget.get<int64_t>() <= 0)
                throw std::invalid_argument("Inspection scratch budget must be a positive integer");
            capture.temporary_memory_budget_bytes = budget.get<uint64_t>();
            capture.replay_identity_control = true;
            if(request.contains("jacobian")) {
                const auto& jacobian=request.at("jacobian");
                if(!jacobian.is_object() || jacobian.value("kind",std::string{})!="final_hidden")
                    throw std::invalid_argument("Jacobian kind must be final_hidden");
                capture.capture_jacobian=true;
                if(jacobian.contains("output_dimension")) {
                    const auto& dimension=jacobian.at("output_dimension");
                    if(!dimension.is_number_integer() || dimension<0 || dimension>=compiled.architecture.d_model)
                        throw std::invalid_argument("Invalid Jacobian output dimension");
                    capture.jacobian_output_dimension=dimension.get<int>();
                }
                if(jacobian.contains("target_position")) {
                    const auto& position=jacobian.at("target_position");
                    if(!position.is_number_integer() || position.get<int64_t>() < -1 || position.get<int64_t>()>std::numeric_limits<int>::max())
                        throw std::invalid_argument("Invalid Jacobian target position");
                    capture.jacobian_target_position=position.get<int>();
                }
            }
            if (tokenizer_identity.empty())throw std::runtime_error("Cannot identify the loaded tokenizer artifact");
            capture.identity.execution_session_id = ctx.logging.session_id;
            capture.identity.tokenizer_fingerprint = tokenizer_identity;
            capture.identity.checkpoint_fingerprint = ctx.loaded_checkpoint_path;
            capture.identity.compiled_config_fingerprint = GRIM::Lenses::inspectionConfigDigest(compiled);
            const auto& prompt = requireJsonString(request, "prompt");
            const auto* input_state = request.contains("reasoning_state") ? &request.at("reasoning_state") : nullptr;
            std::lock_guard<std::mutex> execution_lock(execution_mutex);
            auto result = GRIMText::Training::executePhase2Inspection(ctx, tokenizer, prompt, input_state, capture);
            if (!result.prefill_lens_capture_result)
                throw std::runtime_error("Inspection did not publish its capture result");
            auto report = GRIM::Lenses::inspectionReport(*result.prefill_lens_capture_result,
                compiled, ctx.loaded_checkpoint_path, [&](int id) {
                    return tokenizer.decode(GRIM::Tokenizer::DecodeRequest({id}));
                });
            res.set_content(report.dump(-1, ' ', false, json::error_handler_t::replace), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json({{"error", e.what()}}).dump(), "application/json");
        }
    });

    svr.Post("/internal/generate", [&](const httplib::Request& req, httplib::Response& res) {
        try {
            const json request = json::parse(req.body);
            const std::string prompt = requireJsonString(request, "prompt");
            const auto gen_config = generationHPFromRequest(ctx.config, request);
            std::lock_guard<std::mutex> execution_lock(execution_mutex);
            const auto generated = executeWorkerInference(ctx, tokenizer, prompt, request, gen_config);

            json response = {
                {"model", "grim-text"},
                {"created_at", "2025-11-05T00:00:00Z"},
                {"response", GRIM::HyperParameters::atomInsertionBoundaryProjectionHP(ctx.config).enabled
                    ? generated.text : generated.continuation_text},
                {"done", true},
                {"stats", inferenceStatsJson(generated)}
            };
            res.set_content(response.dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 500;
            EmitModuleError(ModuleId::TrainingOrchestrator,
                std::string("[/internal/generate] ") + e.what(), ctx.global_step);
            res.set_content(json({{"error", std::string(e.what())}}).dump(), "application/json");
        }
    });

    svr.Post("/internal/chat", [&](const httplib::Request& req, httplib::Response& res) {
        try {
            const json request = json::parse(req.body);
            const std::string prompt = chatPromptFromRequest(request, request.contains("reasoning_state"));
            const auto gen_config = generationHPFromRequest(ctx.config, request);
            std::lock_guard<std::mutex> execution_lock(execution_mutex);
            const auto generated = executeWorkerInference(ctx, tokenizer, prompt, request, gen_config);

            json response = {
                {"model", "grim-text"},
                {"created_at", "2025-11-05T00:00:00Z"},
                {"message", {{"role", "assistant"}, {"content", GRIM::HyperParameters::atomInsertionBoundaryProjectionHP(ctx.config).enabled
                    ? generated.text : generated.continuation_text}}},
                {"done", true},
                {"stats", inferenceStatsJson(generated)}
            };
            res.set_content(response.dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 500;
            EmitModuleError(ModuleId::TrainingOrchestrator,
                std::string("[/internal/chat] ") + e.what(), ctx.global_step);
            res.set_content(json({{"error", std::string(e.what())}}).dump(), "application/json");
        }
    });

    std::ostringstream ready;
    ready << "[Phase 2] Inference worker listening on http://127.0.0.1:" << port;
    EmitModuleInfo(ModuleId::TrainingOrchestrator, ready.str(), ctx.global_step);

    if (!svr.listen("127.0.0.1", port)) {
        throw std::runtime_error("train_gpu: inference worker failed to listen on requested port " +
                                 std::to_string(port));
    }
    return 0;
}

} // anonymous namespace

//======================================================//
//  Main Entry Point
//======================================================//

#ifdef _WIN32
// Issue #142b: Windows SEH handler to catch CUDA device errors that bypass C++ exceptions.
// With default /EHsc, CUDA access violations trigger SEH that C++ catch(...) cannot catch.
// This handler ensures we get a log message instead of silent exit.
static LONG WINAPI GrimSEHHandler(EXCEPTION_POINTERS* ep) {
    DWORD code = ep ? ep->ExceptionRecord->ExceptionCode : 0;
    void* addr = ep ? ep->ExceptionRecord->ExceptionAddress : nullptr;
    
    // Log to stderr (most reliable path during crash)
    fprintf(stderr, "\n[FATAL-SEH] Unhandled structured exception!\n");
    fprintf(stderr, "[FATAL-SEH] Exception code: 0x%08lX\n", code);
    fprintf(stderr, "[FATAL-SEH] Exception address: %p\n", addr);
    
    switch (code) {
        case EXCEPTION_ACCESS_VIOLATION:
            fprintf(stderr, "[FATAL-SEH] EXCEPTION_ACCESS_VIOLATION — likely CUDA device error or buffer overflow\n");
            if (ep && ep->ExceptionRecord->NumberParameters >= 2) {
                fprintf(stderr, "[FATAL-SEH] %s address: 0x%p\n",
                    ep->ExceptionRecord->ExceptionInformation[0] == 0 ? "Read from" : "Write to",
                    (void*)ep->ExceptionRecord->ExceptionInformation[1]);
            }
            break;
        case EXCEPTION_STACK_OVERFLOW:
            fprintf(stderr, "[FATAL-SEH] EXCEPTION_STACK_OVERFLOW\n");
            break;
        case 0xC0000409:  // STATUS_STACK_BUFFER_OVERRUN (fast-fail)
            fprintf(stderr, "[FATAL-SEH] STATUS_STACK_BUFFER_OVERRUN — buffer overrun detected by /GS\n");
            break;
        default:
            fprintf(stderr, "[FATAL-SEH] Unknown exception code\n");
            break;
    }
    
    // Check CUDA state
    cudaError_t cuda_err = cudaPeekAtLastError();
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "[FATAL-SEH] Last CUDA error: %s (code=%d)\n",
            cudaGetErrorString(cuda_err), static_cast<int>(cuda_err));
    }
    
    // Also try to log via module system (may fail if state is corrupted)
    EmitModuleError(ModuleId::TrainingOrchestrator,
        std::string("[FATAL-SEH] Exception code=0x") +
        ([](DWORD c) { char buf[16]; snprintf(buf, sizeof(buf), "%08lX", c); return std::string(buf); })(code) +
        " addr=" + ([](void* a) { char buf[24]; snprintf(buf, sizeof(buf), "%p", a); return std::string(buf); })(addr), 0);
    
    fflush(stderr);
    return EXCEPTION_CONTINUE_SEARCH;  // Let default handler terminate
}
#endif

int main(int argc, char** argv) {
#ifdef _WIN32
    // Install SEH handler FIRST — before any CUDA calls.
    // Without this, CUDA device errors (illegal memory access, buffer overflow)
    // cause silent process exit because /EHsc C++ catch(...) cannot catch SEH.
    SetUnhandledExceptionFilter(GrimSEHHandler);
#endif
    
    printBanner();
    
    int exit_code = 0;
    
    try {
        //==================================================
        // PHASE 1: STARTUP
        //==================================================
        printPhaseHeader(1, "Startup");

        EmitModuleInfo(ModuleId::TrainingOrchestrator,
            "[Phase 1] Loading startup config...", 0);
        const auto execution_mode = requestedExecutionMode(argc, argv);
        const auto config_snapshot = GRIM::Config::loadAiConfigSnapshot();
        auto startup_config = GRIM::HyperParameters::finalizeAiConfigSnapshot(
            config_snapshot,
            argc,
            argv,
            execution_mode);
        const auto finalized_execution_mode =
            GRIM::HyperParameters::snapshotExecutionMode(startup_config);
        EmitModuleInfo(
            ModuleId::TrainingOrchestrator,
            std::string("[Phase 1] Execution path selected | requested=") +
                GRIM::HyperParameters::modelExecutionModeToJsonString(execution_mode) +
                " | finalized=" +
                GRIM::HyperParameters::modelExecutionModeToJsonString(finalized_execution_mode),
            0);
        EmitModuleInfo(ModuleId::TrainingOrchestrator,
            "[Phase 1] ✓ Startup config ready from canonical ai_config.json", 0);
        
        auto phase1 = GRIMText::Training::executePhase1(std::move(startup_config));

        if (phase1.outcome == GRIMText::Training::Phase1Outcome::tokenizer_only_complete) {
            return 0;
        }

        auto ctx = std::move(phase1.context);

        if (!ctx.gpu_model.gpu_encoder ||
            !ctx.training_state ||
            !ctx.training_state->initialized) {
            EmitModuleError(ModuleId::TrainingOrchestrator, 
                "Phase 1 failed: GPU model topology or runtime not initialized", 0);
            return 1;
        }

        if (phase1.outcome == GRIMText::Training::Phase1Outcome::ready_for_inference) {
            auto inference_tokenizer = std::move(phase1.inference_tokenizer);
            if (!inference_tokenizer) {
                EmitModuleError(ModuleId::TrainingOrchestrator,
                    "Phase 1 failed: inference tokenizer not initialized", 0);
                return 1;
            }
            printPhaseHeader(2, "Inference Loop");
            const int worker_port = requestedInferenceWorkerPort(argc, argv);
            EmitModuleInfo(ModuleId::TrainingOrchestrator,
                "[Phase 2] Inference context is ready; train_gpu owns request/session generation over Phase1-authored state", 0);
            return runInferenceWorker(ctx, *inference_tokenizer, worker_port);
        }
        
        {
            const auto paths_hp = GRIM::HyperParameters::pathsHP(ctx.config);
            std::ostringstream oss;
            oss << "[Phase 1] ✓ Complete | Model: " << paths_hp.output_model_path
                << " | Train: " << ctx.data.train_views.size()
                << " | Val: " << ctx.data.val_views.size()
                << " | Vocab: " << ctx.data.vocab_size;
            EmitModuleInfo(ModuleId::TrainingOrchestrator, oss.str(), 0);
        }
        
        //==================================================
        // PHASE 2: TRAINING LOOP
        //==================================================
        printPhaseHeader(2, "Training Loop");
        
        bool training_success = GRIMText::Training::executePhase2(ctx);
        
        if (training_success) {
            // Use project logger directly (avoid temporary ostringstream)
            std::string msg = "[Phase 2] ✓ Complete | Steps: ";
            msg += std::to_string(ctx.global_step);
            msg += " | Best val loss: ";
            msg += std::to_string(ctx.best_val_loss);
            if (ctx.auto_stop_triggered) {
                msg += " | Auto-stopped: ";
                msg += ctx.auto_stop_reason;
                msg += " (epoch ";
                msg += std::to_string(ctx.auto_stop_epoch);
                msg += ")";
            }
            EmitModuleInfo(ModuleId::TrainingOrchestrator, msg, ctx.global_step);
        } else {
            EmitModuleError(ModuleId::TrainingOrchestrator,
                "[Phase 2] ✗ Training failed", ctx.global_step);
            exit_code = 1;
        }
        
        //==================================================
        // PHASE 3: CLEANUP
        //==================================================
        printPhaseHeader(3, "Cleanup");
        
        auto cleanup_result = GRIMText::Training::executePhase3(ctx);
        
        if (cleanup_result.success) {
            std::ostringstream oss;
            oss << "[Phase 3] ✓ Complete";
            if (!cleanup_result.final_model_path.empty()) {
                oss << " | Final model: " << cleanup_result.final_model_path;
            }
            oss << " | Duration: " 
                << GRIMText::Training::Internal::formatDuration(
                       cleanup_result.summary.total_duration_seconds);
            EmitModuleInfo(ModuleId::TrainingOrchestrator, oss.str(), ctx.global_step);
        } else {
            std::ostringstream oss;
            oss << "[Phase 3] ✗ Cleanup failed: " << cleanup_result.error_message;
            EmitModuleError(ModuleId::TrainingOrchestrator, oss.str(), ctx.global_step);
            exit_code = 1;
        }
        
    } catch (const std::exception& e) {
        // Rule 20: Print to stderr BEFORE any module logging.
        // The TrainingContext (and its logger) is already destroyed by stack unwinding,
        // so EmitModuleError alone loses the message.
        fprintf(stderr, "\n[FATAL] Unhandled exception: %s\n", e.what());
        fflush(stderr);
        std::ostringstream oss;
        oss << "[FATAL] Unhandled exception: " << e.what();
        EmitModuleError(ModuleId::TrainingOrchestrator, oss.str(), 0);
        exit_code = 1;
    } catch (...) {
        fprintf(stderr, "\n[FATAL] Unknown exception (not std::exception)\n");
        fflush(stderr);
        EmitModuleError(ModuleId::TrainingOrchestrator, "[FATAL] Unknown exception", 0);
        exit_code = 1;
    }
    
    //==================================================
    // FINAL STATUS
    //==================================================
    EmitModuleInfo(ModuleId::TrainingOrchestrator, 
        "════════════════════════════════════════════════════════════", 0);
    if (exit_code == 0) {
        EmitModuleInfo(ModuleId::TrainingOrchestrator, 
            "  TRAINING COMPLETED SUCCESSFULLY", 0);
    } else {
        std::ostringstream oss;
        oss << "  TRAINING FAILED (exit code " << exit_code << ")";
        EmitModuleError(ModuleId::TrainingOrchestrator, oss.str(), 0);
    }
    EmitModuleInfo(ModuleId::TrainingOrchestrator, 
        "════════════════════════════════════════════════════════════", 0);
    
    return exit_code;
}
