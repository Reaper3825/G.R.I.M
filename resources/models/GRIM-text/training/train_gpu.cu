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
    if (request.contains("enable_scratchblock_reasoning")) {
        gen_config.enable_scratchblock_reasoning = request.at("enable_scratchblock_reasoning").get<bool>();
    }
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

std::string chatPromptFromRequest(const json& request) {
    if (!request.contains("messages")) {
        throw std::runtime_error("inference worker chat request missing required field: messages");
    }
    const auto& messages = request.at("messages");
    if (!messages.is_array()) {
        throw std::runtime_error("inference worker chat request field messages is not an array");
    }

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
    prompt += "Assistant: ";
    return prompt;
}

int runInferenceWorker(
    GRIMText::Training::TrainingContext& ctx,
    GRIM::Tokenizer::UniByte& tokenizer,
    int port) {
    httplib::Server svr;
    const auto paths_hp = GRIM::HyperParameters::pathsHP(ctx.config);

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

    svr.Post("/internal/generate", [&](const httplib::Request& req, httplib::Response& res) {
        try {
            const json request = json::parse(req.body);
            const std::string prompt = requireJsonString(request, "prompt");
            const auto gen_config = generationHPFromRequest(ctx.config, request);
            const auto generated = GRIMText::Training::executePhase2TextInference(
                ctx,
                tokenizer,
                prompt,
                gen_config);

            json response = {
                {"model", "grim-text"},
                {"created_at", "2025-11-05T00:00:00Z"},
                {"response", generated.text},
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
            const std::string prompt = chatPromptFromRequest(request);
            const auto gen_config = generationHPFromRequest(ctx.config, request);
            const auto generated = GRIMText::Training::executePhase2TextInference(
                ctx,
                tokenizer,
                prompt,
                gen_config);

            json response = {
                {"model", "grim-text"},
                {"created_at", "2025-11-05T00:00:00Z"},
                {"message", {{"role", "assistant"}, {"content", generated.text}}},
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

        if (!ctx.model) {
            EmitModuleError(ModuleId::TrainingOrchestrator, 
                "Phase 1 failed: model not initialized", 0);
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
