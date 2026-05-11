//======================================================//
//  GRIM-text HTTP Server
//  Ollama-compatible API for GRIM language model
//  
//  Runs locally at: http://127.0.0.1:11435
//  API endpoint: /api/generate
//  
//  Author: GRIM Development Team
//  Date: November 5, 2025
//  Version: 1.0.0
//======================================================//

#define CUDA_GPU_WRAPPER

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#endif

#include "grim_language_model_cuda.hpp"
#include "../Shared/UnigramByte/UniByte.hpp"
#include "../Shared/UnigramByte/AtomTable.hpp"
#include "../Shared/HyperParameters/HyperParameters_GPU.hpp"
#include <iostream>
#include <filesystem>
#include <memory>
#include <chrono>
#include <sstream>
#include <vector>
#include <httplib.h>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;
using namespace GRIM;
// Generation types live in HyperParameters_GPU.hpp (single source of truth).
using GRIM::HyperParameters::GenerationConfig;
using GRIM::HyperParameters::SamplingStrategy;

// Global model + tokenizer
std::unique_ptr<LanguageModel> g_model;
std::unique_ptr<GRIM::Tokenizer::UniByte> g_tokenizer;

GenerationConfig g_generation_defaults = [] {
    GenerationConfig cfg;
    cfg.strategy = SamplingStrategy::TOP_P;
    cfg.max_new_tokens = 256;
    cfg.temperature = 0.8f;
    cfg.top_p = 0.9f;
    cfg.top_k = 50;
    return cfg;
}();

//======================================================//
//  Initialize Model
//======================================================//
bool initializeModel(const HyperParameters::StartupConfig& startup_config,
                     const std::string& model_path,
                     const std::string& vocab_path)
{
    std::cout << "[GRIM-text] Initializing model...\n";
    std::cout << "[GRIM-text] Vocab: " << vocab_path << "\n";
    std::cout << "[GRIM-text] Model: " << model_path << "\n";

    try {
        g_tokenizer = std::make_unique<GRIM::Tokenizer::UniByte>();
        if (!g_tokenizer->load(vocab_path)) {
            std::cerr << "[GRIM-text] ERROR: Failed to load vocabulary\n";
            return false;
        }

        std::cout << "[GRIM-text] Loaded " << g_tokenizer->totalVocabSize() << " tokens\n";
        std::cout << "[GRIM-text] EOS token ID: " << g_tokenizer->eosId() << "\n";
        std::cout << "[GRIM-text] PAD token ID: " << g_tokenizer->padId() << "\n";

        // Start from the same Phase1-loaded hyperparameter snapshot as training.
        // Inference-specific runtime fields are applied after this assignment so
        // they cannot be overwritten by a late architecture slice.
        HyperParameters::LanguageModelConfig config = startup_config.hyperparameters.architecture;
        config.max_seq_len = startup_config.max_seq_len;
        config.vocab_size = g_tokenizer->totalVocabSize();
        config.causal_mask = true;
        config.use_gpu = true;
        config.vocab_path = vocab_path;
        config.infer_vocab_from_file = true;
        config.generation = startup_config.generation;
        
        // CRITICAL: Set correct EOS/PAD tokens from tokenizer
        config.generation.eos_token_id = g_tokenizer->eosId();
        config.generation.pad_token_id = g_tokenizer->padId();
        
        // INFERENCE MODE: Use lightweight inference state (~385MB vs ~15GB training state)
        config.execution_mode = HyperParameters::ModelExecutionMode::INFERENCE;
        
        // INFERENCE-ONLY: batch=1, but sequence capacity comes from the same
        // Phase1-derived max_seq_len so prompt/cache bounds match training config.
        config.max_cached_batch = 1;
        config.max_cached_seq_len = startup_config.max_seq_len;
        config.max_tokens_per_batch = config.max_cached_seq_len;

        config.computeDerivedValues();  // Compute head_dim = d_model / num_heads
        
        g_model = std::make_unique<LanguageModel>(config, startup_config.hyperparameters);
        std::cout << "[GRIM-text] ✓ Model object created\n" << std::flush;

        if (!fs::exists(model_path)) {
            std::cerr << "[GRIM-text] ERROR: Model file missing: " << model_path << "\n" << std::flush;
            return false;
        }
        std::cout << "[GRIM-text] Loading weights from " << model_path << "\n" << std::flush;

        if (!g_model->load(model_path)) {
            std::cerr << "[GRIM-text] ERROR: Failed to load model weights.\n" << std::flush;
            return false;
        }

        std::cout << "[GRIM-text] ✓ Weights loaded successfully\n" << std::flush;

        return true;

    } catch (const std::exception& e) {
        std::cerr << "[GRIM-text] ERROR: " << e.what() << "\n";
        return false;
    }
}

//======================================================//
//  Generate Response
//======================================================//
std::string generateResponse(const std::string& prompt, const GenerationConfig& gen_config_in)
{
    if (!g_model || !g_tokenizer)
        return "Error: Model not initialized";

    try {
        std::cout << "[Generate] Encoding prompt (" << prompt.size() << " chars)..." << std::flush;
        auto start_encode = std::chrono::high_resolution_clock::now();
        auto encoded = g_tokenizer->encodeWithMetadata(prompt);
        auto tokens = std::move(encoded.token_ids);
        auto numeric_values = std::move(encoded.token_numeric_values);
        auto text_features = std::move(encoded.token_text_features);
        auto atom_mask = std::move(encoded.token_atom_mask);
        auto atom_flags = std::move(encoded.token_atom_flags);
        auto prompt_atom_table = encoded.atom_table;
        auto atom_entry_ids = std::move(encoded.atom_entry_ids);
        auto end_encode = std::chrono::high_resolution_clock::now();
        auto encode_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_encode - start_encode).count();
        std::cout << " " << tokens.size() << " tokens (" << encode_ms << "ms)" << std::endl;

        int eos_id = g_tokenizer->eosId();
        if (!tokens.empty() && tokens.back() == eos_id) {
            tokens.pop_back();
            if (numeric_values.empty()) {
                throw std::runtime_error("generateResponse: numeric_values empty while removing EOS");
            }
            numeric_values.pop_back();
            if (text_features.size() < GRIM::Batching::BatchPayload::kTextFeatureDim) {
                throw std::runtime_error("generateResponse: text_features too short while removing EOS");
            }
            text_features.resize(text_features.size() - GRIM::Batching::BatchPayload::kTextFeatureDim);
            if (atom_mask.empty()) {
                throw std::runtime_error("generateResponse: atom_mask empty while removing EOS");
            }
            atom_mask.pop_back();
            if (atom_flags.empty()) {
                throw std::runtime_error("generateResponse: atom_flags empty while removing EOS");
            }
            atom_flags.pop_back();
            if (atom_entry_ids.empty()) {
                throw std::runtime_error("generateResponse: atom_entry_ids empty while removing EOS");
            }
            atom_entry_ids.pop_back();
            std::cout << "[Generate] Removed EOS from prompt, now " << tokens.size() << " tokens" << std::endl;
        }

        GenerationConfig gen_config = gen_config_in;
        
        // CRITICAL: Set EOS/PAD token IDs from tokenizer (not default 0!)
        gen_config.eos_token_id = g_tokenizer->eosId();
        gen_config.pad_token_id = g_tokenizer->padId();
        std::cout << "[Generate] EOS token ID: " << gen_config.eos_token_id 
                  << ", PAD token ID: " << gen_config.pad_token_id << std::endl;
        std::cout << "[Generate] Starting generation (max_tokens=" << gen_config.max_new_tokens << ", temp=" << gen_config.temperature << ")..." << std::endl << std::flush;
        auto start_gen = std::chrono::high_resolution_clock::now();
        const auto& model_cfg = g_model->getConfig();
        const std::vector<int32_t> prompt_token_to_slot_map;
        auto prompt_payload = GRIM::Batching::buildInferenceBatchPayload(
            tokens,
            numeric_values,
            text_features,
            atom_mask,
            atom_flags,
            prompt_atom_table,
            atom_entry_ids,
            prompt_token_to_slot_map,
            model_cfg.vocab_size,
            static_cast<size_t>(model_cfg.max_cached_batch),
            static_cast<size_t>(model_cfg.max_cached_seq_len),
            model_cfg.execution_block_num_slots);
        auto results = g_model->generate(prompt_payload, &gen_config);
        auto end_gen = std::chrono::high_resolution_clock::now();
        auto gen_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_gen - start_gen).count();

        if (results.empty()) {
            std::cout << "[Generate] FAILED - no results" << std::endl;
            return "Error: Generation failed";
        }

        std::cout << "[Generate] Generated " << results[0].token_ids.size() << " tokens in " << gen_ms << "ms" << std::endl;
        std::cout << "[Generate] Tokens/sec: " << (results[0].token_ids.size() * 1000.0 / gen_ms) << std::endl;
        std::cout << "[Generate] Decoding..." << std::flush;
        auto start_decode = std::chrono::high_resolution_clock::now();

        const auto& seq = results[0];
        const GRIM::Tokenizer::AtomTable* atom_tbl = seq.context_atom_table.get();
        std::string output;
        for (size_t i = 0; i < seq.token_ids.size(); ++i) {
            const int tid = seq.token_ids[i];

            if (g_tokenizer->isByteToken(tid)) {
                output.push_back(static_cast<char>(g_tokenizer->byteEncoder().tokenToByte(tid)));
                continue;
            }

            if (g_tokenizer->isAtomToken(tid)) {
                // Context atoms: use atom table raw text
                if (atom_tbl && i < seq.atom_entry_ids.size() &&
                    seq.atom_entry_ids[i] != GRIM::Tokenizer::kAtomEntryNone) {
                    const auto* entry = atom_tbl->getAtom(seq.atom_entry_ids[i]);
                    if (entry) {
                        output += atom_tbl->atomToString(*entry);
                        continue;
                    }
                }

                // Model-generated <NUM>: format the predicted numeric value
                if (i < seq.token_atom_mask.size() && seq.token_atom_mask[i] != 0 &&
                    i < seq.token_numeric_values.size()) {
                    output += GRIM::Tokenizer::formatNumericValue(seq.token_numeric_values[i]);
                    continue;
                }

                output += g_tokenizer->tokenToString(tid);
                continue;
            }

            output += g_tokenizer->tokenToString(tid);
        }

        auto end_decode = std::chrono::high_resolution_clock::now();
        auto decode_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_decode - start_decode).count();
        std::cout << " done (" << decode_ms << "ms)" << std::endl;

        return output;

    } catch (const std::exception& e) {
        std::cout << "[Generate] EXCEPTION: " << e.what() << std::endl;
        return std::string("Error: ") + e.what();
    }
}

//======================================================//
//  MAIN SERVER — LAZY CUDA INITIALIZATION
//======================================================//
int main(int argc, char** argv)
{
#ifdef _WIN32
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);
#endif

    std::cout << "========================================\n";
    std::cout << "  GRIM-text HTTP Server v1.0.0\n";
    std::cout << "  Ollama-compatible API\n";
    std::cout << "========================================\n";

    std::vector<std::string> positionals;
    std::string config_path = "ai_config.json";
    bool explicit_config = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
            explicit_config = true;
        } else {
            positionals.push_back(arg);
        }
    }

    // Try to find ai_config.json from multiple locations
    if (!explicit_config && !std::filesystem::exists(config_path)) {
        config_path = "../../../../../../ai_config.json";  // From Release dir
    }
    if (!explicit_config && !std::filesystem::exists(config_path)) {
        // Try to find GRIM root and look for ai_config.json there
        std::filesystem::path searchPath = std::filesystem::current_path();
        for (int i = 0; i < 10 && searchPath.has_parent_path(); ++i) {
            if (std::filesystem::exists(searchPath / "control") && 
                std::filesystem::exists(searchPath / "resources")) {
                std::filesystem::path configCandidate = searchPath / "ai_config.json";
                if (std::filesystem::exists(configCandidate)) {
                    config_path = configCandidate.string();
                    break;
                }
            }
            searchPath = searchPath.parent_path();
        }
    }

    std::cout << "[GRIM-text] Using config: " << config_path << " (exists: " 
              << (std::filesystem::exists(config_path) ? "yes" : "no") << ")\n";

    std::vector<std::string> startup_args = {argv[0], "--config", config_path};
    std::vector<char*> startup_argv;
    startup_argv.reserve(startup_args.size());
    for (auto& arg : startup_args) {
        startup_argv.push_back(arg.data());
    }

    HyperParameters::StartupConfig startup_config;
    try {
        startup_config = HyperParameters::loadStartupConfig(
            static_cast<int>(startup_argv.size()),
            startup_argv.data());
        g_generation_defaults = startup_config.generation;
    } catch (const std::exception& e) {
        std::cerr << "[GRIM-text] ERROR: Failed to load startup config: " << e.what() << "\n";
        return 1;
    }

    std::string vocab_path = "models/vocab.txt";
    std::string model_path = "models/grim_text.bin";
    int port = 11435;

    std::cout << "[GRIM-text] Config paths loaded successfully\n";
    if (!startup_config.paths.vocab_path.empty()) {
        vocab_path = startup_config.paths.vocab_path;
        std::cout << "[GRIM-text] Using vocab from config: " << vocab_path << "\n";
    }
    if (!startup_config.paths.output_model_path.empty()) {
        model_path = startup_config.paths.output_model_path;
        std::cout << "[GRIM-text] Using model from config: " << model_path << "\n";
    }

    if (positionals.size() >= 1) vocab_path = positionals[0];
    if (positionals.size() >= 2) model_path = positionals[1];
    if (positionals.size() >= 3) port = std::stoi(positionals[2]);

    // Create server (NO CUDA YET)
    httplib::Server svr;

    //==================================================//
    //  API ENDPOINTS (LAZY LOAD MODEL ON FIRST USE)
    //==================================================//

    svr.Get("/", [&](const httplib::Request&, httplib::Response& res) {
        json response = {{"status", "ok"}, {"model", "grim-text"}, {"version", "1.0.0"}};
        res.set_content(response.dump(), "application/json");
    });

    svr.Get("/api/tags", [&](const httplib::Request&, httplib::Response& res) {
        json response = {
            {"models", json::array({
                {{"name", "grim-text"}, {"modified_at", "2025-11-05T00:00:00Z"},
                 {"size", 1024 * 1024 * 768}, {"digest", "grim-text-v1"}}
            })}
        };
        res.set_content(response.dump(), "application/json");
    });

    svr.Post("/api/generate", [&](const httplib::Request& req, httplib::Response& res) {
        try {
            auto request = json::parse(req.body);
            std::string prompt = request.value("prompt", "");
            int max_tokens = request.value("max_tokens", g_generation_defaults.max_new_tokens);
            float temperature = request.value("temperature", g_generation_defaults.temperature);

            if (prompt.empty()) {
                res.status = 400;
                res.set_content(json({{"error", "No prompt provided"}}).dump(), "application/json");
                return;
            }

            // Build GenerationConfig from defaults + request overrides
            GenerationConfig gen_config = g_generation_defaults;
            gen_config.max_new_tokens = max_tokens;
            gen_config.temperature = temperature;
            
            // Parse optional sampling parameters
            if (request.contains("top_p")) gen_config.top_p = request["top_p"].get<float>();
            if (request.contains("top_k")) gen_config.top_k = request["top_k"].get<int>();
            if (request.contains("min_p")) gen_config.min_p = request["min_p"].get<float>();
            if (request.contains("typical_p")) gen_config.typical_p = request["typical_p"].get<float>();
            if (request.contains("repetition_penalty")) gen_config.repetition_penalty = request["repetition_penalty"].get<float>();
            if (request.contains("frequency_penalty")) gen_config.frequency_penalty = request["frequency_penalty"].get<float>();
            if (request.contains("presence_penalty")) gen_config.presence_penalty = request["presence_penalty"].get<float>();
            if (request.contains("no_repeat_ngram_size")) gen_config.no_repeat_ngram_size = request["no_repeat_ngram_size"].get<int>();
            if (request.contains("seed")) gen_config.seed = request["seed"].get<unsigned int>();
            if (request.contains("enable_scratchblock_reasoning")) {
                gen_config.enable_scratchblock_reasoning = request["enable_scratchblock_reasoning"].get<bool>();
            }
            
            // Strategy override: "greedy", "top_k", "top_p", "min_p", "typical", "top_k_top_p"
            if (request.contains("strategy")) {
                std::string strat = request["strategy"].get<std::string>();
                if (strat == "greedy") { gen_config.strategy = SamplingStrategy::GREEDY; gen_config.do_sample = false; }
                else if (strat == "top_k") gen_config.strategy = SamplingStrategy::TOP_K;
                else if (strat == "top_p") gen_config.strategy = SamplingStrategy::TOP_P;
                else if (strat == "min_p") gen_config.strategy = SamplingStrategy::MIN_P;
                else if (strat == "typical") gen_config.strategy = SamplingStrategy::TYPICAL;
                else if (strat == "top_k_top_p") gen_config.strategy = SamplingStrategy::TOP_K_TOP_P;
            }

            std::string text = generateResponse(prompt, gen_config);

            json response = {
                {"model", "grim-text"},
                {"created_at", "2025-11-05T00:00:00Z"},
                {"response", text},
                {"done", true}
            };

            res.set_content(response.dump(), "application/json");

        } catch (const std::exception& e) {
            res.status = 500;
            std::cerr << "[/api/generate] ERROR: " << e.what() << std::endl;
            res.set_content(json({{"error", std::string(e.what())}}).dump(), "application/json");
        }
    });

    svr.Post("/api/chat", [&](const httplib::Request& req, httplib::Response& res) {
        try {
            auto request = json::parse(req.body);
            std::string prompt;

            if (request.contains("messages")) {
                for (const auto& msg : request["messages"]) {
                    std::string role = msg.value("role", "");
                    std::string content = msg.value("content", "");
                    if (role == "user") prompt += "User: " + content + "\n";
                    else if (role == "assistant") prompt += "Assistant: " + content + "\n";
                }
                prompt += "Assistant: ";
            }

            int max_tokens = request.value("max_tokens", g_generation_defaults.max_new_tokens);
            float temperature = request.value("temperature", g_generation_defaults.temperature);

            // Build GenerationConfig from defaults + request overrides
            GenerationConfig gen_config = g_generation_defaults;
            gen_config.max_new_tokens = max_tokens;
            gen_config.temperature = temperature;
            
            // Parse optional sampling parameters
            if (request.contains("top_p")) gen_config.top_p = request["top_p"].get<float>();
            if (request.contains("top_k")) gen_config.top_k = request["top_k"].get<int>();
            if (request.contains("min_p")) gen_config.min_p = request["min_p"].get<float>();
            if (request.contains("typical_p")) gen_config.typical_p = request["typical_p"].get<float>();
            if (request.contains("repetition_penalty")) gen_config.repetition_penalty = request["repetition_penalty"].get<float>();
            if (request.contains("frequency_penalty")) gen_config.frequency_penalty = request["frequency_penalty"].get<float>();
            if (request.contains("presence_penalty")) gen_config.presence_penalty = request["presence_penalty"].get<float>();
            if (request.contains("no_repeat_ngram_size")) gen_config.no_repeat_ngram_size = request["no_repeat_ngram_size"].get<int>();
            if (request.contains("seed")) gen_config.seed = request["seed"].get<unsigned int>();
            if (request.contains("enable_scratchblock_reasoning")) {
                gen_config.enable_scratchblock_reasoning = request["enable_scratchblock_reasoning"].get<bool>();
            }
            if (request.contains("strategy")) {
                std::string strat = request["strategy"].get<std::string>();
                if (strat == "greedy") { gen_config.strategy = SamplingStrategy::GREEDY; gen_config.do_sample = false; }
                else if (strat == "top_k") gen_config.strategy = SamplingStrategy::TOP_K;
                else if (strat == "top_p") gen_config.strategy = SamplingStrategy::TOP_P;
                else if (strat == "min_p") gen_config.strategy = SamplingStrategy::MIN_P;
                else if (strat == "typical") gen_config.strategy = SamplingStrategy::TYPICAL;
                else if (strat == "top_k_top_p") gen_config.strategy = SamplingStrategy::TOP_K_TOP_P;
            }

            std::string text = generateResponse(prompt, gen_config);

            json response = {
                {"model", "grim-text"},
                {"created_at", "2025-11-05T00:00:00Z"},
                {"message", {{"role", "assistant"}, {"content", text}}},
                {"done", true}
            };

            res.set_content(response.dump(), "application/json");

        } catch (const std::exception& e) {
            res.status = 500;
            std::cerr << "[/api/chat] ERROR: " << e.what() << std::endl;
            res.set_content(json({{"error", std::string(e.what())}}).dump(), "application/json");
        }
    });

    //==================================================//
    //  LOAD MODEL FIRST, THEN START SERVER
    //==================================================//

    std::cout << "[GRIM-text] Loading model (this may take 30+ seconds)...\n";
    if (!initializeModel(startup_config, model_path, vocab_path)) {
        std::cerr << "[GRIM-text] ERROR: Model initialization failed\n";
        return 1;
    }

    std::cout << "[GRIM-text] ✓ Model loaded successfully\n";
    std::cout << "[GRIM-text] Starting HTTP server on http://127.0.0.1:" << port << "\n";
    std::cout << "[GRIM-text] Press Ctrl+C to stop.\n";

    // NOW start listening (CUDA already initialized)
    svr.listen("127.0.0.1", port);

#ifdef _WIN32
    WSACleanup();
#endif
    return 0;
}
