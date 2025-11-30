//======================================================//
//  GRIM Data Pipeline - Unified Tool
//  Collect → Verify → Merge → Tokenize
//  
//  Single executable for entire data preparation pipeline
//======================================================//

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#endif

#include <iostream>
#include <filesystem>
#include <regex>
#include <fstream>
#include <chrono>
#include <unordered_set>
#include "web_collector.hpp"
#include "verifier.hpp"
#include "data_preprocessor.hpp"
#include "data_splitter.hpp"
#include "control/training_control_generated.h"
#include "control/ai_config_paths.hpp"  // For loading paths from ai_config.json
#include "checkpoint_data_generated.h"  // FlatBuffers checkpoint schema
#include <flatbuffers/flatbuffers.h>
#include <system_error>
#include "training_paths.hpp"

using namespace GRIM::Training;
namespace fs = std::filesystem;

namespace {
    std::string g_checkpoint_dir;
    std::string g_verified_dir;
    std::string g_output_dir = "data";
    std::string g_raw_dir;
}

// Forward declarations
int runCollect(const std::string& config_path, std::function<void(float)> progressCallback);
int runVerify(const std::string& raw_dir, const std::string& verified_dir);
int runMerge(const std::string& checkpoint_dir, const std::string& verified_dir,
             const std::string& output_dir, bool skip_verification,
             bool retrain_vocab, int vocab_size);

static void printUsage() {
    std::cout << "GRIM Data Pipeline - Unified data preparation tool\n\n";
    std::cout << "Usage: grim_data_pipeline [MODE] [OPTIONS]\n\n";
    std::cout << "Modes:\n";
    std::cout << "  collect      Collect data from web sources\n";
    std::cout << "  verify       Verify collected data quality\n";
    std::cout << "  merge        Merge checkpoints and prepare for training\n";
    std::cout << "  full         Run complete pipeline (collect → verify → merge)\n\n";
    std::cout << "Options:\n";
    std::cout << "  --config <path>          Source configuration (default: source_data.json)\n";
    std::cout << "  --checkpoint-dir <dir>   Directory with checkpoint files\n";
    std::cout << "  --raw-dir <dir>          Raw data directory (default: data/raw)\n";
    std::cout << "  --verified-dir <dir>     Verified data directory (default: data/verified)\n";
    std::cout << "  --output-dir <dir>       Output directory (default: data)\n";
    std::cout << "  --skip-verification      Skip re-verification when merging\n";
    std::cout << "  --retrain-vocab          Force retrain tokenizer (ignores existing vocab)\n";
    std::cout << "  --vocab-size <size>      Vocabulary size for tokenizer (default: 50000)\n\n";
    std::cout << "Examples:\n";
    std::cout << "  grim_data_pipeline collect --config source_data.json\n";
    std::cout << "  grim_data_pipeline verify --raw-dir data/raw\n";
    std::cout << "  grim_data_pipeline merge --checkpoint-dir ../../../data\n";
    std::cout << "  grim_data_pipeline merge --retrain-vocab --vocab-size 30000\n";
    std::cout << "  grim_data_pipeline full --config source_data.json\n";
}

void updateCollectionProgress(float progress) {
    try {
        flatbuffers::FlatBufferBuilder builder(1024);
        auto phase_str = builder.CreateString("Collecting");
        auto error_str = builder.CreateString("");
        
        auto stats = GRIMText::Control::CreateTrainingStats(builder,
            0, 0, 0, 0, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, progress, phase_str, error_str, 0, 0);
        
        auto config = GRIMText::Control::CreateTrainingConfig(builder);
        
        int64_t timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        auto status_response = GRIMText::Control::CreateStatusResponse(builder,
            GRIMText::Control::TrainingState_Collecting,
            stats, config, true, timestamp);
        
        builder.Finish(status_response);
        
        auto statusPath = GRIM::Training::getTrainingStatusFilePath();
        std::error_code mkdirErr;
        fs::create_directories(statusPath.parent_path(), mkdirErr);

        std::ofstream statusFile(statusPath, std::ios::binary);
        if (statusFile.is_open()) {
            statusFile.write(reinterpret_cast<const char*>(builder.GetBufferPointer()), builder.GetSize());
        }
    } catch (...) {}
}

int  runCollect(const std::string& config_path) {
    return runCollect(config_path, nullptr);
}

int runCollect(const std::string& config_path, std::function<void(float)> progressCallback) {
    std::cout << "=== COLLECTING DATA ===\n\n";
    
    if (!fs::exists(config_path)) {
        std::cerr << "ERROR: Config file not found: " << config_path << "\n";
        return 1;
    }
    
    std::cout << "[1/2] Loading sources from " << config_path << "...\n";
    
    WebDataCollector collector;
    collector.setVerbose(true);
    
    // Wire up progress callback if provided
    if (progressCallback) {
        collector.setProgressCallback(progressCallback);
    }
    
    std::ifstream config_file(config_path);
    if (!config_file.is_open()) {
        std::cerr << "ERROR: Cannot open config file\n";
        return 1;
    }
    
    // Parse JSON config and add sources
    std::string config_content((std::istreambuf_iterator<char>(config_file)),
                               std::istreambuf_iterator<char>());
    
    try {
        auto config_json = nlohmann::json::parse(config_content);
        
        if (config_json.contains("data_sources")) {
            for (const auto& source_json : config_json["data_sources"]) {
                if (!source_json.value("enabled", true)) continue;
                
                DataSource source;
                source.name = source_json.value("name", "");
                source.url = source_json.value("url", "");
                source.source_type = sourceTypeFromString(source_json.value("source_type", "unknown"));
                source.enabled = source_json.value("enabled", true);
                source.priority = source_json.value("priority", 5);
                source.fetch_limit = source_json.value("fetch_limit", 100);
                
                if (!source.url.empty()) {
                    collector.addSource(source);
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Failed to parse config: " << e.what() << "\n";
        return 1;
    }
    
    std::string resolved_raw_dir = g_raw_dir.empty() ? GRIM::Config::getCollectedDir() : g_raw_dir;
    if (resolved_raw_dir.empty()) {
        resolved_raw_dir = "data/raw";
    }
    fs::path raw_dir_path(resolved_raw_dir);
    std::error_code mkdirErr;
    fs::create_directories(raw_dir_path, mkdirErr);

    collector.setOutputDir(resolved_raw_dir);

    std::cout << "[2/2] Collecting data...\n";
    size_t collected = collector.collectData();
    std::cout << "\xE2\x9C\x93 Collected entries: " << collected << "\n";

    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
    std::string stamp = std::to_string(timestamp);

    fs::path raw_output_path = raw_dir_path / ("collected_" + stamp + ".jsonl");
    if (collector.saveToJsonl(raw_output_path.string())) {
        std::cout << "\xE2\x9C\x93 Saved raw JSONL: " << raw_output_path.string() << "\n";
    } else {
        std::cerr << "WARNING: Failed to save raw JSONL output." << std::endl;
    }

    if (!g_checkpoint_dir.empty()) {
        fs::path checkpoint_dir = g_checkpoint_dir;
        fs::create_directories(checkpoint_dir, mkdirErr);
        fs::path checkpoint_path = checkpoint_dir / ("checkpoint_" + stamp + ".ckpt");
        if (collector.saveCheckpoint(checkpoint_path.string())) {
            std::cout << "\xE2\x9C\x93 Saved checkpoint: " << checkpoint_path.string() << "\n";
        } else {
            std::cerr << "WARNING: Failed to save checkpoint." << std::endl;
        }
    }

    const auto& stats = collector.getStats();
    std::cout << stats.toString() << std::endl;
    std::cout << "=== COLLECTION COMPLETE ===\n\n";
    return 0;
}

int runVerify(const std::string& raw_dir, const std::string& verified_dir) {
    std::cout << "=== VERIFYING DATA ===\n\n";

    std::string resolved_raw_dir = raw_dir.empty() ? g_raw_dir : raw_dir;
    if (resolved_raw_dir.empty()) {
        resolved_raw_dir = GRIM::Config::getCollectedDir();
    }
    if (resolved_raw_dir.empty()) {
        resolved_raw_dir = "data/raw";
    }

    std::string resolved_verified_dir = verified_dir.empty() ? g_verified_dir : verified_dir;
    if (resolved_verified_dir.empty()) {
        resolved_verified_dir = GRIM::Config::getVerifiedDir();
    }
    if (resolved_verified_dir.empty()) {
        resolved_verified_dir = "data/verified";
    }

    if (!fs::exists(resolved_raw_dir)) {
        std::cerr << "ERROR: Raw directory not found: " << resolved_raw_dir << "\n";
        return 1;
    }
    fs::create_directories(resolved_verified_dir);

    Config verifier_config;
    verifier_config.input_dir = resolved_raw_dir;
    verifier_config.output_dir = resolved_verified_dir;
    verifier_config.verbose_logging = true;
    verifier_config.save_rejected = true;

    Verifier verifier(verifier_config);

    auto unverified_entries = verifier.load_unverified_entries();
    if (unverified_entries.empty()) {
        std::cout << "No new entries to verify in " << resolved_raw_dir << "\n";
        return 0;
    }

    std::cout << "Loaded " << unverified_entries.size() << " unverified entries.\n";
    auto verified_entries = verifier.verify_entries(unverified_entries);
    std::cout << "\xE2\x9C\x93 Verification complete: " << verified_entries.size() << " entries passed filters.\n";

    if (!verifier.save_verified_entries(verified_entries)) {
        std::cerr << "ERROR: Failed to save verified entries to " << resolved_verified_dir << "\n";
        return 1;
    }

    auto stats = verifier.get_stats();
    stats.writeSummaryToLog();
    std::cout << "Saved verification summary.\n";
    std::cout << "High quality: " << stats.high_quality_count
              << ", Medium: " << stats.medium_quality_count
              << ", Low: " << stats.low_quality_count << "\n";
    std::cout << "=== VERIFICATION COMPLETE ===\n\n";
    return 0;
}

int runMerge(const std::string& checkpoint_dir, const std::string& verified_dir,
             const std::string& output_dir, bool skip_verification,
             bool retrain_vocab, int vocab_size) {
    (void)skip_verification;
    (void)retrain_vocab;
    (void)vocab_size;

    std::string resolved_checkpoint_dir = checkpoint_dir.empty() ? g_checkpoint_dir : checkpoint_dir;
    std::string resolved_verified_dir = verified_dir.empty() ? g_verified_dir : verified_dir;
    std::string resolved_output_dir = output_dir.empty() ? g_output_dir : output_dir;

    if (resolved_checkpoint_dir.empty()) {
        resolved_checkpoint_dir = "data/checkpoints";
    }
    if (resolved_verified_dir.empty()) {
        resolved_verified_dir = "data/verified";
    }
    if (resolved_output_dir.empty()) {
        resolved_output_dir = "data";
    }

    // Load checkpoint files if provided (FlatBuffer only)
    std::vector<std::string> checkpoint_files;
    if (!resolved_checkpoint_dir.empty() && fs::exists(resolved_checkpoint_dir)) {
        for (const auto& entry : fs::directory_iterator(resolved_checkpoint_dir)) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().filename().string();
                std::string ext = entry.path().extension().string();
                // Only load .ckpt FlatBuffer files
                if (filename.substr(0, 11) == "checkpoint_" && ext == ".ckpt") {
                    checkpoint_files.push_back(entry.path().string());
                }
            }
        }
    }

    std::cout << "[1/4] Loading data...\n";
    std::cout << "  Checkpoints: " << checkpoint_files.size() << "\n";

    // Load existing verified entries from JSONL files
    std::vector<VerifiedEntry> existing_verified;

    if (fs::exists(resolved_verified_dir)) {
        for (const auto& entry : fs::directory_iterator(resolved_verified_dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".jsonl") {
                std::ifstream file(entry.path());
                std::string line;

                while (std::getline(file, line)) {
                    if (line.empty()) continue;

                    try {
                        nlohmann::json j = nlohmann::json::parse(line);

                        VerifiedEntry ve;
                        ve.content = j["content"].get<std::string>();
                        ve.source_url = j["source_url"].get<std::string>();
                        ve.source_type = j["source_type"].get<std::string>();
                        ve.reliability_score = j.value("reliability_score", 0.8f);
                        ve.verification_time = j.value("verification_time", (time_t)0);

                        existing_verified.push_back(ve);
                    } catch (...) {
                        continue;  // Skip malformed entries
                    }
                }
            }
        }
    }

    std::cout << "  Verified entries: " << existing_verified.size() << "\n";

    // Load checkpoint files (FlatBuffer format only)
    std::vector<VerifiedEntry> checkpoint_entries;
    for (const auto& checkpoint_file : checkpoint_files) {
        try {
            // Load FlatBuffer checkpoint
            std::ifstream file(checkpoint_file, std::ios::binary);
            if (!file.is_open()) continue;

            // Read entire file
            file.seekg(0, std::ios::end);
            size_t fileSize = file.tellg();
            file.seekg(0, std::ios::beg);

            std::vector<uint8_t> buffer(fileSize);
            file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
            file.close();

            // Parse FlatBuffer
            auto checkpoint = GRIMCheckpoint::GetCheckpoint(buffer.data());
            if (!checkpoint || !checkpoint->entries()) {
                std::cerr << "  Warning: Invalid FlatBuffer format in " << checkpoint_file << "\n";
                continue;
            }

            // Convert to VerifiedEntry
            for (const auto* fb_entry : *checkpoint->entries()) {
                if (!fb_entry || !fb_entry->content() || !fb_entry->source_url()) continue;

                VerifiedEntry ve;
                ve.content = fb_entry->content()->str();
                ve.source_url = fb_entry->source_url()->str();
                ve.source_type = fb_entry->source_type() ? fb_entry->source_type()->str() : "web";
                ve.reliability_score = fb_entry->reliability_score();
                ve.verification_time = fb_entry->fetch_timestamp();

                if (!ve.content.empty()) {
                    checkpoint_entries.push_back(ve);
                }
            }

            std::cout << "  Loaded " << checkpoint->entries()->size()
                      << " entries from " << fs::path(checkpoint_file).filename().string()
                      << " (" << (fileSize / 1024) << " KB FlatBuffer)" << std::endl;

        } catch (const std::exception& e) {
            std::cerr << "  Warning: Failed to read " << checkpoint_file
                      << ": " << e.what() << "\n";
            continue;
        }
    }

    std::cout << "  Checkpoint entries: " << checkpoint_entries.size() << "\n\n";

    // Combine all data
    std::vector<VerifiedEntry> all_verified = existing_verified;
    all_verified.insert(all_verified.end(), checkpoint_entries.begin(), checkpoint_entries.end());

    // Deduplicate
    std::cout << "[2/4] Deduplicating...\n";
    std::unordered_set<std::string> seen_hashes;
    std::vector<VerifiedEntry> deduplicated;

    for (const auto& entry : all_verified) {
        std::string hash = std::to_string(std::hash<std::string>{}(entry.content));
        if (seen_hashes.find(hash) == seen_hashes.end()) {
            seen_hashes.insert(hash);
            deduplicated.push_back(entry);
        }
    }

    std::cout << "\xE2\x9C\x93 Deduplicated: " << deduplicated.size() << " unique entries\n\n";

    // Preprocess
    std::cout << "[3/4] Preprocessing...\n";
    PreprocessorConfig prep_config;
    prep_config.remove_html = true;     // Remove HTML tags for clean text
    prep_config.min_length = 75;        // Accept shorter high-quality content
    prep_config.min_words = 15;         // Allow concise but valuable snippets
    prep_config.min_alpha_ratio = 0.5f; // Allow technical/structured content
    prep_config.max_length = 100000;    // Allow long-form content (up to 100K chars)
    prep_config.max_repetition_ratio = 0.5f;
    prep_config.dedup_threshold = 0.9f;
    DataPreprocessor preprocessor(prep_config);

    std::vector<std::string> cleaned_texts;
    for (const auto& entry : deduplicated) {
        std::string clean = preprocessor.preprocess(entry.content);

        if (!preprocessor.passesQualityFilter(clean)) continue;
        if (preprocessor.isDuplicate(clean)) continue;

        clean = preprocessor.addSpecialTokens(clean);
        cleaned_texts.push_back(clean);
    }

    std::cout << "\xE2\x9C\x93 Cleaned: " << cleaned_texts.size() << " entries\n\n";

    // Instead of splitting/tokenizing here, write a merged cleaned cache for training/tokenizer pipeline.
    std::cout << "[4/4] Writing merged verified cache...\n";

    fs::create_directories(resolved_output_dir);
    fs::path cache_path = fs::path(resolved_output_dir) / "merged_verified_cache.jsonl";

    std::ofstream cache_file(cache_path, std::ios::out | std::ios::trunc);
    if (!cache_file.is_open()) {
        std::cerr << "ERROR: Failed to open merged cache file for writing: "
                  << cache_path.string() << "\n";
        return 1;
    }

    for (const auto& text : cleaned_texts) {
        nlohmann::json j;
        j["content"] = text;
        cache_file << j.dump() << "\n";
    }

    cache_file.close();

    std::cout << "\xE2\x9C\x93 Wrote merged verified cache: " << cache_path.string() << "\n";
    std::cout << "=== MERGE COMPLETE (CACHE-ONLY) ===\n";
    std::cout << "Total cached entries: " << cleaned_texts.size() << "\n\n";
    std::cout << "Ready for tokenizer/training pipeline to consume cache.\n";

    return 0;
}

int StartDataCollection(int argc, char** argv, std::function<void(float)> progressCallback) {
    if (argc < 2) {
        printUsage();
        return 1;
    }

    std::string mode = argv[1];

    if (mode == "--help" || mode == "-h") {
        printUsage();
        return 0;
    }

    // Parse options - defaults will be overridden by ai_config.json
    std::string config_path = "";  // Will be loaded from ai_config.json
    std::string checkpoint_dir;
    std::string raw_dir;
    std::string verified_dir;
    std::string output_dir = "data";
    bool skip_verification = false;
    bool retrain_vocab = false;
    int vocab_size = 50000;  // Default vocab size

    // Load paths from ai_config.json (centralized source of truth)
    std::cout << "[DataPipeline] Loading paths from ai_config.json..." << std::endl;
    GRIM::Config::GrimTextPaths grimPaths;
    if (GRIM::Config::loadGrimTextPaths(grimPaths)) {
        // Use source_config path from config (CRITICAL FIX)
        if (!grimPaths.source_config.empty()) {
            config_path = grimPaths.source_config;
            std::cout << "[DataPipeline] ✓ Using source config from ai_config.json: " << config_path << std::endl;
        } else {
            std::cerr << "[DataPipeline] WARNING: source_config not set in ai_config.json" << std::endl;
            config_path = "DataCollection/source_data.json";  // Fallback
        }
        
        // Use checkpoints path from config if available
        if (!grimPaths.checkpoints.empty()) {
            checkpoint_dir = grimPaths.checkpoints;
            std::cout << "[DataPipeline] ✓ Using checkpoints path from ai_config.json: " << checkpoint_dir << std::endl;
        }
        
        // Use collected directory from config
        if (!grimPaths.collected.empty()) {
            raw_dir = grimPaths.collected;
            std::cout << "[DataPipeline] ✓ Using collected dir from ai_config.json: " << raw_dir << std::endl;
        }
        
        // Use verified directory from config
        if (!grimPaths.verified.empty()) {
            verified_dir = grimPaths.verified;
            std::cout << "[DataPipeline] ✓ Using verified dir from ai_config.json: " << verified_dir << std::endl;
        }
        
        // Use training_data path's parent directory as output_dir
        if (!grimPaths.training_data.empty()) {
            fs::path trainingDataPath(grimPaths.training_data);
            output_dir = trainingDataPath.parent_path().string();
            std::cout << "[DataPipeline] ✓ Using output directory from ai_config.json: " << output_dir << std::endl;
        }
    } else {
        std::cout << "[DataPipeline] WARNING: Could not load paths from ai_config.json, using defaults" << std::endl;
    }

    // Command-line arguments override ai_config.json
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        }
        else if (arg == "--checkpoint-dir" && i + 1 < argc) {
            checkpoint_dir = argv[++i];
            std::cout << "[DataPipeline] Checkpoint dir overridden by command line: " << checkpoint_dir << std::endl;
        }
        else if (arg == "--raw-dir" && i + 1 < argc) {
            raw_dir = argv[++i];
        }
        else if (arg == "--verified-dir" && i + 1 < argc) {
            verified_dir = argv[++i];
        }
        else if (arg == "--output-dir" && i + 1 < argc) {
            output_dir = argv[++i];
            std::cout << "[DataPipeline] Output dir overridden by command line: " << output_dir << std::endl;
        }
        else if (arg == "--skip-verification") {
            skip_verification = true;
        }
        else if (arg == "--retrain-vocab") {
            retrain_vocab = true;
            std::cout << "[DataPipeline] Will retrain tokenizer (ignoring existing vocab)\n";
        }
        else if (arg == "--vocab-size" && i + 1 < argc) {
            vocab_size = std::stoi(argv[++i]);
            std::cout << "[DataPipeline] Vocab size set to: " << vocab_size << "\n";
        }
    }

    // Persist paths for helper routines that rely on global context
    g_checkpoint_dir = checkpoint_dir;
    g_verified_dir = verified_dir;
    g_output_dir = output_dir;
    g_raw_dir = raw_dir;
    
    // Run requested mode
    if (mode == "collect") {
        if (config_path.empty()) {
            std::cerr << "ERROR: Source config path not specified and not found in ai_config.json\n";
            return 1;
        }
        return runCollect(config_path, progressCallback);
    }
    else if (mode == "verify") {
        return runVerify(raw_dir, verified_dir);
    }
    else if (mode == "merge") {
        return runMerge(checkpoint_dir, verified_dir, output_dir, skip_verification, retrain_vocab, vocab_size);
    }
    else if (mode == "full") {
        if (config_path.empty()) {
            std::cerr << "ERROR: Source config path not specified and not found in ai_config.json\n";
            return 1;
        }
        
        int result = runCollect(config_path, progressCallback);
        if (result != 0) return result;
        
        result = runVerify(raw_dir, verified_dir);
        if (result != 0) return result;
        
        return runMerge(checkpoint_dir, verified_dir, output_dir, skip_verification, retrain_vocab, vocab_size);
    }
    else {
        std::cerr << "ERROR: Unknown mode '" << mode << "'\n\n";
        printUsage();
        return 1;
    }
}
