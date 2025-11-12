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
#include "../resources/models/GRIM-text/GRIM/grim_tokenizer.hpp"
#include "../../../../control/training_control_generated.h"
#include "../../../../control/ai_config_paths.hpp"  // For loading paths from ai_config.json
#include "checkpoint_data_generated.h"  // FlatBuffers checkpoint schema
#include <flatbuffers/flatbuffers.h>
#include <system_error>
#include "training_paths.hpp"

using namespace GRIM::Training;
namespace fs = std::filesystem;

// Forward declarations
int runCollect(const std::string& config_path, std::function<void(float)> progressCallback);
int runVerify(const std::string& config_path);
int runMerge(const std::string& config_path);

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
    std::cout << "  --skip-verification      Skip re-verification when merging\n\n";
    std::cout << "Examples:\n";
    std::cout << "  grim_data_pipeline collect --config source_data.json\n";
    std::cout << "  grim_data_pipeline verify --raw-dir data/raw\n";
    std::cout << "  grim_data_pipeline merge --checkpoint-dir ../../../data\n";
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
    
    std::cout << "[2/2] Collecting data...\n";
    updateCollectionProgress(0);
    
    size_t collected = collector.collectData();
    updateCollectionProgress(100);
    
    const auto& raw_data = collector.getCollectedData();
    std::cout << "✓ Collected: " << collected << " new entries\n";
    std::cout << "✓ Total entries: " << raw_data.size() << "\n\n";
    
    // Save checkpoint using ai_config path
    std::string checkpoint_dir = "data"; // Default
    GRIM::Config::GrimTextPaths grimPaths;
    if (GRIM::Config::loadGrimTextPaths(grimPaths, "ai_config.json") && !grimPaths.checkpoints.empty()) {
        checkpoint_dir = grimPaths.checkpoints;
        std::cout << "[Checkpoint] Using checkpoint dir from ai_config: " << checkpoint_dir << "\n";
    } else {
        std::cout << "[Checkpoint] WARNING: Using default checkpoint dir: " << checkpoint_dir << "\n";
    }
    
    // Create checkpoint directory if needed
    std::filesystem::create_directories(checkpoint_dir);
    
    // Save as FlatBuffer checkpoint (.ckpt extension)
    std::string checkpoint_file = checkpoint_dir + "/checkpoint_" + std::to_string(raw_data.size()) + ".ckpt";
    if (collector.saveCheckpoint(checkpoint_file)) {
        std::cout << "✓ Saved checkpoint: " << checkpoint_file << "\n";
    }
    
    std::cout << "\nNext: Run 'verify' mode to filter quality data\n";
    
    return 0;
}

int runVerify(const std::string& raw_dir, const std::string& verified_dir) {
    std::cout << "=== VERIFYING DATA ===\n\n";
    
    Config config;
    config.input_dir = raw_dir;
    config.output_dir = verified_dir;
    config.reliability_threshold = 0.3f;
    config.min_length = 50;
    config.max_length = 100000;
    
    config.source_type_weights = {
        {"academic", 1.0f}, {"philosophy", 0.95f}, {"technical", 0.9f},
        {"wikipedia", 0.8f}, {"github", 0.85f}, {"gutenberg", 0.9f},
        {"arxiv", 1.0f}, {"jstor_oa", 1.0f}, {"unknown", 0.6f}
    };
    
    std::cout << "[1/2] Loading entries from " << raw_dir << "...\n";
    
    Verifier verifier(config);
    auto entries = verifier.load_unverified_entries();
    
    std::cout << "✓ Loaded " << entries.size() << " entries\n\n";
    
    std::cout << "[2/2] Verifying quality...\n";
    auto verified = verifier.verify_entries(entries);
    
    std::cout << "✓ Verified " << verified.size() << " entries\n\n";
    
    if (!verifier.save_verified_entries(verified)) {
        std::cerr << "ERROR: Failed to save verified entries\n";
        return 1;
    }
    
    auto stats = verifier.get_stats();
    std::cout << "=== Statistics ===\n";
    std::cout << "Total processed: " << stats.total_processed << "\n";
    std::cout << "Passed: " << stats.passed_verification << "\n";
    std::cout << "Failed: " << stats.failed_verification << "\n";
    std::cout << "Domain rejected: " << stats.domain_rejected << "\n";
    std::cout << "Quality rejected: " << stats.quality_rejected << "\n";
    std::cout << "Duplicates: " << stats.duplicate_rejected << "\n\n";
    
    std::cout << "Next: Run 'merge' mode to prepare training data\n";
    
    return 0;
}

int runMerge(const std::string& checkpoint_dir, const std::string& verified_dir, 
             const std::string& output_dir, bool skip_verification) {
    std::cout << "=== MERGING & PREPARING DATA ===\n\n";
    
    // Load checkpoint files if provided (FlatBuffer only)
    std::vector<std::string> checkpoint_files;
    if (!checkpoint_dir.empty() && fs::exists(checkpoint_dir)) {
        for (const auto& entry : fs::directory_iterator(checkpoint_dir)) {
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
    
    std::cout << "[1/5] Loading data...\n";
    std::cout << "  Checkpoints: " << checkpoint_files.size() << "\n";
    
    // Load existing verified entries from JSONL files
    std::vector<VerifiedEntry> existing_verified;
    
    if (fs::exists(verified_dir)) {
        for (const auto& entry : fs::directory_iterator(verified_dir)) {
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
    std::cout << "[2/5] Deduplicating...\n";
    std::unordered_set<std::string> seen_hashes;
    std::vector<VerifiedEntry> deduplicated;
    
    for (const auto& entry : all_verified) {
        std::string hash = std::to_string(std::hash<std::string>{}(entry.content));
        if (seen_hashes.find(hash) == seen_hashes.end()) {
            seen_hashes.insert(hash);
            deduplicated.push_back(entry);
        }
    }
    
    std::cout << "✓ Deduplicated: " << deduplicated.size() << " unique entries\n\n";
    
    // Preprocess
    std::cout << "[3/5] Preprocessing...\n";
    PreprocessorConfig prep_config;
    prep_config.remove_html = true;     // CRITICAL: Remove HTML tags for clean text
    prep_config.min_length = 75;       // More forgiving - accept shorter HTML extractions
    prep_config.min_words = 15;         // More forgiving - accept brief but quality content
    prep_config.min_alpha_ratio = 0.5f; // More forgiving - accept technical/structured content
    DataPreprocessor preprocessor(prep_config);
    
    std::vector<std::string> cleaned_texts;
    for (const auto& entry : deduplicated) {
        std::string clean = preprocessor.preprocess(entry.content);
        
        if (!preprocessor.passesQualityFilter(clean)) continue;
        if (preprocessor.isDuplicate(clean)) continue;
        
        clean = preprocessor.addSpecialTokens(clean);
        cleaned_texts.push_back(clean);
    }
    
    std::cout << "✓ Cleaned: " << cleaned_texts.size() << " entries\n\n";
    
    // Split data
    std::cout << "[4/5] Splitting train/val/test...\n";
    SplitConfig split_config;
    split_config.train_ratio = 0.8f;
    split_config.val_ratio = 0.1f;
    split_config.test_ratio = 0.1f;
    
    DataSplitter<std::string> splitter(split_config);
    auto split = splitter.split(cleaned_texts);
    
    std::cout << "✓ Train: " << split.train.size() << "\n";
    std::cout << "✓ Val:   " << split.validation.size() << "\n";
    std::cout << "✓ Test:  " << split.test.size() << "\n\n";
    
    // Tokenize
    std::cout << "[5/5] Tokenizing...\n";
    
    GRIM::TokenizerConfig tok_config;
    tok_config.vocab_size = 50000;
    tok_config.max_length = 8192;
    tok_config.special_tokens = {"<pad>", "<unk>", "<s>", "</s>"};
    
    GRIM::GrimTokenizer tokenizer(tok_config);
    
    // Get absolute paths using resource resolver
    auto resource_root = GRIM::Training::resolveResourceRoot();
    auto training_models_dir = resource_root / "models/GRIM-text/training/models";
    auto training_data_dir = resource_root / "models/GRIM-text/training/data";
    auto vocab_path = training_models_dir / "vocab.bin";
    
    // Try to load existing tokenizer
    fs::create_directories(training_models_dir);
    if (fs::exists(vocab_path)) {
        if (tokenizer.load(vocab_path.string())) {
            std::cout << "✓ Loaded existing tokenizer from " << vocab_path.string() << "\n";
        }
    } else {
        std::cout << "  Training new tokenizer...\n";
        tokenizer.trainFromCorpus(split.train, 10000);
        tokenizer.save(vocab_path.string());
        std::cout << "✓ Trained new tokenizer (vocab=" << tokenizer.vocabSize() << ")\n";
    }
    
    // Tokenize all splits
    auto train_tokens = tokenizer.encodeBatch(split.train);
    auto val_tokens = tokenizer.encodeBatch(split.validation);
    auto test_tokens = tokenizer.encodeBatch(split.test);
    
    // Save tokenized data in .grmt format
    fs::create_directories(training_data_dir);
    
    auto save_grmt = [&tokenizer](const std::string& path, const std::vector<std::vector<int>>& data) {
        std::ofstream file(path, std::ios::binary);
        
        // Header: magic, version, num_sequences, vocab_size
        uint32_t magic = 0x474D5254; // "GRMT"
        uint32_t version = 1;
        uint32_t num_sequences = data.size();
        uint32_t vocab_size = tokenizer.vocabSize();
        
        file.write(reinterpret_cast<const char*>(&magic), 4);
        file.write(reinterpret_cast<const char*>(&version), 4);
        file.write(reinterpret_cast<const char*>(&num_sequences), 4);
        file.write(reinterpret_cast<const char*>(&vocab_size), 4);
        
        // Sequences
        for (const auto& seq : data) {
            uint32_t len = seq.size();
            file.write(reinterpret_cast<const char*>(&len), 4);
            file.write(reinterpret_cast<const char*>(seq.data()), len * sizeof(int));
        }
        
        return file.good();
    };
    
    save_grmt((training_data_dir / "training_data.grmt").string(), train_tokens);
    save_grmt((training_data_dir / "validation_data.grmt").string(), val_tokens);
    save_grmt((training_data_dir / "test_data.grmt").string(), test_tokens);
    
    std::cout << "✓ Saved training data in .grmt format\n\n";
    
    std::cout << "=== PIPELINE COMPLETE ===\n";
    std::cout << "✓ Training data: " << (training_data_dir / "training_data.grmt").string() << "\n";
    std::cout << "✓ Vocabulary: " << vocab_path.string() << "\n";
    std::cout << "✓ Total sequences: " << train_tokens.size() << " train, " 
              << val_tokens.size() << " val\n\n";
    std::cout << "Ready for training!\n";
    
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
    }
    
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
        return runMerge(checkpoint_dir, verified_dir, output_dir, skip_verification);
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
        
        return runMerge(checkpoint_dir, verified_dir, output_dir, skip_verification);
    }
    else {
        std::cerr << "ERROR: Unknown mode '" << mode << "'\n\n";
        printUsage();
        return 1;
    }
}
